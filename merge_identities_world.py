"""Final deduplication stage: merge identities that occupy the same ground position.

Point voting can only bridge a gap while its seeded points are still being
tracked, so an animal the detector drops for several hundred frames still comes
back as a fresh identity. This stage catches those.

Every detection is mapped out of image coordinates into the coordinate frame of
the first video frame, by chaining the cached per-frame camera motion. A grazing
animal barely moves in that frame even when it swings right across the image, so
two identities that never coexist and sit at the same ground position are the
same animal, however long the gap between them.

Writes the final identity map, a per-frame identity count CSV and a JSON summary.

Caveat: the chained transform accumulates drift over thousands of frames, so
ground positions far apart in time are only approximate. The distance allowance
grows with the gap to absorb this, which also makes the stage progressively more
permissive; it is a bound on over-counting, not a precise georeference. Real
georeferencing from the drone's telemetry would remove the drift.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracks", type=Path, required=True)
    parser.add_argument("--links", type=Path, required=True,
                        help="*_point_summary.json produced by point_identity_count.py")
    parser.add_argument("--motion-cache", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--min-track-len", type=int, default=3)
    parser.add_argument("--drift-per-frame", type=float, default=0.35,
                        help="Ground-position allowance added per frame of gap, in pixels")
    parser.add_argument("--size-allowance", type=float, default=1.5,
                        help="Base allowance as a multiple of the animal's box width")
    return parser.parse_args()


def cumulative_inverse(transforms: np.ndarray) -> np.ndarray:
    """For each frame, the 2x3 map from that frame's pixels into frame-0 coordinates."""
    out = np.zeros((len(transforms), 2, 3))
    accumulated = np.eye(3)
    out[0] = np.linalg.inv(accumulated)[:2]
    for i in range(1, len(transforms)):
        step = np.eye(3)
        step[:2] = transforms[i]
        accumulated = step @ accumulated
        out[i] = np.linalg.inv(accumulated)[:2]
    return out


def main() -> None:
    args = parse_args()
    transforms = np.load(args.motion_cache)["transforms"]
    to_world = cumulative_inverse(transforms)

    tracks: dict[int, list[tuple]] = defaultdict(list)
    with args.tracks.open() as handle:
        for row in csv.DictReader(handle):
            x1, y1, x2, y2 = (float(row[k]) for k in ("x1", "y1", "x2", "y2"))
            tracks[int(row["track_id"])].append((int(row["frame"]), (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1))
    tracks = {k: sorted(v) for k, v in tracks.items() if len(v) >= args.min_track_len}

    parent = {t: t for t in tracks}

    def find(x, table):
        while table[x] != x:
            table[x] = table[table[x]]
            x = table[x]
        return x

    links = json.loads(args.links.read_text())["links"]
    for link in links:
        a, b = link["pred"], link["succ"]
        if a in parent and b in parent and find(a, parent) != find(b, parent):
            parent[find(b, parent)] = find(a, parent)

    groups: dict[int, list[int]] = defaultdict(list)
    for t in tracks:
        groups[find(t, parent)].append(t)
    before = len(groups)

    info = {}
    for g, members in groups.items():
        pts = []
        for t in members:
            for frame, cx, cy, w in tracks[t]:
                m = to_world[min(frame, len(to_world) - 1)]
                pts.append((frame,
                            m[0, 0] * cx + m[0, 1] * cy + m[0, 2],
                            m[1, 0] * cx + m[1, 1] * cy + m[1, 2], w))
        pts.sort()
        info[g] = (pts[0][0], pts[-1][0],
                   np.array(pts[0][1:3]), np.array(pts[-1][1:3]),
                   float(np.median([p[3] for p in pts])))

    keys = sorted(info)
    candidates = []
    for i in keys:
        _, end_i, _, pos_end_i, width_i = info[i]
        for j in keys:
            if j == i:
                continue
            start_j, _, pos_start_j, _, _ = info[j]
            if start_j <= end_i:
                continue  # coexisted, so they are different animals
            gap = start_j - end_i
            distance = float(np.linalg.norm(pos_start_j - pos_end_i))
            allowance = args.size_allowance * width_i + args.drift_per_frame * gap
            if distance <= allowance:
                candidates.append((distance / allowance, i, j))

    candidates.sort()
    world = {g: g for g in keys}
    used_a, used_b, merges = set(), set(), 0
    for _, i, j in candidates:
        if i in used_a or j in used_b or find(i, world) == find(j, world):
            continue
        used_a.add(i)
        used_b.add(j)
        world[find(j, world)] = find(i, world)
        merges += 1

    identity_of = {}
    final_groups: dict[int, list[int]] = defaultdict(list)
    for g, members in groups.items():
        final_groups[find(g, world)].extend(members)
    for new_id, (_, members) in enumerate(
            sorted(final_groups.items(), key=lambda kv: min(tracks[t][0][0] for t in kv[1])), 1):
        for t in members:
            identity_of[t] = new_id

    per_frame: dict[int, set] = defaultdict(set)
    for t, v in tracks.items():
        for frame, *_ in v:
            per_frame[frame].add(identity_of[t])
    n_frames = max(per_frame) + 1 if per_frame else 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.tracks.stem.replace("_tracks", "")

    # The map from raw track ID to deduplicated identity is the pipeline's real
    # output: render_tracked_video.py labels boxes with it, and tracks absent from
    # it are the flicker fragments dropped before merging.
    identity_map_path = args.out_dir / "identity_map.json"
    identity_map_path.write_text(
        json.dumps({str(t): identity_of[t] for t in sorted(identity_of)}) + "\n")

    seen: set[int] = set()
    with (args.out_dir / f"{stem}_world_counts.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "active_identities", "cumulative_identities"])
        for frame in range(n_frames):
            seen |= per_frame.get(frame, set())
            writer.writerow([frame, len(per_frame.get(frame, set())), len(seen)])

    summary = {
        "input_tracks": len(tracks),
        "identities_after_point_voting": before,
        "identities_after_world_merge": len(final_groups),
        "world_merges": merges,
        "total_reduction_vs_raw_tracks": round(len(tracks) / max(1, len(final_groups)), 2),
        "max_active_identities": max((len(v) for v in per_frame.values()), default=0),
        "parameters": {"drift_per_frame": args.drift_per_frame, "size_allowance": args.size_allowance},
        "outputs": {"identity_map": str(identity_map_path),
                    "counts": str(args.out_dir / f"{stem}_world_counts.csv")},
    }
    (args.out_dir / f"{stem}_world_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
