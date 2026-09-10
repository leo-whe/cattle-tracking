"""Merge fragmented tracks into per-animal identities and produce an honest count.

An online tracker loses an animal and re-acquires it under a new ID many times
per flight, so the cumulative ID total climbs steadily even when the drone is
filming one unchanging herd. On the reference flight, 514 of 776 IDs were born
within 45 frames and 200 px of another track's death: the same animal, counted
again.

This repairs that offline. The end of one track is linked to the start of
another when, after compensating for camera motion, the second starts where the
first was heading. Linked tracks are merged with union-find, and each merged
chain is one animal. The resulting cumulative count rises only when an animal
appears that no existing identity explains.

Camera motion matters because the drone pans: during a 30-frame gap the whole
scene shifts, so an animal that has not moved lands hundreds of pixels away.
Motion is estimated once per frame pair from background feature flow and cached
to ``--motion-cache``, which the later stages reuse.

Outputs a stitched tracks CSV, a per-frame count CSV and a JSON summary.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracks", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--motion-cache", type=Path, default=None,
                        help="npz of per-frame similarity transforms; computed if absent")
    parser.add_argument("--max-gap", type=int, default=90, help="Largest frame gap to bridge")
    parser.add_argument("--max-dist-scale", type=float, default=2.5,
                        help="Link radius as a multiple of the animal's box side")
    parser.add_argument("--max-size-ratio", type=float, default=1.8)
    parser.add_argument("--min-track-len", type=int, default=3,
                        help="Drop tracks shorter than this before stitching")
    parser.add_argument("--flow-width", type=int, default=960, help="Width used for motion estimation")
    parser.add_argument("--no-motion", action="store_true", help="Ablation: skip camera compensation")
    return parser.parse_args()


def estimate_camera_motion(video: Path, width: int, cache: Path | None) -> np.ndarray:
    """Per-frame 2x3 similarity mapping frame i-1 coordinates into frame i."""
    if cache is not None and cache.is_file():
        return np.load(cache)["transforms"]
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {video}")
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    src_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    scale = width / src_w
    transforms = np.tile(np.eye(2, 3, dtype=np.float32), (total, 1, 1))
    ok, prev = capture.read()
    if not ok:
        raise RuntimeError("Empty video")
    prev_small = cv2.cvtColor(cv2.resize(prev, None, fx=scale, fy=scale), cv2.COLOR_BGR2GRAY)
    index = 1
    while index < total:
        ok, frame = capture.read()
        if not ok:
            break
        small = cv2.cvtColor(cv2.resize(frame, None, fx=scale, fy=scale), cv2.COLOR_BGR2GRAY)
        pts = cv2.goodFeaturesToTrack(prev_small, maxCorners=900, qualityLevel=0.01,
                                      minDistance=8, blockSize=7)
        if pts is not None and len(pts) >= 12:
            nxt, status, _ = cv2.calcOpticalFlowPyrLK(prev_small, small, pts, None,
                                                      winSize=(21, 21), maxLevel=3)
            if nxt is not None:
                good_prev = pts[status.ravel() == 1].reshape(-1, 2)
                good_next = nxt[status.ravel() == 1].reshape(-1, 2)
                if len(good_prev) >= 12:
                    matrix, _ = cv2.estimateAffinePartial2D(
                        good_prev, good_next, method=cv2.RANSAC, ransacReprojThreshold=3.0)
                    if matrix is not None:
                        # convert the small-image transform back to full resolution
                        matrix = matrix.astype(np.float32).copy()
                        matrix[:, 2] /= scale
                        transforms[index] = matrix
        prev_small = small
        index += 1
        if index % 500 == 0:
            print(f"camera motion {index}/{total}", flush=True)
    capture.release()
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, transforms=transforms)
    return transforms


def warp_forward(point: np.ndarray, transforms: np.ndarray, start: int, end: int) -> np.ndarray:
    """Map a point from frame ``start`` into frame ``end`` by chaining per-frame motion."""
    p = point.astype(np.float64)
    for i in range(start + 1, min(end, len(transforms) - 1) + 1):
        m = transforms[i]
        p = np.array([m[0, 0] * p[0] + m[0, 1] * p[1] + m[0, 2],
                      m[1, 0] * p[0] + m[1, 1] * p[1] + m[1, 2]])
    return p


def load_tracks(path: Path) -> dict[int, list[tuple]]:
    tracks: dict[int, list[tuple]] = defaultdict(list)
    with path.open() as handle:
        for row in csv.DictReader(handle):
            x1, y1, x2, y2 = (float(row[k]) for k in ("x1", "y1", "x2", "y2"))
            tracks[int(row["track_id"])].append(
                (int(row["frame"]), (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1,
                 float(row["confidence"])))
    for v in tracks.values():
        v.sort()
    return tracks


def main() -> None:
    args = parse_args()
    tracks = load_tracks(args.tracks)
    tracks = {k: v for k, v in tracks.items() if len(v) >= args.min_track_len}
    ids = sorted(tracks)
    if not ids:
        raise ValueError("No tracks left after filtering")

    if args.no_motion:
        capture = cv2.VideoCapture(str(args.video))
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
        transforms = np.tile(np.eye(2, 3, dtype=np.float32), (total, 1, 1))
    else:
        transforms = estimate_camera_motion(args.video, args.flow_width, args.motion_cache)

    starts = {t: tracks[t][0] for t in ids}
    ends = {t: tracks[t][-1] for t in ids}
    span = {t: (tracks[t][0][0], tracks[t][-1][0]) for t in ids}

    # velocity over the last few observations, in image coordinates
    velocity = {}
    for t in ids:
        v = tracks[t]
        k = min(6, len(v) - 1)
        dt = v[-1][0] - v[-1 - k][0]
        velocity[t] = ((v[-1][1] - v[-1 - k][1]) / dt, (v[-1][2] - v[-1 - k][2]) / dt) if dt else (0.0, 0.0)

    by_start = sorted(ids, key=lambda t: span[t][0])
    start_frames = np.array([span[t][0] for t in by_start])

    candidates = []
    for a in ids:
        fe, xe, ye, we, he, _ = ends[a]
        side_a = float(np.sqrt(max(we * he, 1.0)))
        lo = int(np.searchsorted(start_frames, fe + 1))
        hi = int(np.searchsorted(start_frames, fe + args.max_gap, side="right"))
        for b in by_start[lo:hi]:
            if b == a or span[b][0] <= span[a][1]:
                continue  # must not overlap in time
            fs, xs, ys, ws, hs, _ = starts[b]
            side_b = float(np.sqrt(max(ws * hs, 1.0)))
            ratio = max(side_a, side_b) / max(min(side_a, side_b), 1e-6)
            if ratio > args.max_size_ratio:
                continue
            gap = fs - fe
            moved = warp_forward(np.array([xe, ye]), transforms, fe, fs)
            predicted = moved + np.array(velocity[a]) * gap
            dist = float(np.hypot(predicted[0] - xs, predicted[1] - ys))
            limit = args.max_dist_scale * 0.5 * (side_a + side_b) * (1 + gap / 30.0)
            if dist <= limit:
                candidates.append((dist / max(limit, 1e-6), a, b))

    candidates.sort()
    parent = {t: t for t in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    used_pred, used_succ = set(), set()
    links = 0
    for _, a, b in candidates:
        if a in used_succ or b in used_pred:
            continue
        if find(a) == find(b):
            continue
        used_succ.add(a)
        used_pred.add(b)
        parent[find(b)] = find(a)
        links += 1

    groups = defaultdict(list)
    for t in ids:
        groups[find(t)].append(t)
    identity_of = {}
    for new_id, (_, members) in enumerate(sorted(groups.items(), key=lambda kv: min(span[t][0] for t in kv[1])), 1):
        for t in members:
            identity_of[t] = new_id

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.tracks.stem.replace("_tracks", "")
    rows = []
    for t in ids:
        for fr, cx, cy, w, h, conf in tracks[t]:
            rows.append((fr, identity_of[t], t, cx, cy, w, h, conf))
    rows.sort()
    out_csv = args.out_dir / f"{stem}_stitched_tracks.csv"
    with out_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "identity", "orig_track_id", "cx", "cy", "w", "h", "confidence"])
        for r in rows:
            writer.writerow([r[0], r[1], r[2], f"{r[3]:.1f}", f"{r[4]:.1f}",
                             f"{r[5]:.1f}", f"{r[6]:.1f}", f"{r[7]:.4f}"])

    # per-frame active identities and the cumulative distinct-animal curve
    per_frame = defaultdict(set)
    for fr, ident, *_ in rows:
        per_frame[fr].add(ident)
    max_frame = max(per_frame) if per_frame else 0
    seen: set[int] = set()
    counts_csv = args.out_dir / f"{stem}_stitched_counts.csv"
    with counts_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "active_identities", "cumulative_identities"])
        for fr in range(max_frame + 1):
            seen |= per_frame.get(fr, set())
            writer.writerow([fr, len(per_frame.get(fr, set())), len(seen)])

    lengths = [span[t][1] - span[t][0] + 1 for t in ids]
    ident_span = defaultdict(lambda: [10**9, -1])
    for t in ids:
        s = ident_span[identity_of[t]]
        s[0] = min(s[0], span[t][0])
        s[1] = max(s[1], span[t][1])
    ident_len = [b - a + 1 for a, b in ident_span.values()]
    summary = {
        "tracks_csv": str(args.tracks),
        "camera_motion": not args.no_motion,
        "input_tracks": len(ids),
        "links_made": links,
        "identities_after_stitching": len(groups),
        "reduction_factor": round(len(ids) / max(1, len(groups)), 2),
        "median_track_len_frames": float(np.median(lengths)),
        "median_identity_span_frames": float(np.median(ident_len)),
        "max_active_identities": max((len(v) for v in per_frame.values()), default=0),
        "parameters": {"max_gap": args.max_gap, "max_dist_scale": args.max_dist_scale,
                       "max_size_ratio": args.max_size_ratio, "min_track_len": args.min_track_len},
        "outputs": {"tracks": str(out_csv), "counts": str(counts_csv)},
    }
    (args.out_dir / f"{stem}_stitched_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
