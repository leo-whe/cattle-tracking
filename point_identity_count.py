"""Count distinct animals by linking fragmented tracks with point correspondence.

Measured on this footage, points seeded on a cow stay inside that cow's box
99-100 percent of the time for at least 300 frames (10 s). Identity is therefore
far easier to carry with a point tracker than with the detector's own
frame-to-frame association, which loses an animal and re-acquires it under a
fresh ID many times per flight.

The video is walked in overlapping windows. In each window every track that dies
in the first half is seeded with interior points at its death frame; those points
are tracked forward, and any track born later in the gap is scored by the
fraction of the points that land inside its first box. A high vote is direct
evidence that the two tracks are one animal. Links are merged with union-find
over the whole video and written to a JSON summary for the final stage.

Point tracking uses CoTracker3, fetched through ``torch.hub``.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
GRID = [(0.5, 0.5), (0.4, 0.42), (0.6, 0.42), (0.4, 0.58), (0.6, 0.58),
        (0.5, 0.38), (0.5, 0.62), (0.45, 0.5)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracks", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--window", type=int, default=400)
    parser.add_argument("--stride", type=int, default=200)
    parser.add_argument("--max-gap", type=int, default=150)
    parser.add_argument("--vote-threshold", type=float, default=0.5)
    parser.add_argument("--min-track-len", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-frames", type=int, default=None)
    return parser.parse_args()


def load_tracks(path: Path, min_len: int) -> dict[int, list[tuple]]:
    tracks: dict[int, list[tuple]] = defaultdict(list)
    with path.open() as handle:
        for row in csv.DictReader(handle):
            tracks[int(row["track_id"])].append(
                (int(row["frame"]), *(float(row[k]) for k in ("x1", "y1", "x2", "y2"))))
    return {k: sorted(v) for k, v in tracks.items() if len(v) >= min_len}


def decode(video: Path, width: int, limit: int | None) -> tuple[np.ndarray, float]:
    capture = cv2.VideoCapture(str(video))
    src_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if limit:
        total = min(total, limit)
    scale = width / src_w
    height = int(round(src_h * scale / 2) * 2)
    frames = np.empty((total, height, width, 3), dtype=np.uint8)
    i = 0
    while i < total:
        ok, frame = capture.read()
        if not ok:
            break
        frames[i] = cv2.cvtColor(cv2.resize(frame, (width, height)), cv2.COLOR_BGR2RGB)
        i += 1
        if i % 1000 == 0:
            print(f"decoded {i}/{total}", flush=True)
    capture.release()
    return frames[:i], scale


def main() -> None:
    args = parse_args()
    tracks = load_tracks(args.tracks, args.min_track_len)
    frames, scale = decode(args.video, args.width, args.max_frames)
    n_frames = len(frames)
    print(f"{len(tracks)} tracks, {n_frames} frames at {args.width}px", flush=True)

    span = {t: (v[0][0], v[-1][0]) for t, v in tracks.items()}
    first = {t: v[0] for t, v in tracks.items()}
    last = {t: v[-1] for t, v in tracks.items()}
    births_sorted = sorted(tracks, key=lambda t: span[t][0])
    birth_frames = np.array([span[t][0] for t in births_sorted])

    model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(args.device).eval()

    all_pairs: list[tuple[float, int, int, int]] = []
    for start in range(0, n_frames, args.stride):
        stop = min(start + args.window, n_frames)
        if stop - start < 40:
            break
        # tracks that die in the first half of this window, with lookahead inside it
        dying = [t for t in tracks
                 if start <= span[t][1] < start + args.stride and span[t][1] < n_frames - 5]
        if not dying:
            continue
        queries, owner = [], []
        for t in dying:
            fr, x1, y1, x2, y2 = last[t]
            w, h = x2 - x1, y2 - y1
            for fx, fy in GRID:
                queries.append([float(fr - start), (x1 + fx * w) * scale, (y1 + fy * h) * scale])
                owner.append(t)
        owner_arr = np.array(owner)
        clip = torch.from_numpy(frames[start:stop]).permute(0, 3, 1, 2)[None].float().to(args.device)
        q = torch.tensor(queries, dtype=torch.float32, device=args.device)[None]
        with torch.no_grad():
            pred, visible = model(clip, queries=q)
        pred = pred[0].cpu().numpy()
        visible = visible[0].cpu().numpy()
        del clip
        torch.cuda.empty_cache()

        for t in dying:
            fe = span[t][1]
            sel = owner_arr == t
            lo = int(np.searchsorted(birth_frames, fe + 1))
            hi = int(np.searchsorted(birth_frames, fe + args.max_gap, side="right"))
            for b in births_sorted[lo:hi]:
                fs = span[b][0]
                ti = fs - start
                if not (0 <= ti < stop - start) or b == t:
                    continue
                _, bx1, by1, bx2, by2 = first[b]
                box = np.array([bx1, by1, bx2, by2]) * scale
                p = pred[ti, sel]
                v = visible[ti, sel]
                inside = ((p[:, 0] >= box[0]) & (p[:, 0] <= box[2]) &
                          (p[:, 1] >= box[1]) & (p[:, 1] <= box[3]))
                vote = float((inside & (v > 0.5)).sum()) / max(1, int(sel.sum()))
                if vote >= args.vote_threshold:
                    all_pairs.append((vote, t, b, fs - fe))
        print(f"window {start}-{stop}: {len(dying)} deaths, {len(all_pairs)} cumulative links", flush=True)

    parent = {t: t for t in tracks}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    all_pairs.sort(key=lambda r: -r[0])
    used_a, used_b, links = set(), set(), []
    for vote, a, b, gap in all_pairs:
        if a in used_a or b in used_b or find(a) == find(b):
            continue
        used_a.add(a)
        used_b.add(b)
        parent[find(b)] = find(a)
        links.append({"pred": a, "succ": b, "vote": vote, "gap": gap})

    groups = defaultdict(list)
    for t in tracks:
        groups[find(t)].append(t)
    identity_of = {}
    for new_id, (_, members) in enumerate(
            sorted(groups.items(), key=lambda kv: min(span[t][0] for t in kv[1])), 1):
        for t in members:
            identity_of[t] = new_id

    per_frame = defaultdict(set)
    for t, v in tracks.items():
        for fr, *_ in v:
            per_frame[fr].add(identity_of[t])
    seen: set[int] = set()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.tracks.stem.replace("_tracks", "")
    with (args.out_dir / f"{stem}_point_counts.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "active_identities", "cumulative_identities"])
        for fr in range(n_frames):
            seen |= per_frame.get(fr, set())
            writer.writerow([fr, len(per_frame.get(fr, set())), len(seen)])

    ident_span = defaultdict(lambda: [10**9, -1])
    for t in tracks:
        s = ident_span[identity_of[t]]
        s[0] = min(s[0], span[t][0])
        s[1] = max(s[1], span[t][1])
    summary = {
        "point_tracker": "cotracker3_offline",
        "input_tracks": len(tracks),
        "links_confirmed_by_point_voting": len(links),
        "identities": len(groups),
        "reduction_factor": round(len(tracks) / max(1, len(groups)), 2),
        "max_active_identities": max((len(v) for v in per_frame.values()), default=0),
        "median_identity_span_frames": float(np.median([b - a + 1 for a, b in ident_span.values()])),
        "vote_threshold": args.vote_threshold,
        "window": args.window, "stride": args.stride, "max_gap": args.max_gap,
    }
    (args.out_dir / f"{stem}_point_summary.json").write_text(
        json.dumps({**summary, "links": links[:500]}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
