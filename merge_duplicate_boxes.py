#!/usr/bin/env python3
"""Link identities that are two boxes on one animal.

`merge_identities_world.py` merges identities separated in time: if two identities
coexist it treats them as different animals, which is right for fragmentation but
wrong for the other failure mode seen on the nadir flights. At the detector's
operating point a single animal sometimes carries two surviving boxes, a small one
nested inside a larger one. NMS does not remove these because a small box inside a
large one has low IoU; on the 0113 flight 8.7% of boxes were in such a pair.

The test here is intersection-over-smaller rather than IoU, and it has to hold for
most of the frames the two tracks share, so that animals which merely pass close to
each other are not merged. Output is a links file in the same format stitch_links.py
writes, so it can be concatenated with the stitching links.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracks", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--identity-map", type=Path, default=None,
                        help="Raw track -> stitched identity. Pairs are judged between "
                             "identities, not raw tracks; without it one link between two "
                             "raw tracks silently merges the two whole identities they "
                             "belong to, and those merges chain.")
    parser.add_argument("--ios", type=float, default=0.7,
                        help="Intersection-over-smaller above which one box is 'inside' the other")
    parser.add_argument("--min-overlap-frac", type=float, default=0.6,
                        help="Fraction of shared frames that must be nested")
    parser.add_argument("--min-shared", type=int, default=5,
                        help="Minimum shared frames before a pair is judged")
    parser.add_argument("--sample", type=int, default=2, help="Examine every Nth frame")
    args = parser.parse_args()

    ident = {}
    if args.identity_map:
        ident = {int(k): int(v) for k, v in json.loads(args.identity_map.read_text()).items()}

    # frame -> identity -> largest box, so an identity contributes one box per frame
    # and cannot be paired against itself.
    biggest: dict[int, dict[int, tuple]] = defaultdict(dict)
    area_sum: dict[int, float] = defaultdict(float)
    area_n: dict[int, int] = defaultdict(int)
    with args.tracks.open() as handle:
        for row in csv.DictReader(handle):
            frame = int(row["frame"])
            if frame % args.sample:
                continue
            t = int(row["track_id"])
            if ident:
                if t not in ident:
                    continue
                t = ident[t]
            x1, y1, x2, y2 = (float(row[k]) for k in ("x1", "y1", "x2", "y2"))
            area = (x2 - x1) * (y2 - y1)
            area_sum[t] += area
            area_n[t] += 1
            prev = biggest[frame].get(t)
            if prev is None or area > (prev[3] - prev[1]) * (prev[4] - prev[2]):
                biggest[frame][t] = (t, x1, y1, x2, y2)

    mean_area = {t: area_sum[t] / area_n[t] for t in area_sum}
    shared: dict[tuple[int, int], int] = defaultdict(int)
    nested: dict[tuple[int, int], int] = defaultdict(int)
    for by_ident in biggest.values():
        boxes = list(by_ident.values())
        for i in range(len(boxes)):
            ti, ax1, ay1, ax2, ay2 = boxes[i]
            for j in range(i + 1, len(boxes)):
                tj, bx1, by1, bx2, by2 = boxes[j]
                key = (ti, tj) if ti < tj else (tj, ti)
                shared[key] += 1
                ix1, iy1 = max(ax1, bx1), max(ay1, by1)
                ix2, iy2 = min(ax2, bx2), min(ay2, by2)
                if ix2 <= ix1 or iy2 <= iy1:
                    continue
                inter = (ix2 - ix1) * (iy2 - iy1)
                smaller = min((ax2 - ax1) * (ay2 - ay1), (bx2 - bx1) * (by2 - by1))
                if smaller > 0 and inter / smaller >= args.ios:
                    nested[key] += 1

    # One-to-one, non-transitive assignment. Union-find over raw nested pairs chains
    # through a dense herd -- on the 0113 flight it produced groups of up to 14 tracks
    # and put one identity on animals all over the frame. So each absorbed track is
    # merged into exactly one absorbing track, and a track that has been absorbed is
    # never allowed to absorb another, which bounds every merge to a single pair.
    candidates = []
    for key, n in nested.items():
        s = shared[key]
        if s >= args.min_shared and n / s >= args.min_overlap_frac:
            candidates.append((n / s, n, key))
    candidates.sort(reverse=True)

    absorbed: set[int] = set()
    absorbing: set[int] = set()
    links = []
    for frac, n, (a, b) in candidates:
        keep, drop = (a, b) if mean_area[a] >= mean_area[b] else (b, a)
        if keep in absorbed or drop in absorbed or drop in absorbing:
            continue
        absorbed.add(drop)
        absorbing.add(keep)
        links.append({"pred": keep, "succ": drop, "shared": shared[(a, b)],
                      "nested": n, "nested_frac": round(frac, 3)})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "source": str(args.tracks),
        "pairs_examined": len(shared),
        "duplicate_pairs": len(links),
        "parameters": {"ios": args.ios, "min_overlap_frac": args.min_overlap_frac,
                       "min_shared": args.min_shared, "sample": args.sample},
        "links": links,
    }, indent=1) + "\n")
    print(json.dumps({"pairs_examined": len(shared), "duplicate_pairs": len(links)}, indent=2))


if __name__ == "__main__":
    main()
