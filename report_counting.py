"""Turn tracker output into cattle counts and score them against annotated frames.

Consistent IDs come first, then deduplication, then a count. This script
implements the counting half and reports several estimators side by side, because
they answer different questions:

``unique_ids_raw``      every ID the tracker ever emitted. Inflated by
                        fragmentation; kept only as a diagnostic.
``unique_ids_min<k>``   IDs alive for at least k frames. A crude deduplication
                        that removes flicker tracks.
``active_max/p95/med``  instantaneous herd size in view. The right estimator
                        when the whole herd is framed and the count wanted is
                        "how many animals are here", not "how many were ever
                        seen".

With ``--gt`` the count at the annotated frame is scored, so the error reported
is a true count error rather than a comparison between two guesses.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracks", type=Path, nargs="+", required=True, help="tracks_*.csv files")
    parser.add_argument("--gt", type=Path, default=None, help="JSON with gt_count and frame")
    parser.add_argument("--min-lengths", type=int, nargs="+", default=[5, 15, 30])
    parser.add_argument("--out", type=Path, default=None, help="Write the table as JSON here")
    return parser.parse_args()


def load_tracks(path: Path) -> tuple[dict[int, list[int]], Counter]:
    lifetimes: dict[int, list[int]] = defaultdict(list)
    per_frame: Counter = Counter()
    with path.open() as handle:
        for row in csv.DictReader(handle):
            frame = int(row["frame"])
            lifetimes[int(row["track_id"])].append(frame)
            per_frame[frame] += 1
    return lifetimes, per_frame


def estimators(lifetimes: dict[int, list[int]], per_frame: Counter, frames: int,
               min_lengths: list[int], gt_frame: int | None) -> dict[str, float]:
    counts = np.array([per_frame.get(i, 0) for i in range(frames)])
    out: dict[str, float] = {
        "unique_ids_raw": len(lifetimes),
        "active_max": int(counts.max()) if len(counts) else 0,
        "active_p95": float(np.percentile(counts, 95)) if len(counts) else 0.0,
        "active_median": float(np.median(counts)) if len(counts) else 0.0,
        "active_std": float(counts.std()) if len(counts) else 0.0,
    }
    for k in min_lengths:
        out[f"unique_ids_min{k}"] = sum(1 for v in lifetimes.values() if len(v) >= k)
    if gt_frame is not None:
        out["active_at_gt_frame"] = int(per_frame.get(gt_frame, 0))
    return out


def main() -> None:
    args = parse_args()
    gt_count = gt_frame = None
    if args.gt is not None:
        gt = json.loads(args.gt.read_text())
        gt_count, gt_frame = gt["gt_count"], gt["frame"]

    rows = []
    for path in args.tracks:
        lifetimes, per_frame = load_tracks(path)
        frames = max(per_frame) + 1 if per_frame else 0
        stats = estimators(lifetimes, per_frame, frames, args.min_lengths, gt_frame)
        stats["name"] = path.stem.replace("tracks_", "")
        lengths = sorted(len(v) for v in lifetimes.values())
        stats["median_track_len"] = float(np.median(lengths)) if lengths else 0.0
        if gt_count is not None:
            stats["count_err_at_gt_frame"] = stats["active_at_gt_frame"] - gt_count
            stats["count_mape_at_gt_frame"] = abs(stats["count_err_at_gt_frame"]) / gt_count
        rows.append(stats)

    keys = ["name", "unique_ids_raw"] + [f"unique_ids_min{k}" for k in args.min_lengths] + [
        "median_track_len", "active_max", "active_p95", "active_median", "active_std"]
    if gt_count is not None:
        keys += ["active_at_gt_frame", "count_err_at_gt_frame"]
    widths = {k: max(len(k), 12) for k in keys}
    widths["name"] = max(len(r["name"]) for r in rows) + 1
    print(" ".join(k.rjust(widths[k]) for k in keys))
    print("-" * (sum(widths.values()) + len(keys)))
    for r in rows:
        cells = []
        for k in keys:
            v = r.get(k, "")
            cells.append((f"{v:.2f}" if isinstance(v, float) else str(v)).rjust(widths[k]))
        print(" ".join(cells))
    if gt_count is not None:
        print(f"\nGround truth at frame {gt_frame}: {gt_count} cattle")
    if args.out:
        args.out.write_text(json.dumps({"gt_count": gt_count, "gt_frame": gt_frame, "rows": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
