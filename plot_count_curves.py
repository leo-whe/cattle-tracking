"""Plot the competing cattle-count curves for one video.

Shows why an on-screen number can appear to grow without new animals arriving:
the raw cumulative unique-ID curve climbs steadily, while the in-view count and
the deduplicated identity curves do not.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame-counts", type=Path, required=True, help="tracker *_frame_counts.csv")
    parser.add_argument("--stitched-counts", type=Path, default=None)
    parser.add_argument("--point-counts", type=Path, default=None)
    parser.add_argument("--world-counts", type=Path, default=None)
    parser.add_argument("--fps", type=float, default=29.97)
    parser.add_argument("--title", default="")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def read(path: Path, cols: tuple[str, ...]) -> dict[str, list[float]]:
    data: dict[str, list[float]] = {c: [] for c in cols}
    with path.open() as handle:
        for row in csv.DictReader(handle):
            for c in cols:
                data[c].append(float(row[c]))
    return data


def main() -> None:
    args = parse_args()
    base = read(args.frame_counts, ("frame", "active_tracks", "unique_track_ids"))
    t = [f / args.fps for f in base["frame"]]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(t, base["unique_track_ids"], color="#c0392b", lw=2,
            label=f"cumulative raw tracker IDs (diagnostic, not a count) -> {int(base['unique_track_ids'][-1])}")
    ax.plot(t, base["active_tracks"], color="#7f8c8d", lw=0.8, alpha=0.7,
            label="cattle in view per frame")

    for path, colour, name in ((args.stitched_counts, "#e67e22", "motion-compensated stitching"),
                               (args.point_counts, "#27ae60", "point-correspondence voting (CoTracker)"),
                               (args.world_counts, "#2980b9", "final (ground-position + duplicate-box merge)")):
        if path is None or not path.is_file():
            continue
        d = read(path, ("frame", "active_identities", "cumulative_identities"))
        ax.plot([f / args.fps for f in d["frame"]], d["cumulative_identities"],
                color=colour, lw=2,
                label=f"cumulative distinct animals, {name} -> {int(d['cumulative_identities'][-1])}")

    ax.set_xlabel("time (s)")
    ax.set_ylabel("cattle")
    ax.set_title(args.title or "Cattle count over time")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
