"""Re-draw an annotated video from a tracks CSV, without re-running the detector.

Draws the in-view count, and when a deduplicated identity map is supplied it
labels each box with the stable identity rather than the raw track ID, gives each
identity a fixed colour so a re-used identity is visible immediately, and adds a
cumulative distinct-animal figure that levels off instead of climbing.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracks", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--identity-map", type=Path, default=None,
                        help="JSON mapping raw track id -> deduplicated identity")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--smooth", type=int, default=15)
    return parser.parse_args()


def colour_for(identity: int) -> tuple[int, int, int]:
    """Stable, well-spread colour per identity so a re-used ID is visually obvious."""
    hsv = np.uint8([[[(identity * 47) % 180, 200, 255]]])
    b, g, r = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return int(b), int(g), int(r)


def main() -> None:
    args = parse_args()
    identity_of: dict[int, int] = {}
    if args.identity_map and args.identity_map.is_file():
        identity_of = {int(k): int(v) for k, v in json.loads(args.identity_map.read_text()).items()}

    per_frame: dict[int, list] = defaultdict(list)
    dropped = 0
    with args.tracks.open() as handle:
        for row in csv.DictReader(handle):
            tid = int(row["track_id"])
            if identity_of and tid not in identity_of:
                # Tracks too short to survive deduplication are flicker, not animals.
                # Drawing them with their raw id would inflate the distinct-animal total.
                dropped += 1
                continue
            per_frame[int(row["frame"])].append(
                (identity_of.get(tid, tid),
                 *(float(row[k]) for k in ("x1", "y1", "x2", "y2"))))
    if dropped:
        print(f"dropped {dropped} detections from tracks absent from the identity map")

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {args.video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    src_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    out_w = min(src_w, args.width)
    out_h = int(round(src_h * out_w / src_w / 2) * 2)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create {args.out}")

    scale = out_w / src_w
    recent: deque[int] = deque(maxlen=args.smooth)
    seen: set[int] = set()
    index = 0
    while index < total:
        ok, frame = capture.read()
        if not ok:
            break
        frame = cv2.resize(frame, (out_w, out_h))
        rows = per_frame.get(index, [])
        for identity, x1, y1, x2, y2 in rows:
            seen.add(identity)
            colour = colour_for(identity)
            p1 = (int(x1 * scale), int(y1 * scale))
            p2 = (int(x2 * scale), int(y2 * scale))
            cv2.rectangle(frame, p1, p2, colour, 2)
            cv2.putText(frame, str(identity), (p1[0], max(12, p1[1] - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
        recent.append(len(rows))
        text = f"Cattle in view: {len(rows)}  (median of last {len(recent)}: {int(statistics.median(recent))})"
        if identity_of:
            text += f"   |   distinct animals so far: {len(seen)}"
        cv2.putText(frame, text, (24, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
        writer.write(frame)
        index += 1
        if index % 500 == 0:
            print(f"rendered {index}/{total}", flush=True)
    capture.release()
    writer.release()
    print(f"wrote {args.out} ({index} frames)")


if __name__ == "__main__":
    main()
