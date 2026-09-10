"""Detect, track and count cattle in a drone video.

Runs a COCO-pretrained YOLO detector restricted to the ``cow`` class and
associates detections across frames with FastTracker. Writes an annotated MP4,
one CSV row per tracked detection, a per-frame count CSV and a JSON summary.

The burnt-in readout is the number of animals in view, smoothed over a short
window. The cumulative unique-track-ID total is kept in the CSV and JSON as a
diagnostic only: an online tracker that loses an animal and re-acquires it
invents a new ID, so that total grows even when no new animals arrive. Run the
deduplication stages for a distinct-animal count.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from collections import deque
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "yolo26m.pt"
DEFAULT_TRACKER = ROOT / "trackers" / "fasttrack.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Input video")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "tracking")
    parser.add_argument("--start-sec", type=float, default=0.0)
    parser.add_argument("--duration-sec", type=float, default=None)
    parser.add_argument("--max-frames", type=int, default=None, help="Debug limit after start time")
    parser.add_argument("--imgsz", type=int, default=3200,
                        help="Detector input size; must match the animals' apparent size, not the file resolution")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--tracker", default=str(DEFAULT_TRACKER))
    parser.add_argument("--class-names", default="cow", help="Comma-separated classes; use 'all' to disable filtering")
    parser.add_argument("--device", default=None, help="For example 0, cpu; auto-selected by default")
    parser.add_argument("--output-width", type=int, default=1920, help="Annotated video width; 0 keeps source size")
    parser.add_argument("--no-video", action="store_true", help="Only write CSV and JSON")
    return parser.parse_args()


def choose_device(requested: str | None) -> str:
    if requested is not None:
        return requested
    return "0" if torch.cuda.is_available() else "cpu"


def main() -> None:
    args = parse_args()
    if not args.source.is_file():
        raise FileNotFoundError(f"Video not found: {args.source}")
    if args.start_sec < 0 or (args.duration_sec is not None and args.duration_sec <= 0):
        raise ValueError("start-sec must be >= 0 and duration-sec must be > 0")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.source.stem
    video_path = args.output_dir / f"{stem}_tracked.mp4"
    csv_path = args.output_dir / f"{stem}_tracks.csv"
    counts_path = args.output_dir / f"{stem}_frame_counts.csv"
    summary_path = args.output_dir / f"{stem}_summary.json"

    capture = cv2.VideoCapture(str(args.source))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {args.source}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = round(args.start_sec * fps)
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    duration_limit = round(args.duration_sec * fps) if args.duration_sec is not None else None
    limits = [value for value in (duration_limit, args.max_frames) if value is not None]
    frame_limit = min(limits) if limits else total_frames - start_frame

    output_width = width if args.output_width <= 0 else min(width, args.output_width)
    output_height = round(height * output_width / width)
    writer = None
    if not args.no_video:
        writer = cv2.VideoWriter(
            str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (output_width, output_height)
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not create output video: {video_path}")

    device = choose_device(args.device)
    # Use a local checkpoint when one is present, otherwise pass the bare name so
    # ultralytics resolves and downloads it on first use.
    model_ref = str(args.model) if args.model.is_file() else args.model.name
    model = YOLO(model_ref)
    selected_names = (
        None
        if args.class_names is None or args.class_names.strip().lower() in {"all", "*"}
        else [item.strip() for item in args.class_names.split(",")]
    )
    class_ids = None
    if selected_names is not None:
        names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
        name_to_id = {name: class_id for class_id, name in names.items()}
        missing = [name for name in selected_names if name not in name_to_id]
        if missing:
            raise ValueError(f"Unknown classes {missing}; model classes are {model.names}")
        class_ids = [name_to_id[name] for name in selected_names]
    unique_ids: set[int] = set()
    # Rolling window for the burnt-in readout. The cumulative unique-ID total is
    # deliberately not drawn: it only ever grows, because a tracker that loses and
    # re-acquires one animal invents a new ID, so on screen it reads as a herd that
    # keeps growing. It stays in the CSV and JSON as a diagnostic.
    recent: deque[int] = deque(maxlen=15)
    active_counts: list[int] = []
    processed = 0
    started = time.perf_counter()

    try:
        with csv_path.open("w", newline="") as csv_file, counts_path.open("w", newline="") as counts_file:
            fields = ["frame", "time_sec", "track_id", "class_id", "class_name", "confidence", "x1", "y1", "x2", "y2"]
            csv_writer = csv.DictWriter(csv_file, fieldnames=fields)
            csv_writer.writeheader()
            count_writer = csv.DictWriter(
                counts_file, fieldnames=["frame", "time_sec", "active_tracks", "unique_track_ids"]
            )
            count_writer.writeheader()

            while processed < frame_limit:
                ok, frame = capture.read()
                if not ok:
                    break
                result = model.track(
                    frame,
                    persist=True,
                    tracker=args.tracker,
                    imgsz=args.imgsz,
                    conf=args.conf,
                    iou=args.iou,
                    classes=class_ids,
                    device=device,
                    verbose=False,
                )[0]
                boxes = result.boxes
                tracked = boxes is not None and boxes.id is not None
                count = len(boxes) if tracked else 0
                active_counts.append(count)
                source_frame = start_frame + processed

                if tracked:
                    xyxy = boxes.xyxy.cpu().tolist()
                    ids = boxes.id.int().cpu().tolist()
                    classes = boxes.cls.int().cpu().tolist()
                    confidences = boxes.conf.cpu().tolist()
                    unique_ids.update(ids)
                    for coords, track_id, class_id, confidence in zip(xyxy, ids, classes, confidences):
                        csv_writer.writerow(
                            {
                                "frame": source_frame,
                                "time_sec": f"{source_frame / fps:.3f}",
                                "track_id": track_id,
                                "class_id": class_id,
                                "class_name": model.names[class_id],
                                "confidence": f"{confidence:.6f}",
                                "x1": f"{coords[0]:.2f}",
                                "y1": f"{coords[1]:.2f}",
                                "x2": f"{coords[2]:.2f}",
                                "y2": f"{coords[3]:.2f}",
                            }
                        )

                count_writer.writerow(
                    {
                        "frame": source_frame,
                        "time_sec": f"{source_frame / fps:.3f}",
                        "active_tracks": count,
                        "unique_track_ids": len(unique_ids),
                    }
                )

                if writer is not None:
                    annotated = result.plot(labels=True, conf=True, line_width=2, font_size=12)
                    recent.append(count)
                    smoothed = int(statistics.median(recent))
                    cv2.putText(
                        annotated,
                        f"Cattle in view: {count}  (median of last {len(recent)}: {smoothed})",
                        (40, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5,
                        (0, 255, 255),
                        3,
                        cv2.LINE_AA,
                    )
                    if (output_width, output_height) != (width, height):
                        annotated = cv2.resize(annotated, (output_width, output_height), interpolation=cv2.INTER_AREA)
                    writer.write(annotated)
                processed += 1
                if processed % 100 == 0:
                    elapsed = time.perf_counter() - started
                    print(f"Processed {processed}/{frame_limit} frames ({processed / elapsed:.2f} FPS)", flush=True)
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    elapsed = time.perf_counter() - started
    summary = {
        "source": str(args.source.resolve()),
        "model": model_ref,
        "tracker": args.tracker,
        "device": device,
        "source_video": {"width": width, "height": height, "fps": fps, "frames": total_frames},
        "processed": {
            "start_frame": start_frame,
            "frames": processed,
            "start_sec": start_frame / fps,
            "duration_sec": processed / fps,
            "wall_time_sec": elapsed,
            "processing_fps": processed / elapsed if elapsed else 0.0,
        },
        "counts": {
            "unique_track_ids": len(unique_ids),
            "max_active_tracks": max(active_counts, default=0),
            "mean_active_tracks": statistics.fmean(active_counts) if active_counts else 0.0,
            "median_active_tracks": statistics.median(active_counts) if active_counts else 0.0,
        },
        "parameters": {
            "imgsz": args.imgsz,
            "conf": args.conf,
            "iou": args.iou,
            "class_names": selected_names,
        },
        "artifacts": {
            "video": None if args.no_video else str(video_path),
            "tracks_csv": str(csv_path),
            "frame_counts_csv": str(counts_path),
        },
        "caveat": "Unique track IDs are not a ground-truth herd count; ID loss or re-entry can count one animal more than once.",
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
