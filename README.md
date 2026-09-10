# Cattle Tracking and Counting

Detects, tracks and counts cattle in aerial drone video, then deduplicates the
tracker's fragmented IDs into distinct animals.

The pipeline is built around one measured finding: on this footage the count is
limited by **detection resolution**, not by association. Against a hand-verified
frame holding 67 animals, the detector at `imgsz 1280` produced 42; at
`imgsz 3200` it produced 66. Swapping the tracker moved the same number by at most
6. Association still matters, but for identity consistency rather than the count.

## Pipeline

| Stage | Script | Does |
|---|---|---|
| 1 | `track_count_cattle.py` | YOLO detection (COCO `cow`) + FastTracker association; annotated MP4, tracks CSV, per-frame counts |
| 2 | `stitch_tracks.py` | links fragmented tracks by camera-compensated motion; also builds the camera-motion cache stages 4 uses |
| 3 | `point_identity_count.py` | links fragmented tracks by CoTracker3 point correspondence, independently of stage 2 |
| 4 | `merge_identities_world.py` | merges identities sitting at the same ground position; writes the final `identity_map.json` |
| 5 | `render_tracked_video.py` | re-draws the video labelled with deduplicated identities, no detector re-run |
| 6 | `report_counting.py`, `plot_count_curves.py` | counting estimators, optional scoring against an annotated frame, count curves |

Stages 2 and 3 are deliberately independent estimates of the same quantity. On a
full flight they agreed to within 4 percent (332 vs 319 identities from 776 raw
tracks), which is the main evidence that the deduplication is measuring something
real. Stage 4 then closes the long gaps that neither can bridge.

## Environment

Python 3.11 and one CUDA GPU. The detector runs at 4K input, so budget ~10 GB of
VRAM at `imgsz 3200`.

```bash
conda create -n cattle python=3.11 -y
conda activate cattle
pip install -r requirements.txt
```

Two model files are fetched automatically on first run and then cached, so the
first run needs network access:

- `yolo26m.pt`, the COCO-pretrained detector, from the official ultralytics
  assets into the working directory.
- CoTracker3, used by stage 3, through `torch.hub` into `~/.cache/torch/hub`.

Pass `--model /path/to/checkpoint.pt` to use a local detector instead.

## Run

Whole pipeline on one video:

```bash
bash run_pipeline.sh <video> <output-dir> [imgsz] [device]
# e.g.
bash run_pipeline.sh flight.MP4 runs/flight 3200 0
```

Tracking only, which is all that is needed for an in-view count:

```bash
python track_count_cattle.py --source flight.MP4 --output-dir runs/flight \
    --imgsz 3200 --conf 0.25 --device 0
```

Useful flags: `--start-sec` / `--duration-sec` to process a segment, `--no-video`
to skip encoding (roughly 3x faster), `--tracker trackers/bytetrack.yaml` for the
association baseline.

The deduplication stages are also runnable individually; `run_pipeline.sh` is the
reference for their arguments and ordering. Stage 4 requires the motion cache
written by stage 2 and the link summary written by stage 3.

## Choosing `imgsz`

This is the one parameter that has to be set per video. It must place the animals
in the detector's useful size range, so it depends on how large they *appear*, not
on the file resolution.

| Source | Apparent animal size | `imgsz` |
|---|---|---|
| 4K drone, high altitude | tiny (median 42 px in 3840) | 3200 |
| 1080p drone, high altitude | tiny | 2560 |
| 1080p handheld, close | large | 960 |

Upscaling recovers small animals: on 4K footage, recall rose from 0.58 at
`imgsz 1280` to 0.96 at `imgsz 3200`. It also *hurts* when the animals are already
large — on handheld close-range footage, detections fell monotonically from 27 at
`imgsz 960` to 11 at `imgsz 2560`. Lowering `--conf` does not substitute for
resolution: at `imgsz 1280, conf 0.10` recall is still only 0.75 and precision
starts falling.

## Reading the outputs

Two different questions, two different numbers.

**"How many animals are in view?"** Use `active_tracks` from `*_frame_counts.csv`,
or a robust percentile of it (`active_p95`, `active_max`) from
`report_counting.py`. This is the number burnt into the annotated video, and the
one validated at -1 against the annotated frame.

**"How many distinct animals were seen?"** Use `cumulative_identities` from
`*_world_counts.csv` after stage 4.

**Never use `unique_track_ids` as a herd count.** It is cumulative and only ever
rises, because an online tracker that loses an animal and re-acquires it invents a
new ID. On one full flight it reached 776 for roughly 168 distinct animals; each
animal was re-invented 8-11 times. Raising detector recall makes it *worse*, not
better, because more animals means more tracks to fragment. It is kept in the CSV
and JSON as a diagnostic only.

FastTracker is the default because it fragments least: 1.64 tracks per animal with
a median track lifetime of 213.5 frames, against ByteTrack's 2.57 and 16 frames on
the same cached detections — a 13x longer median track, at 140 FPS association.

## Known limitations

- **Nadir viewpoint.** Filmed straight down, the COCO `cow` class barely fires;
  a confidence sweep to 0.01 at `imgsz` 1280 and 1920 returned 0-6 boxes on frames
  that plainly contain cattle. Raising `imgsz` does not fix it. This is a
  viewpoint domain gap and needs a cattle-specific detector. As a consequence,
  low counts on nadir-heavy footage understate true presence.
- **Drift in stage 4.** Ground positions come from chaining per-frame camera
  motion, which accumulates drift over thousands of frames. The distance allowance
  grows with the gap to absorb this, which also makes the stage progressively more
  permissive. Treat its total as a bound on over-counting, not a verified count.
  Georeferencing from drone telemetry would remove the drift.
- **No multi-frame identity ground truth.** HOTA, IDF1 and ID switches are
  therefore not computable here. The count is validated against a single
  hand-verified frame (67 animals, plausible range 65-72); identity quality is
  reported only as fragmentation rate.
