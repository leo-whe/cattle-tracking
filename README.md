# Cattle Tracking and Counting

Detects, tracks and counts cattle in aerial drone video, then deduplicates the
tracker's fragmented IDs into one stable identity per animal.

**Author:** Leo ([@leo-whe](https://github.com/leo-whe)), who designed and built the
pipeline.

This is version 2 of the pipeline. Version 1 used the COCO-pretrained `yolo26m` `cow`
class. On low-altitude, straight-down (nadir) footage that detector finds almost
nothing (recall 0.022 on a verified 46-animal frame). Version 2 bundles a YOLO26m
finetuned for this footage, `weights/cattle_nadir_m.pt`, and adds a duplicate-box
merge to the identity deduplication.

## Pipeline

```
video ─▶ 1 detect + track ─▶ 2 stitch ─▶ 3 links ─▶ 4 ground merge ─▶ 5 dup-box merge ─▶ 6 final map ─▶ 7 render / 8 report
            (YOLO26 +          (camera-     (raw-ID     (same ground     (nested boxes       identity_map_final.json
             FastTracker)       compensated) pairs)      position)        on one animal)
```

| Stage | Script | Does |
|---|---|---|
| 1 | `track_count_cattle.py` | YOLO26 detection + FastTracker association; tracks CSV, per-frame counts, JSON summary |
| 2 | `stitch_tracks.py` | links a track's end to another's start after compensating for camera motion; caches per-frame motion to `camera_motion.npz` |
| 3 | `stitch_links.py` | rewrites the stitched identities as links between raw track IDs |
| 3b | `point_identity_count.py` | *optional* (`POINT_VOTING=1`): CoTracker3 point-correspondence voting, an independent source of links |
| 4 | `merge_identities_world.py` | merges identities that never coexist and sit at the same ground position (frame-0 coordinates); writes `identity_map.json` |
| 5 | `merge_duplicate_boxes.py` | finds coexisting identities that are two nested boxes on one animal (intersection-over-smaller, one-to-one and non-transitive) |
| 6 | `apply_dup_links.py` | applies those pairs and writes `identity_map_final.json` plus the final distinct-animal count curve |
| 7 | `render_tracked_video.py` | redraws the video with one fixed colour and label per deduplicated animal, without re-running the detector |
| 8 | `report_counting.py`, `plot_count_curves.py` | counting estimators (optionally scored against an annotated frame) and count curves |

`run_pipeline.sh` runs all of it in order and is the reference for each stage's
arguments.

## Environment

You need Python 3.11 and one CUDA GPU (about 10 GB of VRAM at `imgsz` 3840).

```bash
conda create -n cattle python=3.11 -y
conda activate cattle
pip install -r requirements.txt
```

`ffmpeg` on `PATH` is optional. When it is there, the rendered video is re-encoded to
H.264.

Model files:

- `weights/cattle_nadir_m.pt` (42 MiB, single class `cattle`) ships with the repo and
  is the default detector.
- `yolo26m.pt` (COCO) is used only by the `coco-highalt` preset. Ultralytics downloads
  it on first use.
- CoTracker3 is used only by the optional stage 3b. It is fetched through `torch.hub`
  into `~/.cache/torch/hub`.

## Run

```bash
bash run_pipeline.sh <video> <output-dir> [preset] [device]
# e.g.
bash run_pipeline.sh DJI_0068.MP4 runs/v0068 default 0
POINT_VOTING=1 bash run_pipeline.sh flight.MP4 runs/flight default 0   # add stage 3b
IMGSZ=2560 CONF=0.20 bash run_pipeline.sh flight.MP4 runs/flight        # override a preset
```

The script reads the video's frame rate. On footage above 45 fps it uses the 60 fps
tracker config and doubles the stitching gap (90 to 180 frames). It also halves the
per-frame drift allowance (0.35 to 0.175 px), so every knob keeps its meaning in
seconds.

To run tracking alone, for an in-view count:

```bash
python track_count_cattle.py --source flight.MP4 --output-dir runs/flight \
    --imgsz 1920 --conf 0.25 --tracker trackers/fasttrack_60fps.yaml --device 0
```

Useful flags: `--start-sec` / `--duration-sec` to process a segment, and `--no-video`
to skip the raw-ID video (roughly 3x faster). `--model yolo26m.pt --class-names cow`
switches to the COCO detector.

## Presets: choosing the operating point

The detector's input size and confidence threshold are the parameters that have to
match the footage. They depend on how large the animals *appear* and on what
background is in view, not on the file's resolution.

| Preset | Detector | `imgsz` | `conf` | Tracker | Use for |
|---|---|---|---|---|---|
| `default` | `cattle_nadir_m` | 1920 | 0.25 | `fasttrack.yaml` / `fasttrack_60fps.yaml` | the recommended setting: nadir and oblique-herd views over pasture |
| `nadir-lowconf` | `cattle_nadir_m` | 3840 | 0.10 | `fasttrack_nadir_lowconf.yaml` | low-altitude nadir over pasture, scattered small animals; recall-limited footage |
| `mixed-terrain` | `cattle_nadir_m` | 1920 | 0.45 | `fasttrack_mixed_terrain.yaml` | flights over gravel, riverbeds or hillsides, where the detector hallucinates |
| `coco-highalt` | COCO `yolo26m` `cow` | 3200 | 0.25 | `fasttrack.yaml` | v1 behaviour: high-altitude oblique 4K over a large herd |

Why `default` is the default. Both of these frames were verified by hand:

| Detector | Nadir frame (46 cattle) | Oblique herd frame (51 cattle) |
|---|---|---|
| COCO `yolo26m`, best `imgsz` | recall **0.022** | recall 0.922, precision 0.940 |
| `cattle_nadir_m` @1920 | recall **0.978**, precision 0.938 | recall **0.941**, precision 0.923 |
| `cattle_nadir_m` @2560 | recall 0.978, precision 0.957 | recall 0.902, precision 1.000 |

@1920 is the only setting that is near-best on both kinds of frame, and it runs at
35 FPS (28 ms/frame, A100). The model's training and evaluation are documented in the
companion `cattle-detection` repository.

Why `mixed-terrain` exists: on bare gravel the detector mistakes stones for animals,
and the error grows steeply with input size. Over five gravel frames verified to be
empty, at conf 0.25, it produced 2 false positives at `imgsz` 1280, 21 at 1920, 135 at
2560 and 366 at 3840. That preset also sets the tracker's `new_track_thresh` (0.60)
above the detector feed (0.45). Low-confidence boxes can then extend an existing track
but never start one.

## Results

These are final deduplicated identities on the flights the pipeline was developed on.
"Raw tracks" means tracker IDs lasting at least 3 frames.

| Flight | Footage | Preset | Raw tracks | Final identities | ID reduction |
|---|---|---|---|---|---|
| Cattle V2 `video-001` | nadir, 30 fps, 150 s | `nadir-lowconf` | 490 | 138 | 3.6x |
| Cattle V2 `0113` | nadir, 30 fps, 66 s | `nadir-lowconf` | 2467 | 307 | 8.0x |
| Texas `0056` | nadir + oblique, 60 fps | `default` | 855 | 148 | 5.8x |
| Texas `0057` | nadir + oblique, 60 fps | `default` | 766 | 153 | 5.0x |
| Texas `0068` | nadir, 60 fps | `default` | 92 | 40 | 2.3x |
| `video-006` | oblique, gravel, pasture | `mixed-terrain` | 104 | 70 | 1.5x |
| `0818` (v1) | oblique high-altitude 4K | `coco-highalt` | 667 | 168 | 4.0x |

Identity stability improved far more than the ID counts alone suggest. On the Cattle V2
flights, the median identity lives 2053 and 1284 frames, against 26 and 19 frames for a
raw track: a 79x and 68x improvement. On 0113 the cumulative identity curve reaches its
final value by frame 1350 and then stays flat for the last 20 s, while the drone keeps
filming the same herd.

## Reading the outputs

```
<output-dir>/
  track/<stem>_tracks.csv          raw tracks: frame, track_id, x1, y1, x2, y2, confidence, ...
  track/<stem>_frame_counts.csv    active_tracks per frame (in-view count), unique_track_ids (diagnostic)
  dedup/identity_map_final.json    raw track_id -> final animal identity  (the main output)
  dedup/identity_map_final_counts.csv  active / cumulative distinct identities per frame
  <stem>_final_h264.mp4            video labelled with the final identities
  counts.json, count_curves.png    counting estimators and curves
```

Different questions call for different numbers.

**"How many animals are in view?"** Use `active_tracks` from `*_frame_counts.csv`, or
a robust percentile of it (`active_p95`, `active_max`) from `report_counting.py`. This
is the number burnt into the video. On v1 it was validated at -1 against a frame with
67 animals verified by hand.

**"How many distinct animals were seen?"** Use `cumulative_identities` from
`identity_map_final_counts.csv`.

**Never use `unique_track_ids` as a herd count.** It only ever rises, because an online
tracker that loses an animal and re-acquires it invents a new ID. On one flight it
reached 776 for roughly 168 animals.

## Known limitations

- **Domain-bound detector.** `cattle_nadir_m` was trained on two flights from one
  ranch on one day. It produces false positives on terrain it never saw: hillsides
  (large low-confidence boxes, conf 0.24-0.41, against 0.94 for a real close-up animal)
  and gravel. On `video-006` about 16% of the final identities are not animals. The fix
  is to pseudo-label those terrains and retrain, not to tune thresholds further.
- **Camera-motion drift.** Ground positions come from chaining per-frame motion
  estimates, which carry a small systematic scale bias that compounds over a flight.
  The ground merge compares identities only a few seconds apart, so it is largely
  unaffected. Statements about absolute ground coverage over a whole flight are not
  reliable, though. Georeferencing from drone telemetry would remove the drift.
- **No multi-frame identity ground truth yet.** HOTA, IDF1 and ID-switch counts are not
  computable. Counts are validated only against single hand-verified frames, so the
  totals in **Results** are unvalidated estimates, not verified herd counts. Annotation
  packages for detection ground truth, hard negatives, a MOT sequence and link review
  have been prepared but are not yet labelled.
