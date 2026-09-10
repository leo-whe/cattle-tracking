#!/usr/bin/env bash
# End-to-end run for one video: track, deduplicate identities, render, report.
#
# Usage: bash run_pipeline.sh <video> <output-dir> [imgsz] [device]
#   imgsz  detector input size, default 3200 (see README: it is per video)
#   device GPU index or "cpu", default 0
set -euo pipefail

VIDEO=${1:?usage: run_pipeline.sh <video> <output-dir> [imgsz] [device]}
OUT=${2:?usage: run_pipeline.sh <video> <output-dir> [imgsz] [device]}
IMGSZ=${3:-3200}
DEVICE=${4:-0}

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
STEM=$(basename "${VIDEO%.*}")
TRACKS="$OUT/track/${STEM}_tracks.csv"
DEDUP="$OUT/dedup"
MOTION="$DEDUP/camera_motion.npz"

# point_identity_count.py takes a torch device string rather than a bare index
if [[ "$DEVICE" =~ ^[0-9]+$ ]]; then TORCH_DEVICE="cuda:$DEVICE"; else TORCH_DEVICE="$DEVICE"; fi

# 1. Detect and track. Writes the tracks CSV, the per-frame count CSV, an annotated MP4.
python "$HERE/track_count_cattle.py" --source "$VIDEO" --output-dir "$OUT/track" \
    --imgsz "$IMGSZ" --conf 0.25 --iou 0.7 --device "$DEVICE"

# 2. Motion-compensated stitching. Also builds the camera-motion cache that step 4 needs.
python "$HERE/stitch_tracks.py" --tracks "$TRACKS" --video "$VIDEO" \
    --out-dir "$DEDUP" --motion-cache "$MOTION"

# 3. Point-correspondence voting, an identity estimate independent of step 2.
python "$HERE/point_identity_count.py" --tracks "$TRACKS" --video "$VIDEO" \
    --out-dir "$DEDUP" --device "$TORCH_DEVICE"

# 4. Ground-position merge: the final identity map and distinct-animal count.
python "$HERE/merge_identities_world.py" --tracks "$TRACKS" \
    --links "$DEDUP/${STEM}_point_summary.json" \
    --motion-cache "$MOTION" --out-dir "$DEDUP"

# 5. Annotated video labelled with the deduplicated identities.
python "$HERE/render_tracked_video.py" --tracks "$TRACKS" --video "$VIDEO" \
    --identity-map "$DEDUP/identity_map.json" --out "$OUT/${STEM}_dedup.mp4"

# 6. Counting estimators, and the curve that shows the deduplication working.
python "$HERE/report_counting.py" --tracks "$TRACKS" --out "$OUT/counts.json"
python "$HERE/plot_count_curves.py" \
    --frame-counts "$OUT/track/${STEM}_frame_counts.csv" \
    --stitched-counts "$DEDUP/${STEM}_stitched_counts.csv" \
    --point-counts "$DEDUP/${STEM}_point_counts.csv" \
    --world-counts "$DEDUP/${STEM}_world_counts.csv" \
    --out "$OUT/count_curves.png"

echo "pipeline finished: $OUT"
