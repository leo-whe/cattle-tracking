#!/usr/bin/env bash
# End-to-end run for one video: detect + track, deduplicate identities, render, report.
#
# Usage: bash run_pipeline.sh <video> <output-dir> [preset] [device]
#
#   preset  detector operating point, default "default" (see README):
#             default        nadir-finetuned detector, imgsz 1920, conf 0.25
#             nadir-lowconf  nadir-finetuned detector, imgsz 3840, conf 0.10
#             mixed-terrain  nadir-finetuned detector, imgsz 1920, conf 0.45
#             coco-highalt   COCO yolo26m "cow" class,  imgsz 3200, conf 0.25
#   device  GPU index or "cpu", default 0
#
# Optional environment overrides: IMGSZ, CONF, MODEL, TRACKER, and POINT_VOTING=1 to
# add the CoTracker3 point-voting links (stage 3b) to the identity merge.
set -euo pipefail

usage="usage: run_pipeline.sh <video> <output-dir> [preset] [device]"
VIDEO=${1:?$usage}
OUT=${2:?$usage}
PRESET=${3:-default}
DEVICE=${4:-0}

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
STEM=$(basename "${VIDEO%.*}")
TRACK_DIR="$OUT/track"
DEDUP="$OUT/dedup"
TRACKS="$TRACK_DIR/${STEM}_tracks.csv"
MOTION="$DEDUP/camera_motion.npz"
mkdir -p "$TRACK_DIR" "$DEDUP"

# Frame-count knobs were tuned at ~30 fps; on 60 fps footage they are doubled (or, for
# per-frame drift, halved) so they keep the same meaning in seconds.
FPS=$(python -c "import cv2,sys; c=cv2.VideoCapture(sys.argv[1]); print(c.get(cv2.CAP_PROP_FPS))" "$VIDEO")
if python -c "import sys; sys.exit(0 if float(sys.argv[1]) > 45 else 1)" "$FPS"; then
    HIGH_FPS=1; MAX_GAP=180; DRIFT=0.175; POINT_GAP=300
else
    HIGH_FPS=0; MAX_GAP=90;  DRIFT=0.35;  POINT_GAP=150
fi

NADIR="$HERE/weights/cattle_nadir_m.pt"
case "$PRESET" in
    default)
        P_MODEL=$NADIR; P_CLASS=cattle; P_IMGSZ=1920; P_CONF=0.25
        if [[ $HIGH_FPS == 1 ]]; then P_TRACKER="$HERE/trackers/fasttrack_60fps.yaml"
        else P_TRACKER="$HERE/trackers/fasttrack.yaml"; fi ;;
    nadir-lowconf)
        P_MODEL=$NADIR; P_CLASS=cattle; P_IMGSZ=3840; P_CONF=0.10
        P_TRACKER="$HERE/trackers/fasttrack_nadir_lowconf.yaml" ;;
    mixed-terrain)
        P_MODEL=$NADIR; P_CLASS=cattle; P_IMGSZ=1920; P_CONF=0.45
        P_TRACKER="$HERE/trackers/fasttrack_mixed_terrain.yaml" ;;
    coco-highalt)
        P_MODEL=yolo26m.pt; P_CLASS=cow; P_IMGSZ=3200; P_CONF=0.25
        P_TRACKER="$HERE/trackers/fasttrack.yaml" ;;
    *) echo "unknown preset: $PRESET"; echo "$usage"; exit 2 ;;
esac
MODEL=${MODEL:-$P_MODEL}; IMGSZ=${IMGSZ:-$P_IMGSZ}; CONF=${CONF:-$P_CONF}
TRACKER=${TRACKER:-$P_TRACKER}
echo "preset=$PRESET fps=$FPS model=$MODEL imgsz=$IMGSZ conf=$CONF tracker=$(basename "$TRACKER")"

# point_identity_count.py takes a torch device string rather than a bare index
if [[ "$DEVICE" =~ ^[0-9]+$ ]]; then TORCH_DEVICE="cuda:$DEVICE"; else TORCH_DEVICE="$DEVICE"; fi

# 1. Detect and track: raw tracks CSV, per-frame counts, JSON summary. The annotated
#    video is drawn later from the deduplicated identities, so it is skipped here.
python "$HERE/track_count_cattle.py" --source "$VIDEO" --output-dir "$TRACK_DIR" \
    --model "$MODEL" --class-names "$P_CLASS" --imgsz "$IMGSZ" --conf "$CONF" --iou 0.7 \
    --tracker "$TRACKER" --device "$DEVICE" --no-video
if [[ $(wc -l < "$TRACKS") -le 1 ]]; then
    echo "no cattle tracked in $VIDEO with preset $PRESET; nothing to deduplicate." >&2
    echo "COCO weights find almost nothing on nadir footage; try the default preset." >&2
    exit 1
fi

# 2. Motion-compensated stitching; also caches the per-frame camera motion.
python "$HERE/stitch_tracks.py" --tracks "$TRACKS" --video "$VIDEO" \
    --out-dir "$DEDUP" --motion-cache "$MOTION" --max-gap "$MAX_GAP"

# 3. Express the stitched identities as links between raw track IDs.
python "$HERE/stitch_links.py" --stitched "$DEDUP/${STEM}_stitched_tracks.csv" \
    --out "$DEDUP/stitch_links.json"
LINKS=("$DEDUP/stitch_links.json")

# 3b. Optional: CoTracker3 point-correspondence voting, an independent link source.
if [[ "${POINT_VOTING:-0}" == 1 ]]; then
    python "$HERE/point_identity_count.py" --tracks "$TRACKS" --video "$VIDEO" \
        --out-dir "$DEDUP" --max-gap "$POINT_GAP" --device "$TORCH_DEVICE"
    LINKS+=("$DEDUP/${STEM}_point_summary.json")
fi

# 4. Ground-position merge of identities that never coexist: identity_map.json.
python "$HERE/merge_identities_world.py" --tracks "$TRACKS" --links "${LINKS[@]}" \
    --motion-cache "$MOTION" --out-dir "$DEDUP" --drift-per-frame "$DRIFT"

# 5. Duplicate-box merge: two coexisting identities nested on one animal.
python "$HERE/merge_duplicate_boxes.py" --tracks "$TRACKS" \
    --identity-map "$DEDUP/identity_map.json" --out "$DEDUP/dup_links.json"

# 6. Final identity map and the distinct-animal count curve.
python "$HERE/apply_dup_links.py" --identity-map "$DEDUP/identity_map.json" \
    --dup-links "$DEDUP/dup_links.json" --out "$DEDUP/identity_map_final.json" \
    --tracks "$TRACKS" | tee "$DEDUP/final_summary.json"

# 7. Annotated video, one stable colour and label per deduplicated animal.
python "$HERE/render_tracked_video.py" --tracks "$TRACKS" --video "$VIDEO" \
    --identity-map "$DEDUP/identity_map_final.json" --out "$OUT/${STEM}_final.mp4"
if command -v ffmpeg >/dev/null 2>&1; then
    ffmpeg -v error -y -i "$OUT/${STEM}_final.mp4" -c:v libx264 -preset medium -crf 23 \
        -pix_fmt yuv420p "$OUT/${STEM}_final_h264.mp4" && rm -f "$OUT/${STEM}_final.mp4"
fi

# 8. Counting estimators, and the curves that show the deduplication working.
python "$HERE/report_counting.py" --tracks "$TRACKS" --out "$OUT/counts.json"
python "$HERE/plot_count_curves.py" --fps "$FPS" \
    --frame-counts "$TRACK_DIR/${STEM}_frame_counts.csv" \
    --stitched-counts "$DEDUP/${STEM}_stitched_counts.csv" \
    --world-counts "$DEDUP/identity_map_final_counts.csv" \
    --out "$OUT/count_curves.png"

echo "pipeline finished: $OUT"
