#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-feature-3dgs}"
CONDA_BIN="${CONDA_BIN:-/home/songliyu/.vocal/miniforge3/bin/conda}"

ITERATIONS="${ITERATIONS:-7000}"
RENDER_ITERATION="${RENDER_ITERATION:-${ITERATIONS}}"
FPS="${FPS:-10}"

DATA_ROOT="${DATA_ROOT:-/media/songliyu/T7_Shield/Documents/feature-3dgs/data/DJI_0544}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/media/songliyu/T7_Shield/Documents/feature-3dgs/output}"
MODEL_PATH="${MODEL_PATH:-${OUTPUT_ROOT}/DJI_0544--lseg-${ITERATIONS}}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

log "Using DATA_ROOT=${DATA_ROOT}"
log "Using MODEL_PATH=${MODEL_PATH}"
log "Using RENDER_ITERATION=${RENDER_ITERATION}"

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "[ERROR] MODEL_PATH not found: ${MODEL_PATH}"
  echo "        This postprocess pipeline requires an existing trained model."
  exit 1
fi

if [[ ! -d "${MODEL_PATH}/point_cloud/iteration_${RENDER_ITERATION}" ]]; then
  echo "[ERROR] Missing trained point cloud iteration_${RENDER_ITERATION} under ${MODEL_PATH}/point_cloud"
  exit 1
fi

# ==================================================================
# [DISABLED] Preprocess and training are intentionally commented out.
# Keep for traceability and quick rollback.
#
# Step 1: convert.py
# Step 2: encode_images.py
# Step 3: train.py
# ==================================================================

# ==============================
# Step 4: render.py
# ==============================
NOVEL_RENDER_DIR="${MODEL_PATH}/novel_views/ours_${RENDER_ITERATION}/renders"
NOVEL_RENDER_COUNT=$(find "${NOVEL_RENDER_DIR}" -maxdepth 1 -type f -name '*.png' 2>/dev/null | wc -l)
if [[ "${NOVEL_RENDER_COUNT}" -gt 0 ]]; then
  log "Step 4/6: render.py skipped (novel_view already exists: ${NOVEL_RENDER_COUNT} frames)"
else
  log "Step 4/6: render.py (novel_view only, skip train/test because no test cameras)"
  "${CONDA_BIN}" run -n "${CONDA_ENV}" python render.py \
    -s "${DATA_ROOT}" \
    -m "${MODEL_PATH}" \
    -f lseg \
    --iteration "${RENDER_ITERATION}" \
    --skip_train \
    --skip_test \
    --novel_view
fi

NOVEL_RENDER_COUNT=$(find "${NOVEL_RENDER_DIR}" -maxdepth 1 -type f -name '*.png' 2>/dev/null | wc -l)
log "Novel-view rendered frames: ${NOVEL_RENDER_COUNT}"
if [[ "${NOVEL_RENDER_COUNT}" -lt 1 ]]; then
  echo "[ERROR] novel_view renders missing under ${NOVEL_RENDER_DIR}"
  exit 1
fi

# ==============================
# Step 5: segmentation.py (no label_src)
# ==============================
log "Step 5/6: segmentation.py (default labels)"
(
  cd encoders/lseg_encoder
  "${CONDA_BIN}" run -n "${CONDA_ENV}" python -u segmentation.py \
    --data "${MODEL_PATH}" \
    --iteration "${RENDER_ITERATION}"
)

# ==============================
# Step 6: videos.py
# ==============================
log "Step 6/6: videos.py"
"${CONDA_BIN}" run -n "${CONDA_ENV}" python videos.py \
  --data "${MODEL_PATH}" \
  --fps "${FPS}" \
  -f lseg \
  --iteration "${RENDER_ITERATION}"

log "Postprocess pipeline completed: ${MODEL_PATH}"
