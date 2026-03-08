#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-feature-3dgs}"
CONDA_BIN="${CONDA_BIN:-/home/songliyu/.vocal/miniforge3/bin/conda}"

ITERATIONS="${ITERATIONS:-7000}"
RENDER_ITERATION="${RENDER_ITERATION:-${ITERATIONS}}"
DATA_ROOT="${DATA_ROOT:-/media/songliyu/T7_Shield/Documents/feature-3dgs/data/DJI_0544}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/media/songliyu/T7_Shield/Documents/feature-3dgs/output}"
MODEL_PATH="${MODEL_PATH:-${OUTPUT_ROOT}/DJI_0544--lseg-${ITERATIONS}}"

SIBR_ROOT="${SIBR_ROOT:-/home/songliyu/Documents/feature-3dgs/SIBR_viewers}"
SIBR_BUILD_DIR="${SIBR_BUILD_DIR:-${SIBR_ROOT}/build}"
SIBR_INSTALL_BIN="${SIBR_INSTALL_BIN:-${SIBR_BUILD_DIR}/install/bin}"
SIBR_VIEWER_BIN="${SIBR_VIEWER_BIN:-${SIBR_INSTALL_BIN}/SIBR_remoteGaussian_app}"

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
  exit 1
fi

if [[ ! -d "${MODEL_PATH}/point_cloud/iteration_${RENDER_ITERATION}" ]]; then
  echo "[ERROR] Missing trained point cloud iteration_${RENDER_ITERATION} under ${MODEL_PATH}/point_cloud"
  exit 1
fi

# ==================================================================
# [DISABLED] Previous postprocess workflow is intentionally commented.
#   Step 4: render.py
#   Step 5: segmentation.py
#   Step 6: videos.py
# ==================================================================

# ==================================================================
# Viewer workflow (LSeg/CLIP only)
# ==================================================================

# Optional (manual, one-time): install Ubuntu dependencies if missing.
# sudo apt install -y \
#   libglew-dev libassimp-dev libboost-all-dev libgtk-3-dev libopencv-dev \
#   libglfw3-dev libavdevice-dev libavcodec-dev libeigen3-dev libxxf86vm-dev libembree-dev

if [[ ! -x "${SIBR_VIEWER_BIN}" ]]; then
  log "SIBR viewer binary not found, building..."
  cmake -B "${SIBR_BUILD_DIR}" "${SIBR_ROOT}" -DCMAKE_BUILD_TYPE=Release
  cmake --build "${SIBR_BUILD_DIR}" -j"$(nproc)" --target install
fi

if [[ ! -x "${SIBR_VIEWER_BIN}" ]]; then
  echo "[ERROR] SIBR viewer binary still missing after build: ${SIBR_VIEWER_BIN}"
  exit 1
fi

log "Launching SIBR viewer: ${SIBR_VIEWER_BIN}"
"${SIBR_VIEWER_BIN}" &
VIEWER_PID=$!

sleep 2

log "Launching view.py (lseg, iteration ${RENDER_ITERATION})"
"${CONDA_BIN}" run -n "${CONDA_ENV}" python view.py \
  -s "${DATA_ROOT}" \
  -m "${MODEL_PATH}" \
  -f lseg \
  --iteration "${RENDER_ITERATION}"

log "view.py exited. Stopping SIBR viewer (pid=${VIEWER_PID})"
kill "${VIEWER_PID}" 2>/dev/null || true

log "Viewer workflow completed."
