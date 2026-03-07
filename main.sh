#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-feature-3dgs}"
CONDA_BIN="${CONDA_BIN:-/home/songliyu/.vocal/miniforge3/bin/conda}"
COLMAP_BIN="${COLMAP_BIN:-/home/songliyu/.vocal/colmap/bin/colmap}"
CUDA_LIB_DIR="${CUDA_LIB_DIR:-/home/songliyu/.vocal/cudas/cuda_12.8.0_570.86.10/targets/x86_64-linux/lib}"

ITERATIONS="${ITERATIONS:-7000}"
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

mkdir -p "${DATA_ROOT}" "${OUTPUT_ROOT}"

if [[ ! -d "${DATA_ROOT}/input" ]]; then
  echo "[ERROR] Missing input directory: ${DATA_ROOT}/input"
  exit 1
fi

INPUT_COUNT=$(find -L "${DATA_ROOT}/input" -maxdepth 1 -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) | wc -l)
log "Input frame count: ${INPUT_COUNT}"

if [[ "${INPUT_COUNT}" -lt 1 ]]; then
  echo "[ERROR] No input images found under ${DATA_ROOT}/input"
  exit 1
fi

if [[ -e "${MODEL_PATH}" ]]; then
  echo "[ERROR] Output model path already exists: ${MODEL_PATH}"
  echo "        Move/delete it or set MODEL_PATH to a new directory before running."
  exit 1
fi

if [[ -d "${CUDA_LIB_DIR}" ]]; then
  export LD_LIBRARY_PATH="${CUDA_LIB_DIR}:${LD_LIBRARY_PATH:-}"
  log "Exported CUDA runtime path for COLMAP: ${CUDA_LIB_DIR}"
fi

# ==============================
# Step 1: COLMAP convert (on external SSD dataset)
# ==============================
log "Step 1/3: convert.py"
rm -rf "${DATA_ROOT}/distorted" "${DATA_ROOT}/sparse" "${DATA_ROOT}/stereo" "${DATA_ROOT}/images"
"${CONDA_BIN}" run -n "${CONDA_ENV}" python convert.py \
  -s "${DATA_ROOT}" \
  --colmap_executable "${COLMAP_BIN}"

# convert.py may swallow non-zero tool exits into large shell codes; verify outputs explicitly.
if [[ ! -f "${DATA_ROOT}/sparse/0/images.bin" && ! -f "${DATA_ROOT}/sparse/0/images.txt" ]]; then
  echo "[ERROR] COLMAP convert did not produce sparse/0/images.{bin,txt}."
  exit 1
fi
if [[ ! -d "${DATA_ROOT}/images" ]]; then
  echo "[ERROR] COLMAP convert did not produce undistorted images directory: ${DATA_ROOT}/images"
  exit 1
fi

# ==============================
# Step 2: LSeg feature encode
# ==============================
log "Step 2/3: LSeg encode_images.py"
rm -rf "${DATA_ROOT}/rgb_feature_langseg"
(
  cd encoders/lseg_encoder
  "${CONDA_BIN}" run -n "${CONDA_ENV}" python -u encode_images.py \
    --backbone clip_vitl16_384 \
    --weights demo_e200.ckpt \
    --widehead --no-scaleinv \
    --outdir "${DATA_ROOT}/rgb_feature_langseg" \
    --test-rgb-dir "${DATA_ROOT}/images" \
    --workers 0
)

FEATURE_COUNT=$(find "${DATA_ROOT}/rgb_feature_langseg" -maxdepth 1 -type f -name '*_fmap_CxHxW.pt' | wc -l)
log "Encoded feature maps: ${FEATURE_COUNT}"
if [[ "${FEATURE_COUNT}" -lt 1 ]]; then
  echo "[ERROR] No encoded feature maps found in ${DATA_ROOT}/rgb_feature_langseg"
  exit 1
fi

# ==============================
# Step 3: Train (original resolution pipeline)
# ==============================
log "Step 3/3: train.py"
"${CONDA_BIN}" run -n "${CONDA_ENV}" python train.py \
  -s "${DATA_ROOT}" \
  -m "${MODEL_PATH}" \
  -f lseg --speedup --iterations "${ITERATIONS}"

log "Experiment completed: ${MODEL_PATH}"
