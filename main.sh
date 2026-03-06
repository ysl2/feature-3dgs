#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="feature-3dgs"
CONDA_BIN="/home/songliyu/.vocal/miniforge3/bin/conda"
COLMAP_BIN="/home/songliyu/.vocal/colmap/bin/colmap"

DATASET_NAME=DJI_0544_0.25
ITERATIONS=7000
DATA_ROOT="data/${DATASET_NAME}"

# ==============================
# Step 0: 降分辨率到 0.25
# ==============================
"${CONDA_BIN}" run -n "${CONDA_ENV}" python scripts/openclaw_downscale_dataset.py \
  --src "$HOME/Templates/DJI-Mini3-Pro/20260208/102MEDIA/DJI_0544/images" \
  --dst "${DATA_ROOT}/images" \
  --scale 0.25 \
  --overwrite

# ==============================
# Step 1: COLMAP convert
# ==============================
mkdir -p "${DATA_ROOT}/input"
rsync -a --delete "${DATA_ROOT}/images/" "${DATA_ROOT}/input/"

# 清理旧的 COLMAP 产物，确保可重复执行
rm -rf "${DATA_ROOT}/distorted" "${DATA_ROOT}/sparse" "${DATA_ROOT}/stereo"

"${CONDA_BIN}" run -n "${CONDA_ENV}" python convert.py -s "${DATA_ROOT}" --colmap_executable "${COLMAP_BIN}"

# ==============================
# Step 2: LSeg 特征编码（必须在 encoders/lseg_encoder 目录执行）
# ==============================
(
  cd encoders/lseg_encoder
  "${CONDA_BIN}" run -n "${CONDA_ENV}" python -u encode_images.py \
    --backbone clip_vitl16_384 \
    --weights demo_e200.ckpt \
    --widehead --no-scaleinv \
    --outdir "../../${DATA_ROOT}/rgb_feature_langseg" \
    --test-rgb-dir "../../${DATA_ROOT}/images" \
    --workers 0
)

# ==============================
# Step 3: 训练
# ==============================
"${CONDA_BIN}" run -n "${CONDA_ENV}" python train.py \
  -s "${DATA_ROOT}" \
  -m "output/${DATASET_NAME}--lseg-${ITERATIONS}" \
  -f lseg --speedup --iterations "${ITERATIONS}"

# 可选：渲染
# "${CONDA_BIN}" run -n "${CONDA_ENV}" python render.py \
#   -s "${DATA_ROOT}" \
#   -m "output/${DATASET_NAME}--lseg-${ITERATIONS}" \
#   -f lseg --iteration "${ITERATIONS}" --novel_view
