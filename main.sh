#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="feature-3dgs"
CONDA_BIN="/home/songliyu/.vocal/miniforge3/bin/conda"

DATASET_NAME=DJI_0544_0.5
ITERATIONS=7000

# ==============================
# Step 0: 降分辨率（已完成，先注释）
# ==============================
# "${CONDA_BIN}" run -n "${CONDA_ENV}" python scripts/openclaw_downscale_dataset.py \
#   --src "$HOME/Templates/DJI-Mini3-Pro/20260208/102MEDIA/DJI_0544/images" \
#   --dst "data/DJI_0544_0.5/images" \
#   --scale 0.5 \
#   --overwrite

# ==============================
# Step 1: 训练流程（使用降分辨率后的数据集）
# ==============================
# rm -rf submodules/diff-gaussian-rasterization-feature/build
# "${CONDA_BIN}" run -n "${CONDA_ENV}" python -m pip install --no-build-isolation submodules/diff-gaussian-rasterization-feature
"${CONDA_BIN}" run -n "${CONDA_ENV}" python train.py -s "data/${DATASET_NAME}" -m "output/${DATASET_NAME}--lseg-${ITERATIONS}" -f lseg --speedup --iterations "${ITERATIONS}"

# "${CONDA_BIN}" run -n "${CONDA_ENV}" python render.py -s "data/${DATASET_NAME}" -m "output/${DATASET_NAME}--lseg-30000" -f lseg --iteration 30000 --skip_test --novel_view
# cd encoders/lseg_encoder
# "${CONDA_BIN}" run -n "${CONDA_ENV}" python -u segmentation.py --data "../../output/${DATASET_NAME}--lseg-30000/" --iteration 30000
# cd ../..
# "${CONDA_BIN}" run -n "${CONDA_ENV}" python videos.py --data "output/${DATASET_NAME}--lseg-30000" --fps 10 -f lseg --iteration 30000
