#!/usr/bin/env bash
set -euo pipefail

: "${DATA_ROOT:?Set DATA_ROOT}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT}"
PYTHON="${PYTHON:-python}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW="${DATA_ROOT}/her2_experimental/images"
TARGETS="${DATA_ROOT}/her2_experimental/mdspace_surrogate_targets"
OUT="${OUTPUT_ROOT}/her2_experimental_raw_direct_512d"

"${PYTHON}" "${ROOT}/scripts/retained/train_her2_experimental_direct.py" train \
  --x_train "${RAW}/train.npy" \
  --x_val "${RAW}/validation.npy" \
  --x_test "${RAW}/test.npy" \
  --coords_train "${TARGETS}/train.npy" \
  --coords_val "${TARGETS}/validation.npy" \
  --coords_test "${TARGETS}/test.npy" \
  --out_dir "${OUT}" \
  --max_atoms 1489 \
  --batch_size 8 --epochs 40 \
  --lr 1e-5 --lr_min 1e-6 \
  --hidden_dim 2048 --mlp_layers 4 --bottleneck_dim 512 \
  --dropout 0.15 --weight_decay 1e-4 \
  --grad_clip 1.0 --huber_beta 5.0 \
  --pca_score_weight 0.0 \
  --min_epochs 5 --early_stop_patience 8 --early_stop_min_delta 1e-4 \
  --seed 42

"${PYTHON}" "${ROOT}/scripts/retained/train_her2_experimental_direct.py" infer \
  --x_path "${RAW}/test.npy" \
  --coords_path "${TARGETS}/test.npy" \
  --out_dir "${OUT}" \
  --checkpoint "${OUT}/cnn_best.pt" \
  --split_name test \
  --max_atoms 1489 --batch_size 16 \
  --hidden_dim 2048 --mlp_layers 4 --bottleneck_dim 512 \
  --dropout 0.15 --save_pred_coords --seed 42
