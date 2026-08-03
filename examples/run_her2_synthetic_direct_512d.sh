#!/usr/bin/env bash
set -euo pipefail

: "${DATA_ROOT:?Set DATA_ROOT}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT}"
PYTHON="${PYTHON:-python}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

"${PYTHON}" "${ROOT}/scripts/retained/train_her2_synthetic_direct.py" \
  --mode train \
  --spi_dir "${DATA_ROOT}/her2_synthetic/images" \
  --pdb_dir "${DATA_ROOT}/her2_synthetic/targets" \
  --splits_path "${DATA_ROOT}/her2_synthetic/splits.json" \
  --out_dir "${OUTPUT_ROOT}/her2_synthetic_direct_512d" \
  --max_atoms 1489 \
  --image_h 128 --image_w 128 \
  --batch_size 8 --epochs 80 \
  --lr 2e-4 --lr_min 1e-5 \
  --hidden_dim 2048 --mlp_layers 4 --bottleneck_dim 512 \
  --dropout 0.15 --weight_decay 1e-4 \
  --grad_clip 1.0 --huber_beta 5.0 --seed 42
