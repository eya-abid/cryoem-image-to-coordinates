#!/usr/bin/env bash
set -euo pipefail

: "${DATA_ROOT:?Set DATA_ROOT}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT}"
PYTHON="${PYTHON:-python}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

"${PYTHON}" "${ROOT}/scripts/retained/train_ak_direct_fullatom.py" \
  --mode train \
  --spi_dir "${DATA_ROOT}/ak/images" \
  --spi_glob '*_projected_normalized.spi' \
  --pdb_dir "${DATA_ROOT}/ak/targets" \
  --pdb_glob '*_rotshift.pdb' \
  --splits_path "${DATA_ROOT}/ak/splits.json" \
  --out_dir "${OUTPUT_ROOT}/ak_direct_fullatom" \
  --max_atoms 1656 \
  --atom_mode all \
  --image_h 128 --image_w 128 \
  --batch_size 8 --epochs 80 \
  --lr 2e-4 --lr_min 1e-5 \
  --hidden_dim 2048 --mlp_layers 4 \
  --dropout 0.15 --weight_decay 1e-4 \
  --grad_clip 1.0 --huber_beta 5.0 --seed 42
