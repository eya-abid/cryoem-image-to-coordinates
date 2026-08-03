# Reproducibility Guide

## Environment

The code is tested with Python 3.11. The historical workstation environment recorded during final consolidation used PyTorch 2.5.1 with CUDA 12.4, NumPy 2.3.3, scikit-learn 1.6.1, pandas 2.2.3, Matplotlib 3.10.0, Pillow 11.1.0, and Biopython 1.85. This inventory is not proof that every historical run used those exact package builds.

Create a fresh environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[training,dev]"
pytest
```

## Retained experimental direct branch

The retained raw-particle direct model used:

- image array shape: `N x 128 x 128`;
- saved input preprocessing: z-score-normalized experimental particles;
- additional model-side normalization: none;
- residual convolutional encoder;
- 512-dimensional bottleneck;
- four-layer coordinate MLP with hidden width 2,048;
- output: `1,489 x 3` ordered C-alpha coordinates;
- Huber coordinate loss with beta 5.0;
- AdamW, learning rate `1e-5`, minimum learning rate `1e-6`;
- weight decay `1e-4`, batch size 8, gradient clip norm 1.0;
- seed 42;
- train/validation/test physical-particle counts: 79,923 / 10,202 / 9,875.

The historical implementation is `scripts/retained/train_her2_experimental_direct.py`. Its command syntax is preserved. The retained baseline used `--pca_score_weight 0.0`.

## Evaluation sequence

1. Run no-alignment inference and save `test_pred_coords.npy`.
2. Compute raw-frame RMSD on all physical test particles.
3. Compute aligned RMSD as a secondary pose diagnostic.
4. Fit target PCA on the declared target reference population.
5. Report prediction spread, paired target-PCA distance, target rank, and top-k recovery.
6. Compare with the training-target mean predictive baseline and a fixed-seed permutation distribution.

Do not silently substitute the test-target population mean for the held-out training-target mean. The former is a transductive population-center diagnostic; the latter is the predictive baseline.

## Determinism

The retained scripts seed Python, NumPy, and PyTorch. GPU kernels can still introduce small nondeterminism. Report the checkpoint selected, device, runtime, dataset sizes, and output paths with every run. Numeric drift must not be used to promote a claim unless it exceeds expected reproducibility variation and survives assignment-aware evaluation.
