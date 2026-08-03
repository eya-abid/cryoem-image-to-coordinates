# Contributing

Contributions should preserve the distinction between known synthetic targets and experimental MDSPACE-derived surrogate targets.

1. Create a focused branch.
2. Add or update tests for behavioral changes.
3. Run `pytest` and `ruff check .`.
4. Record seeds, split sizes, checkpoint selection, input normalization, and output paths for new experiments.
5. Save metrics as CSV or JSON and avoid committing large arrays or checkpoints.

Claims based on experimental HER2 must report assignment-aware diagnostics when discussing particle specificity. Cleaned or trimmed PCA visualizations cannot replace raw coordinate statistics.
