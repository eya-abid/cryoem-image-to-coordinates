# Cryo-EM Image-to-Coordinate Learning

Research code and public reproducibility material for image-to-coordinate prediction in three regimes:

1. **Adenylate kinase (AK):** controlled synthetic images with known coordinate targets.
2. **Synthetic HER2:** controlled synthetic projections with known 1,489-position C-alpha targets.
3. **Experimental HER2:** raw particle images evaluated against MDSPACE-derived 1,489-position C-alpha **surrogate targets**.

The repository supports coordinate regression, raw and rigidly aligned RMSD, target-space PCA diagnostics, and assignment-aware paired-target rank/top-k recovery. It does **not** claim that experimental HER2 predictions are directly observed per-particle atomic structures.

## Scientific Result

The retained experimental raw-particle 512D direct branch reached a mean surrogate-target RMSD of approximately **4.3088 A** on 9,875 physical test particles. Its target-space spread remained compressed and paired-target retrieval was weak. The supported conclusion is therefore narrower than structure recovery: direct image-to-coordinate regression is feasible against a globally inferred surrogate system, but particle-specific conformational assignment remains unresolved.

See [Scientific scope](docs/scientific_scope.md) and [retained results](docs/retained_results.md) before interpreting the metrics.

## Installation

```bash
git clone https://github.com/eya-abid/cryoem-image-to-coordinates.git
cd cryoem-image-to-coordinates
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[training,dev]"
pytest
```

The lightweight evaluation API requires NumPy and scikit-learn. Model training additionally requires PyTorch, Matplotlib, Pillow, pandas, and Biopython.

## Quick Evaluation

Evaluate paired coordinate arrays stored as `(samples, atoms, 3)` or flattened `(samples, atoms*3)` NumPy files:

```bash
cryoem-coords evaluate \
  --predictions predictions.npy \
  --targets targets.npy \
  --output summary.json
```

Run target-PCA assignment diagnostics:

```bash
cryoem-coords assignment \
  --predictions predictions.npy \
  --targets targets.npy \
  --components 10 \
  --output assignment.json
```

Inspect whether an image array is already normalized:

```bash
cryoem-coords inspect-images --images images.npy
```

## Training Scripts

The result-matching monolithic training programs are retained under `scripts/retained/`:

- `train_ak_direct_fullatom.py`
- `train_her2_synthetic_direct.py`
- `train_her2_experimental_direct.py`

They intentionally accept explicit data and output paths. Portable command templates are in `examples/`. The retained experimental baseline used the latter script with `--pca_score_weight 0.0`; despite the historical source filename, it was a coordinate-loss baseline rather than a PCA-score model.

## Repository Map

```text
src/cryoem_image_to_coordinates/  reusable preprocessing and evaluation API
scripts/retained/                 result-matching training implementations
examples/                         portable command templates
configs/                          sanitized retained-run configuration
results/                          public aggregate metric tables
docs/                             scientific scope and reproducibility guidance
tests/                            unit tests
```

## Data Availability

Raw particles, coordinate arrays, checkpoints, and controlled third-party data are not committed. Stable artifact requirements and data boundaries are described in [Data availability](docs/data_availability.md). The public reproducibility record is archived at [Zenodo DOI 10.5281/zenodo.21216716](https://doi.org/10.5281/zenodo.21216716).

## Citation and License

Citation metadata are provided in `CITATION.cff`. Code is released under the MIT License. Public result tables retain their scientific provenance through the cited Zenodo record.
