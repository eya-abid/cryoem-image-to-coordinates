# Data Availability and Artifact Contract

## Included in GitHub

- reusable evaluation code;
- result-matching direct-model training scripts;
- sanitized retained-run configuration;
- aggregate public metric tables;
- command templates, tests, and documentation.

## Not included

- raw or synthetic particle arrays;
- PDB target collections and trajectories;
- experimental MDSPACE-derived coordinate arrays;
- model checkpoints and prediction arrays;
- Scipion project databases;
- controlled or third-party intermediate data.

These artifacts are too large, controlled, or redistribution-dependent for Git. Their expected identifiers, shapes, and roles are documented below.

| Identifier | Expected form | Role |
|---|---|---|
| `AK_SYNTHETIC_IMAGES` | SPI images, 128 x 128 | Controlled AK input |
| `AK_COORDINATE_TARGETS` | ordered PDB coordinates | Known AK targets |
| `HER2_SYNTHETIC_IMAGES` | SPI images, 128 x 128 | Controlled HER2 input |
| `HER2_SYNTHETIC_TARGETS` | 1,489 x 3 C-alpha coordinates | Known synthetic targets |
| `HER2_EXPERIMENTAL_IMAGES` | NumPy arrays, N x 128 x 128 | Raw experimental particles |
| `HER2_EXPERIMENTAL_SURROGATE_TARGETS` | NumPy arrays, N x 4,467 or N x 1,489 x 3 | MDSPACE-derived surrogate targets |

## Data checks before training

1. Verify row counts and ordered pairing for every split.
2. Confirm that train, validation, and test rows correspond to physical-particle splits.
3. Confirm that duplicated raw/teacher-reconstructed views remain in the same split.
4. Record whether saved images are raw, min-max scaled, or z-score normalized.
5. Confirm target atom count and ordering against the reference PDB.
6. Save a manifest and cryptographic checksums outside the run directory.

The archived public companion is available at [DOI 10.5281/zenodo.21216716](https://doi.org/10.5281/zenodo.21216716).
