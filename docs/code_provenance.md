# Code Provenance

The public repository separates reusable code from result-matching historical implementations.

## Reusable API

`src/cryoem_image_to_coordinates` provides explicit preprocessing, PDB I/O, raw/aligned RMSD, and target-PCA assignment metrics. These modules are covered by unit tests and form the recommended interface for new analyses.

## Retained scripts

The programs in `scripts/retained` are sanitized copies of the self-contained sources used by retained direct-model families. Machine-local launchers are not included. Data and output paths must be supplied explicitly.

| Public script | Historical role |
|---|---|
| `train_ak_direct_fullatom.py` | AK full-atom direct image-to-coordinate branch |
| `train_her2_synthetic_direct.py` | Synthetic HER2 1,489-position C-alpha direct branch |
| `train_her2_experimental_direct.py` | Experimental HER2 raw-particle direct branch and related objective controls |

The experimental script's historical filename contained `pcascore`, but the protected baseline set the PCA-score loss weight to zero. Public naming reflects the executed objective rather than the source filename.

## Excluded code

Manuscript assembly utilities, local visualization handoff scripts, obsolete launchers, exploratory branches without retained evidence, and scripts containing fixed workstation layouts are excluded from the primary repository. Their omission prevents local infrastructure from being mistaken for the supported public API.
