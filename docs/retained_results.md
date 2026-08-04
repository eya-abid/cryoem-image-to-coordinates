# Retained Public Results

## Experimental HER2 protected baseline

The primary retained experimental result is the raw-particle direct 512D branch evaluated on 9,875 physical test particles against MDSPACE-derived C-alpha surrogate targets.

| Metric | Value |
|---|---:|
| Mean surrogate-target RMSD | 4.3088 A |
| Median surrogate-target RMSD | 4.0485 A |
| p95 surrogate-target RMSD | 7.4241 A |
| Target-PC3 spread ratio | 0.4818 |
| Paired PC10 mean distance | 2.9088 |
| Median paired-target rank | 3,826 |
| Top-10 recovery | 0.273% |
| Top-100 recovery | 2.329% |

The coordinate model improves global surrogate-target RMSD relative to constant-target controls, but its rank and recovery rates do not establish particle-specific assignment.

The held-out training-target mean baseline has mean RMSD 4.8764 A (bootstrap 95% CI 4.8383-4.9147 A). The raw-direct model is lower for 66.94% of test particles (95% CI 66.02-67.85%) and improves mean RMSD by 0.5676 A (95% CI 0.5428-0.5930 A). A 1,000-derangement fixed-seed random-target distribution has mean RMSD 6.9176 A, standard deviation 0.0176 A, and a 95% interval of 6.8844-6.9507 A.

## Baseline terminology

- **Training-target mean:** a fixed coordinate mean calculated from training physical-particle targets and applied to held-out test particles. This is the predictive constant-target baseline.
- **Test-target mean:** a coordinate mean calculated from the test targets themselves. This is a transductive population-center diagnostic and is reported separately.
- **Random-target control:** a fixed-seed distribution of target permutations. It is not a trained predictor.

## Public tables

- `results/experimental_her2_canonical_stress_table.csv`
- `results/assignment_baselines_summary.csv`
- `results/raw100k_capacity_and_finetune_summary.csv`
- `results/experimental_baseline_reanalysis.csv`

The tables contain aggregate metrics only. Repository checksums and source-provenance
records identify the published files; no archival DOI is currently registered.
