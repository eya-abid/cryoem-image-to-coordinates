# Scientific Scope and Claim Boundaries

## What the repository tests

The central computational question is whether particle-image information can be mapped to an ordered molecular coordinate representation. Three systems separate technical feasibility from experimental ambiguity.

| System | Images | Coordinate targets | Evidential role |
|---|---|---|---|
| AK | Synthetic | Known synthetic coordinates | Controlled benchmark |
| Synthetic HER2 | Synthetic | Known synthetic C-alpha coordinates | Scale and architecture benchmark |
| Experimental HER2 | Raw experimental particles | MDSPACE-derived C-alpha surrogate coordinates | Main experimental challenge |

## What experimental HER2 metrics mean

Experimental HER2 RMSD measures disagreement between a prediction and its associated MDSPACE-derived surrogate coordinate target. It is not RMSD to a directly observed per-particle atomic structure. The upstream target system was inferred globally, and complete split independence of upstream MDSPACE fitting cannot be independently audited from the delivered artifacts.

Residual rigid pose in the delivered coordinate trajectory also cannot be excluded. Raw-frame RMSD is therefore reported as supervised-frame agreement, while rigidly aligned RMSD is a secondary diagnostic separating pose from residual coordinate disagreement.

## Why RMSD is insufficient

A model can reduce global RMSD by predicting near the population center. This may produce moderate coordinate error while failing to recover which region of the target library is paired with each particle. Consequently:

- raw and aligned RMSD test coordinate disagreement;
- target-space PCA and spread ratios test compression or expansion;
- Mahalanobis distance tests whether predictions occupy a target-like region;
- paired-target PCA distance, rank, and top-k recovery test assignment specificity.

PCA is diagnostic rather than primary accuracy evidence. Cleaned or trimmed PCA views must not replace full-set coordinate statistics.

## Supported conclusion

Direct image-to-coordinate prediction is technically feasible and promising in controlled synthetic systems. Against experimental HER2 surrogate targets, the retained direct branch improves coordinate RMSD relative to held-out population-center controls, but assignment-aware metrics remain weak. Full particle-specific conformational assignment is not established.
