# Visualization Assets

Each system directory contains topology metadata and a training-target mean coordinate
array used for PDB export and interpolation. These are derived research artifacts, not
raw particle data.

- AK and synthetic HER2 means derive from their controlled synthetic training targets.
- Experimental HER2 means derive from MDSPACE-derived C-alpha surrogate training
  targets and are not directly observed experimental structures.
- Experimental direct and staged branches retain separate means because their delivered
  coordinate frames differ.

Held-out particles and paired test targets are intentionally excluded from GitHub.
