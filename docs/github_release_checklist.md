# GitHub Release Checklist

- [x] Set the GitHub repository URL in `README.md` and `CITATION.cff`.
- [ ] Confirm the chosen MIT license with the author/institution.
- [ ] Confirm DOI metadata and GitHub repository URL in `CITATION.cff`.
- [ ] Run `pytest` and `ruff check src tests`.
- [ ] Run the secret and large-file audit.
- [ ] Confirm no particle arrays, checkpoints, trajectories, or local absolute paths are tracked.
- [ ] Tag the first public release as `v1.0.0`.
- [ ] Link the GitHub release from the Zenodo record or reserve a new software DOI if required.
