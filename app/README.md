# Conformational Coordinate Lab

This research demonstrator applies the retained direct or staged image-to-coordinate
models to one 128 x 128 cryo-EM particle image and visualizes a linear interpolation
from the training-target mean to the predicted coordinate array.

The interpolation is a visualization of coordinate displacement. It is not molecular
dynamics, a normal mode, or an experimentally observed trajectory.

## Scientific Boundary

| System | Input normalization | Output | Target status |
|---|---|---|---|
| AK | per-image min-max scaling | 1,656 x 3 full-atom coordinates | known synthetic target |
| Synthetic HER2 | per-image min-max scaling | 1,489 x 3 C-alpha coordinates | known synthetic target |
| Experimental HER2 | p1-p99 clipping and per-image z-score | 1,489 x 3 C-alpha coordinates | MDSPACE-derived surrogate target |

Experimental HER2 output is a prediction within a globally inferred surrogate
coordinate system. It is not a directly observed per-particle atomic structure, and
the retained assignment-aware metrics do not establish particle-specific recovery.

## Installation

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[app]"
cd app
npm ci
./vendor_assets.sh
```

Install the retained model weights using [CHECKPOINTS.md](CHECKPOINTS.md). Checkpoints
are deliberately excluded from Git because the nine files total approximately 552 MiB
and their public redistribution must be approved separately.

## Desktop Interface

```bash
cd app
./run_desktop.sh
```

A graphical session is required. For remote machines, use trusted X11 forwarding or
a VNC/remote-desktop session. The desktop interface supports both `Direct` and
`Staged` methods.

## Browser Interface

```bash
cd app
./run_web.sh --host 127.0.0.1 --port 8765
```

Open <http://127.0.0.1:8765>. Binding to a public interface is discouraged unless
authentication and transport security are configured. Uploaded images are processed
in memory and are not persisted by the server.

## Input Boundary

- Accepted formats: SPI, PNG, TIFF, JPEG, or a two-dimensional NPY array.
- Expected input: one centered particle box from the selected system's image domain.
- Inputs not already 128 x 128 are resized for synthetic systems and center-cropped or
  zero-padded for experimental HER2; the response reports the operation.
- Uploaded particles have no paired target, so the application cannot report their
  prediction accuracy.
- Selecting the wrong system applies the wrong normalization, model, topology, and
  target interpretation. Cross-system inference is unsupported.

## Public Assets

The repository includes topology metadata and training-target means required for PDB
export and movement visualization. It excludes held-out particles, raw experimental
images, test coordinate arrays, and checkpoints. Optional example archives can be
placed at `app/assets/<system>/held_out_examples.npz`; their expected schema is
documented in [CHECKPOINTS.md](CHECKPOINTS.md).
