"""Minimal ordered-coordinate PDB input/output helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def read_pdb_coordinates(path: str | Path, atom_name: str | None = "CA") -> np.ndarray:
    """Read ordered ATOM coordinates, optionally retaining one atom name."""
    coordinates = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.startswith("ATOM"):
            continue
        if atom_name is not None and line[12:16].strip() != atom_name:
            continue
        try:
            coordinates.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
        except ValueError as error:
            raise ValueError(f"Invalid coordinate record in {path}: {line}") from error
    if not coordinates:
        raise ValueError(f"No matching ATOM records in {path}")
    return np.asarray(coordinates, dtype=np.float32)


def write_ca_pdb(coordinates: np.ndarray, path: str | Path, chain_id: str = "A") -> None:
    """Write an ordered coordinate array as a C-alpha-only PDB."""
    array = np.asarray(coordinates, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"Expected (atoms,3), got {array.shape}")
    lines = []
    for index, (x, y, z) in enumerate(array, start=1):
        lines.append(
            f"ATOM  {index:5d}  CA  ALA {chain_id}{index:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C"
        )
    lines.extend(("TER", "END"))
    Path(path).write_text("\n".join(lines) + "\n", encoding="ascii")
