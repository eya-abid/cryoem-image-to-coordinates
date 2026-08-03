from pathlib import Path

import numpy as np

from cryoem_image_to_coordinates.pdbio import read_pdb_coordinates, write_ca_pdb


def test_pdb_roundtrip(tmp_path: Path) -> None:
    coordinates = np.array([[1.25, 2.5, -3.75], [4.0, 5.0, 6.0]], dtype=np.float32)
    path = tmp_path / "coordinates.pdb"
    write_ca_pdb(coordinates, path)
    loaded = read_pdb_coordinates(path)
    assert np.allclose(loaded, coordinates, atol=0.001)
