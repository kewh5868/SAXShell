from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import saxshell.saxs.debye.profiles as profiles
from saxshell.saxs.debye import (
    DebyeProfileBuilder,
    compute_debye_intensity,
    discover_available_elements,
    load_structure_file,
    scan_structure_elements,
)


class FakeXrayDB:
    @staticmethod
    def f0(element: str, sin_theta_over_lambda: float) -> np.ndarray:
        base = {
            "H": 1.0,
            "O": 8.0,
            "Cl": 17.0,
            "Zn": 30.0,
        }[element]
        return np.array([base + float(sin_theta_over_lambda)], dtype=float)


def _pdb_atom_line(
    atom_id: int,
    atom_name: str,
    residue_name: str,
    residue_number: int,
    x_coord: float,
    y_coord: float,
    z_coord: float,
    *,
    element: str = "",
) -> str:
    return (
        f"ATOM  {atom_id:5d} {atom_name:<4} {residue_name:>3} A"
        f"{residue_number:4d}    {x_coord:8.3f}{y_coord:8.3f}{z_coord:8.3f}"
        f"  1.00 20.00          {element:>2}"
    )


def test_load_structure_file_preserves_legacy_pdb_element_inference(
    tmp_path: Path,
) -> None:
    pdb_path = tmp_path / "cluster.pdb"
    pdb_path.write_text(
        "\n".join(
            [
                _pdb_atom_line(1, "ZN1", "ZN", 1, 1.0, 2.0, 3.0),
                _pdb_atom_line(2, "CL1", "CL", 1, 4.0, 5.0, 6.0),
                "END",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    coordinates, elements = load_structure_file(pdb_path)

    assert elements == ["Zn", "Cl"]
    np.testing.assert_allclose(
        coordinates,
        np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=float),
    )


def test_scan_structure_elements_matches_loaded_elements(
    tmp_path: Path,
) -> None:
    xyz_path = tmp_path / "cluster.xyz"
    xyz_path.write_text(
        "3\ncomment\nO 0.0 0.0 0.0\nPb 1.0 0.0 0.0\nI 0.0 1.0 0.0\n",
        encoding="utf-8",
    )

    _coordinates, loaded_elements = load_structure_file(xyz_path)
    scanned_elements = scan_structure_elements(xyz_path)

    assert scanned_elements == loaded_elements


def test_compute_debye_intensity_uses_legacy_f0_and_exclusion_logic(
    monkeypatch,
) -> None:
    monkeypatch.setattr(profiles, "xraydb", FakeXrayDB())

    q_values = np.asarray([0.1, 0.3], dtype=float)
    coordinates = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float)
    elements = ["H", "O"]

    result = compute_debye_intensity(coordinates, elements, q_values)
    sin_theta_over_lambda = q_values / (4.0 * np.pi)
    f_h = 1.0 + sin_theta_over_lambda
    f_o = 8.0 + sin_theta_over_lambda
    expected = f_h**2 + f_o**2 + 2.0 * np.sinc(q_values / np.pi) * f_h * f_o
    np.testing.assert_allclose(result, expected)

    excluded = compute_debye_intensity(
        coordinates,
        elements,
        q_values,
        exclude_elements=["O"],
    )
    np.testing.assert_allclose(excluded, f_h**2)


def test_discover_available_elements_collects_unique_cluster_elements(
    tmp_path: Path,
) -> None:
    clusters_dir = tmp_path / "clusters"
    motif_dir = clusters_dir / "A1" / "motif_1"
    motif_dir.mkdir(parents=True)
    (motif_dir / "frame_0001.xyz").write_text(
        "3\ncomment\nC 0.0 0.0 0.0\nH 1.0 0.0 0.0\nO 0.0 1.0 0.0\n",
        encoding="utf-8",
    )
    (motif_dir / "frame_0002.xyz").write_text(
        "2\ncomment\nZn 0.0 0.0 0.0\nCl 1.0 1.0 0.0\n",
        encoding="utf-8",
    )

    assert discover_available_elements(clusters_dir) == [
        "C",
        "Cl",
        "H",
        "O",
        "Zn",
    ]


def test_build_profiles_computes_single_atom_cluster_only_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clusters_dir = tmp_path / "clusters"
    single_atom_dir = clusters_dir / "Zn1"
    single_atom_dir.mkdir(parents=True)
    for index in range(3):
        (single_atom_dir / f"frame_{index:04d}.xyz").write_text(
            "1\ncomment\nZn 0.0 0.0 0.0\n",
            encoding="utf-8",
        )

    q_values = np.asarray([0.1, 0.2, 0.3], dtype=float)
    call_counter = {"count": 0}

    def fake_build_f0_dictionary(elements, q_values):
        del q_values
        return {str(element): np.ones(3, dtype=float) for element in elements}

    def fake_compute_debye_intensity(
        coordinates,
        elements,
        q_values,
        *,
        exclude_elements=None,
        f0_dictionary=None,
    ):
        del coordinates, elements, exclude_elements, f0_dictionary
        call_counter["count"] += 1
        return np.asarray(q_values, dtype=float) + 5.0

    monkeypatch.setattr(
        profiles,
        "build_f0_dictionary",
        fake_build_f0_dictionary,
    )
    monkeypatch.setattr(
        profiles,
        "compute_debye_intensity",
        fake_compute_debye_intensity,
    )

    builder = DebyeProfileBuilder(
        q_values=q_values,
        output_dir=tmp_path / "components",
    )

    components = builder.build_profiles(clusters_dir)

    assert call_counter["count"] == 1
    assert len(components) == 1
    component = components[0]
    np.testing.assert_allclose(component.mean_intensity, q_values + 5.0)
    np.testing.assert_allclose(
        component.std_intensity, np.zeros_like(q_values)
    )
    np.testing.assert_allclose(component.se_intensity, np.zeros_like(q_values))
    assert component.file_count == 3
