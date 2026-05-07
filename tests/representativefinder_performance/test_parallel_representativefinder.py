from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from saxshell.bondanalysis import AngleTripletDefinition, BondPairDefinition
from saxshell.representativefinder import (
    RepresentativeFinderSettings,
    analyze_representative_structure_folder,
)

REFERENCE_PB2I4_DIR = Path(
    "/Users/keithwhite/repos/cluster_extraction/"
    "041_cp2k_pbi2_dmf_0p7M_RT/"
    "clusters_xyz2pdb_splitxyz_f1002_t497p5fs0001/Pb2I4"
)


def _write_xyz_structure(
    path: Path,
    atoms: list[tuple[str, float, float, float]],
) -> None:
    lines = [str(len(atoms)), path.stem]
    for element, x_coord, y_coord, z_coord in atoms:
        lines.append(f"{element} {x_coord:.3f} {y_coord:.3f} {z_coord:.3f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_representative_sample_folder(tmp_path: Path) -> Path:
    stoich_dir = tmp_path / "PbI2"
    stoich_dir.mkdir(parents=True)
    for index, distance in enumerate(
        (1.95, 2.05, 2.15, 2.25, 2.35, 2.45, 2.55, 2.65),
        start=1,
    ):
        _write_xyz_structure(
            stoich_dir / f"candidate_{index:02d}.xyz",
            [
                ("Pb", 0.0, 0.0, 0.0),
                ("I", distance, 0.0, 0.0),
                ("I", 0.0, distance, 0.0),
                ("O", 0.0, 0.0, distance + 0.35),
                ("O", 0.0, 0.0, distance + 2.7),
            ],
        )
    return stoich_dir


def _settings(worker_count: int) -> RepresentativeFinderSettings:
    return RepresentativeFinderSettings(
        bond_pairs=(BondPairDefinition("Pb", "I", 4.2),),
        angle_triplets=(AngleTripletDefinition("Pb", "I", "I", 4.2, 4.2),),
        bond_weight=1.0,
        angle_weight=0.5,
        solvent_weight=0.5,
        parallel_workers=worker_count,
    )


def _link_reference_sample(
    reference_dir: Path,
    sample_dir: Path,
    *,
    sample_count: int,
) -> None:
    sample_dir.mkdir(parents=True)
    for source_path in sorted(reference_dir.glob("*.pdb"))[:sample_count]:
        target_path = sample_dir / source_path.name
        try:
            target_path.symlink_to(source_path)
        except OSError:
            shutil.copy2(source_path, target_path)


def test_parallel_representativefinder_matches_serial_result(tmp_path):
    stoich_dir = _build_representative_sample_folder(tmp_path)

    serial = analyze_representative_structure_folder(
        stoich_dir,
        settings=_settings(1),
        output_dir=tmp_path / "serial_output",
    )
    parallel = analyze_representative_structure_folder(
        stoich_dir,
        settings=_settings(4),
        output_dir=tmp_path / "parallel_output",
    )

    assert parallel.selected_candidate.file_name == (
        serial.selected_candidate.file_name
    )
    assert len(parallel.candidates) == len(serial.candidates)
    serial_scores = {
        candidate.relative_label: candidate.score_total
        for candidate in serial.candidates
    }
    for candidate in parallel.candidates:
        assert candidate.score_total == pytest.approx(
            serial_scores[candidate.relative_label],
            rel=0.0,
            abs=1.0e-12,
        )
    assert parallel.summary_json_path.is_file()
    assert parallel.score_table_path.is_file()


@pytest.mark.skipif(
    not REFERENCE_PB2I4_DIR.is_dir(),
    reason="Pb2I4 representative benchmark reference folder is unavailable.",
)
def test_reference_pb2i4_sample_matches_serial_and_parallel(tmp_path):
    sample_dir = tmp_path / "Pb2I4"
    _link_reference_sample(
        REFERENCE_PB2I4_DIR,
        sample_dir,
        sample_count=32,
    )

    serial = analyze_representative_structure_folder(
        sample_dir,
        settings=_settings(1),
        output_dir=tmp_path / "reference_serial_output",
    )
    parallel = analyze_representative_structure_folder(
        sample_dir,
        settings=_settings(4),
        output_dir=tmp_path / "reference_parallel_output",
    )

    assert len(serial.candidates) == 32
    assert len(parallel.candidates) == 32
    assert parallel.selected_candidate.file_name == (
        serial.selected_candidate.file_name
    )
    assert parallel.selected_candidate.score_total == pytest.approx(
        serial.selected_candidate.score_total,
        rel=0.0,
        abs=1.0e-12,
    )
