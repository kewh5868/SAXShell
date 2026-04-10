from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from matplotlib.backend_bases import MouseButton
from PySide6.QtCore import QItemSelectionModel, Qt
from PySide6.QtGui import QColor
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox, QScrollArea

import saxshell.saxs.electron_density_mapping.workflow as density_workflow
from saxshell.saxs.contrast.electron_density import (
    CONTRAST_SOLVENT_METHOD_DIRECT,
    ContrastSolventDensitySettings,
)
from saxshell.saxs.debye import atomic_form_factor
from saxshell.saxs.electron_density_mapping.ui.main_window import (
    ElectronDensityMappingMainWindow,
    _DebyeComparisonEntry,
    _DebyeScatteringComparisonDialog,
    launch_electron_density_mapping_ui,
)
from saxshell.saxs.electron_density_mapping.ui.viewer import (
    ElectronDensityFourierPreviewPlot,
)
from saxshell.saxs.electron_density_mapping.workflow import (
    ElectronDensityCalculationCanceled,
    ElectronDensityFourierTransformSettings,
    ElectronDensityMeshSettings,
    ElectronDensitySmearingSettings,
    apply_smearing_to_profile_result,
    apply_solvent_contrast_to_profile_result,
    compute_average_debye_scattering_profile_for_input,
    compute_electron_density_profile,
    compute_electron_density_profile_for_input,
    compute_electron_density_scattering_profile,
    compute_single_atom_debye_scattering_profile_for_input,
    inspect_structure_input,
    load_electron_density_structure,
    prepare_electron_density_fourier_transform,
    write_electron_density_profile_outputs,
)
from saxshell.saxs.ui.main_window import AUTO_SNAP_PANES_KEY

xraydb = pytest.importorskip("xraydb")


@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture(autouse=True)
def _accept_default_questions(monkeypatch):
    monkeypatch.setattr(
        "saxshell.saxs.electron_density_mapping.ui.main_window.QMessageBox.question",
        lambda *args, **kwargs: QMessageBox.StandardButton.Yes,
    )


def _write_xyz(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _pdb_atom_line(
    atom_id: int,
    atom_name: str,
    residue_name: str,
    residue_number: int,
    x_coord: float,
    y_coord: float,
    z_coord: float,
    element: str,
) -> str:
    return (
        f"ATOM  {atom_id:5d} {atom_name:<4} {residue_name:>3}  "
        f"{residue_number:4d}    "
        f"{x_coord:8.3f}{y_coord:8.3f}{z_coord:8.3f}"
        f"  1.00  0.00          {element:>2}\n"
    )


def _write_water_debye_waller_frame(
    path: Path,
    *,
    mol1_oxygen_x: float,
    mol1_hydrogen_x: float,
    mol2_oxygen_x: float,
    mol2_hydrogen_x: float,
) -> None:
    path.write_text(
        "".join(
            [
                _pdb_atom_line(
                    1,
                    "O",
                    "HOH",
                    1,
                    mol1_oxygen_x,
                    0.0,
                    0.0,
                    "O",
                ),
                _pdb_atom_line(
                    2,
                    "H1",
                    "HOH",
                    1,
                    mol1_hydrogen_x,
                    0.0,
                    0.0,
                    "H",
                ),
                _pdb_atom_line(
                    3,
                    "O",
                    "HOH",
                    2,
                    mol2_oxygen_x,
                    0.0,
                    0.0,
                    "O",
                ),
                _pdb_atom_line(
                    4,
                    "H1",
                    "HOH",
                    2,
                    mol2_hydrogen_x,
                    0.0,
                    0.0,
                    "H",
                ),
                "END\n",
            ]
        ),
        encoding="utf-8",
    )


def _write_pb_o_pdb_frame(
    path: Path,
    *,
    pb_x: float,
    o_x: float,
) -> None:
    path.write_text(
        "".join(
            [
                _pdb_atom_line(1, "PB", "CLS", 1, pb_x, 0.0, 0.0, "Pb"),
                _pdb_atom_line(2, "O1", "CLS", 1, o_x, 0.0, 0.0, "O"),
                "END\n",
            ]
        ),
        encoding="utf-8",
    )


def _write_project_debye_waller_analysis(
    project_dir: Path,
    tmp_path: Path,
) -> Path:
    from saxshell.saxs.debye_waller.workflow import (
        DebyeWallerWorkflow,
        save_debye_waller_analysis_to_project,
    )

    clusters_dir = tmp_path / "project_dw_clusters"
    structure_dir = clusters_dir / "H2O2"
    structure_dir.mkdir(parents=True)
    _write_water_debye_waller_frame(
        structure_dir / "water_frame_0001.pdb",
        mol1_oxygen_x=0.0,
        mol1_hydrogen_x=1.00,
        mol2_oxygen_x=5.0,
        mol2_hydrogen_x=6.00,
    )
    _write_water_debye_waller_frame(
        structure_dir / "water_frame_0002.pdb",
        mol1_oxygen_x=0.0,
        mol1_hydrogen_x=1.10,
        mol2_oxygen_x=5.2,
        mol2_hydrogen_x=6.35,
    )
    _write_water_debye_waller_frame(
        structure_dir / "water_frame_0004.pdb",
        mol1_oxygen_x=0.0,
        mol1_hydrogen_x=0.95,
        mol2_oxygen_x=4.9,
        mol2_hydrogen_x=6.05,
    )
    _write_water_debye_waller_frame(
        structure_dir / "water_frame_0005.pdb",
        mol1_oxygen_x=0.0,
        mol1_hydrogen_x=1.05,
        mol2_oxygen_x=5.1,
        mol2_hydrogen_x=6.30,
    )
    result = DebyeWallerWorkflow(
        clusters_dir,
        project_dir=project_dir,
        output_dir=tmp_path / "project_dw_out",
        output_basename="electron_density_smearing",
    ).run()
    saved_result = save_debye_waller_analysis_to_project(result, project_dir)
    assert saved_result.artifacts is not None
    return saved_result.artifacts.summary_json_path.resolve()


def _mesh_settings_for_structure(
    structure,
    **overrides,
) -> ElectronDensityMeshSettings:
    payload = {
        "rstep": 0.1,
        "theta_divisions": 120,
        "phi_divisions": 60,
        "rmax": float(structure.rmax),
    }
    payload.update(overrides)
    return ElectronDensityMeshSettings(**payload)


def _baseline_shell_electron_counts(
    structure,
    mesh_settings: ElectronDensityMeshSettings,
) -> np.ndarray:
    mesh_geometry = density_workflow.build_electron_density_mesh(
        structure,
        mesh_settings,
    )
    radial_edges = np.asarray(mesh_geometry.radial_edges, dtype=float)
    shell_count = int(max(len(radial_edges) - 1, 0))
    if shell_count <= 0:
        return np.asarray([], dtype=float)

    radial_distances = np.linalg.norm(
        np.asarray(structure.centered_coordinates, dtype=float),
        axis=1,
    )
    in_domain_mask = radial_distances <= (
        float(mesh_geometry.domain_max_radius) + 1.0e-12
    )
    radial_indices = (
        np.searchsorted(
            radial_edges,
            radial_distances[in_domain_mask],
            side="right",
        )
        - 1
    )
    radial_indices = np.clip(radial_indices, 0, shell_count - 1)
    atomic_numbers = np.asarray(structure.atomic_numbers, dtype=float)[
        in_domain_mask
    ]
    return np.bincount(
        radial_indices,
        weights=atomic_numbers,
        minlength=shell_count,
    ).astype(float, copy=False)


def _profile_result_with_shell_counts(
    result,
    shell_counts: np.ndarray,
):
    shell_count_array = np.asarray(shell_counts, dtype=float)
    density_profile = np.divide(
        shell_count_array,
        np.asarray(result.shell_volumes, dtype=float),
        out=np.zeros_like(shell_count_array, dtype=float),
        where=np.asarray(result.shell_volumes, dtype=float) > 0.0,
    )
    zeros = np.zeros_like(density_profile, dtype=float)
    baseline_result = replace(
        result,
        member_orientation_average_densities=(
            np.asarray(density_profile, dtype=float).copy(),
        ),
        orientation_average_density=np.asarray(density_profile, dtype=float),
        orientation_density_variance=zeros.copy(),
        orientation_density_stddev=zeros.copy(),
        smeared_orientation_average_density=np.asarray(
            density_profile,
            dtype=float,
        ),
        smeared_orientation_density_variance=zeros.copy(),
        smeared_orientation_density_stddev=zeros.copy(),
        shell_volume_average_density=np.asarray(density_profile, dtype=float),
        shell_electron_counts=shell_count_array,
    )
    return apply_smearing_to_profile_result(
        baseline_result,
        result.smearing_settings,
    )


def _assert_shell_electron_total(
    result,
    expected_total: float,
) -> None:
    assert float(np.sum(result.shell_electron_counts)) == pytest.approx(
        float(expected_total)
    )


def _assert_structure_shell_electron_total(result, structure) -> None:
    _assert_shell_electron_total(
        result,
        float(np.sum(structure.atomic_numbers)),
    )


def _assert_density_profile_electron_total(
    result,
    expected_total: float,
) -> None:
    assert float(
        np.sum(
            np.asarray(result.orientation_average_density, dtype=float)
            * np.asarray(result.shell_volumes, dtype=float)
        )
    ) == pytest.approx(float(expected_total))


def _wait_for(condition, qapp, *, timeout_s: float = 10.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        qapp.processEvents()
        if condition():
            return
        time.sleep(0.01)
    raise AssertionError("Timed out waiting for condition.")


def _pane_snap_resize_result(
    splitter,
    pane_snap_filter,
    *,
    target_index: int,
    desired_width: int | None = None,
    other_width: int = 220,
) -> tuple[list[int], list[int]]:
    pane_widget = splitter.widget(target_index)
    assert pane_widget is not None
    click_widget = pane_widget
    if isinstance(pane_widget, QScrollArea):
        click_widget = pane_widget.viewport()
    current_sizes = splitter.sizes()
    current_total = sum(size for size in current_sizes if size > 0)
    assert current_total > 0
    resolved_width = (
        max(720, int(current_total * 0.7))
        if desired_width is None
        else int(desired_width)
    )
    if target_index == 0:
        splitter.setSizes([250, max(current_total - 250, 1)])
    else:
        splitter.setSizes([max(current_total - 250, 1), 250])
    QApplication.processEvents()
    before_sizes = splitter.sizes()

    pane_snap_filter._preferred_width = lambda widget: (
        resolved_width
        if widget is splitter.widget(target_index)
        else int(other_width)
    )
    QTest.mouseClick(
        click_widget,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
        click_widget.rect().center(),
    )
    QApplication.processEvents()
    return before_sizes, splitter.sizes()


def _write_cluster_folder_input(base_dir: Path) -> Path:
    first_dir = base_dir / "PbI2"
    second_dir = base_dir / "PbI3"
    first_dir.mkdir(parents=True)
    second_dir.mkdir(parents=True)
    _write_xyz(
        first_dir / "frame_0001_AAA.xyz",
        [
            "3",
            "PbI2 frame 1",
            "Pb 0.0 0.0 0.0",
            "I 2.8 0.0 0.0",
            "I 0.0 2.8 0.0",
        ],
    )
    _write_xyz(
        second_dir / "frame_0001_AAA.xyz",
        [
            "4",
            "PbI3 frame 1",
            "Pb 0.0 0.0 0.0",
            "I 2.8 0.0 0.0",
            "I 0.0 2.8 0.0",
            "I 0.0 0.0 2.8",
        ],
    )
    return base_dir


def _open_ready_cluster_folder_window(
    qapp,
    tmp_path: Path,
    *,
    folder_name: str,
) -> ElectronDensityMappingMainWindow:
    window = ElectronDensityMappingMainWindow(
        initial_input_path=_write_cluster_folder_input(tmp_path / folder_name)
    )
    _wait_for(
        lambda: window.cluster_group_table.rowCount() == 2,
        qapp,
    )
    window.run_button.click()
    _wait_for(
        lambda: all(
            state.profile_result is not None
            for state in window._cluster_group_states
        )
        and window._calculation_thread is None,
        qapp,
    )
    return window


def _assert_batch_progress_dialog_visible(
    window: ElectronDensityMappingMainWindow,
    *,
    title: str,
    total: int,
):
    dialog = window._batch_operation_progress_dialog
    assert dialog is not None
    assert dialog.isVisible()
    assert dialog.windowTitle() == title
    assert dialog.progress_bar.maximum() == total
    return dialog


def _table_column_index(table, header_text: str) -> int:
    for column_index in range(table.columnCount()):
        header_item = table.horizontalHeaderItem(column_index)
        if header_item is not None and header_item.text() == header_text:
            return int(column_index)
    raise AssertionError(f"Could not find table column {header_text!r}.")


def test_inspect_structure_input_handles_single_files_and_folders(tmp_path):
    folder = tmp_path / "frames"
    folder.mkdir()
    _write_xyz(
        folder / "frame_0002.xyz",
        [
            "2",
            "frame_0002",
            "C 0.0 0.0 0.0",
            "O 1.2 0.0 0.0",
        ],
    )
    _write_xyz(
        folder / "frame_0001.xyz",
        [
            "2",
            "frame_0001",
            "C 0.0 0.0 0.0",
            "O 1.1 0.0 0.0",
        ],
    )
    (folder / "notes.txt").write_text("ignore me\n", encoding="utf-8")

    folder_inspection = inspect_structure_input(folder)
    file_inspection = inspect_structure_input(folder / "frame_0002.xyz")

    assert folder_inspection.input_mode == "folder"
    assert folder_inspection.total_files == 2
    assert folder_inspection.reference_file.name == "frame_0001.xyz"
    assert file_inspection.input_mode == "file"
    assert file_inspection.reference_file.name == "frame_0002.xyz"


def test_load_electron_density_structure_uses_mass_weighted_center_of_mass(
    tmp_path,
):
    structure_path = _write_xyz(
        tmp_path / "pb_h.xyz",
        [
            "2",
            "Pb-H test",
            "H 0.0 0.0 0.0",
            "Pb 2.0 0.0 0.0",
        ],
    )

    structure = load_electron_density_structure(
        structure_path,
        center_mode="reference_element",
        reference_element="O",
    )
    expected_x = (
        float(xraydb.atomic_mass("Pb")) * 2.0
        + float(xraydb.atomic_mass("H")) * 0.0
    ) / (float(xraydb.atomic_mass("Pb")) + float(xraydb.atomic_mass("H")))

    assert structure.center_of_mass[0] == pytest.approx(expected_x)
    assert structure.center_of_mass[0] != pytest.approx(1.0)
    assert structure.reference_element == "Pb"
    assert np.allclose(structure.active_center, [2.0, 0.0, 0.0])
    assert structure.center_mode == "reference_element"
    assert structure.rmax > 0.0
    assert structure.bonds


def test_load_electron_density_structure_tracks_reference_element_centers(
    tmp_path,
):
    structure_path = _write_xyz(
        tmp_path / "reference_centers.xyz",
        [
            "4",
            "reference centers",
            "Pb 0.0 0.0 0.0",
            "Pb 2.0 0.0 0.0",
            "H 0.0 2.0 0.0",
            "O 0.0 0.0 2.0",
        ],
    )

    structure = load_electron_density_structure(
        structure_path,
        center_mode="reference_element",
        reference_element="O",
    )
    reference_centered = load_electron_density_structure(
        structure_path,
        center_mode="reference_element",
    )

    assert structure.reference_element == "O"
    assert np.allclose(structure.geometric_center, [0.5, 0.5, 0.5])
    assert np.allclose(
        structure.reference_element_geometric_center,
        [0.0, 0.0, 2.0],
    )
    assert structure.reference_element_offset_from_geometric_center == (
        pytest.approx(np.sqrt(2.75))
    )
    assert np.allclose(
        structure.active_center,
        structure.reference_element_geometric_center,
    )
    assert reference_centered.reference_element == "Pb"
    assert reference_centered.center_mode == "reference_element"
    assert np.allclose(
        reference_centered.active_center,
        [1.0, 0.0, 0.0],
    )


def test_compute_electron_density_profile_tracks_shell_electrons(tmp_path):
    structure_path = _write_xyz(
        tmp_path / "co.xyz",
        [
            "2",
            "CO",
            "C 0.0 0.0 0.0",
            "O 1.128 0.0 0.0",
        ],
    )
    structure = load_electron_density_structure(structure_path)
    result = compute_electron_density_profile(
        structure,
        _mesh_settings_for_structure(
            structure,
            rstep=0.25,
            theta_divisions=18,
            phi_divisions=12,
        ),
    )

    _assert_structure_shell_electron_total(result, structure)
    assert float(np.sum(result.shell_volumes)) == pytest.approx(
        4.0 / 3.0 * np.pi * (result.mesh_geometry.domain_max_radius**3),
    )
    assert np.all(result.orientation_average_density >= 0.0)
    assert np.all(result.smeared_orientation_average_density >= 0.0)
    assert len(result.radial_centers) == result.mesh_geometry.shell_count
    assert result.mesh_geometry.domain_max_radius == pytest.approx(
        structure.rmax
    )


def test_compute_electron_density_profile_respects_custom_rmax(tmp_path):
    structure_path = _write_xyz(
        tmp_path / "water_cutoff.xyz",
        [
            "3",
            "water cutoff",
            "O 0.0 0.0 0.0",
            "H 0.96 0.0 0.0",
            "H -0.24 0.93 0.0",
        ],
    )
    structure = load_electron_density_structure(structure_path)
    result = compute_electron_density_profile(
        structure,
        _mesh_settings_for_structure(
            structure,
            rstep=0.1,
            theta_divisions=24,
            phi_divisions=18,
            rmax=0.35,
        ),
    )

    assert result.mesh_geometry.domain_max_radius == pytest.approx(0.35)
    assert result.excluded_atom_count > 0
    assert result.excluded_electron_count > 0.0


def test_centered_atom_does_not_create_angular_averaging_singularity(tmp_path):
    structure_path = _write_xyz(
        tmp_path / "single_oxygen.xyz",
        [
            "1",
            "single oxygen",
            "O 0.0 0.0 0.0",
        ],
    )
    structure = load_electron_density_structure(structure_path)
    result = compute_electron_density_profile(
        structure,
        _mesh_settings_for_structure(
            structure,
            rstep=0.2,
            theta_divisions=120,
            phi_divisions=60,
            rmax=0.4,
        ),
    )
    point_deposit_first_shell_density = float(
        structure.atomic_numbers[0] / result.shell_volumes[0]
    )

    _assert_structure_shell_electron_total(result, structure)
    _assert_density_profile_electron_total(
        result,
        float(structure.atomic_numbers[0]),
    )
    assert result.shell_electron_counts[0] == pytest.approx(
        float(structure.atomic_numbers[0])
    )
    assert result.shell_volume_average_density[0] == pytest.approx(
        point_deposit_first_shell_density
    )
    assert result.orientation_average_density[0] < (
        result.shell_volume_average_density[0]
    )
    assert (
        result.orientation_average_density[0]
        < point_deposit_first_shell_density
    )
    assert np.count_nonzero(result.orientation_average_density > 0.0) > 1


def test_single_heavy_atom_is_tagged_into_only_one_shell(tmp_path):
    structure_path = _write_xyz(
        tmp_path / "pbh3_cluster.xyz",
        [
            "4",
            "PbH3 cluster",
            "Pb 0.0 0.0 0.0",
            "H 3.0 0.0 0.0",
            "H 0.0 3.0 0.0",
            "H 0.0 0.0 3.0",
        ],
    )
    structure = load_electron_density_structure(
        structure_path,
        center_mode="reference_element",
        reference_element="Pb",
    )
    mesh_settings = _mesh_settings_for_structure(
        structure,
        rstep=0.25,
        theta_divisions=48,
        phi_divisions=24,
        rmax=3.5,
    )
    result = compute_electron_density_profile(
        structure,
        mesh_settings,
    )
    baseline_counts = _baseline_shell_electron_counts(
        structure,
        mesh_settings,
    )
    lead_electrons = float(xraydb.atomic_number("Pb"))

    assert np.allclose(result.shell_electron_counts, baseline_counts)
    assert result.shell_electron_counts[0] == pytest.approx(lead_electrons)
    assert (
        np.count_nonzero(result.shell_electron_counts >= lead_electrons) == 1
    )
    _assert_structure_shell_electron_total(result, structure)
    _assert_density_profile_electron_total(
        result,
        float(np.sum(structure.atomic_numbers)),
    )


def test_centered_heavy_atom_density_is_spread_across_multiple_shells(
    tmp_path,
):
    structure_path = _write_xyz(
        tmp_path / "single_pb.xyz",
        [
            "1",
            "single lead",
            "Pb 0.0 0.0 0.0",
        ],
    )
    structure = load_electron_density_structure(
        structure_path,
        center_mode="reference_element",
        reference_element="Pb",
    )
    result = compute_electron_density_profile(
        structure,
        _mesh_settings_for_structure(
            structure,
            rstep=0.1,
            theta_divisions=48,
            phi_divisions=24,
            rmax=2.0,
        ),
    )
    point_deposit_first_shell_density = float(
        structure.atomic_numbers[0] / result.shell_volumes[0]
    )

    _assert_structure_shell_electron_total(result, structure)
    _assert_density_profile_electron_total(
        result,
        float(structure.atomic_numbers[0]),
    )
    assert result.shell_electron_counts[0] == pytest.approx(
        float(structure.atomic_numbers[0])
    )
    assert result.orientation_average_density[0] < (
        point_deposit_first_shell_density * 0.1
    )
    assert np.count_nonzero(result.orientation_average_density > 0.0) > 5


def test_small_origin_centered_cluster_assigns_each_atom_to_one_shell(
    tmp_path,
):
    structure_path = _write_xyz(
        tmp_path / "small_origin_cluster.xyz",
        [
            "4",
            "small origin-centered cluster",
            "O 0.0 0.0 0.0",
            "H 0.05 0.0 0.0",
            "C 0.26 0.0 0.0",
            "N 0.51 0.0 0.0",
        ],
    )
    structure = load_electron_density_structure(
        structure_path,
        center_mode="reference_element",
        reference_element="O",
    )
    result = compute_electron_density_profile(
        structure,
        _mesh_settings_for_structure(
            structure,
            rstep=0.25,
            theta_divisions=24,
            phi_divisions=18,
            rmax=0.8,
        ),
    )

    _assert_structure_shell_electron_total(result, structure)
    _assert_density_profile_electron_total(
        result,
        float(np.sum(structure.atomic_numbers)),
    )
    assert np.allclose(result.shell_electron_counts[:4], [9.0, 6.0, 7.0, 0.0])
    assert np.count_nonzero(result.shell_electron_counts) == 3


def test_electron_density_profile_is_invariant_to_angular_mesh_resolution(
    tmp_path,
):
    structure_path = _write_xyz(
        tmp_path / "angular_mesh_invariance.xyz",
        [
            "4",
            "angular mesh invariance",
            "O 0.0 0.0 0.0",
            "H 0.96 0.0 0.0",
            "H -0.24 0.93 0.0",
            "Pb 0.0 0.0 2.4",
        ],
    )
    structure = load_electron_density_structure(structure_path)
    coarse = compute_electron_density_profile(
        structure,
        _mesh_settings_for_structure(
            structure,
            rstep=0.2,
            theta_divisions=8,
            phi_divisions=6,
            rmax=max(float(structure.rmax), 3.0),
        ),
    )
    fine = compute_electron_density_profile(
        structure,
        _mesh_settings_for_structure(
            structure,
            rstep=0.2,
            theta_divisions=96,
            phi_divisions=72,
            rmax=max(float(structure.rmax), 3.0),
        ),
    )

    assert np.allclose(
        coarse.shell_electron_counts,
        fine.shell_electron_counts,
    )
    assert np.allclose(
        coarse.shell_volumes,
        fine.shell_volumes,
    )
    assert np.allclose(
        coarse.orientation_average_density,
        fine.orientation_average_density,
    )
    assert np.allclose(
        coarse.shell_volume_average_density,
        fine.shell_volume_average_density,
    )
    _assert_density_profile_electron_total(
        coarse,
        float(np.sum(structure.atomic_numbers)),
    )
    _assert_density_profile_electron_total(
        fine,
        float(np.sum(structure.atomic_numbers)),
    )


def test_single_heavy_atom_finite_radius_density_changes_downstream_trace(
    tmp_path,
):
    structure_path = _write_xyz(
        tmp_path / "pbh3_trace.xyz",
        [
            "4",
            "PbH3 trace",
            "Pb 0.0 0.0 0.0",
            "H 3.0 0.0 0.0",
            "H 0.0 3.0 0.0",
            "H 0.0 0.0 3.0",
        ],
    )
    structure = load_electron_density_structure(
        structure_path,
        center_mode="reference_element",
        reference_element="Pb",
    )
    mesh_settings = _mesh_settings_for_structure(
        structure,
        rstep=0.2,
        theta_divisions=36,
        phi_divisions=18,
        rmax=3.5,
    )
    result = compute_electron_density_profile(
        structure,
        mesh_settings,
        smearing_settings=ElectronDensitySmearingSettings(
            debye_waller_factor=0.006
        ),
    )
    baseline_result = _profile_result_with_shell_counts(
        result,
        _baseline_shell_electron_counts(structure, mesh_settings),
    )
    settings = ElectronDensityFourierTransformSettings(
        r_min=float(result.radial_centers[0]),
        r_max=float(result.radial_centers[-1]),
        q_min=0.05,
        q_max=1.0,
        q_step=0.05,
        resampling_points=128,
        use_solvent_subtracted_profile=False,
    )

    tagged_transform = compute_electron_density_scattering_profile(
        result,
        settings,
    )
    baseline_transform = compute_electron_density_scattering_profile(
        baseline_result,
        settings,
    )

    assert np.all(np.isfinite(tagged_transform.intensity))
    assert np.all(tagged_transform.intensity >= 0.0)
    assert np.allclose(tagged_transform.q_values, baseline_transform.q_values)
    assert not np.allclose(
        tagged_transform.intensity,
        baseline_transform.intensity,
    )
    assert tagged_transform.intensity[0] < baseline_transform.intensity[0]


def test_single_atom_debye_scattering_profile_for_input_averages_iq(
    tmp_path,
):
    folder = tmp_path / "single_atom_cluster"
    folder.mkdir()
    _write_xyz(
        folder / "frame_0001.xyz",
        [
            "1",
            "single iodine 1",
            "I 0.0 0.0 0.0",
        ],
    )
    _write_xyz(
        folder / "frame_0002.xyz",
        [
            "1",
            "single iodine 2",
            "I 2.5 -1.0 0.2",
        ],
    )

    result = compute_single_atom_debye_scattering_profile_for_input(
        inspect_structure_input(folder),
        ElectronDensityFourierTransformSettings(
            r_min=0.0,
            r_max=1.0,
            q_min=0.1,
            q_max=0.5,
            q_step=0.2,
        ),
    )

    expected_q = np.asarray([0.1, 0.3, 0.5], dtype=float)
    expected_intensity = np.square(atomic_form_factor("I", expected_q))

    assert result.preview.source_mode == "single_atom_debye"
    assert (
        result.preview.source_profile_label == "Single-atom Debye scattering"
    )
    assert np.allclose(result.q_values, expected_q)
    assert np.allclose(result.intensity, expected_intensity)
    assert np.allclose(
        result.scattering_amplitude,
        np.sqrt(expected_intensity),
    )


def test_single_atom_debye_scattering_profile_for_input_computes_each_element_once(
    tmp_path,
    monkeypatch,
):
    folder = tmp_path / "single_atom_cluster"
    folder.mkdir()
    _write_xyz(
        folder / "frame_0001.xyz",
        [
            "1",
            "single iodine 1",
            "I 0.0 0.0 0.0",
        ],
    )
    _write_xyz(
        folder / "frame_0002.xyz",
        [
            "1",
            "single iodine 2",
            "I 2.5 -1.0 0.2",
        ],
    )
    _write_xyz(
        folder / "frame_0003.xyz",
        [
            "1",
            "single iodine 3",
            "I -0.5 1.0 4.0",
        ],
    )

    real_atomic_form_factor = atomic_form_factor
    call_counter = {"count": 0}

    def fake_atomic_form_factor(element, q_values):
        call_counter["count"] += 1
        return real_atomic_form_factor(element, q_values)

    monkeypatch.setattr(
        "saxshell.saxs.electron_density_mapping.workflow.atomic_form_factor",
        fake_atomic_form_factor,
    )

    result = compute_single_atom_debye_scattering_profile_for_input(
        inspect_structure_input(folder),
        ElectronDensityFourierTransformSettings(
            r_min=0.0,
            r_max=1.0,
            q_min=0.1,
            q_max=0.5,
            q_step=0.2,
        ),
    )

    assert call_counter["count"] == 1
    np.testing.assert_allclose(
        result.intensity,
        np.square(real_atomic_form_factor("I", result.q_values)),
    )


def test_average_debye_scattering_profile_for_input_reuses_equivalent_single_atom_traces(
    tmp_path,
    monkeypatch,
):
    folder = tmp_path / "single_atom_cluster"
    folder.mkdir()
    _write_xyz(
        folder / "frame_0001.xyz",
        [
            "1",
            "single iodine 1",
            "I 0.0 0.0 0.0",
        ],
    )
    _write_xyz(
        folder / "frame_0002.xyz",
        [
            "1",
            "single iodine 2",
            "I 2.5 -1.0 0.2",
        ],
    )
    _write_xyz(
        folder / "frame_0003.xyz",
        [
            "1",
            "single iodine 3",
            "I -0.5 1.0 4.0",
        ],
    )

    real_compute_debye_intensity = density_workflow.compute_debye_intensity
    call_counter = {"count": 0}

    def fake_compute_debye_intensity(*args, **kwargs):
        call_counter["count"] += 1
        return real_compute_debye_intensity(*args, **kwargs)

    monkeypatch.setattr(
        "saxshell.saxs.electron_density_mapping.workflow.compute_debye_intensity",
        fake_compute_debye_intensity,
    )

    settings = ElectronDensityFourierTransformSettings(
        r_min=0.0,
        r_max=1.0,
        q_min=0.1,
        q_max=0.5,
        q_step=0.2,
    )
    result = compute_average_debye_scattering_profile_for_input(
        inspect_structure_input(folder),
        settings,
    )

    expected_q = np.asarray([0.1, 0.3, 0.5], dtype=float)
    expected_intensity = np.square(atomic_form_factor("I", expected_q))

    assert call_counter["count"] == 1
    assert result.source_structure_count == 3
    assert result.unique_elements == ("I",)
    assert any(
        "single-atom type" in note.lower()
        or "reused across the average" in note.lower()
        for note in result.notes
    )
    np.testing.assert_allclose(result.q_values, expected_q)
    np.testing.assert_allclose(result.mean_intensity, expected_intensity)
    np.testing.assert_allclose(result.std_intensity, np.zeros_like(expected_q))
    np.testing.assert_allclose(result.se_intensity, np.zeros_like(expected_q))


def test_compute_electron_density_profile_for_folder_averages_and_tracks_variance(
    tmp_path,
):
    folder = tmp_path / "ensemble"
    folder.mkdir()
    first_path = _write_xyz(
        folder / "frame_0001.xyz",
        [
            "3",
            "frame 1",
            "O 0.0 0.0 0.0",
            "H 0.96 0.0 0.0",
            "H -0.24 0.93 0.0",
        ],
    )
    second_path = _write_xyz(
        folder / "frame_0002.xyz",
        [
            "3",
            "frame 2",
            "O 0.0 0.0 0.0",
            "H 1.02 0.0 0.0",
            "H -0.18 0.88 0.0",
        ],
    )
    inspection = inspect_structure_input(folder)
    reference_structure = load_electron_density_structure(
        inspection.reference_file
    )
    first_structure = load_electron_density_structure(first_path)
    second_structure = load_electron_density_structure(second_path)
    mesh_settings = _mesh_settings_for_structure(
        reference_structure,
        rstep=0.1,
        theta_divisions=24,
        phi_divisions=18,
        rmax=max(
            float(first_structure.rmax),
            float(second_structure.rmax),
        ),
    )
    progress_events: list[tuple[int, int, str]] = []

    averaged = compute_electron_density_profile_for_input(
        inspection,
        mesh_settings,
        smearing_settings=ElectronDensitySmearingSettings(
            debye_waller_factor=0.006
        ),
        progress_callback=lambda current, total, message: progress_events.append(
            (current, total, message)
        ),
    )
    first = compute_electron_density_profile(
        first_structure,
        mesh_settings,
        smearing_settings=ElectronDensitySmearingSettings(
            debye_waller_factor=0.006
        ),
    )
    second = compute_electron_density_profile(
        second_structure,
        mesh_settings,
        smearing_settings=ElectronDensitySmearingSettings(
            debye_waller_factor=0.006
        ),
    )
    raw_stack = np.vstack(
        [first.orientation_average_density, second.orientation_average_density]
    )
    smeared_stack = np.vstack(
        [
            first.smeared_orientation_average_density,
            second.smeared_orientation_average_density,
        ]
    )

    assert averaged.input_mode == "folder"
    assert averaged.source_structure_count == 2
    assert len(averaged.member_summaries) == 2
    assert np.allclose(
        averaged.orientation_average_density,
        np.mean(raw_stack, axis=0),
    )
    assert np.allclose(
        averaged.orientation_density_variance,
        np.var(raw_stack, axis=0),
    )
    assert np.allclose(
        averaged.smeared_orientation_average_density,
        np.mean(smeared_stack, axis=0),
    )
    assert np.allclose(
        averaged.smeared_orientation_density_variance,
        np.var(smeared_stack, axis=0),
    )
    _assert_structure_shell_electron_total(first, first_structure)
    _assert_structure_shell_electron_total(second, second_structure)
    _assert_density_profile_electron_total(
        first,
        float(np.sum(first_structure.atomic_numbers)),
    )
    _assert_density_profile_electron_total(
        second,
        float(np.sum(second_structure.atomic_numbers)),
    )
    _assert_shell_electron_total(
        averaged,
        np.mean(
            [
                float(np.sum(first_structure.atomic_numbers)),
                float(np.sum(second_structure.atomic_numbers)),
            ]
        ),
    )
    _assert_density_profile_electron_total(
        averaged,
        np.mean(
            [
                float(np.sum(first_structure.atomic_numbers)),
                float(np.sum(second_structure.atomic_numbers)),
            ]
        ),
    )
    assert progress_events
    assert "Averaging electron density" in progress_events[-2][2]


def test_compute_electron_density_profile_for_input_parallelizes_batches(
    tmp_path,
    monkeypatch,
):
    folder = tmp_path / "parallel_ensemble"
    folder.mkdir()
    for index, oxygen_x in enumerate((0.0, 0.2, 0.4, 0.6), start=1):
        _write_xyz(
            folder / f"frame_{index:04d}.xyz",
            [
                "3",
                f"frame {index}",
                f"O {oxygen_x:.3f} 0.0 0.0",
                f"H {oxygen_x + 0.96:.3f} 0.0 0.0",
                f"H {oxygen_x - 0.24:.3f} 0.93 0.0",
            ],
        )
    inspection = inspect_structure_input(folder)
    reference_structure = load_electron_density_structure(
        inspection.reference_file
    )
    mesh_settings = _mesh_settings_for_structure(
        reference_structure,
        rstep=0.15,
        theta_divisions=24,
        phi_divisions=18,
    )
    real_compute = density_workflow.compute_electron_density_profile
    thread_names: list[str] = []

    def _recording_compute(*args, **kwargs):
        thread_names.append(threading.current_thread().name)
        time.sleep(0.02)
        return real_compute(*args, **kwargs)

    monkeypatch.setattr(
        density_workflow,
        "_resolve_electron_density_worker_count",
        lambda structure_count: 2,
    )
    monkeypatch.setattr(
        density_workflow,
        "compute_electron_density_profile",
        _recording_compute,
    )

    result = compute_electron_density_profile_for_input(
        inspection,
        mesh_settings,
    )

    assert result.source_structure_count == 4
    assert any(name.startswith("electron-density") for name in thread_names)


def test_compute_electron_density_profile_for_input_reuses_reference_structure(
    tmp_path,
    monkeypatch,
):
    folder = tmp_path / "reused_reference"
    folder.mkdir()
    _write_xyz(
        folder / "frame_0001.xyz",
        [
            "3",
            "frame 1",
            "O 0.0 0.0 0.0",
            "H 0.96 0.0 0.0",
            "H -0.24 0.93 0.0",
        ],
    )
    _write_xyz(
        folder / "frame_0002.xyz",
        [
            "3",
            "frame 2",
            "O 0.1 0.0 0.0",
            "H 1.06 0.0 0.0",
            "H -0.14 0.93 0.0",
        ],
    )
    inspection = inspect_structure_input(folder)
    reference_structure = load_electron_density_structure(
        inspection.reference_file
    )
    mesh_settings = _mesh_settings_for_structure(reference_structure)
    real_loader = density_workflow.load_electron_density_structure

    def _guarded_load(file_path, *args, **kwargs):
        if Path(file_path).expanduser().resolve() == inspection.reference_file:
            raise AssertionError(
                "The preloaded reference structure was reloaded."
            )
        return real_loader(file_path, *args, **kwargs)

    monkeypatch.setattr(
        density_workflow,
        "load_electron_density_structure",
        _guarded_load,
    )

    result = compute_electron_density_profile_for_input(
        inspection,
        mesh_settings,
        reference_structure=reference_structure,
    )

    assert result.source_structure_count == 2
    assert result.structure.file_path == inspection.reference_file


def test_resolve_electron_density_worker_count_is_conservative(monkeypatch):
    monkeypatch.setattr(density_workflow.os, "cpu_count", lambda: 12)
    assert density_workflow._resolve_electron_density_worker_count(1) == 1
    assert density_workflow._resolve_electron_density_worker_count(3) == 3
    assert density_workflow._resolve_electron_density_worker_count(20) == 4

    monkeypatch.setattr(density_workflow.os, "cpu_count", lambda: 2)
    assert density_workflow._resolve_electron_density_worker_count(8) == 1


def test_compute_electron_density_profile_for_input_uses_reference_element_mode(
    tmp_path,
):
    folder = tmp_path / "reference_mode_frames"
    folder.mkdir()
    first_path = _write_xyz(
        folder / "frame_0001.xyz",
        [
            "3",
            "frame 1",
            "Pb 0.0 0.0 0.0",
            "O 2.0 0.0 0.0",
            "O 4.0 0.0 0.0",
        ],
    )
    second_path = _write_xyz(
        folder / "frame_0002.xyz",
        [
            "3",
            "frame 2",
            "Pb 1.0 0.0 0.0",
            "O 3.0 0.0 0.0",
            "O 5.0 0.0 0.0",
        ],
    )
    inspection = inspect_structure_input(folder)
    reference_structure = load_electron_density_structure(first_path)
    result = compute_electron_density_profile_for_input(
        inspection,
        _mesh_settings_for_structure(reference_structure, rmax=3.5),
        center_mode="reference_element",
        reference_element="O",
    )

    assert result.structure.center_mode == "reference_element"
    assert result.structure.reference_element == "O"
    assert len(result.member_summaries) == 2
    assert result.member_summaries[0].center_mode == "reference_element"
    assert result.member_summaries[0].reference_element == "O"
    assert np.allclose(
        result.member_summaries[0].active_center,
        [3.0, 0.0, 0.0],
    )
    assert np.allclose(
        result.member_summaries[1].active_center,
        [4.0, 0.0, 0.0],
    )
    assert second_path in result.source_files


def test_compute_electron_density_profile_for_input_locks_contiguous_frame_sets(
    tmp_path,
):
    folder = tmp_path / "contiguous_frames"
    folder.mkdir()
    _write_xyz(
        folder / "frame_0001_AAA.xyz",
        [
            "2",
            "frame 1",
            "Pb 0.0 0.0 0.0",
            "O 2.0 0.0 0.0",
        ],
    )
    _write_xyz(
        folder / "frame_0002_AAA.xyz",
        [
            "2",
            "frame 2",
            "Pb 10.0 0.0 0.0",
            "O 13.0 0.0 0.0",
        ],
    )
    _write_xyz(
        folder / "frame_0004_AAA.xyz",
        [
            "2",
            "frame 4",
            "Pb 20.0 0.0 0.0",
            "O 22.0 0.0 0.0",
        ],
    )
    _write_xyz(
        folder / "frame_0005_AAA.xyz",
        [
            "2",
            "frame 5",
            "Pb 30.0 0.0 0.0",
            "O 34.0 0.0 0.0",
        ],
    )

    inspection = inspect_structure_input(folder)
    reference_structure = load_electron_density_structure(
        inspection.reference_file
    )
    mesh_settings = _mesh_settings_for_structure(
        reference_structure,
        rmax=5.0,
    )

    standard = compute_electron_density_profile_for_input(
        inspection,
        mesh_settings,
        use_contiguous_frame_mode=False,
    )
    locked = compute_electron_density_profile_for_input(
        inspection,
        mesh_settings,
        use_contiguous_frame_mode=True,
    )

    assert locked.contiguous_frame_mode_requested
    assert locked.contiguous_frame_mode_applied
    assert locked.averaging_mode == "contiguous_frame_sets"
    assert len(locked.contiguous_frame_sets) == 2
    assert locked.contiguous_frame_sets[0].frame_ids == (1, 2)
    assert locked.contiguous_frame_sets[1].frame_ids == (4, 5)
    assert locked.contiguous_frame_sets[0].frame_labels == ("0001", "0002")

    standard_offsets = [
        float(entry.active_center[0] - pb_x)
        for entry, pb_x in zip(
            standard.member_summaries,
            (0.0, 10.0, 20.0, 30.0),
        )
    ]
    locked_offsets = [
        float(entry.active_center[0] - pb_x)
        for entry, pb_x in zip(
            locked.member_summaries,
            (0.0, 10.0, 20.0, 30.0),
        )
    ]

    assert standard_offsets[0] != pytest.approx(standard_offsets[1])
    assert standard_offsets[2] != pytest.approx(standard_offsets[3])
    assert locked_offsets[0] == pytest.approx(locked_offsets[1])
    assert locked_offsets[2] == pytest.approx(locked_offsets[3])
    assert locked_offsets[0] != pytest.approx(locked_offsets[2])


def test_compute_electron_density_profile_for_input_falls_back_when_frame_ids_missing(
    tmp_path,
):
    folder = tmp_path / "noncontiguous_names"
    folder.mkdir()
    _write_xyz(
        folder / "cluster_a.xyz",
        [
            "2",
            "cluster a",
            "Pb 0.0 0.0 0.0",
            "O 2.0 0.0 0.0",
        ],
    )
    _write_xyz(
        folder / "cluster_b.xyz",
        [
            "2",
            "cluster b",
            "Pb 0.0 0.0 0.0",
            "O 3.0 0.0 0.0",
        ],
    )

    inspection = inspect_structure_input(folder)
    reference_structure = load_electron_density_structure(
        inspection.reference_file
    )
    mesh_settings = _mesh_settings_for_structure(
        reference_structure,
        rmax=5.0,
    )

    standard = compute_electron_density_profile_for_input(
        inspection,
        mesh_settings,
        use_contiguous_frame_mode=False,
    )
    fallback = compute_electron_density_profile_for_input(
        inspection,
        mesh_settings,
        use_contiguous_frame_mode=True,
    )

    assert fallback.contiguous_frame_mode_requested
    assert not fallback.contiguous_frame_mode_applied
    assert fallback.contiguous_frame_sets == ()
    assert fallback.averaging_mode == "complete_average"
    assert any(
        "Falling back to complete averaging" in note
        for note in fallback.averaging_notes
    )
    assert np.allclose(
        fallback.orientation_average_density,
        standard.orientation_average_density,
    )


def test_compute_electron_density_profile_for_input_can_pin_geometric_tracking_for_contiguous_pdb_frames(
    tmp_path,
):
    folder = tmp_path / "pdb_contiguous_tracking"
    folder.mkdir()
    _write_pb_o_pdb_frame(
        folder / "frame_0001.pdb",
        pb_x=0.0,
        o_x=4.0,
    )
    _write_pb_o_pdb_frame(
        folder / "frame_0002.pdb",
        pb_x=0.0,
        o_x=8.0,
    )

    inspection = inspect_structure_input(folder)
    reference_structure = load_electron_density_structure(
        inspection.reference_file
    )
    standard_mesh = _mesh_settings_for_structure(
        reference_structure,
        rmax=10.0,
        pin_contiguous_geometric_tracking=False,
    )
    pinned_mesh = ElectronDensityMeshSettings(
        rstep=standard_mesh.rstep,
        theta_divisions=standard_mesh.theta_divisions,
        phi_divisions=standard_mesh.phi_divisions,
        rmax=standard_mesh.rmax,
        pin_contiguous_geometric_tracking=True,
    )

    shared = compute_electron_density_profile_for_input(
        inspection,
        standard_mesh,
        use_contiguous_frame_mode=True,
    )
    pinned = compute_electron_density_profile_for_input(
        inspection,
        pinned_mesh,
        use_contiguous_frame_mode=True,
    )

    assert pinned.contiguous_frame_mode_requested
    assert pinned.contiguous_frame_mode_applied
    assert pinned.pinned_geometric_tracking_requested
    assert pinned.pinned_geometric_tracking_applied
    assert any(
        "Pinned each contiguous PDB frame set" in note
        for note in pinned.averaging_notes
    )
    assert pinned.member_summaries[0].active_center[0] == pytest.approx(
        pinned.member_summaries[1].active_center[0]
    )
    assert pinned.member_summaries[0].active_center[0] == pytest.approx(
        pinned.member_summaries[0].center_of_mass[0]
    )
    assert pinned.member_summaries[1].center_of_mass[0] != pytest.approx(
        pinned.member_summaries[1].active_center[0]
    )
    assert shared.member_summaries[1].active_center[0] != pytest.approx(
        pinned.member_summaries[1].active_center[0]
    )


def test_compute_electron_density_profile_for_input_pinned_tracking_falls_back_to_standard_contiguous_lock_for_xyz(
    tmp_path,
):
    folder = tmp_path / "xyz_pinned_tracking_fallback"
    folder.mkdir()
    _write_xyz(
        folder / "frame_0001.xyz",
        [
            "2",
            "frame one",
            "Pb 0.0 0.0 0.0",
            "O 2.0 0.0 0.0",
        ],
    )
    _write_xyz(
        folder / "frame_0002.xyz",
        [
            "2",
            "frame two",
            "Pb 0.0 0.0 0.0",
            "O 4.0 0.0 0.0",
        ],
    )

    inspection = inspect_structure_input(folder)
    mesh_settings = ElectronDensityMeshSettings(
        rstep=0.1,
        theta_divisions=120,
        phi_divisions=60,
        rmax=5.0,
        pin_contiguous_geometric_tracking=True,
    )
    result = compute_electron_density_profile_for_input(
        inspection,
        mesh_settings,
        use_contiguous_frame_mode=True,
    )

    assert result.contiguous_frame_mode_requested
    assert result.contiguous_frame_mode_applied
    assert result.pinned_geometric_tracking_requested
    assert not result.pinned_geometric_tracking_applied
    assert any(
        "requires contiguous PDB frame sets" in note
        for note in result.averaging_notes
    )


def test_compute_electron_density_profile_for_input_skips_visual_metadata_for_non_reference_structures(
    tmp_path,
    monkeypatch,
):
    folder = tmp_path / "batch_visual_metadata"
    folder.mkdir()
    _write_xyz(
        folder / "frame_0001.xyz",
        [
            "2",
            "frame 1",
            "Pb 0.0 0.0 0.0",
            "O 2.0 0.0 0.0",
        ],
    )
    _write_xyz(
        folder / "frame_0002.xyz",
        [
            "2",
            "frame 2",
            "Pb 1.0 0.0 0.0",
            "O 3.0 0.0 0.0",
        ],
    )

    inspection = inspect_structure_input(folder)
    mesh_settings = ElectronDensityMeshSettings(
        rstep=0.25,
        theta_divisions=18,
        phi_divisions=12,
        rmax=5.0,
    )
    calls = {"bonds": 0, "comment": 0}

    def _fake_detect_bonds_records(_atoms):
        calls["bonds"] += 1
        return ()

    def _fake_read_structure_comment(_path):
        calls["comment"] += 1
        return ""

    monkeypatch.setattr(
        "saxshell.saxs.electron_density_mapping.workflow.detect_bonds_records",
        _fake_detect_bonds_records,
    )
    monkeypatch.setattr(
        "saxshell.saxs.electron_density_mapping.workflow.read_structure_comment",
        _fake_read_structure_comment,
    )

    result = compute_electron_density_profile_for_input(
        inspection,
        mesh_settings,
    )

    assert result.source_structure_count == 2
    assert calls["bonds"] == 1
    assert calls["comment"] == 1


def test_prepare_fourier_transform_clamps_qmax_and_computes_intensity(
    tmp_path,
):
    structure_path = _write_xyz(
        tmp_path / "fourier_water.xyz",
        [
            "3",
            "fourier water",
            "O 0.0 0.0 0.0",
            "H 0.96 0.0 0.0",
            "H -0.24 0.93 0.0",
        ],
    )
    structure = load_electron_density_structure(structure_path)
    profile = compute_electron_density_profile(
        structure,
        _mesh_settings_for_structure(
            structure,
            rstep=0.1,
            theta_divisions=24,
            phi_divisions=18,
            rmax=max(float(structure.rmax), 1.5),
        ),
    )
    settings = ElectronDensityFourierTransformSettings(
        r_min=float(profile.radial_centers[0]),
        r_max=float(profile.radial_centers[-1]),
        window_function="lorch",
        resampling_points=64,
        q_min=0.02,
        q_max=500.0,
        q_step=0.01,
    )

    preview = prepare_electron_density_fourier_transform(profile, settings)
    transform = compute_electron_density_scattering_profile(profile, settings)

    assert preview.q_max_was_clamped
    assert preview.settings.q_max == pytest.approx(
        preview.nyquist_q_max_a_inverse
    )
    assert preview.q_grid_is_oversampled
    assert transform.q_values[-1] <= preview.nyquist_q_max_a_inverse + 1.0e-9
    assert np.all(transform.intensity >= 0.0)


def test_prepare_fourier_transform_mirrors_density_profile_by_default(
    tmp_path,
):
    structure_path = _write_xyz(
        tmp_path / "fourier_mirrored.xyz",
        [
            "3",
            "fourier mirrored",
            "O 0.0 0.0 0.0",
            "H 0.96 0.0 0.0",
            "H -0.24 0.93 0.0",
        ],
    )
    structure = load_electron_density_structure(structure_path)
    profile = compute_electron_density_profile(
        structure,
        _mesh_settings_for_structure(
            structure,
            rstep=0.1,
            theta_divisions=24,
            phi_divisions=18,
        ),
    )

    preview = prepare_electron_density_fourier_transform(
        profile,
        ElectronDensityFourierTransformSettings(
            r_max=float(profile.radial_centers[-1]),
            window_function="hanning",
            resampling_points=65,
            q_min=0.02,
            q_max=1.2,
            q_step=0.01,
        ),
    )

    assert preview.settings.domain_mode == "mirrored"
    assert preview.available_r_min == pytest.approx(-preview.available_r_max)
    assert preview.source_radial_values[0] == pytest.approx(
        -preview.source_radial_values[-1]
    )
    assert np.allclose(
        preview.source_density_values,
        preview.source_density_values[::-1],
    )
    assert np.allclose(preview.window_values, preview.window_values[::-1])


def test_prepare_fourier_transform_supports_origin_and_exafs_windows(tmp_path):
    structure_path = _write_xyz(
        tmp_path / "fourier_windows.xyz",
        [
            "3",
            "fourier windows",
            "O 0.0 0.0 0.0",
            "H 0.96 0.0 0.0",
            "H -0.24 0.93 0.0",
        ],
    )
    structure = load_electron_density_structure(structure_path)
    profile = compute_electron_density_profile(
        structure,
        _mesh_settings_for_structure(
            structure,
            rstep=0.1,
            theta_divisions=24,
            phi_divisions=18,
        ),
    )

    for window_name in (
        "hanning",
        "parzen",
        "welch",
        "gaussian",
        "sine",
        "kaiser_bessel",
    ):
        preview = prepare_electron_density_fourier_transform(
            profile,
            ElectronDensityFourierTransformSettings(
                r_min=0.0,
                r_max=float(profile.radial_centers[-1]),
                window_function=window_name,
                resampling_points=65,
                q_min=0.02,
                q_max=6.0,
                q_step=0.05,
            ),
        )
        assert preview.available_r_min == pytest.approx(
            -float(profile.radial_centers[-1])
        )
        assert preview.available_r_max == pytest.approx(
            float(profile.radial_centers[-1])
        )
        assert preview.source_radial_values[0] == pytest.approx(
            -float(profile.radial_centers[-1])
        )
        assert preview.source_radial_values[-1] == pytest.approx(
            float(profile.radial_centers[-1])
        )
        center_index = int(np.argmax(preview.window_values))
        assert center_index in {
            len(preview.window_values) // 2,
            len(preview.window_values) // 2 - 1,
        }
        assert preview.window_values[0] <= preview.window_values[center_index]
        assert preview.window_values[-1] <= preview.window_values[center_index]
        assert np.allclose(preview.window_values, preview.window_values[::-1])


def test_apply_smearing_to_profile_result_reuses_member_profiles_exactly(
    tmp_path,
):
    folder = tmp_path / "ensemble_resmear"
    folder.mkdir()
    _write_xyz(
        folder / "frame_0001.xyz",
        [
            "3",
            "frame 1",
            "O 0.0 0.0 0.0",
            "H 0.96 0.0 0.0",
            "H -0.24 0.93 0.0",
        ],
    )
    _write_xyz(
        folder / "frame_0002.xyz",
        [
            "3",
            "frame 2",
            "O 0.0 0.0 0.0",
            "H 1.02 0.0 0.0",
            "H -0.18 0.88 0.0",
        ],
    )
    inspection = inspect_structure_input(folder)
    reference_structure = load_electron_density_structure(
        inspection.reference_file
    )
    mesh_settings = _mesh_settings_for_structure(
        reference_structure,
        rstep=0.1,
        theta_divisions=24,
        phi_divisions=18,
    )
    result = compute_electron_density_profile_for_input(
        inspection,
        mesh_settings,
        smearing_settings=ElectronDensitySmearingSettings(
            debye_waller_factor=0.006
        ),
    )

    unsmeared = apply_smearing_to_profile_result(
        result,
        ElectronDensitySmearingSettings(debye_waller_factor=0.0),
    )

    assert np.allclose(
        unsmeared.smeared_orientation_average_density,
        result.orientation_average_density,
    )
    assert np.allclose(
        unsmeared.smeared_orientation_density_variance,
        result.orientation_density_variance,
    )
    assert np.allclose(
        unsmeared.smeared_orientation_density_stddev,
        result.orientation_density_stddev,
    )
    assert float(
        np.sum(
            result.smeared_orientation_average_density * result.shell_volumes
        )
    ) <= (
        float(
            np.sum(result.orientation_average_density * result.shell_volumes)
        )
        + 1.0e-9
    )


def test_apply_solvent_contrast_to_profile_result_tracks_cutoff_and_residual(
    tmp_path,
):
    structure_path = _write_xyz(
        tmp_path / "solvent_contrast.xyz",
        [
            "4",
            "solvent contrast",
            "N 0.0 0.0 0.1",
            "H 0.94 0.0 -0.2",
            "H -0.47 0.81 -0.2",
            "H -0.47 -0.81 -0.2",
        ],
    )
    structure = load_electron_density_structure(structure_path)
    result = compute_electron_density_profile(
        structure,
        _mesh_settings_for_structure(
            structure,
            rstep=0.1,
            theta_divisions=24,
            phi_divisions=18,
        ),
    )
    direct_density = (
        float(
            np.max(
                np.asarray(
                    result.smeared_orientation_average_density, dtype=float
                )
            )
        )
        * 0.5
    )

    contrasted = apply_solvent_contrast_to_profile_result(
        result,
        ContrastSolventDensitySettings.from_values(
            method=CONTRAST_SOLVENT_METHOD_DIRECT,
            direct_electron_density_e_per_a3=direct_density,
        ),
        solvent_name="Direct solvent",
    )

    assert contrasted.solvent_contrast is not None
    assert contrasted.solvent_contrast.cutoff_radius_a is not None
    assert np.allclose(
        contrasted.solvent_contrast.solvent_subtracted_smeared_density,
        np.asarray(result.smeared_orientation_average_density, dtype=float)
        - direct_density,
    )


def test_prepare_fourier_transform_uses_solvent_subtracted_profile_when_available(
    tmp_path,
):
    structure_path = _write_xyz(
        tmp_path / "fourier_solvent.xyz",
        [
            "4",
            "fourier solvent",
            "N 0.0 0.0 0.1",
            "H 0.94 0.0 -0.2",
            "H -0.47 0.81 -0.2",
            "H -0.47 -0.81 -0.2",
        ],
    )
    structure = load_electron_density_structure(structure_path)
    result = compute_electron_density_profile(
        structure,
        _mesh_settings_for_structure(
            structure,
            rstep=0.1,
            theta_divisions=24,
            phi_divisions=18,
        ),
    )
    contrasted = apply_solvent_contrast_to_profile_result(
        result,
        ContrastSolventDensitySettings.from_values(
            method=CONTRAST_SOLVENT_METHOD_DIRECT,
            direct_electron_density_e_per_a3=float(
                np.max(
                    np.asarray(
                        result.smeared_orientation_average_density,
                        dtype=float,
                    )
                )
                * 0.5
            ),
        ),
        solvent_name="Direct solvent",
    )
    settings = ElectronDensityFourierTransformSettings(
        r_min=float(contrasted.radial_centers[0]),
        r_max=float(contrasted.radial_centers[-1]),
        window_function="lorch",
        resampling_points=64,
        q_min=0.02,
        q_max=10.0,
        q_step=0.05,
        use_solvent_subtracted_profile=True,
    )

    preview = prepare_electron_density_fourier_transform(contrasted, settings)

    assert preview.source_profile_label.startswith("Solvent-subtracted")
    assert contrasted.solvent_contrast is not None
    expected_positive = np.concatenate(
        (
            [
                float(
                    contrasted.solvent_contrast.solvent_subtracted_smeared_density[
                        0
                    ]
                )
            ],
            np.asarray(
                contrasted.solvent_contrast.solvent_subtracted_smeared_density,
                dtype=float,
            ),
        )
    )
    center_index = len(preview.source_radial_values) // 2
    assert preview.source_radial_values[0] == pytest.approx(
        -float(contrasted.radial_centers[-1])
    )
    assert preview.source_radial_values[center_index] == pytest.approx(0.0)
    assert preview.source_density_values[center_index] == pytest.approx(
        expected_positive[0]
    )
    assert np.allclose(
        preview.source_density_values[center_index:],
        expected_positive,
    )
    assert np.allclose(
        preview.source_density_values[: center_index + 1],
        preview.source_density_values[center_index:][::-1],
    )


def test_write_electron_density_profile_outputs_writes_csv_and_json(tmp_path):
    structure_path = _write_xyz(
        tmp_path / "water.xyz",
        [
            "3",
            "water",
            "O 0.0 0.0 0.0",
            "H 0.96 0.0 0.0",
            "H -0.24 0.93 0.0",
        ],
    )
    structure = load_electron_density_structure(structure_path)
    result = compute_electron_density_profile(
        structure,
        _mesh_settings_for_structure(
            structure,
            rstep=0.3,
            theta_divisions=16,
            phi_divisions=8,
        ),
    )

    artifacts = write_electron_density_profile_outputs(
        result,
        tmp_path / "outputs",
        "water_density_profile",
    )

    assert artifacts.csv_path.is_file()
    assert artifacts.json_path.is_file()
    payload = artifacts.json_path.read_text(encoding="utf-8")
    payload_json = json.loads(payload)
    assert "center_of_mass_a" in payload
    assert "geometric_center_a" in payload
    assert "reference_element" in payload
    assert "active_center_a" in payload
    assert "mesh_settings" in payload
    assert "smearing_settings" in payload
    assert (
        payload_json["density_profile_metadata"][
            "displayed_raw_density_method"
        ]
        == "finite_radius_shell_overlap"
    )
    assert (
        payload_json["density_profile_metadata"]["shell_electron_count_role"]
        == "point_tag_bookkeeping"
    )
    assert (
        "Fourier/scattering outputs"
        in payload_json["density_profile_metadata"]["interpretation_note"]
    )


def test_main_window_updates_mesh_only_when_requested(qapp, tmp_path):
    del qapp
    structure_path = _write_xyz(
        tmp_path / "ethane.xyz",
        [
            "2",
            "ethane_min",
            "C -0.77 0.0 0.0",
            "C 0.77 0.0 0.0",
        ],
    )
    window = ElectronDensityMappingMainWindow(
        initial_input_path=structure_path
    )
    original_settings = window._active_mesh_settings

    window.rstep_spin.setValue(original_settings.rstep + 0.15)
    window.theta_divisions_spin.setValue(
        original_settings.theta_divisions + 10
    )

    assert window._active_mesh_settings == original_settings
    assert "differ from the rendered mesh" in window.pending_mesh_value.text()

    window.update_mesh_button.click()

    assert window._active_mesh_settings.rstep == pytest.approx(
        original_settings.rstep + 0.15
    )
    assert window._active_mesh_settings.theta_divisions == (
        original_settings.theta_divisions + 10
    )
    assert window._active_mesh_geometry.domain_max_radius == pytest.approx(
        window._active_mesh_settings.rmax
    )
    window.close()


def test_main_window_contiguous_notice_tracks_active_center_mode(
    qapp,
    tmp_path,
):
    del qapp
    structure_path = _write_xyz(
        tmp_path / "contiguous_center_notice.xyz",
        [
            "5",
            "contiguous center notice",
            "Pb 0.0 0.0 0.0",
            "Pb 2.0 0.0 0.0",
            "H 0.0 4.0 0.0",
            "H 0.0 6.0 0.0",
            "O 0.0 0.0 2.0",
        ],
    )
    window = ElectronDensityMappingMainWindow(
        initial_input_path=structure_path
    )

    assert "Current active center mode: Geometric Mass Center" in (
        window.contiguous_frame_mode_notice.text()
    )

    window.snap_center_button.click()
    assert (
        "Current active center mode: Nearest atom to geometric mass center"
        in (window.contiguous_frame_mode_notice.text())
    )

    window.snap_reference_center_button.click()
    assert (
        "Current active center mode: Pb reference-element geometric center"
        in (window.contiguous_frame_mode_notice.text())
    )
    window.close()


def test_main_window_pinned_geometric_tracking_requires_pdb_folder_and_geometric_mass_center(
    qapp,
    tmp_path,
):
    del qapp
    single_structure = _write_xyz(
        tmp_path / "single_tracking.xyz",
        [
            "2",
            "single tracking",
            "Pb 0.0 0.0 0.0",
            "O 2.0 0.0 0.0",
        ],
    )
    single_window = ElectronDensityMappingMainWindow(
        initial_input_path=single_structure
    )
    assert not single_window.pinned_geometric_tracking_checkbox.isEnabled()
    single_window.close()

    folder = tmp_path / "pdb_tracking_folder"
    folder.mkdir()
    _write_pb_o_pdb_frame(folder / "frame_0001.pdb", pb_x=0.0, o_x=4.0)
    _write_pb_o_pdb_frame(folder / "frame_0002.pdb", pb_x=0.0, o_x=6.0)
    window = ElectronDensityMappingMainWindow(initial_input_path=folder)

    assert window.pinned_geometric_tracking_checkbox.isEnabled()
    assert window.pinned_geometric_tracking_checkbox.isChecked()
    assert (
        window._mesh_settings_from_controls().pin_contiguous_geometric_tracking
    )

    window.snap_center_button.click()
    assert not window.pinned_geometric_tracking_checkbox.isEnabled()
    assert not window.pinned_geometric_tracking_checkbox.isChecked()
    window.close()


def test_main_window_warns_before_running_with_unconfirmed_mesh_settings(
    qapp,
    tmp_path,
    monkeypatch,
):
    structure_path = _write_xyz(
        tmp_path / "warn_default_mesh.xyz",
        [
            "3",
            "warn default mesh",
            "O 0.0 0.0 0.0",
            "H 0.96 0.0 0.0",
            "H -0.24 0.93 0.0",
        ],
    )
    window = ElectronDensityMappingMainWindow(
        initial_input_path=structure_path
    )
    prompts: list[tuple[str, str]] = []

    def fake_question(*args, **kwargs):
        del kwargs
        prompts.append((str(args[1]), str(args[2])))
        return QMessageBox.StandardButton.No

    monkeypatch.setattr(
        "saxshell.saxs.electron_density_mapping.ui.main_window.QMessageBox.question",
        fake_question,
    )

    window.run_button.click()
    qapp.processEvents()

    assert prompts
    assert prompts[-1][0] == "Mesh Settings Not Updated"
    assert "not updated from the defaults currently shown in the UI" in (
        prompts[-1][1]
    )
    assert "Proceed with the electron-density calculation?" in prompts[-1][1]
    assert window._calculation_thread is None
    assert window._profile_result is None
    window.close()


def test_main_window_mesh_update_skips_defaults_warning_on_run(
    qapp,
    tmp_path,
    monkeypatch,
):
    structure_path = _write_xyz(
        tmp_path / "confirmed_mesh.xyz",
        [
            "3",
            "confirmed mesh",
            "O 0.0 0.0 0.0",
            "H 0.96 0.0 0.0",
            "H -0.24 0.93 0.0",
        ],
    )
    window = ElectronDensityMappingMainWindow(
        initial_input_path=structure_path
    )
    prompts: list[str] = []

    def fake_question(*args, **kwargs):
        del kwargs
        prompts.append(str(args[2]))
        return QMessageBox.StandardButton.Yes

    monkeypatch.setattr(
        "saxshell.saxs.electron_density_mapping.ui.main_window.QMessageBox.question",
        fake_question,
    )

    window.update_mesh_button.click()
    window.run_button.click()
    _wait_for(
        lambda: window._profile_result is not None
        and window._calculation_thread is None,
        qapp,
    )

    assert not prompts
    assert window._profile_result is not None
    window.close()


def test_main_window_preview_mode_marks_title_and_blocks_push(qapp):
    del qapp
    window = ElectronDensityMappingMainWindow(preview_mode=True)

    assert window.windowTitle() == "Electron Density Mapping (Preview)"
    assert "Preview Mode" in window.preview_mode_banner.text()
    assert not window.push_to_model_button.isEnabled()
    assert "Preview mode" in window.push_to_model_status_label.text()
    window.close()


def test_main_window_output_history_defaults_collapsed_and_tracks_outputs(
    qapp,
    tmp_path,
):
    structure_path = _write_xyz(
        tmp_path / "history_capture.xyz",
        [
            "4",
            "history capture",
            "N 0.0 0.0 0.1",
            "H 0.94 0.0 -0.2",
            "H -0.47 0.81 -0.2",
            "H -0.47 -0.81 -0.2",
        ],
    )
    window = ElectronDensityMappingMainWindow(
        initial_input_path=structure_path
    )

    assert not window.output_history_section.is_expanded
    assert window.output_history_table.rowCount() == 0
    assert not window.load_output_history_button.isEnabled()
    assert not window.compare_output_history_button.isEnabled()
    assert (
        window.output_history_table.selectionMode()
        == window.output_history_table.SelectionMode.ExtendedSelection
    )

    window.run_button.click()
    _wait_for(
        lambda: window._profile_result is not None
        and window._calculation_thread is None,
        qapp,
    )

    assert window.output_history_table.rowCount() == 1
    assert window.output_history_table.item(0, 1).text() == "Electron Density"

    window.solvent_method_combo.setCurrentIndex(
        window.solvent_method_combo.findData(CONTRAST_SOLVENT_METHOD_DIRECT)
    )
    window.direct_density_spin.setValue(0.20)
    window.compute_solvent_density_button.click()

    assert window.output_history_table.rowCount() == 2
    assert (
        window.output_history_table.item(0, 1).text() == "Solvent Subtraction"
    )

    window.evaluate_fourier_button.click()

    assert window.output_history_table.rowCount() == 3
    assert window.output_history_table.item(0, 1).text() == "Fourier Transform"
    assert (
        Path(window.output_dir_edit.text())
        / "electron_density_saved_output_history.json"
    ).is_file()
    window.close()


def test_main_window_output_history_restores_preview_entries_in_non_preview_mode(
    qapp,
    tmp_path,
):
    structure_path = _write_xyz(
        tmp_path / "history_restore.xyz",
        [
            "4",
            "history restore",
            "N 0.0 0.0 0.1",
            "H 0.94 0.0 -0.2",
            "H -0.47 0.81 -0.2",
            "H -0.47 -0.81 -0.2",
        ],
    )
    preview_window = ElectronDensityMappingMainWindow(
        initial_input_path=structure_path
    )
    preview_window.run_button.click()
    _wait_for(
        lambda: preview_window._profile_result is not None
        and preview_window._calculation_thread is None,
        qapp,
    )
    preview_window.solvent_method_combo.setCurrentIndex(
        preview_window.solvent_method_combo.findData(
            CONTRAST_SOLVENT_METHOD_DIRECT
        )
    )
    preview_window.direct_density_spin.setValue(0.18)
    preview_window.compute_solvent_density_button.click()
    preview_window.evaluate_fourier_button.click()
    assert preview_window.output_history_table.rowCount() == 3
    preview_window.close()

    restored_window = ElectronDensityMappingMainWindow(
        initial_input_path=structure_path,
        preview_mode=False,
    )
    qapp.processEvents()

    assert restored_window._profile_result is None
    assert restored_window.output_history_table.rowCount() == 3
    assert (
        restored_window.output_history_table.item(0, 1).text()
        == "Fourier Transform"
    )

    restored_window.output_history_table.selectRow(0)
    qapp.processEvents()
    restored_window.load_output_history_button.click()

    assert restored_window._profile_result is not None
    assert restored_window._fourier_result is not None
    assert restored_window._profile_result.solvent_contrast is not None
    restored_window.close()


def test_main_window_push_controls_render_between_fourier_and_saved_outputs(
    qapp,
):
    del qapp
    window = ElectronDensityMappingMainWindow()

    left_layout = window._left_scroll_area.widget().layout()

    assert left_layout.indexOf(window.fourier_section) < left_layout.indexOf(
        window.push_to_model_group
    )
    assert left_layout.indexOf(
        window.push_to_model_group
    ) < left_layout.indexOf(window.output_history_section)
    window.close()


def test_output_history_comparison_dialog_exports_png_and_csv(
    qapp,
    tmp_path,
    monkeypatch,
):
    structure_path = _write_xyz(
        tmp_path / "history_compare.xyz",
        [
            "4",
            "history compare",
            "N 0.0 0.0 0.1",
            "H 0.94 0.0 -0.2",
            "H -0.47 0.81 -0.2",
            "H -0.47 -0.81 -0.2",
        ],
    )
    window = ElectronDensityMappingMainWindow(
        initial_input_path=structure_path
    )
    window.run_button.click()
    _wait_for(
        lambda: window._profile_result is not None
        and window._calculation_thread is None,
        qapp,
    )
    window.solvent_method_combo.setCurrentIndex(
        window.solvent_method_combo.findData(CONTRAST_SOLVENT_METHOD_DIRECT)
    )
    window.direct_density_spin.setValue(0.22)
    window.compute_solvent_density_button.click()
    window.evaluate_fourier_button.click()

    selection_model = window.output_history_table.selectionModel()
    selection_model.clearSelection()
    for row_index in (0, 1):
        selection_model.select(
            window.output_history_table.model().index(row_index, 0),
            QItemSelectionModel.SelectionFlag.Select
            | QItemSelectionModel.SelectionFlag.Rows,
        )
    qapp.processEvents()
    assert window.compare_output_history_button.isEnabled()

    window.compare_output_history_button.click()
    qapp.processEvents()

    dialog = window._output_history_compare_dialog
    assert dialog is not None
    assert len(dialog._entry_plot_widgets) == 2

    png_dir = tmp_path / "png_exports"
    csv_dir = tmp_path / "csv_exports"
    monkeypatch.setattr(
        "saxshell.saxs.electron_density_mapping.ui.main_window.QFileDialog.getExistingDirectory",
        lambda *args, **kwargs: str(png_dir),
    )
    dialog.save_all_png_button.click()

    assert len(list(png_dir.glob("*.png"))) == 10

    monkeypatch.setattr(
        "saxshell.saxs.electron_density_mapping.ui.main_window.QFileDialog.getExistingDirectory",
        lambda *args, **kwargs: str(csv_dir),
    )
    dialog.export_all_csv_button.click()

    assert len(list(csv_dir.glob("*.csv"))) == 2
    dialog.close()
    window.close()


def test_main_window_runs_profile_and_writes_outputs(qapp, tmp_path):
    structure_path = _write_xyz(
        tmp_path / "nh3.xyz",
        [
            "4",
            "ammonia",
            "N 0.0 0.0 0.1",
            "H 0.94 0.0 -0.2",
            "H -0.47 0.81 -0.2",
            "H -0.47 -0.81 -0.2",
        ],
    )
    output_dir = tmp_path / "outputs"
    window = ElectronDensityMappingMainWindow(
        initial_input_path=structure_path
    )
    window.output_dir_edit.setText(str(output_dir))
    window.output_basename_edit.setText("nh3_density")

    window.run_button.click()
    _wait_for(
        lambda: window._profile_result is not None
        and window._calculation_thread is None,
        qapp,
    )

    assert window._profile_result is not None
    assert (output_dir / "nh3_density.csv").is_file()
    assert (output_dir / "nh3_density.json").is_file()
    assert window.profile_plot.current_result is window._profile_result
    assert window.smeared_profile_plot.current_result is window._profile_result
    assert window.profile_plot.figure.axes[0].get_xlabel() == "r (Å)"
    assert window.profile_plot.figure.axes[0].get_ylabel() == "ρ(r) (e/Å³)"
    assert (
        window.smeared_profile_plot.figure.axes[0].get_ylabel()
        == "ρ(r) (e/Å³)"
    )
    assert (
        window._profile_result.smearing_settings.debye_waller_factor
        == pytest.approx(0.006)
    )
    assert window.run_button.isEnabled()
    assert "complete" in window.calculation_progress_message.text().lower()
    window.close()


def test_main_window_folder_run_averages_profiles_and_toggles_variance(
    qapp,
    tmp_path,
):
    folder = tmp_path / "frames"
    folder.mkdir()
    _write_xyz(
        folder / "frame_0001.xyz",
        [
            "3",
            "frame one",
            "O 0.0 0.0 0.0",
            "H 0.96 0.0 0.0",
            "H -0.24 0.93 0.0",
        ],
    )
    _write_xyz(
        folder / "frame_0002.xyz",
        [
            "3",
            "frame two",
            "O 0.0 0.0 0.0",
            "H 1.02 0.0 0.0",
            "H -0.18 0.88 0.0",
        ],
    )
    window = ElectronDensityMappingMainWindow(initial_input_path=folder)

    window.run_button.click()
    _wait_for(
        lambda: window._profile_result is not None
        and window._calculation_thread is None,
        qapp,
    )

    assert window._profile_result.input_mode == "folder"
    assert window._profile_result.source_structure_count == 2
    assert window.contiguous_frame_mode_checkbox.isChecked()
    assert window._profile_result.contiguous_frame_mode_requested
    assert window._profile_result.contiguous_frame_mode_applied
    assert len(window.profile_plot.figure.axes[0].collections) >= 1

    window.show_variance_checkbox.setChecked(False)

    assert len(window.profile_plot.figure.axes[0].collections) == 0
    window.close()


def test_launch_ui_defers_initial_input_loading(qapp, tmp_path):
    structure_path = _write_xyz(
        tmp_path / "deferred_launch.xyz",
        [
            "3",
            "deferred launch",
            "O 0.0 0.0 0.0",
            "H 0.96 0.0 0.0",
            "H -0.24 0.93 0.0",
        ],
    )

    window = launch_electron_density_mapping_ui(
        initial_input_path=structure_path
    )

    try:
        assert window.input_path_edit.text() == str(structure_path.resolve())
        assert window._inspection is None
        assert window._structure is None

        _wait_for(lambda: window._structure is not None, qapp)

        assert window._inspection is not None
        assert window._structure is not None
        assert window._structure.file_path == structure_path.resolve()
    finally:
        window.close()


def test_launch_ui_reports_startup_progress_for_computed_distribution_load(
    qapp,
    tmp_path,
    monkeypatch,
):
    clusters_dir = tmp_path / "clusters"
    cluster_dir = clusters_dir / "2mer"
    cluster_dir.mkdir(parents=True)
    structure_path = _write_xyz(
        cluster_dir / "frame_0001.xyz",
        [
            "3",
            "cluster frame",
            "Pb 0.0 0.0 0.0",
            "I 2.8 0.0 0.0",
            "I 0.0 2.8 0.0",
        ],
    )

    load_started = threading.Event()
    release_load = threading.Event()
    original_load = load_electron_density_structure

    def slow_load(file_path, *args, **kwargs):
        load_started.set()
        assert release_load.wait(timeout=5.0)
        return original_load(file_path, *args, **kwargs)

    monkeypatch.setattr(
        "saxshell.saxs.electron_density_mapping.ui.main_window.load_electron_density_structure",
        slow_load,
    )

    window = launch_electron_density_mapping_ui(
        initial_input_path=clusters_dir,
        preview_mode=False,
    )

    try:
        _wait_for(
            lambda: load_started.is_set()
            and "Loading cluster 1/1"
            in window.calculation_progress_message.text(),
            qapp,
        )

        assert not window.load_input_button.isEnabled()
        assert window.calculation_progress_bar.maximum() == 3
        assert window.calculation_progress_bar.value() == 1
        assert window._cluster_group_states == []
        assert window._workspace_load_progress_dialog is not None
        assert window._workspace_load_progress_dialog.isVisible()
        assert (
            window._workspace_load_progress_dialog.windowTitle()
            == "Loading Electron Density Mapping"
        )
        assert (
            "Loading cluster 1/1"
            in window._workspace_load_progress_dialog.message_label.text()
        )
        assert (
            window._workspace_load_progress_dialog.progress_bar.maximum() == 3
        )
        assert window._workspace_load_progress_dialog.progress_bar.value() == 1

        release_load.set()

        _wait_for(
            lambda: len(window._cluster_group_states) == 1
            and window.load_input_button.isEnabled(),
            qapp,
        )

        assert (
            window._cluster_group_states[0].inspection.reference_file
            == structure_path
        )
        assert "Cluster folders" in window.input_mode_value.text()
        assert window._workspace_load_progress_dialog is not None
        assert not window._workspace_load_progress_dialog.isVisible()
        dialog_output = (
            window._workspace_load_progress_dialog.output_box.toPlainText()
        )
        assert "Discovered 1 stoichiometry folder" in dialog_output
        assert "Loading cluster 1/1" in dialog_output
    finally:
        release_load.set()
        window.close()


def test_load_input_from_edit_reports_progress_for_manual_folder_load(
    qapp,
    tmp_path,
    monkeypatch,
):
    clusters_dir = tmp_path / "clusters"
    cluster_dir = clusters_dir / "2mer"
    cluster_dir.mkdir(parents=True)
    structure_path = _write_xyz(
        cluster_dir / "frame_0001.xyz",
        [
            "3",
            "cluster frame",
            "Pb 0.0 0.0 0.0",
            "I 2.8 0.0 0.0",
            "I 0.0 2.8 0.0",
        ],
    )

    load_started = threading.Event()
    release_load = threading.Event()
    original_load = load_electron_density_structure

    def slow_load(file_path, *args, **kwargs):
        load_started.set()
        assert release_load.wait(timeout=5.0)
        return original_load(file_path, *args, **kwargs)

    monkeypatch.setattr(
        "saxshell.saxs.electron_density_mapping.ui.main_window.load_electron_density_structure",
        slow_load,
    )

    window = ElectronDensityMappingMainWindow(preview_mode=False)

    try:
        window.input_path_edit.setText(str(clusters_dir))
        window._load_input_from_edit()

        _wait_for(
            lambda: load_started.is_set()
            and "Loading cluster 1/1"
            in window.calculation_progress_message.text(),
            qapp,
        )

        assert not window.load_input_button.isEnabled()
        assert window._cluster_group_states == []
        assert window._workspace_load_progress_dialog is not None
        assert window._workspace_load_progress_dialog.isVisible()
        assert (
            "Loading cluster 1/1"
            in window._workspace_load_progress_dialog.message_label.text()
        )

        release_load.set()

        _wait_for(
            lambda: len(window._cluster_group_states) == 1
            and window.load_input_button.isEnabled(),
            qapp,
        )

        assert (
            window._cluster_group_states[0].inspection.reference_file
            == structure_path
        )
        assert window._workspace_load_progress_dialog is not None
        assert not window._workspace_load_progress_dialog.isVisible()
    finally:
        release_load.set()
        window.close()


def test_main_window_can_disable_contiguous_frame_mode(qapp, tmp_path):
    folder = tmp_path / "frames_no_contiguous"
    folder.mkdir()
    _write_xyz(
        folder / "frame_0001_AAA.xyz",
        [
            "2",
            "frame one",
            "Pb 0.0 0.0 0.0",
            "O 2.0 0.0 0.0",
        ],
    )
    _write_xyz(
        folder / "frame_0002_AAA.xyz",
        [
            "2",
            "frame two",
            "Pb 10.0 0.0 0.0",
            "O 13.0 0.0 0.0",
        ],
    )
    window = ElectronDensityMappingMainWindow(initial_input_path=folder)
    assert window.contiguous_frame_mode_checkbox.isChecked()

    window.contiguous_frame_mode_checkbox.setChecked(False)
    window.run_button.click()
    _wait_for(
        lambda: window._profile_result is not None
        and window._calculation_thread is None,
        qapp,
    )

    assert window._profile_result is not None
    assert not window._profile_result.contiguous_frame_mode_requested
    assert not window._profile_result.contiguous_frame_mode_applied
    window.close()


@pytest.mark.parametrize("target_index", [0, 1])
def test_main_window_auto_snap_panes_resizes_clicked_pane(
    qapp,
    target_index,
):
    del qapp
    window = ElectronDensityMappingMainWindow()
    window.resize(1800, 980)
    window.show()
    QApplication.processEvents()

    before_sizes, after_sizes = _pane_snap_resize_result(
        window._pane_splitter,
        window._auto_snap_filter,
        target_index=target_index,
    )

    assert after_sizes[target_index] > before_sizes[target_index] + 20
    window.close()


def test_main_window_auto_snap_setting_defaults_enabled_and_persists(
    qapp,
    monkeypatch,
):
    del qapp

    class _FakeSettings:
        def __init__(self, values: dict[str, object] | None = None):
            self.values = {} if values is None else dict(values)

        def value(self, key, default=None):
            return self.values.get(key, default)

        def setValue(self, key, value):
            self.values[key] = value

    settings_store = _FakeSettings()

    monkeypatch.setattr(
        ElectronDensityMappingMainWindow,
        "_ui_settings",
        lambda self: settings_store,
    )

    first_window = ElectronDensityMappingMainWindow()

    assert first_window.auto_snap_panes_action.isChecked()
    assert first_window._auto_snap_filter.is_enabled()

    first_window.auto_snap_panes_action.trigger()

    assert settings_store.values[AUTO_SNAP_PANES_KEY] is False
    assert not first_window.auto_snap_panes_action.isChecked()
    assert not first_window._auto_snap_filter.is_enabled()

    second_window = ElectronDensityMappingMainWindow()

    assert not second_window.auto_snap_panes_action.isChecked()
    assert not second_window._auto_snap_filter.is_enabled()

    first_window.close()
    second_window.close()


@pytest.mark.parametrize("target_index", [0, 1])
def test_main_window_auto_snap_focuses_clicked_pane_when_its_preferred_width_is_small(
    qapp,
    target_index,
):
    del qapp
    window = ElectronDensityMappingMainWindow()
    window.resize(1800, 980)
    window.show()
    QApplication.processEvents()

    before_sizes, after_sizes = _pane_snap_resize_result(
        window._pane_splitter,
        window._auto_snap_filter,
        target_index=target_index,
        desired_width=220,
        other_width=720,
    )

    assert after_sizes[target_index] > before_sizes[target_index] + 20
    assert after_sizes[1 - target_index] < before_sizes[1 - target_index] - 20
    window.close()


def test_main_window_reports_contiguous_frame_fallback(qapp, tmp_path):
    folder = tmp_path / "frames_fallback"
    folder.mkdir()
    _write_xyz(
        folder / "cluster_a.xyz",
        [
            "2",
            "cluster a",
            "Pb 0.0 0.0 0.0",
            "O 2.0 0.0 0.0",
        ],
    )
    _write_xyz(
        folder / "cluster_b.xyz",
        [
            "2",
            "cluster b",
            "Pb 0.0 0.0 0.0",
            "O 3.0 0.0 0.0",
        ],
    )
    window = ElectronDensityMappingMainWindow(initial_input_path=folder)
    assert window.contiguous_frame_mode_checkbox.isChecked()

    window.run_button.click()
    _wait_for(
        lambda: window._profile_result is not None
        and window._calculation_thread is None,
        qapp,
    )

    assert window._profile_result is not None
    assert window._profile_result.contiguous_frame_mode_requested
    assert not window._profile_result.contiguous_frame_mode_applied
    assert "Falling back to complete averaging" in (
        window.status_text.toPlainText()
    )
    window.close()


def test_main_window_evaluates_fourier_transform_and_toggles_log_axes(
    qapp,
    tmp_path,
):
    structure_path = _write_xyz(
        tmp_path / "fourier_ui.xyz",
        [
            "4",
            "fourier ui",
            "N 0.0 0.0 0.1",
            "H 0.94 0.0 -0.2",
            "H -0.47 0.81 -0.2",
            "H -0.47 -0.81 -0.2",
        ],
    )
    window = ElectronDensityMappingMainWindow(
        initial_input_path=structure_path
    )

    window.run_button.click()
    _wait_for(
        lambda: window._profile_result is not None
        and window._calculation_thread is None,
        qapp,
    )

    assert window.fourier_preview_plot.current_preview is not None
    assert window._fourier_result is None

    window.fourier_qmin_spin.setValue(0.05)
    window.fourier_qmax_spin.setValue(6.0)
    window.fourier_qstep_spin.setValue(0.05)
    window.evaluate_fourier_button.click()

    assert window._fourier_result is not None
    assert window.scattering_plot.current_result is window._fourier_result
    scattering_axis = window.scattering_plot.figure.axes[0]
    assert scattering_axis.get_xlabel() == "q (Å⁻¹)"
    assert scattering_axis.get_xscale() == "log"
    assert scattering_axis.get_yscale() == "log"

    window.fourier_log_q_checkbox.setChecked(False)
    window.fourier_log_intensity_checkbox.setChecked(False)

    scattering_axis = window.scattering_plot.figure.axes[0]
    assert scattering_axis.get_xscale() == "linear"
    assert scattering_axis.get_yscale() == "linear"
    window.close()


def test_main_window_uses_updated_fourier_defaults_and_inherited_q_range(
    qapp,
):
    window = ElectronDensityMappingMainWindow()
    assert window.fourier_qmin_spin.value() == pytest.approx(0.02)
    assert window.fourier_qmax_spin.value() == pytest.approx(1.2)
    assert window.fourier_qstep_spin.value() == pytest.approx(0.01)
    assert window.fourier_resampling_points_spin.value() == 2048

    inherited_window = ElectronDensityMappingMainWindow(
        initial_project_q_min=0.15,
        initial_project_q_max=0.9,
    )
    assert inherited_window.fourier_qmin_spin.value() == pytest.approx(0.15)
    assert inherited_window.fourier_qmax_spin.value() == pytest.approx(0.9)
    assert inherited_window.fourier_qstep_spin.value() == pytest.approx(0.01)
    assert inherited_window.fourier_resampling_points_spin.value() == 2048

    window.close()
    inherited_window.close()


def test_main_window_keeps_fourier_controls_editable_without_cluster_groups(
    qapp,
):
    window = ElectronDensityMappingMainWindow()

    assert not window.apply_fourier_to_all_button.isEnabled()
    assert (
        window.fourier_scope_status_label.text()
        == "Batch Fourier editing becomes available in cluster-folder mode."
    )
    assert window.fourier_rmax_spin.isEnabled()
    assert window.fourier_qmin_spin.isEnabled()
    assert window.fourier_qmax_spin.isEnabled()
    assert window.fourier_qstep_spin.isEnabled()
    assert window.fourier_window_combo.isEnabled()
    assert window.fourier_resampling_points_spin.isEnabled()
    assert window.fourier_use_solvent_subtracted_checkbox.isEnabled()
    assert window.fourier_legacy_mode_checkbox.isEnabled()
    assert not window.fourier_rmin_spin.isEnabled()

    window.fourier_legacy_mode_checkbox.setChecked(True)
    qapp.processEvents()

    assert window.fourier_rmin_spin.isEnabled()
    window.close()


def test_main_window_defaults_to_mirrored_fourier_domain_and_can_toggle_legacy(
    qapp,
):
    window = ElectronDensityMappingMainWindow()
    window.apply_fourier_to_all_button.setChecked(False)
    qapp.processEvents()

    assert not window.fourier_legacy_mode_checkbox.isChecked()
    assert window.fourier_rmin_label.text() == "-r max"
    assert not window.fourier_rmin_spin.isEnabled()

    window.fourier_rmax_spin.setValue(3.25)
    qapp.processEvents()
    assert window.fourier_rmin_spin.value() == pytest.approx(-3.25)

    window.fourier_legacy_mode_checkbox.setChecked(True)
    qapp.processEvents()
    assert window.fourier_rmin_label.text() == "r min"
    assert window.fourier_rmin_spin.isEnabled()
    assert window.fourier_rmin_spin.value() == pytest.approx(0.0)

    window.close()


def test_main_window_exposes_centered_exafs_window_options(qapp):
    del qapp
    window = ElectronDensityMappingMainWindow()
    window_names = {
        str(window.fourier_window_combo.itemData(index))
        for index in range(window.fourier_window_combo.count())
    }

    assert "kaiser_bessel" in window_names
    assert "hanning" in window_names
    assert window.fourier_window_combo.currentData() == "hanning"
    window.close()


def test_cluster_group_loading_uses_fast_atom_count_scan(
    qapp,
    tmp_path,
    monkeypatch,
):
    del qapp
    clusters_dir = tmp_path / "clusters"
    group_dir = clusters_dir / "PbI2"
    group_dir.mkdir(parents=True)
    _write_xyz(
        group_dir / "frame_0001.xyz",
        [
            "3",
            "frame one",
            "Pb 0.0 0.0 0.0",
            "I 2.8 0.0 0.0",
            "I 0.0 2.8 0.0",
        ],
    )
    _write_xyz(
        group_dir / "frame_0002.xyz",
        [
            "4",
            "frame two",
            "Pb 0.0 0.0 0.0",
            "I 2.8 0.0 0.0",
            "I 0.0 2.8 0.0",
            "I 0.0 0.0 2.8",
        ],
    )
    _write_xyz(
        group_dir / "frame_0003.xyz",
        [
            "5",
            "frame three",
            "Pb 0.0 0.0 0.0",
            "I 2.8 0.0 0.0",
            "I 0.0 2.8 0.0",
            "I 0.0 0.0 2.8",
            "I -2.8 0.0 0.0",
        ],
    )

    load_calls: list[str] = []

    def fake_load(file_path, *args, **kwargs):
        del args, kwargs
        resolved = Path(file_path).expanduser().resolve()
        load_calls.append(resolved.name)
        return SimpleNamespace(
            file_path=resolved,
            atom_count=99,
        )

    monkeypatch.setattr(
        "saxshell.saxs.electron_density_mapping.ui.main_window.load_electron_density_structure",
        fake_load,
    )

    window = ElectronDensityMappingMainWindow()

    states = window._cluster_group_states_for_path(clusters_dir)

    assert len(states) == 1
    assert states[0].average_atom_count == pytest.approx(4.0)
    assert load_calls == ["frame_0001.xyz"]
    window.close()


def test_cluster_folder_single_atom_groups_use_direct_debye_scattering(
    qapp,
    tmp_path,
):
    del qapp
    clusters_dir = tmp_path / "clusters"
    single_dir = clusters_dir / "I"
    multi_dir = clusters_dir / "PbI2"
    single_dir.mkdir(parents=True)
    multi_dir.mkdir(parents=True)
    _write_xyz(
        single_dir / "frame_0001.xyz",
        [
            "1",
            "single iodine 1",
            "I 0.0 0.0 0.0",
        ],
    )
    _write_xyz(
        single_dir / "frame_0002.xyz",
        [
            "1",
            "single iodine 2",
            "I 1.0 2.0 3.0",
        ],
    )
    _write_xyz(
        multi_dir / "frame_0001.xyz",
        [
            "3",
            "PbI2 cluster",
            "Pb 0.0 0.0 0.0",
            "I 2.8 0.0 0.0",
            "I 0.0 2.8 0.0",
        ],
    )

    window = ElectronDensityMappingMainWindow(initial_input_path=clusters_dir)

    assert any(
        state.key == "I" and state.single_atom_only
        for state in window._cluster_group_states
    )

    window.run_button.click()
    _wait_for(
        lambda: all(
            (
                state.single_atom_only
                and state.profile_result is None
                and state.transform_result is not None
            )
            or (
                not state.single_atom_only and state.profile_result is not None
            )
            for state in window._cluster_group_states
        )
        and window._calculation_thread is None,
        QApplication.instance(),
    )

    states_by_key = {
        state.key: state for state in window._cluster_group_states
    }
    log_text = window.status_text.toPlainText()

    assert states_by_key["I"].transform_result is not None
    assert states_by_key["I"].profile_result is None
    assert states_by_key["I"].transform_result.preview.source_mode == (
        "single_atom_debye"
    )
    assert states_by_key["PbI2"].profile_result is not None
    assert (
        "Preparing single-atom Debye scattering calculation." not in log_text
    )
    assert "Validating single-atom structure" not in log_text
    assert "Computing single-atom Debye scattering from " not in log_text
    assert "Single-atom Debye scattering profile ready." not in log_text

    row_by_key = {
        window.cluster_group_table.item(row_index, 0).text(): row_index
        for row_index in range(window.cluster_group_table.rowCount())
    }
    assert window.cluster_group_table.item(row_by_key["I"], 6).text() == (
        "Skipped (Debye)"
    )
    assert window.cluster_group_table.item(row_by_key["I"], 7).text() == (
        "Ready (Debye)"
    )

    window.cluster_group_table.selectRow(row_by_key["I"])
    QApplication.instance().processEvents()

    assert window._profile_result is None
    assert window._fourier_result is states_by_key["I"].transform_result
    assert (
        "direct debye scattering" in window.fourier_notice_value.text().lower()
    )
    assert (
        window.scattering_plot.current_result
        is states_by_key["I"].transform_result
    )

    window.fourier_qmax_spin.setValue(1.0)
    window.evaluate_fourier_button.click()

    assert states_by_key["I"].transform_result is not None
    assert states_by_key["I"].transform_result.preview.source_mode == (
        "single_atom_debye"
    )
    assert states_by_key["I"].transform_result.q_values[-1] == pytest.approx(
        1.0
    )
    window.close()


def test_cluster_folder_apply_to_all_fourier_updates_single_atom_debye_rows_with_shared_q_grid(
    qapp,
    tmp_path,
):
    clusters_dir = tmp_path / "clusters"
    single_dir = clusters_dir / "I"
    multi_dir = clusters_dir / "PbI2"
    single_dir.mkdir(parents=True)
    multi_dir.mkdir(parents=True)
    _write_xyz(
        single_dir / "frame_0001.xyz",
        [
            "1",
            "single iodine 1",
            "I 0.0 0.0 0.0",
        ],
    )
    _write_xyz(
        single_dir / "frame_0002.xyz",
        [
            "1",
            "single iodine 2",
            "I 1.0 2.0 3.0",
        ],
    )
    _write_xyz(
        multi_dir / "frame_0001.xyz",
        [
            "3",
            "PbI2 cluster",
            "Pb 0.0 0.0 0.0",
            "I 2.8 0.0 0.0",
            "I 0.0 2.8 0.0",
        ],
    )

    window = ElectronDensityMappingMainWindow(initial_input_path=clusters_dir)

    _wait_for(
        lambda: window.cluster_group_table.rowCount() == 2,
        qapp,
    )
    window.run_button.click()
    _wait_for(
        lambda: window._calculation_thread is None
        and all(
            (state.single_atom_only and state.transform_result is not None)
            or (
                not state.single_atom_only and state.profile_result is not None
            )
            for state in window._cluster_group_states
        ),
        qapp,
    )

    window.fourier_qmin_spin.setValue(0.1)
    window.fourier_qmax_spin.setValue(0.5)
    window.fourier_qstep_spin.setValue(0.2)
    window.evaluate_fourier_button.click()

    expected_q = np.asarray([0.1, 0.3, 0.5], dtype=float)
    states_by_key = {
        state.key: state for state in window._cluster_group_states
    }
    row_by_key = {
        window.cluster_group_table.item(row_index, 0).text(): row_index
        for row_index in range(window.cluster_group_table.rowCount())
    }

    assert states_by_key["I"].transform_result is not None
    assert states_by_key["I"].transform_result.preview.source_mode == (
        "single_atom_debye"
    )
    np.testing.assert_allclose(
        states_by_key["I"].transform_result.q_values,
        expected_q,
    )
    assert states_by_key["PbI2"].profile_result is not None
    assert states_by_key["PbI2"].transform_result is not None
    assert window.fourier_settings_table.item(row_by_key["I"], 1).text() == (
        "Ready (Debye)"
    )
    assert (
        window.fourier_settings_table.item(row_by_key["PbI2"], 1).text()
        == "Ready"
    )
    window.close()


def test_batch_smearing_shows_progress_popup(
    qapp,
    tmp_path,
    monkeypatch,
):
    window = _open_ready_cluster_folder_window(
        qapp,
        tmp_path,
        folder_name="batch_smearing_progress",
    )
    window.apply_smearing_to_all_button.setChecked(True)
    window.smearing_factor_spin.setValue(0.04)

    seen_messages: list[str] = []

    def wrapped_apply(result, settings):
        dialog = _assert_batch_progress_dialog_visible(
            window,
            title="Applying Gaussian Smearing",
            total=len(window._cluster_group_states),
        )
        seen_messages.append(dialog.message_label.text())
        return apply_smearing_to_profile_result(result, settings)

    monkeypatch.setattr(
        "saxshell.saxs.electron_density_mapping.ui.main_window.apply_smearing_to_profile_result",
        wrapped_apply,
    )

    window.apply_smearing_button.click()

    assert len(seen_messages) == len(window._cluster_group_states)
    assert any(
        "Applying Gaussian smearing to" in message for message in seen_messages
    )
    assert window._batch_operation_progress_dialog is not None
    assert not window._batch_operation_progress_dialog.isVisible()
    assert "complete" in window.calculation_progress_message.text().lower()
    window.close()


def test_batch_solvent_contrast_shows_progress_popup(
    qapp,
    tmp_path,
    monkeypatch,
):
    window = _open_ready_cluster_folder_window(
        qapp,
        tmp_path,
        folder_name="batch_contrast_progress",
    )
    base_profile = next(
        state.profile_result
        for state in window._cluster_group_states
        if state.profile_result is not None
    )
    direct_density = float(
        min(
            np.max(
                np.asarray(
                    base_profile.smeared_orientation_average_density,
                    dtype=float,
                )
            )
            * 0.01,
            50.0,
        )
    )
    window.solvent_method_combo.setCurrentIndex(
        window.solvent_method_combo.findData(CONTRAST_SOLVENT_METHOD_DIRECT)
    )
    window.direct_density_spin.setValue(direct_density)
    window.apply_contrast_to_all_button.setChecked(True)

    seen_messages: list[str] = []

    def wrapped_apply(result, settings, *, solvent_name=None):
        dialog = _assert_batch_progress_dialog_visible(
            window,
            title="Applying Electron Density Contrast",
            total=len(window._cluster_group_states),
        )
        seen_messages.append(dialog.message_label.text())
        return apply_solvent_contrast_to_profile_result(
            result,
            settings,
            solvent_name=solvent_name,
        )

    monkeypatch.setattr(
        "saxshell.saxs.electron_density_mapping.ui.main_window.apply_solvent_contrast_to_profile_result",
        wrapped_apply,
    )

    window.compute_solvent_density_button.click()

    assert len(seen_messages) == len(window._cluster_group_states)
    assert any(
        "Applying solvent subtraction to" in message
        for message in seen_messages
    )
    assert window._batch_operation_progress_dialog is not None
    assert not window._batch_operation_progress_dialog.isVisible()
    assert "complete" in window.calculation_progress_message.text().lower()
    window.close()


def test_batch_fourier_evaluation_shows_progress_popup(
    qapp,
    tmp_path,
    monkeypatch,
):
    window = _open_ready_cluster_folder_window(
        qapp,
        tmp_path,
        folder_name="batch_fourier_progress",
    )
    window.apply_fourier_to_all_button.setChecked(True)

    seen_messages: list[str] = []

    def wrapped_compute(result, settings):
        dialog = _assert_batch_progress_dialog_visible(
            window,
            title="Evaluating Fourier Transforms",
            total=len(window._cluster_group_states),
        )
        seen_messages.append(dialog.message_label.text())
        return compute_electron_density_scattering_profile(
            result,
            settings,
        )

    monkeypatch.setattr(
        "saxshell.saxs.electron_density_mapping.ui.main_window.compute_electron_density_scattering_profile",
        wrapped_compute,
    )

    window.evaluate_fourier_button.click()

    assert len(seen_messages) == len(window._cluster_group_states)
    assert any(
        "Evaluating Fourier transform for" in message
        for message in seen_messages
    )
    assert all(
        state.transform_result is not None
        for state in window._cluster_group_states
    )
    assert window._batch_operation_progress_dialog is not None
    assert not window._batch_operation_progress_dialog.isVisible()
    assert "complete" in window.calculation_progress_message.text().lower()
    window.close()


def test_cluster_row_selection_preserves_batch_fourier_transforms(
    qapp,
    tmp_path,
):
    clusters_dir = tmp_path / "batch_fourier_selection"
    clusters_dir.mkdir()
    structures = {
        "PbI2": [
            "3",
            "PbI2 frame 1",
            "Pb 0.0 0.0 0.0",
            "I 2.8 0.0 0.0",
            "I 0.0 2.8 0.0",
        ],
        "PbI3": [
            "4",
            "PbI3 frame 1",
            "Pb 0.0 0.0 0.0",
            "I 2.8 0.0 0.0",
            "I 0.0 2.8 0.0",
            "I 0.0 0.0 2.8",
        ],
        "CsI": [
            "2",
            "CsI frame 1",
            "Cs 0.0 0.0 0.0",
            "I 3.2 0.0 0.0",
        ],
    }
    for name, lines in structures.items():
        stoichiometry_dir = clusters_dir / name
        stoichiometry_dir.mkdir(parents=True)
        _write_xyz(stoichiometry_dir / "frame_0001.xyz", lines)

    window = ElectronDensityMappingMainWindow(initial_input_path=clusters_dir)

    _wait_for(
        lambda: window.cluster_group_table.rowCount() == len(structures),
        qapp,
    )
    window.run_button.click()
    _wait_for(
        lambda: window._calculation_thread is None
        and all(
            state.profile_result is not None
            for state in window._cluster_group_states
        ),
        qapp,
    )

    window.apply_fourier_to_all_button.setChecked(True)
    window.evaluate_fourier_button.click()
    qapp.processEvents()

    assert all(
        state.transform_result is not None
        for state in window._cluster_group_states
    )

    window.apply_debye_to_all_button.setChecked(True)
    qapp.processEvents()
    assert "0/3 target rows" in window.debye_scattering_status_label.text()

    for row_index, expected_state in enumerate(window._cluster_group_states):
        window.cluster_group_table.selectRow(row_index)
        qapp.processEvents()

        active_state = window._active_cluster_group_state()
        assert active_state is not None
        assert active_state.key == expected_state.key
        assert active_state.transform_result is not None
        assert window._fourier_result is active_state.transform_result
        assert (
            window.scattering_plot.current_result
            is active_state.transform_result
        )
        assert window.cluster_group_table.item(row_index, 7).text() == "Ready"
        assert window.fourier_settings_table.item(row_index, 1).text() == (
            "Ready"
        )
        assert all(
            state.transform_result is not None
            for state in window._cluster_group_states
        )
        assert "0/3 target rows" in (
            window.debye_scattering_status_label.text()
        )

    window.close()


def test_cluster_folder_load_defaults_batch_apply_to_all_scopes(
    qapp,
    tmp_path,
):
    window = ElectronDensityMappingMainWindow(
        initial_input_path=_write_cluster_folder_input(
            tmp_path / "default_batch_scope_clusters"
        )
    )

    _wait_for(
        lambda: window.cluster_group_table.rowCount() == 2,
        qapp,
    )

    assert window.apply_smearing_to_all_button.isChecked()
    assert window.apply_contrast_to_all_button.isChecked()
    assert window.apply_fourier_to_all_button.isChecked()
    assert not window.apply_debye_to_all_button.isChecked()

    window.close()


def test_main_window_debye_scattering_pane_builds_cluster_comparison_plot(
    qapp,
    tmp_path,
):
    clusters_dir = _write_cluster_folder_input(tmp_path / "debye_compare")
    window = ElectronDensityMappingMainWindow(initial_input_path=clusters_dir)

    window.run_button.click()
    _wait_for(
        lambda: all(
            state.profile_result is not None
            for state in window._cluster_group_states
        )
        and window._calculation_thread is None,
        qapp,
    )

    window.apply_fourier_to_all_button.setChecked(True)
    window.evaluate_fourier_button.click()
    qapp.processEvents()

    assert window.calculate_debye_scattering_button.isEnabled()
    window.apply_debye_to_all_button.setChecked(True)
    window.calculate_debye_scattering_button.click()
    qapp.processEvents()

    assert all(
        state.debye_scattering_result is not None
        for state in window._cluster_group_states
        if not state.single_atom_only
    )
    for state in window._cluster_group_states:
        if state.single_atom_only:
            continue
        assert state.transform_result is not None
        assert state.debye_scattering_result is not None
        np.testing.assert_allclose(
            state.debye_scattering_result.q_values,
            state.transform_result.q_values,
        )

    assert window.open_debye_scattering_compare_button.isEnabled()
    window.open_debye_scattering_compare_button.click()
    qapp.processEvents()

    dialog = window._debye_scattering_compare_dialog
    assert dialog is not None
    assert dialog._trace_table.rowCount() == 2
    assert dialog.autoscale_button.isChecked()
    assert len(dialog._plot_widget.figure.axes) == 2
    born_axis, debye_axis = dialog._plot_widget.figure.axes
    assert "Born Approximation" in born_axis.get_ylabel()
    assert "Debye Scattering" in debye_axis.get_ylabel()

    dialog.close()
    window.close()


def test_main_window_debye_scattering_progress_bar_updates_for_single_run(
    qapp,
    tmp_path,
    monkeypatch,
):
    structure_path = _write_xyz(
        tmp_path / "debye_progress_single.xyz",
        [
            "3",
            "debye progress single",
            "Pb 0.0 0.0 0.0",
            "I 2.8 0.0 0.0",
            "I 0.0 2.8 0.0",
        ],
    )
    window = ElectronDensityMappingMainWindow(
        initial_input_path=structure_path
    )

    window.run_button.click()
    _wait_for(
        lambda: window._profile_result is not None
        and window._calculation_thread is None,
        qapp,
    )
    window.evaluate_fourier_button.click()
    _wait_for(
        lambda: window._fourier_result is not None
        and window.calculate_debye_scattering_button.isEnabled(),
        qapp,
    )

    progress_snapshots: list[tuple[bool, int, int, str]] = []

    def wrapped_compute(*args, progress_callback=None, **kwargs):
        assert progress_callback is not None

        def wrapped_progress(current, total, message):
            progress_callback(current, total, message)
            progress_snapshots.append(
                (
                    not window.debye_scattering_progress_bar.isHidden(),
                    window.debye_scattering_progress_bar.value(),
                    window.debye_scattering_progress_bar.maximum(),
                    window.debye_scattering_status_label.text(),
                )
            )

        return compute_average_debye_scattering_profile_for_input(
            *args,
            progress_callback=wrapped_progress,
            **kwargs,
        )

    monkeypatch.setattr(
        "saxshell.saxs.electron_density_mapping.ui.main_window.compute_average_debye_scattering_profile_for_input",
        wrapped_compute,
    )

    window.calculate_debye_scattering_button.click()

    assert progress_snapshots
    assert any(visible for visible, *_rest in progress_snapshots)
    assert any(
        maximum == 4 for _visible, _value, maximum, _text in progress_snapshots
    )
    assert any(
        "Debye scattering average calculation" in text or "Debye trace" in text
        for _visible, _value, _maximum, text in progress_snapshots
    )
    assert window.debye_scattering_progress_bar.isHidden()
    assert window._debye_scattering_result is not None
    window.close()


def test_main_window_debye_scattering_progress_bar_updates_for_batch_run(
    qapp,
    tmp_path,
    monkeypatch,
):
    window = _open_ready_cluster_folder_window(
        qapp,
        tmp_path,
        folder_name="debye_progress_batch",
    )
    window.apply_fourier_to_all_button.setChecked(True)
    window.evaluate_fourier_button.click()
    qapp.processEvents()

    progress_snapshots: list[tuple[int, int, str]] = []

    def wrapped_compute(*args, progress_callback=None, **kwargs):
        assert progress_callback is not None

        def wrapped_progress(current, total, message):
            progress_callback(current, total, message)
            progress_snapshots.append(
                (
                    window.debye_scattering_progress_bar.value(),
                    window.debye_scattering_progress_bar.maximum(),
                    window.debye_scattering_status_label.text(),
                )
            )

        return compute_average_debye_scattering_profile_for_input(
            *args,
            progress_callback=wrapped_progress,
            **kwargs,
        )

    monkeypatch.setattr(
        "saxshell.saxs.electron_density_mapping.ui.main_window.compute_average_debye_scattering_profile_for_input",
        wrapped_compute,
    )

    window.apply_debye_to_all_button.setChecked(True)
    window.calculate_debye_scattering_button.click()

    assert progress_snapshots
    assert any(maximum == 8 for _value, maximum, _text in progress_snapshots)
    assert any(
        "Debye 1/2" in text for _value, _maximum, text in progress_snapshots
    )
    assert any(
        "Debye 2/2" in text for _value, _maximum, text in progress_snapshots
    )
    assert window.debye_scattering_progress_bar.isHidden()
    assert all(
        state.debye_scattering_result is not None
        for state in window._cluster_group_states
    )
    window.close()


def test_debye_scattering_comparison_dialog_plots_compatible_born_and_debye_traces(
    qapp,
    tmp_path,
):
    born_path = _write_xyz(
        tmp_path / "born.xyz",
        [
            "3",
            "born cluster",
            "Pb 0.0 0.0 0.0",
            "I 2.8 0.0 0.0",
            "I 0.0 2.8 0.0",
        ],
    )
    born_structure = load_electron_density_structure(born_path)
    born_profile = compute_electron_density_profile(
        born_structure,
        _mesh_settings_for_structure(
            born_structure,
            rstep=0.2,
            theta_divisions=24,
            phi_divisions=18,
            rmax=3.5,
        ),
    )
    born_settings = ElectronDensityFourierTransformSettings(
        r_min=float(born_profile.radial_centers[0]),
        r_max=float(born_profile.radial_centers[-1]),
        q_min=0.1,
        q_max=0.5,
        q_step=0.2,
        resampling_points=32,
        use_solvent_subtracted_profile=False,
    )
    born_result = compute_electron_density_scattering_profile(
        born_profile,
        born_settings,
    )

    debye_folder = tmp_path / "debye"
    debye_folder.mkdir()
    _write_xyz(
        debye_folder / "frame_0001.xyz",
        [
            "1",
            "single iodine 1",
            "I 0.0 0.0 0.0",
        ],
    )
    _write_xyz(
        debye_folder / "frame_0002.xyz",
        [
            "1",
            "single iodine 2",
            "I 1.0 2.0 3.0",
        ],
    )
    debye_result = compute_average_debye_scattering_profile_for_input(
        inspect_structure_input(debye_folder),
        born_settings,
    )

    dialog = _DebyeScatteringComparisonDialog(
        [
            _DebyeComparisonEntry(
                entry_id="pair-1",
                label="PbI2",
                color="#2563eb",
                born_result=born_result,
                debye_result=debye_result,
                info_text="Compatible Born and Debye traces",
            )
        ]
    )

    assert dialog._trace_table.rowCount() == 1
    assert dialog._visible_entries()[0].entry_id == "pair-1"
    assert "Overlaying 1 visible trace pair" in dialog.status_label.text()
    assert len(dialog._plot_widget.figure.axes) == 2
    born_axis, debye_axis = dialog._plot_widget.figure.axes
    assert born_axis.lines[0].get_linestyle() == "-"
    assert debye_axis.lines[0].get_linestyle() == "--"
    dialog.close()


def test_main_window_smearing_commit_updates_smeared_profile(
    qapp,
    tmp_path,
):
    structure_path = _write_xyz(
        tmp_path / "smearing_commit.xyz",
        [
            "4",
            "smearing commit",
            "N 0.0 0.0 0.1",
            "H 0.94 0.0 -0.2",
            "H -0.47 0.81 -0.2",
            "H -0.47 -0.81 -0.2",
        ],
    )
    window = ElectronDensityMappingMainWindow(
        initial_input_path=structure_path
    )

    window.run_button.click()
    _wait_for(
        lambda: window._profile_result is not None
        and window._calculation_thread is None,
        qapp,
    )

    original_result = window._profile_result
    assert original_result is not None
    original_raw = np.asarray(
        original_result.orientation_average_density,
        dtype=float,
    ).copy()
    original_smeared = np.asarray(
        original_result.smeared_orientation_average_density,
        dtype=float,
    ).copy()

    window.smearing_factor_spin.lineEdit().setText("0.040000")
    window.smearing_factor_spin.interpretText()
    qapp.processEvents()

    updated_result = window._profile_result
    assert updated_result is not None
    assert (
        updated_result.smearing_settings.debye_waller_factor
        == pytest.approx(0.04)
    )
    assert np.allclose(
        updated_result.orientation_average_density, original_raw
    )
    assert not np.allclose(
        updated_result.smeared_orientation_average_density,
        original_smeared,
    )
    assert window.smeared_profile_plot.current_result is updated_result
    assert window.profile_plot.current_result is updated_result
    expected_raw_total = float(
        np.sum(
            np.asarray(updated_result.orientation_average_density, dtype=float)
            * np.asarray(updated_result.shell_volumes, dtype=float)
        )
    )
    expected_smeared_total = float(
        np.sum(
            np.asarray(
                updated_result.smeared_orientation_average_density,
                dtype=float,
            )
            * np.asarray(updated_result.shell_volumes, dtype=float)
        )
    )
    assert (
        window.profile_plot.current_integrated_electron_count
        == pytest.approx(expected_raw_total)
    )
    assert (
        window.smeared_profile_plot.current_integrated_electron_count
        == pytest.approx(expected_smeared_total)
    )
    raw_texts = [
        text.get_text() for text in window.profile_plot.figure.axes[0].texts
    ]
    smeared_texts = [
        text.get_text()
        for text in window.smeared_profile_plot.figure.axes[0].texts
    ]
    assert any("Integrated electrons:" in text for text in raw_texts)
    assert any(f"{expected_raw_total:.6f}" in text for text in raw_texts)
    assert any("Integrated electrons:" in text for text in smeared_texts)
    assert any(
        f"{expected_smeared_total:.6f}" in text for text in smeared_texts
    )
    window.close()


def test_main_window_smearing_autosave_adds_reloadable_saved_output_entries(
    qapp,
    tmp_path,
):
    structure_path = _write_xyz(
        tmp_path / "smearing_autosave.xyz",
        [
            "4",
            "smearing autosave",
            "N 0.0 0.0 0.1",
            "H 0.94 0.0 -0.2",
            "H -0.47 0.81 -0.2",
            "H -0.47 -0.81 -0.2",
        ],
    )
    window = ElectronDensityMappingMainWindow(
        initial_input_path=structure_path
    )

    window.run_button.click()
    _wait_for(
        lambda: window._profile_result is not None
        and window._calculation_thread is None,
        qapp,
    )

    assert window.output_history_table.rowCount() == 1
    assert window.output_history_table.item(0, 1).text() == "Electron Density"

    window.smearing_factor_spin.lineEdit().setText("0.040000")
    window.smearing_factor_spin.interpretText()
    qapp.processEvents()

    assert window.output_history_table.rowCount() == 1
    assert window._saved_output_entries[-1].entry_kind == "density"

    window.auto_save_smearing_outputs_checkbox.setChecked(True)
    qapp.processEvents()
    window.smearing_factor_spin.lineEdit().setText("0.020000")
    window.smearing_factor_spin.interpretText()
    qapp.processEvents()

    assert window.output_history_table.rowCount() == 2
    assert window.output_history_table.item(0, 1).text() == "Smearing"
    assert window._saved_output_entries[-1].entry_kind == "smearing"
    assert window._saved_output_entries[
        -1
    ].profile_result.smearing_settings.debye_waller_factor == pytest.approx(
        0.02
    )

    window.output_history_table.selectRow(0)
    qapp.processEvents()
    assert window.load_output_history_button.isEnabled()
    window.load_output_history_button.click()
    qapp.processEvents()

    assert window._profile_result is not None
    assert (
        window._profile_result.smearing_settings.debye_waller_factor
        == pytest.approx(0.02)
    )
    assert window.smearing_factor_spin.value() == pytest.approx(0.02)

    window.auto_save_smearing_outputs_checkbox.setChecked(False)
    qapp.processEvents()
    window.smearing_factor_spin.lineEdit().setText("0.010000")
    window.smearing_factor_spin.interpretText()
    qapp.processEvents()

    assert window.output_history_table.rowCount() == 2
    window.close()


def test_main_window_computes_solvent_contrast_and_updates_fourier_source(
    qapp,
    tmp_path,
):
    structure_path = _write_xyz(
        tmp_path / "solvent_ui.xyz",
        [
            "4",
            "solvent ui",
            "N 0.0 0.0 0.1",
            "H 0.94 0.0 -0.2",
            "H -0.47 0.81 -0.2",
            "H -0.47 -0.81 -0.2",
        ],
    )
    window = ElectronDensityMappingMainWindow(
        initial_input_path=structure_path
    )

    window.run_button.click()
    _wait_for(
        lambda: window._profile_result is not None
        and window._calculation_thread is None,
        qapp,
    )

    assert window._profile_result is not None
    direct_density = float(
        min(
            np.max(
                np.asarray(
                    window._profile_result.smeared_orientation_average_density,
                    dtype=float,
                )
            )
            * 0.01,
            50.0,
        )
    )
    window.solvent_method_combo.setCurrentIndex(
        window.solvent_method_combo.findData(CONTRAST_SOLVENT_METHOD_DIRECT)
    )
    window.direct_density_spin.setValue(direct_density)
    window.compute_solvent_density_button.click()
    qapp.processEvents()

    assert window._profile_result.solvent_contrast is not None
    legend_labels = window.smeared_profile_plot.figure.axes[
        0
    ].get_legend_handles_labels()[1]
    assert any("Direct solvent" in label for label in legend_labels)
    if window._profile_result.solvent_contrast.cutoff_radius_a is not None:
        assert window.fourier_rmax_spin.value() == pytest.approx(
            window._profile_result.solvent_contrast.cutoff_radius_a,
            abs=1.0e-4,
        )
        assert any("Cutoff" in label for label in legend_labels)
    assert (
        window.residual_profile_plot.current_contrast
        is window._profile_result.solvent_contrast
    )
    assert window._fourier_preview is not None
    assert window._fourier_preview.source_profile_label.startswith(
        "Solvent-subtracted"
    )

    window.fourier_use_solvent_subtracted_checkbox.setChecked(False)
    qapp.processEvents()

    assert window._fourier_preview is not None
    assert window._fourier_preview.source_profile_label == "Smeared ρ(r)"
    expected_positive = np.concatenate(
        (
            [
                float(
                    window._profile_result.smeared_orientation_average_density[
                        0
                    ]
                )
            ],
            np.asarray(
                window._profile_result.smeared_orientation_average_density,
                dtype=float,
            ),
        )
    )
    center_index = len(window._fourier_preview.source_radial_values) // 2
    assert window._fourier_preview.source_radial_values[0] == pytest.approx(
        -float(window._profile_result.radial_centers[-1])
    )
    assert window._fourier_preview.source_radial_values[
        center_index
    ] == pytest.approx(0.0)
    assert window._fourier_preview.source_density_values[
        center_index
    ] == pytest.approx(expected_positive[0])
    assert np.allclose(
        window._fourier_preview.source_density_values[center_index:],
        expected_positive,
    )
    assert np.allclose(
        window._fourier_preview.source_density_values[: center_index + 1],
        window._fourier_preview.source_density_values[center_index:][::-1],
    )
    window.close()


def test_fourier_preview_plot_renders_mirrored_profile_and_solvent_subtraction(
    qapp,
    tmp_path,
):
    structure_path = _write_xyz(
        tmp_path / "fourier_plot_aesthetics.xyz",
        [
            "4",
            "fourier plot aesthetics",
            "N 0.0 0.0 0.1",
            "H 0.94 0.0 -0.2",
            "H -0.47 0.81 -0.2",
            "H -0.47 -0.81 -0.2",
        ],
    )
    structure = load_electron_density_structure(structure_path)
    result = compute_electron_density_profile(
        structure,
        _mesh_settings_for_structure(
            structure,
            rstep=0.1,
            theta_divisions=24,
            phi_divisions=18,
        ),
    )
    contrasted = apply_solvent_contrast_to_profile_result(
        result,
        ContrastSolventDensitySettings.from_values(
            method=CONTRAST_SOLVENT_METHOD_DIRECT,
            direct_electron_density_e_per_a3=float(
                np.max(
                    np.asarray(
                        result.smeared_orientation_average_density,
                        dtype=float,
                    )
                )
                * 0.5
            ),
        ),
        solvent_name="Direct solvent",
    )
    settings = ElectronDensityFourierTransformSettings(
        r_min=0.0,
        r_max=float(result.radial_centers[-1]),
        window_function="hanning",
        resampling_points=64,
        q_min=0.02,
        q_max=1.2,
        q_step=0.01,
    )
    preview = prepare_electron_density_fourier_transform(contrasted, settings)

    plot = ElectronDensityFourierPreviewPlot()
    plot.set_preview(preview)

    axis = plot.figure.axes[0]
    assert axis.get_xlabel() == "Mirrored r (Å)"
    assert axis.get_title(loc="left") == "Fourier-Transform Preparation"
    assert len(plot.figure.axes) == 1

    mirrored_source_line = next(
        line
        for line in axis.get_lines()
        if line.get_label().startswith("Windowed Solvent-subtracted")
    )
    x_data = np.asarray(mirrored_source_line.get_xdata(), dtype=float)
    assert x_data[0] == pytest.approx(-settings.r_max)
    assert x_data[-1] == pytest.approx(settings.r_max)
    assert np.min(x_data) < 0.0
    assert np.max(x_data) > 0.0
    trace_labels = [
        line.get_label()
        for line in axis.get_lines()
        if not str(line.get_label()).startswith("_child")
    ]
    assert trace_labels == [mirrored_source_line.get_label()]


def test_main_window_smearing_commit_updates_solvent_residual(
    qapp,
    tmp_path,
):
    structure_path = _write_xyz(
        tmp_path / "solvent_smearing.xyz",
        [
            "4",
            "solvent smearing",
            "N 0.0 0.0 0.1",
            "H 0.94 0.0 -0.2",
            "H -0.47 0.81 -0.2",
            "H -0.47 -0.81 -0.2",
        ],
    )
    window = ElectronDensityMappingMainWindow(
        initial_input_path=structure_path
    )

    window.run_button.click()
    _wait_for(
        lambda: window._profile_result is not None
        and window._calculation_thread is None,
        qapp,
    )

    assert window._profile_result is not None
    direct_density = float(
        min(
            np.max(
                np.asarray(
                    window._profile_result.smeared_orientation_average_density,
                    dtype=float,
                )
            )
            * 0.01,
            50.0,
        )
    )
    window.solvent_method_combo.setCurrentIndex(
        window.solvent_method_combo.findData(CONTRAST_SOLVENT_METHOD_DIRECT)
    )
    window.direct_density_spin.setValue(direct_density)
    window.compute_solvent_density_button.click()
    qapp.processEvents()

    initial_residual = np.asarray(
        window._profile_result.solvent_contrast.solvent_subtracted_smeared_density,
        dtype=float,
    ).copy()

    window.smearing_factor_spin.lineEdit().setText("0.040000")
    window.smearing_factor_spin.interpretText()
    qapp.processEvents()

    assert window._profile_result is not None
    assert window._profile_result.solvent_contrast is not None
    assert (
        window._profile_result.solvent_contrast.solvent_density_e_per_a3
        == pytest.approx(direct_density)
    )
    assert not np.allclose(
        window._profile_result.solvent_contrast.solvent_subtracted_smeared_density,
        initial_residual,
    )
    assert (
        window.residual_profile_plot.current_contrast
        is window._profile_result.solvent_contrast
    )
    window.close()


def test_main_window_defaults_rstep_and_rmax_from_structure(qapp, tmp_path):
    del qapp
    structure_path = _write_xyz(
        tmp_path / "default_mesh.xyz",
        [
            "3",
            "default mesh",
            "O 0.0 0.0 0.0",
            "H 0.96 0.0 0.0",
            "H -0.24 0.93 0.0",
        ],
    )
    window = ElectronDensityMappingMainWindow(
        initial_input_path=structure_path
    )

    assert window.rstep_spin.value() == pytest.approx(0.05)
    assert window.rmax_spin.value() == pytest.approx(
        window._structure.rmax,
        abs=1.0e-4,
    )
    assert "geometric mass center" in (
        window.reset_center_button.text().lower()
    )
    window.close()


def test_main_window_right_pane_uses_visible_vertical_scrollbar(qapp):
    window = ElectronDensityMappingMainWindow()
    window.show()
    qapp.processEvents()

    assert (
        window._right_scroll_area.verticalScrollBarPolicy()
        == Qt.ScrollBarPolicy.ScrollBarAlwaysOn
    )
    assert (
        window._right_scroll_area.horizontalScrollBarPolicy()
        == Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    )
    assert window._right_scroll_area.verticalScrollBar().isVisible()
    window.close()


def test_main_window_expanded_plots_preserve_viewer_height_and_scroll(qapp):
    window = ElectronDensityMappingMainWindow()
    window.show()
    qapp.processEvents()

    minimum_viewer_height = window.structure_viewer.minimumHeight()
    for section in (
        window.profile_section,
        window.smeared_section,
        window.residual_section,
        window.fourier_preview_section,
        window.scattering_section,
    ):
        section.expand()
    qapp.processEvents()

    assert window.structure_viewer.height() >= minimum_viewer_height
    assert window._right_scroll_area.verticalScrollBar().maximum() > 0
    window.close()


def test_main_window_center_controls_snap_and_reset(qapp, tmp_path):
    del qapp
    structure_path = _write_xyz(
        tmp_path / "pb_h_center.xyz",
        [
            "2",
            "Pb-H center",
            "H 0.0 0.0 0.0",
            "Pb 2.0 0.0 0.0",
        ],
    )
    window = ElectronDensityMappingMainWindow(
        initial_input_path=structure_path
    )
    calculated_center = np.asarray(
        window._structure.center_of_mass, dtype=float
    )

    window.snap_center_button.click()

    assert window._structure.center_mode == "nearest_atom"
    assert np.allclose(
        window._structure.active_center,
        window._structure.nearest_atom_coordinates,
    )
    snapped_rmax = window._structure.rmax
    assert window.rmax_spin.value() == pytest.approx(
        snapped_rmax,
        abs=1.0e-4,
    )

    window.reset_center_button.click()

    assert window._structure.center_mode == "center_of_mass"
    assert np.allclose(window._structure.active_center, calculated_center)
    assert window.rmax_spin.value() == pytest.approx(
        window._structure.rmax,
        abs=1.0e-4,
    )
    window.close()


def test_main_window_reference_element_defaults_to_heaviest_and_can_snap(
    qapp,
    tmp_path,
):
    del qapp
    structure_path = _write_xyz(
        tmp_path / "reference_center_ui.xyz",
        [
            "5",
            "reference center ui",
            "Pb 0.0 0.0 0.0",
            "Pb 2.0 0.0 0.0",
            "H 0.0 4.0 0.0",
            "H 0.0 6.0 0.0",
            "O 0.0 0.0 2.0",
        ],
    )
    window = ElectronDensityMappingMainWindow(
        initial_input_path=structure_path
    )

    assert window.reference_element_combo.currentData() == "Pb"
    assert "Å from the total-atom geometric center" in (
        window.reference_element_offset_value.text()
    )

    window.snap_reference_center_button.click()

    assert window._structure.center_mode == "reference_element"
    assert np.allclose(
        window._structure.active_center,
        window._structure.reference_element_geometric_center,
    )

    window.reference_element_combo.setCurrentIndex(
        window.reference_element_combo.findData("H")
    )

    assert window._structure.reference_element == "H"
    assert window._structure.center_mode == "reference_element"
    assert np.allclose(
        window._structure.active_center,
        window._structure.reference_element_geometric_center,
    )
    window.close()


def test_snap_center_preserves_active_viewer_display_settings(qapp, tmp_path):
    del qapp
    structure_path = _write_xyz(
        tmp_path / "snap_preserve.xyz",
        [
            "4",
            "snap preserve",
            "C 0.0 0.0 0.0",
            "H 1.1 0.2 0.0",
            "H -0.2 0.9 0.3",
            "O 2.3 0.0 0.0",
        ],
    )
    window = ElectronDensityMappingMainWindow(
        initial_input_path=structure_path
    )
    viewer = window.structure_viewer

    viewer.atom_contrast_spin.lineEdit().setText("30")
    viewer.mesh_contrast_spin.lineEdit().setText("45")
    viewer.mesh_linewidth_spin.lineEdit().setText("2.70")
    viewer.atom_contrast_spin.interpretText()
    viewer.mesh_contrast_spin.interpretText()
    viewer.mesh_linewidth_spin.interpretText()
    viewer.point_atoms_checkbox.setChecked(True)

    viewer._mesh_color = "#ff6600"
    viewer._update_mesh_color_button_style()
    viewer._view_radius = 7.5
    viewer._view_center = np.asarray([0.2, -0.4, 0.6], dtype=float)
    viewer._scene_rotation = np.asarray(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    previous_centered = np.asarray(
        viewer.current_structure.centered_coordinates,
        dtype=float,
    ).copy()

    window.snap_center_button.click()

    assert viewer._atom_contrast == pytest.approx(0.30)
    assert viewer._mesh_contrast == pytest.approx(0.45)
    assert viewer._mesh_linewidth == pytest.approx(2.7)
    assert viewer._mesh_color == "#ff6600"
    assert viewer._atom_render_mode == "points"
    assert viewer.point_atoms_checkbox.isChecked()
    assert viewer._view_radius == pytest.approx(7.5)
    assert np.allclose(viewer._view_center, [0.2, -0.4, 0.6])
    assert np.allclose(
        viewer._scene_rotation,
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
    )
    assert viewer.current_structure.center_mode == "nearest_atom"
    assert not np.allclose(
        viewer.current_structure.centered_coordinates,
        previous_centered,
    )
    overlay_texts = [
        text
        for text in viewer._axis.texts
        if getattr(text, "get_gid", lambda: None)() == "active-visual-settings"
    ]
    assert len(overlay_texts) == 1
    overlay_text = overlay_texts[0].get_text()
    assert f"ZOOM {viewer._current_zoom_percentage():05.1f}%" in overlay_text
    assert "ATOM 030.0%" in overlay_text
    assert "MESH 045.0%" in overlay_text
    assert "LINE 2.70px" in overlay_text
    assert "#FF6600" in overlay_text
    window.close()


def test_viewer_contrast_controls_allow_full_transparency(qapp, tmp_path):
    del qapp
    structure_path = _write_xyz(
        tmp_path / "co2.xyz",
        [
            "3",
            "carbon dioxide",
            "O -1.16 0.0 0.0",
            "C 0.0 0.0 0.0",
            "O 1.16 0.0 0.0",
        ],
    )
    window = ElectronDensityMappingMainWindow(
        initial_input_path=structure_path
    )

    window.structure_viewer.atom_contrast_spin.setValue(0.0)
    window.structure_viewer.mesh_contrast_spin.setValue(0.0)

    assert window.structure_viewer._atom_contrast == pytest.approx(0.0)
    assert window.structure_viewer._mesh_contrast == pytest.approx(0.0)
    window.close()


def test_viewer_visual_controls_apply_on_commit_and_show_active_overlay(
    qapp,
    tmp_path,
    monkeypatch,
):
    del qapp
    structure_path = _write_xyz(
        tmp_path / "visual_controls.xyz",
        [
            "3",
            "visual controls",
            "O -1.16 0.0 0.0",
            "C 0.0 0.0 0.0",
            "O 1.16 0.0 0.0",
        ],
    )
    window = ElectronDensityMappingMainWindow(
        initial_input_path=structure_path
    )
    viewer = window.structure_viewer

    original_atom_contrast = viewer._atom_contrast
    original_mesh_contrast = viewer._mesh_contrast
    original_mesh_linewidth = viewer._mesh_linewidth

    viewer.atom_contrast_spin.lineEdit().setText("25")
    viewer.mesh_contrast_spin.lineEdit().setText("35")
    viewer.mesh_linewidth_spin.lineEdit().setText("2.40")

    assert viewer._atom_contrast == pytest.approx(original_atom_contrast)
    assert viewer._mesh_contrast == pytest.approx(original_mesh_contrast)
    assert viewer._mesh_linewidth == pytest.approx(original_mesh_linewidth)

    viewer.atom_contrast_spin.interpretText()
    viewer.mesh_contrast_spin.interpretText()
    viewer.mesh_linewidth_spin.interpretText()

    monkeypatch.setattr(
        "saxshell.saxs.electron_density_mapping.ui.viewer.QColorDialog.getColor",
        lambda *args, **kwargs: QColor("#ff6600"),
    )
    viewer.mesh_color_button.click()

    assert viewer._atom_contrast == pytest.approx(0.25)
    assert viewer._mesh_contrast == pytest.approx(0.35)
    assert viewer._mesh_linewidth == pytest.approx(2.4)
    assert viewer._mesh_color == "#ff6600"

    overlay_texts = [
        text
        for text in viewer._axis.texts
        if getattr(text, "get_gid", lambda: None)() == "active-visual-settings"
    ]
    assert len(overlay_texts) == 1
    overlay_text = overlay_texts[0].get_text()
    assert f"ZOOM {viewer._current_zoom_percentage():05.1f}%" in overlay_text
    assert "ATOM 025.0%" in overlay_text
    assert "MESH 035.0%" in overlay_text
    assert "LINE 2.40px" in overlay_text
    assert "#FF6600" in overlay_text
    assert any(
        line.get_color() == "#ff6600" and float(line.get_linewidth()) >= 2.1
        for line in viewer._axis.get_lines()
    )
    window.close()


def test_viewer_supports_point_atoms_and_autoscale(qapp, tmp_path):
    del qapp
    structure_path = _write_xyz(
        tmp_path / "viewer_modes.xyz",
        [
            "4",
            "viewer modes",
            "C 0.0 0.0 0.0",
            "H 0.9 0.0 0.0",
            "H -0.45 0.78 0.0",
            "H -0.45 -0.78 0.0",
        ],
    )
    window = ElectronDensityMappingMainWindow(
        initial_input_path=structure_path
    )
    viewer = window.structure_viewer

    viewer.point_atoms_checkbox.setChecked(True)
    assert viewer._atom_render_mode == "points"

    viewer._view_radius = 50.0
    viewer.autoscale_button.click()

    assert viewer._view_radius == pytest.approx(
        viewer._default_view_radius(viewer.current_structure)
    )
    assert viewer._view_radius < 50.0
    overlay_texts = [
        text
        for text in viewer._axis.texts
        if getattr(text, "get_gid", lambda: None)() == "active-visual-settings"
    ]
    assert len(overlay_texts) == 1
    assert "ZOOM 100.0%" in overlay_texts[0].get_text()
    window.close()


def test_viewer_rebases_default_zoom_around_legacy_two_hundred_percent(
    qapp,
    tmp_path,
):
    del qapp
    structure_path = _write_xyz(
        tmp_path / "viewer_zoom_rebase.xyz",
        [
            "4",
            "viewer zoom rebase",
            "C 0.0 0.0 0.0",
            "H 0.9 0.0 0.0",
            "H -0.45 0.78 0.0",
            "H -0.45 -0.78 0.0",
        ],
    )
    window = ElectronDensityMappingMainWindow(
        initial_input_path=structure_path
    )
    viewer = window.structure_viewer

    assert viewer._view_radius == pytest.approx(
        viewer._legacy_default_view_radius(viewer.current_structure) * 0.5
    )
    assert viewer._current_zoom_percentage() == pytest.approx(
        100.0,
        abs=0.01,
    )
    window.close()


def test_viewer_scroll_wheel_no_longer_zooms_structure(
    qapp,
    tmp_path,
    monkeypatch,
):
    structure_path = _write_xyz(
        tmp_path / "zoom_perf.xyz",
        [
            "4",
            "zoom perf",
            "C 0.0 0.0 0.0",
            "H 0.9 0.0 0.0",
            "H -0.45 0.78 0.0",
            "H -0.45 -0.78 0.0",
        ],
    )
    window = ElectronDensityMappingMainWindow(
        initial_input_path=structure_path
    )
    viewer = window.structure_viewer
    axis_reference_calls: list[str] = []
    draw_calls: list[float] = []

    monkeypatch.setattr(
        viewer,
        "_draw_axis_reference",
        lambda: axis_reference_calls.append("axis-reference"),
    )
    monkeypatch.setattr(
        viewer.canvas,
        "draw_idle",
        lambda: draw_calls.append(viewer._view_radius),
    )

    initial_radius = viewer._view_radius
    viewer._handle_scroll(
        SimpleNamespace(
            inaxes=viewer._axis,
            step=1.0,
            button="up",
        )
    )

    assert viewer._view_radius == pytest.approx(initial_radius)
    assert not viewer._view_update_timer.isActive()
    assert draw_calls == []
    assert axis_reference_calls == []
    window.close()


def test_viewer_rotation_drag_preserves_valid_scene_rotation(qapp, tmp_path):
    del qapp
    structure_path = _write_xyz(
        tmp_path / "tetra.xyz",
        [
            "4",
            "tetra",
            "C 0.0 0.0 0.0",
            "H 0.9 0.9 0.9",
            "H -0.9 -0.9 0.9",
            "H -0.9 0.9 -0.9",
        ],
    )
    window = ElectronDensityMappingMainWindow(
        initial_input_path=structure_path
    )
    viewer = window.structure_viewer

    press_event = SimpleNamespace(
        x=120.0,
        y=140.0,
        inaxes=viewer._axis,
        button=MouseButton.LEFT,
    )
    viewer._handle_press(press_event)
    viewer._handle_motion(
        SimpleNamespace(
            x=980.0,
            y=1520.0,
            inaxes=viewer._axis,
        )
    )
    viewer._handle_release(SimpleNamespace())

    rotation = viewer._scene_rotation
    assert np.all(np.isfinite(rotation))
    assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-6)
    assert float(np.linalg.det(rotation)) == pytest.approx(1.0, abs=1.0e-6)
    assert not np.allclose(rotation, np.eye(3))
    window.close()


def test_viewer_origin_guides_follow_structure_rotation(qapp, tmp_path):
    del qapp
    structure_path = _write_xyz(
        tmp_path / "near_origin.xyz",
        [
            "4",
            "near origin",
            "C -0.8 0.0 0.0",
            "C 0.4 0.9 0.3",
            "C 0.2 -0.7 0.8",
            "H 0.0 0.0 4.2",
        ],
    )
    window = ElectronDensityMappingMainWindow(
        initial_input_path=structure_path
    )
    viewer = window.structure_viewer

    initial_guides = [
        line
        for line in viewer._axis.get_lines()
        if line.get_gid() == "origin-guide"
    ]
    assert len(initial_guides) == 3
    initial_segments = sorted(
        (
            tuple(np.round(line.get_data_3d()[0], 6)),
            tuple(np.round(line.get_data_3d()[1], 6)),
            tuple(np.round(line.get_data_3d()[2], 6)),
        )
        for line in initial_guides
    )

    viewer._handle_press(
        SimpleNamespace(
            x=180.0,
            y=210.0,
            inaxes=viewer._axis,
            button=MouseButton.LEFT,
        )
    )
    viewer._handle_motion(
        SimpleNamespace(
            x=760.0,
            y=860.0,
            inaxes=viewer._axis,
        )
    )
    viewer._handle_release(SimpleNamespace())

    rotated_guides = [
        line
        for line in viewer._axis.get_lines()
        if line.get_gid() == "origin-guide"
    ]
    assert len(rotated_guides) == 3
    rotated_segments = sorted(
        (
            tuple(np.round(line.get_data_3d()[0], 6)),
            tuple(np.round(line.get_data_3d()[1], 6)),
            tuple(np.round(line.get_data_3d()[2], 6)),
        )
        for line in rotated_guides
    )

    assert rotated_segments != initial_segments
    window.close()


def test_smearing_settings_can_disable_or_smooth_profile(tmp_path):
    structure_path = _write_xyz(
        tmp_path / "smoothable.xyz",
        [
            "4",
            "smoothable",
            "O 0.0 0.0 0.0",
            "H 0.95 0.0 0.0",
            "H -0.25 0.92 0.0",
            "H 0.0 0.0 1.8",
        ],
    )
    structure = load_electron_density_structure(structure_path)
    mesh_settings = _mesh_settings_for_structure(
        structure,
        rstep=0.1,
        theta_divisions=24,
        phi_divisions=18,
    )

    no_smear = compute_electron_density_profile(
        structure,
        mesh_settings,
        smearing_settings=ElectronDensitySmearingSettings(
            debye_waller_factor=0.0
        ),
    )
    smeared = compute_electron_density_profile(
        structure,
        mesh_settings,
        smearing_settings=ElectronDensitySmearingSettings(
            debye_waller_factor=0.006
        ),
    )

    assert np.allclose(
        no_smear.smeared_orientation_average_density,
        no_smear.orientation_average_density,
    )
    assert not np.allclose(
        smeared.smeared_orientation_average_density,
        smeared.orientation_average_density,
    )


def test_smearing_strength_changes_profile_without_increasing_total(
    tmp_path,
):
    structure_path = _write_xyz(
        tmp_path / "smearing_strength.xyz",
        [
            "4",
            "smearing strength",
            "O 0.0 0.0 0.0",
            "H 0.95 0.0 0.0",
            "H -0.25 0.92 0.0",
            "H 0.0 0.0 1.8",
        ],
    )
    structure = load_electron_density_structure(structure_path)
    mesh_settings = _mesh_settings_for_structure(
        structure,
        rstep=0.1,
        theta_divisions=24,
        phi_divisions=18,
    )

    baseline = compute_electron_density_profile(
        structure,
        mesh_settings,
        smearing_settings=ElectronDensitySmearingSettings(
            debye_waller_factor=0.0
        ),
    )
    light_smear = compute_electron_density_profile(
        structure,
        mesh_settings,
        smearing_settings=ElectronDensitySmearingSettings(
            debye_waller_factor=0.001
        ),
    )
    heavy_smear = compute_electron_density_profile(
        structure,
        mesh_settings,
        smearing_settings=ElectronDensitySmearingSettings(
            debye_waller_factor=0.04
        ),
    )

    baseline_density = np.asarray(
        baseline.orientation_average_density,
        dtype=float,
    )
    light_density = np.asarray(
        light_smear.smeared_orientation_average_density,
        dtype=float,
    )
    heavy_density = np.asarray(
        heavy_smear.smeared_orientation_average_density,
        dtype=float,
    )
    shell_volumes = np.asarray(baseline.shell_volumes, dtype=float)
    baseline_total = float(np.sum(baseline_density * shell_volumes))
    light_total = float(np.sum(light_density * shell_volumes))
    heavy_total = float(np.sum(heavy_density * shell_volumes))

    assert light_total <= baseline_total + 1.0e-9
    assert heavy_total <= baseline_total + 1.0e-9
    assert not np.allclose(light_density, heavy_density)
    assert np.linalg.norm(light_density - baseline_density) < np.linalg.norm(
        heavy_density - baseline_density
    )


def _build_distribution_workspace_inputs(
    tmp_path,
) -> tuple[Path, Path, Path]:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    distribution_root = project_dir / "saved_distributions" / "dist_demo"
    distribution_root.mkdir(parents=True)
    (distribution_root / "distribution.json").write_text(
        json.dumps(
            {
                "distribution_id": "dist_demo",
                "component_artifacts_ready": False,
                "prior_artifacts_ready": True,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    clusters_dir = tmp_path / "clusters"
    first_dir = clusters_dir / "PbI2"
    second_dir = clusters_dir / "PbI3"
    first_dir.mkdir(parents=True)
    second_dir.mkdir(parents=True)
    _write_xyz(
        first_dir / "frame_0001.xyz",
        [
            "3",
            "PbI2 cluster",
            "Pb 0.0 0.0 0.0",
            "I 2.8 0.0 0.0",
            "I 0.0 2.8 0.0",
        ],
    )
    _write_xyz(
        second_dir / "frame_0001.xyz",
        [
            "4",
            "PbI3 cluster",
            "Pb 0.0 0.0 0.0",
            "I 2.8 0.0 0.0",
            "I 0.0 2.8 0.0",
            "I 0.0 0.0 2.8",
        ],
    )
    return project_dir, distribution_root, clusters_dir


def test_main_window_restores_non_pushed_workspace_state(
    qapp,
    tmp_path,
):
    project_dir, distribution_root, clusters_dir = (
        _build_distribution_workspace_inputs(tmp_path)
    )
    workspace_state_path = (
        distribution_root / "electron_density_mapping" / "workspace_state.json"
    )
    summary_path = (
        distribution_root
        / "electron_density_mapping"
        / "born_approximation_component_summary.json"
    )

    window = ElectronDensityMappingMainWindow(
        initial_project_dir=project_dir,
        initial_input_path=clusters_dir,
        initial_output_dir=distribution_root / "electron_density_mapping",
        initial_distribution_id="dist_demo",
        initial_distribution_root_dir=distribution_root,
        preview_mode=False,
    )

    window.run_button.click()
    _wait_for(
        lambda: all(
            state.profile_result is not None
            for state in window._cluster_group_states
        )
        and window._calculation_thread is None,
        qapp,
    )
    window.apply_fourier_to_all_button.setChecked(True)
    window.evaluate_fourier_button.click()
    qapp.processEvents()

    assert all(
        state.transform_result is not None
        for state in window._cluster_group_states
    )
    assert workspace_state_path.is_file()
    assert not summary_path.is_file()
    window.close()

    restored = ElectronDensityMappingMainWindow(
        initial_project_dir=project_dir,
        initial_input_path=clusters_dir,
        initial_output_dir=distribution_root / "electron_density_mapping",
        initial_distribution_id="dist_demo",
        initial_distribution_root_dir=distribution_root,
        preview_mode=False,
    )
    qapp.processEvents()

    assert all(
        state.profile_result is not None
        for state in restored._cluster_group_states
    )
    assert all(
        state.transform_result is not None
        for state in restored._cluster_group_states
    )
    assert restored.push_to_model_button.isEnabled()
    restored.close()


def test_main_window_restores_smearing_autosave_workspace_setting(
    qapp,
    tmp_path,
):
    project_dir, distribution_root, clusters_dir = (
        _build_distribution_workspace_inputs(tmp_path)
    )
    workspace_state_path = (
        distribution_root / "electron_density_mapping" / "workspace_state.json"
    )

    window = ElectronDensityMappingMainWindow(
        initial_project_dir=project_dir,
        initial_input_path=clusters_dir,
        initial_output_dir=distribution_root / "electron_density_mapping",
        initial_distribution_id="dist_demo",
        initial_distribution_root_dir=distribution_root,
        preview_mode=False,
    )

    window.run_button.click()
    _wait_for(
        lambda: all(
            state.profile_result is not None
            for state in window._cluster_group_states
        )
        and window._calculation_thread is None,
        qapp,
    )

    window.auto_save_smearing_outputs_checkbox.setChecked(True)
    qapp.processEvents()

    assert workspace_state_path.is_file()
    workspace_payload = json.loads(
        workspace_state_path.read_text(encoding="utf-8")
    )
    assert workspace_payload["auto_save_smearing_outputs"] is True
    window.close()

    restored = ElectronDensityMappingMainWindow(
        initial_project_dir=project_dir,
        initial_input_path=clusters_dir,
        initial_output_dir=distribution_root / "electron_density_mapping",
        initial_distribution_id="dist_demo",
        initial_distribution_root_dir=distribution_root,
        preview_mode=False,
    )
    _wait_for(
        lambda: restored.cluster_group_table.rowCount() == 2,
        qapp,
    )

    assert restored.auto_save_smearing_outputs_checkbox.isChecked()
    restored.close()


def test_main_window_restores_debye_scattering_workspace_state(
    qapp,
    tmp_path,
):
    project_dir, distribution_root, clusters_dir = (
        _build_distribution_workspace_inputs(tmp_path)
    )
    workspace_state_path = (
        distribution_root / "electron_density_mapping" / "workspace_state.json"
    )

    window = ElectronDensityMappingMainWindow(
        initial_project_dir=project_dir,
        initial_input_path=clusters_dir,
        initial_output_dir=distribution_root / "electron_density_mapping",
        initial_distribution_id="dist_demo",
        initial_distribution_root_dir=distribution_root,
        preview_mode=False,
    )

    window.run_button.click()
    _wait_for(
        lambda: all(
            state.profile_result is not None
            for state in window._cluster_group_states
        )
        and window._calculation_thread is None,
        qapp,
    )
    window.apply_fourier_to_all_button.setChecked(True)
    window.evaluate_fourier_button.click()
    qapp.processEvents()
    window.apply_debye_to_all_button.setChecked(True)
    window.calculate_debye_scattering_button.click()
    qapp.processEvents()

    assert workspace_state_path.is_file()
    workspace_payload = json.loads(
        workspace_state_path.read_text(encoding="utf-8")
    )
    assert all(
        group_payload.get("debye_scattering_result") is not None
        for group_payload in workspace_payload["cluster_groups"]
    )
    window.close()

    restored = ElectronDensityMappingMainWindow(
        initial_project_dir=project_dir,
        initial_input_path=clusters_dir,
        initial_output_dir=distribution_root / "electron_density_mapping",
        initial_distribution_id="dist_demo",
        initial_distribution_root_dir=distribution_root,
        preview_mode=False,
    )
    qapp.processEvents()

    assert all(
        state.debye_scattering_result is not None
        for state in restored._cluster_group_states
    )
    assert restored.open_debye_scattering_compare_button.isEnabled()
    restored.open_debye_scattering_compare_button.click()
    qapp.processEvents()
    assert restored._debye_scattering_compare_dialog is not None
    assert (
        restored._debye_scattering_compare_dialog._trace_table.rowCount() == 2
    )
    restored._debye_scattering_compare_dialog.close()
    restored.close()


def test_main_window_restores_single_atom_debye_workspace_state(
    qapp,
    tmp_path,
):
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    distribution_root = tmp_path / "distribution"
    distribution_root.mkdir()
    clusters_dir = tmp_path / "clusters"
    single_dir = clusters_dir / "I"
    multi_dir = clusters_dir / "PbI2"
    single_dir.mkdir(parents=True)
    multi_dir.mkdir(parents=True)
    _write_xyz(
        single_dir / "frame_0001.xyz",
        [
            "1",
            "single iodine 1",
            "I 0.0 0.0 0.0",
        ],
    )
    _write_xyz(
        single_dir / "frame_0002.xyz",
        [
            "1",
            "single iodine 2",
            "I 1.0 2.0 3.0",
        ],
    )
    _write_xyz(
        multi_dir / "frame_0001.xyz",
        [
            "3",
            "PbI2 cluster",
            "Pb 0.0 0.0 0.0",
            "I 2.8 0.0 0.0",
            "I 0.0 2.8 0.0",
        ],
    )
    workspace_state_path = (
        distribution_root / "electron_density_mapping" / "workspace_state.json"
    )

    window = ElectronDensityMappingMainWindow(
        initial_project_dir=project_dir,
        initial_input_path=clusters_dir,
        initial_output_dir=distribution_root / "electron_density_mapping",
        initial_distribution_id="dist_demo",
        initial_distribution_root_dir=distribution_root,
        preview_mode=False,
    )
    window.run_button.click()
    _wait_for(
        lambda: window._calculation_thread is None
        and window._cluster_group_state_by_key("I") is not None
        and window._cluster_group_state_by_key("I").transform_result
        is not None,
        qapp,
    )

    assert workspace_state_path.is_file()
    assert window._cluster_group_state_by_key("I").transform_result is not None
    window.close()

    restored = ElectronDensityMappingMainWindow(
        initial_project_dir=project_dir,
        initial_input_path=clusters_dir,
        initial_output_dir=distribution_root / "electron_density_mapping",
        initial_distribution_id="dist_demo",
        initial_distribution_root_dir=distribution_root,
        preview_mode=False,
    )
    _wait_for(
        lambda: restored._cluster_group_state_by_key("I") is not None
        and restored._cluster_group_state_by_key("I").transform_result
        is not None,
        qapp,
    )

    assert restored._cluster_group_state_by_key("I").profile_result is None
    assert (
        restored._cluster_group_state_by_key("I").transform_result is not None
    )
    row_by_key = {
        restored.cluster_group_table.item(row_index, 0).text(): row_index
        for row_index in range(restored.cluster_group_table.rowCount())
    }
    assert restored.cluster_group_table.item(row_by_key["I"], 7).text() == (
        "Ready (Debye)"
    )
    restored.close()


def test_main_window_imports_project_debye_waller_terms_for_smearing_defaults(
    qapp,
    tmp_path,
):
    del qapp
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    summary_path = _write_project_debye_waller_analysis(project_dir, tmp_path)

    window = ElectronDensityMappingMainWindow(initial_project_dir=project_dir)

    assert window._project_debye_waller_source_path == summary_path
    assert window._active_smearing_settings.uses_pair_specific_terms
    assert len(window._active_smearing_settings.pair_specific_terms) > 0
    assert (
        window._active_smearing_settings.pair_specific_terms
        == window._active_smearing_settings.imported_pair_specific_terms
    )
    assert "Pair-specific Debye-Waller terms are selected by default" in (
        window.smearing_summary_value.text()
    )
    window.close()


def test_profile_result_serialization_preserves_project_debye_waller_terms(
    qapp,
    tmp_path,
):
    del qapp
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    _write_project_debye_waller_analysis(project_dir, tmp_path)

    window = ElectronDensityMappingMainWindow(initial_project_dir=project_dir)
    structure_path = _write_xyz(
        tmp_path / "smearing_pair_terms.xyz",
        [
            "3",
            "pair-specific smearing terms",
            "O 0.0 0.0 0.0",
            "H 0.9 0.0 0.0",
            "H -0.3 0.8 0.0",
        ],
    )
    structure = load_electron_density_structure(structure_path)
    profile_result = compute_electron_density_profile(
        structure,
        _mesh_settings_for_structure(structure, rstep=0.2),
        smearing_settings=window._smearing_settings_from_controls(),
    )

    payload = window._serialize_profile_result(profile_result)
    restored = window._deserialize_profile_result(payload)

    assert restored.smearing_settings.uses_pair_specific_terms
    assert (
        restored.smearing_settings.debye_waller_mode
        == profile_result.smearing_settings.debye_waller_mode
    )
    assert (
        restored.smearing_settings.pair_specific_terms
        == profile_result.smearing_settings.pair_specific_terms
    )
    assert (
        restored.smearing_settings.imported_pair_specific_terms
        == profile_result.smearing_settings.imported_pair_specific_terms
    )
    window.close()


def test_main_window_push_to_model_writes_and_restores_born_components(
    qapp,
    tmp_path,
):
    project_dir, distribution_root, clusters_dir = (
        _build_distribution_workspace_inputs(tmp_path)
    )

    pushed_payloads: list[dict[str, object]] = []
    window = ElectronDensityMappingMainWindow(
        initial_project_dir=project_dir,
        initial_input_path=clusters_dir,
        initial_output_dir=distribution_root / "electron_density_mapping",
        initial_distribution_id="dist_demo",
        initial_distribution_root_dir=distribution_root,
        preview_mode=False,
    )
    window.born_components_built.connect(
        lambda payload: pushed_payloads.append(dict(payload))
    )

    assert window.windowTitle() == "Electron Density Mapping"
    assert "Computed Distribution Mode" in window.preview_mode_banner.text()
    assert window.cluster_group_table.rowCount() == 2
    assert not window.push_to_model_button.isEnabled()

    window.run_button.click()
    _wait_for(
        lambda: all(
            state.profile_result is not None
            for state in window._cluster_group_states
        )
        and window._calculation_thread is None,
        qapp,
    )

    for row_index in range(window.cluster_group_table.rowCount()):
        window.cluster_group_table.selectRow(row_index)
        qapp.processEvents()
        window.evaluate_fourier_button.click()
        assert window._active_cluster_group_state() is not None
        assert (
            window._active_cluster_group_state().transform_result is not None
        )

    assert window.push_to_model_button.isEnabled()
    window.push_to_model_button.click()

    component_map_path = distribution_root / "md_saxs_map.json"
    summary_path = (
        distribution_root
        / "electron_density_mapping"
        / "born_approximation_component_summary.json"
    )
    assert component_map_path.is_file()
    assert summary_path.is_file()
    component_map = json.loads(component_map_path.read_text(encoding="utf-8"))
    assert component_map["saxs_map"]["PbI2"]["no_motif"] == "PbI2_no_motif.txt"
    assert component_map["saxs_map"]["PbI3"]["no_motif"] == "PbI3_no_motif.txt"
    assert pushed_payloads
    assert pushed_payloads[-1]["distribution_id"] == "dist_demo"

    restored = ElectronDensityMappingMainWindow(
        initial_project_dir=project_dir,
        initial_input_path=clusters_dir,
        initial_output_dir=distribution_root / "electron_density_mapping",
        initial_distribution_id="dist_demo",
        initial_distribution_root_dir=distribution_root,
        preview_mode=False,
    )
    qapp.processEvents()

    assert restored.cluster_group_table.rowCount() == 2
    assert all(
        state.transform_result is not None
        for state in restored._cluster_group_states
    )
    assert restored.push_to_model_button.isEnabled()
    assert restored._selected_cluster_group_key == "PbI3"
    restored.close()
    window.close()


def test_reset_calculations_clears_persisted_workspace_state_and_history(
    qapp,
    tmp_path,
    monkeypatch,
):
    project_dir, distribution_root, clusters_dir = (
        _build_distribution_workspace_inputs(tmp_path)
    )
    workspace_state_path = (
        distribution_root / "electron_density_mapping" / "workspace_state.json"
    )
    saved_history_path = (
        distribution_root
        / "electron_density_mapping"
        / "saved_output_history.json"
    )
    preview_history_path = (
        distribution_root
        / "electron_density_mapping"
        / "electron_density_saved_output_history.json"
    )
    window = ElectronDensityMappingMainWindow(
        initial_project_dir=project_dir,
        initial_input_path=clusters_dir,
        initial_output_dir=distribution_root / "electron_density_mapping",
        initial_distribution_id="dist_demo",
        initial_distribution_root_dir=distribution_root,
        preview_mode=False,
    )

    window.run_button.click()
    _wait_for(
        lambda: all(
            state.profile_result is not None
            for state in window._cluster_group_states
        )
        and window._calculation_thread is None,
        qapp,
    )
    window.apply_fourier_to_all_button.setChecked(True)
    window.evaluate_fourier_button.click()
    qapp.processEvents()

    assert workspace_state_path.is_file()
    assert saved_history_path.is_file()

    monkeypatch.setattr(
        "saxshell.saxs.electron_density_mapping.ui.main_window.QInputDialog.getText",
        lambda *args, **kwargs: ("RESET", True),
    )
    window.reset_calculations_button.click()
    qapp.processEvents()

    assert not workspace_state_path.exists()
    assert not saved_history_path.exists()
    assert not preview_history_path.exists()
    assert window.output_history_table.rowCount() == 0
    window.close()

    restored = ElectronDensityMappingMainWindow(
        initial_project_dir=project_dir,
        initial_input_path=clusters_dir,
        initial_output_dir=distribution_root / "electron_density_mapping",
        initial_distribution_id="dist_demo",
        initial_distribution_root_dir=distribution_root,
        preview_mode=False,
    )
    qapp.processEvents()

    assert all(
        state.profile_result is None and state.transform_result is None
        for state in restored._cluster_group_states
    )
    assert restored.output_history_table.rowCount() == 0
    assert not restored.push_to_model_button.isEnabled()
    restored.close()


def test_cluster_group_table_starts_collapsed(qapp, tmp_path):
    clusters_dir = tmp_path / "clusters"
    first_dir = clusters_dir / "PbI2"
    second_dir = clusters_dir / "PbI3"
    first_dir.mkdir(parents=True)
    second_dir.mkdir(parents=True)
    _write_xyz(
        first_dir / "frame_0001.xyz",
        [
            "3",
            "PbI2 cluster",
            "Pb 0.0 0.0 0.0",
            "I 2.8 0.0 0.0",
            "I 0.0 2.8 0.0",
        ],
    )
    _write_xyz(
        second_dir / "frame_0001.xyz",
        [
            "4",
            "PbI3 cluster",
            "Pb 0.0 0.0 0.0",
            "I 2.8 0.0 0.0",
            "I 0.0 2.8 0.0",
            "I 0.0 0.0 2.8",
        ],
    )

    window = ElectronDensityMappingMainWindow(initial_input_path=clusters_dir)

    assert window.cluster_group_table.rowCount() == 2
    assert not window.cluster_group_table_section.is_expanded

    window.cluster_group_table_section.expand()

    assert window.cluster_group_table_section.is_expanded
    window.close()


def test_cluster_group_selection_preserves_zoom_percentage(
    qapp,
    tmp_path,
):
    clusters_dir = tmp_path / "clusters"
    first_dir = clusters_dir / "PbI2"
    second_dir = clusters_dir / "PbI3"
    first_dir.mkdir(parents=True)
    second_dir.mkdir(parents=True)
    _write_xyz(
        first_dir / "frame_0001.xyz",
        [
            "3",
            "PbI2 cluster",
            "Pb 0.0 0.0 0.0",
            "I 2.8 0.0 0.0",
            "I 0.0 2.8 0.0",
        ],
    )
    _write_xyz(
        second_dir / "frame_0001.xyz",
        [
            "4",
            "PbI3 cluster",
            "Pb 0.0 0.0 0.0",
            "I 2.8 0.0 0.0",
            "I 0.0 2.8 0.0",
            "I 0.0 0.0 2.8",
        ],
    )

    window = ElectronDensityMappingMainWindow(initial_input_path=clusters_dir)
    viewer = window.structure_viewer

    viewer._view_radius = (
        viewer._default_view_radius(viewer.current_structure) / 1.6
    )
    zoom_percentage = viewer._current_zoom_percentage()

    window.cluster_group_table.selectRow(1)
    qapp.processEvents()

    assert window._selected_cluster_group_key == "PbI3"
    assert viewer.current_structure is not None
    assert viewer._current_zoom_percentage() == pytest.approx(zoom_percentage)
    assert viewer._view_radius == pytest.approx(
        viewer._default_view_radius(viewer.current_structure)
        / (zoom_percentage / 100.0)
    )
    window.close()


def test_cluster_group_initial_selection_starts_at_hundred_percent_zoom(
    qapp,
    tmp_path,
):
    clusters_dir = _write_cluster_folder_input(tmp_path / "clusters")
    window = ElectronDensityMappingMainWindow(initial_input_path=clusters_dir)
    viewer = window.structure_viewer

    assert viewer.current_structure is not None
    assert viewer.current_mesh_geometry is not None
    assert viewer._current_zoom_percentage() == pytest.approx(
        100.0,
        abs=0.01,
    )

    overlay_texts = [
        text
        for text in viewer._axis.texts
        if getattr(text, "get_gid", lambda: None)() == "active-visual-settings"
    ]
    assert len(overlay_texts) == 1
    assert "ZOOM 100.0%" in overlay_texts[0].get_text()
    window.close()


def test_cluster_group_shared_mesh_rmax_survives_reference_element_updates(
    qapp,
    tmp_path,
):
    clusters_dir = tmp_path / "clusters"
    first_dir = clusters_dir / "I2"
    second_dir = clusters_dir / "PbI2"
    first_dir.mkdir(parents=True)
    second_dir.mkdir(parents=True)
    _write_xyz(
        first_dir / "frame_0001.xyz",
        [
            "2",
            "I2 cluster",
            "I 0.0 0.0 0.0",
            "I 2.8 0.0 0.0",
        ],
    )
    _write_xyz(
        second_dir / "frame_0001.xyz",
        [
            "3",
            "PbI2 cluster",
            "Pb 0.0 0.0 0.0",
            "I 2.8 0.0 0.0",
            "I 0.0 2.8 0.0",
        ],
    )

    window = ElectronDensityMappingMainWindow(initial_input_path=clusters_dir)
    viewer = window.structure_viewer
    expected_rmax = max(
        float(state.reference_structure.rmax)
        for state in window._cluster_group_states
    )

    def mesh_line_count() -> int:
        return sum(
            1
            for line in viewer._axis.get_lines()
            if str(line.get_color()).lower() == str(viewer._mesh_color).lower()
        )

    assert window.rmax_spin.value() == pytest.approx(expected_rmax, abs=1.0e-4)
    assert window._active_mesh_settings.rmax == pytest.approx(
        expected_rmax,
        abs=1.0e-4,
    )
    assert window._mesh_settings_from_controls().rmax == pytest.approx(
        expected_rmax,
        abs=1.0e-4,
    )
    assert viewer.current_mesh_geometry is not None
    assert viewer.current_mesh_geometry.domain_max_radius == pytest.approx(
        expected_rmax
    )
    assert mesh_line_count() > 0

    window._apply_center_mode("reference_element")
    qapp.processEvents()
    window.update_mesh_button.click()
    qapp.processEvents()

    assert window._active_mesh_settings.rmax == pytest.approx(
        expected_rmax,
        abs=1.0e-4,
    )
    assert viewer.current_mesh_geometry is not None
    assert viewer.current_mesh_geometry.domain_max_radius == pytest.approx(
        expected_rmax,
        abs=1.0e-4,
    )

    window.cluster_group_table.selectRow(1)
    qapp.processEvents()

    assert viewer.current_mesh_geometry is not None
    assert viewer.current_mesh_geometry.domain_max_radius == pytest.approx(
        expected_rmax,
        abs=1.0e-4,
    )
    assert viewer._current_zoom_percentage() == pytest.approx(
        100.0,
        abs=0.01,
    )
    assert mesh_line_count() > 0
    window.close()


def test_cluster_group_reload_resets_zoom_to_hundred_percent(
    qapp,
    tmp_path,
):
    initial_clusters = _write_cluster_folder_input(
        tmp_path / "clusters_initial"
    )
    replacement_clusters = _write_cluster_folder_input(
        tmp_path / "clusters_replacement"
    )
    window = ElectronDensityMappingMainWindow(
        initial_input_path=initial_clusters
    )
    viewer = window.structure_viewer

    viewer._view_radius = (
        viewer._default_view_radius(viewer.current_structure) / 0.13
    )
    viewer._draw_view(reset_view=False)
    assert viewer._current_zoom_percentage() == pytest.approx(13.0)

    window._load_input_path(replacement_clusters)
    qapp.processEvents()

    assert viewer.current_structure is not None
    assert viewer.current_mesh_geometry is not None
    assert viewer._current_zoom_percentage() == pytest.approx(100.0)
    window.close()


def test_cluster_group_table_center_element_defaults_to_heaviest_per_stoichiometry(
    qapp,
    tmp_path,
):
    clusters_dir = tmp_path / "clusters"
    first_dir = clusters_dir / "NaCl"
    second_dir = clusters_dir / "PbI2"
    first_dir.mkdir(parents=True)
    second_dir.mkdir(parents=True)
    _write_xyz(
        first_dir / "frame_0001.xyz",
        [
            "2",
            "NaCl cluster",
            "Na 0.0 0.0 0.0",
            "Cl 2.5 0.0 0.0",
        ],
    )
    _write_xyz(
        second_dir / "frame_0001.xyz",
        [
            "3",
            "PbI2 cluster",
            "Pb 0.0 0.0 0.0",
            "I 2.8 0.0 0.0",
            "I 0.0 2.8 0.0",
        ],
    )

    window = ElectronDensityMappingMainWindow(initial_input_path=clusters_dir)
    center_element_column = _table_column_index(
        window.cluster_group_table,
        "Center Element",
    )
    row_by_key = {
        window.cluster_group_table.item(row_index, 0).text(): row_index
        for row_index in range(window.cluster_group_table.rowCount())
    }

    nacl_combo = window.cluster_group_table.cellWidget(
        row_by_key["NaCl"],
        center_element_column,
    )
    pbi2_combo = window.cluster_group_table.cellWidget(
        row_by_key["PbI2"],
        center_element_column,
    )

    assert nacl_combo is not None
    assert pbi2_combo is not None
    assert nacl_combo.currentData() == "Cl"
    assert pbi2_combo.currentData() == "Pb"

    window.cluster_group_table.selectRow(row_by_key["NaCl"])
    qapp.processEvents()
    assert window.reference_element_combo.currentData() == "Cl"

    window.cluster_group_table.selectRow(row_by_key["PbI2"])
    qapp.processEvents()
    assert window.reference_element_combo.currentData() == "Pb"
    window.close()


def test_cluster_group_reference_element_dropdown_is_per_row_and_used_on_group_run(
    qapp,
    tmp_path,
):
    clusters_dir = tmp_path / "clusters"
    first_dir = clusters_dir / "PbI2"
    second_dir = clusters_dir / "CsI"
    first_dir.mkdir(parents=True)
    second_dir.mkdir(parents=True)
    _write_xyz(
        first_dir / "frame_0001.xyz",
        [
            "3",
            "PbI2 cluster",
            "Pb 0.0 0.0 0.0",
            "I 2.8 0.0 0.0",
            "I 0.0 2.8 0.0",
        ],
    )
    _write_xyz(
        second_dir / "frame_0001.xyz",
        [
            "2",
            "CsI cluster",
            "Cs 0.0 0.0 0.0",
            "I 2.8 0.0 0.0",
        ],
    )

    window = ElectronDensityMappingMainWindow(initial_input_path=clusters_dir)
    center_element_column = _table_column_index(
        window.cluster_group_table,
        "Center Element",
    )
    row_by_key = {
        window.cluster_group_table.item(row_index, 0).text(): row_index
        for row_index in range(window.cluster_group_table.rowCount())
    }

    window._apply_center_mode("reference_element")

    pbi2_combo = window.cluster_group_table.cellWidget(
        row_by_key["PbI2"],
        center_element_column,
    )
    assert pbi2_combo is not None
    pbi2_combo.setCurrentIndex(max(pbi2_combo.findData("I"), 0))
    qapp.processEvents()

    window.cluster_group_table.selectRow(row_by_key["CsI"])
    qapp.processEvents()
    csi_combo = window.cluster_group_table.cellWidget(
        row_by_key["CsI"],
        center_element_column,
    )
    assert csi_combo is not None
    assert csi_combo.currentData() == "Cs"
    assert window.reference_element_combo.currentData() == "Cs"

    window.cluster_group_table.selectRow(row_by_key["PbI2"])
    qapp.processEvents()
    assert window.reference_element_combo.currentData() == "I"

    window.run_button.click()
    _wait_for(
        lambda: all(
            state.profile_result is not None
            for state in window._cluster_group_states
        )
        and window._calculation_thread is None,
        qapp,
    )

    states_by_key = {
        state.key: state for state in window._cluster_group_states
    }
    assert states_by_key["PbI2"].profile_result is not None
    assert states_by_key["CsI"].profile_result is not None
    assert (
        states_by_key["PbI2"].profile_result.structure.reference_element == "I"
    )
    assert (
        states_by_key["CsI"].profile_result.structure.reference_element == "Cs"
    )
    window.close()


def test_cluster_group_mesh_settings_remain_shared_after_row_switches(
    qapp,
    tmp_path,
):
    clusters_dir = _write_cluster_folder_input(tmp_path / "clusters")
    window = ElectronDensityMappingMainWindow(initial_input_path=clusters_dir)

    window.rstep_spin.setValue(0.2)
    window.theta_divisions_spin.setValue(36)
    window.phi_divisions_spin.setValue(24)
    window.rmax_spin.setValue(7.5)
    window.update_mesh_button.click()
    window.run_button.click()
    _wait_for(
        lambda: all(
            state.profile_result is not None
            for state in window._cluster_group_states
        )
        and window._calculation_thread is None,
        qapp,
    )

    stored_mesh_settings = window._cluster_group_states[
        0
    ].profile_result.mesh_geometry.settings
    assert stored_mesh_settings.rstep == pytest.approx(0.2)

    window.rstep_spin.setValue(0.35)
    window.theta_divisions_spin.setValue(48)
    window.phi_divisions_spin.setValue(30)
    window.rmax_spin.setValue(8.25)
    window.update_mesh_button.click()

    window.cluster_group_table.selectRow(1)
    qapp.processEvents()

    assert window._active_mesh_settings.rstep == pytest.approx(0.35)
    assert window._active_mesh_settings.theta_divisions == 48
    assert window._active_mesh_settings.phi_divisions == 30
    assert window._active_mesh_settings.rmax == pytest.approx(8.25)
    assert window.rstep_spin.value() == pytest.approx(0.35)
    assert window.theta_divisions_spin.value() == 48
    assert window.phi_divisions_spin.value() == 30
    assert window.rmax_spin.value() == pytest.approx(8.25)

    window.cluster_group_table.selectRow(0)
    qapp.processEvents()

    assert window._active_mesh_settings.rstep == pytest.approx(0.35)
    assert window._active_mesh_settings.theta_divisions == 48
    assert window._active_mesh_settings.phi_divisions == 30
    assert window._active_mesh_settings.rmax == pytest.approx(8.25)
    assert window.rstep_spin.value() == pytest.approx(0.35)
    assert window.theta_divisions_spin.value() == 48
    assert window.phi_divisions_spin.value() == 30
    assert window.rmax_spin.value() == pytest.approx(8.25)
    window.close()


def test_cluster_group_table_stores_center_mode_and_reference_details(
    qapp,
    tmp_path,
):
    clusters_dir = _write_cluster_folder_input(tmp_path / "clusters")
    window = ElectronDensityMappingMainWindow(initial_input_path=clusters_dir)
    table = window.cluster_group_table
    mode_column = _table_column_index(table, "Center Mode")
    ref_column = _table_column_index(table, "Center Ref")
    element_column = _table_column_index(table, "Center Element")

    for row_index, state in enumerate(window._cluster_group_states):
        mode_item = table.item(row_index, mode_column)
        ref_item = table.item(row_index, ref_column)
        combo = table.cellWidget(row_index, element_column)
        assert mode_item is not None
        assert ref_item is not None
        assert mode_item.text() == "Mass COM"
        assert ref_item.text() == "COM"
        assert (
            combo.currentData() == state.reference_structure.reference_element
        )

    window._apply_center_mode("nearest_atom")
    qapp.processEvents()

    for row_index, state in enumerate(window._cluster_group_states):
        mode_item = table.item(row_index, mode_column)
        ref_item = table.item(row_index, ref_column)
        assert mode_item is not None
        assert ref_item is not None
        expected_reference = (
            f"{state.reference_structure.nearest_atom_element} atom "
            f"#{state.reference_structure.nearest_atom_index + 1}"
        )
        assert mode_item.text() == "Nearest Atom"
        assert ref_item.text() == expected_reference
        assert "Nearest atom" in ref_item.toolTip()

    window._apply_center_mode("reference_element")
    qapp.processEvents()

    for row_index, state in enumerate(window._cluster_group_states):
        mode_item = table.item(row_index, mode_column)
        ref_item = table.item(row_index, ref_column)
        combo = table.cellWidget(row_index, element_column)
        assert mode_item is not None
        assert ref_item is not None
        assert (
            combo.currentData() == state.reference_structure.reference_element
        )
        assert mode_item.text() == "Ref Element"
        assert (
            ref_item.text()
            == f"{state.reference_structure.reference_element} geom center"
        )
        assert (
            state.reference_structure.reference_element in ref_item.toolTip()
        )
    window.close()


def test_cluster_group_row_switch_keeps_mesh_overlay_visible(
    qapp,
    tmp_path,
):
    clusters_dir = _write_cluster_folder_input(tmp_path / "mesh_clusters")
    window = ElectronDensityMappingMainWindow(initial_input_path=clusters_dir)
    viewer = window.structure_viewer

    viewer._mesh_color = "#ff6600"
    viewer._update_mesh_color_button_style()
    viewer._draw_view(reset_view=False)

    def mesh_line_count() -> int:
        return sum(
            1
            for line in viewer._axis.get_lines()
            if str(line.get_color()).lower() == "#ff6600"
        )

    assert viewer.current_mesh_geometry is not None
    assert mesh_line_count() > 0

    window.cluster_group_table.selectRow(1)
    qapp.processEvents()

    assert viewer.current_mesh_geometry is not None
    assert mesh_line_count() > 0

    window.cluster_group_table.selectRow(0)
    qapp.processEvents()

    assert viewer.current_mesh_geometry is not None
    assert mesh_line_count() > 0
    window.close()


def test_cluster_group_selection_restores_per_cluster_viewer_state(
    qapp,
    tmp_path,
):
    clusters_dir = tmp_path / "clusters"
    first_dir = clusters_dir / "PbI2"
    second_dir = clusters_dir / "PbI3"
    first_dir.mkdir(parents=True)
    second_dir.mkdir(parents=True)
    _write_xyz(
        first_dir / "frame_0001.xyz",
        [
            "3",
            "PbI2 cluster",
            "Pb 0.0 0.0 0.0",
            "I 2.8 0.0 0.0",
            "I 0.0 2.8 0.0",
        ],
    )
    _write_xyz(
        second_dir / "frame_0001.xyz",
        [
            "4",
            "PbI3 cluster",
            "Pb 0.0 0.0 0.0",
            "I 2.8 0.0 0.0",
            "I 0.0 2.8 0.0",
            "I 0.0 0.0 2.8",
        ],
    )

    window = ElectronDensityMappingMainWindow(initial_input_path=clusters_dir)
    viewer = window.structure_viewer

    viewer.atom_contrast_spin.setValue(25.0)
    viewer.mesh_contrast_spin.setValue(35.0)
    viewer.mesh_linewidth_spin.setValue(2.4)
    viewer.show_mesh_checkbox.setChecked(False)
    viewer.point_atoms_checkbox.setChecked(False)
    viewer._mesh_color = "#ff6600"
    viewer._update_mesh_color_button_style()
    viewer._view_center = np.asarray([0.3, -0.5, 0.8], dtype=float)
    viewer._view_radius = 6.75
    viewer._scene_rotation = np.asarray(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    viewer._draw_view(reset_view=False)

    window.cluster_group_table.selectRow(1)
    qapp.processEvents()

    viewer.atom_contrast_spin.setValue(70.0)
    viewer.mesh_contrast_spin.setValue(80.0)
    viewer.mesh_linewidth_spin.setValue(1.1)
    viewer.show_mesh_checkbox.setChecked(True)
    viewer.point_atoms_checkbox.setChecked(True)
    viewer._mesh_color = "#0088cc"
    viewer._update_mesh_color_button_style()
    viewer._view_center = np.asarray([-0.2, 0.4, -0.6], dtype=float)
    viewer._view_radius = 4.25
    viewer._scene_rotation = np.asarray(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    viewer._draw_view(reset_view=False)

    window.cluster_group_table.selectRow(0)
    qapp.processEvents()

    assert window._selected_cluster_group_key == "PbI2"
    assert viewer._atom_contrast == pytest.approx(0.25)
    assert viewer._mesh_contrast == pytest.approx(0.35)
    assert viewer._mesh_linewidth == pytest.approx(2.4)
    assert viewer._mesh_color == "#ff6600"
    assert not viewer.show_mesh_checkbox.isChecked()
    assert not viewer.point_atoms_checkbox.isChecked()
    assert viewer._view_radius == pytest.approx(6.75)
    assert np.allclose(viewer._view_center, [0.3, -0.5, 0.8])
    assert np.allclose(
        viewer._scene_rotation,
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
    )

    window.cluster_group_table.selectRow(1)
    qapp.processEvents()

    assert window._selected_cluster_group_key == "PbI3"
    assert viewer._atom_contrast == pytest.approx(0.70)
    assert viewer._mesh_contrast == pytest.approx(0.80)
    assert viewer._mesh_linewidth == pytest.approx(1.1)
    assert viewer._mesh_color == "#0088cc"
    assert viewer.show_mesh_checkbox.isChecked()
    assert viewer.point_atoms_checkbox.isChecked()
    assert viewer._view_radius == pytest.approx(4.25)
    assert np.allclose(viewer._view_center, [-0.2, 0.4, -0.6])
    assert np.allclose(
        viewer._scene_rotation,
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ],
    )
    window.close()


def test_cluster_view_cache_prewarm_prioritizes_more_complex_clusters(
    qapp,
    tmp_path,
):
    clusters_dir = tmp_path / "clusters"
    clusters_dir.mkdir(parents=True)
    cluster_specs = {
        "PbI1": [
            "2",
            "PbI1 cluster",
            "Pb 0.0 0.0 0.0",
            "I 2.8 0.0 0.0",
        ],
        "PbI4": [
            "5",
            "PbI4 cluster",
            "Pb 0.0 0.0 0.0",
            "I 2.8 0.0 0.0",
            "I 0.0 2.8 0.0",
            "I 0.0 0.0 2.8",
            "I -2.8 0.0 0.0",
        ],
        "PbI2": [
            "3",
            "PbI2 cluster",
            "Pb 0.0 0.0 0.0",
            "I 2.8 0.0 0.0",
            "I 0.0 2.8 0.0",
        ],
    }
    for name, lines in cluster_specs.items():
        cluster_dir = clusters_dir / name
        cluster_dir.mkdir()
        _write_xyz(cluster_dir / "frame_0001.xyz", lines)

    window = ElectronDensityMappingMainWindow(initial_input_path=clusters_dir)
    viewer = window.structure_viewer
    viewer._scene_cache_max_entries = 2
    viewer._scene_cache_max_bytes = 10**9

    window._schedule_cluster_view_cache_prewarm()
    _wait_for(
        lambda: not window._cluster_view_cache_prewarm_queue
        and not window._cluster_view_cache_prewarm_timer.isActive(),
        qapp,
    )

    cached_keys = set(viewer._scene_cache.keys())
    active_key = window._selected_cluster_group_key
    assert active_key in cached_keys
    assert "PbI4" in cached_keys
    assert len(cached_keys) == 2
    skipped_key = min(
        (
            state.key
            for state in window._cluster_group_states
            if state.key != active_key and state.key != "PbI4"
        ),
        key=lambda key: next(
            window._cluster_group_render_complexity(state)
            for state in window._cluster_group_states
            if state.key == key
        ),
    )
    assert skipped_key not in cached_keys
    window.close()


def test_cluster_group_switch_preserves_mesh_update_confirmation(
    qapp,
    tmp_path,
    monkeypatch,
):
    clusters_dir = tmp_path / "clusters"
    first_dir = clusters_dir / "PbI2"
    second_dir = clusters_dir / "PbI3"
    first_dir.mkdir(parents=True)
    second_dir.mkdir(parents=True)
    _write_xyz(
        first_dir / "frame_0001.xyz",
        [
            "3",
            "PbI2 cluster",
            "Pb 0.0 0.0 0.0",
            "I 2.8 0.0 0.0",
            "I 0.0 2.8 0.0",
        ],
    )
    _write_xyz(
        second_dir / "frame_0001.xyz",
        [
            "4",
            "PbI3 cluster",
            "Pb 0.0 0.0 0.0",
            "I 2.8 0.0 0.0",
            "I 0.0 2.8 0.0",
            "I 0.0 0.0 2.8",
        ],
    )

    window = ElectronDensityMappingMainWindow(initial_input_path=clusters_dir)
    prompts: list[str] = []

    def fake_question(*args, **kwargs):
        del kwargs
        prompts.append(str(args[2]))
        return QMessageBox.StandardButton.Yes

    monkeypatch.setattr(
        "saxshell.saxs.electron_density_mapping.ui.main_window.QMessageBox.question",
        fake_question,
    )

    window.update_mesh_button.click()
    window.cluster_group_table.selectRow(1)
    qapp.processEvents()
    window.run_button.click()
    _wait_for(
        lambda: all(
            state.profile_result is not None
            for state in window._cluster_group_states
        )
        and window._calculation_thread is None,
        qapp,
    )

    assert not prompts
    window.close()


def test_cluster_group_switch_warns_after_unapplied_mesh_edits(
    qapp,
    tmp_path,
    monkeypatch,
):
    clusters_dir = tmp_path / "clusters"
    first_dir = clusters_dir / "PbI2"
    second_dir = clusters_dir / "PbI3"
    first_dir.mkdir(parents=True)
    second_dir.mkdir(parents=True)
    _write_xyz(
        first_dir / "frame_0001.xyz",
        [
            "3",
            "PbI2 cluster",
            "Pb 0.0 0.0 0.0",
            "I 2.8 0.0 0.0",
            "I 0.0 2.8 0.0",
        ],
    )
    _write_xyz(
        second_dir / "frame_0001.xyz",
        [
            "4",
            "PbI3 cluster",
            "Pb 0.0 0.0 0.0",
            "I 2.8 0.0 0.0",
            "I 0.0 2.8 0.0",
            "I 0.0 0.0 2.8",
        ],
    )

    window = ElectronDensityMappingMainWindow(initial_input_path=clusters_dir)
    prompts: list[tuple[str, str]] = []

    def fake_question(*args, **kwargs):
        del kwargs
        prompts.append((str(args[1]), str(args[2])))
        return QMessageBox.StandardButton.No

    monkeypatch.setattr(
        "saxshell.saxs.electron_density_mapping.ui.main_window.QMessageBox.question",
        fake_question,
    )

    window.update_mesh_button.click()
    window.rstep_spin.setValue(window.rstep_spin.value() + 0.05)
    window.cluster_group_table.selectRow(1)
    qapp.processEvents()
    window.run_button.click()
    qapp.processEvents()

    assert prompts
    assert prompts[-1][0] == "Mesh Settings Not Updated"
    assert window._calculation_thread is None
    window.close()


def test_cluster_group_selection_syncs_fourier_rmax_to_solvent_cutoff(
    qapp,
    tmp_path,
):
    clusters_dir = tmp_path / "clusters"
    first_dir = clusters_dir / "PbI2"
    second_dir = clusters_dir / "PbI3"
    first_dir.mkdir(parents=True)
    second_dir.mkdir(parents=True)
    _write_xyz(
        first_dir / "frame_0001.xyz",
        [
            "3",
            "PbI2 cluster",
            "Pb 0.0 0.0 0.0",
            "I 2.8 0.0 0.0",
            "I 0.0 2.8 0.0",
        ],
    )
    _write_xyz(
        second_dir / "frame_0001.xyz",
        [
            "4",
            "PbI3 cluster",
            "Pb 0.0 0.0 0.0",
            "I 2.8 0.0 0.0",
            "I 0.0 2.8 0.0",
            "I 0.0 0.0 2.8",
        ],
    )

    window = ElectronDensityMappingMainWindow(initial_input_path=clusters_dir)

    assert window.cluster_group_table.rowCount() == 2
    window.apply_contrast_to_all_button.setChecked(False)

    window.run_button.click()
    _wait_for(
        lambda: all(
            state.profile_result is not None
            for state in window._cluster_group_states
        )
        and window._calculation_thread is None,
        qapp,
    )

    window.solvent_method_combo.setCurrentIndex(
        window.solvent_method_combo.findData(CONTRAST_SOLVENT_METHOD_DIRECT)
    )

    cutoff_by_key: dict[str, float] = {}
    for row_index, state in enumerate(window._cluster_group_states):
        window.cluster_group_table.selectRow(row_index)
        qapp.processEvents()

        if row_index == 0:
            window.evaluate_fourier_button.click()
            assert state.transform_result is not None

        assert window._profile_result is not None
        direct_density = float(
            np.max(
                np.asarray(
                    window._profile_result.smeared_orientation_average_density,
                    dtype=float,
                )
            )
            * 0.5
        )
        window.direct_density_spin.setValue(direct_density)
        window.compute_solvent_density_button.click()
        qapp.processEvents()

        assert state.profile_result is not None
        assert state.profile_result.solvent_contrast is not None
        assert (
            state.profile_result.solvent_contrast.cutoff_radius_a is not None
        )
        cutoff_by_key[state.key] = float(
            state.profile_result.solvent_contrast.cutoff_radius_a
        )
        assert window.fourier_rmax_spin.value() == pytest.approx(
            cutoff_by_key[state.key],
            abs=1.0e-4,
        )
        assert state.transform_result is None

    for row_index, state in enumerate(window._cluster_group_states):
        window.cluster_group_table.selectRow(row_index)
        qapp.processEvents()
        assert window.fourier_rmax_spin.value() == pytest.approx(
            cutoff_by_key[state.key],
            abs=1.0e-4,
        )

    window.close()


def test_cluster_folder_manual_mode_runs_selected_row_and_locks_mesh(
    qapp,
    tmp_path,
):
    clusters_dir = _write_cluster_folder_input(tmp_path / "manual_clusters")
    window = ElectronDensityMappingMainWindow(initial_input_path=clusters_dir)
    window.update_mesh_button.click()
    window.manual_mode_checkbox.setChecked(True)
    window.cluster_group_table.selectRow(1)
    qapp.processEvents()

    window.run_button.click()
    _wait_for(
        lambda: window._calculation_thread is None
        and window._cluster_group_states[1].profile_result is not None,
        qapp,
    )

    assert window._cluster_group_states[0].profile_result is None
    assert window._cluster_group_states[1].profile_result is not None
    assert "1/2" in window.cluster_completion_tracker_label.text()
    assert not window.rstep_spin.isEnabled()
    assert not window.theta_divisions_spin.isEnabled()
    assert not window.update_mesh_button.isEnabled()

    window.manual_mode_checkbox.setChecked(False)
    qapp.processEvents()

    assert not window.rstep_spin.isEnabled()
    assert not window.update_mesh_button.isEnabled()

    window.manual_mode_checkbox.setChecked(True)
    window.cluster_group_table.selectRow(0)
    qapp.processEvents()
    window.run_button.click()
    _wait_for(
        lambda: window._calculation_thread is None
        and all(
            state.profile_result is not None
            for state in window._cluster_group_states
        ),
        qapp,
    )

    assert window.cluster_completion_indicator.text() == "COMPLETE"
    assert "2/2" in window.cluster_completion_tracker_label.text()
    window.close()


def test_reset_calculated_densities_requires_secondary_auth_and_unlocks_mesh(
    qapp,
    tmp_path,
    monkeypatch,
):
    clusters_dir = _write_cluster_folder_input(tmp_path / "reset_clusters")
    window = ElectronDensityMappingMainWindow(initial_input_path=clusters_dir)
    window.update_mesh_button.click()
    window.manual_mode_checkbox.setChecked(True)
    window.run_button.click()
    _wait_for(
        lambda: window._calculation_thread is None
        and window._cluster_group_states[0].profile_result is not None,
        qapp,
    )

    monkeypatch.setattr(
        "saxshell.saxs.electron_density_mapping.ui.main_window.QMessageBox.critical",
        lambda *args, **kwargs: QMessageBox.StandardButton.Ok,
    )
    monkeypatch.setattr(
        "saxshell.saxs.electron_density_mapping.ui.main_window.QInputDialog.getText",
        lambda *args, **kwargs: ("not-reset", True),
    )
    window.reset_calculations_button.click()
    qapp.processEvents()

    assert window._cluster_group_states[0].profile_result is not None
    assert not window.rstep_spin.isEnabled()

    monkeypatch.setattr(
        "saxshell.saxs.electron_density_mapping.ui.main_window.QInputDialog.getText",
        lambda *args, **kwargs: ("RESET", True),
    )
    window.reset_calculations_button.click()
    qapp.processEvents()

    assert all(
        state.profile_result is None and state.transform_result is None
        for state in window._cluster_group_states
    )
    assert window.cluster_completion_indicator.text() == "PENDING"
    assert "0/2" in window.cluster_completion_tracker_label.text()
    assert window.rstep_spin.isEnabled()
    assert window.update_mesh_button.isEnabled()
    window.close()


def test_stop_active_electron_density_calculation_keeps_window_open(
    qapp,
    tmp_path,
    monkeypatch,
):
    structure_path = _write_xyz(
        tmp_path / "stop_density.xyz",
        [
            "3",
            "stop density",
            "O 0.0 0.0 0.0",
            "H 0.96 0.0 0.0",
            "H -0.24 0.93 0.0",
        ],
    )
    original_compute = compute_electron_density_profile_for_input

    def slow_compute(
        *args, progress_callback=None, cancel_callback=None, **kwargs
    ):
        for step in range(10):
            if progress_callback is not None:
                progress_callback(step, 10, f"Slow step {step + 1}/10")
            time.sleep(0.03)
            if cancel_callback is not None and cancel_callback():
                raise ElectronDensityCalculationCanceled(
                    "Electron-density calculation was stopped by the user."
                )
        return original_compute(
            *args,
            progress_callback=progress_callback,
            cancel_callback=cancel_callback,
            **kwargs,
        )

    monkeypatch.setattr(
        "saxshell.saxs.electron_density_mapping.ui.main_window.compute_electron_density_profile_for_input",
        slow_compute,
    )

    window = ElectronDensityMappingMainWindow(
        initial_input_path=structure_path
    )
    window.update_mesh_button.click()
    window.run_button.click()
    _wait_for(lambda: window._calculation_thread is not None, qapp)
    _wait_for(lambda: window.stop_calculation_button.isEnabled(), qapp)

    window.stop_calculation_button.click()
    _wait_for(lambda: window._calculation_thread is None, qapp)

    assert window._profile_result is None
    assert "stopped" in window.calculation_progress_message.text().lower()
    assert "Slow step" not in window.status_text.toPlainText()
    assert window.run_button.isEnabled()
    assert not window.stop_calculation_button.isEnabled()
    window.close()


def test_cluster_folder_run_uses_contiguous_frame_mode_per_stoichiometry(
    qapp,
    tmp_path,
):
    clusters_dir = tmp_path / "clusters"
    first_dir = clusters_dir / "PbI2"
    second_dir = clusters_dir / "PbI3"
    first_dir.mkdir(parents=True)
    second_dir.mkdir(parents=True)
    _write_xyz(
        first_dir / "frame_0001_AAA.xyz",
        [
            "3",
            "PbI2 frame 1",
            "Pb 0.0 0.0 0.0",
            "I 2.8 0.0 0.0",
            "I 0.0 2.8 0.0",
        ],
    )
    _write_xyz(
        first_dir / "frame_0002_AAA.xyz",
        [
            "3",
            "PbI2 frame 2",
            "Pb 10.0 0.0 0.0",
            "I 12.9 0.0 0.0",
            "I 10.0 2.9 0.0",
        ],
    )
    _write_xyz(
        second_dir / "frame_0005_AAA.xyz",
        [
            "4",
            "PbI3 frame 5",
            "Pb 0.0 0.0 0.0",
            "I 2.8 0.0 0.0",
            "I 0.0 2.8 0.0",
            "I 0.0 0.0 2.8",
        ],
    )
    _write_xyz(
        second_dir / "frame_0006_AAA.xyz",
        [
            "4",
            "PbI3 frame 6",
            "Pb 10.0 0.0 0.0",
            "I 12.9 0.0 0.0",
            "I 10.0 2.9 0.0",
            "I 10.0 0.0 2.9",
        ],
    )

    window = ElectronDensityMappingMainWindow(initial_input_path=clusters_dir)
    assert window.contiguous_frame_mode_checkbox.isChecked()

    window.run_button.click()
    _wait_for(
        lambda: all(
            state.profile_result is not None
            for state in window._cluster_group_states
        )
        and window._calculation_thread is None,
        qapp,
    )

    assert all(
        state.profile_result is not None
        and state.profile_result.contiguous_frame_mode_applied
        and len(state.profile_result.contiguous_frame_sets) == 1
        for state in window._cluster_group_states
    )
    window.close()


def test_cluster_folder_run_reports_specific_progress_with_overall_bar(
    qapp,
    tmp_path,
):
    clusters_dir = tmp_path / "progress_clusters"
    first_dir = clusters_dir / "PbI2"
    second_dir = clusters_dir / "PbI3"
    first_dir.mkdir(parents=True)
    second_dir.mkdir(parents=True)
    _write_xyz(
        first_dir / "frame_0001_AAA.xyz",
        [
            "3",
            "PbI2 frame 1",
            "Pb 0.0 0.0 0.0",
            "I 2.8 0.0 0.0",
            "I 0.0 2.8 0.0",
        ],
    )
    _write_xyz(
        first_dir / "frame_0002_AAA.xyz",
        [
            "3",
            "PbI2 frame 2",
            "Pb 9.0 0.0 0.0",
            "I 11.9 0.0 0.0",
            "I 9.0 2.9 0.0",
        ],
    )
    _write_xyz(
        second_dir / "frame_0005_AAA.xyz",
        [
            "4",
            "PbI3 frame 5",
            "Pb 0.0 0.0 0.0",
            "I 2.8 0.0 0.0",
            "I 0.0 2.8 0.0",
            "I 0.0 0.0 2.8",
        ],
    )
    _write_xyz(
        second_dir / "frame_0006_AAA.xyz",
        [
            "4",
            "PbI3 frame 6",
            "Pb 9.0 0.0 0.0",
            "I 11.9 0.0 0.0",
            "I 9.0 2.9 0.0",
            "I 9.0 0.0 2.9",
        ],
    )

    window = ElectronDensityMappingMainWindow(initial_input_path=clusters_dir)

    window.run_button.click()
    _wait_for(
        lambda: all(
            state.profile_result is not None
            for state in window._cluster_group_states
        )
        and window._calculation_thread is None,
        qapp,
    )

    log_text = window.status_text.toPlainText()

    assert not window.calculation_overall_progress_bar.isHidden()
    assert window.calculation_overall_progress_bar.maximum() == 2
    assert window.calculation_overall_progress_bar.value() == 2
    assert "2/2 groups complete" in (
        window.calculation_overall_progress_message.text().lower()
    )
    assert (
        "Prepared 2 electron-density profiles across 2 stoichiometry folders."
        in log_text
    )
    assert "PbI2: Contiguous-frame evaluation locked 1 frame set" in log_text
    assert "PbI3: Contiguous-frame evaluation locked 1 frame set" in log_text
    assert (
        "Preparing ensemble electron-density calculation for 2 structures"
        not in (log_text)
    )
    assert "Loading structure 1/2: frame_0001_AAA.xyz" not in log_text
    window.close()
