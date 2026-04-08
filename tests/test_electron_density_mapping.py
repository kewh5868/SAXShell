from __future__ import annotations

import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from matplotlib.backend_bases import MouseButton
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QApplication

from saxshell.saxs.contrast.electron_density import (
    CONTRAST_SOLVENT_METHOD_DIRECT,
    ContrastSolventDensitySettings,
)
from saxshell.saxs.electron_density_mapping.ui.main_window import (
    ElectronDensityMappingMainWindow,
)
from saxshell.saxs.electron_density_mapping.workflow import (
    ElectronDensityFourierTransformSettings,
    ElectronDensityMeshSettings,
    ElectronDensitySmearingSettings,
    apply_smearing_to_profile_result,
    apply_solvent_contrast_to_profile_result,
    compute_electron_density_profile,
    compute_electron_density_profile_for_input,
    compute_electron_density_scattering_profile,
    inspect_structure_input,
    load_electron_density_structure,
    prepare_electron_density_fourier_transform,
    write_electron_density_profile_outputs,
)

xraydb = pytest.importorskip("xraydb")


@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


def _write_xyz(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


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


def _wait_for(condition, qapp, *, timeout_s: float = 10.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        qapp.processEvents()
        if condition():
            return
        time.sleep(0.01)
    raise AssertionError("Timed out waiting for condition.")


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

    structure = load_electron_density_structure(structure_path)
    expected_x = (
        float(xraydb.atomic_mass("Pb")) * 2.0
        + float(xraydb.atomic_mass("H")) * 0.0
    ) / (float(xraydb.atomic_mass("Pb")) + float(xraydb.atomic_mass("H")))

    assert structure.center_of_mass[0] == pytest.approx(expected_x)
    assert structure.center_of_mass[0] != pytest.approx(1.0)
    assert np.allclose(structure.active_center, structure.center_of_mass)
    assert structure.center_mode == "center_of_mass"
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

    structure = load_electron_density_structure(structure_path)
    reference_centered = load_electron_density_structure(
        structure_path,
        center_mode="reference_element",
    )

    assert structure.reference_element == "Pb"
    assert np.allclose(structure.geometric_center, [0.5, 0.5, 0.5])
    assert np.allclose(
        structure.reference_element_geometric_center,
        [1.0, 0.0, 0.0],
    )
    assert structure.reference_element_offset_from_geometric_center == (
        pytest.approx(np.sqrt(0.75))
    )
    assert reference_centered.center_mode == "reference_element"
    assert np.allclose(
        reference_centered.active_center,
        structure.reference_element_geometric_center,
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

    assert float(np.sum(result.shell_electron_counts)) == pytest.approx(
        float(np.sum(structure.atomic_numbers))
    )
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
    expected_first_shell_density = float(
        structure.atomic_numbers[0] / result.shell_volumes[0]
    )

    assert result.shell_electron_counts[0] == pytest.approx(
        float(structure.atomic_numbers[0])
    )
    assert result.orientation_average_density[0] == pytest.approx(
        expected_first_shell_density
    )
    assert result.orientation_average_density[0] == pytest.approx(
        result.shell_volume_average_density[0]
    )


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
    mesh_settings = _mesh_settings_for_structure(
        reference_structure,
        rstep=0.1,
        theta_divisions=24,
        phi_divisions=18,
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
        load_electron_density_structure(first_path),
        mesh_settings,
        smearing_settings=ElectronDensitySmearingSettings(
            debye_waller_factor=0.006
        ),
    )
    second = compute_electron_density_profile(
        load_electron_density_structure(second_path),
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
    assert progress_events
    assert "Averaging electron density" in progress_events[-2][2]


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
        assert preview.available_r_min == pytest.approx(0.0)
        assert preview.source_radial_values[0] == pytest.approx(0.0)
        center_index = int(np.argmax(preview.window_values))
        assert center_index in {
            len(preview.window_values) // 2,
            len(preview.window_values) // 2 - 1,
        }
        assert preview.window_values[0] <= preview.window_values[center_index]
        assert preview.window_values[-1] <= preview.window_values[center_index]


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
    assert preview.source_radial_values[0] == pytest.approx(0.0)
    assert preview.source_density_values[0] == pytest.approx(
        contrasted.solvent_contrast.solvent_subtracted_smeared_density[0]
    )
    assert np.allclose(
        preview.source_density_values[1:],
        contrasted.solvent_contrast.solvent_subtracted_smeared_density,
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
    assert "center_of_mass_a" in payload
    assert "geometric_center_a" in payload
    assert "reference_element" in payload
    assert "active_center_a" in payload
    assert "mesh_settings" in payload
    assert "smearing_settings" in payload


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
    assert len(window.profile_plot.figure.axes[0].collections) >= 1

    window.show_variance_checkbox.setChecked(False)

    assert len(window.profile_plot.figure.axes[0].collections) == 0
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
    assert window._fourier_preview.source_radial_values[0] == pytest.approx(
        0.0
    )
    assert window._fourier_preview.source_density_values[0] == pytest.approx(
        window._profile_result.smeared_orientation_average_density[0]
    )
    assert np.allclose(
        window._fourier_preview.source_density_values[1:],
        window._profile_result.smeared_orientation_average_density,
    )
    window.close()


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

    assert window.rstep_spin.value() == pytest.approx(0.1)
    assert window.rmax_spin.value() == pytest.approx(
        window._structure.rmax,
        abs=1.0e-4,
    )
    assert "calculated center" in window.reset_center_button.text().lower()
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
