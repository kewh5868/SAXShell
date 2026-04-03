from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("xraydb")

import saxshell.saxs.contrast.representatives as contrast_representatives_module  # noqa: E402
from saxshell.saxs.contrast.debye import (  # noqa: E402
    build_contrast_component_profiles,
    compute_contrast_debye_intensity,
)
from saxshell.saxs.contrast.electron_density import (  # noqa: E402
    CONTRAST_SOLVENT_METHOD_DIRECT,
    CONTRAST_SOLVENT_METHOD_NEAT,
    CONTRAST_SOLVENT_METHOD_REFERENCE,
    ContrastGeometryDensitySettings,
    ContrastSolventDensitySettings,
    compute_contrast_geometry_and_electron_density,
)
from saxshell.saxs.contrast.representatives import (  # noqa: E402
    analyze_contrast_representatives,
)
from saxshell.saxs.contrast.settings import (  # noqa: E402
    COMPONENT_BUILD_MODE_CONTRAST,
    COMPONENT_BUILD_MODE_NO_CONTRAST,
    ContrastRepresentativeSamplerSettings,
)
from saxshell.saxs.debye import ClusterBin  # noqa: E402
from saxshell.saxs.debye.profiles import compute_debye_intensity  # noqa: E402
from saxshell.saxs.project_manager import (  # noqa: E402
    SAXSProjectManager,
    project_artifact_paths,
)


def _write_xyz(
    path: Path, rows: list[tuple[str, float, float, float]]
) -> None:
    path.write_text(
        "\n".join(
            [
                str(len(rows)),
                path.stem,
                *(
                    f"{element} {x:.6f} {y:.6f} {z:.6f}"
                    for element, x, y, z in rows
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _build_contrast_cluster_fixture(tmp_path: Path) -> tuple[Path, Path]:
    project_dir = tmp_path / "contrast_project"
    project_dir.mkdir()
    clusters_dir = tmp_path / "clusters"
    clusters_dir.mkdir()

    pbi2_dir = clusters_dir / "PbI2"
    pbi2_dir.mkdir()
    _write_xyz(
        pbi2_dir / "frame_0001.xyz",
        [
            ("Pb", 0.0, 0.0, 0.0),
            ("I", 3.0, 0.0, 0.0),
            ("I", 0.7, 1.8, 0.0),
            ("O", 0.0, 0.0, 2.6),
            ("O", 0.0, 0.0, 5.1),
        ],
    )
    _write_xyz(
        pbi2_dir / "frame_0002.xyz",
        [
            ("Pb", 0.0, 0.0, 0.0),
            ("I", 2.1, 0.0, 0.0),
            ("I", 0.0, 2.1, 0.0),
            ("O", 0.0, 0.0, 2.8),
            ("O", 0.0, 0.0, 5.2),
        ],
    )
    _write_xyz(
        pbi2_dir / "frame_0003.xyz",
        [
            ("Pb", 0.0, 0.0, 0.0),
            ("I", 1.5, 0.0, 0.0),
            ("I", -1.4, 1.6, 0.0),
            ("O", 0.0, 0.0, 3.1),
        ],
    )

    motif_dir = clusters_dir / "PbI3" / "motif_1"
    motif_dir.mkdir(parents=True)
    _write_xyz(
        motif_dir / "frame_0001.xyz",
        [
            ("Pb", 0.0, 0.0, 0.0),
            ("I", 2.2, 0.0, 0.0),
            ("I", 0.0, 2.2, 0.0),
            ("I", 0.0, 0.0, 2.2),
        ],
    )
    return project_dir, clusters_dir


def _build_reference_solvent_box(
    tmp_path: Path,
    *,
    edge_length: float = 24.0,
    spacing: float = 4.0,
) -> Path:
    coordinates: list[tuple[str, float, float, float]] = []
    samples = int(edge_length // spacing)
    for x_index in range(samples + 1):
        for y_index in range(samples + 1):
            for z_index in range(samples + 1):
                coordinates.append(
                    (
                        "O",
                        float(x_index) * spacing,
                        float(y_index) * spacing,
                        float(z_index) * spacing,
                    )
                )
    reference_path = tmp_path / "reference_solvent.xyz"
    _write_xyz(reference_path, coordinates)
    return reference_path


def _build_large_contrast_cluster_fixture(
    tmp_path: Path,
    *,
    file_count: int = 18,
) -> tuple[Path, Path]:
    project_dir = tmp_path / "contrast_project_large"
    project_dir.mkdir()
    clusters_dir = tmp_path / "clusters_large"
    clusters_dir.mkdir()
    pbi2_dir = clusters_dir / "PbI2"
    pbi2_dir.mkdir()
    for index in range(file_count):
        shift = float(index % 6) * 0.06
        shell_shift = float(index % 5) * 0.05
        _write_xyz(
            pbi2_dir / f"frame_{index + 1:04d}.xyz",
            [
                ("Pb", 0.0, 0.0, 0.0),
                ("I", 2.05 + shift, 0.0, 0.0),
                ("I", 0.0, 2.05 + shift, 0.0),
                ("O", 0.0, 0.0, 2.75 + shell_shift),
                ("O", 0.0, 0.0, 5.15 + shell_shift),
            ],
        )
    return project_dir, clusters_dir


def _build_hydrogen_contrast_cluster_fixture(
    tmp_path: Path,
) -> tuple[Path, Path]:
    project_dir = tmp_path / "contrast_project_h"
    project_dir.mkdir()
    clusters_dir = tmp_path / "clusters_h"
    clusters_dir.mkdir()

    structure_dir = clusters_dir / "PbIH2"
    structure_dir.mkdir()
    _write_xyz(
        structure_dir / "frame_0001.xyz",
        [
            ("Pb", 0.0, 0.0, 0.0),
            ("I", 2.6, 0.0, 0.0),
            ("H", 0.0, 1.1, 0.0),
            ("H", 0.0, -1.1, 0.0),
        ],
    )
    _write_xyz(
        structure_dir / "frame_0002.xyz",
        [
            ("Pb", 0.0, 0.0, 0.0),
            ("I", 2.4, 0.0, 0.0),
            ("H", 0.0, 1.0, 0.2),
            ("H", 0.0, -1.0, -0.2),
        ],
    )
    return project_dir, clusters_dir


def _build_contrast_project_settings(tmp_path: Path):
    manager = SAXSProjectManager()
    project_dir, clusters_dir = _build_contrast_cluster_fixture(tmp_path)
    settings = manager.create_project(project_dir)
    settings.model_only_mode = True
    settings.clusters_dir = str(clusters_dir)
    settings.use_experimental_grid = False
    settings.q_min = 0.05
    settings.q_max = 0.30
    settings.q_points = 8
    settings.selected_model_template = (
        "template_pd_likelihood_monosq_decoupled"
    )
    settings.component_build_mode = COMPONENT_BUILD_MODE_CONTRAST
    manager.save_project(settings)
    return manager, settings


def test_analyze_contrast_representatives_selects_existing_files_and_writes_outputs(
    tmp_path,
):
    project_dir, clusters_dir = _build_contrast_cluster_fixture(tmp_path)
    progress_updates: list[tuple[int, int, str]] = []
    logs: list[str] = []

    result = analyze_contrast_representatives(
        project_dir,
        clusters_dir,
        progress_callback=lambda processed, total, message: progress_updates.append(
            (processed, total, message)
        ),
        log_callback=logs.append,
    )

    assert result.project_dir == project_dir.resolve()
    assert result.clusters_dir == clusters_dir.resolve()
    assert (
        result.output_dir
        == (project_dir / "contrast_workflow" / "representatives").resolve()
    )
    assert len(result.bin_results) == 2
    assert result.summary_json_path.is_file()
    assert result.summary_table_path.is_file()
    assert result.summary_text_path.is_file()
    assert progress_updates[-1][0] == 2
    assert progress_updates[-1][1] == 2
    assert "complete" in progress_updates[-1][2].lower()
    assert any(
        "selected frame_0002.xyz for pbi2" in log.lower() for log in logs
    )

    pbi2_result = next(
        bin_result
        for bin_result in result.bin_results
        if bin_result.structure == "PbI2"
    )
    assert pbi2_result.selected_file.name == "frame_0002.xyz"
    assert pbi2_result.copied_representative_file.is_file()
    assert pbi2_result.screening_json_path.is_file()
    assert pbi2_result.screening_table_path.is_file()
    assert (
        pbi2_result.selected_candidate.descriptor.direct_solvent_atom_count
        == 1
    )
    assert (
        pbi2_result.selected_candidate.descriptor.outer_solvent_atom_count == 1
    )

    screening_payload = json.loads(
        pbi2_result.screening_json_path.read_text(encoding="utf-8")
    )
    assert screening_payload["selected_file"].endswith("frame_0002.xyz")
    assert (
        "mean_direct_solvent_coordination"
        in screening_payload["target_summary"]["solvent_metrics"]
    )
    assert len(screening_payload["candidates"]) == 3

    summary_payload = json.loads(
        result.summary_json_path.read_text(encoding="utf-8")
    )
    assert len(summary_payload["bin_results"]) == 2
    assert summary_payload["bin_results"][0]["screening_json_path"].endswith(
        ".json"
    )
    assert (
        "Contrast representative selection complete"
        in result.summary_text_path.read_text(encoding="utf-8")
    )


def test_analyze_contrast_representatives_sampling_is_seeded_and_bounded(
    tmp_path,
):
    project_dir, clusters_dir = _build_large_contrast_cluster_fixture(tmp_path)
    sampler_settings = ContrastRepresentativeSamplerSettings.from_values(
        full_scan_threshold=0,
        target_distribution_samples=7,
        minimum_candidate_samples=4,
        max_candidate_samples=6,
        candidate_batch_size=2,
        random_seed=17,
        convergence_patience=2,
        improvement_tolerance=0.0001,
        stratify_sampling=True,
    )

    first = analyze_contrast_representatives(
        project_dir,
        clusters_dir,
        sampler_settings=sampler_settings,
    )
    second = analyze_contrast_representatives(
        project_dir,
        clusters_dir,
        sampler_settings=sampler_settings,
    )

    assert len(first.bin_results) == 1
    assert len(second.bin_results) == 1
    first_bin = first.bin_results[0]
    second_bin = second.bin_results[0]
    assert first_bin.selection_strategy == "monte_carlo_sampling"
    assert first_bin.distribution_sample_count <= 7
    assert first_bin.sampled_candidate_count <= 6
    assert first_bin.sampled_candidate_count == len(first_bin.candidates)
    assert first_bin.selected_file == second_bin.selected_file

    screening_payload = json.loads(
        first_bin.screening_json_path.read_text(encoding="utf-8")
    )
    assert screening_payload["selection_strategy"] == "monte_carlo_sampling"
    assert screening_payload["distribution_sample_count"] <= 7
    assert screening_payload["sampled_candidate_count"] <= 6
    assert screening_payload["sampler_settings"]["random_seed"] == 17
    assert len(screening_payload["candidates"]) <= 6


def test_sampling_logs_true_lowest_score_even_below_convergence_tolerance(
    tmp_path,
    monkeypatch,
):
    source_dir = tmp_path / "clusters" / "Pb2I3"
    source_dir.mkdir(parents=True)
    file_paths = (
        source_dir / "frame_27308_AAB.pdb",
        source_dir / "frame_27534_AAB.pdb",
        source_dir / "frame_28001_AAB.pdb",
        source_dir / "frame_28100_AAB.pdb",
        source_dir / "frame_28200_AAB.pdb",
    )
    for file_path in file_paths:
        file_path.write_text("MODEL        1\nENDMDL\n", encoding="utf-8")

    cluster_bin = ClusterBin(
        structure="Pb2I3",
        motif="no_motif",
        source_dir=source_dir,
        files=file_paths,
        representative=None,
    )
    target_summary = (
        contrast_representatives_module.ContrastRepresentativeTargetSummary(
            pair_contact_distance_medians={},
            bond_length_medians={},
            angle_medians={},
            coordination_medians={},
            solvent_metrics={},
        )
    )
    score_lookup = {
        file_paths[0]: 0.0500,
        file_paths[1]: 0.0432,
        file_paths[2]: 0.0424,
        file_paths[3]: 0.0490,
        file_paths[4]: 0.0510,
    }
    sample_call_count = {"count": 0}

    def fake_sample_indices(
        total_count,
        sample_count,
        *,
        rng,
        stratify,
        shuffle,
    ):
        del total_count, sample_count, rng, stratify, shuffle
        sample_call_count["count"] += 1
        if sample_call_count["count"] == 1:
            return (0, 1, 2, 3)
        return (0, 1, 2)

    monkeypatch.setattr(
        contrast_representatives_module,
        "_sample_indices",
        fake_sample_indices,
    )
    monkeypatch.setattr(
        contrast_representatives_module,
        "_load_parsed_structure_cached",
        lambda file_path, **kwargs: SimpleNamespace(file_path=file_path),
    )
    monkeypatch.setattr(
        contrast_representatives_module,
        "_evaluate_descriptor_candidates",
        lambda parsed_structures, *, expected_core_counts: (
            {},
            target_summary,
            tuple(),
        ),
    )
    monkeypatch.setattr(
        contrast_representatives_module,
        "_describe_candidate_cached",
        lambda file_path, **kwargs: SimpleNamespace(file_path=file_path),
    )

    def fake_candidate_for_descriptor(descriptor, *, target_summary):
        del target_summary
        score = score_lookup[descriptor.file_path]
        return contrast_representatives_module.ContrastRepresentativeCandidate(
            descriptor=SimpleNamespace(file_path=descriptor.file_path),
            score_total=score,
            score_bond=score,
            score_angle=score,
            score_coordination=score,
            score_solvent=score,
        )

    monkeypatch.setattr(
        contrast_representatives_module,
        "_candidate_for_descriptor",
        fake_candidate_for_descriptor,
    )

    logs: list[str] = []
    sampler_settings = ContrastRepresentativeSamplerSettings.from_values(
        full_scan_threshold=0,
        target_distribution_samples=4,
        minimum_candidate_samples=1,
        max_candidate_samples=3,
        candidate_batch_size=1,
        random_seed=17,
        convergence_patience=10,
        improvement_tolerance=0.0025,
        stratify_sampling=False,
    )

    (
        _target_summary,
        candidates,
        selection_strategy,
        distribution_sample_count,
        sampled_candidate_count,
    ) = contrast_representatives_module._sampled_descriptor_candidates(
        cluster_bin,
        expected_core_counts={"Pb": 2, "I": 3},
        sampler_settings=sampler_settings,
        log_callback=logs.append,
        parsed_cache={},
        issues=[],
    )

    progress_logs = [
        log
        for log in logs
        if "current best descriptor-distance match is" in log.lower()
    ]
    assert selection_strategy == "monte_carlo_sampling"
    assert distribution_sample_count == 4
    assert sampled_candidate_count == 3
    assert candidates[0].descriptor.file_path == file_paths[2]
    assert progress_logs
    assert "frame_28001_aab.pdb" in progress_logs[-1].lower()
    assert "0.0424" in progress_logs[-1]
    assert "lower is better" in progress_logs[-1].lower()


def test_compute_contrast_debye_intensity_applies_solvent_displacement_form_factor():
    coordinates = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [2.4, 0.0, 0.0],
        ],
        dtype=float,
    )
    q_values = np.asarray([0.05, 0.10, 0.20], dtype=float)
    base_f0_dictionary = {"O": np.asarray([8.0, 7.4, 6.6], dtype=float)}

    base_intensity = compute_debye_intensity(
        coordinates,
        ["O", "O"],
        q_values,
        f0_dictionary=base_f0_dictionary,
    )
    contrast_intensity = compute_contrast_debye_intensity(
        coordinates,
        ["O", "O"],
        q_values,
        cluster_density_e_per_a3=1.6,
        solvent_density_e_per_a3=0.4,
        f0_dictionary=base_f0_dictionary,
    )

    radius_a = 0.66
    excluded_volume_a3 = (4.0 / 3.0) * np.pi * radius_a**3
    qr_values = q_values * radius_a
    sphere_form_factor = np.ones_like(q_values)
    nonzero_mask = np.abs(qr_values) > 1.0e-12
    sphere_form_factor[nonzero_mask] = (
        3.0
        * (
            np.sin(qr_values[nonzero_mask])
            - qr_values[nonzero_mask] * np.cos(qr_values[nonzero_mask])
        )
        / (qr_values[nonzero_mask] ** 3)
    )
    effective_f0_dictionary = {
        "O": base_f0_dictionary["O"]
        - (0.4 * excluded_volume_a3 * sphere_form_factor)
    }
    expected_intensity = compute_debye_intensity(
        coordinates,
        ["O", "O"],
        q_values,
        f0_dictionary=effective_f0_dictionary,
    )
    np.testing.assert_allclose(
        contrast_intensity,
        expected_intensity,
        rtol=1e-10,
        atol=1e-10,
    )
    assert not np.allclose(
        contrast_intensity,
        base_intensity * (((1.6 - 0.4) / 1.6) ** 2),
    )


def test_compute_contrast_debye_intensity_matches_vacuum_when_solvent_density_is_zero():
    coordinates = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [2.4, 0.0, 0.0],
        ],
        dtype=float,
    )
    q_values = np.asarray([0.05, 0.10, 0.20], dtype=float)
    base_f0_dictionary = {"O": np.asarray([8.0, 7.4, 6.6], dtype=float)}

    base_intensity = compute_debye_intensity(
        coordinates,
        ["O", "O"],
        q_values,
        f0_dictionary=base_f0_dictionary,
    )
    contrast_intensity = compute_contrast_debye_intensity(
        coordinates,
        ["O", "O"],
        q_values,
        cluster_density_e_per_a3=1.6,
        solvent_density_e_per_a3=0.0,
        f0_dictionary=base_f0_dictionary,
    )

    np.testing.assert_allclose(
        contrast_intensity,
        base_intensity,
        rtol=1e-10,
        atol=1e-10,
    )


def test_compute_contrast_geometry_density_with_neat_solvent_estimate_writes_outputs(
    tmp_path,
):
    project_dir, clusters_dir = _build_contrast_cluster_fixture(tmp_path)
    representative_result = analyze_contrast_representatives(
        project_dir,
        clusters_dir,
    )

    density_result = compute_contrast_geometry_and_electron_density(
        representative_result,
        ContrastGeometryDensitySettings(
            solvent=ContrastSolventDensitySettings.from_values(
                method=CONTRAST_SOLVENT_METHOD_NEAT,
                solvent_formula="H2O",
                solvent_density_g_per_ml=1.0,
            )
        ),
    )

    assert density_result.output_dir == representative_result.output_dir
    assert density_result.geometry_dir.is_dir()
    assert density_result.density_dir.is_dir()
    assert density_result.summary_json_path.is_file()
    assert density_result.summary_table_path.is_file()
    assert density_result.summary_text_path.is_file()
    assert len(density_result.bin_results) == 2

    pbi2_result = next(
        bin_result
        for bin_result in density_result.bin_results
        if bin_result.structure == "PbI2"
    )
    assert pbi2_result.mesh_json_path.is_file()
    assert pbi2_result.density_json_path.is_file()
    assert pbi2_result.mesh.volume_a3 > 0.0
    assert pbi2_result.mesh.surface_area_a2 > 0.0
    assert len(pbi2_result.mesh.vertices) >= 4
    assert len(pbi2_result.mesh.faces) >= 4
    assert (
        pbi2_result.cluster_electron_density.electron_density_e_per_a3
        > pbi2_result.solvent_electron_density.electron_density_e_per_a3
    )
    assert (
        pbi2_result.solvent_electron_density.method
        == CONTRAST_SOLVENT_METHOD_NEAT
    )
    assert pbi2_result.solvent_electron_density.formula == "H2O"

    mesh_payload = json.loads(
        pbi2_result.mesh_json_path.read_text(encoding="utf-8")
    )
    density_payload = json.loads(
        pbi2_result.density_json_path.read_text(encoding="utf-8")
    )
    summary_payload = json.loads(
        density_result.summary_json_path.read_text(encoding="utf-8")
    )

    assert mesh_payload["construction_method"] in {
        "expanded_convex_hull",
        "padded_bounding_box",
    }
    assert (
        density_payload["cluster_electron_density"][
            "electron_density_e_per_a3"
        ]
        > 0.0
    )
    assert (
        density_payload["solvent_electron_density"]["method"]
        == CONTRAST_SOLVENT_METHOD_NEAT
    )
    assert len(summary_payload["bin_results"]) == 2
    assert "Contrast geometry and electron-density calculation complete" in (
        density_result.summary_text_path.read_text(encoding="utf-8")
    )


def test_build_contrast_component_profiles_writes_component_outputs(tmp_path):
    project_dir, clusters_dir = _build_contrast_cluster_fixture(tmp_path)
    representative_result = analyze_contrast_representatives(
        project_dir,
        clusters_dir,
    )
    density_result = compute_contrast_geometry_and_electron_density(
        representative_result,
        ContrastGeometryDensitySettings(
            solvent=ContrastSolventDensitySettings.from_values(
                method=CONTRAST_SOLVENT_METHOD_NEAT,
                solvent_formula="H2O",
                solvent_density_g_per_ml=1.0,
            )
        ),
    )
    q_values = np.linspace(0.05, 0.30, 8)
    output_dir = tmp_path / "contrast_components"
    metadata_dir = tmp_path / "contrast_metadata"
    component_map_path = tmp_path / "md_saxs_map.json"

    build_result = build_contrast_component_profiles(
        representative_result,
        density_result,
        q_values=q_values,
        output_dir=output_dir,
        metadata_dir=metadata_dir,
        component_map_path=component_map_path,
    )

    assert len(build_result.trace_results) == 2
    assert build_result.component_map_path == component_map_path.resolve()
    assert build_result.summary_json_path.is_file()
    assert build_result.summary_table_path.is_file()
    assert build_result.summary_text_path.is_file()
    assert component_map_path.is_file()

    component_map_payload = json.loads(
        component_map_path.read_text(encoding="utf-8")
    )
    assert set(component_map_payload["saxs_map"]) == {"PbI2", "PbI3"}

    first_trace = build_result.trace_results[0]
    assert first_trace.profile_path.is_file()
    written_data = np.loadtxt(first_trace.profile_path, comments="#")
    written_data = np.atleast_2d(np.asarray(written_data, dtype=float))
    np.testing.assert_allclose(written_data[:, 0], q_values)
    np.testing.assert_allclose(written_data[:, 1], first_trace.intensity)
    assert first_trace.contrast_scale_factor > 0.0

    summary_payload = json.loads(
        build_result.summary_json_path.read_text(encoding="utf-8")
    )
    assert len(summary_payload["trace_results"]) == 2
    assert (
        "Contrast Debye scattering build complete"
        in build_result.summary_text_path.read_text(encoding="utf-8")
    )


def test_compute_contrast_geometry_density_with_reference_solvent_structure(
    tmp_path,
):
    project_dir, clusters_dir = _build_contrast_cluster_fixture(tmp_path)
    representative_result = analyze_contrast_representatives(
        project_dir,
        clusters_dir,
    )
    reference_solvent_path = _build_reference_solvent_box(tmp_path)

    density_result = compute_contrast_geometry_and_electron_density(
        representative_result,
        ContrastGeometryDensitySettings(
            solvent=ContrastSolventDensitySettings.from_values(
                method=CONTRAST_SOLVENT_METHOD_REFERENCE,
                reference_structure_file=reference_solvent_path,
            )
        ),
    )

    pbi2_result = next(
        bin_result
        for bin_result in density_result.bin_results
        if bin_result.structure == "PbI2"
    )
    assert (
        pbi2_result.solvent_electron_density.method
        == CONTRAST_SOLVENT_METHOD_REFERENCE
    )
    assert (
        pbi2_result.solvent_electron_density.reference_structure_file
        == reference_solvent_path.resolve()
    )
    assert pbi2_result.solvent_electron_density.atom_count is not None
    assert pbi2_result.solvent_electron_density.atom_count > 0
    assert pbi2_result.solvent_electron_density.reference_box_spans is not None
    assert pbi2_result.solvent_electron_density.electron_density_e_per_a3 > 0.0

    density_payload = json.loads(
        pbi2_result.density_json_path.read_text(encoding="utf-8")
    )
    assert (
        density_payload["solvent_electron_density"]["method"]
        == CONTRAST_SOLVENT_METHOD_REFERENCE
    )
    assert density_payload["solvent_electron_density"]["atom_count"] > 0


def test_compute_contrast_geometry_density_with_vacuum_solvent_preset(
    tmp_path,
):
    project_dir, clusters_dir = _build_contrast_cluster_fixture(tmp_path)
    representative_result = analyze_contrast_representatives(
        project_dir,
        clusters_dir,
    )

    density_result = compute_contrast_geometry_and_electron_density(
        representative_result,
        ContrastGeometryDensitySettings(
            solvent=ContrastSolventDensitySettings.from_values(
                method=CONTRAST_SOLVENT_METHOD_NEAT,
                solvent_formula="Vacuum",
                solvent_density_g_per_ml=0.0,
            )
        ),
    )

    pbi2_result = next(
        bin_result
        for bin_result in density_result.bin_results
        if bin_result.structure == "PbI2"
    )
    assert (
        pbi2_result.solvent_electron_density.electron_density_e_per_a3
        == pytest.approx(0.0)
    )
    assert (
        pbi2_result.solvent_electron_density.total_electrons
        == pytest.approx(0.0)
    )
    assert pbi2_result.solvent_electron_density.formula == "Vacuum"


def test_compute_contrast_geometry_density_with_direct_electron_density_value(
    tmp_path,
):
    project_dir, clusters_dir = _build_contrast_cluster_fixture(tmp_path)
    representative_result = analyze_contrast_representatives(
        project_dir,
        clusters_dir,
    )

    density_result = compute_contrast_geometry_and_electron_density(
        representative_result,
        ContrastGeometryDensitySettings(
            solvent=ContrastSolventDensitySettings.from_values(
                method=CONTRAST_SOLVENT_METHOD_DIRECT,
                direct_electron_density_e_per_a3=0.275,
            )
        ),
    )

    pbi2_result = next(
        bin_result
        for bin_result in density_result.bin_results
        if bin_result.structure == "PbI2"
    )
    assert (
        pbi2_result.solvent_electron_density.method
        == CONTRAST_SOLVENT_METHOD_DIRECT
    )
    assert (
        pbi2_result.solvent_electron_density.electron_density_e_per_a3
        == pytest.approx(0.275)
    )

    density_payload = json.loads(
        pbi2_result.density_json_path.read_text(encoding="utf-8")
    )
    assert (
        density_payload["solvent_electron_density"]["method"]
        == CONTRAST_SOLVENT_METHOD_DIRECT
    )
    assert density_payload["solvent_electron_density"][
        "electron_density_e_per_a3"
    ] == pytest.approx(0.275)


def test_contrast_hydrogen_exclusion_applies_to_density_and_debye(tmp_path):
    project_dir, clusters_dir = _build_hydrogen_contrast_cluster_fixture(
        tmp_path
    )
    representative_result = analyze_contrast_representatives(
        project_dir,
        clusters_dir,
    )
    density_all = compute_contrast_geometry_and_electron_density(
        representative_result,
        ContrastGeometryDensitySettings(
            solvent=ContrastSolventDensitySettings.from_values(
                method=CONTRAST_SOLVENT_METHOD_NEAT,
                solvent_formula="Vacuum",
                solvent_density_g_per_ml=0.0,
            )
        ),
    )
    density_no_h = compute_contrast_geometry_and_electron_density(
        representative_result,
        ContrastGeometryDensitySettings(
            solvent=ContrastSolventDensitySettings.from_values(
                method=CONTRAST_SOLVENT_METHOD_NEAT,
                solvent_formula="Vacuum",
                solvent_density_g_per_ml=0.0,
            ),
            exclude_elements=("H",),
        ),
    )
    q_values = np.linspace(0.05, 0.30, 8)
    build_all = build_contrast_component_profiles(
        representative_result,
        density_all,
        q_values=q_values,
        output_dir=tmp_path / "contrast_components_all",
        metadata_dir=tmp_path / "contrast_metadata_all",
        component_map_path=tmp_path / "md_saxs_map_all.json",
    )
    build_no_h = build_contrast_component_profiles(
        representative_result,
        density_no_h,
        q_values=q_values,
        output_dir=tmp_path / "contrast_components_no_h",
        metadata_dir=tmp_path / "contrast_metadata_no_h",
        component_map_path=tmp_path / "md_saxs_map_no_h.json",
    )

    density_bin_all = density_all.bin_results[0]
    density_bin_no_h = density_no_h.bin_results[0]
    assert (
        density_bin_no_h.cluster_electron_density.atom_count
        < density_bin_all.cluster_electron_density.atom_count
    )
    assert (
        density_bin_no_h.cluster_electron_density.total_electrons
        < density_bin_all.cluster_electron_density.total_electrons
    )

    trace_all = build_all.trace_results[0]
    trace_no_h = build_no_h.trace_results[0]
    assert trace_no_h.atom_count < trace_all.atom_count
    assert trace_no_h.element_counts.get("H", 0) == 0
    assert not np.allclose(trace_no_h.intensity, trace_all.intensity)


def test_reference_solvent_structure_must_be_larger_than_largest_mesh(
    tmp_path,
):
    project_dir, clusters_dir = _build_contrast_cluster_fixture(tmp_path)
    representative_result = analyze_contrast_representatives(
        project_dir,
        clusters_dir,
    )
    too_small_reference = tmp_path / "small_reference.xyz"
    _write_xyz(
        too_small_reference,
        [
            ("O", 0.0, 0.0, 0.0),
            ("O", 3.0, 0.0, 0.0),
            ("O", 0.0, 3.0, 0.0),
            ("O", 0.0, 0.0, 3.0),
        ],
    )

    with pytest.raises(
        ValueError,
        match="not larger than the largest retained representative mesh",
    ):
        compute_contrast_geometry_and_electron_density(
            representative_result,
            ContrastGeometryDensitySettings(
                solvent=ContrastSolventDensitySettings.from_values(
                    method=CONTRAST_SOLVENT_METHOD_REFERENCE,
                    reference_structure_file=too_small_reference,
                )
            ),
        )


def test_project_manager_build_scattering_components_persists_contrast_distribution(
    tmp_path,
):
    manager, settings = _build_contrast_project_settings(tmp_path)

    build_result = manager.build_scattering_components(settings)
    artifact_paths = project_artifact_paths(
        settings,
        storage_mode="distribution",
        allow_legacy_fallback=False,
    )
    metadata_payload = json.loads(
        artifact_paths.distribution_metadata_file.read_text(encoding="utf-8")
    )

    assert len(build_result.component_entries) == 2
    assert build_result.model_map_path == artifact_paths.component_map_file
    assert artifact_paths.component_dir.is_dir()
    assert sorted(
        path.name for path in artifact_paths.component_dir.glob("*.txt")
    ) == [entry.profile_file for entry in build_result.component_entries]
    assert artifact_paths.contrast_dir.is_dir()
    assert (artifact_paths.contrast_dir / "representative_structures").is_dir()
    assert (artifact_paths.contrast_dir / "screening").is_dir()
    assert (artifact_paths.contrast_dir / "geometry").is_dir()
    assert (artifact_paths.contrast_dir / "electron_density").is_dir()
    assert (
        artifact_paths.contrast_dir / "debye" / "component_summary.json"
    ).is_file()
    assert (
        metadata_payload["component_build_mode"]
        == COMPONENT_BUILD_MODE_CONTRAST
    )
    assert metadata_payload["component_artifacts_ready"] is True


def test_project_manager_builds_contrast_components_for_predicted_structure_bins(
    tmp_path,
    monkeypatch,
):
    manager, settings = _build_contrast_project_settings(tmp_path)
    settings.use_predicted_structure_weights = True
    manager.save_project(settings)

    predicted_dir = tmp_path / "predicted_structures"
    predicted_dir.mkdir(parents=True, exist_ok=True)
    predicted_structure = predicted_dir / "04_rank01_PbI4.xyz"
    _write_xyz(
        predicted_structure,
        [
            ("Pb", 0.0, 0.0, 0.0),
            ("I", 2.6, 0.0, 0.0),
            ("I", -2.6, 0.0, 0.0),
            ("I", 0.0, 2.6, 0.0),
            ("I", 0.0, 0.0, 2.6),
        ],
    )

    def fake_predicted_payload(self, settings, *, cluster_inventory):
        del self, settings
        observed_weight = 0.75 / max(len(cluster_inventory.cluster_bins), 1)
        return (
            SimpleNamespace(dataset_file=tmp_path / "mock_predicted.json"),
            {
                (cluster_bin.structure, cluster_bin.motif): observed_weight
                for cluster_bin in cluster_inventory.cluster_bins
            },
            [
                {
                    "prediction": SimpleNamespace(label="PbI4"),
                    "motif": "predicted_rank01",
                    "weight": 0.25,
                    "source_path": predicted_structure,
                }
            ],
            ["Pb", "I"],
        )

    monkeypatch.setattr(
        SAXSProjectManager,
        "_predicted_structure_weight_payload",
        fake_predicted_payload,
    )

    build_result = manager.build_scattering_components(settings)
    artifact_paths = project_artifact_paths(
        settings,
        storage_mode="distribution",
        allow_legacy_fallback=False,
    )
    selection_payload = json.loads(
        (artifact_paths.contrast_dir / "selection_summary.json").read_text(
            encoding="utf-8"
        )
    )
    debye_payload = json.loads(
        (
            artifact_paths.contrast_dir / "debye" / "component_summary.json"
        ).read_text(encoding="utf-8")
    )

    predicted_entry = next(
        entry
        for entry in build_result.component_entries
        if entry.structure == "PbI4"
    )
    predicted_bin = next(
        bin_result
        for bin_result in selection_payload["bin_results"]
        if bin_result["structure"] == "PbI4"
    )
    predicted_trace = next(
        trace
        for trace in debye_payload["trace_results"]
        if trace["structure"] == "PbI4"
    )

    assert build_result.used_predicted_structure_weights is True
    assert build_result.predicted_component_count == 1
    assert predicted_entry.motif == "predicted_rank01"
    assert predicted_entry.representative.endswith(predicted_structure.name)
    assert (
        artifact_paths.component_dir / predicted_entry.profile_file
    ).is_file()
    assert predicted_bin["selected_file"].endswith(predicted_structure.name)
    assert predicted_trace["representative_file"].endswith(
        predicted_structure.name
    )


def test_generate_prior_weights_for_contrast_distribution_uses_representative_traces(
    tmp_path,
):
    manager, settings = _build_contrast_project_settings(tmp_path)

    build_result = manager.build_scattering_components(settings)
    prior_result = manager.generate_prior_weights(settings)
    artifact_paths = project_artifact_paths(
        settings,
        storage_mode="distribution",
        allow_legacy_fallback=False,
    )

    prior_payload = json.loads(
        artifact_paths.prior_weights_file.read_text(encoding="utf-8")
    )
    summary_payload = json.loads(
        (
            artifact_paths.contrast_dir / "debye" / "component_summary.json"
        ).read_text(encoding="utf-8")
    )
    trace_by_key = {
        (str(trace["structure"]), str(trace["motif"])): trace
        for trace in summary_payload["trace_results"]
    }

    assert (
        prior_result.md_prior_weights_path == artifact_paths.prior_weights_file
    )
    assert {
        (entry.structure, entry.motif, entry.representative)
        for entry in prior_result.component_entries
    } == {
        (
            entry.structure,
            entry.motif,
            entry.representative,
        )
        for entry in build_result.component_entries
    }

    pbi2_payload = prior_payload["structures"]["PbI2"]["no_motif"]
    pbi3_payload = prior_payload["structures"]["PbI3"]["motif_1"]
    representative_root = (
        artifact_paths.contrast_dir / "representative_structures"
    ).resolve()

    assert pbi2_payload["count"] == 3
    assert pbi2_payload["weight"] == pytest.approx(0.75)
    assert pbi2_payload["representative"] == "PbI2__frame_0002.xyz"
    assert (
        pbi2_payload["profile_file"]
        == trace_by_key[("PbI2", "no_motif")]["profile_file"]
    )
    assert pbi2_payload["source_dir"] == str(representative_root)
    assert pbi2_payload["source_file"] == str(
        (representative_root / "PbI2__frame_0002.xyz").resolve()
    )

    assert pbi3_payload["count"] == 1
    assert pbi3_payload["weight"] == pytest.approx(0.25)
    assert pbi3_payload["representative"] == "PbI3__motif_1__frame_0001.xyz"
    assert (
        pbi3_payload["profile_file"]
        == trace_by_key[("PbI3", "motif_1")]["profile_file"]
    )
    assert pbi3_payload["source_dir"] == str(representative_root)
    assert pbi3_payload["source_file"] == str(
        (representative_root / "PbI3__motif_1__frame_0001.xyz").resolve()
    )


def test_generate_prior_weights_keeps_no_contrast_cluster_mapping_unchanged(
    tmp_path,
):
    manager = SAXSProjectManager()
    project_dir, clusters_dir = _build_contrast_cluster_fixture(tmp_path)
    settings = manager.create_project(project_dir)
    settings.model_only_mode = True
    settings.clusters_dir = str(clusters_dir)
    settings.use_experimental_grid = False
    settings.q_min = 0.05
    settings.q_max = 0.30
    settings.q_points = 8
    settings.selected_model_template = (
        "template_pd_likelihood_monosq_decoupled"
    )
    settings.component_build_mode = COMPONENT_BUILD_MODE_NO_CONTRAST
    manager.save_project(settings)

    manager.generate_prior_weights(settings)
    artifact_paths = project_artifact_paths(
        settings,
        storage_mode="distribution",
        allow_legacy_fallback=False,
    )
    prior_payload = json.loads(
        artifact_paths.prior_weights_file.read_text(encoding="utf-8")
    )

    pbi2_payload = prior_payload["structures"]["PbI2"]["no_motif"]
    pbi3_payload = prior_payload["structures"]["PbI3"]["motif_1"]

    assert pbi2_payload["count"] == 3
    assert pbi2_payload["weight"] == pytest.approx(0.75)
    assert pbi2_payload["representative"] == "frame_0001.xyz"
    assert pbi2_payload["source_dir"] == str((clusters_dir / "PbI2").resolve())
    assert pbi2_payload["source_file"] == str(
        (clusters_dir / "PbI2" / "frame_0001.xyz").resolve()
    )

    assert pbi3_payload["count"] == 1
    assert pbi3_payload["weight"] == pytest.approx(0.25)
    assert pbi3_payload["representative"] == "frame_0001.xyz"
    assert pbi3_payload["source_dir"] == str(
        (clusters_dir / "PbI3" / "motif_1").resolve()
    )
    assert pbi3_payload["source_file"] == str(
        (clusters_dir / "PbI3" / "motif_1" / "frame_0001.xyz").resolve()
    )
