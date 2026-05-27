from __future__ import annotations

import os
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

from saxshell.bondanalysis import (
    AngleTripletDefinition,
    BondAnalysisPreset,
    BondPairDefinition,
    CoordinationNumberDefinition,
)
from saxshell.bondanalysis.ui.batch_queue_window import (
    BondAnalysisBatchJob,
    BondAnalysisBatchQueueWindow,
    BondAnalysisBatchWorker,
)
from saxshell.saxs.project_manager import SAXSProjectManager


@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


def _write_xyz_cluster(
    path: Path,
    atoms: list[tuple[str, float, float, float]],
) -> None:
    lines = [str(len(atoms)), path.stem]
    for element, x_coord, y_coord, z_coord in atoms:
        lines.append(f"{element} {x_coord:.3f} {y_coord:.3f} {z_coord:.3f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_project_with_clusters(
    tmp_path: Path,
    name: str,
) -> tuple[Path, Path]:
    manager = SAXSProjectManager()
    project_dir = tmp_path / name
    settings = manager.create_project(project_dir)
    clusters_dir = project_dir / "clusters_splitxyz0001"
    pbi2_dir = clusters_dir / "PbI2"
    pbo_dir = clusters_dir / "PbO"
    pbi2_dir.mkdir(parents=True)
    pbo_dir.mkdir(parents=True)
    _write_xyz_cluster(
        pbi2_dir / "frame_0000_AAA.xyz",
        [
            ("Pb", 0.0, 0.0, 0.0),
            ("I", 2.0, 0.0, 0.0),
            ("I", 0.0, 2.0, 0.0),
        ],
    )
    _write_xyz_cluster(
        pbo_dir / "frame_0001_AAA.xyz",
        [
            ("Pb", 0.0, 0.0, 0.0),
            ("O", 1.8, 0.0, 0.0),
        ],
    )
    settings.clusters_dir = str(clusters_dir.resolve())
    manager.save_project(settings)
    return project_dir, clusters_dir


def _preset() -> BondAnalysisPreset:
    return BondAnalysisPreset(
        name="Pb-I",
        bond_pairs=(BondPairDefinition("Pb", "I", 2.5),),
        angle_triplets=(AngleTripletDefinition("Pb", "I", "I", 2.5, 2.5),),
        coordination_numbers=(CoordinationNumberDefinition("Pb", "I", 2.5),),
    )


def test_batch_queue_prefills_current_project_and_builds_jobs(
    qapp,
    tmp_path,
    monkeypatch,
):
    del qapp
    monkeypatch.setenv(
        "SAXSHELL_BONDANALYSIS_PRESETS_PATH",
        str(tmp_path / "presets.json"),
    )
    project_dir, clusters_dir = _build_project_with_clusters(
        tmp_path,
        "MAPBI_DMSO_project",
    )

    window = BondAnalysisBatchQueueWindow(initial_project_dir=project_dir)

    assert window.queue_table.rowCount() == 1
    assert window.queue_table.item(0, window.COL_PROJECT).text() == str(
        project_dir.resolve()
    )
    assert window.queue_table.item(0, window.COL_CLUSTERS).text() == str(
        clusters_dir.resolve()
    )
    assert window._row_preset_name(0) == "DMSO"

    [(item_id, job)] = window.queue_jobs_in_order()
    assert item_id
    assert job.project_dir == project_dir.resolve()
    assert job.clusters_dir == clusters_dir.resolve()
    assert (
        job.output_dir
        == (
            project_dir / "analysis" / "bondanalysis" / "clusters_splitxyz0001"
        ).resolve()
    )
    assert job.preset_name == "DMSO"
    assert not hasattr(window, "manifest_csv_edit")
    window.close()


def test_batch_queue_can_apply_one_preset_to_all_rows(
    qapp,
    tmp_path,
    monkeypatch,
):
    del qapp
    monkeypatch.setenv(
        "SAXSHELL_BONDANALYSIS_PRESETS_PATH",
        str(tmp_path / "presets.json"),
    )
    window = BondAnalysisBatchQueueWindow()
    window.add_queue_item()
    window.add_queue_item()
    dmf_index = window.global_preset_combo.findData("DMF")

    window.global_preset_combo.setCurrentIndex(dmf_index)
    window._apply_preset_to_all()

    assert window._row_preset_name(0) == "DMF"
    assert window._row_preset_name(1) == "DMF"
    window.close()


def test_batch_worker_writes_outputs_to_project_folder(qapp, tmp_path):
    del qapp
    project_dir, clusters_dir = _build_project_with_clusters(
        tmp_path,
        "batch_project",
    )
    output_dir = project_dir / "analysis" / "bondanalysis" / "queue_run"
    job = BondAnalysisBatchJob(
        project_dir=project_dir.resolve(),
        clusters_dir=clusters_dir.resolve(),
        output_dir=output_dir,
        preset_name="Pb-I",
        preset=_preset(),
    )
    worker = BondAnalysisBatchWorker(
        [("queue-item", job)],
    )
    summaries = []
    failures = []
    log_messages = []
    worker.finished.connect(summaries.append)
    worker.failed.connect(lambda item_id, message: failures.append(message))
    worker.log.connect(log_messages.append)

    worker.run()

    assert failures == []
    assert len(summaries) == 1
    summary = summaries[0]
    assert not hasattr(summary.results[0], "result")
    assert summary.csv_count > 0
    assert summary.output_dirs == (output_dir,)
    assert not (
        project_dir
        / "analysis"
        / "bondanalysis"
        / "bondanalysis_batch_csv_manifest.csv"
    ).exists()

    csv_paths = set(summary.results[0].csv_files)
    relative_paths = {
        str(path.relative_to(summary.results[0].output_dir))
        for path in csv_paths
    }
    assert all(path.exists() for path in csv_paths)
    assert "all_clusters/Pb_I_histogram.csv" in relative_paths
    assert "all_clusters/Pb_I_distribution.csv" in relative_paths
    assert "all_clusters/Pb_I_I_histogram.csv" in relative_paths
    assert "all_clusters/CN_Pb_I_histogram.csv" in relative_paths
    assert "all_clusters/CN_Pb_I_coordination.csv" in relative_paths
    assert summary.results[0].results_index_path.exists()
    assert any(
        f"Output directory: {output_dir}" in message
        for message in log_messages
    )
    assert any(
        f"All-cluster tables: {output_dir / 'all_clusters'}" in message
        for message in log_messages
    )
    assert any(
        f"Per-cluster-type tables: {output_dir / 'cluster_types'}" in message
        for message in log_messages
    )
    assert any(
        f"Comparison tables: {output_dir / 'comparisons'}" in message
        for message in log_messages
    )
