from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QAbstractItemView, QApplication, QWidget

import saxshell.pdf.debyer.ui.batch_queue_window as batch_queue_module
import saxshell.pdf.debyer.workflow as debyer_workflow
import saxshell.pdfsetup as pdfsetup_module
from saxshell import saxshell as saxshell_module
from saxshell.pdf.debyer.ui.batch_queue_window import (
    DebyerPDFBatchItem,
    DebyerPDFBatchItemWidget,
    DebyerPDFBatchQueueWindow,
    DebyerPDFBatchWorker,
    DebyerPDFExistingPartialsJob,
    DebyerPDFExistingPartialsWorker,
)
from saxshell.pdf.debyer.ui.main_window import (
    DebyerPDFMainWindow,
    DebyerPDFWorker,
)
from saxshell.pdf.debyer.workflow import (
    DebyerPDFCalculation,
    DebyerPDFSettings,
    DebyerPDFWorkflow,
    calculate_number_density,
    compute_experimental_fit_metrics,
    convert_distribution_values,
    fit_coordination_peak_from_r,
    list_saved_debyer_calculations,
    load_debyer_calculation,
    parse_debyer_output_file,
)
from saxshell.saxs.project_manager import SAXSProjectManager


@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


def _write_fake_debyer(path: Path, *, sleep_seconds: float = 0.0) -> Path:
    script = """#!/usr/bin/env python3
import re
import sys
import time
from pathlib import Path

import numpy as np


def main(argv):
    if "--help" in argv or len(argv) <= 1:
        print("fake debyer help")
        return 0

    output = None
    mode = "PDF"
    input_file = None
    for arg in argv[1:]:
        if arg.startswith("--output="):
            output = Path(arg.split("=", 1)[1])
        elif arg in ("--PDF", "--RDF", "--rPDF"):
            mode = arg[2:]
        elif not arg.startswith("--"):
            input_file = Path(arg)

    if output is None or input_file is None:
        print("missing output or input", file=sys.stderr)
        return 2

    digits = re.findall(r"(\\d+)", input_file.stem)
    frame_index = int(digits[-1]) if digits else 0
    scale = 1.0 + frame_index
    sleep_seconds = __SLEEP_SECONDS__
    if sleep_seconds > 0.0:
        time.sleep(sleep_seconds)
    r_values = np.array([0.5, 1.0, 1.5], dtype=float)
    partial_pbpb = scale * np.array([0.10, 0.12, 0.14], dtype=float)
    partial_pbi = scale * np.array([0.55, 0.60, 0.65], dtype=float)
    partial_ii = scale * np.array([0.35, 0.38, 0.41], dtype=float)
    total = partial_pbpb + partial_pbi + partial_ii

    if mode == "RDF":
        total = total * 2.0
        partial_pbpb = partial_pbpb * 2.0
        partial_pbi = partial_pbi * 2.0
        partial_ii = partial_ii * 2.0
    elif mode == "rPDF":
        total = total - 0.25
        partial_pbpb = partial_pbpb - 0.05
        partial_pbi = partial_pbi - 0.10
        partial_ii = partial_ii - 0.10

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        handle.write("# fake debyer output\\n")
        handle.write("# sum Pb-Pb Pb-I I-I\\n")
        for row in zip(r_values, total, partial_pbpb, partial_pbi, partial_ii):
            handle.write(" ".join(f"{value:.6f}" for value in row) + "\\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
"""
    script = script.replace("__SLEEP_SECONDS__", repr(float(sleep_seconds)))
    path.write_text(script, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def _build_frames_dir(
    tmp_path: Path,
    *,
    frame_count: int = 2,
    xyz: str | None = None,
) -> Path:
    frames_dir = tmp_path / "splitxyz_f0_t0fs"
    frames_dir.mkdir(parents=True)
    xyz_text = (
        xyz
        or "3\n"
        "frame\n"
        "Pb 0.0 0.0 0.0\n"
        "I 2.0 0.0 0.0\n"
        "I 0.0 2.0 0.0\n"
    )
    for index in range(frame_count):
        (frames_dir / f"frame_{index:04d}.xyz").write_text(
            xyz_text,
            encoding="utf-8",
        )
    return frames_dir


def _pdb_atom_line(
    serial: int,
    atom_name: str,
    residue_name: str,
    residue_id: int,
    element: str,
    x: float,
    y: float,
    z: float,
) -> str:
    return (
        f"ATOM  {serial:5d} {atom_name:<4s} {residue_name:>3s} "
        f"A{residue_id:4d}    {x:8.3f}{y:8.3f}{z:8.3f}"
        f"  1.00  0.00          {element:>2s}\n"
    )


def _build_pdb_frames_dir(tmp_path: Path, *, frame_count: int = 2) -> Path:
    frames_dir = tmp_path / "splitpdb_f0_t0fs"
    frames_dir.mkdir(parents=True)
    pdb_text = "".join(
        [
            _pdb_atom_line(1, "PB", "PER", 1, "Pb", 0.0, 0.0, 0.0),
            _pdb_atom_line(2, "I1", "PER", 1, "I", 2.0, 0.0, 0.0),
            _pdb_atom_line(3, "O1", "DMS", 2, "O", 0.0, 2.0, 0.0),
            _pdb_atom_line(4, "C1", "DMS", 2, "C", 0.0, 0.0, 2.0),
            "END\n",
        ]
    )
    for index in range(frame_count):
        (frames_dir / f"frame_{index:04d}.pdb").write_text(
            pdb_text,
            encoding="utf-8",
        )
    return frames_dir


def _write_pbc_source_file(frames_dir: Path, token: str = "12x10x8") -> Path:
    source = frames_dir.parent / f"sample_pbc_{token}-pos-1.xyz"
    source.write_text(
        "1\nsource\nPb 0.0 0.0 0.0\n",
        encoding="utf-8",
    )
    return source


def test_convert_distribution_values_from_pdf_mode():
    r_values = np.array([0.5, 1.0, 1.5], dtype=float)
    g_values = np.array([1.0, 1.1, 1.2], dtype=float)
    rho0 = 0.2
    converted_r = convert_distribution_values(
        g_values,
        r_values=r_values,
        rho0=rho0,
        source_mode="PDF",
        target_representation="R(r)",
    )
    converted_g = convert_distribution_values(
        g_values,
        r_values=r_values,
        rho0=rho0,
        source_mode="PDF",
        target_representation="G(r)",
    )
    expected_r = 4.0 * np.pi * rho0 * (r_values**2) * g_values
    expected_g = 4.0 * np.pi * rho0 * r_values * (g_values - 1.0)
    assert np.allclose(converted_r, expected_r)
    assert np.allclose(converted_g, expected_g)


def test_infers_default_solute_elements_from_frame_elements():
    assert debyer_workflow.infer_default_solute_elements(
        {"Pb": 1, "I": 2}
    ) == ("Pb", "I")
    assert debyer_workflow.infer_default_solute_elements(
        {"Cs": 1, "Pb": 1, "I": 3}
    ) == ("Cs", "Pb", "I")
    assert (
        debyer_workflow.infer_default_solute_elements({"Na": 1, "Cl": 1}) == ()
    )


def test_pdfsetup_rejects_pdb_frame_folders(tmp_path):
    frames_dir = _build_pdb_frames_dir(tmp_path)

    with pytest.raises(ValueError, match="require XYZ frame files"):
        debyer_workflow.inspect_frames_dir(frames_dir)


def test_pdf_batch_item_uses_saved_or_manual_xyz_settings(qapp, tmp_path):
    project_dir = tmp_path / "project"
    frames_dir = _build_frames_dir(
        tmp_path,
        xyz=(
            "3\n"
            "frame\n"
            "Cs 0.0 0.0 0.0\n"
            "Pb 2.0 0.0 0.0\n"
            "I 0.0 2.0 0.0\n"
        ),
    )
    widget = DebyerPDFBatchItemWidget(
        DebyerPDFBatchItem(
            item_id="batch-item",
            project_dir=project_dir,
            frames_dir=frames_dir,
            filename_prefix="stored_pdf",
            box_dimensions=(12.0, 10.0, 8.0),
            atom_count=3,
            solute_elements=("Cs", "Pb", "I"),
        )
    )
    widget.to_edit.setText("25")

    settings = widget.settings()

    assert not hasattr(widget, "inspect_button")
    assert not hasattr(widget, "inspect_frames")
    assert settings.project_dir == project_dir.resolve()
    assert settings.frames_dir == frames_dir.resolve()
    assert settings.filename_prefix == "stored_pdf"
    assert settings.atom_count == 3
    assert settings.box_dimensions == pytest.approx((12.0, 10.0, 8.0))
    assert settings.to_value == pytest.approx(4.0)
    assert settings.solute_elements == ("Cs", "Pb", "I")


def test_pdf_batch_queue_window_keeps_collapsible_reorderable_items(
    qapp,
    tmp_path,
):
    first_frames_dir = _build_frames_dir(tmp_path / "first")
    second_frames_dir = _build_frames_dir(tmp_path / "second")
    window = DebyerPDFBatchQueueWindow()

    first = window.add_queue_item(
        DebyerPDFBatchItem(
            item_id="first",
            project_dir=tmp_path / "first_project",
            frames_dir=first_frames_dir,
            filename_prefix="first_pdf",
            box_dimensions=(10.0, 10.0, 10.0),
            atom_count=3,
        )
    )
    second = window.add_queue_item(
        DebyerPDFBatchItem(
            item_id="second",
            project_dir=tmp_path / "second_project",
            frames_dir=second_frames_dir,
            filename_prefix="second_pdf",
            box_dimensions=(11.0, 11.0, 11.0),
            atom_count=3,
        )
    )

    assert (
        window.queue_list.dragDropMode()
        == QAbstractItemView.DragDropMode.InternalMove
    )
    assert first.settings_group.isHidden()
    assert second.settings_group.isHidden()
    first._set_settings_visible(True)
    assert not first.settings_group.isHidden()
    first._set_settings_visible(False)
    assert first.settings_group.isHidden()
    window.queue_list.setCurrentItem(window.queue_list.item(0))
    window._refresh_item_selection_styles()
    assert first.header_frame.property("selected") is True
    assert second.header_frame.property("selected") is False
    assert [
        item_id for item_id, _settings in window.queue_settings_in_order()
    ] == [
        "first",
        "second",
    ]
    assert second.item().display_name() == "second_project"


def test_pdf_batch_queue_adds_multiple_selected_project_folders(
    qapp,
    tmp_path,
    monkeypatch,
):
    project_a = tmp_path / "project_a"
    project_b = tmp_path / "project_b"
    project_a.mkdir()
    project_b.mkdir()
    window = DebyerPDFBatchQueueWindow()
    monkeypatch.setattr(
        batch_queue_module,
        "_choose_existing_directories",
        lambda *_args, **_kwargs: (project_a.resolve(), project_b.resolve()),
    )

    window._choose_project_to_add()

    assert window.queue_list.count() == 2
    project_dirs = []
    for row in range(window.queue_list.count()):
        item_id = str(
            window.queue_list.item(row).data(Qt.ItemDataRole.UserRole)
        )
        project_dirs.append(window._widgets_by_id[item_id].item().project_dir)
    assert project_dirs == [project_a.resolve(), project_b.resolve()]


def test_pdf_batch_queue_shows_progress_dialog_for_multiple_project_load(
    qapp,
    tmp_path,
    monkeypatch,
):
    project_a = tmp_path / "project_a"
    project_b = tmp_path / "project_b"
    project_a.mkdir()
    project_b.mkdir()
    window = DebyerPDFBatchQueueWindow()
    monkeypatch.setattr(
        batch_queue_module,
        "_choose_existing_directories",
        lambda *_args, **_kwargs: (project_a.resolve(), project_b.resolve()),
    )
    dialogs: list[object] = []

    class FakeProgressDialog:
        def __init__(self, label, cancel_text, minimum, maximum, parent):
            del cancel_text, parent
            self.labels = [label]
            self.values: list[int] = []
            self.minimum = minimum
            self.maximum = maximum
            self.closed = False
            self.shown = False
            dialogs.append(self)

        def setWindowTitle(self, title):
            self.title = title

        def setWindowModality(self, modality):
            self.modality = modality

        def setMinimumDuration(self, duration):
            self.minimum_duration = duration

        def setAutoClose(self, enabled):
            self.auto_close = enabled

        def setAutoReset(self, enabled):
            self.auto_reset = enabled

        def setValue(self, value):
            self.values.append(value)

        def show(self):
            self.shown = True

        def setLabelText(self, label):
            self.labels.append(label)

        def close(self):
            self.closed = True

    monkeypatch.setattr(
        batch_queue_module,
        "QProgressDialog",
        FakeProgressDialog,
    )

    window._choose_project_to_add()

    assert len(dialogs) == 1
    dialog = dialogs[0]
    assert dialog.maximum == 2
    assert dialog.shown is True
    assert dialog.closed is True
    assert dialog.values[-1] == 2
    assert any("project_a" in label for label in dialog.labels)
    assert any("project_b" in label for label in dialog.labels)
    assert window.queue_list.count() == 2


def test_pdf_batch_queue_prefills_project_debyer_defaults(
    qapp,
    tmp_path,
    monkeypatch,
):
    frames_dir = _build_frames_dir(
        tmp_path / "frames",
        xyz=(
            "3\n"
            "frame\n"
            "Cs 0.0 0.0 0.0\n"
            "Pb 2.0 0.0 0.0\n"
            "I 0.0 2.0 0.0\n"
        ),
    )
    project_dir = tmp_path / "project_defaults"
    manager = SAXSProjectManager()
    settings = manager.create_project(project_dir)
    settings.frames_dir = str(frames_dir)
    project_file = manager.save_project(settings)
    payload = json.loads(project_file.read_text(encoding="utf-8"))
    payload["debyer_pdf_settings"] = {
        "filename_prefix": "stored_pdf",
        "mode": "PDF",
        "from_value": 0.7,
        "to_value": 3.5,
        "step_value": 0.02,
        "box_dimensions": [12.0, 10.0, 8.0],
        "atom_count": 3,
        "solute_elements": ["Cs", "Pb", "I"],
        "parallel_jobs": 2,
    }
    project_file.write_text(json.dumps(payload), encoding="utf-8")
    window = DebyerPDFBatchQueueWindow()
    monkeypatch.setattr(
        batch_queue_module,
        "_choose_existing_directories",
        lambda *_args, **_kwargs: (project_dir.resolve(),),
    )

    window._choose_project_to_add()

    assert window.queue_list.count() == 1
    item_id = str(window.queue_list.item(0).data(Qt.ItemDataRole.UserRole))
    widget = window._widgets_by_id[item_id]
    queued_settings = widget.settings()
    assert queued_settings.project_dir == project_dir.resolve()
    assert queued_settings.frames_dir == frames_dir.resolve()
    assert queued_settings.filename_prefix == "stored_pdf"
    assert queued_settings.from_value == pytest.approx(0.7)
    assert queued_settings.to_value == pytest.approx(3.5)
    assert queued_settings.step_value == pytest.approx(0.02)
    assert queued_settings.box_dimensions == pytest.approx((12.0, 10.0, 8.0))
    assert queued_settings.atom_count == 3
    assert queued_settings.solute_elements == ("Cs", "Pb", "I")
    assert queued_settings.max_parallel_jobs == 2


def test_pdf_batch_queue_append_mode_uses_project_solute_jobs(
    qapp,
    tmp_path,
):
    project_dir = tmp_path / "project"
    window = DebyerPDFBatchQueueWindow()
    item_widget = window.add_queue_item(
        DebyerPDFBatchItem(
            item_id="existing",
            project_dir=project_dir,
            solute_elements=("Pb",),
        )
    )
    append_index = window.queue_mode_combo.findData("append_grouped")

    window.queue_mode_combo.setCurrentIndex(append_index)
    jobs = window.existing_partials_jobs_in_order()

    assert window.run_button.text() == "Append Grouped Partial Columns"
    assert window.add_frames_button.isEnabled() is False
    assert item_widget.frames_dir_edit.isEnabled() is False
    assert item_widget.solute_elements_edit.isEnabled() is True
    assert jobs == [
        (
            "existing",
            DebyerPDFExistingPartialsJob(
                project_dir=project_dir.resolve(),
                solute_elements=("Pb",),
            ),
        )
    ]

    item_widget.solute_elements_edit.setText("Cs, Pb, I")
    item_widget._on_editor_changed()
    jobs = window.existing_partials_jobs_in_order()

    assert jobs[0][1].solute_elements == ("Cs", "Pb", "I")


def test_compute_experimental_fit_metrics_interpolates_model_gr():
    metrics = compute_experimental_fit_metrics(
        model_r_values=np.array([0.0, 1.0, 2.0], dtype=float),
        model_g_values=np.array([1.0, 2.0, 3.0], dtype=float),
        experimental_r_values=np.array([0.5, 1.5], dtype=float),
        experimental_g_values=np.array([1.5, 2.5], dtype=float),
    )

    assert metrics is not None
    assert metrics.r_squared == pytest.approx(1.0)
    assert metrics.rmse == pytest.approx(0.0)
    assert metrics.mae == pytest.approx(0.0)
    assert metrics.point_count == 2
    assert metrics.r_min == pytest.approx(0.5)
    assert metrics.r_max == pytest.approx(1.5)


def test_fit_coordination_peak_from_r_recovers_gaussian_area():
    r_values = np.linspace(1.5, 3.5, 121)
    expected_cn = 4.25
    center = 2.62
    sigma = 0.11
    baseline = 0.35 + 0.08 * (r_values - center)
    r_values_distribution = baseline + (
        expected_cn
        / (sigma * np.sqrt(2.0 * np.pi))
        * np.exp(-0.5 * ((r_values - center) / sigma) ** 2)
    )

    result = fit_coordination_peak_from_r(
        r_values=r_values,
        r_distribution_values=r_values_distribution,
        r_min=2.2,
        r_max=3.05,
        initial_center=2.6,
        initial_sigma=0.12,
    )

    assert result.coordination_number == pytest.approx(expected_cn, rel=0.03)
    assert result.center == pytest.approx(center, abs=0.02)
    assert result.sigma == pytest.approx(sigma, abs=0.02)
    assert result.r_squared > 0.99


def test_running_debyer_average_matches_batch_average_without_frame_cache():
    r_values = np.linspace(0.1, 6.0, 240, dtype=float)
    batch_outputs = []
    running_average = debyer_workflow._RunningDebyerAverage()

    for index in range(250):
        scale = 1.0 + float(index)
        columns = {
            "sum": scale * np.sin(r_values),
            "Pb-I": scale * np.cos(r_values),
        }
        if index >= 25:
            columns["I-I"] = scale * (r_values**2)
        batch_outputs.append((r_values, columns))
        running_average.add_frame(r_values, columns)

    batch_r, batch_columns, batch_values = (
        debyer_workflow._average_frame_outputs(batch_outputs)
    )
    running_r, running_columns, running_values = running_average.average()
    raw_cache_bytes = sum(
        r_array.nbytes + sum(values.nbytes for values in columns.values())
        for r_array, columns in batch_outputs
    )

    assert np.allclose(running_r, batch_r)
    assert running_columns == batch_columns
    for key in batch_columns:
        assert np.allclose(running_values[key], batch_values[key])
    assert running_average.memory_bytes < raw_cache_bytes / 100


def test_debyer_workflow_averages_and_persists_calculation(tmp_path):
    fake_debyer = _write_fake_debyer(tmp_path / "debyer")
    frames_dir = _build_frames_dir(tmp_path)
    project_dir = tmp_path / "project"
    settings = DebyerPDFSettings(
        project_dir=project_dir,
        frames_dir=frames_dir,
        filename_prefix="demo_pdf",
        mode="PDF",
        from_value=0.5,
        to_value=15.0,
        step_value=0.01,
        box_dimensions=(10.0, 10.0, 10.0),
        atom_count=3,
        store_frame_outputs=False,
        solute_elements=("Pb",),
    )
    workflow = DebyerPDFWorkflow(settings, debyer_executable=fake_debyer)
    progress_messages: list[str] = []
    preview_results = []
    result = workflow.run(
        progress_callback=lambda _processed, _total, message: progress_messages.append(
            message
        ),
        preview_callback=preview_results.append,
    )

    assert result.frame_count == 2
    assert result.mode == "PDF"
    assert result.store_frame_outputs is False
    assert result.frame_output_dir is None
    assert result.averaged_output_file.is_file()
    assert np.allclose(result.r_values, np.array([0.5, 1.0, 1.5]))
    assert sorted(result.partial_values) == ["I-I", "Pb-I", "Pb-Pb"]
    assert np.allclose(
        result.total_values,
        np.array([1.5, 1.65, 1.8]),
    )
    assert result.rho0 == pytest.approx(
        calculate_number_density(3, (10.0, 10.0, 10.0))
    )
    assert progress_messages
    assert any("remaining" in message for message in progress_messages)
    assert preview_results
    assert preview_results[-1].processed_frame_count == result.frame_count
    assert preview_results[-1].averaged_output_file.is_file()
    averaged_text = result.averaged_output_file.read_text(encoding="utf-8")
    assert "# created_at:" in averaged_text
    assert "# processed_frames:" in averaged_text
    assert "# total_frames:" in averaged_text
    assert "# columns: sum" in averaged_text
    _output_r, output_values = parse_debyer_output_file(
        result.averaged_output_file
    )
    assert "solute-solute" in output_values
    assert "solute-solvent" in output_values
    assert "solvent-solvent" in output_values
    assert np.allclose(
        output_values["solute-solute"],
        result.partial_values["Pb-Pb"],
    )
    assert np.allclose(
        output_values["solute-solvent"],
        result.partial_values["Pb-I"],
    )
    assert np.allclose(
        output_values["solvent-solvent"],
        result.partial_values["I-I"],
    )

    summaries = list_saved_debyer_calculations(project_dir)
    assert len(summaries) == 1
    loaded = load_debyer_calculation(summaries[0].calculation_dir)
    assert loaded.filename_prefix == "demo_pdf"
    assert np.allclose(loaded.total_values, result.total_values)
    assert loaded.solute_elements == ("Pb",)
    assert sorted(loaded.partial_values) == ["I-I", "Pb-I", "Pb-Pb"]
    assert loaded.processed_frame_count == loaded.frame_count
    assert "Pb-I" in loaded.partial_peak_markers
    assert loaded.partial_peak_markers["Pb-I"]
    assert loaded.peak_finder_settings.max_peak_count >= 0


def test_pdf_batch_worker_runs_projects_in_sequence(qapp, tmp_path):
    fake_debyer = _write_fake_debyer(tmp_path / "debyer")
    first_frames_dir = _build_frames_dir(tmp_path / "first")
    second_frames_dir = _build_frames_dir(tmp_path / "second")
    entries = [
        (
            "first",
            DebyerPDFSettings(
                project_dir=tmp_path / "first_project",
                frames_dir=first_frames_dir,
                filename_prefix="first_pdf",
                mode="PDF",
                from_value=0.5,
                to_value=5.0,
                step_value=0.01,
                box_dimensions=(10.0, 10.0, 10.0),
                atom_count=3,
                solute_elements=("Pb", "I"),
            ),
        ),
        (
            "second",
            DebyerPDFSettings(
                project_dir=tmp_path / "second_project",
                frames_dir=second_frames_dir,
                filename_prefix="second_pdf",
                mode="PDF",
                from_value=0.5,
                to_value=5.0,
                step_value=0.01,
                box_dimensions=(11.0, 11.0, 11.0),
                atom_count=3,
                solute_elements=("Pb", "I"),
            ),
        ),
    ]
    worker = DebyerPDFBatchWorker(entries, debyer_executable=fake_debyer)
    started_items: list[str] = []
    finished_results: list[DebyerPDFCalculation] = []
    worker.item_started.connect(
        lambda item_id, _index, _total: started_items.append(item_id)
    )
    worker.finished.connect(lambda results: finished_results.extend(results))

    worker.run()

    assert started_items == ["first", "second"]
    assert [result.filename_prefix for result in finished_results] == [
        "first_pdf",
        "second_pdf",
    ]
    assert all(
        result.averaged_output_file.is_file() for result in finished_results
    )
    assert all(not result.is_partial_average for result in finished_results)
    for (_item_id, settings), result in zip(entries, finished_results):
        summaries = list_saved_debyer_calculations(settings.project_dir)
        assert len(summaries) == 1
        assert summaries[0].filename_prefix == result.filename_prefix
        loaded = load_debyer_calculation(summaries[0].calculation_dir)
        assert loaded.project_dir == settings.project_dir.resolve()
        assert loaded.frames_dir == settings.frames_dir.resolve()
        assert loaded.filename_prefix == result.filename_prefix
        assert loaded.mode == result.mode
        assert loaded.frame_count == result.frame_count
        assert loaded.processed_frame_count == loaded.frame_count
        assert np.allclose(loaded.total_values, result.total_values)
        assert loaded.solute_elements == ("Pb", "I")
        _output_r, output_values = parse_debyer_output_file(
            loaded.averaged_output_file
        )
        assert "solute-solute" in output_values
        assert "solute-solvent" not in output_values
        assert "solvent-solvent" not in output_values
        assert np.allclose(
            output_values["solute-solute"],
            sum(loaded.partial_values.values()),
        )
        assert (loaded.calculation_dir / "calculation.json").is_file()
        assert loaded.averaged_output_file.is_file()


def test_pdf_existing_partials_worker_updates_saved_calculations(
    qapp,
    tmp_path,
):
    fake_debyer = _write_fake_debyer(tmp_path / "debyer")
    frames_dir = _build_frames_dir(
        tmp_path,
        xyz=("2\n" "frame\n" "Na 0.0 0.0 0.0\n" "Cl 2.0 0.0 0.0\n"),
    )
    project_dir = tmp_path / "project"
    result = DebyerPDFWorkflow(
        DebyerPDFSettings(
            project_dir=project_dir,
            frames_dir=frames_dir,
            filename_prefix="existing_average",
            mode="PDF",
            from_value=0.5,
            to_value=5.0,
            step_value=0.01,
            box_dimensions=(10.0, 10.0, 10.0),
            atom_count=2,
            solute_elements=(),
        ),
        debyer_executable=fake_debyer,
    ).run()
    _r_values, before_values = parse_debyer_output_file(
        result.averaged_output_file
    )
    assert "solute-solute" not in before_values
    worker = DebyerPDFExistingPartialsWorker(
        [
            (
                "project",
                DebyerPDFExistingPartialsJob(
                    project_dir=project_dir,
                    solute_elements=("Pb",),
                ),
            )
        ]
    )
    updated_results: list[DebyerPDFCalculation] = []
    progress_messages: list[str] = []
    worker.item_progress.connect(
        lambda _item_id, _processed, _total, message: progress_messages.append(
            message
        )
    )
    worker.finished.connect(lambda results: updated_results.extend(results))

    worker.run()

    assert len(updated_results) == 1
    assert progress_messages
    _r_values, after_values = parse_debyer_output_file(
        result.averaged_output_file
    )
    assert "solute-solute" in after_values
    assert "solute-solvent" in after_values
    assert "solvent-solvent" in after_values
    loaded = load_debyer_calculation(result.calculation_dir)
    assert loaded.solute_elements == ("Pb",)
    assert sorted(loaded.partial_values) == ["I-I", "Pb-I", "Pb-Pb"]


def test_debyer_workflow_parallel_jobs_match_serial_average(tmp_path):
    fake_debyer = _write_fake_debyer(tmp_path / "debyer")
    frames_dir = _build_frames_dir(tmp_path, frame_count=8)

    serial = DebyerPDFWorkflow(
        DebyerPDFSettings(
            project_dir=tmp_path / "serial_project",
            frames_dir=frames_dir,
            filename_prefix="serial_pdf",
            mode="PDF",
            from_value=0.5,
            to_value=15.0,
            step_value=0.01,
            box_dimensions=(10.0, 10.0, 10.0),
            atom_count=3,
            store_frame_outputs=False,
            solute_elements=("Pb",),
            max_parallel_jobs=1,
        ),
        debyer_executable=fake_debyer,
    ).run()
    parallel = DebyerPDFWorkflow(
        DebyerPDFSettings(
            project_dir=tmp_path / "parallel_project",
            frames_dir=frames_dir,
            filename_prefix="parallel_pdf",
            mode="PDF",
            from_value=0.5,
            to_value=15.0,
            step_value=0.01,
            box_dimensions=(10.0, 10.0, 10.0),
            atom_count=3,
            store_frame_outputs=False,
            solute_elements=("Pb",),
            max_parallel_jobs=4,
        ),
        debyer_executable=fake_debyer,
    ).run()

    assert parallel.parallel_jobs == 4
    assert parallel.processed_frame_count == serial.processed_frame_count == 8
    assert np.allclose(parallel.total_values, serial.total_values)
    assert parallel.partial_values.keys() == serial.partial_values.keys()
    for key in parallel.partial_values:
        assert np.allclose(
            parallel.partial_values[key],
            serial.partial_values[key],
        )
    loaded = load_debyer_calculation(parallel.calculation_dir)
    assert loaded.parallel_jobs == 4
    averaged_text = parallel.averaged_output_file.read_text(encoding="utf-8")
    assert "# parallel_jobs: 4" in averaged_text


def test_debyer_workflow_defaults_solute_elements_when_omitted(tmp_path):
    fake_debyer = _write_fake_debyer(tmp_path / "debyer")
    frames_dir = _build_frames_dir(tmp_path)
    result = DebyerPDFWorkflow(
        DebyerPDFSettings(
            project_dir=tmp_path / "project",
            frames_dir=frames_dir,
            filename_prefix="default_solute_pdf",
            mode="PDF",
            from_value=0.5,
            to_value=15.0,
            step_value=0.01,
            box_dimensions=(10.0, 10.0, 10.0),
            atom_count=3,
            store_frame_outputs=False,
            solute_elements=(),
        ),
        debyer_executable=fake_debyer,
    ).run()

    assert result.solute_elements == ("Pb", "I")
    loaded = load_debyer_calculation(result.calculation_dir)
    assert loaded.solute_elements == ("Pb", "I")


def test_debyer_workflow_checkpoints_sparse_running_averages(tmp_path):
    fake_debyer = _write_fake_debyer(tmp_path / "debyer")
    frames_dir = _build_frames_dir(tmp_path, frame_count=12)
    project_dir = tmp_path / "project"
    workflow = DebyerPDFWorkflow(
        DebyerPDFSettings(
            project_dir=project_dir,
            frames_dir=frames_dir,
            filename_prefix="sparse_preview_pdf",
            mode="PDF",
            from_value=0.5,
            to_value=15.0,
            step_value=0.01,
            box_dimensions=(10.0, 10.0, 10.0),
            atom_count=3,
            store_frame_outputs=False,
            solute_elements=("Pb",),
        ),
        debyer_executable=fake_debyer,
    )
    preview_counts: list[int | None] = []

    result = workflow.run(
        preview_callback=lambda calculation: preview_counts.append(
            calculation.processed_frame_count
        )
    )

    assert preview_counts == [12]
    assert result.processed_frame_count == 12
    assert result.is_partial_average is False


def test_debyer_workflow_respects_live_preview_decision(tmp_path):
    fake_debyer = _write_fake_debyer(tmp_path / "debyer")
    frames_dir = _build_frames_dir(tmp_path, frame_count=12)
    project_dir = tmp_path / "project"
    workflow = DebyerPDFWorkflow(
        DebyerPDFSettings(
            project_dir=project_dir,
            frames_dir=frames_dir,
            filename_prefix="live_preview_pdf",
            mode="PDF",
            from_value=0.5,
            to_value=15.0,
            step_value=0.01,
            box_dimensions=(10.0, 10.0, 10.0),
            atom_count=3,
            store_frame_outputs=False,
            solute_elements=("Pb",),
        ),
        debyer_executable=fake_debyer,
    )
    previews = []
    preview_enabled = {"value": False}

    def update_preview_toggle(processed, _total, _message):
        if processed == 6:
            preview_enabled["value"] = True
        elif processed == 7:
            preview_enabled["value"] = False

    def should_preview(processed, _total, checkpoint_due):
        if processed == 6:
            assert checkpoint_due is False
        return preview_enabled["value"]

    result = workflow.run(
        progress_callback=update_preview_toggle,
        preview_callback=previews.append,
        preview_decision_callback=should_preview,
    )

    assert [preview.processed_frame_count for preview in previews] == [6]
    assert result.processed_frame_count == 12
    assert result.is_partial_average is False
    loaded = load_debyer_calculation(result.calculation_dir)
    assert loaded.processed_frame_count == 12


def test_debyer_worker_preview_toggle_requests_next_average(tmp_path):
    settings = DebyerPDFSettings(
        project_dir=tmp_path / "project",
        frames_dir=tmp_path / "frames",
        filename_prefix="worker_preview",
    )
    worker = DebyerPDFWorker(settings, preview_enabled=False)

    assert worker._should_emit_preview(1, 10, True) is False

    worker.set_preview_enabled(True)
    assert worker._should_emit_preview(2, 10, False) is True
    worker._emit_preview(object())
    assert worker._should_emit_preview(3, 10, False) is False
    assert worker._should_emit_preview(10, 10, True) is True

    worker.set_preview_enabled(False)
    assert worker._should_emit_preview(10, 10, True) is False


def test_debyer_window_clamps_rejected_r_range_maximum_to_half_min_box(
    qapp,
    tmp_path,
):
    del qapp
    frames_dir = _build_frames_dir(tmp_path)
    project_dir = tmp_path / "project"
    window = DebyerPDFMainWindow()
    window.project_dir_edit.setText(str(project_dir))
    window.frames_dir_edit.setText(str(frames_dir))
    window.filename_prefix_edit.setText("box_limited_pdf")
    window.from_edit.setText("0.5")
    window.to_edit.setText("15.0")
    window.step_edit.setText("0.01")
    window.box_a_edit.setText("20.0")
    window.box_b_edit.setText("8.0")
    window.box_c_edit.setText("12.0")
    window.atom_count_edit.setText("3")

    settings = window._build_settings()

    assert settings.to_value == pytest.approx(4.0)
    assert window.to_edit.text() == "4"
    assert (
        "half of the minimum box dimension"
        in window.statusBar().currentMessage()
    )
    window.close()


def test_debyer_window_defaults_solutes_from_inspected_frames(qapp, tmp_path):
    del qapp
    pb_i_frames = _build_frames_dir(tmp_path / "pb_i")
    cs_pb_i_frames = _build_frames_dir(
        tmp_path / "cs_pb_i",
        xyz=(
            "4\n"
            "frame\n"
            "Cs 0.0 0.0 0.0\n"
            "Pb 2.0 0.0 0.0\n"
            "I 0.0 2.0 0.0\n"
            "I 0.0 0.0 2.0\n"
        ),
    )
    window = DebyerPDFMainWindow()

    window.frames_dir_edit.setText(str(pb_i_frames))
    window._inspect_frames_dir()
    assert window.solute_elements_edit.text() == "Pb, I"
    assert "Default solutes: Pb, I" in window.frames_summary_label.text()

    window.solute_elements_edit.clear()
    window.frames_dir_edit.setText(str(cs_pb_i_frames))
    window._inspect_frames_dir()
    assert window.solute_elements_edit.text() == "Cs, Pb, I"
    assert "Default solutes: Cs, Pb, I" in window.frames_summary_label.text()
    window.close()


def test_debyer_window_splitter_handle_is_grabbable(qapp):
    del qapp
    window = DebyerPDFMainWindow()

    tab_names = [
        window.result_tabs.tabText(index)
        for index in range(window.result_tabs.count())
    ]
    assert "Shape Function Analysis" in tab_names
    assert "Fit" in tab_names
    assert tab_names.index("Shape Function Analysis") < tab_names.index("Fit")
    assert window._main_splitter.handleWidth() >= 12
    assert window._main_splitter.handle(1).toolTip()
    assert window._main_splitter.widget(1).minimumSizeHint().width() < 800
    assert (
        window.findChild(QWidget, "pdfPlotControls").minimumSizeHint().width()
        < 800
    )
    window.close()


def test_debyer_window_fits_coordination_number_from_r_trace(qapp, tmp_path):
    r_values = np.linspace(1.5, 3.5, 121)
    expected_cn = 3.75
    center = 2.55
    sigma = 0.13
    r_distribution = 0.2 + (
        expected_cn
        / (sigma * np.sqrt(2.0 * np.pi))
        * np.exp(-0.5 * ((r_values - center) / sigma) ** 2)
    )
    calculation = DebyerPDFCalculation(
        calculation_id="fit_demo",
        calculation_dir=tmp_path,
        created_at="2026-05-13T00:00:00",
        project_dir=tmp_path,
        frames_dir=tmp_path,
        frame_format="xyz",
        frame_count=1,
        filename_prefix="fit_demo",
        mode="RDF",
        from_value=1.5,
        to_value=3.5,
        step_value=float(r_values[1] - r_values[0]),
        box_dimensions=(20.0, 20.0, 20.0),
        box_source=None,
        box_source_kind=None,
        atom_count=2,
        rho0=1.0,
        store_frame_outputs=False,
        frame_output_dir=None,
        averaged_output_file=tmp_path / "averaged_raw.txt",
        solute_elements=("Pb",),
        parallel_jobs=1,
        r_values=r_values,
        total_values=r_distribution,
        partial_values={"Pb-I": r_distribution},
    )
    window = DebyerPDFMainWindow()
    window._apply_loaded_calculation(calculation)
    qapp.processEvents()

    trace_index = window.coordination_fit_trace_combo.findData("partial:Pb-I")
    assert trace_index >= 0
    window.coordination_fit_trace_combo.setCurrentIndex(trace_index)
    window.coordination_fit_r_min_spin.setValue(2.1)
    window.coordination_fit_r_max_spin.setValue(3.0)
    window.coordination_fit_center_spin.setValue(2.5)
    window.coordination_fit_sigma_spin.setValue(0.12)
    window._fit_coordination_number()
    qapp.processEvents()

    assert window.representation_combo.currentText() == "R(r)"
    assert window.coordination_fit_results_table.rowCount() == 1
    fitted_cn = float(window.coordination_fit_results_table.item(0, 5).text())
    assert fitted_cn == pytest.approx(expected_cn, rel=0.05)
    assert "CN =" in window.coordination_fit_status_label.text()
    window.close()


def test_debyer_window_applies_solute_groups_after_calculation(
    qapp,
    tmp_path,
):
    fake_debyer = _write_fake_debyer(tmp_path / "debyer")
    frames_dir = _build_frames_dir(
        tmp_path,
        xyz=("2\n" "frame\n" "Na 0.0 0.0 0.0\n" "Cl 2.0 0.0 0.0\n"),
    )
    project_dir = tmp_path / "project"
    result = DebyerPDFWorkflow(
        DebyerPDFSettings(
            project_dir=project_dir,
            frames_dir=frames_dir,
            filename_prefix="forgot_solutes",
            mode="PDF",
            from_value=0.5,
            to_value=15.0,
            step_value=0.01,
            box_dimensions=(10.0, 10.0, 10.0),
            atom_count=2,
            store_frame_outputs=False,
            solute_elements=(),
        ),
        debyer_executable=fake_debyer,
    ).run()
    assert result.solute_elements == ()

    window = DebyerPDFMainWindow(initial_project_dir=project_dir)
    qapp.processEvents()
    assert not any(
        window.trace_table.item(row, 3) is not None
        and window.trace_table.item(row, 3).text() == "Group"
        for row in range(window.trace_table.rowCount())
    )

    window.solute_elements_edit.setText("Pb")
    window._apply_solute_groups_from_ui()
    qapp.processEvents()

    assert window._current_calculation is not None
    assert window._current_calculation.solute_elements == ("Pb",)
    assert any(
        window.trace_table.item(row, 3) is not None
        and window.trace_table.item(row, 3).text() == "Group"
        and window.trace_table.item(row, 2) is not None
        and window.trace_table.item(row, 2).text() == "solute-solvent"
        for row in range(window.trace_table.rowCount())
    )
    loaded = load_debyer_calculation(result.calculation_dir)
    assert loaded.solute_elements == ("Pb",)
    _output_r, output_values = parse_debyer_output_file(
        loaded.averaged_output_file
    )
    assert "solute-solute" in output_values
    assert "solute-solvent" in output_values
    assert "solvent-solvent" in output_values
    assert sorted(loaded.partial_values) == ["I-I", "Pb-I", "Pb-Pb"]
    assert "Solute elements: Pb" in window.calculation_info_label.text()
    window.close()


def test_debyer_workflow_cancellation_saves_partial_average(tmp_path):
    fake_debyer = _write_fake_debyer(tmp_path / "debyer")
    frames_dir = _build_frames_dir(tmp_path)
    project_dir = tmp_path / "project"
    stop_requested = {"value": False}
    workflow = DebyerPDFWorkflow(
        DebyerPDFSettings(
            project_dir=project_dir,
            frames_dir=frames_dir,
            filename_prefix="cancelled_pdf",
            mode="PDF",
            from_value=0.5,
            to_value=15.0,
            step_value=0.01,
            box_dimensions=(10.0, 10.0, 10.0),
            atom_count=3,
            store_frame_outputs=False,
            solute_elements=("Pb",),
        ),
        debyer_executable=fake_debyer,
    )

    def request_stop_after_first_frame(processed, _total, _message):
        if processed >= 1:
            stop_requested["value"] = True

    result = workflow.run(
        progress_callback=request_stop_after_first_frame,
        cancel_callback=lambda: stop_requested["value"],
    )

    assert result.processed_frame_count == 1
    assert result.frame_count == 2
    assert result.is_partial_average is True
    assert np.allclose(result.total_values, np.array([1.0, 1.1, 1.2]))

    loaded = load_debyer_calculation(result.calculation_dir)
    assert loaded.processed_frame_count == 1
    assert loaded.frame_count == 2
    assert loaded.is_partial_average is True
    assert np.allclose(loaded.total_values, result.total_values)


def test_debyer_load_backfills_missing_peak_metadata(tmp_path):
    fake_debyer = _write_fake_debyer(tmp_path / "debyer")
    frames_dir = _build_frames_dir(tmp_path)
    project_dir = tmp_path / "project"
    workflow = DebyerPDFWorkflow(
        DebyerPDFSettings(
            project_dir=project_dir,
            frames_dir=frames_dir,
            filename_prefix="legacy_pdf",
            mode="PDF",
            from_value=0.5,
            to_value=15.0,
            step_value=0.01,
            box_dimensions=(11.0, 11.0, 11.0),
            atom_count=3,
            store_frame_outputs=False,
            solute_elements=("Pb",),
        ),
        debyer_executable=fake_debyer,
    )
    result = workflow.run()
    metadata_path = result.calculation_dir / "calculation.json"
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload.pop("partial_peak_markers", None)
    payload.pop("peak_finder_settings", None)
    metadata_path.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    loaded = load_debyer_calculation(result.calculation_dir)
    refreshed_payload = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert loaded.partial_peak_markers["Pb-Pb"]
    assert "partial_peak_markers" in refreshed_payload
    assert "peak_finder_settings" in refreshed_payload


def test_debyer_window_loads_saved_calculation(qapp, tmp_path, monkeypatch):
    fake_debyer = _write_fake_debyer(tmp_path / "debyer")
    frames_dir = _build_frames_dir(tmp_path)
    project_dir = tmp_path / "project"
    workflow = DebyerPDFWorkflow(
        DebyerPDFSettings(
            project_dir=project_dir,
            frames_dir=frames_dir,
            filename_prefix="ui_demo",
            mode="PDF",
            from_value=0.5,
            to_value=15.0,
            step_value=0.01,
            box_dimensions=(12.0, 12.0, 12.0),
            atom_count=3,
            store_frame_outputs=True,
            solute_elements=("Pb",),
        ),
        debyer_executable=fake_debyer,
    )
    workflow.run()

    window = DebyerPDFMainWindow(initial_project_dir=project_dir)
    qapp.processEvents()
    assert window.saved_calculations_combo.count() == 1
    assert window.trace_table.rowCount() >= 4
    assert "Frames averaged:" in window.calculation_info_label.text()
    assert window.project_dir_edit.text() == str(project_dir.resolve())
    assert window.update_plot_during_run_checkbox.isChecked() is True
    assert window.tag_font_family_combo.currentFont().family()
    assert window.axis_font_family_combo.currentFont().family()
    assert "Estimated time remaining" in window.time_estimate_label.text()
    partial_row = None
    average_row = None
    grouped_row = None
    for row in range(window.trace_table.rowCount()):
        kind_item = window.trace_table.item(row, 3)
        label_item = window.trace_table.item(row, 2)
        if kind_item is None or label_item is None:
            continue
        if kind_item.text() == "Average" and label_item.text() == "Average":
            average_row = row
        if (
            kind_item.text() == "Group"
            and label_item.text() == "solute-solvent"
        ):
            grouped_row = row
        if kind_item.text() == "Partial" and label_item.text() == "Pb-I":
            partial_row = row
    assert average_row is not None
    assert grouped_row is not None
    assert partial_row is not None
    group_colors = {
        key: window._trace_colors[key]
        for key in (
            "group:solute-solute",
            "group:solute-solvent",
            "group:solvent-solvent",
        )
    }
    assert group_colors == {
        "group:solute-solute": "#cc79a7",
        "group:solute-solvent": "#e69f00",
        "group:solvent-solvent": "#009e73",
    }
    assert len(set(group_colors.values())) == 3
    average_tag_box = window.trace_table.cellWidget(average_row, 1)
    grouped_tag_box = window.trace_table.cellWidget(grouped_row, 1)
    assert average_tag_box is not None
    assert grouped_tag_box is not None
    assert average_tag_box.isChecked() is True
    assert grouped_tag_box.isChecked() is False
    peaks_item = window.trace_table.item(partial_row, 4)
    assert peaks_item is not None
    assert peaks_item.text() != "—"
    window._set_trace_tag_visible("partial:Pb-I", True)
    qapp.processEvents()
    assert window._tag_artist_records
    pb_i_targets = {
        record["target_trace_key"]
        for record in window._tag_artist_records
        if record["pair_label"] == "Pb-I"
    }
    assert pb_i_targets == {"average"}
    window._set_trace_tag_visible("group:solute-solvent", True)
    qapp.processEvents()
    pb_i_targets = {
        record["target_trace_key"]
        for record in window._tag_artist_records
        if record["pair_label"] == "Pb-I"
    }
    assert pb_i_targets == {"average"}
    window._set_trace_visible("group:solute-solvent", True)
    qapp.processEvents()
    pb_i_targets = {
        record["target_trace_key"]
        for record in window._tag_artist_records
        if record["pair_label"] == "Pb-I"
    }
    assert pb_i_targets == {"average", "group:solute-solvent"}
    group_trace_rows = window._current_calculation.target_peak_markers.get(
        "group:solute-solvent",
        {},
    )
    assert "Pb-I" not in group_trace_rows
    original_peak_summary = peaks_item.text()
    original_marker_value = window._current_calculation.partial_peak_markers[
        "Pb-I"
    ][0].r_value
    new_marker_value = 0.5 if original_marker_value != 0.5 else 1.0
    window._selected_tag = {
        "pair_label": "Pb-I",
        "marker_index": 0,
        "target_trace_key": "average",
    }
    window._drag_state = {
        "mode": "marker",
        "pair_label": "Pb-I",
        "marker_index": 0,
        "target_trace_key": "average",
    }

    class _FakeEvent:
        def __init__(self, xdata, ydata, inaxes, button=1):
            self.xdata = xdata
            self.ydata = ydata
            self.inaxes = inaxes
            self.button = button

    window._on_plot_motion(
        _FakeEvent(
            xdata=new_marker_value,
            ydata=1.5,
            inaxes=window.figure.axes[0],
        )
    )
    window._on_plot_release(
        _FakeEvent(new_marker_value, 1.5, window.figure.axes[0])
    )
    qapp.processEvents()
    updated_peaks_item = window.trace_table.item(partial_row, 4)
    assert updated_peaks_item is not None
    assert updated_peaks_item.text() != original_peak_summary
    assert f"{new_marker_value:.2f}" in updated_peaks_item.text()
    assert window._current_calculation.partial_peak_markers["Pb-I"][
        0
    ].r_value == pytest.approx(new_marker_value)
    group_override = window._current_calculation.target_peak_markers[
        "group:solute-solvent"
    ]["Pb-I"]
    assert group_override[0].r_value == pytest.approx(original_marker_value)
    assert group_override[0].r_value != pytest.approx(new_marker_value)
    window._set_trace_visible("average", False)
    qapp.processEvents()
    pb_i_targets = {
        record["target_trace_key"]
        for record in window._tag_artist_records
        if record["pair_label"] == "Pb-I"
    }
    assert pb_i_targets == {"group:solute-solvent"}
    window._set_trace_visible("average", True)
    qapp.processEvents()
    target_records = [
        record
        for record in window._tag_artist_records
        if record["pair_label"] == "Pb-I"
        and record["target_trace_key"] == "average"
    ]
    assert target_records
    for record in window._tag_artist_records:
        monkeypatch.setattr(
            record["marker_artist"],
            "contains",
            lambda _event: (False, {}),
        )
        monkeypatch.setattr(
            record["annotation"],
            "contains",
            lambda _event: (False, {}),
        )
    monkeypatch.setattr(
        target_records[-1]["annotation"],
        "contains",
        lambda _event: (True, {}),
    )
    window._on_plot_press(_FakeEvent(0.0, 0.0, window.figure.axes[0]))
    assert window._selected_tag == {
        "pair_label": "Pb-I",
        "marker_index": 0,
        "target_trace_key": "average",
    }
    assert window._drag_state is None
    original_marker_count = len(
        window._current_calculation.partial_peak_markers["Pb-I"]
    )
    assert original_marker_count >= 1
    window._set_selected_tag("Pb-I", 0, target_trace_key="average")
    window._delete_selected_tag()
    qapp.processEvents()
    assert (
        len(window._current_calculation.partial_peak_markers["Pb-I"])
        == original_marker_count - 1
    )
    window._reset_partial_peak_markers("partial:Pb-I")
    qapp.processEvents()
    assert (
        len(window._current_calculation.partial_peak_markers["Pb-I"])
        == original_marker_count
    )
    window.close()


def test_debyer_window_loads_experimental_gr_trace(qapp, tmp_path):
    fake_debyer = _write_fake_debyer(tmp_path / "debyer")
    frames_dir = _build_frames_dir(tmp_path)
    project_dir = tmp_path / "project"
    workflow = DebyerPDFWorkflow(
        DebyerPDFSettings(
            project_dir=project_dir,
            frames_dir=frames_dir,
            filename_prefix="experimental_demo",
            mode="PDF",
            from_value=0.5,
            to_value=15.0,
            step_value=0.01,
            box_dimensions=(12.0, 12.0, 12.0),
            atom_count=3,
            store_frame_outputs=False,
            solute_elements=("Pb",),
        ),
        debyer_executable=fake_debyer,
    )
    workflow.run()
    experimental_path = tmp_path / "experimental_gr.txt"
    experimental_path.write_text(
        "r(A) g(r)\n" "0.5 1.5\n" "1.0 1.65\n" "1.5 1.8\n",
        encoding="utf-8",
    )

    window = DebyerPDFMainWindow(initial_project_dir=project_dir)
    qapp.processEvents()
    window._load_experimental_file(experimental_path)
    qapp.processEvents()

    experimental_row = None
    for row in range(window.trace_table.rowCount()):
        kind_item = window.trace_table.item(row, 3)
        if kind_item is not None and kind_item.text() == "Experimental":
            experimental_row = row
            break
    assert experimental_row is not None
    visible_box = window.trace_table.cellWidget(experimental_row, 0)
    assert visible_box is not None
    assert visible_box.isChecked() is True
    experimental_line = next(
        line
        for line in window.figure.axes[0].get_lines()
        if line.get_label().startswith("Experimental g(r)")
    )
    assert experimental_line.get_linestyle() == "--"
    assert "R^2 = 1.0000" in window._experimental_fit_metrics_text()
    assert any(
        "R^2 = 1.0000" in text_artist.get_text()
        for text_artist in window.figure.axes[0].texts
    )

    window._toggle_experimental_trace()
    qapp.processEvents()
    assert window._trace_visibility["experimental"] is False
    assert window.experimental_toggle_button.text() == "Show Experimental"

    window.fit_box_checkbox.setChecked(False)
    qapp.processEvents()
    assert not any(
        "R^2 =" in text_artist.get_text()
        for text_artist in window.figure.axes[0].texts
    )
    window.close()


def test_debyer_window_can_export_active_traces(qapp, tmp_path, monkeypatch):
    fake_debyer = _write_fake_debyer(tmp_path / "debyer")
    frames_dir = _build_frames_dir(tmp_path)
    project_dir = tmp_path / "project"
    workflow = DebyerPDFWorkflow(
        DebyerPDFSettings(
            project_dir=project_dir,
            frames_dir=frames_dir,
            filename_prefix="export_demo",
            mode="PDF",
            from_value=0.5,
            to_value=15.0,
            step_value=0.01,
            box_dimensions=(12.0, 12.0, 12.0),
            atom_count=3,
            store_frame_outputs=False,
            solute_elements=("Pb",),
        ),
        debyer_executable=fake_debyer,
    )
    workflow.run()

    window = DebyerPDFMainWindow(initial_project_dir=project_dir)
    qapp.processEvents()
    window._toggle_partial_traces()
    qapp.processEvents()

    export_path = tmp_path / "active_traces_export.txt"

    def _fake_get_save_file_name(*_args, **_kwargs):
        return str(export_path), "Text files (*.txt)"

    monkeypatch.setattr(
        "saxshell.pdf.debyer.ui.main_window.QFileDialog.getSaveFileName",
        _fake_get_save_file_name,
    )
    window._export_active_traces()

    assert export_path.is_file()
    export_text = export_path.read_text(encoding="utf-8")
    assert "# columns: r Average I-I Pb-I Pb-Pb" in export_text
    exported_data = np.loadtxt(export_path, comments="#")
    assert exported_data.shape == (3, 5)
    window.close()


def test_saxshell_pdfsetup_subcommand_delegates(monkeypatch):
    captured: dict[str, object] = {}

    def _fake_main(argv=None):
        captured["argv"] = argv
        return 17

    monkeypatch.setattr(pdfsetup_module, "main", _fake_main)
    result = saxshell_module.main(["pdfsetup", "--", "/tmp/example_project"])
    assert result == 17
    assert captured["argv"] == ["/tmp/example_project"]
