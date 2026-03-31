from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

import saxshell.pdfsetup as pdfsetup_module
from saxshell import saxshell as saxshell_module
from saxshell.pdf.debyer.ui.main_window import DebyerPDFMainWindow
from saxshell.pdf.debyer.workflow import (
    DebyerPDFSettings,
    DebyerPDFWorkflow,
    calculate_number_density,
    convert_distribution_values,
    list_saved_debyer_calculations,
    load_debyer_calculation,
)


@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


def _write_fake_debyer(path: Path) -> Path:
    script = """#!/usr/bin/env python3
import re
import sys
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
    path.write_text(script, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def _build_frames_dir(tmp_path: Path) -> Path:
    frames_dir = tmp_path / "splitxyz_f0fs"
    frames_dir.mkdir()
    xyz = (
        "3\n" "frame\n" "Pb 0.0 0.0 0.0\n" "I 2.0 0.0 0.0\n" "I 0.0 2.0 0.0\n"
    )
    for index in range(2):
        (frames_dir / f"frame_{index:04d}.xyz").write_text(
            xyz,
            encoding="utf-8",
        )
    return frames_dir


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

    summaries = list_saved_debyer_calculations(project_dir)
    assert len(summaries) == 1
    loaded = load_debyer_calculation(summaries[0].calculation_dir)
    assert loaded.filename_prefix == "demo_pdf"
    assert np.allclose(loaded.total_values, result.total_values)
    assert loaded.solute_elements == ("Pb",)
    assert loaded.processed_frame_count == loaded.frame_count
    assert "Pb-I" in loaded.partial_peak_markers
    assert loaded.partial_peak_markers["Pb-I"]
    assert loaded.peak_finder_settings.max_peak_count >= 0


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
