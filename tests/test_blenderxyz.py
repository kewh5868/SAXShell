from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

import saxshell.toolbox.blender.ui.main_window as blender_ui_module
from saxshell.toolbox.blender.common import (
    ATOM_STYLE_DEFAULTS,
    DEFAULT_ATOM_STYLE,
    OrientationSpec,
    style_atom_color,
    style_neutral_bond_color,
    style_split_bond_color,
)
from saxshell.toolbox.blender.ui.main_window import (
    BlenderXYZRendererMainWindow,
    _preview_atom_color,
    _preview_bond_color,
    _preview_neutral_bond_color,
)
from saxshell.toolbox.blender.workflow import (
    BlenderXYZRenderSettings,
    BlenderXYZRenderWorkflow,
    build_photoshoot_orientations,
    build_render_output_paths,
    load_preview_structure,
    resolve_blender_executable,
)


@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


def _write_xyz(path: Path, comment: str) -> Path:
    path.write_text(
        "5\n"
        f"{comment}\n"
        "C 0.0 0.0 0.0\n"
        "O 1.2 0.0 0.0\n"
        "N -1.0 0.3 0.4\n"
        "S 0.0 1.5 0.1\n"
        "H 0.0 -1.0 0.0\n",
        encoding="utf-8",
    )
    return path


def _write_fake_blender(path: Path) -> Path:
    script = """#!/usr/bin/env python3
import sys
from pathlib import Path


def format_angle(value):
    text = f"{float(value):.1f}"
    return "0.0" if text == "-0.0" else text


def main(argv):
    args = argv[1:]
    forwarded = args[args.index("--") + 1 :]
    input_path = Path(forwarded[forwarded.index("--input") + 1])
    output_dir = Path(forwarded[forwarded.index("--output-dir") + 1])
    orientations = []
    for index, token in enumerate(forwarded):
        if token == "--orientation":
            orientations.append(forwarded[index + 1])

    output_dir.mkdir(parents=True, exist_ok=True)
    for encoded in orientations:
        key, x_value, y_value, z_value = encoded.split(":")
        output_path = output_dir / (
            f"{input_path.stem}_{key}"
            f"_rx{format_angle(x_value)}"
            f"_ry{format_angle(y_value)}"
            f"_rz{format_angle(z_value)}.png"
        )
        output_path.write_text("fake png payload\\n", encoding="utf-8")
        print(f"rendered {output_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
"""
    path.write_text(script, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


class _FakeQSettings:
    _store: dict[str, object] = {}

    def __init__(self, *args, **kwargs) -> None:
        del args, kwargs

    def value(self, key, default=None, type=None):
        value = self._store.get(key, default)
        if type is bool:
            return bool(value)
        return value

    def setValue(self, key, value) -> None:
        if isinstance(value, tuple):
            value = list(value)
        if isinstance(value, list):
            value = [str(item) for item in value]
        self._store[key] = value


@pytest.fixture()
def fake_blender_ui_settings(monkeypatch):
    _FakeQSettings._store = {}
    monkeypatch.setattr(blender_ui_module, "QSettings", _FakeQSettings)
    return _FakeQSettings


def test_build_render_output_paths_include_orientation_metadata(tmp_path):
    xyz_path = _write_xyz(tmp_path / "sample.xyz", "Sample")
    output_dir = tmp_path / "renders"
    orientations = (
        OrientationSpec(
            key="isometric",
            label="Isometric",
            source="preset",
            x_degrees=35.3,
            y_degrees=0.0,
            z_degrees=45.0,
            enabled=True,
        ),
        OrientationSpec(
            key="photoshoot_01",
            label="Photoshoot 1",
            source="photoshoot",
            x_degrees=-18.0,
            y_degrees=26.0,
            z_degrees=-28.0,
            enabled=True,
        ),
    )

    output_paths = build_render_output_paths(
        xyz_path, output_dir, orientations
    )

    assert output_paths[0].name == "sample_isometric_rx35.3_ry0.0_rz45.0.png"
    assert output_paths[1].name == (
        "sample_photoshoot_01_rx-18.0_ry26.0_rz-28.0.png"
    )
    assert "_publication" not in output_paths[0].name


def test_build_photoshoot_orientations_returns_three_views(tmp_path):
    xyz_path = _write_xyz(tmp_path / "molecule.xyz", "Preview")
    structure = load_preview_structure(xyz_path)

    orientations = build_photoshoot_orientations(structure.atoms)

    assert len(orientations) == 3
    assert [orientation.source for orientation in orientations] == [
        "photoshoot",
        "photoshoot",
        "photoshoot",
    ]
    assert [orientation.key for orientation in orientations] == [
        "photoshoot_01",
        "photoshoot_02",
        "photoshoot_03",
    ]


def test_resolve_blender_executable_accepts_app_bundle(tmp_path):
    app_bundle = tmp_path / "Blender.app"
    executable = app_bundle / "Contents" / "MacOS" / "Blender"
    executable.parent.mkdir(parents=True)
    executable.write_text("#!/bin/sh\n", encoding="utf-8")

    resolved = resolve_blender_executable(app_bundle)

    assert resolved == str(executable.resolve())


def test_blender_render_workflow_runs_fake_batch_renderer(tmp_path):
    fake_blender = _write_fake_blender(tmp_path / "blender")
    xyz_path = _write_xyz(tmp_path / "molecule.xyz", "Rendered Title")
    output_dir = tmp_path / "renders"
    orientations = (
        OrientationSpec(
            key="isometric",
            label="Isometric",
            source="preset",
            x_degrees=35.264,
            y_degrees=0.0,
            z_degrees=45.0,
            enabled=True,
        ),
        OrientationSpec(
            key="custom_pose",
            label="Custom Pose",
            source="custom",
            x_degrees=12.0,
            y_degrees=-28.0,
            z_degrees=5.0,
            enabled=True,
        ),
    )
    workflow = BlenderXYZRenderWorkflow(
        BlenderXYZRenderSettings(
            input_path=xyz_path,
            output_dir=output_dir,
            orientations=orientations,
            title="Custom Figure",
            render_title=False,
            blender_executable=fake_blender,
        )
    )

    result = workflow.run()

    assert output_dir.is_dir()
    assert len(result.output_paths) == 2
    assert result.output_paths[0].is_file()
    assert result.output_paths[1].is_file()
    assert "--output-dir" in result.command
    assert result.command.count("--orientation") == 2
    assert "--atom-style" in result.command
    assert (
        result.command[result.command.index("--atom-style") + 1]
        == "paper_gloss"
    )
    assert "--render-quality" in result.command
    assert (
        "rendered molecule_isometric_rx35.3_ry0.0_rz45.0.png" in result.stdout
    )


def test_blender_window_keeps_destination_folder_and_loads_orientations(
    qapp,
    tmp_path,
    fake_blender_ui_settings,
):
    del qapp
    del fake_blender_ui_settings
    first_xyz = _write_xyz(tmp_path / "first.xyz", "First Comment")
    second_xyz = _write_xyz(tmp_path / "second.xyz", "Second Comment")
    manual_output_dir = tmp_path / "manual_renders"

    window = BlenderXYZRendererMainWindow(initial_input_path=first_xyz)
    window.include_presets_box.setChecked(True)
    window.include_photoshoot_box.setChecked(True)
    window._rebuild_orientations(select_index=0)
    labels = [
        window.orientation_table.item(row, 1).text()
        for row in range(window.orientation_table.rowCount())
    ]
    assert "Isometric" in labels
    assert "Photoshoot 1" in labels

    window.output_dir_edit.setText(str(manual_output_dir))
    window.set_input_path(second_xyz)

    assert window.title_edit.text() == "Second Comment"
    assert window.output_dir_edit.text() == str(manual_output_dir)

    window.include_presets_box.setChecked(False)
    updated_labels = [
        window.orientation_table.item(row, 1).text()
        for row in range(window.orientation_table.rowCount())
    ]
    assert "Isometric" not in updated_labels
    assert "Photoshoot 1" in updated_labels
    window.close()


def test_blender_window_persists_recent_xyz_history(
    qapp,
    tmp_path,
    fake_blender_ui_settings,
):
    del qapp
    del fake_blender_ui_settings
    first_xyz = _write_xyz(tmp_path / "first.xyz", "First Comment")
    second_xyz = _write_xyz(tmp_path / "second.xyz", "Second Comment")

    first_window = BlenderXYZRendererMainWindow()
    first_window.set_input_path(first_xyz)
    first_window.set_input_path(second_xyz)

    first_recent_paths = [
        first_window.recent_input_combo.itemData(index)
        for index in range(first_window.recent_input_combo.count())
    ]
    assert first_recent_paths == [
        str(second_xyz.resolve()),
        str(first_xyz.resolve()),
    ]
    first_window.close()

    second_window = BlenderXYZRendererMainWindow()
    second_recent_paths = [
        second_window.recent_input_combo.itemData(index)
        for index in range(second_window.recent_input_combo.count())
    ]
    assert second_recent_paths == [
        str(second_xyz.resolve()),
        str(first_xyz.resolve()),
    ]

    second_window.recent_input_combo.setCurrentIndex(1)
    second_window._open_selected_recent_input()

    assert second_window.input_edit.text() == str(first_xyz.resolve())
    second_window.close()


def test_blender_window_inline_preview_snaps_to_selected_orientation(
    qapp,
    tmp_path,
    fake_blender_ui_settings,
):
    del qapp
    del fake_blender_ui_settings
    xyz_path = _write_xyz(tmp_path / "preview.xyz", "Preview Comment")

    window = BlenderXYZRendererMainWindow(initial_input_path=xyz_path)

    target_row = 1 if window.orientation_table.rowCount() > 1 else 0
    target_label = window.orientation_table.item(target_row, 1).text()
    target_source = window.orientation_table.item(target_row, 2).text()
    window.orientation_table.selectRow(target_row)

    assert target_label in window.orientation_preview.orientation_label.text()
    assert (
        window.orientation_preview.source_label.text() == target_source.title()
    )

    window.orientation_preview.x_spin.setValue(
        window.orientation_preview.x_spin.value() + 7.0
    )

    assert window.orientation_table.item(target_row, 2).text() == "custom"
    assert float(
        window.orientation_table.item(target_row, 3).text()
    ) == pytest.approx(window.orientation_preview.x_spin.value())
    window.close()


def test_blender_window_collects_style_and_quality_defaults(
    qapp,
    tmp_path,
    fake_blender_ui_settings,
):
    del qapp
    del fake_blender_ui_settings
    xyz_path = _write_xyz(tmp_path / "styled.xyz", "Styled Comment")
    output_dir = tmp_path / "styled-renders"

    window = BlenderXYZRendererMainWindow(initial_input_path=xyz_path)
    window.output_dir_edit.setText(str(output_dir))
    window.atom_style_combo.setCurrentIndex(
        window.atom_style_combo.findData("monochrome")
    )
    window.render_quality_combo.setCurrentIndex(
        window.render_quality_combo.findData("draft")
    )

    settings = window._collect_settings()

    assert settings.atom_style == "monochrome"
    assert settings.render_quality == "draft"
    assert settings.samples == 128
    assert settings.bond_color_mode == "neutral"
    assert settings.camera_margin == pytest.approx(1.05)
    window.close()


def test_blender_window_defaults_to_paper_gloss_style(
    qapp,
    tmp_path,
    fake_blender_ui_settings,
):
    del qapp
    del fake_blender_ui_settings
    xyz_path = _write_xyz(tmp_path / "default-style.xyz", "Default Style")

    window = BlenderXYZRendererMainWindow(initial_input_path=xyz_path)
    settings = window._collect_settings()

    assert settings.atom_style == DEFAULT_ATOM_STYLE
    assert settings.atom_scale == pytest.approx(
        float(ATOM_STYLE_DEFAULTS[DEFAULT_ATOM_STYLE]["atom_scale"])
    )
    assert settings.bond_radius == pytest.approx(
        float(ATOM_STYLE_DEFAULTS[DEFAULT_ATOM_STYLE]["bond_radius"])
    )
    assert settings.bond_color_mode == str(
        ATOM_STYLE_DEFAULTS[DEFAULT_ATOM_STYLE]["bond_color_mode"]
    )
    assert "Glossy" in window.style_hint_label.text()
    window.close()


def test_preview_style_colors_stay_in_sync_with_shared_render_helpers():
    assert _preview_atom_color("C", atom_style="toon_matte") == pytest.approx(
        style_atom_color("C", atom_style="toon_matte")[:3]
    )
    assert _preview_bond_color(
        "Se", atom_style="crystal_cartoon"
    ) == pytest.approx(
        style_split_bond_color("Se", atom_style="crystal_cartoon")[:3]
    )
    assert _preview_neutral_bond_color("soft_studio") == pytest.approx(
        style_neutral_bond_color("soft_studio")[:3]
    )
