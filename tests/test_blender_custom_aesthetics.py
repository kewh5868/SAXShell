from __future__ import annotations

import importlib
import json
import os
import sys
import types
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

import saxshell.toolbox.blender.ui.main_window as blender_ui_module
from saxshell.toolbox.blender.common import (
    COVALENT_RADII,
    AtomAppearanceOverride,
    BondThresholdSpec,
    CustomAestheticSpec,
    atom_style_base,
    atom_style_label,
    encode_custom_aesthetic_arg,
    normalize_atom_style,
    parse_custom_aesthetic_arg,
    set_custom_aesthetics,
    style_atom_color,
    style_display_radius,
)
from saxshell.toolbox.blender.ui.main_window import (
    AtomAestheticEditorDialog,
    BlenderXYZRendererMainWindow,
)
from saxshell.toolbox.blender.workflow import (
    BlenderXYZRenderSettings,
    BlenderXYZRenderWorkflow,
    build_bond_thresholds_for_structure,
)

pytestmark = pytest.mark.filterwarnings(
    "ignore:This figure includes Axes that are not compatible with tight_layout.*:UserWarning"
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


def _import_render_script_module(monkeypatch):
    class _FakeVector(tuple):
        def __new__(cls, values):
            return super().__new__(cls, values)

    fake_bpy = types.ModuleType("bpy")
    fake_bpy.context = types.SimpleNamespace(
        view_layer=types.SimpleNamespace(update=lambda: None)
    )
    fake_mathutils = types.ModuleType("mathutils")
    fake_mathutils.Vector = _FakeVector

    monkeypatch.setitem(sys.modules, "bpy", fake_bpy)
    monkeypatch.setitem(sys.modules, "mathutils", fake_mathutils)
    sys.modules.pop("saxshell.toolbox.blender.render_xyz_publication", None)
    return importlib.import_module(
        "saxshell.toolbox.blender.render_xyz_publication"
    )


class _FakeQSettings:
    _store: dict[str, object] = {}

    def __init__(self, *args, **kwargs) -> None:
        del args, kwargs

    def value(self, key, default=None, type=None):
        value = self._store.get(key, default)
        if type is bool:
            return bool(value)
        if type is float:
            return float(value)
        return value

    def setValue(self, key, value) -> None:
        self._store[key] = value

    def remove(self, key) -> None:
        self._store.pop(key, None)


@pytest.fixture()
def fake_blender_ui_settings(monkeypatch):
    _FakeQSettings._store = {}
    set_custom_aesthetics(())
    monkeypatch.setattr(blender_ui_module, "QSettings", _FakeQSettings)
    yield _FakeQSettings
    set_custom_aesthetics(())


def test_custom_aesthetic_roundtrip_overrides_colors_and_sizes():
    spec = CustomAestheticSpec(
        key="My Highlighted Sulfur",
        name="My Highlighted Sulfur",
        base_style="toon_matte",
        overrides=(
            AtomAppearanceOverride(
                element="S",
                color=(0.12, 0.34, 0.56, 1.0),
                size_scale=1.25,
            ),
        ),
    )

    loaded = parse_custom_aesthetic_arg(encode_custom_aesthetic_arg(spec))
    set_custom_aesthetics((loaded,))

    assert normalize_atom_style(loaded.key) == loaded.key
    assert atom_style_base(loaded.key) == "toon_matte"
    assert atom_style_label(loaded.key) == "My Highlighted Sulfur"
    assert style_atom_color("S", atom_style=loaded.key) == pytest.approx(
        (0.12, 0.34, 0.56, 1.0)
    )
    assert style_display_radius("S", atom_style=loaded.key) == pytest.approx(
        max(COVALENT_RADII["S"] * 1.25, 0.18)
    )


def test_atom_aesthetic_editor_dialog_returns_named_custom_spec(qapp):
    del qapp
    set_custom_aesthetics(())
    dialog = AtomAestheticEditorDialog(
        active_style="toon_matte",
        elements=("C", "S"),
    )

    try:
        dialog.name_edit.setText("Poster Sulfur")
        dialog.seed_preset_combo.setCurrentIndex(
            dialog.seed_preset_combo.findData("poster_pop")
        )
        dialog._apply_selected_preset()
        dialog._set_row_color(1, (0.22, 0.44, 0.66, 1.0))
        sulfur_scale = dialog._row_scale_spin(1)
        assert sulfur_scale is not None
        sulfur_scale.setValue(1.11)

        spec = dialog.custom_aesthetic(existing_specs=())
        overrides = spec.override_lookup()

        assert spec.name == "Poster Sulfur"
        assert spec.base_style == "poster_pop"
        assert spec.key == "custom_poster_sulfur"
        assert overrides["S"].color == pytest.approx((0.22, 0.44, 0.66, 1.0))
        assert overrides["S"].size_scale == pytest.approx(1.11)
    finally:
        dialog.close()


def test_workflow_command_includes_custom_aesthetic_payload(tmp_path):
    xyz_path = _write_xyz(tmp_path / "custom.xyz", "Custom Style")
    spec = CustomAestheticSpec(
        key="custom_lab_palette",
        name="Lab Palette",
        base_style="soft_studio",
        overrides=(
            AtomAppearanceOverride(
                element="O",
                color=(0.18, 0.58, 0.88, 1.0),
                size_scale=0.93,
            ),
        ),
    )
    workflow = BlenderXYZRenderWorkflow(
        BlenderXYZRenderSettings(
            input_path=xyz_path,
            output_dir=tmp_path / "renders",
            blender_executable="blender",
            atom_style=spec.key,
            custom_aesthetics=(spec,),
        )
    )

    command = workflow.build_command()

    assert command[command.index("--atom-style") + 1] == spec.key
    encoded = command[command.index("--custom-aesthetic") + 1]
    payload = json.loads(encoded)
    assert payload["key"] == spec.key
    assert payload["base_style"] == "soft_studio"


def test_workflow_command_includes_sample_floor_override(tmp_path):
    xyz_path = _write_xyz(tmp_path / "sample-floor.xyz", "Sample Floor")
    workflow = BlenderXYZRenderWorkflow(
        BlenderXYZRenderSettings(
            input_path=xyz_path,
            output_dir=tmp_path / "renders",
            blender_executable="blender",
            atom_style="toon_matte",
            sample_floor_override=32,
        )
    )

    command = workflow.build_command()

    assert command[command.index("--sample-floor") + 1] == "32"


def test_blender_window_loads_saved_custom_aesthetics_across_sessions(
    qapp,
    tmp_path,
    fake_blender_ui_settings,
):
    del qapp
    del fake_blender_ui_settings
    xyz_path = _write_xyz(tmp_path / "style-session.xyz", "Style Session")
    spec = CustomAestheticSpec(
        key="custom_session_palette",
        name="Session Palette",
        base_style="cpk",
        overrides=(
            AtomAppearanceOverride(
                element="N",
                color=(0.31, 0.62, 0.93, 1.0),
                size_scale=0.88,
            ),
        ),
    )

    first = BlenderXYZRendererMainWindow(initial_input_path=xyz_path)
    first._custom_aesthetics_by_key[spec.key] = spec
    first._persist_custom_aesthetics()
    first.close()

    second = BlenderXYZRendererMainWindow()
    try:
        assert second.atom_style_combo.findData(spec.key) >= 0
        assert (
            second._custom_aesthetics_by_key[spec.key].name
            == "Session Palette"
        )
    finally:
        second.close()


def test_blender_window_persists_bond_thresholds_across_sessions(
    qapp,
    tmp_path,
    fake_blender_ui_settings,
):
    del qapp
    del fake_blender_ui_settings
    xyz_path = _write_xyz(tmp_path / "bond-thresholds.xyz", "Thresholds")

    first = BlenderXYZRendererMainWindow(initial_input_path=xyz_path)
    defaults = build_bond_thresholds_for_structure(
        first._preview_structure.atoms
    )
    modified: list[BondThresholdSpec] = []
    target_key = defaults[0].pair_key
    for spec in defaults:
        if spec.pair_key == target_key:
            modified.append(
                BondThresholdSpec(
                    element_a=spec.element_a,
                    element_b=spec.element_b,
                    min_length=spec.min_length + 0.05,
                    max_length=spec.max_length + 0.40,
                )
            )
        else:
            modified.append(spec)
    modified_tuple = tuple(modified)
    first._persist_bond_thresholds_for_path(
        xyz_path,
        modified_tuple,
        default_thresholds=defaults,
    )
    first.close()

    second = BlenderXYZRendererMainWindow(initial_input_path=xyz_path)
    try:
        lookup = {spec.pair_key: spec for spec in second._bond_thresholds}
        assert lookup[target_key].min_length == pytest.approx(
            defaults[0].min_length + 0.05
        )
        assert lookup[target_key].max_length == pytest.approx(
            defaults[0].max_length + 0.40
        )
    finally:
        second.close()


def test_render_script_preserves_none_atom_scale_for_custom_aesthetic(
    monkeypatch,
    tmp_path,
):
    render_module = _import_render_script_module(monkeypatch)
    xyz_path = _write_xyz(tmp_path / "render-custom.xyz", "Render Custom")
    output_path = tmp_path / "render-custom.png"
    spec = CustomAestheticSpec(
        key="custom_render_semantics",
        name="Render Semantics",
        base_style="toon_matte",
        overrides=(
            AtomAppearanceOverride(
                element="S",
                color=(0.85, 0.72, 0.15, 1.0),
                size_scale=1.27,
            ),
        ),
    )
    captured: dict[str, object] = {}
    bounds = (
        types.SimpleNamespace(z=0.0),
        types.SimpleNamespace(z=1.0),
    )

    monkeypatch.setattr(
        render_module,
        "parse_structure",
        lambda path, hide_hydrogen=False: ("Render Custom", []),
    )
    monkeypatch.setattr(render_module, "clear_scene", lambda: None)
    monkeypatch.setattr(render_module, "setup_render", lambda **kwargs: None)
    monkeypatch.setattr(render_module, "setup_world", lambda **kwargs: None)
    monkeypatch.setattr(
        render_module, "setup_lighting", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        render_module, "create_camera_rig", lambda: (object(), object())
    )
    monkeypatch.setattr(render_module, "world_bounds", lambda objects: bounds)
    monkeypatch.setattr(
        render_module, "_apply_orientation", lambda root, orientation: None
    )
    monkeypatch.setattr(
        render_module,
        "frame_camera",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(render_module, "render_scene", lambda path: None)
    monkeypatch.setattr(render_module, "save_blend_scene", lambda path: None)

    def _fake_build_structure(
        atoms,
        *,
        atom_style,
        atom_scale,
        bond_radius,
        bond_threshold,
        pair_thresholds,
        bond_color_mode,
    ):
        del (
            atoms,
            bond_radius,
            bond_threshold,
            pair_thresholds,
            bond_color_mode,
        )
        captured["atom_style"] = atom_style
        captured["atom_scale"] = atom_scale
        return object(), [object()]

    monkeypatch.setattr(
        render_module, "build_structure", _fake_build_structure
    )

    exit_code = render_module.main(
        [
            "--",
            "--input",
            str(xyz_path),
            "--output",
            str(output_path),
            "--hide-title",
            "--atom-style",
            spec.key,
            "--custom-aesthetic",
            encode_custom_aesthetic_arg(spec),
        ]
    )

    assert exit_code == 0
    assert captured["atom_style"] == spec.key
    assert captured["atom_scale"] is None
