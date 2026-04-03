from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

import numpy as np
from matplotlib.backend_bases import MouseButton
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from PySide6.QtCore import QObject, QSettings, Qt, QThread, Signal, Slot
from PySide6.QtGui import QColor, QFont, QPixmap, QResizeEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFontComboBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from saxshell.saxs.ui.branding import (
    configure_saxshell_application,
    load_saxshell_icon,
    prepare_saxshell_application_identity,
)
from saxshell.toolbox.blender.common import (
    ATOM_STYLE_LABELS,
    COVALENT_RADII,
    AtomAppearanceOverride,
    CustomAestheticSpec,
    DEFAULT_ATOM_STYLE,
    DEFAULT_LEGEND_FONT,
    DEFAULT_LIGHTING_LEVEL,
    DEFAULT_RENDER_QUALITY,
    LIGHTING_LEVEL_LABELS,
    RENDER_QUALITY_LABELS,
    BondThresholdSpec,
    OrientationSpec,
    atom_style_base,
    atom_style_defaults,
    atom_style_description,
    atom_style_is_custom,
    atom_style_label,
    available_atom_style_labels,
    deserialize_custom_aesthetic,
    encode_bond_threshold_arg,
    get_custom_aesthetic,
    normalize_atom_style,
    normalize_builtin_atom_style,
    normalize_lighting_level,
    normalize_render_quality,
    parse_bond_threshold_arg,
    sanitize_orientation_key,
    sanitize_custom_aesthetic_key,
    serialize_custom_aesthetic,
    set_custom_aesthetics,
    style_atom_color,
    style_atom_size_scale,
    style_display_radius,
    style_neutral_bond_color,
    style_split_bond_color,
)
from saxshell.toolbox.blender.workflow import (
    BlenderPreviewStructure,
    BlenderXYZRenderResult,
    BlenderXYZRenderSettings,
    BlenderXYZRenderWorkflow,
    apply_orientation_to_positions,
    build_bond_thresholds_for_structure,
    build_default_orientation_catalog,
    compose_euler_degrees,
    infer_title,
    load_preview_structure,
    read_structure_comment,
    resolve_blender_executable,
    resolve_desktop_dir,
    suggest_output_dir,
)
from saxshell.toolbox.blender.ui.reference_atoms import (
    REFERENCE_ATOM_BACKGROUND,
    REFERENCE_ATOM_LABELS,
    reference_atom_path,
)

_OPEN_WINDOWS: list["BlenderXYZRendererMainWindow"] = []

_OUTPUT_DIR_KEY = "toolbox/blender/output_dir"
_BLENDER_EXECUTABLE_KEY = "toolbox/blender/blender_executable"
_INCLUDE_PRESETS_KEY = "toolbox/blender/include_presets"
_INCLUDE_PHOTOSHOOT_KEY = "toolbox/blender/include_photoshoot"
_RENDER_TITLE_KEY = "toolbox/blender/render_title"
_RECENT_INPUT_FILES_KEY = "toolbox/blender/recent_input_files"
_ATOM_STYLE_KEY = "toolbox/blender/atom_style"
_CUSTOM_AESTHETICS_KEY = "toolbox/blender/custom_aesthetics"
_BOND_THRESHOLD_OVERRIDES_KEY = "toolbox/blender/bond_threshold_overrides"
_RENDER_QUALITY_KEY = "toolbox/blender/render_quality"
_PREVIEW_BACKGROUND_KEY = "toolbox/blender/preview_background"
_REFERENCE_BACKGROUND_KEY = "toolbox/blender/reference_background"
_NUDGE_INCREMENT_KEY = "toolbox/blender/nudge_increment"
_SAVE_BLEND_FILES_KEY = "toolbox/blender/save_blend_files"
_LEGEND_FONT_KEY = "toolbox/blender/legend_font"
_MAX_RECENT_INPUT_FILES = 10
_ORIENTATION_ENABLED_COLUMN = 0
_ORIENTATION_LEGEND_COLUMN = 1
_ORIENTATION_NAME_COLUMN = 2
_ORIENTATION_SOURCE_COLUMN = 3
_ORIENTATION_LIGHTING_COLUMN = 4
_ORIENTATION_STYLE_COLUMN = 5
_ORIENTATION_QUALITY_COLUMN = 6
_ORIENTATION_X_COLUMN = 7
_ORIENTATION_Y_COLUMN = 8
_ORIENTATION_Z_COLUMN = 9
_LIGHTING_LEVEL_TABLE_LABELS = {
    value: str(value) for value in LIGHTING_LEVEL_LABELS
}
_RENDER_MATCHED_VIEW_AZIM = -90.0
_RENDER_MATCHED_VIEW_ELEV = 0.0


def _preview_palette(atom_style: str) -> dict[str, object]:
    style = atom_style_base(atom_style)
    if style == "paper_gloss":
        return {
            "background": "#fdfbf8",
            "bond": "#91889f",
            "edge": (0.16, 0.14, 0.20),
            "shadow": (0.26, 0.17, 0.30),
            "label": "Paper Gloss",
        }
    if style == "soft_studio":
        return {
            "background": "#f8f6f3",
            "bond": "#887f74",
            "edge": (0.23, 0.21, 0.22),
            "shadow": (0.20, 0.18, 0.18),
            "label": "Soft Studio",
        }
    if style == "flat_diagram":
        return {
            "background": "#ffffff",
            "bond": "#636a74",
            "edge": (0.14, 0.17, 0.20),
            "shadow": (0.0, 0.0, 0.0),
            "label": "Flat Diagram",
        }
    if style == "toon_matte":
        return {
            "background": "#fff8ef",
            "bond": "#5f6570",
            "edge": (0.14, 0.15, 0.18),
            "shadow": (0.12, 0.12, 0.15),
            "label": "Toon Matte",
        }
    if style == "poster_pop":
        return {
            "background": "#fff6ea",
            "bond": "#555d68",
            "edge": (0.10, 0.12, 0.16),
            "shadow": (0.14, 0.12, 0.14),
            "label": "Poster Pop",
        }
    if style == "pastel_cartoon":
        return {
            "background": "#fffaf6",
            "bond": "#897f76",
            "edge": (0.22, 0.20, 0.21),
            "shadow": (0.15, 0.14, 0.15),
            "label": "Pastel Cartoon",
        }
    if style == "crystal_flat":
        return {
            "background": "#ffffff",
            "bond": "#5f6671",
            "edge": (0.17, 0.18, 0.20),
            "shadow": (0.12, 0.12, 0.13),
            "label": "Crystal Flat",
        }
    if style == "crystal_cartoon":
        return {
            "background": "#fffdf7",
            "bond": "#596273",
            "edge": (0.13, 0.14, 0.17),
            "shadow": (0.17, 0.16, 0.18),
            "label": "Crystal Cartoon",
        }
    if style == "crystal_shadow_gloss":
        return {
            "background": "#fff9f0",
            "bond": "#6b6370",
            "edge": (0.14, 0.13, 0.16),
            "shadow": (0.25, 0.18, 0.20),
            "label": "Crystal Shadow Gloss",
        }
    if style == "monochrome":
        return {
            "background": "#f7f7f7",
            "bond": "#7a7d84",
            "edge": (0.16, 0.17, 0.19),
            "shadow": (0.0, 0.0, 0.0),
            "label": "Monochrome",
        }
    if style == "cpk":
        return {
            "background": "#f4f7fb",
            "bond": "#67707c",
            "edge": (0.18, 0.22, 0.30),
            "shadow": (0.17, 0.21, 0.28),
            "label": "CPK",
        }
    return {
        "background": "#ffffff",
        "bond": "#6a7380",
        "edge": (0.15, 0.17, 0.21),
        "shadow": (0.16, 0.19, 0.24),
        "label": "VESTA-like",
    }


def _color_to_hex(
    color: tuple[float, float, float, float] | tuple[float, float, float],
) -> str:
    channels = tuple(float(channel) for channel in tuple(color)[:3])
    return "#{:02x}{:02x}{:02x}".format(
        int(max(0.0, min(1.0, channels[0])) * 255.0),
        int(max(0.0, min(1.0, channels[1])) * 255.0),
        int(max(0.0, min(1.0, channels[2])) * 255.0),
    )


def _qcolor_to_rgba(color: QColor) -> tuple[float, float, float, float]:
    return (
        float(color.redF()),
        float(color.greenF()),
        float(color.blueF()),
        float(color.alphaF()),
    )


def _rgba_to_qcolor(
    color: tuple[float, float, float, float],
) -> QColor:
    return QColor.fromRgbF(
        float(color[0]),
        float(color[1]),
        float(color[2]),
        float(color[3]),
    )


def _preview_atom_color(
    element: str,
    *,
    atom_style: str,
) -> tuple[float, float, float]:
    return style_atom_color(element, atom_style=atom_style)[:3]


def _preview_bond_color(
    element: str,
    *,
    atom_style: str,
) -> tuple[float, float, float]:
    return style_split_bond_color(element, atom_style=atom_style)[:3]


def _preview_neutral_bond_color(atom_style: str) -> tuple[float, float, float]:
    return style_neutral_bond_color(atom_style)[:3]


def _preview_structure_elements(
    structure: BlenderPreviewStructure,
) -> list[str]:
    seen: set[str] = set()
    elements: list[str] = []
    for atom in structure.atoms:
        if atom.element in seen:
            continue
        seen.add(atom.element)
        elements.append(atom.element)
    return elements


class OrientationPreviewWidget(QWidget):
    """Inline structure preview and orientation editor."""

    orientationChanged = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._structure: BlenderPreviewStructure | None = None
        self._orientation: OrientationSpec | None = None
        self._positions = np.zeros((0, 3), dtype=float)
        self._atom_style = DEFAULT_ATOM_STYLE
        self._reference_background_color = REFERENCE_ATOM_BACKGROUND
        self._reference_pixmap_cache: dict[tuple[str, int, str], QPixmap] = {}
        self._updating_controls = False
        self._axis = None
        self._axis_reference = None
        self._view_azim = -58.0
        self._view_elev = 22.0
        self._view_center = np.zeros(3, dtype=float)
        self._view_radius = 1.0
        self._interaction_mode = "rotate"
        self._background_override: str | None = None
        self._drag_mode: str | None = None
        self._drag_start_xy = (0.0, 0.0)
        self._drag_last_xy = (0.0, 0.0)
        self._drag_start_azim = self._view_azim
        self._drag_start_elev = self._view_elev
        self._drag_start_center = self._view_center.copy()
        self._drag_start_radius = self._view_radius
        self._build_ui()
        self.clear_preview()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        toolbar_row = QHBoxLayout()
        toolbar_row.setContentsMargins(0, 0, 0, 0)
        toolbar_row.setSpacing(6)
        toolbar_row.addWidget(QLabel("View Tool"))
        self.rotate_tool_button = QPushButton("Rotate")
        self.rotate_tool_button.setCheckable(True)
        self.rotate_tool_button.clicked.connect(
            lambda _checked=False: self._set_interaction_mode("rotate")
        )
        toolbar_row.addWidget(self.rotate_tool_button)
        self.pan_tool_button = QPushButton("Pan")
        self.pan_tool_button.setCheckable(True)
        self.pan_tool_button.clicked.connect(
            lambda _checked=False: self._set_interaction_mode("pan")
        )
        toolbar_row.addWidget(self.pan_tool_button)
        self.zoom_tool_button = QPushButton("Zoom")
        self.zoom_tool_button.setCheckable(True)
        self.zoom_tool_button.clicked.connect(
            lambda _checked=False: self._set_interaction_mode("zoom")
        )
        toolbar_row.addWidget(self.zoom_tool_button)
        self.reset_view_button = QPushButton("Reset View")
        self.reset_view_button.clicked.connect(self.reset_view)
        self.reset_view_button.setEnabled(False)
        toolbar_row.addWidget(self.reset_view_button)
        toolbar_row.addSpacing(14)
        toolbar_row.addWidget(QLabel("Nudge"))
        self.nudge_step_spin = QDoubleSpinBox()
        self.nudge_step_spin.setRange(0.1, 180.0)
        self.nudge_step_spin.setDecimals(1)
        self.nudge_step_spin.setSingleStep(0.5)
        self.nudge_step_spin.setSuffix(" deg")
        self.nudge_step_spin.setValue(5.0)
        self.nudge_step_spin.setFixedWidth(92)
        toolbar_row.addWidget(self.nudge_step_spin)
        self.x_minus_button = QPushButton("X-")
        self.x_minus_button.clicked.connect(
            lambda _checked=False: self._nudge_structure_rotation("x", -1.0)
        )
        toolbar_row.addWidget(self.x_minus_button)
        self.x_plus_button = QPushButton("X+")
        self.x_plus_button.clicked.connect(
            lambda _checked=False: self._nudge_structure_rotation("x", 1.0)
        )
        toolbar_row.addWidget(self.x_plus_button)
        self.y_minus_button = QPushButton("Y-")
        self.y_minus_button.clicked.connect(
            lambda _checked=False: self._nudge_structure_rotation("y", -1.0)
        )
        toolbar_row.addWidget(self.y_minus_button)
        self.y_plus_button = QPushButton("Y+")
        self.y_plus_button.clicked.connect(
            lambda _checked=False: self._nudge_structure_rotation("y", 1.0)
        )
        toolbar_row.addWidget(self.y_plus_button)
        self.z_minus_button = QPushButton("Z-")
        self.z_minus_button.clicked.connect(
            lambda _checked=False: self._nudge_structure_rotation("z", -1.0)
        )
        toolbar_row.addWidget(self.z_minus_button)
        self.z_plus_button = QPushButton("Z+")
        self.z_plus_button.clicked.connect(
            lambda _checked=False: self._nudge_structure_rotation("z", 1.0)
        )
        toolbar_row.addWidget(self.z_plus_button)
        toolbar_row.addStretch(1)
        self.rotation_readout_label = QLabel()
        self.rotation_readout_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        self.rotation_readout_label.setMinimumWidth(250)
        self.rotation_readout_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        toolbar_row.addWidget(self.rotation_readout_label)
        layout.addLayout(toolbar_row)

        angle_row = QHBoxLayout()
        angle_row.setContentsMargins(0, 0, 0, 0)
        angle_row.setSpacing(6)
        angle_row.addWidget(QLabel("Set Rotation"))
        self.x_toolbar_spin = self._build_toolbar_angle_spinbox(
            lambda value: self._handle_toolbar_angle_changed("x", value)
        )
        angle_row.addWidget(QLabel("X"))
        angle_row.addWidget(self.x_toolbar_spin)
        self.y_toolbar_spin = self._build_toolbar_angle_spinbox(
            lambda value: self._handle_toolbar_angle_changed("y", value)
        )
        angle_row.addWidget(QLabel("Y"))
        angle_row.addWidget(self.y_toolbar_spin)
        self.z_toolbar_spin = self._build_toolbar_angle_spinbox(
            lambda value: self._handle_toolbar_angle_changed("z", value)
        )
        angle_row.addWidget(QLabel("Z"))
        angle_row.addWidget(self.z_toolbar_spin)
        angle_row.addStretch(1)
        layout.addLayout(angle_row)

        self.figure = Figure(figsize=(8.2, 6.8))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(420)
        self.canvas.mpl_connect(
            "button_press_event", self._handle_canvas_press
        )
        self.canvas.mpl_connect(
            "button_release_event", self._handle_canvas_release
        )
        self.canvas.mpl_connect(
            "motion_notify_event", self._handle_canvas_motion
        )
        self.canvas.mpl_connect("scroll_event", self._handle_canvas_scroll)

        preview_row = QHBoxLayout()
        preview_row.setContentsMargins(0, 0, 0, 0)
        preview_row.setSpacing(10)
        preview_row.addWidget(self.canvas, stretch=1)
        preview_row.addWidget(
            self._build_reference_panel(),
            stretch=0,
            alignment=Qt.AlignmentFlag.AlignTop,
        )
        layout.addLayout(preview_row, stretch=1)

        self.controls_group = QGroupBox("Selected Orientation")
        controls_layout = QFormLayout(self.controls_group)

        self.source_label = QLabel()
        controls_layout.addRow("Source", self.source_label)

        self.preview_style_label = QLabel()
        controls_layout.addRow("Aesthetic", self.preview_style_label)

        self.name_edit = QLineEdit()
        self.name_edit.textEdited.connect(self._handle_name_changed)
        controls_layout.addRow("Name", self.name_edit)

        self.enabled_box = QCheckBox("Enabled for rendering")
        self.enabled_box.toggled.connect(self._handle_enabled_changed)
        controls_layout.addRow("", self.enabled_box)

        self.x_spin = self._build_angle_spinbox(self._handle_angles_changed)
        self.y_spin = self._build_angle_spinbox(self._handle_angles_changed)
        self.z_spin = self._build_angle_spinbox(self._handle_angles_changed)
        self.nudge_step_spin.valueChanged.connect(
            self._handle_nudge_step_changed
        )
        controls_layout.addRow("Rotate X", self.x_spin)
        controls_layout.addRow("Rotate Y", self.y_spin)
        controls_layout.addRow("Rotate Z", self.z_spin)
        layout.addWidget(self.controls_group)
        layout.setStretch(2, 4)
        layout.setStretch(3, 2)
        self._handle_nudge_step_changed(self.nudge_step_spin.value())
        self._set_interaction_mode("rotate")
        self._set_nudge_buttons_enabled(False)
        self._set_toolbar_angle_spins_enabled(False)

    def _build_reference_panel(self) -> QWidget:
        panel = QGroupBox("Render Reference")
        panel.setMinimumWidth(250)
        panel.setMaximumWidth(300)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        self.reference_style_label = QLabel("Aesthetic: None")
        self.reference_style_label.setWordWrap(True)
        layout.addWidget(self.reference_style_label)

        self.reference_lighting_label = QLabel("Lighting: None")
        self.reference_lighting_label.setWordWrap(True)
        layout.addWidget(self.reference_lighting_label)

        images_row = QHBoxLayout()
        images_row.setContentsMargins(0, 0, 0, 0)
        images_row.setSpacing(8)

        dark_card = QVBoxLayout()
        dark_card.setContentsMargins(0, 0, 0, 0)
        dark_card.setSpacing(4)
        self.reference_dark_title_label = QLabel(REFERENCE_ATOM_LABELS["C"])
        self.reference_dark_title_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )
        self.reference_dark_title_label.setWordWrap(True)
        dark_card.addWidget(self.reference_dark_title_label)
        self.reference_dark_image_label = QLabel("Select an orientation row")
        self.reference_dark_image_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )
        self.reference_dark_image_label.setMinimumSize(114, 114)
        self.reference_dark_image_label.setFrameShape(QFrame.Shape.StyledPanel)
        self.reference_dark_image_label.setWordWrap(True)
        dark_card.addWidget(self.reference_dark_image_label, stretch=1)
        images_row.addLayout(dark_card, stretch=1)

        light_card = QVBoxLayout()
        light_card.setContentsMargins(0, 0, 0, 0)
        light_card.setSpacing(4)
        self.reference_light_title_label = QLabel(REFERENCE_ATOM_LABELS["S"])
        self.reference_light_title_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )
        self.reference_light_title_label.setWordWrap(True)
        light_card.addWidget(self.reference_light_title_label)
        self.reference_light_image_label = QLabel("Select an orientation row")
        self.reference_light_image_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )
        self.reference_light_image_label.setMinimumSize(114, 114)
        self.reference_light_image_label.setFrameShape(QFrame.Shape.StyledPanel)
        self.reference_light_image_label.setWordWrap(True)
        light_card.addWidget(self.reference_light_image_label, stretch=1)
        images_row.addLayout(light_card, stretch=1)

        layout.addLayout(images_row)

        self.reference_status_label = QLabel("")
        self.reference_status_label.setWordWrap(True)
        self.reference_status_label.setStyleSheet("color: #4b5563;")
        layout.addWidget(self.reference_status_label)

        self.reference_hint_label = QLabel(
            "Transparent Blender carbon and sulfur renders composited on "
            "the selected reference background."
        )
        self.reference_hint_label.setWordWrap(True)
        self.reference_hint_label.setStyleSheet("color: #4b5563;")
        layout.addWidget(self.reference_hint_label)
        layout.addStretch(1)
        self._apply_reference_background()
        return panel

    def detach_controls_group(self) -> QGroupBox:
        layout = self.layout()
        if layout is not None:
            layout.removeWidget(self.controls_group)
        self.controls_group.setParent(None)
        return self.controls_group

    def _build_angle_spinbox(self, slot) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(-360.0, 360.0)
        spin.setDecimals(1)
        spin.setSingleStep(1.0)
        spin.valueChanged.connect(slot)
        return spin

    def _build_toolbar_angle_spinbox(self, slot) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(-360.0, 360.0)
        spin.setDecimals(1)
        spin.setSingleStep(1.0)
        spin.setFixedWidth(88)
        spin.valueChanged.connect(slot)
        return spin

    def _set_toolbar_angle_spins_enabled(self, enabled: bool) -> None:
        for spin in (
            self.x_toolbar_spin,
            self.y_toolbar_spin,
            self.z_toolbar_spin,
        ):
            spin.setEnabled(enabled)

    def _sync_toolbar_angle_spins_from_controls(self) -> None:
        for toolbar_spin, control_spin in (
            (self.x_toolbar_spin, self.x_spin),
            (self.y_toolbar_spin, self.y_spin),
            (self.z_toolbar_spin, self.z_spin),
        ):
            toolbar_spin.blockSignals(True)
            toolbar_spin.setValue(float(control_spin.value()))
            toolbar_spin.blockSignals(False)

    def _set_angle_controls(
        self,
        x_degrees: float,
        y_degrees: float,
        z_degrees: float,
        *,
        preserve_name: bool = True,
        redraw: bool = True,
    ) -> None:
        if self._orientation is None:
            return
        self._updating_controls = True
        self.x_spin.setValue(float(x_degrees))
        self.y_spin.setValue(float(y_degrees))
        self.z_spin.setValue(float(z_degrees))
        self._sync_toolbar_angle_spins_from_controls()
        self._updating_controls = False
        self._update_rotation_readout()
        self._emit_orientation(preserve_name=preserve_name)
        if redraw:
            self._draw_preview()

    def _set_interaction_mode(self, mode: str) -> None:
        if mode not in {"rotate", "pan", "zoom"}:
            return
        self._interaction_mode = mode
        self.rotate_tool_button.setChecked(mode == "rotate")
        self.pan_tool_button.setChecked(mode == "pan")
        self.zoom_tool_button.setChecked(mode == "zoom")
        self._update_canvas_cursor()

    def _set_nudge_buttons_enabled(self, enabled: bool) -> None:
        for button in (
            self.x_minus_button,
            self.x_plus_button,
            self.y_minus_button,
            self.y_plus_button,
            self.z_minus_button,
            self.z_plus_button,
        ):
            button.setEnabled(enabled)

    def _nudge_structure_rotation(
        self, axis_name: str, step_sign: float
    ) -> None:
        if self._orientation is None:
            return
        spin_map = {
            "x": self.x_spin,
            "y": self.y_spin,
            "z": self.z_spin,
        }
        spin = spin_map.get(axis_name)
        if spin is None:
            return
        step_size = float(self.nudge_step_spin.value())
        value = float(spin.value()) + (float(step_sign) * step_size)
        while value > 360.0:
            value -= 720.0
        while value < -360.0:
            value += 720.0
        spin.setValue(value)

    def _update_rotation_readout(self) -> None:
        if self._orientation is None:
            self.rotation_readout_label.setText(
                "Rotation X 0.0 deg  Y 0.0 deg  Z 0.0 deg"
            )
            return
        self.rotation_readout_label.setText(
            "Rotation "
            f"X {self.x_spin.value():.1f} deg  "
            f"Y {self.y_spin.value():.1f} deg  "
            f"Z {self.z_spin.value():.1f} deg"
        )

    @Slot(float)
    def _handle_nudge_step_changed(self, value: float) -> None:
        step = max(float(value), 0.1)
        for spin in (
            self.x_spin,
            self.y_spin,
            self.z_spin,
            self.x_toolbar_spin,
            self.y_toolbar_spin,
            self.z_toolbar_spin,
        ):
            spin.setSingleStep(step)

    @Slot(float)
    def _handle_toolbar_angle_changed(
        self, axis_name: str, value: float
    ) -> None:
        if self._updating_controls:
            return
        if self._orientation is None:
            return
        x_value = float(self.x_spin.value())
        y_value = float(self.y_spin.value())
        z_value = float(self.z_spin.value())
        if axis_name == "x":
            x_value = float(value)
        elif axis_name == "y":
            y_value = float(value)
        elif axis_name == "z":
            z_value = float(value)
        else:
            return
        self._set_angle_controls(
            x_value,
            y_value,
            z_value,
            preserve_name=True,
        )

    def set_nudge_increment(self, degrees: float) -> None:
        self.nudge_step_spin.setValue(max(float(degrees), 0.1))

    def nudge_increment(self) -> float:
        return float(self.nudge_step_spin.value())

    def set_background_override(self, color_value: str | None) -> None:
        self._background_override = (
            (str(color_value).strip() or None) if color_value else None
        )
        if self._structure is None or self._orientation is None:
            self.clear_preview()
            return
        self._draw_preview()

    def background_override(self) -> str | None:
        return self._background_override

    def effective_background_color(self) -> str:
        if self._background_override:
            return self._background_override
        palette = _preview_palette(self._atom_style)
        return str(palette["background"])

    def set_reference_background_color(self, color_value: str | None) -> None:
        color_text = str(color_value).strip() if color_value else ""
        self._reference_background_color = (
            color_text or REFERENCE_ATOM_BACKGROUND
        )
        self._apply_reference_background()

    def reference_background_color(self) -> str:
        return self._reference_background_color

    def _reference_image_labels(self) -> tuple[QLabel, ...]:
        labels: list[QLabel] = []
        for name in (
            "reference_dark_image_label",
            "reference_light_image_label",
        ):
            label = getattr(self, name, None)
            if isinstance(label, QLabel):
                labels.append(label)
        return tuple(labels)

    def _apply_reference_background(self) -> None:
        labels = self._reference_image_labels()
        if not labels:
            return
        for label in labels:
            label.setStyleSheet(
                "background-color: "
                f"{self._reference_background_color}; "
                "border: 1px solid #c8d0da;"
            )
        if any(label.pixmap() is not None for label in labels):
            self._refresh_reference_preview()

    def _reference_pixmap(
        self,
        atom_style: str,
        lighting_level: int,
        element: str,
    ) -> QPixmap | None:
        cache_key = (
            normalize_atom_style(atom_style),
            normalize_lighting_level(lighting_level),
            str(element).strip().upper(),
        )
        cached = self._reference_pixmap_cache.get(cache_key)
        if cached is not None:
            return cached
        path = reference_atom_path(*cache_key)
        if not path.is_file():
            return None
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            return None
        self._reference_pixmap_cache[cache_key] = pixmap
        return pixmap

    def _refresh_reference_preview(self) -> None:
        labels = self._reference_image_labels()
        if not labels:
            return
        if self._orientation is None:
            self.reference_style_label.setText("Aesthetic: None")
            self.reference_lighting_label.setText("Lighting: None")
            self.reference_status_label.setText("")
            for label in labels:
                label.clear()
                label.setText("Select an orientation row")
            return

        lighting_level = self._orientation.effective_lighting_level(
            DEFAULT_LIGHTING_LEVEL
        )
        self.reference_style_label.setText(
            "Aesthetic: "
            f"{atom_style_label(self._atom_style)}"
        )
        self.reference_lighting_label.setText(
            "Lighting: "
            f"{LIGHTING_LEVEL_LABELS[lighting_level]}"
        )
        if atom_style_is_custom(self._atom_style):
            self.reference_status_label.setText(
                "Reference swatches are only generated for built-in "
                "aesthetics. Use the live preview to inspect custom atom "
                "colors and sizes."
            )
            for label in labels:
                label.clear()
                label.setText("Unavailable")
            return
        self.reference_status_label.setText("")
        missing_elements: list[str] = []
        for element, label in (
            ("C", self.reference_dark_image_label),
            ("S", self.reference_light_image_label),
        ):
            pixmap = self._reference_pixmap(
                self._atom_style,
                lighting_level,
                element,
            )
            if pixmap is None:
                label.clear()
                label.setText("Missing swatch")
                missing_elements.append(element)
                continue
            scaled = pixmap.scaled(
                label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            label.setText("")
            label.setPixmap(scaled)
        if missing_elements:
            names = ", ".join(
                REFERENCE_ATOM_LABELS[element]
                for element in missing_elements
            )
            self.reference_status_label.setText(
                f"Reference render not found for {names}. "
                "Run generate_reference_atoms.py to rebuild the swatches."
            )

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        if self._orientation is not None:
            self._refresh_reference_preview()

    def _update_canvas_cursor(self, *, dragging: bool = False) -> None:
        if dragging:
            if self._drag_mode in {"rotate", "pan"}:
                cursor = Qt.CursorShape.ClosedHandCursor
            else:
                cursor = Qt.CursorShape.SizeVerCursor
        elif self._interaction_mode == "zoom":
            cursor = Qt.CursorShape.SizeVerCursor
        else:
            cursor = Qt.CursorShape.OpenHandCursor
        self.canvas.setCursor(cursor)

    def clear_preview(self) -> None:
        self._structure = None
        self._orientation = None
        self._positions = np.zeros((0, 3), dtype=float)
        self._updating_controls = True
        self.source_label.setText("None")
        self.preview_style_label.setText(atom_style_label(self._atom_style))
        self.name_edit.clear()
        self.name_edit.setReadOnly(True)
        self.enabled_box.setChecked(False)
        for spin in (self.x_spin, self.y_spin, self.z_spin):
            spin.setValue(0.0)
            spin.setEnabled(False)
        self._sync_toolbar_angle_spins_from_controls()
        self._set_toolbar_angle_spins_enabled(False)
        self.name_edit.setEnabled(False)
        self.enabled_box.setEnabled(False)
        self._set_nudge_buttons_enabled(False)
        self.reset_view_button.setEnabled(False)
        self._updating_controls = False
        self._update_rotation_readout()
        self._refresh_reference_preview()

        self.figure.clear()
        axis = self.figure.add_subplot(111, projection="3d")
        self._axis = axis
        self._axis_reference = None
        self._view_center = np.zeros(3, dtype=float)
        self._view_radius = 1.0
        self._view_azim = _RENDER_MATCHED_VIEW_AZIM
        self._view_elev = _RENDER_MATCHED_VIEW_ELEV
        background_color = self.effective_background_color()
        self.figure.set_facecolor(background_color)
        axis.set_facecolor(background_color)
        try:
            axis.set_proj_type("ortho")
        except Exception:
            pass
        axis.grid(False)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_zticks([])
        self._apply_view_to_axis()
        self._update_canvas_cursor()
        axis.axis("off")
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def set_preview(
        self,
        structure: BlenderPreviewStructure,
        orientation: OrientationSpec,
        *,
        row_index: int,
        row_count: int,
        atom_style: str,
    ) -> None:
        reset_view = (
            self._structure is None
            or self._structure.input_path != structure.input_path
        )
        self._structure = structure
        self._orientation = orientation
        self._positions = np.asarray(
            [atom.position for atom in structure.atoms],
            dtype=float,
        )
        self._atom_style = normalize_atom_style(atom_style)
        self._updating_controls = True
        self.source_label.setText(orientation.source.title())
        self.preview_style_label.setText(atom_style_label(self._atom_style))
        self.name_edit.setEnabled(True)
        self.enabled_box.setEnabled(True)
        self.name_edit.setReadOnly(orientation.source != "custom")
        self.name_edit.setText(orientation.label)
        self.enabled_box.setChecked(orientation.enabled)
        self.x_spin.setEnabled(True)
        self.y_spin.setEnabled(True)
        self.z_spin.setEnabled(True)
        self._set_toolbar_angle_spins_enabled(True)
        self._set_nudge_buttons_enabled(True)
        self.x_spin.setValue(float(orientation.x_degrees))
        self.y_spin.setValue(float(orientation.y_degrees))
        self.z_spin.setValue(float(orientation.z_degrees))
        self._sync_toolbar_angle_spins_from_controls()
        self.reset_view_button.setEnabled(True)
        self._updating_controls = False
        self._update_rotation_readout()
        self._refresh_reference_preview()
        self._draw_preview(reset_view=reset_view)

    def refresh_style(self, atom_style: str) -> None:
        self._atom_style = normalize_atom_style(atom_style)
        self.preview_style_label.setText(atom_style_label(self._atom_style))
        self._refresh_reference_preview()
        if self._structure is not None and self._orientation is not None:
            self._draw_preview()

    @Slot()
    def reset_view(self) -> None:
        if self._structure is None or self._orientation is None:
            return
        self._draw_preview(reset_view=True)

    def _emit_orientation(self, *, preserve_name: bool = False) -> None:
        if self._orientation is None:
            return
        label = (
            self._orientation.label
            if preserve_name
            else (self.name_edit.text().strip() or self._orientation.label)
        )
        self._orientation = OrientationSpec(
            key=sanitize_orientation_key(label or self._orientation.key),
            label=label,
            source=self._orientation.source,
            x_degrees=float(self.x_spin.value()),
            y_degrees=float(self.y_spin.value()),
            z_degrees=float(self.z_spin.value()),
            enabled=self.enabled_box.isChecked(),
            atom_style=self._orientation.atom_style,
            render_quality=self._orientation.render_quality,
            lighting_level=self._orientation.lighting_level,
            save_legend=self._orientation.save_legend,
        )
        self.orientationChanged.emit(self._orientation)

    def _draw_preview(self, *, reset_view: bool = False) -> None:
        if self._structure is None or self._orientation is None:
            self.clear_preview()
            return

        rotated = apply_orientation_to_positions(
            self._positions,
            self._orientation,
        )
        x_values = rotated[:, 0]
        y_values = rotated[:, 1]
        z_values = rotated[:, 2]
        palette = _preview_palette(self._atom_style)
        background_color = self.effective_background_color()
        style_defaults = atom_style_defaults(self._atom_style)

        self.figure.clear()
        axis = self.figure.add_subplot(111, projection="3d")
        self._axis = axis
        self._axis_reference = self.figure.add_axes(
            [0.05, 0.06, 0.13, 0.13],
            projection="3d",
        )
        self._axis_reference.set_in_layout(False)
        self.figure.set_facecolor(background_color)
        axis.set_facecolor(background_color)
        try:
            axis.set_proj_type("ortho")
        except Exception:
            pass
        axis.grid(False)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_zticks([])
        axis.set_xlabel("")
        axis.set_ylabel("")
        axis.set_zlabel("")
        try:
            axis.set_box_aspect((1.0, 1.0, 1.0))
        except Exception:
            pass
        for axis_obj in (axis.xaxis, axis.yaxis, axis.zaxis):
            try:
                axis_obj.line.set_alpha(0.0)
                axis_obj.pane.set_alpha(0.0)
            except Exception:
                pass

        depth_min = float(y_values.min())
        depth_span = max(float(y_values.max()) - depth_min, 1.0e-6)
        bond_color_mode = str(style_defaults["bond_color_mode"])
        neutral_bond_color = _preview_neutral_bond_color(self._atom_style)
        bond_width = max(
            2.4,
            10.0 * float(style_defaults["bond_radius"]),
        )

        for left_index, right_index in sorted(
            self._structure.bonds,
            key=lambda pair: float(y_values[pair[0]] + y_values[pair[1]]),
        ):
            depth = (
                float(y_values[left_index]) + float(y_values[right_index])
            ) * 0.5
            depth_factor = (depth - depth_min) / depth_span
            start = rotated[left_index]
            end = rotated[right_index]
            midpoint = (start + end) * 0.5

            # Draw a faint neutral underlay first so bonds stay visible in
            # the quick preview even when atom colors are bright or pale.
            axis.plot(
                (start[0], end[0]),
                (start[1], end[1]),
                (start[2], end[2]),
                color=neutral_bond_color,
                linewidth=bond_width + 1.4,
                alpha=0.26 + depth_factor * 0.10,
                solid_capstyle="round",
            )

            if bond_color_mode == "split":
                left_color = _preview_bond_color(
                    self._structure.atoms[int(left_index)].element,
                    atom_style=self._atom_style,
                )
                right_color = _preview_bond_color(
                    self._structure.atoms[int(right_index)].element,
                    atom_style=self._atom_style,
                )
                axis.plot(
                    (start[0], midpoint[0]),
                    (start[1], midpoint[1]),
                    (start[2], midpoint[2]),
                    color=left_color,
                    linewidth=bond_width,
                    alpha=0.72 + depth_factor * 0.14,
                    solid_capstyle="round",
                )
                axis.plot(
                    (midpoint[0], end[0]),
                    (midpoint[1], end[1]),
                    (midpoint[2], end[2]),
                    color=right_color,
                    linewidth=bond_width,
                    alpha=0.72 + depth_factor * 0.14,
                    solid_capstyle="round",
                )
            else:
                axis.plot(
                    (start[0], end[0]),
                    (start[1], end[1]),
                    (start[2], end[2]),
                    color=neutral_bond_color,
                    linewidth=bond_width,
                    alpha=0.68 + depth_factor * 0.14,
                    solid_capstyle="round",
                )

        for atom_index in np.argsort(y_values):
            atom = self._structure.atoms[int(atom_index)]
            depth_factor = (
                float(y_values[atom_index]) - depth_min
            ) / depth_span
            color = _preview_atom_color(
                atom.element,
                atom_style=self._atom_style,
            )
            size = (
                style_display_radius(
                    atom.element,
                    atom_style=self._atom_style,
                )
                * 30.0
            ) ** 2
            axis.scatter(
                [x_values[atom_index]],
                [y_values[atom_index]],
                [z_values[atom_index]],
                s=size * (1.04 + depth_factor * 0.24),
                c=[palette["shadow"]],
                alpha=0.10,
                linewidths=0.0,
                zorder=2,
            )
            axis.scatter(
                [x_values[atom_index]],
                [y_values[atom_index]],
                [z_values[atom_index]],
                s=size * (0.84 + depth_factor * 0.34),
                c=[color],
                alpha=0.94,
                edgecolors=[palette["edge"]],
                linewidths=1.0,
                zorder=3,
                depthshade=False,
            )

        legend_handles: list[Line2D] = []
        for element in _preview_structure_elements(self._structure):
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="None",
                    markersize=7.5,
                    markerfacecolor=_preview_atom_color(
                        element,
                        atom_style=self._atom_style,
                    ),
                    markeredgecolor=palette["edge"],
                    markeredgewidth=1.0,
                    label=element,
                )
            )
        if legend_handles:
            legend = axis.legend(
                handles=legend_handles,
                loc="upper left",
                bbox_to_anchor=(1.01, 0.98),
                frameon=True,
                fancybox=True,
                framealpha=0.94,
                borderpad=0.5,
                labelspacing=0.35,
                handletextpad=0.5,
                borderaxespad=0.0,
                fontsize=8.5,
            )
            frame = legend.get_frame()
            frame.set_facecolor(background_color)
            frame.set_edgecolor("#c6ccd4")
            frame.set_linewidth(0.8)

        minima = rotated.min(axis=0)
        maxima = rotated.max(axis=0)
        center = (minima + maxima) * 0.5
        radius = max(float(np.max(maxima - minima)) * 0.6, 1.0)
        if reset_view:
            self._view_center = center
            self._view_radius = radius
            # Match Blender's export camera: orthographic, looking down the
            # structure y-axis with z-up.
            self._view_azim = _RENDER_MATCHED_VIEW_AZIM
            self._view_elev = _RENDER_MATCHED_VIEW_ELEV
        self._apply_view_to_axis()
        axis.axis("off")
        self.figure.tight_layout(rect=(0.0, 0.0, 0.95, 1.0))
        self.canvas.draw_idle()

    def _apply_view_to_axis(self) -> None:
        if self._axis is None:
            return
        self._axis.set_xlim(
            self._view_center[0] - self._view_radius,
            self._view_center[0] + self._view_radius,
        )
        self._axis.set_ylim(
            self._view_center[1] - self._view_radius,
            self._view_center[1] + self._view_radius,
        )
        self._axis.set_zlim(
            self._view_center[2] - self._view_radius,
            self._view_center[2] + self._view_radius,
        )
        self._axis.view_init(elev=self._view_elev, azim=self._view_azim)
        self._draw_axis_reference()

    def _draw_axis_reference(self) -> None:
        if self._axis_reference is None:
            return
        axis_ref = self._axis_reference
        axis_ref.cla()
        axis_ref.set_facecolor((1.0, 1.0, 1.0, 0.0))
        axis_ref.patch.set_alpha(0.0)
        try:
            axis_ref.set_proj_type("ortho")
        except Exception:
            pass
        axis_ref.grid(False)
        axis_ref.set_xticks([])
        axis_ref.set_yticks([])
        axis_ref.set_zticks([])
        try:
            axis_ref.set_box_aspect((1.0, 1.0, 1.0))
        except Exception:
            pass
        for axis_obj in (axis_ref.xaxis, axis_ref.yaxis, axis_ref.zaxis):
            try:
                axis_obj.line.set_alpha(0.0)
                axis_obj.pane.set_alpha(0.0)
            except Exception:
                pass
        axis_ref.view_init(elev=self._view_elev, azim=self._view_azim)
        axis_length = 0.8
        label_scale = 1.15
        if self._orientation is None:
            axis_vectors = np.eye(3, dtype=float) * axis_length
        else:
            axis_vectors = apply_orientation_to_positions(
                np.eye(3, dtype=float) * axis_length,
                self._orientation,
            )
        for vector, color, label in (
            (axis_vectors[0], "#d9534f", "x"),
            (axis_vectors[1], "#3aa655", "y"),
            (axis_vectors[2], "#3b7ddd", "z"),
        ):
            axis_ref.plot(
                (0.0, float(vector[0])),
                (0.0, float(vector[1])),
                (0.0, float(vector[2])),
                color=color,
                linewidth=2.0,
            )
            axis_ref.text(
                float(vector[0]) * label_scale,
                float(vector[1]) * label_scale,
                float(vector[2]) * label_scale,
                label,
                color=color,
                fontsize=8,
            )
        axis_ref.scatter(
            [0.0], [0.0], [0.0], s=10, c=["#6b7280"], depthshade=False
        )
        axis_ref.set_xlim(-1.0, 1.0)
        axis_ref.set_ylim(-1.0, 1.0)
        axis_ref.set_zlim(-1.0, 1.0)

    def _view_basis_vectors(self) -> tuple[np.ndarray, np.ndarray]:
        azimuth = np.radians(self._view_azim)
        elevation = np.radians(self._view_elev)
        forward = np.array(
            [
                np.cos(elevation) * np.cos(azimuth),
                np.cos(elevation) * np.sin(azimuth),
                np.sin(elevation),
            ],
            dtype=float,
        )
        world_up = np.array([0.0, 0.0, 1.0], dtype=float)
        right = np.cross(forward, world_up)
        norm = float(np.linalg.norm(right))
        if norm < 1.0e-8:
            right = np.array([1.0, 0.0, 0.0], dtype=float)
        else:
            right /= norm
        up = np.cross(right, forward)
        up_norm = float(np.linalg.norm(up))
        if up_norm < 1.0e-8:
            up = np.array([0.0, 1.0, 0.0], dtype=float)
        else:
            up /= up_norm
        return right, up

    def _handle_canvas_press(self, event) -> None:
        if self._axis is None or event.inaxes not in {
            self._axis,
            self._axis_reference,
        }:
            return
        if event.button is MouseButton.LEFT:
            self._drag_mode = self._interaction_mode
        elif event.button in (MouseButton.MIDDLE, MouseButton.RIGHT):
            self._drag_mode = "pan"
        else:
            self._drag_mode = None
            return
        self._drag_start_xy = (float(event.x), float(event.y))
        self._drag_last_xy = self._drag_start_xy
        self._drag_start_azim = self._view_azim
        self._drag_start_elev = self._view_elev
        self._drag_start_center = self._view_center.copy()
        self._drag_start_radius = self._view_radius
        self._update_canvas_cursor(dragging=True)

    def _handle_canvas_release(self, _event) -> None:
        self._drag_mode = None
        self._update_canvas_cursor()

    def _handle_canvas_motion(self, event) -> None:
        if self._axis is None or self._drag_mode is None:
            return
        delta_x = float(event.x) - self._drag_start_xy[0]
        delta_y = float(event.y) - self._drag_start_xy[1]
        step_delta_x = float(event.x) - self._drag_last_xy[0]
        step_delta_y = float(event.y) - self._drag_last_xy[1]
        width = max(float(self.canvas.width()), 1.0)
        height = max(float(self.canvas.height()), 1.0)
        if self._drag_mode == "rotate":
            if self._orientation is None:
                return
            rotation_scale = 120.0 / max(min(width, height), 1.0)
            x_value, y_value, z_value = compose_euler_degrees(
                self._orientation.euler_degrees,
                (
                    step_delta_y * rotation_scale * 0.82,
                    0.0,
                    step_delta_x * rotation_scale,
                ),
            )
            self._set_angle_controls(
                x_value,
                y_value,
                z_value,
                preserve_name=True,
                redraw=False,
            )
            self._drag_last_xy = (float(event.x), float(event.y))
        elif self._drag_mode == "pan":
            data_scale = (2.0 * self._view_radius) / max(width, height)
            right, up = self._view_basis_vectors()
            self._view_center = (
                self._drag_start_center
                + right * (-delta_x * data_scale)
                + up * (delta_y * data_scale)
            )
        elif self._drag_mode == "zoom":
            zoom_factor = float(np.exp(delta_y / height * 2.0))
            self._view_radius = float(
                np.clip(self._drag_start_radius * zoom_factor, 0.2, 1.0e5)
            )
        if self._drag_mode == "rotate":
            return
        self._apply_view_to_axis()
        self.canvas.draw_idle()

    def _handle_canvas_scroll(self, event) -> None:
        if self._axis is None or event.inaxes not in {
            self._axis,
            self._axis_reference,
        }:
            return
        step = float(getattr(event, "step", 0.0))
        if step == 0.0:
            if event.button == "up":
                step = 1.0
            elif event.button == "down":
                step = -1.0
        if step == 0.0:
            return
        if step > 0.0:
            factor = 0.9**step
        else:
            factor = 1.1 ** abs(step)
        self._view_radius = float(
            np.clip(self._view_radius * factor, 0.2, 1.0e5)
        )
        self._apply_view_to_axis()
        self.canvas.draw_idle()

    @Slot(str)
    def _handle_name_changed(self, _text: str) -> None:
        if self._updating_controls or self._orientation is None:
            return
        self._emit_orientation()
        self._draw_preview()

    @Slot(bool)
    def _handle_enabled_changed(self, _checked: bool) -> None:
        if self._updating_controls or self._orientation is None:
            return
        self._emit_orientation(preserve_name=True)

    @Slot(float)
    def _handle_angles_changed(self, _value: float) -> None:
        if self._updating_controls or self._orientation is None:
            return
        self._sync_toolbar_angle_spins_from_controls()
        self._update_rotation_readout()
        self._emit_orientation(preserve_name=True)
        self._draw_preview()


class BlenderXYZRenderWorker(QObject):
    log = Signal(str)
    progress = Signal(int, int, str)
    status = Signal(str)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, settings: BlenderXYZRenderSettings) -> None:
        super().__init__()
        self.settings = settings

    @Slot()
    def run(self) -> None:
        workflow = BlenderXYZRenderWorkflow(self.settings)
        try:
            prepared = workflow.prepare_settings()
            total_orientations = len(prepared.orientations)
            command = workflow.build_command(prepared)
            self.status.emit(
                f"Launching Blender render for {total_orientations} orientations..."
            )
            self.progress.emit(0, total_orientations, "Launching Blender...")
            self.log.emit(f"Input: {prepared.input_path}")
            self.log.emit(f"Output folder: {prepared.output_dir}")
            self.log.emit(f"Blender: {prepared.blender_executable}")
            self.log.emit(f"Atom style: {prepared.atom_style}")
            self.log.emit(f"Render quality: {prepared.render_quality}")
            self.log.emit(
                f"Bond threshold pairs: {len(prepared.bond_thresholds)}"
            )
            self.log.emit(f"Command: {shlex.join(command)}")

            def _handle_progress_event(
                event_name: str,
                current: int,
                total: int,
                label: str,
            ) -> None:
                item_label = label or f"Orientation {current}"
                if event_name == "start":
                    self.status.emit(
                        f"Rendering {current} of {total}: {item_label}"
                    )
                    self.progress.emit(
                        max(current - 1, 0),
                        total,
                        f"Rendering {current}/{total}: {item_label}",
                    )
                    self.log.emit(f"Rendering {current}/{total}: {item_label}")
                    return
                self.status.emit(
                    f"Rendered {current} of {total}: {item_label}"
                )
                self.progress.emit(
                    current,
                    total,
                    f"Rendered {current}/{total}: {item_label}",
                )

            result = workflow.run_streaming(
                line_callback=self.log.emit,
                progress_callback=_handle_progress_event,
            )
        except Exception as exc:
            self.failed.emit(str(exc))
            return

        for output_path in result.output_paths:
            self.log.emit(f"Rendered: {output_path}")
        for blend_path in result.blend_paths:
            self.log.emit(f"Saved Blender scene: {blend_path}")
        for legend_path in result.legend_paths:
            self.log.emit(f"Saved atom legend: {legend_path}")
        self.progress.emit(
            len(result.output_paths),
            len(result.output_paths),
            "Blender render complete",
        )
        self.finished.emit(result)


class BondThresholdEditorDialog(QDialog):
    """Edit per-pair bond search lengths for the active structure."""

    def __init__(
        self,
        thresholds: tuple[BondThresholdSpec, ...] | list[BondThresholdSpec],
        *,
        default_thresholds: (
            tuple[BondThresholdSpec, ...] | list[BondThresholdSpec]
        ),
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Bond Thresholds")
        self.resize(720, 460)
        self._default_thresholds = {
            spec.pair_key: spec for spec in default_thresholds
        }

        layout = QVBoxLayout(self)
        intro = QLabel(
            "VESTA-style bond searching uses minimum and maximum lengths for each "
            "atom-pair type. These values are seeded here from covalent-radius "
            "sums so you can fine-tune the active structure before previewing "
            "or rendering."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        self.table = QTableWidget(len(thresholds), 4, self)
        self.table.setHorizontalHeaderLabels(
            [
                "Pair",
                "Seed Max (\u00c5)",
                "Min (\u00c5)",
                "Max (\u00c5)",
            ]
        )
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(
            0,
            QHeaderView.ResizeMode.Stretch,
        )
        for column in (1, 2, 3):
            self.table.horizontalHeader().setSectionResizeMode(
                column,
                QHeaderView.ResizeMode.ResizeToContents,
            )
        layout.addWidget(self.table, stretch=1)

        for row_index, spec in enumerate(thresholds):
            pair_item = QTableWidgetItem(f"{spec.element_a}-{spec.element_b}")
            pair_item.setFlags(
                Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
            )
            pair_item.setData(Qt.ItemDataRole.UserRole, spec.pair_key)
            self.table.setItem(row_index, 0, pair_item)

            seed_spec = self._default_thresholds.get(spec.pair_key, spec)
            seed_item = QTableWidgetItem(f"{seed_spec.max_length:.3f}")
            seed_item.setFlags(
                Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
            )
            seed_item.setTextAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
            self.table.setItem(row_index, 1, seed_item)

            self.table.setCellWidget(
                row_index,
                2,
                self._build_spinbox(spec.min_length),
            )
            self.table.setCellWidget(
                row_index,
                3,
                self._build_spinbox(spec.max_length),
            )

        controls_row = QHBoxLayout()
        self.reset_defaults_button = QPushButton("Reset Seed Defaults")
        self.reset_defaults_button.clicked.connect(self._reset_defaults)
        controls_row.addWidget(self.reset_defaults_button)
        controls_row.addStretch(1)
        layout.addLayout(controls_row)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _build_spinbox(self, value: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox(self.table)
        spin.setDecimals(3)
        spin.setRange(0.0, 50.0)
        spin.setSingleStep(0.05)
        spin.setValue(float(value))
        spin.setAlignment(Qt.AlignmentFlag.AlignRight)
        return spin

    @Slot()
    def _reset_defaults(self) -> None:
        for row_index in range(self.table.rowCount()):
            pair_item = self.table.item(row_index, 0)
            if pair_item is None:
                continue
            pair_key = pair_item.data(Qt.ItemDataRole.UserRole)
            if not pair_key:
                continue
            seed_spec = self._default_thresholds.get(tuple(pair_key))
            if seed_spec is None:
                continue
            min_spin = self.table.cellWidget(row_index, 2)
            max_spin = self.table.cellWidget(row_index, 3)
            if isinstance(min_spin, QDoubleSpinBox):
                min_spin.setValue(float(seed_spec.min_length))
            if isinstance(max_spin, QDoubleSpinBox):
                max_spin.setValue(float(seed_spec.max_length))

    def bond_thresholds(self) -> tuple[BondThresholdSpec, ...]:
        thresholds: list[BondThresholdSpec] = []
        for row_index in range(self.table.rowCount()):
            pair_item = self.table.item(row_index, 0)
            min_spin = self.table.cellWidget(row_index, 2)
            max_spin = self.table.cellWidget(row_index, 3)
            if (
                pair_item is None
                or not isinstance(min_spin, QDoubleSpinBox)
                or not isinstance(max_spin, QDoubleSpinBox)
            ):
                continue
            pair_key = pair_item.data(Qt.ItemDataRole.UserRole)
            if not pair_key or len(pair_key) != 2:
                continue
            thresholds.append(
                BondThresholdSpec(
                    element_a=str(pair_key[0]),
                    element_b=str(pair_key[1]),
                    min_length=float(min_spin.value()),
                    max_length=float(max_spin.value()),
                )
            )
        return tuple(thresholds)


class AtomAestheticEditorDialog(QDialog):
    """Edit per-element atom colors and size scales for a custom aesthetic."""

    def __init__(
        self,
        *,
        active_style: str,
        elements: tuple[str, ...] | list[str],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._elements = tuple(elements)
        self._active_style = normalize_atom_style(active_style)
        self._active_custom = get_custom_aesthetic(self._active_style)
        self._active_base_style = atom_style_base(self._active_style)
        self.setWindowTitle("Atom Colors and Sizes")
        self.resize(760, 520)

        layout = QVBoxLayout(self)
        intro = QLabel(
            "Adjust per-element atom colors and sizes for the elements present "
            "in the active structure. Built-in aesthetics stay read-only; "
            "saving here creates or updates a named custom aesthetic that can "
            "be recalled across sessions. Elements not listed in this table fall "
            "back to the chosen preset when reused on another structure."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        form = QFormLayout()
        self.name_edit = QLineEdit()
        default_name = (
            self._active_custom.name
            if self._active_custom is not None
            else f"{atom_style_label(self._active_style)} Custom"
        )
        self.name_edit.setText(default_name)
        form.addRow("Custom Name", self.name_edit)

        self.seed_preset_combo = QComboBox()
        for key, label in ATOM_STYLE_LABELS.items():
            self.seed_preset_combo.addItem(label, key)
        self.seed_preset_combo.setCurrentIndex(
            max(
                self.seed_preset_combo.findData(self._active_base_style),
                0,
            )
        )
        form.addRow("Preset Seed", self.seed_preset_combo)
        layout.addLayout(form)

        controls_row = QHBoxLayout()
        self.apply_preset_button = QPushButton("Load Preset Values")
        self.apply_preset_button.clicked.connect(self._apply_selected_preset)
        controls_row.addWidget(self.apply_preset_button)
        self.reset_active_button = QPushButton("Reset To Active Aesthetic")
        self.reset_active_button.clicked.connect(self._reset_to_active_style)
        controls_row.addWidget(self.reset_active_button)
        controls_row.addStretch(1)
        layout.addLayout(controls_row)

        self.table = QTableWidget(len(self._elements), 4, self)
        self.table.setHorizontalHeaderLabels(
            [
                "Element",
                "Color",
                "Size Scale",
                "Radius (\u00c5)",
            ]
        )
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(
            0,
            QHeaderView.ResizeMode.ResizeToContents,
        )
        self.table.horizontalHeader().setSectionResizeMode(
            1,
            QHeaderView.ResizeMode.Stretch,
        )
        self.table.horizontalHeader().setSectionResizeMode(
            2,
            QHeaderView.ResizeMode.ResizeToContents,
        )
        self.table.horizontalHeader().setSectionResizeMode(
            3,
            QHeaderView.ResizeMode.ResizeToContents,
        )
        layout.addWidget(self.table, stretch=1)

        for row_index, element in enumerate(self._elements):
            element_item = QTableWidgetItem(element)
            element_item.setFlags(
                Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
            )
            self.table.setItem(row_index, 0, element_item)

            color_button = QPushButton()
            color_button.clicked.connect(
                lambda _checked=False, row=row_index: self._choose_row_color(row)
            )
            self.table.setCellWidget(row_index, 1, color_button)

            scale_spin = QDoubleSpinBox(self.table)
            scale_spin.setDecimals(3)
            scale_spin.setRange(0.05, 4.0)
            scale_spin.setSingleStep(0.02)
            scale_spin.setAlignment(Qt.AlignmentFlag.AlignRight)
            scale_spin.valueChanged.connect(
                lambda _value, row=row_index: self._update_radius_cell(row)
            )
            self.table.setCellWidget(row_index, 2, scale_spin)

            radius_item = QTableWidgetItem()
            radius_item.setFlags(
                Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
            )
            radius_item.setTextAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
            self.table.setItem(row_index, 3, radius_item)

        self._apply_style_values(self._active_style)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        save_button = buttons.button(QDialogButtonBox.StandardButton.Ok)
        if save_button is not None:
            save_button.setText("Save Custom Aesthetic")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _row_element(self, row_index: int) -> str:
        item = self.table.item(row_index, 0)
        return item.text().strip() if item is not None else "X"

    def _row_color_button(self, row_index: int) -> QPushButton | None:
        widget = self.table.cellWidget(row_index, 1)
        return widget if isinstance(widget, QPushButton) else None

    def _row_scale_spin(self, row_index: int) -> QDoubleSpinBox | None:
        widget = self.table.cellWidget(row_index, 2)
        return widget if isinstance(widget, QDoubleSpinBox) else None

    def _row_rgba(
        self,
        row_index: int,
    ) -> tuple[float, float, float, float]:
        button = self._row_color_button(row_index)
        data = button.property("rgba") if button is not None else None
        if isinstance(data, tuple) and len(data) == 4:
            return tuple(float(channel) for channel in data)
        element = self._row_element(row_index)
        return style_atom_color(element, atom_style=self._active_style)

    def _set_row_color(
        self,
        row_index: int,
        rgba: tuple[float, float, float, float],
    ) -> None:
        button = self._row_color_button(row_index)
        if button is None:
            return
        hex_value = _color_to_hex(rgba)
        text_color = "#111827"
        if (
            (rgba[0] * 0.299 + rgba[1] * 0.587 + rgba[2] * 0.114)
            < 0.52
        ):
            text_color = "#f9fafb"
        button.setProperty("rgba", rgba)
        button.setText(hex_value)
        button.setStyleSheet(
            "background-color: "
            f"{hex_value}; "
            f"color: {text_color}; "
            "border: 1px solid #aeb7c2; padding: 4px 8px;"
        )

    def _update_radius_cell(self, row_index: int) -> None:
        scale_spin = self._row_scale_spin(row_index)
        radius_item = self.table.item(row_index, 3)
        if scale_spin is None or radius_item is None:
            return
        element = self._row_element(row_index)
        radius = max(
            COVALENT_RADII.get(element, 0.85) * float(scale_spin.value()),
            0.18,
        )
        radius_item.setText(f"{radius:.3f}")

    def _apply_style_values(self, atom_style: str) -> None:
        style = normalize_atom_style(atom_style)
        for row_index, element in enumerate(self._elements):
            self._set_row_color(
                row_index,
                style_atom_color(element, atom_style=style),
            )
            scale_spin = self._row_scale_spin(row_index)
            if scale_spin is not None:
                scale_spin.setValue(
                    style_atom_size_scale(element, atom_style=style)
                )
            self._update_radius_cell(row_index)

    @Slot()
    def _apply_selected_preset(self) -> None:
        preset = normalize_builtin_atom_style(
            str(self.seed_preset_combo.currentData())
        )
        self._apply_style_values(preset)

    @Slot()
    def _reset_to_active_style(self) -> None:
        self._apply_style_values(self._active_style)
        self.seed_preset_combo.setCurrentIndex(
            max(
                self.seed_preset_combo.findData(self._active_base_style),
                0,
            )
        )

    def _choose_row_color(self, row_index: int) -> None:
        current = _rgba_to_qcolor(self._row_rgba(row_index))
        color = QColorDialog.getColor(
            current,
            self,
            f"Select {self._row_element(row_index)} Color",
        )
        if not color.isValid():
            return
        self._set_row_color(row_index, _qcolor_to_rgba(color))

    def custom_aesthetic(
        self,
        *,
        existing_specs: tuple[CustomAestheticSpec, ...] | list[CustomAestheticSpec],
    ) -> CustomAestheticSpec:
        name = self.name_edit.text().strip()
        if not name:
            raise ValueError("Enter a name for the custom aesthetic.")
        key = (
            self._active_custom.key
            if self._active_custom is not None
            else sanitize_custom_aesthetic_key(name)
        )
        existing_by_name = {
            spec.name.strip().lower(): spec.key
            for spec in existing_specs
        }
        existing_key = existing_by_name.get(name.lower())
        if existing_key is not None and existing_key != key:
            raise ValueError(
                "Choose a different custom aesthetic name. That name is already in use."
            )
        base_style = normalize_builtin_atom_style(
            str(self.seed_preset_combo.currentData())
        )
        overrides = tuple(
            AtomAppearanceOverride(
                element=self._row_element(row_index),
                color=self._row_rgba(row_index),
                size_scale=float(self._row_scale_spin(row_index).value()),
            )
            for row_index in range(self.table.rowCount())
            if self._row_scale_spin(row_index) is not None
        )
        return CustomAestheticSpec(
            key=key,
            name=name,
            base_style=base_style,
            overrides=overrides,
        )


class BlenderXYZRendererMainWindow(QMainWindow):
    """Launcher window for batch Blender XYZ rendering."""

    def __init__(
        self,
        initial_input_path: str | Path | None = None,
        *,
        initial_blender_executable: str | Path | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._settings_store = QSettings()
        self._custom_aesthetics_by_key = self._load_custom_aesthetics_from_settings()
        self._run_thread: QThread | None = None
        self._run_worker: BlenderXYZRenderWorker | None = None
        self._preview_structure: BlenderPreviewStructure | None = None
        self._bond_thresholds: tuple[BondThresholdSpec, ...] = ()
        self._custom_orientations: list[OrientationSpec] = []
        self._last_suggested_title = ""
        self._orientation_table_updating = False
        self.orientation_preview: OrientationPreviewWidget | None = None
        self.orientation_controls_group: QGroupBox | None = None
        self._build_ui()
        self._load_persistent_state(initial_blender_executable)
        if initial_input_path is not None:
            self.set_input_path(initial_input_path)
        self._refresh_blender_hint()

    def closeEvent(self, event) -> None:
        if self._run_thread is not None and self._run_thread.isRunning():
            QMessageBox.warning(
                self,
                "Blender Structure Renderer",
                "Please wait for the current Blender render job to finish "
                "before closing this window.",
            )
            event.ignore()
            return
        super().closeEvent(event)

    def _load_custom_aesthetics_from_settings(
        self,
    ) -> dict[str, CustomAestheticSpec]:
        raw_value = self._settings_store.value(_CUSTOM_AESTHETICS_KEY, "[]")
        if isinstance(raw_value, (list, tuple)):
            payload_text = json.dumps(list(raw_value))
        else:
            payload_text = str(raw_value or "[]")
        try:
            payload = json.loads(payload_text)
        except Exception:
            payload = []
        specs: list[CustomAestheticSpec] = []
        if isinstance(payload, list):
            for item in payload:
                try:
                    specs.append(deserialize_custom_aesthetic(item))
                except Exception:
                    continue
        set_custom_aesthetics(tuple(specs))
        return {spec.key: spec for spec in specs}

    def _persist_custom_aesthetics(self) -> None:
        specs = tuple(
            self._custom_aesthetics_by_key[key]
            for key in sorted(
                self._custom_aesthetics_by_key,
                key=lambda item: (
                    self._custom_aesthetics_by_key[item].name.lower(),
                    item,
                ),
            )
        )
        set_custom_aesthetics(specs)
        self._settings_store.setValue(
            _CUSTOM_AESTHETICS_KEY,
            json.dumps(
                [serialize_custom_aesthetic(spec) for spec in specs],
                sort_keys=True,
            ),
        )

    def _saved_bond_threshold_map(self) -> dict[str, list[str]]:
        raw_value = self._settings_store.value(
            _BOND_THRESHOLD_OVERRIDES_KEY,
            "{}",
        )
        payload_text = (
            json.dumps(list(raw_value))
            if isinstance(raw_value, (list, tuple))
            else str(raw_value or "{}")
        )
        try:
            payload = json.loads(payload_text)
        except Exception:
            payload = {}
        if not isinstance(payload, dict):
            return {}
        normalized: dict[str, list[str]] = {}
        for path_text, encoded in payload.items():
            if not isinstance(path_text, str):
                continue
            if isinstance(encoded, str):
                values = [encoded] if encoded.strip() else []
            elif isinstance(encoded, list):
                values = [str(item) for item in encoded if str(item).strip()]
            else:
                values = []
            normalized[path_text] = values
        return normalized

    def _persist_bond_threshold_map(
        self,
        data: dict[str, list[str]],
    ) -> None:
        self._settings_store.setValue(
            _BOND_THRESHOLD_OVERRIDES_KEY,
            json.dumps(data, sort_keys=True),
        )

    def _saved_bond_thresholds_for_path(
        self,
        input_path: str | Path,
    ) -> tuple[BondThresholdSpec, ...]:
        resolved = str(Path(input_path).expanduser().resolve())
        encoded = self._saved_bond_threshold_map().get(resolved, [])
        specs: list[BondThresholdSpec] = []
        for item in encoded:
            try:
                specs.append(parse_bond_threshold_arg(item))
            except Exception:
                continue
        return tuple(specs)

    def _persist_bond_thresholds_for_path(
        self,
        input_path: str | Path,
        thresholds: tuple[BondThresholdSpec, ...],
        *,
        default_thresholds: tuple[BondThresholdSpec, ...],
    ) -> None:
        resolved = str(Path(input_path).expanduser().resolve())
        mapping = self._saved_bond_threshold_map()
        if tuple(thresholds) == tuple(default_thresholds):
            mapping.pop(resolved, None)
        else:
            mapping[resolved] = [
                encode_bond_threshold_arg(spec) for spec in thresholds
            ]
        self._persist_bond_threshold_map(mapping)

    def _build_ui(self) -> None:
        self.setWindowTitle("SAXSShell (Blender Structure Renderer)")
        self.setWindowIcon(load_saxshell_icon())
        self.resize(1340, 960)

        central = QWidget(self)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(10)

        top_splitter = QSplitter(Qt.Orientation.Horizontal)
        top_splitter.setChildrenCollapsible(False)
        right_pane = self._build_right_orientation_pane()
        left_pane = self._build_left_controls_pane()
        top_splitter.addWidget(left_pane)
        top_splitter.addWidget(right_pane)
        top_splitter.setStretchFactor(0, 1)
        top_splitter.setStretchFactor(1, 2)
        top_splitter.setSizes([380, 960])

        content_splitter = QSplitter(Qt.Orientation.Vertical)
        content_splitter.setChildrenCollapsible(False)
        content_splitter.addWidget(top_splitter)
        content_splitter.addWidget(self._build_log_group())
        content_splitter.setStretchFactor(0, 5)
        content_splitter.setStretchFactor(1, 2)
        content_splitter.setSizes([760, 200])
        root_layout.addWidget(content_splitter, stretch=1)

        self.setCentralWidget(central)
        self.statusBar().showMessage("Ready")

    def _build_left_controls_pane(self) -> QScrollArea:
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        intro = QLabel(
            "Select an XYZ or PDB file, choose a destination folder, preview or edit "
            "orientations in the inline viewer, then render a batch of "
            "publication images. The renderer includes preset axis views, "
            "computed photoshoot views, custom Euler-angle orientations, and "
            "style presets tuned for higher-quality atom renders."
        )
        intro.setWordWrap(True)
        intro.setFrameShape(QFrame.Shape.StyledPanel)
        layout.addWidget(intro)
        layout.addWidget(self._build_paths_group())
        layout.addWidget(self._build_bond_thresholds_group())
        layout.addWidget(self._build_style_group())
        layout.addWidget(self._build_orientation_sources_group())
        if self.orientation_controls_group is not None:
            layout.addWidget(self.orientation_controls_group)
        layout.addWidget(self._build_run_group())
        layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        return scroll

    def _build_right_orientation_pane(self) -> QWidget:
        right_splitter = QSplitter(Qt.Orientation.Vertical)
        right_splitter.setChildrenCollapsible(False)
        right_splitter.addWidget(self._build_visualizer_group())
        right_splitter.addWidget(self._build_orientation_table_group())
        right_splitter.setStretchFactor(0, 4)
        right_splitter.setStretchFactor(1, 2)
        right_splitter.setSizes([620, 300])
        return right_splitter

    def _build_paths_group(self) -> QGroupBox:
        group = QGroupBox("Files")
        layout = QFormLayout(group)

        self.blender_edit = QLineEdit()
        self.blender_edit.textChanged.connect(
            lambda _text: self._refresh_blender_hint()
        )
        self.blender_edit.editingFinished.connect(
            self._persist_blender_executable
        )
        self.blender_browse_button = QPushButton("Browse...")
        self.blender_browse_button.clicked.connect(
            self._browse_blender_executable
        )
        blender_row = QHBoxLayout()
        blender_row.addWidget(self.blender_edit, stretch=1)
        blender_row.addWidget(self.blender_browse_button)
        layout.addRow("Blender", self._wrap_layout(blender_row))

        self.blender_hint_label = QLabel()
        self.blender_hint_label.setWordWrap(True)
        self.blender_hint_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout.addRow("", self.blender_hint_label)

        self.input_edit = QLineEdit()
        self.input_browse_button = QPushButton("Browse...")
        self.input_browse_button.clicked.connect(self._browse_input_file)
        input_row = QHBoxLayout()
        input_row.addWidget(self.input_edit, stretch=1)
        input_row.addWidget(self.input_browse_button)
        layout.addRow("Structure File", self._wrap_layout(input_row))

        self.recent_input_combo = QComboBox()
        self.recent_input_combo.activated.connect(
            lambda _index: self._open_selected_recent_input()
        )
        self.open_recent_button = QPushButton("Open")
        self.open_recent_button.clicked.connect(
            self._open_selected_recent_input
        )
        self.clear_recent_button = QPushButton("Clear")
        self.clear_recent_button.clicked.connect(
            self._clear_recent_input_history
        )
        recent_row = QHBoxLayout()
        recent_row.addWidget(self.recent_input_combo, stretch=1)
        recent_row.addWidget(self.open_recent_button)
        recent_row.addWidget(self.clear_recent_button)
        layout.addRow("Recent Files", self._wrap_layout(recent_row))

        self.title_edit = QLineEdit()
        self.title_edit.setPlaceholderText(
            "Used when the title overlay checkbox is enabled"
        )
        layout.addRow("Title", self.title_edit)

        self.render_title_box = QCheckBox("Render title overlay")
        self.render_title_box.toggled.connect(self._persist_render_title)
        layout.addRow("", self.render_title_box)

        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.editingFinished.connect(self._persist_output_dir)
        self.output_dir_browse_button = QPushButton("Browse...")
        self.output_dir_browse_button.clicked.connect(self._browse_output_dir)
        output_row = QHBoxLayout()
        output_row.addWidget(self.output_dir_edit, stretch=1)
        output_row.addWidget(self.output_dir_browse_button)
        layout.addRow("Destination Folder", self._wrap_layout(output_row))
        return group

    def _build_bond_thresholds_group(self) -> QGroupBox:
        group = QGroupBox("Bond Thresholds")
        layout = QVBoxLayout(group)
        self.bond_thresholds_summary_label = QLabel()
        self.bond_thresholds_summary_label.setWordWrap(True)
        layout.addWidget(self.bond_thresholds_summary_label)
        self.edit_bond_thresholds_button = QPushButton(
            "Edit Bond Thresholds..."
        )
        self.edit_bond_thresholds_button.clicked.connect(
            self._edit_bond_thresholds
        )
        layout.addWidget(self.edit_bond_thresholds_button)
        self._refresh_bond_threshold_summary()
        return group

    def _build_style_group(self) -> QGroupBox:
        group = QGroupBox("Appearance")
        layout = QFormLayout(group)

        self.atom_style_combo = QComboBox()
        self._refresh_atom_style_combo_options()
        self.atom_style_combo.currentIndexChanged.connect(
            self._handle_atom_style_changed
        )
        layout.addRow("Default Aesthetic", self.atom_style_combo)

        aesthetic_editor_row = QHBoxLayout()
        aesthetic_editor_row.setContentsMargins(0, 0, 0, 0)
        aesthetic_editor_row.setSpacing(6)
        self.edit_aesthetic_button = QPushButton(
            "Edit Atom Colors and Sizes..."
        )
        self.edit_aesthetic_button.clicked.connect(
            self._edit_selected_aesthetic
        )
        aesthetic_editor_row.addWidget(self.edit_aesthetic_button)
        aesthetic_editor_row.addStretch(1)
        layout.addRow("Custom Aesthetic", self._wrap_layout(aesthetic_editor_row))

        self.render_quality_combo = QComboBox()
        for key, label in RENDER_QUALITY_LABELS.items():
            self.render_quality_combo.addItem(label, key)
        self.render_quality_combo.currentIndexChanged.connect(
            self._persist_render_quality
        )
        layout.addRow("Default Render Quality", self.render_quality_combo)

        self.legend_font_combo = QFontComboBox()
        self.legend_font_combo.currentFontChanged.connect(
            self._persist_legend_font
        )
        layout.addRow("Legend Font", self.legend_font_combo)

        preview_background_row = QHBoxLayout()
        preview_background_row.setContentsMargins(0, 0, 0, 0)
        preview_background_row.setSpacing(6)
        self.preview_background_chip = QFrame()
        self.preview_background_chip.setFrameShape(QFrame.Shape.StyledPanel)
        self.preview_background_chip.setFixedSize(24, 24)
        preview_background_row.addWidget(self.preview_background_chip)
        self.preview_background_value_label = QLabel()
        preview_background_row.addWidget(
            self.preview_background_value_label, stretch=1
        )
        self.preview_background_button = QPushButton("Choose...")
        self.preview_background_button.clicked.connect(
            self._choose_preview_background
        )
        preview_background_row.addWidget(self.preview_background_button)
        self.preview_background_reset_button = QPushButton("Use Style Default")
        self.preview_background_reset_button.clicked.connect(
            self._reset_preview_background
        )
        preview_background_row.addWidget(self.preview_background_reset_button)
        layout.addRow(
            "Preview Background",
            self._wrap_layout(preview_background_row),
        )

        reference_background_row = QHBoxLayout()
        reference_background_row.setContentsMargins(0, 0, 0, 0)
        reference_background_row.setSpacing(6)
        self.reference_background_chip = QFrame()
        self.reference_background_chip.setFrameShape(
            QFrame.Shape.StyledPanel
        )
        self.reference_background_chip.setFixedSize(24, 24)
        reference_background_row.addWidget(self.reference_background_chip)
        self.reference_background_value_label = QLabel()
        reference_background_row.addWidget(
            self.reference_background_value_label, stretch=1
        )
        self.reference_background_button = QPushButton("Choose...")
        self.reference_background_button.clicked.connect(
            self._choose_reference_background
        )
        reference_background_row.addWidget(self.reference_background_button)
        self.reference_background_reset_button = QPushButton(
            "Use White Default"
        )
        self.reference_background_reset_button.clicked.connect(
            self._reset_reference_background
        )
        reference_background_row.addWidget(
            self.reference_background_reset_button
        )
        layout.addRow(
            "Reference Background",
            self._wrap_layout(reference_background_row),
        )

        self.style_hint_label = QLabel()
        self.style_hint_label.setWordWrap(True)
        layout.addRow("", self.style_hint_label)
        return group

    def _build_orientation_sources_group(self) -> QGroupBox:
        group = QGroupBox("Orientation Sources")
        layout = QVBoxLayout(group)

        self.include_presets_box = QCheckBox("Include preset orientations")
        self.include_presets_box.toggled.connect(
            self._handle_orientation_source_changed
        )
        layout.addWidget(self.include_presets_box)
        self.include_photoshoot_box = QCheckBox(
            "Include computed photoshoot orientations"
        )
        self.include_photoshoot_box.toggled.connect(
            self._handle_orientation_source_changed
        )
        layout.addWidget(self.include_photoshoot_box)
        return group

    def _build_visualizer_group(self) -> QGroupBox:
        group = QGroupBox("Visualizer")
        layout = QVBoxLayout(group)
        self.orientation_preview = OrientationPreviewWidget()
        self.orientation_preview.setMinimumHeight(460)
        self.orientation_controls_group = (
            self.orientation_preview.detach_controls_group()
        )
        self.orientation_preview.orientationChanged.connect(
            self._handle_preview_orientation_changed
        )
        self.orientation_preview.nudge_step_spin.valueChanged.connect(
            self._persist_nudge_increment
        )
        layout.addWidget(self.orientation_preview, stretch=1)
        return group

    def _build_orientation_table_group(self) -> QGroupBox:
        group = QGroupBox("Render Orientations")
        layout = QVBoxLayout(group)
        self.orientation_table = QTableWidget(0, 10)
        self.orientation_table.setHorizontalHeaderLabels(
            [
                "Enabled",
                "Legend",
                "Name",
                "Source",
                "Lighting",
                "Aesthetic",
                "Quality",
                "X",
                "Y",
                "Z",
            ]
        )
        self.orientation_table.verticalHeader().setVisible(False)
        self.orientation_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.orientation_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.orientation_table.horizontalHeader().setSectionResizeMode(
            _ORIENTATION_ENABLED_COLUMN,
            QHeaderView.ResizeMode.ResizeToContents,
        )
        self.orientation_table.horizontalHeader().setSectionResizeMode(
            _ORIENTATION_LEGEND_COLUMN,
            QHeaderView.ResizeMode.ResizeToContents,
        )
        self.orientation_table.horizontalHeader().setSectionResizeMode(
            _ORIENTATION_NAME_COLUMN,
            QHeaderView.ResizeMode.Stretch,
        )
        self.orientation_table.horizontalHeader().setSectionResizeMode(
            _ORIENTATION_SOURCE_COLUMN,
            QHeaderView.ResizeMode.ResizeToContents,
        )
        self.orientation_table.horizontalHeader().setSectionResizeMode(
            _ORIENTATION_LIGHTING_COLUMN,
            QHeaderView.ResizeMode.ResizeToContents,
        )
        self.orientation_table.horizontalHeader().setSectionResizeMode(
            _ORIENTATION_STYLE_COLUMN,
            QHeaderView.ResizeMode.ResizeToContents,
        )
        self.orientation_table.horizontalHeader().setSectionResizeMode(
            _ORIENTATION_QUALITY_COLUMN,
            QHeaderView.ResizeMode.ResizeToContents,
        )
        for column in (
            _ORIENTATION_X_COLUMN,
            _ORIENTATION_Y_COLUMN,
            _ORIENTATION_Z_COLUMN,
        ):
            self.orientation_table.horizontalHeader().setSectionResizeMode(
                column,
                QHeaderView.ResizeMode.ResizeToContents,
            )
        self.orientation_table.itemSelectionChanged.connect(
            self._sync_preview_from_selection
        )
        self.orientation_table.itemChanged.connect(
            self._handle_orientation_table_item_changed
        )
        layout.addWidget(self.orientation_table, stretch=1)

        button_row = QHBoxLayout()
        self.add_custom_button = QPushButton("Add Custom Orientation")
        self.add_custom_button.clicked.connect(self._add_custom_orientation)
        button_row.addWidget(self.add_custom_button)
        self.duplicate_orientation_button = QPushButton("Duplicate Selected")
        self.duplicate_orientation_button.clicked.connect(
            self._duplicate_selected_orientation
        )
        button_row.addWidget(self.duplicate_orientation_button)
        self.remove_custom_button = QPushButton("Remove Custom")
        self.remove_custom_button.clicked.connect(self._remove_selected_custom)
        button_row.addWidget(self.remove_custom_button)
        self.reset_orientations_button = QPushButton("Reset Orientations")
        self.reset_orientations_button.clicked.connect(
            self._reset_orientations
        )
        button_row.addWidget(self.reset_orientations_button)
        button_row.addStretch(1)
        layout.addLayout(button_row)
        return group

    def _build_log_group(self) -> QGroupBox:
        group = QGroupBox("Log")
        layout = QVBoxLayout(group)
        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        layout.addWidget(self.log_edit, stretch=1)

        button_row = QHBoxLayout()
        button_row.addStretch(1)
        self.clear_log_button = QPushButton("Clear Log")
        self.clear_log_button.clicked.connect(self.log_edit.clear)
        button_row.addWidget(self.clear_log_button)
        layout.addLayout(button_row)
        return group

    def _build_run_group(self) -> QGroupBox:
        group = QGroupBox("Render")
        layout = QVBoxLayout(group)

        self.save_blend_files_box = QCheckBox(
            "Save a .blend file beside each rendered PNG"
        )
        self.save_blend_files_box.toggled.connect(
            self._persist_save_blend_files
        )
        layout.addWidget(self.save_blend_files_box)

        self.render_button = QPushButton("Render Selected Orientations")
        self.render_button.clicked.connect(self._start_render)
        layout.addWidget(self.render_button)

        self.render_progress_bar = QProgressBar()
        self.render_progress_bar.setRange(0, 1)
        self.render_progress_bar.setValue(0)
        self.render_progress_bar.setTextVisible(True)
        self.render_progress_bar.setFormat("Ready")
        layout.addWidget(self.render_progress_bar)
        return group

    def _wrap_layout(self, layout) -> QWidget:
        widget = QWidget()
        widget.setLayout(layout)
        return widget

    def _populate_combo_options(
        self,
        combo: QComboBox,
        options: dict[object, str],
        *,
        current_data: object | None = None,
    ) -> None:
        previous = combo.currentData() if current_data is None else current_data
        combo.blockSignals(True)
        combo.clear()
        for key, label in options.items():
            combo.addItem(label, key)
        target_index = combo.findData(previous)
        if target_index < 0:
            target_index = 0 if combo.count() > 0 else -1
        if target_index >= 0:
            combo.setCurrentIndex(target_index)
        combo.blockSignals(False)

    def _atom_style_options(self) -> dict[str, str]:
        return available_atom_style_labels()

    def _refresh_atom_style_combo_options(self) -> None:
        if not hasattr(self, "atom_style_combo"):
            return
        self._populate_combo_options(
            self.atom_style_combo,
            self._atom_style_options(),
        )

    def _refresh_orientation_style_combo_options(self) -> None:
        if not hasattr(self, "orientation_table"):
            return
        options = self._atom_style_options()
        for row_index in range(self.orientation_table.rowCount()):
            combo = self.orientation_table.cellWidget(
                row_index,
                _ORIENTATION_STYLE_COLUMN,
            )
            if isinstance(combo, QComboBox):
                current_data = combo.currentData()
                self._populate_combo_options(
                    combo,
                    options,
                    current_data=current_data,
                )

    def _refresh_all_atom_style_controls(
        self,
        *,
        selected_style: str | None = None,
    ) -> None:
        selected = normalize_atom_style(
            selected_style or str(self.atom_style_combo.currentData())
        )
        self._refresh_atom_style_combo_options()
        index = self.atom_style_combo.findData(selected)
        if index >= 0:
            self.atom_style_combo.blockSignals(True)
            self.atom_style_combo.setCurrentIndex(index)
            self.atom_style_combo.blockSignals(False)
        self._refresh_orientation_style_combo_options()

    def _set_render_progress(
        self,
        value: int,
        total: int,
        text: str,
    ) -> None:
        maximum = max(int(total), 1)
        clamped_value = max(0, min(int(value), maximum))
        self.render_progress_bar.setRange(0, maximum)
        self.render_progress_bar.setValue(clamped_value)
        self.render_progress_bar.setFormat(text)

    def _load_persistent_state(
        self,
        initial_blender_executable: str | Path | None,
    ) -> None:
        saved_output_dir = self._settings_store.value(
            _OUTPUT_DIR_KEY,
            str(suggest_output_dir()),
        )
        self.output_dir_edit.setText(str(saved_output_dir))
        saved_render_title = self._settings_store.value(
            _RENDER_TITLE_KEY,
            False,
            type=bool,
        )
        self.render_title_box.setChecked(bool(saved_render_title))
        saved_atom_style = normalize_atom_style(
            str(
                self._settings_store.value(
                    _ATOM_STYLE_KEY,
                    DEFAULT_ATOM_STYLE,
                )
            )
        )
        self.atom_style_combo.setCurrentIndex(
            max(self.atom_style_combo.findData(saved_atom_style), 0)
        )
        saved_render_quality = normalize_render_quality(
            str(
                self._settings_store.value(
                    _RENDER_QUALITY_KEY,
                    DEFAULT_RENDER_QUALITY,
                )
            )
        )
        self.render_quality_combo.setCurrentIndex(
            max(self.render_quality_combo.findData(saved_render_quality), 0)
        )
        saved_legend_font = (
            str(
                self._settings_store.value(
                    _LEGEND_FONT_KEY,
                    DEFAULT_LEGEND_FONT,
                )
            ).strip()
            or DEFAULT_LEGEND_FONT
        )
        self.legend_font_combo.setCurrentFont(QFont(saved_legend_font))
        self._refresh_style_hint()
        self.save_blend_files_box.setChecked(
            self._settings_store.value(
                _SAVE_BLEND_FILES_KEY,
                False,
                type=bool,
            )
        )
        saved_preview_background = str(
            self._settings_store.value(
                _PREVIEW_BACKGROUND_KEY,
                "",
            )
        ).strip()
        self.orientation_preview.set_background_override(
            saved_preview_background or None
        )
        saved_reference_background = str(
            self._settings_store.value(
                _REFERENCE_BACKGROUND_KEY,
                REFERENCE_ATOM_BACKGROUND,
            )
        ).strip()
        self.orientation_preview.set_reference_background_color(
            saved_reference_background or REFERENCE_ATOM_BACKGROUND
        )
        saved_nudge_increment = self._settings_store.value(
            _NUDGE_INCREMENT_KEY,
            5.0,
            type=float,
        )
        self.orientation_preview.set_nudge_increment(saved_nudge_increment)
        self._refresh_preview_background_controls()
        self._refresh_reference_background_controls()
        self.include_presets_box.setChecked(
            self._settings_store.value(
                _INCLUDE_PRESETS_KEY,
                True,
                type=bool,
            )
        )
        self.include_photoshoot_box.setChecked(
            self._settings_store.value(
                _INCLUDE_PHOTOSHOOT_KEY,
                True,
                type=bool,
            )
        )
        saved_blender = self._settings_store.value(
            _BLENDER_EXECUTABLE_KEY,
            "",
        )
        candidate = initial_blender_executable or saved_blender or None
        self.blender_edit.setText(resolve_blender_executable(candidate))
        self._refresh_recent_input_history()

    def _persist_output_dir(self) -> None:
        text = self.output_dir_edit.text().strip() or str(
            resolve_desktop_dir()
        )
        self.output_dir_edit.setText(text)
        self._settings_store.setValue(_OUTPUT_DIR_KEY, text)

    def _persist_blender_executable(self) -> None:
        text = self.blender_edit.text().strip()
        if text:
            self._settings_store.setValue(_BLENDER_EXECUTABLE_KEY, text)

    def _persist_render_title(self, checked: bool) -> None:
        self._settings_store.setValue(_RENDER_TITLE_KEY, bool(checked))

    def _persist_save_blend_files(self, checked: bool) -> None:
        self._settings_store.setValue(_SAVE_BLEND_FILES_KEY, bool(checked))

    @Slot(QFont)
    def _persist_legend_font(self, font: QFont) -> None:
        family = font.family().strip() or DEFAULT_LEGEND_FONT
        self._settings_store.setValue(_LEGEND_FONT_KEY, family)

    @Slot(int)
    def _persist_render_quality(self, _index: int) -> None:
        render_quality = normalize_render_quality(
            str(self.render_quality_combo.currentData())
        )
        self._settings_store.setValue(_RENDER_QUALITY_KEY, render_quality)
        self._refresh_style_hint()

    @Slot(float)
    def _persist_nudge_increment(self, value: float) -> None:
        self._settings_store.setValue(_NUDGE_INCREMENT_KEY, float(value))

    @Slot(int)
    def _handle_atom_style_changed(self, _index: int) -> None:
        atom_style = normalize_atom_style(
            str(self.atom_style_combo.currentData())
        )
        self._settings_store.setValue(_ATOM_STYLE_KEY, atom_style)
        self._refresh_style_hint()
        if self.orientation_table.rowCount() == 0:
            self.orientation_preview.refresh_style(atom_style)
        else:
            self._sync_preview_from_selection()
        self._refresh_preview_background_controls()
        self._refresh_reference_background_controls()

    def _refresh_style_hint(self) -> None:
        atom_style = self._selected_atom_style()
        render_quality = self._selected_render_quality()
        description = atom_style_description(atom_style)
        quality_label = RENDER_QUALITY_LABELS[render_quality]
        self.style_hint_label.setText(
            f"{description} {quality_label} render quality tunes sampling, "
            "framing, and baseline lighting automatically. New and reset rows "
            f"start at lighting level {DEFAULT_LIGHTING_LEVEL}, and each table "
            "row can override lighting, style, and quality before rendering."
        )

    def _refresh_bond_threshold_summary(self) -> None:
        if not hasattr(self, "bond_thresholds_summary_label"):
            return
        if self._preview_structure is None or not self._bond_thresholds:
            self.bond_thresholds_summary_label.setText(
                "Load a structure to edit VESTA-style minimum and maximum bond "
                "lengths for each atom-pair type."
            )
            if hasattr(self, "edit_bond_thresholds_button"):
                self.edit_bond_thresholds_button.setEnabled(False)
            return
        pair_count = len(self._bond_thresholds)
        preview_pairs = ", ".join(
            f"{spec.element_a}-{spec.element_b}"
            for spec in self._bond_thresholds[:4]
        )
        if pair_count > 4:
            preview_pairs = f"{preview_pairs}, ..."
        self.bond_thresholds_summary_label.setText(
            f"Using {pair_count} pair-specific bond rules for this structure: "
            f"{preview_pairs}. Edit minimum and maximum lengths here to match "
            "the bond drawing you want in both preview and Blender output."
        )
        if hasattr(self, "edit_bond_thresholds_button"):
            self.edit_bond_thresholds_button.setEnabled(True)

    @Slot()
    def _edit_bond_thresholds(self) -> None:
        if self._preview_structure is None:
            self._show_error(
                "Load a structure file before editing bond thresholds."
            )
            return
        default_thresholds = build_bond_thresholds_for_structure(
            self._preview_structure.atoms
        )
        dialog = BondThresholdEditorDialog(
            self._bond_thresholds or default_thresholds,
            default_thresholds=default_thresholds,
            parent=self,
        )
        if dialog.exec() != int(QDialog.DialogCode.Accepted):
            return
        self._bond_thresholds = dialog.bond_thresholds()
        self._preview_structure = load_preview_structure(
            self._preview_structure.input_path,
            pair_thresholds=self._bond_thresholds,
        )
        self._persist_bond_thresholds_for_path(
            self._preview_structure.input_path,
            self._bond_thresholds,
            default_thresholds=default_thresholds,
        )
        self._refresh_bond_threshold_summary()
        self._custom_orientations = self._extract_custom_orientations()
        self._sync_preview_from_selection()
        self.statusBar().showMessage(
            "Updated bond thresholds for the active structure."
        )

    @Slot()
    def _edit_selected_aesthetic(self) -> None:
        if self._preview_structure is None:
            self._show_error(
                "Load a structure file before editing atom colors and sizes."
            )
            return
        active_style = self._selected_atom_style()
        dialog = AtomAestheticEditorDialog(
            active_style=active_style,
            elements=tuple(_preview_structure_elements(self._preview_structure)),
            parent=self,
        )
        if dialog.exec() != int(QDialog.DialogCode.Accepted):
            return
        try:
            spec = dialog.custom_aesthetic(
                existing_specs=tuple(self._custom_aesthetics_by_key.values())
            )
        except Exception as exc:
            self._show_error(str(exc))
            return

        self._custom_aesthetics_by_key[spec.key] = spec
        self._persist_custom_aesthetics()
        self._refresh_all_atom_style_controls(selected_style=spec.key)
        self._settings_store.setValue(_ATOM_STYLE_KEY, spec.key)
        selected_index = self.atom_style_combo.findData(spec.key)
        if selected_index >= 0:
            self.atom_style_combo.setCurrentIndex(selected_index)

        row_index = self._selected_row()
        if row_index >= 0:
            row_style_combo = self.orientation_table.cellWidget(
                row_index,
                _ORIENTATION_STYLE_COLUMN,
            )
            if isinstance(row_style_combo, QComboBox):
                style_index = row_style_combo.findData(spec.key)
                if style_index >= 0:
                    row_style_combo.setCurrentIndex(style_index)
        self._refresh_style_hint()
        self.statusBar().showMessage(
            f"Saved custom aesthetic '{spec.name}'."
        )

    def _refresh_preview_background_controls(self) -> None:
        override = self.orientation_preview.background_override()
        effective = self.orientation_preview.effective_background_color()
        if override:
            label = f"{effective} (custom)"
        else:
            label = f"{effective} (style default)"
        self.preview_background_value_label.setText(label)
        self.preview_background_chip.setStyleSheet(
            "background-color: " f"{effective}; " "border: 1px solid #aeb7c2;"
        )
        self.preview_background_reset_button.setEnabled(bool(override))

    def _refresh_reference_background_controls(self) -> None:
        color_value = self.orientation_preview.reference_background_color()
        is_default = color_value == REFERENCE_ATOM_BACKGROUND
        label = (
            f"{color_value} (white default)"
            if is_default
            else f"{color_value} (custom)"
        )
        self.reference_background_value_label.setText(label)
        self.reference_background_chip.setStyleSheet(
            "background-color: "
            f"{color_value}; "
            "border: 1px solid #aeb7c2;"
        )
        self.reference_background_reset_button.setEnabled(not is_default)

    @Slot()
    def _choose_preview_background(self) -> None:
        color = QColorDialog.getColor(
            QColor(self.orientation_preview.effective_background_color()),
            self,
            "Select Preview Background",
        )
        if not color.isValid():
            return
        color_name = str(color.name())
        self.orientation_preview.set_background_override(color_name)
        self._settings_store.setValue(_PREVIEW_BACKGROUND_KEY, color_name)
        self._refresh_preview_background_controls()

    @Slot()
    def _reset_preview_background(self) -> None:
        self.orientation_preview.set_background_override(None)
        self._settings_store.remove(_PREVIEW_BACKGROUND_KEY)
        self._refresh_preview_background_controls()

    @Slot()
    def _choose_reference_background(self) -> None:
        color = QColorDialog.getColor(
            QColor(self.orientation_preview.reference_background_color()),
            self,
            "Select Reference Background",
        )
        if not color.isValid():
            return
        color_name = str(color.name())
        self.orientation_preview.set_reference_background_color(color_name)
        self._settings_store.setValue(_REFERENCE_BACKGROUND_KEY, color_name)
        self._refresh_reference_background_controls()

    @Slot()
    def _reset_reference_background(self) -> None:
        self.orientation_preview.set_reference_background_color(
            REFERENCE_ATOM_BACKGROUND
        )
        self._settings_store.remove(_REFERENCE_BACKGROUND_KEY)
        self._refresh_reference_background_controls()

    def _refresh_blender_hint(self) -> None:
        blender_text = self.blender_edit.text().strip()
        if blender_text and blender_text != "blender":
            self.blender_hint_label.setText(
                "Using the selected Blender executable."
            )
            return
        detected = resolve_blender_executable(None)
        if detected == "blender":
            self.blender_hint_label.setText(
                "Blender was not auto-detected. If rendering fails, browse "
                "to the Blender executable or .app bundle."
            )
            return
        self.blender_hint_label.setText(f"Auto-detected Blender: {detected}")

    def _recent_input_paths(self) -> list[str]:
        raw_value = self._settings_store.value(_RECENT_INPUT_FILES_KEY, [])
        if raw_value is None:
            candidates: list[str] = []
        elif isinstance(raw_value, str):
            candidates = [raw_value] if raw_value.strip() else []
        elif isinstance(raw_value, (list, tuple)):
            candidates = [str(item) for item in raw_value if str(item).strip()]
        else:
            candidates = []

        recent_paths: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            try:
                normalized = str(Path(candidate).expanduser().resolve())
            except Exception:
                normalized = str(Path(candidate).expanduser())
            if normalized in seen:
                continue
            if not Path(normalized).is_file():
                continue
            seen.add(normalized)
            recent_paths.append(normalized)
        return recent_paths[:_MAX_RECENT_INPUT_FILES]

    def _refresh_recent_input_history(self) -> None:
        recent_paths = self._recent_input_paths()
        self._settings_store.setValue(_RECENT_INPUT_FILES_KEY, recent_paths)
        self.recent_input_combo.blockSignals(True)
        self.recent_input_combo.clear()
        if recent_paths:
            for path in recent_paths:
                self.recent_input_combo.addItem(path, path)
            self.recent_input_combo.setCurrentIndex(0)
        else:
            self.recent_input_combo.addItem("No recent structure files", None)
            self.recent_input_combo.setCurrentIndex(0)
        self.recent_input_combo.blockSignals(False)

        has_recent = bool(recent_paths)
        self.recent_input_combo.setEnabled(has_recent)
        self.open_recent_button.setEnabled(has_recent)
        self.clear_recent_button.setEnabled(has_recent)

    def _remember_recent_input_path(self, input_path: str | Path) -> None:
        normalized = str(Path(input_path).expanduser().resolve())
        recent = [
            path for path in self._recent_input_paths() if path != normalized
        ]
        recent.insert(0, normalized)
        self._settings_store.setValue(
            _RECENT_INPUT_FILES_KEY,
            recent[:_MAX_RECENT_INPUT_FILES],
        )
        self._refresh_recent_input_history()

    @Slot()
    def _open_selected_recent_input(self) -> None:
        data = self.recent_input_combo.currentData()
        if not data:
            return
        recent_path = Path(str(data)).expanduser()
        if not recent_path.is_file():
            self._settings_store.setValue(
                _RECENT_INPUT_FILES_KEY,
                [
                    path
                    for path in self._recent_input_paths()
                    if path != str(recent_path)
                ],
            )
            self._refresh_recent_input_history()
            self._show_error(
                f"Recent structure file was not found:\n{recent_path}"
            )
            return
        self.set_input_path(recent_path)

    @Slot()
    def _clear_recent_input_history(self) -> None:
        self._settings_store.setValue(_RECENT_INPUT_FILES_KEY, [])
        self._refresh_recent_input_history()

    def _set_orientation_rows(
        self,
        orientations: list[OrientationSpec] | tuple[OrientationSpec, ...],
        *,
        select_index: int | None = None,
    ) -> None:
        self._orientation_table_updating = True
        self.orientation_table.blockSignals(True)
        self.orientation_table.setRowCount(0)
        for row_index, orientation in enumerate(orientations):
            self.orientation_table.insertRow(row_index)
            self._set_orientation_row(row_index, orientation)

        if (
            select_index is not None
            and 0 <= select_index < self.orientation_table.rowCount()
        ):
            self.orientation_table.selectRow(select_index)
        elif self.orientation_table.rowCount() > 0:
            self.orientation_table.selectRow(0)
        self.orientation_table.blockSignals(False)
        self._orientation_table_updating = False
        self._sync_preview_from_selection()

    def _orientation_with_table_defaults(
        self,
        orientation: OrientationSpec,
    ) -> OrientationSpec:
        return OrientationSpec(
            key=orientation.key,
            label=orientation.label,
            source=orientation.source,
            x_degrees=orientation.x_degrees,
            y_degrees=orientation.y_degrees,
            z_degrees=orientation.z_degrees,
            enabled=orientation.enabled,
            atom_style=orientation.effective_atom_style(
                self._selected_atom_style()
            ),
            render_quality=orientation.effective_render_quality(
                self._selected_render_quality()
            ),
            lighting_level=orientation.effective_lighting_level(
                DEFAULT_LIGHTING_LEVEL
            ),
            save_legend=bool(orientation.save_legend),
        )

    def _selected_atom_style(self) -> str:
        return normalize_atom_style(str(self.atom_style_combo.currentData()))

    def _selected_render_quality(self) -> str:
        return normalize_render_quality(
            str(self.render_quality_combo.currentData())
        )

    def _table_combo_value(
        self,
        row_index: int,
        column: int,
        *,
        normalizer,
        fallback,
    ):
        widget = self.orientation_table.cellWidget(row_index, column)
        if not isinstance(widget, QComboBox):
            return fallback
        return normalizer(widget.currentData())

    def _orientation_from_row(self, row_index: int) -> OrientationSpec | None:
        if not (0 <= row_index < self.orientation_table.rowCount()):
            return None
        enabled_item = self.orientation_table.item(
            row_index, _ORIENTATION_ENABLED_COLUMN
        )
        legend_item = self.orientation_table.item(
            row_index, _ORIENTATION_LEGEND_COLUMN
        )
        name_item = self.orientation_table.item(
            row_index, _ORIENTATION_NAME_COLUMN
        )
        source_item = self.orientation_table.item(
            row_index, _ORIENTATION_SOURCE_COLUMN
        )
        x_item = self.orientation_table.item(row_index, _ORIENTATION_X_COLUMN)
        y_item = self.orientation_table.item(row_index, _ORIENTATION_Y_COLUMN)
        z_item = self.orientation_table.item(row_index, _ORIENTATION_Z_COLUMN)
        if (
            enabled_item is None
            or legend_item is None
            or name_item is None
            or source_item is None
            or x_item is None
            or y_item is None
            or z_item is None
        ):
            return None
        label = name_item.text().strip() or f"Orientation {row_index + 1}"
        source = source_item.text().strip() or "custom"
        stored_key = name_item.data(Qt.ItemDataRole.UserRole)
        key = sanitize_orientation_key(str(stored_key or label))
        return OrientationSpec(
            key=key,
            label=label,
            source=source,
            x_degrees=float(x_item.text()),
            y_degrees=float(y_item.text()),
            z_degrees=float(z_item.text()),
            enabled=enabled_item.checkState() == Qt.CheckState.Checked,
            atom_style=self._table_combo_value(
                row_index,
                _ORIENTATION_STYLE_COLUMN,
                normalizer=normalize_atom_style,
                fallback=self._selected_atom_style(),
            ),
            render_quality=self._table_combo_value(
                row_index,
                _ORIENTATION_QUALITY_COLUMN,
                normalizer=normalize_render_quality,
                fallback=self._selected_render_quality(),
            ),
            lighting_level=self._table_combo_value(
                row_index,
                _ORIENTATION_LIGHTING_COLUMN,
                normalizer=normalize_lighting_level,
                fallback=DEFAULT_LIGHTING_LEVEL,
            ),
            save_legend=legend_item.checkState() == Qt.CheckState.Checked,
        )

    def _create_orientation_table_combo(
        self,
        *,
        row_index: int,
        column: int,
        options: dict[object, str],
    ) -> QComboBox:
        combo = QComboBox(self.orientation_table)
        for key, label in options.items():
            combo.addItem(label, key)
        combo.currentIndexChanged.connect(
            lambda _index, row=row_index: self._handle_orientation_table_combo_changed(
                row
            )
        )
        self.orientation_table.setCellWidget(row_index, column, combo)
        return combo

    def _set_orientation_row(
        self,
        row_index: int,
        orientation: OrientationSpec,
    ) -> None:
        orientation = self._orientation_with_table_defaults(orientation)
        enabled_item = self.orientation_table.item(
            row_index, _ORIENTATION_ENABLED_COLUMN
        )
        if enabled_item is None:
            enabled_item = QTableWidgetItem()
            enabled_item.setFlags(
                Qt.ItemFlag.ItemIsEnabled
                | Qt.ItemFlag.ItemIsSelectable
                | Qt.ItemFlag.ItemIsUserCheckable
            )
            self.orientation_table.setItem(
                row_index,
                _ORIENTATION_ENABLED_COLUMN,
                enabled_item,
            )
        enabled_item.setCheckState(
            Qt.CheckState.Checked
            if orientation.enabled
            else Qt.CheckState.Unchecked
        )

        legend_item = self.orientation_table.item(
            row_index, _ORIENTATION_LEGEND_COLUMN
        )
        if legend_item is None:
            legend_item = QTableWidgetItem()
            legend_item.setFlags(
                Qt.ItemFlag.ItemIsEnabled
                | Qt.ItemFlag.ItemIsSelectable
                | Qt.ItemFlag.ItemIsUserCheckable
            )
            self.orientation_table.setItem(
                row_index,
                _ORIENTATION_LEGEND_COLUMN,
                legend_item,
            )
        legend_item.setCheckState(
            Qt.CheckState.Checked
            if orientation.save_legend
            else Qt.CheckState.Unchecked
        )

        name_item = self.orientation_table.item(
            row_index, _ORIENTATION_NAME_COLUMN
        )
        if name_item is None:
            name_item = QTableWidgetItem()
            name_item.setFlags(
                Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
            )
            self.orientation_table.setItem(
                row_index,
                _ORIENTATION_NAME_COLUMN,
                name_item,
            )
        name_item.setText(orientation.label)
        name_item.setData(Qt.ItemDataRole.UserRole, orientation.key)

        source_item = self.orientation_table.item(
            row_index,
            _ORIENTATION_SOURCE_COLUMN,
        )
        if source_item is None:
            source_item = QTableWidgetItem()
            source_item.setFlags(
                Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
            )
            self.orientation_table.setItem(
                row_index,
                _ORIENTATION_SOURCE_COLUMN,
                source_item,
            )
        source_item.setText(orientation.source)

        lighting_combo = self.orientation_table.cellWidget(
            row_index,
            _ORIENTATION_LIGHTING_COLUMN,
        )
        if not isinstance(lighting_combo, QComboBox):
            lighting_combo = self._create_orientation_table_combo(
                row_index=row_index,
                column=_ORIENTATION_LIGHTING_COLUMN,
                options=_LIGHTING_LEVEL_TABLE_LABELS,
            )
        lighting_index = max(
            lighting_combo.findData(
                orientation.effective_lighting_level(DEFAULT_LIGHTING_LEVEL)
            ),
            0,
        )
        lighting_combo.blockSignals(True)
        lighting_combo.setCurrentIndex(lighting_index)
        lighting_combo.blockSignals(False)

        style_combo = self.orientation_table.cellWidget(
            row_index,
            _ORIENTATION_STYLE_COLUMN,
        )
        if not isinstance(style_combo, QComboBox):
            style_combo = self._create_orientation_table_combo(
                row_index=row_index,
                column=_ORIENTATION_STYLE_COLUMN,
                options=self._atom_style_options(),
            )
        style_index = max(
            style_combo.findData(orientation.effective_atom_style()),
            0,
        )
        style_combo.blockSignals(True)
        style_combo.setCurrentIndex(style_index)
        style_combo.blockSignals(False)

        quality_combo = self.orientation_table.cellWidget(
            row_index,
            _ORIENTATION_QUALITY_COLUMN,
        )
        if not isinstance(quality_combo, QComboBox):
            quality_combo = self._create_orientation_table_combo(
                row_index=row_index,
                column=_ORIENTATION_QUALITY_COLUMN,
                options=RENDER_QUALITY_LABELS,
            )
        quality_index = max(
            quality_combo.findData(orientation.effective_render_quality()),
            0,
        )
        quality_combo.blockSignals(True)
        quality_combo.setCurrentIndex(quality_index)
        quality_combo.blockSignals(False)

        for column, value in zip(
            (
                _ORIENTATION_X_COLUMN,
                _ORIENTATION_Y_COLUMN,
                _ORIENTATION_Z_COLUMN,
            ),
            orientation.euler_degrees,
        ):
            angle_item = self.orientation_table.item(row_index, column)
            if angle_item is None:
                angle_item = QTableWidgetItem()
                angle_item.setFlags(
                    Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
                )
                angle_item.setTextAlignment(
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                )
                self.orientation_table.setItem(row_index, column, angle_item)
            angle_item.setText(f"{value:.1f}")

    def _extract_orientations_from_table(self) -> list[OrientationSpec]:
        orientations: list[OrientationSpec] = []
        for row_index in range(self.orientation_table.rowCount()):
            orientation = self._orientation_from_row(row_index)
            if orientation is None:
                continue
            if orientation.source == "custom":
                orientation = OrientationSpec(
                    key=sanitize_orientation_key(orientation.label),
                    label=orientation.label,
                    source=orientation.source,
                    x_degrees=orientation.x_degrees,
                    y_degrees=orientation.y_degrees,
                    z_degrees=orientation.z_degrees,
                    enabled=orientation.enabled,
                    atom_style=orientation.atom_style,
                    render_quality=orientation.render_quality,
                    lighting_level=orientation.lighting_level,
                    save_legend=orientation.save_legend,
                )
            orientations.append(orientation)
        return orientations

    def _extract_custom_orientations(self) -> list[OrientationSpec]:
        return [
            orientation
            for orientation in self._extract_orientations_from_table()
            if orientation.source == "custom"
        ]

    def _selected_row(self) -> int:
        selected_rows = self.orientation_table.selectionModel().selectedRows()
        if not selected_rows:
            return -1
        return int(selected_rows[0].row())

    @Slot()
    def _sync_preview_from_selection(self) -> None:
        row_index = self._selected_row()
        if (
            self._preview_structure is None
            or row_index < 0
            or self.orientation_table.rowCount() == 0
        ):
            self.orientation_preview.clear_preview()
            return
        orientation = self._orientation_from_row(row_index)
        if orientation is None:
            self.orientation_preview.clear_preview()
            return
        self.orientation_preview.set_preview(
            self._preview_structure,
            orientation,
            row_index=row_index,
            row_count=self.orientation_table.rowCount(),
            atom_style=orientation.effective_atom_style(
                self._selected_atom_style()
            ),
        )

    @Slot(object)
    def _handle_preview_orientation_changed(
        self,
        orientation: OrientationSpec,
    ) -> None:
        row_index = self._selected_row()
        if row_index < 0:
            return
        existing = self._orientation_from_row(row_index)
        if existing is None:
            return

        promoted_to_custom = existing.source != "custom" and (
            existing.label != orientation.label
            or existing.x_degrees != orientation.x_degrees
            or existing.y_degrees != orientation.y_degrees
            or existing.z_degrees != orientation.z_degrees
        )
        if promoted_to_custom:
            orientation = OrientationSpec(
                key=sanitize_orientation_key(f"{orientation.label}_custom"),
                label=orientation.label,
                source="custom",
                x_degrees=orientation.x_degrees,
                y_degrees=orientation.y_degrees,
                z_degrees=orientation.z_degrees,
                enabled=orientation.enabled,
                atom_style=existing.atom_style,
                render_quality=existing.render_quality,
                lighting_level=existing.lighting_level,
                save_legend=existing.save_legend,
            )
            self.statusBar().showMessage(
                "Edited preset orientation was converted into a custom orientation."
            )
        elif existing.source != "custom":
            orientation = OrientationSpec(
                key=existing.key,
                label=existing.label,
                source=existing.source,
                x_degrees=orientation.x_degrees,
                y_degrees=orientation.y_degrees,
                z_degrees=orientation.z_degrees,
                enabled=orientation.enabled,
                atom_style=existing.atom_style,
                render_quality=existing.render_quality,
                lighting_level=existing.lighting_level,
                save_legend=existing.save_legend,
            )
        else:
            orientation = OrientationSpec(
                key=sanitize_orientation_key(orientation.label),
                label=orientation.label,
                source="custom",
                x_degrees=orientation.x_degrees,
                y_degrees=orientation.y_degrees,
                z_degrees=orientation.z_degrees,
                enabled=orientation.enabled,
                atom_style=existing.atom_style,
                render_quality=existing.render_quality,
                lighting_level=existing.lighting_level,
                save_legend=existing.save_legend,
            )

        self._orientation_table_updating = True
        self.orientation_table.blockSignals(True)
        self._set_orientation_row(row_index, orientation)
        self.orientation_table.blockSignals(False)
        self._orientation_table_updating = False
        self._custom_orientations = self._extract_custom_orientations()
        self._sync_preview_from_selection()

    @Slot(object)
    def _handle_orientation_table_item_changed(
        self,
        item: QTableWidgetItem,
    ) -> None:
        if self._orientation_table_updating:
            return
        if item.column() in {
            _ORIENTATION_ENABLED_COLUMN,
            _ORIENTATION_LEGEND_COLUMN,
        }:
            self._custom_orientations = self._extract_custom_orientations()
        if item.row() == self._selected_row():
            self._sync_preview_from_selection()

    @Slot(int)
    def _handle_orientation_table_combo_changed(
        self,
        row_index: int,
    ) -> None:
        if self._orientation_table_updating:
            return
        if self._selected_row() != row_index:
            self.orientation_table.selectRow(row_index)
        self._custom_orientations = self._extract_custom_orientations()
        self._sync_preview_from_selection()

    def _rebuild_orientations(
        self, *, select_index: int | None = None
    ) -> None:
        if self._preview_structure is None:
            self.orientation_table.setRowCount(0)
            self.orientation_preview.clear_preview()
            return
        generated = list(
            build_default_orientation_catalog(
                self._preview_structure,
                include_presets=self.include_presets_box.isChecked(),
                include_photoshoot=self.include_photoshoot_box.isChecked(),
            )
        )
        generated = [
            self._orientation_with_table_defaults(orientation)
            for orientation in generated
        ]
        generated.extend(
            self._orientation_with_table_defaults(orientation)
            for orientation in self._custom_orientations
        )
        self._set_orientation_rows(generated, select_index=select_index)

    def set_input_path(self, path: str | Path) -> None:
        input_path = Path(path).expanduser().resolve()
        self.input_edit.setText(str(input_path))

        try:
            saved_thresholds = self._saved_bond_thresholds_for_path(input_path)
            self._preview_structure = load_preview_structure(
                input_path,
                pair_thresholds=saved_thresholds or None,
            )
        except Exception as exc:
            self._preview_structure = None
            self._bond_thresholds = ()
            self.orientation_preview.clear_preview()
            self._refresh_bond_threshold_summary()
            self._show_error(str(exc))
            return

        self._bond_thresholds = self._preview_structure.bond_thresholds
        self._refresh_bond_threshold_summary()

        try:
            comment = read_structure_comment(input_path)
        except Exception:
            comment = ""
        suggested_title = infer_title(
            input_path,
            structure_comment=comment,
            explicit_title=None,
        )
        current_title = self.title_edit.text().strip()
        if not current_title or current_title == self._last_suggested_title:
            self.title_edit.setText(suggested_title)
        self._last_suggested_title = suggested_title

        self._custom_orientations = []
        self._rebuild_orientations(select_index=0)
        self._remember_recent_input_path(input_path)
        self.statusBar().showMessage(
            f"Loaded {input_path.name} with "
            f"{len(self._preview_structure.atoms)} atoms"
        )

    @Slot()
    def _browse_blender_executable(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Select Blender Executable",
            str(Path.home()),
        )
        if selected:
            self.blender_edit.setText(selected)
            self._persist_blender_executable()
            self._refresh_blender_hint()

    @Slot()
    def _browse_input_file(self) -> None:
        start_dir = self.input_edit.text().strip() or str(Path.home())
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Select Structure File",
            start_dir,
            "Structure files (*.xyz *.pdb *.ent);;XYZ files (*.xyz);;PDB files (*.pdb *.ent);;All files (*)",
        )
        if selected:
            self.set_input_path(selected)

    @Slot()
    def _browse_output_dir(self) -> None:
        start_dir = self.output_dir_edit.text().strip() or str(
            resolve_desktop_dir()
        )
        selected = QFileDialog.getExistingDirectory(
            self,
            "Select Destination Folder",
            start_dir,
        )
        if selected:
            self.output_dir_edit.setText(str(Path(selected).expanduser()))
            self._persist_output_dir()

    @Slot(bool)
    def _handle_orientation_source_changed(self, _checked: bool) -> None:
        self._settings_store.setValue(
            _INCLUDE_PRESETS_KEY,
            self.include_presets_box.isChecked(),
        )
        self._settings_store.setValue(
            _INCLUDE_PHOTOSHOOT_KEY,
            self.include_photoshoot_box.isChecked(),
        )
        self._custom_orientations = self._extract_custom_orientations()
        self._rebuild_orientations(select_index=self._selected_row())

    def _apply_orientation_list(
        self,
        orientations: list[OrientationSpec],
        *,
        select_index: int | None = None,
    ) -> None:
        self._custom_orientations = [
            self._orientation_with_table_defaults(orientation)
            for orientation in orientations
            if orientation.source == "custom"
        ]
        self._set_orientation_rows(
            [
                self._orientation_with_table_defaults(orientation)
                for orientation in orientations
            ],
            select_index=select_index,
        )

    @Slot()
    def _add_custom_orientation(self) -> None:
        if self._preview_structure is None:
            self._show_error(
                "Select a structure file before adding orientations."
            )
            return
        orientations = self._extract_orientations_from_table()
        if orientations:
            base = orientations[max(self._selected_row(), 0)]
        else:
            base = OrientationSpec(
                key="custom_01",
                label="Custom 1",
                source="custom",
                x_degrees=0.0,
                y_degrees=0.0,
                z_degrees=0.0,
                enabled=True,
                atom_style=self._selected_atom_style(),
                render_quality=self._selected_render_quality(),
                lighting_level=DEFAULT_LIGHTING_LEVEL,
                save_legend=False,
            )
        custom_count = 1 + sum(
            1 for orientation in orientations if orientation.source == "custom"
        )
        custom = OrientationSpec(
            key=f"custom_{custom_count:02d}",
            label=f"Custom {custom_count}",
            source="custom",
            x_degrees=base.x_degrees,
            y_degrees=base.y_degrees,
            z_degrees=base.z_degrees,
            enabled=True,
            atom_style=base.effective_atom_style(self._selected_atom_style()),
            render_quality=base.effective_render_quality(
                self._selected_render_quality()
            ),
            lighting_level=base.effective_lighting_level(
                DEFAULT_LIGHTING_LEVEL
            ),
            save_legend=base.save_legend,
        )
        orientations.append(custom)
        self._apply_orientation_list(
            orientations,
            select_index=len(orientations) - 1,
        )
        self.statusBar().showMessage(
            "Added a custom orientation. Tune it with the inline preview controls."
        )

    def _next_duplicate_label(
        self,
        base_label: str,
        orientations: list[OrientationSpec],
    ) -> str:
        base_duplicate_label = f"{base_label} Copy"
        existing_labels = {orientation.label for orientation in orientations}
        if base_duplicate_label not in existing_labels:
            return base_duplicate_label
        suffix = 2
        while f"{base_duplicate_label} {suffix}" in existing_labels:
            suffix += 1
        return f"{base_duplicate_label} {suffix}"

    @Slot()
    def _duplicate_selected_orientation(self) -> None:
        row_index = self._selected_row()
        if row_index < 0:
            self.statusBar().showMessage(
                "Select an orientation row before duplicating it."
            )
            return
        orientations = self._extract_orientations_from_table()
        if row_index >= len(orientations):
            return
        base = orientations[row_index]
        duplicate_label = self._next_duplicate_label(base.label, orientations)
        duplicate = OrientationSpec(
            key=sanitize_orientation_key(duplicate_label),
            label=duplicate_label,
            source="custom",
            x_degrees=base.x_degrees,
            y_degrees=base.y_degrees,
            z_degrees=base.z_degrees,
            enabled=base.enabled,
            atom_style=base.effective_atom_style(self._selected_atom_style()),
            render_quality=base.effective_render_quality(
                self._selected_render_quality()
            ),
            lighting_level=base.effective_lighting_level(
                DEFAULT_LIGHTING_LEVEL
            ),
            save_legend=base.save_legend,
        )
        orientations.insert(row_index + 1, duplicate)
        self._apply_orientation_list(
            orientations,
            select_index=row_index + 1,
        )
        self.statusBar().showMessage(
            "Duplicated the selected orientation. Adjust its style, quality, or angles as needed."
        )

    @Slot()
    def _remove_selected_custom(self) -> None:
        row_index = self._selected_row()
        if row_index < 0:
            return
        orientations = self._extract_orientations_from_table()
        if orientations[row_index].source != "custom":
            self.statusBar().showMessage(
                "Preset and photoshoot orientations can be disabled instead of removed."
            )
            return
        del orientations[row_index]
        self._apply_orientation_list(
            orientations,
            select_index=max(row_index - 1, 0),
        )

    @Slot()
    def _reset_orientations(self) -> None:
        self._custom_orientations = []
        self._rebuild_orientations(select_index=0)

    def _append_log(self, message: str) -> None:
        self.log_edit.append(message)

    def _collect_settings(self) -> BlenderXYZRenderSettings:
        input_text = self.input_edit.text().strip()
        if not input_text:
            raise ValueError("Select a structure input file before rendering.")

        output_dir_text = self.output_dir_edit.text().strip()
        if not output_dir_text:
            output_dir_text = str(resolve_desktop_dir())
            self.output_dir_edit.setText(output_dir_text)
        self._persist_output_dir()

        orientations = tuple(self._extract_orientations_from_table())
        if not orientations:
            raise ValueError(
                "Choose at least one orientation before rendering."
            )
        if not any(orientation.enabled for orientation in orientations):
            raise ValueError(
                "Enable at least one orientation before rendering."
            )

        atom_style = self._selected_atom_style()
        render_quality = self._selected_render_quality()
        legend_font_family = (
            self.legend_font_combo.currentFont().family().strip()
            or DEFAULT_LEGEND_FONT
        )

        return BlenderXYZRenderSettings(
            input_path=Path(input_text),
            output_dir=Path(output_dir_text),
            orientations=orientations,
            title=self.title_edit.text().strip() or None,
            blender_executable=self.blender_edit.text().strip() or None,
            atom_style=atom_style,
            render_quality=render_quality,
            render_title=self.render_title_box.isChecked(),
            bond_thresholds=self._bond_thresholds,
            transparent=True,
            save_blend_file=self.save_blend_files_box.isChecked(),
            legend_font_family=legend_font_family,
            custom_aesthetics=tuple(self._custom_aesthetics_by_key.values()),
        )

    @Slot()
    def _start_render(self) -> None:
        if self._run_thread is not None:
            return
        try:
            settings = self._collect_settings()
        except Exception as exc:
            self._show_error(str(exc))
            return

        self.log_edit.clear()
        enabled_count = sum(
            1 for orientation in settings.orientations if orientation.enabled
        )
        self._set_render_progress(
            0,
            enabled_count,
            f"Queued 0/{enabled_count} renders",
        )
        self._set_running_state(True)
        self.statusBar().showMessage("Launching Blender render...")

        self._run_thread = QThread(self)
        self._run_worker = BlenderXYZRenderWorker(settings)
        self._run_worker.moveToThread(self._run_thread)
        self._run_thread.started.connect(self._run_worker.run)
        self._run_worker.log.connect(self._append_log)
        self._run_worker.progress.connect(self._handle_render_progress)
        self._run_worker.status.connect(self.statusBar().showMessage)
        self._run_worker.finished.connect(self._handle_render_finished)
        self._run_worker.failed.connect(self._handle_render_failed)
        self._run_worker.finished.connect(self._run_thread.quit)
        self._run_worker.failed.connect(self._run_thread.quit)
        self._run_thread.finished.connect(self._cleanup_run_thread)
        self._run_thread.start()

    @Slot(object)
    def _handle_render_finished(self, result: BlenderXYZRenderResult) -> None:
        self._set_render_progress(
            len(result.output_paths),
            len(result.output_paths),
            f"Completed {len(result.output_paths)}/{len(result.output_paths)} renders",
        )
        artifact_parts = [f"{len(result.output_paths)} PNG renders"]
        if result.legend_paths:
            artifact_parts.append(f"{len(result.legend_paths)} legend images")
        if result.blend_paths:
            artifact_parts.append(f"{len(result.blend_paths)} .blend files")
        artifact_text = ", ".join(artifact_parts)
        status_message = f"Saved {artifact_text} to {result.output_dir}"
        dialog_message = f"Saved {artifact_text} to:\n{result.output_dir}"
        self.statusBar().showMessage(status_message)
        QMessageBox.information(
            self,
            "Render Complete",
            dialog_message,
        )

    @Slot(str)
    def _handle_render_failed(self, message: str) -> None:
        self.render_progress_bar.setFormat("Render failed")
        self.statusBar().showMessage("Render failed")
        self._append_log(message)
        self._show_error(message)

    @Slot(int, int, str)
    def _handle_render_progress(
        self,
        value: int,
        total: int,
        text: str,
    ) -> None:
        self._set_render_progress(value, total, text)

    @Slot()
    def _cleanup_run_thread(self) -> None:
        self._set_running_state(False)
        if self._run_worker is not None:
            self._run_worker.deleteLater()
        if self._run_thread is not None:
            self._run_thread.deleteLater()
        self._run_worker = None
        self._run_thread = None

    def _set_running_state(self, running: bool) -> None:
        for widget in (
            self.blender_edit,
            self.input_edit,
            self.recent_input_combo,
            self.title_edit,
            self.output_dir_edit,
            self.atom_style_combo,
            self.edit_aesthetic_button,
            self.render_quality_combo,
            self.legend_font_combo,
            self.edit_bond_thresholds_button,
            self.preview_background_button,
            self.preview_background_reset_button,
            self.reference_background_button,
            self.reference_background_reset_button,
            self.blender_browse_button,
            self.input_browse_button,
            self.output_dir_browse_button,
            self.open_recent_button,
            self.render_title_box,
            self.save_blend_files_box,
            self.include_presets_box,
            self.include_photoshoot_box,
            self.add_custom_button,
            self.duplicate_orientation_button,
            self.remove_custom_button,
            self.reset_orientations_button,
            self.render_button,
            self.orientation_table,
            self.orientation_preview,
        ):
            widget.setEnabled(not running)
        self.clear_log_button.setEnabled(not running)
        has_recent = bool(self._recent_input_paths())
        self.recent_input_combo.setEnabled((not running) and has_recent)
        self.open_recent_button.setEnabled((not running) and has_recent)
        self.clear_recent_button.setEnabled((not running) and has_recent)

    def _show_error(self, message: str) -> None:
        QMessageBox.critical(self, "Blender Structure Renderer", message)


def launch_blender_xyz_renderer_ui(
    *,
    input_path: str | Path | None = None,
    blender_executable: str | Path | None = None,
) -> BlenderXYZRendererMainWindow:
    app = QApplication.instance()
    if app is None:
        prepare_saxshell_application_identity()
        app = QApplication(sys.argv)
    configure_saxshell_application(app)
    window = BlenderXYZRendererMainWindow(
        initial_input_path=input_path,
        initial_blender_executable=blender_executable,
    )
    window.show()
    window.raise_()
    _OPEN_WINDOWS.append(window)
    return window


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="blenderxyz",
        description=(
            "Launch a Qt window for previewing and rendering multiple "
            "orientation-aware Blender structure images to a destination folder."
        ),
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        type=Path,
        help="Optional XYZ or PDB file to prefill in the window.",
    )
    parser.add_argument(
        "--blender-executable",
        type=Path,
        help="Optional Blender executable or .app bundle to prefill.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    app = QApplication.instance()
    created_app = app is None
    if app is None:
        prepare_saxshell_application_identity()
        app = QApplication(sys.argv)
    launch_blender_xyz_renderer_ui(
        input_path=args.input_path,
        blender_executable=args.blender_executable,
    )
    if created_app:
        assert app is not None
        return int(app.exec())
    return 0


__all__ = [
    "BlenderXYZRendererMainWindow",
    "launch_blender_xyz_renderer_ui",
    "main",
]
