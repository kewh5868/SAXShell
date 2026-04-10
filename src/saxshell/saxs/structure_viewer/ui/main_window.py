from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from saxshell.saxs.electron_density_mapping.workflow import (
    ElectronDensityMeshGeometry,
    ElectronDensityMeshSettings,
    ElectronDensityStructure,
    build_electron_density_mesh,
    load_electron_density_structure,
    recenter_electron_density_structure,
)
from saxshell.saxs.ui.branding import (
    configure_saxshell_application,
    load_saxshell_icon,
    prepare_saxshell_application_identity,
)

from .widget import StructureViewerWidget

_OPEN_WINDOWS: list["StructureViewerMainWindow"] = []


class StructureViewerMainWindow(QMainWindow):
    """Standalone harness for the reusable structure-viewer widget."""

    def __init__(
        self,
        *,
        initial_input_path: str | Path | None = None,
    ) -> None:
        super().__init__()
        self._structure: ElectronDensityStructure | None = None
        self._active_mesh_settings = ElectronDensityMeshSettings(rstep=0.05)
        self._active_mesh_geometry: ElectronDensityMeshGeometry | None = None

        self.setWindowTitle("Structure Viewer")
        self.setWindowIcon(load_saxshell_icon())
        self.resize(1380, 900)
        self._build_menu_bar()
        self._build_ui()

        if initial_input_path is not None:
            self.input_path_edit.setText(
                str(Path(initial_input_path).expanduser().resolve())
            )
            self._load_input_from_edit()

    def _build_menu_bar(self) -> None:
        file_menu = self.menuBar().addMenu("File")

        self.open_structure_action = QAction("Open Structure...", self)
        self.open_structure_action.triggered.connect(self._choose_input_file)
        file_menu.addAction(self.open_structure_action)

        self.reload_structure_action = QAction(
            "Reload Current Structure", self
        )
        self.reload_structure_action.setEnabled(False)
        self.reload_structure_action.triggered.connect(
            self._reload_current_structure
        )
        file_menu.addAction(self.reload_structure_action)

        file_menu.addSeparator()
        close_action = QAction("Close", self)
        close_action.triggered.connect(self.close)
        file_menu.addAction(close_action)

    def _build_ui(self) -> None:
        central = QWidget(self)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(8)
        self.setCentralWidget(central)

        self._pane_splitter = QSplitter(Qt.Orientation.Horizontal, central)
        root_layout.addWidget(self._pane_splitter, stretch=1)

        self._left_scroll_area = QScrollArea(self)
        self._left_scroll_area.setWidgetResizable(True)
        self._left_scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self._left_scroll_area.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.setSpacing(10)
        left_layout.addWidget(self._build_input_group())
        left_layout.addWidget(self._build_mesh_group())
        left_layout.addStretch(1)
        self._left_scroll_area.setWidget(left_container)
        self._pane_splitter.addWidget(self._left_scroll_area)

        right_container = QWidget(self)
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(10, 10, 10, 10)
        right_layout.setSpacing(8)

        self.viewer_status_label = QLabel(
            "Load a single XYZ or PDB structure to exercise the isolated "
            "Structure Viewer widget."
        )
        self.viewer_status_label.setWordWrap(True)
        right_layout.addWidget(self.viewer_status_label)

        self.structure_viewer = StructureViewerWidget(right_container)
        right_layout.addWidget(self.structure_viewer, stretch=1)

        self._pane_splitter.addWidget(right_container)
        self._pane_splitter.setStretchFactor(0, 0)
        self._pane_splitter.setStretchFactor(1, 1)
        self._pane_splitter.setSizes([420, 960])

    def _build_input_group(self) -> QWidget:
        group = QGroupBox("Input")
        layout = QVBoxLayout(group)

        path_row = QHBoxLayout()
        self.input_path_edit = QLineEdit()
        self.input_path_edit.setPlaceholderText(
            "Choose one XYZ or PDB structure file"
        )
        self.input_path_edit.returnPressed.connect(self._load_input_from_edit)
        path_row.addWidget(self.input_path_edit, stretch=1)

        browse_button = QPushButton("Choose File")
        browse_button.clicked.connect(self._choose_input_file)
        path_row.addWidget(browse_button)
        layout.addLayout(path_row)

        self.load_input_button = QPushButton("Load Structure")
        self.load_input_button.clicked.connect(self._load_input_from_edit)
        layout.addWidget(self.load_input_button)

        form = QFormLayout()
        form.addRow(QLabel("Input mode:"))
        self.input_mode_value = QLabel("No structure loaded")
        self.input_mode_value.setWordWrap(True)
        form.addRow(self.input_mode_value)

        form.addRow(QLabel("Source file:"))
        self.reference_file_value = QLabel("Unavailable")
        self.reference_file_value.setWordWrap(True)
        form.addRow(self.reference_file_value)

        form.addRow(QLabel("Structure summary:"))
        self.structure_summary_value = QLabel(
            "Load a structure to populate atoms, element counts, center of "
            "mass, active center, and rmax."
        )
        self.structure_summary_value.setWordWrap(True)
        form.addRow(self.structure_summary_value)
        layout.addLayout(form)
        return group

    def _build_mesh_group(self) -> QWidget:
        group = QGroupBox("Mesh and Origin")
        layout = QFormLayout(group)

        self.rstep_spin = QDoubleSpinBox()
        self.rstep_spin.setRange(0.01, 1000.0)
        self.rstep_spin.setDecimals(4)
        self.rstep_spin.setSingleStep(0.01)
        self.rstep_spin.setValue(self._active_mesh_settings.rstep)
        self.rstep_spin.valueChanged.connect(self._refresh_mesh_notice)
        layout.addRow("rstep (A)", self.rstep_spin)

        self.theta_divisions_spin = QSpinBox()
        self.theta_divisions_spin.setRange(2, 720)
        self.theta_divisions_spin.setValue(
            self._active_mesh_settings.theta_divisions
        )
        self.theta_divisions_spin.valueChanged.connect(
            self._refresh_mesh_notice
        )
        layout.addRow("Theta divisions", self.theta_divisions_spin)

        self.phi_divisions_spin = QSpinBox()
        self.phi_divisions_spin.setRange(2, 720)
        self.phi_divisions_spin.setValue(
            self._active_mesh_settings.phi_divisions
        )
        self.phi_divisions_spin.valueChanged.connect(self._refresh_mesh_notice)
        layout.addRow("Phi divisions", self.phi_divisions_spin)

        self.rmax_spin = QDoubleSpinBox()
        self.rmax_spin.setRange(0.01, 100000.0)
        self.rmax_spin.setDecimals(4)
        self.rmax_spin.setSingleStep(0.1)
        self.rmax_spin.setValue(self._active_mesh_settings.rmax)
        self.rmax_spin.valueChanged.connect(self._refresh_mesh_notice)
        layout.addRow("rmax (A)", self.rmax_spin)

        layout.addRow(QLabel("Center mode:"))
        self.center_mode_value = QLabel("Mass-weighted center of mass")
        self.center_mode_value.setWordWrap(True)
        layout.addRow(self.center_mode_value)

        layout.addRow(QLabel("Calculated center:"))
        self.calculated_center_value = QLabel("Unavailable")
        self.calculated_center_value.setWordWrap(True)
        layout.addRow(self.calculated_center_value)

        layout.addRow(QLabel("Active center:"))
        self.active_center_value = QLabel("Unavailable")
        self.active_center_value.setWordWrap(True)
        layout.addRow(self.active_center_value)

        layout.addRow(QLabel("Nearest atom:"))
        self.nearest_atom_value = QLabel("Unavailable")
        self.nearest_atom_value.setWordWrap(True)
        layout.addRow(self.nearest_atom_value)

        self.reference_element_combo = QComboBox()
        self.reference_element_combo.setEnabled(False)
        self.reference_element_combo.currentIndexChanged.connect(
            self._handle_reference_element_changed
        )
        layout.addRow("Reference element", self.reference_element_combo)

        layout.addRow(QLabel("Total-atom geometric center:"))
        self.geometric_center_value = QLabel("Unavailable")
        self.geometric_center_value.setWordWrap(True)
        layout.addRow(self.geometric_center_value)

        layout.addRow(QLabel("Reference-element geometric center:"))
        self.reference_element_center_value = QLabel("Unavailable")
        self.reference_element_center_value.setWordWrap(True)
        layout.addRow(self.reference_element_center_value)

        layout.addRow(QLabel("Reference-center offset:"))
        self.reference_element_offset_value = QLabel("Unavailable")
        self.reference_element_offset_value.setWordWrap(True)
        layout.addRow(self.reference_element_offset_value)

        layout.addRow(QLabel("Snap center:"))
        center_button_row = QWidget(group)
        center_button_grid = QGridLayout(center_button_row)
        center_button_grid.setContentsMargins(0, 0, 0, 0)
        center_button_grid.setSpacing(4)

        self.reset_center_button = QPushButton("Calculated Center")
        self.reset_center_button.setCheckable(True)
        self.reset_center_button.clicked.connect(
            lambda _checked=False: self._apply_center_mode("center_of_mass")
        )
        center_button_grid.addWidget(self.reset_center_button, 0, 0)

        self.snap_center_button = QPushButton("Nearest Atom")
        self.snap_center_button.setCheckable(True)
        self.snap_center_button.clicked.connect(
            lambda _checked=False: self._apply_center_mode("nearest_atom")
        )
        center_button_grid.addWidget(self.snap_center_button, 0, 1)

        self.snap_reference_center_button = QPushButton("Reference Element")
        self.snap_reference_center_button.setCheckable(True)
        self.snap_reference_center_button.clicked.connect(
            lambda _checked=False: self._apply_center_mode("reference_element")
        )
        center_button_grid.addWidget(self.snap_reference_center_button, 1, 0)

        self._center_snap_button_group = QButtonGroup(group)
        self._center_snap_button_group.setExclusive(True)
        self._center_snap_button_group.addButton(self.reset_center_button)
        self._center_snap_button_group.addButton(self.snap_center_button)
        self._center_snap_button_group.addButton(
            self.snap_reference_center_button
        )
        self.reset_center_button.setChecked(True)
        layout.addRow(center_button_row)

        self.update_mesh_button = QPushButton("Update Mesh Settings")
        self.update_mesh_button.clicked.connect(self._apply_mesh_from_controls)
        layout.addRow(self.update_mesh_button)

        layout.addRow(QLabel("Active mesh:"))
        self.active_mesh_value = QLabel(
            "Load a structure to render a spherical mesh overlay."
        )
        self.active_mesh_value.setWordWrap(True)
        layout.addRow(self.active_mesh_value)

        layout.addRow(QLabel("Pending fields:"))
        self.pending_mesh_value = QLabel(
            "Pending field values match the rendered mesh."
        )
        self.pending_mesh_value.setWordWrap(True)
        layout.addRow(self.pending_mesh_value)
        return group

    @Slot()
    def _choose_input_file(self) -> None:
        start_dir = str(self._suggest_input_dir())
        selected_path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "Choose Structure File",
            start_dir,
            "Structure Files (*.xyz *.pdb)",
        )
        if not selected_path:
            return
        self.input_path_edit.setText(selected_path)
        self._load_input_from_edit()

    def _suggest_input_dir(self) -> Path:
        current_text = self.input_path_edit.text().strip()
        if current_text:
            candidate = Path(current_text).expanduser()
            if candidate.is_file():
                return candidate.parent
            if candidate.is_dir():
                return candidate
        return Path.cwd()

    @Slot()
    def _load_input_from_edit(self) -> None:
        raw_text = self.input_path_edit.text().strip()
        if not raw_text:
            self._show_error(
                "No Structure Selected",
                "Choose an XYZ or PDB file before loading the viewer.",
            )
            return
        self._load_structure_path(Path(raw_text))

    @Slot()
    def _reload_current_structure(self) -> None:
        if self._structure is None:
            self._show_error(
                "No Structure Loaded",
                "Load a structure before reloading it.",
            )
            return
        self._load_structure_path(self._structure.file_path)

    def _load_structure_path(self, path: Path) -> None:
        try:
            structure = load_electron_density_structure(path)
        except Exception as exc:
            self._show_error("Structure Load Error", str(exc))
            return

        self._structure = structure
        self.input_path_edit.setText(str(structure.file_path))
        self.input_mode_value.setText("Single structure file")
        self.reference_file_value.setText(str(structure.file_path))
        self.reload_structure_action.setEnabled(True)
        self._sync_controls_to_structure()
        self._apply_mesh_settings(self._mesh_settings_from_controls())
        self.structure_viewer.set_structure(
            self._structure,
            mesh_geometry=self._active_mesh_geometry,
            reset_view=True,
        )
        self.viewer_status_label.setText(
            f"Previewing {structure.display_label} from {structure.file_path.name}."
        )
        self.statusBar().showMessage(f"Loaded {structure.file_path.name}")

    @staticmethod
    def _format_point(values: np.ndarray) -> str:
        array = np.asarray(values, dtype=float)
        return f"({array[0]:.3f}, {array[1]:.3f}, {array[2]:.3f}) A"

    @staticmethod
    def _format_mesh_settings_summary(
        settings: ElectronDensityMeshSettings,
    ) -> str:
        normalized = settings.normalized()
        return (
            f"rstep={normalized.rstep:.3f} A, "
            f"theta={normalized.theta_divisions}, "
            f"phi={normalized.phi_divisions}, "
            f"rmax={normalized.rmax:.3f} A"
        )

    @staticmethod
    def _center_mode_label_for_structure(
        structure: ElectronDensityStructure | None,
    ) -> str:
        if structure is None:
            return "Unavailable"
        if structure.center_mode == "center_of_mass":
            return "Mass-weighted center of mass"
        if structure.center_mode == "nearest_atom":
            return "Nearest atom to calculated center"
        return (
            f"{structure.reference_element} reference-element "
            "geometric center"
        )

    def _mesh_settings_from_controls(self) -> ElectronDensityMeshSettings:
        return ElectronDensityMeshSettings(
            rstep=float(self.rstep_spin.value()),
            theta_divisions=int(self.theta_divisions_spin.value()),
            phi_divisions=int(self.phi_divisions_spin.value()),
            rmax=float(self.rmax_spin.value()),
        ).normalized()

    def _selected_reference_element(self) -> str | None:
        selected = (
            self.reference_element_combo.currentData()
            or self.reference_element_combo.currentText()
        )
        text = str(selected or "").strip()
        return text or None

    def _sync_reference_element_controls(self) -> None:
        combo = self.reference_element_combo
        combo.blockSignals(True)
        combo.clear()
        if self._structure is None:
            combo.setEnabled(False)
            combo.blockSignals(False)
            return
        for element in sorted(self._structure.element_counts):
            count = int(self._structure.element_counts[element])
            combo.addItem(f"{element} ({count})", element)
        combo.setCurrentIndex(
            max(combo.findData(self._structure.reference_element), 0)
        )
        combo.setEnabled(combo.count() > 0)
        combo.blockSignals(False)

    def _sync_controls_to_structure(self) -> None:
        if self._structure is None:
            return
        self.rmax_spin.blockSignals(True)
        try:
            self.rmax_spin.setValue(max(float(self._structure.rmax), 0.01))
        finally:
            self.rmax_spin.blockSignals(False)
        self._sync_reference_element_controls()
        self._refresh_center_display()
        self._refresh_structure_summary()
        self._refresh_active_mesh_display()
        self._refresh_mesh_notice()

    def _refresh_structure_summary(self) -> None:
        if self._structure is None:
            self.structure_summary_value.setText(
                "Load a structure to populate atoms, element counts, center "
                "of mass, reference-element center, current center, and rmax."
            )
            return
        structure = self._structure
        self.structure_summary_value.setText(
            f"{structure.atom_count} atoms, "
            + ", ".join(
                f"{element} x{count}"
                for element, count in structure.element_counts.items()
            )
            + "; calculated center="
            + self._format_point(structure.center_of_mass)
            + f"; {structure.reference_element} geometric center="
            + self._format_point(structure.reference_element_geometric_center)
            + "; active center="
            + self._format_point(structure.active_center)
            + f"; rmax={structure.rmax:.3f} A"
        )

    def _refresh_center_display(self) -> None:
        if self._structure is None:
            self.center_mode_value.setText("Mass-weighted center of mass")
            self.calculated_center_value.setText("Unavailable")
            self.active_center_value.setText("Unavailable")
            self.nearest_atom_value.setText("Unavailable")
            self.geometric_center_value.setText("Unavailable")
            self.reference_element_center_value.setText("Unavailable")
            self.reference_element_offset_value.setText("Unavailable")
            self.reference_element_combo.setEnabled(False)
            self.reset_center_button.setChecked(True)
            self.snap_center_button.setChecked(False)
            self.snap_reference_center_button.setChecked(False)
            return
        structure = self._structure
        self.center_mode_value.setText(
            self._center_mode_label_for_structure(structure)
        )
        self.reset_center_button.setChecked(
            structure.center_mode == "center_of_mass"
        )
        self.snap_center_button.setChecked(
            structure.center_mode == "nearest_atom"
        )
        self.snap_reference_center_button.setChecked(
            structure.center_mode == "reference_element"
        )
        self.calculated_center_value.setText(
            self._format_point(structure.center_of_mass)
        )
        self.geometric_center_value.setText(
            self._format_point(structure.geometric_center)
        )
        self.reference_element_center_value.setText(
            f"{structure.reference_element}: "
            f"{self._format_point(structure.reference_element_geometric_center)}"
        )
        self.reference_element_offset_value.setText(
            f"{structure.reference_element_offset_from_geometric_center:.3f} A "
            "from the total-atom geometric center"
        )
        self.active_center_value.setText(
            self._format_point(structure.active_center)
        )
        self.nearest_atom_value.setText(
            f"#{structure.nearest_atom_index + 1} "
            f"{structure.nearest_atom_element} at "
            f"{self._format_point(structure.nearest_atom_coordinates)}; "
            f"{structure.nearest_atom_distance:.3f} A from the calculated center"
        )

    def _refresh_active_mesh_display(self) -> None:
        center_label = self._center_mode_label_for_structure(self._structure)
        center_text = (
            self._format_point(self._structure.active_center)
            if self._structure is not None
            else "Unavailable"
        )
        if self._active_mesh_geometry is None:
            self.active_mesh_value.setText(
                "No active mesh is rendered yet. The current settings are "
                f"{self._format_mesh_settings_summary(self._active_mesh_settings)}, "
                f"center mode={center_label}, center={center_text}."
            )
            return
        geometry = self._active_mesh_geometry
        self.active_mesh_value.setText(
            f"{self._format_mesh_settings_summary(geometry.settings)}, "
            f"shells={geometry.shell_count}, "
            f"domain=0 to {geometry.domain_max_radius:.3f} A, "
            f"center mode={center_label}, center={center_text}"
        )

    @Slot()
    def _refresh_mesh_notice(self) -> None:
        if self._structure is None:
            self.pending_mesh_value.setText(
                "Load a structure to compare pending mesh fields to the "
                "rendered mesh."
            )
            self.pending_mesh_value.setStyleSheet("color: #475569;")
            return
        pending = self._mesh_settings_from_controls()
        if pending != self._active_mesh_settings:
            self.pending_mesh_value.setText(
                "Pending field values differ from the rendered mesh. Press "
                "Update Mesh Settings to redraw the spherical overlay."
            )
            self.pending_mesh_value.setStyleSheet("color: #92400e;")
            return
        self.pending_mesh_value.setText(
            "Pending field values match the rendered mesh."
        )
        self.pending_mesh_value.setStyleSheet("color: #166534;")

    @Slot()
    def _apply_mesh_from_controls(self) -> None:
        if self._structure is None:
            self._show_error(
                "No Structure Loaded",
                "Load a structure before updating the mesh overlay.",
            )
            return
        self._apply_mesh_settings(self._mesh_settings_from_controls())
        self.statusBar().showMessage("Updated mesh settings")

    def _apply_mesh_settings(
        self,
        settings: ElectronDensityMeshSettings,
        *,
        preserve_viewer_display: bool = False,
    ) -> None:
        self._active_mesh_settings = settings.normalized()
        self._active_mesh_geometry = None
        if self._structure is not None:
            self._active_mesh_geometry = build_electron_density_mesh(
                self._structure,
                self._active_mesh_settings,
            )
        self._refresh_active_mesh_display()
        self._refresh_mesh_notice()
        if self._structure is None:
            return
        if preserve_viewer_display:
            self.structure_viewer.set_structure_preserving_display(
                self._structure,
                mesh_geometry=self._active_mesh_geometry,
            )
            return
        self.structure_viewer.set_structure(
            self._structure,
            mesh_geometry=self._active_mesh_geometry,
            reset_view=False,
        )

    def _apply_center_mode(self, center_mode: str) -> None:
        if self._structure is None:
            self._show_error(
                "No Structure Loaded",
                "Load a structure before changing the active center.",
            )
            return
        try:
            self._structure = recenter_electron_density_structure(
                self._structure,
                center_mode=center_mode,
                reference_element=self._selected_reference_element(),
            )
        except Exception as exc:
            self._show_error("Center Update Error", str(exc))
            return
        self._sync_controls_to_structure()
        self._apply_mesh_settings(
            self._mesh_settings_from_controls(),
            preserve_viewer_display=True,
        )
        self.statusBar().showMessage("Updated active center")

    @Slot()
    def _handle_reference_element_changed(self) -> None:
        if self._structure is None:
            return
        reference_element = self._selected_reference_element()
        if (
            reference_element is None
            or reference_element == self._structure.reference_element
        ):
            return
        previous_structure = self._structure
        try:
            self._structure = recenter_electron_density_structure(
                self._structure,
                center_mode=self._structure.center_mode,
                reference_element=reference_element,
            )
        except Exception as exc:
            self._show_error("Reference Element Error", str(exc))
            self._sync_reference_element_controls()
            return
        active_center_changed = not np.allclose(
            previous_structure.active_center,
            self._structure.active_center,
        ) or not np.isclose(previous_structure.rmax, self._structure.rmax)
        self._sync_controls_to_structure()
        if active_center_changed:
            self._apply_mesh_settings(
                self._mesh_settings_from_controls(),
                preserve_viewer_display=True,
            )
            self.statusBar().showMessage("Updated reference-element center")
            return
        self.statusBar().showMessage("Updated reference element")

    def _show_error(self, title: str, message: str) -> None:
        QMessageBox.critical(self, title, message)


def _forget_open_window(window: StructureViewerMainWindow) -> None:
    if window in _OPEN_WINDOWS:
        _OPEN_WINDOWS.remove(window)


def launch_structure_viewer_ui(
    *,
    initial_input_path: str | Path | None = None,
) -> StructureViewerMainWindow:
    app = QApplication.instance()
    if app is None:
        prepare_saxshell_application_identity()
        app = QApplication(sys.argv)
    configure_saxshell_application(app)
    window = StructureViewerMainWindow(
        initial_input_path=(
            None
            if initial_input_path is None
            else Path(initial_input_path).expanduser().resolve()
        )
    )
    window.show()
    window.raise_()
    _OPEN_WINDOWS.append(window)
    window.destroyed.connect(
        lambda _obj=None, win=window: _forget_open_window(win)
    )
    return window


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="structureviewer",
        description=(
            "Launch the standalone SAXSShell Structure Viewer widget harness."
        ),
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        type=Path,
        help="Optional XYZ or PDB file to prefill in the window.",
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
    launch_structure_viewer_ui(initial_input_path=args.input_path)
    if created_app:
        assert app is not None
        return int(app.exec())
    return 0


__all__ = [
    "StructureViewerMainWindow",
    "build_parser",
    "launch_structure_viewer_ui",
    "main",
]
