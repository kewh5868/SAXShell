from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLayout,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from saxshell.saxs.contrast.electron_density import (
    CONTRAST_SOLVENT_METHOD_DIRECT,
    CONTRAST_SOLVENT_METHOD_NEAT,
    CONTRAST_SOLVENT_METHOD_REFERENCE,
    ContrastSolventDensitySettings,
)
from saxshell.saxs.contrast.solvents import (
    ContrastSolventPreset,
    delete_custom_solvent_preset,
    load_solvent_presets,
    ordered_solvent_preset_names,
    save_custom_solvent_preset,
)
from saxshell.saxs.electron_density_mapping.ui.viewer import (
    ElectronDensityFourierPreviewPlot,
    ElectronDensityProfileOverlay,
    ElectronDensityProfilePlot,
    ElectronDensityResidualPlot,
    ElectronDensityScatteringPlot,
    ElectronDensityStructureViewer,
)
from saxshell.saxs.electron_density_mapping.workflow import (
    ElectronDensityFourierTransformPreview,
    ElectronDensityFourierTransformSettings,
    ElectronDensityInputInspection,
    ElectronDensityMeshGeometry,
    ElectronDensityMeshSettings,
    ElectronDensityOutputArtifacts,
    ElectronDensityProfileResult,
    ElectronDensityScatteringTransformResult,
    ElectronDensitySmearingSettings,
    ElectronDensityStructure,
    apply_smearing_to_profile_result,
    apply_solvent_contrast_to_profile_result,
    build_electron_density_mesh,
    compute_electron_density_profile_for_input,
    compute_electron_density_scattering_profile,
    inspect_structure_input,
    load_electron_density_structure,
    prepare_electron_density_fourier_transform,
    recenter_electron_density_structure,
    suggest_output_basename,
    suggest_output_dir,
    write_electron_density_profile_outputs,
)
from saxshell.saxs.ui.branding import (
    configure_saxshell_application,
    load_saxshell_icon,
    prepare_saxshell_application_identity,
)

_OPEN_WINDOWS: list["ElectronDensityMappingMainWindow"] = []


class _CollapsibleSection(QWidget):
    """A titled, toggle-collapsible container for a single body
    widget."""

    toggled = Signal(bool)

    def __init__(
        self,
        title: str,
        body: QWidget,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 2, 0, 2)
        header_layout.setSpacing(4)
        self._toggle_button = QToolButton()
        self._toggle_button.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self._toggle_button.setArrowType(Qt.ArrowType.RightArrow)
        self._toggle_button.setText(title)
        self._toggle_button.setAutoRaise(True)
        self._toggle_button.clicked.connect(self._toggle)
        header_layout.addWidget(self._toggle_button)
        header_layout.addStretch(1)
        self._body = body
        self._body.setVisible(False)
        layout.addWidget(header)
        layout.addWidget(self._body)

    def _toggle(self) -> None:
        self.set_expanded(not self._body.isVisible())

    def set_expanded(self, expanded: bool) -> None:
        requested = bool(expanded)
        if self._body.isVisible() == requested:
            return
        self._body.setVisible(requested)
        self._toggle_button.setArrowType(
            Qt.ArrowType.DownArrow if requested else Qt.ArrowType.RightArrow
        )
        self._body.updateGeometry()
        self.updateGeometry()
        self.toggled.emit(requested)

    def expand(self) -> None:
        self.set_expanded(True)

    def collapse(self) -> None:
        self.set_expanded(False)

    @property
    def is_expanded(self) -> bool:
        return self._body.isVisible()


class ElectronDensityCalculationWorker(QObject):
    progress = Signal(int, int, str)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        *,
        inspection: ElectronDensityInputInspection,
        mesh_settings: ElectronDensityMeshSettings,
        smearing_settings: ElectronDensitySmearingSettings,
        center_mode: str,
        reference_element: str | None,
        output_dir: str,
        output_basename: str,
    ) -> None:
        super().__init__()
        self._inspection = inspection
        self._mesh_settings = mesh_settings
        self._smearing_settings = smearing_settings
        self._center_mode = str(center_mode)
        self._reference_element = (
            None
            if reference_element is None
            else str(reference_element).strip() or None
        )
        self._output_dir = str(output_dir)
        self._output_basename = str(output_basename)

    @Slot()
    def run(self) -> None:
        try:
            result = compute_electron_density_profile_for_input(
                self._inspection,
                self._mesh_settings,
                smearing_settings=self._smearing_settings,
                center_mode=self._center_mode,
                reference_element=self._reference_element,
                progress_callback=self.progress.emit,
            )
            artifacts = None
            if self._output_dir.strip():
                structure_count = len(self._inspection.structure_files)
                progress_total = structure_count * 2 + 3
                self.progress.emit(
                    progress_total,
                    progress_total,
                    "Writing electron-density outputs.",
                )
                artifacts = write_electron_density_profile_outputs(
                    result,
                    self._output_dir,
                    self._output_basename,
                )
            self.finished.emit(
                {
                    "result": result,
                    "artifacts": artifacts,
                }
            )
        except Exception as exc:
            self.failed.emit(str(exc))


class ElectronDensityMappingMainWindow(QMainWindow):
    """Interactive supporting tool for radial electron-density
    inspection."""

    def __init__(
        self,
        *,
        initial_project_dir: Path | None = None,
        initial_input_path: Path | None = None,
    ) -> None:
        super().__init__()
        self._project_dir = (
            None
            if initial_project_dir is None
            else Path(initial_project_dir).expanduser().resolve()
        )
        self._inspection: ElectronDensityInputInspection | None = None
        self._structure: ElectronDensityStructure | None = None
        self._active_mesh_settings = ElectronDensityMeshSettings()
        self._active_mesh_geometry: ElectronDensityMeshGeometry | None = None
        self._active_smearing_settings = ElectronDensitySmearingSettings()
        self._active_fourier_settings = (
            ElectronDensityFourierTransformSettings()
        )
        self._solvent_presets: dict[str, ContrastSolventPreset] = {}
        self._active_contrast_settings: (
            ContrastSolventDensitySettings | None
        ) = None
        self._active_contrast_name: str | None = None
        self._profile_result: ElectronDensityProfileResult | None = None
        self._fourier_preview: (
            ElectronDensityFourierTransformPreview | None
        ) = None
        self._fourier_result: (
            ElectronDensityScatteringTransformResult | None
        ) = None
        self._calculation_thread: QThread | None = None
        self._calculation_worker: ElectronDensityCalculationWorker | None = (
            None
        )
        self._last_progress_message = ""

        self.setWindowTitle("Electron Density Mapping")
        self.setWindowIcon(load_saxshell_icon())
        self.resize(1520, 960)
        self._build_ui()
        self._set_initial_defaults()
        if initial_input_path is not None:
            self.input_path_edit.setText(
                str(Path(initial_input_path).expanduser().resolve())
            )
            self._load_input_from_edit()

    def _build_ui(self) -> None:
        central = QWidget(self)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(8)
        splitter = QSplitter(Qt.Orientation.Horizontal, central)
        root_layout.addWidget(splitter, stretch=1)
        self.setCentralWidget(central)

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
        self._left_scroll_area.setWidget(left_container)
        splitter.addWidget(self._left_scroll_area)

        intro = QLabel(
            "This supporting application previews a single XYZ or PDB "
            "structure, centers it at a mass-weighted origin or a nearby "
            "atom, and computes both raw and Gaussian-smeared spherical "
            "electron-density profiles."
        )
        intro.setWordWrap(True)
        left_layout.addWidget(intro)

        left_layout.addWidget(self._build_input_group())
        left_layout.addWidget(self._build_output_group())
        self.mesh_section = _CollapsibleSection(
            "Mesh Settings", self._build_mesh_group(), left_container
        )
        left_layout.addWidget(self.mesh_section)
        left_layout.addWidget(self._build_actions_group())
        left_layout.addWidget(self._build_smearing_group())
        self.contrast_section = _CollapsibleSection(
            "Electron Density Contrast",
            self._build_contrast_group(),
            left_container,
        )
        left_layout.addWidget(self.contrast_section)
        self.fourier_section = _CollapsibleSection(
            "Fourier Transform",
            self._build_fourier_transform_group(),
            left_container,
        )
        left_layout.addWidget(self.fourier_section)
        left_layout.addWidget(self._build_status_group(), stretch=1)
        left_layout.addStretch(1)

        self._right_scroll_area = QScrollArea(self)
        self._right_scroll_area.setWidgetResizable(True)
        self._right_scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._right_scroll_area.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOn
        )
        self._right_scroll_area.verticalScrollBar().setSingleStep(32)
        self._right_panel = QWidget()
        right_layout = QVBoxLayout(self._right_panel)
        right_layout.setContentsMargins(10, 10, 10, 10)
        right_layout.setSpacing(10)
        right_layout.setSizeConstraint(QLayout.SizeConstraint.SetMinAndMaxSize)

        plot_options_row = QHBoxLayout()
        plot_options_row.setContentsMargins(0, 0, 0, 0)
        plot_options_row.setSpacing(6)
        self.show_variance_checkbox = QCheckBox("Show Variance Shading")
        self.show_variance_checkbox.setChecked(True)
        self.show_variance_checkbox.toggled.connect(
            self._toggle_variance_shading
        )
        plot_options_row.addWidget(self.show_variance_checkbox)
        self.auto_expand_checkbox = QCheckBox("Auto-expand plots")
        self.auto_expand_checkbox.setChecked(True)
        self.auto_expand_checkbox.setToolTip(
            "Automatically expand a plot panel when its data is updated."
        )
        plot_options_row.addWidget(self.auto_expand_checkbox)
        plot_options_row.addStretch(1)
        self.export_plot_traces_button = QPushButton("Export Plot Traces")
        self.export_plot_traces_button.setToolTip(
            "Export all active plot traces (raw density, smeared density, "
            "Fourier preview, and scattering) into a single CSV file."
        )
        self.export_plot_traces_button.clicked.connect(
            self._export_plot_traces
        )
        plot_options_row.addWidget(self.export_plot_traces_button)
        right_layout.addLayout(plot_options_row)

        self.profile_plot = ElectronDensityProfilePlot(
            self._right_panel,
            title="Orientation-Averaged Radial Electron Density ρ(r)",
            legend_label="Orientation-averaged density",
            trace_color="#1d4ed8",
            fill_color="#60a5fa",
            profile_attribute="orientation_average_density",
            spread_attribute="orientation_density_stddev",
            as_step_trace=True,
        )
        self.profile_section = _CollapsibleSection(
            "Orientation-Averaged Radial Electron Density ρ(r)",
            self.profile_plot,
            self._right_panel,
        )
        right_layout.addWidget(self.profile_section)

        self.smeared_profile_plot = ElectronDensityProfilePlot(
            self._right_panel,
            title="Gaussian-Smeared Radial Electron Density ρ(r)",
            legend_label="Smeared density",
            trace_color="#15803d",
            fill_color="#86efac",
            profile_attribute="smeared_orientation_average_density",
            spread_attribute="smeared_orientation_density_stddev",
            as_step_trace=False,
        )
        self.smeared_section = _CollapsibleSection(
            "Gaussian-Smeared Radial Electron Density ρ(r)",
            self.smeared_profile_plot,
            self._right_panel,
        )
        right_layout.addWidget(self.smeared_section)

        self.residual_profile_plot = ElectronDensityResidualPlot(
            self._right_panel
        )
        self.residual_section = _CollapsibleSection(
            "Solvent-Subtracted Residual",
            self.residual_profile_plot,
            self._right_panel,
        )
        right_layout.addWidget(self.residual_section)

        self.fourier_preview_plot = ElectronDensityFourierPreviewPlot(
            self._right_panel
        )
        self.fourier_preview_section = _CollapsibleSection(
            "Fourier Transform Preview",
            self.fourier_preview_plot,
            self._right_panel,
        )
        right_layout.addWidget(self.fourier_preview_section)

        self.scattering_plot = ElectronDensityScatteringPlot(self._right_panel)
        self.scattering_section = _CollapsibleSection(
            "Scattering Profile I(q)",
            self.scattering_plot,
            self._right_panel,
        )
        right_layout.addWidget(self.scattering_section)

        self.structure_viewer = ElectronDensityStructureViewer(
            self._right_panel
        )
        right_layout.addWidget(self.structure_viewer, stretch=1)
        for section in (
            self.profile_section,
            self.smeared_section,
            self.residual_section,
            self.fourier_preview_section,
            self.scattering_section,
        ):
            section.toggled.connect(self._refresh_right_panel_layout)
        self.profile_plot.set_variance_visible(True)
        self.smeared_profile_plot.set_variance_visible(True)
        self.residual_profile_plot.set_variance_visible(True)

        self._right_scroll_area.setWidget(self._right_panel)
        self._refresh_right_panel_layout()
        splitter.addWidget(self._right_scroll_area)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([430, 1090])

    @Slot()
    def _refresh_right_panel_layout(self, *_args: object) -> None:
        layout = self._right_panel.layout()
        if layout is None:
            return
        layout.activate()
        minimum_height = layout.minimumSize().height()
        if minimum_height > 0:
            self._right_panel.setMinimumHeight(minimum_height)
        self._right_panel.updateGeometry()
        self._right_panel.adjustSize()

    def _build_input_group(self) -> QWidget:
        group = QGroupBox("Input")
        layout = QVBoxLayout(group)

        path_row = QHBoxLayout()
        self.input_path_edit = QLineEdit()
        self.input_path_edit.setPlaceholderText(
            "Choose one XYZ/PDB file or a folder of XYZ/PDB files"
        )
        self.input_path_edit.returnPressed.connect(self._load_input_from_edit)
        path_row.addWidget(self.input_path_edit, stretch=1)

        browse_file_button = QPushButton("Choose File")
        browse_file_button.clicked.connect(self._choose_input_file)
        path_row.addWidget(browse_file_button)

        browse_folder_button = QPushButton("Choose Folder")
        browse_folder_button.clicked.connect(self._choose_input_folder)
        path_row.addWidget(browse_folder_button)
        layout.addLayout(path_row)

        self.load_input_button = QPushButton("Load Input")
        self.load_input_button.clicked.connect(self._load_input_from_edit)
        layout.addWidget(self.load_input_button)

        form = QFormLayout()
        self.input_mode_value = QLabel("No structure loaded")
        self.input_mode_value.setWordWrap(True)
        form.addRow("Input mode", self.input_mode_value)

        self.reference_file_value = QLabel("Unavailable")
        self.reference_file_value.setWordWrap(True)
        form.addRow("Reference file", self.reference_file_value)

        self.structure_summary_value = QLabel(
            "Load a structure to populate atoms, element counts, center of "
            "mass, active center, and rmax."
        )
        self.structure_summary_value.setWordWrap(True)
        form.addRow("Structure summary", self.structure_summary_value)
        layout.addLayout(form)
        return group

    def _build_output_group(self) -> QWidget:
        group = QGroupBox("Output")
        layout = QFormLayout(group)

        output_dir_field = QWidget(group)
        output_dir_row = QHBoxLayout()
        output_dir_row.setContentsMargins(0, 0, 0, 0)
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setPlaceholderText(
            "Choose a folder for the profile CSV and JSON summary"
        )
        output_dir_row.addWidget(self.output_dir_edit, stretch=1)
        output_dir_button = QPushButton("Browse")
        output_dir_button.clicked.connect(self._choose_output_dir)
        output_dir_row.addWidget(output_dir_button)
        output_dir_field.setLayout(output_dir_row)
        layout.addRow("Output directory", output_dir_field)

        self.output_basename_edit = QLineEdit()
        self.output_basename_edit.setPlaceholderText(
            "electron_density_profile"
        )
        layout.addRow("Output basename", self.output_basename_edit)
        return group

    def _build_mesh_group(self) -> QWidget:
        group = QGroupBox("Mesh Settings")
        layout = QFormLayout(group)

        self.rstep_spin = QDoubleSpinBox()
        self.rstep_spin.setRange(0.01, 1000.0)
        self.rstep_spin.setDecimals(4)
        self.rstep_spin.setSingleStep(0.01)
        self.rstep_spin.setValue(self._active_mesh_settings.rstep)
        self.rstep_spin.valueChanged.connect(self._refresh_mesh_notice)
        layout.addRow("rstep (Å)", self.rstep_spin)

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
        layout.addRow("rmax (Å)", self.rmax_spin)

        layout.addRow(QLabel("Center mode:"))
        self.center_mode_value = QLabel("Calculated center of mass")
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

        center_button_row = QWidget(group)
        center_button_layout = QHBoxLayout(center_button_row)
        center_button_layout.setContentsMargins(0, 0, 0, 0)
        center_button_layout.setSpacing(6)
        self.snap_center_button = QPushButton("Snap Center to Nearest Atom")
        self.snap_center_button.clicked.connect(
            lambda _checked=False: self._apply_center_mode("nearest_atom")
        )
        center_button_layout.addWidget(self.snap_center_button)
        self.reset_center_button = QPushButton("Reset to Calculated Center")
        self.reset_center_button.clicked.connect(
            lambda _checked=False: self._apply_center_mode("center_of_mass")
        )
        center_button_layout.addWidget(self.reset_center_button)
        self.snap_reference_center_button = QPushButton(
            "Snap Center to Reference Element"
        )
        self.snap_reference_center_button.clicked.connect(
            lambda _checked=False: self._apply_center_mode("reference_element")
        )
        center_button_layout.addWidget(self.snap_reference_center_button)
        layout.addRow(center_button_row)

        self.update_mesh_button = QPushButton("Update Mesh Settings")
        self.update_mesh_button.clicked.connect(self._apply_mesh_from_controls)
        layout.addRow(self.update_mesh_button)

        layout.addRow(QLabel("Active mesh:"))
        self.active_mesh_value = QLabel()
        self.active_mesh_value.setWordWrap(True)
        layout.addRow(self.active_mesh_value)

        layout.addRow(QLabel("Pending fields:"))
        self.pending_mesh_value = QLabel()
        self.pending_mesh_value.setWordWrap(True)
        layout.addRow(self.pending_mesh_value)
        return group

    def _build_actions_group(self) -> QWidget:
        group = QGroupBox("Actions")
        layout = QVBoxLayout(group)
        self.run_button = QPushButton("Run Electron Density Calculation")
        self.run_button.clicked.connect(self._run_calculation)
        layout.addWidget(self.run_button)

        self.calculation_progress_message = QLabel("Idle")
        self.calculation_progress_message.setWordWrap(True)
        layout.addWidget(self.calculation_progress_message)

        self.calculation_progress_bar = QProgressBar()
        self.calculation_progress_bar.setRange(0, 1)
        self.calculation_progress_bar.setValue(0)
        layout.addWidget(self.calculation_progress_bar)
        return group

    def _build_smearing_group(self) -> QWidget:
        group = QGroupBox("Smearing")
        layout = QFormLayout(group)

        intro = QLabel(
            "Apply a Gaussian kernel to the raw radial density profile to "
            "soften boxy shell-to-shell transitions. A value of 0 disables "
            "smearing."
        )
        intro.setWordWrap(True)
        layout.addRow(intro)

        self.smearing_factor_spin = QDoubleSpinBox()
        self.smearing_factor_spin.setRange(0.0, 100.0)
        self.smearing_factor_spin.setDecimals(6)
        self.smearing_factor_spin.setSingleStep(0.001)
        self.smearing_factor_spin.setKeyboardTracking(False)
        self.smearing_factor_spin.setValue(
            self._active_smearing_settings.debye_waller_factor
        )
        self.smearing_factor_spin.valueChanged.connect(
            self._apply_smearing_from_controls
        )
        layout.addRow("Debye-Waller factor (Å²)", self.smearing_factor_spin)

        self.smearing_sigma_value = QLabel()
        self.smearing_sigma_value.setWordWrap(True)
        layout.addRow("Gaussian sigma", self.smearing_sigma_value)

        self.smearing_summary_value = QLabel()
        self.smearing_summary_value.setWordWrap(True)
        layout.addRow("Behavior", self.smearing_summary_value)
        return group

    def _build_contrast_group(self) -> QWidget:
        group = QGroupBox("Electron Density Contrast")
        layout = QFormLayout(group)

        intro = QLabel(
            "Estimate a flat solvent electron density using the contrast Debye "
            "workflow solvent options, then compare that value against the "
            "smeared radial density profile."
        )
        intro.setWordWrap(True)
        layout.addRow(intro)

        self.solvent_method_combo = QComboBox()
        self.solvent_method_combo.addItem(
            "Estimate from Solvent Formula and Density",
            userData=CONTRAST_SOLVENT_METHOD_NEAT,
        )
        self.solvent_method_combo.addItem(
            "Reference Solvent Structure (XYZ/PDB)",
            userData=CONTRAST_SOLVENT_METHOD_REFERENCE,
        )
        self.solvent_method_combo.addItem(
            "Direct Electron Density Value",
            userData=CONTRAST_SOLVENT_METHOD_DIRECT,
        )
        self.solvent_method_combo.currentIndexChanged.connect(
            self._sync_density_method_controls
        )
        layout.addRow("Compute option", self.solvent_method_combo)

        self.solvent_preset_combo = QComboBox()
        self.solvent_preset_combo.currentIndexChanged.connect(
            self._load_selected_solvent_preset
        )
        self.save_custom_solvent_button = QPushButton("Save Custom Solvent")
        self.save_custom_solvent_button.clicked.connect(
            self._save_current_solvent_preset
        )
        self.delete_custom_solvent_button = QPushButton(
            "Delete Custom Solvent"
        )
        self.delete_custom_solvent_button.clicked.connect(
            self._delete_current_solvent_preset
        )
        solvent_preset_row = QWidget(group)
        solvent_preset_layout = QHBoxLayout(solvent_preset_row)
        solvent_preset_layout.setContentsMargins(0, 0, 0, 0)
        solvent_preset_layout.setSpacing(6)
        solvent_preset_layout.addWidget(self.solvent_preset_combo, stretch=1)
        solvent_preset_layout.addWidget(self.save_custom_solvent_button)
        solvent_preset_layout.addWidget(self.delete_custom_solvent_button)
        layout.addRow("Saved solvents", solvent_preset_row)

        self.solvent_formula_edit = QLineEdit()
        self.solvent_formula_edit.setPlaceholderText(
            "Examples: H2O, Vacuum, C3H7NO (DMF), C2H6OS (DMSO)"
        )
        layout.addRow("Solvent formula", self.solvent_formula_edit)

        self.solvent_density_spin = QDoubleSpinBox()
        self.solvent_density_spin.setDecimals(6)
        self.solvent_density_spin.setRange(0.0, 100.0)
        self.solvent_density_spin.setSingleStep(0.01)
        self.solvent_density_spin.setValue(1.0)
        self.solvent_density_spin.setKeyboardTracking(False)
        layout.addRow("Density (g/mL)", self.solvent_density_spin)

        self.direct_density_spin = QDoubleSpinBox()
        self.direct_density_spin.setDecimals(6)
        self.direct_density_spin.setRange(0.0, 100.0)
        self.direct_density_spin.setSingleStep(0.001)
        self.direct_density_spin.setValue(0.334)
        self.direct_density_spin.setKeyboardTracking(False)
        layout.addRow("Direct density (e-/ Å³)", self.direct_density_spin)

        self.reference_solvent_file_edit = QLineEdit()
        self.reference_solvent_file_edit.setPlaceholderText(
            "Choose a reference solvent XYZ or PDB file"
        )
        reference_row = QWidget(group)
        reference_layout = QHBoxLayout(reference_row)
        reference_layout.setContentsMargins(0, 0, 0, 0)
        reference_layout.setSpacing(6)
        reference_layout.addWidget(
            self.reference_solvent_file_edit,
            stretch=1,
        )
        self.reference_solvent_browse_button = QPushButton("Browse…")
        self.reference_solvent_browse_button.clicked.connect(
            self._choose_reference_solvent_file
        )
        reference_layout.addWidget(self.reference_solvent_browse_button)
        layout.addRow("Reference solvent file", reference_row)

        self.solvent_method_hint_label = QLabel()
        self.solvent_method_hint_label.setWordWrap(True)
        layout.addRow("", self.solvent_method_hint_label)

        self.compute_solvent_density_button = QPushButton(
            "Compute Solvent Electron Density"
        )
        self.compute_solvent_density_button.clicked.connect(
            self._compute_solvent_contrast
        )
        layout.addRow(self.compute_solvent_density_button)

        layout.addRow(QLabel("Active contrast:"))
        self.active_contrast_value = QLabel(
            "No active solvent electron density yet."
        )
        self.active_contrast_value.setWordWrap(True)
        layout.addRow(self.active_contrast_value)

        layout.addRow(QLabel("Notes:"))
        self.contrast_notice_value = QLabel(
            "Compute a solvent electron density to add the solvent line, "
            "cutoff, and residual traces."
        )
        self.contrast_notice_value.setWordWrap(True)
        layout.addRow(self.contrast_notice_value)
        return group

    def _build_fourier_transform_group(self) -> QWidget:
        group = QGroupBox("Fourier Transform")
        outer = QVBoxLayout(group)
        outer.setSpacing(6)

        intro = QLabel(
            "Prepare a spherical Born-approximation transform of the smeared "
            "electron-density profile into q-space. The preview panel shows "
            "the active bounds, the selected real-space window over that interval, "
            "and the resampled data before the transform is evaluated."
        )
        intro.setWordWrap(True)
        outer.addWidget(intro)

        # --- two-column grid for all numeric inputs ---
        grid = QGridLayout()
        grid.setSpacing(4)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)

        # Left column header
        grid.addWidget(QLabel("r range (Å)"), 0, 0)
        # Right column header
        grid.addWidget(QLabel("q range (Å⁻¹)"), 0, 1)

        # r min
        r_min_cell = QWidget()
        r_min_form = QFormLayout(r_min_cell)
        r_min_form.setContentsMargins(0, 0, 0, 0)
        r_min_form.setSpacing(2)
        self.fourier_rmin_spin = QDoubleSpinBox()
        self.fourier_rmin_spin.setRange(0.0, 100000.0)
        self.fourier_rmin_spin.setDecimals(4)
        self.fourier_rmin_spin.setSingleStep(0.05)
        self.fourier_rmin_spin.setKeyboardTracking(False)
        self.fourier_rmin_spin.setValue(self._active_fourier_settings.r_min)
        self.fourier_rmin_spin.valueChanged.connect(
            self._refresh_fourier_preview_from_controls
        )
        r_min_form.addRow("r min", self.fourier_rmin_spin)
        # r max
        self.fourier_rmax_spin = QDoubleSpinBox()
        self.fourier_rmax_spin.setRange(0.01, 100000.0)
        self.fourier_rmax_spin.setDecimals(4)
        self.fourier_rmax_spin.setSingleStep(0.1)
        self.fourier_rmax_spin.setKeyboardTracking(False)
        self.fourier_rmax_spin.setValue(self._active_fourier_settings.r_max)
        self.fourier_rmax_spin.valueChanged.connect(
            self._refresh_fourier_preview_from_controls
        )
        r_min_form.addRow("r max", self.fourier_rmax_spin)
        grid.addWidget(r_min_cell, 1, 0)

        # q min / q max / q step
        q_cell = QWidget()
        q_form = QFormLayout(q_cell)
        q_form.setContentsMargins(0, 0, 0, 0)
        q_form.setSpacing(2)
        self.fourier_qmin_spin = QDoubleSpinBox()
        self.fourier_qmin_spin.setRange(0.0, 1000.0)
        self.fourier_qmin_spin.setDecimals(4)
        self.fourier_qmin_spin.setSingleStep(0.02)
        self.fourier_qmin_spin.setKeyboardTracking(False)
        self.fourier_qmin_spin.setValue(self._active_fourier_settings.q_min)
        self.fourier_qmin_spin.valueChanged.connect(
            self._refresh_fourier_preview_from_controls
        )
        q_form.addRow("q min", self.fourier_qmin_spin)
        self.fourier_qmax_spin = QDoubleSpinBox()
        self.fourier_qmax_spin.setRange(0.01, 1000.0)
        self.fourier_qmax_spin.setDecimals(4)
        self.fourier_qmax_spin.setSingleStep(0.1)
        self.fourier_qmax_spin.setKeyboardTracking(False)
        self.fourier_qmax_spin.setValue(self._active_fourier_settings.q_max)
        self.fourier_qmax_spin.valueChanged.connect(
            self._refresh_fourier_preview_from_controls
        )
        q_form.addRow("q max", self.fourier_qmax_spin)
        self.fourier_qstep_spin = QDoubleSpinBox()
        self.fourier_qstep_spin.setRange(0.0001, 1000.0)
        self.fourier_qstep_spin.setDecimals(4)
        self.fourier_qstep_spin.setSingleStep(0.01)
        self.fourier_qstep_spin.setKeyboardTracking(False)
        self.fourier_qstep_spin.setValue(self._active_fourier_settings.q_step)
        self.fourier_qstep_spin.valueChanged.connect(
            self._refresh_fourier_preview_from_controls
        )
        q_form.addRow("q step", self.fourier_qstep_spin)
        grid.addWidget(q_cell, 1, 1)

        # Window / resampling side-by-side in second grid row
        win_cell = QWidget()
        win_form = QFormLayout(win_cell)
        win_form.setContentsMargins(0, 0, 0, 0)
        win_form.setSpacing(2)
        self.fourier_window_combo = QComboBox()
        self.fourier_window_combo.addItem("None", "none")
        self.fourier_window_combo.addItem("Lorch", "lorch")
        self.fourier_window_combo.addItem("Cosine", "cosine")
        self.fourier_window_combo.addItem("Hanning", "hanning")
        self.fourier_window_combo.addItem("Parzen", "parzen")
        self.fourier_window_combo.addItem("Welch", "welch")
        self.fourier_window_combo.addItem("Gaussian", "gaussian")
        self.fourier_window_combo.addItem("Sine", "sine")
        self.fourier_window_combo.addItem("Kaiser-Bessel", "kaiser_bessel")
        default_window_index = max(
            self.fourier_window_combo.findData(
                self._active_fourier_settings.window_function
            ),
            0,
        )
        self.fourier_window_combo.setCurrentIndex(default_window_index)
        self.fourier_window_combo.currentIndexChanged.connect(
            self._refresh_fourier_preview_from_controls
        )
        win_form.addRow("Window", self.fourier_window_combo)
        grid.addWidget(win_cell, 2, 0)

        res_cell = QWidget()
        res_form = QFormLayout(res_cell)
        res_form.setContentsMargins(0, 0, 0, 0)
        res_form.setSpacing(2)
        self.fourier_resampling_points_spin = QSpinBox()
        self.fourier_resampling_points_spin.setRange(8, 32768)
        self.fourier_resampling_points_spin.setSingleStep(128)
        self.fourier_resampling_points_spin.setValue(
            self._active_fourier_settings.resampling_points
        )
        self.fourier_resampling_points_spin.valueChanged.connect(
            self._refresh_fourier_preview_from_controls
        )
        res_form.addRow("Resample pts", self.fourier_resampling_points_spin)
        grid.addWidget(res_cell, 2, 1)

        self.fourier_use_solvent_subtracted_checkbox = QCheckBox(
            "Use solvent-subtracted profile"
        )
        self.fourier_use_solvent_subtracted_checkbox.setChecked(
            self._active_fourier_settings.use_solvent_subtracted_profile
        )
        self.fourier_use_solvent_subtracted_checkbox.toggled.connect(
            self._refresh_fourier_preview_from_controls
        )
        grid.addWidget(
            self.fourier_use_solvent_subtracted_checkbox, 3, 0, 1, 2
        )

        outer.addLayout(grid)

        # Log-scale checkboxes
        log_row = QWidget(group)
        log_layout = QHBoxLayout(log_row)
        log_layout.setContentsMargins(0, 0, 0, 0)
        log_layout.setSpacing(6)
        self.fourier_log_q_checkbox = QCheckBox("Log q axis")
        self.fourier_log_q_checkbox.setChecked(
            self._active_fourier_settings.log_q_axis
        )
        self.fourier_log_q_checkbox.toggled.connect(
            self._apply_fourier_axis_scale_preferences
        )
        log_layout.addWidget(self.fourier_log_q_checkbox)
        self.fourier_log_intensity_checkbox = QCheckBox("Log intensity axis")
        self.fourier_log_intensity_checkbox.setChecked(
            self._active_fourier_settings.log_intensity_axis
        )
        self.fourier_log_intensity_checkbox.toggled.connect(
            self._apply_fourier_axis_scale_preferences
        )
        log_layout.addWidget(self.fourier_log_intensity_checkbox)
        log_layout.addStretch(1)
        outer.addWidget(log_row)

        self.evaluate_fourier_button = QPushButton(
            "Evaluate Fourier Transform"
        )
        self.evaluate_fourier_button.clicked.connect(
            self._evaluate_fourier_transform
        )
        outer.addWidget(self.evaluate_fourier_button)

        info_form = QFormLayout()
        info_form.setSpacing(4)
        info_form.addRow(QLabel("Available r range:"))
        self.fourier_available_range_value = QLabel(
            "Awaiting mesh/profile domain."
        )
        self.fourier_available_range_value.setWordWrap(True)
        info_form.addRow(self.fourier_available_range_value)

        info_form.addRow(QLabel("Sampling:"))
        self.fourier_nyquist_value = QLabel(
            "Awaiting transform sampling settings."
        )
        self.fourier_nyquist_value.setWordWrap(True)
        info_form.addRow(self.fourier_nyquist_value)

        info_form.addRow(QLabel("Notes:"))
        self.fourier_notice_value = QLabel(
            "Run the density calculation to prepare a q-space transform preview."
        )
        self.fourier_notice_value.setWordWrap(True)
        info_form.addRow(self.fourier_notice_value)
        outer.addLayout(info_form)
        return group

    def _build_status_group(self) -> QWidget:
        group = QGroupBox("Status")
        layout = QVBoxLayout(group)
        self.status_text = QPlainTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMinimumHeight(210)
        layout.addWidget(self.status_text)
        return group

    def _set_initial_defaults(self) -> None:
        if self._project_dir is not None:
            self.output_dir_edit.setText(
                str(
                    suggest_output_dir(
                        self._project_dir,
                        project_dir=self._project_dir,
                    )
                )
            )
        self._refresh_center_display()
        self._refresh_active_mesh_display()
        self._refresh_mesh_notice()
        self._refresh_smearing_display()
        self._reload_solvent_presets(selected_name="Water")
        self._sync_density_method_controls()
        self._refresh_contrast_display()
        self._refresh_fourier_info_labels(None)
        self._append_status(
            "Waiting for an XYZ or PDB structure. Folder mode will preview and "
            "average across every valid structure in the folder when you run the calculation."
        )

    @Slot(bool)
    def _toggle_variance_shading(self, checked: bool) -> None:
        self.profile_plot.set_variance_visible(bool(checked))
        self.smeared_profile_plot.set_variance_visible(bool(checked))
        self.residual_profile_plot.set_variance_visible(bool(checked))

    def _mesh_settings_from_controls(self) -> ElectronDensityMeshSettings:
        return ElectronDensityMeshSettings(
            rstep=float(self.rstep_spin.value()),
            theta_divisions=int(self.theta_divisions_spin.value()),
            phi_divisions=int(self.phi_divisions_spin.value()),
            rmax=float(self.rmax_spin.value()),
        ).normalized()

    def _smearing_settings_from_controls(
        self,
    ) -> ElectronDensitySmearingSettings:
        return ElectronDensitySmearingSettings(
            debye_waller_factor=float(self.smearing_factor_spin.value()),
        ).normalized()

    def _fourier_settings_from_controls(
        self,
    ) -> ElectronDensityFourierTransformSettings:
        return ElectronDensityFourierTransformSettings(
            r_min=float(self.fourier_rmin_spin.value()),
            r_max=float(self.fourier_rmax_spin.value()),
            window_function=str(
                self.fourier_window_combo.currentData()
                or self.fourier_window_combo.currentText()
            ),
            resampling_points=int(self.fourier_resampling_points_spin.value()),
            q_min=float(self.fourier_qmin_spin.value()),
            q_max=float(self.fourier_qmax_spin.value()),
            q_step=float(self.fourier_qstep_spin.value()),
            use_solvent_subtracted_profile=bool(
                self.fourier_use_solvent_subtracted_checkbox.isChecked()
            ),
            log_q_axis=bool(self.fourier_log_q_checkbox.isChecked()),
            log_intensity_axis=bool(
                self.fourier_log_intensity_checkbox.isChecked()
            ),
        ).normalized()

    @staticmethod
    def _format_point(values: np.ndarray) -> str:
        array = np.asarray(values, dtype=float)
        return f"({array[0]:.3f}, {array[1]:.3f}, {array[2]:.3f}) Å"

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
        selected_index = max(
            combo.findData(self._structure.reference_element), 0
        )
        combo.setCurrentIndex(selected_index)
        combo.setEnabled(combo.count() > 0)
        combo.blockSignals(False)

    def _reset_density_results(self) -> None:
        self._profile_result = None
        self._fourier_preview = None
        self._fourier_result = None
        self.profile_plot.draw_placeholder()
        self.smeared_profile_plot.draw_placeholder()
        self.residual_profile_plot.draw_placeholder()
        self.fourier_preview_plot.draw_placeholder()
        self.scattering_plot.draw_placeholder()

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
            + f"; rmax={structure.rmax:.3f} Å"
        )

    def _refresh_center_display(self) -> None:
        if self._structure is None:
            self.center_mode_value.setText("Calculated center of mass")
            self.calculated_center_value.setText("Unavailable")
            self.active_center_value.setText("Unavailable")
            self.nearest_atom_value.setText("Unavailable")
            self.geometric_center_value.setText("Unavailable")
            self.reference_element_center_value.setText("Unavailable")
            self.reference_element_offset_value.setText("Unavailable")
            self.reference_element_combo.setEnabled(False)
            return
        structure = self._structure
        if structure.center_mode == "center_of_mass":
            mode_label = "Mass-weighted center of mass"
        elif structure.center_mode == "nearest_atom":
            mode_label = "Nearest atom to calculated center"
        else:
            mode_label = (
                f"{structure.reference_element} reference-element "
                "geometric center"
            )
        self.center_mode_value.setText(mode_label)
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
            f"{structure.reference_element_offset_from_geometric_center:.3f} Å "
            "from the total-atom geometric center"
        )
        self.active_center_value.setText(
            self._format_point(structure.active_center)
        )
        self.nearest_atom_value.setText(
            f"#{structure.nearest_atom_index + 1} "
            f"{structure.nearest_atom_element} at "
            f"{self._format_point(structure.nearest_atom_coordinates)}; "
            f"{structure.nearest_atom_distance:.3f} Å from the calculated center"
        )

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
            updated_structure = recenter_electron_density_structure(
                previous_structure,
                center_mode=previous_structure.center_mode,
                reference_element=reference_element,
            )
        except Exception as exc:
            self._show_error("Reference Element Error", str(exc))
            self._sync_reference_element_controls()
            return
        self._structure = updated_structure
        active_center_changed = not np.allclose(
            previous_structure.active_center,
            updated_structure.active_center,
        ) or not np.isclose(previous_structure.rmax, updated_structure.rmax)
        self._sync_controls_to_structure()
        self._refresh_active_mesh_display()
        if active_center_changed:
            self._reset_density_results()
            self._apply_mesh_settings(
                self._mesh_settings_from_controls(),
                announce=False,
                preserve_viewer_display=True,
            )
            self._sync_fourier_controls_to_domain(reset_bounds=True)
            self._refresh_fourier_info_labels(None)
            self._refresh_contrast_display()
            self._append_status(
                "Updated the active center to the "
                f"{updated_structure.reference_element} reference-element "
                "geometric center."
            )
            self.statusBar().showMessage("Updated reference-element center")
        else:
            self._append_status(
                "Updated the reference element to "
                f"{updated_structure.reference_element}; offset from the "
                "total-atom geometric center = "
                f"{updated_structure.reference_element_offset_from_geometric_center:.3f} Å."
            )
            self.statusBar().showMessage("Updated reference element")

    def _refresh_smearing_display(self) -> None:
        settings = self._smearing_settings_from_controls()
        self._active_smearing_settings = settings
        self.smearing_sigma_value.setText(f"{settings.gaussian_sigma_a:.4f} Å")
        if settings.debye_waller_factor <= 0.0:
            self.smearing_summary_value.setText(
                "Smearing is disabled. The lower plot will match the raw profile."
            )
        else:
            self.smearing_summary_value.setText(
                "The lower plot applies a Gaussian kernel to the raw radial "
                f"density profile with sigma={settings.gaussian_sigma_a:.4f} Å."
            )

    def _current_smeared_profile_overlay(
        self,
    ) -> ElectronDensityProfileOverlay | None:
        if (
            self._profile_result is None
            or self._profile_result.solvent_contrast is None
        ):
            return None
        contrast = self._profile_result.solvent_contrast
        return ElectronDensityProfileOverlay(
            solvent_density_e_per_a3=float(contrast.solvent_density_e_per_a3),
            solvent_density_label=contrast.legend_label,
            cutoff_radius_a=contrast.cutoff_radius_a,
            cutoff_label=contrast.cutoff_label,
            solvent_subtracted_density=np.asarray(
                contrast.solvent_subtracted_smeared_density,
                dtype=float,
            ),
            solvent_subtracted_label="Solvent-subtracted ρ(r)",
        )

    def _refresh_profile_plots(self) -> None:
        if self._profile_result is None:
            self.profile_plot.draw_placeholder()
            self.smeared_profile_plot.draw_placeholder()
            self.residual_profile_plot.draw_placeholder()
            return
        self.profile_plot.set_profile(self._profile_result)
        self.smeared_profile_plot.set_profile(
            self._profile_result,
            overlay=self._current_smeared_profile_overlay(),
        )
        self.residual_profile_plot.set_residual_profile(
            self._profile_result,
            self._profile_result.solvent_contrast,
        )
        self._toggle_variance_shading(self.show_variance_checkbox.isChecked())
        if self.auto_expand_checkbox.isChecked():
            self.profile_section.expand()
            self.smeared_section.expand()
            if self._profile_result.solvent_contrast is not None:
                self.residual_section.expand()

    def _selected_solvent_preset_name(self) -> str | None:
        return self.solvent_preset_combo.currentData()

    def _reload_solvent_presets(
        self,
        *,
        selected_name: str | None = None,
    ) -> None:
        previous_name = selected_name or self._selected_solvent_preset_name()
        self._solvent_presets = load_solvent_presets()
        self.solvent_preset_combo.blockSignals(True)
        self.solvent_preset_combo.clear()
        self.solvent_preset_combo.addItem("Custom entry", None)
        selected_index = 0
        for index, preset_name in enumerate(
            ordered_solvent_preset_names(self._solvent_presets),
            start=1,
        ):
            preset = self._solvent_presets[preset_name]
            label = (
                preset.name if preset.builtin else f"{preset.name} (Custom)"
            )
            self.solvent_preset_combo.addItem(label, preset_name)
            if previous_name == preset_name:
                selected_index = index
        self.solvent_preset_combo.setCurrentIndex(selected_index)
        self.solvent_preset_combo.blockSignals(False)
        self._load_selected_solvent_preset()

    @Slot()
    def _load_selected_solvent_preset(self) -> None:
        preset_name = self._selected_solvent_preset_name()
        preset = self._solvent_presets.get(preset_name or "")
        if preset is None:
            self.delete_custom_solvent_button.setEnabled(False)
            return
        self.solvent_formula_edit.setText(preset.formula)
        self.solvent_density_spin.setValue(preset.density_g_per_ml)
        self.delete_custom_solvent_button.setEnabled(not preset.builtin)

    @Slot()
    def _save_current_solvent_preset(self) -> None:
        suggested_name = self._selected_solvent_preset_name() or ""
        preset_name, accepted = QInputDialog.getText(
            self,
            "Save Custom Solvent",
            "Custom solvent name:",
            text=suggested_name,
        )
        if not accepted:
            return
        name = str(preset_name).strip()
        if not name:
            QMessageBox.warning(
                self,
                "Save Custom Solvent",
                "Enter a solvent name before saving.",
            )
            return
        formula = self.solvent_formula_edit.text().strip()
        density = float(self.solvent_density_spin.value())
        try:
            preset = ContrastSolventPreset(
                name=name,
                formula=formula,
                density_g_per_ml=density,
                builtin=False,
            )
        except ValueError as exc:
            QMessageBox.warning(
                self,
                "Save Custom Solvent",
                str(exc),
            )
            return
        if name in self._solvent_presets:
            response = QMessageBox.question(
                self,
                "Overwrite custom solvent?",
                f"A solvent named '{name}' already exists. Overwrite it?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if response != QMessageBox.StandardButton.Yes:
                return
        save_custom_solvent_preset(preset)
        self._reload_solvent_presets(selected_name=name)
        self._append_status(f"Saved custom solvent preset: {name}.")

    @Slot()
    def _delete_current_solvent_preset(self) -> None:
        preset_name = self._selected_solvent_preset_name()
        if preset_name is None:
            QMessageBox.information(
                self,
                "Delete Custom Solvent",
                "Select a saved custom solvent first.",
            )
            return
        preset = self._solvent_presets.get(preset_name)
        if preset is None or preset.builtin:
            QMessageBox.information(
                self,
                "Delete Custom Solvent",
                "Only custom solvents can be deleted.",
            )
            return
        response = QMessageBox.question(
            self,
            "Delete Custom Solvent",
            f"Delete the custom solvent preset '{preset_name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if response != QMessageBox.StandardButton.Yes:
            return
        delete_custom_solvent_preset(preset_name)
        self._reload_solvent_presets(selected_name="Water")
        self._append_status(f"Deleted custom solvent preset: {preset_name}.")

    @Slot()
    def _sync_density_method_controls(self) -> None:
        method = str(self.solvent_method_combo.currentData() or "").strip()
        using_direct = method == CONTRAST_SOLVENT_METHOD_DIRECT
        using_reference = method == CONTRAST_SOLVENT_METHOD_REFERENCE
        using_neat = method == CONTRAST_SOLVENT_METHOD_NEAT
        for widget in (
            self.solvent_preset_combo,
            self.solvent_formula_edit,
            self.solvent_density_spin,
            self.save_custom_solvent_button,
            self.delete_custom_solvent_button,
        ):
            widget.setEnabled(using_neat)
        self.reference_solvent_file_edit.setEnabled(using_reference)
        self.reference_solvent_browse_button.setEnabled(using_reference)
        self.direct_density_spin.setEnabled(using_direct)
        if using_reference:
            self.solvent_method_hint_label.setText(
                "Reference structure mode estimates a uniform solvent electron "
                "density from the full XYZ/PDB coordinate box spanned by the selected file."
            )
        elif using_direct:
            self.solvent_method_hint_label.setText(
                "Direct value mode uses the electron density you provide in "
                "e-/ Å³ without requiring a solvent structure file or formula. "
                "Use 0.0 e-/ Å³ to model vacuum."
            )
        else:
            self.solvent_method_hint_label.setText(
                "Quick estimate mode uses the selected solvent stoichiometry and "
                "density. Built-in presets include Water, Vacuum, DMF, and DMSO."
            )

    @Slot()
    def _choose_reference_solvent_file(self) -> None:
        start_dir = (
            str(
                Path(self.reference_solvent_file_edit.text())
                .expanduser()
                .resolve()
                .parent
            )
            if self.reference_solvent_file_edit.text().strip()
            else str(self._project_dir or Path.cwd())
        )
        selected_path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "Choose Reference Solvent Structure",
            start_dir,
            "Structure files (*.pdb *.xyz);;All files (*)",
        )
        if not selected_path:
            return
        self.reference_solvent_file_edit.setText(
            str(Path(selected_path).expanduser().resolve())
        )

    def _contrast_settings_from_controls(
        self,
    ) -> ContrastSolventDensitySettings:
        method = str(self.solvent_method_combo.currentData() or "").strip()
        if method == CONTRAST_SOLVENT_METHOD_REFERENCE:
            reference_path = self.reference_solvent_file_edit.text().strip()
            if not reference_path:
                raise ValueError(
                    "Choose a reference solvent XYZ or PDB file before computing the solvent electron density."
                )
            return ContrastSolventDensitySettings.from_values(
                method=CONTRAST_SOLVENT_METHOD_REFERENCE,
                reference_structure_file=reference_path,
            )
        if method == CONTRAST_SOLVENT_METHOD_DIRECT:
            direct_density = float(self.direct_density_spin.value())
            if direct_density < 0.0:
                raise ValueError(
                    "Enter a non-negative direct solvent electron density before computing the solvent contrast."
                )
            return ContrastSolventDensitySettings.from_values(
                method=CONTRAST_SOLVENT_METHOD_DIRECT,
                direct_electron_density_e_per_a3=direct_density,
            )
        formula = self.solvent_formula_edit.text().strip()
        if not formula:
            raise ValueError(
                "Enter a solvent stoichiometry formula before computing the solvent electron density."
            )
        return ContrastSolventDensitySettings.from_values(
            method=CONTRAST_SOLVENT_METHOD_NEAT,
            solvent_formula=formula,
            solvent_density_g_per_ml=self.solvent_density_spin.value(),
        )

    def _contrast_display_name_from_controls(self) -> str:
        method = str(self.solvent_method_combo.currentData() or "").strip()
        if method == CONTRAST_SOLVENT_METHOD_REFERENCE:
            reference_path = self.reference_solvent_file_edit.text().strip()
            return (
                Path(reference_path).stem
                if reference_path
                else "Reference solvent"
            )
        if method == CONTRAST_SOLVENT_METHOD_DIRECT:
            return "Direct solvent"
        preset_name = self._selected_solvent_preset_name()
        if preset_name:
            return preset_name
        return self.solvent_formula_edit.text().strip() or "Solvent"

    def _refresh_contrast_display(self) -> None:
        if self._active_contrast_settings is None:
            self.active_contrast_value.setText(
                "No active solvent electron density yet."
            )
            self.active_contrast_value.setStyleSheet("color: #475569;")
            self.contrast_notice_value.setText(
                "Compute a solvent electron density to add the solvent line, "
                "cutoff, and residual traces."
            )
            self.contrast_notice_value.setStyleSheet("color: #475569;")
            return
        contrast = (
            None
            if self._profile_result is None
            else self._profile_result.solvent_contrast
        )
        if contrast is None:
            active_name = self._active_contrast_name or "Configured solvent"
            self.active_contrast_value.setText(
                f"{active_name} is configured and will be applied to the next available density profile."
            )
            self.active_contrast_value.setStyleSheet("color: #92400e;")
            self.contrast_notice_value.setText(
                "Run the density calculation or click Compute Solvent Electron Density "
                "to populate the comparison overlays."
            )
            self.contrast_notice_value.setStyleSheet("color: #92400e;")
            return
        cutoff_text = (
            f"highest-r cutoff={contrast.cutoff_radius_a:.3f} Å."
            if contrast.cutoff_radius_a is not None
            else "no high-r solvent crossing was found."
        )
        self.active_contrast_value.setText(
            f"{contrast.legend_label}; {cutoff_text}"
        )
        self.active_contrast_value.setStyleSheet("color: #166534;")
        self.contrast_notice_value.setText(
            "The smeared plot shows the solvent line, the gray solvent-subtracted trace, "
            "and the cutoff marker. The residual panel updates when smearing changes."
        )
        self.contrast_notice_value.setStyleSheet("color: #166534;")

    def _apply_active_contrast_to_profile(
        self,
        *,
        show_error: bool,
        announce: bool,
    ) -> bool:
        if self._profile_result is None:
            if show_error:
                self._show_error(
                    "No Density Profile",
                    "Run the electron-density calculation before computing the solvent contrast.",
                )
            return False
        if self._active_contrast_settings is None:
            return False
        try:
            self._profile_result = apply_solvent_contrast_to_profile_result(
                self._profile_result,
                self._active_contrast_settings,
                solvent_name=self._active_contrast_name,
            )
        except Exception as exc:
            self._refresh_contrast_display()
            if show_error:
                self._show_error("Solvent Contrast Error", str(exc))
            else:
                self._append_status(
                    f"Could not apply the solvent electron density: {exc}"
                )
                self.statusBar().showMessage(
                    "Could not apply the solvent electron density",
                    5000,
                )
            return False
        self._refresh_profile_plots()
        self._refresh_contrast_display()
        self._refresh_fourier_preview_from_controls(clear_transform=True)
        contrast = self._profile_result.solvent_contrast
        if announce and contrast is not None:
            cutoff_text = (
                f"Highest-r cutoff={contrast.cutoff_radius_a:.3f} Å."
                if contrast.cutoff_radius_a is not None
                else "No highest-r solvent crossing was found."
            )
            self._append_status(
                f"Applied solvent electron density {contrast.legend_label}. {cutoff_text}"
            )
        return True

    @Slot()
    def _compute_solvent_contrast(self) -> None:
        try:
            self._active_contrast_settings = (
                self._contrast_settings_from_controls()
            )
            self._active_contrast_name = (
                self._contrast_display_name_from_controls()
            )
        except Exception as exc:
            self._show_error("Solvent Contrast Error", str(exc))
            return
        if not self._apply_active_contrast_to_profile(
            show_error=True,
            announce=True,
        ):
            return
        self.statusBar().showMessage("Updated solvent electron density")

    def _apply_smearing_from_controls(self, *_args: object) -> None:
        self._refresh_smearing_display()
        if self._profile_result is None:
            return
        current_factor = float(
            self._profile_result.smearing_settings.debye_waller_factor
        )
        pending_factor = float(
            self._active_smearing_settings.debye_waller_factor
        )
        if abs(current_factor - pending_factor) <= 1.0e-12:
            return
        self._profile_result = apply_smearing_to_profile_result(
            self._profile_result,
            self._active_smearing_settings,
        )
        self._refresh_profile_plots()
        self._refresh_contrast_display()
        self.statusBar().showMessage("Updated Gaussian smearing preview")
        self._append_status(
            "Updated Gaussian smearing preview to "
            f"factor={self._profile_result.smearing_settings.debye_waller_factor:.6f} Å² "
            f"(sigma={self._profile_result.smearing_settings.gaussian_sigma_a:.4f} Å)."
        )
        self._refresh_fourier_preview_from_controls(clear_transform=True)

    def _available_fourier_r_domain(self) -> tuple[float, float] | None:
        if self._profile_result is not None:
            radial_values = np.asarray(
                self._profile_result.radial_centers, dtype=float
            )
            if radial_values.size >= 2:
                return 0.0, float(radial_values[-1])
        if self._active_mesh_geometry is None:
            return None
        radial_edges = np.asarray(
            self._active_mesh_geometry.radial_edges, dtype=float
        )
        if radial_edges.size < 2:
            return None
        radial_centers = (radial_edges[:-1] + radial_edges[1:]) * 0.5
        if radial_centers.size < 1:
            return None
        return 0.0, float(radial_centers[-1])

    def _sync_fourier_controls_to_domain(
        self,
        *,
        reset_bounds: bool,
    ) -> None:
        available_domain = self._available_fourier_r_domain()
        if available_domain is None:
            return
        domain_min, domain_max = available_domain
        minimum_span = max(min(domain_max - domain_min, 1.0e-3), 1.0e-6)
        self.fourier_rmin_spin.blockSignals(True)
        self.fourier_rmax_spin.blockSignals(True)
        try:
            self.fourier_rmin_spin.setRange(
                domain_min,
                max(domain_max - minimum_span, domain_min),
            )
            self.fourier_rmax_spin.setRange(
                min(domain_min + minimum_span, domain_max),
                domain_max,
            )
            current_rmin = float(self.fourier_rmin_spin.value())
            current_rmax = float(self.fourier_rmax_spin.value())
            if (
                reset_bounds
                or current_rmin < domain_min
                or current_rmin >= domain_max
            ):
                current_rmin = domain_min
            if (
                reset_bounds
                or current_rmax > domain_max
                or current_rmax <= current_rmin
            ):
                current_rmax = domain_max
            current_rmin = min(current_rmin, domain_max - minimum_span)
            current_rmax = max(current_rmax, current_rmin + minimum_span)
            self.fourier_rmin_spin.setValue(current_rmin)
            self.fourier_rmax_spin.setValue(min(current_rmax, domain_max))
        finally:
            self.fourier_rmin_spin.blockSignals(False)
            self.fourier_rmax_spin.blockSignals(False)

    def _sync_fourier_controls_to_settings(
        self,
        settings: ElectronDensityFourierTransformSettings,
    ) -> None:
        normalized = settings.normalized()
        widgets = (
            self.fourier_rmin_spin,
            self.fourier_rmax_spin,
            self.fourier_qmin_spin,
            self.fourier_qmax_spin,
            self.fourier_qstep_spin,
            self.fourier_resampling_points_spin,
            self.fourier_window_combo,
            self.fourier_use_solvent_subtracted_checkbox,
            self.fourier_log_q_checkbox,
            self.fourier_log_intensity_checkbox,
        )
        for widget in widgets:
            widget.blockSignals(True)
        try:
            self.fourier_rmin_spin.setValue(float(normalized.r_min))
            self.fourier_rmax_spin.setValue(float(normalized.r_max))
            window_index = max(
                self.fourier_window_combo.findData(normalized.window_function),
                0,
            )
            self.fourier_window_combo.setCurrentIndex(window_index)
            self.fourier_resampling_points_spin.setValue(
                int(normalized.resampling_points)
            )
            self.fourier_qmin_spin.setValue(float(normalized.q_min))
            self.fourier_qmax_spin.setValue(float(normalized.q_max))
            self.fourier_qstep_spin.setValue(float(normalized.q_step))
            self.fourier_use_solvent_subtracted_checkbox.setChecked(
                bool(normalized.use_solvent_subtracted_profile)
            )
            self.fourier_log_q_checkbox.setChecked(bool(normalized.log_q_axis))
            self.fourier_log_intensity_checkbox.setChecked(
                bool(normalized.log_intensity_axis)
            )
        finally:
            for widget in widgets:
                widget.blockSignals(False)

    def _refresh_fourier_info_labels(
        self,
        preview: ElectronDensityFourierTransformPreview | None,
    ) -> None:
        available_domain = self._available_fourier_r_domain()
        if available_domain is None:
            self.fourier_available_range_value.setText(
                "Awaiting mesh/profile domain."
            )
        else:
            self.fourier_available_range_value.setText(
                f"{available_domain[0]:.4f} to {available_domain[1]:.4f} Å"
            )
        if preview is None:
            try:
                settings = self._fourier_settings_from_controls()
            except Exception as exc:
                self.fourier_nyquist_value.setText(
                    "Adjust the Fourier-transform bounds to satisfy the requested sampling."
                )
                self.fourier_notice_value.setText(str(exc))
                self.fourier_notice_value.setStyleSheet("color: #991b1b;")
                return
            self._active_fourier_settings = settings
            span = max(settings.r_max - settings.r_min, 1.0e-12)
            resampling_step = span / max(settings.resampling_points - 1, 1)
            nyquist_qmax = float(np.pi / max(resampling_step, 1.0e-12))
            independent_qstep = float(np.pi / span)
            self.fourier_nyquist_value.setText(
                f"Δr={resampling_step:.4f} Å, qmax≈{nyquist_qmax:.3f} Å⁻¹, "
                f"independent Δq≈{independent_qstep:.3f} Å⁻¹."
            )
            if self._profile_result is None:
                self.fourier_notice_value.setText(
                    "Run the density calculation to prepare the Fourier-transform preview."
                )
                self.fourier_notice_value.setStyleSheet("color: #475569;")
            else:
                self.fourier_notice_value.setText(
                    "Adjust the transform settings, then evaluate the q-space scattering profile."
                )
                self.fourier_notice_value.setStyleSheet("color: #475569;")
            return
        self._active_fourier_settings = preview.settings
        self.fourier_nyquist_value.setText(
            f"Δr={preview.resampling_step_a:.4f} Å, "
            f"qmax≈{preview.nyquist_q_max_a_inverse:.3f} Å⁻¹, "
            f"independent Δq≈{preview.independent_q_step_a_inverse:.3f} Å⁻¹."
        )
        if preview.notes:
            self.fourier_notice_value.setText(
                f"Source: {preview.source_profile_label}. "
                + " ".join(preview.notes)
            )
            self.fourier_notice_value.setStyleSheet("color: #92400e;")
        else:
            self.fourier_notice_value.setText(
                "Source: "
                f"{preview.source_profile_label}. "
                "Requested transform settings are within the current sampling limits."
            )
            self.fourier_notice_value.setStyleSheet("color: #166534;")

    def _refresh_fourier_preview_from_controls(
        self,
        *_args: object,
        clear_transform: bool = True,
    ) -> None:
        try:
            settings = self._fourier_settings_from_controls()
        except Exception as exc:
            self._active_fourier_settings = (
                ElectronDensityFourierTransformSettings()
            )
            self.fourier_notice_value.setText(str(exc))
            self.fourier_notice_value.setStyleSheet("color: #991b1b;")
            self.fourier_preview_plot.draw_placeholder()
            if clear_transform:
                self._fourier_result = None
                self.scattering_plot.draw_placeholder()
            return
        self._active_fourier_settings = settings
        if self._profile_result is None:
            self._fourier_preview = None
            self.fourier_preview_plot.draw_placeholder()
            if clear_transform:
                self._fourier_result = None
                self.scattering_plot.draw_placeholder()
            self._refresh_fourier_info_labels(None)
            return
        try:
            preview = prepare_electron_density_fourier_transform(
                self._profile_result,
                settings,
            )
        except Exception as exc:
            self._fourier_preview = None
            self.fourier_preview_plot.draw_placeholder()
            if clear_transform:
                self._fourier_result = None
                self.scattering_plot.draw_placeholder()
            self._refresh_fourier_info_labels(None)
            self.fourier_notice_value.setText(str(exc))
            self.fourier_notice_value.setStyleSheet("color: #991b1b;")
            return
        self._fourier_preview = preview
        self.fourier_preview_plot.set_preview(preview)
        if self.auto_expand_checkbox.isChecked():
            self.fourier_preview_section.expand()
        self._refresh_fourier_info_labels(preview)
        if clear_transform:
            self._fourier_result = None
            self.scattering_plot.draw_placeholder()

    @Slot(bool)
    def _apply_fourier_axis_scale_preferences(self, _checked: bool) -> None:
        if self._fourier_result is None:
            return
        self.scattering_plot.set_transform_result(
            self._fourier_result,
            log_q_axis=self.fourier_log_q_checkbox.isChecked(),
            log_intensity_axis=self.fourier_log_intensity_checkbox.isChecked(),
        )

    @Slot()
    def _evaluate_fourier_transform(self) -> None:
        if self._profile_result is None:
            self._show_error(
                "No Density Profile",
                "Run the electron-density calculation before evaluating the Fourier transform.",
            )
            return
        try:
            transform_result = compute_electron_density_scattering_profile(
                self._profile_result,
                self._fourier_settings_from_controls(),
            )
        except Exception as exc:
            self._show_error("Fourier Transform Error", str(exc))
            return
        self._fourier_preview = transform_result.preview
        self._fourier_result = transform_result
        self._sync_fourier_controls_to_settings(
            transform_result.preview.settings
        )
        self._refresh_fourier_info_labels(transform_result.preview)
        self.fourier_preview_plot.set_preview(transform_result.preview)
        self.scattering_plot.set_transform_result(
            transform_result,
            log_q_axis=self.fourier_log_q_checkbox.isChecked(),
            log_intensity_axis=self.fourier_log_intensity_checkbox.isChecked(),
        )
        if self.auto_expand_checkbox.isChecked():
            self.fourier_preview_section.expand()
            self.scattering_section.expand()
        self._append_status(
            "Evaluated the Born-approximation q-space transform with "
            f"window={transform_result.preview.settings.window_function}, "
            f"r={transform_result.preview.settings.r_min:.4f} to "
            f"{transform_result.preview.settings.r_max:.4f} Å, "
            f"{len(transform_result.q_values)} q points."
        )
        for note in transform_result.preview.notes:
            self._append_status(note)
        self.statusBar().showMessage("Evaluated Fourier transform")

    def _sync_controls_to_structure(self) -> None:
        if self._structure is None:
            return
        self.rmax_spin.setValue(max(float(self._structure.rmax), 0.01))
        self._sync_reference_element_controls()
        self._refresh_center_display()
        self._refresh_structure_summary()

    def _refresh_active_mesh_display(self) -> None:
        if self._active_mesh_geometry is None:
            self.active_mesh_value.setText(
                "No active structure yet. The current defaults are "
                f"rstep={self._active_mesh_settings.rstep:.3f} Å, "
                f"theta={self._active_mesh_settings.theta_divisions}, "
                f"phi={self._active_mesh_settings.phi_divisions}, "
                f"rmax={self._active_mesh_settings.rmax:.3f} Å."
            )
            return
        geometry = self._active_mesh_geometry
        self.active_mesh_value.setText(
            f"rstep={geometry.settings.rstep:.3f} Å, "
            f"theta={geometry.settings.theta_divisions}, "
            f"phi={geometry.settings.phi_divisions}, "
            f"rmax={geometry.settings.rmax:.3f} Å, "
            f"shells={geometry.shell_count}, "
            f"domain=0 to {geometry.domain_max_radius:.3f} Å, "
            f"center={self._format_point(self._structure.active_center) if self._structure is not None else 'Unavailable'}"
        )

    @Slot()
    def _refresh_mesh_notice(self) -> None:
        pending = self._mesh_settings_from_controls()
        if pending == self._active_mesh_settings:
            self.pending_mesh_value.setText(
                "Pending field values match the rendered mesh."
            )
            self.pending_mesh_value.setStyleSheet("color: #166534;")
            return
        self.pending_mesh_value.setText(
            "Pending field values differ from the rendered mesh. Press "
            "Update Mesh Settings to redraw the spherical overlay."
        )
        self.pending_mesh_value.setStyleSheet("color: #92400e;")

    def _load_input_path(self, path: Path) -> None:
        inspection = inspect_structure_input(path)
        structure = load_electron_density_structure(inspection.reference_file)
        self._inspection = inspection
        self._structure = structure
        self._reset_density_results()
        self._reset_progress_display("Idle")

        if inspection.input_mode == "folder":
            self.input_mode_value.setText(
                f"Folder ({inspection.total_files} files). Previewing "
                f"{inspection.reference_file.name}; running will average all valid structures."
            )
        else:
            self.input_mode_value.setText("Single structure file")
        self.reference_file_value.setText(str(inspection.reference_file))
        self._sync_controls_to_structure()

        self.output_dir_edit.setText(
            str(
                suggest_output_dir(
                    inspection.selection_path,
                    project_dir=self._project_dir,
                )
            )
        )
        self.output_basename_edit.setText(suggest_output_basename(inspection))

        self._apply_mesh_settings(
            self._mesh_settings_from_controls(),
            announce=False,
        )
        self._sync_fourier_controls_to_domain(reset_bounds=True)
        self._refresh_fourier_info_labels(None)
        self._refresh_contrast_display()
        self.structure_viewer.set_structure(
            structure,
            mesh_geometry=self._active_mesh_geometry,
            reset_view=True,
        )

        self._append_status(
            f"Loaded {structure.file_path} with {structure.atom_count} atoms "
            f"using the {self.center_mode_value.text().lower()} as the active origin."
        )
        for note in inspection.notes():
            self._append_status(note)
        self.statusBar().showMessage(f"Loaded {structure.file_path.name}")

    def _apply_mesh_settings(
        self,
        settings: ElectronDensityMeshSettings,
        *,
        announce: bool,
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
        if self._structure is not None:
            if preserve_viewer_display:
                self.structure_viewer.set_structure_preserving_display(
                    self._structure,
                    mesh_geometry=self._active_mesh_geometry,
                )
            else:
                self.structure_viewer.set_structure(
                    self._structure,
                    mesh_geometry=self._active_mesh_geometry,
                    reset_view=False,
                )
        if self._profile_result is None:
            self._sync_fourier_controls_to_domain(reset_bounds=False)
            self._refresh_fourier_info_labels(None)
        if announce and self._active_mesh_geometry is not None:
            self._append_status(
                "Updated the spherical mesh overlay with "
                f"rstep={self._active_mesh_settings.rstep:.3f} Å, "
                f"theta={self._active_mesh_settings.theta_divisions}, "
                f"phi={self._active_mesh_settings.phi_divisions}, "
                f"rmax={self._active_mesh_settings.rmax:.3f} Å."
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
        self._reset_density_results()
        self._sync_controls_to_structure()
        self._apply_mesh_settings(
            self._mesh_settings_from_controls(),
            announce=False,
            preserve_viewer_display=True,
        )
        self._sync_fourier_controls_to_domain(reset_bounds=True)
        self._refresh_fourier_info_labels(None)
        self._refresh_contrast_display()
        if self._structure.center_mode == "nearest_atom":
            self._append_status(
                "Snapped the active center to the nearest atom: "
                f"{self._structure.nearest_atom_element} "
                f"(atom #{self._structure.nearest_atom_index + 1})."
            )
        elif self._structure.center_mode == "reference_element":
            self._append_status(
                "Snapped the active center to the "
                f"{self._structure.reference_element} reference-element "
                "geometric center."
            )
        else:
            self._append_status(
                "Reset the active center to the calculated mass-weighted center of mass."
            )
        self.statusBar().showMessage("Updated active center")

    @Slot()
    def _apply_mesh_from_controls(self) -> None:
        self._apply_mesh_settings(
            self._mesh_settings_from_controls(), announce=True
        )
        self.statusBar().showMessage("Updated mesh overlay")

    @Slot()
    def _choose_input_file(self) -> None:
        selected, _filter = QFileDialog.getOpenFileName(
            self,
            "Select XYZ or PDB file",
            str(Path.home()),
            "Structure files (*.xyz *.pdb);;All files (*)",
        )
        if not selected:
            return
        self.input_path_edit.setText(
            str(Path(selected).expanduser().resolve())
        )
        self._load_input_from_edit()

    @Slot()
    def _choose_input_folder(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "Select folder of XYZ/PDB files",
            str(Path.home()),
        )
        if not selected:
            return
        self.input_path_edit.setText(
            str(Path(selected).expanduser().resolve())
        )
        self._load_input_from_edit()

    @Slot()
    def _choose_output_dir(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "Select output directory",
            str(Path.home()),
        )
        if not selected:
            return
        self.output_dir_edit.setText(
            str(Path(selected).expanduser().resolve())
        )

    @Slot()
    def _load_input_from_edit(self) -> None:
        text = self.input_path_edit.text().strip()
        if not text:
            self._show_error(
                "Input Required",
                "Choose a structure file or folder before loading.",
            )
            return
        try:
            self._load_input_path(Path(text))
        except Exception as exc:
            self._show_error("Input Error", str(exc))

    def _output_basename(self) -> str:
        basename = self.output_basename_edit.text().strip()
        return basename or "electron_density_profile"

    def _write_outputs(
        self,
        result: ElectronDensityProfileResult,
    ) -> ElectronDensityOutputArtifacts | None:
        output_dir_text = self.output_dir_edit.text().strip()
        if not output_dir_text:
            return None
        return write_electron_density_profile_outputs(
            result,
            output_dir_text,
            self._output_basename(),
        )

    def _reset_progress_display(self, message: str = "Idle") -> None:
        self.calculation_progress_message.setText(str(message))
        self.calculation_progress_bar.setRange(0, 1)
        self.calculation_progress_bar.setValue(0)

    def _set_calculation_running(self, running: bool) -> None:
        enabled = not bool(running)
        for widget in (
            self.load_input_button,
            self.run_button,
            self.update_mesh_button,
            self.snap_center_button,
            self.reset_center_button,
            self.rstep_spin,
            self.theta_divisions_spin,
            self.phi_divisions_spin,
            self.rmax_spin,
            self.smearing_factor_spin,
            self.solvent_method_combo,
            self.solvent_preset_combo,
            self.save_custom_solvent_button,
            self.delete_custom_solvent_button,
            self.solvent_formula_edit,
            self.solvent_density_spin,
            self.direct_density_spin,
            self.reference_solvent_file_edit,
            self.reference_solvent_browse_button,
            self.compute_solvent_density_button,
            self.fourier_rmin_spin,
            self.fourier_rmax_spin,
            self.fourier_window_combo,
            self.fourier_resampling_points_spin,
            self.fourier_qmin_spin,
            self.fourier_qmax_spin,
            self.fourier_qstep_spin,
            self.fourier_use_solvent_subtracted_checkbox,
            self.fourier_log_q_checkbox,
            self.fourier_log_intensity_checkbox,
            self.evaluate_fourier_button,
            self.input_path_edit,
            self.output_dir_edit,
            self.output_basename_edit,
        ):
            widget.setEnabled(enabled)

    @Slot(int, int, str)
    def _on_calculation_progress(
        self,
        current: int,
        total: int,
        message: str,
    ) -> None:
        bounded_total = max(int(total), 1)
        bounded_current = min(max(int(current), 0), bounded_total)
        self.calculation_progress_bar.setRange(0, bounded_total)
        self.calculation_progress_bar.setValue(bounded_current)
        text = str(message).strip()
        if text:
            self.calculation_progress_message.setText(text)
            self.statusBar().showMessage(text)
            if text != self._last_progress_message:
                self._append_status(text)
                self._last_progress_message = text

    @Slot(object)
    def _on_calculation_finished(self, payload: object) -> None:
        data = payload if isinstance(payload, dict) else {}
        result = data.get("result")
        artifacts = data.get("artifacts")
        if not isinstance(result, ElectronDensityProfileResult):
            self._on_calculation_failed(
                "Electron-density calculation finished without a valid result."
            )
            return

        self._profile_result = result
        self._fourier_result = None
        if self._active_contrast_settings is not None:
            self._apply_active_contrast_to_profile(
                show_error=False,
                announce=False,
            )
        self._refresh_profile_plots()
        self._refresh_contrast_display()
        self._sync_fourier_controls_to_domain(reset_bounds=False)
        self._refresh_fourier_preview_from_controls(clear_transform=True)
        self.calculation_progress_bar.setRange(0, 1)
        self.calculation_progress_bar.setValue(1)
        self.calculation_progress_message.setText(
            "Electron-density calculation complete."
        )
        self._set_calculation_running(False)

        active_result = self._profile_result
        average_total_electrons = float(
            np.sum(active_result.shell_electron_counts)
        )
        if active_result.input_mode == "folder":
            self._append_status(
                f"Computed average radial density across {active_result.source_structure_count} structures: "
                f"{len(active_result.radial_centers)} shells and an average of "
                f"{average_total_electrons:.0f} electrons placed on the mesh per structure."
            )
        else:
            self._append_status(
                f"Computed radial density for {active_result.structure.file_path.name}: "
                f"{len(active_result.radial_centers)} shells, "
                f"{average_total_electrons:.0f} total electrons placed on the mesh."
            )
        self._append_status(
            "Applied Gaussian smearing with "
            f"factor={active_result.smearing_settings.debye_waller_factor:.6f} Å² "
            f"(sigma={active_result.smearing_settings.gaussian_sigma_a:.4f} Å)."
        )
        if active_result.solvent_contrast is not None:
            cutoff_text = (
                f"highest-r cutoff={active_result.solvent_contrast.cutoff_radius_a:.3f} Å."
                if active_result.solvent_contrast.cutoff_radius_a is not None
                else "no highest-r solvent crossing was found."
            )
            self._append_status(
                f"Applied solvent comparison with {active_result.solvent_contrast.legend_label}; "
                f"{cutoff_text}"
            )
        if active_result.source_structure_count > 1:
            mean_rmax = float(
                np.mean(
                    [entry.rmax for entry in active_result.member_summaries]
                )
            )
            max_rmax = float(
                np.max(
                    [entry.rmax for entry in active_result.member_summaries]
                )
            )
            self._append_status(
                f"Ensemble rmax values ranged up to {max_rmax:.3f} Å "
                f"(mean {mean_rmax:.3f} Å), and variance shading can be toggled in the plot area."
            )
        if active_result.excluded_atom_count > 0:
            self._append_status(
                f"Excluded {active_result.excluded_atom_count} atoms and "
                f"{active_result.excluded_electron_count:.0f} electrons beyond "
                f"rmax={self._active_mesh_settings.rmax:.3f} Å."
            )
        if isinstance(artifacts, ElectronDensityOutputArtifacts):
            self._append_status(f"Wrote profile CSV to {artifacts.csv_path}")
            self._append_status(f"Wrote summary JSON to {artifacts.json_path}")
        self.statusBar().showMessage("Electron-density calculation complete")

    @Slot(str)
    def _on_calculation_failed(self, message: str) -> None:
        self._set_calculation_running(False)
        self._reset_progress_display("Calculation failed.")
        self._show_error("Calculation Error", str(message))

    @Slot()
    def _cleanup_calculation_thread(self) -> None:
        if self._calculation_worker is not None:
            self._calculation_worker.deleteLater()
            self._calculation_worker = None
        if self._calculation_thread is not None:
            self._calculation_thread.deleteLater()
            self._calculation_thread = None

    @Slot()
    def _run_calculation(self) -> None:
        if self._structure is None or self._inspection is None:
            self._show_error(
                "No Structure Loaded",
                "Load an XYZ or PDB structure before running the calculation.",
            )
            return
        if self._active_mesh_geometry is None:
            self._apply_mesh_settings(
                self._mesh_settings_from_controls(),
                announce=False,
            )
        if (
            self._calculation_thread is not None
            and self._calculation_thread.isRunning()
        ):
            self._show_error(
                "Calculation Already Running",
                "Wait for the current electron-density calculation to finish before starting another run.",
            )
            return
        self._last_progress_message = ""
        self._set_calculation_running(True)
        self.calculation_progress_bar.setRange(
            0,
            max(len(self._inspection.structure_files) * 2 + 3, 1),
        )
        self.calculation_progress_bar.setValue(0)
        self.calculation_progress_message.setText(
            "Preparing electron-density calculation..."
        )
        self.statusBar().showMessage("Computing electron-density profile...")
        self._calculation_thread = QThread(self)
        self._calculation_worker = ElectronDensityCalculationWorker(
            inspection=self._inspection,
            mesh_settings=self._active_mesh_settings,
            smearing_settings=self._smearing_settings_from_controls(),
            center_mode=self._structure.center_mode,
            reference_element=self._structure.reference_element,
            output_dir=self.output_dir_edit.text().strip(),
            output_basename=self._output_basename(),
        )
        self._calculation_worker.moveToThread(self._calculation_thread)
        self._calculation_thread.started.connect(self._calculation_worker.run)
        self._calculation_worker.progress.connect(
            self._on_calculation_progress
        )
        self._calculation_worker.finished.connect(
            self._on_calculation_finished
        )
        self._calculation_worker.failed.connect(self._on_calculation_failed)
        self._calculation_worker.finished.connect(
            self._calculation_thread.quit
        )
        self._calculation_worker.failed.connect(self._calculation_thread.quit)
        self._calculation_thread.finished.connect(
            self._cleanup_calculation_thread
        )
        self._calculation_thread.start(QThread.Priority.LowPriority)

    @Slot()
    def _export_plot_traces(self) -> None:
        if self._profile_result is None and self._fourier_result is None:
            self._show_error(
                "No Data to Export",
                "Run the electron-density calculation before exporting plot traces.",
            )
            return

        # Resolve default output directory
        default_dir: Path | None = None
        if self._project_dir is not None:
            from saxshell.saxs.project_manager.project import (
                build_project_paths,
            )

            paths = build_project_paths(self._project_dir)
            default_dir = paths.exported_data_dir
        else:
            output_dir_text = self.output_dir_edit.text().strip()
            if output_dir_text:
                default_dir = Path(output_dir_text).expanduser().resolve()

        if default_dir is not None:
            chosen_dir = default_dir
        else:
            chosen = QFileDialog.getExistingDirectory(
                self,
                "Choose Export Directory for Plot Traces",
                "",
            )
            if not chosen:
                return
            chosen_dir = Path(chosen).expanduser().resolve()

        chosen_dir.mkdir(parents=True, exist_ok=True)
        basename = self._output_basename()
        csv_path = chosen_dir / f"{basename}_plot_traces.csv"

        # Collect all trace columns
        headers: list[str] = []
        columns: list[list[str]] = []

        result = self._profile_result
        if result is not None:
            r = np.asarray(result.radial_centers, dtype=float)
            raw = np.asarray(result.orientation_average_density, dtype=float)
            raw_std = np.asarray(
                result.orientation_density_stddev, dtype=float
            )
            smeared = np.asarray(
                result.smeared_orientation_average_density, dtype=float
            )
            smeared_std = np.asarray(
                result.smeared_orientation_density_stddev, dtype=float
            )
            headers += [
                "r_center_a",
                "raw_density_e_per_a3",
                "raw_stddev_e_per_a3",
                "smeared_density_e_per_a3",
                "smeared_stddev_e_per_a3",
            ]
            columns += [
                [f"{v:.10g}" for v in r],
                [f"{v:.10g}" for v in raw],
                [f"{v:.10g}" for v in raw_std],
                [f"{v:.10g}" for v in smeared],
                [f"{v:.10g}" for v in smeared_std],
            ]
            if result.solvent_contrast is not None:
                residual = np.asarray(
                    result.solvent_contrast.solvent_subtracted_smeared_density,
                    dtype=float,
                )
                headers += [
                    "solvent_density_e_per_a3",
                    "solvent_subtracted_density_e_per_a3",
                ]
                columns += [
                    [
                        f"{float(result.solvent_contrast.solvent_density_e_per_a3):.10g}"
                        for _value in r
                    ],
                    [f"{v:.10g}" for v in residual],
                ]

        preview = self._fourier_preview
        if preview is not None:
            r_pre = np.asarray(preview.resampled_r_values, dtype=float)
            d_pre = np.asarray(preview.resampled_density_values, dtype=float)
            w_pre = np.asarray(preview.windowed_density_values, dtype=float)
            headers += [
                "fourier_preview_r_a",
                "fourier_preview_resampled_density",
                "fourier_preview_windowed_density",
            ]
            columns += [
                [f"{v:.10g}" for v in r_pre],
                [f"{v:.10g}" for v in d_pre],
                [f"{v:.10g}" for v in w_pre],
            ]

        ft = self._fourier_result
        if ft is not None:
            q_vals = np.asarray(ft.q_values, dtype=float)
            amp = np.asarray(ft.scattering_amplitude, dtype=float)
            intensity = np.asarray(ft.intensity, dtype=float)
            headers += [
                "q_a_inv",
                "scattering_amplitude",
                "intensity",
            ]
            columns += [
                [f"{v:.10g}" for v in q_vals],
                [f"{v:.10g}" for v in amp],
                [f"{v:.10g}" for v in intensity],
            ]

        max_rows = max((len(col) for col in columns), default=0)
        try:
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(headers)
                for row_idx in range(max_rows):
                    writer.writerow(
                        col[row_idx] if row_idx < len(col) else ""
                        for col in columns
                    )
        except Exception as exc:
            self._show_error("Export Failed", str(exc))
            return

        self._append_status(f"Exported plot traces to {csv_path}")
        self.statusBar().showMessage(f"Exported plot traces to {csv_path}")

    def _append_status(self, message: str) -> None:
        text = str(message).strip()
        if not text:
            return
        existing = self.status_text.toPlainText().strip()
        if existing:
            self.status_text.appendPlainText("")
        self.status_text.appendPlainText(text)

    def _show_error(self, title: str, message: str) -> None:
        QMessageBox.critical(self, title, message)
        self.statusBar().showMessage(message, 5000)


def _forget_open_window(window: ElectronDensityMappingMainWindow) -> None:
    if window in _OPEN_WINDOWS:
        _OPEN_WINDOWS.remove(window)


def launch_electron_density_mapping_ui(
    *,
    initial_project_dir: str | Path | None = None,
    initial_input_path: str | Path | None = None,
) -> ElectronDensityMappingMainWindow:
    app = QApplication.instance()
    if app is None:
        prepare_saxshell_application_identity()
        app = QApplication(sys.argv)
    configure_saxshell_application(app)
    window = ElectronDensityMappingMainWindow(
        initial_project_dir=(
            None
            if initial_project_dir is None
            else Path(initial_project_dir).expanduser().resolve()
        ),
        initial_input_path=(
            None
            if initial_input_path is None
            else Path(initial_input_path).expanduser().resolve()
        ),
    )
    window.show()
    window.raise_()
    _OPEN_WINDOWS.append(window)
    window.destroyed.connect(
        lambda _obj=None, win=window: _forget_open_window(win)
    )
    return window


__all__ = [
    "ElectronDensityMappingMainWindow",
    "launch_electron_density_mapping_ui",
]
