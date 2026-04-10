from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
from PySide6.QtCore import (
    QItemSelectionModel,
    QObject,
    QSettings,
    Qt,
    QThread,
    QTimer,
    Signal,
    Slot,
)
from PySide6.QtGui import QAction, QCloseEvent, QColor
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
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
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from saxshell.saxs.contrast.electron_density import (
    CONTRAST_SOLVENT_METHOD_DIRECT,
    CONTRAST_SOLVENT_METHOD_NEAT,
    CONTRAST_SOLVENT_METHOD_REFERENCE,
    ContrastElectronDensityEstimate,
    ContrastSolventDensitySettings,
)
from saxshell.saxs.contrast.solvents import (
    ContrastSolventPreset,
    delete_custom_solvent_preset,
    load_solvent_presets,
    ordered_solvent_preset_names,
    save_custom_solvent_preset,
)
from saxshell.saxs.debye import discover_cluster_bins
from saxshell.saxs.debye_waller import (
    find_saved_project_debye_waller_analysis,
    load_debye_waller_analysis_result,
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
    ElectronDensityCalculationCanceled,
    ElectronDensityContiguousFrameSetSummary,
    ElectronDensityDebyeScatteringAverageResult,
    ElectronDensityDebyeWallerPairTerm,
    ElectronDensityFourierTransformPreview,
    ElectronDensityFourierTransformSettings,
    ElectronDensityInputInspection,
    ElectronDensityMeshGeometry,
    ElectronDensityMeshSettings,
    ElectronDensityOutputArtifacts,
    ElectronDensityProfileResult,
    ElectronDensityScatteringTransformResult,
    ElectronDensitySmearingSettings,
    ElectronDensitySolventContrastResult,
    ElectronDensityStructure,
    apply_smearing_to_profile_result,
    apply_solvent_contrast_to_profile_result,
    build_electron_density_mesh,
    compute_average_debye_scattering_profile_for_input,
    compute_electron_density_profile_for_input,
    compute_electron_density_scattering_profile,
    compute_single_atom_debye_scattering_profile_for_input,
    inspect_structure_input,
    load_electron_density_structure,
    prepare_electron_density_fourier_transform,
    prepare_single_atom_debye_scattering_preview,
    recenter_electron_density_structure,
    suggest_output_basename,
    suggest_output_dir,
    write_electron_density_profile_outputs,
)
from saxshell.saxs.ui._pane_snap import PaneSnapFilter
from saxshell.saxs.ui.branding import (
    configure_saxshell_application,
    load_saxshell_icon,
    prepare_saxshell_application_identity,
)
from saxshell.saxs.ui.progress_dialog import SAXSProgressDialog

_OPEN_WINDOWS: list["ElectronDensityMappingMainWindow"] = []
_CLUSTER_TRACE_COLORS = (
    "#b45309",
    "#2563eb",
    "#15803d",
    "#9333ea",
    "#dc2626",
    "#0891b2",
)
AUTO_SNAP_PANES_KEY = "auto_snap_panes_enabled"
_FT_COLUMN_STOICHIOMETRY = 0
_FT_COLUMN_STATUS = 1
_FT_COLUMN_PROFILE = 2
_FT_COLUMN_RMIN = 3
_FT_COLUMN_RMAX = 4
_FT_COLUMN_QMIN = 5
_FT_COLUMN_QMAX = 6
_FT_COLUMN_QSTEP = 7
_FT_COLUMN_RESAMPLE = 8
_FT_COLUMN_WINDOW = 9
_CLUSTER_GROUP_COLUMN_STOICHIOMETRY = 0
_CLUSTER_GROUP_COLUMN_FILES = 1
_CLUSTER_GROUP_COLUMN_AVG_ATOMS = 2
_CLUSTER_GROUP_COLUMN_CENTER_MODE = 3
_CLUSTER_GROUP_COLUMN_CENTER_REFERENCE = 4
_CLUSTER_GROUP_COLUMN_CENTER_ELEMENT = 5
_CLUSTER_GROUP_COLUMN_DENSITY = 6
_CLUSTER_GROUP_COLUMN_FOURIER = 7
_CLUSTER_GROUP_COLUMN_TRACE_COLOR = 8
_CLUSTER_GROUP_COLUMN_SMEARING = 9
_CLUSTER_GROUP_COLUMN_SOLVENT = 10
_CLUSTER_GROUP_COLUMN_CUTOFF = 11
_CLUSTER_GROUP_COLUMN_FT_SETTINGS = 12
_CLUSTER_GROUP_COLUMN_REFERENCE_FILE = 13


def _quick_structure_atom_count(file_path: Path) -> int | None:
    resolved = Path(file_path).expanduser().resolve()
    try:
        with resolved.open(
            "r",
            encoding="utf-8",
            errors="ignore",
        ) as handle:
            if resolved.suffix.lower() == ".xyz":
                atom_count_line = handle.readline()
                atom_count = int(atom_count_line.strip())
                return atom_count if atom_count > 0 else None
            if resolved.suffix.lower() == ".pdb":
                atom_count = sum(
                    1
                    for line in handle
                    if line.startswith("ATOM") or line.startswith("HETATM")
                )
                return atom_count if atom_count > 0 else None
    except Exception:
        return None
    return None


def _cluster_trace_color_for_index(index: int) -> str:
    return _CLUSTER_TRACE_COLORS[index % len(_CLUSTER_TRACE_COLORS)]


@dataclass(slots=True)
class _ClusterDensityGroupState:
    key: str
    display_name: str
    structure_name: str
    motif_name: str
    source_dir: Path
    inspection: ElectronDensityInputInspection
    reference_structure: ElectronDensityStructure
    average_atom_count: float
    single_atom_only: bool
    trace_color: str
    profile_result: ElectronDensityProfileResult | None = None
    transform_result: ElectronDensityScatteringTransformResult | None = None
    debye_scattering_result: (
        ElectronDensityDebyeScatteringAverageResult | None
    ) = None
    fourier_settings: ElectronDensityFourierTransformSettings | None = None
    solvent_density_e_per_a3: float | None = None
    solvent_cutoff_radius_a: float | None = None


@dataclass(slots=True)
class _SavedOutputEntry:
    entry_id: str
    created_at: str
    entry_kind: str
    input_path: Path | None
    output_basename: str
    preview_mode: bool
    group_key: str | None
    group_label: str | None
    use_contiguous_frame_mode: bool
    profile_result: ElectronDensityProfileResult
    fourier_settings: ElectronDensityFourierTransformSettings
    transform_result: ElectronDensityScatteringTransformResult | None = None


@dataclass(slots=True)
class _ElectronDensityWorkspaceLoadPayload:
    selection_path: Path
    cluster_states: tuple[_ClusterDensityGroupState, ...] = ()
    inspection: ElectronDensityInputInspection | None = None
    structure: ElectronDensityStructure | None = None


@dataclass(slots=True, frozen=True)
class _DebyeComparisonEntry:
    entry_id: str
    label: str
    color: str
    born_result: ElectronDensityScatteringTransformResult
    debye_result: ElectronDensityDebyeScatteringAverageResult
    info_text: str


def _build_cluster_group_states_for_path(
    path: Path,
    *,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> list[_ClusterDensityGroupState]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_dir():
        return []
    try:
        cluster_bins = discover_cluster_bins(resolved)
    except Exception:
        return []
    if not cluster_bins:
        return []

    total_steps = len(cluster_bins) * 2 + 1
    if progress_callback is not None:
        progress_callback(
            0,
            total_steps,
            f"Discovered {len(cluster_bins)} stoichiometry folder"
            f"{'' if len(cluster_bins) == 1 else 's'} in {resolved.name}.",
        )

    states: list[_ClusterDensityGroupState] = []
    for index, cluster_bin in enumerate(cluster_bins, start=1):
        display_name = (
            cluster_bin.structure
            if cluster_bin.motif == "no_motif"
            else f"{cluster_bin.structure}/{cluster_bin.motif}"
        )
        inspect_step = (index - 1) * 2 + 1
        prepared_step = inspect_step + 1
        if progress_callback is not None:
            progress_callback(
                inspect_step,
                total_steps,
                f"Loading cluster {index}/{len(cluster_bins)}: {display_name}.",
            )
        try:
            inspection = inspect_structure_input(cluster_bin.source_dir)
            reference_structure = load_electron_density_structure(
                inspection.reference_file
            )
        except Exception:
            continue

        atom_counts: list[int] = []
        for structure_file in inspection.structure_files:
            atom_count = _quick_structure_atom_count(structure_file)
            if atom_count is not None:
                atom_counts.append(atom_count)
        if not atom_counts:
            continue

        single_atom_only = len(atom_counts) == inspection.total_files and all(
            int(atom_count) == 1 for atom_count in atom_counts
        )
        states.append(
            _ClusterDensityGroupState(
                key=display_name,
                display_name=display_name,
                structure_name=cluster_bin.structure,
                motif_name=cluster_bin.motif,
                source_dir=cluster_bin.source_dir,
                inspection=inspection,
                reference_structure=reference_structure,
                average_atom_count=float(np.mean(atom_counts)),
                single_atom_only=single_atom_only,
                trace_color=_cluster_trace_color_for_index(index - 1),
            )
        )
        if progress_callback is not None:
            progress_callback(
                prepared_step,
                total_steps,
                f"Prepared cluster {index}/{len(cluster_bins)}: {display_name}.",
            )
    return states


def _build_workspace_load_payload(
    path: Path,
    *,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> _ElectronDensityWorkspaceLoadPayload:
    resolved = Path(path).expanduser().resolve()
    cluster_states = _build_cluster_group_states_for_path(
        resolved,
        progress_callback=progress_callback,
    )
    if cluster_states:
        return _ElectronDensityWorkspaceLoadPayload(
            selection_path=resolved,
            cluster_states=tuple(cluster_states),
        )

    total_steps = 4
    if progress_callback is not None:
        progress_callback(
            0,
            total_steps,
            f"Inspecting inherited input {resolved.name}.",
        )
    inspection = inspect_structure_input(resolved)
    if progress_callback is not None:
        file_count = inspection.total_files
        progress_callback(
            1,
            total_steps,
            f"Found {file_count} structure file"
            f"{'' if file_count == 1 else 's'} in {resolved.name}.",
        )
        progress_callback(
            2,
            total_steps,
            f"Loading reference structure {inspection.reference_file.name}.",
        )
    structure = load_electron_density_structure(inspection.reference_file)
    if progress_callback is not None:
        progress_callback(
            3,
            total_steps,
            "Preparing the inherited electron-density workspace.",
        )
        progress_callback(
            4,
            total_steps,
            f"Loaded {inspection.reference_file.name}.",
        )
    return _ElectronDensityWorkspaceLoadPayload(
        selection_path=resolved,
        inspection=inspection,
        structure=structure,
    )


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
        self._expanded = False
        self._body.setVisible(self._expanded)
        layout.addWidget(header)
        layout.addWidget(self._body)

    def _toggle(self) -> None:
        self.set_expanded(not self._expanded)

    def set_expanded(self, expanded: bool) -> None:
        requested = bool(expanded)
        if self._expanded == requested:
            return
        self._expanded = requested
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

    def add_header_widget(self, widget: QWidget) -> None:
        """Append a widget to the right end of the section header
        row."""
        header = self.layout().itemAt(0).widget()
        header.layout().addWidget(widget)

    @property
    def is_expanded(self) -> bool:
        return self._expanded


class _OverlayFigureWidget(QWidget):
    """A simple matplotlib figure widget used for multi-trace
    overlays."""

    def __init__(
        self,
        title: str,
        *,
        figsize: tuple[float, float] = (8.4, 3.4),
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as _FC
        from matplotlib.backends.backend_qtagg import (
            NavigationToolbar2QT as _NT,
        )
        from matplotlib.figure import Figure as _Fig

        self._title = title
        self.figure = _Fig(figsize=figsize)
        self.canvas = _FC(self.figure)
        self.toolbar = _NT(self.canvas, self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self.toolbar)
        self.canvas.setMinimumHeight(260)
        layout.addWidget(self.canvas, stretch=1)
        self._draw_placeholder()

    def _draw_placeholder(self) -> None:
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.text(
            0.5,
            0.5,
            "No visible entries.",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        self.canvas.draw_idle()


class _SavedOutputComparisonDialog(QDialog):
    def __init__(
        self,
        entries: list[_SavedOutputEntry],
        *,
        show_variance_shading: bool,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.setWindowTitle("Saved Output Comparison")
        self.resize(1380, 980)
        self._entries = tuple(entries)
        self._entry_plot_widgets = self._entries
        self._show_variance_shading = bool(show_variance_shading)
        self._trace_colors: dict[str, str] = {
            entry.entry_id: _cluster_trace_color_for_index(i)
            for i, entry in enumerate(entries)
        }
        self._trace_visibility: dict[str, bool] = {
            entry.entry_id: True for entry in entries
        }
        self._build_ui()
        self._refresh_plots()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(8)

        controls = QHBoxLayout()
        controls.setSpacing(6)
        summary = QLabel(
            f"Comparing {len(self._entries)} saved output set"
            f"{'' if len(self._entries) == 1 else 's'}."
        )
        summary.setWordWrap(True)
        controls.addWidget(summary, stretch=1)
        self.save_all_png_button = QPushButton("Save All PNGs")
        self.save_all_png_button.clicked.connect(self._save_all_pngs)
        controls.addWidget(self.save_all_png_button)
        self.export_all_csv_button = QPushButton("Export All CSVs")
        self.export_all_csv_button.clicked.connect(self._export_all_csvs)
        controls.addWidget(self.export_all_csv_button)
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.close)
        controls.addWidget(self.close_button)
        outer.addLayout(controls)

        self.status_label = QLabel(
            "All selected entries are overlaid on shared axes. "
            "Use the trace table to show/hide or recolour individual traces."
        )
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: #475569;")
        outer.addWidget(self.status_label)

        splitter = QSplitter(Qt.Orientation.Vertical, self)

        # ---- overlay plot panels ----
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        container = QWidget()
        plot_layout = QVBoxLayout(container)
        plot_layout.setContentsMargins(4, 4, 4, 4)
        plot_layout.setSpacing(8)

        self._raw_plot = _OverlayFigureWidget(
            "Orientation-Averaged Radial Electron Density ρ(r)"
        )
        self._smeared_plot = _OverlayFigureWidget(
            "Gaussian-Smeared Radial Electron Density ρ(r)"
        )
        self._residual_plot = _OverlayFigureWidget(
            "Solvent-Subtracted Residual Δρ(r)"
        )
        self._fourier_plot = _OverlayFigureWidget(
            "Fourier-Transform Preparation (Windowed Source)"
        )
        self._scatter_plot = _OverlayFigureWidget("q-Space Scattering Profile")
        for widget in (
            self._raw_plot,
            self._smeared_plot,
            self._residual_plot,
            self._fourier_plot,
            self._scatter_plot,
        ):
            plot_layout.addWidget(widget)
        plot_layout.addStretch(1)
        scroll_area.setWidget(container)
        splitter.addWidget(scroll_area)

        # ---- trace table ----
        table_container = QGroupBox("Active Traces")
        table_layout = QVBoxLayout(table_container)
        self._trace_table = QTableWidget(0, 4)
        self._trace_table.setHorizontalHeaderLabels(
            ["Visible", "Color", "Label", "Info"]
        )
        self._trace_table.verticalHeader().setVisible(False)
        self._trace_table.setSelectionMode(
            QTableWidget.SelectionMode.NoSelection
        )
        hh = self._trace_table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        hh.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self._trace_table.setMinimumHeight(120)
        table_layout.addWidget(self._trace_table)
        splitter.addWidget(table_container)

        splitter.setSizes([700, 220])
        outer.addWidget(splitter, stretch=1)

    # ------------------------------------------------------------------
    # Trace table helpers (PDF-tool style)
    # ------------------------------------------------------------------

    def _refresh_trace_table(self) -> None:
        self._trace_table.setRowCount(0)
        for row, entry in enumerate(self._entries):
            self._trace_table.insertRow(row)
            key = entry.entry_id

            visible_box = QCheckBox()
            visible_box.setChecked(self._trace_visibility.get(key, True))
            visible_box.toggled.connect(
                lambda checked, k=key: self._set_trace_visible(k, checked)
            )
            self._trace_table.setCellWidget(row, 0, visible_box)

            color_btn = QPushButton()
            color_btn.clicked.connect(
                lambda _=False, k=key: self._choose_trace_color(k)
            )
            self._configure_color_button(
                color_btn, self._trace_colors.get(key, "#333333")
            )
            self._trace_table.setCellWidget(row, 1, color_btn)

            label_item = QTableWidgetItem(
                ElectronDensityMappingMainWindow._saved_output_entry_heading(
                    entry, index=row + 1
                )
            )
            label_item.setFlags(
                label_item.flags() & ~Qt.ItemFlag.ItemIsEditable
            )
            self._trace_table.setItem(row, 2, label_item)

            info_item = QTableWidgetItem(
                ElectronDensityMappingMainWindow._saved_output_entry_summary_text(
                    entry
                )
            )
            info_item.setFlags(info_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._trace_table.setItem(row, 3, info_item)

    def _configure_color_button(self, button: QPushButton, color: str) -> None:
        qc = QColor(color)
        text_color = "#000000" if qc.lightnessF() > 0.55 else "#ffffff"
        button.setText(color)
        button.setStyleSheet(
            f"QPushButton {{background-color:{color};color:{text_color};"
            "padding:3px 8px;}}"
        )

    def _choose_trace_color(self, entry_id: str) -> None:
        current = self._trace_colors.get(entry_id, "#333333")
        chosen = QColorDialog.getColor(
            QColor(current), self, "Choose trace color"
        )
        if not chosen.isValid():
            return
        self._trace_colors[entry_id] = chosen.name()
        self._refresh_trace_table()
        self._refresh_plots()

    def _set_trace_visible(self, entry_id: str, visible: bool) -> None:
        self._trace_visibility[entry_id] = bool(visible)
        self._refresh_plots()

    # ------------------------------------------------------------------
    # Overlay plot drawing
    # ------------------------------------------------------------------

    def _refresh_plots(self) -> None:
        self._refresh_trace_table()
        self._draw_raw_density()
        self._draw_smeared_density()
        self._draw_residual_density()
        self._draw_fourier_preview()
        self._draw_scattering_profile()

    def _visible_entries(
        self,
    ) -> list[tuple[_SavedOutputEntry, str]]:
        return [
            (e, self._trace_colors.get(e.entry_id, "#333333"))
            for e in self._entries
            if self._trace_visibility.get(e.entry_id, True)
        ]

    def _draw_raw_density(self) -> None:
        pairs = self._visible_entries()
        fig = self._raw_plot.figure
        fig.clear()
        ax = fig.add_subplot(111)
        if not pairs:
            ax.text(
                0.5,
                0.5,
                "No visible entries.",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_axis_off()
            self._raw_plot.canvas.draw_idle()
            return
        for entry, color in pairs:
            result = entry.profile_result
            radial = np.asarray(result.radial_centers, dtype=float)
            density = np.asarray(
                result.orientation_average_density, dtype=float
            )
            label = (
                ElectronDensityMappingMainWindow._saved_output_context_label(
                    entry
                )
            )
            ax.step(
                radial,
                density,
                where="mid",
                color=color,
                linewidth=1.8,
                label=label,
            )
            if self._show_variance_shading:
                spread = np.asarray(
                    result.orientation_density_stddev, dtype=float
                )
                if np.any(spread > 0.0):
                    ax.fill_between(
                        radial,
                        np.maximum(density - spread, 0.0),
                        density + spread,
                        step="mid",
                        alpha=0.14,
                        color=color,
                    )
        ax.set_xlabel("r (Å)")
        ax.set_ylabel("ρ(r) (e/Å³)")
        ax.set_title("Orientation-Averaged Radial Electron Density ρ(r)")
        ax.grid(True, alpha=0.28)
        ax.legend(loc="upper right", frameon=True, fontsize=8)
        fig.tight_layout()
        self._raw_plot.canvas.draw_idle()

    def _draw_smeared_density(self) -> None:
        pairs = self._visible_entries()
        fig = self._smeared_plot.figure
        fig.clear()
        ax = fig.add_subplot(111)
        if not pairs:
            ax.text(
                0.5,
                0.5,
                "No visible entries.",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_axis_off()
            self._smeared_plot.canvas.draw_idle()
            return
        for entry, color in pairs:
            result = entry.profile_result
            radial = np.asarray(result.radial_centers, dtype=float)
            density = np.asarray(
                result.smeared_orientation_average_density, dtype=float
            )
            label = (
                ElectronDensityMappingMainWindow._saved_output_context_label(
                    entry
                )
            )
            ax.plot(radial, density, color=color, linewidth=2.0, label=label)
            if self._show_variance_shading:
                spread = np.asarray(
                    result.smeared_orientation_density_stddev, dtype=float
                )
                if np.any(spread > 0.0):
                    ax.fill_between(
                        radial,
                        np.maximum(density - spread, 0.0),
                        density + spread,
                        alpha=0.14,
                        color=color,
                    )
            if result.solvent_contrast is not None:
                contrast = result.solvent_contrast
                ax.axhline(
                    float(contrast.solvent_density_e_per_a3),
                    color=color,
                    linewidth=1.2,
                    linestyle=":",
                    alpha=0.65,
                )
        ax.set_xlabel("r (Å)")
        ax.set_ylabel("ρ(r) (e/Å³)")
        ax.set_title("Gaussian-Smeared Radial Electron Density ρ(r)")
        ax.grid(True, alpha=0.28)
        ax.legend(loc="upper right", frameon=True, fontsize=8)
        fig.tight_layout()
        self._smeared_plot.canvas.draw_idle()

    def _draw_residual_density(self) -> None:
        pairs = self._visible_entries()
        fig = self._residual_plot.figure
        fig.clear()
        ax = fig.add_subplot(111)
        has_residual = False
        for entry, color in pairs:
            result = entry.profile_result
            if result.solvent_contrast is None:
                continue
            contrast = result.solvent_contrast
            radial = np.asarray(result.radial_centers, dtype=float)
            residual = np.asarray(
                contrast.solvent_subtracted_smeared_density, dtype=float
            )
            label = (
                ElectronDensityMappingMainWindow._saved_output_context_label(
                    entry
                )
            )
            ax.plot(
                radial,
                residual,
                color=color,
                linewidth=1.8,
                linestyle="--",
                label=label,
            )
            if self._show_variance_shading:
                spread = np.asarray(
                    result.smeared_orientation_density_stddev, dtype=float
                )
                if np.any(spread > 0.0):
                    ax.fill_between(
                        radial,
                        residual - spread,
                        residual + spread,
                        alpha=0.12,
                        color=color,
                    )
            has_residual = True
        if not has_residual:
            ax.text(
                0.5,
                0.5,
                "No solvent contrast data available for selected entries.",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_axis_off()
            self._residual_plot.canvas.draw_idle()
            return
        ax.axhline(
            0.0, color="#94a3b8", linewidth=1.4, linestyle=":", alpha=0.9
        )
        ax.set_xlabel("r (Å)")
        ax.set_ylabel("Δρ(r) (e/Å³)")
        ax.set_title("Solvent-Subtracted Residual Δρ(r)")
        ax.grid(True, alpha=0.28)
        ax.legend(loc="upper right", frameon=True, fontsize=8)
        fig.tight_layout()
        self._residual_plot.canvas.draw_idle()

    def _draw_fourier_preview(self) -> None:
        pairs = self._visible_entries()
        fig = self._fourier_plot.figure
        fig.clear()
        ax = fig.add_subplot(111)
        has_preview = False
        for entry, color in pairs:
            preview = ElectronDensityMappingMainWindow._preview_for_saved_output_entry(
                entry
            )
            if preview is None:
                continue
            resampled_r = np.asarray(preview.resampled_r_values, dtype=float)
            windowed = np.asarray(preview.windowed_density_values, dtype=float)
            label = (
                ElectronDensityMappingMainWindow._saved_output_context_label(
                    entry
                )
            )
            ax.plot(
                resampled_r, windowed, color=color, linewidth=1.8, label=label
            )
            has_preview = True
        if not has_preview:
            ax.text(
                0.5,
                0.5,
                "No Fourier preview data available.",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_axis_off()
            self._fourier_plot.canvas.draw_idle()
            return
        ax.set_xlabel("r (Å)")
        ax.set_ylabel("ρ(r) (e/Å³)")
        ax.set_title("Fourier-Transform Preparation (Windowed Source)")
        ax.grid(True, alpha=0.28)
        ax.legend(loc="upper right", frameon=True, fontsize=8)
        fig.tight_layout()
        self._fourier_plot.canvas.draw_idle()

    def _draw_scattering_profile(self) -> None:
        pairs = self._visible_entries()
        fig = self._scatter_plot.figure
        fig.clear()
        ax = fig.add_subplot(111)
        has_scatter = False
        use_log_q = any(
            bool(e.fourier_settings.log_q_axis)
            for e, _ in pairs
            if e.transform_result is not None
        )
        use_log_i = any(
            bool(e.fourier_settings.log_intensity_axis)
            for e, _ in pairs
            if e.transform_result is not None
        )
        for entry, color in pairs:
            if entry.transform_result is None:
                continue
            result = entry.transform_result
            q = np.asarray(result.q_values, dtype=float)
            intensity = np.asarray(result.intensity, dtype=float)
            mask = np.ones_like(q, dtype=bool)
            if use_log_q:
                mask &= q > 0.0
            if use_log_i:
                mask &= intensity > 0.0
            if not mask.any():
                continue
            label = (
                entry.group_label
                or ElectronDensityMappingMainWindow._saved_output_context_label(
                    entry
                )
            )
            ax.plot(
                q[mask],
                intensity[mask],
                color=color,
                linewidth=1.8,
                label=label,
            )
            has_scatter = True
        if not has_scatter:
            ax.text(
                0.5,
                0.5,
                "No Fourier transform results available for selected entries.",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_axis_off()
            self._scatter_plot.canvas.draw_idle()
            return
        if use_log_q:
            ax.set_xscale("log")
        if use_log_i:
            ax.set_yscale("log")
        ax.set_xlabel("q (Å⁻¹)", labelpad=10.0)
        ax.set_ylabel("Intensity (arb. units)")
        ax.set_title("q-Space Scattering Profile")
        ax.grid(True, which="both", alpha=0.28)
        ax.legend(loc="upper right", frameon=True, fontsize=8)
        fig.tight_layout()
        self._scatter_plot.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Save / export
    # ------------------------------------------------------------------

    @Slot()
    def _save_all_pngs(self) -> None:
        chosen = QFileDialog.getExistingDirectory(
            self, "Choose Directory for Saved Output PNGs", ""
        )
        if not chosen:
            return
        export_dir = Path(chosen).expanduser().resolve()
        export_dir.mkdir(parents=True, exist_ok=True)
        panels = [
            ("raw_density", self._raw_plot),
            ("smeared_density", self._smeared_plot),
            ("residual_density", self._residual_plot),
            ("fourier_preview", self._fourier_plot),
            ("scattering_profile", self._scatter_plot),
        ]
        written_count = 0
        original_visibility = dict(self._trace_visibility)
        try:
            for index, entry in enumerate(self._entries, start=1):
                self._trace_visibility = {
                    candidate.entry_id: candidate.entry_id == entry.entry_id
                    for candidate in self._entries
                }
                self._refresh_plots()
                stem = ElectronDensityMappingMainWindow._saved_output_entry_file_stem(
                    entry,
                    index=index,
                )
                for suffix, widget in panels:
                    widget.figure.savefig(
                        export_dir / f"{stem}_{suffix}.png",
                        dpi=200,
                        bbox_inches="tight",
                    )
                    written_count += 1
        except Exception as exc:
            QMessageBox.critical(self, "PNG Export Failed", str(exc))
            return
        finally:
            self._trace_visibility = original_visibility
            self._refresh_plots()
        self.status_label.setText(
            f"Saved {written_count} PNG file"
            f"{'' if written_count == 1 else 's'} to {export_dir}."
        )

    @Slot()
    def _export_all_csvs(self) -> None:
        chosen = QFileDialog.getExistingDirectory(
            self, "Choose Directory for Saved Output CSVs", ""
        )
        if not chosen:
            return
        export_dir = Path(chosen).expanduser().resolve()
        export_dir.mkdir(parents=True, exist_ok=True)
        written_count = 0
        try:
            for index, entry in enumerate(self._entries, start=1):
                preview = ElectronDensityMappingMainWindow._preview_for_saved_output_entry(
                    entry
                )
                csv_path = export_dir / (
                    ElectronDensityMappingMainWindow._saved_output_entry_file_stem(
                        entry, index=index
                    )
                    + ".csv"
                )
                ElectronDensityMappingMainWindow._write_plot_trace_csv(
                    csv_path,
                    profile_result=entry.profile_result,
                    fourier_preview=preview,
                    transform_result=entry.transform_result,
                )
                written_count += 1
        except Exception as exc:
            QMessageBox.critical(self, "CSV Export Failed", str(exc))
            return
        self.status_label.setText(
            f"Exported {written_count} CSV file"
            f"{'' if written_count == 1 else 's'} to {export_dir}."
        )


class _DebyeScatteringComparisonDialog(QDialog):
    def __init__(
        self,
        entries: list[_DebyeComparisonEntry],
        *,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.setWindowTitle("Debye Scattering Comparison")
        self.resize(1320, 900)
        self._entries = tuple(entries)
        self._trace_colors = {
            entry.entry_id: str(entry.color) for entry in self._entries
        }
        self._trace_visibility = {
            entry.entry_id: True for entry in self._entries
        }
        self._build_ui()
        self._refresh_plot()

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(8)

        controls = QHBoxLayout()
        controls.setSpacing(6)
        summary = QLabel(
            f"Comparing {len(self._entries)} Born/Debye trace pair"
            f"{'' if len(self._entries) == 1 else 's'}."
        )
        summary.setWordWrap(True)
        controls.addWidget(summary, stretch=1)

        self.log_x_checkbox = QCheckBox("Log X")
        self.log_x_checkbox.setChecked(True)
        self.log_x_checkbox.toggled.connect(self._refresh_plot)
        controls.addWidget(self.log_x_checkbox)

        self.log_y_checkbox = QCheckBox("Log Y")
        self.log_y_checkbox.setChecked(True)
        self.log_y_checkbox.toggled.connect(self._refresh_plot)
        controls.addWidget(self.log_y_checkbox)

        self.legend_button = QPushButton("Legend")
        self.legend_button.setCheckable(True)
        self.legend_button.setChecked(True)
        self.legend_button.toggled.connect(self._refresh_plot)
        controls.addWidget(self.legend_button)

        self.autoscale_button = QPushButton("Autoscale Overlay")
        self.autoscale_button.setCheckable(True)
        self.autoscale_button.setChecked(True)
        self.autoscale_button.toggled.connect(self._refresh_plot)
        controls.addWidget(self.autoscale_button)

        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.close)
        controls.addWidget(self.close_button)
        outer.addLayout(controls)

        self.status_label = QLabel(
            "Born-approximation traces use the left axis and solid lines. "
            "Debye scattering traces use the right axis and dashed lines. "
            "Use the table below to overlay selected trace pairs."
        )
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: #475569;")
        outer.addWidget(self.status_label)

        splitter = QSplitter(Qt.Orientation.Vertical, self)
        self._plot_widget = _OverlayFigureWidget(
            "Born Approximation vs Debye Scattering",
            figsize=(8.8, 4.4),
        )
        splitter.addWidget(self._plot_widget)

        table_container = QGroupBox("Trace Pairs")
        table_layout = QVBoxLayout(table_container)
        self._trace_table = QTableWidget(0, 5)
        self._trace_table.setHorizontalHeaderLabels(
            ["Visible", "Color", "Label", "Born", "Debye"]
        )
        self._trace_table.verticalHeader().setVisible(False)
        self._trace_table.setSelectionMode(
            QTableWidget.SelectionMode.NoSelection
        )
        header = self._trace_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        self._trace_table.setMinimumHeight(160)
        table_layout.addWidget(self._trace_table)
        splitter.addWidget(table_container)
        splitter.setSizes([680, 220])
        outer.addWidget(splitter, stretch=1)

    def _configure_color_button(
        self,
        button: QPushButton,
        color: str,
    ) -> None:
        qc = QColor(color)
        text_color = "#000000" if qc.lightnessF() > 0.55 else "#ffffff"
        button.setText(color)
        button.setStyleSheet(
            f"QPushButton {{background-color:{color};color:{text_color};"
            "padding:3px 8px;}}"
        )

    def _choose_trace_color(self, entry_id: str) -> None:
        current = self._trace_colors.get(entry_id, "#333333")
        chosen = QColorDialog.getColor(
            QColor(current), self, "Choose trace color"
        )
        if not chosen.isValid():
            return
        self._trace_colors[entry_id] = chosen.name()
        self._refresh_trace_table()
        self._refresh_plot()

    def _set_trace_visible(self, entry_id: str, visible: bool) -> None:
        self._trace_visibility[entry_id] = bool(visible)
        self._refresh_plot()

    def _refresh_trace_table(self) -> None:
        self._trace_table.setRowCount(0)
        for row_index, entry in enumerate(self._entries):
            self._trace_table.insertRow(row_index)
            key = entry.entry_id

            visible_box = QCheckBox()
            visible_box.setChecked(self._trace_visibility.get(key, True))
            visible_box.toggled.connect(
                lambda checked, entry_id=key: self._set_trace_visible(
                    entry_id,
                    checked,
                )
            )
            self._trace_table.setCellWidget(row_index, 0, visible_box)

            color_button = QPushButton()
            color_button.clicked.connect(
                lambda _=False, entry_id=key: self._choose_trace_color(
                    entry_id
                )
            )
            self._configure_color_button(
                color_button,
                self._trace_colors.get(key, "#333333"),
            )
            self._trace_table.setCellWidget(row_index, 1, color_button)

            for column_index, value in enumerate(
                (
                    entry.label,
                    self._born_status_text(entry.born_result),
                    self._debye_status_text(entry.debye_result),
                ),
                start=2,
            ):
                item = QTableWidgetItem(value)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                if column_index == 2:
                    item.setToolTip(entry.info_text)
                self._trace_table.setItem(row_index, column_index, item)

    @staticmethod
    def _born_status_text(
        result: ElectronDensityScatteringTransformResult,
    ) -> str:
        q_values = np.asarray(result.q_values, dtype=float)
        if q_values.size == 0:
            return "Unavailable"
        return (
            f"{len(q_values)} points, q={float(q_values[0]):.4f}"
            f"–{float(q_values[-1]):.4f} Å⁻¹"
        )

    @staticmethod
    def _debye_status_text(
        result: ElectronDensityDebyeScatteringAverageResult,
    ) -> str:
        q_values = np.asarray(result.q_values, dtype=float)
        return (
            f"{result.source_structure_count} structure"
            f"{'' if result.source_structure_count == 1 else 's'}, "
            f"{len(q_values)} q points"
        )

    def _visible_entries(self) -> list[_DebyeComparisonEntry]:
        return [
            entry
            for entry in self._entries
            if self._trace_visibility.get(entry.entry_id, True)
        ]

    def _refresh_plot(self) -> None:
        self._refresh_trace_table()
        fig = self._plot_widget.figure
        fig.clear()
        visible_entries = self._visible_entries()
        if not visible_entries:
            axis = fig.add_subplot(111)
            axis.text(
                0.5,
                0.5,
                "No visible comparison traces.",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
            axis.set_axis_off()
            self._plot_widget.canvas.draw_idle()
            self.status_label.setText(
                "Born-approximation traces use the left axis and solid lines. "
                "Debye scattering traces use the right axis and dashed lines. "
                "Use the table below to overlay selected trace pairs."
            )
            return

        base_axis = fig.add_subplot(111)
        born_axis = base_axis
        debye_axis = base_axis.twinx()
        plotted_lines: list[object] = []

        positive_q = True
        born_positive_i = True
        debye_positive_i = True
        for entry in visible_entries:
            color = self._trace_colors.get(entry.entry_id, entry.color)
            born_q = np.asarray(entry.born_result.q_values, dtype=float)
            born_i = np.asarray(entry.born_result.intensity, dtype=float)
            debye_q = np.asarray(entry.debye_result.q_values, dtype=float)
            debye_i = np.asarray(
                entry.debye_result.mean_intensity, dtype=float
            )
            positive_q = positive_q and bool(
                np.all(born_q > 0.0) and np.all(debye_q > 0.0)
            )
            born_positive_i = born_positive_i and bool(np.all(born_i > 0.0))
            debye_positive_i = debye_positive_i and bool(np.all(debye_i > 0.0))
            (born_line,) = born_axis.plot(
                born_q,
                born_i,
                color=color,
                linewidth=1.8,
                label=f"{entry.label} · Born",
            )
            (debye_line,) = debye_axis.plot(
                debye_q,
                debye_i,
                color=color,
                linewidth=1.8,
                linestyle="--",
                label=f"{entry.label} · Debye",
            )
            plotted_lines.extend([born_line, debye_line])

        if self.log_x_checkbox.isChecked() and positive_q:
            born_axis.set_xscale("log")
            debye_axis.set_xscale("log")
        born_axis.set_yscale(
            "log"
            if self.log_y_checkbox.isChecked() and born_positive_i
            else "linear"
        )
        debye_axis.set_yscale(
            "log"
            if self.log_y_checkbox.isChecked() and debye_positive_i
            else "linear"
        )
        born_axis.set_xlabel("q (Å⁻¹)")
        born_axis.set_ylabel("Born Approximation Intensity (arb. units)")
        debye_axis.set_ylabel("Debye Scattering Intensity (arb. units)")
        born_axis.set_title("Born Approximation vs Debye Scattering")
        born_axis.grid(True, which="both", alpha=0.28)

        if self.autoscale_button.isChecked():
            self._autoscale_to_born_range(born_axis, debye_axis)

        if plotted_lines and self.legend_button.isChecked():
            born_axis.legend(
                plotted_lines,
                [line.get_label() for line in plotted_lines],
                loc="best",
                fontsize=8,
            )
        fig.tight_layout()
        self._plot_widget.canvas.draw_idle()
        self.status_label.setText(
            f"Overlaying {len(visible_entries)} visible trace pair"
            f"{'' if len(visible_entries) == 1 else 's'}."
        )

    def _autoscale_to_born_range(self, born_axis, debye_axis) -> None:
        born_lines = [
            line for line in born_axis.get_lines() if line.get_visible()
        ]
        if not born_lines:
            return
        born_q_values = np.concatenate(
            [
                np.asarray(line.get_xdata(orig=False), dtype=float)
                for line in born_lines
            ]
        )
        born_q_values = born_q_values[np.isfinite(born_q_values)]
        if born_q_values.size == 0:
            return
        q_min = float(np.nanmin(born_q_values))
        q_max = float(np.nanmax(born_q_values))
        born_axis.set_xlim(q_min, q_max)
        debye_axis.set_xlim(q_min, q_max)
        self._autoscale_axis_y(born_axis, q_min, q_max)
        self._normalize_debye_axis(born_axis, debye_axis)

    def _normalize_debye_axis(self, born_axis, debye_axis) -> None:
        born_lines = [
            line for line in born_axis.get_lines() if line.get_visible()
        ]
        debye_lines = [
            line for line in debye_axis.get_lines() if line.get_visible()
        ]
        if not born_lines or not debye_lines:
            return
        born_q = np.concatenate(
            [
                np.asarray(line.get_xdata(orig=False), dtype=float)
                for line in born_lines
            ]
        )
        born_i = np.concatenate(
            [
                np.asarray(line.get_ydata(orig=False), dtype=float)
                for line in born_lines
            ]
        )
        debye_q = np.concatenate(
            [
                np.asarray(line.get_xdata(orig=False), dtype=float)
                for line in debye_lines
            ]
        )
        debye_i = np.concatenate(
            [
                np.asarray(line.get_ydata(orig=False), dtype=float)
                for line in debye_lines
            ]
        )
        overlap_mask = (born_q >= float(np.nanmin(debye_q))) & (
            born_q <= float(np.nanmax(debye_q))
        )
        if np.any(overlap_mask):
            born_i = born_i[overlap_mask]
        born_i = born_i[np.isfinite(born_i)]
        debye_i = debye_i[np.isfinite(debye_i)]
        if self.log_y_checkbox.isChecked():
            born_i = born_i[born_i > 0.0]
            debye_i = debye_i[debye_i > 0.0]
        if born_i.size == 0 or debye_i.size == 0:
            return
        debye_axis.set_ylim(
            self._aligned_y_limits(
                born_axis.get_ylim(),
                float(np.nanmin(born_i)),
                float(np.nanmax(born_i)),
                float(np.nanmin(debye_i)),
                float(np.nanmax(debye_i)),
                log_scale=self.log_y_checkbox.isChecked(),
            )
        )

    def _autoscale_axis_y(
        self,
        axis,
        q_min: float,
        q_max: float,
    ) -> None:
        y_segments: list[np.ndarray] = []
        log_scale = self.log_y_checkbox.isChecked()
        for line in axis.get_lines():
            if not line.get_visible():
                continue
            x_data = np.asarray(line.get_xdata(orig=False), dtype=float)
            y_data = np.asarray(line.get_ydata(orig=False), dtype=float)
            mask = (
                np.isfinite(x_data)
                & np.isfinite(y_data)
                & (x_data >= q_min)
                & (x_data <= q_max)
            )
            if log_scale:
                mask &= y_data > 0.0
            if np.any(mask):
                y_segments.append(y_data[mask])
        if not y_segments:
            return
        y_values = np.concatenate(y_segments)
        y_min = float(np.nanmin(y_values))
        y_max = float(np.nanmax(y_values))
        if np.isclose(y_min, y_max):
            padding = max(abs(y_min) * 0.05, 1e-12)
            axis.set_ylim(y_min - padding, y_max + padding)
            return
        if log_scale:
            axis.set_ylim(y_min / 1.15, y_max * 1.15)
        else:
            padding = 0.05 * (y_max - y_min)
            axis.set_ylim(y_min - padding, y_max + padding)

    @staticmethod
    def _aligned_y_limits(
        left_limits: tuple[float, float],
        born_min: float,
        born_max: float,
        debye_min: float,
        debye_max: float,
        *,
        log_scale: bool,
    ) -> tuple[float, float]:
        if log_scale:
            if (
                min(
                    left_limits[0],
                    left_limits[1],
                    born_min,
                    born_max,
                    debye_min,
                    debye_max,
                )
                <= 0.0
            ):
                log_scale = False
        if not log_scale:
            left_low, left_high = left_limits
            born_low, born_high = sorted((born_min, born_max))
            debye_low, debye_high = sorted((debye_min, debye_max))
            if np.isclose(left_high, left_low) or np.isclose(
                born_high,
                born_low,
            ):
                padding = max(abs(debye_low) * 0.1, 1e-12)
                return debye_low - padding, debye_high + padding
            p0 = (born_low - left_low) / (left_high - left_low)
            p1 = (born_high - left_low) / (left_high - left_low)
            if np.isclose(p1, p0):
                padding = max(abs(debye_low) * 0.1, 1e-12)
                return debye_low - padding, debye_high + padding
            delta = (debye_high - debye_low) / (p1 - p0)
            right_low = debye_low - p0 * delta
            right_high = right_low + delta
            return right_low, right_high

        left_logs = np.log10(np.asarray(left_limits, dtype=float))
        born_logs = np.log10(
            np.asarray(sorted((born_min, born_max)), dtype=float)
        )
        debye_logs = np.log10(
            np.asarray(sorted((debye_min, debye_max)), dtype=float)
        )
        if np.isclose(left_logs[1], left_logs[0]) or np.isclose(
            born_logs[1],
            born_logs[0],
        ):
            return debye_min / 1.2, debye_max * 1.2
        p0 = (born_logs[0] - left_logs[0]) / (left_logs[1] - left_logs[0])
        p1 = (born_logs[1] - left_logs[0]) / (left_logs[1] - left_logs[0])
        if np.isclose(p1, p0):
            return debye_min / 1.2, debye_max * 1.2
        delta = (debye_logs[1] - debye_logs[0]) / (p1 - p0)
        right_low_log = debye_logs[0] - p0 * delta
        right_high_log = right_low_log + delta
        return 10**right_low_log, 10**right_high_log


class ElectronDensityWorkspaceLoadWorker(QObject):
    progress = Signal(int, int, str)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, path: Path) -> None:
        super().__init__()
        self._path = Path(path).expanduser().resolve()

    @Slot()
    def run(self) -> None:
        try:
            payload = _build_workspace_load_payload(
                self._path,
                progress_callback=self.progress.emit,
            )
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        self.finished.emit(payload)


class ElectronDensityCalculationWorker(QObject):
    progress = Signal(int, int, str)
    overall_progress = Signal(int, int, str)
    finished = Signal(object)
    canceled = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        *,
        inspection: ElectronDensityInputInspection | None,
        reference_structure: ElectronDensityStructure | None,
        mesh_settings: ElectronDensityMeshSettings,
        smearing_settings: ElectronDensitySmearingSettings,
        center_mode: str,
        reference_element: str | None,
        output_dir: str,
        output_basename: str,
        use_contiguous_frame_mode: bool,
        grouped_inspections: tuple[
            tuple[str, ElectronDensityInputInspection],
            ...,
        ] = (),
        grouped_reference_structures: tuple[
            tuple[str, ElectronDensityStructure],
            ...,
        ] = (),
        single_atom_group_keys: tuple[str, ...] = (),
        debye_scattering_settings: (
            ElectronDensityFourierTransformSettings | None
        ) = None,
    ) -> None:
        super().__init__()
        self._inspection = inspection
        self._reference_structure = reference_structure
        self._grouped_inspections = tuple(grouped_inspections)
        self._grouped_reference_structures = {
            str(group_key): structure
            for group_key, structure in grouped_reference_structures
        }
        self._single_atom_group_keys = {
            str(group_key) for group_key in single_atom_group_keys
        }
        self._mesh_settings = mesh_settings
        self._smearing_settings = smearing_settings
        self._center_mode = str(center_mode)
        self._reference_element = (
            None
            if reference_element is None
            else str(reference_element).strip() or None
        )
        self._use_contiguous_frame_mode = bool(use_contiguous_frame_mode)
        self._output_dir = str(output_dir)
        self._output_basename = str(output_basename)
        self._debye_scattering_settings = (
            ElectronDensityFourierTransformSettings()
            if debye_scattering_settings is None
            else debye_scattering_settings.normalized()
        )
        self._cancel_requested = False

    @Slot()
    def cancel(self) -> None:
        self._cancel_requested = True

    @staticmethod
    def _safe_group_basename(label: str) -> str:
        normalized = "".join(
            character if character.isalnum() or character in "._-" else "_"
            for character in str(label).strip()
        )
        return normalized.strip("._") or "cluster_group"

    @staticmethod
    def _group_progress_prefix(
        group_index: int,
        group_count: int,
        group_label: str,
    ) -> str:
        return (
            f"Cluster {group_index}/{group_count} "
            f"[{str(group_label).strip() or 'cluster_group'}]"
        )

    @Slot()
    def run(self) -> None:
        group_results: dict[str, ElectronDensityProfileResult] = {}
        group_transform_results: dict[
            str, ElectronDensityScatteringTransformResult
        ] = {}
        group_artifacts: dict[str, ElectronDensityOutputArtifacts] = {}
        try:
            if self._grouped_inspections:
                group_count = len(self._grouped_inspections)
                self.overall_progress.emit(
                    0,
                    max(group_count, 1),
                    "Overall cluster progress: "
                    f"0/{group_count} groups complete.",
                )
                for group_index, (group_key, inspection) in enumerate(
                    self._grouped_inspections,
                    start=1,
                ):
                    group_label = str(group_key).strip() or "cluster_group"
                    progress_prefix = self._group_progress_prefix(
                        group_index,
                        group_count,
                        group_label,
                    )
                    self.overall_progress.emit(
                        group_index - 1,
                        max(group_count, 1),
                        "Overall cluster progress: "
                        f"{group_index - 1}/{group_count} groups complete. "
                        f"Running {group_label}.",
                    )

                    def emit_group_progress(
                        current: int,
                        total: int,
                        message: str,
                        *,
                        _prefix: str = progress_prefix,
                    ) -> None:
                        self.progress.emit(
                            int(current),
                            int(total),
                            f"{_prefix}: {str(message).strip()}",
                        )

                    if group_key in self._single_atom_group_keys:
                        transform_result = compute_single_atom_debye_scattering_profile_for_input(
                            inspection,
                            self._debye_scattering_settings,
                            progress_callback=emit_group_progress,
                            cancel_callback=lambda: self._cancel_requested,
                        )
                        if self._cancel_requested:
                            raise ElectronDensityCalculationCanceled(
                                "Electron-density calculation was stopped by the user."
                            )
                        group_transform_results[group_key] = transform_result
                    else:
                        result = compute_electron_density_profile_for_input(
                            inspection,
                            self._mesh_settings,
                            smearing_settings=self._smearing_settings,
                            reference_structure=self._grouped_reference_structures.get(
                                group_key
                            ),
                            center_mode=self._center_mode,
                            reference_element=None,
                            use_contiguous_frame_mode=self._use_contiguous_frame_mode,
                            progress_callback=emit_group_progress,
                            cancel_callback=lambda: self._cancel_requested,
                        )
                        if self._cancel_requested:
                            raise ElectronDensityCalculationCanceled(
                                "Electron-density calculation was stopped by the user."
                            )
                        group_results[group_key] = result
                        if self._output_dir.strip():
                            artifact = write_electron_density_profile_outputs(
                                result,
                                self._output_dir,
                                (
                                    f"{self._output_basename}_"
                                    f"{self._safe_group_basename(group_key)}"
                                ),
                            )
                            if self._cancel_requested:
                                raise ElectronDensityCalculationCanceled(
                                    "Electron-density calculation was stopped by the user."
                                )
                            group_artifacts[group_key] = artifact
                    self.overall_progress.emit(
                        group_index,
                        max(group_count, 1),
                        "Overall cluster progress: "
                        f"{group_index}/{group_count} groups complete. "
                        f"Finished {group_label}.",
                    )
                    if self._cancel_requested and group_index < group_count:
                        raise ElectronDensityCalculationCanceled(
                            "Electron-density calculation was stopped by the user."
                        )
                self.finished.emit(
                    {
                        "group_results": group_results,
                        "group_transform_results": group_transform_results,
                        "group_artifacts": group_artifacts,
                    }
                )
                return
            if self._inspection is None:
                raise ValueError("No input inspection was provided.")
            result = compute_electron_density_profile_for_input(
                self._inspection,
                self._mesh_settings,
                smearing_settings=self._smearing_settings,
                reference_structure=self._reference_structure,
                center_mode=self._center_mode,
                reference_element=self._reference_element,
                use_contiguous_frame_mode=self._use_contiguous_frame_mode,
                progress_callback=self.progress.emit,
                cancel_callback=lambda: self._cancel_requested,
            )
            if self._cancel_requested:
                raise ElectronDensityCalculationCanceled(
                    "Electron-density calculation was stopped by the user."
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
                if self._cancel_requested:
                    raise ElectronDensityCalculationCanceled(
                        "Electron-density calculation was stopped by the user."
                    )
                artifacts = write_electron_density_profile_outputs(
                    result,
                    self._output_dir,
                    self._output_basename,
                )
                if self._cancel_requested:
                    raise ElectronDensityCalculationCanceled(
                        "Electron-density calculation was stopped by the user."
                    )
            self.finished.emit(
                {
                    "result": result,
                    "artifacts": artifacts,
                }
            )
        except ElectronDensityCalculationCanceled:
            self.canceled.emit(
                {
                    "group_results": group_results,
                    "group_transform_results": group_transform_results,
                    "group_artifacts": group_artifacts,
                }
            )
        except Exception as exc:
            self.failed.emit(str(exc))


class ElectronDensityMappingMainWindow(QMainWindow):
    """Interactive supporting tool for radial electron-density
    inspection."""

    born_components_built = Signal(object)
    cancel_calculation_requested = Signal()

    @staticmethod
    def _default_fourier_settings() -> ElectronDensityFourierTransformSettings:
        return ElectronDensityFourierTransformSettings(
            r_min=-1.0,
            r_max=1.0,
            domain_mode="mirrored",
            window_function="hanning",
            q_min=0.02,
            q_max=1.2,
            q_step=0.01,
            resampling_points=2048,
        )

    def __init__(
        self,
        *,
        initial_project_dir: Path | None = None,
        initial_input_path: Path | None = None,
        initial_output_dir: Path | None = None,
        initial_project_q_min: float | None = None,
        initial_project_q_max: float | None = None,
        initial_distribution_id: str | None = None,
        initial_distribution_root_dir: Path | None = None,
        initial_use_predicted_structure_weights: bool = False,
        preview_mode: bool = True,
        restore_saved_distribution_state_on_init: bool = True,
    ) -> None:
        super().__init__()
        self._preview_mode = bool(preview_mode)
        self._project_dir = (
            None
            if initial_project_dir is None
            else Path(initial_project_dir).expanduser().resolve()
        )
        self._initial_output_dir = (
            None
            if initial_output_dir is None
            else Path(initial_output_dir).expanduser().resolve()
        )
        self._project_q_min = (
            None
            if initial_project_q_min is None
            else float(initial_project_q_min)
        )
        self._project_q_max = (
            None
            if initial_project_q_max is None
            else float(initial_project_q_max)
        )
        self._distribution_id = (
            str(initial_distribution_id or "").strip() or None
        )
        self._distribution_root_dir = (
            None
            if initial_distribution_root_dir is None
            else Path(initial_distribution_root_dir).expanduser().resolve()
        )
        self._use_predicted_structure_weights = bool(
            initial_use_predicted_structure_weights
        )
        self._inspection: ElectronDensityInputInspection | None = None
        self._structure: ElectronDensityStructure | None = None
        self._cluster_group_states: list[_ClusterDensityGroupState] = []
        self._selected_cluster_group_key: str | None = None
        self._restoring_saved_distribution_state = False
        self._restoring_saved_output_history = False
        self._mesh_settings_confirmed_for_run = False
        self._manual_mesh_lock_settings: ElectronDensityMeshSettings | None = (
            None
        )
        self._current_group_run_manual = False
        self._auto_snap_panes_enabled = self._load_auto_snap_panes_setting()
        self._active_mesh_settings = ElectronDensityMeshSettings()
        self._active_mesh_geometry: ElectronDensityMeshGeometry | None = None
        self._active_smearing_settings = ElectronDensitySmearingSettings()
        self._active_fourier_settings = self._default_fourier_settings()
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
        self._debye_scattering_result: (
            ElectronDensityDebyeScatteringAverageResult | None
        ) = None
        self._calculation_ui_running = False
        self._calculation_cancel_requested = False
        self._calculation_thread: QThread | None = None
        self._calculation_worker: ElectronDensityCalculationWorker | None = (
            None
        )
        self._workspace_load_thread: QThread | None = None
        self._workspace_load_worker: (
            ElectronDensityWorkspaceLoadWorker | None
        ) = None
        self._workspace_load_progress_dialog: SAXSProgressDialog | None = None
        self._batch_operation_progress_dialog: SAXSProgressDialog | None = None
        self._restore_distribution_state_after_workspace_load = False
        self._saved_output_entries: list[_SavedOutputEntry] = []
        self._output_history_compare_dialog: QDialog | None = None
        self._debye_scattering_compare_dialog: QDialog | None = None
        self._updating_fourier_settings_table = False
        self._syncing_cluster_selection = False
        self._deferred_initial_input_path: Path | None = None
        self._deferred_restore_saved_distribution_state = False
        self._last_workspace_load_message = ""
        self._cluster_view_cache_prewarm_queue: list[str] = []
        self._cluster_view_cache_prewarm_timer = QTimer(self)
        self._cluster_view_cache_prewarm_timer.setSingleShot(True)
        self._cluster_view_cache_prewarm_timer.timeout.connect(
            self._prime_next_cluster_view_cache_entry
        )
        self._initial_workspace_load_timer = QTimer(self)
        self._initial_workspace_load_timer.setSingleShot(True)
        self._initial_workspace_load_timer.timeout.connect(
            self._run_deferred_initial_workspace_load
        )
        self._project_debye_waller_source_path: Path | None = None

        self._apply_preview_mode_title()
        self.setWindowIcon(load_saxshell_icon())
        self.resize(1520, 960)
        self._build_ui()
        self._load_project_debye_waller_terms()
        self._set_initial_defaults()
        if initial_input_path is not None:
            self.input_path_edit.setText(
                str(Path(initial_input_path).expanduser().resolve())
            )
            self._load_input_path(Path(initial_input_path))
        if restore_saved_distribution_state_on_init:
            self._restore_saved_distribution_state_if_available()
        else:
            self._update_push_to_model_state()

    def _build_ui(self) -> None:
        self._build_menu_bar()
        central = QWidget(self)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(8)
        self._pane_splitter = QSplitter(Qt.Orientation.Horizontal, central)
        root_layout.addWidget(self._pane_splitter, stretch=1)
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
        self._pane_splitter.addWidget(self._left_scroll_area)

        self.preview_mode_banner = QLabel()
        self.preview_mode_banner.setWordWrap(True)
        self.preview_mode_banner.setStyleSheet(
            "QLabel { padding: 8px; border: 1px solid #cbd5e1; "
            "background: #f8fafc; }"
        )
        left_layout.addWidget(self.preview_mode_banner)

        left_layout.addWidget(self._build_input_group())
        left_layout.addWidget(self._build_output_group())
        self.mesh_section = _CollapsibleSection(
            "Mesh Settings", self._build_mesh_group(), left_container
        )
        left_layout.addWidget(self.mesh_section)
        left_layout.addWidget(self._build_actions_group())
        self.smearing_section = _CollapsibleSection(
            "Smearing",
            self._build_smearing_group(),
            left_container,
        )
        left_layout.addWidget(self.smearing_section)
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
        self.push_to_model_group = self._build_push_to_model_group()
        left_layout.addWidget(self.push_to_model_group)
        self.debye_scattering_group = self._build_debye_scattering_group()
        left_layout.addWidget(self.debye_scattering_group)
        self.output_history_section = _CollapsibleSection(
            "Saved Outputs",
            self._build_output_history_group(),
            left_container,
        )
        self.output_history_section.collapse()
        left_layout.addWidget(self.output_history_section)
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
        self.show_all_cluster_transforms_checkbox = QCheckBox(
            "Show All Cluster Transforms"
        )
        self.show_all_cluster_transforms_checkbox.setChecked(False)
        self.show_all_cluster_transforms_checkbox.toggled.connect(
            self._refresh_scattering_plot
        )
        plot_options_row.addWidget(self.show_all_cluster_transforms_checkbox)
        plot_options_row.addStretch(1)
        self.collapse_expand_button = QPushButton("Expand All")
        self.collapse_expand_button.setToolTip(
            "Expand or collapse all plot panels at once"
        )
        self.collapse_expand_button.clicked.connect(
            self._toggle_all_plot_sections
        )
        plot_options_row.addWidget(self.collapse_expand_button)
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
        self.fourier_preview_info_label = QLabel()
        self.fourier_preview_info_label.setStyleSheet(
            "color: #334155; font-size: 11px;"
        )
        self.fourier_preview_section.add_header_widget(
            self.fourier_preview_info_label
        )

        self.scattering_plot = ElectronDensityScatteringPlot(self._right_panel)
        self.scattering_section = _CollapsibleSection(
            "Scattering Profile I(q)",
            self.scattering_plot,
            self._right_panel,
        )
        right_layout.addWidget(self.scattering_section)
        self.scattering_info_label = QLabel()
        self.scattering_info_label.setStyleSheet(
            "color: #7c2d12; font-size: 11px;"
        )
        self.scattering_section.add_header_widget(self.scattering_info_label)

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
            section.toggled.connect(self._sync_collapse_expand_button)
        self.profile_plot.set_variance_visible(True)
        self.smeared_profile_plot.set_variance_visible(True)
        self.residual_profile_plot.set_variance_visible(True)

        self._right_scroll_area.setWidget(self._right_panel)
        self._refresh_right_panel_layout()
        self._pane_splitter.addWidget(self._right_scroll_area)
        self._pane_splitter.setStretchFactor(0, 0)
        self._pane_splitter.setStretchFactor(1, 1)
        self._pane_splitter.setSizes([430, 1090])
        self._auto_snap_filter = PaneSnapFilter(
            self._pane_splitter,
            self._left_scroll_area,
            self._right_scroll_area,
            self,
        )
        self._set_auto_snap_panes_enabled(
            self._auto_snap_panes_enabled,
            persist=False,
        )
        self._refresh_preview_mode_banner()

    def _build_menu_bar(self) -> None:
        settings_menu = self.menuBar().addMenu("Settings")
        self.auto_snap_panes_action = QAction("Auto-Snap Panes", self)
        self.auto_snap_panes_action.setCheckable(True)
        self.auto_snap_panes_action.setChecked(
            bool(self._auto_snap_panes_enabled)
        )
        self.auto_snap_panes_action.triggered.connect(
            self._toggle_auto_snap_panes
        )
        settings_menu.addAction(self.auto_snap_panes_action)

    def _apply_preview_mode_title(self) -> None:
        title = "Electron Density Mapping"
        if self._preview_mode:
            title += " (Preview)"
        self.setWindowTitle(title)

    def _refresh_preview_mode_banner(self) -> None:
        if self._preview_mode:
            self.preview_mode_banner.setText(
                "Preview Mode: this tool can inspect standalone structures, "
                "folders, or cluster folders without saving model components "
                "into a computed distribution. Use Build SAXS Components from "
                "the main UI to launch the linked workflow."
            )
            self.preview_mode_banner.setToolTip(
                "Preview mode does not push component traces into the active "
                "SAXS model."
            )
            return
        distribution_text = (
            self._distribution_id
            if self._distribution_id is not None
            else "active computed distribution"
        )
        self.preview_mode_banner.setText(
            "Computed Distribution Mode: this run is linked to "
            f"{distribution_text}. Output defaults and q-range context are "
            "inherited from the active SAXS project."
        )
        self.preview_mode_banner.setToolTip(
            "This workflow was launched from Build SAXS Components for the "
            "active computed distribution."
        )

    def set_preview_mode(self, preview_mode: bool) -> None:
        self._preview_mode = bool(preview_mode)
        self._apply_preview_mode_title()
        self._refresh_preview_mode_banner()

    def schedule_initial_workspace_load(
        self,
        *,
        initial_input_path: Path | None = None,
        restore_saved_distribution_state: bool | None = None,
    ) -> None:
        self._deferred_initial_input_path = (
            None
            if initial_input_path is None
            else Path(initial_input_path).expanduser().resolve()
        )
        self._deferred_restore_saved_distribution_state = bool(
            (not self._preview_mode)
            if restore_saved_distribution_state is None
            else restore_saved_distribution_state
        )
        if self._deferred_initial_input_path is not None:
            self.input_path_edit.setText(
                str(self._deferred_initial_input_path)
            )
        if (
            self._deferred_initial_input_path is None
            and not self._deferred_restore_saved_distribution_state
        ):
            return
        self._reset_progress_display("Loading inherited workspace...")
        self.statusBar().showMessage(
            "Loading inherited electron-density workspace"
        )
        self._initial_workspace_load_timer.start(0)

    @Slot()
    def _run_deferred_initial_workspace_load(self) -> None:
        input_path = self._deferred_initial_input_path
        restore_saved_distribution_state = (
            self._deferred_restore_saved_distribution_state
        )
        self._deferred_initial_input_path = None
        self._deferred_restore_saved_distribution_state = False
        if input_path is not None:
            self._start_workspace_load(
                input_path,
                restore_saved_distribution_state=restore_saved_distribution_state,
                progress_message=(
                    "Loading inherited electron-density workspace..."
                ),
            )
            return
        if restore_saved_distribution_state:
            restore_message = (
                "Restoring saved electron-density mapping state..."
            )
            self.calculation_progress_bar.setRange(0, 0)
            self.calculation_progress_bar.setFormat("")
            self.calculation_progress_message.setText(
                "Restoring saved computed-distribution state..."
            )
            self.statusBar().showMessage(restore_message)
            self._set_workspace_load_progress_busy_message(restore_message)
            QApplication.processEvents()
            if self._restore_saved_distribution_state_if_available():
                return
            self._close_workspace_load_progress_dialog()
            self._reset_progress_display("Idle")
        elif input_path is None:
            self._reset_progress_display("Idle")

    def _start_workspace_load(
        self,
        input_path: Path,
        *,
        restore_saved_distribution_state: bool,
        progress_message: str = "Loading electron-density workspace...",
    ) -> None:
        if (
            self._workspace_load_thread is not None
            and self._workspace_load_thread.isRunning()
        ):
            return
        progress_text = (
            str(progress_message).strip()
            or "Loading electron-density workspace..."
        )
        self._restore_distribution_state_after_workspace_load = bool(
            restore_saved_distribution_state
        )
        self._set_calculation_running(True)
        self.stop_calculation_button.setEnabled(False)
        self.calculation_progress_bar.setRange(0, 0)
        self.calculation_progress_bar.setFormat("")
        self.calculation_progress_message.setText(progress_text)
        self.statusBar().showMessage(progress_text)
        self._begin_workspace_load_progress(progress_text)

        self._workspace_load_thread = QThread()
        self._workspace_load_worker = ElectronDensityWorkspaceLoadWorker(
            input_path
        )
        self._workspace_load_worker.moveToThread(self._workspace_load_thread)
        self._workspace_load_thread.started.connect(
            self._workspace_load_worker.run
        )
        self._workspace_load_worker.progress.connect(
            self._on_workspace_load_progress
        )
        self._workspace_load_worker.finished.connect(
            self._on_workspace_load_finished
        )
        self._workspace_load_worker.failed.connect(
            self._on_workspace_load_failed
        )
        self._workspace_load_worker.finished.connect(
            self._workspace_load_thread.quit
        )
        self._workspace_load_worker.failed.connect(
            self._workspace_load_thread.quit
        )
        self._workspace_load_thread.finished.connect(
            self._workspace_load_worker.deleteLater
        )
        self._workspace_load_thread.finished.connect(
            self._workspace_load_thread.deleteLater
        )
        self._workspace_load_thread.finished.connect(
            self._clear_workspace_load_handles
        )
        self._workspace_load_thread.start(QThread.Priority.LowPriority)

    @Slot()
    def _clear_workspace_load_handles(self) -> None:
        self._workspace_load_worker = None
        self._workspace_load_thread = None

    def _ensure_workspace_load_progress_dialog(self) -> SAXSProgressDialog:
        if self._workspace_load_progress_dialog is None:
            self._workspace_load_progress_dialog = SAXSProgressDialog(self)
        return self._workspace_load_progress_dialog

    def _begin_workspace_load_progress(self, message: str) -> None:
        dialog = self._ensure_workspace_load_progress_dialog()
        self._last_workspace_load_message = ""
        dialog.begin_busy(
            str(message).strip() or "Loading electron-density workspace...",
            title="Loading Electron Density Mapping",
        )
        QApplication.processEvents()

    def _set_workspace_load_progress_busy_message(self, message: str) -> None:
        dialog = self._ensure_workspace_load_progress_dialog()
        stripped = (
            str(message).strip() or "Loading electron-density workspace..."
        )
        dialog.setWindowTitle("Loading Electron Density Mapping")
        dialog.progress_bar.setRange(0, 0)
        dialog.progress_bar.setValue(0)
        dialog.progress_bar.setFormat("")
        dialog.message_label.setText(stripped)
        dialog.show()
        dialog.raise_()
        QApplication.processEvents()

    def _append_workspace_load_output(self, message: str) -> None:
        stripped = str(message).strip()
        if (
            not stripped
            or stripped == self._last_workspace_load_message
            or self._workspace_load_progress_dialog is None
        ):
            return
        self._last_workspace_load_message = stripped
        self._workspace_load_progress_dialog.append_output(stripped)

    def _close_workspace_load_progress_dialog(self) -> None:
        self._last_workspace_load_message = ""
        if self._workspace_load_progress_dialog is not None:
            self._workspace_load_progress_dialog.close()

    def _ensure_batch_operation_progress_dialog(self) -> SAXSProgressDialog:
        if self._batch_operation_progress_dialog is None:
            dialog = SAXSProgressDialog(self)
            dialog.setModal(True)
            dialog.setWindowModality(Qt.WindowModality.WindowModal)
            self._batch_operation_progress_dialog = dialog
        return self._batch_operation_progress_dialog

    def _begin_batch_operation_progress(
        self,
        *,
        total: int,
        message: str,
        title: str,
    ) -> None:
        bounded_total = max(int(total), 1)
        stripped = str(message).strip() or "Preparing batch update..."
        dialog = self._ensure_batch_operation_progress_dialog()
        dialog.begin(
            bounded_total,
            stripped,
            unit_label="stoichiometries",
            title=title,
        )
        self._on_calculation_progress(0, bounded_total, stripped)
        self.statusBar().showMessage(stripped)
        QApplication.processEvents()

    def _update_batch_operation_progress(
        self,
        current: int,
        total: int,
        message: str,
    ) -> None:
        bounded_total = max(int(total), 1)
        stripped = str(message).strip() or "Processing stoichiometry..."
        self._on_calculation_progress(current, bounded_total, stripped)
        if self._batch_operation_progress_dialog is not None:
            self._batch_operation_progress_dialog.update_progress(
                current,
                bounded_total,
                stripped,
                unit_label="stoichiometries",
            )
        self.statusBar().showMessage(stripped)
        QApplication.processEvents()

    def _close_batch_operation_progress_dialog(self) -> None:
        if self._batch_operation_progress_dialog is not None:
            self._batch_operation_progress_dialog.close()

    @Slot(int, int, str)
    def _on_workspace_load_progress(
        self,
        current: int,
        total: int,
        message: str,
    ) -> None:
        self._on_calculation_progress(current, total, message)
        stripped = str(message).strip()
        if self._workspace_load_progress_dialog is not None:
            self._workspace_load_progress_dialog.update_progress(
                current,
                total,
                stripped or "Loading inherited electron-density workspace...",
                unit_label="steps",
            )
        self._append_workspace_load_output(stripped)
        if stripped:
            self.statusBar().showMessage(stripped)
        QApplication.processEvents()

    @Slot(object)
    def _on_workspace_load_finished(self, payload: object) -> None:
        try:
            if not isinstance(payload, _ElectronDensityWorkspaceLoadPayload):
                raise RuntimeError(
                    "Workspace loader returned an unexpected payload type."
                )
            self._apply_workspace_load_payload(payload)
            if self._restore_distribution_state_after_workspace_load:
                restore_message = (
                    "Restoring saved electron-density mapping state..."
                )
                self.calculation_progress_bar.setRange(0, 0)
                self.calculation_progress_bar.setFormat("")
                self.calculation_progress_message.setText(
                    "Restoring saved computed-distribution state..."
                )
                self.statusBar().showMessage(restore_message)
                self._set_workspace_load_progress_busy_message(restore_message)
                QApplication.processEvents()
                self._restore_saved_distribution_state_if_available()
            self._reset_progress_display("Idle")
        except Exception as exc:
            self._handle_workspace_load_failure(str(exc))
            return
        finally:
            self._restore_distribution_state_after_workspace_load = False
            self._set_calculation_running(False)
            self._close_workspace_load_progress_dialog()

    @Slot(str)
    def _on_workspace_load_failed(self, message: str) -> None:
        self._handle_workspace_load_failure(message)

    def _handle_workspace_load_failure(self, message: str) -> None:
        self._restore_distribution_state_after_workspace_load = False
        self._set_calculation_running(False)
        self._reset_progress_display("Workspace load failed.")
        self._append_status(f"Workspace load failed: {message}")
        self._close_workspace_load_progress_dialog()
        self._show_error("Input Error", message)

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

    def _plot_sections(self) -> tuple[_CollapsibleSection, ...]:
        return (
            self.profile_section,
            self.smeared_section,
            self.residual_section,
            self.fourier_preview_section,
            self.scattering_section,
        )

    @Slot()
    def _sync_collapse_expand_button(self, *_args: object) -> None:
        any_expanded = any(s.is_expanded for s in self._plot_sections())
        self.collapse_expand_button.setText(
            "Collapse All" if any_expanded else "Expand All"
        )

    @Slot()
    def _toggle_all_plot_sections(self) -> None:
        sections = self._plot_sections()
        any_expanded = any(s.is_expanded for s in sections)
        for section in sections:
            section.set_expanded(not any_expanded)

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
        form.addRow(QLabel("Input mode:"))
        self.input_mode_value = QLabel("No structure loaded")
        self.input_mode_value.setWordWrap(True)
        form.addRow(self.input_mode_value)

        form.addRow(QLabel("Reference file:"))
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
        self.center_mode_value = QLabel("Geometric Mass Center")
        self.center_mode_value.setWordWrap(True)
        layout.addRow(self.center_mode_value)

        layout.addRow(QLabel("Geometric mass center:"))
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

        self.reset_center_button = QPushButton("Geometric Mass Center")
        self.reset_center_button.setCheckable(True)
        self.reset_center_button.setToolTip(
            "Reset active center to the geometric mass center"
        )
        self.reset_center_button.clicked.connect(
            lambda _checked=False: self._apply_center_mode("center_of_mass")
        )
        center_button_grid.addWidget(self.reset_center_button, 0, 0)

        self.snap_center_button = QPushButton("Nearest Atom")
        self.snap_center_button.setCheckable(True)
        self.snap_center_button.setToolTip(
            "Snap active center to the atom nearest to the geometric mass center"
        )
        self.snap_center_button.clicked.connect(
            lambda _checked=False: self._apply_center_mode("nearest_atom")
        )
        center_button_grid.addWidget(self.snap_center_button, 0, 1)

        self.snap_reference_center_button = QPushButton("Reference Element")
        self.snap_reference_center_button.setCheckable(True)
        self.snap_reference_center_button.setToolTip(
            "Snap active center to the geometric center of the reference element"
        )
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

        self.pinned_geometric_tracking_checkbox = QCheckBox(
            "Pin Geometric Tracking Across Contiguous PDB Frames"
        )
        self.pinned_geometric_tracking_checkbox.setToolTip(
            "For folder and stoichiometry-folder runs with contiguous PDB "
            "frame_<NNNN> names, reuse the first frame's geometric mass "
            "center coordinates across each contiguous set instead of "
            "recomputing the center for every frame."
        )
        self.pinned_geometric_tracking_checkbox.setChecked(
            bool(self._active_mesh_settings.pin_contiguous_geometric_tracking)
        )
        self.pinned_geometric_tracking_checkbox.toggled.connect(
            self._handle_pinned_geometric_tracking_toggled
        )
        layout.addRow(self.pinned_geometric_tracking_checkbox)

        self.pinned_geometric_tracking_notice = QLabel()
        self.pinned_geometric_tracking_notice.setWordWrap(True)
        self.pinned_geometric_tracking_notice.setStyleSheet("color: #475569;")
        layout.addRow(self.pinned_geometric_tracking_notice)

        self.contiguous_frame_mode_checkbox = QCheckBox(
            "Use Contiguous Frame Evaluation"
        )
        self.contiguous_frame_mode_checkbox.setChecked(True)
        self.contiguous_frame_mode_checkbox.setToolTip(
            "When averaging multiple structures, group files by frame_<NNNN> "
            "sequences and lock each contiguous set to a shared center offset "
            "relative to the heaviest-element geometric center. Files without "
            "frame_<NNNN> identifiers fall back to complete averaging."
        )
        self.contiguous_frame_mode_checkbox.toggled.connect(
            self._handle_contiguous_frame_mode_toggled
        )
        layout.addRow(self.contiguous_frame_mode_checkbox)

        self.contiguous_frame_mode_notice = QLabel(
            "Default: on. Folder and cluster-folder runs will try contiguous "
            "frame grouping from frame_<NNNN> file names, then fall back to "
            "complete averaging with a status notice when that naming scheme "
            "is unavailable."
        )
        self.contiguous_frame_mode_notice.setWordWrap(True)
        self.contiguous_frame_mode_notice.setStyleSheet("color: #475569;")
        layout.addRow(self.contiguous_frame_mode_notice)

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
        self._refresh_pinned_geometric_tracking_controls()
        return group

    def _build_actions_group(self) -> QWidget:
        group = QGroupBox("Actions")
        layout = QVBoxLayout(group)
        self.cluster_group_status_label = QLabel(
            "Single-file and folder inputs run directly. Loading a folder of "
            "stoichiometry subfolders will populate a per-cluster summary table here."
        )
        self.cluster_group_status_label.setWordWrap(True)
        layout.addWidget(self.cluster_group_status_label)

        cluster_group_table_body = QWidget(group)
        cluster_group_table_layout = QVBoxLayout(cluster_group_table_body)
        cluster_group_table_layout.setContentsMargins(0, 0, 0, 0)
        cluster_group_table_layout.setSpacing(0)

        self.cluster_group_table = QTableWidget(0, 14)
        self.cluster_group_table.setHorizontalHeaderLabels(
            [
                "Stoichiometry",
                "Files",
                "Avg Atoms",
                "Center Mode",
                "Center Ref",
                "Center Element",
                "Density",
                "Fourier",
                "Trace Color",
                "Smearing",
                "Solvent",
                "Cutoff",
                "FT Settings",
                "Reference File",
            ]
        )
        self.cluster_group_table.verticalHeader().setVisible(False)
        self.cluster_group_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self.cluster_group_table.setSelectionMode(
            QTableWidget.SelectionMode.ExtendedSelection
        )
        self.cluster_group_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers
        )
        self.cluster_group_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        self.cluster_group_table.horizontalHeader().setStretchLastSection(True)
        self.cluster_group_table.setMinimumHeight(180)
        self.cluster_group_table.itemSelectionChanged.connect(
            self._handle_cluster_group_selection_changed
        )
        cluster_group_table_layout.addWidget(self.cluster_group_table)

        self.cluster_group_table_section = _CollapsibleSection(
            "Stoichiometry Table",
            cluster_group_table_body,
            group,
        )
        self.cluster_group_table_section.collapse()
        layout.addWidget(self.cluster_group_table_section)

        self.cluster_batch_scope_label = QLabel()
        self.cluster_batch_scope_label.setWordWrap(True)
        self.cluster_batch_scope_label.setStyleSheet("color: #475569;")
        layout.addWidget(self.cluster_batch_scope_label)

        self.manual_mode_checkbox = QCheckBox(
            "Manual Mode (selected stoichiometry only)"
        )
        self.manual_mode_checkbox.setToolTip(
            "Run only the actively selected stoichiometry row. After the "
            "first manual calculation succeeds, mesh settings stay locked "
            "until you reset the calculated densities."
        )
        self.manual_mode_checkbox.toggled.connect(
            self._refresh_manual_mode_notice
        )
        layout.addWidget(self.manual_mode_checkbox)

        self.manual_mode_notice_label = QLabel()
        self.manual_mode_notice_label.setWordWrap(True)
        self.manual_mode_notice_label.setStyleSheet("color: #475569;")
        layout.addWidget(self.manual_mode_notice_label)

        self.cluster_completion_indicator = QLabel("PENDING")
        self.cluster_completion_indicator.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )
        self.cluster_completion_indicator.setMinimumWidth(92)
        self.cluster_completion_indicator.setStyleSheet(
            "background-color: #94a3b8; color: white; padding: 4px 8px; "
            "border-radius: 8px; font-weight: 600;"
        )
        self.cluster_completion_tracker_label = QLabel(
            "Computed stoichiometries: n/a"
        )
        self.cluster_completion_tracker_label.setWordWrap(True)
        tracker_row = QWidget(group)
        tracker_layout = QHBoxLayout(tracker_row)
        tracker_layout.setContentsMargins(0, 0, 0, 0)
        tracker_layout.setSpacing(8)
        tracker_layout.addWidget(self.cluster_completion_indicator, stretch=0)
        tracker_layout.addWidget(
            self.cluster_completion_tracker_label,
            stretch=1,
        )
        layout.addWidget(tracker_row)

        self.run_button = QPushButton("Run Electron Density Calculation")
        self.run_button.clicked.connect(self._run_calculation)
        self.stop_calculation_button = QPushButton("Stop Active Calculation")
        self.stop_calculation_button.clicked.connect(
            self._request_calculation_stop
        )
        run_row = QWidget(group)
        run_layout = QHBoxLayout(run_row)
        run_layout.setContentsMargins(0, 0, 0, 0)
        run_layout.setSpacing(6)
        run_layout.addWidget(self.run_button, stretch=1)
        run_layout.addWidget(self.stop_calculation_button, stretch=0)
        layout.addWidget(run_row)

        self.reset_calculations_button = QPushButton(
            "Reset Calculated Densities"
        )
        self.reset_calculations_button.clicked.connect(
            self._reset_calculated_densities
        )
        layout.addWidget(self.reset_calculations_button)

        self.calculation_overall_progress_message = QLabel(
            "Overall progress: idle"
        )
        self.calculation_overall_progress_message.setWordWrap(True)
        self.calculation_overall_progress_message.setHidden(True)
        layout.addWidget(self.calculation_overall_progress_message)

        self.calculation_overall_progress_bar = QProgressBar()
        self.calculation_overall_progress_bar.setRange(0, 1)
        self.calculation_overall_progress_bar.setValue(0)
        self.calculation_overall_progress_bar.setFormat(
            "%v / %m cluster groups"
        )
        self.calculation_overall_progress_bar.setHidden(True)
        layout.addWidget(self.calculation_overall_progress_bar)

        self.calculation_progress_message = QLabel("Idle")
        self.calculation_progress_message.setWordWrap(True)
        layout.addWidget(self.calculation_progress_message)

        self.calculation_progress_bar = QProgressBar()
        self.calculation_progress_bar.setRange(0, 1)
        self.calculation_progress_bar.setValue(0)
        self.calculation_progress_bar.setFormat("%v / %m steps")
        layout.addWidget(self.calculation_progress_bar)
        return group

    def _build_push_to_model_group(self) -> QWidget:
        group = QGroupBox("Push to Model")
        layout = QVBoxLayout(group)
        self.push_to_model_status_label = QLabel()
        self.push_to_model_status_label.setWordWrap(True)
        layout.addWidget(self.push_to_model_status_label)

        self.push_to_model_button = QPushButton("Push to Model")
        self.push_to_model_button.setToolTip(
            "Write the active Born-approximation component traces into the "
            "linked computed distribution so the main SAXS UI can load them."
        )
        self.push_to_model_button.clicked.connect(
            self._push_components_to_model
        )
        layout.addWidget(self.push_to_model_button)
        self._update_push_to_model_state()
        return group

    def _build_debye_scattering_group(self) -> QWidget:
        group = QGroupBox("Debye Scattering Calculation")
        layout = QVBoxLayout(group)
        self.debye_scattering_summary_label = QLabel(
            "Compute averaged Debye scattering traces on the same q-grid as "
            "the current Born-approximation I(Q) results, then open a "
            "deployable comparison plot with paired overlays."
        )
        self.debye_scattering_summary_label.setWordWrap(True)
        layout.addWidget(self.debye_scattering_summary_label)

        self.debye_scattering_status_label = QLabel()
        self.debye_scattering_status_label.setWordWrap(True)
        layout.addWidget(self.debye_scattering_status_label)

        self.debye_scattering_progress_bar = QProgressBar()
        self.debye_scattering_progress_bar.setRange(0, 1)
        self.debye_scattering_progress_bar.setValue(0)
        self.debye_scattering_progress_bar.setFormat("%v / %m steps")
        self.debye_scattering_progress_bar.setHidden(True)
        layout.addWidget(self.debye_scattering_progress_bar)

        controls = QHBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(6)

        self.calculate_debye_scattering_button = QPushButton(
            "Calculate Debye Scattering"
        )
        self.calculate_debye_scattering_button.clicked.connect(
            self._calculate_debye_scattering_action
        )
        controls.addWidget(self.calculate_debye_scattering_button)

        self.apply_debye_to_all_button = QCheckBox("Apply to All Rows")
        self.apply_debye_to_all_button.toggled.connect(
            self._refresh_debye_scattering_group
        )
        controls.addWidget(self.apply_debye_to_all_button)

        self.open_debye_scattering_compare_button = QPushButton(
            "Open Comparison Plot"
        )
        self.open_debye_scattering_compare_button.clicked.connect(
            self._open_debye_scattering_comparison_plot
        )
        controls.addWidget(self.open_debye_scattering_compare_button)
        controls.addStretch(1)
        layout.addLayout(controls)

        self._refresh_debye_scattering_group()
        return group

    @staticmethod
    def _debye_scattering_progress_total_for_inspection(
        inspection: ElectronDensityInputInspection,
    ) -> int:
        return max(len(tuple(inspection.structure_files)) * 2 + 2, 1)

    def _reset_debye_scattering_progress(self) -> None:
        if not hasattr(self, "debye_scattering_progress_bar"):
            return
        self.debye_scattering_progress_bar.setRange(0, 1)
        self.debye_scattering_progress_bar.setValue(0)
        self.debye_scattering_progress_bar.setFormat("%v / %m steps")
        self.debye_scattering_progress_bar.setHidden(True)

    def _begin_debye_scattering_progress(
        self,
        *,
        total: int,
        message: str,
    ) -> None:
        if not hasattr(self, "debye_scattering_progress_bar"):
            return
        bounded_total = max(int(total), 1)
        stripped = (
            str(message).strip() or "Preparing Debye scattering calculation..."
        )
        self.debye_scattering_progress_bar.setHidden(False)
        self.debye_scattering_progress_bar.setRange(0, bounded_total)
        self.debye_scattering_progress_bar.setValue(0)
        self.debye_scattering_progress_bar.setFormat("%v / %m steps")
        self.debye_scattering_status_label.setText(stripped)
        self.statusBar().showMessage(stripped)
        QApplication.processEvents()

    def _update_debye_scattering_progress(
        self,
        current: int,
        total: int,
        message: str,
    ) -> None:
        if not hasattr(self, "debye_scattering_progress_bar"):
            return
        bounded_total = max(int(total), 1)
        bounded_current = min(max(int(current), 0), bounded_total)
        stripped = str(message).strip()
        self.debye_scattering_progress_bar.setHidden(False)
        self.debye_scattering_progress_bar.setRange(0, bounded_total)
        self.debye_scattering_progress_bar.setValue(bounded_current)
        self.debye_scattering_progress_bar.setFormat("%v / %m steps")
        if stripped:
            self.debye_scattering_status_label.setText(stripped)
            self.statusBar().showMessage(stripped)
        QApplication.processEvents()

    @staticmethod
    def _cluster_trace_color(index: int) -> str:
        return _cluster_trace_color_for_index(index)

    def _active_cluster_group_state(self) -> _ClusterDensityGroupState | None:
        return self._cluster_group_state_by_key(
            str(self._selected_cluster_group_key or "")
        )

    def _cluster_group_state_by_key(
        self,
        group_key: str,
    ) -> _ClusterDensityGroupState | None:
        for state in self._cluster_group_states:
            if state.key == group_key:
                return state
        return None

    def _cluster_group_render_complexity(
        self,
        state: _ClusterDensityGroupState,
    ) -> float:
        structure = state.reference_structure
        bond_count = len(structure.bonds)
        atom_count = float(max(structure.atom_count, state.average_atom_count))
        return atom_count + bond_count * 0.35

    def _schedule_cluster_view_cache_prewarm(self) -> None:
        self._cluster_view_cache_prewarm_timer.stop()
        self._cluster_view_cache_prewarm_queue = []
        if len(self._cluster_group_states) <= 1:
            return
        active_key = self._selected_cluster_group_key
        ranked_states = sorted(
            self._cluster_group_states,
            key=self._cluster_group_render_complexity,
            reverse=True,
        )
        self._cluster_view_cache_prewarm_queue = [
            state.key for state in ranked_states if state.key != active_key
        ]
        if self._cluster_view_cache_prewarm_queue:
            self._cluster_view_cache_prewarm_timer.start(0)

    def _prime_next_cluster_view_cache_entry(self) -> None:
        if not self._cluster_view_cache_prewarm_queue:
            return
        scene_key = self._cluster_view_cache_prewarm_queue.pop(0)
        if scene_key == self._selected_cluster_group_key:
            if self._cluster_view_cache_prewarm_queue:
                self._cluster_view_cache_prewarm_timer.start(0)
            return
        target_state = next(
            (
                state
                for state in self._cluster_group_states
                if state.key == scene_key
            ),
            None,
        )
        if target_state is None:
            if self._cluster_view_cache_prewarm_queue:
                self._cluster_view_cache_prewarm_timer.start(0)
            return
        mesh_geometry = (
            None
            if target_state.profile_result is None
            else target_state.profile_result.mesh_geometry
        )
        self.structure_viewer.prewarm_scene(
            target_state.key,
            structure=target_state.reference_structure,
            mesh_geometry=mesh_geometry,
            complexity_score=self._cluster_group_render_complexity(
                target_state
            ),
        )
        if self._cluster_view_cache_prewarm_queue:
            self._cluster_view_cache_prewarm_timer.start(0)

    def _selected_cluster_group_rows(self) -> list[int]:
        selection_model = self.cluster_group_table.selectionModel()
        if selection_model is None:
            return []
        return sorted(
            {
                index.row()
                for index in selection_model.selectedRows()
                if 0 <= index.row() < len(self._cluster_group_states)
            }
        )

    def _batch_target_cluster_group_states(
        self,
        *,
        apply_to_all: bool,
    ) -> tuple[list[_ClusterDensityGroupState], str, list[str]]:
        if apply_to_all:
            return (
                list(self._cluster_group_states),
                f"all {len(self._cluster_group_states)} stoichiometr"
                f"{'y' if len(self._cluster_group_states) == 1 else 'ies'}",
                [],
            )
        selected_rows = self._selected_cluster_group_rows()
        if not selected_rows:
            active_state = self._active_cluster_group_state()
            if active_state is None:
                return [], "0 selected stoichiometries", []
            return (
                [active_state],
                "1 selected stoichiometry",
                [active_state.key],
            )
        selected_states = [
            self._cluster_group_states[row_index]
            for row_index in selected_rows
        ]
        return (
            selected_states,
            f"{len(selected_states)} selected stoichiometr"
            f"{'y' if len(selected_states) == 1 else 'ies'}",
            [state.key for state in selected_states],
        )

    def _restore_cluster_group_selection(
        self,
        group_keys: list[str],
        *,
        preferred_current_key: str | None = None,
    ) -> None:
        if not self._cluster_group_states:
            return
        selection_model = self.cluster_group_table.selectionModel()
        if selection_model is None:
            return
        key_set = {str(key) for key in group_keys if str(key).strip()}
        if not key_set:
            return
        self.cluster_group_table.blockSignals(True)
        try:
            selection_model.clearSelection()
            current_row = -1
            fallback_row = -1
            for row_index, state in enumerate(self._cluster_group_states):
                if state.key in key_set:
                    index = self.cluster_group_table.model().index(
                        row_index,
                        0,
                    )
                    selection_model.select(
                        index,
                        QItemSelectionModel.SelectionFlag.Select
                        | QItemSelectionModel.SelectionFlag.Rows,
                    )
                    if fallback_row < 0:
                        fallback_row = row_index
                if state.key == preferred_current_key:
                    current_row = row_index
            if current_row < 0:
                current_row = fallback_row
            if current_row >= 0:
                self.cluster_group_table.setCurrentCell(current_row, 0)
        finally:
            self.cluster_group_table.blockSignals(False)

    def _set_current_cluster_group_row(
        self,
        group_key: str,
        *,
        preserve_selection: bool,
    ) -> None:
        if not self._cluster_group_states:
            return
        selection_model = self.cluster_group_table.selectionModel()
        if selection_model is None:
            return
        target_row = -1
        for row_index, state in enumerate(self._cluster_group_states):
            if state.key == group_key:
                target_row = row_index
                break
        if target_row < 0:
            return
        index = self.cluster_group_table.model().index(target_row, 0)
        self.cluster_group_table.blockSignals(True)
        try:
            if preserve_selection:
                selection_model.select(
                    index,
                    QItemSelectionModel.SelectionFlag.Select
                    | QItemSelectionModel.SelectionFlag.Rows,
                )
            else:
                selection_model.clearSelection()
                selection_model.select(
                    index,
                    QItemSelectionModel.SelectionFlag.Select
                    | QItemSelectionModel.SelectionFlag.Rows,
                )
            self.cluster_group_table.setCurrentCell(target_row, 0)
        finally:
            self.cluster_group_table.blockSignals(False)

    @staticmethod
    def _solvent_metadata_for_result(
        result: ElectronDensityProfileResult | None,
    ) -> tuple[float | None, float | None]:
        contrast = None if result is None else result.solvent_contrast
        density = (
            None
            if contrast is None
            else float(contrast.solvent_density_e_per_a3)
        )
        cutoff = (
            None
            if contrast is None or contrast.cutoff_radius_a is None
            else float(contrast.cutoff_radius_a)
        )
        return density, cutoff

    def _sync_cluster_state_solvent_metadata(
        self,
        state: _ClusterDensityGroupState,
    ) -> None:
        density, cutoff = self._solvent_metadata_for_result(
            state.profile_result
        )
        state.solvent_density_e_per_a3 = density
        state.solvent_cutoff_radius_a = cutoff

    def _cluster_state_fourier_settings(
        self,
        state: _ClusterDensityGroupState,
    ) -> ElectronDensityFourierTransformSettings:
        if state.fourier_settings is not None:
            return state.fourier_settings.normalized()
        if state.transform_result is not None:
            return state.transform_result.preview.settings
        try:
            base_settings = self._fourier_settings_from_controls()
        except Exception:
            base_settings = self._active_fourier_settings
        if state.profile_result is not None and not state.single_atom_only:
            return self._constrain_fourier_settings_for_result(
                state.profile_result,
                base_settings,
                prefer_solvent_cutoff=bool(state.solvent_cutoff_radius_a),
            )
        return base_settings.normalized()

    def _set_cluster_state_fourier_settings(
        self,
        state: _ClusterDensityGroupState,
        settings: ElectronDensityFourierTransformSettings,
        *,
        prefer_solvent_cutoff: bool,
    ) -> ElectronDensityFourierTransformSettings:
        current_settings = self._cluster_state_fourier_settings(state)
        next_settings = settings.normalized()
        if state.profile_result is not None and not state.single_atom_only:
            next_settings = self._constrain_fourier_settings_for_result(
                state.profile_result,
                next_settings,
                prefer_solvent_cutoff=prefer_solvent_cutoff,
            )
        if current_settings != next_settings:
            state.transform_result = None
            state.debye_scattering_result = None
            self._close_debye_scattering_compare_dialog()
        state.fourier_settings = next_settings
        return next_settings

    @staticmethod
    def _fourier_profile_label_for_settings(
        settings: ElectronDensityFourierTransformSettings,
        *,
        single_atom_only: bool,
    ) -> str:
        if single_atom_only:
            return "Debye"
        return (
            "solvent" if settings.use_solvent_subtracted_profile else "smeared"
        )

    def _shared_fourier_settings_for_state(
        self,
        state: _ClusterDensityGroupState,
        reference_settings: ElectronDensityFourierTransformSettings,
    ) -> ElectronDensityFourierTransformSettings:
        row_settings = self._cluster_state_fourier_settings(state)
        return ElectronDensityFourierTransformSettings(
            r_min=row_settings.r_min,
            r_max=row_settings.r_max,
            domain_mode=reference_settings.domain_mode,
            window_function=reference_settings.window_function,
            resampling_points=reference_settings.resampling_points,
            q_min=reference_settings.q_min,
            q_max=reference_settings.q_max,
            q_step=reference_settings.q_step,
            use_solvent_subtracted_profile=(
                reference_settings.use_solvent_subtracted_profile
            ),
            log_q_axis=row_settings.log_q_axis,
            log_intensity_axis=row_settings.log_intensity_axis,
        ).normalized()

    def _fourier_domain_mode_from_controls(self) -> str:
        if (
            hasattr(self, "fourier_legacy_mode_checkbox")
            and self.fourier_legacy_mode_checkbox.isChecked()
        ):
            return "legacy"
        return "mirrored"

    def _refresh_fourier_domain_mode_controls(self) -> None:
        if not hasattr(self, "fourier_rmin_label"):
            return
        mirrored_mode = self._fourier_domain_mode_from_controls() == "mirrored"
        self.fourier_rmin_label.setText("-r max" if mirrored_mode else "r min")
        if hasattr(self, "fourier_settings_table"):
            header_item = self.fourier_settings_table.horizontalHeaderItem(
                _FT_COLUMN_RMIN
            )
            if header_item is not None:
                header_item.setText("-r max" if mirrored_mode else "r min")
        self._sync_mirrored_fourier_rmin_to_rmax()

    def _sync_mirrored_fourier_rmin_to_rmax(self, *_args: object) -> None:
        if (
            not hasattr(self, "fourier_rmin_spin")
            or self._fourier_domain_mode_from_controls() != "mirrored"
        ):
            return
        target_rmin = -float(self.fourier_rmax_spin.value())
        self.fourier_rmin_spin.blockSignals(True)
        try:
            self.fourier_rmin_spin.setValue(target_rmin)
        finally:
            self.fourier_rmin_spin.blockSignals(False)

    @Slot(bool)
    def _handle_fourier_domain_mode_toggled(self, checked: bool) -> None:
        self._refresh_fourier_domain_mode_controls()
        if checked:
            self.fourier_rmin_spin.blockSignals(True)
            try:
                self.fourier_rmin_spin.setValue(
                    max(float(self.fourier_rmin_spin.value()), 0.0)
                )
            finally:
                self.fourier_rmin_spin.blockSignals(False)
        else:
            self._sync_mirrored_fourier_rmin_to_rmax()
        self._sync_fourier_controls_to_domain(reset_bounds=False)
        self._refresh_fourier_table_interaction_state()
        self._refresh_fourier_preview_from_controls()

    def _refresh_cluster_views_after_batch_update(
        self,
        target_states: list[_ClusterDensityGroupState],
        *,
        selected_keys: list[str] | None = None,
    ) -> None:
        active_key = self._selected_cluster_group_key or (
            None if not target_states else target_states[0].key
        )
        keys_to_restore = (
            list(selected_keys)
            if selected_keys
            else ([] if active_key is None else [active_key])
        )
        self._populate_cluster_group_table()
        self._restore_cluster_group_selection(
            keys_to_restore,
            preferred_current_key=active_key,
        )
        if active_key is not None:
            self._set_active_cluster_group(active_key)
        self._schedule_cluster_view_cache_prewarm()
        self._refresh_debye_scattering_group()

    def _clear_cluster_group_states(self) -> None:
        self._cluster_view_cache_prewarm_timer.stop()
        self._cluster_view_cache_prewarm_queue = []
        self.structure_viewer.clear_scene_cache()
        self._cluster_group_states = []
        self._selected_cluster_group_key = None
        self._manual_mesh_lock_settings = None
        self.cluster_group_table.setRowCount(0)
        if hasattr(self, "fourier_settings_table"):
            self.fourier_settings_table.setRowCount(0)
        self.cluster_group_table_section.collapse()
        self.cluster_group_status_label.setText(
            "Single-file and folder inputs run directly. Loading a folder of "
            "stoichiometry subfolders will populate a per-cluster summary table here."
        )
        if hasattr(self, "cluster_batch_scope_label"):
            self.cluster_batch_scope_label.setText(
                "Batch solvent, smearing, and Fourier actions become "
                "available after loading cluster folders."
            )
        for button, checked in (
            (getattr(self, "apply_smearing_to_all_button", None), True),
            (getattr(self, "apply_contrast_to_all_button", None), True),
            (getattr(self, "apply_fourier_to_all_button", None), True),
            (getattr(self, "apply_debye_to_all_button", None), False),
        ):
            if button is None:
                continue
            button.blockSignals(True)
            button.setChecked(checked)
            button.blockSignals(False)
        self.show_all_cluster_transforms_checkbox.setChecked(False)
        self.show_all_cluster_transforms_checkbox.setEnabled(False)
        self._debye_scattering_result = None
        self._close_debye_scattering_compare_dialog()
        self._refresh_smearing_scope_status(False)
        self._refresh_contrast_scope_status(False)
        self._refresh_fourier_scope_status(False)
        self._refresh_run_action_state()
        self._refresh_mesh_notice()
        self._refresh_debye_scattering_group()

    def _clear_cluster_group_outputs(
        self,
        *,
        clear_manual_mesh_lock: bool,
    ) -> int:
        cleared_count = 0
        for state in self._cluster_group_states:
            if (
                state.profile_result is not None
                or state.transform_result is not None
                or state.debye_scattering_result is not None
                or state.solvent_density_e_per_a3 is not None
                or state.solvent_cutoff_radius_a is not None
            ):
                cleared_count += 1
            state.profile_result = None
            state.transform_result = None
            state.debye_scattering_result = None
            state.solvent_density_e_per_a3 = None
            state.solvent_cutoff_radius_a = None
        if clear_manual_mesh_lock:
            self._manual_mesh_lock_settings = None
        self._debye_scattering_result = None
        self._close_debye_scattering_compare_dialog()
        if self._cluster_group_states:
            self._populate_cluster_group_table()
        else:
            self._refresh_run_action_state()
        self._refresh_mesh_notice()
        self._update_push_to_model_state()
        self._refresh_debye_scattering_group()
        return cleared_count

    def _clear_cluster_group_outputs_for_keys(
        self,
        group_keys: set[str],
    ) -> int:
        resolved_keys = {
            str(key).strip() for key in group_keys if str(key).strip()
        }
        if not resolved_keys:
            return 0
        cleared_count = 0
        active_key = str(self._selected_cluster_group_key or "").strip()
        for state in self._cluster_group_states:
            if state.key not in resolved_keys:
                continue
            if (
                state.profile_result is not None
                or state.transform_result is not None
                or state.debye_scattering_result is not None
                or state.solvent_density_e_per_a3 is not None
                or state.solvent_cutoff_radius_a is not None
            ):
                cleared_count += 1
            state.profile_result = None
            state.transform_result = None
            state.debye_scattering_result = None
            state.solvent_density_e_per_a3 = None
            state.solvent_cutoff_radius_a = None
        if self._manual_mesh_lock_settings is not None and not any(
            self._cluster_group_is_complete(state)
            for state in self._cluster_group_states
        ):
            self._manual_mesh_lock_settings = None
        if active_key in resolved_keys or not self._cluster_group_states:
            self._debye_scattering_result = None
        self._close_debye_scattering_compare_dialog()
        self._populate_cluster_group_table()
        if active_key in resolved_keys:
            self._set_active_cluster_group(active_key)
        else:
            self._refresh_run_action_state()
            self._refresh_mesh_notice()
            self._update_push_to_model_state()
            self._refresh_debye_scattering_group()
        return cleared_count

    @staticmethod
    def _distribution_prefit_snapshot_count(
        distribution_root_dir: Path,
    ) -> int:
        prefit_dir = distribution_root_dir / "prefit"
        if not prefit_dir.is_dir():
            return 0
        return len(
            [
                path
                for path in prefit_dir.iterdir()
                if path.is_dir() and (path / "prefit_state.json").is_file()
            ]
        )

    @staticmethod
    def _distribution_dream_run_count(distribution_root_dir: Path) -> int:
        runtime_dir = distribution_root_dir / "dream" / "runtime_scripts"
        if not runtime_dir.is_dir():
            return 0
        return len([path for path in runtime_dir.iterdir() if path.is_dir()])

    def _distribution_push_lock_reason(self) -> str | None:
        if self._preview_mode:
            return (
                "Preview mode does not save component traces into a computed "
                "distribution."
            )
        if self._distribution_root_dir is None:
            return "This window is not linked to a computed distribution."
        prefit_count = self._distribution_prefit_snapshot_count(
            self._distribution_root_dir
        )
        if prefit_count > 0:
            return (
                "Push is locked because this computed distribution already has "
                f"{prefit_count} saved prefit snapshot"
                f"{'' if prefit_count == 1 else 's'}."
            )
        dream_count = self._distribution_dream_run_count(
            self._distribution_root_dir
        )
        if dream_count > 0:
            return (
                "Push is locked because this computed distribution already has "
                f"{dream_count} saved DREAM run"
                f"{'' if dream_count == 1 else 's'}."
            )
        return None

    def _project_q_range_mismatch_text(self) -> str | None:
        settings = None
        active_state = self._active_cluster_group_state()
        if (
            active_state is not None
            and active_state.transform_result is not None
        ):
            settings = active_state.transform_result.preview.settings
        elif self._fourier_result is not None:
            settings = self._fourier_result.preview.settings
        else:
            try:
                settings = self._fourier_settings_from_controls()
            except Exception:
                settings = None
        if settings is None:
            return None
        mismatches: list[str] = []
        if (
            self._project_q_min is not None
            and abs(float(settings.q_min) - float(self._project_q_min))
            > 1.0e-9
        ):
            mismatches.append(
                f"q min {settings.q_min:.6g} vs project {self._project_q_min:.6g}"
            )
        if (
            self._project_q_max is not None
            and abs(float(settings.q_max) - float(self._project_q_max))
            > 1.0e-9
        ):
            mismatches.append(
                f"q max {settings.q_max:.6g} vs project {self._project_q_max:.6g}"
            )
        if not mismatches:
            return None
        return (
            "Fourier q-range differs from the inherited project q-range: "
            + "; ".join(mismatches)
            + ". The pushed model will use the transform q-grid as written."
        )

    def _update_push_to_model_state(self) -> None:
        if not hasattr(self, "push_to_model_button"):
            return
        lock_reason = self._distribution_push_lock_reason()
        pending_count = sum(
            1
            for state in self._cluster_group_states
            if state.transform_result is None
        )
        if lock_reason is not None:
            enabled = False
            status_text = lock_reason
        elif not self._cluster_group_states:
            enabled = False
            status_text = (
                "Load a folder of cluster bins, calculate each density, and "
                "evaluate each Born-approximation transform before pushing."
            )
        elif pending_count > 0:
            enabled = False
            status_text = (
                f"{pending_count} cluster transform"
                f"{'' if pending_count == 1 else 's'} still need to be evaluated."
            )
        else:
            enabled = True
            status_text = (
                "All cluster transforms are ready to be written into the "
                "linked computed distribution."
            )
            mismatch_text = self._project_q_range_mismatch_text()
            if mismatch_text:
                status_text += " " + mismatch_text
        self.push_to_model_button.setEnabled(enabled)
        self.push_to_model_status_label.setText(status_text)

    def _is_calculation_running(self) -> bool:
        return bool(self._calculation_ui_running)

    def _cluster_group_is_complete(
        self,
        state: _ClusterDensityGroupState,
    ) -> bool:
        if state.single_atom_only:
            return state.transform_result is not None
        return state.profile_result is not None

    def _completed_cluster_group_count(self) -> int:
        return sum(
            1
            for state in self._cluster_group_states
            if self._cluster_group_is_complete(state)
        )

    def _has_calculated_density_outputs(self) -> bool:
        if any(
            self._cluster_group_is_complete(state)
            for state in self._cluster_group_states
        ):
            return True
        return (
            self._profile_result is not None
            or self._fourier_result is not None
        )

    def _manual_mode_enabled_for_run(self) -> bool:
        return bool(
            self._cluster_group_states
            and self.manual_mode_checkbox.isChecked()
        )

    def _refresh_manual_mode_notice(self) -> None:
        if not hasattr(self, "manual_mode_notice_label"):
            return
        if not self._cluster_group_states:
            self.manual_mode_notice_label.setText(
                "Manual mode becomes available after loading a cluster-folder input."
            )
            return
        if self._manual_mesh_lock_settings is not None:
            self.manual_mode_notice_label.setText(
                "Manual-mode calculations have locked the mesh at "
                f"{self._format_mesh_settings_summary(self._manual_mesh_lock_settings)}. "
                "Reset the calculated densities to edit the mesh again."
            )
            return
        if self.manual_mode_checkbox.isChecked():
            self.manual_mode_notice_label.setText(
                "Manual mode will run only the selected stoichiometry row. "
                "The first successful manual calculation locks the mesh "
                "settings until reset."
            )
            return
        self.manual_mode_notice_label.setText(
            "Automatic mode preserves the current behavior and runs every "
            "stoichiometry row in one pass."
        )

    def _refresh_cluster_completion_tracker(self) -> None:
        if not hasattr(self, "cluster_completion_indicator"):
            return
        total = len(self._cluster_group_states)
        if total <= 0:
            self.cluster_completion_indicator.setText("PENDING")
            self.cluster_completion_indicator.setStyleSheet(
                "background-color: #94a3b8; color: white; padding: 4px 8px; "
                "border-radius: 8px; font-weight: 600;"
            )
            self.cluster_completion_tracker_label.setText(
                "Computed stoichiometries: n/a"
            )
            return
        completed = self._completed_cluster_group_count()
        all_complete = completed >= total
        if all_complete:
            label = "COMPLETE"
            style = (
                "background-color: #15803d; color: white; padding: 4px 8px; "
                "border-radius: 8px; font-weight: 600;"
            )
        elif completed > 0:
            label = "ACTIVE"
            style = (
                "background-color: #b45309; color: white; padding: 4px 8px; "
                "border-radius: 8px; font-weight: 600;"
            )
        else:
            label = "PENDING"
            style = (
                "background-color: #475569; color: white; padding: 4px 8px; "
                "border-radius: 8px; font-weight: 600;"
            )
        self.cluster_completion_indicator.setText(label)
        self.cluster_completion_indicator.setStyleSheet(style)
        self.cluster_completion_tracker_label.setText(
            "Computed stoichiometries: "
            f"{completed}/{total}"
            + (" (all complete)" if all_complete else "")
        )

    def _refresh_mesh_control_lock(self) -> None:
        if not hasattr(self, "rstep_spin"):
            return
        enabled = not self._is_calculation_running() and (
            self._manual_mesh_lock_settings is None
        )
        for widget in (
            self.rstep_spin,
            self.theta_divisions_spin,
            self.phi_divisions_spin,
            self.rmax_spin,
            self.update_mesh_button,
        ):
            widget.setEnabled(enabled)

    def _refresh_run_action_state(self) -> None:
        if not hasattr(self, "run_button"):
            return
        running = self._is_calculation_running()
        pending_thread_teardown = (
            (not running)
            and self._calculation_worker is not None
            and self._calculation_thread is not None
            and not getattr(
                self._calculation_worker, "_cancel_requested", False
            )
        )
        has_cluster_groups = bool(self._cluster_group_states)
        if not has_cluster_groups and self.manual_mode_checkbox.isChecked():
            self.manual_mode_checkbox.blockSignals(True)
            self.manual_mode_checkbox.setChecked(False)
            self.manual_mode_checkbox.blockSignals(False)
        self.manual_mode_checkbox.setEnabled(
            has_cluster_groups and not running
        )
        for widget in (
            getattr(self, "apply_smearing_to_all_button", None),
            getattr(self, "apply_contrast_to_all_button", None),
            getattr(self, "apply_fourier_to_all_button", None),
            getattr(self, "apply_debye_to_all_button", None),
        ):
            if widget is None:
                continue
            widget.setEnabled(has_cluster_groups and not running)
        self.stop_calculation_button.setEnabled(
            running or pending_thread_teardown
        )
        self.reset_calculations_button.setEnabled(
            (not running) and self._has_calculated_density_outputs()
        )
        self._refresh_manual_mode_notice()
        self._refresh_cluster_completion_tracker()
        self._refresh_mesh_control_lock()
        self._refresh_fourier_table_interaction_state()
        self._refresh_debye_scattering_group()

    def _update_cluster_group_status(self) -> None:
        if not self._cluster_group_states:
            self.cluster_group_status_label.setText(
                "Single-file and folder inputs run directly. Loading a folder of "
                "stoichiometry subfolders will populate a per-cluster summary table here."
            )
            if hasattr(self, "cluster_batch_scope_label"):
                self.cluster_batch_scope_label.setText(
                    "Batch solvent, smearing, and Fourier actions become "
                    "available after loading cluster folders."
                )
            self._refresh_run_action_state()
            return
        density_ready = sum(
            1
            for state in self._cluster_group_states
            if not state.single_atom_only and state.profile_result is not None
        )
        debye_only = sum(
            1 for state in self._cluster_group_states if state.single_atom_only
        )
        scattering_ready = sum(
            1
            for state in self._cluster_group_states
            if state.transform_result is not None
        )
        self.cluster_group_status_label.setText(
            f"Cluster-folder mode: {len(self._cluster_group_states)} stoichiometries, "
            f"{density_ready} density profiles ready, "
            f"{debye_only} Debye-only groups, "
            f"{scattering_ready} scattering profiles ready."
        )
        if hasattr(self, "cluster_batch_scope_label"):
            selected_rows = self._selected_cluster_group_rows()
            if selected_rows:
                scope_text = (
                    f"Current selected stoichiometry rows: "
                    f"{len(selected_rows)}."
                )
            else:
                scope_text = "Current selected stoichiometry rows: none."
            self.cluster_batch_scope_label.setText(
                "Select stoichiometry rows to target apply-to-selected "
                "actions. Turn on Apply to All within Smearing, Electron "
                "Density Contrast, or Fourier Transform to update every "
                "stoichiometry instead. " + scope_text
            )
        self._refresh_run_action_state()

    @staticmethod
    def _apply_scope_status_text(apply_to_all: bool) -> str:
        if apply_to_all:
            return "Selection will be applied to all."
        return "Selection will be applied to selected."

    @Slot(bool)
    def _refresh_smearing_scope_status(
        self, _checked: bool | None = None
    ) -> None:
        if hasattr(self, "smearing_scope_status_label"):
            self.smearing_scope_status_label.setText(
                self._apply_scope_status_text(
                    bool(self.apply_smearing_to_all_button.isChecked())
                )
            )

    def _auto_save_smearing_outputs_enabled(self) -> bool:
        if not hasattr(self, "auto_save_smearing_outputs_checkbox"):
            return False
        return bool(self.auto_save_smearing_outputs_checkbox.isChecked())

    @Slot(bool)
    def _on_auto_save_smearing_outputs_toggled(
        self,
        _checked: bool | None = None,
    ) -> None:
        self._sync_workspace_state()

    def _capture_smearing_saved_output_if_enabled(
        self,
        *,
        group_state: _ClusterDensityGroupState | None = None,
        profile_result: ElectronDensityProfileResult | None = None,
    ) -> bool:
        if not self._auto_save_smearing_outputs_enabled():
            return False
        self._capture_saved_output_entry(
            "smearing",
            group_state=group_state,
            profile_result=profile_result,
        )
        return True

    @Slot(bool)
    def _refresh_contrast_scope_status(
        self, _checked: bool | None = None
    ) -> None:
        if hasattr(self, "contrast_scope_status_label"):
            self.contrast_scope_status_label.setText(
                self._apply_scope_status_text(
                    bool(self.apply_contrast_to_all_button.isChecked())
                )
            )

    @Slot(bool)
    def _refresh_fourier_scope_status(
        self, _checked: bool | None = None
    ) -> None:
        if hasattr(self, "fourier_scope_status_label"):
            if not self._cluster_group_states:
                self.fourier_scope_status_label.setText(
                    "Batch Fourier editing becomes available in "
                    "cluster-folder mode."
                )
                return
            self.fourier_scope_status_label.setText(
                self._apply_scope_status_text(
                    bool(self.apply_fourier_to_all_button.isChecked())
                )
            )

    def _refresh_fourier_table_interaction_state(self) -> None:
        if not hasattr(self, "fourier_settings_table"):
            return
        has_cluster_groups = bool(self._cluster_group_states)
        apply_to_all = bool(
            has_cluster_groups and self.apply_fourier_to_all_button.isChecked()
        )
        running = self._is_calculation_running()
        for widget in (
            self.fourier_rmax_spin,
            self.fourier_qmin_spin,
            self.fourier_qmax_spin,
            self.fourier_qstep_spin,
            self.fourier_window_combo,
            self.fourier_resampling_points_spin,
            self.fourier_use_solvent_subtracted_checkbox,
            self.fourier_legacy_mode_checkbox,
        ):
            widget.setEnabled(not apply_to_all and not running)
        self.fourier_rmin_spin.setEnabled(
            (
                not apply_to_all
                and not running
                and self._fourier_domain_mode_from_controls() == "legacy"
            )
        )
        self._refresh_fourier_domain_mode_controls()
        self.fourier_settings_table.setEnabled(
            has_cluster_groups and not running
        )
        self.fourier_settings_table.setEditTriggers(
            (
                QTableWidget.EditTrigger.DoubleClicked
                | QTableWidget.EditTrigger.EditKeyPressed
                | QTableWidget.EditTrigger.SelectedClicked
            )
            if apply_to_all and not running
            else QTableWidget.EditTrigger.NoEditTriggers
        )
        self._populate_fourier_settings_table()

    def _populate_fourier_settings_table(self) -> None:
        if not hasattr(self, "fourier_settings_table"):
            return
        active_key = self._selected_cluster_group_key
        self._updating_fourier_settings_table = True
        self.fourier_settings_table.blockSignals(True)
        try:
            self.fourier_settings_table.setRowCount(
                len(self._cluster_group_states)
            )
            for row_index, state in enumerate(self._cluster_group_states):
                settings = self._cluster_state_fourier_settings(state)
                status_text = (
                    "Ready (Debye)"
                    if state.single_atom_only
                    and state.transform_result is not None
                    else (
                        "Pending (Debye)"
                        if state.single_atom_only
                        else (
                            "Ready"
                            if state.transform_result is not None
                            else "Pending"
                        )
                    )
                )
                values = [
                    state.display_name,
                    status_text,
                    self._fourier_profile_label_for_settings(
                        settings,
                        single_atom_only=state.single_atom_only,
                    ),
                    f"{settings.r_min:.6g}",
                    f"{settings.r_max:.6g}",
                    f"{settings.q_min:.6g}",
                    f"{settings.q_max:.6g}",
                    f"{settings.q_step:.6g}",
                    str(int(settings.resampling_points)),
                    str(settings.window_function),
                ]
                editable_columns = {
                    _FT_COLUMN_PROFILE,
                    _FT_COLUMN_RMAX,
                    _FT_COLUMN_QMIN,
                    _FT_COLUMN_QMAX,
                    _FT_COLUMN_QSTEP,
                    _FT_COLUMN_RESAMPLE,
                    _FT_COLUMN_WINDOW,
                }
                if settings.domain_mode == "legacy":
                    editable_columns.add(_FT_COLUMN_RMIN)
                for column_index, value in enumerate(values):
                    item = QTableWidgetItem(value)
                    flags = (
                        Qt.ItemFlag.ItemIsEnabled
                        | Qt.ItemFlag.ItemIsSelectable
                    )
                    if (
                        self.apply_fourier_to_all_button.isChecked()
                        and column_index in editable_columns
                        and not state.single_atom_only
                    ) or (
                        self.apply_fourier_to_all_button.isChecked()
                        and state.single_atom_only
                        and column_index
                        in {
                            _FT_COLUMN_QMIN,
                            _FT_COLUMN_QMAX,
                            _FT_COLUMN_QSTEP,
                            _FT_COLUMN_RESAMPLE,
                            _FT_COLUMN_WINDOW,
                        }
                    ):
                        flags |= Qt.ItemFlag.ItemIsEditable
                    item.setFlags(flags)
                    self.fourier_settings_table.setItem(
                        row_index,
                        column_index,
                        item,
                    )
            if active_key is not None:
                self._sync_fourier_settings_table_selection(active_key)
        finally:
            self.fourier_settings_table.blockSignals(False)
            self._updating_fourier_settings_table = False

    def _sync_fourier_settings_table_selection(self, group_key: str) -> None:
        if (
            not hasattr(self, "fourier_settings_table")
            or not self._cluster_group_states
        ):
            return
        self._syncing_cluster_selection = True
        self.fourier_settings_table.blockSignals(True)
        try:
            target_row = -1
            for row_index, state in enumerate(self._cluster_group_states):
                if state.key == group_key:
                    target_row = row_index
                    break
            if target_row >= 0:
                self.fourier_settings_table.setCurrentCell(target_row, 0)
                self.fourier_settings_table.selectRow(target_row)
        finally:
            self.fourier_settings_table.blockSignals(False)
            self._syncing_cluster_selection = False

    def _resolve_fourier_window_value(self, raw_value: str) -> str:
        candidate = (
            str(raw_value).strip().lower().replace(" ", "_").replace("-", "_")
        )
        for index in range(self.fourier_window_combo.count()):
            data = str(self.fourier_window_combo.itemData(index) or "")
            text = (
                str(self.fourier_window_combo.itemText(index))
                .strip()
                .lower()
                .replace(" ", "_")
                .replace("-", "_")
            )
            if candidate == data or candidate == text:
                return data
        raise ValueError("Choose a supported Fourier window name.")

    @Slot()
    def _handle_fourier_settings_table_selection_changed(self) -> None:
        if self._syncing_cluster_selection:
            return
        selected_ranges = self.fourier_settings_table.selectedRanges()
        if not selected_ranges:
            return
        row_index = int(selected_ranges[0].topRow())
        if row_index < 0 or row_index >= len(self._cluster_group_states):
            return
        group_key = self._cluster_group_states[row_index].key
        self._syncing_cluster_selection = True
        try:
            self._set_current_cluster_group_row(
                group_key,
                preserve_selection=True,
            )
        finally:
            self._syncing_cluster_selection = False
        self._set_active_cluster_group(group_key)

    @Slot(QTableWidgetItem)
    def _handle_fourier_settings_table_item_changed(
        self,
        item: QTableWidgetItem,
    ) -> None:
        if (
            self._updating_fourier_settings_table
            or not self.apply_fourier_to_all_button.isChecked()
        ):
            return
        row_index = int(item.row())
        column_index = int(item.column())
        if row_index < 0 or row_index >= len(self._cluster_group_states):
            return
        state = self._cluster_group_states[row_index]
        current_settings = self._cluster_state_fourier_settings(state)
        try:
            new_settings = ElectronDensityFourierTransformSettings(
                r_min=(
                    float(item.text())
                    if column_index == _FT_COLUMN_RMIN
                    else current_settings.r_min
                ),
                r_max=(
                    float(item.text())
                    if column_index == _FT_COLUMN_RMAX
                    else current_settings.r_max
                ),
                domain_mode=current_settings.domain_mode,
                window_function=(
                    self._resolve_fourier_window_value(item.text())
                    if column_index == _FT_COLUMN_WINDOW
                    else current_settings.window_function
                ),
                resampling_points=(
                    int(float(item.text()))
                    if column_index == _FT_COLUMN_RESAMPLE
                    else current_settings.resampling_points
                ),
                q_min=(
                    float(item.text())
                    if column_index == _FT_COLUMN_QMIN
                    else current_settings.q_min
                ),
                q_max=(
                    float(item.text())
                    if column_index == _FT_COLUMN_QMAX
                    else current_settings.q_max
                ),
                q_step=(
                    float(item.text())
                    if column_index == _FT_COLUMN_QSTEP
                    else current_settings.q_step
                ),
                use_solvent_subtracted_profile=(
                    ("solvent" in str(item.text()).strip().lower())
                    if column_index == _FT_COLUMN_PROFILE
                    else current_settings.use_solvent_subtracted_profile
                ),
                log_q_axis=current_settings.log_q_axis,
                log_intensity_axis=current_settings.log_intensity_axis,
            ).normalized()
        except Exception:
            self._populate_fourier_settings_table()
            return
        if column_index in {
            _FT_COLUMN_PROFILE,
            _FT_COLUMN_QMIN,
            _FT_COLUMN_QMAX,
            _FT_COLUMN_QSTEP,
            _FT_COLUMN_RESAMPLE,
            _FT_COLUMN_WINDOW,
        }:
            for target_state in self._cluster_group_states:
                self._set_cluster_state_fourier_settings(
                    target_state,
                    self._shared_fourier_settings_for_state(
                        target_state,
                        new_settings,
                    ),
                    prefer_solvent_cutoff=False,
                )
        else:
            self._set_cluster_state_fourier_settings(
                state,
                new_settings,
                prefer_solvent_cutoff=False,
            )
        active_key = self._selected_cluster_group_key
        self._populate_cluster_group_table()
        self._populate_fourier_settings_table()
        if active_key is not None:
            self._set_active_cluster_group(active_key)
        self._sync_workspace_state()

    @Slot(bool)
    def _handle_fourier_apply_to_all_toggled(self, checked: bool) -> None:
        self._refresh_fourier_scope_status(checked)
        if checked and self._cluster_group_states:
            reference_state = (
                self._active_cluster_group_state()
                or self._cluster_group_states[0]
            )
            reference_settings = self._cluster_state_fourier_settings(
                reference_state
            )
            for state in self._cluster_group_states:
                self._set_cluster_state_fourier_settings(
                    state,
                    self._shared_fourier_settings_for_state(
                        state,
                        reference_settings,
                    ),
                    prefer_solvent_cutoff=False,
                )
            active_key = self._selected_cluster_group_key
            self._populate_cluster_group_table()
            self._populate_fourier_settings_table()
            if active_key is not None:
                self._set_active_cluster_group(active_key)
        elif self._selected_cluster_group_key is not None:
            self._set_active_cluster_group(self._selected_cluster_group_key)
        self._refresh_fourier_table_interaction_state()
        self._sync_workspace_state()

    def _use_contiguous_frame_mode(self) -> bool:
        if not hasattr(self, "contiguous_frame_mode_checkbox"):
            return True
        return bool(self.contiguous_frame_mode_checkbox.isChecked())

    def _use_pinned_geometric_tracking(self) -> bool:
        return bool(self.pinned_geometric_tracking_checkbox.isChecked())

    def _current_input_supports_pinned_geometric_tracking(self) -> bool:
        if self._cluster_group_states:
            return bool(self._cluster_group_states) and all(
                state.inspection.input_mode == "folder"
                and state.inspection.total_files > 1
                and set(state.inspection.format_counts).issubset({"pdb"})
                for state in self._cluster_group_states
            )
        return bool(
            self._inspection is not None
            and self._inspection.input_mode == "folder"
            and self._inspection.total_files > 1
            and set(self._inspection.format_counts).issubset({"pdb"})
        )

    @Slot(bool)
    def _handle_contiguous_frame_mode_toggled(self, _checked: bool) -> None:
        self._refresh_contiguous_frame_mode_notice()
        self._refresh_pinned_geometric_tracking_controls()
        self._refresh_mesh_notice()
        self._sync_workspace_state()

    @Slot(bool)
    def _handle_pinned_geometric_tracking_toggled(
        self, _checked: bool
    ) -> None:
        self._refresh_pinned_geometric_tracking_controls()
        self._refresh_contiguous_frame_mode_notice()
        self._refresh_mesh_notice()
        self._sync_workspace_state()

    def _refresh_pinned_geometric_tracking_controls(self) -> None:
        center_mode = (
            "center_of_mass"
            if self._structure is None
            else str(self._structure.center_mode)
        )
        supports_tracking = (
            self._current_input_supports_pinned_geometric_tracking()
        )
        enabled = bool(
            center_mode == "center_of_mass"
            and self._use_contiguous_frame_mode()
            and supports_tracking
        )
        if not enabled and self.pinned_geometric_tracking_checkbox.isChecked():
            self.pinned_geometric_tracking_checkbox.blockSignals(True)
            try:
                self.pinned_geometric_tracking_checkbox.setChecked(False)
            finally:
                self.pinned_geometric_tracking_checkbox.blockSignals(False)
        self.pinned_geometric_tracking_checkbox.setEnabled(enabled)
        if center_mode != "center_of_mass":
            notice = (
                "Pinned geometric tracking is available only while "
                "Geometric Mass Center mode is selected."
            )
        elif not self._use_contiguous_frame_mode():
            notice = (
                "Pinned geometric tracking also requires contiguous-frame "
                "evaluation to be enabled."
            )
        elif not supports_tracking:
            notice = (
                "Pinned geometric tracking applies to folder or cluster-folder "
                "runs with contiguous PDB frame_<NNNN> files."
            )
        elif self._use_pinned_geometric_tracking():
            notice = (
                "Pinned geometric tracking is on. Each contiguous PDB frame set "
                "will reuse the first frame's geometric mass center coordinates."
            )
        else:
            notice = (
                "Optional: pin each contiguous PDB frame set to the first "
                "frame's geometric mass center coordinates to capture "
                "displacements without recentering every frame."
            )
        self.pinned_geometric_tracking_notice.setText(notice)

    @staticmethod
    def _center_mode_label_for_structure(
        structure: ElectronDensityStructure | None,
    ) -> str:
        if structure is None:
            return "Unavailable"
        if structure.center_mode == "center_of_mass":
            return "Geometric Mass Center"
        if structure.center_mode == "nearest_atom":
            return "Nearest atom to geometric mass center"
        return (
            f"{structure.reference_element} reference-element "
            "geometric center"
        )

    @staticmethod
    def _center_mode_table_text_for_structure(
        structure: ElectronDensityStructure | None,
    ) -> str:
        if structure is None:
            return "Unavailable"
        if structure.center_mode == "center_of_mass":
            return "Geom Mass"
        if structure.center_mode == "nearest_atom":
            return "Nearest Atom"
        return "Ref Element"

    @staticmethod
    def _center_reference_table_text_for_structure(
        structure: ElectronDensityStructure | None,
    ) -> str:
        if structure is None:
            return "Unavailable"
        if structure.center_mode == "center_of_mass":
            return "Geom Mass"
        if structure.center_mode == "nearest_atom":
            return (
                f"{structure.nearest_atom_element} atom "
                f"#{structure.nearest_atom_index + 1}"
            )
        return f"{structure.reference_element} geom center"

    def _center_reference_tooltip_for_structure(
        self,
        structure: ElectronDensityStructure | None,
    ) -> str:
        if structure is None:
            return "Unavailable"
        if structure.center_mode == "center_of_mass":
            return (
                "Geometric mass center at "
                f"{self._format_point(structure.active_center)}. "
                "Nearest atom to that point: "
                f"{structure.nearest_atom_element} atom "
                f"#{structure.nearest_atom_index + 1}, "
                f"{structure.nearest_atom_distance:.3f} Å away."
            )
        if structure.center_mode == "nearest_atom":
            return (
                "Nearest atom to the geometric mass center: "
                f"{structure.nearest_atom_element} atom "
                f"#{structure.nearest_atom_index + 1}, "
                f"{structure.nearest_atom_distance:.3f} Å away. "
                f"Active center = {self._format_point(structure.active_center)}."
            )
        return (
            "Reference-element center uses "
            f"{structure.reference_element}; "
            "reference-element geometric center = "
            f"{self._format_point(structure.reference_element_geometric_center)}. "
            "Offset from the total-atom geometric center = "
            f"{structure.reference_element_offset_from_geometric_center:.3f} Å."
        )

    @staticmethod
    def _format_mesh_settings_summary(
        settings: ElectronDensityMeshSettings,
    ) -> str:
        normalized = settings.normalized()
        summary = (
            f"rstep={normalized.rstep:.3f} Å, "
            f"theta={normalized.theta_divisions}, "
            f"phi={normalized.phi_divisions}, "
            f"rmax={normalized.rmax:.3f} Å"
        )
        if normalized.pin_contiguous_geometric_tracking:
            summary += ", pinned tracking=on"
        return summary

    def _set_mesh_settings_confirmed_for_run(
        self,
        confirmed: bool,
    ) -> None:
        self._mesh_settings_confirmed_for_run = bool(confirmed)
        if hasattr(self, "pending_mesh_value"):
            self._refresh_mesh_notice()

    def _mesh_confirmation_carries_forward_on_auto_apply(self) -> bool:
        return bool(
            self._mesh_settings_confirmed_for_run
            and self._mesh_settings_from_controls()
            == self._active_mesh_settings
        )

    def _refresh_contiguous_frame_mode_notice(self) -> None:
        base_notice = (
            "Default: on. Folder and cluster-folder runs will try contiguous "
            "frame grouping from frame_<NNNN> file names and preserve the "
            "active center mode from the mesh settings as a shared offset "
            "relative to the heaviest-element geometric center."
        )
        tooltip = (
            "When averaging multiple structures, group files by frame_<NNNN> "
            "sequences and lock each contiguous set to a shared active-center "
            "offset relative to the heaviest-element geometric center."
        )
        if self._structure is None:
            center_notice = (
                " Load a structure to show which active center contiguous-frame "
                "evaluation will use."
            )
        else:
            center_label = self._center_mode_label_for_structure(
                self._structure
            )
            center_notice = (
                " Current active center mode: "
                f"{center_label}; active center="
                f"{self._format_point(self._structure.active_center)}."
            )
            tooltip += (
                " Current active center mode: "
                f"{center_label}; active center="
                f"{self._format_point(self._structure.active_center)}."
            )
        self.contiguous_frame_mode_notice.setText(
            base_notice
            + center_notice
            + (
                " Pinned geometric tracking is ready for contiguous PDB sets."
                if self._use_pinned_geometric_tracking()
                else ""
            )
            + " Files without frame_<NNNN> identifiers fall back to complete "
            "averaging with a status notice."
        )
        self.contiguous_frame_mode_checkbox.setToolTip(tooltip)

    @staticmethod
    def _format_contiguous_frame_set(
        frame_set: ElectronDensityContiguousFrameSetSummary,
    ) -> str:
        series_text = (
            f"{frame_set.series_label}: "
            if str(frame_set.series_label).strip()
            and str(frame_set.series_label).strip().lower() != "default"
            else ""
        )
        if frame_set.frame_count <= 1:
            return (
                f"{series_text}frame {frame_set.frame_range_label} "
                f"({frame_set.frame_count} file)"
            )
        return (
            f"{series_text}frames {frame_set.frame_range_label} "
            f"({frame_set.frame_count} files)"
        )

    def _append_averaging_status(
        self,
        result: ElectronDensityProfileResult,
        *,
        prefix: str = "",
    ) -> None:
        label_prefix = str(prefix).strip()
        if label_prefix:
            label_prefix = label_prefix.rstrip() + " "
        if result.contiguous_frame_mode_applied:
            if result.pinned_geometric_tracking_applied:
                self._append_status(
                    f"{label_prefix}Contiguous-frame evaluation pinned "
                    f"{len(result.contiguous_frame_sets)} frame set"
                    f"{'' if len(result.contiguous_frame_sets) == 1 else 's'} "
                    "to the first frame's geometric mass center coordinates."
                )
            else:
                center_label = self._center_mode_label_for_structure(
                    result.structure
                )
                self._append_status(
                    f"{label_prefix}Contiguous-frame evaluation locked "
                    f"{len(result.contiguous_frame_sets)} frame set"
                    f"{'' if len(result.contiguous_frame_sets) == 1 else 's'} "
                    "to shared centers relative to the heaviest-element anchor "
                    "using the active center mode from the mesh settings: "
                    f"{center_label}."
                )
            for note in result.averaging_notes:
                self._append_status(f"{label_prefix}{note}")
            for frame_set in result.contiguous_frame_sets:
                self._append_status(
                    f"{label_prefix}{self._format_contiguous_frame_set(frame_set)}"
                )
        elif result.contiguous_frame_mode_requested:
            for note in result.averaging_notes:
                self._append_status(f"{label_prefix}{note}")

    @staticmethod
    def _profile_overlay_for_result(
        result: ElectronDensityProfileResult | None,
    ) -> ElectronDensityProfileOverlay | None:
        if result is None or result.solvent_contrast is None:
            return None
        contrast = result.solvent_contrast
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

    @staticmethod
    def _format_saved_output_timestamp(timestamp: str) -> str:
        try:
            return datetime.fromisoformat(str(timestamp)).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        except Exception:
            return str(timestamp).strip() or "Unknown time"

    @staticmethod
    def _saved_output_entry_kind_label(entry_kind: str) -> str:
        normalized = str(entry_kind).strip().lower()
        if normalized == "fourier_transform":
            return "Fourier Transform"
        if normalized == "smearing":
            return "Smearing"
        if normalized == "solvent_subtraction":
            return "Solvent Subtraction"
        return "Electron Density"

    @staticmethod
    def _saved_output_context_label(entry: _SavedOutputEntry) -> str:
        return (
            str(entry.group_label).strip()
            or entry.profile_result.structure.file_path.name
        )

    @staticmethod
    def _saved_output_entry_heading(
        entry: _SavedOutputEntry,
        *,
        index: int | None = None,
    ) -> str:
        prefix = "" if index is None else f"{index}. "
        return (
            prefix
            + ElectronDensityMappingMainWindow._saved_output_context_label(
                entry
            )
            + " · "
            + ElectronDensityMappingMainWindow._saved_output_entry_kind_label(
                entry.entry_kind
            )
        )

    @staticmethod
    def _saved_output_entry_summary_text(entry: _SavedOutputEntry) -> str:
        result = entry.profile_result
        mesh_settings = result.mesh_geometry.settings
        solvent_text = (
            result.solvent_contrast.legend_label
            if result.solvent_contrast is not None
            else "No solvent subtraction"
        )
        fourier_text = (
            "Evaluated"
            if entry.transform_result is not None
            else "Preview only"
        )
        return (
            f"Saved {ElectronDensityMappingMainWindow._format_saved_output_timestamp(entry.created_at)} "
            f"in {'Preview' if entry.preview_mode else 'Computed Distribution'} mode. "
            f"Averaging: {str(result.averaging_mode).replace('_', ' ')}. "
            f"Mesh: rstep={mesh_settings.rstep:.3f} Å, rmax={mesh_settings.rmax:.3f} Å. "
            f"Smearing factor={result.smearing_settings.debye_waller_factor:.6g} Å². "
            f"Solvent: {solvent_text}. "
            f"Fourier: {fourier_text} with window={entry.fourier_settings.window_function}, "
            f"r={entry.fourier_settings.r_min:.3f}–{entry.fourier_settings.r_max:.3f} Å."
        )

    @staticmethod
    def _safe_saved_output_slug(value: str) -> str:
        normalized = "".join(
            character if character.isalnum() or character in "._-" else "_"
            for character in str(value).strip()
        )
        return normalized.strip("._") or "saved_output"

    @staticmethod
    def _saved_output_entry_file_stem(
        entry: _SavedOutputEntry,
        *,
        index: int,
    ) -> str:
        context = ElectronDensityMappingMainWindow._saved_output_context_label(
            entry
        )
        return ElectronDensityMappingMainWindow._safe_saved_output_slug(
            f"{index:02d}_{entry.entry_kind}_{context}_{entry.entry_id}"
        )

    def _update_output_history_summary(self) -> None:
        if not hasattr(self, "output_history_summary_label"):
            return
        count = len(self._saved_output_entries)
        history_path = self._saved_output_history_write_path()
        if count == 0:
            persistence_text = (
                f" Persisted history will be written to {history_path}."
                if history_path is not None
                else " A writable output directory or computed distribution link is needed for persistence."
            )
            self.output_history_summary_label.setText(
                "Density calculations, solvent-subtracted outputs, Fourier "
                "evaluations, and optional smearing snapshots will be captured "
                "here for reload and comparison." + persistence_text
            )
            return
        self.output_history_summary_label.setText(
            f"{count} saved output set"
            f"{'' if count == 1 else 's'} ready. "
            "Use Command/Ctrl-click to compare multiple entries in a separate window."
        )

    def _output_history_selected_rows(self) -> list[int]:
        if not hasattr(self, "output_history_table"):
            return []
        selection_model = self.output_history_table.selectionModel()
        if selection_model is None:
            return []
        return sorted(
            {index.row() for index in selection_model.selectedRows()}
        )

    def _selected_output_history_entries(self) -> list[_SavedOutputEntry]:
        selected_entries: list[_SavedOutputEntry] = []
        for row_index in self._output_history_selected_rows():
            item = self.output_history_table.item(row_index, 0)
            entry_id = (
                None if item is None else item.data(Qt.ItemDataRole.UserRole)
            )
            for entry in self._saved_output_entries:
                if entry.entry_id == entry_id:
                    selected_entries.append(entry)
                    break
        return selected_entries

    def _update_output_history_actions(self) -> None:
        if not hasattr(self, "load_output_history_button"):
            return
        selected_count = len(self._output_history_selected_rows())
        self.load_output_history_button.setEnabled(selected_count == 1)
        self.compare_output_history_button.setEnabled(selected_count >= 1)

    def _populate_output_history_table(self) -> None:
        if not hasattr(self, "output_history_table"):
            return
        entries = list(reversed(self._saved_output_entries))
        self.output_history_table.blockSignals(True)
        self.output_history_table.setRowCount(len(entries))
        for row_index, entry in enumerate(entries):
            result = entry.profile_result
            mesh_settings = result.mesh_geometry.settings
            solvent_text = (
                result.solvent_contrast.legend_label
                if result.solvent_contrast is not None
                else "None"
            )
            averaging_text = str(result.averaging_mode).replace("_", " ")
            if result.contiguous_frame_mode_applied:
                averaging_text += (
                    " · pinned geometric"
                    if result.pinned_geometric_tracking_applied
                    else " · contiguous"
                )
            elif result.contiguous_frame_mode_requested:
                averaging_text += " · fallback"
            values = [
                self._format_saved_output_timestamp(entry.created_at),
                self._saved_output_entry_kind_label(entry.entry_kind),
                self._saved_output_context_label(entry),
                averaging_text,
                (
                    f"rstep={mesh_settings.rstep:.3f} Å, "
                    f"rmax={mesh_settings.rmax:.3f} Å"
                ),
                f"{result.smearing_settings.debye_waller_factor:.6g} Å²",
                solvent_text,
                (
                    f"{entry.fourier_settings.window_function}, "
                    f"r={entry.fourier_settings.r_min:.3f}–"
                    f"{entry.fourier_settings.r_max:.3f} Å"
                    + (
                        ""
                        if entry.transform_result is not None
                        else " (preview)"
                    )
                ),
            ]
            for column_index, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setFlags(
                    Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
                )
                if column_index == 0:
                    item.setData(Qt.ItemDataRole.UserRole, entry.entry_id)
                    item.setToolTip(
                        self._saved_output_entry_summary_text(entry)
                    )
                self.output_history_table.setItem(
                    row_index, column_index, item
                )
        self.output_history_table.blockSignals(False)
        self._update_output_history_summary()
        self._update_output_history_actions()

    def _clear_saved_output_history(self, *, announce: bool) -> None:
        self._saved_output_entries = []
        if hasattr(self, "output_history_table"):
            self.output_history_table.setRowCount(0)
        self._update_output_history_summary()
        self._update_output_history_actions()
        if announce:
            self._append_status("Cleared the saved output history.")

    @staticmethod
    def _cluster_group_cutoff_text(
        state: _ClusterDensityGroupState,
    ) -> str:
        if state.single_atom_only:
            return "n/a"
        if state.profile_result is None:
            return "Pending"
        if state.solvent_density_e_per_a3 is None:
            return "Pending"
        if state.solvent_cutoff_radius_a is None:
            return "Not found"
        return f"{state.solvent_cutoff_radius_a:.6g}"

    def _cluster_group_fourier_settings_text(
        self,
        state: _ClusterDensityGroupState,
    ) -> str:
        transform = state.transform_result
        settings = (
            transform.preview.settings
            if transform is not None
            else self._cluster_state_fourier_settings(state)
        )
        if state.single_atom_only:
            return (
                f"Debye - q={settings.q_min:.3f}-{settings.q_max:.3f} "
                f"1/A, dq={settings.q_step:.3f}"
            )
        preview_source = (
            str(transform.preview.source_profile_label).lower()
            if transform is not None
            else self._fourier_profile_label_for_settings(
                settings,
                single_atom_only=False,
            )
        )
        profile_source = (
            "solvent" if "solvent-subtracted" in preview_source else "smeared"
        )
        return (
            f"{profile_source} - {settings.window_function} - "
            f"r={settings.r_min:.3f}-{settings.r_max:.3f} A - "
            f"q={settings.q_min:.3f}-{settings.q_max:.3f} 1/A"
        )

    def _populate_cluster_group_table(self) -> None:
        active_key = str(self._selected_cluster_group_key or "").strip()
        self.cluster_group_table.blockSignals(True)
        self.cluster_group_table.setRowCount(0)
        self.cluster_group_table.setRowCount(len(self._cluster_group_states))
        for row_index, state in enumerate(self._cluster_group_states):
            structure = state.reference_structure
            if state.single_atom_only:
                profile_status = "Skipped (Debye)"
                transform_status = (
                    "Ready (Debye)"
                    if state.transform_result is not None
                    else "Auto on Run"
                )
            else:
                profile_status = (
                    "Ready" if state.profile_result is not None else "Pending"
                )
                transform_status = (
                    "Ready"
                    if state.transform_result is not None
                    else "Pending"
                )
            center_element_combo = QComboBox(self.cluster_group_table)
            for element in sorted(structure.element_counts):
                count = int(structure.element_counts[element])
                center_element_combo.addItem(f"{element} ({count})", element)
            center_element_combo.setCurrentIndex(
                max(
                    center_element_combo.findData(structure.reference_element),
                    0,
                )
            )
            center_element_combo.setToolTip(
                "Stored reference-element center for this stoichiometry. "
                "This selection is used whenever reference-element centering "
                "is active. Current center mode: "
                f"{self._center_mode_label_for_structure(structure)}. "
                f"{self._center_reference_tooltip_for_structure(structure)}"
            )
            center_element_combo.currentIndexChanged.connect(
                lambda _index, group_key=state.key, combo=center_element_combo: (
                    self._handle_cluster_group_reference_element_changed(
                        group_key,
                        combo,
                    )
                )
            )
            self.cluster_group_table.setCellWidget(
                row_index,
                _CLUSTER_GROUP_COLUMN_CENTER_ELEMENT,
                center_element_combo,
            )
            values = [
                (_CLUSTER_GROUP_COLUMN_STOICHIOMETRY, state.display_name),
                (
                    _CLUSTER_GROUP_COLUMN_FILES,
                    str(state.inspection.total_files),
                ),
                (
                    _CLUSTER_GROUP_COLUMN_AVG_ATOMS,
                    f"{state.average_atom_count:.1f}",
                ),
                (
                    _CLUSTER_GROUP_COLUMN_CENTER_MODE,
                    self._center_mode_table_text_for_structure(structure),
                ),
                (
                    _CLUSTER_GROUP_COLUMN_CENTER_REFERENCE,
                    self._center_reference_table_text_for_structure(structure),
                ),
                (_CLUSTER_GROUP_COLUMN_DENSITY, profile_status),
                (_CLUSTER_GROUP_COLUMN_FOURIER, transform_status),
                (_CLUSTER_GROUP_COLUMN_TRACE_COLOR, state.trace_color),
                (
                    _CLUSTER_GROUP_COLUMN_SMEARING,
                    (
                        f"{state.profile_result.smearing_settings.debye_waller_factor:.6g}"
                        if state.profile_result is not None
                        else f"{self.smearing_factor_spin.value():.6g}"
                    ),
                ),
                (
                    _CLUSTER_GROUP_COLUMN_SOLVENT,
                    (
                        f"{state.solvent_density_e_per_a3:.6g}"
                        if state.solvent_density_e_per_a3 is not None
                        else "Pending"
                    ),
                ),
                (
                    _CLUSTER_GROUP_COLUMN_CUTOFF,
                    self._cluster_group_cutoff_text(state),
                ),
                (
                    _CLUSTER_GROUP_COLUMN_FT_SETTINGS,
                    self._cluster_group_fourier_settings_text(state),
                ),
                (
                    _CLUSTER_GROUP_COLUMN_REFERENCE_FILE,
                    state.inspection.reference_file.name,
                ),
            ]
            for column_index, value in values:
                item = QTableWidgetItem(value)
                item.setFlags(
                    Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
                )
                if column_index == _CLUSTER_GROUP_COLUMN_TRACE_COLOR:
                    item.setToolTip(
                        f"Transform trace color: {state.trace_color}"
                    )
                if column_index == _CLUSTER_GROUP_COLUMN_FT_SETTINGS:
                    item.setToolTip(str(value))
                if column_index == _CLUSTER_GROUP_COLUMN_CENTER_MODE:
                    item.setToolTip(
                        self._center_mode_label_for_structure(structure)
                    )
                if column_index == _CLUSTER_GROUP_COLUMN_CENTER_REFERENCE:
                    item.setToolTip(
                        self._center_reference_tooltip_for_structure(structure)
                    )
                self.cluster_group_table.setItem(row_index, column_index, item)
        if active_key:
            for row_index, state in enumerate(self._cluster_group_states):
                if state.key == active_key:
                    self.cluster_group_table.selectRow(row_index)
                    break
        self.cluster_group_table.blockSignals(False)
        self.show_all_cluster_transforms_checkbox.setEnabled(
            len(self._cluster_group_states) > 1
        )
        self._populate_fourier_settings_table()
        self._update_cluster_group_status()
        self._update_push_to_model_state()
        self._refresh_debye_scattering_group()

    def _set_active_cluster_group(
        self,
        group_key: str,
        *,
        preserve_zoom: bool = True,
    ) -> None:
        target_state = self._cluster_group_state_by_key(group_key)
        if target_state is None:
            return
        preserved_zoom_percentage = (
            self.structure_viewer._current_zoom_percentage()
            if preserve_zoom
            else None
        )
        preserve_mesh_confirmation = (
            self._mesh_confirmation_carries_forward_on_auto_apply()
        )
        self._selected_cluster_group_key = group_key
        self._inspection = target_state.inspection
        self._structure = target_state.reference_structure
        self._debye_scattering_result = target_state.debye_scattering_result
        self.input_mode_value.setText(
            "Cluster folders "
            f"({len(self._cluster_group_states)} stoichiometries); "
            f"active row = {target_state.display_name}"
        )
        self.reference_file_value.setText(
            str(target_state.inspection.reference_file)
        )
        self._sync_controls_to_structure(sync_mesh_rmax=False)
        active_fourier_settings = (
            target_state.transform_result.preview.settings
            if target_state.transform_result is not None
            else self._cluster_state_fourier_settings(target_state)
        )
        if target_state.single_atom_only:
            self._profile_result = None
            self._fourier_preview = None
            self._fourier_result = target_state.transform_result
            self.profile_plot.draw_placeholder()
            self.smeared_profile_plot.draw_placeholder()
            self.residual_profile_plot.draw_placeholder()
            self.fourier_preview_plot.draw_placeholder()
            self._apply_mesh_settings(
                self._active_mesh_settings,
                announce=False,
                update_viewer=False,
            )
            self._set_mesh_settings_confirmed_for_run(
                preserve_mesh_confirmation
            )
            self._sync_fourier_controls_to_settings(active_fourier_settings)
            self._refresh_fourier_info_labels(None)
            self._refresh_scattering_plot()
            self._refresh_contrast_display()
        elif target_state.profile_result is None:
            self._reset_density_results()
            self._refresh_scattering_plot()
            self._apply_mesh_settings(
                self._active_mesh_settings,
                announce=False,
                update_viewer=False,
            )
            self._set_mesh_settings_confirmed_for_run(
                preserve_mesh_confirmation
            )
            self._sync_fourier_controls_to_settings(active_fourier_settings)
            self._refresh_fourier_info_labels(None)
        else:
            self._profile_result = target_state.profile_result
            self._apply_mesh_settings(
                self._active_mesh_settings,
                announce=False,
                update_viewer=False,
            )
            self._set_mesh_settings_confirmed_for_run(
                preserve_mesh_confirmation
            )
            self._refresh_profile_plots()
            self._sync_fourier_controls_to_domain(reset_bounds=False)
            self._sync_fourier_controls_to_settings(active_fourier_settings)
            if target_state.transform_result is None:
                self._refresh_fourier_preview_from_controls(
                    clear_transform=True
                )
            else:
                self._display_fourier_preview(
                    target_state.transform_result.preview
                )
                self._fourier_result = target_state.transform_result
            self._refresh_scattering_plot()
        self.structure_viewer.set_structure(
            self._structure,
            mesh_geometry=self._active_mesh_geometry,
            reset_view=True,
            preserve_zoom_percentage=preserved_zoom_percentage,
            scene_key=target_state.key,
            render_complexity=self._cluster_group_render_complexity(
                target_state
            ),
        )
        self._sync_fourier_settings_table_selection(group_key)
        self._refresh_debye_scattering_group()
        self._update_push_to_model_state()
        self.statusBar().showMessage(
            f"Selected cluster group {target_state.display_name}"
        )
        self._schedule_cluster_view_cache_prewarm()

    @Slot()
    def _handle_cluster_group_selection_changed(self) -> None:
        current_row = int(self.cluster_group_table.currentRow())
        if 0 <= current_row < len(self._cluster_group_states):
            self._set_active_cluster_group(
                self._cluster_group_states[current_row].key
            )
            self._update_cluster_group_status()
            return
        selected_rows = self._selected_cluster_group_rows()
        if not selected_rows:
            return
        self._set_active_cluster_group(
            self._cluster_group_states[selected_rows[0]].key
        )
        self._update_cluster_group_status()

    def _cluster_group_states_for_path(
        self,
        path: Path,
    ) -> list[_ClusterDensityGroupState]:
        return _build_cluster_group_states_for_path(path)

    def _build_smearing_group(self) -> QWidget:
        group = QGroupBox("Smearing")
        layout = QFormLayout(group)

        intro = QLabel(
            "Apply a Gaussian kernel to the raw radial density profile to "
            "soften boxy shell-to-shell transitions. A value of 0 disables "
            "smearing. Edit the factor, then click Apply Smearing for the "
            "selected stoichiometries or turn on Apply to All to reuse the "
            "same factor everywhere."
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
        self.smearing_summary_value.setStyleSheet("color: #475569;")
        layout.addRow(self.smearing_summary_value)

        action_row = QWidget(group)
        action_layout = QHBoxLayout(action_row)
        action_layout.setContentsMargins(0, 0, 0, 0)
        action_layout.setSpacing(6)
        self.apply_smearing_button = QPushButton("Apply Smearing")
        self.apply_smearing_button.clicked.connect(self._apply_smearing_action)
        action_layout.addWidget(self.apply_smearing_button)
        self.apply_smearing_to_all_button = QCheckBox("Apply to All")
        self.apply_smearing_to_all_button.setChecked(True)
        self.apply_smearing_to_all_button.toggled.connect(
            self._refresh_smearing_scope_status
        )
        self.update_all_smearing_button = self.apply_smearing_to_all_button
        action_layout.addWidget(self.apply_smearing_to_all_button)
        action_layout.addSpacing(20)
        self.smearing_scope_status_label = QLabel()
        self.smearing_scope_status_label.setWordWrap(True)
        self.smearing_scope_status_label.setStyleSheet("color: #475569;")
        action_layout.addWidget(self.smearing_scope_status_label, stretch=1)
        layout.addRow(action_row)
        self.auto_save_smearing_outputs_checkbox = QCheckBox(
            "Auto-save smearing snapshots to Saved Outputs"
        )
        self.auto_save_smearing_outputs_checkbox.setChecked(False)
        self.auto_save_smearing_outputs_checkbox.setToolTip(
            "When enabled, each smearing re-evaluation is saved as a reloadable "
            "Saved Outputs entry."
        )
        self.auto_save_smearing_outputs_checkbox.toggled.connect(
            self._on_auto_save_smearing_outputs_toggled
        )
        layout.addRow(self.auto_save_smearing_outputs_checkbox)
        self._refresh_smearing_scope_status(False)
        return group

    def _build_contrast_group(self) -> QWidget:
        group = QGroupBox("Electron Density Contrast")
        layout = QFormLayout(group)

        intro = QLabel(
            "Estimate a flat solvent electron density using the contrast Debye "
            "workflow solvent options, then compare that value against the "
            "smeared radial density profile. Apply to All reuses the same "
            "solvent settings across every stoichiometry while keeping each "
            "row's own solvent-derived cutoff."
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
        self.direct_density_spin.setDecimals(9)
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

        action_row = QWidget(group)
        action_layout = QHBoxLayout(action_row)
        action_layout.setContentsMargins(0, 0, 0, 0)
        action_layout.setSpacing(6)
        self.compute_solvent_density_button = QPushButton(
            "Apply Electron Density Contrast"
        )
        self.compute_solvent_density_button.clicked.connect(
            self._apply_electron_density_contrast_action
        )
        action_layout.addWidget(self.compute_solvent_density_button)
        self.apply_contrast_to_all_button = QCheckBox("Apply to All")
        self.apply_contrast_to_all_button.setChecked(True)
        self.apply_contrast_to_all_button.toggled.connect(
            self._refresh_contrast_scope_status
        )
        self.compute_all_solvent_density_button = (
            self.apply_contrast_to_all_button
        )
        action_layout.addWidget(self.apply_contrast_to_all_button)
        action_layout.addSpacing(20)
        self.contrast_scope_status_label = QLabel()
        self.contrast_scope_status_label.setWordWrap(True)
        self.contrast_scope_status_label.setStyleSheet("color: #475569;")
        action_layout.addWidget(self.contrast_scope_status_label, stretch=1)
        layout.addRow(action_row)
        self._refresh_contrast_scope_status(False)

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
            "the mirrored real-space source used by the transform. Mirrored "
            "mode is the default: it reflects the profile about r = 0 and "
            "evaluates the windowed transform over -rmax to rmax. Toggle "
            "legacy mode to restore the historical rmin to rmax behavior. In "
            "Apply to All mode, the table becomes the editable per-stoichiometry "
            "Fourier settings view: q settings stay shared, while each row "
            "keeps its own r range."
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
        self.fourier_legacy_mode_checkbox = QCheckBox(
            "Legacy r min to r max transform"
        )
        self.fourier_legacy_mode_checkbox.setChecked(
            self._active_fourier_settings.domain_mode == "legacy"
        )
        self.fourier_legacy_mode_checkbox.toggled.connect(
            self._handle_fourier_domain_mode_toggled
        )
        r_min_form.addRow(self.fourier_legacy_mode_checkbox)
        self.fourier_rmin_label = QLabel("r min")
        self.fourier_rmin_spin = QDoubleSpinBox()
        self.fourier_rmin_spin.setRange(-100000.0, 100000.0)
        self.fourier_rmin_spin.setDecimals(4)
        self.fourier_rmin_spin.setSingleStep(0.05)
        self.fourier_rmin_spin.setKeyboardTracking(False)
        self.fourier_rmin_spin.setValue(self._active_fourier_settings.r_min)
        self.fourier_rmin_spin.valueChanged.connect(
            self._refresh_fourier_preview_from_controls
        )
        r_min_form.addRow(self.fourier_rmin_label, self.fourier_rmin_spin)
        # r max
        self.fourier_rmax_spin = QDoubleSpinBox()
        self.fourier_rmax_spin.setRange(0.01, 100000.0)
        self.fourier_rmax_spin.setDecimals(4)
        self.fourier_rmax_spin.setSingleStep(0.1)
        self.fourier_rmax_spin.setKeyboardTracking(False)
        self.fourier_rmax_spin.setValue(self._active_fourier_settings.r_max)
        self.fourier_rmax_spin.valueChanged.connect(
            self._sync_mirrored_fourier_rmin_to_rmax
        )
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

        action_row = QWidget(group)
        action_layout = QHBoxLayout(action_row)
        action_layout.setContentsMargins(0, 0, 0, 0)
        action_layout.setSpacing(6)
        self.evaluate_fourier_button = QPushButton(
            "Evaluate Fourier Transform"
        )
        self.evaluate_fourier_button.clicked.connect(
            self._evaluate_fourier_transform_action
        )
        action_layout.addWidget(self.evaluate_fourier_button)
        self.apply_fourier_to_all_button = QCheckBox("Apply to All")
        self.apply_fourier_to_all_button.setChecked(True)
        self.apply_fourier_to_all_button.toggled.connect(
            self._handle_fourier_apply_to_all_toggled
        )
        self.evaluate_all_fourier_button = self.apply_fourier_to_all_button
        action_layout.addWidget(self.apply_fourier_to_all_button)
        action_layout.addSpacing(20)
        self.fourier_scope_status_label = QLabel()
        self.fourier_scope_status_label.setWordWrap(True)
        self.fourier_scope_status_label.setStyleSheet("color: #475569;")
        action_layout.addWidget(self.fourier_scope_status_label, stretch=1)
        outer.addWidget(action_row)

        self.fourier_settings_table = QTableWidget(0, 10)
        self.fourier_settings_table.setHorizontalHeaderLabels(
            [
                "Stoichiometry",
                "Status",
                "Profile",
                "r min",
                "r max",
                "q min",
                "q max",
                "q step",
                "Resample pts",
                "Window",
            ]
        )
        self.fourier_settings_table.verticalHeader().setVisible(False)
        self.fourier_settings_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self.fourier_settings_table.setSelectionMode(
            QTableWidget.SelectionMode.SingleSelection
        )
        self.fourier_settings_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers
        )
        self.fourier_settings_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        self.fourier_settings_table.horizontalHeader().setStretchLastSection(
            True
        )
        self.fourier_settings_table.setMinimumHeight(180)
        self.fourier_settings_table.itemSelectionChanged.connect(
            self._handle_fourier_settings_table_selection_changed
        )
        self.fourier_settings_table.itemChanged.connect(
            self._handle_fourier_settings_table_item_changed
        )
        outer.addWidget(self.fourier_settings_table)
        self._refresh_fourier_domain_mode_controls()
        self._refresh_fourier_scope_status(False)
        self._refresh_fourier_table_interaction_state()

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

    def _build_output_history_group(self) -> QWidget:
        group = QGroupBox("Saved Output Sets")
        layout = QVBoxLayout(group)
        self.output_history_summary_label = QLabel(
            "Density calculations, solvent-subtracted outputs, and Fourier "
            "evaluations will be captured here for reload and comparison."
        )
        self.output_history_summary_label.setWordWrap(True)
        layout.addWidget(self.output_history_summary_label)

        self.output_history_table = QTableWidget(0, 8)
        self.output_history_table.setHorizontalHeaderLabels(
            [
                "Saved",
                "Type",
                "Context",
                "Averaging",
                "Mesh",
                "Smearing",
                "Solvent",
                "Fourier",
            ]
        )
        self.output_history_table.verticalHeader().setVisible(False)
        self.output_history_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self.output_history_table.setSelectionMode(
            QTableWidget.SelectionMode.ExtendedSelection
        )
        self.output_history_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers
        )
        self.output_history_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        self.output_history_table.horizontalHeader().setStretchLastSection(
            True
        )
        self.output_history_table.setMinimumHeight(200)
        self.output_history_table.itemSelectionChanged.connect(
            self._update_output_history_actions
        )
        self.output_history_table.itemDoubleClicked.connect(
            lambda *_args: self._load_selected_output_history_entry()
        )
        layout.addWidget(self.output_history_table)

        controls = QHBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(6)
        self.load_output_history_button = QPushButton("Load Selected")
        self.load_output_history_button.clicked.connect(
            self._load_selected_output_history_entry
        )
        controls.addWidget(self.load_output_history_button)
        self.compare_output_history_button = QPushButton("Compare Selected")
        self.compare_output_history_button.clicked.connect(
            self._compare_selected_output_history_entries
        )
        controls.addWidget(self.compare_output_history_button)
        controls.addStretch(1)
        layout.addLayout(controls)
        self._update_output_history_summary()
        self._update_output_history_actions()
        return group

    def _set_initial_defaults(self) -> None:
        if self._initial_output_dir is not None:
            self.output_dir_edit.setText(str(self._initial_output_dir))
        elif self._project_dir is not None:
            self.output_dir_edit.setText(
                str(
                    suggest_output_dir(
                        self._project_dir,
                        project_dir=self._project_dir,
                    )
                )
            )
        if self._project_q_min is not None:
            self.fourier_qmin_spin.setValue(float(self._project_q_min))
        if self._project_q_max is not None:
            self.fourier_qmax_spin.setValue(float(self._project_q_max))
        self._refresh_center_display()
        self._refresh_active_mesh_display()
        self._refresh_mesh_notice()
        self._refresh_smearing_display()
        self._reload_solvent_presets(selected_name="Water")
        self._sync_density_method_controls()
        self._refresh_contrast_display()
        self._refresh_fourier_info_labels(None)
        self._refresh_run_action_state()
        self._refresh_debye_scattering_group()
        self._append_status(
            "Waiting for an XYZ or PDB structure. Folder mode will preview and "
            "average across every valid structure in the folder when you run the calculation."
        )

    @staticmethod
    def _project_debye_waller_pair_term(
        payload: object,
    ) -> ElectronDensityDebyeWallerPairTerm:
        return ElectronDensityDebyeWallerPairTerm(
            scope=str(getattr(payload, "scope", "")),
            type_definition=str(getattr(payload, "type_definition", "")),
            pair_label_a=str(getattr(payload, "pair_label_a", "")),
            pair_label_b=str(getattr(payload, "pair_label_b", "")),
            pair_label=str(getattr(payload, "pair_label", "")),
            mean_distance_a=float(getattr(payload, "mean_distance_a", 0.0)),
            sigma_a=float(getattr(payload, "sigma_mean", 0.0)),
            sigma_std_a=float(getattr(payload, "sigma_std", 0.0)),
            sigma_squared_a2=float(
                getattr(payload, "sigma_squared_mean", 0.0)
            ),
            sigma_squared_std_a2=float(
                getattr(payload, "sigma_squared_std", 0.0)
            ),
            b_factor_a2=float(getattr(payload, "b_factor_mean", 0.0)),
            b_factor_std_a2=float(getattr(payload, "b_factor_std", 0.0)),
        ).normalized()

    def _load_project_debye_waller_terms(self) -> None:
        self._project_debye_waller_source_path = None
        if self._project_dir is None:
            return
        try:
            summary_path = find_saved_project_debye_waller_analysis(
                self._project_dir
            )
            if summary_path is None:
                return
            result = load_debye_waller_analysis_result(summary_path)
        except Exception as exc:
            self._append_status(
                "Could not load project Debye-Waller terms for electron-density smearing: "
                f"{exc}"
            )
            return
        imported_terms = tuple(
            self._project_debye_waller_pair_term(entry)
            for entry in result.aggregated_pair_summaries
        )
        if not imported_terms:
            return
        self._project_debye_waller_source_path = summary_path
        self._active_smearing_settings = (
            self._active_smearing_settings.with_pair_specific_terms(
                imported_terms,
                imported_terms=imported_terms,
                prefer_pair_specific=True,
            )
        )
        self._append_status(
            "Imported "
            f"{len(imported_terms)} aggregated Debye-Waller pair term"
            f"{'' if len(imported_terms) == 1 else 's'} from "
            f"{summary_path}."
        )
        if self._project_q_min is not None or self._project_q_max is not None:
            self._append_status(
                "Inherited project q-range context: "
                f"{self._project_q_min if self._project_q_min is not None else 'unset'} "
                f"to {self._project_q_max if self._project_q_max is not None else 'unset'} A^-1."
            )

    def _load_auto_snap_panes_setting(self) -> bool:
        raw_value = self._ui_settings().value(
            AUTO_SNAP_PANES_KEY,
            True,
        )
        if isinstance(raw_value, bool):
            return raw_value
        return str(raw_value).strip().lower() not in {
            "",
            "0",
            "false",
            "no",
            "off",
        }

    @Slot(bool)
    def _toggle_auto_snap_panes(self, enabled: bool) -> None:
        self._set_auto_snap_panes_enabled(enabled, persist=True)

    def set_auto_snap_enabled(self, enabled: bool) -> None:
        self._set_auto_snap_panes_enabled(enabled, persist=False)

    def _set_auto_snap_panes_enabled(
        self,
        enabled: bool,
        *,
        persist: bool,
    ) -> None:
        self._auto_snap_panes_enabled = bool(enabled)
        if hasattr(self, "auto_snap_panes_action"):
            self.auto_snap_panes_action.blockSignals(True)
            self.auto_snap_panes_action.setChecked(
                self._auto_snap_panes_enabled
            )
            self.auto_snap_panes_action.blockSignals(False)
        if hasattr(self, "_auto_snap_filter"):
            self._auto_snap_filter.set_enabled(self._auto_snap_panes_enabled)
        if persist:
            self._ui_settings().setValue(
                AUTO_SNAP_PANES_KEY,
                self._auto_snap_panes_enabled,
            )
            self.statusBar().showMessage(
                "Auto-snap panes "
                + ("enabled" if self._auto_snap_panes_enabled else "disabled")
            )

    def _ui_settings(self) -> QSettings:
        return QSettings("SAXShell", "SAXS")

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
            pin_contiguous_geometric_tracking=self._use_pinned_geometric_tracking(),
        ).normalized()

    def _sync_pinned_geometric_tracking_control_to_settings(
        self,
        settings: ElectronDensityMeshSettings,
    ) -> None:
        normalized = settings.normalized()
        self.pinned_geometric_tracking_checkbox.blockSignals(True)
        try:
            self.pinned_geometric_tracking_checkbox.setChecked(
                bool(normalized.pin_contiguous_geometric_tracking)
            )
        finally:
            self.pinned_geometric_tracking_checkbox.blockSignals(False)
        self._refresh_pinned_geometric_tracking_controls()

    def _sync_mesh_controls_to_settings(
        self,
        settings: ElectronDensityMeshSettings,
    ) -> None:
        normalized = settings.normalized()
        for widget, value in (
            (self.rstep_spin, float(normalized.rstep)),
            (self.theta_divisions_spin, int(normalized.theta_divisions)),
            (self.phi_divisions_spin, int(normalized.phi_divisions)),
            (self.rmax_spin, float(normalized.rmax)),
        ):
            widget.blockSignals(True)
            try:
                widget.setValue(value)
            finally:
                widget.blockSignals(False)
        self._sync_pinned_geometric_tracking_control_to_settings(normalized)

    def _cluster_group_mesh_settings_for_states(
        self,
        cluster_states: list[_ClusterDensityGroupState],
    ) -> ElectronDensityMeshSettings:
        shared_rmax = max(
            (
                max(float(state.reference_structure.rmax), 0.01)
                for state in cluster_states
            ),
            default=0.01,
        )
        current_settings = self._active_mesh_settings.normalized()
        return ElectronDensityMeshSettings(
            rstep=float(current_settings.rstep),
            theta_divisions=int(current_settings.theta_divisions),
            phi_divisions=int(current_settings.phi_divisions),
            rmax=float(shared_rmax),
            pin_contiguous_geometric_tracking=bool(
                current_settings.pin_contiguous_geometric_tracking
            ),
        ).normalized()

    def _smearing_settings_from_controls(
        self,
    ) -> ElectronDensitySmearingSettings:
        return ElectronDensitySmearingSettings(
            debye_waller_factor=float(self.smearing_factor_spin.value()),
            debye_waller_mode=str(
                self._active_smearing_settings.debye_waller_mode
            ),
            pair_specific_terms=tuple(
                self._active_smearing_settings.pair_specific_terms
            ),
            imported_pair_specific_terms=tuple(
                self._active_smearing_settings.imported_pair_specific_terms
            ),
        ).normalized()

    def _fourier_settings_from_controls(
        self,
    ) -> ElectronDensityFourierTransformSettings:
        return ElectronDensityFourierTransformSettings(
            r_min=float(self.fourier_rmin_spin.value()),
            r_max=float(self.fourier_rmax_spin.value()),
            domain_mode=self._fourier_domain_mode_from_controls(),
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
        self._debye_scattering_result = None
        self._close_debye_scattering_compare_dialog()
        self.profile_plot.draw_placeholder()
        self.smeared_profile_plot.draw_placeholder()
        self.residual_profile_plot.draw_placeholder()
        self.fourier_preview_plot.draw_placeholder()
        self.fourier_preview_info_label.setText("")
        self._refresh_scattering_plot()
        if hasattr(self, "reset_calculations_button"):
            self._refresh_run_action_state()
        self._refresh_debye_scattering_group()

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
            + "; geometric mass center="
            + self._format_point(structure.center_of_mass)
            + f"; {structure.reference_element} geometric center="
            + self._format_point(structure.reference_element_geometric_center)
            + "; active center="
            + self._format_point(structure.active_center)
            + f"; rmax={structure.rmax:.3f} Å"
        )

    def _refresh_center_display(self) -> None:
        if self._structure is None:
            self.center_mode_value.setText("Geometric Mass Center")
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
            self._refresh_pinned_geometric_tracking_controls()
            self._refresh_contiguous_frame_mode_notice()
            return
        structure = self._structure
        mode_label = self._center_mode_label_for_structure(structure)
        self.center_mode_value.setText(mode_label)
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
            f"{structure.nearest_atom_distance:.3f} Å from the geometric mass center"
        )
        self._refresh_pinned_geometric_tracking_controls()
        self._refresh_contiguous_frame_mode_notice()

    @Slot()
    def _handle_reference_element_changed(self) -> None:
        if self._structure is None:
            return
        preserve_mesh_confirmation = (
            self._mesh_confirmation_carries_forward_on_auto_apply()
        )
        active_cluster_state = self._active_cluster_group_state()
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
        if active_cluster_state is not None:
            active_cluster_state.reference_structure = updated_structure
        active_center_changed = not np.allclose(
            previous_structure.active_center,
            updated_structure.active_center,
        ) or not np.isclose(previous_structure.rmax, updated_structure.rmax)
        self._sync_controls_to_structure(
            sync_mesh_rmax=(active_cluster_state is None)
        )
        if active_center_changed:
            if active_cluster_state is None:
                self._reset_density_results()
                self._apply_mesh_settings(
                    self._mesh_settings_from_controls(),
                    announce=False,
                    preserve_viewer_display=True,
                )
            else:
                cleared_count = self._clear_cluster_group_outputs_for_keys(
                    {active_cluster_state.key}
                )
                self._set_mesh_settings_confirmed_for_run(
                    preserve_mesh_confirmation
                )
                self._sync_fourier_controls_to_domain(reset_bounds=True)
                self._refresh_fourier_info_labels(None)
                self._refresh_contrast_display()
                self._append_status(
                    "Updated "
                    f"{active_cluster_state.display_name} to the "
                    f"{updated_structure.reference_element} reference-element "
                    "center."
                    + (
                        " Cleared that stoichiometry's saved outputs so the "
                        "plots stay aligned with the new center."
                        if cleared_count > 0
                        else ""
                    )
                )
                self.statusBar().showMessage(
                    "Updated stoichiometry reference-element center"
                )
                self._sync_workspace_state()
                return
            self._set_mesh_settings_confirmed_for_run(
                preserve_mesh_confirmation
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
            if active_cluster_state is not None:
                self._populate_cluster_group_table()
            self._apply_mesh_settings(
                self._active_mesh_settings,
                announce=False,
                preserve_viewer_display=True,
            )
            self._set_mesh_settings_confirmed_for_run(
                preserve_mesh_confirmation
            )
            self._append_status(
                "Updated the reference element to "
                f"{updated_structure.reference_element}; offset from the "
                "total-atom geometric center = "
                f"{updated_structure.reference_element_offset_from_geometric_center:.3f} Å."
            )
            self.statusBar().showMessage("Updated reference element")
        self._sync_workspace_state()

    def _handle_cluster_group_reference_element_changed(
        self,
        group_key: str,
        combo: QComboBox,
    ) -> None:
        state = self._cluster_group_state_by_key(group_key)
        if state is None:
            return
        selected = combo.currentData() or combo.currentText()
        reference_element = str(selected or "").strip() or None
        if (
            reference_element is None
            or reference_element == state.reference_structure.reference_element
        ):
            return
        is_active_group = group_key == self._selected_cluster_group_key
        preserve_mesh_confirmation = (
            self._mesh_confirmation_carries_forward_on_auto_apply()
            if is_active_group
            else False
        )
        previous_structure = state.reference_structure
        try:
            updated_structure = recenter_electron_density_structure(
                previous_structure,
                center_mode=previous_structure.center_mode,
                reference_element=reference_element,
            )
        except Exception as exc:
            self._show_error("Reference Element Error", str(exc))
            self._populate_cluster_group_table()
            return
        state.reference_structure = updated_structure
        active_center_changed = not np.allclose(
            previous_structure.active_center,
            updated_structure.active_center,
        ) or not np.isclose(previous_structure.rmax, updated_structure.rmax)
        if is_active_group:
            self._structure = updated_structure
            self._sync_controls_to_structure(sync_mesh_rmax=False)
        if active_center_changed:
            cleared_count = self._clear_cluster_group_outputs_for_keys(
                {group_key}
            )
            if is_active_group:
                self._set_mesh_settings_confirmed_for_run(
                    preserve_mesh_confirmation
                )
                self._sync_fourier_controls_to_domain(reset_bounds=True)
                self._refresh_fourier_info_labels(None)
                self._refresh_contrast_display()
            self._append_status(
                "Updated "
                f"{state.display_name} to the "
                f"{updated_structure.reference_element} reference-element "
                "center."
                + (
                    " Cleared that stoichiometry's saved outputs so the plots "
                    "stay aligned with the new center."
                    if cleared_count > 0
                    else ""
                )
            )
        else:
            self._populate_cluster_group_table()
            if is_active_group:
                self._apply_mesh_settings(
                    self._active_mesh_settings,
                    announce=False,
                    preserve_viewer_display=True,
                )
                self._set_mesh_settings_confirmed_for_run(
                    preserve_mesh_confirmation
                )
            self._append_status(
                "Updated "
                f"{state.display_name} to use "
                f"{updated_structure.reference_element} as its stored "
                "reference element."
            )
        if is_active_group:
            self.statusBar().showMessage(
                "Updated stoichiometry reference element"
            )
        self._sync_workspace_state()

    def _refresh_smearing_display(self) -> None:
        settings = self._smearing_settings_from_controls()
        self._active_smearing_settings = settings
        self.smearing_sigma_value.setText(f"{settings.gaussian_sigma_a:.4f} Å")
        if settings.uses_pair_specific_terms:
            source_text = (
                ""
                if self._project_debye_waller_source_path is None
                else f" from {self._project_debye_waller_source_path.name}"
            )
            self.smearing_summary_value.setText(
                "Pair-specific Debye-Waller terms are selected by default: "
                f"{len(settings.pair_specific_terms)} imported pair type"
                f"{'' if len(settings.pair_specific_terms) == 1 else 's'}"
                + source_text
                + ". TODO: the electron-density smearing preview still uses "
                "the universal Gaussian factor until pair-aware smearing is "
                "implemented in a future update."
            )
        elif settings.debye_waller_factor <= 0.0:
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
        return self._profile_overlay_for_result(self._profile_result)

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
        active_cluster_state = self._active_cluster_group_state()
        if (
            active_cluster_state is not None
            and active_cluster_state.single_atom_only
        ):
            self.active_contrast_value.setText(
                "Single-atom Debye mode does not use solvent electron density."
            )
            self.active_contrast_value.setStyleSheet("color: #475569;")
            self.contrast_notice_value.setText(
                "This cluster bin skips electron-density and solvent-contrast "
                "evaluation and uses direct Debye scattering for I(Q)."
            )
            self.contrast_notice_value.setStyleSheet("color: #475569;")
            return
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
                "Run the density calculation, then click Apply Electron "
                "Density Contrast to populate the comparison overlays."
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
        active_cluster_state = self._active_cluster_group_state()
        preserved_fourier_settings = (
            None
            if active_cluster_state is None
            else self._cluster_state_fourier_settings(active_cluster_state)
        )
        if self._profile_result is None:
            if (
                active_cluster_state is not None
                and active_cluster_state.single_atom_only
            ):
                message = (
                    "Single-atom cluster groups use direct Debye scattering "
                    "and do not support solvent-contrast overlays."
                )
                self._refresh_contrast_display()
                if show_error:
                    self._show_error(
                        "Solvent Contrast Unavailable",
                        message,
                    )
                return False
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
        if active_cluster_state is not None:
            active_cluster_state.transform_result = None
            active_cluster_state.debye_scattering_result = None
        else:
            self._debye_scattering_result = None
        self._close_debye_scattering_compare_dialog()
        self._sync_fourier_controls_to_domain(reset_bounds=False)
        self._refresh_profile_plots()
        self._refresh_contrast_display()
        contrast = self._profile_result.solvent_contrast
        cutoff_rmax = None
        if active_cluster_state is not None:
            active_cluster_state.profile_result = self._profile_result
            self._sync_cluster_state_solvent_metadata(active_cluster_state)
            updated_settings = self._set_cluster_state_fourier_settings(
                active_cluster_state,
                preserved_fourier_settings
                or self._cluster_state_fourier_settings(active_cluster_state),
                prefer_solvent_cutoff=True,
            )
            cutoff_rmax = (
                float(updated_settings.r_max)
                if contrast is not None
                and contrast.cutoff_radius_a is not None
                else None
            )
            self._sync_fourier_controls_to_settings(updated_settings)
            self._populate_cluster_group_table()
        else:
            cutoff_rmax = self._sync_fourier_rmax_to_solvent_cutoff()
        self._refresh_fourier_preview_from_controls(clear_transform=True)
        if contrast is not None:
            self._capture_saved_output_entry(
                "solvent_subtraction",
                group_state=active_cluster_state,
                profile_result=self._profile_result,
            )
        if announce and contrast is not None:
            cutoff_text = (
                f"Highest-r cutoff={contrast.cutoff_radius_a:.3f} Å."
                if contrast.cutoff_radius_a is not None
                else "No highest-r solvent crossing was found."
            )
            if cutoff_rmax is not None:
                cutoff_text += f" Set transform r max to {cutoff_rmax:.3f} Å."
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

    @Slot()
    def _apply_electron_density_contrast_action(self) -> None:
        if self._cluster_group_states:
            self._apply_solvent_contrast_to_target_clusters(
                apply_to_all=bool(
                    self.apply_contrast_to_all_button.isChecked()
                )
            )
            return
        self._compute_solvent_contrast()

    def _apply_solvent_contrast_to_target_clusters(
        self,
        *,
        apply_to_all: bool,
    ) -> None:
        if not self._cluster_group_states:
            self._show_error(
                "No Stoichiometry Table",
                "Load a cluster-folder input before applying solvent "
                "subtraction across stoichiometries.",
            )
            return
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
        (
            target_states,
            scope_label,
            selected_keys,
        ) = self._batch_target_cluster_group_states(apply_to_all=apply_to_all)
        if not target_states:
            self._show_error(
                "No Stoichiometries Selected",
                "Select at least one stoichiometry row before applying "
                "solvent subtraction.",
            )
            return
        updated_count = 0
        skipped_pending = 0
        skipped_debye = 0
        failures: list[str] = []
        total_targets = len(target_states)
        error_payload: tuple[str, str] | None = None
        self._begin_batch_operation_progress(
            total=total_targets,
            message=(
                "Preparing batch solvent subtraction across "
                f"{scope_label}..."
            ),
            title="Applying Electron Density Contrast",
        )
        try:
            for index, state in enumerate(target_states, start=1):
                self._update_batch_operation_progress(
                    index - 1,
                    total_targets,
                    "Applying solvent subtraction to "
                    f"{state.display_name} ({index}/{total_targets}).",
                )
                if state.single_atom_only:
                    skipped_debye += 1
                    self._update_batch_operation_progress(
                        index,
                        total_targets,
                        "Skipped Debye-only stoichiometry "
                        f"{state.display_name} ({index}/{total_targets}).",
                    )
                    continue
                if state.profile_result is None:
                    skipped_pending += 1
                    self._update_batch_operation_progress(
                        index,
                        total_targets,
                        "Skipped pending stoichiometry "
                        f"{state.display_name} ({index}/{total_targets}).",
                    )
                    continue
                preserved_fourier_settings = (
                    self._cluster_state_fourier_settings(state)
                )
                try:
                    state.profile_result = (
                        apply_solvent_contrast_to_profile_result(
                            state.profile_result,
                            self._active_contrast_settings,
                            solvent_name=self._active_contrast_name,
                        )
                    )
                except Exception as exc:
                    failures.append(f"{state.display_name}: {exc}")
                    self._update_batch_operation_progress(
                        index,
                        total_targets,
                        "Solvent subtraction failed for "
                        f"{state.display_name} ({index}/{total_targets}).",
                    )
                    continue
                state.transform_result = None
                state.debye_scattering_result = None
                self._sync_cluster_state_solvent_metadata(state)
                self._set_cluster_state_fourier_settings(
                    state,
                    preserved_fourier_settings,
                    prefer_solvent_cutoff=True,
                )
                self._capture_saved_output_entry(
                    "solvent_subtraction",
                    group_state=state,
                    profile_result=state.profile_result,
                )
                updated_count += 1
                self._update_batch_operation_progress(
                    index,
                    total_targets,
                    "Applied solvent subtraction to "
                    f"{state.display_name} ({index}/{total_targets}).",
                )
            if updated_count <= 0:
                self._update_batch_operation_progress(
                    total_targets,
                    total_targets,
                    "Batch solvent subtraction finished without updates.",
                )
                if failures:
                    error_payload = (
                        "Batch Solvent Contrast Error",
                        "\n".join(failures[:6]),
                    )
                else:
                    error_payload = (
                        "No Solvent Targets Updated",
                        "Only stoichiometries with computed density "
                        "profiles can receive solvent subtraction. "
                        "Single-atom Debye rows are skipped.",
                    )
            else:
                self._update_batch_operation_progress(
                    total_targets,
                    total_targets,
                    "Refreshing solvent-subtracted stoichiometry views...",
                )
                self._close_debye_scattering_compare_dialog()
                self._refresh_cluster_views_after_batch_update(
                    target_states,
                    selected_keys=selected_keys,
                )
                self._update_batch_operation_progress(
                    total_targets,
                    total_targets,
                    "Batch solvent subtraction complete.",
                )
        finally:
            self._close_batch_operation_progress_dialog()
        if error_payload is not None:
            self._show_error(*error_payload)
            return
        summary_parts = [
            f"Applied solvent subtraction to {updated_count} stoichiometr"
            f"{'y' if updated_count == 1 else 'ies'}"
        ]
        if skipped_pending > 0:
            summary_parts.append(
                f"skipped {skipped_pending} pending density row"
                f"{'' if skipped_pending == 1 else 's'}"
            )
        if skipped_debye > 0:
            summary_parts.append(
                f"skipped {skipped_debye} Debye-only row"
                f"{'' if skipped_debye == 1 else 's'}"
            )
        if failures:
            summary_parts.append(
                f"{len(failures)} row"
                f"{'' if len(failures) == 1 else 's'} failed"
            )
        self._append_status(
            "Batch solvent update: "
            + "; ".join(summary_parts)
            + f" across {scope_label}. Each updated stoichiometry kept its "
            "own solvent cutoff for later Fourier evaluation."
        )
        for failure in failures:
            self._append_status(f"Batch solvent warning: {failure}")
        self.statusBar().showMessage(
            "Updated solvent subtraction across batch"
        )

    def _apply_smearing_from_controls(self, *_args: object) -> None:
        self._refresh_smearing_display()
        if self._profile_result is None:
            return
        active_cluster_state = self._active_cluster_group_state()
        preserved_fourier_settings = (
            None
            if active_cluster_state is None
            else self._cluster_state_fourier_settings(active_cluster_state)
        )
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
        if active_cluster_state is not None:
            active_cluster_state.profile_result = self._profile_result
            active_cluster_state.transform_result = None
            active_cluster_state.debye_scattering_result = None
            self._sync_cluster_state_solvent_metadata(active_cluster_state)
            updated_settings = self._set_cluster_state_fourier_settings(
                active_cluster_state,
                preserved_fourier_settings
                or self._cluster_state_fourier_settings(active_cluster_state),
                prefer_solvent_cutoff=True,
            )
            self._sync_fourier_controls_to_settings(updated_settings)
        else:
            self._sync_fourier_rmax_to_solvent_cutoff()
        self._sync_fourier_controls_to_domain(reset_bounds=False)
        self._refresh_profile_plots()
        self._populate_cluster_group_table()
        self._refresh_contrast_display()
        self.statusBar().showMessage("Updated Gaussian smearing preview")
        self._append_status(
            "Updated Gaussian smearing preview to "
            f"factor={self._profile_result.smearing_settings.debye_waller_factor:.6f} Å² "
            f"(sigma={self._profile_result.smearing_settings.gaussian_sigma_a:.4f} Å)."
        )
        self._refresh_fourier_preview_from_controls(clear_transform=True)
        if not self._capture_smearing_saved_output_if_enabled(
            group_state=active_cluster_state,
            profile_result=self._profile_result,
        ):
            self._sync_workspace_state()

    @Slot()
    def _apply_smearing_action(self) -> None:
        if self._cluster_group_states:
            self._apply_smearing_to_target_clusters(
                apply_to_all=bool(
                    self.apply_smearing_to_all_button.isChecked()
                )
            )
            return
        self._apply_smearing_from_controls()

    def _apply_smearing_to_target_clusters(
        self,
        *,
        apply_to_all: bool,
    ) -> None:
        if not self._cluster_group_states:
            self._show_error(
                "No Stoichiometry Table",
                "Load a cluster-folder input before applying smearing across stoichiometries.",
            )
            return
        self._refresh_smearing_display()
        (
            target_states,
            scope_label,
            selected_keys,
        ) = self._batch_target_cluster_group_states(apply_to_all=apply_to_all)
        if not target_states:
            self._show_error(
                "No Stoichiometries Selected",
                "Select at least one stoichiometry row before applying "
                "smearing.",
            )
            return
        updated_count = 0
        skipped_pending = 0
        skipped_debye = 0
        failures: list[str] = []
        total_targets = len(target_states)
        error_payload: tuple[str, str] | None = None
        self._begin_batch_operation_progress(
            total=total_targets,
            message=(
                "Preparing batch smearing update across " f"{scope_label}..."
            ),
            title="Applying Gaussian Smearing",
        )
        try:
            for index, state in enumerate(target_states, start=1):
                self._update_batch_operation_progress(
                    index - 1,
                    total_targets,
                    "Applying Gaussian smearing to "
                    f"{state.display_name} ({index}/{total_targets}).",
                )
                if state.single_atom_only:
                    skipped_debye += 1
                    self._update_batch_operation_progress(
                        index,
                        total_targets,
                        "Skipped Debye-only stoichiometry "
                        f"{state.display_name} ({index}/{total_targets}).",
                    )
                    continue
                if state.profile_result is None:
                    skipped_pending += 1
                    self._update_batch_operation_progress(
                        index,
                        total_targets,
                        "Skipped pending stoichiometry "
                        f"{state.display_name} ({index}/{total_targets}).",
                    )
                    continue
                preserved_fourier_settings = (
                    self._cluster_state_fourier_settings(state)
                )
                try:
                    state.profile_result = apply_smearing_to_profile_result(
                        state.profile_result,
                        self._active_smearing_settings,
                    )
                except Exception as exc:
                    failures.append(f"{state.display_name}: {exc}")
                    self._update_batch_operation_progress(
                        index,
                        total_targets,
                        "Gaussian smearing failed for "
                        f"{state.display_name} ({index}/{total_targets}).",
                    )
                    continue
                state.transform_result = None
                state.debye_scattering_result = None
                self._sync_cluster_state_solvent_metadata(state)
                self._set_cluster_state_fourier_settings(
                    state,
                    preserved_fourier_settings,
                    prefer_solvent_cutoff=True,
                )
                self._capture_smearing_saved_output_if_enabled(
                    group_state=state,
                    profile_result=state.profile_result,
                )
                updated_count += 1
                self._update_batch_operation_progress(
                    index,
                    total_targets,
                    "Applied Gaussian smearing to "
                    f"{state.display_name} ({index}/{total_targets}).",
                )
            if updated_count <= 0:
                self._update_batch_operation_progress(
                    total_targets,
                    total_targets,
                    "Batch smearing finished without updates.",
                )
                if failures:
                    error_payload = (
                        "Batch Smearing Error",
                        "\n".join(failures[:6]),
                    )
                else:
                    error_payload = (
                        "No Smearing Targets Updated",
                        "Only stoichiometries with computed density "
                        "profiles can be updated. Single-atom Debye rows "
                        "are skipped.",
                    )
            else:
                self._update_batch_operation_progress(
                    total_targets,
                    total_targets,
                    "Refreshing smeared stoichiometry views...",
                )
                self._close_debye_scattering_compare_dialog()
                self._refresh_cluster_views_after_batch_update(
                    target_states,
                    selected_keys=selected_keys,
                )
                self._update_batch_operation_progress(
                    total_targets,
                    total_targets,
                    "Batch smearing complete.",
                )
        finally:
            self._close_batch_operation_progress_dialog()
        if error_payload is not None:
            self._show_error(*error_payload)
            return
        summary_parts = [
            f"Applied smearing factor={self._active_smearing_settings.debye_waller_factor:.6f} A^2 "
            f"to {updated_count} stoichiometr"
            f"{'y' if updated_count == 1 else 'ies'}"
        ]
        if skipped_pending > 0:
            summary_parts.append(
                f"skipped {skipped_pending} pending density row"
                f"{'' if skipped_pending == 1 else 's'}"
            )
        if skipped_debye > 0:
            summary_parts.append(
                f"skipped {skipped_debye} Debye-only row"
                f"{'' if skipped_debye == 1 else 's'}"
            )
        if failures:
            summary_parts.append(
                f"{len(failures)} row"
                f"{'' if len(failures) == 1 else 's'} failed"
            )
        self._append_status(
            "Batch smearing update: "
            + "; ".join(summary_parts)
            + f" across {scope_label}."
        )
        for failure in failures:
            self._append_status(f"Batch smearing warning: {failure}")
        self.statusBar().showMessage("Updated smearing across batch")
        self._sync_workspace_state()

    def _available_fourier_r_domain(self) -> tuple[float, float] | None:
        domain_max: float | None = None
        if self._profile_result is not None:
            radial_values = np.asarray(
                self._profile_result.radial_centers, dtype=float
            )
            if radial_values.size >= 2:
                domain_max = float(radial_values[-1])
        elif self._active_mesh_geometry is not None:
            radial_edges = np.asarray(
                self._active_mesh_geometry.radial_edges, dtype=float
            )
            if radial_edges.size >= 2:
                radial_centers = (radial_edges[:-1] + radial_edges[1:]) * 0.5
                if radial_centers.size >= 1:
                    domain_max = float(radial_centers[-1])
        if domain_max is None:
            return None
        if self._fourier_domain_mode_from_controls() == "mirrored":
            return -domain_max, domain_max
        return 0.0, domain_max

    def _sync_fourier_controls_to_domain(
        self,
        *,
        reset_bounds: bool,
    ) -> None:
        available_domain = self._available_fourier_r_domain()
        if available_domain is None:
            return
        domain_min, domain_max = available_domain
        mirrored_mode = self._fourier_domain_mode_from_controls() == "mirrored"
        minimum_span = max(min(domain_max - domain_min, 1.0e-3), 1.0e-6)
        self.fourier_rmin_spin.blockSignals(True)
        self.fourier_rmax_spin.blockSignals(True)
        try:
            current_rmin = float(self.fourier_rmin_spin.value())
            current_rmax = float(self.fourier_rmax_spin.value())
            if mirrored_mode:
                mirrored_domain_max = abs(float(domain_max))
                minimum_radius = max(min(mirrored_domain_max, 1.0e-3), 1.0e-6)
                self.fourier_rmin_spin.setRange(
                    -mirrored_domain_max,
                    -minimum_radius,
                )
                self.fourier_rmax_spin.setRange(
                    minimum_radius,
                    mirrored_domain_max,
                )
                if (
                    reset_bounds
                    or current_rmax > mirrored_domain_max
                    or current_rmax < minimum_radius
                ):
                    current_rmax = mirrored_domain_max
                current_rmax = float(
                    np.clip(
                        current_rmax,
                        minimum_radius,
                        mirrored_domain_max,
                    )
                )
                current_rmin = -current_rmax
                self.fourier_rmin_spin.setValue(current_rmin)
                self.fourier_rmax_spin.setValue(current_rmax)
            else:
                self.fourier_rmin_spin.setRange(
                    domain_min,
                    max(domain_max - minimum_span, domain_min),
                )
                self.fourier_rmax_spin.setRange(
                    min(domain_min + minimum_span, domain_max),
                    domain_max,
                )
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
        self._refresh_fourier_domain_mode_controls()

    def _sync_fourier_controls_to_settings(
        self,
        settings: ElectronDensityFourierTransformSettings,
    ) -> None:
        normalized = settings.normalized()
        widgets = (
            self.fourier_legacy_mode_checkbox,
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
            self.fourier_legacy_mode_checkbox.setChecked(
                normalized.domain_mode == "legacy"
            )
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
        self._refresh_fourier_domain_mode_controls()

    def _sync_fourier_rmax_to_solvent_cutoff(self) -> float | None:
        if (
            self._profile_result is None
            or self._profile_result.solvent_contrast is None
            or self._profile_result.solvent_contrast.cutoff_radius_a is None
        ):
            return None
        available_domain = self._available_fourier_r_domain()
        if available_domain is None:
            return None
        domain_min, domain_max = available_domain
        mirrored_mode = self._fourier_domain_mode_from_controls() == "mirrored"
        minimum_span = max(min(domain_max - domain_min, 1.0e-3), 1.0e-6)
        minimum_radius = max(min(abs(float(domain_max)), 1.0e-3), 1.0e-6)
        current_rmin = (
            -float(self.fourier_rmax_spin.value())
            if mirrored_mode
            else float(self.fourier_rmin_spin.value())
        )
        target_rmax = float(
            np.clip(
                float(self._profile_result.solvent_contrast.cutoff_radius_a),
                (
                    minimum_radius
                    if mirrored_mode
                    else current_rmin + minimum_span
                ),
                abs(float(domain_max)) if mirrored_mode else domain_max,
            )
        )
        if mirrored_mode:
            self.fourier_rmin_spin.blockSignals(True)
            try:
                self.fourier_rmin_spin.setValue(-target_rmax)
            finally:
                self.fourier_rmin_spin.blockSignals(False)
        self.fourier_rmax_spin.blockSignals(True)
        try:
            self.fourier_rmax_spin.setValue(target_rmax)
        finally:
            self.fourier_rmax_spin.blockSignals(False)
        return target_rmax

    def _refresh_fourier_info_labels(
        self,
        preview: ElectronDensityFourierTransformPreview | None,
    ) -> None:
        active_cluster_state = self._active_cluster_group_state()
        active_settings = (
            None
            if active_cluster_state is None
            else self._cluster_state_fourier_settings(active_cluster_state)
        )
        if (
            active_cluster_state is not None
            and active_cluster_state.single_atom_only
        ):
            if preview is None:
                if active_cluster_state.transform_result is not None:
                    settings = (
                        active_cluster_state.transform_result.preview.settings
                    )
                else:
                    settings = (
                        active_settings
                        if active_settings is not None
                        else self._fourier_settings_from_controls()
                    )
            else:
                settings = preview.settings
            self._active_fourier_settings = settings
            q_min = float(settings.q_min)
            q_max = float(settings.q_max)
            q_step = float(settings.q_step)
            q_count = len(
                np.arange(
                    q_min,
                    q_max + q_step * 0.5,
                    q_step,
                    dtype=float,
                )
            )
            self.fourier_available_range_value.setText("Direct Debye mode")
            self.fourier_nyquist_value.setText(
                f"q={q_min:.4f} to {q_max:.4f} Å⁻¹, Δq={q_step:.4f} Å⁻¹, {q_count} points."
            )
            self.fourier_preview_info_label.setText(
                "Direct Debye scattering  ·  no r-space preview"
            )
            if active_cluster_state.transform_result is None:
                self.fourier_notice_value.setText(
                    "This cluster bin contains only single-atom structures. "
                    "Run the calculation or click Evaluate Fourier Transform "
                    "to build a direct Debye I(Q) trace. The r-range "
                    "and window controls are ignored."
                )
                self.fourier_notice_value.setStyleSheet("color: #92400e;")
            else:
                self.fourier_notice_value.setText(
                    "This cluster bin uses direct Debye scattering. "
                    "Electron-density, solvent-contrast, and Fourier-profile preparation "
                    "are skipped; only the q-grid and axis-scale settings apply."
                )
                self.fourier_notice_value.setStyleSheet("color: #166534;")
            return
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
            if active_settings is not None:
                settings = active_settings
            else:
                try:
                    settings = self._fourier_settings_from_controls()
                except Exception as exc:
                    self.fourier_nyquist_value.setText(
                        "Adjust the Fourier-transform bounds to satisfy the requested sampling."
                    )
                    self.fourier_notice_value.setText(str(exc))
                    self.fourier_notice_value.setStyleSheet("color: #991b1b;")
                    self.fourier_preview_info_label.setText("")
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
            self.fourier_preview_info_label.setText(
                f"Δr={resampling_step:.4f} Å  ·  "
                f"qmax≈{nyquist_qmax:.3f} Å⁻¹  ·  "
                f"Δq≈{independent_qstep:.3f} Å⁻¹"
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
        self.fourier_preview_info_label.setText(
            f"Δr={preview.resampling_step_a:.4f} Å  ·  "
            f"qmax≈{preview.nyquist_q_max_a_inverse:.3f} Å⁻¹  ·  "
            f"Δq≈{preview.independent_q_step_a_inverse:.3f} Å⁻¹"
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

    def _display_fourier_preview(
        self,
        preview: ElectronDensityFourierTransformPreview | None,
    ) -> None:
        self._fourier_preview = preview
        if preview is None:
            self.fourier_preview_plot.draw_placeholder()
            self._refresh_fourier_info_labels(None)
            return
        self.fourier_preview_plot.set_preview(preview)
        if self.auto_expand_checkbox.isChecked():
            self.fourier_preview_section.expand()
        self._refresh_fourier_info_labels(preview)

    def _refresh_fourier_preview_from_controls(
        self,
        *_args: object,
        clear_transform: bool = True,
    ) -> None:
        active_cluster_state = self._active_cluster_group_state()
        if clear_transform and active_cluster_state is not None:
            active_cluster_state.debye_scattering_result = None
        if (
            active_cluster_state is not None
            and active_cluster_state.single_atom_only
        ):
            try:
                settings = self._fourier_settings_from_controls()
            except Exception:
                settings = self._default_fourier_settings()
            self._active_fourier_settings = (
                self._set_cluster_state_fourier_settings(
                    active_cluster_state,
                    settings,
                    prefer_solvent_cutoff=False,
                )
            )
            self._fourier_preview = None
            self.fourier_preview_plot.draw_placeholder()
            if clear_transform:
                self._fourier_result = None
                self._debye_scattering_result = None
                self._refresh_scattering_plot()
                self._close_debye_scattering_compare_dialog()
            self._populate_cluster_group_table()
            self._refresh_fourier_info_labels(None)
            self._sync_workspace_state()
            return
        try:
            settings = self._fourier_settings_from_controls()
        except Exception as exc:
            self._active_fourier_settings = self._default_fourier_settings()
            self.fourier_notice_value.setText(str(exc))
            self.fourier_notice_value.setStyleSheet("color: #991b1b;")
            self.fourier_preview_plot.draw_placeholder()
            if clear_transform:
                self._fourier_result = None
                self._debye_scattering_result = None
                self._refresh_scattering_plot()
                self._close_debye_scattering_compare_dialog()
            self._sync_workspace_state()
            return
        if active_cluster_state is not None:
            settings = self._set_cluster_state_fourier_settings(
                active_cluster_state,
                settings,
                prefer_solvent_cutoff=bool(
                    active_cluster_state.solvent_cutoff_radius_a
                ),
            )
            self._populate_cluster_group_table()
        self._active_fourier_settings = settings
        if self._profile_result is None:
            self._fourier_preview = None
            self.fourier_preview_plot.draw_placeholder()
            if clear_transform:
                self._fourier_result = None
                self._debye_scattering_result = None
                self._refresh_scattering_plot()
                self._close_debye_scattering_compare_dialog()
            self._refresh_fourier_info_labels(None)
            self._sync_workspace_state()
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
                self._debye_scattering_result = None
                self._refresh_scattering_plot()
                self._close_debye_scattering_compare_dialog()
            self._refresh_fourier_info_labels(None)
            self.fourier_notice_value.setText(str(exc))
            self.fourier_notice_value.setStyleSheet("color: #991b1b;")
            self._sync_workspace_state()
            return
        self._display_fourier_preview(preview)
        if clear_transform:
            self._fourier_result = None
            self._debye_scattering_result = None
            self._refresh_scattering_plot()
            self._close_debye_scattering_compare_dialog()
        self._sync_workspace_state()

    def _additional_scattering_series(self) -> list[dict[str, object]]:
        if (
            not self._cluster_group_states
            or not self.show_all_cluster_transforms_checkbox.isChecked()
        ):
            return []
        active_key = self._selected_cluster_group_key
        overlays: list[dict[str, object]] = []
        for state in self._cluster_group_states:
            if state.transform_result is None or state.key == active_key:
                continue
            overlays.append(
                {
                    "label": state.display_name,
                    "q_values": np.asarray(
                        state.transform_result.q_values,
                        dtype=float,
                    ),
                    "intensity": np.asarray(
                        state.transform_result.intensity,
                        dtype=float,
                    ),
                    "color": state.trace_color,
                }
            )
        return overlays

    def _refresh_scattering_plot(self) -> None:
        if self._fourier_result is None:
            self.scattering_plot.draw_placeholder()
            self.scattering_info_label.setText("")
            return
        active_cluster_state = self._active_cluster_group_state()
        uses_single_atom_debye = (
            active_cluster_state is not None
            and active_cluster_state.single_atom_only
        )
        self.scattering_plot.set_transform_result(
            self._fourier_result,
            log_q_axis=self.fourier_log_q_checkbox.isChecked(),
            log_intensity_axis=self.fourier_log_intensity_checkbox.isChecked(),
            additional_series=self._additional_scattering_series(),
            primary_label=(
                "Direct Debye intensity"
                if uses_single_atom_debye
                else (
                    "Born-approximation intensity"
                    if not self._cluster_group_states
                    else (
                        self._active_cluster_group_state().display_name
                        if self._active_cluster_group_state() is not None
                        else "Born-approximation intensity"
                    )
                )
            ),
            primary_color=(
                "#2563eb"
                if uses_single_atom_debye
                else (
                    "#b45309"
                    if self._active_cluster_group_state() is None
                    else self._active_cluster_group_state().trace_color
                )
            ),
        )
        settings = self._fourier_result.preview.settings
        if uses_single_atom_debye:
            self.scattering_info_label.setText(
                f"Direct Debye scattering  ·  "
                f"q: {settings.q_min:.3f}–{settings.q_max:.3f} Å⁻¹  ·  "
                f"Δq={settings.q_step:.3f} Å⁻¹  ·  "
                f"{len(self._fourier_result.q_values)} q points"
            )
            return
        self.scattering_info_label.setText(
            f"Window: {settings.window_function}  ·  "
            f"r: {settings.r_min:.3f}–{settings.r_max:.3f} Å  ·  "
            f"{len(self._fourier_result.q_values)} q points"
        )

    @Slot()
    def _evaluate_fourier_transform_action(self) -> None:
        if self._cluster_group_states:
            self._evaluate_fourier_transform_for_target_clusters(
                apply_to_all=bool(self.apply_fourier_to_all_button.isChecked())
            )
            return
        self._evaluate_fourier_transform()

    @Slot()
    def _evaluate_fourier_transform(self) -> None:
        active_cluster_state = self._active_cluster_group_state()
        try:
            settings = (
                self._cluster_state_fourier_settings(active_cluster_state)
                if active_cluster_state is not None
                else self._fourier_settings_from_controls()
            )
        except Exception as exc:
            self._show_error("Fourier Transform Error", str(exc))
            return
        if (
            active_cluster_state is not None
            and active_cluster_state.single_atom_only
        ):
            try:
                transform_result = (
                    compute_single_atom_debye_scattering_profile_for_input(
                        active_cluster_state.inspection,
                        settings,
                    )
                )
            except Exception as exc:
                self._show_error("Debye Scattering Error", str(exc))
                return
            self._fourier_preview = None
            self._fourier_result = transform_result
            self._debye_scattering_result = None
            active_cluster_state.fourier_settings = (
                transform_result.preview.settings
            )
            active_cluster_state.transform_result = transform_result
            active_cluster_state.debye_scattering_result = None
            self._sync_fourier_controls_to_settings(
                transform_result.preview.settings
            )
            self.fourier_preview_plot.draw_placeholder()
            self._refresh_fourier_info_labels(None)
            self._populate_cluster_group_table()
            self._refresh_scattering_plot()
            self._append_status(
                "Built direct Debye I(Q) for single-atom cluster group "
                f"{active_cluster_state.display_name}: "
                f"{len(transform_result.q_values)} q points."
            )
            self._update_push_to_model_state()
            self._refresh_debye_scattering_group()
            self.statusBar().showMessage("Built single-atom Debye scattering")
            self._close_debye_scattering_compare_dialog()
            self._sync_workspace_state()
            return
        if self._profile_result is None:
            self._show_error(
                "No Density Profile",
                "Run the electron-density calculation before evaluating the Fourier transform.",
            )
            return
        try:
            transform_result = compute_electron_density_scattering_profile(
                self._profile_result,
                settings,
            )
        except Exception as exc:
            self._show_error("Fourier Transform Error", str(exc))
            return
        self._fourier_preview = transform_result.preview
        self._fourier_result = transform_result
        self._debye_scattering_result = None
        self._sync_fourier_controls_to_settings(
            transform_result.preview.settings
        )
        self._refresh_fourier_info_labels(transform_result.preview)
        self.fourier_preview_plot.set_preview(transform_result.preview)
        active_cluster_state = self._active_cluster_group_state()
        if active_cluster_state is not None:
            active_cluster_state.fourier_settings = (
                transform_result.preview.settings
            )
            active_cluster_state.transform_result = transform_result
            active_cluster_state.debye_scattering_result = None
            self._populate_cluster_group_table()
        self._capture_saved_output_entry(
            "fourier_transform",
            group_state=active_cluster_state,
            profile_result=self._profile_result,
            transform_result=transform_result,
        )
        self._refresh_scattering_plot()
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
        self._update_push_to_model_state()
        self._refresh_debye_scattering_group()
        self.statusBar().showMessage("Evaluated Fourier transform")
        self._close_debye_scattering_compare_dialog()
        self._sync_workspace_state()

    def _evaluate_fourier_transform_for_target_clusters(
        self,
        *,
        apply_to_all: bool,
    ) -> None:
        if not self._cluster_group_states:
            self._show_error(
                "No Stoichiometry Table",
                "Load a cluster-folder input before evaluating Fourier "
                "transforms across stoichiometries.",
            )
            return
        (
            target_states,
            scope_label,
            selected_keys,
        ) = self._batch_target_cluster_group_states(apply_to_all=apply_to_all)
        if not target_states:
            self._show_error(
                "No Stoichiometries Selected",
                "Select at least one stoichiometry row before evaluating "
                "the Fourier transform.",
            )
            return
        density_count = 0
        debye_count = 0
        skipped_pending = 0
        failures: list[str] = []
        total_targets = len(target_states)
        error_payload: tuple[str, str] | None = None
        self._begin_batch_operation_progress(
            total=total_targets,
            message=(
                "Preparing batch Fourier evaluation across "
                f"{scope_label}..."
            ),
            title="Evaluating Fourier Transforms",
        )
        try:
            for index, state in enumerate(target_states, start=1):
                self._update_batch_operation_progress(
                    index - 1,
                    total_targets,
                    "Evaluating Fourier transform for "
                    f"{state.display_name} ({index}/{total_targets}).",
                )
                settings = self._cluster_state_fourier_settings(state)
                try:
                    if state.single_atom_only:
                        transform_result = compute_single_atom_debye_scattering_profile_for_input(
                            state.inspection,
                            settings,
                        )
                        state.fourier_settings = (
                            transform_result.preview.settings
                        )
                        state.transform_result = transform_result
                        state.debye_scattering_result = None
                        debye_count += 1
                        self._update_batch_operation_progress(
                            index,
                            total_targets,
                            "Built direct Debye scattering for "
                            f"{state.display_name} ({index}/{total_targets}).",
                        )
                        continue
                    if state.profile_result is None:
                        skipped_pending += 1
                        self._update_batch_operation_progress(
                            index,
                            total_targets,
                            "Skipped pending stoichiometry "
                            f"{state.display_name} ({index}/{total_targets}).",
                        )
                        continue
                    transform_result = (
                        compute_electron_density_scattering_profile(
                            state.profile_result,
                            settings,
                        )
                    )
                except Exception as exc:
                    failures.append(f"{state.display_name}: {exc}")
                    self._update_batch_operation_progress(
                        index,
                        total_targets,
                        "Fourier evaluation failed for "
                        f"{state.display_name} ({index}/{total_targets}).",
                    )
                    continue
                state.fourier_settings = transform_result.preview.settings
                state.transform_result = transform_result
                state.debye_scattering_result = None
                self._capture_saved_output_entry(
                    "fourier_transform",
                    group_state=state,
                    profile_result=state.profile_result,
                    transform_result=transform_result,
                )
                density_count += 1
                self._update_batch_operation_progress(
                    index,
                    total_targets,
                    "Evaluated Fourier transform for "
                    f"{state.display_name} ({index}/{total_targets}).",
                )
            if density_count <= 0 and debye_count <= 0:
                self._update_batch_operation_progress(
                    total_targets,
                    total_targets,
                    "Batch Fourier evaluation finished without updates.",
                )
                if failures:
                    error_payload = (
                        "Batch Fourier Transform Error",
                        "\n".join(failures[:6]),
                    )
                else:
                    error_payload = (
                        "No Fourier Targets Updated",
                        "Only stoichiometries with computed density "
                        "profiles or single-atom Debye rows can be "
                        "evaluated.",
                    )
            else:
                self._update_batch_operation_progress(
                    total_targets,
                    total_targets,
                    "Refreshing Fourier-transform stoichiometry views...",
                )
                self._close_debye_scattering_compare_dialog()
                self._refresh_cluster_views_after_batch_update(
                    target_states,
                    selected_keys=selected_keys,
                )
                self._update_batch_operation_progress(
                    total_targets,
                    total_targets,
                    "Batch Fourier evaluation complete.",
                )
        finally:
            self._close_batch_operation_progress_dialog()
        if error_payload is not None:
            self._show_error(*error_payload)
            return
        summary_parts: list[str] = []
        if density_count > 0:
            summary_parts.append(
                f"{density_count} Born-approximation transform"
                f"{'' if density_count == 1 else 's'}"
            )
        if debye_count > 0:
            summary_parts.append(
                f"{debye_count} direct Debye trace"
                f"{'' if debye_count == 1 else 's'}"
            )
        if skipped_pending > 0:
            summary_parts.append(
                f"skipped {skipped_pending} pending density row"
                f"{'' if skipped_pending == 1 else 's'}"
            )
        if failures:
            summary_parts.append(
                f"{len(failures)} row"
                f"{'' if len(failures) == 1 else 's'} failed"
            )
        self._append_status(
            "Batch Fourier evaluation: "
            + "; ".join(summary_parts)
            + f" across {scope_label}. Each density-based transform used the "
            "stored Fourier settings for that row, including its own r range."
        )
        for failure in failures:
            self._append_status(f"Batch Fourier warning: {failure}")
        self.statusBar().showMessage(
            "Evaluated Fourier transforms across batch"
        )
        self._sync_workspace_state()

    @Slot(bool)
    def _apply_fourier_axis_scale_preferences(self, _checked: bool) -> None:
        if self._fourier_result is None:
            return
        self._refresh_scattering_plot()

    def _sync_controls_to_structure(
        self,
        *,
        sync_mesh_rmax: bool = True,
    ) -> None:
        if self._structure is None:
            return
        if sync_mesh_rmax:
            self.rmax_spin.setValue(max(float(self._structure.rmax), 0.01))
        self._sync_reference_element_controls()
        self._refresh_center_display()
        self._refresh_structure_summary()

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
                f"center mode={center_label}, "
                f"center={center_text}."
            )
            return
        geometry = self._active_mesh_geometry
        self.active_mesh_value.setText(
            f"{self._format_mesh_settings_summary(geometry.settings)}, "
            f"shells={geometry.shell_count}, "
            f"domain=0 to {geometry.domain_max_radius:.3f} Å, "
            f"center mode={center_label}, "
            f"center={center_text}"
        )

    @Slot()
    def _refresh_mesh_notice(self) -> None:
        if self._manual_mesh_lock_settings is not None:
            self.pending_mesh_value.setText(
                "Mesh settings are locked at "
                f"{self._format_mesh_settings_summary(self._manual_mesh_lock_settings)} "
                "while manual-mode stoichiometry calculations remain active. "
                "Reset the calculated densities to edit the mesh."
            )
            self.pending_mesh_value.setStyleSheet("color: #92400e;")
            return
        pending = self._mesh_settings_from_controls()
        if pending != self._active_mesh_settings:
            self.pending_mesh_value.setText(
                "Pending field values differ from the rendered mesh. Press "
                "Update Mesh Settings to redraw the spherical overlay before "
                "running, or continue with the last updated mesh."
            )
            self.pending_mesh_value.setStyleSheet("color: #92400e;")
            return
        if self._mesh_settings_confirmed_for_run:
            self.pending_mesh_value.setText(
                "Pending field values match the rendered mesh."
            )
            self.pending_mesh_value.setStyleSheet("color: #166534;")
            return
        self.pending_mesh_value.setText(
            "Mesh settings still reflect the current default values shown in "
            "the UI. Press Update Mesh Settings to confirm them before "
            "running, or continue from the run warning."
        )
        self.pending_mesh_value.setStyleSheet("color: #92400e;")

    def _load_cluster_group_input(
        self,
        path: Path,
        *,
        cluster_states: list[_ClusterDensityGroupState] | None = None,
    ) -> bool:
        if cluster_states is None:
            cluster_states = self._cluster_group_states_for_path(path)
        if not cluster_states:
            return False
        single_atom_count = sum(
            1 for state in cluster_states if state.single_atom_only
        )
        self._cluster_group_states = cluster_states
        for state in self._cluster_group_states:
            if state.fourier_settings is None:
                state.fourier_settings = (
                    self._active_fourier_settings.normalized()
                )
        self._selected_cluster_group_key = cluster_states[0].key
        self._reset_density_results()
        self._reset_progress_display("Idle")
        self.input_mode_value.setText(
            f"Cluster folders ({len(cluster_states)} stoichiometries)"
        )
        self.reference_file_value.setText(
            str(cluster_states[0].inspection.reference_file)
        )
        self._active_mesh_settings = (
            self._cluster_group_mesh_settings_for_states(cluster_states)
        )
        self._sync_mesh_controls_to_settings(self._active_mesh_settings)
        self._set_mesh_settings_confirmed_for_run(False)
        self.output_dir_edit.setText(
            str(
                self._initial_output_dir
                or suggest_output_dir(
                    path,
                    project_dir=self._project_dir,
                )
            )
        )
        self.output_basename_edit.setText(
            f"{path.name}_cluster_average_density"
        )
        self.smearing_factor_spin.setValue(0.001)
        self._populate_cluster_group_table()
        self.cluster_group_table.selectRow(0)
        self._set_active_cluster_group(
            cluster_states[0].key,
            preserve_zoom=False,
        )
        self._sync_pinned_geometric_tracking_control_to_settings(
            self._active_mesh_settings
        )
        self._schedule_cluster_view_cache_prewarm()
        self._append_status(
            f"Loaded cluster-folder input {path} with {len(cluster_states)} stoichiometries."
        )
        self._append_status(
            "Cluster-folder mode computes an averaged electron-density profile "
            "for each stoichiometry subfolder and lets you switch plots by row."
        )
        if single_atom_count > 0:
            self._append_status(
                f"{single_atom_count} stoichiometry folder"
                f"{'' if single_atom_count == 1 else 's'} contain only single-atom "
                "structures and will use direct Debye scattering instead of "
                "electron-density and Fourier-profile evaluation."
            )
        self.statusBar().showMessage(
            f"Loaded cluster folders from {path.name}"
        )
        return True

    def _apply_workspace_load_payload(
        self,
        payload: _ElectronDensityWorkspaceLoadPayload,
    ) -> None:
        path = Path(payload.selection_path).expanduser().resolve()
        self._clear_saved_output_history(announce=False)
        self._clear_cluster_group_states()
        if self._load_cluster_group_input(
            path,
            cluster_states=list(payload.cluster_states),
        ):
            self._restore_saved_output_history_if_available()
            return
        inspection = payload.inspection
        structure = payload.structure
        if inspection is None or structure is None:
            raise RuntimeError(
                "Workspace loader returned an incomplete structure payload."
            )
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
        self._sync_pinned_geometric_tracking_control_to_settings(
            self._active_mesh_settings
        )

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
        self._set_mesh_settings_confirmed_for_run(False)
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
        self._restore_saved_output_history_if_available()

    def _load_input_path(
        self,
        path: Path,
        *,
        asynchronous: bool = False,
        restore_saved_distribution_state: bool = False,
    ) -> None:
        resolved_path = Path(path).expanduser().resolve()
        if asynchronous:
            self._start_workspace_load(
                resolved_path,
                restore_saved_distribution_state=restore_saved_distribution_state,
                progress_message="Loading electron-density input...",
            )
            return
        self._apply_workspace_load_payload(
            _build_workspace_load_payload(resolved_path)
        )

    def _component_summary_path(self) -> Path | None:
        if self._distribution_root_dir is None:
            return None
        return (
            self._distribution_root_dir
            / "electron_density_mapping"
            / "born_approximation_component_summary.json"
        )

    def _workspace_state_path(self) -> Path | None:
        if self._distribution_root_dir is None:
            return None
        return (
            self._distribution_root_dir
            / "electron_density_mapping"
            / "workspace_state.json"
        )

    def _component_artifact_targets(self) -> tuple[Path, Path] | None:
        if self._distribution_root_dir is None:
            return None
        if self._use_predicted_structure_weights:
            return (
                self._distribution_root_dir
                / "scattering_components_predicted_structures",
                self._distribution_root_dir
                / "md_saxs_map_predicted_structures.json",
            )
        return (
            self._distribution_root_dir / "scattering_components",
            self._distribution_root_dir / "md_saxs_map.json",
        )

    @staticmethod
    def _component_profile_filename(structure: str, motif: str) -> str:
        safe_name = f"{structure}_{motif}".replace("/", "_")
        return f"{safe_name}.txt"

    @staticmethod
    def _serialize_structure(
        structure: ElectronDensityStructure,
    ) -> dict[str, object]:
        return {
            "file_path": str(structure.file_path),
            "display_label": str(structure.display_label),
            "structure_comment": str(structure.structure_comment),
            "coordinates": np.asarray(
                structure.coordinates, dtype=float
            ).tolist(),
            "centered_coordinates": np.asarray(
                structure.centered_coordinates,
                dtype=float,
            ).tolist(),
            "elements": list(structure.elements),
            "element_counts": dict(structure.element_counts),
            "atomic_numbers": np.asarray(
                structure.atomic_numbers,
                dtype=float,
            ).tolist(),
            "atomic_masses": np.asarray(
                structure.atomic_masses,
                dtype=float,
            ).tolist(),
            "center_of_mass": np.asarray(
                structure.center_of_mass,
                dtype=float,
            ).tolist(),
            "geometric_center": np.asarray(
                structure.geometric_center,
                dtype=float,
            ).tolist(),
            "reference_element": str(structure.reference_element),
            "reference_element_geometric_center": np.asarray(
                structure.reference_element_geometric_center,
                dtype=float,
            ).tolist(),
            "reference_element_offset_from_geometric_center": float(
                structure.reference_element_offset_from_geometric_center
            ),
            "active_center": np.asarray(
                structure.active_center,
                dtype=float,
            ).tolist(),
            "center_mode": str(structure.center_mode),
            "nearest_atom_index": int(structure.nearest_atom_index),
            "nearest_atom_distance": float(structure.nearest_atom_distance),
            "bonds": [list(bond) for bond in structure.bonds],
            "rmax": float(structure.rmax),
        }

    @staticmethod
    def _deserialize_structure(
        payload: dict[str, object]
    ) -> ElectronDensityStructure:
        return ElectronDensityStructure(
            file_path=Path(str(payload.get("file_path") or ""))
            .expanduser()
            .resolve(),
            display_label=str(payload.get("display_label") or ""),
            structure_comment=str(payload.get("structure_comment") or ""),
            coordinates=np.asarray(
                payload.get("coordinates") or [], dtype=float
            ),
            centered_coordinates=np.asarray(
                payload.get("centered_coordinates") or [],
                dtype=float,
            ),
            elements=tuple(
                str(value) for value in (payload.get("elements") or [])
            ),
            element_counts={
                str(key): int(value)
                for key, value in dict(
                    payload.get("element_counts") or {}
                ).items()
            },
            atomic_numbers=np.asarray(
                payload.get("atomic_numbers") or [],
                dtype=float,
            ),
            atomic_masses=np.asarray(
                payload.get("atomic_masses") or [],
                dtype=float,
            ),
            center_of_mass=np.asarray(
                payload.get("center_of_mass") or [0.0, 0.0, 0.0],
                dtype=float,
            ),
            geometric_center=np.asarray(
                payload.get("geometric_center") or [0.0, 0.0, 0.0],
                dtype=float,
            ),
            reference_element=str(payload.get("reference_element") or ""),
            reference_element_geometric_center=np.asarray(
                payload.get("reference_element_geometric_center")
                or [0.0, 0.0, 0.0],
                dtype=float,
            ),
            reference_element_offset_from_geometric_center=float(
                payload.get("reference_element_offset_from_geometric_center")
                or 0.0
            ),
            active_center=np.asarray(
                payload.get("active_center") or [0.0, 0.0, 0.0],
                dtype=float,
            ),
            center_mode=str(payload.get("center_mode") or "center_of_mass"),
            nearest_atom_index=int(payload.get("nearest_atom_index") or 0),
            nearest_atom_distance=float(
                payload.get("nearest_atom_distance") or 0.0
            ),
            bonds=tuple(
                (int(bond[0]), int(bond[1]))
                for bond in (payload.get("bonds") or [])
                if isinstance(bond, (list, tuple)) and len(bond) >= 2
            ),
            rmax=float(payload.get("rmax") or 0.0),
        )

    @staticmethod
    def _serialize_mesh_geometry(
        geometry: ElectronDensityMeshGeometry,
    ) -> dict[str, object]:
        return {
            "settings": geometry.settings.to_dict(),
            "domain_max_radius": float(geometry.domain_max_radius),
            "radial_edges": np.asarray(
                geometry.radial_edges, dtype=float
            ).tolist(),
            "theta_edges": np.asarray(
                geometry.theta_edges, dtype=float
            ).tolist(),
            "phi_edges": np.asarray(geometry.phi_edges, dtype=float).tolist(),
        }

    @staticmethod
    def _deserialize_mesh_geometry(
        payload: dict[str, object],
    ) -> ElectronDensityMeshGeometry:
        settings_payload = dict(payload.get("settings") or {})
        default_settings = ElectronDensityMeshSettings()
        return ElectronDensityMeshGeometry(
            settings=ElectronDensityMeshSettings(
                rstep=float(
                    settings_payload.get("rstep_a") or default_settings.rstep
                ),
                theta_divisions=int(
                    settings_payload.get("theta_divisions") or 120
                ),
                phi_divisions=int(settings_payload.get("phi_divisions") or 60),
                rmax=float(settings_payload.get("rmax_a") or 1.0),
                pin_contiguous_geometric_tracking=bool(
                    settings_payload.get(
                        "pin_contiguous_geometric_tracking",
                        default_settings.pin_contiguous_geometric_tracking,
                    )
                ),
            ),
            domain_max_radius=float(payload.get("domain_max_radius") or 0.0),
            radial_edges=np.asarray(
                payload.get("radial_edges") or [], dtype=float
            ),
            theta_edges=np.asarray(
                payload.get("theta_edges") or [], dtype=float
            ),
            phi_edges=np.asarray(payload.get("phi_edges") or [], dtype=float),
        )

    @staticmethod
    def _deserialize_mesh_settings(
        payload: dict[str, object],
    ) -> ElectronDensityMeshSettings:
        default_settings = ElectronDensityMeshSettings()
        return ElectronDensityMeshSettings(
            rstep=float(payload.get("rstep_a") or default_settings.rstep),
            theta_divisions=int(payload.get("theta_divisions") or 120),
            phi_divisions=int(payload.get("phi_divisions") or 60),
            rmax=float(payload.get("rmax_a") or 1.0),
            pin_contiguous_geometric_tracking=bool(
                payload.get(
                    "pin_contiguous_geometric_tracking",
                    default_settings.pin_contiguous_geometric_tracking,
                )
            ),
        ).normalized()

    @staticmethod
    def _deserialize_density_estimate(
        payload: dict[str, object],
    ) -> ContrastElectronDensityEstimate:
        reference_path = payload.get("reference_structure_file")
        reference_box_spans = payload.get("reference_box_spans")
        translated_center = payload.get("translated_volume_center")
        return ContrastElectronDensityEstimate(
            method=str(payload.get("method") or ""),
            label=str(payload.get("label") or ""),
            volume_a3=float(payload.get("volume_a3") or 0.0),
            total_electrons=float(payload.get("total_electrons") or 0.0),
            electron_density_e_per_a3=float(
                payload.get("electron_density_e_per_a3") or 0.0
            ),
            electron_density_e_per_cm3=float(
                payload.get("electron_density_e_per_cm3") or 0.0
            ),
            atom_count=(
                None
                if payload.get("atom_count") is None
                else int(payload.get("atom_count"))
            ),
            element_counts={
                str(key): int(value)
                for key, value in dict(
                    payload.get("element_counts") or {}
                ).items()
            },
            formula=(
                None
                if payload.get("formula") is None
                else str(payload.get("formula"))
            ),
            source_density_g_per_cm3=(
                None
                if payload.get("source_density_g_per_cm3") is None
                else float(payload.get("source_density_g_per_cm3"))
            ),
            reference_structure_file=(
                None
                if reference_path is None
                else Path(str(reference_path)).expanduser().resolve()
            ),
            reference_box_spans=(
                None
                if not isinstance(reference_box_spans, list)
                else tuple(float(value) for value in reference_box_spans[:3])
            ),
            translated_volume_center=(
                None
                if not isinstance(translated_center, list)
                else tuple(float(value) for value in translated_center[:3])
            ),
        )

    @staticmethod
    def _deserialize_debye_waller_pair_term(
        payload: dict[str, object],
    ) -> ElectronDensityDebyeWallerPairTerm:
        return ElectronDensityDebyeWallerPairTerm(
            scope=str(payload.get("scope") or ""),
            type_definition=str(payload.get("type_definition") or ""),
            pair_label_a=str(payload.get("pair_label_a") or ""),
            pair_label_b=str(payload.get("pair_label_b") or ""),
            pair_label=str(payload.get("pair_label") or ""),
            mean_distance_a=float(payload.get("mean_distance_a") or 0.0),
            sigma_a=float(payload.get("sigma_a") or 0.0),
            sigma_std_a=float(payload.get("sigma_std_a") or 0.0),
            sigma_squared_a2=float(payload.get("sigma_squared_a2") or 0.0),
            sigma_squared_std_a2=float(
                payload.get("sigma_squared_std_a2") or 0.0
            ),
            b_factor_a2=float(payload.get("b_factor_a2") or 0.0),
            b_factor_std_a2=float(payload.get("b_factor_std_a2") or 0.0),
        ).normalized()

    @staticmethod
    def _serialize_profile_result(
        result: ElectronDensityProfileResult,
    ) -> dict[str, object]:
        return {
            "structure": ElectronDensityMappingMainWindow._serialize_structure(
                result.structure
            ),
            "input_mode": str(result.input_mode),
            "source_files": [str(path) for path in result.source_files],
            "source_structure_count": int(result.source_structure_count),
            "averaging_mode": str(result.averaging_mode),
            "contiguous_frame_mode_requested": bool(
                result.contiguous_frame_mode_requested
            ),
            "contiguous_frame_mode_applied": bool(
                result.contiguous_frame_mode_applied
            ),
            "pinned_geometric_tracking_requested": bool(
                result.pinned_geometric_tracking_requested
            ),
            "pinned_geometric_tracking_applied": bool(
                result.pinned_geometric_tracking_applied
            ),
            "averaging_notes": [str(note) for note in result.averaging_notes],
            "contiguous_frame_sets": [
                frame_set.to_dict()
                for frame_set in result.contiguous_frame_sets
            ],
            "mesh_geometry": ElectronDensityMappingMainWindow._serialize_mesh_geometry(
                result.mesh_geometry
            ),
            "smearing_settings": result.smearing_settings.to_dict(),
            "radial_centers": np.asarray(
                result.radial_centers, dtype=float
            ).tolist(),
            "orientation_average_density": np.asarray(
                result.orientation_average_density,
                dtype=float,
            ).tolist(),
            "orientation_density_variance": np.asarray(
                result.orientation_density_variance,
                dtype=float,
            ).tolist(),
            "orientation_density_stddev": np.asarray(
                result.orientation_density_stddev,
                dtype=float,
            ).tolist(),
            "smeared_orientation_average_density": np.asarray(
                result.smeared_orientation_average_density,
                dtype=float,
            ).tolist(),
            "smeared_orientation_density_variance": np.asarray(
                result.smeared_orientation_density_variance,
                dtype=float,
            ).tolist(),
            "smeared_orientation_density_stddev": np.asarray(
                result.smeared_orientation_density_stddev,
                dtype=float,
            ).tolist(),
            "shell_volume_average_density": np.asarray(
                result.shell_volume_average_density,
                dtype=float,
            ).tolist(),
            "shell_electron_counts": np.asarray(
                result.shell_electron_counts,
                dtype=float,
            ).tolist(),
            "shell_volumes": np.asarray(
                result.shell_volumes,
                dtype=float,
            ).tolist(),
            "excluded_atom_count": int(result.excluded_atom_count),
            "excluded_electron_count": float(result.excluded_electron_count),
            "solvent_contrast": (
                None
                if result.solvent_contrast is None
                else result.solvent_contrast.to_dict()
            ),
        }

    @staticmethod
    def _deserialize_profile_result(
        payload: dict[str, object],
    ) -> ElectronDensityProfileResult:
        smearing_payload = dict(payload.get("smearing_settings") or {})
        pair_specific_terms = tuple(
            ElectronDensityMappingMainWindow._deserialize_debye_waller_pair_term(
                dict(entry)
            )
            for entry in (smearing_payload.get("pair_specific_terms") or [])
            if isinstance(entry, dict)
        )
        imported_pair_specific_terms = tuple(
            ElectronDensityMappingMainWindow._deserialize_debye_waller_pair_term(
                dict(entry)
            )
            for entry in (
                smearing_payload.get("imported_pair_specific_terms") or []
            )
            if isinstance(entry, dict)
        )
        solvent_payload = payload.get("solvent_contrast")
        solvent_contrast = None
        if isinstance(solvent_payload, dict):
            settings_payload = dict(solvent_payload.get("settings") or {})
            solvent_contrast = ElectronDensitySolventContrastResult(
                settings=ContrastSolventDensitySettings.from_values(
                    method=settings_payload.get("method"),
                    solvent_formula=settings_payload.get("solvent_formula"),
                    solvent_density_g_per_ml=settings_payload.get(
                        "solvent_density_g_per_ml"
                    ),
                    reference_structure_file=settings_payload.get(
                        "reference_structure_file"
                    ),
                    direct_electron_density_e_per_a3=settings_payload.get(
                        "direct_electron_density_e_per_a3"
                    ),
                ),
                solvent_name=str(solvent_payload.get("solvent_name") or ""),
                density_estimate=ElectronDensityMappingMainWindow._deserialize_density_estimate(
                    dict(solvent_payload.get("density_estimate") or {})
                ),
                solvent_density_e_per_a3=float(
                    solvent_payload.get("solvent_density_e_per_a3") or 0.0
                ),
                cutoff_radius_a=(
                    None
                    if solvent_payload.get("cutoff_radius_a") is None
                    else float(solvent_payload.get("cutoff_radius_a"))
                ),
                solvent_subtracted_smeared_density=np.asarray(
                    solvent_payload.get(
                        "solvent_subtracted_smeared_density_e_per_a3"
                    )
                    or [],
                    dtype=float,
                ),
            )
        orientation_average_density = np.asarray(
            payload.get("orientation_average_density") or [],
            dtype=float,
        )
        contiguous_frame_sets = tuple(
            ElectronDensityContiguousFrameSetSummary(
                series_label=str(entry.get("series_label") or "default"),
                frame_ids=tuple(
                    int(value) for value in (entry.get("frame_ids") or [])
                ),
                frame_labels=tuple(
                    str(value) for value in (entry.get("frame_labels") or [])
                ),
                file_paths=tuple(
                    Path(str(value)).expanduser().resolve()
                    for value in (entry.get("file_paths") or [])
                ),
            )
            for entry in (payload.get("contiguous_frame_sets") or [])
            if isinstance(entry, dict)
        )
        return ElectronDensityProfileResult(
            structure=ElectronDensityMappingMainWindow._deserialize_structure(
                dict(payload.get("structure") or {})
            ),
            input_mode=str(payload.get("input_mode") or "folder"),
            source_files=tuple(
                Path(str(value)).expanduser().resolve()
                for value in (payload.get("source_files") or [])
            ),
            source_structure_count=int(
                payload.get("source_structure_count") or 0
            ),
            member_summaries=tuple(),
            member_orientation_average_densities=(
                np.asarray(orientation_average_density, dtype=float).copy(),
            ),
            mesh_geometry=ElectronDensityMappingMainWindow._deserialize_mesh_geometry(
                dict(payload.get("mesh_geometry") or {})
            ),
            smearing_settings=ElectronDensitySmearingSettings(
                debye_waller_factor=float(
                    smearing_payload.get("debye_waller_factor_a2") or 0.0
                ),
                debye_waller_mode=str(
                    smearing_payload.get("debye_waller_mode") or "universal"
                ),
                pair_specific_terms=pair_specific_terms,
                imported_pair_specific_terms=imported_pair_specific_terms,
            ).normalized(),
            radial_centers=np.asarray(
                payload.get("radial_centers") or [],
                dtype=float,
            ),
            orientation_average_density=orientation_average_density,
            orientation_density_variance=np.asarray(
                payload.get("orientation_density_variance") or [],
                dtype=float,
            ),
            orientation_density_stddev=np.asarray(
                payload.get("orientation_density_stddev") or [],
                dtype=float,
            ),
            smeared_orientation_average_density=np.asarray(
                payload.get("smeared_orientation_average_density") or [],
                dtype=float,
            ),
            smeared_orientation_density_variance=np.asarray(
                payload.get("smeared_orientation_density_variance") or [],
                dtype=float,
            ),
            smeared_orientation_density_stddev=np.asarray(
                payload.get("smeared_orientation_density_stddev") or [],
                dtype=float,
            ),
            shell_volume_average_density=np.asarray(
                payload.get("shell_volume_average_density") or [],
                dtype=float,
            ),
            shell_electron_counts=np.asarray(
                payload.get("shell_electron_counts") or [],
                dtype=float,
            ),
            shell_volumes=np.asarray(
                payload.get("shell_volumes") or [],
                dtype=float,
            ),
            excluded_atom_count=int(payload.get("excluded_atom_count") or 0),
            excluded_electron_count=float(
                payload.get("excluded_electron_count") or 0.0
            ),
            averaging_mode=str(
                payload.get("averaging_mode") or "complete_average"
            ),
            contiguous_frame_mode_requested=bool(
                payload.get("contiguous_frame_mode_requested", False)
            ),
            contiguous_frame_mode_applied=bool(
                payload.get("contiguous_frame_mode_applied", False)
            ),
            pinned_geometric_tracking_requested=bool(
                payload.get("pinned_geometric_tracking_requested", False)
            ),
            pinned_geometric_tracking_applied=bool(
                payload.get("pinned_geometric_tracking_applied", False)
            ),
            averaging_notes=tuple(
                str(note) for note in (payload.get("averaging_notes") or [])
            ),
            contiguous_frame_sets=contiguous_frame_sets,
            solvent_contrast=solvent_contrast,
        )

    @staticmethod
    def _serialize_transform_result(
        result: ElectronDensityScatteringTransformResult,
    ) -> dict[str, object]:
        return {
            "settings": result.preview.settings.to_dict(),
            "q_values": np.asarray(result.q_values, dtype=float).tolist(),
            "scattering_amplitude": np.asarray(
                result.scattering_amplitude,
                dtype=float,
            ).tolist(),
            "intensity": np.asarray(result.intensity, dtype=float).tolist(),
        }

    @staticmethod
    def _deserialize_transform_result(
        payload: dict[str, object],
        profile_result: ElectronDensityProfileResult | None,
        *,
        single_atom_only: bool = False,
    ) -> ElectronDensityScatteringTransformResult | None:
        settings_payload = dict(payload.get("settings") or {})
        domain_mode = settings_payload.get("domain_mode")
        inferred_domain_mode = (
            "legacy" if domain_mode is None else str(domain_mode)
        )
        settings = ElectronDensityFourierTransformSettings(
            r_min=float(settings_payload.get("r_min_a") or 0.0),
            r_max=float(settings_payload.get("r_max_a") or 1.0),
            domain_mode=inferred_domain_mode,
            window_function=str(
                settings_payload.get("window_function") or "none"
            ),
            resampling_points=int(
                settings_payload.get("resampling_points") or 2048
            ),
            q_min=float(settings_payload.get("q_min_a_inverse") or 0.02),
            q_max=float(settings_payload.get("q_max_a_inverse") or 1.2),
            q_step=float(settings_payload.get("q_step_a_inverse") or 0.01),
            use_solvent_subtracted_profile=bool(
                settings_payload.get("use_solvent_subtracted_profile", True)
            ),
            log_q_axis=bool(settings_payload.get("log_q_axis", True)),
            log_intensity_axis=bool(
                settings_payload.get("log_intensity_axis", True)
            ),
        )
        if single_atom_only:
            preview = prepare_single_atom_debye_scattering_preview(settings)
        else:
            if profile_result is None:
                return None
            try:
                preview = prepare_electron_density_fourier_transform(
                    profile_result,
                    settings,
                )
            except Exception:
                return None
        return ElectronDensityScatteringTransformResult(
            preview=preview,
            q_values=np.asarray(payload.get("q_values") or [], dtype=float),
            scattering_amplitude=np.asarray(
                payload.get("scattering_amplitude") or [],
                dtype=float,
            ),
            intensity=np.asarray(payload.get("intensity") or [], dtype=float),
        )

    @staticmethod
    def _serialize_debye_scattering_result(
        result: ElectronDensityDebyeScatteringAverageResult,
    ) -> dict[str, object]:
        return {
            "q_values": np.asarray(result.q_values, dtype=float).tolist(),
            "mean_intensity": np.asarray(
                result.mean_intensity,
                dtype=float,
            ).tolist(),
            "std_intensity": np.asarray(
                result.std_intensity,
                dtype=float,
            ).tolist(),
            "se_intensity": np.asarray(
                result.se_intensity,
                dtype=float,
            ).tolist(),
            "source_files": [str(path) for path in result.source_files],
            "source_structure_count": int(result.source_structure_count),
            "unique_elements": [
                str(value) for value in result.unique_elements
            ],
            "notes": [str(note) for note in result.notes],
        }

    @staticmethod
    def _deserialize_debye_scattering_result(
        payload: dict[str, object],
    ) -> ElectronDensityDebyeScatteringAverageResult:
        return ElectronDensityDebyeScatteringAverageResult(
            q_values=np.asarray(payload.get("q_values") or [], dtype=float),
            mean_intensity=np.asarray(
                payload.get("mean_intensity") or [],
                dtype=float,
            ),
            std_intensity=np.asarray(
                payload.get("std_intensity") or [],
                dtype=float,
            ),
            se_intensity=np.asarray(
                payload.get("se_intensity") or [],
                dtype=float,
            ),
            source_files=tuple(
                Path(str(value)).expanduser().resolve()
                for value in (payload.get("source_files") or [])
            ),
            source_structure_count=int(
                payload.get("source_structure_count") or 0
            ),
            unique_elements=tuple(
                str(value) for value in (payload.get("unique_elements") or [])
            ),
            notes=tuple(str(value) for value in (payload.get("notes") or [])),
        )

    @staticmethod
    def _deserialize_fourier_settings(
        payload: dict[str, object],
    ) -> ElectronDensityFourierTransformSettings:
        domain_mode = payload.get("domain_mode")
        return ElectronDensityFourierTransformSettings(
            r_min=float(payload.get("r_min_a") or 0.0),
            r_max=float(payload.get("r_max_a") or 1.0),
            domain_mode=(
                "legacy" if domain_mode is None else str(domain_mode)
            ),
            window_function=str(payload.get("window_function") or "none"),
            resampling_points=int(payload.get("resampling_points") or 2048),
            q_min=float(payload.get("q_min_a_inverse") or 0.02),
            q_max=float(payload.get("q_max_a_inverse") or 1.2),
            q_step=float(payload.get("q_step_a_inverse") or 0.01),
            use_solvent_subtracted_profile=bool(
                payload.get("use_solvent_subtracted_profile", True)
            ),
            log_q_axis=bool(payload.get("log_q_axis", True)),
            log_intensity_axis=bool(payload.get("log_intensity_axis", True)),
        ).normalized()

    @staticmethod
    def _serialize_saved_output_entry(
        entry: _SavedOutputEntry,
    ) -> dict[str, object]:
        return {
            "entry_id": entry.entry_id,
            "created_at": entry.created_at,
            "entry_kind": entry.entry_kind,
            "input_path": (
                None if entry.input_path is None else str(entry.input_path)
            ),
            "output_basename": entry.output_basename,
            "preview_mode": bool(entry.preview_mode),
            "group_key": entry.group_key,
            "group_label": entry.group_label,
            "use_contiguous_frame_mode": bool(entry.use_contiguous_frame_mode),
            "profile_result": ElectronDensityMappingMainWindow._serialize_profile_result(
                entry.profile_result
            ),
            "fourier_settings": entry.fourier_settings.to_dict(),
            "transform_result": (
                None
                if entry.transform_result is None
                else ElectronDensityMappingMainWindow._serialize_transform_result(
                    entry.transform_result
                )
            ),
        }

    @staticmethod
    def _deserialize_saved_output_entry(
        payload: dict[str, object],
    ) -> _SavedOutputEntry | None:
        profile_payload = payload.get("profile_result")
        if not isinstance(profile_payload, dict):
            return None
        profile_result = (
            ElectronDensityMappingMainWindow._deserialize_profile_result(
                profile_payload
            )
        )
        fourier_settings = (
            ElectronDensityMappingMainWindow._deserialize_fourier_settings(
                dict(payload.get("fourier_settings") or {})
            )
        )
        transform_payload = payload.get("transform_result")
        transform_result = None
        if isinstance(transform_payload, dict):
            transform_result = (
                ElectronDensityMappingMainWindow._deserialize_transform_result(
                    transform_payload,
                    profile_result,
                    single_atom_only=False,
                )
            )
        input_path = payload.get("input_path")
        return _SavedOutputEntry(
            entry_id=str(payload.get("entry_id") or ""),
            created_at=str(payload.get("created_at") or ""),
            entry_kind=str(payload.get("entry_kind") or "density"),
            input_path=(
                None
                if input_path is None or not str(input_path).strip()
                else Path(str(input_path)).expanduser().resolve()
            ),
            output_basename=str(payload.get("output_basename") or ""),
            preview_mode=bool(payload.get("preview_mode", True)),
            group_key=(
                None
                if payload.get("group_key") is None
                else str(payload.get("group_key"))
            ),
            group_label=(
                None
                if payload.get("group_label") is None
                else str(payload.get("group_label"))
            ),
            use_contiguous_frame_mode=bool(
                payload.get("use_contiguous_frame_mode", True)
            ),
            profile_result=profile_result,
            fourier_settings=fourier_settings,
            transform_result=transform_result,
        )

    def _current_input_path(self) -> Path | None:
        text = self.input_path_edit.text().strip()
        if not text:
            return None
        try:
            return Path(text).expanduser().resolve()
        except Exception:
            return None

    def _preview_saved_output_history_path(self) -> Path | None:
        output_dir_text = self.output_dir_edit.text().strip()
        if not output_dir_text:
            return None
        try:
            output_dir = Path(output_dir_text).expanduser().resolve()
        except Exception:
            return None
        return output_dir / "electron_density_saved_output_history.json"

    def _distribution_saved_output_history_path(self) -> Path | None:
        if self._distribution_root_dir is None:
            return None
        return (
            self._distribution_root_dir
            / "electron_density_mapping"
            / "saved_output_history.json"
        )

    def _saved_output_history_write_path(self) -> Path | None:
        if not self._preview_mode:
            history_path = self._distribution_saved_output_history_path()
            if history_path is not None:
                return history_path
        return self._preview_saved_output_history_path()

    def _saved_output_history_restore_candidates(self) -> list[Path]:
        candidates: list[Path] = []
        distribution_history = self._distribution_saved_output_history_path()
        preview_history = self._preview_saved_output_history_path()
        if not self._preview_mode and distribution_history is not None:
            candidates.append(distribution_history)
        if preview_history is not None and preview_history not in candidates:
            candidates.append(preview_history)
        return candidates

    def _write_saved_output_history(self) -> None:
        history_path = self._saved_output_history_write_path()
        current_input_path = self._current_input_path()
        if history_path is None or current_input_path is None:
            self._update_output_history_summary()
            return
        payload = {
            "schema_version": 1,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "preview_mode": self._preview_mode,
            "distribution_id": self._distribution_id,
            "input_path": str(current_input_path),
            "entries": [
                self._serialize_saved_output_entry(entry)
                for entry in self._saved_output_entries
            ],
        }
        try:
            history_path.parent.mkdir(parents=True, exist_ok=True)
            history_path.write_text(
                json.dumps(payload, indent=2) + "\n",
                encoding="utf-8",
            )
        except Exception as exc:
            self._append_status(
                f"Could not write saved output history to {history_path}: {exc}"
            )
        self._update_output_history_summary()

    def _restore_saved_output_history_if_available(self) -> None:
        current_input_path = self._current_input_path()
        if current_input_path is None:
            return
        for history_path in self._saved_output_history_restore_candidates():
            if not history_path.is_file():
                continue
            try:
                payload = json.loads(history_path.read_text(encoding="utf-8"))
            except Exception as exc:
                self._append_status(
                    f"Could not read saved output history from {history_path}: {exc}"
                )
                continue
            saved_input_path = str(payload.get("input_path") or "").strip()
            if saved_input_path:
                try:
                    resolved_saved_input = (
                        Path(saved_input_path).expanduser().resolve()
                    )
                except Exception:
                    resolved_saved_input = None
                if (
                    resolved_saved_input is not None
                    and resolved_saved_input != current_input_path
                ):
                    continue
            entries = [
                entry
                for entry in (
                    self._deserialize_saved_output_entry(dict(entry_payload))
                    for entry_payload in (payload.get("entries") or [])
                    if isinstance(entry_payload, dict)
                )
                if entry is not None
            ]
            if not entries:
                continue
            self._restoring_saved_output_history = True
            try:
                self._saved_output_entries = entries
                self._populate_output_history_table()
            finally:
                self._restoring_saved_output_history = False
            self._append_status(
                f"Restored {len(entries)} saved output set"
                f"{'' if len(entries) == 1 else 's'} from {history_path}."
            )
            return

    @staticmethod
    def _preview_for_saved_output_entry(
        entry: _SavedOutputEntry,
    ) -> ElectronDensityFourierTransformPreview | None:
        if entry.transform_result is not None:
            return entry.transform_result.preview
        try:
            return prepare_electron_density_fourier_transform(
                entry.profile_result,
                entry.fourier_settings,
            )
        except Exception:
            return None

    @staticmethod
    def _constrain_fourier_settings_for_result(
        result: ElectronDensityProfileResult,
        settings: ElectronDensityFourierTransformSettings,
        *,
        prefer_solvent_cutoff: bool = False,
    ) -> ElectronDensityFourierTransformSettings:
        normalized = settings.normalized()
        radial_values = np.asarray(result.radial_centers, dtype=float)
        if radial_values.size < 1:
            return normalized
        domain_min = 0.0
        domain_max = float(radial_values[-1])
        if domain_max <= domain_min:
            return normalized
        if normalized.domain_mode == "mirrored":
            minimum_radius = max(min(domain_max, 1.0e-3), 1.0e-6)
            constrained_r_max = float(
                np.clip(
                    float(normalized.r_max),
                    minimum_radius,
                    domain_max,
                )
            )
            if (
                prefer_solvent_cutoff
                and result.solvent_contrast is not None
                and result.solvent_contrast.cutoff_radius_a is not None
            ):
                constrained_r_max = float(
                    np.clip(
                        float(result.solvent_contrast.cutoff_radius_a),
                        minimum_radius,
                        domain_max,
                    )
                )
            constrained_r_min = -constrained_r_max
        else:
            minimum_span = max(min(domain_max - domain_min, 1.0e-3), 1.0e-6)
            constrained_r_min = float(
                np.clip(
                    float(normalized.r_min),
                    domain_min,
                    max(domain_max - minimum_span, domain_min),
                )
            )
            constrained_r_max = float(
                np.clip(
                    float(normalized.r_max),
                    constrained_r_min + minimum_span,
                    domain_max,
                )
            )
            if (
                prefer_solvent_cutoff
                and result.solvent_contrast is not None
                and result.solvent_contrast.cutoff_radius_a is not None
            ):
                constrained_r_max = float(
                    np.clip(
                        float(result.solvent_contrast.cutoff_radius_a),
                        constrained_r_min + minimum_span,
                        domain_max,
                    )
                )
        return ElectronDensityFourierTransformSettings(
            r_min=constrained_r_min,
            r_max=constrained_r_max,
            domain_mode=normalized.domain_mode,
            window_function=normalized.window_function,
            resampling_points=normalized.resampling_points,
            q_min=normalized.q_min,
            q_max=normalized.q_max,
            q_step=normalized.q_step,
            use_solvent_subtracted_profile=(
                normalized.use_solvent_subtracted_profile
            ),
            log_q_axis=normalized.log_q_axis,
            log_intensity_axis=normalized.log_intensity_axis,
        ).normalized()

    def _capture_saved_output_entry(
        self,
        entry_kind: str,
        *,
        group_state: _ClusterDensityGroupState | None = None,
        profile_result: ElectronDensityProfileResult | None = None,
        transform_result: (
            ElectronDensityScatteringTransformResult | None
        ) = None,
    ) -> None:
        if self._restoring_saved_output_history:
            return
        result = profile_result
        if result is None and group_state is not None:
            result = group_state.profile_result
        if result is None:
            result = self._profile_result
        if result is None:
            return
        current_transform = transform_result
        if current_transform is None and group_state is not None:
            current_transform = group_state.transform_result
        elif current_transform is None:
            current_transform = self._fourier_result
        if current_transform is not None:
            fourier_settings = current_transform.preview.settings
        else:
            if group_state is not None:
                fourier_settings = self._cluster_state_fourier_settings(
                    group_state
                )
            else:
                try:
                    fourier_settings = self._fourier_settings_from_controls()
                except Exception:
                    fourier_settings = self._active_fourier_settings
                fourier_settings = self._constrain_fourier_settings_for_result(
                    result,
                    fourier_settings,
                )
        snapshot = _SavedOutputEntry(
            entry_id=datetime.now().strftime("%Y%m%dT%H%M%S%f"),
            created_at=datetime.now().isoformat(timespec="seconds"),
            entry_kind=str(entry_kind).strip() or "density",
            input_path=self._current_input_path(),
            output_basename=self._output_basename(),
            preview_mode=self._preview_mode,
            group_key=None if group_state is None else group_state.key,
            group_label=(
                None if group_state is None else group_state.display_name
            ),
            use_contiguous_frame_mode=self._use_contiguous_frame_mode(),
            profile_result=result,
            fourier_settings=fourier_settings,
            transform_result=current_transform,
        )
        self._saved_output_entries.append(snapshot)
        self._populate_output_history_table()
        self._write_saved_output_history()
        self._sync_workspace_state()

    def _restore_contrast_controls_from_result(
        self,
        result: ElectronDensityProfileResult,
    ) -> None:
        contrast = result.solvent_contrast
        if contrast is None:
            self._active_contrast_settings = None
            self._active_contrast_name = None
            self._refresh_contrast_display()
            return
        settings = contrast.settings
        self.solvent_method_combo.blockSignals(True)
        try:
            method_index = max(
                self.solvent_method_combo.findData(settings.method),
                0,
            )
            self.solvent_method_combo.setCurrentIndex(method_index)
        finally:
            self.solvent_method_combo.blockSignals(False)
        self._sync_density_method_controls()
        self.solvent_preset_combo.blockSignals(True)
        try:
            if settings.method == CONTRAST_SOLVENT_METHOD_NEAT:
                preset_index = max(
                    self.solvent_preset_combo.findData(contrast.solvent_name),
                    0,
                )
                self.solvent_preset_combo.setCurrentIndex(preset_index)
            else:
                self.solvent_preset_combo.setCurrentIndex(0)
        finally:
            self.solvent_preset_combo.blockSignals(False)
        self.solvent_formula_edit.setText(settings.solvent_formula or "")
        self.solvent_density_spin.setValue(
            float(settings.solvent_density_g_per_ml or 0.0)
        )
        self.reference_solvent_file_edit.setText(
            ""
            if settings.reference_structure_file is None
            else str(settings.reference_structure_file)
        )
        self.direct_density_spin.setValue(
            float(settings.direct_electron_density_e_per_a3 or 0.0)
        )
        self._active_contrast_settings = settings
        self._active_contrast_name = (
            contrast.solvent_name or contrast.legend_label
        )
        self._refresh_contrast_display()

    def _restore_saved_output_entry(
        self,
        entry: _SavedOutputEntry,
        *,
        announce: bool,
    ) -> bool:
        target_state = None
        row_index = -1
        if entry.group_key is not None and self._cluster_group_states:
            for index, state in enumerate(self._cluster_group_states):
                if state.key == entry.group_key:
                    target_state = state
                    row_index = index
                    break
            if target_state is None:
                self._show_error(
                    "Saved Output Unavailable",
                    "This saved output references a stoichiometry that is not "
                    "available in the current input.",
                )
                return False

        self._restoring_saved_output_history = True
        try:
            if target_state is not None:
                target_state.profile_result = entry.profile_result
                target_state.transform_result = entry.transform_result
                target_state.debye_scattering_result = None
                self._sync_cluster_state_solvent_metadata(target_state)
                self._populate_cluster_group_table()
                if row_index >= 0:
                    self.cluster_group_table.blockSignals(True)
                    try:
                        self.cluster_group_table.selectRow(row_index)
                    finally:
                        self.cluster_group_table.blockSignals(False)
                self._set_active_cluster_group(target_state.key)
            else:
                self._profile_result = entry.profile_result
                self._fourier_result = entry.transform_result
                self._debye_scattering_result = None
                self._structure = entry.profile_result.structure
                self._active_mesh_settings = (
                    entry.profile_result.mesh_geometry.settings
                )
                self._active_mesh_geometry = entry.profile_result.mesh_geometry
                self.reference_file_value.setText(
                    str(self._structure.file_path)
                )
                self._sync_reference_element_controls()
                self._refresh_center_display()
                self._refresh_structure_summary()
                self._refresh_active_mesh_display()
                self._refresh_mesh_notice()
                self.structure_viewer.set_structure(
                    self._structure,
                    mesh_geometry=self._active_mesh_geometry,
                    reset_view=True,
                )

            self.contiguous_frame_mode_checkbox.setChecked(
                bool(entry.use_contiguous_frame_mode)
            )

            self.smearing_factor_spin.blockSignals(True)
            try:
                self.smearing_factor_spin.setValue(
                    float(
                        entry.profile_result.smearing_settings.debye_waller_factor
                    )
                )
            finally:
                self.smearing_factor_spin.blockSignals(False)
            self._active_smearing_settings = (
                entry.profile_result.smearing_settings
            )
            self._refresh_smearing_display()

            mesh_settings = entry.profile_result.mesh_geometry.settings
            for widget, value in (
                (self.rstep_spin, float(mesh_settings.rstep)),
                (
                    self.theta_divisions_spin,
                    int(mesh_settings.theta_divisions),
                ),
                (self.phi_divisions_spin, int(mesh_settings.phi_divisions)),
                (self.rmax_spin, float(mesh_settings.rmax)),
            ):
                widget.blockSignals(True)
                try:
                    widget.setValue(value)
                finally:
                    widget.blockSignals(False)
            self._active_mesh_settings = mesh_settings
            self._active_mesh_geometry = entry.profile_result.mesh_geometry
            self._sync_pinned_geometric_tracking_control_to_settings(
                mesh_settings
            )
            self._refresh_active_mesh_display()
            self._refresh_mesh_notice()
            self._set_mesh_settings_confirmed_for_run(True)

            self._restore_contrast_controls_from_result(entry.profile_result)

            self._profile_result = entry.profile_result
            self._fourier_result = entry.transform_result
            self._sync_fourier_controls_to_domain(reset_bounds=False)
            self._sync_fourier_controls_to_settings(entry.fourier_settings)
            self._refresh_profile_plots()
            self._refresh_contrast_display()
            self._refresh_fourier_preview_from_controls(
                clear_transform=entry.transform_result is None
            )
            if entry.transform_result is not None:
                self._fourier_result = entry.transform_result
            self._refresh_scattering_plot()
            self._update_push_to_model_state()
            self._close_debye_scattering_compare_dialog()
            self._refresh_debye_scattering_group()
        finally:
            self._restoring_saved_output_history = False

        if announce:
            self._append_status(
                "Loaded saved output: "
                + self._saved_output_entry_heading(entry)
                + "."
            )
            self.statusBar().showMessage("Loaded saved output")
        return True

    @Slot()
    def _load_selected_output_history_entry(self) -> None:
        selected_entries = self._selected_output_history_entries()
        if len(selected_entries) != 1:
            self._show_error(
                "Select One Saved Output",
                "Select exactly one saved output row to load it into the main pane.",
            )
            return
        self._restore_saved_output_entry(selected_entries[0], announce=True)

    @Slot()
    def _compare_selected_output_history_entries(self) -> None:
        selected_entries = self._selected_output_history_entries()
        if not selected_entries:
            self._show_error(
                "No Saved Outputs Selected",
                "Select one or more saved output rows before opening the comparison window.",
            )
            return
        if self._output_history_compare_dialog is not None:
            self._output_history_compare_dialog.close()
        dialog = _SavedOutputComparisonDialog(
            selected_entries,
            show_variance_shading=self.show_variance_checkbox.isChecked(),
            parent=self,
        )
        dialog.finished.connect(
            lambda _result: setattr(
                self,
                "_output_history_compare_dialog",
                None,
            )
        )
        self._output_history_compare_dialog = dialog
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _close_debye_scattering_compare_dialog(self) -> None:
        dialog = self._debye_scattering_compare_dialog
        if dialog is None:
            return
        self._debye_scattering_compare_dialog = None
        dialog.close()

    @staticmethod
    def _debye_result_matches_transform(
        transform_result: ElectronDensityScatteringTransformResult | None,
        debye_result: ElectronDensityDebyeScatteringAverageResult | None,
    ) -> bool:
        if transform_result is None or debye_result is None:
            return False
        born_q = np.asarray(transform_result.q_values, dtype=float)
        debye_q = np.asarray(debye_result.q_values, dtype=float)
        return bool(
            born_q.shape == debye_q.shape
            and born_q.size > 0
            and np.allclose(
                born_q,
                debye_q,
                rtol=1.0e-9,
                atol=1.0e-12,
            )
        )

    @staticmethod
    def _is_density_fourier_transform(
        transform_result: ElectronDensityScatteringTransformResult | None,
    ) -> bool:
        return bool(
            transform_result is not None
            and transform_result.preview.source_mode == "density_fourier"
        )

    @staticmethod
    def _debye_comparison_info_text(
        label: str,
        born_result: ElectronDensityScatteringTransformResult,
        debye_result: ElectronDensityDebyeScatteringAverageResult,
    ) -> str:
        q_values = np.asarray(born_result.q_values, dtype=float)
        q_text = "q grid unavailable"
        if q_values.size > 0:
            q_text = (
                f"q={float(q_values[0]):.4f}"
                f"–{float(q_values[-1]):.4f} Å⁻¹ "
                f"({len(q_values)} points)"
            )
        notes = " ".join(
            str(note).strip()
            for note in debye_result.notes
            if str(note).strip()
        )
        base = (
            f"{label}: {debye_result.source_structure_count} structure"
            f"{'' if debye_result.source_structure_count == 1 else 's'}; "
            f"{q_text}."
        )
        return base if not notes else base + " " + notes

    def _debye_comparison_entries(self) -> list[_DebyeComparisonEntry]:
        if self._cluster_group_states:
            entries: list[_DebyeComparisonEntry] = []
            for state in self._cluster_group_states:
                born_result = state.transform_result
                debye_result = state.debye_scattering_result
                if (
                    state.single_atom_only
                    or not self._is_density_fourier_transform(born_result)
                    or not self._debye_result_matches_transform(
                        born_result,
                        debye_result,
                    )
                ):
                    continue
                entries.append(
                    _DebyeComparisonEntry(
                        entry_id=state.key,
                        label=state.display_name,
                        color=state.trace_color,
                        born_result=born_result,
                        debye_result=debye_result,
                        info_text=self._debye_comparison_info_text(
                            state.display_name,
                            born_result,
                            debye_result,
                        ),
                    )
                )
            return entries

        born_result = self._fourier_result
        debye_result = self._debye_scattering_result
        if not self._debye_result_matches_transform(born_result, debye_result):
            return []
        if not self._is_density_fourier_transform(born_result):
            return []
        current_input = self._current_input_path()
        label = (
            current_input.name
            if current_input is not None
            else self._output_basename()
        )
        return [
            _DebyeComparisonEntry(
                entry_id="active",
                label=label,
                color="#b45309",
                born_result=born_result,
                debye_result=debye_result,
                info_text=self._debye_comparison_info_text(
                    label,
                    born_result,
                    debye_result,
                ),
            )
        ]

    def _refresh_debye_scattering_group(self) -> None:
        if not hasattr(self, "debye_scattering_status_label"):
            return
        running = self._is_calculation_running()
        has_clusters = bool(self._cluster_group_states)
        self.apply_debye_to_all_button.setVisible(has_clusters)
        self.apply_debye_to_all_button.setEnabled(has_clusters and not running)

        comparison_entries = self._debye_comparison_entries()
        self.open_debye_scattering_compare_button.setEnabled(
            (not running) and bool(comparison_entries)
        )

        if has_clusters:
            (
                target_states,
                scope_label,
                _selected_keys,
            ) = self._batch_target_cluster_group_states(
                apply_to_all=bool(self.apply_debye_to_all_button.isChecked())
            )
            eligible_states = [
                state
                for state in target_states
                if (
                    not state.single_atom_only
                    and self._is_density_fourier_transform(
                        state.transform_result
                    )
                )
            ]
            ready_count = sum(
                1
                for state in eligible_states
                if self._debye_result_matches_transform(
                    state.transform_result,
                    state.debye_scattering_result,
                )
            )
            stale_count = sum(
                1
                for state in eligible_states
                if (
                    state.debye_scattering_result is not None
                    and not self._debye_result_matches_transform(
                        state.transform_result,
                        state.debye_scattering_result,
                    )
                )
            )
            skipped_single_atom = sum(
                1 for state in target_states if state.single_atom_only
            )
            skipped_pending = max(
                len(target_states)
                - len(eligible_states)
                - skipped_single_atom,
                0,
            )
            self.calculate_debye_scattering_button.setEnabled(
                (not running) and bool(eligible_states)
            )
            if not target_states:
                self.debye_scattering_status_label.setText(
                    "Select a stoichiometry row with a Born-approximation "
                    "Fourier trace to compute a matching Debye average."
                )
                return
            if eligible_states:
                message = (
                    f"{ready_count}/{len(eligible_states)} target row"
                    f"{'' if len(eligible_states) == 1 else 's'} across "
                    f"{scope_label} already have Debye averages on their "
                    "Born q-grid."
                )
                if stale_count > 0:
                    message += (
                        f" {stale_count} row"
                        f"{'' if stale_count == 1 else 's'} need recalculation "
                        "because the Born q-grid changed."
                    )
                if comparison_entries:
                    message += (
                        f" {len(comparison_entries)} comparison pair"
                        f"{'' if len(comparison_entries) == 1 else 's'} are "
                        "ready for overlay."
                    )
                else:
                    message += " Compute a Debye average to unlock the comparison plot."
                self.debye_scattering_status_label.setText(message)
                return
            if skipped_single_atom > 0 and skipped_pending == 0:
                self.debye_scattering_status_label.setText(
                    "The current target rows already use direct Debye "
                    "scattering only, so a separate Born-vs-Debye "
                    "comparison trace is not needed."
                )
                return
            self.debye_scattering_status_label.setText(
                "Evaluate the Born-approximation Fourier transform for the "
                f"target rows in {scope_label} first. Debye averages will "
                "reuse that exact q-grid."
            )
            return

        born_result = self._fourier_result
        has_density_born = self._is_density_fourier_transform(born_result)
        self.calculate_debye_scattering_button.setEnabled(
            (not running) and self._inspection is not None and has_density_born
        )
        if self._inspection is None:
            self.debye_scattering_status_label.setText(
                "Load a structure or folder input before computing Debye scattering."
            )
            return
        if not has_density_born:
            self.debye_scattering_status_label.setText(
                "Evaluate the Born-approximation Fourier transform first. "
                "The Debye average will reuse that q-grid exactly."
            )
            return
        q_values = np.asarray(born_result.q_values, dtype=float)
        ready = self._debye_result_matches_transform(
            born_result,
            self._debye_scattering_result,
        )
        stale = self._debye_scattering_result is not None and not ready
        q_text = (
            "q grid unavailable"
            if q_values.size == 0
            else (
                f"q={float(q_values[0]):.4f}"
                f"–{float(q_values[-1]):.4f} Å⁻¹ "
                f"({len(q_values)} points)"
            )
        )
        message = (
            f"Ready to compute a Debye average across "
            f"{self._inspection.total_files} structure"
            f"{'' if self._inspection.total_files == 1 else 's'} on the "
            f"active Born q-grid ({q_text})."
        )
        if ready:
            message = (
                f"Debye scattering average ready for the active Born trace "
                f"({q_text}). Open Comparison Plot to overlay the pair."
            )
        elif stale:
            message += " A previous Debye average exists but no longer matches the current Born q-grid."
        self.debye_scattering_status_label.setText(message)

    @Slot()
    def _calculate_debye_scattering_action(self) -> None:
        if self._is_calculation_running():
            self._show_error(
                "Calculation Running",
                "Wait for the active electron-density calculation to finish "
                "before computing Debye scattering.",
            )
            return
        if self._cluster_group_states:
            self._calculate_debye_scattering_for_target_clusters(
                apply_to_all=bool(self.apply_debye_to_all_button.isChecked())
            )
            return
        if self._inspection is None:
            self._show_error(
                "No Structure Loaded",
                "Load a structure or folder input before computing Debye scattering.",
            )
            return
        if not self._is_density_fourier_transform(self._fourier_result):
            self._show_error(
                "Born Transform Required",
                "Evaluate the Born-approximation Fourier transform before "
                "computing the Debye comparison trace.",
            )
            return
        self._close_debye_scattering_compare_dialog()
        progress_total = self._debye_scattering_progress_total_for_inspection(
            self._inspection
        )
        self._set_calculation_running(True)
        self.stop_calculation_button.setEnabled(False)
        self._begin_debye_scattering_progress(
            total=progress_total,
            message="Preparing Debye scattering average calculation...",
        )
        result: ElectronDensityDebyeScatteringAverageResult | None = None
        error_message: str | None = None
        try:
            result = compute_average_debye_scattering_profile_for_input(
                self._inspection,
                q_values=np.asarray(
                    self._fourier_result.q_values, dtype=float
                ),
                progress_callback=self._update_debye_scattering_progress,
            )
        except Exception as exc:
            error_message = str(exc)
        finally:
            self._set_calculation_running(False)
            self.stop_calculation_button.setEnabled(False)
            self._reset_debye_scattering_progress()
        if error_message is not None:
            self._show_error("Debye Scattering Error", error_message)
            return
        if result is None:
            self._show_error(
                "Debye Scattering Error",
                "Debye scattering calculation finished without a result.",
            )
            return
        self._debye_scattering_result = result
        self._refresh_debye_scattering_group()
        q_values = np.asarray(result.q_values, dtype=float)
        self._append_status(
            "Computed Debye scattering average on the active Born q-grid: "
            f"{result.source_structure_count} structure"
            f"{'' if result.source_structure_count == 1 else 's'}, "
            f"{len(q_values)} q points."
        )
        for note in result.notes:
            self._append_status(note)
        self.statusBar().showMessage("Debye scattering average ready")
        self._sync_workspace_state()

    def _calculate_debye_scattering_for_target_clusters(
        self,
        *,
        apply_to_all: bool,
    ) -> None:
        if not self._cluster_group_states:
            self._show_error(
                "No Stoichiometry Table",
                "Load a cluster-folder input before computing Debye "
                "comparison traces across stoichiometries.",
            )
            return
        (
            target_states,
            scope_label,
            selected_keys,
        ) = self._batch_target_cluster_group_states(apply_to_all=apply_to_all)
        updated_count = 0
        skipped_pending = 0
        skipped_single_atom = 0
        failures: list[str] = []
        eligible_states = [
            state
            for state in target_states
            if (
                not state.single_atom_only
                and self._is_density_fourier_transform(state.transform_result)
            )
        ]
        self._close_debye_scattering_compare_dialog()
        progress_total = sum(
            self._debye_scattering_progress_total_for_inspection(
                state.inspection
            )
            for state in eligible_states
        )
        if progress_total > 0:
            skipped_count = len(target_states) - len(eligible_states)
            progress_message = (
                "Preparing Debye scattering averages across "
                f"{len(eligible_states)} target row"
                f"{'' if len(eligible_states) == 1 else 's'} in "
                f"{scope_label}."
            )
            if skipped_count > 0:
                progress_message += (
                    f" {skipped_count} row"
                    f"{'' if skipped_count == 1 else 's'} will be skipped."
                )
            self._set_calculation_running(True)
            self.stop_calculation_button.setEnabled(False)
            self._begin_debye_scattering_progress(
                total=progress_total,
                message=progress_message,
            )
        progress_offset = 0
        eligible_index = 0
        try:
            for state in target_states:
                if state.single_atom_only:
                    skipped_single_atom += 1
                    state.debye_scattering_result = None
                    continue
                born_result = state.transform_result
                if not self._is_density_fourier_transform(born_result):
                    skipped_pending += 1
                    state.debye_scattering_result = None
                    continue
                eligible_index += 1
                state_progress_total = (
                    self._debye_scattering_progress_total_for_inspection(
                        state.inspection
                    )
                )

                def emit_state_progress(
                    current: int,
                    total: int,
                    message: str,
                    *,
                    _state=state,
                    _offset=progress_offset,
                    _scope_label=scope_label,
                    _scope_index=eligible_index,
                    _scope_total=len(eligible_states),
                ) -> None:
                    self._update_debye_scattering_progress(
                        _offset
                        + min(max(int(current), 0), max(int(total), 1)),
                        progress_total,
                        "Debye "
                        f"{_scope_index}/{_scope_total} "
                        f"[{_state.display_name}] in {_scope_label}: "
                        f"{str(message).strip()}",
                    )

                try:
                    result = (
                        compute_average_debye_scattering_profile_for_input(
                            state.inspection,
                            q_values=np.asarray(
                                born_result.q_values,
                                dtype=float,
                            ),
                            progress_callback=emit_state_progress,
                        )
                    )
                except Exception as exc:
                    failures.append(f"{state.display_name}: {exc}")
                    progress_offset += state_progress_total
                    if progress_total > 0:
                        self._update_debye_scattering_progress(
                            progress_offset,
                            progress_total,
                            f"Debye {eligible_index}/{len(eligible_states)} "
                            f"[{state.display_name}] in {scope_label} failed.",
                        )
                    continue
                state.debye_scattering_result = result
                if state.key == self._selected_cluster_group_key:
                    self._debye_scattering_result = result
                updated_count += 1
                progress_offset += state_progress_total
        finally:
            if progress_total > 0:
                self._set_calculation_running(False)
                self.stop_calculation_button.setEnabled(False)
                self._reset_debye_scattering_progress()
        if updated_count <= 0:
            self._refresh_debye_scattering_group()
            if failures:
                self._show_error(
                    "Debye Scattering Error",
                    "\n".join(failures[:6]),
                )
                return
            if skipped_single_atom > 0 and skipped_pending == 0:
                self._show_error(
                    "No Debye Targets Updated",
                    "The selected target rows already use direct Debye "
                    "scattering only, so a separate Born-vs-Debye "
                    "comparison trace is not needed.",
                )
                return
            self._show_error(
                "Born Transform Required",
                "Evaluate the Born-approximation Fourier transform for the "
                "target rows before computing Debye comparison traces.",
            )
            return
        self._refresh_cluster_views_after_batch_update(
            target_states,
            selected_keys=selected_keys,
        )
        summary_parts = [
            f"computed {updated_count} Debye scattering average"
            f"{'' if updated_count == 1 else 's'}"
        ]
        if skipped_pending > 0:
            summary_parts.append(
                f"skipped {skipped_pending} row"
                f"{'' if skipped_pending == 1 else 's'} without a Born trace"
            )
        if skipped_single_atom > 0:
            summary_parts.append(
                f"skipped {skipped_single_atom} direct-Debye row"
                f"{'' if skipped_single_atom == 1 else 's'}"
            )
        if failures:
            summary_parts.append(
                f"{len(failures)} row"
                f"{'' if len(failures) == 1 else 's'} failed"
            )
        self._append_status(
            "Debye scattering batch update: "
            + "; ".join(summary_parts)
            + f" across {scope_label}. Each trace reused the q-grid from its "
            "matching Born-approximation transform."
        )
        for failure in failures:
            self._append_status(f"Debye scattering warning: {failure}")
        self.statusBar().showMessage("Debye scattering averages ready")
        self._sync_workspace_state()

    @Slot()
    def _open_debye_scattering_comparison_plot(self) -> None:
        entries = self._debye_comparison_entries()
        if not entries:
            self._show_error(
                "No Debye Comparison Traces",
                "Compute at least one matching Born and Debye trace before "
                "opening the comparison plot.",
            )
            return
        self._close_debye_scattering_compare_dialog()
        dialog = _DebyeScatteringComparisonDialog(entries, parent=self)
        dialog.finished.connect(
            lambda _result: setattr(
                self,
                "_debye_scattering_compare_dialog",
                None,
            )
        )
        self._debye_scattering_compare_dialog = dialog
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _build_component_summary_payload(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "distribution_id": self._distribution_id,
            "project_dir": (
                None if self._project_dir is None else str(self._project_dir)
            ),
            "input_path": self.input_path_edit.text().strip() or None,
            "output_dir": self.output_dir_edit.text().strip() or None,
            "output_basename": self.output_basename_edit.text().strip()
            or None,
            "use_contiguous_frame_mode": self._use_contiguous_frame_mode(),
            "auto_save_smearing_outputs": (
                self._auto_save_smearing_outputs_enabled()
            ),
            "active_group_key": self._selected_cluster_group_key,
            "show_all_cluster_transforms": bool(
                self.show_all_cluster_transforms_checkbox.isChecked()
            ),
            "manual_mode_enabled": bool(self.manual_mode_checkbox.isChecked()),
            "manual_mesh_lock_settings": (
                None
                if self._manual_mesh_lock_settings is None
                else self._manual_mesh_lock_settings.to_dict()
            ),
            "active_mesh_settings": self._active_mesh_settings.to_dict(),
            "cluster_groups": [
                {
                    "key": state.key,
                    "display_name": state.display_name,
                    "structure_name": state.structure_name,
                    "motif_name": state.motif_name,
                    "source_dir": str(state.source_dir),
                    "single_atom_only": bool(state.single_atom_only),
                    "trace_color": state.trace_color,
                    "average_atom_count": float(state.average_atom_count),
                    "solvent_density_e_per_a3": state.solvent_density_e_per_a3,
                    "solvent_cutoff_radius_a": state.solvent_cutoff_radius_a,
                    "reference_structure": self._serialize_structure(
                        state.reference_structure
                    ),
                    "fourier_settings": (
                        None
                        if state.fourier_settings is None
                        else state.fourier_settings.to_dict()
                    ),
                    "profile_result": (
                        None
                        if state.profile_result is None
                        else self._serialize_profile_result(
                            state.profile_result
                        )
                    ),
                    "transform_result": (
                        None
                        if state.transform_result is None
                        else self._serialize_transform_result(
                            state.transform_result
                        )
                    ),
                    "debye_scattering_result": (
                        None
                        if state.debye_scattering_result is None
                        else self._serialize_debye_scattering_result(
                            state.debye_scattering_result
                        )
                    ),
                }
                for state in self._cluster_group_states
            ],
        }

    def _write_component_trace_file(
        self,
        state: _ClusterDensityGroupState,
        component_dir: Path,
    ) -> str:
        transform_result = state.transform_result
        if transform_result is None:
            raise ValueError(
                f"Cluster {state.display_name} does not have a Fourier transform yet."
            )
        profile_file = self._component_profile_filename(
            state.structure_name,
            state.motif_name,
        )
        output_path = component_dir / profile_file
        header = (
            f"# Number of files: {state.inspection.total_files}\n"
            "# Columns: q, S(q)_avg, S(q)_std, S(q)_se\n"
        )
        q_values = np.asarray(transform_result.q_values, dtype=float)
        intensity = np.asarray(transform_result.intensity, dtype=float)
        data = np.column_stack(
            [
                q_values,
                intensity,
                np.zeros_like(q_values, dtype=float),
                np.zeros_like(q_values, dtype=float),
            ]
        )
        np.savetxt(
            output_path,
            data,
            comments="",
            header=header,
            fmt=["%.8f", "%.8f", "%.8f", "%.8f"],
        )
        return profile_file

    def _write_distribution_component_summary(
        self,
        summary_path: Path,
    ) -> None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(self._build_component_summary_payload(), indent=2)
            + "\n",
            encoding="utf-8",
        )

    def _distribution_state_restore_candidates(self) -> list[Path]:
        candidates: list[Path] = []
        workspace_state_path = self._workspace_state_path()
        if workspace_state_path is not None:
            candidates.append(workspace_state_path)
        summary_path = self._component_summary_path()
        if summary_path is not None and summary_path not in candidates:
            candidates.append(summary_path)
        return candidates

    def _delete_saved_output_history_storage(
        self,
        *,
        announce: bool,
    ) -> None:
        removed_paths: list[Path] = []
        for history_path in self._saved_output_history_restore_candidates():
            if not history_path.is_file():
                continue
            try:
                history_path.unlink()
            except Exception as exc:
                self._append_status(
                    "Could not remove saved output history from "
                    f"{history_path}: {exc}"
                )
                continue
            removed_paths.append(history_path)
        if announce and removed_paths:
            self._append_status(
                "Removed saved output history from "
                + ", ".join(str(path) for path in removed_paths)
                + "."
            )

    def _delete_workspace_state(
        self,
        *,
        announce: bool,
    ) -> None:
        workspace_state_path = self._workspace_state_path()
        if workspace_state_path is None or not workspace_state_path.is_file():
            return
        try:
            workspace_state_path.unlink()
        except Exception as exc:
            self._append_status(
                "Could not remove the saved workspace state from "
                f"{workspace_state_path}: {exc}"
            )
            return
        if announce:
            self._append_status(
                f"Removed the saved workspace state from {workspace_state_path}."
            )

    def _sync_workspace_state(self) -> None:
        if (
            self._preview_mode
            or self._restoring_saved_distribution_state
            or self._restoring_saved_output_history
        ):
            return
        workspace_state_path = self._workspace_state_path()
        current_input_path = self._current_input_path()
        if workspace_state_path is None or current_input_path is None:
            return
        if not self._has_calculated_density_outputs():
            self._delete_workspace_state(announce=False)
            return
        payload = self._build_component_summary_payload()
        payload["state_kind"] = "workspace_session"
        try:
            workspace_state_path.parent.mkdir(parents=True, exist_ok=True)
            workspace_state_path.write_text(
                json.dumps(payload, indent=2) + "\n",
                encoding="utf-8",
            )
        except Exception as exc:
            self._append_status(
                "Could not write workspace state to "
                f"{workspace_state_path}: {exc}"
            )

    def _update_distribution_metadata_after_push(
        self,
        *,
        component_dir: Path,
        component_map_path: Path,
        summary_path: Path,
    ) -> None:
        if self._distribution_root_dir is None:
            return
        metadata_path = self._distribution_root_dir / "distribution.json"
        if not metadata_path.is_file():
            return
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
        payload["updated_at"] = datetime.now().isoformat(timespec="seconds")
        payload["component_artifacts_ready"] = bool(
            component_dir.is_dir()
            and any(component_dir.glob("*.txt"))
            and component_map_path.is_file()
        )
        payload["electron_density_component_summary"] = str(summary_path)
        metadata_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @Slot()
    def _push_components_to_model(self) -> None:
        lock_reason = self._distribution_push_lock_reason()
        if lock_reason is not None:
            self._show_error("Push to Model unavailable", lock_reason)
            return
        artifact_targets = self._component_artifact_targets()
        summary_path = self._component_summary_path()
        if artifact_targets is None or summary_path is None:
            self._show_error(
                "Push to Model unavailable",
                "This window is not linked to a computed distribution.",
            )
            return
        if not self._cluster_group_states:
            self._show_error(
                "Push to Model unavailable",
                "Load a cluster-folder input before pushing Born-approximation traces.",
            )
            return
        pending = [
            state.display_name
            for state in self._cluster_group_states
            if state.transform_result is None
        ]
        if pending:
            self._show_error(
                "Push to Model unavailable",
                "Evaluate the Fourier transform for every cluster before pushing. "
                "Pending: " + ", ".join(pending),
            )
            return
        component_dir, component_map_path = artifact_targets
        component_dir.mkdir(parents=True, exist_ok=True)
        saxs_map: dict[str, dict[str, str]] = {}
        for state in self._cluster_group_states:
            profile_file = self._write_component_trace_file(
                state,
                component_dir,
            )
            saxs_map.setdefault(state.structure_name, {})
            saxs_map[state.structure_name][state.motif_name] = profile_file
        component_map_path.write_text(
            json.dumps({"saxs_map": saxs_map}, indent=2) + "\n",
            encoding="utf-8",
        )
        self._write_distribution_component_summary(summary_path)
        self._update_distribution_metadata_after_push(
            component_dir=component_dir,
            component_map_path=component_map_path,
            summary_path=summary_path,
        )
        mismatch_text = self._project_q_range_mismatch_text()
        if mismatch_text:
            self._append_status(mismatch_text)
        self._append_status(
            "Pushed Born-approximation component traces into the linked "
            f"computed distribution: {component_map_path}"
        )
        self.statusBar().showMessage(
            "Born-approximation components pushed to model"
        )
        self.born_components_built.emit(
            {
                "project_dir": (
                    None
                    if self._project_dir is None
                    else str(self._project_dir)
                ),
                "distribution_id": self._distribution_id,
                "distribution_dir": (
                    None
                    if self._distribution_root_dir is None
                    else str(self._distribution_root_dir)
                ),
                "component_dir": str(component_dir),
                "component_map_path": str(component_map_path),
                "component_summary_path": str(summary_path),
            }
        )
        self._update_push_to_model_state()

    def _restore_distribution_state_from_payload(
        self,
        payload: dict[str, object],
        *,
        source_path: Path,
    ) -> bool:
        if not self._cluster_group_states:
            return False
        state_by_key = {
            state.key: state for state in self._cluster_group_states
        }
        restored_count = 0
        self._restoring_saved_distribution_state = True
        try:
            self.contiguous_frame_mode_checkbox.setChecked(
                bool(payload.get("use_contiguous_frame_mode", True))
            )
            if hasattr(self, "auto_save_smearing_outputs_checkbox"):
                self.auto_save_smearing_outputs_checkbox.setChecked(
                    bool(payload.get("auto_save_smearing_outputs", False))
                )
            self.manual_mode_checkbox.blockSignals(True)
            try:
                self.manual_mode_checkbox.setChecked(
                    bool(payload.get("manual_mode_enabled", False))
                )
            finally:
                self.manual_mode_checkbox.blockSignals(False)
            manual_mesh_lock_payload = payload.get("manual_mesh_lock_settings")
            if isinstance(manual_mesh_lock_payload, dict):
                try:
                    self._manual_mesh_lock_settings = (
                        self._deserialize_mesh_settings(
                            manual_mesh_lock_payload
                        )
                    )
                except Exception:
                    self._manual_mesh_lock_settings = None
            else:
                self._manual_mesh_lock_settings = None
            restored_mesh_settings = None
            active_mesh_settings_payload = payload.get("active_mesh_settings")
            if isinstance(active_mesh_settings_payload, dict):
                try:
                    restored_mesh_settings = self._deserialize_mesh_settings(
                        active_mesh_settings_payload
                    )
                except Exception:
                    restored_mesh_settings = None
            for group_payload in payload.get("cluster_groups") or []:
                if not isinstance(group_payload, dict):
                    continue
                key = str(group_payload.get("key") or "").strip()
                state = state_by_key.get(key)
                if state is None:
                    continue
                state.trace_color = str(
                    group_payload.get("trace_color") or state.trace_color
                )
                state.average_atom_count = float(
                    group_payload.get("average_atom_count")
                    or state.average_atom_count
                )
                state.solvent_density_e_per_a3 = (
                    None
                    if group_payload.get("solvent_density_e_per_a3") is None
                    else float(group_payload.get("solvent_density_e_per_a3"))
                )
                state.solvent_cutoff_radius_a = (
                    None
                    if group_payload.get("solvent_cutoff_radius_a") is None
                    else float(group_payload.get("solvent_cutoff_radius_a"))
                )
                reference_structure_payload = group_payload.get(
                    "reference_structure"
                )
                if isinstance(reference_structure_payload, dict):
                    try:
                        state.reference_structure = (
                            self._deserialize_structure(
                                reference_structure_payload
                            )
                        )
                    except Exception:
                        pass
                profile_payload = group_payload.get("profile_result")
                if isinstance(profile_payload, dict):
                    state.profile_result = self._deserialize_profile_result(
                        profile_payload
                    )
                    if not isinstance(reference_structure_payload, dict):
                        state.reference_structure = (
                            state.profile_result.structure
                        )
                    self._sync_cluster_state_solvent_metadata(state)
                elif state.solvent_cutoff_radius_a is None:
                    legacy_cutoff = group_payload.get(
                        "solvent_intercept_e_per_a3"
                    )
                    state.solvent_cutoff_radius_a = (
                        None if legacy_cutoff is None else float(legacy_cutoff)
                    )
                transform_payload = group_payload.get("transform_result")
                if isinstance(transform_payload, dict):
                    state.transform_result = (
                        self._deserialize_transform_result(
                            transform_payload,
                            state.profile_result,
                            single_atom_only=state.single_atom_only,
                        )
                    )
                else:
                    state.transform_result = None
                debye_payload = group_payload.get("debye_scattering_result")
                if isinstance(debye_payload, dict):
                    try:
                        state.debye_scattering_result = (
                            self._deserialize_debye_scattering_result(
                                debye_payload
                            )
                        )
                    except Exception:
                        state.debye_scattering_result = None
                else:
                    state.debye_scattering_result = None
                fourier_settings_payload = group_payload.get(
                    "fourier_settings"
                )
                if isinstance(fourier_settings_payload, dict):
                    state.fourier_settings = (
                        self._deserialize_fourier_settings(
                            fourier_settings_payload
                        )
                    )
                elif state.transform_result is not None:
                    state.fourier_settings = (
                        state.transform_result.preview.settings
                    )
                elif state.fourier_settings is None:
                    state.fourier_settings = (
                        self._active_fourier_settings.normalized()
                    )
                restored_count += 1
            if restored_mesh_settings is None:
                active_key = str(payload.get("active_group_key") or "").strip()
                fallback_state = self._cluster_group_state_by_key(active_key)
                if (
                    fallback_state is None
                    or fallback_state.profile_result is None
                ):
                    fallback_state = next(
                        (
                            state
                            for state in self._cluster_group_states
                            if state.profile_result is not None
                        ),
                        None,
                    )
                if (
                    fallback_state is not None
                    and fallback_state.profile_result is not None
                ):
                    restored_mesh_settings = (
                        fallback_state.profile_result.mesh_geometry.settings
                    )
            if restored_mesh_settings is not None:
                self._active_mesh_settings = restored_mesh_settings
                self._sync_mesh_controls_to_settings(restored_mesh_settings)
                self._set_mesh_settings_confirmed_for_run(True)
            self.show_all_cluster_transforms_checkbox.setChecked(
                bool(payload.get("show_all_cluster_transforms"))
            )
            self._populate_cluster_group_table()
            active_key = str(payload.get("active_group_key") or "").strip()
            if active_key and active_key in state_by_key:
                for row_index, state in enumerate(self._cluster_group_states):
                    if state.key == active_key:
                        self.cluster_group_table.selectRow(row_index)
                        break
                self._set_active_cluster_group(active_key)
            elif self._cluster_group_states:
                self.cluster_group_table.selectRow(0)
                self._set_active_cluster_group(
                    self._cluster_group_states[0].key
                )
        finally:
            self._restoring_saved_distribution_state = False
        self._refresh_manual_mode_notice()
        self._refresh_mesh_notice()
        if restored_count > 0:
            self._append_status(
                "Restored saved electron-density mapping state for "
                f"{restored_count} cluster group"
                f"{'' if restored_count == 1 else 's'} from {source_path}."
            )
        self._update_push_to_model_state()
        return restored_count > 0

    def _restore_saved_distribution_state_if_available(self) -> bool:
        if self._preview_mode:
            self._update_push_to_model_state()
            return False
        for state_path in self._distribution_state_restore_candidates():
            if not state_path.is_file():
                continue
            try:
                payload = json.loads(state_path.read_text(encoding="utf-8"))
            except Exception as exc:
                self._append_status(
                    "Could not restore saved electron-density mapping state "
                    f"from {state_path}: {exc}"
                )
                continue
            saved_input_path = str(payload.get("input_path") or "").strip()
            resolved_saved_input: Path | None = None
            if saved_input_path:
                try:
                    resolved_saved_input = (
                        Path(saved_input_path).expanduser().resolve()
                    )
                except Exception:
                    resolved_saved_input = None
            current_input_path = self._current_input_path()
            if (
                resolved_saved_input is not None
                and current_input_path is not None
                and resolved_saved_input != current_input_path
            ):
                continue
            if (
                resolved_saved_input is not None
                and current_input_path is None
                and not self._cluster_group_states
            ):
                try:
                    self.input_path_edit.setText(str(resolved_saved_input))
                    self._load_input_path(
                        resolved_saved_input,
                        asynchronous=True,
                        restore_saved_distribution_state=True,
                    )
                except Exception as exc:
                    self._append_status(
                        "Could not reload saved cluster input "
                        f"{resolved_saved_input}: {exc}"
                    )
                    continue
                return True
            if self._restore_distribution_state_from_payload(
                payload,
                source_path=state_path,
            ):
                return False
        self._update_push_to_model_state()
        return False

    def _apply_mesh_settings(
        self,
        settings: ElectronDensityMeshSettings,
        *,
        announce: bool,
        preserve_viewer_display: bool = False,
        update_viewer: bool = True,
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
        if self._structure is not None and update_viewer:
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
        preserve_mesh_confirmation = (
            self._mesh_confirmation_carries_forward_on_auto_apply()
        )
        active_cluster_state = self._active_cluster_group_state()
        if self._cluster_group_states:
            updated_structures: dict[str, ElectronDensityStructure] = {}
            try:
                for state in self._cluster_group_states:
                    updated_structures[state.key] = (
                        recenter_electron_density_structure(
                            state.reference_structure,
                            center_mode=center_mode,
                            reference_element=state.reference_structure.reference_element,
                        )
                    )
            except Exception as exc:
                self._show_error("Center Update Error", str(exc))
                return
            for state in self._cluster_group_states:
                state.reference_structure = updated_structures[state.key]
            if active_cluster_state is not None:
                self._structure = updated_structures[active_cluster_state.key]
            self._reset_density_results()
            self._clear_cluster_group_outputs(clear_manual_mesh_lock=True)
            self._sync_controls_to_structure(sync_mesh_rmax=False)
            self._apply_mesh_settings(
                self._active_mesh_settings,
                announce=False,
                preserve_viewer_display=True,
            )
        else:
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
            self._sync_controls_to_structure(sync_mesh_rmax=True)
            self._apply_mesh_settings(
                self._mesh_settings_from_controls(),
                announce=False,
                preserve_viewer_display=True,
            )
        self._set_mesh_settings_confirmed_for_run(preserve_mesh_confirmation)
        self._sync_fourier_controls_to_domain(reset_bounds=True)
        self._refresh_fourier_info_labels(None)
        self._refresh_contrast_display()
        if self._cluster_group_states:
            if self._structure.center_mode == "nearest_atom":
                self._append_status(
                    "Snapped the active center for every stoichiometry to "
                    "its nearest atom."
                )
            elif self._structure.center_mode == "reference_element":
                self._append_status(
                    "Snapped every stoichiometry to its selected "
                    "reference-element geometric center."
                )
            else:
                self._append_status(
                    "Reset the active center for every stoichiometry to its "
                    "geometric mass center."
                )
        elif self._structure.center_mode == "nearest_atom":
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
                "Reset the active center to the geometric mass center."
            )
        self.statusBar().showMessage("Updated active center")
        self._sync_workspace_state()

    @Slot()
    def _apply_mesh_from_controls(self) -> None:
        if self._manual_mesh_lock_settings is not None:
            self.statusBar().showMessage(
                "Mesh settings are locked until the calculated densities are reset.",
                5000,
            )
            return
        self._apply_mesh_settings(
            self._mesh_settings_from_controls(), announce=True
        )
        self._set_mesh_settings_confirmed_for_run(True)
        self.statusBar().showMessage("Updated mesh overlay")
        self._sync_workspace_state()

    @Slot()
    def _request_calculation_stop(self) -> None:
        if self._calculation_worker is None:
            return
        self._calculation_cancel_requested = True
        self.stop_calculation_button.setEnabled(False)
        self._calculation_worker.cancel()
        if not self._is_calculation_running():
            if not self._cluster_group_states:
                self._reset_density_results()
                self._set_calculation_overall_progress_visible(False)
            self.calculation_progress_bar.setRange(0, 1)
            self.calculation_progress_bar.setValue(0)
            self.calculation_progress_bar.setFormat("%v / %m steps")
            self.calculation_progress_message.setText(
                "Electron-density calculation stopped."
            )
            self.statusBar().showMessage(
                "Discarding pending electron-density result...",
                5000,
            )
            self._append_status(
                "Discarded the pending electron-density calculation result."
            )
            return
        self.cancel_calculation_requested.emit()
        self.calculation_progress_message.setText(
            "Stopping electron-density calculation..."
        )
        self.statusBar().showMessage(
            "Stopping electron-density calculation...",
            5000,
        )
        self._append_status(
            "Requested a stop for the active electron-density calculation."
        )

    @Slot()
    def _reset_calculated_densities(self) -> None:
        if self._is_calculation_running():
            self._show_error(
                "Calculation Running",
                "Stop the active electron-density calculation before resetting the results.",
            )
            return
        if not self._has_calculated_density_outputs():
            return
        response = QMessageBox.question(
            self,
            "Reset Calculated Densities",
            "This will clear the active electron-density calculations and "
            "unlock the mesh settings for a fresh run. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if response != QMessageBox.StandardButton.Yes:
            return
        confirmation_text, accepted = QInputDialog.getText(
            self,
            "Secondary Authentication",
            "Type RESET to clear the calculated densities:",
            QLineEdit.EchoMode.Normal,
            "",
        )
        if not accepted:
            return
        if str(confirmation_text).strip().upper() != "RESET":
            self._show_error(
                "Reset Canceled",
                "Secondary authentication failed. Type RESET to clear the calculated densities.",
            )
            return
        cleared_cluster_outputs = self._clear_cluster_group_outputs(
            clear_manual_mesh_lock=True
        )
        self._reset_density_results()
        self._refresh_contrast_display()
        self._refresh_fourier_info_labels(None)
        self._reset_progress_display("Idle")
        self._clear_saved_output_history(announce=False)
        self._delete_saved_output_history_storage(announce=False)
        self._delete_workspace_state(announce=False)
        if self._cluster_group_states:
            active_key = (
                self._selected_cluster_group_key
                or self._cluster_group_states[0].key
            )
            self._set_active_cluster_group(active_key)
        cleared_text = (
            f"Cleared {cleared_cluster_outputs} stoichiometry calculation"
            f"{'' if cleared_cluster_outputs == 1 else 's'}."
            if cleared_cluster_outputs > 0
            else "Cleared the active electron-density results."
        )
        self._append_status(
            cleared_text + " Mesh settings are editable again."
        )
        self.statusBar().showMessage("Cleared calculated densities")

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
            self._load_input_path(Path(text), asynchronous=True)
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

    def _set_calculation_overall_progress_visible(
        self,
        visible: bool,
    ) -> None:
        show = bool(visible)
        self.calculation_overall_progress_message.setHidden(not show)
        self.calculation_overall_progress_bar.setHidden(not show)

    def _reset_progress_display(self, message: str = "Idle") -> None:
        self.calculation_progress_message.setText(str(message))
        self.calculation_progress_bar.setRange(0, 1)
        self.calculation_progress_bar.setValue(0)
        self.calculation_progress_bar.setFormat("%v / %m steps")
        self.calculation_overall_progress_message.setText(
            "Overall progress: idle"
        )
        self.calculation_overall_progress_bar.setRange(0, 1)
        self.calculation_overall_progress_bar.setValue(0)
        self.calculation_overall_progress_bar.setFormat(
            "%v / %m cluster groups"
        )
        self._set_calculation_overall_progress_visible(False)

    def _set_calculation_running(self, running: bool) -> None:
        self._calculation_ui_running = bool(running)
        enabled = not bool(running)
        for widget in (
            self.load_input_button,
            self.run_button,
            self.cluster_group_table,
            self.contiguous_frame_mode_checkbox,
            self.pinned_geometric_tracking_checkbox,
            self.manual_mode_checkbox,
            self.update_mesh_button,
            self.snap_center_button,
            self.reset_center_button,
            self.snap_reference_center_button,
            self.rstep_spin,
            self.theta_divisions_spin,
            self.phi_divisions_spin,
            self.rmax_spin,
            self.smearing_factor_spin,
            self.auto_save_smearing_outputs_checkbox,
            self.solvent_method_combo,
            self.solvent_preset_combo,
            self.save_custom_solvent_button,
            self.delete_custom_solvent_button,
            self.solvent_formula_edit,
            self.solvent_density_spin,
            self.direct_density_spin,
            self.reference_solvent_file_edit,
            self.reference_solvent_browse_button,
            self.apply_smearing_button,
            self.apply_smearing_to_all_button,
            self.compute_solvent_density_button,
            self.apply_contrast_to_all_button,
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
            self.apply_fourier_to_all_button,
            self.calculate_debye_scattering_button,
            self.apply_debye_to_all_button,
            self.open_debye_scattering_compare_button,
            self.fourier_settings_table,
            self.show_all_cluster_transforms_checkbox,
            self.evaluate_fourier_button,
            self.push_to_model_button,
            self.input_path_edit,
            self.output_dir_edit,
            self.output_basename_edit,
        ):
            widget.setEnabled(enabled)
        pending_thread_teardown = (
            not bool(running)
            and self._calculation_worker is not None
            and self._calculation_thread is not None
            and not getattr(
                self._calculation_worker, "_cancel_requested", False
            )
        )
        self.stop_calculation_button.setEnabled(
            bool(running) or pending_thread_teardown
        )
        self.reset_calculations_button.setEnabled(
            (not bool(running)) and self._has_calculated_density_outputs()
        )
        self._refresh_mesh_control_lock()
        if enabled:
            self._refresh_pinned_geometric_tracking_controls()
            self._update_push_to_model_state()
        self._refresh_fourier_table_interaction_state()
        self._refresh_run_action_state()

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
        self.calculation_progress_bar.setFormat("%v / %m steps")
        text = str(message).strip()
        if text:
            self.calculation_progress_message.setText(text)

    @Slot(int, int, str)
    def _on_calculation_overall_progress(
        self,
        current: int,
        total: int,
        message: str,
    ) -> None:
        if self._current_group_run_manual:
            return
        bounded_total = max(int(total), 1)
        bounded_current = min(max(int(current), 0), bounded_total)
        self._set_calculation_overall_progress_visible(True)
        self.calculation_overall_progress_bar.setRange(0, bounded_total)
        self.calculation_overall_progress_bar.setValue(bounded_current)
        self.calculation_overall_progress_bar.setFormat(
            "%v / %m cluster groups"
        )
        text = str(message).strip()
        if text:
            self.calculation_overall_progress_message.setText(text)

    def _apply_group_run_payload(
        self,
        data: dict[str, object],
    ) -> tuple[
        list[_ClusterDensityGroupState],
        list[_ClusterDensityGroupState],
        dict[str, ElectronDensityOutputArtifacts],
    ]:
        group_results = data.get("group_results")
        group_transform_results = data.get("group_transform_results")
        group_artifacts = data.get("group_artifacts")
        updated_density_states: list[_ClusterDensityGroupState] = []
        updated_transform_states: list[_ClusterDensityGroupState] = []
        written_artifacts: dict[str, ElectronDensityOutputArtifacts] = {}
        for state in self._cluster_group_states:
            preserved_fourier_settings = self._cluster_state_fourier_settings(
                state
            )
            result = (
                group_results.get(state.key)
                if isinstance(group_results, dict)
                else None
            )
            if isinstance(result, ElectronDensityProfileResult):
                state.profile_result = result
                state.transform_result = None
                state.debye_scattering_result = None
                state.solvent_density_e_per_a3 = None
                state.solvent_cutoff_radius_a = None
                self._capture_saved_output_entry(
                    "density",
                    group_state=state,
                    profile_result=result,
                )
                if self._active_contrast_settings is not None:
                    updated_result = apply_solvent_contrast_to_profile_result(
                        state.profile_result,
                        self._active_contrast_settings,
                        solvent_name=self._active_contrast_name,
                    )
                    state.profile_result = updated_result
                    self._sync_cluster_state_solvent_metadata(state)
                    if updated_result.solvent_contrast is not None:
                        self._capture_saved_output_entry(
                            "solvent_subtraction",
                            group_state=state,
                            profile_result=updated_result,
                        )
                self._set_cluster_state_fourier_settings(
                    state,
                    preserved_fourier_settings,
                    prefer_solvent_cutoff=bool(state.solvent_cutoff_radius_a),
                )
                updated_density_states.append(state)
                continue
            transform_result = (
                group_transform_results.get(state.key)
                if isinstance(group_transform_results, dict)
                else None
            )
            if state.single_atom_only and isinstance(
                transform_result,
                ElectronDensityScatteringTransformResult,
            ):
                state.profile_result = None
                state.fourier_settings = transform_result.preview.settings
                state.transform_result = transform_result
                state.debye_scattering_result = None
                state.solvent_density_e_per_a3 = None
                state.solvent_cutoff_radius_a = None
                updated_transform_states.append(state)
        if isinstance(group_artifacts, dict):
            for group_key, artifacts in group_artifacts.items():
                if isinstance(artifacts, ElectronDensityOutputArtifacts):
                    written_artifacts[str(group_key)] = artifacts
        self._populate_cluster_group_table()
        active_state = self._active_cluster_group_state()
        if active_state is not None:
            self._set_active_cluster_group(active_state.key)
        return (
            updated_density_states,
            updated_transform_states,
            written_artifacts,
        )

    @Slot(object)
    def _on_calculation_finished(self, payload: object) -> None:
        if self._calculation_cancel_requested or (
            self._calculation_worker is not None
            and getattr(self._calculation_worker, "_cancel_requested", False)
        ):
            self._on_calculation_canceled(payload)
            return
        if self._calculation_worker is not None and getattr(
            self._calculation_worker, "_cancel_requested", False
        ):
            self._on_calculation_canceled(payload)
            return
        self._calculation_cancel_requested = False
        data = payload if isinstance(payload, dict) else {}
        group_results = data.get("group_results")
        if isinstance(group_results, dict):
            (
                updated_density_states,
                updated_transform_states,
                written_artifacts,
            ) = self._apply_group_run_payload(data)
            if self._current_group_run_manual and (
                updated_density_states or updated_transform_states
            ):
                self._manual_mesh_lock_settings = self._active_mesh_settings
                self._refresh_mesh_notice()
            completed_count = self._completed_cluster_group_count()
            cluster_group_count = max(len(self._cluster_group_states), 1)
            self._set_calculation_overall_progress_visible(True)
            self.calculation_overall_progress_bar.setRange(
                0,
                cluster_group_count,
            )
            self.calculation_overall_progress_bar.setValue(completed_count)
            self.calculation_overall_progress_bar.setFormat(
                "%v / %m cluster groups"
            )
            self.calculation_overall_progress_message.setText(
                "Overall cluster progress: "
                f"{completed_count}/{len(self._cluster_group_states)} "
                "groups complete."
            )
            self.calculation_progress_bar.setRange(0, 1)
            self.calculation_progress_bar.setValue(1)
            self.calculation_progress_bar.setFormat("%v / %m steps")
            self.calculation_progress_message.setText(
                (
                    "Manual stoichiometry calculation complete."
                    if self._current_group_run_manual
                    else "Cluster-folder scattering preparation complete."
                )
            )
            self._set_calculation_running(False)
            summary_parts: list[str] = []
            if updated_density_states:
                summary_parts.append(
                    f"{len(updated_density_states)} electron-density profile"
                    f"{'' if len(updated_density_states) == 1 else 's'}"
                )
            if updated_transform_states:
                summary_parts.append(
                    f"{len(updated_transform_states)} direct Debye scattering trace"
                    f"{'' if len(updated_transform_states) == 1 else 's'}"
                )
            if not summary_parts:
                summary_parts.append("no cluster outputs")
            if self._current_group_run_manual:
                active_label = (
                    updated_density_states[0].display_name
                    if updated_density_states
                    else (
                        updated_transform_states[0].display_name
                        if updated_transform_states
                        else "the selected stoichiometry"
                    )
                )
                self._append_status(
                    "Prepared "
                    + " and ".join(summary_parts)
                    + f" for {active_label}. "
                    f"{completed_count}/{len(self._cluster_group_states)} "
                    "stoichiometry groups are now complete."
                )
            else:
                self._append_status(
                    "Prepared "
                    + " and ".join(summary_parts)
                    + f" across {len(self._cluster_group_states)} stoichiometry folders."
                )
            for state in updated_density_states:
                self._append_averaging_status(
                    state.profile_result,
                    prefix=f"{state.display_name}:",
                )
            for state in updated_transform_states:
                self._append_status(
                    f"{state.display_name}: built direct Debye I(Q) for a "
                    "single-atom cluster bin and skipped electron-density "
                    "and Fourier-profile evaluation."
                )
            for group_key, artifacts in sorted(written_artifacts.items()):
                self._append_status(
                    f"Wrote {group_key} CSV to {artifacts.csv_path}"
                )
                self._append_status(
                    f"Wrote {group_key} JSON to {artifacts.json_path}"
                )
            self.statusBar().showMessage(
                (
                    "Manual stoichiometry calculation complete"
                    if self._current_group_run_manual
                    else "Cluster-folder scattering preparation complete"
                )
            )
            self._current_group_run_manual = False
            self._refresh_run_action_state()
            self._schedule_cluster_view_cache_prewarm()
            self._sync_workspace_state()
            return
        result = data.get("result")
        artifacts = data.get("artifacts")
        if not isinstance(result, ElectronDensityProfileResult):
            self._on_calculation_failed(
                "Electron-density calculation finished without a valid result."
            )
            return

        self._profile_result = result
        self._fourier_result = None
        self._debye_scattering_result = None
        active_cluster_state = self._active_cluster_group_state()
        preserved_fourier_settings = (
            None
            if active_cluster_state is None
            else self._cluster_state_fourier_settings(active_cluster_state)
        )
        if active_cluster_state is not None:
            active_cluster_state.profile_result = result
            active_cluster_state.transform_result = None
            active_cluster_state.debye_scattering_result = None
            active_cluster_state.solvent_density_e_per_a3 = None
            active_cluster_state.solvent_cutoff_radius_a = None
        self._capture_saved_output_entry(
            "density",
            group_state=active_cluster_state,
            profile_result=result,
        )
        if self._active_contrast_settings is not None:
            self._apply_active_contrast_to_profile(
                show_error=False,
                announce=False,
            )
            if (
                active_cluster_state is not None
                and self._profile_result is not None
                and self._profile_result.solvent_contrast is not None
            ):
                active_cluster_state.profile_result = self._profile_result
                self._sync_cluster_state_solvent_metadata(active_cluster_state)
        if (
            active_cluster_state is not None
            and preserved_fourier_settings is not None
        ):
            updated_settings = self._set_cluster_state_fourier_settings(
                active_cluster_state,
                preserved_fourier_settings,
                prefer_solvent_cutoff=bool(
                    active_cluster_state.solvent_cutoff_radius_a
                ),
            )
            self._sync_fourier_controls_to_settings(updated_settings)
        self._refresh_profile_plots()
        self._populate_cluster_group_table()
        self._refresh_contrast_display()
        self._sync_fourier_controls_to_domain(reset_bounds=False)
        self._refresh_fourier_preview_from_controls(clear_transform=True)
        self._close_debye_scattering_compare_dialog()
        self._set_calculation_overall_progress_visible(False)
        self.calculation_progress_bar.setRange(0, 1)
        self.calculation_progress_bar.setValue(1)
        self.calculation_progress_bar.setFormat("%v / %m steps")
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
        self._append_averaging_status(active_result)
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
        self._current_group_run_manual = False
        self._refresh_run_action_state()
        self.statusBar().showMessage("Electron-density calculation complete")
        self._sync_workspace_state()

    @Slot(str)
    def _on_calculation_failed(self, message: str) -> None:
        self._calculation_cancel_requested = False
        self._set_calculation_running(False)
        self._reset_progress_display("Calculation failed.")
        self._current_group_run_manual = False
        self._refresh_run_action_state()
        self._show_error("Calculation Error", str(message))

    @Slot(object)
    def _on_calculation_canceled(self, payload: object) -> None:
        self._calculation_cancel_requested = False
        data = payload if isinstance(payload, dict) else {}
        group_results = data.get("group_results")
        if self._cluster_group_states and isinstance(group_results, dict):
            (
                updated_density_states,
                updated_transform_states,
                written_artifacts,
            ) = self._apply_group_run_payload(data)
            completed_count = self._completed_cluster_group_count()
            total_count = max(len(self._cluster_group_states), 1)
            self._set_calculation_overall_progress_visible(True)
            self.calculation_overall_progress_bar.setRange(0, total_count)
            self.calculation_overall_progress_bar.setValue(completed_count)
            self.calculation_overall_progress_bar.setFormat(
                "%v / %m cluster groups"
            )
            self.calculation_overall_progress_message.setText(
                "Overall cluster progress: "
                f"{completed_count}/{len(self._cluster_group_states)} "
                "groups complete."
            )
            for state in updated_density_states:
                self._append_averaging_status(
                    state.profile_result,
                    prefix=f"{state.display_name}:",
                )
            for state in updated_transform_states:
                self._append_status(
                    f"{state.display_name}: built direct Debye I(Q) for a "
                    "single-atom cluster bin and skipped electron-density "
                    "and Fourier-profile evaluation."
                )
            for group_key, artifacts in sorted(written_artifacts.items()):
                self._append_status(
                    f"Wrote {group_key} CSV to {artifacts.csv_path}"
                )
                self._append_status(
                    f"Wrote {group_key} JSON to {artifacts.json_path}"
                )
        else:
            self._reset_density_results()
            self._set_calculation_overall_progress_visible(False)
        self.calculation_progress_bar.setRange(0, 1)
        self.calculation_progress_bar.setValue(0)
        self.calculation_progress_bar.setFormat("%v / %m steps")
        self.calculation_progress_message.setText(
            "Electron-density calculation stopped."
        )
        self._set_calculation_running(False)
        self._current_group_run_manual = False
        self._refresh_run_action_state()
        self._schedule_cluster_view_cache_prewarm()
        self._append_status("Stopped the active electron-density calculation.")
        self.statusBar().showMessage("Electron-density calculation stopped")

    @Slot()
    def _cleanup_calculation_thread(self) -> None:
        if self._calculation_worker is not None:
            self._calculation_worker.deleteLater()
            self._calculation_worker = None
        if self._calculation_thread is not None:
            self._calculation_thread.deleteLater()
            self._calculation_thread = None
        self.stop_calculation_button.setEnabled(False)
        self._refresh_run_action_state()

    @Slot()
    def _run_calculation(self) -> None:
        if self._structure is None or self._inspection is None:
            if not self._cluster_group_states:
                self._show_error(
                    "No Structure Loaded",
                    "Load an XYZ or PDB structure before running the calculation.",
                )
                return
        target_group_states = list(self._cluster_group_states)
        self._current_group_run_manual = False
        if self._cluster_group_states and self._manual_mode_enabled_for_run():
            active_state = self._active_cluster_group_state()
            if active_state is None:
                self._show_error(
                    "No Stoichiometry Selected",
                    "Select a stoichiometry row before running manual mode.",
                )
                return
            target_group_states = [active_state]
            self._current_group_run_manual = True
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
        pending_mesh_settings = self._mesh_settings_from_controls()
        needs_mesh_confirmation = (
            pending_mesh_settings != self._active_mesh_settings
            or not self._mesh_settings_confirmed_for_run
        )
        if needs_mesh_confirmation:
            warning_lines: list[str] = []
            if not self._mesh_settings_confirmed_for_run:
                warning_lines.append(
                    "Mesh settings were not updated from the defaults "
                    "currently shown in the UI."
                )
            if pending_mesh_settings != self._active_mesh_settings:
                warning_lines.append(
                    "Pending mesh field values differ from the active mesh, "
                    "so this calculation will use the last updated mesh "
                    "instead of the values currently shown in the fields."
                )
            warning_lines.append(
                "Active mesh: "
                f"{self._format_mesh_settings_summary(self._active_mesh_settings)}; "
                "center mode="
                f"{self._center_mode_label_for_structure(self._structure)}."
            )
            if pending_mesh_settings != self._active_mesh_settings:
                warning_lines.append(
                    "Pending fields: "
                    f"{self._format_mesh_settings_summary(pending_mesh_settings)}."
                )
            response = QMessageBox.question(
                self,
                "Mesh Settings Not Updated",
                " ".join(warning_lines)
                + " Proceed with the electron-density calculation?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if response != QMessageBox.StandardButton.Yes:
                self.statusBar().showMessage(
                    "Electron-density calculation canceled",
                    5000,
                )
                self._append_status(
                    "Canceled electron-density calculation so the mesh "
                    "settings can be reviewed."
                )
                return
            self._set_mesh_settings_confirmed_for_run(True)
        self._initial_workspace_load_timer.stop()
        self._deferred_initial_input_path = None
        self._deferred_restore_saved_distribution_state = False
        self._calculation_cancel_requested = False
        self._set_calculation_running(True)
        self.stop_calculation_button.setEnabled(True)
        grouped_run = bool(self._cluster_group_states)
        if grouped_run:
            group_total = max(len(target_group_states), 1)
            self._set_calculation_overall_progress_visible(True)
            if self._current_group_run_manual:
                completed_count = self._completed_cluster_group_count()
                self.calculation_overall_progress_bar.setRange(
                    0,
                    max(len(self._cluster_group_states), 1),
                )
                self.calculation_overall_progress_bar.setValue(completed_count)
                self.calculation_overall_progress_bar.setFormat(
                    "%v / %m cluster groups"
                )
                self.calculation_overall_progress_message.setText(
                    "Overall cluster progress: "
                    f"{completed_count}/{len(self._cluster_group_states)} "
                    "groups complete. Manual mode is running the selected row."
                )
            else:
                self.calculation_overall_progress_bar.setRange(0, group_total)
                self.calculation_overall_progress_bar.setValue(0)
                self.calculation_overall_progress_bar.setFormat(
                    "%v / %m cluster groups"
                )
                self.calculation_overall_progress_message.setText(
                    "Overall cluster progress: "
                    f"0/{len(target_group_states)} groups complete."
                )
            self.calculation_progress_bar.setRange(0, 1)
            self.calculation_progress_bar.setValue(0)
            self.calculation_progress_bar.setFormat("%v / %m steps")
            self.calculation_progress_message.setText(
                (
                    "Preparing manual stoichiometry calculation..."
                    if self._current_group_run_manual
                    else "Waiting for the first cluster-group progress update..."
                )
            )
        else:
            self._set_calculation_overall_progress_visible(False)
            self.calculation_progress_bar.setRange(
                0,
                max(len(self._inspection.structure_files) * 2 + 3, 1),
            )
            self.calculation_progress_bar.setValue(0)
            self.calculation_progress_bar.setFormat("%v / %m steps")
            self.calculation_progress_message.setText(
                "Preparing electron-density calculation..."
            )
        self.statusBar().showMessage("Computing electron-density profile...")
        self._calculation_thread = QThread(self)
        self._calculation_worker = ElectronDensityCalculationWorker(
            inspection=self._inspection,
            reference_structure=self._structure,
            mesh_settings=self._active_mesh_settings,
            smearing_settings=self._smearing_settings_from_controls(),
            center_mode=self._structure.center_mode,
            reference_element=self._structure.reference_element,
            output_dir=self.output_dir_edit.text().strip(),
            output_basename=self._output_basename(),
            use_contiguous_frame_mode=self._use_contiguous_frame_mode(),
            grouped_inspections=tuple(
                (state.key, state.inspection) for state in target_group_states
            ),
            grouped_reference_structures=tuple(
                (state.key, state.reference_structure)
                for state in target_group_states
            ),
            single_atom_group_keys=tuple(
                state.key
                for state in target_group_states
                if state.single_atom_only
            ),
            debye_scattering_settings=self._fourier_settings_from_controls(),
        )
        self._calculation_worker.moveToThread(self._calculation_thread)
        self._calculation_thread.started.connect(self._calculation_worker.run)
        self.cancel_calculation_requested.connect(
            self._calculation_worker.cancel
        )
        self._calculation_worker.progress.connect(
            self._on_calculation_progress
        )
        self._calculation_worker.overall_progress.connect(
            self._on_calculation_overall_progress
        )
        self._calculation_worker.finished.connect(
            self._on_calculation_finished
        )
        self._calculation_worker.canceled.connect(
            self._on_calculation_canceled
        )
        self._calculation_worker.failed.connect(self._on_calculation_failed)
        self._calculation_worker.finished.connect(
            self._calculation_thread.quit
        )
        self._calculation_worker.canceled.connect(
            self._calculation_thread.quit
        )
        self._calculation_worker.failed.connect(self._calculation_thread.quit)
        self._calculation_thread.finished.connect(
            self._cleanup_calculation_thread
        )
        self._calculation_thread.start(QThread.Priority.LowPriority)

    @staticmethod
    def _build_plot_trace_columns(
        *,
        profile_result: ElectronDensityProfileResult | None,
        fourier_preview: ElectronDensityFourierTransformPreview | None,
        transform_result: ElectronDensityScatteringTransformResult | None,
    ) -> tuple[list[str], list[list[str]]]:
        headers: list[str] = []
        columns: list[list[str]] = []

        result = profile_result
        if result is not None:
            r_values = np.asarray(result.radial_centers, dtype=float)
            raw_values = np.asarray(
                result.orientation_average_density,
                dtype=float,
            )
            raw_std = np.asarray(
                result.orientation_density_stddev,
                dtype=float,
            )
            smeared_values = np.asarray(
                result.smeared_orientation_average_density,
                dtype=float,
            )
            smeared_std = np.asarray(
                result.smeared_orientation_density_stddev,
                dtype=float,
            )
            headers += [
                "r_center_a",
                "raw_density_e_per_a3",
                "raw_stddev_e_per_a3",
                "smeared_density_e_per_a3",
                "smeared_stddev_e_per_a3",
            ]
            columns += [
                [f"{value:.10g}" for value in r_values],
                [f"{value:.10g}" for value in raw_values],
                [f"{value:.10g}" for value in raw_std],
                [f"{value:.10g}" for value in smeared_values],
                [f"{value:.10g}" for value in smeared_std],
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
                        for _value in r_values
                    ],
                    [f"{value:.10g}" for value in residual],
                ]

        preview = fourier_preview
        if preview is not None:
            preview_r = np.asarray(preview.resampled_r_values, dtype=float)
            preview_density = np.asarray(
                preview.resampled_density_values,
                dtype=float,
            )
            windowed_density = np.asarray(
                preview.windowed_density_values,
                dtype=float,
            )
            headers += [
                "fourier_preview_r_a",
                "fourier_preview_resampled_density",
                "fourier_preview_windowed_density",
            ]
            columns += [
                [f"{value:.10g}" for value in preview_r],
                [f"{value:.10g}" for value in preview_density],
                [f"{value:.10g}" for value in windowed_density],
            ]

        ft_result = transform_result
        if ft_result is not None:
            q_values = np.asarray(ft_result.q_values, dtype=float)
            amplitude = np.asarray(
                ft_result.scattering_amplitude,
                dtype=float,
            )
            intensity = np.asarray(ft_result.intensity, dtype=float)
            headers += [
                "q_a_inv",
                "scattering_amplitude",
                "intensity",
            ]
            columns += [
                [f"{value:.10g}" for value in q_values],
                [f"{value:.10g}" for value in amplitude],
                [f"{value:.10g}" for value in intensity],
            ]
        return headers, columns

    @staticmethod
    def _write_plot_trace_csv(
        csv_path: Path,
        *,
        profile_result: ElectronDensityProfileResult | None,
        fourier_preview: ElectronDensityFourierTransformPreview | None,
        transform_result: ElectronDensityScatteringTransformResult | None,
    ) -> None:
        headers, columns = (
            ElectronDensityMappingMainWindow._build_plot_trace_columns(
                profile_result=profile_result,
                fourier_preview=fourier_preview,
                transform_result=transform_result,
            )
        )
        max_rows = max((len(column) for column in columns), default=0)
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(headers)
            for row_index in range(max_rows):
                writer.writerow(
                    column[row_index] if row_index < len(column) else ""
                    for column in columns
                )

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
        try:
            self._write_plot_trace_csv(
                csv_path,
                profile_result=self._profile_result,
                fourier_preview=self._fourier_preview,
                transform_result=self._fourier_result,
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

    def closeEvent(self, event: QCloseEvent) -> None:
        self._initial_workspace_load_timer.stop()
        self._cluster_view_cache_prewarm_timer.stop()
        self._cluster_view_cache_prewarm_queue = []
        self.structure_viewer.clear_scene_cache()
        self._sync_workspace_state()
        self._close_workspace_load_progress_dialog()
        self._close_batch_operation_progress_dialog()
        if (
            self._calculation_worker is not None
            and self._calculation_thread is not None
            and self._calculation_thread.isRunning()
        ):
            self._calculation_worker.cancel()
            self.cancel_calculation_requested.emit()
            self._calculation_thread.quit()
            self._calculation_thread.wait(1000)
        if (
            self._workspace_load_thread is not None
            and self._workspace_load_thread.isRunning()
        ):
            self._workspace_load_thread.quit()
            self._workspace_load_thread.wait(1000)
        super().closeEvent(event)


def _forget_open_window(window: ElectronDensityMappingMainWindow) -> None:
    if window in _OPEN_WINDOWS:
        _OPEN_WINDOWS.remove(window)


def launch_electron_density_mapping_ui(
    *,
    initial_project_dir: str | Path | None = None,
    initial_input_path: str | Path | None = None,
    initial_output_dir: str | Path | None = None,
    initial_project_q_min: float | None = None,
    initial_project_q_max: float | None = None,
    initial_distribution_id: str | None = None,
    initial_distribution_root_dir: str | Path | None = None,
    initial_use_predicted_structure_weights: bool = False,
    preview_mode: bool = True,
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
        initial_input_path=None,
        initial_output_dir=(
            None
            if initial_output_dir is None
            else Path(initial_output_dir).expanduser().resolve()
        ),
        initial_project_q_min=initial_project_q_min,
        initial_project_q_max=initial_project_q_max,
        initial_distribution_id=initial_distribution_id,
        initial_distribution_root_dir=(
            None
            if initial_distribution_root_dir is None
            else Path(initial_distribution_root_dir).expanduser().resolve()
        ),
        initial_use_predicted_structure_weights=(
            initial_use_predicted_structure_weights
        ),
        preview_mode=preview_mode,
        restore_saved_distribution_state_on_init=False,
    )
    window.schedule_initial_workspace_load(
        initial_input_path=(
            None
            if initial_input_path is None
            else Path(initial_input_path).expanduser().resolve()
        ),
        restore_saved_distribution_state=not preview_mode,
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
