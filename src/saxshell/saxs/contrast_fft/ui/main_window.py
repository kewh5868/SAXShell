from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import (
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from PySide6.QtCore import (
    QObject,
    QSettings,
    Qt,
    QThread,
    QTimer,
    Signal,
    Slot,
)
from PySide6.QtGui import QAction, QCloseEvent
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from saxshell.fullrmc import load_rmc_project_source
from saxshell.plotting import Q_A_INVERSE_LABEL
from saxshell.saxs.born_refinement.backend import (
    build_shared_q_grid,
    compute_constant_weight_debye_intensity,
)
from saxshell.saxs.contrast.electron_density import (
    ANGSTROM3_PER_CM3,
    CONTRAST_SOLVENT_METHOD_DIRECT,
    CONTRAST_SOLVENT_METHOD_NEAT,
    CONTRAST_SOLVENT_METHOD_REFERENCE,
    ContrastElectronDensityEstimate,
    ContrastSolventDensitySettings,
    _direct_solvent_electron_density,
    _neat_solvent_electron_density,
)
from saxshell.saxs.contrast.solvents import (
    ContrastSolventPreset,
    delete_custom_solvent_preset,
    load_solvent_presets,
    ordered_solvent_preset_names,
    save_custom_solvent_preset,
)
from saxshell.saxs.contrast_fft import (
    ContrastFFTResult,
    ContrastFFTSettings,
    ContrastFFTTiming,
    compute_contrast_fft_intensity,
    default_contrast_fft_settings,
)
from saxshell.saxs.debye import discover_cluster_bins
from saxshell.saxs.electron_density_mapping.ui.viewer import (
    ElectronDensityStructureViewer,
)
from saxshell.saxs.electron_density_mapping.workflow import (
    ElectronDensityFourierTransformSettings,
    ElectronDensityMeshSettings,
    ElectronDensitySmearingSettings,
    ElectronDensityStructure,
    apply_solvent_contrast_to_profile_result,
    build_electron_density_mesh,
    compute_electron_density_profile,
    compute_electron_density_scattering_profile,
    inspect_structure_input,
    legacy_born_average_default_fourier_settings,
    legacy_born_average_default_mesh_settings,
    legacy_born_average_default_smearing_settings,
    load_electron_density_structure,
)
from saxshell.saxs.ui._pane_snap import PaneSnapFilter
from saxshell.saxs.ui.branding import (
    configure_saxshell_application,
    prepare_saxshell_application_identity,
)
from saxshell.saxs.ui.progress_dialog import SAXSProgressDialog

AUTO_SNAP_PANES_KEY = "contrast_fft_auto_snap_panes_enabled"
_SOLVENT_PRESET_NONE = "__none__"
_OPEN_WINDOWS: list["FFTBornApproximationMainWindow"] = []


class _FFTCancelledError(RuntimeError):
    """Raised when the 3D FFT worker is cancelled cooperatively."""


@dataclass(slots=True, frozen=True)
class _FFTProfileTarget:
    key: str
    display_name: str
    structure_name: str
    motif_name: str
    file_count: int
    reference_file: Path
    source_files: tuple[Path, ...]
    representative: str | None
    source_mode: str
    solvent_mode: str


@dataclass(slots=True, frozen=True)
class _FFTProfileComputationResult:
    target: _FFTProfileTarget
    q_values: np.ndarray
    fft_result: ContrastFFTResult
    legacy_q_values: np.ndarray | None
    legacy_intensity: np.ndarray | None
    exact_debye_intensity: np.ndarray | None
    legacy_elapsed_seconds: float | None
    debye_elapsed_seconds: float | None


@dataclass(slots=True, frozen=True)
class _FFTComputationPayload:
    q_values: np.ndarray
    profile_results: tuple[_FFTProfileComputationResult, ...]


class _CollapsibleSection(QWidget):
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

        header = QWidget(self)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 2, 0, 2)
        header_layout.setSpacing(4)
        self._toggle_button = QToolButton(self)
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

    @property
    def is_expanded(self) -> bool:
        return self._expanded


class _FFTComparisonPlot(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.figure = Figure(figsize=(8.4, 3.6))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self.toolbar)
        self.canvas.setMinimumHeight(300)
        layout.addWidget(self.canvas, stretch=1)
        self.draw_placeholder()

    def draw_placeholder(self) -> None:
        self.figure.clear()
        axis = self.figure.add_subplot(111)
        axis.text(
            0.5,
            0.58,
            "Run the 3D FFT Born calculation to populate the q-space comparison plot.",
            ha="center",
            va="center",
            wrap=True,
            transform=axis.transAxes,
        )
        axis.text(
            0.5,
            0.39,
            "Optional overlays can include the legacy 1D Born approximation, "
            "exact Debye scattering, and the zero-contrast kernel-corrected FFT diagnostic.",
            ha="center",
            va="center",
            wrap=True,
            alpha=0.78,
            transform=axis.transAxes,
        )
        axis.set_axis_off()
        self.canvas.draw_idle()

    def set_curves(
        self,
        *,
        q_values: np.ndarray,
        primary_values: np.ndarray,
        primary_label: str,
        additional_series: list[dict[str, object]],
        log_q_axis: bool,
        log_intensity_axis: bool,
        show_legend: bool,
    ) -> None:
        self.figure.clear()
        axis = self.figure.add_subplot(111)
        self._plot_series(
            axis,
            q_values=np.asarray(q_values, dtype=float),
            intensity=np.asarray(primary_values, dtype=float),
            label=str(primary_label),
            color="#1d4ed8",
            linestyle="-",
            linewidth=2.3,
            log_q_axis=log_q_axis,
            log_intensity_axis=log_intensity_axis,
        )
        for series in additional_series:
            self._plot_series(
                axis,
                q_values=np.asarray(series.get("q_values"), dtype=float),
                intensity=np.asarray(series.get("intensity"), dtype=float),
                label=str(series.get("label") or "Comparison"),
                color=str(series.get("color") or "#64748b"),
                linestyle=str(series.get("linestyle") or "--"),
                linewidth=float(series.get("linewidth") or 1.6),
                log_q_axis=log_q_axis,
                log_intensity_axis=log_intensity_axis,
            )
        if log_q_axis:
            axis.set_xscale("log")
        if log_intensity_axis:
            axis.set_yscale("log")
        axis.set_xlabel(Q_A_INVERSE_LABEL, labelpad=10.0)
        axis.set_ylabel("Intensity (arb. units)")
        axis.set_title("3D FFT Born Approximation")
        axis.grid(True, which="both", alpha=0.28)
        handles, labels = axis.get_legend_handles_labels()
        if show_legend and handles:
            axis.legend(loc="lower left", frameon=True)
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _plot_series(
        self,
        axis,
        *,
        q_values: np.ndarray,
        intensity: np.ndarray,
        label: str,
        color: str,
        linestyle: str,
        linewidth: float,
        log_q_axis: bool,
        log_intensity_axis: bool,
    ) -> None:
        mask = np.ones_like(q_values, dtype=bool)
        if log_q_axis:
            mask &= q_values > 0.0
        if log_intensity_axis:
            mask &= intensity > 0.0
        filtered_q = q_values[mask]
        filtered_intensity = intensity[mask]
        if filtered_q.size == 0 or filtered_intensity.size == 0:
            return
        axis.plot(
            filtered_q,
            filtered_intensity,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            label=label,
        )


class _FFTShellCountPlot(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.figure = Figure(figsize=(8.4, 2.8))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self.toolbar)
        self.canvas.setMinimumHeight(240)
        layout.addWidget(self.canvas, stretch=1)
        self.draw_placeholder()

    def draw_placeholder(self) -> None:
        self.figure.clear()
        axis = self.figure.add_subplot(111)
        axis.text(
            0.5,
            0.5,
            "q-shell population diagnostics will appear after the FFT run.",
            ha="center",
            va="center",
            wrap=True,
            transform=axis.transAxes,
        )
        axis.set_axis_off()
        self.canvas.draw_idle()

    def set_counts(
        self, q_values: np.ndarray, q_shell_counts: np.ndarray
    ) -> None:
        self.figure.clear()
        axis = self.figure.add_subplot(111)
        axis.plot(
            np.asarray(q_values, dtype=float),
            np.asarray(q_shell_counts, dtype=int),
            color="#0f766e",
            linewidth=1.9,
        )
        axis.set_xlabel(Q_A_INVERSE_LABEL)
        axis.set_ylabel("Shell count")
        axis.set_title("3D FFT q-Shell Population")
        axis.grid(True, alpha=0.28)
        self.figure.tight_layout()
        self.canvas.draw_idle()


class _FFTRealSpaceVisualizer(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.figure = Figure(figsize=(8.4, 3.3))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self.toolbar)
        self.canvas.setMinimumHeight(260)
        layout.addWidget(self.canvas, stretch=1)
        self.draw_placeholder()

    def draw_placeholder(self) -> None:
        self.figure.clear()
        axis = self.figure.add_subplot(111, projection="3d")
        axis.text2D(
            0.5,
            0.58,
            "Load a structure to preview its centered 3D real-space geometry.",
            ha="center",
            va="center",
            wrap=True,
            transform=axis.transAxes,
        )
        axis.text2D(
            0.5,
            0.39,
            "After the 3D FFT run, this panel adds the FFT box volume and a zoomed 3D structure view.",
            ha="center",
            va="center",
            wrap=True,
            alpha=0.78,
            transform=axis.transAxes,
        )
        axis.set_axis_off()
        self.canvas.draw_idle()

    def set_structure_preview(
        self,
        structure: ElectronDensityStructure | None,
        *,
        fft_result: ContrastFFTResult | None = None,
        contrast_summary: str | None = None,
    ) -> None:
        if structure is None:
            self.draw_placeholder()
            return
        coordinates = np.asarray(structure.centered_coordinates, dtype=float)
        if coordinates.size == 0:
            self.draw_placeholder()
            return
        self.figure.clear()
        if fft_result is None:
            axis = self.figure.add_subplot(111, projection="3d")
            self._draw_scene(
                axis,
                coordinates=coordinates,
                title="Centered Structure Preview",
            )
            axis.text2D(
                0.02,
                0.98,
                "Run the 3D FFT calculation to overlay the FFT box volume and spacing diagnostics.",
                ha="left",
                va="top",
                wrap=True,
                fontsize=9.0,
                bbox={
                    "boxstyle": "round,pad=0.3",
                    "facecolor": "#ffffff",
                    "edgecolor": "#cbd5e1",
                    "alpha": 0.94,
                },
                transform=axis.transAxes,
            )
            self.figure.tight_layout()
            self.canvas.draw_idle()
            return
        full_axis = self.figure.add_subplot(121, projection="3d")
        zoom_axis = self.figure.add_subplot(122, projection="3d")
        self._draw_scene(
            full_axis,
            coordinates=coordinates,
            title="FFT Box Volume",
            box_lengths=fft_result.box_lengths_a,
            fit_to_box=True,
        )
        self._draw_scene(
            zoom_axis,
            coordinates=coordinates,
            title="Structure Zoom",
        )
        summary_lines = [
            (
                "Box (Å): "
                + " × ".join(
                    f"{value:.1f}" for value in fft_result.box_lengths_a
                )
            ),
            (
                "Spacing: "
                f"{fft_result.voxel_spacing_a[0]:.3f} Å, "
                f"q_Nyquist={fft_result.q_nyquist_a_inverse:.3f} Å⁻¹"
            ),
        ]
        if contrast_summary:
            summary_lines.append(str(contrast_summary).strip())
        full_axis.text2D(
            0.02,
            0.02,
            "\n".join(summary_lines),
            ha="left",
            va="bottom",
            fontsize=8.8,
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "#ffffff",
                "edgecolor": "#cbd5e1",
                "alpha": 0.94,
            },
            transform=full_axis.transAxes,
        )
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _draw_scene(
        self,
        axis,
        *,
        coordinates: np.ndarray,
        title: str,
        box_lengths: tuple[float, float, float] | None = None,
        fit_to_box: bool = False,
    ) -> None:
        xyz = np.asarray(coordinates[:, :3], dtype=float)
        radial_distance = np.linalg.norm(xyz, axis=1)
        axis.scatter(
            xyz[:, 0],
            xyz[:, 1],
            xyz[:, 2],
            s=26.0,
            c=radial_distance,
            cmap="viridis",
            alpha=0.90,
            edgecolors="#0f172a",
            linewidths=0.2,
            depthshade=True,
        )
        if box_lengths is not None:
            self._draw_fft_box(axis, box_lengths=box_lengths)
            if fit_to_box:
                half_lengths = 0.5 * np.asarray(box_lengths, dtype=float)
                margin = max(float(np.max(half_lengths)) * 0.08, 1.0)
                self._set_3d_limits(
                    axis,
                    (
                        -float(half_lengths[0]) - margin,
                        float(half_lengths[0]) + margin,
                    ),
                    (
                        -float(half_lengths[1]) - margin,
                        float(half_lengths[1]) + margin,
                    ),
                    (
                        -float(half_lengths[2]) - margin,
                        float(half_lengths[2]) + margin,
                    ),
                )
        if not fit_to_box:
            x_min = float(np.min(xyz[:, 0]))
            x_max = float(np.max(xyz[:, 0]))
            y_min = float(np.min(xyz[:, 1]))
            y_max = float(np.max(xyz[:, 1]))
            z_min = float(np.min(xyz[:, 2]))
            z_max = float(np.max(xyz[:, 2]))
            pad = (
                max(
                    x_max - x_min,
                    y_max - y_min,
                    z_max - z_min,
                    1.0,
                )
                * 0.18
            )
            self._set_3d_limits(
                axis,
                (x_min - pad, x_max + pad),
                (y_min - pad, y_max + pad),
                (z_min - pad, z_max + pad),
            )
        axis.set_xlabel("x (Å)")
        axis.set_ylabel("y (Å)")
        axis.set_zlabel("z (Å)")
        axis.set_title(title)
        axis.view_init(
            elev=20.0 if fit_to_box else 24.0,
            azim=36.0 if fit_to_box else 48.0,
        )
        axis.grid(True, alpha=0.24)

    def _draw_fft_box(
        self,
        axis,
        *,
        box_lengths: tuple[float, float, float],
    ) -> None:
        half_lengths = 0.5 * np.asarray(box_lengths, dtype=float)
        x_half, y_half, z_half = (
            float(half_lengths[0]),
            float(half_lengths[1]),
            float(half_lengths[2]),
        )
        corners = np.asarray(
            [
                [-x_half, -y_half, -z_half],
                [x_half, -y_half, -z_half],
                [x_half, y_half, -z_half],
                [-x_half, y_half, -z_half],
                [-x_half, -y_half, z_half],
                [x_half, -y_half, z_half],
                [x_half, y_half, z_half],
                [-x_half, y_half, z_half],
            ],
            dtype=float,
        )
        edges = (
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        )
        for start, stop in edges:
            axis.plot(
                [corners[start, 0], corners[stop, 0]],
                [corners[start, 1], corners[stop, 1]],
                [corners[start, 2], corners[stop, 2]],
                color="#1d4ed8",
                linestyle="--",
                linewidth=1.2,
                alpha=0.95,
            )

    def _set_3d_limits(
        self,
        axis,
        x_limits: tuple[float, float],
        y_limits: tuple[float, float],
        z_limits: tuple[float, float],
    ) -> None:
        axis.set_xlim(*x_limits)
        axis.set_ylim(*y_limits)
        axis.set_zlim(*z_limits)
        spans = np.asarray(
            [
                max(float(x_limits[1] - x_limits[0]), 1.0),
                max(float(y_limits[1] - y_limits[0]), 1.0),
                max(float(z_limits[1] - z_limits[0]), 1.0),
            ],
            dtype=float,
        )
        axis.set_box_aspect(tuple(float(value) for value in spans))


class _FFTComputationWorker(QObject):
    progress = Signal(str)
    finished = Signal(object)
    failed = Signal(str)
    cancelled = Signal(str)

    def __init__(
        self,
        *,
        targets: tuple[_FFTProfileTarget, ...],
        fft_settings: ContrastFFTSettings,
        legacy_mesh_settings: ElectronDensityMeshSettings | None,
        legacy_smearing_settings: ElectronDensitySmearingSettings | None,
        legacy_fourier_settings: (
            ElectronDensityFourierTransformSettings | None
        ),
        active_contrast_settings: ContrastSolventDensitySettings | None,
        active_contrast_name: str | None,
        q_min: float,
        q_max: float,
        q_step: float,
        compare_legacy_1d: bool,
        compare_exact_debye: bool,
    ) -> None:
        super().__init__()
        self._targets = tuple(targets)
        self._fft_settings = fft_settings
        self._legacy_mesh_settings = (
            None
            if legacy_mesh_settings is None
            else legacy_mesh_settings.normalized()
        )
        self._legacy_smearing_settings = (
            None
            if legacy_smearing_settings is None
            else legacy_smearing_settings.normalized()
        )
        self._legacy_fourier_settings = (
            None
            if legacy_fourier_settings is None
            else legacy_fourier_settings.normalized()
        )
        self._active_contrast_settings = (
            None
            if active_contrast_settings is None
            else ContrastSolventDensitySettings.from_values(
                **active_contrast_settings.to_dict()
            )
        )
        self._active_contrast_name = (
            None if active_contrast_name is None else str(active_contrast_name)
        )
        self._q_min = float(q_min)
        self._q_max = float(q_max)
        self._q_step = float(q_step)
        self._compare_legacy_1d = bool(compare_legacy_1d)
        self._compare_exact_debye = bool(compare_exact_debye)
        self._cancel_requested = False

    def cancel(self) -> None:
        self._cancel_requested = True

    def _raise_if_cancelled(self) -> None:
        if self._cancel_requested:
            raise _FFTCancelledError("3D FFT Born calculation cancelled.")

    @Slot()
    def run(self) -> None:
        try:
            if not self._targets:
                raise ValueError(
                    "No 3D FFT profile targets were prepared for the calculation."
                )
            self._raise_if_cancelled()
            self.progress.emit("Preparing shared q grid...")
            q_values = build_shared_q_grid(
                self._q_min,
                self._q_max,
                q_step=self._q_step,
            )
            profile_results: list[_FFTProfileComputationResult] = []
            total_targets = len(self._targets)
            for target_index, target in enumerate(self._targets, start=1):
                self._raise_if_cancelled()
                profile_results.append(
                    self._compute_profile_result_for_target(
                        target,
                        q_values=q_values,
                        target_index=target_index,
                        total_targets=total_targets,
                    )
                )
            self._raise_if_cancelled()
            self.progress.emit("Updating 3D FFT Born outputs...")
            self.finished.emit(
                _FFTComputationPayload(
                    q_values=np.asarray(q_values, dtype=float),
                    profile_results=tuple(profile_results),
                )
            )
        except _FFTCancelledError as exc:
            self.cancelled.emit(str(exc))
        except Exception as exc:  # pragma: no cover - defensive UI worker path
            self.failed.emit(str(exc))

    def _compute_profile_result_for_target(
        self,
        target: _FFTProfileTarget,
        *,
        q_values: np.ndarray,
        target_index: int,
        total_targets: int,
    ) -> _FFTProfileComputationResult:
        self._raise_if_cancelled()
        self.progress.emit(
            "Running 3D FFT Born approximation for "
            f"{target.display_name} ({target_index}/{total_targets})..."
        )
        fft_results: list[ContrastFFTResult] = []
        legacy_q_values: np.ndarray | None = None
        legacy_intensities: list[np.ndarray] = []
        exact_debye_intensities: list[np.ndarray] = []
        legacy_elapsed_seconds = 0.0
        debye_elapsed_seconds = 0.0

        for file_index, file_path in enumerate(target.source_files, start=1):
            self._raise_if_cancelled()
            self.progress.emit(
                f"Loading structure {file_index}/{len(target.source_files)} "
                f"for {target.display_name}: {file_path.name}"
            )
            structure = load_electron_density_structure(file_path)
            coordinates = np.asarray(
                structure.centered_coordinates,
                dtype=float,
            )
            weights = np.asarray(structure.atomic_numbers, dtype=float)
            fft_results.append(
                compute_contrast_fft_intensity(
                    coordinates,
                    weights,
                    q_values,
                    self._fft_settings,
                    elements=structure.elements,
                    cancelled=lambda: self._cancel_requested,
                )
            )
            self._raise_if_cancelled()
            if self._compare_legacy_1d:
                legacy_start = perf_counter()
                mesh_settings = self._legacy_mesh_settings
                if mesh_settings is None:
                    mesh_settings = legacy_born_average_default_mesh_settings(
                        structure
                    )
                profile = compute_electron_density_profile(
                    structure,
                    mesh_settings,
                    smearing_settings=(
                        self._legacy_smearing_settings
                        or legacy_born_average_default_smearing_settings()
                    ),
                )
                if self._active_contrast_settings is not None:
                    profile = apply_solvent_contrast_to_profile_result(
                        profile,
                        self._active_contrast_settings,
                        solvent_name=self._active_contrast_name,
                    )
                legacy_fourier_template = self._legacy_fourier_settings
                if legacy_fourier_template is None:
                    legacy_fourier_template = (
                        legacy_born_average_default_fourier_settings(
                            r_max=float(profile.radial_centers[-1]),
                            q_min=float(q_values[0]),
                            q_max=float(q_values[-1]),
                            q_step=float(np.median(np.diff(q_values))),
                        )
                    )
                legacy_result = compute_electron_density_scattering_profile(
                    profile,
                    ElectronDensityFourierTransformSettings(
                        r_min=float(legacy_fourier_template.r_min),
                        r_max=float(legacy_fourier_template.r_max),
                        domain_mode=str(legacy_fourier_template.domain_mode),
                        window_function=str(
                            legacy_fourier_template.window_function
                        ),
                        resampling_points=int(
                            legacy_fourier_template.resampling_points
                        ),
                        q_min=float(q_values[0]),
                        q_max=float(q_values[-1]),
                        q_step=float(np.median(np.diff(q_values))),
                        use_solvent_subtracted_profile=bool(
                            self._active_contrast_settings is not None
                            or legacy_fourier_template.use_solvent_subtracted_profile
                        ),
                        log_q_axis=bool(legacy_fourier_template.log_q_axis),
                        log_intensity_axis=bool(
                            legacy_fourier_template.log_intensity_axis
                        ),
                    ).normalized(),
                )
                legacy_q_values = np.asarray(
                    legacy_result.q_values,
                    dtype=float,
                )
                legacy_intensities.append(
                    np.asarray(legacy_result.intensity, dtype=float)
                )
                legacy_elapsed_seconds += float(perf_counter() - legacy_start)
            if self._compare_exact_debye:
                self._raise_if_cancelled()
                debye_start = perf_counter()
                exact_debye_intensities.append(
                    np.asarray(
                        compute_constant_weight_debye_intensity(
                            coordinates,
                            weights,
                            q_values,
                        ),
                        dtype=float,
                    )
                )
                debye_elapsed_seconds += float(perf_counter() - debye_start)

        aggregated_fft_result = self._aggregate_fft_results(
            q_values,
            fft_results,
        )
        return _FFTProfileComputationResult(
            target=target,
            q_values=np.asarray(q_values, dtype=float),
            fft_result=aggregated_fft_result,
            legacy_q_values=(
                None
                if legacy_q_values is None
                else np.asarray(legacy_q_values, dtype=float)
            ),
            legacy_intensity=(
                None
                if not legacy_intensities
                else np.nanmean(np.vstack(legacy_intensities), axis=0)
            ),
            exact_debye_intensity=(
                None
                if not exact_debye_intensities
                else np.nanmean(np.vstack(exact_debye_intensities), axis=0)
            ),
            legacy_elapsed_seconds=(
                None
                if not legacy_intensities
                else legacy_elapsed_seconds / float(len(legacy_intensities))
            ),
            debye_elapsed_seconds=(
                None
                if not exact_debye_intensities
                else debye_elapsed_seconds
                / float(len(exact_debye_intensities))
            ),
        )

    def _aggregate_fft_results(
        self,
        q_values: np.ndarray,
        fft_results: list[ContrastFFTResult],
    ) -> ContrastFFTResult:
        if not fft_results:
            raise ValueError("No FFT results were available to aggregate.")
        if len(fft_results) == 1:
            return fft_results[0]
        first = fft_results[0]

        def _mean_array(values: list[np.ndarray]) -> np.ndarray:
            return np.nanmean(np.stack(values, axis=0), axis=0)

        def _mean_scalar(name: str) -> float:
            return float(
                np.mean(
                    [float(getattr(result, name)) for result in fft_results],
                    dtype=float,
                )
            )

        def _mean_triplet(name: str) -> tuple[float, float, float]:
            return tuple(
                float(
                    np.mean(
                        [
                            float(getattr(result, name)[axis])
                            for result in fft_results
                        ],
                        dtype=float,
                    )
                )
                for axis in range(3)
            )

        mean_shell_counts = np.rint(
            np.mean(
                [
                    np.asarray(result.q_shell_counts, dtype=float)
                    for result in fft_results
                ],
                axis=0,
            )
        ).astype(int)
        nonempty = np.flatnonzero(mean_shell_counts > 0)
        first_nonempty_q = (
            None
            if nonempty.size == 0
            else float(np.asarray(q_values, dtype=float)[int(nonempty[0])])
        )
        return ContrastFFTResult(
            settings=first.settings,
            q_values=np.asarray(q_values, dtype=float),
            raw_intensity=_mean_array(
                [
                    np.asarray(result.raw_intensity, dtype=float)
                    for result in fft_results
                ]
            ),
            kernel_corrected_intensity=_mean_array(
                [
                    np.asarray(
                        result.kernel_corrected_intensity,
                        dtype=float,
                    )
                    for result in fft_results
                ]
            ),
            q_shell_counts=np.asarray(mean_shell_counts, dtype=int),
            density_integral=_mean_scalar("density_integral"),
            expected_weight=_mean_scalar("expected_weight"),
            contrast_density_integral=_mean_scalar(
                "contrast_density_integral"
            ),
            expected_contrast_weight=_mean_scalar("expected_contrast_weight"),
            solvent_exclusion_volume_a3=_mean_scalar(
                "solvent_exclusion_volume_a3"
            ),
            grid_shape=first.grid_shape,
            box_lengths_a=_mean_triplet("box_lengths_a"),
            voxel_spacing_a=_mean_triplet("voxel_spacing_a"),
            q_nyquist_a_inverse=_mean_scalar("q_nyquist_a_inverse"),
            q_frequency_step_a_inverse=_mean_triplet(
                "q_frequency_step_a_inverse"
            ),
            q_convention=first.q_convention,
            uses_two_pi_frequency_conversion=bool(
                first.uses_two_pi_frequency_conversion
            ),
            density_subtraction_active=bool(first.density_subtraction_active),
            first_nonempty_q_a_inverse=first_nonempty_q,
            solvent_density_e_per_a3=_mean_scalar("solvent_density_e_per_a3"),
            contrast_mode=first.contrast_mode,
            kernel_correction_supported=all(
                bool(result.kernel_correction_supported)
                for result in fft_results
            ),
            kernel_correction_applied=all(
                bool(result.kernel_correction_applied)
                for result in fft_results
            ),
            kernel_correction_model=first.kernel_correction_model,
            timing=type(first.timing)(
                atomic_density_seconds=float(
                    np.mean(
                        [
                            result.timing.atomic_density_seconds
                            for result in fft_results
                        ],
                        dtype=float,
                    )
                ),
                contrast_density_seconds=float(
                    np.mean(
                        [
                            result.timing.contrast_density_seconds
                            for result in fft_results
                        ],
                        dtype=float,
                    )
                ),
                fft_seconds=float(
                    np.mean(
                        [result.timing.fft_seconds for result in fft_results],
                        dtype=float,
                    )
                ),
                shell_average_seconds=float(
                    np.mean(
                        [
                            result.timing.shell_average_seconds
                            for result in fft_results
                        ],
                        dtype=float,
                    )
                ),
                total_seconds=float(
                    np.mean(
                        [
                            result.timing.total_seconds
                            for result in fft_results
                        ],
                        dtype=float,
                    )
                ),
            ),
        )


class FFTBornApproximationMainWindow(QMainWindow):
    born_components_built = Signal(object)

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
        initial_use_representative_structures: bool = False,
        preview_mode: bool = True,
    ) -> None:
        super().__init__()
        self._preview_mode = bool(preview_mode)
        self._project_dir = initial_project_dir
        self._output_dir = initial_output_dir
        self._project_q_min = initial_project_q_min
        self._project_q_max = initial_project_q_max
        self._distribution_id = (
            None
            if initial_distribution_id is None
            else str(initial_distribution_id)
        )
        self._distribution_root_dir = initial_distribution_root_dir
        self._use_predicted_structure_weights = bool(
            initial_use_predicted_structure_weights
        )
        self._prefer_representative_structures = bool(
            initial_use_representative_structures
        )
        self._auto_snap_panes_enabled = self._load_auto_snap_setting()
        self._deferred_initial_input_path = initial_input_path
        self._loaded_input_path: Path | None = None
        self._loaded_reference_file: Path | None = None
        self._loaded_structure: ElectronDensityStructure | None = None
        self._loaded_structure_count = 0
        self._available_profile_targets: dict[
            tuple[str, str],
            tuple[_FFTProfileTarget, ...],
        ] = {}
        self._current_profile_targets: tuple[_FFTProfileTarget, ...] = ()
        self._reference_structure_cache: dict[
            str, ElectronDensityStructure
        ] = {}
        self._active_profile_key: str | None = None
        self._solvent_presets: dict[str, ContrastSolventPreset] = {}
        self._active_contrast_settings: (
            ContrastSolventDensitySettings | None
        ) = None
        self._active_contrast_estimate: (
            ContrastElectronDensityEstimate | None
        ) = None
        self._active_contrast_name: str | None = None
        self._active_solvent_density_e_per_a3 = 0.0
        self._current_payload: _FFTComputationPayload | None = None
        self._computed_profile_results: dict[
            str, _FFTProfileComputationResult
        ] = {}
        self._computed_profile_run_signature: dict[str, object] | None = None
        self._restoring_workspace_state = False
        self._contrast_controls_dirty = False
        self._curve_legend_visible = True
        self._compute_thread: QThread | None = None
        self._compute_worker: _FFTComputationWorker | None = None
        self._progress_dialog: SAXSProgressDialog | None = None
        self._close_requested_while_running = False
        self._build_ui()
        self._build_menu_bar()
        self._apply_preview_mode_title()
        self._refresh_preview_mode_banner()
        self._reload_solvent_presets(selected_name="Water")
        self._sync_density_method_controls()
        self._refresh_contrast_display()
        self._connect_trace_configuration_controls()
        self._update_curve_legend_button_text()
        self._sync_kernel_correction_option()
        self._update_push_to_model_state()
        if self._deferred_initial_input_path is not None:
            QTimer.singleShot(0, self._load_deferred_input)

    def _build_ui(self) -> None:
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
        left_container = QWidget(self)
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(8, 8, 8, 8)
        left_layout.setSpacing(10)
        self.preview_mode_banner = QLabel()
        self.preview_mode_banner.setWordWrap(True)
        self.preview_mode_banner.setStyleSheet(
            "QLabel { background: #f8fafc; border: 1px solid #cbd5e1; "
            "border-radius: 8px; padding: 10px; }"
        )
        left_layout.addWidget(self.preview_mode_banner)
        left_layout.addWidget(self._build_input_group())
        self.fft_settings_section = _CollapsibleSection(
            "3D FFT Settings",
            self._build_fft_settings_group(),
            self,
        )
        self.fft_settings_section.expand()
        left_layout.addWidget(self.fft_settings_section)
        self.legacy_1d_settings_section = _CollapsibleSection(
            "1D FFT Comparison Settings",
            self._build_legacy_1d_settings_group(),
            self,
        )
        left_layout.addWidget(self.legacy_1d_settings_section)
        self.contrast_section = _CollapsibleSection(
            "Electron Density Contrast",
            self._build_contrast_group(),
            self,
        )
        self.contrast_section.expand()
        left_layout.addWidget(self.contrast_section)
        self.comparison_section = _CollapsibleSection(
            "Comparison Overlays",
            self._build_comparison_group(),
            self,
        )
        self.comparison_section.expand()
        left_layout.addWidget(self.comparison_section)
        left_layout.addWidget(self._build_plot_options_group())
        left_layout.addWidget(self._build_actions_group())
        left_layout.addWidget(self._build_log_group(), stretch=1)
        left_layout.addStretch(1)
        self._left_scroll_area.setWidget(left_container)
        self._pane_splitter.addWidget(self._left_scroll_area)

        self._right_scroll_area = QScrollArea(self)
        self._right_scroll_area.setWidgetResizable(True)
        self._right_scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self._right_scroll_area.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        right_container = QWidget(self)
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(8, 8, 8, 8)
        right_layout.setSpacing(10)
        self.structure_viewer = ElectronDensityStructureViewer(self)
        structure_group = QGroupBox("Structure Viewer")
        structure_layout = QVBoxLayout(structure_group)
        structure_layout.addWidget(self.structure_viewer)
        right_layout.addWidget(structure_group)
        self.curve_plot = _FFTComparisonPlot(self)
        curve_group = QGroupBox("q-Space Curves")
        curve_layout = QVBoxLayout(curve_group)
        curve_controls = QHBoxLayout()
        curve_controls.setContentsMargins(0, 0, 0, 0)
        curve_controls.setSpacing(6)
        self.toggle_curve_legend_button = QPushButton("Hide Legend")
        self.toggle_curve_legend_button.clicked.connect(
            self._toggle_curve_legend
        )
        curve_controls.addWidget(self.toggle_curve_legend_button)
        self.export_curve_csv_button = QPushButton("Export Plot CSV")
        self.export_curve_csv_button.clicked.connect(
            self._export_q_space_curves_csv
        )
        curve_controls.addWidget(self.export_curve_csv_button)
        curve_controls.addStretch(1)
        curve_layout.addLayout(curve_controls)
        curve_layout.addWidget(self.curve_plot)
        right_layout.addWidget(curve_group)
        self.fft_box_visualizer = _FFTRealSpaceVisualizer(self)
        fft_visualizer_group = QGroupBox("FFT Real-Space Visualizer")
        fft_visualizer_layout = QVBoxLayout(fft_visualizer_group)
        fft_visualizer_layout.addWidget(self.fft_box_visualizer)
        right_layout.addWidget(fft_visualizer_group)
        self.shell_count_plot = _FFTShellCountPlot(self)
        shell_group = QGroupBox("FFT Shell Diagnostics")
        shell_layout = QVBoxLayout(shell_group)
        shell_layout.addWidget(self.shell_count_plot)
        right_layout.addWidget(shell_group)
        self.result_summary_box = QPlainTextEdit(self)
        self.result_summary_box.setReadOnly(True)
        self.result_summary_box.setMinimumHeight(180)
        summary_group = QGroupBox("Run Summary")
        summary_layout = QVBoxLayout(summary_group)
        summary_layout.addWidget(self.result_summary_box)
        right_layout.addWidget(summary_group)
        right_layout.addStretch(1)
        self._right_scroll_area.setWidget(right_container)
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

    def _build_input_group(self) -> QWidget:
        group = QGroupBox("Input")
        layout = QVBoxLayout(group)
        path_row = QHBoxLayout()
        self.input_path_edit = QLineEdit(self)
        path_row.addWidget(self.input_path_edit, stretch=1)
        browse_file_button = QPushButton("Open File...")
        browse_file_button.clicked.connect(self._browse_input_file)
        path_row.addWidget(browse_file_button)
        browse_folder_button = QPushButton("Open Folder...")
        browse_folder_button.clicked.connect(self._browse_input_folder)
        path_row.addWidget(browse_folder_button)
        layout.addLayout(path_row)
        load_row = QHBoxLayout()
        self.load_input_button = QPushButton("Load Input")
        self.load_input_button.clicked.connect(self._load_input_from_edit)
        load_row.addWidget(self.load_input_button)
        load_row.addStretch(1)
        layout.addLayout(load_row)
        source_grid = QGridLayout()
        source_grid.setHorizontalSpacing(8)
        source_grid.setVerticalSpacing(6)
        self.structure_source_combo = QComboBox(self)
        self.structure_source_combo.addItem(
            "Average cluster folders / input structures",
            "average",
        )
        self.structure_source_combo.addItem(
            "Representative structures",
            "representative",
        )
        self.structure_source_combo.currentIndexChanged.connect(
            self._on_structure_source_mode_changed
        )
        source_grid.addWidget(QLabel("Structure source"), 0, 0)
        source_grid.addWidget(self.structure_source_combo, 0, 1)
        self.representative_solvent_mode_combo = QComboBox(self)
        self.representative_solvent_mode_combo.currentIndexChanged.connect(
            self._refresh_available_profile_targets
        )
        source_grid.addWidget(QLabel("Representative solvent"), 1, 0)
        source_grid.addWidget(self.representative_solvent_mode_combo, 1, 1)
        self.active_profile_combo = QComboBox(self)
        self.active_profile_combo.currentIndexChanged.connect(
            self._on_active_profile_changed
        )
        source_grid.addWidget(QLabel("Active profile"), 2, 0)
        source_grid.addWidget(self.active_profile_combo, 2, 1)
        layout.addLayout(source_grid)
        self.structure_source_hint = QLabel(
            "Average structure mode is active until representative targets or cluster bins are detected."
        )
        self.structure_source_hint.setWordWrap(True)
        layout.addWidget(self.structure_source_hint)
        self.loaded_input_summary = QLabel(
            "Load a structure file or folder to inspect the 3D FFT Born setup."
        )
        self.loaded_input_summary.setWordWrap(True)
        layout.addWidget(self.loaded_input_summary)
        return group

    def _build_fft_settings_group(self) -> QWidget:
        panel = QWidget(self)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        default_settings = default_contrast_fft_settings()
        intro = QLabel(
            "These controls define the shared q grid and the Cartesian voxel grid "
            "used by the 3D FFT Born approximation. The q-range inherits from "
            "Project Setup when the window is launched from the main UI."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)
        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)
        tooltips = {
            "q_min": (
                "Lower bound of the shared q grid in Å^-1. When the 3D FFT "
                "window is launched from the main UI, this inherits the "
                "project q-range when that value is available."
            ),
            "q_max": (
                "Upper bound of the shared q grid in Å^-1. Keep this below the "
                "useful FFT Nyquist limit for the chosen voxel spacing."
            ),
            "q_step": (
                "Spacing between neighboring q samples in the comparison plot. "
                "This shared q grid is reused for the 3D FFT result and the "
                "optional 1D Born and Debye overlays."
            ),
            "spacing": (
                "Real-space voxel spacing for the Cartesian 3D density grid. "
                "Smaller spacing improves real-space detail and raises the FFT "
                "Nyquist limit, but increases memory and compute cost."
            ),
            "sigma": (
                "Gaussian deposition width used when mapping atomic electron "
                "density onto the voxel grid. Larger sigma smooths the real-space "
                "map and suppresses high-q scattering."
            ),
            "minimum_box_length": (
                "Minimum side length of the FFT box in Å before odd-grid "
                "rounding. Larger boxes improve low-q sampling because the FFT "
                "frequency spacing scales roughly as 2π/L."
            ),
            "padding": (
                "Extra vacuum padding added around the structure before the FFT "
                "box is built. Padding helps reduce boundary coupling and "
                "periodic-image contamination."
            ),
        }

        def _add_setting(
            row: int,
            label_text: str,
            widget: QWidget,
            tooltip: str,
            column: int = 0,
        ) -> None:
            label = QLabel(label_text)
            label.setToolTip(tooltip)
            widget.setToolTip(tooltip)
            grid.addWidget(label, row, column)
            grid.addWidget(widget, row, column + 1)

        self.q_min_spin = QDoubleSpinBox(self)
        self.q_min_spin.setRange(0.0, 100.0)
        self.q_min_spin.setDecimals(4)
        self.q_min_spin.setValue(
            0.01 if self._project_q_min is None else float(self._project_q_min)
        )
        self.q_max_spin = QDoubleSpinBox(self)
        self.q_max_spin.setRange(0.0, 100.0)
        self.q_max_spin.setDecimals(4)
        self.q_max_spin.setValue(
            1.20 if self._project_q_max is None else float(self._project_q_max)
        )
        self.q_step_spin = QDoubleSpinBox(self)
        self.q_step_spin.setRange(1.0e-4, 10.0)
        self.q_step_spin.setDecimals(4)
        self.q_step_spin.setValue(0.01)
        self.spacing_spin = QDoubleSpinBox(self)
        self.spacing_spin.setRange(0.1, 25.0)
        self.spacing_spin.setDecimals(3)
        self.spacing_spin.setValue(float(default_settings.spacing_a))
        self.sigma_spin = QDoubleSpinBox(self)
        self.sigma_spin.setRange(0.0, 25.0)
        self.sigma_spin.setDecimals(3)
        self.sigma_spin.setValue(float(default_settings.gaussian_sigma_a))
        self.min_box_length_spin = QDoubleSpinBox(self)
        self.min_box_length_spin.setRange(1.0, 5000.0)
        self.min_box_length_spin.setDecimals(3)
        self.min_box_length_spin.setValue(
            float(default_settings.minimum_box_length_a)
        )
        self.padding_spin = QDoubleSpinBox(self)
        self.padding_spin.setRange(0.0, 500.0)
        self.padding_spin.setDecimals(3)
        self.padding_spin.setValue(float(default_settings.padding_a))
        _add_setting(0, "q min (Å⁻¹)", self.q_min_spin, tooltips["q_min"])
        _add_setting(0, "q max (Å⁻¹)", self.q_max_spin, tooltips["q_max"], 2)
        _add_setting(1, "q step (Å⁻¹)", self.q_step_spin, tooltips["q_step"])
        _add_setting(
            1,
            "Voxel spacing (Å)",
            self.spacing_spin,
            tooltips["spacing"],
            2,
        )
        _add_setting(
            2,
            "Gaussian sigma (Å)",
            self.sigma_spin,
            tooltips["sigma"],
        )
        _add_setting(
            2,
            "Minimum box length (Å)",
            self.min_box_length_spin,
            tooltips["minimum_box_length"],
            2,
        )
        _add_setting(
            3,
            "Extra padding (Å)",
            self.padding_spin,
            tooltips["padding"],
        )
        layout.addLayout(grid)
        return panel

    def _build_legacy_1d_settings_group(self) -> QWidget:
        panel = QWidget(self)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        mesh_defaults = legacy_born_average_default_mesh_settings()
        smearing_defaults = legacy_born_average_default_smearing_settings()
        fourier_defaults = legacy_born_average_default_fourier_settings(
            r_max=float(mesh_defaults.rmax),
            q_min=float(
                0.01 if self._project_q_min is None else self._project_q_min
            ),
            q_max=float(
                1.20 if self._project_q_max is None else self._project_q_max
            ),
            q_step=0.01,
        )

        intro = QLabel(
            "These controls configure the legacy 1D Born overlay that can be "
            "plotted alongside the 3D FFT result. The q grid stays shared with "
            "the 3D FFT calculation, and any active solvent contrast is reused "
            "for the 1D q-space curve."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        mesh_grid = QGridLayout()
        mesh_grid.setHorizontalSpacing(8)
        mesh_grid.setVerticalSpacing(6)
        mesh_grid.setColumnStretch(1, 1)
        mesh_grid.setColumnStretch(3, 1)
        self.legacy_1d_rstep_spin = QDoubleSpinBox(self)
        self.legacy_1d_rstep_spin.setRange(0.01, 5.0)
        self.legacy_1d_rstep_spin.setDecimals(3)
        self.legacy_1d_rstep_spin.setValue(float(mesh_defaults.rstep))
        self.legacy_1d_theta_spin = QSpinBox(self)
        self.legacy_1d_theta_spin.setRange(4, 720)
        self.legacy_1d_theta_spin.setValue(int(mesh_defaults.theta_divisions))
        self.legacy_1d_phi_spin = QSpinBox(self)
        self.legacy_1d_phi_spin.setRange(4, 360)
        self.legacy_1d_phi_spin.setValue(int(mesh_defaults.phi_divisions))
        self.legacy_1d_rmax_spin = QDoubleSpinBox(self)
        self.legacy_1d_rmax_spin.setRange(0.01, 5000.0)
        self.legacy_1d_rmax_spin.setDecimals(3)
        self.legacy_1d_rmax_spin.setValue(float(mesh_defaults.rmax))
        mesh_grid.addWidget(QLabel("rstep (Å)"), 0, 0)
        mesh_grid.addWidget(self.legacy_1d_rstep_spin, 0, 1)
        mesh_grid.addWidget(QLabel("Theta divisions"), 0, 2)
        mesh_grid.addWidget(self.legacy_1d_theta_spin, 0, 3)
        mesh_grid.addWidget(QLabel("Phi divisions"), 1, 0)
        mesh_grid.addWidget(self.legacy_1d_phi_spin, 1, 1)
        mesh_grid.addWidget(QLabel("rmax (Å)"), 1, 2)
        mesh_grid.addWidget(self.legacy_1d_rmax_spin, 1, 3)
        layout.addLayout(mesh_grid)

        transform_grid = QGridLayout()
        transform_grid.setHorizontalSpacing(8)
        transform_grid.setVerticalSpacing(6)
        transform_grid.setColumnStretch(1, 1)
        transform_grid.setColumnStretch(3, 1)
        self.legacy_1d_smearing_factor_spin = QDoubleSpinBox(self)
        self.legacy_1d_smearing_factor_spin.setRange(0.0, 500.0)
        self.legacy_1d_smearing_factor_spin.setDecimals(6)
        self.legacy_1d_smearing_factor_spin.setValue(
            float(smearing_defaults.debye_waller_factor)
        )
        self.legacy_1d_domain_combo = QComboBox(self)
        self.legacy_1d_domain_combo.addItem("One-sided (legacy)", "legacy")
        self.legacy_1d_domain_combo.addItem("Mirrored", "mirrored")
        self.legacy_1d_domain_combo.setCurrentIndex(
            0 if str(fourier_defaults.domain_mode) == "legacy" else 1
        )
        self.legacy_1d_window_combo = QComboBox(self)
        for label, value in (
            ("None", "none"),
            ("Lorch", "lorch"),
            ("Cosine", "cosine"),
            ("Hanning", "hanning"),
            ("Parzen", "parzen"),
            ("Welch", "welch"),
            ("Gaussian", "gaussian"),
            ("Sine", "sine"),
            ("Kaiser-Bessel", "kaiser_bessel"),
        ):
            self.legacy_1d_window_combo.addItem(label, value)
            if value == str(fourier_defaults.window_function):
                self.legacy_1d_window_combo.setCurrentIndex(
                    self.legacy_1d_window_combo.count() - 1
                )
        self.legacy_1d_resampling_points_spin = QSpinBox(self)
        self.legacy_1d_resampling_points_spin.setRange(8, 200000)
        self.legacy_1d_resampling_points_spin.setValue(
            int(fourier_defaults.resampling_points)
        )
        self.legacy_1d_shared_q_note = QLabel(
            "The 1D overlay always uses the shared 3D FFT q range and q step for direct plot comparison."
        )
        self.legacy_1d_shared_q_note.setWordWrap(True)
        self.legacy_1d_contrast_note = QLabel(
            "No active solvent contrast is currently being reused by the 1D comparison curve."
        )
        self.legacy_1d_contrast_note.setWordWrap(True)
        transform_grid.addWidget(QLabel("Debye-Waller factor (Å²)"), 0, 0)
        transform_grid.addWidget(self.legacy_1d_smearing_factor_spin, 0, 1)
        transform_grid.addWidget(QLabel("Transform domain"), 0, 2)
        transform_grid.addWidget(self.legacy_1d_domain_combo, 0, 3)
        transform_grid.addWidget(QLabel("Window"), 1, 0)
        transform_grid.addWidget(self.legacy_1d_window_combo, 1, 1)
        transform_grid.addWidget(QLabel("Resample pts"), 1, 2)
        transform_grid.addWidget(self.legacy_1d_resampling_points_spin, 1, 3)
        layout.addLayout(transform_grid)
        layout.addWidget(self.legacy_1d_shared_q_note)
        layout.addWidget(self.legacy_1d_contrast_note)
        return panel

    def _build_contrast_group(self) -> QWidget:
        panel = QWidget(self)
        layout = QGridLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(6)
        layout.setColumnStretch(1, 1)

        intro = QLabel(
            "Set up an optional flat solvent electron-density subtraction for "
            "the 3D FFT Born approximation. These settings stay separate from "
            "the legacy 1D Born workflow and must be applied before the FFT run."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro, 0, 0, 1, 2)

        self.solvent_method_combo = QComboBox(self)
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
        layout.addWidget(QLabel("Compute option"), 1, 0)
        layout.addWidget(self.solvent_method_combo, 1, 1)

        self.solvent_preset_combo = QComboBox(self)
        self.solvent_preset_combo.currentIndexChanged.connect(
            self._load_selected_solvent_preset
        )
        self.save_custom_solvent_button = QPushButton("Save Custom")
        self.save_custom_solvent_button.clicked.connect(
            self._save_current_solvent_preset
        )
        self.delete_custom_solvent_button = QPushButton("Delete Custom")
        self.delete_custom_solvent_button.clicked.connect(
            self._delete_current_solvent_preset
        )
        solvent_row = QWidget(self)
        solvent_row_layout = QHBoxLayout(solvent_row)
        solvent_row_layout.setContentsMargins(0, 0, 0, 0)
        solvent_row_layout.setSpacing(6)
        solvent_row_layout.addWidget(self.solvent_preset_combo, stretch=1)
        solvent_row_layout.addWidget(self.save_custom_solvent_button)
        solvent_row_layout.addWidget(self.delete_custom_solvent_button)
        layout.addWidget(QLabel("Saved solvents"), 2, 0)
        layout.addWidget(solvent_row, 2, 1)

        self.solvent_formula_edit = QLineEdit(self)
        self.solvent_formula_edit.setPlaceholderText(
            "Examples: H2O, Vacuum, C3H7NO (DMF), C2H6OS (DMSO)"
        )
        layout.addWidget(QLabel("Solvent formula"), 3, 0)
        layout.addWidget(self.solvent_formula_edit, 3, 1)

        self.solvent_density_spin = QDoubleSpinBox(self)
        self.solvent_density_spin.setDecimals(6)
        self.solvent_density_spin.setRange(0.0, 100.0)
        self.solvent_density_spin.setSingleStep(0.01)
        self.solvent_density_spin.setValue(1.0)
        self.solvent_density_spin.setKeyboardTracking(False)
        layout.addWidget(QLabel("Density (g/mL)"), 4, 0)
        layout.addWidget(self.solvent_density_spin, 4, 1)

        self.direct_density_spin = QDoubleSpinBox(self)
        self.direct_density_spin.setDecimals(9)
        self.direct_density_spin.setRange(0.0, 100.0)
        self.direct_density_spin.setSingleStep(0.001)
        self.direct_density_spin.setValue(0.334)
        self.direct_density_spin.setKeyboardTracking(False)
        layout.addWidget(QLabel("Direct density (e-/Å³)"), 5, 0)
        layout.addWidget(self.direct_density_spin, 5, 1)

        self.reference_solvent_file_edit = QLineEdit(self)
        self.reference_solvent_file_edit.setPlaceholderText(
            "Choose a reference solvent XYZ or PDB file"
        )
        self.reference_solvent_browse_button = QPushButton("Browse…")
        self.reference_solvent_browse_button.clicked.connect(
            self._choose_reference_solvent_file
        )
        reference_row = QWidget(self)
        reference_layout = QHBoxLayout(reference_row)
        reference_layout.setContentsMargins(0, 0, 0, 0)
        reference_layout.setSpacing(6)
        reference_layout.addWidget(self.reference_solvent_file_edit, stretch=1)
        reference_layout.addWidget(self.reference_solvent_browse_button)
        layout.addWidget(QLabel("Reference solvent file"), 6, 0)
        layout.addWidget(reference_row, 6, 1)

        contrast_fft_defaults = default_contrast_fft_settings()
        self.exclusion_radius_scale_spin = QDoubleSpinBox(self)
        self.exclusion_radius_scale_spin.setRange(0.1, 10.0)
        self.exclusion_radius_scale_spin.setDecimals(3)
        self.exclusion_radius_scale_spin.setValue(
            float(contrast_fft_defaults.exclusion_radius_scale)
        )
        layout.addWidget(QLabel("Exclusion radius scale"), 7, 0)
        layout.addWidget(self.exclusion_radius_scale_spin, 7, 1)

        self.exclusion_radius_padding_spin = QDoubleSpinBox(self)
        self.exclusion_radius_padding_spin.setRange(0.0, 25.0)
        self.exclusion_radius_padding_spin.setDecimals(3)
        self.exclusion_radius_padding_spin.setValue(
            float(contrast_fft_defaults.exclusion_radius_padding_a)
        )
        layout.addWidget(QLabel("Exclusion radius padding (Å)"), 8, 0)
        layout.addWidget(self.exclusion_radius_padding_spin, 8, 1)

        self.solvent_method_hint_label = QLabel(self)
        self.solvent_method_hint_label.setWordWrap(True)
        layout.addWidget(self.solvent_method_hint_label, 9, 0, 1, 2)

        self.apply_contrast_button = QPushButton(
            "Apply Electron Density Contrast"
        )
        self.apply_contrast_button.clicked.connect(
            self._apply_contrast_settings
        )
        layout.addWidget(self.apply_contrast_button, 10, 0, 1, 2)

        self.active_contrast_value = QLabel(
            "No active solvent electron density contrast yet."
        )
        self.active_contrast_value.setWordWrap(True)
        layout.addWidget(QLabel("Active contrast"), 11, 0)
        layout.addWidget(self.active_contrast_value, 11, 1)

        self.contrast_notice_value = QLabel(
            "Apply contrast settings to use a solvent-density subtraction in the next 3D FFT Born calculation."
        )
        self.contrast_notice_value.setWordWrap(True)
        layout.addWidget(QLabel("Notes"), 12, 0)
        layout.addWidget(self.contrast_notice_value, 12, 1)
        return panel

    def _build_comparison_group(self) -> QWidget:
        panel = QWidget(self)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        self.compare_legacy_checkbox = QCheckBox(
            "Overlay 1D Born Approximation (Average)"
        )
        self.compare_legacy_checkbox.setChecked(True)
        self.compare_legacy_checkbox.toggled.connect(
            self._refresh_plot_controls
        )
        self.compare_exact_debye_checkbox = QCheckBox(
            "Overlay exact Debye scattering"
        )
        self.compare_exact_debye_checkbox.setChecked(False)
        self.compare_exact_debye_checkbox.toggled.connect(
            self._refresh_plot_controls
        )
        self.show_kernel_corrected_checkbox = QCheckBox(
            "Show kernel-corrected FFT overlay (zero-contrast diagnostic)"
        )
        self.show_kernel_corrected_checkbox.setChecked(False)
        self.show_kernel_corrected_checkbox.setToolTip(
            "Kernel correction removes the Gaussian voxel-deposition response "
            "from the zero-contrast FFT intensity so it can be compared against "
            "the point-scatterer Debye limit. It is not used for solvent-contrast production curves."
        )
        self.show_kernel_corrected_checkbox.toggled.connect(
            self._refresh_plot_controls
        )
        layout.addWidget(self.compare_legacy_checkbox)
        layout.addWidget(self.compare_exact_debye_checkbox)
        layout.addWidget(self.show_kernel_corrected_checkbox)
        return panel

    def _build_plot_options_group(self) -> QWidget:
        group = QGroupBox("Plot Options")
        layout = QVBoxLayout(group)
        self.log_q_checkbox = QCheckBox("Log q axis")
        self.log_q_checkbox.setChecked(True)
        self.log_q_checkbox.toggled.connect(self._refresh_plot_controls)
        self.log_intensity_checkbox = QCheckBox("Log intensity axis")
        self.log_intensity_checkbox.setChecked(True)
        self.log_intensity_checkbox.toggled.connect(
            self._refresh_plot_controls
        )
        layout.addWidget(self.log_q_checkbox)
        layout.addWidget(self.log_intensity_checkbox)
        return group

    def _build_actions_group(self) -> QWidget:
        group = QGroupBox("Actions")
        layout = QHBoxLayout(group)
        self.compute_button = QPushButton("Compute 3D FFT Born Approximation")
        self.compute_button.clicked.connect(self._start_calculation)
        layout.addWidget(self.compute_button)
        self.push_to_model_button = QPushButton("Push to Model")
        self.push_to_model_button.clicked.connect(
            self._push_components_to_model
        )
        layout.addWidget(self.push_to_model_button)
        self.clear_results_button = QPushButton("Clear Results")
        self.clear_results_button.clicked.connect(self._clear_results)
        layout.addWidget(self.clear_results_button)
        return group

    def _build_log_group(self) -> QWidget:
        group = QGroupBox("Status Log")
        layout = QVBoxLayout(group)
        self.status_log_box = QPlainTextEdit(self)
        self.status_log_box.setReadOnly(True)
        self.status_log_box.setMinimumHeight(180)
        layout.addWidget(self.status_log_box)
        return group

    def _connect_trace_configuration_controls(self) -> None:
        for spinbox in (
            self.q_min_spin,
            self.q_max_spin,
            self.q_step_spin,
            self.spacing_spin,
            self.sigma_spin,
            self.min_box_length_spin,
            self.padding_spin,
            self.exclusion_radius_scale_spin,
            self.exclusion_radius_padding_spin,
        ):
            spinbox.valueChanged.connect(self._on_trace_configuration_changed)
        for widget in (
            self.solvent_method_combo,
            self.solvent_preset_combo,
        ):
            widget.currentIndexChanged.connect(
                self._on_contrast_controls_changed
            )
        for widget in (
            self.solvent_formula_edit,
            self.reference_solvent_file_edit,
        ):
            widget.textChanged.connect(self._on_contrast_controls_changed)
        for spinbox in (
            self.solvent_density_spin,
            self.direct_density_spin,
        ):
            spinbox.valueChanged.connect(self._on_contrast_controls_changed)

    def _on_trace_configuration_changed(self, *_args: object) -> None:
        if self._restoring_workspace_state:
            return
        self._refresh_fft_box_visualizer()
        self._update_push_to_model_state()

    def _on_contrast_controls_changed(self, *_args: object) -> None:
        if self._restoring_workspace_state:
            return
        self._contrast_controls_dirty = True
        if self._computed_profile_results:
            self.contrast_notice_value.setText(
                "Electron density contrast controls changed. Apply the new "
                "contrast or recompute the 3D FFT traces before pushing these "
                "components to the model."
            )
            self.contrast_notice_value.setStyleSheet("color: #b45309;")
        self._update_push_to_model_state()

    def _legacy_1d_mesh_settings_from_controls(
        self,
    ) -> ElectronDensityMeshSettings:
        return ElectronDensityMeshSettings(
            rstep=float(self.legacy_1d_rstep_spin.value()),
            theta_divisions=int(self.legacy_1d_theta_spin.value()),
            phi_divisions=int(self.legacy_1d_phi_spin.value()),
            rmax=float(self.legacy_1d_rmax_spin.value()),
        ).normalized()

    def _legacy_1d_smearing_settings_from_controls(
        self,
    ) -> ElectronDensitySmearingSettings:
        return ElectronDensitySmearingSettings(
            debye_waller_factor=float(
                self.legacy_1d_smearing_factor_spin.value()
            )
        ).normalized()

    def _legacy_1d_fourier_settings_from_controls(
        self,
    ) -> ElectronDensityFourierTransformSettings:
        mesh_settings = self._legacy_1d_mesh_settings_from_controls()
        return ElectronDensityFourierTransformSettings(
            r_min=0.0,
            r_max=float(mesh_settings.rmax),
            domain_mode=str(
                self.legacy_1d_domain_combo.currentData() or "legacy"
            ),
            window_function=str(
                self.legacy_1d_window_combo.currentData() or "none"
            ),
            resampling_points=int(
                self.legacy_1d_resampling_points_spin.value()
            ),
            q_min=float(self.q_min_spin.value()),
            q_max=float(self.q_max_spin.value()),
            q_step=float(self.q_step_spin.value()),
            use_solvent_subtracted_profile=bool(
                self._active_contrast_settings is not None
            ),
            log_q_axis=bool(self.log_q_checkbox.isChecked()),
            log_intensity_axis=bool(self.log_intensity_checkbox.isChecked()),
        ).normalized()

    def _sync_legacy_1d_defaults_to_structure(
        self,
        structure: ElectronDensityStructure,
    ) -> None:
        mesh_defaults = legacy_born_average_default_mesh_settings(structure)
        self.legacy_1d_rstep_spin.setValue(float(mesh_defaults.rstep))
        self.legacy_1d_theta_spin.setValue(int(mesh_defaults.theta_divisions))
        self.legacy_1d_phi_spin.setValue(int(mesh_defaults.phi_divisions))
        self.legacy_1d_rmax_spin.setValue(float(mesh_defaults.rmax))

    def _refresh_legacy_1d_contrast_note(self) -> None:
        if self._active_contrast_settings is None:
            self.legacy_1d_contrast_note.setText(
                "No active solvent contrast is currently being reused by the 1D comparison curve."
            )
            self.legacy_1d_contrast_note.setStyleSheet("color: #475569;")
            return
        if abs(float(self._active_solvent_density_e_per_a3)) <= 1.0e-15:
            self.legacy_1d_contrast_note.setText(
                "The 1D comparison curve reuses the active zero-density contrast, so both 1D and 3D curves stay in the bare-density limit."
            )
            self.legacy_1d_contrast_note.setStyleSheet("color: #166534;")
            return
        self.legacy_1d_contrast_note.setText(
            "The 1D comparison curve reuses the active solvent-density subtraction before its Fourier transform so the q-space comparison stays contrast-matched."
        )
        self.legacy_1d_contrast_note.setStyleSheet("color: #166534;")

    @Slot()
    def _on_structure_source_mode_changed(self) -> None:
        self._prefer_representative_structures = (
            self._current_structure_source_mode() == "representative"
        )
        self._refresh_available_profile_targets()

    def _selected_solvent_preset_token(self) -> object:
        return self.solvent_preset_combo.currentData()

    def _selected_solvent_preset_name(self) -> str | None:
        token = self._selected_solvent_preset_token()
        if token in {None, _SOLVENT_PRESET_NONE}:
            return None
        return str(token).strip() or None

    def _reload_solvent_presets(self, *, selected_name: str | None) -> None:
        self._solvent_presets = load_solvent_presets()
        previous_name = (
            selected_name
            if selected_name is not None
            else self._selected_solvent_preset_name()
        )
        self.solvent_preset_combo.blockSignals(True)
        self.solvent_preset_combo.clear()
        self.solvent_preset_combo.addItem("Custom entry", None)
        self.solvent_preset_combo.addItem("None", _SOLVENT_PRESET_NONE)
        selected_index = 0
        if previous_name == _SOLVENT_PRESET_NONE:
            selected_index = 1
        for index, preset_name in enumerate(
            ordered_solvent_preset_names(self._solvent_presets),
            start=2,
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
        if self._selected_solvent_preset_token() == _SOLVENT_PRESET_NONE:
            self.delete_custom_solvent_button.setEnabled(False)
            self._sync_density_method_controls()
            return
        preset_name = self._selected_solvent_preset_name()
        preset = self._solvent_presets.get(preset_name or "")
        if preset is None:
            self.delete_custom_solvent_button.setEnabled(False)
            self._sync_density_method_controls()
            return
        self.solvent_formula_edit.setText(preset.formula)
        self.solvent_density_spin.setValue(preset.density_g_per_ml)
        self.delete_custom_solvent_button.setEnabled(not preset.builtin)
        self._sync_density_method_controls()

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
            QMessageBox.warning(self, "Save Custom Solvent", str(exc))
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

    def _clear_solvent_contrast_requested_from_controls(self) -> bool:
        method = str(self.solvent_method_combo.currentData() or "").strip()
        return (
            method == CONTRAST_SOLVENT_METHOD_NEAT
            and self._selected_solvent_preset_token() == _SOLVENT_PRESET_NONE
        )

    @Slot()
    def _sync_density_method_controls(self) -> None:
        method = str(self.solvent_method_combo.currentData() or "").strip()
        using_neat = method == CONTRAST_SOLVENT_METHOD_NEAT
        using_reference = method == CONTRAST_SOLVENT_METHOD_REFERENCE
        using_direct = method == CONTRAST_SOLVENT_METHOD_DIRECT
        for widget in (
            self.solvent_preset_combo,
            self.solvent_formula_edit,
            self.solvent_density_spin,
            self.save_custom_solvent_button,
        ):
            widget.setEnabled(using_neat)
        self.delete_custom_solvent_button.setEnabled(
            using_neat
            and self._selected_solvent_preset_name() is not None
            and not self._solvent_presets.get(
                self._selected_solvent_preset_name() or "",
                ContrastSolventPreset("", "Vacuum", 0.0, builtin=True),
            ).builtin
        )
        self.reference_solvent_file_edit.setEnabled(using_reference)
        self.reference_solvent_browse_button.setEnabled(using_reference)
        self.direct_density_spin.setEnabled(using_direct)
        if using_reference:
            self.solvent_method_hint_label.setText(
                "Reference structure mode estimates a uniform solvent electron density "
                "from the full XYZ/PDB coordinate box spanned by the selected file."
            )
        elif using_direct:
            self.solvent_method_hint_label.setText(
                "Direct value mode uses the electron density you provide in e-/Å³. "
                "Use 0.0 e-/Å³ to model vacuum without solvent subtraction."
            )
        elif self._clear_solvent_contrast_requested_from_controls():
            self.solvent_method_hint_label.setText(
                "The None solvent option clears the active solvent-density subtraction "
                "and returns the 3D FFT workflow to bare atomic density."
            )
        else:
            self.solvent_method_hint_label.setText(
                "Quick estimate mode uses the selected solvent stoichiometry and density. "
                "Built-in presets include Water, Vacuum, DMF, and DMSO."
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

    def _estimate_reference_solvent_density(
        self,
        settings: ContrastSolventDensitySettings,
    ) -> ContrastElectronDensityEstimate:
        reference_file = settings.reference_structure_file
        if reference_file is None:
            raise ValueError(
                "Choose a reference solvent XYZ or PDB file before computing the solvent electron density."
            )
        structure = load_electron_density_structure(reference_file)
        coordinates = np.asarray(structure.coordinates, dtype=float)
        spans = np.max(coordinates, axis=0) - np.min(coordinates, axis=0)
        volume_a3 = float(np.prod(spans))
        if volume_a3 <= 0.0:
            raise ValueError(
                "The reference solvent structure must span a positive 3D coordinate box."
            )
        total_electrons = float(np.sum(structure.atomic_numbers, dtype=float))
        density_e_per_a3 = total_electrons / volume_a3
        return ContrastElectronDensityEstimate(
            method=CONTRAST_SOLVENT_METHOD_REFERENCE,
            label="Reference solvent structure",
            volume_a3=volume_a3,
            total_electrons=total_electrons,
            electron_density_e_per_a3=float(density_e_per_a3),
            electron_density_e_per_cm3=float(
                density_e_per_a3 * ANGSTROM3_PER_CM3
            ),
            atom_count=structure.atom_count,
            element_counts=dict(sorted(structure.element_counts.items())),
            reference_structure_file=reference_file,
            reference_box_spans=(
                float(spans[0]),
                float(spans[1]),
                float(spans[2]),
            ),
        )

    def _estimate_solvent_density(
        self,
        settings: ContrastSolventDensitySettings,
    ) -> ContrastElectronDensityEstimate:
        if settings.method == CONTRAST_SOLVENT_METHOD_REFERENCE:
            return self._estimate_reference_solvent_density(settings)
        if settings.method == CONTRAST_SOLVENT_METHOD_DIRECT:
            return _direct_solvent_electron_density(settings, volume_a3=1.0)
        return _neat_solvent_electron_density(settings, volume_a3=1.0)

    @Slot()
    def _apply_contrast_settings(self) -> None:
        self._apply_contrast_settings_from_controls(announce=True)

    def _apply_contrast_settings_from_controls(
        self,
        *,
        announce: bool,
    ) -> bool:
        if self._clear_solvent_contrast_requested_from_controls():
            self._active_contrast_settings = None
            self._active_contrast_estimate = None
            self._active_contrast_name = None
            self._active_solvent_density_e_per_a3 = 0.0
            self._contrast_controls_dirty = False
            self._sync_kernel_correction_option()
            self._refresh_contrast_display()
            self._refresh_fft_box_visualizer()
            self._update_push_to_model_state()
            if announce:
                self._append_status(
                    "Cleared the active solvent-density subtraction for the 3D FFT Born workflow."
                )
                self.statusBar().showMessage("Cleared 3D FFT solvent contrast")
            return True
        try:
            self._active_contrast_settings = (
                self._contrast_settings_from_controls()
            )
            self._active_contrast_estimate = self._estimate_solvent_density(
                self._active_contrast_settings
            )
            self._active_contrast_name = (
                self._contrast_display_name_from_controls()
            )
            self._active_solvent_density_e_per_a3 = float(
                self._active_contrast_estimate.electron_density_e_per_a3
            )
        except Exception as exc:
            self._show_error("Solvent Contrast Error", str(exc))
            return False
        self._contrast_controls_dirty = False
        self._sync_kernel_correction_option()
        self._refresh_contrast_display()
        self._refresh_fft_box_visualizer()
        self._update_push_to_model_state()
        if announce:
            self._append_status(
                "Applied solvent-density setup "
                f"{self._active_contrast_name or 'Solvent'} at "
                f"{self._active_solvent_density_e_per_a3:.6f} e/Å³."
            )
            self.statusBar().showMessage("Applied 3D FFT solvent contrast")
        return True

    def _refresh_contrast_display(self) -> None:
        estimate = self._active_contrast_estimate
        if estimate is None:
            self.active_contrast_value.setText(
                "No active solvent electron density contrast. The 3D FFT run will use bare atomic density only."
            )
            self.active_contrast_value.setStyleSheet("color: #475569;")
            self.contrast_notice_value.setText(
                "Apply contrast settings to enable constant solvent-density subtraction inside the atomic exclusion mask."
            )
            self.contrast_notice_value.setStyleSheet("color: #475569;")
            self._refresh_legacy_1d_contrast_note()
            return
        active_name = self._active_contrast_name or estimate.label
        self.active_contrast_value.setText(
            f"{active_name}: {estimate.electron_density_e_per_a3:.6f} e/Å³"
        )
        if abs(float(estimate.electron_density_e_per_a3)) <= 1.0e-15:
            self.active_contrast_value.setStyleSheet("color: #166534;")
            self.contrast_notice_value.setText(
                "This is effectively a zero-contrast run, so the kernel-corrected FFT overlay can be used as a point-scatterer diagnostic."
            )
            self.contrast_notice_value.setStyleSheet("color: #166534;")
            self._refresh_legacy_1d_contrast_note()
            return
        self.active_contrast_value.setStyleSheet("color: #166534;")
        self.contrast_notice_value.setText(
            "The next 3D FFT Born calculation will subtract this constant solvent electron density inside the union of atomic exclusion spheres."
        )
        self.contrast_notice_value.setStyleSheet("color: #166534;")
        self._refresh_legacy_1d_contrast_note()

    def _active_contrast_summary_for_visualizer(self) -> str | None:
        estimate = self._active_contrast_estimate
        if estimate is None:
            return None
        if abs(float(estimate.electron_density_e_per_a3)) <= 1.0e-15:
            return "Active contrast: zero-density / vacuum"
        return (
            f"Active contrast: {estimate.electron_density_e_per_a3:.4f} e/Å³, "
            f"scale={self.exclusion_radius_scale_spin.value():.3f}, "
            f"padding={self.exclusion_radius_padding_spin.value():.3f} Å"
        )

    def _make_profile_target_key(
        self,
        *,
        structure_name: str,
        motif_name: str,
        source_mode: str,
        solvent_mode: str,
    ) -> str:
        return (
            f"{source_mode}|{solvent_mode}|{structure_name.strip()}|"
            f"{motif_name.strip() or 'no_motif'}"
        )

    def _project_source(self):
        if self._project_dir is None:
            return None
        try:
            return load_rmc_project_source(self._project_dir)
        except Exception:
            return None

    def _project_average_input_path(self) -> Path | None:
        if self._project_dir is None:
            return None
        try:
            from saxshell.saxs.project_manager.project import (
                SAXSProjectManager,
            )

            settings = SAXSProjectManager().load_project(self._project_dir)
        except Exception:
            return None
        clusters_dir = settings.resolved_clusters_dir
        if clusters_dir is not None and clusters_dir.exists():
            return clusters_dir
        return None

    def _cluster_targets_from_input_path(
        self,
        input_path: Path,
    ) -> tuple[_FFTProfileTarget, ...]:
        try:
            cluster_bins = discover_cluster_bins(input_path)
        except Exception:
            cluster_bins = []
        if cluster_bins:
            return tuple(
                _FFTProfileTarget(
                    key=self._make_profile_target_key(
                        structure_name=cluster_bin.structure,
                        motif_name=cluster_bin.motif,
                        source_mode="average",
                        solvent_mode="input",
                    ),
                    display_name=(
                        cluster_bin.structure
                        if cluster_bin.motif == "no_motif"
                        else f"{cluster_bin.structure}/{cluster_bin.motif}"
                    ),
                    structure_name=cluster_bin.structure,
                    motif_name=cluster_bin.motif,
                    file_count=len(cluster_bin.files),
                    reference_file=cluster_bin.files[0],
                    source_files=tuple(cluster_bin.files),
                    representative=cluster_bin.representative,
                    source_mode="average",
                    solvent_mode="input",
                )
                for cluster_bin in cluster_bins
            )
        inspection = inspect_structure_input(input_path)
        structure_name = (
            input_path.stem if input_path.is_file() else input_path.name
        )
        return (
            _FFTProfileTarget(
                key=self._make_profile_target_key(
                    structure_name=structure_name,
                    motif_name="no_motif",
                    source_mode="average",
                    solvent_mode="input",
                ),
                display_name=structure_name,
                structure_name=structure_name,
                motif_name="no_motif",
                file_count=int(inspection.total_files),
                reference_file=inspection.reference_file,
                source_files=tuple(inspection.structure_files),
                representative=inspection.reference_file.name,
                source_mode="average",
                solvent_mode="input",
            ),
        )

    def _representative_targets_from_project_source(
        self,
    ) -> dict[tuple[str, str], tuple[_FFTProfileTarget, ...]]:
        def source_solvent_mode(entry) -> str:
            normalized = str(
                getattr(entry, "source_solvent_mode", "") or ""
            ).strip()
            if normalized == "nosolv":
                return "none"
            if normalized == "fullsolv":
                return "full"
            return "partial"

        project_source = self._project_source()
        if project_source is None:
            return {}
        targets: dict[tuple[str, str], list[_FFTProfileTarget]] = {}
        metadata = project_source.representative_selection
        solvent_metadata = project_source.solvent_handling
        if metadata is not None:
            source_entries: list[tuple[object, Path]] = []
            source_modes: set[str] = set()
            for entry in metadata.representative_entries:
                source_file = Path(entry.source_file).expanduser().resolve()
                if not source_file.is_file():
                    continue
                source_entries.append((entry, source_file))
                source_modes.add(source_solvent_mode(entry))
            if source_entries:
                aggregate_mode = (
                    next(iter(source_modes))
                    if len(source_modes) == 1
                    else "partial"
                )
                source_targets: list[_FFTProfileTarget] = []
                for entry, source_file in source_entries:
                    source_targets.append(
                        _FFTProfileTarget(
                            key=self._make_profile_target_key(
                                structure_name=entry.structure,
                                motif_name=entry.motif,
                                source_mode="representative",
                                solvent_mode=aggregate_mode,
                            ),
                            display_name=(
                                entry.structure
                                if entry.motif == "no_motif"
                                else f"{entry.structure}/{entry.motif}"
                            ),
                            structure_name=entry.structure,
                            motif_name=entry.motif,
                            file_count=1,
                            reference_file=source_file,
                            source_files=(source_file,),
                            representative=(
                                entry.source_file_name or source_file.name
                            ),
                            source_mode="representative",
                            solvent_mode=aggregate_mode,
                        )
                    )
                has_built_mode_overlap = bool(
                    solvent_metadata is not None
                    and solvent_metadata.entries
                    and aggregate_mode in {"none", "full"}
                )
                if not has_built_mode_overlap:
                    targets[("representative", aggregate_mode)] = (
                        source_targets
                    )
        if solvent_metadata is not None and solvent_metadata.entries:
            no_solvent_targets: list[_FFTProfileTarget] = []
            full_solvent_targets: list[_FFTProfileTarget] = []
            for entry in solvent_metadata.entries:
                no_solvent_path = (
                    Path(entry.no_solvent_pdb).expanduser().resolve()
                )
                completed_path = (
                    Path(entry.completed_pdb).expanduser().resolve()
                )
                display_name = (
                    entry.structure
                    if entry.motif == "no_motif"
                    else f"{entry.structure}/{entry.motif}"
                )
                if no_solvent_path.is_file():
                    no_solvent_targets.append(
                        _FFTProfileTarget(
                            key=self._make_profile_target_key(
                                structure_name=entry.structure,
                                motif_name=entry.motif,
                                source_mode="representative",
                                solvent_mode="none",
                            ),
                            display_name=display_name,
                            structure_name=entry.structure,
                            motif_name=entry.motif,
                            file_count=1,
                            reference_file=no_solvent_path,
                            source_files=(no_solvent_path,),
                            representative=no_solvent_path.name,
                            source_mode="representative",
                            solvent_mode="none",
                        )
                    )
                if completed_path.is_file():
                    full_solvent_targets.append(
                        _FFTProfileTarget(
                            key=self._make_profile_target_key(
                                structure_name=entry.structure,
                                motif_name=entry.motif,
                                source_mode="representative",
                                solvent_mode="full",
                            ),
                            display_name=display_name,
                            structure_name=entry.structure,
                            motif_name=entry.motif,
                            file_count=1,
                            reference_file=completed_path,
                            source_files=(completed_path,),
                            representative=completed_path.name,
                            source_mode="representative",
                            solvent_mode="full",
                        )
                    )
            if no_solvent_targets:
                targets[("representative", "none")] = no_solvent_targets
            if full_solvent_targets:
                targets[("representative", "full")] = full_solvent_targets
        return {key: tuple(value) for key, value in targets.items() if value}

    def _resolve_available_profile_targets(
        self,
        input_path: Path,
    ) -> dict[tuple[str, str], tuple[_FFTProfileTarget, ...]]:
        targets: dict[tuple[str, str], tuple[_FFTProfileTarget, ...]] = {}
        average_error: Exception | None = None
        average_input_candidates: list[Path] = []
        project_average_input = self._project_average_input_path()
        if project_average_input is not None:
            average_input_candidates.append(project_average_input)
        if not any(
            candidate.resolve() == input_path.resolve()
            for candidate in average_input_candidates
        ):
            average_input_candidates.append(input_path)
        for average_input in average_input_candidates:
            try:
                average_targets = self._cluster_targets_from_input_path(
                    average_input
                )
            except Exception as exc:
                if average_error is None:
                    average_error = exc
                continue
            if average_targets:
                targets[("average", "input")] = average_targets
                break
        targets.update(self._representative_targets_from_project_source())
        if not targets and average_error is not None:
            raise average_error
        return targets

    def _current_structure_source_mode(self) -> str:
        return str(self.structure_source_combo.currentData() or "average")

    def _current_representative_solvent_mode(self) -> str:
        return str(
            self.representative_solvent_mode_combo.currentData() or "partial"
        )

    def _active_profile_target(self) -> _FFTProfileTarget | None:
        if not self._current_profile_targets:
            return None
        for target in self._current_profile_targets:
            if target.key == self._active_profile_key:
                return target
        return self._current_profile_targets[0]

    def _load_structure_for_target(
        self,
        target: _FFTProfileTarget | None,
    ) -> ElectronDensityStructure | None:
        if target is None:
            return None
        cache_key = str(target.reference_file)
        cached = self._reference_structure_cache.get(cache_key)
        if cached is not None:
            return cached
        try:
            structure = load_electron_density_structure(target.reference_file)
        except Exception:
            return None
        self._reference_structure_cache[cache_key] = structure
        return structure

    def _set_active_profile_target(
        self,
        target: _FFTProfileTarget | None,
    ) -> None:
        self._active_profile_key = None if target is None else target.key
        if target is None:
            self._loaded_reference_file = None
            self._loaded_structure = None
            self._loaded_structure_count = 0
            self._refresh_fft_box_visualizer()
            return
        self._loaded_reference_file = target.reference_file
        self._loaded_structure_count = int(target.file_count)
        self._loaded_structure = self._load_structure_for_target(target)
        if self._loaded_structure is not None:
            preview_mesh = self._build_preview_mesh_geometry(
                self._loaded_structure
            )
            self._sync_legacy_1d_defaults_to_structure(self._loaded_structure)
            self.structure_viewer.set_structure(
                self._loaded_structure,
                mesh_geometry=preview_mesh,
                scene_key=str(target.reference_file),
            )
            self.structure_viewer.mesh_contrast_spin.setValue(90.0)
            self.structure_viewer.mesh_linewidth_spin.setValue(1.6)
        self._update_loaded_input_summary()
        self._refresh_fft_box_visualizer()
        self._refresh_plot_controls()
        self._sync_workspace_state()

    def _update_loaded_input_summary(self) -> None:
        if self._loaded_input_path is None:
            self.loaded_input_summary.setText(
                "Load a structure file or folder to inspect the 3D FFT Born setup."
            )
            return
        target = self._active_profile_target()
        if target is None:
            self.loaded_input_summary.setText(
                f"Loaded input from {self._loaded_input_path}, but no eligible 3D FFT profile targets were detected."
            )
            return
        self.loaded_input_summary.setText(
            f"Loaded {len(self._current_profile_targets)} active 3D FFT profile "
            f"target(s) from {self._loaded_input_path.name}.\n"
            f"Selected profile: {target.display_name} using "
            f"{target.file_count} structure file(s) from "
            f"{target.source_mode} / {target.solvent_mode} mode."
        )

    def _populate_active_profile_combo(self) -> None:
        self.active_profile_combo.blockSignals(True)
        self.active_profile_combo.clear()
        selected_index = 0
        for index, target in enumerate(self._current_profile_targets):
            label = (
                f"{target.display_name} ({target.file_count} file"
                f"{'' if target.file_count == 1 else 's'})"
            )
            self.active_profile_combo.addItem(label, target.key)
            if target.key == self._active_profile_key:
                selected_index = index
        self.active_profile_combo.setEnabled(
            bool(self._current_profile_targets)
        )
        if self._current_profile_targets:
            self.active_profile_combo.setCurrentIndex(selected_index)
        self.active_profile_combo.blockSignals(False)

    @Slot()
    def _refresh_available_profile_targets(self) -> None:
        if self._loaded_input_path is None:
            self.structure_source_combo.setEnabled(False)
            self.representative_solvent_mode_combo.setEnabled(False)
            self.active_profile_combo.setEnabled(False)
            return
        available_sources = {
            source_mode
            for source_mode, _solvent_mode in self._available_profile_targets
        }
        self.structure_source_combo.blockSignals(True)
        self.structure_source_combo.setEnabled(bool(available_sources))
        preferred_source_mode = (
            "representative"
            if self._prefer_representative_structures
            and "representative" in available_sources
            else "average"
        )
        current_source_mode = self._current_structure_source_mode()
        if (
            not self._current_profile_targets
            and preferred_source_mode in available_sources
        ):
            current_source_mode = preferred_source_mode
        elif current_source_mode not in available_sources:
            current_source_mode = preferred_source_mode
        target_source_index = self.structure_source_combo.findData(
            current_source_mode
        )
        if target_source_index < 0:
            target_source_index = self.structure_source_combo.findData(
                "average"
            )
        self.structure_source_combo.setCurrentIndex(
            max(target_source_index, 0)
        )
        self.structure_source_combo.blockSignals(False)

        representative_modes = [
            solvent_mode
            for source_mode, solvent_mode in self._available_profile_targets
            if source_mode == "representative"
        ]
        representative_labels = {
            "none": "No solvent representative",
            "partial": "Partial / source representative",
            "full": "Full solvent representative",
        }
        current_rep_mode = self._current_representative_solvent_mode()
        if current_rep_mode not in representative_modes:
            current_rep_mode = (
                "partial"
                if "partial" in representative_modes
                else (
                    representative_modes[0]
                    if representative_modes
                    else "partial"
                )
            )
        self.representative_solvent_mode_combo.blockSignals(True)
        self.representative_solvent_mode_combo.clear()
        if representative_modes:
            for solvent_mode in ("none", "partial", "full"):
                if solvent_mode not in representative_modes:
                    continue
                self.representative_solvent_mode_combo.addItem(
                    representative_labels[solvent_mode],
                    solvent_mode,
                )
            rep_index = self.representative_solvent_mode_combo.findData(
                current_rep_mode
            )
            self.representative_solvent_mode_combo.setCurrentIndex(
                max(rep_index, 0)
            )
        else:
            self.representative_solvent_mode_combo.addItem(
                "No representative solvent variants available",
                None,
            )
            self.representative_solvent_mode_combo.setCurrentIndex(0)
        using_representatives = (
            self._current_structure_source_mode() == "representative"
            and bool(representative_modes)
        )
        self.representative_solvent_mode_combo.setEnabled(
            using_representatives
        )
        self.representative_solvent_mode_combo.blockSignals(False)

        selected_mode = self._current_structure_source_mode()
        selected_solvent_mode = (
            self._current_representative_solvent_mode()
            if selected_mode == "representative"
            else "input"
        )
        self._current_profile_targets = self._available_profile_targets.get(
            (selected_mode, selected_solvent_mode),
            (),
        )
        if self._active_profile_key is None or all(
            target.key != self._active_profile_key
            for target in self._current_profile_targets
        ):
            self._active_profile_key = (
                None
                if not self._current_profile_targets
                else self._current_profile_targets[0].key
            )
        self._populate_active_profile_combo()
        if selected_mode == "representative" and representative_modes:
            self.structure_source_hint.setText(
                "Representative mode is active. Choose no-solvent, partial/source, or full-solvent representative files when those project artifacts are available."
            )
        elif selected_mode == "representative":
            self.structure_source_hint.setText(
                "Representative mode was requested, but no saved representative structures are currently available for this project."
            )
        else:
            self.structure_source_hint.setText(
                "Average structure mode is active. Each profile target averages all structure files in its active cluster bin or loaded input folder."
            )
        self._set_active_profile_target(self._active_profile_target())
        self._update_push_to_model_state()

    @Slot()
    def _on_active_profile_changed(self) -> None:
        target_key = str(self.active_profile_combo.currentData() or "").strip()
        if not target_key:
            self._set_active_profile_target(None)
            return
        for target in self._current_profile_targets:
            if target.key == target_key:
                self._set_active_profile_target(target)
                return
        self._set_active_profile_target(self._active_profile_target())

    def _active_profile_result(self) -> _FFTProfileComputationResult | None:
        if self._active_profile_key is None:
            return None
        return self._computed_profile_results.get(self._active_profile_key)

    def _legacy_overlay_label(self) -> str:
        if self._active_contrast_settings is None:
            return "1D Born Approximation (Average)"
        return "1D Born Approximation (Average, matched contrast)"

    def _refresh_fft_box_visualizer(self) -> None:
        active_result = self._active_profile_result()
        self.fft_box_visualizer.set_structure_preview(
            self._loaded_structure,
            fft_result=(
                None if active_result is None else active_result.fft_result
            ),
            contrast_summary=self._active_contrast_summary_for_visualizer(),
        )

    def _update_curve_legend_button_text(self) -> None:
        self.toggle_curve_legend_button.setText(
            "Hide Legend" if self._curve_legend_visible else "Show Legend"
        )

    @Slot()
    def _toggle_curve_legend(self) -> None:
        self._curve_legend_visible = not self._curve_legend_visible
        self._update_curve_legend_button_text()
        self._refresh_plot_controls()

    def _current_curve_series(
        self,
    ) -> list[tuple[str, np.ndarray, np.ndarray]]:
        active_result = self._active_profile_result()
        if active_result is None:
            return []
        primary_label = (
            "3D FFT Born Approximation (solvent contrast)"
            if active_result.fft_result.density_subtraction_active
            else "3D FFT Born Approximation"
        )
        series: list[tuple[str, np.ndarray, np.ndarray]] = [
            (
                primary_label,
                np.asarray(active_result.q_values, dtype=float),
                np.asarray(
                    active_result.fft_result.raw_intensity, dtype=float
                ),
            )
        ]
        if (
            self.show_kernel_corrected_checkbox.isChecked()
            and active_result.fft_result.kernel_correction_supported
        ):
            series.append(
                (
                    "3D FFT kernel-corrected (diagnostic)",
                    np.asarray(active_result.q_values, dtype=float),
                    np.asarray(
                        active_result.fft_result.kernel_corrected_intensity,
                        dtype=float,
                    ),
                )
            )
        if (
            self.compare_legacy_checkbox.isChecked()
            and active_result.legacy_q_values is not None
            and active_result.legacy_intensity is not None
        ):
            series.append(
                (
                    self._legacy_overlay_label(),
                    np.asarray(active_result.legacy_q_values, dtype=float),
                    np.asarray(active_result.legacy_intensity, dtype=float),
                )
            )
        if (
            self.compare_exact_debye_checkbox.isChecked()
            and active_result.exact_debye_intensity is not None
        ):
            series.append(
                (
                    "Exact Debye scattering",
                    np.asarray(active_result.q_values, dtype=float),
                    np.asarray(
                        active_result.exact_debye_intensity, dtype=float
                    ),
                )
            )
        return series

    @Slot()
    def _export_q_space_curves_csv(self) -> None:
        series = self._current_curve_series()
        if not series:
            self._show_error(
                "No q-Space Data",
                "Run the 3D FFT Born approximation before exporting plot data.",
            )
            return
        start_dir = str(self._output_dir or self._project_dir or Path.cwd())
        selected_path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export q-Space Curves CSV",
            str(Path(start_dir) / "fft_born_q_space_curves.csv"),
            "CSV files (*.csv);;All files (*)",
        )
        if not selected_path:
            return
        path = Path(selected_path).expanduser().resolve()
        max_length = max(
            int(q_values.size) for _label, q_values, _intensity in series
        )
        header: list[str] = []
        for label, _q_values, _intensity in series:
            prefix = (
                str(label)
                .lower()
                .replace(" ", "_")
                .replace(",", "")
                .replace("(", "")
                .replace(")", "")
                .replace("/", "_")
                .replace("-", "_")
            )
            header.extend([f"{prefix}_q_a_inverse", f"{prefix}_intensity"])
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            for index in range(max_length):
                row: list[object] = []
                for _label, q_values, intensity in series:
                    if index < q_values.size:
                        row.append(f"{float(q_values[index]):.8f}")
                        row.append(f"{float(intensity[index]):.8e}")
                    else:
                        row.extend(["", ""])
                writer.writerow(row)
        self._append_status(f"Exported q-space plot data CSV to {path}.")
        self.statusBar().showMessage("Exported q-space plot data CSV")

    def _append_status(self, message: str) -> None:
        text = str(message).strip()
        if not text:
            return
        self.status_log_box.appendPlainText(text)
        self.statusBar().showMessage(text)

    def _apply_preview_mode_title(self) -> None:
        title = "3D FFT Born Approximation"
        if self._preview_mode:
            title += " (Preview)"
        self.setWindowTitle(title)

    def _refresh_preview_mode_banner(self) -> None:
        if self._preview_mode:
            self.preview_mode_banner.setText(
                "Preview Mode: inspect a structure file or folder and compare the "
                "3D FFT Born approximation against optional legacy 1D Born and exact Debye references."
            )
            self.preview_mode_banner.setToolTip(
                "Preview mode does not push 3D FFT Born outputs into an active SAXS model."
            )
            return
        distribution_text = (
            self._distribution_id
            if self._distribution_id is not None
            else "active computed distribution"
        )
        self.preview_mode_banner.setText(
            "Computed Distribution Mode: this run is linked to "
            f"{distribution_text}. Use this window to evaluate the separate 3D FFT Born workflow before full component-export integration."
        )
        self.preview_mode_banner.setToolTip(
            "This window was launched from Build SAXS Components using the 3D FFT Born Approximation mode."
        )

    def _browse_input_file(self) -> None:
        selected, _filter = QFileDialog.getOpenFileName(
            self,
            "Open Structure File",
            str(self._project_dir or Path.home()),
            "Structure Files (*.pdb *.xyz);;All Files (*)",
        )
        if selected:
            self.input_path_edit.setText(selected)

    def _browse_input_folder(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "Open Structure Folder",
            str(self._project_dir or Path.home()),
        )
        if selected:
            self.input_path_edit.setText(selected)

    @Slot()
    def _load_input_from_edit(self) -> None:
        text = str(self.input_path_edit.text()).strip()
        if not text:
            self._show_error(
                "Input required",
                "Choose a structure file or folder before loading the 3D FFT Born input.",
            )
            return
        self._load_input_path(Path(text).expanduser().resolve())

    def _load_input_path(self, path: Path) -> None:
        try:
            available_targets = self._resolve_available_profile_targets(path)
        except Exception as exc:
            self._show_error(
                "Load failed",
                f"Could not load 3D FFT Born input targets:\n{exc}",
            )
            return
        if not available_targets:
            self._show_error(
                "Load failed",
                "No eligible 3D FFT Born profile targets were discovered for the selected input.",
            )
            return
        self._loaded_input_path = path
        self._available_profile_targets = dict(available_targets)
        self._current_payload = None
        self._computed_profile_results = {}
        self._computed_profile_run_signature = None
        self.curve_plot.draw_placeholder()
        self.shell_count_plot.draw_placeholder()
        self.result_summary_box.clear()
        self._restoring_workspace_state = True
        try:
            self._refresh_available_profile_targets()
        finally:
            self._restoring_workspace_state = False
        self._restore_workspace_state_from_disk()
        self._update_push_to_model_state()
        self._append_status(
            f"Loaded 3D FFT Born input from {path} with "
            f"{sum(len(targets) for targets in self._available_profile_targets.values())} "
            "discovered profile target(s)."
        )

    def _build_preview_mesh_geometry(
        self,
        structure: ElectronDensityStructure,
    ):
        return build_electron_density_mesh(
            structure,
            legacy_born_average_default_mesh_settings(structure),
        )

    def _load_deferred_input(self) -> None:
        if self._deferred_initial_input_path is None:
            return
        self.input_path_edit.setText(str(self._deferred_initial_input_path))
        self._load_input_path(self._deferred_initial_input_path)
        self._deferred_initial_input_path = None

    def _current_fft_settings(self) -> ContrastFFTSettings:
        return ContrastFFTSettings(
            spacing_a=float(self.spacing_spin.value()),
            gaussian_sigma_a=float(self.sigma_spin.value()),
            minimum_box_length_a=float(self.min_box_length_spin.value()),
            padding_a=float(self.padding_spin.value()),
            solvent_density_e_per_a3=float(
                self._active_solvent_density_e_per_a3
            ),
            exclusion_radius_scale=float(
                self.exclusion_radius_scale_spin.value()
            ),
            exclusion_radius_padding_a=float(
                self.exclusion_radius_padding_spin.value()
            ),
        )

    @staticmethod
    def _signature_float(value: object) -> float:
        return round(float(value), 12)

    def _current_trace_configuration_signature(self) -> dict[str, object]:
        fft_settings = self._current_fft_settings().normalized()
        return {
            "target_keys": [
                target.key for target in self._current_profile_targets
            ],
            "structure_source_mode": self._current_structure_source_mode(),
            "representative_solvent_mode": (
                self._current_representative_solvent_mode()
                if self._current_structure_source_mode() == "representative"
                else "input"
            ),
            "q_min": self._signature_float(self.q_min_spin.value()),
            "q_max": self._signature_float(self.q_max_spin.value()),
            "q_step": self._signature_float(self.q_step_spin.value()),
            "fft_settings": {
                "spacing_a": self._signature_float(fft_settings.spacing_a),
                "gaussian_sigma_a": self._signature_float(
                    fft_settings.gaussian_sigma_a
                ),
                "minimum_box_length_a": self._signature_float(
                    fft_settings.minimum_box_length_a
                ),
                "padding_a": self._signature_float(fft_settings.padding_a),
                "support_sigma": self._signature_float(
                    fft_settings.support_sigma
                ),
                "solvent_density_e_per_a3": self._signature_float(
                    fft_settings.solvent_density_e_per_a3
                ),
                "exclusion_radius_scale": self._signature_float(
                    fft_settings.exclusion_radius_scale
                ),
                "exclusion_radius_padding_a": self._signature_float(
                    fft_settings.exclusion_radius_padding_a
                ),
                "use_cubic_box": bool(fft_settings.use_cubic_box),
            },
            "active_contrast_settings": (
                None
                if self._active_contrast_settings is None
                else self._active_contrast_settings.to_dict()
            ),
            "active_solvent_density_e_per_a3": self._signature_float(
                self._active_solvent_density_e_per_a3
            ),
        }

    def _requested_q_values_from_controls(self) -> np.ndarray:
        return build_shared_q_grid(
            float(self.q_min_spin.value()),
            float(self.q_max_spin.value()),
            q_step=float(self.q_step_spin.value()),
        )

    def _computed_profile_q_grid_matches_requested_grid(self) -> bool:
        if not self._computed_profile_results:
            return False
        try:
            requested_q_values = self._requested_q_values_from_controls()
        except Exception:
            return False
        for result in self._computed_profile_results.values():
            q_values = np.asarray(result.q_values, dtype=float)
            if q_values.shape != requested_q_values.shape:
                return False
            if not np.allclose(
                q_values,
                requested_q_values,
                rtol=1.0e-10,
                atol=1.0e-12,
            ):
                return False
        return True

    def _results_match_current_configuration(self) -> bool:
        if not self._computed_profile_results:
            return False
        if self._contrast_controls_dirty:
            return False
        run_signature = self._computed_profile_run_signature
        if run_signature is None:
            return True
        return (
            run_signature == self._current_trace_configuration_signature()
            and self._computed_profile_q_grid_matches_requested_grid()
        )

    @Slot()
    def _start_calculation(self) -> None:
        if (
            self._compute_thread is not None
            and self._compute_thread.isRunning()
        ):
            return
        if not self._current_profile_targets:
            self._show_error(
                "No input loaded",
                "Load a structure file or folder with at least one active "
                "profile target before running the 3D FFT Born approximation.",
            )
            return
        q_min = float(self.q_min_spin.value())
        q_max = float(self.q_max_spin.value())
        q_step = float(self.q_step_spin.value())
        if q_max <= q_min:
            self._show_error(
                "Invalid q range",
                "q max must be greater than q min before starting the "
                "3D FFT Born calculation.",
            )
            return
        if self._contrast_controls_dirty and not (
            self._apply_contrast_settings_from_controls(announce=True)
        ):
            return
        self.compute_button.setEnabled(False)
        self.load_input_button.setEnabled(False)
        self._close_requested_while_running = False
        self._begin_progress_dialog("Running 3D FFT Born Approximation...")
        self._append_status("Starting 3D FFT Born approximation calculation.")
        self._compute_thread = QThread(self)
        self._compute_worker = _FFTComputationWorker(
            targets=tuple(self._current_profile_targets),
            fft_settings=self._current_fft_settings(),
            legacy_mesh_settings=self._legacy_1d_mesh_settings_from_controls(),
            legacy_smearing_settings=(
                self._legacy_1d_smearing_settings_from_controls()
            ),
            legacy_fourier_settings=(
                self._legacy_1d_fourier_settings_from_controls()
            ),
            active_contrast_settings=self._active_contrast_settings,
            active_contrast_name=self._active_contrast_name,
            q_min=q_min,
            q_max=q_max,
            q_step=q_step,
            compare_legacy_1d=self.compare_legacy_checkbox.isChecked(),
            compare_exact_debye=self.compare_exact_debye_checkbox.isChecked(),
        )
        self._compute_worker.moveToThread(self._compute_thread)
        self._compute_thread.started.connect(self._compute_worker.run)
        self._compute_worker.progress.connect(self._on_worker_progress)
        self._compute_worker.finished.connect(self._on_worker_finished)
        self._compute_worker.failed.connect(self._on_worker_failed)
        self._compute_worker.cancelled.connect(self._on_worker_cancelled)
        self._compute_worker.finished.connect(self._compute_thread.quit)
        self._compute_worker.failed.connect(self._compute_thread.quit)
        self._compute_worker.cancelled.connect(self._compute_thread.quit)
        self._compute_thread.finished.connect(self._compute_worker.deleteLater)
        self._compute_thread.finished.connect(self._compute_thread.deleteLater)
        self._compute_thread.finished.connect(self._clear_worker_handles)
        self._compute_thread.finished.connect(
            self._finalize_close_after_worker
        )
        self._compute_thread.start(QThread.Priority.LowPriority)

    @Slot(str)
    def _on_worker_progress(self, message: str) -> None:
        self._append_status(message)
        self._set_progress_message(message)

    @Slot(object)
    def _on_worker_finished(self, payload: object) -> None:
        self.compute_button.setEnabled(True)
        self.load_input_button.setEnabled(True)
        self._close_progress_dialog()
        if not isinstance(payload, _FFTComputationPayload):
            self._show_error(
                "Calculation failed",
                "The 3D FFT Born calculation returned an unexpected payload.",
            )
            return
        self._current_payload = payload
        self._computed_profile_results = {
            result.target.key: result for result in payload.profile_results
        }
        self._computed_profile_run_signature = (
            self._current_trace_configuration_signature()
        )
        if (
            self._active_profile_key is None
            or self._active_profile_key not in self._computed_profile_results
        ) and payload.profile_results:
            self._active_profile_key = payload.profile_results[0].target.key
        self._refresh_plot_controls()
        self._refresh_fft_box_visualizer()
        self._update_result_summary(payload)
        active_result = self._active_profile_result()
        if active_result is not None:
            self._append_status(
                "3D FFT Nyquist limit: "
                f"{active_result.fft_result.q_nyquist_a_inverse:.6f} Å^-1 using voxel spacing "
                f"{active_result.fft_result.voxel_spacing_a[0]:.3f} Å."
            )
        self._sync_workspace_state()
        self._update_push_to_model_state()
        self._append_status("3D FFT Born approximation calculation complete.")

    @Slot(str)
    def _on_worker_failed(self, message: str) -> None:
        self.compute_button.setEnabled(True)
        self.load_input_button.setEnabled(True)
        self._close_progress_dialog()
        if self._close_requested_while_running:
            return
        self._show_error(
            "Calculation failed",
            "The 3D FFT Born approximation could not be completed:\n"
            + str(message).strip(),
        )

    @Slot(str)
    def _on_worker_cancelled(self, message: str) -> None:
        self.compute_button.setEnabled(True)
        self.load_input_button.setEnabled(True)
        self._close_progress_dialog()
        if self._close_requested_while_running:
            return
        text = str(message).strip() or "3D FFT Born calculation cancelled."
        self._append_status(text)
        self.statusBar().showMessage(text)

    @Slot()
    def _clear_worker_handles(self) -> None:
        self._compute_worker = None
        self._compute_thread = None

    @Slot()
    def _finalize_close_after_worker(self) -> None:
        if not self._close_requested_while_running:
            return
        self._close_progress_dialog()
        self.hide()
        self.deleteLater()

    def _update_result_summary(self, payload: _FFTComputationPayload) -> None:
        active_result = self._active_profile_result()
        if active_result is None:
            self.result_summary_box.clear()
            return
        fft = active_result.fft_result
        lines = [
            ("Computed profiles: " f"{len(payload.profile_results)}"),
            ("Active profile: " f"{active_result.target.display_name}"),
            f"Input: {self._loaded_reference_file or 'None'}",
            f"Atoms: {0 if self._loaded_structure is None else self._loaded_structure.atom_count}",
            f"Target file count: {active_result.target.file_count}",
            (
                "Solvent density contrast: "
                f"{fft.solvent_density_e_per_a3:.6f} e/Å³"
            ),
            f"Contrast mode: {fft.contrast_mode}",
            f"Grid shape: {'x'.join(str(value) for value in fft.grid_shape)}",
            (
                "Box lengths (Å): "
                + ", ".join(f"{value:.3f}" for value in fft.box_lengths_a)
            ),
            f"q Nyquist: {fft.q_nyquist_a_inverse:.6f} Å⁻¹",
            (
                "First non-empty q bin: "
                + (
                    "None"
                    if fft.first_nonempty_q_a_inverse is None
                    else f"{fft.first_nonempty_q_a_inverse:.4f} Å⁻¹"
                )
            ),
            (
                "Atomic density integral / expected weight: "
                f"{fft.density_integral:.6f} / {fft.expected_weight:.6f}"
            ),
            (
                "Contrast density integral / expected contrast weight: "
                f"{fft.contrast_density_integral:.6f} / {fft.expected_contrast_weight:.6f}"
            ),
            (
                "Timing (s): deposit="
                f"{fft.timing.atomic_density_seconds:.4f}, contrast="
                f"{fft.timing.contrast_density_seconds:.4f}, fft="
                f"{fft.timing.fft_seconds:.4f}, shells="
                f"{fft.timing.shell_average_seconds:.4f}, total="
                f"{fft.timing.total_seconds:.4f}"
            ),
        ]
        if active_result.legacy_elapsed_seconds is not None:
            lines.append(
                "Legacy 1D Born comparison time (s): "
                f"{active_result.legacy_elapsed_seconds:.4f}"
            )
            if self._active_contrast_settings is not None:
                lines.append(
                    "Legacy 1D Born overlay reused the active solvent contrast before its Fourier transform."
                )
        if active_result.debye_elapsed_seconds is not None:
            lines.append(
                f"Exact Debye comparison time (s): {active_result.debye_elapsed_seconds:.4f}"
            )
        if fft.kernel_correction_supported:
            lines.append(
                "Kernel correction is available for this zero-contrast run as a diagnostic overlay."
            )
        else:
            lines.append(
                "Kernel correction is disabled because solvent-contrast subtraction is active."
            )
        self.result_summary_box.setPlainText("\n".join(lines))

    @Slot()
    def _refresh_plot_controls(self) -> None:
        active_result = self._active_profile_result()
        if active_result is None:
            self.curve_plot.draw_placeholder()
            self.shell_count_plot.draw_placeholder()
            self._refresh_fft_box_visualizer()
            return
        additional_series: list[dict[str, object]] = []
        if (
            self.show_kernel_corrected_checkbox.isChecked()
            and active_result.fft_result.kernel_correction_supported
        ):
            additional_series.append(
                {
                    "q_values": active_result.q_values,
                    "intensity": active_result.fft_result.kernel_corrected_intensity,
                    "label": "3D FFT kernel-corrected (diagnostic)",
                    "color": "#7c3aed",
                    "linestyle": "--",
                }
            )
        if (
            self.compare_legacy_checkbox.isChecked()
            and active_result.legacy_q_values is not None
            and active_result.legacy_intensity is not None
        ):
            additional_series.append(
                {
                    "q_values": active_result.legacy_q_values,
                    "intensity": active_result.legacy_intensity,
                    "label": self._legacy_overlay_label(),
                    "color": "#b45309",
                    "linestyle": "-.",
                }
            )
        if (
            self.compare_exact_debye_checkbox.isChecked()
            and active_result.exact_debye_intensity is not None
        ):
            additional_series.append(
                {
                    "q_values": active_result.q_values,
                    "intensity": active_result.exact_debye_intensity,
                    "label": "Exact Debye scattering",
                    "color": "#0f172a",
                    "linestyle": ":",
                    "linewidth": 1.9,
                }
            )
        primary_label = (
            "3D FFT Born Approximation (solvent contrast)"
            if active_result.fft_result.density_subtraction_active
            else "3D FFT Born Approximation"
        )
        self.curve_plot.set_curves(
            q_values=active_result.q_values,
            primary_values=active_result.fft_result.raw_intensity,
            primary_label=primary_label,
            additional_series=additional_series,
            log_q_axis=self.log_q_checkbox.isChecked(),
            log_intensity_axis=self.log_intensity_checkbox.isChecked(),
            show_legend=self._curve_legend_visible,
        )
        self.shell_count_plot.set_counts(
            active_result.q_values,
            active_result.fft_result.q_shell_counts,
        )
        self._refresh_fft_box_visualizer()

    def _sync_kernel_correction_option(self) -> None:
        solvent_density = float(self._active_solvent_density_e_per_a3)
        enabled = abs(solvent_density) <= 1.0e-15
        self.show_kernel_corrected_checkbox.setEnabled(enabled)
        if not enabled:
            self.show_kernel_corrected_checkbox.setChecked(False)

    @Slot()
    def _clear_results(self) -> None:
        self._current_payload = None
        self._computed_profile_results = {}
        self._computed_profile_run_signature = None
        self.curve_plot.draw_placeholder()
        self.shell_count_plot.draw_placeholder()
        self.result_summary_box.clear()
        self._refresh_fft_box_visualizer()
        self._sync_workspace_state()
        self._update_push_to_model_state()
        self._append_status("Cleared 3D FFT Born outputs.")

    def _state_dir(self) -> Path | None:
        if self._output_dir is not None:
            return Path(self._output_dir).expanduser().resolve()
        if self._distribution_root_dir is None:
            return None
        return (
            Path(self._distribution_root_dir).expanduser().resolve()
            / "born_approximation_3d_fft"
        )

    def _workspace_state_path(self) -> Path | None:
        state_dir = self._state_dir()
        if state_dir is None:
            return None
        return state_dir / "workspace_state.json"

    def _component_summary_path(self) -> Path | None:
        state_dir = self._state_dir()
        if state_dir is None:
            return None
        return state_dir / "born_approximation_3d_fft_component_summary.json"

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

    def _serialize_fft_result(
        self,
        result: ContrastFFTResult,
    ) -> dict[str, object]:
        return {
            "settings": {
                "spacing_a": float(result.settings.spacing_a),
                "gaussian_sigma_a": float(result.settings.gaussian_sigma_a),
                "minimum_box_length_a": float(
                    result.settings.minimum_box_length_a
                ),
                "padding_a": float(result.settings.padding_a),
                "support_sigma": float(result.settings.support_sigma),
                "solvent_density_e_per_a3": float(
                    result.settings.solvent_density_e_per_a3
                ),
                "exclusion_radius_scale": float(
                    result.settings.exclusion_radius_scale
                ),
                "exclusion_radius_padding_a": float(
                    result.settings.exclusion_radius_padding_a
                ),
                "use_cubic_box": bool(result.settings.use_cubic_box),
            },
            "q_values": np.asarray(result.q_values, dtype=float).tolist(),
            "raw_intensity": np.asarray(
                result.raw_intensity,
                dtype=float,
            ).tolist(),
            "kernel_corrected_intensity": np.asarray(
                result.kernel_corrected_intensity,
                dtype=float,
            ).tolist(),
            "q_shell_counts": np.asarray(
                result.q_shell_counts,
                dtype=int,
            ).tolist(),
            "density_integral": float(result.density_integral),
            "expected_weight": float(result.expected_weight),
            "contrast_density_integral": float(
                result.contrast_density_integral
            ),
            "expected_contrast_weight": float(result.expected_contrast_weight),
            "solvent_exclusion_volume_a3": float(
                result.solvent_exclusion_volume_a3
            ),
            "grid_shape": [int(value) for value in result.grid_shape],
            "box_lengths_a": [float(value) for value in result.box_lengths_a],
            "voxel_spacing_a": [
                float(value) for value in result.voxel_spacing_a
            ],
            "q_nyquist_a_inverse": float(result.q_nyquist_a_inverse),
            "q_frequency_step_a_inverse": [
                float(value) for value in result.q_frequency_step_a_inverse
            ],
            "q_convention": str(result.q_convention),
            "uses_two_pi_frequency_conversion": bool(
                result.uses_two_pi_frequency_conversion
            ),
            "density_subtraction_active": bool(
                result.density_subtraction_active
            ),
            "first_nonempty_q_a_inverse": (
                None
                if result.first_nonempty_q_a_inverse is None
                else float(result.first_nonempty_q_a_inverse)
            ),
            "solvent_density_e_per_a3": float(result.solvent_density_e_per_a3),
            "contrast_mode": str(result.contrast_mode),
            "kernel_correction_supported": bool(
                result.kernel_correction_supported
            ),
            "kernel_correction_applied": bool(
                result.kernel_correction_applied
            ),
            "kernel_correction_model": result.kernel_correction_model,
            "timing": {
                "atomic_density_seconds": float(
                    result.timing.atomic_density_seconds
                ),
                "contrast_density_seconds": float(
                    result.timing.contrast_density_seconds
                ),
                "fft_seconds": float(result.timing.fft_seconds),
                "shell_average_seconds": float(
                    result.timing.shell_average_seconds
                ),
                "total_seconds": float(result.timing.total_seconds),
            },
        }

    def _deserialize_fft_result(
        self,
        payload: dict[str, object],
    ) -> ContrastFFTResult:
        settings_payload = dict(payload.get("settings", {}))
        settings = ContrastFFTSettings(
            spacing_a=float(settings_payload.get("spacing_a", 2.5)),
            gaussian_sigma_a=float(
                settings_payload.get("gaussian_sigma_a", 0.75)
            ),
            minimum_box_length_a=float(
                settings_payload.get("minimum_box_length_a", 640.0)
            ),
            padding_a=float(settings_payload.get("padding_a", 24.0)),
            support_sigma=float(settings_payload.get("support_sigma", 4.0)),
            solvent_density_e_per_a3=float(
                settings_payload.get("solvent_density_e_per_a3", 0.0)
            ),
            exclusion_radius_scale=float(
                settings_payload.get("exclusion_radius_scale", 1.0)
            ),
            exclusion_radius_padding_a=float(
                settings_payload.get("exclusion_radius_padding_a", 0.0)
            ),
            use_cubic_box=bool(settings_payload.get("use_cubic_box", True)),
        ).normalized()
        timing_payload = dict(payload.get("timing", {}))
        return ContrastFFTResult(
            settings=settings,
            q_values=np.asarray(payload.get("q_values", []), dtype=float),
            raw_intensity=np.asarray(
                payload.get("raw_intensity", []),
                dtype=float,
            ),
            kernel_corrected_intensity=np.asarray(
                payload.get("kernel_corrected_intensity", []),
                dtype=float,
            ),
            q_shell_counts=np.asarray(
                payload.get("q_shell_counts", []),
                dtype=int,
            ),
            density_integral=float(payload.get("density_integral", 0.0)),
            expected_weight=float(payload.get("expected_weight", 0.0)),
            contrast_density_integral=float(
                payload.get("contrast_density_integral", 0.0)
            ),
            expected_contrast_weight=float(
                payload.get("expected_contrast_weight", 0.0)
            ),
            solvent_exclusion_volume_a3=float(
                payload.get("solvent_exclusion_volume_a3", 0.0)
            ),
            grid_shape=tuple(
                int(value) for value in payload.get("grid_shape", (1, 1, 1))
            ),
            box_lengths_a=tuple(
                float(value)
                for value in payload.get("box_lengths_a", (0.0, 0.0, 0.0))
            ),
            voxel_spacing_a=tuple(
                float(value)
                for value in payload.get("voxel_spacing_a", (0.0, 0.0, 0.0))
            ),
            q_nyquist_a_inverse=float(payload.get("q_nyquist_a_inverse", 0.0)),
            q_frequency_step_a_inverse=tuple(
                float(value)
                for value in payload.get(
                    "q_frequency_step_a_inverse",
                    (0.0, 0.0, 0.0),
                )
            ),
            q_convention=str(payload.get("q_convention", "")).strip(),
            uses_two_pi_frequency_conversion=bool(
                payload.get("uses_two_pi_frequency_conversion", True)
            ),
            density_subtraction_active=bool(
                payload.get("density_subtraction_active", False)
            ),
            first_nonempty_q_a_inverse=(
                None
                if payload.get("first_nonempty_q_a_inverse") is None
                else float(payload.get("first_nonempty_q_a_inverse", 0.0))
            ),
            solvent_density_e_per_a3=float(
                payload.get("solvent_density_e_per_a3", 0.0)
            ),
            contrast_mode=str(payload.get("contrast_mode", "")).strip(),
            kernel_correction_supported=bool(
                payload.get("kernel_correction_supported", False)
            ),
            kernel_correction_applied=bool(
                payload.get("kernel_correction_applied", False)
            ),
            kernel_correction_model=payload.get("kernel_correction_model"),
            timing=ContrastFFTTiming(
                atomic_density_seconds=float(
                    timing_payload.get("atomic_density_seconds", 0.0)
                ),
                contrast_density_seconds=float(
                    timing_payload.get("contrast_density_seconds", 0.0)
                ),
                fft_seconds=float(timing_payload.get("fft_seconds", 0.0)),
                shell_average_seconds=float(
                    timing_payload.get("shell_average_seconds", 0.0)
                ),
                total_seconds=float(timing_payload.get("total_seconds", 0.0)),
            ),
        )

    def _serialize_profile_result(
        self,
        result: _FFTProfileComputationResult,
    ) -> dict[str, object]:
        return {
            "target": {
                "key": result.target.key,
                "display_name": result.target.display_name,
                "structure_name": result.target.structure_name,
                "motif_name": result.target.motif_name,
                "file_count": int(result.target.file_count),
                "reference_file": str(result.target.reference_file),
                "source_files": [
                    str(path) for path in result.target.source_files
                ],
                "representative": result.target.representative,
                "source_mode": result.target.source_mode,
                "solvent_mode": result.target.solvent_mode,
            },
            "q_values": np.asarray(result.q_values, dtype=float).tolist(),
            "fft_result": self._serialize_fft_result(result.fft_result),
            "legacy_q_values": (
                None
                if result.legacy_q_values is None
                else np.asarray(result.legacy_q_values, dtype=float).tolist()
            ),
            "legacy_intensity": (
                None
                if result.legacy_intensity is None
                else np.asarray(result.legacy_intensity, dtype=float).tolist()
            ),
            "exact_debye_intensity": (
                None
                if result.exact_debye_intensity is None
                else np.asarray(
                    result.exact_debye_intensity,
                    dtype=float,
                ).tolist()
            ),
            "legacy_elapsed_seconds": result.legacy_elapsed_seconds,
            "debye_elapsed_seconds": result.debye_elapsed_seconds,
        }

    def _deserialize_profile_result(
        self,
        payload: dict[str, object],
    ) -> _FFTProfileComputationResult:
        target_payload = dict(payload.get("target", {}))
        target = _FFTProfileTarget(
            key=str(target_payload.get("key", "")).strip(),
            display_name=str(target_payload.get("display_name", "")).strip(),
            structure_name=str(
                target_payload.get("structure_name", "")
            ).strip(),
            motif_name=str(
                target_payload.get("motif_name", "no_motif")
            ).strip()
            or "no_motif",
            file_count=int(target_payload.get("file_count", 1)),
            reference_file=Path(
                str(target_payload.get("reference_file", "")).strip()
            )
            .expanduser()
            .resolve(),
            source_files=tuple(
                Path(str(path)).expanduser().resolve()
                for path in target_payload.get("source_files", [])
            ),
            representative=(
                None
                if target_payload.get("representative") is None
                else str(target_payload.get("representative", "")).strip()
            ),
            source_mode=str(
                target_payload.get("source_mode", "average")
            ).strip()
            or "average",
            solvent_mode=str(
                target_payload.get("solvent_mode", "input")
            ).strip()
            or "input",
        )
        return _FFTProfileComputationResult(
            target=target,
            q_values=np.asarray(payload.get("q_values", []), dtype=float),
            fft_result=self._deserialize_fft_result(
                dict(payload.get("fft_result", {}))
            ),
            legacy_q_values=(
                None
                if payload.get("legacy_q_values") is None
                else np.asarray(
                    payload.get("legacy_q_values", []), dtype=float
                )
            ),
            legacy_intensity=(
                None
                if payload.get("legacy_intensity") is None
                else np.asarray(
                    payload.get("legacy_intensity", []), dtype=float
                )
            ),
            exact_debye_intensity=(
                None
                if payload.get("exact_debye_intensity") is None
                else np.asarray(
                    payload.get("exact_debye_intensity", []),
                    dtype=float,
                )
            ),
            legacy_elapsed_seconds=(
                None
                if payload.get("legacy_elapsed_seconds") is None
                else float(payload.get("legacy_elapsed_seconds", 0.0))
            ),
            debye_elapsed_seconds=(
                None
                if payload.get("debye_elapsed_seconds") is None
                else float(payload.get("debye_elapsed_seconds", 0.0))
            ),
        )

    def _build_component_summary_payload(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "distribution_id": self._distribution_id,
            "project_dir": (
                None if self._project_dir is None else str(self._project_dir)
            ),
            "input_path": (
                None
                if self._loaded_input_path is None
                else str(self._loaded_input_path)
            ),
            "structure_source_mode": self._current_structure_source_mode(),
            "prefer_representative_structures": bool(
                self._prefer_representative_structures
            ),
            "representative_solvent_mode": (
                self._current_representative_solvent_mode()
            ),
            "active_profile_key": self._active_profile_key,
            "active_contrast_name": self._active_contrast_name,
            "active_contrast_settings": (
                None
                if self._active_contrast_settings is None
                else self._active_contrast_settings.to_dict()
            ),
            "active_contrast_estimate": (
                None
                if self._active_contrast_estimate is None
                else self._active_contrast_estimate.to_dict()
            ),
            "active_solvent_density_e_per_a3": float(
                self._active_solvent_density_e_per_a3
            ),
            "q_min": float(self.q_min_spin.value()),
            "q_max": float(self.q_max_spin.value()),
            "q_step": float(self.q_step_spin.value()),
            "fft_settings": {
                "spacing_a": float(self.spacing_spin.value()),
                "gaussian_sigma_a": float(self.sigma_spin.value()),
                "minimum_box_length_a": float(
                    self.min_box_length_spin.value()
                ),
                "padding_a": float(self.padding_spin.value()),
                "exclusion_radius_scale": float(
                    self.exclusion_radius_scale_spin.value()
                ),
                "exclusion_radius_padding_a": float(
                    self.exclusion_radius_padding_spin.value()
                ),
            },
            "legacy_mesh_settings": (
                self._legacy_1d_mesh_settings_from_controls().to_dict()
            ),
            "legacy_smearing_settings": (
                self._legacy_1d_smearing_settings_from_controls().to_dict()
            ),
            "legacy_fourier_settings": (
                self._legacy_1d_fourier_settings_from_controls().to_dict()
            ),
            "compare_legacy_1d": bool(
                self.compare_legacy_checkbox.isChecked()
            ),
            "compare_exact_debye": bool(
                self.compare_exact_debye_checkbox.isChecked()
            ),
            "show_kernel_corrected": bool(
                self.show_kernel_corrected_checkbox.isChecked()
            ),
            "log_q_axis": bool(self.log_q_checkbox.isChecked()),
            "log_intensity_axis": bool(
                self.log_intensity_checkbox.isChecked()
            ),
            "curve_legend_visible": bool(self._curve_legend_visible),
            "trace_configuration_signature": (
                self._computed_profile_run_signature
                if self._computed_profile_run_signature is not None
                else self._current_trace_configuration_signature()
            ),
            "profile_results": [
                self._serialize_profile_result(result)
                for result in self._computed_profile_results.values()
            ],
        }

    def _restore_contrast_estimate(
        self,
        payload: dict[str, object] | None,
    ) -> ContrastElectronDensityEstimate | None:
        if not isinstance(payload, dict):
            return None
        reference_path = payload.get("reference_structure_file")
        reference_box_spans = payload.get("reference_box_spans")
        translated_volume_center = payload.get("translated_volume_center")
        try:
            return ContrastElectronDensityEstimate(
                method=str(payload.get("method", "")).strip(),
                label=str(payload.get("label", "")).strip(),
                volume_a3=float(payload.get("volume_a3", 0.0)),
                total_electrons=float(payload.get("total_electrons", 0.0)),
                electron_density_e_per_a3=float(
                    payload.get("electron_density_e_per_a3", 0.0)
                ),
                electron_density_e_per_cm3=float(
                    payload.get("electron_density_e_per_cm3", 0.0)
                ),
                atom_count=(
                    None
                    if payload.get("atom_count") is None
                    else int(payload.get("atom_count", 0))
                ),
                element_counts={
                    str(key): int(value)
                    for key, value in dict(
                        payload.get("element_counts", {})
                    ).items()
                },
                formula=(
                    None
                    if payload.get("formula") is None
                    else str(payload.get("formula", "")).strip() or None
                ),
                source_density_g_per_cm3=(
                    None
                    if payload.get("source_density_g_per_cm3") is None
                    else float(payload.get("source_density_g_per_cm3", 0.0))
                ),
                reference_structure_file=(
                    None
                    if reference_path is None
                    else Path(str(reference_path)).expanduser().resolve()
                ),
                reference_box_spans=(
                    None
                    if reference_box_spans is None
                    else tuple(float(value) for value in reference_box_spans)
                ),
                translated_volume_center=(
                    None
                    if translated_volume_center is None
                    else tuple(
                        float(value) for value in translated_volume_center
                    )
                ),
            )
        except Exception:
            return None

    def _restore_workspace_controls_from_payload(
        self,
        payload: dict[str, object],
    ) -> None:
        self.q_min_spin.setValue(
            float(payload.get("q_min", self.q_min_spin.value()))
        )
        self.q_max_spin.setValue(
            float(payload.get("q_max", self.q_max_spin.value()))
        )
        self.q_step_spin.setValue(
            float(payload.get("q_step", self.q_step_spin.value()))
        )
        fft_settings = payload.get("fft_settings")
        if isinstance(fft_settings, dict):
            self.spacing_spin.setValue(
                float(fft_settings.get("spacing_a", self.spacing_spin.value()))
            )
            self.sigma_spin.setValue(
                float(
                    fft_settings.get(
                        "gaussian_sigma_a",
                        self.sigma_spin.value(),
                    )
                )
            )
            self.min_box_length_spin.setValue(
                float(
                    fft_settings.get(
                        "minimum_box_length_a",
                        self.min_box_length_spin.value(),
                    )
                )
            )
            self.padding_spin.setValue(
                float(fft_settings.get("padding_a", self.padding_spin.value()))
            )
            self.exclusion_radius_scale_spin.setValue(
                float(
                    fft_settings.get(
                        "exclusion_radius_scale",
                        self.exclusion_radius_scale_spin.value(),
                    )
                )
            )
            self.exclusion_radius_padding_spin.setValue(
                float(
                    fft_settings.get(
                        "exclusion_radius_padding_a",
                        self.exclusion_radius_padding_spin.value(),
                    )
                )
            )
        legacy_mesh = payload.get("legacy_mesh_settings")
        if isinstance(legacy_mesh, dict):
            self.legacy_1d_rstep_spin.setValue(
                float(
                    legacy_mesh.get(
                        "rstep_a",
                        self.legacy_1d_rstep_spin.value(),
                    )
                )
            )
            self.legacy_1d_theta_spin.setValue(
                int(
                    legacy_mesh.get(
                        "theta_divisions",
                        self.legacy_1d_theta_spin.value(),
                    )
                )
            )
            self.legacy_1d_phi_spin.setValue(
                int(
                    legacy_mesh.get(
                        "phi_divisions",
                        self.legacy_1d_phi_spin.value(),
                    )
                )
            )
            self.legacy_1d_rmax_spin.setValue(
                float(
                    legacy_mesh.get(
                        "rmax_a",
                        self.legacy_1d_rmax_spin.value(),
                    )
                )
            )
        legacy_smearing = payload.get("legacy_smearing_settings")
        if isinstance(legacy_smearing, dict):
            self.legacy_1d_smearing_factor_spin.setValue(
                float(
                    legacy_smearing.get(
                        "debye_waller_factor_a2",
                        self.legacy_1d_smearing_factor_spin.value(),
                    )
                )
            )
        legacy_fourier = payload.get("legacy_fourier_settings")
        if isinstance(legacy_fourier, dict):
            domain_index = self.legacy_1d_domain_combo.findData(
                legacy_fourier.get("domain_mode", "legacy")
            )
            if domain_index >= 0:
                self.legacy_1d_domain_combo.setCurrentIndex(domain_index)
            window_index = self.legacy_1d_window_combo.findData(
                legacy_fourier.get("window_function", "none")
            )
            if window_index >= 0:
                self.legacy_1d_window_combo.setCurrentIndex(window_index)
            self.legacy_1d_resampling_points_spin.setValue(
                int(
                    legacy_fourier.get(
                        "resampling_points",
                        self.legacy_1d_resampling_points_spin.value(),
                    )
                )
            )
        self.compare_legacy_checkbox.setChecked(
            bool(payload.get("compare_legacy_1d", True))
        )
        self.compare_exact_debye_checkbox.setChecked(
            bool(payload.get("compare_exact_debye", False))
        )
        self.log_q_checkbox.setChecked(bool(payload.get("log_q_axis", True)))
        self.log_intensity_checkbox.setChecked(
            bool(payload.get("log_intensity_axis", True))
        )
        self._curve_legend_visible = bool(
            payload.get("curve_legend_visible", True)
        )
        self._update_curve_legend_button_text()

        contrast_settings_payload = payload.get("active_contrast_settings")
        self._active_contrast_settings = None
        self._active_contrast_estimate = None
        self._active_contrast_name = None
        self._active_solvent_density_e_per_a3 = 0.0
        if isinstance(contrast_settings_payload, dict):
            restored_settings = ContrastSolventDensitySettings.from_values(
                **contrast_settings_payload
            )
            method_index = self.solvent_method_combo.findData(
                restored_settings.method
            )
            if method_index >= 0:
                self.solvent_method_combo.setCurrentIndex(method_index)
            self.solvent_formula_edit.setText(
                restored_settings.solvent_formula or ""
            )
            self.solvent_density_spin.setValue(
                float(restored_settings.solvent_density_g_per_ml or 0.0)
            )
            self.direct_density_spin.setValue(
                float(
                    restored_settings.direct_electron_density_e_per_a3 or 0.0
                )
            )
            self.reference_solvent_file_edit.setText(
                ""
                if restored_settings.reference_structure_file is None
                else str(restored_settings.reference_structure_file)
            )
            self._active_contrast_settings = restored_settings
            self._active_contrast_name = (
                str(payload.get("active_contrast_name", "")).strip() or None
            )
            self._active_contrast_estimate = self._restore_contrast_estimate(
                payload.get("active_contrast_estimate")
                if isinstance(payload.get("active_contrast_estimate"), dict)
                else None
            )
            if self._active_contrast_estimate is None:
                try:
                    self._active_contrast_estimate = (
                        self._estimate_solvent_density(restored_settings)
                    )
                except Exception:
                    self._active_contrast_estimate = None
            if self._active_contrast_estimate is not None:
                self._active_solvent_density_e_per_a3 = float(
                    self._active_contrast_estimate.electron_density_e_per_a3
                )
            else:
                self._active_solvent_density_e_per_a3 = float(
                    payload.get("active_solvent_density_e_per_a3", 0.0)
                )
        self._sync_density_method_controls()
        self._sync_kernel_correction_option()
        self.show_kernel_corrected_checkbox.setChecked(
            bool(payload.get("show_kernel_corrected", False))
            and self.show_kernel_corrected_checkbox.isEnabled()
        )
        self._contrast_controls_dirty = False
        self._refresh_contrast_display()

    def _delete_workspace_state(self, *, announce: bool) -> None:
        workspace_state_path = self._workspace_state_path()
        if workspace_state_path is None or not workspace_state_path.is_file():
            return
        try:
            workspace_state_path.unlink()
        except Exception as exc:
            if announce:
                self._append_status(
                    "Could not remove the saved 3D FFT workspace state from "
                    f"{workspace_state_path}: {exc}"
                )
            return
        if announce:
            self._append_status(
                f"Removed the saved 3D FFT workspace state from {workspace_state_path}."
            )

    def _sync_workspace_state(self) -> None:
        if self._preview_mode or self._restoring_workspace_state:
            return
        workspace_state_path = self._workspace_state_path()
        if workspace_state_path is None:
            return
        if not self._computed_profile_results:
            self._delete_workspace_state(announce=False)
            return
        if not self._results_match_current_configuration():
            return
        payload = self._build_component_summary_payload()
        payload["state_kind"] = "workspace_session"
        workspace_state_path.parent.mkdir(parents=True, exist_ok=True)
        workspace_state_path.write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )

    def _restore_workspace_state_from_disk(self) -> None:
        if self._preview_mode:
            return
        workspace_state_path = self._workspace_state_path()
        if workspace_state_path is None or not workspace_state_path.is_file():
            return
        try:
            payload = json.loads(
                workspace_state_path.read_text(encoding="utf-8")
            )
        except Exception as exc:
            self._append_status(
                "Could not read the saved 3D FFT workspace state from "
                f"{workspace_state_path}: {exc}"
            )
            return
        self._restoring_workspace_state = True
        try:
            self._prefer_representative_structures = bool(
                payload.get(
                    "prefer_representative_structures",
                    self._prefer_representative_structures,
                )
            )
            self._restore_workspace_controls_from_payload(payload)
            source_mode = str(payload.get("structure_source_mode", "")).strip()
            if source_mode:
                index = self.structure_source_combo.findData(source_mode)
                if index >= 0:
                    self.structure_source_combo.setCurrentIndex(index)
            solvent_mode = str(
                payload.get("representative_solvent_mode", "")
            ).strip()
            if solvent_mode:
                index = self.representative_solvent_mode_combo.findData(
                    solvent_mode
                )
                if index >= 0:
                    self.representative_solvent_mode_combo.setCurrentIndex(
                        index
                    )
            self._refresh_available_profile_targets()
            restored_results: list[_FFTProfileComputationResult] = []
            for entry in payload.get("profile_results", []):
                if not isinstance(entry, dict):
                    continue
                try:
                    restored_results.append(
                        self._deserialize_profile_result(dict(entry))
                    )
                except Exception:
                    continue
            self._computed_profile_results = {
                result.target.key: result for result in restored_results
            }
            self._current_payload = (
                None
                if not restored_results
                else _FFTComputationPayload(
                    q_values=np.asarray(
                        restored_results[0].q_values,
                        dtype=float,
                    ),
                    profile_results=tuple(restored_results),
                )
            )
            restored_signature = payload.get("trace_configuration_signature")
            self._computed_profile_run_signature = (
                dict(restored_signature)
                if restored_results and isinstance(restored_signature, dict)
                else None
            )
            restored_key = str(payload.get("active_profile_key", "")).strip()
            if restored_key:
                self._active_profile_key = restored_key
            if self._active_profile_key not in self._computed_profile_results:
                self._active_profile_key = (
                    None
                    if not restored_results
                    else restored_results[0].target.key
                )
            self._populate_active_profile_combo()
            self._set_active_profile_target(self._active_profile_target())
            if (
                restored_results
                and self._computed_profile_run_signature is None
            ):
                self._computed_profile_run_signature = (
                    self._current_trace_configuration_signature()
                )
            if self._current_payload is not None:
                self._update_result_summary(self._current_payload)
            self._refresh_plot_controls()
        finally:
            self._restoring_workspace_state = False
        self._update_push_to_model_state()

    def _update_push_to_model_state(self) -> None:
        enabled = bool(
            (not self._preview_mode)
            and self._distribution_root_dir is not None
            and self._computed_profile_results
            and self._results_match_current_configuration()
        )
        self.push_to_model_button.setEnabled(enabled)

    def _ensure_linked_distribution_ready_for_push(self) -> None:
        if (
            self._preview_mode
            or self._distribution_root_dir is None
            or self._project_dir is None
        ):
            return
        metadata_path = self._distribution_root_dir / "distribution.json"
        prior_weights_path = self._distribution_root_dir / (
            "md_prior_weights_predicted_structures.json"
            if self._use_predicted_structure_weights
            else "md_prior_weights.json"
        )
        if metadata_path.is_file() and prior_weights_path.is_file():
            return
        from saxshell.saxs.contrast.settings import (
            COMPONENT_BUILD_MODE_BORN_APPROXIMATION_3D_FFT,
        )
        from saxshell.saxs.project_manager.project import (
            SAXSProjectManager,
            project_artifact_paths,
        )

        project_manager = SAXSProjectManager()
        settings = project_manager.load_project(self._project_dir)
        settings.component_build_mode = (
            COMPONENT_BUILD_MODE_BORN_APPROXIMATION_3D_FFT
        )
        settings.use_predicted_structure_weights = bool(
            self._use_predicted_structure_weights
        )
        settings.use_representative_structures = (
            self._current_structure_source_mode() == "representative"
        )
        artifact_paths = project_artifact_paths(
            settings,
            storage_mode="distribution",
            allow_legacy_fallback=False,
        )
        if artifact_paths.root_dir.resolve() != (
            self._distribution_root_dir.expanduser().resolve()
        ):
            raise ValueError(
                "The linked computed distribution no longer matches the "
                "active project settings. Reopen the 3D FFT Born workflow "
                "from Project Setup and push again."
            )
        project_manager.generate_prior_weights(settings)

    def _write_component_trace_file(
        self,
        result: _FFTProfileComputationResult,
        component_dir: Path,
    ) -> str:
        profile_file = self._component_profile_filename(
            result.target.structure_name,
            result.target.motif_name,
        )
        output_path = component_dir / profile_file
        q_values = np.asarray(result.q_values, dtype=float)
        intensity = np.asarray(result.fft_result.raw_intensity, dtype=float)
        data = np.column_stack(
            [
                q_values,
                intensity,
                np.zeros_like(q_values, dtype=float),
                np.zeros_like(q_values, dtype=float),
            ]
        )
        header = (
            f"# Number of files: {result.target.file_count}\n"
            "# Columns: q, S(q)_avg, S(q)_std, S(q)_se\n"
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
        self, summary_path: Path
    ) -> None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(self._build_component_summary_payload(), indent=2)
            + "\n",
            encoding="utf-8",
        )

    @Slot()
    def _push_components_to_model(self) -> None:
        if self._preview_mode or self._distribution_root_dir is None:
            self._show_error(
                "Push to Model unavailable",
                "This 3D FFT Born window is not linked to a computed distribution.",
            )
            return
        if not self._computed_profile_results:
            self._show_error(
                "Push to Model unavailable",
                "Compute at least one 3D FFT Born profile set before pushing the outputs into the model.",
            )
            return
        if not self._results_match_current_configuration():
            self._show_error(
                "Recompute 3D FFT Born traces",
                "The active 3D FFT source, q range, grid settings, or electron density contrast has changed since these traces were computed. Recompute the traces before pushing them into the model.",
            )
            return
        artifact_targets = self._component_artifact_targets()
        summary_path = self._component_summary_path()
        if artifact_targets is None or summary_path is None:
            self._show_error(
                "Push to Model unavailable",
                "The linked computed distribution paths are not available.",
            )
            return
        self._ensure_linked_distribution_ready_for_push()
        component_dir, component_map_path = artifact_targets
        component_dir.mkdir(parents=True, exist_ok=True)
        saxs_map: dict[str, dict[str, str]] = {}
        for result in self._computed_profile_results.values():
            profile_file = self._write_component_trace_file(
                result,
                component_dir,
            )
            saxs_map.setdefault(result.target.structure_name, {})
            saxs_map[result.target.structure_name][
                result.target.motif_name
            ] = profile_file
        component_map_path.write_text(
            json.dumps({"saxs_map": saxs_map}, indent=2) + "\n",
            encoding="utf-8",
        )
        self._write_distribution_component_summary(summary_path)
        if self._project_dir is not None:
            from saxshell.saxs.contrast.settings import (
                COMPONENT_BUILD_MODE_BORN_APPROXIMATION_3D_FFT,
            )
            from saxshell.saxs.project_manager.project import (
                SAXSProjectManager,
                project_artifact_paths,
            )

            project_manager = SAXSProjectManager()
            settings = project_manager.load_project(self._project_dir)
            settings.component_build_mode = (
                COMPONENT_BUILD_MODE_BORN_APPROXIMATION_3D_FFT
            )
            settings.use_predicted_structure_weights = bool(
                self._use_predicted_structure_weights
            )
            settings.use_representative_structures = (
                self._current_structure_source_mode() == "representative"
            )
            artifact_paths = project_artifact_paths(
                settings,
                storage_mode="distribution",
                allow_legacy_fallback=False,
            )
            if artifact_paths.root_dir.resolve() == (
                self._distribution_root_dir.expanduser().resolve()
            ):
                project_manager._write_distribution_metadata(
                    settings,
                    artifact_paths=artifact_paths,
                    built_component_source_mode=(
                        self._current_structure_source_mode()
                    ),
                )
                project_manager.save_project(
                    settings,
                    refresh_registered_paths=False,
                )
            else:
                raise ValueError(
                    "The linked computed distribution no longer matches the "
                    "active project settings. Reopen the 3D FFT Born workflow "
                    "from Project Setup and push again."
                )
        self._append_status(
            "Pushed 3D FFT Born component traces into the linked computed "
            f"distribution: {component_map_path}"
        )
        self.statusBar().showMessage("3D FFT Born components pushed to model")
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

    def _show_error(self, title: str, message: str) -> None:
        QMessageBox.critical(self, title, message)
        self.statusBar().showMessage(title)

    def _ensure_progress_dialog(self) -> SAXSProgressDialog:
        if self._progress_dialog is None:
            dialog = SAXSProgressDialog(self)
            dialog.setModal(True)
            dialog.setWindowModality(Qt.WindowModality.WindowModal)
            self._progress_dialog = dialog
        return self._progress_dialog

    def _begin_progress_dialog(self, message: str) -> None:
        dialog = self._ensure_progress_dialog()
        dialog.begin_busy(
            str(message).strip() or "Running 3D FFT Born Approximation...",
            title="Calculating 3D FFT Born Approximation",
        )
        QApplication.processEvents()

    def _set_progress_message(self, message: str) -> None:
        dialog = self._ensure_progress_dialog()
        stripped = (
            str(message).strip() or "Running 3D FFT Born Approximation..."
        )
        dialog.setWindowTitle("Calculating 3D FFT Born Approximation")
        dialog.progress_bar.setRange(0, 0)
        dialog.progress_bar.setValue(0)
        dialog.progress_bar.setFormat("")
        dialog.message_label.setText(stripped)
        dialog.show()
        dialog.raise_()
        QApplication.processEvents()

    def _close_progress_dialog(self) -> None:
        if self._progress_dialog is not None:
            self._progress_dialog.close()

    def _ui_settings(self) -> QSettings:
        return QSettings("SAXShell", "SAXS")

    def _load_auto_snap_setting(self) -> bool:
        raw_value = self._ui_settings().value(AUTO_SNAP_PANES_KEY, True)
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

    def closeEvent(
        self, event: QCloseEvent
    ) -> None:  # noqa: N802 - Qt override
        if (
            self._compute_thread is not None
            and self._compute_thread.isRunning()
        ):
            self._close_requested_while_running = True
            self._close_progress_dialog()
            if self._compute_worker is not None:
                self._compute_worker.cancel()
            self._compute_thread.quit()
            self._compute_thread.wait(1000)
            if (
                self._compute_thread is not None
                and self._compute_thread.isRunning()
            ):
                self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
                self.hide()
                event.accept()
                return
            self._close_requested_while_running = False
        else:
            self._close_requested_while_running = False
        self._close_progress_dialog()
        if self._progress_dialog is not None:
            self._progress_dialog.deleteLater()
            self._progress_dialog = None
        self._reference_structure_cache.clear()
        self._available_profile_targets = {}
        self._current_profile_targets = ()
        super().closeEvent(event)


def _forget_open_window(window: FFTBornApproximationMainWindow) -> None:
    global _OPEN_WINDOWS
    _OPEN_WINDOWS = [
        existing for existing in _OPEN_WINDOWS if existing is not window
    ]


def launch_3d_fft_born_approximation_ui(
    *,
    initial_project_dir: str | Path | None = None,
    initial_input_path: str | Path | None = None,
    initial_output_dir: str | Path | None = None,
    initial_project_q_min: float | None = None,
    initial_project_q_max: float | None = None,
    initial_distribution_id: str | None = None,
    initial_distribution_root_dir: str | Path | None = None,
    initial_use_predicted_structure_weights: bool = False,
    initial_use_representative_structures: bool = False,
    preview_mode: bool = True,
) -> FFTBornApproximationMainWindow:
    app = QApplication.instance()
    if app is None:
        prepare_saxshell_application_identity()
        app = QApplication(sys.argv)
    configure_saxshell_application(app)
    window = FFTBornApproximationMainWindow(
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
        initial_use_representative_structures=(
            initial_use_representative_structures
        ),
        preview_mode=preview_mode,
    )
    window.show()
    window.raise_()
    _OPEN_WINDOWS.append(window)
    window.destroyed.connect(
        lambda _obj=None, win=window: _forget_open_window(win)
    )
    return window


__all__ = [
    "FFTBornApproximationMainWindow",
    "launch_3d_fft_born_approximation_ui",
]
