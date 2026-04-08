from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib.backend_bases import MouseButton
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import (
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from saxshell.saxs.electron_density_mapping.workflow import (
    ElectronDensityFourierTransformPreview,
    ElectronDensityMeshGeometry,
    ElectronDensityProfileResult,
    ElectronDensityScatteringTransformResult,
    ElectronDensitySolventContrastResult,
    ElectronDensityStructure,
)
from saxshell.toolbox.blender.common import (
    DEFAULT_ATOM_STYLE,
    atom_style_defaults,
    style_atom_color,
    style_display_radius,
    style_neutral_bond_color,
    style_split_bond_color,
)

_BACKGROUND_COLOR = "#fdfbf8"
_EDGE_COLOR = (0.16, 0.14, 0.20)
_SHADOW_COLOR = (0.26, 0.17, 0.30)
_AXIS_COLORS = (
    ("x", "#d9534f"),
    ("y", "#3aa655"),
    ("z", "#3b7ddd"),
)
_DEFAULT_MESH_COLOR = "#2b8cbe"
_DEFAULT_MESH_LINEWIDTH = 1.35
_CAMERA_AZIM = -62.0
_CAMERA_ELEV = 22.0
_DEFAULT_ATOM_CONTRAST = 1.0
_DEFAULT_MESH_CONTRAST = 0.60
_ORIGIN_GUIDE_CUTOFF_A = 3.0
_POINT_ATOM_SIZE = 26.0
_ACTIVE_READOUT_COLOR = "#9ad8ff"
_ACTIVE_READOUT_BACKGROUND = (0.05, 0.10, 0.17, 0.78)
_VIEW_UPDATE_INTERVAL_MS = 16


@dataclass(slots=True)
class ElectronDensityProfileOverlay:
    solvent_density_e_per_a3: float | None = None
    solvent_density_label: str | None = None
    cutoff_radius_a: float | None = None
    cutoff_label: str | None = None
    solvent_subtracted_density: np.ndarray | None = None
    solvent_subtracted_label: str | None = None


def _clamp_fraction(value: float) -> float:
    return float(max(0.0, min(1.0, float(value))))


def _rotation_matrix_from_axis_angle(
    axis: np.ndarray,
    angle_radians: float,
) -> np.ndarray:
    axis_vector = np.asarray(axis, dtype=float)
    norm = float(np.linalg.norm(axis_vector))
    if norm <= 1.0e-12 or abs(float(angle_radians)) <= 1.0e-12:
        return np.eye(3, dtype=float)
    axis_vector = axis_vector / norm
    x_value, y_value, z_value = axis_vector
    cosine = float(np.cos(angle_radians))
    sine = float(np.sin(angle_radians))
    one_minus_cosine = 1.0 - cosine
    return np.asarray(
        [
            [
                cosine + x_value * x_value * one_minus_cosine,
                x_value * y_value * one_minus_cosine - z_value * sine,
                x_value * z_value * one_minus_cosine + y_value * sine,
            ],
            [
                y_value * x_value * one_minus_cosine + z_value * sine,
                cosine + y_value * y_value * one_minus_cosine,
                y_value * z_value * one_minus_cosine - x_value * sine,
            ],
            [
                z_value * x_value * one_minus_cosine - y_value * sine,
                z_value * y_value * one_minus_cosine + x_value * sine,
                cosine + z_value * z_value * one_minus_cosine,
            ],
        ],
        dtype=float,
    )


def _orthonormalize_rotation(matrix: np.ndarray) -> np.ndarray:
    left, _singular_values, right = np.linalg.svd(
        np.asarray(matrix, dtype=float)
    )
    rotation = left @ right
    if float(np.linalg.det(rotation)) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right
    return np.asarray(rotation, dtype=float)


def _sample_values(values: np.ndarray, *, max_count: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.size <= max_count:
        return array
    indices = np.linspace(
        0,
        array.size - 1,
        max_count,
        dtype=int,
    )
    return array[np.unique(indices)]


class ElectronDensityProfilePlot(QWidget):
    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        title: str,
        legend_label: str,
        trace_color: str,
        fill_color: str | None,
        profile_attribute: str,
        spread_attribute: str | None = None,
        as_step_trace: bool,
    ) -> None:
        super().__init__(parent)
        self.current_result: ElectronDensityProfileResult | None = None
        self.current_overlay: ElectronDensityProfileOverlay | None = None
        self._title = str(title)
        self._legend_label = str(legend_label)
        self._trace_color = str(trace_color)
        self._fill_color = None if fill_color is None else str(fill_color)
        self._profile_attribute = str(profile_attribute)
        self._spread_attribute = (
            None if spread_attribute is None else str(spread_attribute)
        )
        self._as_step_trace = bool(as_step_trace)
        self._show_variance_shading = True
        self.figure = Figure(figsize=(8.4, 3.4))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self._build_ui()
        self.draw_placeholder()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self.toolbar)
        self.canvas.setMinimumHeight(280)
        layout.addWidget(self.canvas, stretch=1)

    def draw_placeholder(self) -> None:
        self.current_result = None
        self.current_overlay = None
        self.figure.clear()
        axis = self.figure.add_subplot(111)
        axis.text(
            0.5,
            0.58,
            "Run the electron-density calculation to populate this profile.",
            ha="center",
            va="center",
            wrap=True,
            transform=axis.transAxes,
        )
        axis.text(
            0.5,
            0.40,
            "The raw panel shows the shell-averaged profile. The smoothed panel "
            "applies the current Gaussian smearing settings to the same radial trace.",
            ha="center",
            va="center",
            wrap=True,
            alpha=0.78,
            transform=axis.transAxes,
        )
        axis.set_axis_off()
        self.canvas.draw_idle()

    def set_profile(
        self,
        result: ElectronDensityProfileResult | None,
        *,
        overlay: ElectronDensityProfileOverlay | None = None,
    ) -> None:
        if result is None:
            self.draw_placeholder()
            return
        self.current_result = result
        self.current_overlay = overlay
        self.figure.clear()
        axis = self.figure.add_subplot(111)
        radial_values = np.asarray(result.radial_centers, dtype=float)
        density_values = np.asarray(
            getattr(result, self._profile_attribute),
            dtype=float,
        )
        spread_values = None
        if self._show_variance_shading and self._spread_attribute is not None:
            spread_values = np.asarray(
                getattr(result, self._spread_attribute),
                dtype=float,
            )
        if self._as_step_trace:
            axis.step(
                radial_values,
                density_values,
                where="mid",
                color=self._trace_color,
                linewidth=2.2,
                label=self._legend_label,
            )
            if spread_values is not None and np.any(
                np.asarray(spread_values, dtype=float) > 0.0
            ):
                lower = np.maximum(density_values - spread_values, 0.0)
                upper = density_values + spread_values
                axis.fill_between(
                    radial_values,
                    lower,
                    upper,
                    step="mid",
                    alpha=0.18,
                    color=self._fill_color or self._trace_color,
                    label="Variance shading",
                )
            elif (
                self._spread_attribute is None and self._fill_color is not None
            ):
                axis.fill_between(
                    radial_values,
                    density_values,
                    step="mid",
                    alpha=0.12,
                    color=self._fill_color,
                )
        else:
            axis.plot(
                radial_values,
                density_values,
                color=self._trace_color,
                linewidth=2.4,
                label=self._legend_label,
            )
            if spread_values is not None and np.any(
                np.asarray(spread_values, dtype=float) > 0.0
            ):
                lower = np.maximum(density_values - spread_values, 0.0)
                upper = density_values + spread_values
                axis.fill_between(
                    radial_values,
                    lower,
                    upper,
                    alpha=0.18,
                    color=self._fill_color or self._trace_color,
                    label="Variance shading",
                )
            elif (
                self._spread_attribute is None and self._fill_color is not None
            ):
                axis.fill_between(
                    radial_values,
                    density_values,
                    alpha=0.10,
                    color=self._fill_color,
                )
        if (
            overlay is not None
            and overlay.solvent_subtracted_density is not None
        ):
            solvent_subtracted = np.asarray(
                overlay.solvent_subtracted_density,
                dtype=float,
            )
            if solvent_subtracted.shape == density_values.shape:
                if self._as_step_trace:
                    axis.step(
                        radial_values,
                        solvent_subtracted,
                        where="mid",
                        color="#64748b",
                        linewidth=1.8,
                        linestyle="--",
                        alpha=0.95,
                        label=(
                            overlay.solvent_subtracted_label
                            or "Solvent-subtracted ρ(r)"
                        ),
                    )
                else:
                    axis.plot(
                        radial_values,
                        solvent_subtracted,
                        color="#64748b",
                        linewidth=1.9,
                        linestyle="--",
                        alpha=0.95,
                        label=(
                            overlay.solvent_subtracted_label
                            or "Solvent-subtracted ρ(r)"
                        ),
                    )
        if (
            overlay is not None
            and overlay.solvent_density_e_per_a3 is not None
        ):
            axis.axhline(
                float(overlay.solvent_density_e_per_a3),
                color="#334155",
                linewidth=1.8,
                linestyle="--",
                alpha=0.95,
                label=overlay.solvent_density_label or "Solvent density",
            )
        if overlay is not None and overlay.cutoff_radius_a is not None:
            axis.axvline(
                float(overlay.cutoff_radius_a),
                color="crimson",
                linewidth=2.6,
                linestyle="--",
                alpha=0.95,
                label=overlay.cutoff_label or "Cutoff",
            )
        axis.set_xlabel("r (Å)")
        axis.set_ylabel("ρ(r) (e/Å³)")
        axis.set_title(self._title)
        axis.grid(True, alpha=0.28)
        axis.legend(loc="upper right", frameon=True)
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def set_variance_visible(self, visible: bool) -> None:
        self._show_variance_shading = bool(visible)
        if self.current_result is not None:
            self.set_profile(self.current_result, overlay=self.current_overlay)


class ElectronDensityResidualPlot(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.current_result: ElectronDensityProfileResult | None = None
        self.current_contrast: ElectronDensitySolventContrastResult | None = (
            None
        )
        self._show_variance_shading = True
        self.figure = Figure(figsize=(8.4, 3.2))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self._build_ui()
        self.draw_placeholder()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self.toolbar)
        self.canvas.setMinimumHeight(260)
        layout.addWidget(self.canvas, stretch=1)

    def draw_placeholder(self) -> None:
        self.current_result = None
        self.current_contrast = None
        self.figure.clear()
        axis = self.figure.add_subplot(111)
        axis.text(
            0.5,
            0.60,
            "Compute the solvent electron density to populate the residual plot.",
            ha="center",
            va="center",
            wrap=True,
            transform=axis.transAxes,
        )
        axis.text(
            0.5,
            0.40,
            "This panel shows the solvent-subtracted smeared profile, "
            "which is also available as the optional Fourier-transform source.",
            ha="center",
            va="center",
            wrap=True,
            alpha=0.78,
            transform=axis.transAxes,
        )
        axis.set_axis_off()
        self.canvas.draw_idle()

    def set_residual_profile(
        self,
        result: ElectronDensityProfileResult | None,
        contrast: ElectronDensitySolventContrastResult | None,
    ) -> None:
        if result is None or contrast is None:
            self.draw_placeholder()
            return
        self.current_result = result
        self.current_contrast = contrast
        self.figure.clear()
        axis = self.figure.add_subplot(111)
        radial_values = np.asarray(result.radial_centers, dtype=float)
        residual_values = np.asarray(
            contrast.solvent_subtracted_smeared_density,
            dtype=float,
        )
        axis.plot(
            radial_values,
            residual_values,
            color="#64748b",
            linewidth=2.1,
            linestyle="--",
            label="Smeared ρ(r) - solvent",
        )
        if self._show_variance_shading:
            spread_values = np.asarray(
                result.smeared_orientation_density_stddev,
                dtype=float,
            )
            if np.any(spread_values > 0.0):
                axis.fill_between(
                    radial_values,
                    residual_values - spread_values,
                    residual_values + spread_values,
                    alpha=0.16,
                    color="#94a3b8",
                    label="Variance shading",
                )
        axis.axhline(
            0.0,
            color="#94a3b8",
            linewidth=1.5,
            linestyle=":",
            alpha=0.95,
            label="Solvent baseline",
        )
        if contrast.cutoff_radius_a is not None:
            axis.axvline(
                float(contrast.cutoff_radius_a),
                color="crimson",
                linewidth=2.6,
                linestyle="--",
                alpha=0.95,
                label=contrast.cutoff_label or "Cutoff",
            )
        axis.set_xlabel("r (Å)")
        axis.set_ylabel("Δρ(r) (e/Å³)")
        axis.set_title("Solvent-Subtracted Residual")
        axis.grid(True, alpha=0.28)
        axis.legend(loc="upper right", frameon=True)
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def set_variance_visible(self, visible: bool) -> None:
        self._show_variance_shading = bool(visible)
        if (
            self.current_result is not None
            and self.current_contrast is not None
        ):
            self.set_residual_profile(
                self.current_result,
                self.current_contrast,
            )


class ElectronDensityFourierPreviewPlot(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.current_preview: ElectronDensityFourierTransformPreview | None = (
            None
        )
        self.figure = Figure(figsize=(8.4, 3.4))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self._build_ui()
        self.draw_placeholder()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self.toolbar)
        self.canvas.setMinimumHeight(280)
        layout.addWidget(self.canvas, stretch=1)

    def draw_placeholder(self) -> None:
        self.current_preview = None
        self.figure.clear()
        axis = self.figure.add_subplot(111)
        axis.text(
            0.5,
            0.60,
            "Run the electron-density calculation to prepare the Fourier-transform preview.",
            ha="center",
            va="center",
            wrap=True,
            transform=axis.transAxes,
        )
        axis.text(
            0.5,
            0.40,
            "This panel will show the smeared ρ(r), the selected transform bounds, "
            "the resampled points, and the active modification window.",
            ha="center",
            va="center",
            wrap=True,
            alpha=0.78,
            transform=axis.transAxes,
        )
        axis.set_axis_off()
        self.canvas.draw_idle()

    def set_preview(
        self,
        preview: ElectronDensityFourierTransformPreview | None,
    ) -> None:
        if preview is None:
            self.draw_placeholder()
            return
        self.current_preview = preview
        self.figure.clear()
        axis = self.figure.add_subplot(111)
        settings = preview.settings
        full_r = np.asarray(preview.source_radial_values, dtype=float)
        full_density = np.asarray(preview.source_density_values, dtype=float)
        resampled_r = np.asarray(preview.resampled_r_values, dtype=float)
        resampled_density = np.asarray(
            preview.resampled_density_values, dtype=float
        )
        windowed_density = np.asarray(
            preview.windowed_density_values, dtype=float
        )
        window_values = np.asarray(preview.window_values, dtype=float)

        axis.plot(
            full_r,
            full_density,
            color="#9bd5a6",
            linewidth=1.6,
            alpha=0.95,
            label=preview.source_profile_label,
        )
        axis.axvspan(
            float(settings.r_min),
            float(settings.r_max),
            color="#dbeafe",
            alpha=0.45,
            label="Transform bounds",
        )
        axis.plot(
            resampled_r,
            resampled_density,
            color="#22c55e",
            linewidth=1.3,
            alpha=0.70,
            label=f"Resampled {preview.source_profile_label}",
        )
        axis.plot(
            resampled_r,
            windowed_density,
            color="#15803d",
            linewidth=2.2,
            label=f"Windowed {preview.source_profile_label}",
        )
        sample_stride = max(int(np.ceil(len(resampled_r) / 80.0)), 1)
        axis.scatter(
            resampled_r[::sample_stride],
            resampled_density[::sample_stride],
            s=10,
            color="#14532d",
            alpha=0.55,
            label="Resampled points",
            zorder=3,
        )
        window_axis = axis.twinx()
        window_axis.plot(
            resampled_r,
            window_values,
            color="#7c3aed",
            linewidth=1.5,
            linestyle="--",
            alpha=0.95,
            label=f"Window ({settings.window_function})",
        )
        window_axis.set_ylabel("Window")
        window_axis.set_ylim(-0.05, 1.10)
        axis.set_xlabel("r (Å)")
        axis.set_ylabel("ρ(r) (e/Å³)")
        axis.set_title("Fourier-Transform Preparation")
        axis.grid(True, alpha=0.28)
        handles, labels = axis.get_legend_handles_labels()
        window_handles, window_labels = window_axis.get_legend_handles_labels()
        axis.legend(
            handles + window_handles,
            labels + window_labels,
            loc="upper right",
            frameon=True,
            fontsize=8.1,
        )
        axis.text(
            0.02,
            0.02,
            (
                f"Δr={preview.resampling_step_a:.4f} Å | "
                f"Nyquist qmax≈{preview.nyquist_q_max_a_inverse:.3f} Å⁻¹ | "
                f"Independent Δq≈{preview.independent_q_step_a_inverse:.3f} Å⁻¹"
            ),
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=8.0,
            color="#334155",
            bbox={
                "facecolor": "#f8fafc",
                "edgecolor": "#cbd5e1",
                "boxstyle": "round,pad=0.28",
                "alpha": 0.92,
            },
        )
        self.figure.tight_layout()
        self.canvas.draw_idle()


class ElectronDensityScatteringPlot(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.current_result: (
            ElectronDensityScatteringTransformResult | None
        ) = None
        self.figure = Figure(figsize=(8.4, 3.4))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self._build_ui()
        self.draw_placeholder()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self.toolbar)
        self.canvas.setMinimumHeight(280)
        layout.addWidget(self.canvas, stretch=1)

    def draw_placeholder(self) -> None:
        self.current_result = None
        self.figure.clear()
        axis = self.figure.add_subplot(111)
        axis.text(
            0.5,
            0.60,
            "Evaluate the Fourier transform to populate the q-space scattering profile.",
            ha="center",
            va="center",
            wrap=True,
            transform=axis.transAxes,
        )
        axis.text(
            0.5,
            0.40,
            "The q-space plot will show Intensity (arb. units) versus q (Å⁻¹), "
            "with optional log scaling on each axis.",
            ha="center",
            va="center",
            wrap=True,
            alpha=0.78,
            transform=axis.transAxes,
        )
        axis.set_axis_off()
        self.canvas.draw_idle()

    def set_transform_result(
        self,
        result: ElectronDensityScatteringTransformResult | None,
        *,
        log_q_axis: bool,
        log_intensity_axis: bool,
    ) -> None:
        if result is None:
            self.draw_placeholder()
            return
        self.current_result = result
        self.figure.clear()
        axis = self.figure.add_subplot(111)
        q_values = np.asarray(result.q_values, dtype=float)
        intensity = np.asarray(result.intensity, dtype=float)
        positive_mask = np.ones_like(q_values, dtype=bool)
        if log_q_axis:
            positive_mask &= q_values > 0.0
        if log_intensity_axis:
            positive_mask &= intensity > 0.0
        plot_q = q_values[positive_mask]
        plot_intensity = intensity[positive_mask]
        if plot_q.size == 0 or plot_intensity.size == 0:
            axis.text(
                0.5,
                0.5,
                "No positive q/intensity values are available for the requested log scaling.",
                ha="center",
                va="center",
                wrap=True,
                transform=axis.transAxes,
            )
            axis.set_axis_off()
            self.canvas.draw_idle()
            return
        axis.plot(
            plot_q,
            plot_intensity,
            color="#b45309",
            linewidth=2.2,
            label="Born-approximation intensity",
        )
        if log_q_axis:
            axis.set_xscale("log")
        if log_intensity_axis:
            axis.set_yscale("log")
        axis.set_xlabel("q (Å⁻¹)", labelpad=10.0)
        axis.set_ylabel("Intensity (arb. units)")
        axis.set_title("q-Space Scattering Profile")
        axis.grid(True, which="both", alpha=0.28)
        axis.legend(loc="upper right", frameon=True)
        settings = result.preview.settings
        axis.text(
            0.02,
            0.02,
            (
                f"Window={settings.window_function} | "
                f"r={settings.r_min:.3f} to {settings.r_max:.3f} Å | "
                f"{len(result.q_values)} q points"
            ),
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=8.0,
            color="#7c2d12",
            bbox={
                "facecolor": "#fff7ed",
                "edgecolor": "#fdba74",
                "boxstyle": "round,pad=0.28",
                "alpha": 0.92,
            },
        )
        self.figure.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
        self.canvas.draw_idle()


class _PassiveWheelFigureCanvas(FigureCanvas):
    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt override
        event.ignore()


class ElectronDensityStructureViewer(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.current_structure: ElectronDensityStructure | None = None
        self.current_mesh_geometry: ElectronDensityMeshGeometry | None = None
        self.figure = Figure(figsize=(8.4, 6.2))
        self.canvas = _PassiveWheelFigureCanvas(self.figure)
        self._axis = None
        self._axis_reference = None
        self._view_center = np.zeros(3, dtype=float)
        self._view_radius = 1.0
        self._interaction_mode = "rotate"
        self._drag_mode: str | None = None
        self._drag_start_xy = (0.0, 0.0)
        self._drag_start_center = np.zeros(3, dtype=float)
        self._drag_start_radius = 1.0
        self._drag_start_rotation = np.eye(3, dtype=float)
        self._scene_rotation = np.eye(3, dtype=float)
        self._mesh_visible = True
        self._atom_contrast = _DEFAULT_ATOM_CONTRAST
        self._mesh_contrast = _DEFAULT_MESH_CONTRAST
        self._mesh_linewidth = _DEFAULT_MESH_LINEWIDTH
        self._mesh_color = _DEFAULT_MESH_COLOR
        self._atom_render_mode = "balls"
        self._pending_view_update = False
        self._pending_axis_reference_refresh = False
        self._view_update_timer = QTimer(self)
        self._view_update_timer.setSingleShot(True)
        self._view_update_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._view_update_timer.timeout.connect(self._flush_view_update)
        self._build_ui()
        self._connect_canvas_events()
        self.draw_placeholder()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        toolbar_row = QHBoxLayout()
        toolbar_row.setContentsMargins(0, 0, 0, 0)
        toolbar_row.setSpacing(6)
        self.rotate_button = QPushButton("Rotate")
        self.rotate_button.setCheckable(True)
        self.rotate_button.clicked.connect(
            lambda _checked=False: self._set_interaction_mode("rotate")
        )
        toolbar_row.addWidget(self.rotate_button)

        self.pan_button = QPushButton("Pan")
        self.pan_button.setCheckable(True)
        self.pan_button.clicked.connect(
            lambda _checked=False: self._set_interaction_mode("pan")
        )
        toolbar_row.addWidget(self.pan_button)

        self.zoom_button = QPushButton("Zoom")
        self.zoom_button.setCheckable(True)
        self.zoom_button.clicked.connect(
            lambda _checked=False: self._set_interaction_mode("zoom")
        )
        toolbar_row.addWidget(self.zoom_button)

        self.autoscale_button = QPushButton("Autoscale")
        self.autoscale_button.clicked.connect(self.autoscale_view)
        toolbar_row.addWidget(self.autoscale_button)

        self.reset_view_button = QPushButton("Reset View")
        self.reset_view_button.clicked.connect(self.reset_view)
        toolbar_row.addWidget(self.reset_view_button)

        self.show_mesh_checkbox = QCheckBox("Show Mesh Overlay")
        self.show_mesh_checkbox.setChecked(True)
        self.show_mesh_checkbox.toggled.connect(self._handle_mesh_toggle)
        toolbar_row.addWidget(self.show_mesh_checkbox)
        toolbar_row.addStretch(1)
        layout.addLayout(toolbar_row)

        contrast_row = QHBoxLayout()
        contrast_row.setContentsMargins(0, 0, 0, 0)
        contrast_row.setSpacing(6)
        contrast_row.addWidget(QLabel("Atom Contrast"))
        self.atom_contrast_spin = QDoubleSpinBox(self)
        self.atom_contrast_spin.setRange(0.0, 100.0)
        self.atom_contrast_spin.setDecimals(0)
        self.atom_contrast_spin.setSingleStep(5.0)
        self.atom_contrast_spin.setSuffix("%")
        self.atom_contrast_spin.setKeyboardTracking(False)
        self.atom_contrast_spin.setValue(self._atom_contrast * 100.0)
        self.atom_contrast_spin.valueChanged.connect(
            self._handle_atom_contrast_changed
        )
        contrast_row.addWidget(self.atom_contrast_spin)

        contrast_row.addWidget(QLabel("Mesh Contrast"))
        self.mesh_contrast_spin = QDoubleSpinBox(self)
        self.mesh_contrast_spin.setRange(0.0, 100.0)
        self.mesh_contrast_spin.setDecimals(0)
        self.mesh_contrast_spin.setSingleStep(5.0)
        self.mesh_contrast_spin.setSuffix("%")
        self.mesh_contrast_spin.setKeyboardTracking(False)
        self.mesh_contrast_spin.setValue(self._mesh_contrast * 100.0)
        self.mesh_contrast_spin.valueChanged.connect(
            self._handle_mesh_contrast_changed
        )
        contrast_row.addWidget(self.mesh_contrast_spin)

        contrast_row.addWidget(QLabel("Mesh Width"))
        self.mesh_linewidth_spin = QDoubleSpinBox(self)
        self.mesh_linewidth_spin.setRange(0.2, 10.0)
        self.mesh_linewidth_spin.setDecimals(2)
        self.mesh_linewidth_spin.setSingleStep(0.1)
        self.mesh_linewidth_spin.setSuffix(" px")
        self.mesh_linewidth_spin.setKeyboardTracking(False)
        self.mesh_linewidth_spin.setValue(self._mesh_linewidth)
        self.mesh_linewidth_spin.valueChanged.connect(
            self._handle_mesh_linewidth_changed
        )
        contrast_row.addWidget(self.mesh_linewidth_spin)

        self.mesh_color_button = QPushButton("Mesh Color")
        self.mesh_color_button.clicked.connect(self._choose_mesh_color)
        contrast_row.addWidget(self.mesh_color_button)

        self.point_atoms_checkbox = QCheckBox("Point Atoms")
        self.point_atoms_checkbox.toggled.connect(
            self._handle_point_atoms_toggle
        )
        contrast_row.addWidget(self.point_atoms_checkbox)
        contrast_row.addStretch(1)
        layout.addLayout(contrast_row)
        self._update_mesh_color_button_style()

        self.canvas.setMinimumHeight(520)
        layout.addWidget(self.canvas, stretch=1)
        self.setMinimumHeight(self.minimumSizeHint().height())
        self._set_interaction_mode("rotate")

    def _connect_canvas_events(self) -> None:
        self.canvas.mpl_connect("button_press_event", self._handle_press)
        self.canvas.mpl_connect("button_release_event", self._handle_release)
        self.canvas.mpl_connect("motion_notify_event", self._handle_motion)

    def _set_interaction_mode(self, mode: str) -> None:
        if mode not in {"rotate", "pan", "zoom"}:
            return
        self._interaction_mode = mode
        self.rotate_button.setChecked(mode == "rotate")
        self.pan_button.setChecked(mode == "pan")
        self.zoom_button.setChecked(mode == "zoom")
        self._update_canvas_cursor()

    def _update_canvas_cursor(self, *, dragging: bool = False) -> None:
        if dragging:
            if self._drag_mode == "zoom":
                cursor = Qt.CursorShape.SizeVerCursor
            else:
                cursor = Qt.CursorShape.ClosedHandCursor
        elif self._interaction_mode == "zoom":
            cursor = Qt.CursorShape.SizeVerCursor
        else:
            cursor = Qt.CursorShape.OpenHandCursor
        self.canvas.setCursor(cursor)

    def draw_placeholder(self) -> None:
        self.current_structure = None
        self._clear_pending_view_update()
        self.figure.clear()
        axis = self.figure.add_subplot(111)
        axis.text(
            0.5,
            0.57,
            "Load an XYZ or PDB structure to preview atoms, bonds, and the "
            "active-origin mesh frame.",
            ha="center",
            va="center",
            wrap=True,
            transform=axis.transAxes,
        )
        axis.text(
            0.5,
            0.40,
            "The viewer uses Blender-inspired atom colors and bond styling, "
            "with mouse-driven rotate, pan, and zoom controls.",
            ha="center",
            va="center",
            wrap=True,
            alpha=0.78,
            transform=axis.transAxes,
        )
        axis.set_axis_off()
        self.canvas.draw_idle()

    def set_structure(
        self,
        structure: ElectronDensityStructure | None,
        *,
        mesh_geometry: ElectronDensityMeshGeometry | None = None,
        reset_view: bool = True,
    ) -> None:
        if structure is None:
            self.draw_placeholder()
            return
        self.current_structure = structure
        self.current_mesh_geometry = mesh_geometry
        self._draw_view(reset_view=reset_view)

    def set_structure_preserving_display(
        self,
        structure: ElectronDensityStructure | None,
        *,
        mesh_geometry: ElectronDensityMeshGeometry | None = None,
    ) -> None:
        if structure is None:
            self.draw_placeholder()
            return
        self.current_structure = structure
        self.current_mesh_geometry = mesh_geometry
        self._atom_contrast = _clamp_fraction(
            float(self.atom_contrast_spin.value()) / 100.0
        )
        self._mesh_contrast = _clamp_fraction(
            float(self.mesh_contrast_spin.value()) / 100.0
        )
        self._mesh_linewidth = max(
            float(self.mesh_linewidth_spin.value()), 0.2
        )
        self._atom_render_mode = (
            "points" if self.point_atoms_checkbox.isChecked() else "balls"
        )
        self._draw_view(reset_view=False)

    def set_mesh_geometry(
        self,
        mesh_geometry: ElectronDensityMeshGeometry | None,
    ) -> None:
        self.current_mesh_geometry = mesh_geometry
        if self.current_structure is not None:
            self._draw_view(reset_view=False)

    def autoscale_view(self) -> None:
        if self.current_structure is None:
            return
        self._view_center = np.zeros(3, dtype=float)
        self._view_radius = self._default_view_radius(self.current_structure)
        self._schedule_view_update(immediate=True)

    def reset_view(self) -> None:
        if self.current_structure is None:
            return
        self._draw_view(reset_view=True)

    def _default_view_radius(
        self,
        structure: ElectronDensityStructure,
    ) -> float:
        max_structure_radius = float(
            np.max(np.linalg.norm(structure.centered_coordinates, axis=1))
        )
        mesh_radius = (
            0.0
            if self.current_mesh_geometry is None
            else float(self.current_mesh_geometry.domain_max_radius)
        )
        return max(
            max_structure_radius * 1.55,
            mesh_radius * 1.15,
            1.2,
        )

    def _draw_view(self, *, reset_view: bool) -> None:
        structure = self.current_structure
        if structure is None:
            self.draw_placeholder()
            return

        self._clear_pending_view_update()
        self.figure.clear()
        axis = self.figure.add_subplot(
            111,
            projection="3d",
            computed_zorder=False,
        )
        axis_reference = self.figure.add_axes(
            [0.05, 0.05, 0.14, 0.14],
            projection="3d",
            computed_zorder=False,
        )
        axis_reference.set_in_layout(False)
        self._axis = axis
        self._axis_reference = axis_reference
        self.figure.set_facecolor(_BACKGROUND_COLOR)
        axis.set_facecolor(_BACKGROUND_COLOR)
        axis.grid(False)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_zticks([])
        axis.set_xlabel("")
        axis.set_ylabel("")
        axis.set_zlabel("")
        try:
            axis.set_proj_type("ortho")
        except Exception:
            pass
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

        self._draw_bonds(axis, structure)
        self._draw_atoms(axis, structure)
        if self._mesh_visible and self.current_mesh_geometry is not None:
            self._draw_mesh_overlay(axis, self.current_mesh_geometry)
        self._draw_origin_guides(axis, structure)
        self._draw_origin_axes(axis, structure)
        self._draw_active_settings_readout(axis)
        self._draw_legend(axis, structure)

        if reset_view:
            self._scene_rotation = np.eye(3, dtype=float)
            self._view_center = np.zeros(3, dtype=float)
            self._view_radius = self._default_view_radius(structure)
        self._apply_view_to_axes(refresh_axis_reference=True)
        axis.set_title(structure.display_label)
        axis.axis("off")
        self.figure.subplots_adjust(
            left=0.01,
            right=0.86,
            bottom=0.02,
            top=0.94,
        )
        self.canvas.draw_idle()

    def _clear_pending_view_update(self) -> None:
        self._pending_view_update = False
        self._pending_axis_reference_refresh = False
        self._view_update_timer.stop()

    def _draw_atoms(self, axis, structure: ElectronDensityStructure) -> None:
        if self._atom_contrast <= 0.0:
            return
        coordinates = self._transform_coordinates(
            np.asarray(structure.centered_coordinates, dtype=float)
        )
        y_values = coordinates[:, 1]
        depth_min = float(np.min(y_values))
        depth_span = max(float(np.max(y_values) - depth_min), 1.0e-6)
        for atom_index in np.argsort(y_values):
            element = structure.elements[int(atom_index)]
            depth_factor = (
                float(y_values[atom_index]) - depth_min
            ) / depth_span
            color = style_atom_color(element, atom_style=DEFAULT_ATOM_STYLE)
            size = (
                style_display_radius(
                    element,
                    atom_style=DEFAULT_ATOM_STYLE,
                )
                * 30.0
            ) ** 2
            point = coordinates[int(atom_index)]
            if self._atom_render_mode == "points":
                axis.scatter(
                    [point[0]],
                    [point[1]],
                    [point[2]],
                    s=_POINT_ATOM_SIZE * (0.90 + depth_factor * 0.22),
                    c=[color],
                    alpha=0.95 * self._atom_contrast,
                    edgecolors="none",
                    linewidths=0.0,
                    zorder=4,
                    depthshade=False,
                )
                continue
            axis.scatter(
                [point[0]],
                [point[1]],
                [point[2]],
                s=size * (1.06 + depth_factor * 0.22),
                c=[_SHADOW_COLOR],
                alpha=0.10 * self._atom_contrast,
                linewidths=0.0,
                zorder=3,
            )
            axis.scatter(
                [point[0]],
                [point[1]],
                [point[2]],
                s=size * (0.84 + depth_factor * 0.34),
                c=[color],
                alpha=0.94 * self._atom_contrast,
                edgecolors=[_EDGE_COLOR],
                linewidths=1.0,
                zorder=4,
                depthshade=False,
            )

    def _draw_bonds(self, axis, structure: ElectronDensityStructure) -> None:
        if self._atom_contrast <= 0.0:
            return
        coordinates = self._transform_coordinates(
            np.asarray(structure.centered_coordinates, dtype=float)
        )
        if coordinates.size == 0:
            return
        style_defaults = atom_style_defaults(DEFAULT_ATOM_STYLE)
        bond_width = max(2.4, 10.0 * float(style_defaults["bond_radius"]))
        neutral_bond_color = style_neutral_bond_color(DEFAULT_ATOM_STYLE)
        y_values = coordinates[:, 1]
        depth_min = float(np.min(y_values))
        depth_span = max(float(np.max(y_values) - depth_min), 1.0e-6)
        for left_index, right_index in sorted(
            structure.bonds,
            key=lambda pair: float(y_values[pair[0]] + y_values[pair[1]]),
        ):
            start = coordinates[int(left_index)]
            end = coordinates[int(right_index)]
            midpoint = (start + end) * 0.5
            depth_factor = (
                float(y_values[left_index] + y_values[right_index]) * 0.5
                - depth_min
            ) / depth_span
            axis.plot(
                (start[0], end[0]),
                (start[1], end[1]),
                (start[2], end[2]),
                color=neutral_bond_color,
                linewidth=bond_width + 1.4,
                alpha=(0.24 + depth_factor * 0.10) * self._atom_contrast,
                solid_capstyle="round",
                zorder=1,
            )
            left_color = style_split_bond_color(
                structure.elements[int(left_index)],
                atom_style=DEFAULT_ATOM_STYLE,
            )
            right_color = style_split_bond_color(
                structure.elements[int(right_index)],
                atom_style=DEFAULT_ATOM_STYLE,
            )
            axis.plot(
                (start[0], midpoint[0]),
                (start[1], midpoint[1]),
                (start[2], midpoint[2]),
                color=left_color,
                linewidth=bond_width,
                alpha=(0.74 + depth_factor * 0.12) * self._atom_contrast,
                solid_capstyle="round",
                zorder=2,
            )
            axis.plot(
                (midpoint[0], end[0]),
                (midpoint[1], end[1]),
                (midpoint[2], end[2]),
                color=right_color,
                linewidth=bond_width,
                alpha=(0.74 + depth_factor * 0.12) * self._atom_contrast,
                solid_capstyle="round",
                zorder=2,
            )

    def _draw_origin_axes(
        self, axis, structure: ElectronDensityStructure
    ) -> None:
        mesh_radius = (
            0.0
            if self.current_mesh_geometry is None
            else float(self.current_mesh_geometry.domain_max_radius)
        )
        coordinates = np.asarray(structure.centered_coordinates, dtype=float)
        atom_radius = float(np.max(np.linalg.norm(coordinates, axis=1)))
        axis_length = max(mesh_radius * 0.32, atom_radius * 0.55, 1.0)
        for vector, (label, color) in zip(
            self._transform_coordinates(np.eye(3, dtype=float) * axis_length),
            _AXIS_COLORS,
            strict=False,
        ):
            axis.plot(
                (-vector[0], vector[0]),
                (-vector[1], vector[1]),
                (-vector[2], vector[2]),
                color=color,
                linewidth=1.4,
                alpha=0.90,
                zorder=10,
            )
            axis.text(
                vector[0] * 1.08,
                vector[1] * 1.08,
                vector[2] * 1.08,
                label,
                color=color,
                fontsize=8,
                zorder=11,
            )
        axis.scatter(
            [0.0],
            [0.0],
            [0.0],
            s=22,
            c=["#111827"],
            depthshade=False,
            zorder=11,
        )

    def _draw_origin_guides(
        self,
        axis,
        structure: ElectronDensityStructure,
    ) -> None:
        coordinates = np.asarray(structure.centered_coordinates, dtype=float)
        if coordinates.size == 0:
            return
        distances = np.linalg.norm(coordinates, axis=1)
        nearby_indices = np.flatnonzero(distances <= _ORIGIN_GUIDE_CUTOFF_A)
        if nearby_indices.size == 0:
            return
        transformed = self._transform_coordinates(coordinates[nearby_indices])
        for point in transformed:
            (line,) = axis.plot(
                (0.0, point[0]),
                (0.0, point[1]),
                (0.0, point[2]),
                color="#0f172a",
                linewidth=1.0,
                alpha=0.58,
                linestyle=(0, (4.0, 3.0)),
                zorder=9,
            )
            line.set_gid("origin-guide")

    def _draw_legend(self, axis, structure: ElectronDensityStructure) -> None:
        legend_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markersize=5.0 if self._atom_render_mode == "points" else 7.5,
                markerfacecolor=style_atom_color(
                    element, atom_style=DEFAULT_ATOM_STYLE
                ),
                markeredgecolor=_EDGE_COLOR,
                markeredgewidth=(
                    0.0 if self._atom_render_mode == "points" else 1.0
                ),
                label=f"{element} ({structure.element_counts[element]})",
            )
            for element in sorted(structure.element_counts)
        ]
        if np.any(
            np.linalg.norm(
                np.asarray(structure.centered_coordinates, dtype=float),
                axis=1,
            )
            <= _ORIGIN_GUIDE_CUTOFF_A
        ):
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    color="#0f172a",
                    linewidth=1.0,
                    alpha=0.58,
                    linestyle=(0, (4.0, 3.0)),
                    label="Origin guides (≤ 3 Å)",
                )
            )
        if self._mesh_visible and self.current_mesh_geometry is not None:
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    color=self._mesh_color,
                    linewidth=self._mesh_linewidth,
                    alpha=self._mesh_contrast,
                    label="Spherical mesh",
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
                fontsize=8.2,
            )
            frame = legend.get_frame()
            frame.set_facecolor(_BACKGROUND_COLOR)
            frame.set_edgecolor("#c6ccd4")
            frame.set_linewidth(0.8)

    def _draw_mesh_overlay(
        self,
        axis,
        mesh_geometry: ElectronDensityMeshGeometry,
    ) -> None:
        if self._mesh_contrast <= 0.0:
            return
        radii = _sample_values(
            np.asarray(mesh_geometry.radial_edges[1:], dtype=float),
            max_count=8,
        )
        theta_edges = _sample_values(
            np.asarray(mesh_geometry.theta_edges[1:-1], dtype=float),
            max_count=10,
        )
        phi_edges = _sample_values(
            np.asarray(mesh_geometry.phi_edges[:-1], dtype=float),
            max_count=12,
        )
        full_phi = np.linspace(0.0, 2.0 * np.pi, 150)
        full_theta = np.linspace(0.0, np.pi, 120)
        for radius in radii:
            for theta in theta_edges:
                sin_theta = float(np.sin(theta))
                cos_theta = float(np.cos(theta))
                x_values = radius * sin_theta * np.cos(full_phi)
                y_values = radius * sin_theta * np.sin(full_phi)
                z_values = np.full_like(full_phi, radius * cos_theta)
                transformed = self._transform_coordinates(
                    np.column_stack((x_values, y_values, z_values))
                )
                axis.plot(
                    transformed[:, 0],
                    transformed[:, 1],
                    transformed[:, 2],
                    color=self._mesh_color,
                    linewidth=self._mesh_linewidth,
                    alpha=0.26 * self._mesh_contrast,
                    zorder=8,
                )
            for phi in phi_edges:
                sin_theta = np.sin(full_theta)
                x_values = radius * sin_theta * np.cos(phi)
                y_values = radius * sin_theta * np.sin(phi)
                z_values = radius * np.cos(full_theta)
                transformed = self._transform_coordinates(
                    np.column_stack((x_values, y_values, z_values))
                )
                axis.plot(
                    transformed[:, 0],
                    transformed[:, 1],
                    transformed[:, 2],
                    color=self._mesh_color,
                    linewidth=self._mesh_linewidth,
                    alpha=0.26 * self._mesh_contrast,
                    zorder=8,
                )

        theta_centers = _sample_values(
            (
                np.asarray(mesh_geometry.theta_edges[:-1], dtype=float)
                + np.asarray(mesh_geometry.theta_edges[1:], dtype=float)
            )
            * 0.5,
            max_count=6,
        )
        phi_centers = _sample_values(
            (
                np.asarray(mesh_geometry.phi_edges[:-1], dtype=float)
                + np.asarray(mesh_geometry.phi_edges[1:], dtype=float)
            )
            * 0.5,
            max_count=8,
        )
        radial_values = np.linspace(
            0.0,
            float(mesh_geometry.domain_max_radius),
            28,
        )
        for theta in theta_centers:
            sin_theta = float(np.sin(theta))
            cos_theta = float(np.cos(theta))
            for phi in phi_centers:
                x_values = radial_values * sin_theta * np.cos(phi)
                y_values = radial_values * sin_theta * np.sin(phi)
                z_values = radial_values * cos_theta
                transformed = self._transform_coordinates(
                    np.column_stack((x_values, y_values, z_values))
                )
                axis.plot(
                    transformed[:, 0],
                    transformed[:, 1],
                    transformed[:, 2],
                    color=self._mesh_color,
                    linewidth=max(self._mesh_linewidth * 0.88, 0.2),
                    alpha=0.22 * self._mesh_contrast,
                    zorder=8,
                )

    def _draw_active_settings_readout(self, axis) -> None:
        readout = "\n".join(
            (
                f"ATOM {self._atom_contrast * 100.0:05.1f}%",
                f"MESH {self._mesh_contrast * 100.0:05.1f}%",
                f"LINE {self._mesh_linewidth:04.2f}px",
                f"COLOR {self._mesh_color.upper()}",
            )
        )
        artist = axis.text2D(
            0.02,
            0.98,
            readout,
            transform=axis.transAxes,
            ha="left",
            va="top",
            color=_ACTIVE_READOUT_COLOR,
            fontsize=8.8,
            fontfamily="DejaVu Sans Mono",
            fontweight="bold",
            bbox={
                "facecolor": _ACTIVE_READOUT_BACKGROUND,
                "edgecolor": "none",
                "boxstyle": "round,pad=0.32",
            },
            zorder=14,
        )
        artist.set_gid("active-visual-settings")

    def _update_mesh_color_button_style(self) -> None:
        self.mesh_color_button.setStyleSheet(
            "QPushButton {"
            f"background-color: {self._mesh_color};"
            "color: white;"
            "border: 1px solid #475569;"
            "padding: 4px 8px;"
            "}"
        )
        self.mesh_color_button.setText(
            f"Mesh Color {self._mesh_color.upper()}"
        )

    def _choose_mesh_color(self) -> None:
        selected = QColorDialog.getColor(
            QColor(self._mesh_color),
            self,
            "Select Mesh Color",
        )
        if not selected.isValid():
            return
        self._mesh_color = str(selected.name())
        self._update_mesh_color_button_style()
        if self.current_structure is not None:
            self._draw_view(reset_view=False)

    def _schedule_view_update(
        self,
        *,
        refresh_axis_reference: bool = False,
        immediate: bool = False,
    ) -> None:
        if self._axis is None:
            return
        self._pending_view_update = True
        self._pending_axis_reference_refresh = (
            self._pending_axis_reference_refresh
            or bool(refresh_axis_reference)
        )
        if immediate:
            self._view_update_timer.stop()
            self._flush_view_update()
            return
        if not self._view_update_timer.isActive():
            self._view_update_timer.start(_VIEW_UPDATE_INTERVAL_MS)

    def _flush_view_update(self) -> None:
        if self._axis is None or not self._pending_view_update:
            return
        refresh_axis_reference = self._pending_axis_reference_refresh
        self._pending_view_update = False
        self._pending_axis_reference_refresh = False
        self._apply_view_to_axes(refresh_axis_reference=refresh_axis_reference)
        self.canvas.draw_idle()

    def _apply_view_to_axes(
        self,
        *,
        refresh_axis_reference: bool = True,
    ) -> None:
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
        self._axis.view_init(elev=_CAMERA_ELEV, azim=_CAMERA_AZIM)
        if refresh_axis_reference:
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
        axis_ref.view_init(elev=_CAMERA_ELEV, azim=_CAMERA_AZIM)
        axis_length = 0.8
        label_scale = 1.15
        axis_vectors = self._transform_coordinates(
            np.eye(3, dtype=float) * axis_length
        )
        for vector, (label, color) in zip(
            axis_vectors,
            _AXIS_COLORS,
            strict=False,
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
            [0.0],
            [0.0],
            [0.0],
            s=10,
            c=["#6b7280"],
            depthshade=False,
        )
        axis_ref.set_xlim(-1.0, 1.0)
        axis_ref.set_ylim(-1.0, 1.0)
        axis_ref.set_zlim(-1.0, 1.0)

    def _transform_coordinates(self, coordinates: np.ndarray) -> np.ndarray:
        array = np.asarray(coordinates, dtype=float)
        if array.size == 0:
            return array.reshape((-1, 3))
        return np.asarray(array @ self._scene_rotation.T, dtype=float)

    def _camera_basis_vectors(self) -> tuple[np.ndarray, np.ndarray]:
        azimuth = np.radians(_CAMERA_AZIM)
        elevation = np.radians(_CAMERA_ELEV)
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
        right_norm = float(np.linalg.norm(right))
        if right_norm < 1.0e-8:
            right = np.array([1.0, 0.0, 0.0], dtype=float)
        else:
            right /= right_norm
        up = np.cross(right, forward)
        up_norm = float(np.linalg.norm(up))
        if up_norm < 1.0e-8:
            up = np.array([0.0, 1.0, 0.0], dtype=float)
        else:
            up /= up_norm
        return right, up

    def _handle_press(self, event) -> None:
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
        self._drag_start_center = self._view_center.copy()
        self._drag_start_radius = self._view_radius
        self._drag_start_rotation = self._scene_rotation.copy()
        self._update_canvas_cursor(dragging=True)

    def _handle_release(self, _event) -> None:
        self._drag_mode = None
        if self._view_update_timer.isActive():
            self._view_update_timer.stop()
            self._flush_view_update()
        self._update_canvas_cursor()

    def _handle_motion(self, event) -> None:
        if self._axis is None or self._drag_mode is None:
            return
        delta_x = float(event.x) - self._drag_start_xy[0]
        delta_y = float(event.y) - self._drag_start_xy[1]
        width = max(float(self.canvas.width()), 1.0)
        height = max(float(self.canvas.height()), 1.0)
        if self._drag_mode == "rotate":
            right, up = self._camera_basis_vectors()
            horizontal_rotation = _rotation_matrix_from_axis_angle(
                up,
                -delta_x * np.pi / width,
            )
            vertical_rotation = _rotation_matrix_from_axis_angle(
                right,
                delta_y * np.pi / height,
            )
            self._scene_rotation = _orthonormalize_rotation(
                horizontal_rotation
                @ vertical_rotation
                @ self._drag_start_rotation
            )
            self._draw_view(reset_view=False)
            return
        elif self._drag_mode == "pan":
            data_scale = (2.0 * self._view_radius) / max(width, height)
            right, up = self._camera_basis_vectors()
            self._view_center = (
                self._drag_start_center
                + right * (-delta_x * data_scale)
                + up * (delta_y * data_scale)
            )
        elif self._drag_mode == "zoom":
            zoom_factor = float(np.exp(delta_y / height * 2.0))
            self._view_radius = float(
                np.clip(self._drag_start_radius * zoom_factor, 0.2, 1.0e6)
            )
        self._schedule_view_update(refresh_axis_reference=False)

    def _handle_scroll(self, event) -> None:
        del event

    def _handle_mesh_toggle(self, checked: bool) -> None:
        self._mesh_visible = bool(checked)
        if self.current_structure is not None:
            self._draw_view(reset_view=False)

    def _handle_atom_contrast_changed(self, value: float) -> None:
        self._atom_contrast = _clamp_fraction(float(value) / 100.0)
        if self.current_structure is not None:
            self._draw_view(reset_view=False)

    def _handle_mesh_contrast_changed(self, value: float) -> None:
        self._mesh_contrast = _clamp_fraction(float(value) / 100.0)
        if self.current_structure is not None:
            self._draw_view(reset_view=False)

    def _handle_mesh_linewidth_changed(self, value: float) -> None:
        self._mesh_linewidth = max(float(value), 0.2)
        if self.current_structure is not None:
            self._draw_view(reset_view=False)

    def _handle_point_atoms_toggle(self, checked: bool) -> None:
        self._atom_render_mode = "points" if checked else "balls"
        if self.current_structure is not None:
            self._draw_view(reset_view=False)


__all__ = [
    "ElectronDensityFourierPreviewPlot",
    "ElectronDensityProfileOverlay",
    "ElectronDensityProfilePlot",
    "ElectronDensityResidualPlot",
    "ElectronDensityScatteringPlot",
    "ElectronDensityStructureViewer",
]
