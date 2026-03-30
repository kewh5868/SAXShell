from __future__ import annotations

import math

import numpy as np
from matplotlib import colormaps
from matplotlib import colors as mcolors
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import (
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from saxshell.clusterdynamics.workflow import ClusterDynamicsResult

PLOT_COLORMAPS = ("viridis", "magma", "cividis", "inferno", "turbo")
DISPLAY_MODE_LABELS = {
    "count": "Counts / bin",
    "fraction": "Fraction / bin",
    "mean_count": "Mean count / frame",
}
DISPLAY_MODE_COLORBAR_LABELS = {
    "count": "Clusters in bin",
    "fraction": "Cluster fraction",
    "mean_count": "Mean clusters per frame",
}
OVERLAY_SERIES = (
    ("None", None),
    ("Temperature", "temperature"),
    ("Potential Energy", "potential"),
    ("Kinetic Energy", "kinetic"),
)
OVERLAY_COLORS = {
    "temperature": "#1f77b4",
    "potential": "#2e8b57",
    "kinetic": "#c0392b",
}


class ClusterDynamicsPlotPanel(QWidget):
    """Interactive time-binned cluster heatmap panel."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._result: ClusterDynamicsResult | None = None
        self._build_ui()
        self.refresh_plot()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)

        controls_widget = QWidget()
        controls = QHBoxLayout(controls_widget)
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(8)

        controls.addWidget(QLabel("Heatmap"))
        self.display_mode_combo = QComboBox()
        for mode, label in DISPLAY_MODE_LABELS.items():
            self.display_mode_combo.addItem(label, mode)
        self.display_mode_combo.setCurrentIndex(1)
        self.display_mode_combo.currentIndexChanged.connect(
            lambda _index: self.refresh_plot()
        )
        controls.addWidget(self.display_mode_combo)

        controls.addWidget(QLabel("Units"))
        self.time_unit_combo = QComboBox()
        self.time_unit_combo.addItem("fs", "fs")
        self.time_unit_combo.addItem("ps", "ps")
        self.time_unit_combo.currentIndexChanged.connect(
            lambda _index: self.refresh_plot()
        )
        controls.addWidget(self.time_unit_combo)

        controls.addWidget(QLabel("Colormap"))
        self.colormap_combo = QComboBox()
        for cmap_name in PLOT_COLORMAPS:
            self.colormap_combo.addItem(cmap_name, cmap_name)
        self.colormap_combo.currentIndexChanged.connect(
            lambda _index: self.refresh_plot()
        )
        controls.addWidget(self.colormap_combo)

        controls.addWidget(QLabel("Lower q"))
        self.lower_quantile_spin = QDoubleSpinBox()
        self.lower_quantile_spin.setDecimals(2)
        self.lower_quantile_spin.setRange(0.0, 0.95)
        self.lower_quantile_spin.setSingleStep(0.05)
        self.lower_quantile_spin.setValue(0.05)
        self.lower_quantile_spin.valueChanged.connect(
            self._on_quantile_changed
        )
        controls.addWidget(self.lower_quantile_spin)

        controls.addWidget(QLabel("Upper q"))
        self.upper_quantile_spin = QDoubleSpinBox()
        self.upper_quantile_spin.setDecimals(2)
        self.upper_quantile_spin.setRange(0.05, 1.0)
        self.upper_quantile_spin.setSingleStep(0.05)
        self.upper_quantile_spin.setValue(0.95)
        self.upper_quantile_spin.valueChanged.connect(
            self._on_quantile_changed
        )
        controls.addWidget(self.upper_quantile_spin)

        controls.addWidget(QLabel("Overlay"))
        self.overlay_combo = QComboBox()
        for label, data in OVERLAY_SERIES:
            self.overlay_combo.addItem(label, data)
        self.overlay_combo.currentIndexChanged.connect(
            lambda _index: self.refresh_plot()
        )
        controls.addWidget(self.overlay_combo)
        controls.addStretch(1)

        root.addWidget(controls_widget)

        self.figure = Figure(figsize=(9.2, 7.2))
        self.canvas = FigureCanvas(self.figure)
        root.addWidget(NavigationToolbar(self.canvas, self))
        root.addWidget(self.canvas, stretch=1)

    def set_result(self, result: ClusterDynamicsResult | None) -> None:
        self._result = result
        has_energy = bool(
            result is not None and result.energy_data is not None
        )
        self.overlay_combo.setEnabled(has_energy)
        if not has_energy:
            self.overlay_combo.setCurrentIndex(0)
        self.refresh_plot()

    def refresh_plot(self) -> None:
        self.figure.clear()
        if self._result is None:
            axis = self.figure.add_subplot(111)
            self._draw_placeholder(
                axis,
                "Run the analysis to render the cluster-distribution heatmap.",
            )
            self.canvas.draw_idle()
            return

        if self._result.bin_count == 0:
            axis = self.figure.add_subplot(111)
            self._draw_placeholder(
                axis,
                "No time bins are available for the current selection.",
            )
            self.canvas.draw_idle()
            return

        matrix = self._result.matrix(self._display_mode())
        if matrix.size == 0 or len(self._result.cluster_labels) == 0:
            axis = self.figure.add_subplot(111)
            self._draw_placeholder(
                axis,
                "No clusters were detected in the selected time window.",
            )
            self.canvas.draw_idle()
            return

        overlay_name = self.overlay_combo.currentData()
        show_overlay = bool(
            overlay_name is not None and self._result.energy_data is not None
        )

        if show_overlay:
            grid = self.figure.add_gridspec(
                2,
                1,
                height_ratios=[4.0, 1.2],
                hspace=0.08,
            )
            heatmap_axis = self.figure.add_subplot(grid[0, 0])
            overlay_axis = self.figure.add_subplot(
                grid[1, 0],
                sharex=heatmap_axis,
            )
        else:
            heatmap_axis = self.figure.add_subplot(111)
            overlay_axis = None

        time_unit = self.time_unit_combo.currentData()
        time_edges = self._result.time_edges(time_unit)
        cmap = colormaps[self.colormap_combo.currentData()]
        norm = self._quantile_norm(matrix)

        image = heatmap_axis.imshow(
            matrix,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            extent=(
                float(time_edges[0]),
                float(time_edges[-1]),
                -0.5,
                len(self._result.cluster_labels) - 0.5,
            ),
            cmap=cmap,
            norm=norm,
        )
        colorbar = self.figure.colorbar(image, ax=heatmap_axis, pad=0.02)
        colorbar.set_label(DISPLAY_MODE_COLORBAR_LABELS[self._display_mode()])

        label_step = max(
            1,
            int(math.ceil(len(self._result.cluster_labels) / 24)),
        )
        tick_positions = np.arange(
            0, len(self._result.cluster_labels), label_step
        )
        heatmap_axis.set_yticks(tick_positions)
        heatmap_axis.set_yticklabels(
            [self._result.cluster_labels[index] for index in tick_positions]
        )
        heatmap_axis.set_ylabel("Cluster label")
        heatmap_axis.set_xlim(float(time_edges[0]), float(time_edges[-1]))
        heatmap_axis.set_title(
            "Time-Binned Cluster Distribution "
            f"({DISPLAY_MODE_LABELS[self._display_mode()]})"
        )
        if overlay_axis is None:
            heatmap_axis.set_xlabel(f"Time ({time_unit})")
        else:
            heatmap_axis.tick_params(labelbottom=False)

        if overlay_axis is not None and overlay_name is not None:
            x_values, y_values, y_label = self._result.energy_series(
                overlay_name,
                unit=time_unit,
            )
            overlay_axis.plot(
                x_values,
                y_values,
                color=OVERLAY_COLORS.get(overlay_name, "#333333"),
                linewidth=1.5,
            )
            overlay_axis.set_ylabel(y_label)
            overlay_axis.set_xlabel(f"Time ({time_unit})")
            overlay_axis.grid(alpha=0.25, linestyle=":")

        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _display_mode(self) -> str:
        value = self.display_mode_combo.currentData()
        return "fraction" if value is None else str(value)

    def _on_quantile_changed(self) -> None:
        lower = self.lower_quantile_spin.value()
        upper = self.upper_quantile_spin.value()
        if lower >= upper:
            if self.sender() is self.lower_quantile_spin:
                self.upper_quantile_spin.blockSignals(True)
                self.upper_quantile_spin.setValue(min(lower + 0.05, 1.0))
                self.upper_quantile_spin.blockSignals(False)
            else:
                self.lower_quantile_spin.blockSignals(True)
                self.lower_quantile_spin.setValue(max(upper - 0.05, 0.0))
                self.lower_quantile_spin.blockSignals(False)
        self.refresh_plot()

    def _quantile_norm(self, matrix: np.ndarray) -> mcolors.Normalize:
        values = np.asarray(matrix, dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return mcolors.Normalize(vmin=0.0, vmax=1.0)

        positive = finite[finite > 0.0]
        if positive.size:
            finite = positive

        lower_q = float(self.lower_quantile_spin.value())
        upper_q = float(self.upper_quantile_spin.value())
        vmin = float(np.quantile(finite, lower_q))
        vmax = float(np.quantile(finite, upper_q))
        if vmax <= vmin:
            vmin = float(np.min(finite))
            vmax = float(np.max(finite))
        if vmax <= vmin:
            vmax = vmin + 1.0
        return mcolors.Normalize(vmin=vmin, vmax=vmax)

    @staticmethod
    def _draw_placeholder(axis, message: str) -> None:
        axis.text(
            0.5,
            0.5,
            message,
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_frame_on(False)


__all__ = ["ClusterDynamicsPlotPanel"]
