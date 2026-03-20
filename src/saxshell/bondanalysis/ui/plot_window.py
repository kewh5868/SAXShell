from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib import colormaps
from matplotlib import colors as mcolors
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import (
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QColorDialog,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from saxshell.bondanalysis.results import (
    BondAnalysisPlotRequest,
    export_plot_request_csv,
    recommended_plot_request_filename,
)


class BondAnalysisPlotTab(QWidget):
    """One bondanalysis plot tab inside the shared plotting window."""

    def __init__(
        self,
        plot_request: BondAnalysisPlotRequest,
        default_output_dir: str | Path,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.plot_request = plot_request
        self.default_output_dir = Path(default_output_dir)
        self._series_colors = self._default_series_colors()
        self._series_states = self._initial_series_states()
        self._updating_series_list = False
        self._build_ui()
        self.refresh_plot()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        self.controls_widget = QWidget()
        controls = QHBoxLayout(self.controls_widget)
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(8)

        self.save_button = QPushButton("Save Plot Data As...")
        self.save_button.clicked.connect(self.save_plot_data_as)
        controls.addWidget(self.save_button)

        controls.addWidget(QLabel("Bin size"))
        self.bin_size_spin = QDoubleSpinBox()
        self.bin_size_spin.setDecimals(3)
        self.bin_size_spin.setRange(0.001, 1000000.0)
        self.bin_size_spin.setSingleStep(0.05)
        self.bin_size_spin.setValue(self._default_bin_size())
        self.bin_size_spin.valueChanged.connect(
            lambda _value: self.refresh_plot()
        )
        controls.addWidget(self.bin_size_spin)

        controls.addWidget(QLabel("Transparency"))
        self.transparency_spin = QDoubleSpinBox()
        self.transparency_spin.setDecimals(2)
        self.transparency_spin.setRange(0.05, 1.0)
        self.transparency_spin.setSingleStep(0.05)
        self.transparency_spin.setValue(0.45)
        self.transparency_spin.valueChanged.connect(
            lambda _value: self.refresh_plot()
        )
        controls.addWidget(self.transparency_spin)

        self.series_color_label = QLabel("Overlay colors")
        controls.addWidget(self.series_color_label)
        self.series_color_container = QWidget()
        self.series_color_layout = QHBoxLayout(self.series_color_container)
        self.series_color_layout.setContentsMargins(0, 0, 0, 0)
        self.series_color_layout.setSpacing(6)
        self.series_color_list = QListWidget()
        self.series_color_list.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.series_color_list.setDragDropMode(
            QAbstractItemView.DragDropMode.InternalMove
        )
        self.series_color_list.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.series_color_list.setDragEnabled(True)
        self.series_color_list.setAcceptDrops(True)
        self.series_color_list.setDropIndicatorShown(True)
        self.series_color_list.setMaximumHeight(90)
        self.series_color_list.setMinimumWidth(240)
        self.series_color_list.setToolTip(
            "Drag items to change histogram stacking order. "
            "Double-click an item to change its color."
        )
        self.series_color_list.itemDoubleClicked.connect(
            self._choose_series_color_for_item
        )
        self.series_color_list.model().rowsMoved.connect(
            self._on_series_order_changed
        )
        self.series_color_layout.addWidget(self.series_color_list)
        controls.addWidget(self.series_color_container)
        controls.addStretch(1)

        root.addWidget(self.controls_widget)

        self.figure = Figure(figsize=(8.4, 6.4))
        self.canvas = FigureCanvas(self.figure)
        self.axis = self.figure.add_subplot(111)
        root.addWidget(NavigationToolbar(self.canvas, self))
        root.addWidget(self.canvas, stretch=1)

        self._refresh_series_color_list()

    def save_plot_data_as(self) -> None:
        default_filename = recommended_plot_request_filename(self.plot_request)
        default_path = self.default_output_dir / default_filename
        selected_path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save Bondanalysis Plot Data",
            str(default_path),
            "CSV Files (*.csv)",
        )
        if not selected_path:
            return
        csv_path = export_plot_request_csv(self.plot_request, selected_path)
        QMessageBox.information(
            self,
            "Bondanalysis Plot Data",
            f"Saved plot data to:\n{csv_path}",
        )

    def refresh_plot(self) -> None:
        self.axis.clear()
        non_empty_series = self._ordered_series_states()
        self._update_overlay_controls(non_empty_series)

        if not non_empty_series:
            self.axis.text(
                0.5,
                0.5,
                "No computed values were found for this selection.",
                ha="center",
                va="center",
                transform=self.axis.transAxes,
            )
            self.axis.set_title(self.plot_request.title)
            self.axis.set_xlabel(self.plot_request.xlabel)
            self.axis.set_ylabel("Count")
            self.canvas.draw_idle()
            return

        combined_values = np.concatenate(
            [series["values"] for series in non_empty_series]
        )
        histogram_edges = self._histogram_edges(combined_values)
        stats = self._distribution_stats(combined_values, histogram_edges)

        if len(non_empty_series) == 1:
            series = non_empty_series[0]
            self.axis.hist(
                series["values"],
                bins=histogram_edges,
                color="#2e8b57",
                edgecolor="black",
                linewidth=0.8,
                alpha=1.0,
                label=series["label"],
            )
        else:
            for index, series in enumerate(non_empty_series):
                self.axis.hist(
                    series["values"],
                    bins=histogram_edges,
                    color=series["color"],
                    edgecolor="black",
                    linewidth=0.8,
                    alpha=self.transparency_spin.value(),
                    label=series["label"],
                )

        self.axis.axvline(
            stats["mean"],
            color="black",
            linestyle="--",
            linewidth=1.4,
            label="Mean",
        )
        self.axis.axvline(
            stats["median"],
            color="red",
            linestyle="--",
            linewidth=1.4,
            label="Median",
        )

        self.axis.set_title(self.plot_request.title)
        self.axis.set_xlabel(self.plot_request.xlabel)
        self.axis.set_ylabel("Count")

        legend = self.axis.legend(loc="upper right", frameon=True)
        stats_y = self._stats_box_y(legend)
        self.axis.text(
            0.98,
            stats_y,
            "\n".join(
                (
                    f"Mean: {stats['mean']:.3f}",
                    f"Median: {stats['median']:.3f}",
                    f"Mode: {stats['mode']:.3f}",
                )
            ),
            ha="right",
            va="top",
            transform=self.axis.transAxes,
            bbox={
                "boxstyle": "round",
                "facecolor": "white",
                "edgecolor": "black",
                "alpha": 0.92,
            },
        )

        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _update_overlay_controls(self, non_empty_series: list[dict]) -> None:
        show_overlay_controls = len(non_empty_series) > 1
        self.transparency_spin.setEnabled(show_overlay_controls)
        self.series_color_label.setVisible(show_overlay_controls)
        self.series_color_container.setVisible(show_overlay_controls)
        self._refresh_series_color_list(non_empty_series)

    def _refresh_series_color_list(
        self, non_empty_series: list[dict] | None = None
    ) -> None:
        series = (
            non_empty_series
            if non_empty_series is not None
            else self._ordered_series_states()
        )
        self._updating_series_list = True
        self.series_color_list.clear()
        if len(series) <= 1:
            self._updating_series_list = False
            return

        for entry in series:
            item = QListWidgetItem(entry["label"])
            item.setData(Qt.ItemDataRole.UserRole, entry["key"])
            self._style_series_color_item(item, entry["color"])
            self.series_color_list.addItem(item)
        self._updating_series_list = False

    def _choose_series_color_for_item(self, item: QListWidgetItem) -> None:
        state = self._series_state_from_key(
            str(item.data(Qt.ItemDataRole.UserRole))
        )
        if state is None:
            return
        initial = QColor(state["color"])
        selected = QColorDialog.getColor(
            initial,
            self,
            f"Select color for {state['label']}",
        )
        if not selected.isValid():
            return
        state["color"] = selected.name()
        self._style_series_color_item(item, selected.name())
        self.refresh_plot()

    @staticmethod
    def _style_series_color_item(item: QListWidgetItem, color: str) -> None:
        qcolor = QColor(color)
        item.setBackground(qcolor)
        item.setForeground(
            QColor("black") if qcolor.lightnessF() > 0.62 else QColor("white")
        )

    def _default_bin_size(self) -> float:
        combined_values = np.concatenate(
            [
                series.values
                for series in self.plot_request.series
                if series.values.size > 0
            ]
            or [np.array([0.0, 1.0], dtype=float)]
        )
        value_range = float(np.max(combined_values) - np.min(combined_values))
        if value_range <= 0:
            return 0.1
        recommended = value_range / 24.0
        return max(0.01, round(recommended, 3))

    def _default_series_colors(self) -> list[str]:
        color_map = colormaps["tab10"]
        return [
            mcolors.to_hex(color_map(index % 10))
            for index in range(max(1, len(self.plot_request.series)))
        ]

    @staticmethod
    def _fallback_series_color(index: int) -> str:
        return mcolors.to_hex(colormaps["tab10"](index % 10))

    def _initial_series_states(self) -> list[dict[str, object]]:
        states: list[dict[str, object]] = []
        non_empty_series = [
            entry
            for entry in self.plot_request.series
            if entry.values.size > 0
        ]
        while len(self._series_colors) < len(non_empty_series):
            self._series_colors.append(
                self._fallback_series_color(len(self._series_colors))
            )
        for index, series in enumerate(non_empty_series):
            states.append(
                {
                    "key": f"series-{index}",
                    "label": series.label,
                    "values": series.values,
                    "color": self._series_colors[index],
                }
            )
        return states

    def _ordered_series_states(self) -> list[dict[str, object]]:
        if self.series_color_list.count() <= 1:
            return list(self._series_states)
        ordered_keys = [
            str(
                self.series_color_list.item(index).data(
                    Qt.ItemDataRole.UserRole
                )
            )
            for index in range(self.series_color_list.count())
        ]
        lookup = {str(entry["key"]): entry for entry in self._series_states}
        ordered = [lookup[key] for key in ordered_keys if key in lookup]
        if len(ordered) == len(self._series_states):
            return ordered
        return list(self._series_states)

    def _series_state_from_key(self, key: str) -> dict[str, object] | None:
        for entry in self._series_states:
            if str(entry["key"]) == key:
                return entry
        return None

    def _on_series_order_changed(self, *_args) -> None:
        if self._updating_series_list:
            return
        self._series_states = self._ordered_series_states()
        self.refresh_plot()

    def _histogram_edges(self, values: np.ndarray) -> np.ndarray:
        bin_size = max(self.bin_size_spin.value(), 1e-6)
        value_min = float(np.min(values))
        value_max = float(np.max(values))
        if np.isclose(value_min, value_max):
            half_width = bin_size / 2.0
            return np.array([value_min - half_width, value_min + half_width])

        bin_count = max(1, int(np.ceil((value_max - value_min) / bin_size)))
        edges = value_min + np.arange(bin_count + 1) * bin_size
        if edges[-1] < value_max:
            edges = np.append(edges, edges[-1] + bin_size)
        return edges

    @staticmethod
    def _distribution_stats(
        values: np.ndarray, edges: np.ndarray
    ) -> dict[str, float]:
        mean_value = float(np.mean(values))
        median_value = float(np.median(values))
        counts, histogram_edges = np.histogram(values, bins=edges)
        if counts.size == 0:
            mode_value = mean_value
        else:
            peak_index = int(np.argmax(counts))
            mode_value = float(
                0.5
                * (
                    histogram_edges[peak_index]
                    + histogram_edges[peak_index + 1]
                )
            )
        return {
            "mean": mean_value,
            "median": median_value,
            "mode": mode_value,
        }

    @staticmethod
    def _stats_box_y(legend) -> float:
        if legend is None:
            return 0.98
        legend_entries = max(1, len(legend.get_texts()))
        return max(0.18, 0.98 - 0.075 * legend_entries)


class BondAnalysisPlotWindow(QMainWindow):
    """Shared tabbed bondanalysis plotting workspace."""

    def __init__(
        self,
        plot_request: BondAnalysisPlotRequest,
        default_output_dir: str | Path,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.default_output_dir = Path(default_output_dir)
        self.tab_widget = QTabWidget()
        self.tab_widget.setTabsClosable(True)
        self.tab_widget.tabCloseRequested.connect(self._close_tab)
        self.tab_widget.currentChanged.connect(self._sync_active_tab)
        self.setCentralWidget(self.tab_widget)
        self.resize(1040, 820)

        self._next_tab_shortcut = QShortcut(
            QKeySequence(Qt.Key.Key_Right), self
        )
        self._next_tab_shortcut.activated.connect(self._select_next_tab)
        self._previous_tab_shortcut = QShortcut(
            QKeySequence(Qt.Key.Key_Left), self
        )
        self._previous_tab_shortcut.activated.connect(
            self._select_previous_tab
        )

        self.add_plot_request(plot_request)

    def add_plot_request(self, plot_request: BondAnalysisPlotRequest) -> None:
        plot_tab = BondAnalysisPlotTab(
            plot_request,
            default_output_dir=self.default_output_dir,
            parent=self,
        )
        tab_index = self.tab_widget.addTab(
            plot_tab, self._unique_tab_label(plot_request.title)
        )
        self.tab_widget.setCurrentIndex(tab_index)
        self._sync_active_tab(tab_index)

    @property
    def current_plot_tab(self) -> BondAnalysisPlotTab | None:
        widget = self.tab_widget.currentWidget()
        if isinstance(widget, BondAnalysisPlotTab):
            return widget
        return None

    @property
    def plot_request(self) -> BondAnalysisPlotRequest:
        current = self.current_plot_tab
        if current is None:
            raise RuntimeError("No active bondanalysis plot tab is available.")
        return current.plot_request

    @property
    def controls_widget(self) -> QWidget:
        return self.current_plot_tab.controls_widget

    @property
    def bin_size_spin(self) -> QDoubleSpinBox:
        return self.current_plot_tab.bin_size_spin

    @property
    def transparency_spin(self) -> QDoubleSpinBox:
        return self.current_plot_tab.transparency_spin

    @property
    def series_color_container(self) -> QWidget:
        return self.current_plot_tab.series_color_container

    @property
    def series_color_list(self) -> QListWidget:
        return self.current_plot_tab.series_color_list

    @property
    def axis(self):
        return self.current_plot_tab.axis

    def save_plot_data_as(self) -> None:
        current = self.current_plot_tab
        if current is not None:
            current.save_plot_data_as()

    def refresh_plot(self) -> None:
        current = self.current_plot_tab
        if current is not None:
            current.refresh_plot()

    def _close_tab(self, index: int) -> None:
        widget = self.tab_widget.widget(index)
        self.tab_widget.removeTab(index)
        if widget is not None:
            widget.deleteLater()
        if self.tab_widget.count() == 0:
            self.close()
            return
        self._sync_active_tab(self.tab_widget.currentIndex())

    def _select_next_tab(self) -> None:
        count = self.tab_widget.count()
        if count <= 1:
            return
        next_index = (self.tab_widget.currentIndex() + 1) % count
        self.tab_widget.setCurrentIndex(next_index)

    def _select_previous_tab(self) -> None:
        count = self.tab_widget.count()
        if count <= 1:
            return
        next_index = (self.tab_widget.currentIndex() - 1) % count
        self.tab_widget.setCurrentIndex(next_index)

    def _sync_active_tab(self, _index: int) -> None:
        current = self.current_plot_tab
        if current is None:
            self.setWindowTitle("Bond Analysis Plots")
            return
        self.setWindowTitle(
            f"Bond Analysis Plots - {current.plot_request.title}"
        )

    def _unique_tab_label(self, title: str) -> str:
        base = self._tab_label(title)
        existing = {
            self.tab_widget.tabText(index)
            for index in range(self.tab_widget.count())
        }
        if base not in existing:
            return base
        suffix = 2
        while f"{base} ({suffix})" in existing:
            suffix += 1
        return f"{base} ({suffix})"

    @staticmethod
    def _tab_label(title: str) -> str:
        if len(title) <= 36:
            return title
        return title[:33] + "..."
