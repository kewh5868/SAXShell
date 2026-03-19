from __future__ import annotations

import json
import math
import re

import numpy as np
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg,
    NavigationToolbar2QT,
)
from matplotlib.figure import Figure
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from scipy import stats

from saxshell.saxs.dream import BASE_DISTRIBUTIONS, DreamParameterEntry


class WeightDistributionPreviewWindow(QMainWindow):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self) -> None:
        self.setWindowTitle("DREAM Weight Prior Preview")
        self.resize(900, 620)

        central = QWidget()
        layout = QVBoxLayout(central)
        self.figure = Figure(figsize=(8, 5))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setCentralWidget(central)

    def plot_entries(self, entries: list[DreamParameterEntry]) -> None:
        self.figure.clear()
        axis = self.figure.add_subplot(111)
        x_limits: list[tuple[float, float]] = []
        plotted = 0

        for entry in entries:
            distribution = getattr(stats, entry.distribution)
            x_min, x_max = _distribution_domain(entry)
            if not np.isfinite(x_min) or not np.isfinite(x_max):
                continue
            x_values = np.linspace(x_min, x_max, 300)
            y_values = distribution.pdf(x_values, **entry.dist_params)
            if not np.all(np.isfinite(y_values)):
                continue
            legend_label = entry.param
            if entry.structure.strip():
                legend_label = f"{entry.param} ({entry.structure})"
            axis.plot(
                x_values,
                y_values,
                linewidth=1.6,
                label=legend_label,
            )
            x_limits.append((x_min, x_max))
            plotted += 1

        if plotted == 0:
            axis.text(
                0.5,
                0.5,
                "No valid w<##> prior distributions are available to preview.",
                ha="center",
                va="center",
            )
            axis.set_axis_off()
        else:
            axis.set_xlabel("Value")
            axis.set_ylabel("Density")
            axis.set_title("Weight prior distributions")
            axis.legend(loc="best", fontsize="small")
            axis.set_xlim(
                min(limit[0] for limit in x_limits),
                max(limit[1] for limit in x_limits),
            )

        self.figure.tight_layout()
        self.canvas.draw()


class DistributionSetupWindow(QMainWindow):
    saved = Signal(list)

    def __init__(
        self,
        entries: list[DreamParameterEntry],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._entries = entries
        self._has_existing_parameter_map = False
        self._was_saved = False
        self._weight_preview_window: WeightDistributionPreviewWindow | None = (
            None
        )
        self._build_ui()
        self.load_entries(entries)

    def _build_ui(self) -> None:
        self.setWindowTitle("SAXS DREAM Prior Setup")
        self.resize(1200, 720)

        central = QWidget()
        root = QHBoxLayout(central)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        top_row = QHBoxLayout()
        top_row.addStretch(1)
        self.preview_weight_priors_button = QPushButton(
            "Preview Weight Priors"
        )
        self.preview_weight_priors_button.setToolTip(
            "Open a shared density plot of all current w<##> prior "
            "distributions from the table."
        )
        self.preview_weight_priors_button.clicked.connect(
            self._show_weight_prior_preview
        )
        top_row.addWidget(self.preview_weight_priors_button)
        left_layout.addLayout(top_row)
        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(
            [
                "Structure",
                "Motif",
                "Param Type",
                "Param",
                "Value",
                "Vary",
                "Distribution",
                "Distribution Params",
            ]
        )
        self.table.cellClicked.connect(self._on_row_selected)
        self.table.cellChanged.connect(self._on_table_changed)
        left_layout.addWidget(self.table)

        button_row = QHBoxLayout()
        self.set_all_vary_on_button = QPushButton("Set All Vary On")
        self.set_all_vary_on_button.clicked.connect(
            lambda: self._set_all_vary(True)
        )
        button_row.addWidget(self.set_all_vary_on_button)
        self.set_all_vary_off_button = QPushButton("Set All Vary Off")
        self.set_all_vary_off_button.clicked.connect(
            lambda: self._set_all_vary(False)
        )
        button_row.addWidget(self.set_all_vary_off_button)
        save_button = QPushButton("Save Parameter Map")
        save_button.clicked.connect(self._emit_saved)
        button_row.addWidget(save_button)
        button_row.addStretch(1)
        left_layout.addLayout(button_row)

        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setMinimumHeight(160)
        left_layout.addWidget(self.console)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        self.figure = Figure(figsize=(6, 5))
        self.canvas = FigureCanvasQTAgg(self.figure)
        right_layout.addWidget(self.canvas)

        root.addWidget(left, stretch=3)
        root.addWidget(right, stretch=2)
        self.setCentralWidget(central)

    def load_entries(
        self,
        entries: list[DreamParameterEntry],
        *,
        has_existing_parameter_map: bool | None = None,
    ) -> None:
        if has_existing_parameter_map is not None:
            self._has_existing_parameter_map = bool(has_existing_parameter_map)
            self._was_saved = False
        self.table.blockSignals(True)
        self.table.setRowCount(len(entries))
        for row, entry in enumerate(entries):
            params = self._normalize_distribution_params(
                entry.distribution,
                entry.dist_params,
                entry.value,
            )
            self.table.setItem(row, 0, QTableWidgetItem(entry.structure))
            self.table.setItem(row, 1, QTableWidgetItem(entry.motif))
            self.table.setItem(row, 2, QTableWidgetItem(entry.param_type))
            self.table.setItem(row, 3, QTableWidgetItem(entry.param))
            self.table.setItem(
                row,
                4,
                QTableWidgetItem(f"{entry.value:.6g}"),
            )
            vary_box = QCheckBox()
            vary_box.setChecked(entry.vary)
            self.table.setCellWidget(row, 5, vary_box)
            combo = QComboBox()
            combo.addItems(list(BASE_DISTRIBUTIONS))
            combo.setCurrentText(entry.distribution)
            combo.currentTextChanged.connect(
                lambda _text, selected_row=row: self._on_distribution_changed(
                    selected_row
                )
            )
            self.table.setCellWidget(row, 6, combo)
            self.table.setItem(
                row,
                7,
                QTableWidgetItem(json.dumps(params, sort_keys=True)),
            )
        self.table.blockSignals(False)
        self.table.resizeColumnsToContents()
        self._entries = entries
        if entries:
            self._plot_entry(entries[0])

    def current_entries(self) -> list[DreamParameterEntry]:
        entries: list[DreamParameterEntry] = []
        for row in range(self.table.rowCount()):
            distribution = self.table.cellWidget(row, 6).currentText()
            value = float(self.table.item(row, 4).text())
            params = self._normalize_distribution_params(
                distribution,
                self._parse_params(self.table.item(row, 7).text()),
                value,
            )
            entries.append(
                DreamParameterEntry(
                    structure=self.table.item(row, 0).text(),
                    motif=self.table.item(row, 1).text(),
                    param_type=self.table.item(row, 2).text(),
                    param=self.table.item(row, 3).text(),
                    value=value,
                    vary=self.table.cellWidget(row, 5).isChecked(),
                    distribution=distribution,
                    dist_params=params,
                )
            )
        return entries

    def _emit_saved(self) -> None:
        entries = self.current_entries()
        self._was_saved = True
        self._has_existing_parameter_map = True
        self.saved.emit(entries)
        self.console.append("Saved current DREAM parameter map.")
        QMessageBox.information(
            self,
            "Saved",
            "The DREAM parameter map was updated.",
        )

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if self._has_existing_parameter_map or self._was_saved:
            super().closeEvent(event)
            return
        response = QMessageBox.question(
            self,
            "Quit without saving parameter map?",
            (
                "No DREAM parameter map has been saved for this project yet. "
                "Are you sure you want to quit Edit Priors without saving "
                "the parameter map?"
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if response == QMessageBox.StandardButton.Yes:
            super().closeEvent(event)
            return
        event.ignore()

    def _on_row_selected(self, row: int, _column: int) -> None:
        self._plot_entry(self.current_entries()[row])

    def _on_distribution_changed(self, row: int) -> None:
        distribution = self.table.cellWidget(row, 6).currentText()
        value = float(self.table.item(row, 4).text())
        params = self._normalize_distribution_params(
            distribution,
            self._parse_params(self.table.item(row, 7).text()),
            value,
        )
        self.table.blockSignals(True)
        self.table.item(row, 7).setText(json.dumps(params, sort_keys=True))
        self.table.blockSignals(False)
        entry = self.current_entries()[row]
        self.console.append(
            f"Distribution for {entry.param} set to {entry.distribution}."
        )
        self._plot_entry(entry)

    def _on_table_changed(self, row: int, column: int) -> None:
        if column == 7:
            try:
                entry = self.current_entries()[row]
            except Exception as exc:
                self.console.append(
                    f"Invalid distribution parameter JSON: {exc}"
                )
                return
            self._plot_entry(entry)

    def _set_all_vary(self, enabled: bool) -> None:
        for row in range(self.table.rowCount()):
            vary_box = self.table.cellWidget(row, 5)
            if isinstance(vary_box, QCheckBox):
                vary_box.setChecked(enabled)
        self.console.append(
            "Set all DREAM vary flags " + ("on." if enabled else "off.")
        )

    def _show_weight_prior_preview(self) -> None:
        weight_entries = [
            entry
            for entry in self.current_entries()
            if re.fullmatch(r"w\d+", entry.param.strip())
        ]
        if not weight_entries:
            QMessageBox.information(
                self,
                "No weight priors available",
                "No w<##> prior distributions are currently available in the table.",
            )
            return
        if self._weight_preview_window is None:
            self._weight_preview_window = WeightDistributionPreviewWindow()
        self._weight_preview_window.plot_entries(weight_entries)
        self._weight_preview_window.show()
        self._weight_preview_window.raise_()
        self._weight_preview_window.activateWindow()
        self.console.append(
            "Opened shared preview for all current w<##> prior distributions."
        )

    def _plot_entry(self, entry: DreamParameterEntry) -> None:
        self.figure.clear()
        axis = self.figure.add_subplot(111)
        try:
            distribution = getattr(stats, entry.distribution)
            x_min, x_max = _distribution_domain(entry)
            x_values = np.linspace(x_min, x_max, 200)
            y_values = distribution.pdf(x_values, **entry.dist_params)
            axis.plot(x_values, y_values, color="black")
            axis.axvline(entry.value, color="tab:red", linestyle="--")
            axis.set_title(f"{entry.param}: {entry.distribution}")
            axis.set_xlabel("Value")
            axis.set_ylabel("Density")
        except Exception as exc:
            axis.text(
                0.5,
                0.5,
                f"Unable to plot distribution:\n{exc}",
                ha="center",
                va="center",
            )
            axis.set_axis_off()
        self.figure.tight_layout()
        self.canvas.draw()

    @staticmethod
    def _parse_params(text: str) -> dict[str, float]:
        payload = json.loads(text or "{}")
        return {str(key): float(value) for key, value in payload.items()}

    @staticmethod
    def _normalize_distribution_params(
        distribution: str,
        raw_params: dict[str, float],
        value: float,
    ) -> dict[str, float]:
        if distribution not in BASE_DISTRIBUTIONS:
            return {
                str(key): float(param_value)
                for key, param_value in raw_params.items()
            }
        params = _distribution_defaults_for_value(distribution, value)
        for key in list(params):
            if key in raw_params:
                params[key] = float(raw_params[key])
        return params


def _distribution_domain(entry: DreamParameterEntry) -> tuple[float, float]:
    distribution = getattr(stats, entry.distribution)
    x_min = distribution.ppf(0.001, **entry.dist_params)
    x_max = distribution.ppf(0.999, **entry.dist_params)
    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_min == x_max:
        return entry.value - 1.0, entry.value + 1.0
    return float(x_min), float(x_max)


def _distribution_defaults_for_value(
    distribution: str,
    value: float,
) -> dict[str, float]:
    eps = 1e-6
    cv_default = 1 / math.e
    if distribution == "lognorm":
        scale_val = value if value > 0 else eps
        return {
            "loc": 0.0,
            "scale": scale_val,
            "s": math.sqrt(math.log(1 + cv_default**2)),
        }
    if distribution == "norm":
        return {
            "loc": value,
            "scale": value / math.e if value > 0 else eps,
        }
    if distribution == "uniform":
        if value > 0:
            half_range = value * cv_default
            return {
                "loc": max(value - half_range, 0.0),
                "scale": 2 * half_range,
            }
        return {"loc": 0.0, "scale": eps}
    return dict(BASE_DISTRIBUTIONS[distribution])
