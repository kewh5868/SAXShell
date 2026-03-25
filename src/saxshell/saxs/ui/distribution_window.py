from __future__ import annotations

import copy
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
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
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

SMART_PRIOR_PRESET_ITEMS: tuple[tuple[str, str], ...] = (
    ("Very Strict", "very_strict"),
    ("Strict", "strict"),
    ("Proportional (Current Default)", "proportional"),
    ("Lenient", "lenient"),
    ("Very Lenient", "very_lenient"),
    ("Strict Small / Lenient Large", "strict_small_lenient_large"),
    ("Lenient Small / Strict Large", "lenient_small_strict_large"),
)
SMART_PRIOR_INDIVIDUAL_STATUS_ITEMS: tuple[tuple[str, str], ...] = (
    ("Custom / Manual", "custom"),
    ("Very Strict", "very_strict"),
    ("Strict", "strict"),
    ("Proportional", "proportional"),
    ("Lenient", "lenient"),
    ("Very Lenient", "very_lenient"),
)
SMART_PRIOR_APPLY_SCOPE_ITEMS: tuple[tuple[str, str], ...] = (
    ("All Structures", "all"),
    ("Selected Structures", "selected"),
)
SMART_PRIOR_SPREAD_FACTORS: dict[str, float] = {
    "very_strict": 0.4,
    "strict": 0.65,
    "proportional": 1.0,
    "lenient": 1.5,
    "very_lenient": 2.25,
}
SMART_PRIOR_STATUS_LABELS: dict[str, str] = {
    value: label for label, value in SMART_PRIOR_INDIVIDUAL_STATUS_ITEMS
}


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
        self._suppress_vary_warning = False
        self._suppress_status_change = False
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
        top_row.addWidget(QLabel("Smart prior preset"))
        self.smart_prior_preset_combo = QComboBox()
        for label, value in SMART_PRIOR_PRESET_ITEMS:
            self.smart_prior_preset_combo.addItem(label, userData=value)
        self.smart_prior_preset_combo.setToolTip(
            "Apply a preset tightening or relaxation pattern to the current "
            "prior distributions. The current table settings act as the "
            "baseline."
        )
        top_row.addWidget(self.smart_prior_preset_combo)
        top_row.addWidget(QLabel("Apply to"))
        self.smart_prior_apply_scope_combo = QComboBox()
        for label, value in SMART_PRIOR_APPLY_SCOPE_ITEMS:
            self.smart_prior_apply_scope_combo.addItem(label, userData=value)
        self.smart_prior_apply_scope_combo.setToolTip(
            "Choose whether the selected preset should affect every "
            "structure in the table or only the currently selected "
            "structure rows. Size-aware mixed presets always apply to all "
            "structures so their relative ranking remains meaningful."
        )
        top_row.addWidget(self.smart_prior_apply_scope_combo)
        self.apply_smart_prior_preset_button = QPushButton(
            "Apply Smart Preset"
        )
        self.apply_smart_prior_preset_button.setToolTip(
            "Adjust the current prior widths using the selected preset. "
            "Size-aware presets use effective-radius parameters to identify "
            "relatively small and large cluster weights."
        )
        self.apply_smart_prior_preset_button.clicked.connect(
            self._apply_smart_prior_preset
        )
        top_row.addWidget(self.apply_smart_prior_preset_button)
        top_row.addStretch(1)
        left_layout.addLayout(top_row)
        self.table = QTableWidget(0, 9)
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
                "Smart Preset Status",
            ]
        )
        self.table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.table.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.table.cellClicked.connect(self._on_row_selected)
        self.table.cellChanged.connect(self._on_table_changed)
        left_layout.addWidget(self.table)

        button_row = QHBoxLayout()
        self.select_recommended_vary_button = QPushButton(
            "Select Weights + Model Params"
        )
        self.select_recommended_vary_button.setToolTip(
            "Enable vary for weights and other model parameters, but keep "
            "effective-radius parameters turned off."
        )
        self.select_recommended_vary_button.clicked.connect(
            self._set_recommended_vary_selection
        )
        button_row.addWidget(self.select_recommended_vary_button)
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
            vary_box.toggled.connect(
                lambda checked, selected_row=row: self._on_vary_toggled(
                    selected_row,
                    checked,
                )
            )
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
            status_combo = QComboBox()
            for label, value in SMART_PRIOR_INDIVIDUAL_STATUS_ITEMS:
                status_combo.addItem(label, userData=value)
            status_value = self._normalized_smart_status(
                getattr(entry, "smart_preset_status", "custom")
            )
            status_index = status_combo.findData(status_value)
            if status_index < 0:
                status_index = status_combo.findData("custom")
            self._suppress_status_change = True
            try:
                status_combo.setCurrentIndex(max(status_index, 0))
            finally:
                self._suppress_status_change = False
            status_combo.currentIndexChanged.connect(
                lambda _index, selected_row=row, combo=status_combo: (
                    self._on_smart_status_changed(selected_row, combo)
                )
            )
            self.table.setCellWidget(row, 8, status_combo)
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
                    smart_preset_status=self._row_smart_status(row),
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
        self._set_group_status_for_row(row, "custom")
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
            self._set_group_status_for_row(row, "custom")
            self._plot_entry(entry)
            return
        if column == 4:
            self._set_group_status_for_row(row, "custom")

    def _set_all_vary(self, enabled: bool) -> None:
        self._set_vary_state_for_rows(lambda _row_index, _param_name: enabled)
        self.console.append(
            "Set all DREAM vary flags " + ("on." if enabled else "off.")
        )

    def _apply_smart_prior_preset(self) -> None:
        preset_mode = (
            str(
                self.smart_prior_preset_combo.currentData() or "proportional"
            ).strip()
            or "proportional"
        )
        apply_scope = (
            str(
                self.smart_prior_apply_scope_combo.currentData() or "all"
            ).strip()
            or "all"
        )
        effective_scope = (
            "all"
            if preset_mode
            in {
                "strict_small_lenient_large",
                "lenient_small_strict_large",
            }
            else apply_scope
        )
        entries = self.current_entries()
        try:
            updated_entries = self._smart_adjusted_entries(
                entries,
                preset_mode=preset_mode,
                apply_scope=effective_scope,
                selected_rows=self._selected_row_indexes(),
            )
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Unable to apply smart prior preset",
                str(exc),
            )
            self.console.append(
                "Unable to apply smart prior preset: " + str(exc)
            )
            return
        self.load_entries(
            updated_entries,
            has_existing_parameter_map=self._has_existing_parameter_map,
        )
        preset_label = str(
            self.smart_prior_preset_combo.currentText() or preset_mode
        ).strip()
        scope_label = str(
            self.smart_prior_apply_scope_combo.currentText() or apply_scope
        ).strip()
        if preset_mode in {
            "strict_small_lenient_large",
            "lenient_small_strict_large",
        }:
            scope_label = "All Structures"
        self.console.append(
            f"Applied smart prior preset: {preset_label} ({scope_label})."
        )

    def _set_recommended_vary_selection(self) -> None:
        self._set_vary_state_for_rows(
            lambda _row_index, param_name: (
                not self._is_effective_radius_parameter(param_name)
            )
        )
        self.console.append(
            "Enabled vary for weights and model parameters while keeping "
            "effective-radius parameters off."
        )

    def _set_vary_state_for_rows(
        self,
        selector,
    ) -> None:
        self._suppress_vary_warning = True
        try:
            for row in range(self.table.rowCount()):
                vary_box = self.table.cellWidget(row, 5)
                if not isinstance(vary_box, QCheckBox):
                    continue
                param_name = self.table.item(row, 3).text().strip()
                vary_box.setChecked(bool(selector(row, param_name)))
        finally:
            self._suppress_vary_warning = False

    def _on_vary_toggled(self, row: int, checked: bool) -> None:
        if self._suppress_vary_warning or not checked:
            return
        if row < 0 or row >= self.table.rowCount():
            return
        param_name = self.table.item(row, 3).text().strip()
        if not self._is_effective_radius_parameter(param_name):
            return
        message = (
            "It is not recommended to vary effective-radius parameters in "
            "the DREAM prior map.\n\n"
            f"Selected parameter: {param_name}"
        )
        self.console.append(
            "Warning: effective-radius parameters are not recommended for "
            f"DREAM variation ({param_name})."
        )
        QMessageBox.warning(
            self,
            "Effective radius variation warning",
            message,
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

    def _smart_adjusted_entries(
        self,
        entries: list[DreamParameterEntry],
        *,
        preset_mode: str,
        apply_scope: str = "all",
        selected_rows: list[int] | None = None,
    ) -> list[DreamParameterEntry]:
        updated_entries = [
            DreamParameterEntry.from_dict(entry.to_dict()) for entry in entries
        ]
        row_groups = self._row_group_keys(updated_entries)
        target_groups = self._target_group_keys(
            updated_entries,
            apply_scope=apply_scope,
            selected_rows=list(selected_rows or []),
        )
        if preset_mode in SMART_PRIOR_SPREAD_FACTORS:
            factor = SMART_PRIOR_SPREAD_FACTORS[preset_mode]
            for row_index, entry in enumerate(updated_entries):
                if row_groups[row_index] not in target_groups:
                    continue
                entry.dist_params = self._adjust_distribution_params(
                    entry,
                    factor=factor,
                )
                entry.smart_preset_status = preset_mode
            return updated_entries

        weight_radius_map = self._weight_radius_map(updated_entries)
        if len(weight_radius_map) < 2:
            raise ValueError(
                "Size-aware smart priors require at least two effective-"
                "radius values tied to weight parameters."
            )
        size_groups = self._weight_size_groups(weight_radius_map)
        if preset_mode == "strict_small_lenient_large":
            factor_by_group = {
                "small": SMART_PRIOR_SPREAD_FACTORS["strict"],
                "large": SMART_PRIOR_SPREAD_FACTORS["lenient"],
                "neutral": SMART_PRIOR_SPREAD_FACTORS["proportional"],
            }
        elif preset_mode == "lenient_small_strict_large":
            factor_by_group = {
                "small": SMART_PRIOR_SPREAD_FACTORS["lenient"],
                "large": SMART_PRIOR_SPREAD_FACTORS["strict"],
                "neutral": SMART_PRIOR_SPREAD_FACTORS["proportional"],
            }
        else:
            raise ValueError(f"Unknown smart prior preset: {preset_mode}")

        weight_by_group = self._group_weight_params(updated_entries)
        if preset_mode == "strict_small_lenient_large":
            status_by_group = {
                "small": "strict",
                "large": "lenient",
                "neutral": "proportional",
            }
        else:
            status_by_group = {
                "small": "lenient",
                "large": "strict",
                "neutral": "proportional",
            }

        for row_index, entry in enumerate(updated_entries):
            group_key = row_groups[row_index]
            weight_param = weight_by_group.get(group_key)
            if weight_param is None:
                size_group = "neutral"
                factor = SMART_PRIOR_SPREAD_FACTORS["proportional"]
            else:
                size_group = size_groups.get(weight_param, "neutral")
                factor = factor_by_group.get(
                    size_group,
                    SMART_PRIOR_SPREAD_FACTORS["proportional"],
                )
            entry.dist_params = self._adjust_distribution_params(
                entry,
                factor=factor,
            )
            entry.smart_preset_status = status_by_group.get(
                size_group,
                "proportional",
            )
        return updated_entries

    def _on_smart_status_changed(
        self,
        row: int,
        combo: QComboBox,
    ) -> None:
        if self._suppress_status_change:
            return
        preset_mode = self._normalized_smart_status(combo.currentData())
        if preset_mode == "custom":
            self._set_group_status_for_row(row, "custom")
            self.console.append(
                "Marked smart prior status as Custom / Manual for "
                f"{self._row_status_label(row)}."
            )
            return
        entries = self.current_entries()
        try:
            updated_entries = self._smart_adjusted_entries(
                entries,
                preset_mode=preset_mode,
                apply_scope="selected",
                selected_rows=[row],
            )
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Unable to apply row smart prior preset",
                str(exc),
            )
            self.console.append(
                "Unable to apply row smart prior preset: " + str(exc)
            )
            self._set_group_status_for_row(row, self._row_smart_status(row))
            return
        self.load_entries(
            updated_entries,
            has_existing_parameter_map=self._has_existing_parameter_map,
        )
        self.console.append(
            "Applied row smart prior preset: "
            f"{SMART_PRIOR_STATUS_LABELS.get(preset_mode, preset_mode)} "
            f"for {self._row_status_label(row)}."
        )

    def _selected_row_indexes(self) -> list[int]:
        return sorted({index.row() for index in self.table.selectedIndexes()})

    def _row_group_keys(
        self,
        entries: list[DreamParameterEntry],
    ) -> list[tuple[str, str, str]]:
        return [
            self._entry_group_key(entry, row_index)
            for row_index, entry in enumerate(entries)
        ]

    def _target_group_keys(
        self,
        entries: list[DreamParameterEntry],
        *,
        apply_scope: str,
        selected_rows: list[int],
    ) -> set[tuple[str, str, str]]:
        row_groups = self._row_group_keys(entries)
        if apply_scope == "all":
            return set(row_groups)
        if apply_scope != "selected":
            raise ValueError(f"Unknown smart prior apply scope: {apply_scope}")
        if not selected_rows:
            raise ValueError(
                "Select one or more structure rows before applying a smart "
                "prior preset to selected structures."
            )
        return {
            row_groups[row]
            for row in selected_rows
            if 0 <= row < len(row_groups)
        }

    def _group_weight_params(
        self,
        entries: list[DreamParameterEntry],
    ) -> dict[tuple[str, str, str], str]:
        mapping: dict[tuple[str, str, str], str] = {}
        for row_index, entry in enumerate(entries):
            weight_name = self._weight_param_name_for_entry(entry)
            if weight_name is None:
                continue
            mapping[self._entry_group_key(entry, row_index)] = weight_name
        return mapping

    @staticmethod
    def _entry_group_key(
        entry: DreamParameterEntry,
        row_index: int,
    ) -> tuple[str, str, str]:
        structure = str(entry.structure).strip()
        motif = str(entry.motif).strip()
        if structure or motif:
            return ("structure", structure, motif)
        return ("row", str(row_index), "")

    @staticmethod
    def _normalized_smart_status(value: object) -> str:
        text = str(value or "custom").strip() or "custom"
        valid = {
            status for _label, status in SMART_PRIOR_INDIVIDUAL_STATUS_ITEMS
        }
        return text if text in valid else "custom"

    def _row_smart_status(self, row: int) -> str:
        combo = self.table.cellWidget(row, 8)
        if isinstance(combo, QComboBox):
            return self._normalized_smart_status(combo.currentData())
        return "custom"

    def _set_group_status_for_row(
        self,
        row: int,
        status: str,
    ) -> None:
        entries = self.current_entries()
        if row < 0 or row >= len(entries):
            return
        target_key = self._entry_group_key(entries[row], row)
        normalized_status = self._normalized_smart_status(status)
        self._suppress_status_change = True
        try:
            for row_index, entry in enumerate(entries):
                if self._entry_group_key(entry, row_index) != target_key:
                    continue
                combo = self.table.cellWidget(row_index, 8)
                if not isinstance(combo, QComboBox):
                    continue
                combo_index = combo.findData(normalized_status)
                if combo_index >= 0:
                    combo.setCurrentIndex(combo_index)
        finally:
            self._suppress_status_change = False

    def _row_status_label(self, row: int) -> str:
        structure_item = self.table.item(row, 0)
        motif_item = self.table.item(row, 1)
        param_item = self.table.item(row, 3)
        structure = (
            structure_item.text().strip() if structure_item is not None else ""
        )
        motif = motif_item.text().strip() if motif_item is not None else ""
        param = param_item.text().strip() if param_item is not None else ""
        if structure or motif:
            if motif:
                return f"{structure or 'No structure'} / {motif}"
            return structure
        return param or f"row {row + 1}"

    @staticmethod
    def _adjust_distribution_params(
        entry: DreamParameterEntry,
        *,
        factor: float,
    ) -> dict[str, float]:
        params = copy.deepcopy(dict(entry.dist_params))
        bounded_factor = max(float(factor), 1e-6)
        epsilon = 1e-9
        if entry.distribution == "lognorm":
            params["s"] = max(
                float(params.get("s", epsilon)) * bounded_factor, epsilon
            )
            return params
        if entry.distribution == "norm":
            params["scale"] = max(
                float(params.get("scale", epsilon)) * bounded_factor,
                epsilon,
            )
            return params
        if entry.distribution == "uniform":
            current_scale = max(float(params.get("scale", epsilon)), epsilon)
            center = float(params.get("loc", 0.0)) + current_scale / 2.0
            updated_scale = max(current_scale * bounded_factor, epsilon)
            params["scale"] = updated_scale
            params["loc"] = center - updated_scale / 2.0
            return params
        return params

    @classmethod
    def _weight_radius_map(
        cls,
        entries: list[DreamParameterEntry],
    ) -> dict[str, float]:
        sphere_map: dict[str, float] = {}
        ellipsoid_axes: dict[str, dict[str, float]] = {}
        for entry in entries:
            param_name = str(entry.param).strip()
            sphere_match = re.fullmatch(r"r_eff_(w\d+)", param_name)
            if sphere_match:
                sphere_map[sphere_match.group(1)] = max(
                    float(entry.value), 0.0
                )
                continue
            ellipsoid_match = re.fullmatch(r"([abc])_eff_(w\d+)", param_name)
            if ellipsoid_match:
                axis_name = ellipsoid_match.group(1)
                weight_name = ellipsoid_match.group(2)
                ellipsoid_axes.setdefault(weight_name, {})[axis_name] = max(
                    float(entry.value),
                    0.0,
                )
        for weight_name, axes in ellipsoid_axes.items():
            if {"a", "b", "c"} <= set(axes):
                sphere_map[weight_name] = float(
                    np.cbrt(axes["a"] * axes["b"] * axes["c"])
                )
        return sphere_map

    @staticmethod
    def _weight_size_groups(
        weight_radius_map: dict[str, float],
    ) -> dict[str, str]:
        radii = np.asarray(list(weight_radius_map.values()), dtype=float)
        median_radius = float(np.median(radii))
        tolerance = max(median_radius * 1e-9, 1e-9)
        groups: dict[str, str] = {}
        for weight_name, radius in weight_radius_map.items():
            if radius < median_radius - tolerance:
                groups[weight_name] = "small"
            elif radius > median_radius + tolerance:
                groups[weight_name] = "large"
            else:
                groups[weight_name] = "neutral"
        return groups

    @staticmethod
    def _weight_param_name_for_entry(
        entry: DreamParameterEntry,
    ) -> str | None:
        param_name = str(entry.param).strip()
        if re.fullmatch(r"w\d+", param_name):
            return param_name
        return None

    @staticmethod
    def _is_effective_radius_parameter(param_name: str) -> bool:
        name = str(param_name).strip()
        return bool(
            name == "eff_r"
            or name.startswith("r_eff_")
            or name.startswith("a_eff_")
            or name.startswith("b_eff_")
            or name.startswith("c_eff_")
        )


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
