from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import datetime
from html import escape as html_escape
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg,
    NavigationToolbar2QT,
)
from matplotlib.figure import Figure
from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import (
    QAction,
    QBrush,
    QColor,
    QFont,
    QKeySequence,
    QPainter,
    QPen,
    QShortcut,
    QTextDocument,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QSpinBox,
    QSplitter,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QTableWidget,
    QTableWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from saxshell.saxs.ui.branding import (
    configure_saxshell_application,
    prepare_saxshell_application_identity,
)
from saxshell.uvvis_fitting.model import (
    DEFAULT_MONTE_CARLO_SEED,
    PEAK_COLORS,
    FitResult,
    MonteCarloFitRecord,
    MonteCarloResult,
    MonteCarloSettings,
    PeakComponent,
    SweepFitRecord,
    UVVisDataset,
    component_parameter_value,
    evaluate_components,
    evaluate_fit_result,
    fit_components,
    is_relation_constraint,
    load_session_payload,
    load_uvvis_file,
    next_peak_label,
    parse_interval_constraint,
    parse_sweep_range,
    relation_constraint_value,
)
from saxshell.uvvis_fitting.model import (
    run_monte_carlo_fit as run_monte_carlo_fit_model,
)
from saxshell.uvvis_fitting.model import (
    run_parameter_sweep,
    save_fit_bundle,
    session_payload,
    set_component_parameter,
    write_uvvis_fit_csv,
)

PEAK_REFERENCE_PATTERN = re.compile(r"@([A-Za-z][A-Za-z0-9_]*)")
VALID_REF_COLOR = "#0072B2"
INVALID_REF_COLOR = "#C3263A"
SECTION_DIVIDER_COLOR = "#202020"


class ParameterGroupHeader(QHeaderView):
    """Header that paints strong vertical dividers between parameter
    groups."""

    def __init__(
        self,
        orientation: Qt.Orientation,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(orientation, parent)
        self._section_dividers: set[int] = set()

    def set_section_dividers(self, columns: set[int]) -> None:
        self._section_dividers = set(columns)
        self.viewport().update()

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        painter = QPainter(self.viewport())
        pen = QPen(QColor(SECTION_DIVIDER_COLOR))
        pen.setWidth(3)
        painter.setPen(pen)
        for col in sorted(self._section_dividers):
            if self.isSectionHidden(col):
                continue
            x_pos = (
                self.sectionViewportPosition(col) + self.sectionSize(col) - 1
            )
            if -2 <= x_pos <= self.viewport().width() + 2:
                painter.drawLine(x_pos, 0, x_pos, self.viewport().height())


class ParameterGroupTableWidget(QTableWidget):
    """Table that paints vertical dividers through parameter
    sections."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._section_dividers: set[int] = set()

    def set_section_dividers(self, columns: set[int]) -> None:
        self._section_dividers = set(columns)
        header = self.horizontalHeader()
        if isinstance(header, ParameterGroupHeader):
            header.set_section_dividers(self._section_dividers)
        self.viewport().update()

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        painter = QPainter(self.viewport())
        pen = QPen(QColor(SECTION_DIVIDER_COLOR))
        pen.setWidth(3)
        painter.setPen(pen)
        for col in sorted(self._section_dividers):
            if self.isColumnHidden(col):
                continue
            x_pos = (
                self.columnViewportPosition(col) + self.columnWidth(col) - 1
            )
            if -2 <= x_pos <= self.viewport().width() + 2:
                painter.drawLine(x_pos, 0, x_pos, self.viewport().height())


class ConstraintCellDelegate(QStyledItemDelegate):
    """Render constraint cells with @<peak> tokens colored by
    validity."""

    def __init__(self, window: "UVVisFitMainWindow") -> None:
        super().__init__(window)
        self._window = window

    def paint(self, painter, option, index) -> None:
        text = str(index.data(Qt.ItemDataRole.DisplayRole) or "")
        if "@" not in text:
            super().paint(painter, option, index)
            return
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        widget = opt.widget
        style = widget.style() if widget else QApplication.style()
        opt.text = ""
        style.drawControl(
            QStyle.ControlElement.CE_ItemViewItem, opt, painter, widget
        )

        labels = {component.label for component in self._window.components}

        def _replace(match: re.Match[str]) -> str:
            label = match.group(1)
            color = VALID_REF_COLOR if label in labels else INVALID_REF_COLOR
            return (
                f'<span style="color:{color};font-weight:600">'
                f"@{html_escape(label)}</span>"
            )

        html = PEAK_REFERENCE_PATTERN.sub(_replace, html_escape(text))
        doc = QTextDocument()
        doc.setDocumentMargin(0)
        doc.setHtml(f"<div>{html}</div>")
        rect = style.subElementRect(
            QStyle.SubElement.SE_ItemViewItemText,
            opt,
            widget,
        )
        painter.save()
        painter.translate(rect.topLeft())
        doc.setTextWidth(rect.width())
        doc.drawContents(painter)
        painter.restore()


DEFAULT_UVVIS_DIR = Path(
    "/Users/keithwhite/repos/pvskspecies_figures/data/processed_data/"
    "uvvis/maintext/figure_01/uvvis_molar_absorptivity"
)
DEFAULT_SAVE_DIR = DEFAULT_UVVIS_DIR / "saxsshell_uvvis_fits"
SETTINGS_ORG = "saxshell"
SETTINGS_APP = "uvvisfit"
SETTINGS_KEY_LAST_SAVE_DIR = "last_save_dir"
_OPEN_WINDOWS: list[QMainWindow] = []


HELP_TEXT = """\
<h3>Workflow</h3>
<ol>
<li><b>Load Data</b> — open a UV-Vis text file (.txt / .csv / .dat).</li>
<li><b>Set the fit range</b> in the <i>Fit min</i> / <i>Fit max</i> boxes
(top right). The shaded vertical band on the plot shows the active range.
Data outside the band is still visible so peaks can be grabbed.</li>
<li><b>Add Peak</b> — the cursor becomes a crosshair; click on the plot to
place a new peak. Center = click X, amplitude = click Y, FWHM defaults to
about 5% of the data X span. Press <b>Esc</b> to cancel placement.</li>
<li>Drag the top square handle to adjust amplitude+center; drag the
triangular side handles to adjust FWHM. Locked parameters do not move.</li>
<li><b>Fit</b> refines all unlocked parameters using lmfit
(least-squares).</li>
</ol>

<h3>Constraint shorthand (per-parameter cells)</h3>
<table cellspacing="6">
<tr><td><b>blank</b></td>
<td>parameter is unconstrained (only its sign / physical limit applies).</td></tr>
<tr><td><b>[min, max]</b></td>
<td>bounded interval, e.g. <code>[200, 300]</code> on a center cell.</td></tr>
<tr><td><b>@A</b></td>
<td>tie this parameter to peak A's same parameter (e.g. set on B.fwhm to
force B.fwhm == A.fwhm).</td></tr>
<tr><td><b>@A + 50</b></td>
<td>tie with an offset (e.g. center B = center A + 50 nm). Operators
supported: <code>+ - * /</code>.</td></tr>
</table>
<p>When you enter an <code>@&lt;peak&gt;</code> reference, the constrained
parameter <b>snaps immediately</b> to the referenced peak's value. The
<code>@&lt;peak&gt;</code> token appears <span style="color:#0072B2">
<b>blue</b></span> when the reference is valid and
<span style="color:#C3263A"><b>red</b></span> when the peak does not exist.</p>
<p><b>Area constraints</b> apply as fit-time penalties because area is a
derived quantity (function of amplitude, FWHM, eta).</p>

<h3>Pseudo-Voigt parameters</h3>
<ul>
<li><b>Amplitude</b> — peak height at the center.</li>
<li><b>Area</b> — integrated intensity (derived from amplitude, FWHM, η).</li>
<li><b>Center</b> — peak location on the X axis.</li>
<li><b>FWHM</b> — full width at half maximum.</li>
<li><b>η (Lorentz fraction, 0–1)</b> — Gauss–Lorentz mixing ratio.
<code>η = 0</code> is pure Gaussian, <code>η = 1</code> is pure Lorentzian.
Pseudo-Voigt = <code>(1 − η)·Gaussian + η·Lorentzian</code> (CasaXPS
convention).</li>
</ul>

<h3>Locking</h3>
<p>Each lock checkbox freezes that parameter at its current value during
the fit. Drag handles also respect locks. An <code>@</code> tie acts as
an effective lock during fitting because the parameter follows the
referenced peak.</p>

<h3>Range Fit</h3>
<p>Double-click any numeric value cell (Amp, Area, Center, FWHM, Eta)
or select a value cell and press <b>Range Fit</b> to sweep that parameter
across <code>[min, max, steps]</code>. Each step writes a folder
containing <code>fit.json</code>, <code>peaks.csv</code>, and
<code>curve.csv</code>. A chi-squared vs. parameter chart appears; select
a row and <b>Apply Selected</b> to load that fit.</p>

<h3>Monte Carlo</h3>
<p><b>Monte Carlo</b> first obtains a reference fit, then creates repeated
synthetic spectra by adding random noise to that fitted envelope, refits each
synthetic spectrum using the same locks, bounds, and peak ties, and reports
per-peak error bars from the retained best-fit population. The default noise
estimate comes from the RMS residual; use a manual sigma when instrument noise
is known. Each run writes a Markdown fit report plus CSV exports for the
ranked population, parameter error bars, and trace/error bands.</p>

<h3>Fit metrics</h3>
<p>After a fit, the bottom-right of the main plot displays:
<code>R²</code>, total <code>χ²</code>, reduced <code>χ²</code>,
<code>RMS</code> of the residual, and <code>max |r|</code> (largest
absolute residual). The residual subplot below the main plot shows the
residual within the fit range. Its y-axis is fixed (no auto-rescale on
each refresh); click <b>Rescale Residual Y</b> to re-fit it to the
current residual.</p>

<h3>Saving</h3>
<p><b>Save Fit</b> opens the save dialog seeded at the current session
path (overwrites with confirmation). <b>Save As</b> seeds the dialog
with a fresh timestamped filename so the previous fit is preserved.
Both write a JSON session plus a companion CSV (header carries fit
metadata + per-peak parameters, body is x/data/total/residual/peak_*).
<b>Export Current Fit Folder</b> writes <code>fit.json</code> +
<code>peaks.csv</code> + <code>curve.csv</code> into a directory.</p>

<h3>Importing peaks</h3>
<p><b>File > Import Peaks From Fit...</b> loads peaks (positions, FWHM,
η, constraints, locks) from a previously saved <code>.json</code> while
keeping the currently loaded dataset and fit range. The previous peak
list is wiped — use <b>Undo</b> to restore it.</p>
"""


TABLE_COLUMNS: list[tuple[str, str | None, str]] = [
    ("Peak", None, "label"),
    ("Amp", "amplitude", "value"),
    ("Lock", "amplitude", "lock"),
    ("Amp Constraint", "amplitude", "constraint"),
    ("Area", "area", "value"),
    ("Lock", "area", "lock"),
    ("Area Constraint", "area", "constraint"),
    ("Center", "center", "value"),
    ("Lock", "center", "lock"),
    ("Center Constraint", "center", "constraint"),
    ("FWHM", "fwhm", "value"),
    ("Lock", "fwhm", "lock"),
    ("FWHM Constraint", "fwhm", "constraint"),
    ("η (Lorentz frac)", "eta", "value"),
    ("Lock", "eta", "lock"),
    ("η Constraint", "eta", "constraint"),
]
PARAMETER_SECTION_BOUNDARY_COLUMNS = {
    index
    for index, (_title, _parameter, kind) in enumerate(TABLE_COLUMNS)
    if kind in {"label", "constraint"}
}
VALUE_COLUMNS = {
    index: parameter
    for index, (_title, parameter, kind) in enumerate(TABLE_COLUMNS)
    if parameter is not None and kind == "value"
}
LOCK_COLUMNS = {
    index: parameter
    for index, (_title, parameter, kind) in enumerate(TABLE_COLUMNS)
    if parameter is not None and kind == "lock"
}
CONSTRAINT_COLUMNS = {
    index: parameter
    for index, (_title, parameter, kind) in enumerate(TABLE_COLUMNS)
    if parameter is not None and kind == "constraint"
}


class ParameterRangeDialog(QDialog):
    def __init__(
        self,
        *,
        peak_label: str,
        parameter: str,
        current_value: float,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Range fit {peak_label} {parameter}")
        layout = QVBoxLayout(self)
        form = QFormLayout()
        span = max(abs(float(current_value)) * 0.2, 1.0)
        default_min = float(current_value) - span
        default_max = float(current_value) + span
        if parameter in {"amplitude", "area", "fwhm", "eta"}:
            default_min = max(default_min, 0.0)
        if parameter == "eta":
            default_max = min(default_max, 1.0)
        self.range_edit = QLineEdit(
            f"[{default_min:.6g}, {default_max:.6g}, 12]"
        )
        form.addRow("[min, max, steps]", self.range_edit)
        layout.addLayout(form)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def values(self) -> tuple[float, float, int]:
        return parse_sweep_range(self.range_edit.text())


class RangeSweepResultsDialog(QDialog):
    def __init__(
        self,
        *,
        peak_label: str,
        parameter: str,
        records: list[SweepFitRecord],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._records = records
        self._parameter = parameter
        self._peak_label = peak_label
        self._selected_record: SweepFitRecord | None = None
        self.setWindowTitle(f"Range fit results: {peak_label} {parameter}")
        self.resize(1200, 640)

        layout = QVBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Vertical)
        self.figure = Figure(figsize=(11.5, 3.6), constrained_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        splitter.addWidget(self.canvas)
        self.tree = QTreeWidget()
        self.tree.setColumnCount(3)
        self.tree.setHeaderLabels(["Value", "Chi-squared", "Output folder"])
        self.tree.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.tree.itemSelectionChanged.connect(self._on_selection_changed)
        for index, record in enumerate(records):
            item = QTreeWidgetItem(
                [
                    f"{record.value:.8g}",
                    f"{record.chisq:.6g}",
                    str(record.output_dir),
                ]
            )
            item.setData(0, Qt.ItemDataRole.UserRole, index)
            self.tree.addTopLevelItem(item)
        self.tree.header().setSectionResizeMode(
            0,
            QHeaderView.ResizeMode.ResizeToContents,
        )
        self.tree.header().setSectionResizeMode(
            1,
            QHeaderView.ResizeMode.ResizeToContents,
        )
        self.tree.header().setSectionResizeMode(
            2, QHeaderView.ResizeMode.Stretch
        )
        splitter.addWidget(self.tree)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter)

        button_row = QHBoxLayout()
        button_row.addStretch(1)
        self.apply_button = QPushButton("Apply Selected")
        self.apply_button.clicked.connect(self._apply_selected)
        button_row.addWidget(self.apply_button)
        keep_button = QPushButton("Keep Current")
        keep_button.clicked.connect(self.reject)
        button_row.addWidget(keep_button)
        layout.addLayout(button_row)
        self._build_axes()
        self._draw_chi_plot()
        self._draw_model_plot(None)

    @property
    def selected_record(self) -> SweepFitRecord | None:
        return self._selected_record

    def _build_axes(self) -> None:
        self.figure.clear()
        gs = self.figure.add_gridspec(1, 2, width_ratios=[3, 2], wspace=0.28)
        self.model_ax = self.figure.add_subplot(gs[0])
        self.chi_ax = self.figure.add_subplot(gs[1])

    def _draw_chi_plot(self) -> None:
        ax = self.chi_ax
        ax.clear()
        values = [record.value for record in self._records]
        chisq = [record.chisq for record in self._records]
        ax.plot(values, chisq, marker="o", color="#0072B2", linewidth=1.4)
        ax.set_xlabel(f"{self._peak_label} {self._parameter}")
        ax.set_ylabel("Chi-squared")
        ax.grid(True, alpha=0.25)
        self.canvas.draw_idle()

    def _draw_model_plot(self, record: SweepFitRecord | None) -> None:
        ax = self.model_ax
        ax.clear()
        ax.grid(True, alpha=0.18)
        if record is None:
            ax.text(
                0.5,
                0.5,
                "Select a sweep step to preview its model",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color="0.4",
            )
            ax.set_axis_off()
            self.canvas.draw_idle()
            return
        result = record.result
        ax.plot(
            result.x_fit,
            result.y_data,
            color="#1f1f1f",
            linewidth=1.2,
            label="Data",
            zorder=3,
        )
        for component, curve in zip(
            result.components,
            result.component_curves,
            strict=False,
        ):
            ax.plot(
                result.x_fit,
                curve,
                color=component.color,
                linewidth=1.2,
                label=f"Peak {component.label}",
                alpha=0.9,
            )
        ax.plot(
            result.x_fit,
            result.total,
            color="#C3263A",
            linewidth=1.6,
            label="Total",
            zorder=4,
        )
        ax.set_xlabel("Wavelength")
        ax.set_ylabel("Absorbance")
        ax.set_title(
            f"{self._peak_label} {self._parameter} = {record.value:.4g}",
            fontsize=10,
        )
        ax.legend(loc="best", fontsize=8)
        self.canvas.draw_idle()

    def _on_selection_changed(self) -> None:
        items = self.tree.selectedItems()
        if not items:
            self._selected_record = None
            self._draw_model_plot(None)
            return
        index = int(items[0].data(0, Qt.ItemDataRole.UserRole))
        self._selected_record = self._records[index]
        self._draw_model_plot(self._selected_record)

    def _apply_selected(self) -> None:
        if self._selected_record is None:
            QMessageBox.information(
                self,
                "Range Fit",
                "Select a generated fit before applying it.",
            )
            return
        self.accept()


class MonteCarloSettingsDialog(QDialog):
    def __init__(
        self,
        *,
        dataset: UVVisDataset,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Monte Carlo fitting")
        layout = QVBoxLayout(self)
        form = QFormLayout()
        x_span = max(dataset.x_max - dataset.x_min, 1.0)

        self.iterations_spin = QSpinBox()
        self.iterations_spin.setRange(1, 100000)
        self.iterations_spin.setValue(100)
        form.addRow("Fits", self.iterations_spin)

        self.keep_percent_spin = QDoubleSpinBox()
        self.keep_percent_spin.setRange(1.0, 100.0)
        self.keep_percent_spin.setDecimals(1)
        self.keep_percent_spin.setSuffix("%")
        self.keep_percent_spin.setValue(80.0)
        form.addRow("Retain best", self.keep_percent_spin)

        self.noise_scale_spin = QDoubleSpinBox()
        self.noise_scale_spin.setRange(0.0, 1000.0)
        self.noise_scale_spin.setDecimals(3)
        self.noise_scale_spin.setSingleStep(0.1)
        self.noise_scale_spin.setValue(1.0)
        form.addRow("Noise scale", self.noise_scale_spin)

        self.noise_sigma_edit = QLineEdit()
        self.noise_sigma_edit.setPlaceholderText("blank = residual RMS")
        form.addRow("Manual σ", self.noise_sigma_edit)

        self.amplitude_jitter_spin = QDoubleSpinBox()
        self.amplitude_jitter_spin.setRange(0.0, 1000.0)
        self.amplitude_jitter_spin.setDecimals(1)
        self.amplitude_jitter_spin.setSuffix("%")
        self.amplitude_jitter_spin.setValue(25.0)
        form.addRow("Amp start jitter", self.amplitude_jitter_spin)

        self.center_jitter_spin = QDoubleSpinBox()
        self.center_jitter_spin.setRange(0.0, 1.0e9)
        self.center_jitter_spin.setDecimals(4)
        self.center_jitter_spin.setSingleStep(max(x_span * 0.005, 0.1))
        self.center_jitter_spin.setValue(max(x_span * 0.02, 0.1))
        form.addRow("Center start jitter", self.center_jitter_spin)

        self.fwhm_jitter_spin = QDoubleSpinBox()
        self.fwhm_jitter_spin.setRange(0.0, 1000.0)
        self.fwhm_jitter_spin.setDecimals(1)
        self.fwhm_jitter_spin.setSuffix("%")
        self.fwhm_jitter_spin.setValue(25.0)
        form.addRow("FWHM start jitter", self.fwhm_jitter_spin)

        self.eta_jitter_spin = QDoubleSpinBox()
        self.eta_jitter_spin.setRange(0.0, 1.0)
        self.eta_jitter_spin.setDecimals(3)
        self.eta_jitter_spin.setSingleStep(0.025)
        self.eta_jitter_spin.setValue(0.1)
        form.addRow("η start jitter", self.eta_jitter_spin)

        self.max_nfev_spin = QSpinBox()
        self.max_nfev_spin.setRange(1, 1000000)
        self.max_nfev_spin.setValue(1000)
        form.addRow("Max evals / fit", self.max_nfev_spin)

        self.seed_edit = QLineEdit()
        self.seed_edit.setText(str(DEFAULT_MONTE_CARLO_SEED))
        self.seed_edit.setPlaceholderText("blank = random")
        form.addRow("Seed", self.seed_edit)

        layout.addLayout(form)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def settings(self) -> MonteCarloSettings:
        sigma_text = self.noise_sigma_edit.text().strip()
        seed_text = self.seed_edit.text().strip()
        return MonteCarloSettings(
            iterations=self.iterations_spin.value(),
            keep_fraction=self.keep_percent_spin.value() / 100.0,
            seed=None if not seed_text else int(seed_text),
            noise_scale=self.noise_scale_spin.value(),
            noise_sigma=None if not sigma_text else float(sigma_text),
            amplitude_jitter=self.amplitude_jitter_spin.value() / 100.0,
            center_jitter=self.center_jitter_spin.value(),
            fwhm_jitter=self.fwhm_jitter_spin.value() / 100.0,
            eta_jitter=self.eta_jitter_spin.value(),
            max_nfev=self.max_nfev_spin.value(),
        )


class MonteCarloResultsDialog(QDialog):
    def __init__(
        self,
        *,
        result: MonteCarloResult,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._result = result
        self._selected_record: MonteCarloFitRecord | None = result.best_record
        self.setWindowTitle("Monte Carlo results")
        self.resize(1280, 760)

        layout = QVBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Vertical)
        self.figure = Figure(figsize=(11.5, 3.8), constrained_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        splitter.addWidget(self.canvas)

        bottom = QSplitter(Qt.Orientation.Horizontal)
        self.tree = QTreeWidget()
        self.tree.setColumnCount(6)
        self.tree.setHeaderLabels(
            ["Rank", "Iteration", "Chi-squared", "Red chi", "Kept", "Nfev"]
        )
        self.tree.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.tree.itemSelectionChanged.connect(self._on_selection_changed)
        for rank, record in enumerate(result.records, start=1):
            item = QTreeWidgetItem(
                [
                    str(rank),
                    str(record.index),
                    f"{record.chisq:.6g}",
                    f"{record.redchi:.6g}",
                    "yes" if record.retained else "",
                    str(record.result.nfev),
                ]
            )
            item.setData(0, Qt.ItemDataRole.UserRole, rank - 1)
            if record.retained:
                for column in range(self.tree.columnCount()):
                    item.setBackground(column, QBrush(QColor("#E8F2FB")))
            self.tree.addTopLevelItem(item)
        for column in range(self.tree.columnCount()):
            self.tree.header().setSectionResizeMode(
                column,
                QHeaderView.ResizeMode.ResizeToContents,
            )
        bottom.addWidget(self.tree)

        self.summary_table = QTableWidget()
        self.summary_table.setColumnCount(10)
        self.summary_table.setHorizontalHeaderLabels(
            [
                "Peak",
                "Parameter",
                "Best",
                "Median",
                "-err",
                "+err",
                "Std",
                "Mean",
                "Min",
                "Max",
            ]
        )
        self.summary_table.verticalHeader().setVisible(False)
        self.summary_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self._populate_summary_table()
        bottom.addWidget(self.summary_table)
        bottom.setStretchFactor(0, 2)
        bottom.setStretchFactor(1, 3)
        splitter.addWidget(bottom)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter)

        button_row = QHBoxLayout()
        self.summary_label = QLabel(self._summary_text())
        button_row.addWidget(self.summary_label)
        button_row.addStretch(1)
        apply_best_button = QPushButton("Apply Best")
        apply_best_button.clicked.connect(self._apply_best)
        button_row.addWidget(apply_best_button)
        self.apply_selected_button = QPushButton("Apply Selected")
        self.apply_selected_button.clicked.connect(self._apply_selected)
        button_row.addWidget(self.apply_selected_button)
        keep_button = QPushButton("Keep Current")
        keep_button.clicked.connect(self.reject)
        button_row.addWidget(keep_button)
        layout.addLayout(button_row)

        self._build_axes()
        if self.tree.topLevelItemCount():
            self.tree.setCurrentItem(self.tree.topLevelItem(0))
        self._draw_plots()

    @property
    def selected_record(self) -> MonteCarloFitRecord | None:
        return self._selected_record

    def _summary_text(self) -> str:
        output = self._result.output_dir
        path_text = "" if output is None else f"  Output: {output}"
        return (
            f"{self._result.completed}/{self._result.attempted} fits completed; "
            f"{len(self._result.retained_records)} retained.{path_text}"
        )

    def _populate_summary_table(self) -> None:
        self.summary_table.setRowCount(len(self._result.summaries))
        for row, summary in enumerate(self._result.summaries):
            values = [
                summary.peak_label,
                summary.parameter,
                summary.best,
                summary.median,
                summary.lower_error,
                summary.upper_error,
                summary.std,
                summary.mean,
                summary.minimum,
                summary.maximum,
            ]
            for col, value in enumerate(values):
                if isinstance(value, float):
                    text = f"{value:.8g}"
                else:
                    text = str(value)
                item = QTableWidgetItem(text)
                if col in {0, 1}:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.summary_table.setItem(row, col, item)
        self.summary_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        self.summary_table.horizontalHeader().setSectionResizeMode(
            9,
            QHeaderView.ResizeMode.Stretch,
        )

    def _build_axes(self) -> None:
        self.figure.clear()
        gs = self.figure.add_gridspec(1, 2, width_ratios=[3, 2], wspace=0.28)
        self.model_ax = self.figure.add_subplot(gs[0])
        self.chi_ax = self.figure.add_subplot(gs[1])

    def _draw_plots(self) -> None:
        self._draw_model_plot()
        self._draw_chi_plot()
        self.canvas.draw_idle()

    def _draw_model_plot(self) -> None:
        ax = self.model_ax
        ax.clear()
        ax.grid(True, alpha=0.18)
        record = self._selected_record
        if record is None:
            ax.set_axis_off()
            return
        fit = record.result
        ax.plot(
            fit.x_fit,
            fit.y_data,
            color="#1f1f1f",
            linewidth=1.0,
            label="Synthetic data",
            zorder=3,
        )
        for component, curve in zip(
            fit.components, fit.component_curves, strict=False
        ):
            ax.plot(
                fit.x_fit,
                curve,
                color=component.color,
                linewidth=1.1,
                alpha=0.9,
                label=f"Peak {component.label}",
            )
        ax.plot(
            fit.x_fit,
            fit.total,
            color="#C3263A",
            linewidth=1.6,
            label="Total",
            zorder=4,
        )
        ax.set_title(f"Fit rank {self._record_rank(record)}", fontsize=10)
        ax.set_xlabel("Wavelength")
        ax.set_ylabel("Absorbance")
        ax.legend(loc="best", fontsize=8)

    def _draw_chi_plot(self) -> None:
        ax = self.chi_ax
        ax.clear()
        ranks = np.arange(1, len(self._result.records) + 1)
        chisq = np.asarray([record.chisq for record in self._result.records])
        colors = [
            "#0072B2" if record.retained else "#9AA3AA"
            for record in self._result.records
        ]
        ax.scatter(ranks, chisq, c=colors, s=24, alpha=0.9)
        if self._selected_record is not None:
            rank = self._record_rank(self._selected_record)
            ax.scatter(
                [rank],
                [self._selected_record.chisq],
                c=["#C3263A"],
                s=54,
                zorder=5,
            )
        ax.set_xlabel("Rank")
        ax.set_ylabel("Chi-squared")
        ax.grid(True, alpha=0.25)

    def _record_rank(self, record: MonteCarloFitRecord) -> int:
        for rank, candidate in enumerate(self._result.records, start=1):
            if candidate is record:
                return rank
        return 0

    def _on_selection_changed(self) -> None:
        items = self.tree.selectedItems()
        if not items:
            self._selected_record = None
            self._draw_plots()
            return
        index = int(items[0].data(0, Qt.ItemDataRole.UserRole))
        self._selected_record = self._result.records[index]
        self._draw_plots()

    def _apply_best(self) -> None:
        self._selected_record = self._result.best_record
        self.accept()

    def _apply_selected(self) -> None:
        if self._selected_record is None:
            QMessageBox.information(
                self,
                "Monte Carlo",
                "Select a Monte Carlo fit before applying it.",
            )
            return
        self.accept()


class UVVisNavigationToolbar(NavigationToolbar2QT):
    """Navigation toolbar whose Home action resets the UV-vis plot
    baseline."""

    def __init__(
        self,
        canvas: FigureCanvasQTAgg,
        parent: QWidget,
        *,
        home_callback,
    ) -> None:
        self._home_callback = home_callback
        super().__init__(canvas, parent)

    def home(self, *args) -> None:
        self._home_callback()
        super().update()
        self.push_current()


class UVVisFitMainWindow(QMainWindow):
    def __init__(
        self,
        *,
        initial_input_path: str | Path | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("SAXSShell UV-Vis Peak Fitter")
        self.resize(1260, 860)
        self.dataset: UVVisDataset | None = None
        self.components: list[PeakComponent] = []
        self.last_result: FitResult | None = None
        self.last_monte_carlo_result: MonteCarloResult | None = None
        self.current_session_path: Path | None = None
        self._updating_table = False
        self._drag_state: tuple[str, str] | None = None
        self._pending_peak_placement = False
        self._selected_peak_label: str | None = None
        self._residual_ylim: tuple[float, float] | None = None
        self._plot_view_dataset_key: tuple[object, ...] | None = None
        self._last_save_dir = self._read_last_save_dir()
        self._undo_stack: list[dict] = []
        self._undo_limit = 100
        self._build_ui()
        if initial_input_path is not None:
            self.load_data_path(initial_input_path)

    def _build_ui(self) -> None:
        self._build_menu_bar()
        root = QWidget(self)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        controls = QHBoxLayout()
        self.load_button = QPushButton("Load Data")
        self.load_button.clicked.connect(self.open_data_dialog)
        controls.addWidget(self.load_button)
        self.open_button = QPushButton("Open Fit")
        self.open_button.clicked.connect(self.open_session_dialog)
        controls.addWidget(self.open_button)
        self.save_button = QPushButton("Save Fit")
        self.save_button.clicked.connect(self.save_session_dialog)
        controls.addWidget(self.save_button)
        self.save_as_button = QPushButton("Save As")
        self.save_as_button.setToolTip(
            "Save under a new timestamped filename so the existing fit is "
            "preserved."
        )
        self.save_as_button.clicked.connect(self.save_session_as_dialog)
        controls.addWidget(self.save_as_button)
        controls.addSpacing(16)
        self.add_peak_button = QPushButton("Add Peak")
        self.add_peak_button.clicked.connect(self.add_peak)
        controls.addWidget(self.add_peak_button)
        self.remove_peak_button = QPushButton("Remove Peak")
        self.remove_peak_button.clicked.connect(self.remove_selected_peak)
        controls.addWidget(self.remove_peak_button)
        self.undo_button = QPushButton("Undo")
        self.undo_button.setToolTip("Revert the last change (Ctrl+Z)")
        self.undo_button.clicked.connect(self.undo)
        self.undo_button.setEnabled(False)
        controls.addWidget(self.undo_button)
        self._undo_shortcut = QShortcut(QKeySequence.StandardKey.Undo, self)
        self._undo_shortcut.activated.connect(self.undo)
        controls.addSpacing(16)
        self.fit_button = QPushButton("Fit")
        self.fit_button.clicked.connect(self.run_fit)
        controls.addWidget(self.fit_button)
        self.range_fit_button = QPushButton("Range Fit")
        self.range_fit_button.clicked.connect(self.run_selected_range_fit)
        controls.addWidget(self.range_fit_button)
        self.monte_carlo_button = QPushButton("Monte Carlo")
        self.monte_carlo_button.clicked.connect(self.run_monte_carlo_fit)
        controls.addWidget(self.monte_carlo_button)
        self.rescale_residual_button = QPushButton("Rescale Residual Y")
        self.rescale_residual_button.setToolTip(
            "Resize the residual plot's y-axis to fit the current residual."
        )
        self.rescale_residual_button.clicked.connect(self._rescale_residual_y)
        controls.addWidget(self.rescale_residual_button)
        controls.addStretch(1)
        controls.addWidget(QLabel("Fit min"))
        self.fit_min_spin = QDoubleSpinBox()
        self.fit_min_spin.setDecimals(4)
        self.fit_min_spin.setRange(-1.0e9, 1.0e9)
        self.fit_min_spin.valueChanged.connect(self._refresh_plot)
        controls.addWidget(self.fit_min_spin)
        controls.addWidget(QLabel("Fit max"))
        self.fit_max_spin = QDoubleSpinBox()
        self.fit_max_spin.setDecimals(4)
        self.fit_max_spin.setRange(-1.0e9, 1.0e9)
        self.fit_max_spin.valueChanged.connect(self._refresh_plot)
        controls.addWidget(self.fit_max_spin)
        layout.addLayout(controls)

        splitter = QSplitter(Qt.Orientation.Vertical)
        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        self.figure = Figure(figsize=(8.0, 4.8))
        self.figure.subplots_adjust(
            left=0.085,
            right=0.985,
            top=0.94,
            bottom=0.11,
            hspace=0.08,
        )
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = UVVisNavigationToolbar(
            self.canvas,
            self,
            home_callback=self._home_plot_view,
        )
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas)
        splitter.addWidget(plot_widget)

        self.table = ParameterGroupTableWidget()
        self.table.setHorizontalHeader(
            ParameterGroupHeader(Qt.Orientation.Horizontal, self.table)
        )
        self.table.setColumnCount(len(TABLE_COLUMNS))
        self.table.setHorizontalHeaderLabels(
            [title for title, _p, _k in TABLE_COLUMNS]
        )
        self.table.set_section_dividers(PARAMETER_SECTION_BOUNDARY_COLUMNS)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.table.itemChanged.connect(self._on_table_item_changed)
        self.table.cellDoubleClicked.connect(
            self._on_table_cell_double_clicked
        )
        self.table.itemSelectionChanged.connect(
            self._on_table_selection_changed
        )
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        for col in CONSTRAINT_COLUMNS:
            self.table.horizontalHeader().setSectionResizeMode(
                col,
                QHeaderView.ResizeMode.Stretch,
            )
        self._constraint_delegate = ConstraintCellDelegate(self)
        for col in CONSTRAINT_COLUMNS:
            self.table.setItemDelegateForColumn(col, self._constraint_delegate)
        eta_tooltip = (
            "η (eta): Gauss–Lorentz mixing fraction in the pseudo-Voigt.\n"
            "0 = pure Gaussian, 1 = pure Lorentzian.\n"
            "Pseudo-Voigt = (1 − η)·Gaussian + η·Lorentzian."
        )
        for col, (_title, parameter, _kind) in enumerate(TABLE_COLUMNS):
            if parameter == "eta":
                header_item = self.table.horizontalHeaderItem(col)
                if header_item is not None:
                    header_item.setToolTip(eta_tooltip)
        splitter.addWidget(self.table)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter)
        self.setCentralWidget(root)
        self.statusBar().showMessage("Ready")
        self._refresh_table()
        self._refresh_plot()

    def _build_menu_bar(self) -> None:
        file_menu = self.menuBar().addMenu("File")
        load_action = QAction("Load UV-Vis Data...", self)
        load_action.triggered.connect(self.open_data_dialog)
        file_menu.addAction(load_action)
        open_action = QAction("Open Saved Fit...", self)
        open_action.triggered.connect(self.open_session_dialog)
        file_menu.addAction(open_action)
        import_peaks_action = QAction("Import Peaks From Fit...", self)
        import_peaks_action.setToolTip(
            "Replace the current peak list with the peaks from another saved "
            "fit, keeping the loaded dataset and fit range."
        )
        import_peaks_action.triggered.connect(self.import_peaks_dialog)
        file_menu.addAction(import_peaks_action)
        save_action = QAction("Save Fit...", self)
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        save_action.triggered.connect(self.save_session_dialog)
        file_menu.addAction(save_action)
        save_as_action = QAction("Save Fit As...", self)
        save_as_action.setShortcut(QKeySequence.StandardKey.SaveAs)
        save_as_action.setToolTip(
            "Save the current fit to a new file, suggesting a timestamped "
            "filename so the existing fit is not overwritten."
        )
        save_as_action.triggered.connect(self.save_session_as_dialog)
        file_menu.addAction(save_as_action)
        export_action = QAction("Export Current Fit Folder...", self)
        export_action.triggered.connect(self.export_fit_folder_dialog)
        file_menu.addAction(export_action)
        file_menu.addSeparator()
        close_action = QAction("Close", self)
        close_action.triggered.connect(self.close)
        file_menu.addAction(close_action)

        fit_menu = self.menuBar().addMenu("Fit")
        fit_action = QAction("Run Fit", self)
        fit_action.triggered.connect(self.run_fit)
        fit_menu.addAction(fit_action)
        range_action = QAction("Run Range Fit for Selected Parameter", self)
        range_action.triggered.connect(self.run_selected_range_fit)
        fit_menu.addAction(range_action)
        monte_carlo_action = QAction("Run Monte Carlo Fit", self)
        monte_carlo_action.triggered.connect(self.run_monte_carlo_fit)
        fit_menu.addAction(monte_carlo_action)

        help_menu = self.menuBar().addMenu("Help")
        constraints_action = QAction("Constraints && Fit Shorthand...", self)
        constraints_action.triggered.connect(self.show_help_dialog)
        help_menu.addAction(constraints_action)

    def open_data_dialog(self) -> None:
        start_dir = (
            DEFAULT_UVVIS_DIR if DEFAULT_UVVIS_DIR.exists() else Path.home()
        )
        selected, _filter = QFileDialog.getOpenFileName(
            self,
            "Load UV-Vis data",
            str(start_dir),
            "Text data (*.txt *.csv *.dat);;All files (*)",
        )
        if selected:
            self.load_data_path(selected)

    def load_data_path(self, path: str | Path) -> None:
        try:
            dataset = load_uvvis_file(path)
        except Exception as exc:
            self._show_error("Load UV-Vis Data", str(exc))
            return
        self.dataset = dataset
        self.components = []
        self.last_result = None
        self.last_monte_carlo_result = None
        self.current_session_path = None
        self._pending_peak_placement = False
        self._residual_ylim = None
        self._reset_plot_view()
        self._selected_peak_label = None
        self._set_fit_window(dataset.x_min, dataset.x_max)
        self._clear_undo()
        self._refresh_table()
        self._refresh_plot()
        self.statusBar().showMessage(
            f"Loaded {dataset.source_path} — use Add Peak to place components."
        )

    def open_session_dialog(self) -> None:
        selected, _filter = QFileDialog.getOpenFileName(
            self,
            "Open saved UV-Vis fit",
            str(
                DEFAULT_UVVIS_DIR
                if DEFAULT_UVVIS_DIR.exists()
                else Path.home()
            ),
            "UV-Vis fit (*.json);;All files (*)",
        )
        if selected:
            self.load_session_path(selected)

    def load_session_path(self, path: str | Path) -> None:
        resolved = Path(path).expanduser().resolve()
        try:
            payload = json.loads(resolved.read_text(encoding="utf-8"))
            dataset, components, fit_min, fit_max = load_session_payload(
                payload
            )
        except Exception as exc:
            self._show_error("Open Saved Fit", str(exc))
            return
        self.dataset = dataset
        self.components = components
        self.last_result = None
        self.last_monte_carlo_result = None
        self.current_session_path = resolved
        self._residual_ylim = None
        self._reset_plot_view()
        self._set_fit_window(
            dataset.x_min if fit_min is None else fit_min,
            dataset.x_max if fit_max is None else fit_max,
        )
        self._clear_undo()
        self._refresh_table()
        self._refresh_plot()
        self.statusBar().showMessage(f"Opened {resolved}")

    def import_peaks_dialog(self) -> None:
        if self.dataset is None:
            self._show_error(
                "Import Peaks",
                "Load UV-Vis data before importing peaks.",
            )
            return
        start_dir = self._last_save_dir
        if not start_dir.exists():
            start_dir = (
                DEFAULT_UVVIS_DIR
                if DEFAULT_UVVIS_DIR.exists()
                else Path.home()
            )
        selected, _filter = QFileDialog.getOpenFileName(
            self,
            "Import peaks from saved fit",
            str(start_dir),
            "UV-Vis fit (*.json);;All files (*)",
        )
        if selected:
            self.import_peaks_from_path(selected)

    def import_peaks_from_path(self, path: str | Path) -> None:
        resolved = Path(path).expanduser().resolve()
        try:
            payload = json.loads(resolved.read_text(encoding="utf-8"))
            peaks_payload = list(payload.get("peaks", []) or [])
            new_components = [
                PeakComponent.from_dict(dict(item)) for item in peaks_payload
            ]
        except Exception as exc:
            self._show_error("Import Peaks", str(exc))
            return
        if not new_components:
            self._show_error(
                "Import Peaks",
                "No peaks found in selected file.",
            )
            return
        self._push_undo_snapshot()
        self.components = new_components
        self.last_result = None
        self.last_monte_carlo_result = None
        self._selected_peak_label = None
        self._refresh_table()
        self._refresh_plot()
        self.statusBar().showMessage(
            f"Imported {len(new_components)} peaks from {resolved.name}"
        )

    def save_session_dialog(self) -> None:
        if self.dataset is None:
            self._show_error("Save Fit", "Load UV-Vis data before saving.")
            return
        if self.current_session_path is not None:
            start_path = self.current_session_path
        else:
            target_dir = self._last_save_dir
            try:
                target_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                target_dir = Path.home()
            stem = Path(self.dataset.source_path).stem or "uvvis_fit"
            start_path = target_dir / f"{stem}.uvvis_fit.json"
        selected, _filter = QFileDialog.getSaveFileName(
            self,
            "Save UV-Vis fit",
            str(start_path),
            "UV-Vis fit (*.json);;All files (*)",
        )
        if selected:
            self.save_session_path(selected)

    def save_session_as_dialog(self) -> None:
        if self.dataset is None:
            self._show_error(
                "Save Fit As",
                "Load UV-Vis data before saving.",
            )
            return
        target_dir = self._last_save_dir
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            target_dir = Path.home()
        stem = Path(self.dataset.source_path).stem or "uvvis_fit"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suggested = target_dir / f"{stem}_{timestamp}.uvvis_fit.json"
        selected, _filter = QFileDialog.getSaveFileName(
            self,
            "Save UV-Vis fit as a new file",
            str(suggested),
            "UV-Vis fit (*.json);;All files (*)",
        )
        if selected:
            self.save_session_path(selected)

    def save_session_path(self, path: str | Path) -> None:
        if self.dataset is None:
            self._show_error("Save Fit", "Load UV-Vis data before saving.")
            return
        resolved = Path(path).expanduser().resolve()
        if resolved.suffix.lower() != ".json":
            resolved = resolved.with_suffix(".json")
        try:
            payload = session_payload(
                self.dataset,
                self.components,
                fit_min=self.fit_min_spin.value(),
                fit_max=self.fit_max_spin.value(),
                result=self.last_result,
            )
            if self.last_monte_carlo_result is not None:
                payload["monte_carlo"] = self.last_monte_carlo_result.to_dict(
                    include_records=False
                )
            resolved.write_text(
                json.dumps(payload, indent=2), encoding="utf-8"
            )
            csv_path = write_uvvis_fit_csv(
                resolved.with_suffix(".csv"),
                self.dataset,
                self.components,
                result=self.last_result,
                fit_min=self.fit_min_spin.value(),
                fit_max=self.fit_max_spin.value(),
            )
        except Exception as exc:
            self._show_error("Save Fit", str(exc))
            return
        self.current_session_path = resolved
        self._last_save_dir = resolved.parent
        self._write_last_save_dir(resolved.parent)
        self.statusBar().showMessage(f"Saved {resolved} (+ {csv_path.name})")

    def export_fit_folder_dialog(self) -> None:
        if self.dataset is None:
            self._show_error(
                "Export Fit", "Load UV-Vis data before exporting."
            )
            return
        selected = QFileDialog.getExistingDirectory(
            self,
            "Export current fit folder",
            str(Path(self.dataset.source_path).parent),
        )
        if not selected:
            return
        try:
            save_fit_bundle(
                selected,
                self.dataset,
                self.components,
                result=self.last_result,
                fit_min=self.fit_min_spin.value(),
                fit_max=self.fit_max_spin.value(),
            )
        except Exception as exc:
            self._show_error("Export Fit", str(exc))
            return
        self.statusBar().showMessage(f"Exported fit files to {selected}")

    def add_peak(self) -> None:
        if self.dataset is None:
            self._show_error(
                "Add Peak", "Load UV-Vis data before adding peaks."
            )
            return
        self._enter_peak_placement_mode()

    def _enter_peak_placement_mode(self) -> None:
        self._pending_peak_placement = True
        self.canvas.setCursor(Qt.CursorShape.CrossCursor)
        self.canvas.setFocus()
        self.statusBar().showMessage(
            "Click on the plot to place the new peak (Esc to cancel)."
        )

    def _exit_peak_placement_mode(self) -> None:
        self._pending_peak_placement = False
        self.canvas.unsetCursor()

    def _place_peak_at(self, x: float, y: float) -> None:
        dataset = self.dataset
        if dataset is None:
            return
        self._push_undo_snapshot()
        center = float(np.clip(float(x), dataset.x_min, dataset.x_max))
        amplitude = max(float(y), 0.0)
        fwhm = max((dataset.x_max - dataset.x_min) / 20.0, 1e-9)
        label = next_peak_label(self.components)
        component = PeakComponent(
            label=label,
            amplitude=amplitude,
            center=center,
            fwhm=fwhm,
            eta=0.2,
            color=PEAK_COLORS[len(self.components) % len(PEAK_COLORS)],
        )
        self.components.append(component)
        self.last_result = None
        self.last_monte_carlo_result = None
        self._refresh_table()
        self._refresh_plot()
        self.statusBar().showMessage(
            f"Added peak {label} at center={center:.3f}, amplitude={amplitude:.3g}"
        )

    def remove_selected_peak(self) -> None:
        row = self.table.currentRow()
        if row < 0 or row >= len(self.components):
            self._show_error(
                "Remove Peak", "Select a peak row before removing."
            )
            return
        self._push_undo_snapshot()
        label = self.components[row].label
        del self.components[row]
        self.last_result = None
        self.last_monte_carlo_result = None
        self._refresh_table()
        self._refresh_plot()
        self.statusBar().showMessage(f"Removed peak {label}")

    def run_fit(self) -> None:
        if self.dataset is None:
            self._show_error("Fit", "Load UV-Vis data before fitting.")
            return
        self._push_undo_snapshot()
        try:
            result = fit_components(
                self.dataset,
                self.components,
                fit_min=self.fit_min_spin.value(),
                fit_max=self.fit_max_spin.value(),
            )
        except Exception as exc:
            self._show_error("Fit", str(exc))
            return
        self.components = result.components
        self.last_result = result
        self.last_monte_carlo_result = None
        self._refresh_table()
        self._refresh_plot()
        self.statusBar().showMessage(
            f"Fit complete: chi-squared {result.chisq:.6g}, "
            f"redchi {result.redchi:.6g}"
        )

    def run_selected_range_fit(self) -> None:
        if self.dataset is None:
            self._show_error(
                "Range Fit", "Load UV-Vis data before range fitting."
            )
            return
        row = self.table.currentRow()
        col = self.table.currentColumn()
        if row < 0 or row >= len(self.components) or col not in VALUE_COLUMNS:
            self._show_error(
                "Range Fit",
                "Select an amplitude, area, center, FWHM, or eta value cell.",
            )
            return
        component = self.components[row]
        parameter = VALUE_COLUMNS[col]
        current_value = self._component_parameter_value(component, parameter)
        dialog = ParameterRangeDialog(
            peak_label=component.label,
            parameter=parameter,
            current_value=current_value,
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            lower, upper, steps = dialog.values()
        except Exception as exc:
            self._show_error("Range Fit", str(exc))
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_root = (
            Path(self.dataset.source_path).expanduser().resolve().parent
            / "uvvis_range_fits"
            / f"{timestamp}_{component.label}_{parameter}"
        )
        try:
            records = run_parameter_sweep(
                self.dataset,
                self.components,
                peak_label=component.label,
                parameter=parameter,
                lower=lower,
                upper=upper,
                steps=steps,
                output_root=output_root,
                fit_min=self.fit_min_spin.value(),
                fit_max=self.fit_max_spin.value(),
            )
        except Exception as exc:
            self._show_error("Range Fit", str(exc))
            return
        results_dialog = RangeSweepResultsDialog(
            peak_label=component.label,
            parameter=parameter,
            records=records,
            parent=self,
        )
        if results_dialog.exec() == QDialog.DialogCode.Accepted:
            selected = results_dialog.selected_record
            if selected is not None:
                self._push_undo_snapshot()
                self.components = selected.result.components
                self.last_result = selected.result
                self.last_monte_carlo_result = None
                self._refresh_table()
                self._refresh_plot()
                self.statusBar().showMessage(
                    f"Applied range fit from {selected.output_dir}"
                )
                return
        self.statusBar().showMessage(f"Range fits written to {output_root}")

    def run_monte_carlo_fit(self) -> None:
        if self.dataset is None:
            self._show_error("Monte Carlo", "Load UV-Vis data before fitting.")
            return
        if not self.components:
            self._show_error(
                "Monte Carlo", "Add at least one peak before fitting."
            )
            return
        dialog = MonteCarloSettingsDialog(dataset=self.dataset, parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            settings = dialog.settings()
        except Exception as exc:
            self._show_error("Monte Carlo", str(exc))
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_root = (
            Path(self.dataset.source_path).expanduser().resolve().parent
            / "uvvis_monte_carlo_fits"
            / f"{timestamp}_monte_carlo"
        )
        self.statusBar().showMessage("Running Monte Carlo fits...")
        progress_dialog = QProgressDialog(
            "Preparing Monte Carlo reference fit...",
            None,
            0,
            settings.iterations,
            self,
        )
        progress_dialog.setWindowTitle("Monte Carlo fitting")
        progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        progress_dialog.setMinimumDuration(0)
        progress_dialog.setAutoClose(False)
        progress_dialog.setAutoReset(False)
        progress_dialog.setValue(0)
        progress_dialog.show()
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QApplication.processEvents()
        error_message: str | None = None
        monte_carlo_result: MonteCarloResult | None = None

        def _update_monte_carlo_progress(
            processed: int,
            total: int,
            message: str,
        ) -> None:
            total = max(int(total), 1)
            processed = max(0, min(int(processed), total))
            progress_dialog.setMaximum(total)
            progress_dialog.setValue(processed)
            progress_dialog.setLabelText(message)
            self.statusBar().showMessage(message)
            QApplication.processEvents()

        try:
            monte_carlo_result = run_monte_carlo_fit_model(
                self.dataset,
                self.components,
                settings=settings,
                fit_min=self.fit_min_spin.value(),
                fit_max=self.fit_max_spin.value(),
                reference_result=self.last_result,
                output_root=output_root,
                progress_callback=_update_monte_carlo_progress,
            )
        except Exception as exc:
            error_message = str(exc)
        finally:
            QApplication.restoreOverrideCursor()
            progress_dialog.close()
        if error_message is not None:
            self._show_error("Monte Carlo", error_message)
            return
        if monte_carlo_result is None:
            self._show_error(
                "Monte Carlo", "Monte Carlo fitting did not finish."
            )
            return
        results_dialog = MonteCarloResultsDialog(
            result=monte_carlo_result,
            parent=self,
        )
        if results_dialog.exec() == QDialog.DialogCode.Accepted:
            selected = results_dialog.selected_record
            if selected is not None:
                self._push_undo_snapshot()
                self.components = [
                    component.copy()
                    for component in selected.result.components
                ]
                self.last_result = evaluate_fit_result(
                    self.dataset,
                    self.components,
                    fit_min=self.fit_min_spin.value(),
                    fit_max=self.fit_max_spin.value(),
                    message=(
                        "Monte Carlo parameter set evaluated on original data."
                    ),
                )
                self.last_monte_carlo_result = monte_carlo_result
                self._refresh_table()
                self._refresh_plot()
                self.statusBar().showMessage(
                    f"Applied Monte Carlo fit from {monte_carlo_result.output_dir}"
                )
                return
        self.last_monte_carlo_result = monte_carlo_result
        self.statusBar().showMessage(
            f"Monte Carlo fits written to {monte_carlo_result.output_dir}"
        )

    def _component_parameter_value(
        self,
        component: PeakComponent,
        parameter: str,
    ) -> float:
        if parameter == "area":
            return component.area
        return float(getattr(component, parameter))

    def _value_clipping(
        self,
        component: PeakComponent,
        parameter: str,
    ) -> tuple[bool, str]:
        value = component_parameter_value(component, parameter)
        tol = 1e-6
        constraint = component.constraints.get(parameter, "")
        interval = parse_interval_constraint(constraint)
        if interval is not None:
            lower, upper = interval
            if abs(value - lower) <= tol * max(abs(lower), 1.0):
                return (
                    True,
                    f"{parameter} pinned at constraint lower {lower:.6g}",
                )
            if abs(value - upper) <= tol * max(abs(upper), 1.0):
                return (
                    True,
                    f"{parameter} pinned at constraint upper {upper:.6g}",
                )
        if parameter == "eta":
            if value <= 0.0 + tol:
                return True, "η = 0 (pure Gaussian)"
            if value >= 1.0 - tol:
                return True, "η = 1 (pure Lorentzian)"
        if parameter in {"amplitude", "area"} and value <= tol:
            return True, f"{parameter} pinned at zero"
        if parameter == "fwhm" and value <= 1e-6:
            return True, "FWHM pinned at minimum"
        dataset = self.dataset
        if dataset is not None:
            x_span = dataset.x_max - dataset.x_min
            if parameter == "fwhm" and x_span > 0:
                if value >= 4.0 * x_span - tol * max(x_span, 1.0):
                    return (
                        True,
                        f"FWHM at upper bound 4·span ({4.0 * x_span:.6g})",
                    )
        return False, ""

    def _label_brushes(self, color_hex: str) -> tuple[QBrush, QBrush]:
        bg = QColor(color_hex)
        if not bg.isValid():
            bg = QColor("#999999")
        luminance = (
            0.2126 * bg.red() + 0.7152 * bg.green() + 0.0722 * bg.blue()
        ) / 255.0
        fg = QColor("#FFFFFF") if luminance < 0.55 else QColor("#1A1A1A")
        return QBrush(bg), QBrush(fg)

    def _select_peak_row(self, label: str) -> None:
        for row, component in enumerate(self.components):
            if component.label == label:
                if self.table.currentRow() != row:
                    self.table.setCurrentCell(row, 0)
                return

    def _on_table_selection_changed(self) -> None:
        row = self.table.currentRow()
        if 0 <= row < len(self.components):
            new_label = self.components[row].label
        else:
            new_label = None
        if new_label != self._selected_peak_label:
            self._selected_peak_label = new_label
            self._refresh_plot()

    def _snap_relation_constraint(
        self,
        component: PeakComponent,
        parameter: str,
        text: str,
    ) -> None:
        if not is_relation_constraint(text):
            return
        values_by_label = {
            other.label: component_parameter_value(other, parameter)
            for other in self.components
            if other.label != component.label
        }
        try:
            snapped = relation_constraint_value(text, values_by_label)
        except Exception:
            return
        set_component_parameter(component, parameter, snapped)

    def _forward_propagate(self, parameter: str) -> None:
        """Fixed-point: enforce all relational constraints on `parameter`."""
        for _ in range(16):
            changed_any = False
            for component in self.components:
                text = component.constraints.get(parameter, "")
                if not is_relation_constraint(text):
                    continue
                values_by_label = {
                    other.label: component_parameter_value(other, parameter)
                    for other in self.components
                    if other.label != component.label
                }
                try:
                    target = relation_constraint_value(text, values_by_label)
                except Exception:
                    continue
                current = component_parameter_value(component, parameter)
                if abs(target - current) > 1e-9:
                    set_component_parameter(component, parameter, target)
                    changed_any = True
            if not changed_any:
                break

    def _backward_invert_chain(
        self,
        edited_label: str,
        parameter: str,
    ) -> None:
        """When the edited peak's own constraint is relational, walk up
        the chain and update each referenced peak so the user's new
        value is consistent.

        Assumes linear expressions of a single referenced peak.
        """
        visited: set[str] = set()
        current_label = edited_label
        while current_label not in visited:
            visited.add(current_label)
            component = self._component_by_label(current_label)
            if component is None:
                return
            text = component.constraints.get(parameter, "")
            if not is_relation_constraint(text):
                return
            refs = set(PEAK_REFERENCE_PATTERN.findall(text))
            if len(refs) != 1:
                return
            ref_label = next(iter(refs))
            ref_component = self._component_by_label(ref_label)
            if ref_component is None:
                return
            value_now = component_parameter_value(component, parameter)
            try:
                b = relation_constraint_value(text, {ref_label: 0.0})
                ab = relation_constraint_value(text, {ref_label: 1.0})
            except Exception:
                return
            slope = ab - b
            if abs(slope) < 1e-12:
                return
            new_ref_value = (value_now - b) / slope
            set_component_parameter(ref_component, parameter, new_ref_value)
            current_label = ref_label

    def _propagate_value_change(
        self, edited_label: str, parameter: str
    ) -> None:
        """Backward-invert chain (if edited peak is itself constrained),
        then forward-propagate so every relational dependent stays in
        sync."""
        self._backward_invert_chain(edited_label, parameter)
        self._forward_propagate(parameter)

    def _set_fit_window(self, fit_min: float, fit_max: float) -> None:
        self.fit_min_spin.blockSignals(True)
        self.fit_max_spin.blockSignals(True)
        lower = min(float(fit_min), float(fit_max))
        upper = max(float(fit_min), float(fit_max))
        self.fit_min_spin.setValue(lower)
        self.fit_max_spin.setValue(upper)
        self.fit_min_spin.blockSignals(False)
        self.fit_max_spin.blockSignals(False)

    def _refresh_table(self) -> None:
        self._updating_table = True
        self.table.setRowCount(len(self.components))
        for row, component in enumerate(self.components):
            for col, (_title, parameter, kind) in enumerate(TABLE_COLUMNS):
                if kind == "label":
                    item = self._table_item(component.label, editable=False)
                    bg, fg = self._label_brushes(component.color)
                    item.setBackground(bg)
                    item.setForeground(fg)
                    font = QFont(item.font())
                    font.setBold(True)
                    item.setFont(font)
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                elif kind == "value" and parameter is not None:
                    value = self._component_parameter_value(
                        component, parameter
                    )
                    item = self._table_item(f"{value:.8g}", editable=True)
                    clipping, reason = self._value_clipping(
                        component, parameter
                    )
                    if clipping:
                        item.setBackground(QBrush(QColor("#FCE19A")))
                        item.setForeground(QBrush(QColor("#7A3E00")))
                        item.setToolTip(
                            f"{reason}\nDouble-click to run a range fit."
                        )
                    else:
                        item.setToolTip("Double-click to run a range fit.")
                elif kind == "lock" and parameter is not None:
                    item = self._checkbox_item(
                        checked=component.locked.get(parameter, False)
                    )
                elif kind == "constraint" and parameter is not None:
                    item = self._table_item(
                        component.constraints.get(parameter, ""),
                        editable=True,
                    )
                else:
                    item = self._table_item("", editable=False)
                self.table.setItem(row, col, item)
        self._updating_table = False

    def _table_item(self, text: str, *, editable: bool) -> QTableWidgetItem:
        item = QTableWidgetItem(text)
        flags = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
        if editable:
            flags |= Qt.ItemFlag.ItemIsEditable
        item.setFlags(flags)
        return item

    def _checkbox_item(self, *, checked: bool) -> QTableWidgetItem:
        item = QTableWidgetItem("")
        item.setFlags(
            Qt.ItemFlag.ItemIsEnabled
            | Qt.ItemFlag.ItemIsSelectable
            | Qt.ItemFlag.ItemIsUserCheckable
        )
        item.setCheckState(
            Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        )
        return item

    def _on_table_item_changed(self, item: QTableWidgetItem) -> None:
        if self._updating_table:
            return
        row = item.row()
        col = item.column()
        if row < 0 or row >= len(self.components):
            return
        component = self.components[row]
        self._push_undo_snapshot()
        try:
            if col in VALUE_COLUMNS:
                parameter = VALUE_COLUMNS[col]
                set_component_parameter(
                    component,
                    parameter,
                    float(item.text().strip()),
                )
                self._propagate_value_change(component.label, parameter)
            elif col in LOCK_COLUMNS:
                component.locked[LOCK_COLUMNS[col]] = (
                    item.checkState() == Qt.CheckState.Checked
                )
            elif col in CONSTRAINT_COLUMNS:
                parameter = CONSTRAINT_COLUMNS[col]
                new_text = item.text().strip()
                component.constraints[parameter] = new_text
                self._snap_relation_constraint(component, parameter, new_text)
                self._forward_propagate(parameter)
            else:
                self._undo_stack.pop()
                self._update_undo_button()
                return
        except Exception as exc:
            if self._undo_stack:
                self._undo_stack.pop()
                self._update_undo_button()
            self._show_error("Peak Table", str(exc))
            self._refresh_table()
            return
        self.last_result = None
        self.last_monte_carlo_result = None
        self._refresh_table()
        self._refresh_plot()

    def _on_table_cell_double_clicked(self, row: int, col: int) -> None:
        if row < 0 or row >= len(self.components) or col not in VALUE_COLUMNS:
            return
        self.table.setCurrentCell(row, col)
        self.run_selected_range_fit()

    def _refresh_plot(self) -> None:
        dataset = self.dataset
        plot_view = self._current_plot_view(dataset)
        self.figure.clear()
        gs = self.figure.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.08)
        ax = self.figure.add_subplot(gs[0])
        residual_ax = self.figure.add_subplot(gs[1], sharex=ax)
        self.ax = ax
        self.residual_ax = residual_ax
        if self.dataset is None:
            ax.text(
                0.5,
                0.5,
                "Load UV-Vis data",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color="0.35",
            )
            ax.set_axis_off()
            residual_ax.set_axis_off()
            self.canvas.draw_idle()
            self._reset_plot_view()
            return

        fit_lo = float(self.fit_min_spin.value())
        fit_hi = float(self.fit_max_spin.value())
        ax.axvspan(
            fit_lo,
            fit_hi,
            color="#f1f3f5",
            alpha=0.65,
            zorder=0,
            label="Fit range",
        )
        for value in (fit_lo, fit_hi):
            ax.axvline(
                value,
                color="#5b6770",
                linewidth=1.0,
                linestyle="--",
                alpha=0.85,
                zorder=1,
            )
        ax.axhline(
            0.0,
            color="#5b6770",
            linewidth=0.8,
            alpha=0.6,
            zorder=1,
        )
        ax.plot(
            dataset.x,
            dataset.y,
            color="#1f1f1f",
            linewidth=1.2,
            label="Data",
            zorder=3,
        )
        component_curves, total = evaluate_components(
            dataset.x, self.components
        )
        for component, curve in zip(
            self.components,
            component_curves,
            strict=False,
        ):
            selected = component.label == self._selected_peak_label
            ax.plot(
                dataset.x,
                curve,
                color=component.color,
                linewidth=2.6 if selected else 1.4,
                label=f"Peak {component.label}",
                alpha=1.0 if selected else 0.95,
                zorder=5 if selected else 2,
            )
            self._draw_component_handles(ax, component, selected=selected)
        if self.components:
            ax.plot(
                dataset.x,
                total,
                color="#C3263A",
                linewidth=1.8,
                label="Total",
                zorder=4,
            )
        if self.last_result is not None:
            metrics_text = self._format_fit_metrics(self.last_result)
            ax.text(
                0.985,
                0.025,
                metrics_text,
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=9,
                color="0.15",
                bbox=dict(
                    facecolor="white",
                    edgecolor="0.65",
                    boxstyle="round,pad=0.45",
                    alpha=0.92,
                ),
                family="monospace",
                zorder=8,
            )
        ax.set_xlabel("")
        ax.set_ylabel(dataset.y_label)
        if dataset.source_path:
            ax.set_title(Path(dataset.source_path).name)
        ax.grid(True, alpha=0.18)
        ax.legend(loc="best", fontsize=8)
        ax.tick_params(axis="x", labelbottom=False)
        self._draw_residual_axis(residual_ax, fit_lo, fit_hi)
        if plot_view is None:
            self._apply_fixed_view_limits(ax)
        else:
            xlim, ylim = plot_view
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
        self._plot_view_dataset_key = self._dataset_view_key(dataset)
        self.canvas.draw_idle()

        if not hasattr(self, "_mpl_connections"):
            self._mpl_connections = [
                self.canvas.mpl_connect(
                    "button_press_event", self._on_plot_press
                ),
                self.canvas.mpl_connect(
                    "motion_notify_event", self._on_plot_motion
                ),
                self.canvas.mpl_connect(
                    "button_release_event",
                    self._on_plot_release,
                ),
            ]

    def _format_fit_metrics(self, result: FitResult) -> str:
        residual = np.asarray(result.residual, dtype=float)
        y_data = np.asarray(result.y_data, dtype=float)
        ss_res = float(np.sum(residual**2))
        y_mean = float(np.mean(y_data)) if y_data.size else 0.0
        ss_tot = float(np.sum((y_data - y_mean) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        rms = math.sqrt(ss_res / max(residual.size, 1))
        max_abs = float(np.nanmax(np.abs(residual))) if residual.size else 0.0
        return (
            f"R²       = {r2:.4f}\n"
            f"χ²       = {result.chisq:.4g}\n"
            f"red χ²   = {result.redchi:.4g}\n"
            f"RMS resid = {rms:.4g}\n"
            f"max |r|  = {max_abs:.4g}"
        )

    def _draw_residual_axis(
        self,
        residual_ax,
        fit_lo: float,
        fit_hi: float,
    ) -> None:
        dataset = self.dataset
        if dataset is None:
            return
        residual_ax.axhline(0.0, color="#5b6770", linewidth=0.8, zorder=1)
        residual_ax.axvspan(
            fit_lo,
            fit_hi,
            color="#f1f3f5",
            alpha=0.65,
            zorder=0,
        )
        for value in (fit_lo, fit_hi):
            residual_ax.axvline(
                value,
                color="#5b6770",
                linewidth=1.0,
                linestyle="--",
                alpha=0.85,
                zorder=1,
            )
        if self.last_result is not None:
            residual_ax.plot(
                self.last_result.x_fit,
                self.last_result.residual,
                color="#1f1f1f",
                linewidth=1.0,
                zorder=3,
            )
            if self._residual_ylim is None:
                m = float(np.nanmax(np.abs(self.last_result.residual)))
                if not np.isfinite(m) or m <= 0:
                    m = max(dataset.y_scale * 0.05, 1.0)
                self._residual_ylim = (-1.15 * m, 1.15 * m)
        residual_ax.set_xlim(dataset.x_min, dataset.x_max)
        if self._residual_ylim is not None:
            residual_ax.set_ylim(*self._residual_ylim)
        else:
            residual_ax.set_ylim(-1.0, 1.0)
        residual_ax.set_xlabel(dataset.x_label)
        residual_ax.set_ylabel("Residual")
        residual_ax.grid(True, alpha=0.15)

    def _rescale_residual_y(self) -> None:
        if self.last_result is None:
            self._residual_ylim = None
        else:
            m = float(np.nanmax(np.abs(self.last_result.residual)))
            if not np.isfinite(m) or m <= 0:
                m = 1.0
            self._residual_ylim = (-1.15 * m, 1.15 * m)
        self._refresh_plot()

    def _apply_fixed_view_limits(self, ax) -> None:
        dataset = self.dataset
        if dataset is None:
            return
        ax.set_xlim(dataset.x_min, dataset.x_max)
        y_lo = float(np.nanmin(dataset.y))
        y_hi = float(np.nanmax(dataset.y))
        if not np.isfinite(y_lo) or not np.isfinite(y_hi) or y_hi <= y_lo:
            y_lo, y_hi = 0.0, max(float(dataset.y_scale), 1.0)
        span = y_hi - y_lo
        pad = span * 0.05 if span > 0 else max(abs(y_hi), 1.0) * 0.05
        ax.set_ylim(y_lo - pad, y_hi + pad)

    def _dataset_view_key(
        self,
        dataset: UVVisDataset,
    ) -> tuple[object, ...]:
        return (
            str(dataset.source_path),
            int(dataset.x.size),
            int(dataset.y.size),
            float(dataset.x_min),
            float(dataset.x_max),
        )

    def _current_plot_view(
        self,
        dataset: UVVisDataset | None,
    ) -> tuple[tuple[float, float], tuple[float, float]] | None:
        if dataset is None:
            return None
        if self._plot_view_dataset_key != self._dataset_view_key(dataset):
            return None
        ax = getattr(self, "ax", None)
        if ax is None:
            return None
        xlim = tuple(float(value) for value in ax.get_xlim())
        ylim = tuple(float(value) for value in ax.get_ylim())
        if not all(np.isfinite(value) for value in (*xlim, *ylim)):
            return None
        return xlim, ylim

    def _reset_plot_view(self) -> None:
        self._plot_view_dataset_key = None

    def _home_plot_view(self) -> None:
        self._reset_plot_view()
        self._refresh_plot()

    def _draw_component_handles(
        self,
        ax,
        component: PeakComponent,
        *,
        selected: bool = False,
    ) -> None:
        center = component.center
        amplitude = component.amplitude
        left = center - component.fwhm / 2.0
        right = center + component.fwhm / 2.0
        edge_color = "#1a1a1a" if selected else "white"
        edge_width = 1.4 if selected else 0.7
        size_top = 60 if selected else 34
        size_side = 52 if selected else 30
        ax.scatter(
            [center],
            [amplitude],
            s=size_top,
            marker="s",
            color=component.color,
            edgecolor=edge_color,
            linewidth=edge_width,
            zorder=7 if selected else 6,
        )
        ax.scatter(
            [left, right],
            [amplitude / 2.0, amplitude / 2.0],
            s=size_side,
            marker="^",
            color=component.color,
            edgecolor=edge_color,
            linewidth=edge_width,
            zorder=7 if selected else 6,
        )

    def _on_plot_press(self, event) -> None:
        if self.dataset is None or event.inaxes is not getattr(
            self, "ax", None
        ):
            return
        if getattr(self.toolbar, "mode", ""):
            return
        if self._pending_peak_placement:
            if event.xdata is None or event.ydata is None:
                return
            self._exit_peak_placement_mode()
            self._place_peak_at(float(event.xdata), float(event.ydata))
            return
        picked = self._pick_component_handle(event)
        self._drag_state = picked
        if picked is not None:
            self._push_undo_snapshot()
            self._select_peak_row(picked[0])

    def keyPressEvent(self, event) -> None:
        if self._pending_peak_placement and event.key() == Qt.Key.Key_Escape:
            self._exit_peak_placement_mode()
            self.statusBar().showMessage("Peak placement cancelled.")
            event.accept()
            return
        super().keyPressEvent(event)

    def _on_plot_motion(self, event) -> None:
        if (
            self._drag_state is None
            or event.xdata is None
            or event.ydata is None
        ):
            return
        if event.inaxes is not getattr(self, "ax", None):
            return
        label, mode = self._drag_state
        component = self._component_by_label(label)
        if component is None:
            return
        changed_params: list[str] = []
        if mode == "width":
            if not component.locked.get("fwhm", False):
                component.fwhm = max(
                    2.0 * abs(float(event.xdata) - component.center),
                    1e-9,
                )
                changed_params.append("fwhm")
        else:
            if not component.locked.get("center", False):
                component.center = float(event.xdata)
                changed_params.append("center")
            if not component.locked.get("amplitude", False):
                component.amplitude = max(float(event.ydata), 0.0)
                changed_params.append("amplitude")
        for parameter in changed_params:
            self._propagate_value_change(label, parameter)
        self.last_result = None
        self.last_monte_carlo_result = None
        self._refresh_table()
        self._refresh_plot()

    def _on_plot_release(self, _event) -> None:
        self._drag_state = None

    def _pick_component_handle(self, event) -> tuple[str, str] | None:
        ax = getattr(self, "ax", None)
        if ax is None:
            return None
        best_distance = math.inf
        best: tuple[str, str] | None = None
        for component in self.components:
            points = [
                (component.center, component.amplitude, "center"),
                (
                    component.center - component.fwhm / 2.0,
                    component.amplitude / 2.0,
                    "width",
                ),
                (
                    component.center + component.fwhm / 2.0,
                    component.amplitude / 2.0,
                    "width",
                ),
            ]
            for x_value, y_value, mode in points:
                px, py = ax.transData.transform((x_value, y_value))
                distance = math.hypot(float(event.x) - px, float(event.y) - py)
                if distance < best_distance:
                    best_distance = distance
                    best = (component.label, mode)
        return best if best_distance <= 18.0 else None

    def _component_by_label(self, label: str) -> PeakComponent | None:
        return next(
            (
                component
                for component in self.components
                if component.label == label
            ),
            None,
        )

    def _show_error(self, title: str, message: str) -> None:
        QMessageBox.critical(self, title, message)
        self.statusBar().showMessage(message, 5000)

    def show_help_dialog(self) -> None:
        QMessageBox.information(
            self,
            "Constraints & Fit Shorthand",
            HELP_TEXT,
        )

    def _push_undo_snapshot(self) -> None:
        self._undo_stack.append(
            {
                "components": [
                    component.copy() for component in self.components
                ],
                "last_result": self.last_result,
                "last_monte_carlo_result": self.last_monte_carlo_result,
                "fit_min": float(self.fit_min_spin.value()),
                "fit_max": float(self.fit_max_spin.value()),
                "selected": self._selected_peak_label,
                "residual_ylim": self._residual_ylim,
            }
        )
        if len(self._undo_stack) > self._undo_limit:
            self._undo_stack = self._undo_stack[-self._undo_limit :]
        self._update_undo_button()

    def _clear_undo(self) -> None:
        self._undo_stack.clear()
        self._update_undo_button()

    def _update_undo_button(self) -> None:
        self.undo_button.setEnabled(bool(self._undo_stack))

    def undo(self) -> None:
        if not self._undo_stack:
            return
        snap = self._undo_stack.pop()
        self.components = [
            component.copy() for component in snap["components"]
        ]
        self.last_result = snap["last_result"]
        self.last_monte_carlo_result = snap.get("last_monte_carlo_result")
        self._set_fit_window(snap["fit_min"], snap["fit_max"])
        self._selected_peak_label = snap["selected"]
        self._residual_ylim = snap["residual_ylim"]
        self._pending_peak_placement = False
        self.canvas.unsetCursor()
        self._refresh_table()
        self._refresh_plot()
        self._update_undo_button()
        self.statusBar().showMessage("Reverted last change.")

    def _read_last_save_dir(self) -> Path:
        settings = QSettings(SETTINGS_ORG, SETTINGS_APP)
        stored = settings.value(SETTINGS_KEY_LAST_SAVE_DIR, "", type=str)
        if stored:
            return Path(stored)
        return DEFAULT_SAVE_DIR

    def _write_last_save_dir(self, directory: Path) -> None:
        settings = QSettings(SETTINGS_ORG, SETTINGS_APP)
        settings.setValue(SETTINGS_KEY_LAST_SAVE_DIR, str(directory))


def launch_uvvis_fitting_ui(
    initial_input_path: str | Path | None = None,
) -> UVVisFitMainWindow:
    app = QApplication.instance()
    if app is None:
        prepare_saxshell_application_identity()
        app = QApplication(sys.argv)
    configure_saxshell_application(app)
    window = UVVisFitMainWindow(initial_input_path=initial_input_path)
    window.show()
    window.raise_()
    _OPEN_WINDOWS.append(window)
    return window


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="uvvisfit",
        description="Launch the SAXSShell UV-Vis pseudo-Voigt fitting UI.",
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        type=Path,
        help="Optional UV-Vis text file or saved fit JSON to open.",
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
    window = launch_uvvis_fitting_ui()
    if args.input_path is not None:
        path = Path(args.input_path)
        if path.suffix.lower() == ".json":
            window.load_session_path(path)
        else:
            window.load_data_path(path)
    if created_app:
        assert app is not None
        return int(app.exec())
    return 0


__all__ = [
    "UVVisFitMainWindow",
    "build_parser",
    "launch_uvvis_fitting_ui",
    "main",
]
