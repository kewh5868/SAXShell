from __future__ import annotations

import csv
import json
import pickle
import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from inspect import Parameter, signature
from pathlib import Path
from typing import cast

import numpy as np
from matplotlib import colormaps
from matplotlib.colors import to_hex
from PySide6.QtCore import (
    QEventLoop,
    QObject,
    QRect,
    QSettings,
    QSize,
    Qt,
    QThread,
    QTimer,
    QUrl,
    Signal,
    Slot,
)
from PySide6.QtGui import (
    QAction,
    QColor,
    QDesktopServices,
    QFont,
    QKeySequence,
    QShortcut,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFontComboBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from saxshell.saxs._model_templates import (
    list_template_specs,
    load_template_spec,
)
from saxshell.saxs.contrast.settings import (
    COMPONENT_BUILD_MODE_CONTRAST,
    component_build_mode_label,
    normalize_component_build_mode,
)
from saxshell.saxs.dream import (
    DreamParameterEntry,
    DreamRunBundle,
    DreamRunSettings,
    SAXSDreamResultsLoader,
    SAXSDreamWorkflow,
    load_dream_settings,
    load_parameter_map,
)
from saxshell.saxs.model_report import (
    DreamFilterReportView,
    DreamModelReportContext,
    PriorHistogramRequest,
    ReportComponentPlotData,
    ReportComponentSeries,
    export_dream_model_report_pptx,
)
from saxshell.saxs.prefit import (
    ClusterGeometryMetadataRow,
    PrefitEvaluation,
    PrefitParameterEntry,
    PrefitScaleRecommendation,
    SAXSPrefitWorkflow,
)
from saxshell.saxs.prefit.workflow import (
    SOLUTE_VOLUME_FRACTION_PARAMETER_NAMES,
    SOLVENT_VOLUME_FRACTION_PARAMETER_NAMES,
    SOLVENT_WEIGHT_PARAMETER_NAMES,
    normalize_requested_q_range_to_supported,
    q_range_boundary_tolerance,
)
from saxshell.saxs.project_manager import (
    ClusterImportResult,
    ExperimentalDataSummary,
    PowerPointExportSettings,
    ProjectSettings,
    SAXSProjectManager,
    build_project_paths,
    effective_q_range_for_settings,
    export_prior_histogram_npy,
    export_prior_histogram_table,
    load_built_component_q_range,
    load_experimental_data_file,
    project_artifact_paths,
)
from saxshell.saxs.solute_volume_fraction import DISPLAY_FRACTION_DECIMALS
from saxshell.saxs.solution_scattering_estimator import (
    SolutionScatteringEstimate,
)
from saxshell.saxs.stoichiometry import (
    build_stoichiometry_target,
    evaluate_weighted_stoichiometry,
    stoichiometry_deviation_text,
    stoichiometry_ratio_text,
    stoichiometry_target_text,
)
from saxshell.saxs.template_installation import (
    format_validation_report,
    install_template_candidate,
)
from saxshell.saxs.ui.branding import (
    build_saxshell_brand_widget,
    configure_saxshell_application,
    create_saxshell_startup_splash,
    load_saxshell_icon,
    prepare_saxshell_application_identity,
)
from saxshell.saxs.ui.distribution_window import DistributionSetupWindow
from saxshell.saxs.ui.dream_tab import DreamTab
from saxshell.saxs.ui.dream_violin_export_dialog import DreamViolinExportDialog
from saxshell.saxs.ui.prefit_tab import PrefitTab
from saxshell.saxs.ui.prior_histogram_window import PriorHistogramWindow
from saxshell.saxs.ui.progress_dialog import SAXSProgressDialog
from saxshell.saxs.ui.project_setup_tab import ProjectSetupTab
from saxshell.saxs.ui.solute_volume_fraction_widget import (
    AttenuationEstimateToolWindow,
    FluorescenceEstimateToolWindow,
    NumberDensityEstimateToolWindow,
    SoluteVolumeFractionToolWindow,
)
from saxshell.version import __version__

GITHUB_REPOSITORY_URL = "https://github.com/kewh5868/SAXSShell"
CONTACT_EMAIL = "keith.white@colorado.edu"
RECENT_PROJECTS_KEY = "recent_project_dirs"
CONSOLE_AUTOSCROLL_KEY = "console_autoscroll_enabled"
MAX_RECENT_PROJECTS = 10
PROJECT_LOAD_PREP_STEPS = 4
PROJECT_LOAD_TOTAL_STEPS = 12
DEFAULT_WINDOW_PRESET_KEY = "laptop_14"
REPO_ROOT = Path(__file__).resolve().parents[4]
EQUIVALENT_SPHERE_MIX_TEMPLATE_NAMES = {
    "template_pydream_poly_lma_hs_mix_approx",
    "template_pydream_poly_lma_hs_legacy",
}
POWERPOINT_COLOR_MAP_OPTIONS = (
    "viridis",
    "plasma",
    "cividis",
    "magma",
    "inferno",
    "turbo",
)


@dataclass(frozen=True, slots=True)
class RuntimeBundleOpener:
    label: str
    stored_value: str
    launch_target: str
    launch_mode: str


@dataclass(frozen=True, slots=True)
class WindowLayoutPreset:
    key: str
    label: str
    width: int
    height: int
    ui_scale: float


WINDOW_LAYOUT_PRESETS = (
    WindowLayoutPreset(
        key="laptop_13",
        label="13-inch Laptop (Compact)",
        width=1180,
        height=760,
        ui_scale=0.95,
    ),
    WindowLayoutPreset(
        key="laptop_14",
        label="14-inch Laptop / MacBook Pro",
        width=1280,
        height=820,
        ui_scale=1.0,
    ),
    WindowLayoutPreset(
        key="laptop_16",
        label="15-inch / 16-inch Laptop",
        width=1440,
        height=900,
        ui_scale=1.05,
    ),
    WindowLayoutPreset(
        key="display_1080p",
        label="External Display (1080p)",
        width=1500,
        height=880,
        ui_scale=1.0,
    ),
    WindowLayoutPreset(
        key="display_1440p",
        label="External Display (1440p / QHD)",
        width=1680,
        height=980,
        ui_scale=1.1,
    ),
)
WINDOW_LAYOUT_PRESET_MAP = {
    preset.key: preset for preset in WINDOW_LAYOUT_PRESETS
}


@dataclass(frozen=True, slots=True)
class FitQualityMetrics:
    rmse: float
    mean_abs_residual: float
    r_squared: float


@dataclass(frozen=True, slots=True)
class DreamSavedRunRecord:
    run_dir: Path
    display_label: str


@dataclass(frozen=True, slots=True)
class DreamFitConstraintComparison:
    prefit_metrics: FitQualityMetrics
    dream_metrics: FitQualityMetrics
    fixed_parameters: tuple[str, ...]
    fixed_non_weight_parameters: tuple[str, ...]
    fixed_weight_parameters: tuple[str, ...]


@dataclass(slots=True)
class ProjectLoadPrefitPayload:
    workflow: SAXSPrefitWorkflow | None
    evaluation: PrefitEvaluation | None
    scale_recommendation: PrefitScaleRecommendation | None = None
    workflow_error: str | None = None
    preview_error: str | None = None


@dataclass(slots=True)
class ProjectLoadDreamPayload:
    workflow: SAXSDreamWorkflow | None
    preset_names: list[str]
    selected_preset: str | None
    settings: DreamRunSettings | None
    parameter_map_entries: list[DreamParameterEntry]
    error: str | None = None


@dataclass(slots=True)
class ProjectLoadPayload:
    settings: ProjectSettings
    warnings: tuple[str, ...]
    prefit: ProjectLoadPrefitPayload
    dream: ProjectLoadDreamPayload


@dataclass(frozen=True, slots=True)
class TemplateInstallRequest:
    model_name: str
    template_path: Path
    model_description: str


class MainUISettingsDialog(QDialog):
    def __init__(
        self,
        *,
        dream_settings: DreamRunSettings,
        powerpoint_settings: PowerPointExportSettings,
        powerpoint_enabled: bool,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._powerpoint_enabled = bool(powerpoint_enabled)
        self.setWindowTitle("Main UI Settings")
        self.resize(760, 780)
        layout = QVBoxLayout(self)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        self._build_dream_output_tab(dream_settings)
        self._build_powerpoint_tab(powerpoint_settings)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _build_dream_output_tab(self, settings: DreamRunSettings) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        form_layout = QFormLayout()

        self.verbose_checkbox = QCheckBox("Verbose sampler output")
        self.verbose_checkbox.setChecked(settings.verbose)
        self.verbose_checkbox.setToolTip(
            "Enable or disable verbose DREAM sampler progress output."
        )
        self.interval_spin = QDoubleSpinBox()
        self.interval_spin.setRange(0.1, 30.0)
        self.interval_spin.setDecimals(1)
        self.interval_spin.setSingleStep(0.1)
        self.interval_spin.setValue(settings.verbose_output_interval_seconds)
        self.interval_spin.setToolTip(
            "Minimum number of seconds between DREAM runtime output "
            "updates shown in the UI while verbose output is enabled."
        )
        self.interval_spin.setEnabled(self.verbose_checkbox.isChecked())
        self.verbose_checkbox.toggled.connect(self.interval_spin.setEnabled)
        form_layout.addRow(self.verbose_checkbox)
        form_layout.addRow("Output interval (s)", self.interval_spin)
        layout.addLayout(form_layout)
        layout.addStretch(1)
        self.tabs.addTab(tab, "DREAM Output")

    def _build_powerpoint_tab(
        self,
        settings: PowerPointExportSettings,
    ) -> None:
        tab = QWidget()
        tab_layout = QVBoxLayout(tab)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        tab_layout.addWidget(scroll_area)

        container = QWidget()
        scroll_area.setWidget(container)
        layout = QVBoxLayout(container)

        note_label = QLabel(
            "Adjust the PowerPoint export styling, slide content, and "
            "supplemental files generated with the report."
        )
        note_label.setWordWrap(True)
        layout.addWidget(note_label)
        self._powerpoint_disabled_label = QLabel(
            "Load a project to edit PowerPoint export settings."
        )
        self._powerpoint_disabled_label.setWordWrap(True)
        self._powerpoint_disabled_label.setVisible(
            not self._powerpoint_enabled
        )
        layout.addWidget(self._powerpoint_disabled_label)

        appearance_group = QGroupBox("Appearance")
        appearance_layout = QFormLayout(appearance_group)
        self.powerpoint_font_combo = QFontComboBox()
        self.powerpoint_font_combo.setCurrentFont(QFont(settings.font_family))
        appearance_layout.addRow("Font family", self.powerpoint_font_combo)

        self.component_cmap_combo = QComboBox()
        self._populate_color_map_combo(
            self.component_cmap_combo,
            settings.component_color_map,
        )
        appearance_layout.addRow(
            "Component trace palette",
            self.component_cmap_combo,
        )

        self.prior_cmap_combo = QComboBox()
        self._populate_color_map_combo(
            self.prior_cmap_combo,
            settings.prior_histogram_color_map,
        )
        appearance_layout.addRow(
            "Prior histogram palette",
            self.prior_cmap_combo,
        )

        self.solvent_sort_prior_cmap_combo = QComboBox()
        self._populate_color_map_combo(
            self.solvent_sort_prior_cmap_combo,
            settings.solvent_sort_histogram_color_map,
        )
        appearance_layout.addRow(
            "Solvent-sort histogram palette",
            self.solvent_sort_prior_cmap_combo,
        )
        layout.addWidget(appearance_group)

        colors_group = QGroupBox("Colors")
        colors_layout = QGridLayout(colors_group)
        colors_layout.setColumnStretch(1, 1)
        color_rows = [
            ("Text", settings.text_color, "powerpoint_text_color_button"),
            (
                "Experimental trace",
                settings.experimental_trace_color,
                "powerpoint_experimental_color_button",
            ),
            (
                "Model trace",
                settings.model_trace_color,
                "powerpoint_model_color_button",
            ),
            (
                "Residual trace",
                settings.residual_trace_color,
                "powerpoint_residual_color_button",
            ),
            (
                "Solvent trace",
                settings.solvent_trace_color,
                "powerpoint_solvent_color_button",
            ),
            (
                "Structure factor trace",
                settings.structure_factor_color,
                "powerpoint_structure_factor_color_button",
            ),
            (
                "Table header fill",
                settings.table_header_fill,
                "powerpoint_table_header_fill_button",
            ),
            (
                "Table even row fill",
                settings.table_even_row_fill,
                "powerpoint_table_even_fill_button",
            ),
            (
                "Table odd row fill",
                settings.table_odd_row_fill,
                "powerpoint_table_odd_fill_button",
            ),
            (
                "Table header rule",
                settings.table_rule_color,
                "powerpoint_table_rule_color_button",
            ),
        ]
        for row_index, (label, color, attribute_name) in enumerate(color_rows):
            color_label = QLabel(label)
            button = QPushButton()
            self._configure_color_button(button, color=color, label=label)
            button.clicked.connect(
                lambda checked=False, btn=button, title=label: self._choose_color(
                    btn,
                    title,
                )
            )
            setattr(self, attribute_name, button)
            colors_layout.addWidget(color_label, row_index, 0)
            colors_layout.addWidget(button, row_index, 1)
        layout.addWidget(colors_group)

        content_group = QGroupBox("Slides and Summary Content")
        content_layout = QVBoxLayout(content_group)
        self.include_prior_histograms_checkbox = QCheckBox(
            "Include prior histogram slides"
        )
        self.include_prior_histograms_checkbox.setChecked(
            settings.include_prior_histograms
        )
        content_layout.addWidget(self.include_prior_histograms_checkbox)

        self.include_initial_traces_checkbox = QCheckBox(
            "Include initial SAXS traces slide"
        )
        self.include_initial_traces_checkbox.setChecked(
            settings.include_initial_traces
        )
        content_layout.addWidget(self.include_initial_traces_checkbox)

        self.include_prefit_model_checkbox = QCheckBox(
            "Include prefit model slide"
        )
        self.include_prefit_model_checkbox.setChecked(
            settings.include_prefit_model
        )
        content_layout.addWidget(self.include_prefit_model_checkbox)

        self.include_prefit_parameters_checkbox = QCheckBox(
            "Include prefit parameter table slides"
        )
        self.include_prefit_parameters_checkbox.setChecked(
            settings.include_prefit_parameters
        )
        content_layout.addWidget(self.include_prefit_parameters_checkbox)

        self.include_geometry_table_checkbox = QCheckBox(
            "Include computed geometry parameter slides"
        )
        self.include_geometry_table_checkbox.setChecked(
            settings.include_geometry_table
        )
        content_layout.addWidget(self.include_geometry_table_checkbox)

        self.include_estimator_metrics_checkbox = QCheckBox(
            "Include estimator metrics slides"
        )
        self.include_estimator_metrics_checkbox.setChecked(
            settings.include_estimator_metrics
        )
        content_layout.addWidget(self.include_estimator_metrics_checkbox)

        self.include_dream_settings_checkbox = QCheckBox(
            "Include DREAM settings and assessment slides"
        )
        self.include_dream_settings_checkbox.setChecked(
            settings.include_dream_settings
        )
        content_layout.addWidget(self.include_dream_settings_checkbox)

        self.include_dream_prior_table_checkbox = QCheckBox(
            "Include DREAM prior distribution slides"
        )
        self.include_dream_prior_table_checkbox.setChecked(
            settings.include_dream_prior_table
        )
        content_layout.addWidget(self.include_dream_prior_table_checkbox)

        self.include_dream_output_model_checkbox = QCheckBox(
            "Include DREAM output model slides"
        )
        self.include_dream_output_model_checkbox.setChecked(
            settings.include_dream_output_model
        )
        content_layout.addWidget(self.include_dream_output_model_checkbox)

        self.include_posterior_comparisons_checkbox = QCheckBox(
            "Include posterior comparison plots"
        )
        self.include_posterior_comparisons_checkbox.setChecked(
            settings.include_posterior_comparisons
        )
        content_layout.addWidget(self.include_posterior_comparisons_checkbox)

        self.include_output_summary_checkbox = QCheckBox(
            "Include output summary text"
        )
        self.include_output_summary_checkbox.setChecked(
            settings.include_output_summary
        )
        content_layout.addWidget(self.include_output_summary_checkbox)

        self.include_directory_summary_checkbox = QCheckBox(
            "Include directory summary text"
        )
        self.include_directory_summary_checkbox.setChecked(
            settings.include_directory_summary
        )
        content_layout.addWidget(self.include_directory_summary_checkbox)
        layout.addWidget(content_group)

        output_group = QGroupBox("Supplemental Output Data")
        output_layout = QVBoxLayout(output_group)
        self.generate_manifest_checkbox = QCheckBox(
            "Generate report manifest JSON"
        )
        self.generate_manifest_checkbox.setChecked(settings.generate_manifest)
        output_layout.addWidget(self.generate_manifest_checkbox)

        self.export_figure_assets_checkbox = QCheckBox(
            "Keep rendered figure PNG assets"
        )
        self.export_figure_assets_checkbox.setChecked(
            settings.export_figure_assets
        )
        output_layout.addWidget(self.export_figure_assets_checkbox)
        layout.addWidget(output_group)

        self._powerpoint_groups = (
            appearance_group,
            colors_group,
            content_group,
            output_group,
        )
        for group in self._powerpoint_groups:
            group.setEnabled(self._powerpoint_enabled)
        layout.addStretch(1)
        self.tabs.addTab(tab, "PowerPoint Export")

    @staticmethod
    def _populate_color_map_combo(
        combo: QComboBox,
        selected_name: str,
    ) -> None:
        for name in POWERPOINT_COLOR_MAP_OPTIONS:
            combo.addItem(name, name)
        current_name = str(selected_name).strip() or "viridis"
        index = combo.findData(current_name)
        combo.setCurrentIndex(index if index >= 0 else 0)

    @staticmethod
    def _configure_color_button(
        button: QPushButton,
        *,
        color: str,
        label: str,
    ) -> None:
        normalized = str(color).strip() or "#000000"
        qcolor = QColor(normalized)
        foreground = "#ffffff"
        if qcolor.isValid() and qcolor.lightness() > 128:
            foreground = "#000000"
        button.setText(normalized.upper())
        button.setToolTip(f"{label}: {normalized.upper()}")
        button.setMinimumWidth(120)
        button.setStyleSheet(
            "QPushButton {"
            f"background-color: {normalized};"
            f"color: {foreground};"
            "border: 1px solid #666666;"
            "padding: 4px 8px;"
            "}"
        )

    def _choose_color(self, button: QPushButton, label: str) -> None:
        chosen = QColorDialog.getColor(
            QColor(button.text().strip()),
            self,
            f"Choose {label.lower()}",
        )
        if not chosen.isValid():
            return
        self._configure_color_button(
            button,
            color=chosen.name(),
            label=label,
        )

    def dream_output_values(self) -> tuple[bool, float]:
        return (
            bool(self.verbose_checkbox.isChecked()),
            float(self.interval_spin.value()),
        )

    def powerpoint_settings_value(self) -> PowerPointExportSettings:
        return PowerPointExportSettings(
            font_family=(
                self.powerpoint_font_combo.currentFont().family().strip()
                or "Arial"
            ),
            component_color_map=str(
                self.component_cmap_combo.currentData() or "viridis"
            ),
            prior_histogram_color_map=str(
                self.prior_cmap_combo.currentData() or "viridis"
            ),
            solvent_sort_histogram_color_map=str(
                self.solvent_sort_prior_cmap_combo.currentData() or "summer"
            ),
            text_color=self.powerpoint_text_color_button.text().strip(),
            experimental_trace_color=(
                self.powerpoint_experimental_color_button.text().strip()
            ),
            model_trace_color=(
                self.powerpoint_model_color_button.text().strip()
            ),
            residual_trace_color=(
                self.powerpoint_residual_color_button.text().strip()
            ),
            solvent_trace_color=(
                self.powerpoint_solvent_color_button.text().strip()
            ),
            structure_factor_color=(
                self.powerpoint_structure_factor_color_button.text().strip()
            ),
            table_header_fill=(
                self.powerpoint_table_header_fill_button.text().strip()
            ),
            table_even_row_fill=(
                self.powerpoint_table_even_fill_button.text().strip()
            ),
            table_odd_row_fill=(
                self.powerpoint_table_odd_fill_button.text().strip()
            ),
            table_rule_color=(
                self.powerpoint_table_rule_color_button.text().strip()
            ),
            include_prior_histograms=bool(
                self.include_prior_histograms_checkbox.isChecked()
            ),
            include_initial_traces=bool(
                self.include_initial_traces_checkbox.isChecked()
            ),
            include_prefit_model=bool(
                self.include_prefit_model_checkbox.isChecked()
            ),
            include_prefit_parameters=bool(
                self.include_prefit_parameters_checkbox.isChecked()
            ),
            include_geometry_table=bool(
                self.include_geometry_table_checkbox.isChecked()
            ),
            include_estimator_metrics=bool(
                self.include_estimator_metrics_checkbox.isChecked()
            ),
            include_dream_settings=bool(
                self.include_dream_settings_checkbox.isChecked()
            ),
            include_dream_prior_table=bool(
                self.include_dream_prior_table_checkbox.isChecked()
            ),
            include_dream_output_model=bool(
                self.include_dream_output_model_checkbox.isChecked()
            ),
            include_posterior_comparisons=bool(
                self.include_posterior_comparisons_checkbox.isChecked()
            ),
            include_output_summary=bool(
                self.include_output_summary_checkbox.isChecked()
            ),
            include_directory_summary=bool(
                self.include_directory_summary_checkbox.isChecked()
            ),
            generate_manifest=bool(
                self.generate_manifest_checkbox.isChecked()
            ),
            export_figure_assets=bool(
                self.export_figure_assets_checkbox.isChecked()
            ),
        )


class DreamSavedRunPreviewDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setModal(False)
        self.setWindowTitle("Saved DREAM Run Preview")
        self.resize(960, 780)

        layout = QVBoxLayout(self)

        description = QLabel(
            "Review the selected saved DREAM run's settings and prior "
            "parameter map before loading its analysis results."
        )
        description.setWordWrap(True)
        layout.addWidget(description)

        self.summary_group = QGroupBox("Saved Run Settings")
        summary_layout = QVBoxLayout(self.summary_group)
        self.summary_box = QTextEdit()
        self.summary_box.setReadOnly(True)
        self.summary_box.setMinimumHeight(280)
        summary_layout.addWidget(self.summary_box)
        layout.addWidget(self.summary_group)

        self.parameter_map_group = QGroupBox("Saved Prior Parameter Map")
        parameter_map_layout = QVBoxLayout(self.parameter_map_group)
        self.parameter_map_status_label = QLabel()
        self.parameter_map_status_label.setWordWrap(True)
        parameter_map_layout.addWidget(self.parameter_map_status_label)
        self.parameter_map_table = QTableWidget(0, 8)
        self.parameter_map_table.setHorizontalHeaderLabels(
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
        self.parameter_map_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.parameter_map_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.parameter_map_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.parameter_map_table.setMinimumHeight(320)
        header = self.parameter_map_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setStretchLastSection(True)
        parameter_map_layout.addWidget(self.parameter_map_table)
        layout.addWidget(self.parameter_map_group, stretch=1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.close)
        layout.addWidget(buttons)

    def set_preview(
        self,
        *,
        display_label: str | None,
        summary_text: str,
        parameter_map_entries: list[DreamParameterEntry],
    ) -> None:
        title = "Saved DREAM Run Preview"
        if display_label:
            title = f"{title}: {display_label}"
        self.setWindowTitle(title)
        self.summary_box.setPlainText(summary_text)
        self._set_parameter_map_entries(parameter_map_entries)

    def _set_parameter_map_entries(
        self,
        entries: list[DreamParameterEntry],
    ) -> None:
        varying_count = sum(1 for entry in entries if entry.vary)
        fixed_count = len(entries) - varying_count
        if entries:
            self.parameter_map_status_label.setText(
                "Entries: "
                f"{len(entries)} total, {varying_count} varying, "
                f"{fixed_count} fixed."
            )
        else:
            self.parameter_map_status_label.setText(
                "No saved prior parameter-map entries were found for this run."
            )
        self.parameter_map_table.setRowCount(len(entries))
        for row, entry in enumerate(entries):
            self.parameter_map_table.setItem(
                row,
                0,
                QTableWidgetItem(entry.structure),
            )
            self.parameter_map_table.setItem(
                row,
                1,
                QTableWidgetItem(entry.motif),
            )
            self.parameter_map_table.setItem(
                row,
                2,
                QTableWidgetItem(entry.param_type),
            )
            self.parameter_map_table.setItem(
                row,
                3,
                QTableWidgetItem(entry.param),
            )
            self.parameter_map_table.setItem(
                row,
                4,
                QTableWidgetItem(f"{entry.value:.6g}"),
            )
            self.parameter_map_table.setItem(
                row,
                5,
                QTableWidgetItem("Yes" if entry.vary else "No"),
            )
            self.parameter_map_table.setItem(
                row,
                6,
                QTableWidgetItem(entry.distribution),
            )
            self.parameter_map_table.setItem(
                row,
                7,
                QTableWidgetItem(
                    self._format_distribution_params(entry.dist_params)
                ),
            )
        if entries:
            self.parameter_map_table.resizeRowsToContents()

    @staticmethod
    def _format_distribution_params(params: dict[str, float]) -> str:
        if not params:
            return ""
        return ", ".join(
            f"{key}={value:.6g}" for key, value in sorted(params.items())
        )


class InstallModelDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Install Model")
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        form_layout = QFormLayout()

        self.model_name_edit = QLineEdit()
        self.model_name_edit.setPlaceholderText("Example: Custom Poly LMA")
        form_layout.addRow("Model name", self.model_name_edit)

        self.template_path_edit = QLineEdit()
        self.template_path_edit.setPlaceholderText(
            "Select a Python template file"
        )
        browse_button = QPushButton("Browse…")
        browse_button.clicked.connect(self._choose_template_file)
        path_row = QWidget()
        path_layout = QHBoxLayout(path_row)
        path_layout.setContentsMargins(0, 0, 0, 0)
        path_layout.addWidget(self.template_path_edit, stretch=1)
        path_layout.addWidget(browse_button)
        form_layout.addRow("Template .py", path_row)

        self.description_edit = QTextEdit()
        self.description_edit.setMinimumHeight(140)
        self.description_edit.setPlaceholderText(
            "Describe the model shown in the template selector tooltip."
        )
        form_layout.addRow("Description", self.description_edit)

        layout.addLayout(form_layout)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _choose_template_file(self) -> None:
        selected, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "Select template Python file",
            str(Path.home()),
            "Python files (*.py)",
        )
        if selected:
            self.template_path_edit.setText(selected)

    def selected_request(self) -> TemplateInstallRequest:
        model_name = self.model_name_edit.text().strip()
        if not model_name:
            raise ValueError("Enter a model name before installing.")
        template_path_text = self.template_path_edit.text().strip()
        if not template_path_text:
            raise ValueError("Select a template Python file to install.")
        template_path = Path(template_path_text).expanduser().resolve()
        if not template_path.is_file():
            raise FileNotFoundError(
                f"Template file not found: {template_path}"
            )
        model_description = self.description_edit.toPlainText().strip()
        if not model_description:
            raise ValueError("Enter a model description before installing.")
        return TemplateInstallRequest(
            model_name=model_name,
            template_path=template_path,
            model_description=model_description,
        )


class SAXSProjectTaskWorker(QObject):
    """Background worker for cluster import and project build tasks."""

    progress = Signal(int, int, str)
    finished = Signal(str, object)
    failed = Signal(str, str)

    def __init__(
        self,
        task_name: str,
        task_fn: Callable[[Callable[[int, int, str], None]], object],
    ) -> None:
        super().__init__()
        self.task_name = task_name
        self.task_fn = task_fn

    @Slot()
    def run(self) -> None:
        try:
            result = self.task_fn(self._emit_progress)
        except Exception as exc:
            self.failed.emit(self.task_name, str(exc))
            return
        self.finished.emit(self.task_name, result)

    def _emit_progress(self, processed: int, total: int, message: str) -> None:
        self.progress.emit(int(processed), int(total), str(message))


class SAXSDreamRunWorker(QObject):
    """Background worker for DREAM runtime execution."""

    status = Signal(str)
    output = Signal(str)
    finished = Signal(str, object)
    failed = Signal(str)

    def __init__(
        self,
        project_dir: str | Path,
        bundle: DreamRunBundle,
        *,
        verbose_output_interval_seconds: float = 5.0,
    ) -> None:
        super().__init__()
        self.project_dir = str(Path(project_dir).expanduser().resolve())
        self.bundle = bundle
        self.verbose_output_interval_seconds = max(
            float(verbose_output_interval_seconds),
            0.1,
        )

    @Slot()
    def run(self) -> None:
        try:
            self.status.emit("Executing DREAM runtime bundle...")
            workflow = SAXSDreamWorkflow(self.project_dir)
            result = workflow.run_bundle(
                self.bundle,
                output_callback=self.output.emit,
                output_interval_seconds=self.verbose_output_interval_seconds,
            )
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        self.finished.emit(str(self.bundle.run_dir), result)


class SAXSMainWindow(QMainWindow):
    """Main Qt window for SAXS project setup, prefit, and DREAM
    refinement."""

    DREAM_REFRESH_DELAY_MS = 75
    DREAM_REFRESH_STYLE = 1
    DREAM_REFRESH_VIOLIN = 2
    DREAM_REFRESH_SUMMARY = 3
    DREAM_REFRESH_FULL = 4

    def __init__(
        self,
        initial_project_dir: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.project_manager = SAXSProjectManager()
        self.current_settings: ProjectSettings | None = None
        self.prefit_workflow: SAXSPrefitWorkflow | None = None
        self.dream_workflow: SAXSDreamWorkflow | None = None
        self.distribution_window: DistributionSetupWindow | None = None
        self._dream_saved_run_preview_dialog: (
            DreamSavedRunPreviewDialog | None
        ) = None
        self._last_results_loader: SAXSDreamResultsLoader | None = None
        self._last_written_dream_bundle: DreamRunBundle | None = None
        self._dream_workflow_project_dir: str | None = None
        self._dream_parameter_map_saved_in_session = False
        self._active_dream_settings_snapshot: DreamRunSettings | None = None
        self._applied_dream_analysis_settings: DreamRunSettings | None = None
        self._last_dream_summary = None
        self._current_dream_preset_name: str | None = None
        self._last_dream_filter_assessments: list[dict[str, object]] = []
        self._last_dream_filter_recommendation: dict[str, object] | None = None
        self._last_dream_constraint_warning_signature: (
            tuple[object, ...] | None
        ) = None
        self._prior_histogram_windows: list[PriorHistogramWindow] = []
        self._task_thread: QThread | None = None
        self._task_worker: SAXSProjectTaskWorker | None = None
        self._progress_dialog: SAXSProgressDialog | None = None
        self._active_task_name: str | None = None
        self._active_task_settings: ProjectSettings | None = None
        self._dream_task_thread: QThread | None = None
        self._dream_task_worker: SAXSDreamRunWorker | None = None
        self._dream_progress_dialog: SAXSProgressDialog | None = None
        self._active_dream_run_settings: DreamRunSettings | None = None
        self._loaded_dream_run_dir: Path | None = None
        self._warn_on_prefit_template_change = True
        self._warn_on_large_prefit_parameter_count = True
        self._prefit_sequence_pending_manual_edits = False
        self._prefit_sequence_baseline_entries: list[PrefitParameterEntry] = []
        self._syncing_template_controls = False
        self._prefit_missing_components_warning_shown = False
        self._updating_deprecated_template_visibility = False
        self._show_deprecated_templates = False
        self._console_autoscroll_enabled = (
            self._load_console_autoscroll_setting()
        )
        self._ui_scale = 1.0
        self._base_font_point_size = self._resolve_base_font_point_size()
        self._scale_shortcuts: list[QShortcut] = []
        self._child_tool_windows: list[object] = []
        self._contrast_mode_tool_window: object | None = None
        self._solute_volume_fraction_tool_window: (
            SoluteVolumeFractionToolWindow | None
        ) = None
        self._number_density_tool_window: (
            NumberDensityEstimateToolWindow | None
        ) = None
        self._attenuation_tool_window: AttenuationEstimateToolWindow | None = (
            None
        )
        self._fluorescence_tool_window: (
            FluorescenceEstimateToolWindow | None
        ) = None
        self._last_solution_scattering_estimate: (
            SolutionScatteringEstimate | None
        ) = None
        self._pending_prefit_sf_approximation_change: (
            tuple[str, str] | None
        ) = None
        self._prefit_cluster_geometry_sync_rows_snapshot: (
            list[dict[str, object]] | None
        ) = None
        self._prefit_cluster_geometry_sync_radii_type: str | None = None
        self._prefit_cluster_geometry_sync_ionic_radius_type: str | None = None
        self._pending_dream_refresh_scope = 0
        self._dream_refresh_timer = QTimer(self)
        self._dream_refresh_timer.setSingleShot(True)
        self._dream_refresh_timer.setInterval(self.DREAM_REFRESH_DELAY_MS)
        self._dream_refresh_timer.timeout.connect(
            self._flush_pending_dream_refresh
        )
        self._build_ui()
        self._capture_scale_baselines(self)
        self._register_scale_shortcuts()
        self._apply_ui_scale(announce=False)
        self._refresh_template_selectors()
        self.dream_tab.set_available_settings_presets([], None)
        if initial_project_dir is not None:
            self.load_project(initial_project_dir)
        else:
            self.project_setup_tab.set_project_selected(False)
            self.project_setup_tab.draw_component_plot(None)
            self.project_setup_tab.draw_prior_plot(None)
            self.prefit_tab.plot_evaluation(None)
            self.dream_tab.clear_plots()

    def _build_ui(self) -> None:
        self.setWindowTitle("SAXSShell")
        self.setWindowIcon(load_saxshell_icon())
        self.resize(self._default_window_size())

        self._build_menu_bar()
        self.tabs = QTabWidget()
        self.tabs.setCornerWidget(
            build_saxshell_brand_widget(self.tabs),
            Qt.Corner.TopLeftCorner,
        )
        self.project_setup_tab = ProjectSetupTab()
        self.prefit_tab = PrefitTab()
        self.dream_tab = DreamTab()
        self.project_setup_tab.set_console_autoscroll_enabled(
            self._console_autoscroll_enabled
        )
        self.prefit_tab.set_console_autoscroll_enabled(
            self._console_autoscroll_enabled
        )
        self.dream_tab.set_console_autoscroll_enabled(
            self._console_autoscroll_enabled
        )
        self.tabs.addTab(self.project_setup_tab, "Project Setup")
        self.tabs.addTab(self.prefit_tab, "SAXS Prefit")
        self.tabs.addTab(self.dream_tab, "SAXS DREAM Fit")
        self.setCentralWidget(self.tabs)
        self.statusBar().showMessage("Ready")

        self.project_setup_tab.create_project_requested.connect(
            self.create_project_from_tab
        )
        self.project_setup_tab.open_project_requested.connect(
            self.open_project_from_dialog
        )
        self.project_setup_tab.autosave_project_requested.connect(
            self._autosave_project_from_tab
        )
        self.project_setup_tab.open_mdtrajectory_requested.connect(
            self._open_mdtrajectory_tool
        )
        self.project_setup_tab.open_xyz2pdb_requested.connect(
            self._open_xyz2pdb_tool
        )
        self.project_setup_tab.open_cluster_requested.connect(
            self._open_cluster_tool
        )
        self.project_setup_tab.model_only_mode_changed.connect(
            self._on_model_only_mode_changed
        )
        self.project_setup_tab.predicted_structure_weights_changed.connect(
            self._on_predicted_structure_weights_changed
        )
        self.project_setup_tab.load_distribution_requested.connect(
            self._load_saved_distribution
        )
        self.project_setup_tab.view_active_contrast_distribution_requested.connect(
            self._open_active_contrast_distribution_view
        )
        self.project_setup_tab.scan_clusters_requested.connect(
            self.scan_clusters_from_tab
        )
        self.project_setup_tab.build_components_requested.connect(
            self.build_project_components
        )
        self.project_setup_tab.build_prior_weights_requested.connect(
            self.build_prior_weights
        )
        self.project_setup_tab.install_model_requested.connect(
            self.install_model_template
        )
        self.project_setup_tab.template_selection_changed.connect(
            self._on_project_setup_template_selected
        )
        self.project_setup_tab.change_template_requested.connect(
            self._on_change_template_requested
        )
        self.project_setup_tab.show_deprecated_templates_changed.connect(
            self._on_show_deprecated_templates_changed
        )
        self.project_setup_tab.prior_mode_combo.currentTextChanged.connect(
            lambda _text: self._refresh_prior_plot()
        )
        self.project_setup_tab.generate_prior_plot_requested.connect(
            self.show_prior_histogram_window
        )
        self.project_setup_tab.save_component_plot_data_requested.connect(
            self.save_component_plot_data
        )
        self.project_setup_tab.save_prior_plot_data_requested.connect(
            self.save_prior_plot_data
        )
        self.project_setup_tab.save_prior_png_requested.connect(
            self.save_prior_plot_png
        )

        self.prefit_tab.template_changed.connect(
            self._on_prefit_template_changed
        )
        self.prefit_tab.change_template_requested.connect(
            self._on_change_template_requested
        )
        self.prefit_tab.show_deprecated_templates_changed.connect(
            self._on_show_deprecated_templates_changed
        )
        self.prefit_tab.field_interaction_requested.connect(
            self._on_prefit_field_interaction_requested
        )
        self.prefit_tab.autosave_toggled.connect(self._on_autosave_changed)
        self.prefit_tab.sequence_history_toggled.connect(
            self._on_prefit_sequence_history_changed
        )
        self.prefit_tab.parameter_table_edited.connect(
            self._on_prefit_parameter_table_edited
        )
        self.prefit_tab.update_model_requested.connect(
            self.update_prefit_model
        )
        self.prefit_tab.run_fit_requested.connect(self.run_prefit)
        self.prefit_tab.apply_recommended_scale_requested.connect(
            self.apply_recommended_scale_settings
        )
        self.prefit_tab.save_plot_data_requested.connect(
            self.save_prefit_plot_data
        )
        self.prefit_tab.set_best_prefit_requested.connect(
            self.set_best_prefit_parameters
        )
        self.prefit_tab.reset_best_prefit_requested.connect(
            self.reset_parameters_to_best_prefit
        )
        self.prefit_tab.save_fit_requested.connect(self.save_prefit)
        self.prefit_tab.restore_state_requested.connect(
            self.restore_prefit_state
        )
        self.prefit_tab.reset_requested.connect(self.reset_prefit_entries)
        self.prefit_tab.parameter_reset_requested.connect(
            self.reset_single_prefit_parameter
        )
        self.prefit_tab.compute_cluster_geometry_requested.connect(
            self.compute_prefit_cluster_geometry
        )
        self.prefit_tab.update_cluster_geometry_requested.connect(
            self.update_prefit_cluster_geometry
        )
        self.prefit_tab.cluster_geometry_mapping_changed.connect(
            self._on_prefit_cluster_geometry_changed
        )
        self.prefit_tab.cluster_geometry_sf_approximation_changed.connect(
            self._on_prefit_cluster_geometry_sf_approximation_changed
        )
        self.prefit_tab.cluster_geometry_radii_type_changed.connect(
            self._on_prefit_cluster_geometry_radii_type_changed
        )
        self.prefit_tab.cluster_geometry_ionic_radius_type_changed.connect(
            self._on_prefit_cluster_geometry_ionic_radius_type_changed
        )
        self.prefit_tab.solute_volume_fraction_widget.estimate_calculated.connect(
            self._on_solution_scattering_estimate_calculated
        )

        self.dream_tab.edit_parameter_map_requested.connect(
            self.open_distribution_editor
        )
        self.dream_tab.save_settings_requested.connect(
            self.save_dream_settings
        )
        self.dream_tab.write_runtime_requested.connect(self.write_dream_bundle)
        self.dream_tab.preview_runtime_requested.connect(
            self.preview_dream_runtime_bundle
        )
        self.dream_tab.run_dream_requested.connect(self.run_dream_bundle)
        self.dream_tab.load_results_requested.connect(
            self.load_selected_results
        )
        self.dream_tab.preview_saved_run_requested.connect(
            self.preview_selected_dream_run
        )
        self.dream_tab.save_report_requested.connect(self.save_dream_report)
        self.dream_tab.recycle_output_requested.connect(
            self.recycle_dream_output_to_prefit
        )
        self.dream_tab.export_model_report_requested.connect(
            self.export_dream_model_report
        )
        self.dream_tab.save_model_fit_requested.connect(
            self.save_dream_model_fit
        )
        self.dream_tab.save_violin_data_requested.connect(
            self.save_dream_violin_data
        )
        self.dream_tab.settings_preset_changed.connect(
            self._on_dream_settings_preset_changed
        )
        self.dream_tab.apply_filter_requested.connect(
            self._apply_dream_filter_settings
        )
        self.dream_tab.visualization_settings_changed.connect(
            self._on_dream_analysis_settings_changed
        )
        self.dream_tab.results_settings_changed.connect(
            self._on_dream_analysis_settings_changed
        )
        self.dream_tab.summary_settings_changed.connect(
            self._on_dream_analysis_settings_changed
        )
        self.dream_tab.violin_data_settings_changed.connect(
            self._on_dream_analysis_settings_changed
        )
        self.dream_tab.violin_style_settings_changed.connect(
            self._on_dream_analysis_settings_changed
        )
        self._refresh_recent_projects_menu()
        self._update_file_menu_state()

    def _build_menu_bar(self) -> None:
        menu_bar = self.menuBar()

        self.file_menu = menu_bar.addMenu("File")
        self.create_project_action = QAction("Create Project", self)
        self.create_project_action.triggered.connect(
            self._create_project_from_menu
        )
        self.file_menu.addAction(self.create_project_action)

        self.open_project_action = QAction("Open Existing Project...", self)
        self.open_project_action.triggered.connect(
            self._open_project_from_menu
        )
        self.file_menu.addAction(self.open_project_action)

        self.open_recent_menu = self.file_menu.addMenu("Open Recent Project")

        self.save_project_action = QAction("Save Project", self)
        save_shortcuts = QKeySequence.keyBindings(
            QKeySequence.StandardKey.Save
        )
        if save_shortcuts:
            self.save_project_action.setShortcuts(save_shortcuts)
        self.save_project_action.triggered.connect(self.save_project_state)
        self.file_menu.addAction(self.save_project_action)

        self.save_project_as_action = QAction("Save Project As...", self)
        self.save_project_as_action.triggered.connect(self.save_project_as)
        self.file_menu.addAction(self.save_project_as_action)

        self.tools_menu = menu_bar.addMenu("Tools")
        self.mdtrajectory_action = QAction(
            "Open MD Trajectory Extraction", self
        )
        self.mdtrajectory_action.triggered.connect(
            self._open_mdtrajectory_tool
        )
        self.tools_menu.addAction(self.mdtrajectory_action)

        self.xyz2pdb_action = QAction("Open XYZ -> PDB Conversion", self)
        self.xyz2pdb_action.triggered.connect(self._open_xyz2pdb_tool)
        self.tools_menu.addAction(self.xyz2pdb_action)

        self.cluster_action = QAction("Open Cluster Extraction", self)
        self.cluster_action.triggered.connect(self._open_cluster_tool)
        self.tools_menu.addAction(self.cluster_action)

        self.bondanalysis_action = QAction("Open Bond Analysis", self)
        self.bondanalysis_action.triggered.connect(
            self._open_bondanalysis_tool
        )
        self.tools_menu.addAction(self.bondanalysis_action)

        self.cluster_dynamics_menu = self.tools_menu.addMenu(
            "Cluster Dynamics"
        )
        self.clusterdynamics_action = QAction(
            "Open Cluster Dynamics (only)",
            self,
        )
        self.clusterdynamics_action.triggered.connect(
            self._open_clusterdynamics_tool
        )
        self.cluster_dynamics_menu.addAction(self.clusterdynamics_action)

        self.clusterdynamicsml_action = QAction(
            "Open Cluster Dynamics (ML)",
            self,
        )
        self.clusterdynamicsml_action.triggered.connect(
            self._open_clusterdynamicsml_tool
        )
        self.cluster_dynamics_menu.addAction(self.clusterdynamicsml_action)

        self.pdf_menu = self.tools_menu.addMenu("PDF")
        self.pdfsetup_action = QAction("Open PDF Calculation", self)
        self.pdfsetup_action.triggered.connect(self._open_pdfsetup_tool)
        self.pdf_menu.addAction(self.pdfsetup_action)

        self.fullrmc_action = QAction("Open fullrmc Setup", self)
        self.fullrmc_action.triggered.connect(self._open_fullrmc_tool)
        self.pdf_menu.addAction(self.fullrmc_action)

        self.blenderxyz_action = QAction(
            "Open Blender XYZ Renderer",
            self,
        )
        self.blenderxyz_action.triggered.connect(self._open_blenderxyz_tool)
        self.tools_menu.addAction(self.blenderxyz_action)

        self.contrast_mode_action = QAction(
            "Open SAXS Contrast Mode",
            self,
        )
        self.contrast_mode_action.triggered.connect(
            self._open_contrast_mode_tool
        )
        self.tools_menu.addAction(self.contrast_mode_action)

        self.estimation_menu = self.tools_menu.addMenu("Estimation")
        self.volume_fraction_action = QAction(
            "Open Volume Fraction Estimate", self
        )
        self.volume_fraction_action.triggered.connect(
            self._open_solute_volume_fraction_tool
        )
        self.estimation_menu.addAction(self.volume_fraction_action)
        self.number_density_action = QAction(
            "Open Number Density Estimate", self
        )
        self.number_density_action.triggered.connect(
            self._open_number_density_tool
        )
        self.estimation_menu.addAction(self.number_density_action)
        self.attenuation_estimate_action = QAction(
            "Open Attenuation Estimate",
            self,
        )
        self.attenuation_estimate_action.triggered.connect(
            self._open_attenuation_tool
        )
        self.estimation_menu.addAction(self.attenuation_estimate_action)
        self.fluorescence_estimate_action = QAction(
            "Open Fluorescence Estimate",
            self,
        )
        self.fluorescence_estimate_action.triggered.connect(
            self._open_fluorescence_tool
        )
        self.estimation_menu.addAction(self.fluorescence_estimate_action)
        self.settings_menu = menu_bar.addMenu("Settings")
        self.console_autoscroll_action = QAction(
            "Autoscroll Console Output",
            self,
        )
        self.console_autoscroll_action.setCheckable(True)
        self.console_autoscroll_action.setChecked(
            bool(self._console_autoscroll_enabled)
        )
        self.console_autoscroll_action.triggered.connect(
            self._toggle_console_autoscroll
        )
        self.settings_menu.addAction(self.console_autoscroll_action)
        self.main_ui_settings_action = QAction(
            "Main UI Settings...",
            self,
        )
        self.main_ui_settings_action.triggered.connect(
            self._open_main_ui_settings_dialog
        )
        self.settings_menu.addAction(self.main_ui_settings_action)
        self.dream_output_settings_action = self.main_ui_settings_action
        self.window_presets_menu = self.settings_menu.addMenu("Window Presets")
        self.auto_fit_window_action = QAction(
            "Auto Fit Current Screen",
            self,
        )
        self.auto_fit_window_action.triggered.connect(
            self._apply_auto_window_layout_preset
        )
        self.window_presets_menu.addAction(self.auto_fit_window_action)
        self.window_presets_menu.addSeparator()
        self.window_preset_actions: dict[str, QAction] = {}
        for preset in WINDOW_LAYOUT_PRESETS:
            action = QAction(preset.label, self)
            action.triggered.connect(
                lambda checked=False, preset_key=preset.key: (
                    self._apply_window_layout_preset(preset_key)
                )
            )
            self.window_presets_menu.addAction(action)
            self.window_preset_actions[preset.key] = action

        self.help_menu = menu_bar.addMenu("Help")
        self.version_info_action = QAction("Version Information", self)
        self.version_info_action.triggered.connect(
            self._show_version_information
        )
        self.help_menu.addAction(self.version_info_action)

        self.github_action = QAction("Open GitHub Repository", self)
        self.github_action.triggered.connect(self._open_github_repository)
        self.help_menu.addAction(self.github_action)

        self.contact_action = QAction("Contact Developer", self)
        self.contact_action.triggered.connect(self._show_contact_information)
        self.help_menu.addAction(self.contact_action)

    def _resolve_base_font_point_size(self) -> float:
        font = self.font()
        point_size = font.pointSizeF()
        if point_size <= 0:
            app = QApplication.instance()
            if app is not None:
                point_size = app.font().pointSizeF()
        return point_size if point_size > 0 else 12.0

    def _register_scale_shortcuts(self) -> None:
        shortcut_map = [
            ("Meta+=", lambda: self._adjust_ui_scale(0.1)),
            ("Meta++", lambda: self._adjust_ui_scale(0.1)),
            ("Ctrl+=", lambda: self._adjust_ui_scale(0.1)),
            ("Ctrl++", lambda: self._adjust_ui_scale(0.1)),
            ("Meta+-", lambda: self._adjust_ui_scale(-0.1)),
            ("Ctrl+-", lambda: self._adjust_ui_scale(-0.1)),
            ("Meta+0", self._reset_ui_scale),
            ("Ctrl+0", self._reset_ui_scale),
        ]
        for sequence, handler in shortcut_map:
            shortcut = QShortcut(QKeySequence(sequence), self)
            shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
            shortcut.activated.connect(handler)
            self._scale_shortcuts.append(shortcut)

    def _adjust_ui_scale(self, delta: float) -> None:
        self._set_ui_scale(self._ui_scale + delta)

    def _reset_ui_scale(self) -> None:
        self._set_ui_scale(1.0)

    def _set_ui_scale(self, scale: float) -> None:
        bounded = max(0.7, min(1.6, round(float(scale), 2)))
        if abs(bounded - self._ui_scale) < 1e-9:
            return
        self._ui_scale = bounded
        self._apply_ui_scale(announce=True)

    def _apply_ui_scale(self, *, announce: bool) -> None:
        scaled_font = self.font()
        scaled_font.setPointSizeF(self._base_font_point_size * self._ui_scale)
        self.setFont(scaled_font)
        self._apply_scale_to_widget_tree(self)
        if announce:
            self.statusBar().showMessage(
                f"Interface scale: {int(round(self._ui_scale * 100))}%"
            )

    def _capture_scale_baselines(self, widget: QWidget) -> None:
        if bool(widget.property("_saxs_skip_scale")):
            return
        if widget.property("_saxs_base_min_width") is None:
            widget.setProperty("_saxs_base_min_width", widget.minimumWidth())
            widget.setProperty("_saxs_base_min_height", widget.minimumHeight())
        if isinstance(widget, QSplitter):
            if widget.property("_saxs_base_handle_width") is None:
                widget.setProperty(
                    "_saxs_base_handle_width", widget.handleWidth()
                )
        for child in widget.findChildren(
            QWidget, options=Qt.FindChildOption.FindDirectChildrenOnly
        ):
            self._capture_scale_baselines(child)

    def _apply_scale_to_widget_tree(self, widget: QWidget) -> None:
        if bool(widget.property("_saxs_skip_scale")):
            return
        base_min_width = widget.property("_saxs_base_min_width")
        base_min_height = widget.property("_saxs_base_min_height")
        if isinstance(base_min_width, int) and base_min_width > 0:
            widget.setMinimumWidth(
                max(1, round(base_min_width * self._ui_scale))
            )
        if isinstance(base_min_height, int) and base_min_height > 0:
            widget.setMinimumHeight(
                max(1, round(base_min_height * self._ui_scale))
            )
        if isinstance(widget, QSplitter):
            base_handle_width = widget.property("_saxs_base_handle_width")
            if isinstance(base_handle_width, int) and base_handle_width > 0:
                widget.setHandleWidth(
                    max(2, round(base_handle_width * self._ui_scale))
                )
        for child in widget.findChildren(
            QWidget, options=Qt.FindChildOption.FindDirectChildrenOnly
        ):
            self._apply_scale_to_widget_tree(child)

    def _current_available_geometry(self) -> QRect | None:
        screen = self.screen()
        if screen is None and self.windowHandle() is not None:
            screen = self.windowHandle().screen()
        if screen is None:
            app = QApplication.instance()
            screen = app.primaryScreen() if app is not None else None
        return screen.availableGeometry() if screen is not None else None

    def _fit_window_size_to_current_screen(self, size: QSize) -> QSize:
        available = self._current_available_geometry()
        if available is None:
            return size
        usable_width = max(640, available.width() - 48)
        usable_height = max(520, available.height() - 64)
        return QSize(
            min(size.width(), usable_width),
            min(size.height(), usable_height),
        )

    def _recommended_window_layout_preset(self) -> WindowLayoutPreset:
        available = self._current_available_geometry()
        if available is None:
            return WINDOW_LAYOUT_PRESET_MAP[DEFAULT_WINDOW_PRESET_KEY]
        if available.width() <= 1366 or available.height() <= 820:
            return WINDOW_LAYOUT_PRESET_MAP["laptop_13"]
        if available.width() <= 1600 or available.height() <= 940:
            return WINDOW_LAYOUT_PRESET_MAP["laptop_14"]
        if available.width() <= 1920 or available.height() <= 1100:
            return WINDOW_LAYOUT_PRESET_MAP["display_1080p"]
        return WINDOW_LAYOUT_PRESET_MAP["display_1440p"]

    def _apply_auto_window_layout_preset(self) -> None:
        preset = self._recommended_window_layout_preset()
        self._apply_window_layout_preset(preset.key)

    def _apply_window_layout_preset(self, preset_key: str) -> None:
        preset = WINDOW_LAYOUT_PRESET_MAP.get(str(preset_key).strip())
        if preset is None:
            raise ValueError(f"Unknown window preset: {preset_key}")
        target_size = self._fit_window_size_to_current_screen(
            QSize(preset.width, preset.height)
        )
        self.resize(target_size)
        self._set_ui_scale(preset.ui_scale)
        self.statusBar().showMessage(
            "Applied window preset: "
            f"{preset.label} ({target_size.width()} x {target_size.height()}, "
            f"{int(round(preset.ui_scale * 100))}% scale)"
        )

    def _default_window_size(self) -> QSize:
        default_preset = WINDOW_LAYOUT_PRESET_MAP[DEFAULT_WINDOW_PRESET_KEY]
        return self._fit_window_size_to_current_screen(
            QSize(default_preset.width, default_preset.height)
        )

    def create_project_from_tab(self) -> None:
        try:
            project_dir = self.project_setup_tab.project_dir()
            if project_dir is None:
                self._show_error(
                    "Create project failed",
                    "Select a project directory and enter a project folder name first.",
                )
                return
            if project_dir.exists():
                if not project_dir.is_dir():
                    self._show_error(
                        "Create project failed",
                        "The selected project path already exists and is not a directory.",
                    )
                    return
                response = QMessageBox.warning(
                    self,
                    "Project folder already exists",
                    (
                        f"The folder\n{project_dir}\n\nalready exists. "
                        "Creating this project can overwrite the existing "
                        "SAXS project files in that folder. Do you want to continue?"
                    ),
                    QMessageBox.StandardButton.Yes
                    | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                if response != QMessageBox.StandardButton.Yes:
                    self.statusBar().showMessage("Project creation canceled")
                    return
            project_name = project_dir.name
            settings = self.project_manager.create_project(
                project_dir,
                project_name=project_name,
            )
            self.current_settings = settings
            self._apply_project_settings(settings)
            self._remember_recent_project(settings.project_dir)
            self.project_setup_tab.append_summary(
                f"Created project {settings.project_name} at {settings.project_dir}"
            )
            self.statusBar().showMessage("Project created")
        except Exception as exc:
            self._show_error("Create project failed", str(exc))

    def _create_project_from_menu(self) -> None:
        self.tabs.setCurrentWidget(self.project_setup_tab)
        self.create_project_from_tab()

    def open_project_from_dialog(self) -> None:
        try:
            selected_path = self.project_setup_tab.open_project_dir()
            if selected_path is not None:
                self.load_project(self._validated_project_dir(selected_path))
                return
            selected = QFileDialog.getExistingDirectory(
                self,
                "Open SAXS project folder",
                str(Path.home()),
            )
            if selected:
                self.load_project(self._validated_project_dir(selected))
        except Exception as exc:
            self._show_error("Open project failed", str(exc))

    def _open_project_from_menu(self) -> None:
        try:
            start_dir = (
                self.current_settings.project_dir
                if self.current_settings is not None
                else str(Path.home())
            )
            selected = QFileDialog.getExistingDirectory(
                self,
                "Open SAXS project folder",
                start_dir,
            )
            if selected:
                self.tabs.setCurrentWidget(self.project_setup_tab)
                self.load_project(self._validated_project_dir(selected))
        except Exception as exc:
            self._show_error("Open project failed", str(exc))

    def load_project(self, project_dir: str | Path) -> None:
        resolved_project_dir = Path(project_dir).expanduser().resolve()
        project_file = (
            resolved_project_dir
            if resolved_project_dir.name.endswith(".json")
            else build_project_paths(resolved_project_dir).project_file
        )
        processed_steps = 0
        payload: ProjectLoadPayload | None = None

        def advance_step(
            message: str,
            *,
            log_message: str | None = None,
        ) -> None:
            nonlocal processed_steps
            processed_steps += 1
            self._update_project_load_progress(
                processed_steps,
                PROJECT_LOAD_TOTAL_STEPS,
                message,
                log_message=log_message,
            )

        self._begin_project_load_progress(
            PROJECT_LOAD_TOTAL_STEPS,
            f"Opening project {resolved_project_dir.name}...",
        )
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self._append_project_load_output(
                f"Loading project settings from {project_file}"
            )
            payload = self._prepare_project_load_payload(
                resolved_project_dir,
                selected_dream_preset=(
                    self.dream_tab.selected_settings_preset_name()
                ),
                progress_callback=advance_step,
            )
            processed_steps = PROJECT_LOAD_PREP_STEPS
            self._last_solution_scattering_estimate = None
            self.current_settings = payload.settings
            self._apply_project_settings(
                payload.settings,
                progress_callback=advance_step,
                log_callback=self._append_project_load_output,
                prefit_payload=payload.prefit,
                dream_payload=payload.dream,
            )
            self._append_project_load_output(
                f"Recording recent project: {payload.settings.project_dir}"
            )
            self._remember_recent_project(payload.settings.project_dir)
            self._append_project_load_output(
                f"Loaded project {payload.settings.project_name}"
            )
            for message in payload.warnings:
                self._append_project_load_output(message)
            self.project_setup_tab.finish_activity_progress(
                "Project load complete."
            )
            self.statusBar().showMessage(
                "Project loaded with registered-path warnings"
                if payload.warnings
                else "Project loaded"
            )
        except Exception:
            self.project_setup_tab.finish_activity_progress(
                "Project load failed."
            )
            raise
        finally:
            QApplication.restoreOverrideCursor()
            self._close_progress_dialog()

    def _prepare_project_load_payload(
        self,
        project_dir: Path,
        *,
        selected_dream_preset: str | None,
        progress_callback: Callable[[str], None] | None = None,
    ) -> ProjectLoadPayload:
        result: ProjectLoadPayload | None = None
        failure_message: str | None = None
        worker_thread = QThread(self)
        worker: SAXSProjectTaskWorker | None = None
        loop = QEventLoop(self)

        def on_progress(
            processed: int,
            total: int,
            message: str,
        ) -> None:
            if progress_callback is None:
                return
            del total
            self._update_project_load_progress(
                processed,
                PROJECT_LOAD_TOTAL_STEPS,
                message,
                log_message=message,
            )

        def on_finished(task_name: str, worker_result: object) -> None:
            nonlocal failure_message, result
            del task_name
            if isinstance(worker_result, ProjectLoadPayload):
                result = worker_result
            else:
                failure_message = (
                    "Project loader returned an unexpected payload type."
                )
            loop.quit()

        def on_failed(task_name: str, message: str) -> None:
            nonlocal failure_message
            del task_name
            failure_message = message
            loop.quit()

        def run_task(
            worker_progress: Callable[[int, int, str], None],
        ) -> ProjectLoadPayload:
            return self._build_project_load_payload(
                project_dir,
                selected_dream_preset=selected_dream_preset,
                progress_callback=worker_progress,
            )

        try:
            worker = SAXSProjectTaskWorker("load_project_payload", run_task)
            worker.moveToThread(worker_thread)
            worker_thread.started.connect(worker.run)
            worker.progress.connect(on_progress)
            worker.finished.connect(on_finished)
            worker.failed.connect(on_failed)
            worker.finished.connect(worker_thread.quit)
            worker.failed.connect(worker_thread.quit)
            worker_thread.start()
            loop.exec()
        finally:
            if worker_thread.isRunning():
                worker_thread.quit()
                worker_thread.wait(1500)
            if worker is not None:
                worker.deleteLater()
            worker_thread.deleteLater()
        if failure_message is not None:
            raise RuntimeError(failure_message)
        if result is None:
            raise RuntimeError(
                "Project loading did not return a result payload."
            )
        return result

    def _build_project_load_payload(
        self,
        project_dir: Path,
        *,
        selected_dream_preset: str | None,
        progress_callback: Callable[[int, int, str], None],
    ) -> ProjectLoadPayload:
        manager = SAXSProjectManager()
        progress_callback(
            1,
            PROJECT_LOAD_PREP_STEPS,
            "Loading project settings...",
        )
        settings = manager.load_project(project_dir)
        progress_callback(
            2,
            PROJECT_LOAD_PREP_STEPS,
            "Preparing Prefit workflow and preview...",
        )
        prefit_payload = self._build_project_load_prefit_payload(settings)
        progress_callback(
            3,
            PROJECT_LOAD_PREP_STEPS,
            "Preparing DREAM workflow...",
        )
        dream_payload = self._build_project_load_dream_payload(
            settings,
            selected_dream_preset=selected_dream_preset,
        )
        progress_callback(
            4,
            PROJECT_LOAD_PREP_STEPS,
            "Checking registered project paths...",
        )
        warnings = tuple(manager.registered_folder_warnings(settings))
        return ProjectLoadPayload(
            settings=settings,
            warnings=warnings,
            prefit=prefit_payload,
            dream=dream_payload,
        )

    def _build_project_load_prefit_payload(
        self,
        settings: ProjectSettings,
    ) -> ProjectLoadPrefitPayload:
        try:
            workflow = SAXSPrefitWorkflow(
                settings.project_dir,
                template_name=settings.selected_model_template,
            )
        except Exception as exc:
            return ProjectLoadPrefitPayload(
                workflow=None,
                evaluation=None,
                workflow_error=str(exc),
            )
        evaluation: PrefitEvaluation | None = None
        preview_error: str | None = None
        scale_recommendation: PrefitScaleRecommendation | None = None
        try:
            evaluation = workflow.evaluate()
        except Exception as exc:
            preview_error = str(exc)
        else:
            try:
                scale_recommendation = workflow.recommend_scale_settings()
            except Exception:
                scale_recommendation = None
        return ProjectLoadPrefitPayload(
            workflow=workflow,
            evaluation=evaluation,
            scale_recommendation=scale_recommendation,
            preview_error=preview_error,
        )

    def _build_project_load_dream_payload(
        self,
        settings: ProjectSettings,
        *,
        selected_dream_preset: str | None,
    ) -> ProjectLoadDreamPayload:
        if settings.model_only_mode:
            return ProjectLoadDreamPayload(
                workflow=None,
                preset_names=[],
                selected_preset=None,
                settings=None,
                parameter_map_entries=[],
                error=(
                    "DREAM is disabled in Model Only Mode. Disable Model "
                    "Only Mode and add experimental SAXS data to enable "
                    "DREAM."
                ),
            )
        try:
            workflow = SAXSDreamWorkflow(settings.project_dir)
        except Exception as exc:
            return ProjectLoadDreamPayload(
                workflow=None,
                preset_names=[],
                selected_preset=None,
                settings=None,
                parameter_map_entries=[],
                error=str(exc),
            )
        if not workflow.prefit_workflow.can_run_prefit():
            return ProjectLoadDreamPayload(
                workflow=None,
                preset_names=[],
                selected_preset=None,
                settings=None,
                parameter_map_entries=[],
                error=(
                    "DREAM requires experimental SAXS data and an enabled "
                    "prefit workflow."
                ),
            )
        try:
            preset_names = workflow.list_settings_presets()
            active_preset = (
                selected_dream_preset
                if selected_dream_preset in preset_names
                else None
            )
            dream_settings = workflow.load_settings_preset(active_preset)
        except Exception as exc:
            return ProjectLoadDreamPayload(
                workflow=None,
                preset_names=[],
                selected_preset=None,
                settings=None,
                parameter_map_entries=[],
                error=str(exc),
            )
        try:
            parameter_map_entries = workflow.load_parameter_map(
                persist_if_missing=False
            )
        except Exception:
            parameter_map_entries = []
        return ProjectLoadDreamPayload(
            workflow=workflow,
            preset_names=preset_names,
            selected_preset=active_preset,
            settings=dream_settings,
            parameter_map_entries=parameter_map_entries,
        )

        if (
            settings is not None
            and settings.resolved_clusters_dir is not None
            and settings.resolved_clusters_dir.is_dir()
        ):
            self.project_setup_tab.append_summary(
                "Refreshing cluster inventory from "
                f"{settings.resolved_clusters_dir}"
            )
            self.project_setup_tab.request_cluster_scan()

    def scan_clusters_from_tab(self) -> None:
        clusters_dir = self.project_setup_tab.clusters_dir()
        if clusters_dir is None:
            self._show_error(
                "Cluster import failed",
                "Select a clusters directory first.",
            )
            return
        self._start_project_task(
            "scan_clusters",
            lambda progress: self.project_manager.scan_cluster_inventory(
                clusters_dir,
                progress_callback=progress,
            ),
            start_message="Importing cluster files...",
        )

    def build_project_components(self) -> None:
        try:
            settings = self._settings_from_project_tab()
            if not self._confirm_default_q_range_for_component_build():
                return
            build_mode_label = component_build_mode_label(
                settings.component_build_mode
            )
            if settings.component_build_mode == COMPONENT_BUILD_MODE_CONTRAST:
                self._save_settings(
                    settings,
                    status_message=(
                        "Project auto-saved before launching the contrast-mode "
                        "SAXS workflow"
                    ),
                )
                self.current_settings = settings
                self.project_setup_tab.append_summary(
                    "Build SAXS Components requested in Contrast Mode.\n"
                    "Launching the separate contrast-mode workflow window "
                    "with the current project context.\n"
                    "Run the representative analysis, electron-density step, "
                    "and contrast Debye build from that window when you are "
                    "ready.\n"
                    "Switch Component Build Mode back to No Contrast Mode to "
                    "use the existing SAXS component builder."
                )
                self._open_contrast_mode_tool()
                self.statusBar().showMessage(
                    "Opened SAXS contrast-mode workflow"
                )
                return
            self._save_settings(
                settings,
                status_message="Project auto-saved before building SAXS components",
            )
            self.current_settings = settings
            self._start_project_task(
                "build_components",
                lambda progress: (
                    settings,
                    self.project_manager.build_scattering_components(
                        settings,
                        progress_callback=progress,
                    ),
                ),
                start_message=(
                    f"Building SAXS components ({build_mode_label})..."
                ),
                settings=settings,
            )
        except Exception as exc:
            self._show_error("Build failed", str(exc))

    def _current_project_has_built_components(self) -> bool:
        try:
            settings = self._settings_from_project_tab()
        except Exception:
            if self.current_settings is not None:
                settings = ProjectSettings.from_dict(
                    self.current_settings.to_dict()
                )
            else:
                project_dir = self.project_setup_tab.project_dir()
                if project_dir is None:
                    return False
                settings = ProjectSettings(
                    project_name=project_dir.name,
                    project_dir=str(project_dir),
                    use_predicted_structure_weights=bool(
                        self.project_setup_tab.use_predicted_structure_weights()
                    ),
                )
        if settings is None:
            return False
        component_dir = project_artifact_paths(settings).component_dir
        return component_dir.is_dir() and any(component_dir.glob("*.txt"))

    def _on_prefit_field_interaction_requested(self) -> None:
        if self._current_project_has_built_components():
            self._prefit_missing_components_warning_shown = False
            return
        message = (
            "Build SAXS components in the Project Setup tab before editing "
            "Prefit fields for this model."
        )
        if self._prefit_missing_components_warning_shown:
            self.statusBar().showMessage(message, 5000)
            return
        self._prefit_missing_components_warning_shown = True
        QMessageBox.warning(
            self,
            "Build SAXS components first",
            message,
        )
        self.statusBar().showMessage(message, 5000)

    def _confirm_default_q_range_for_component_build(self) -> bool:
        if (
            not self.project_setup_tab.q_range_matches_loaded_experimental_defaults()
        ):
            return True
        default_range = self.project_setup_tab.default_experimental_q_range()
        if default_range is None:
            return True
        q_min, q_max = default_range
        response = QMessageBox.warning(
            self,
            "Build SAXS components with default q-range?",
            (
                "The q-range still matches the full experimental-data "
                f"default ({q_min:.6g} to {q_max:.6g}). If you intended to "
                "crop the SAXS range, adjust q min and q max before "
                "building.\n\n"
                "Continue building SAXS components with the default q-range?"
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if response == QMessageBox.StandardButton.Yes:
            return True
        self.statusBar().showMessage(
            "SAXS component build canceled so the q-range can be adjusted.",
            5000,
        )
        return False

    def build_prior_weights(self) -> None:
        try:
            settings = self._settings_from_project_tab()
            self._save_settings(
                settings,
                status_message="Project auto-saved before generating prior weights",
            )
            self.current_settings = settings
            self._start_project_task(
                "build_prior_weights",
                lambda progress: (
                    settings,
                    self.project_manager.generate_prior_weights(
                        settings,
                        progress_callback=progress,
                    ),
                ),
                start_message="Generating prior weights...",
                settings=settings,
            )
        except Exception as exc:
            self._show_error("Generate prior weights failed", str(exc))

    def install_model_template(self) -> None:
        try:
            request = self._prompt_template_install_request()
            if request is None:
                return
            try:
                installed = install_template_candidate(
                    request.template_path,
                    model_name=request.model_name,
                    model_description=request.model_description,
                )
            except FileExistsError:
                response = QMessageBox.warning(
                    self,
                    "Template already exists",
                    (
                        "A template with this installed model name already "
                        "exists. Do you want to overwrite it?"
                    ),
                    QMessageBox.StandardButton.Yes
                    | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                if response != QMessageBox.StandardButton.Yes:
                    self.statusBar().showMessage(
                        "Template installation canceled"
                    )
                    return
                installed = install_template_candidate(
                    request.template_path,
                    model_name=request.model_name,
                    model_description=request.model_description,
                    overwrite=True,
                )
            report = format_validation_report(installed.validation_result)
            self._refresh_template_selectors(
                selected_name=self.project_setup_tab.selected_template_name(),
                active_name=self._active_template_name(),
            )
            self.project_setup_tab.append_summary(
                "Installed model template "
                f"{request.model_name} from {request.template_path}"
            )
            QMessageBox.information(
                self,
                "Model installed",
                report
                + "\n\nInstalled template:\n"
                + str(installed.installed_template_path),
            )
            self.statusBar().showMessage("Model installed")
        except Exception as exc:
            self._show_error("Install model failed", str(exc))

    def _prompt_template_install_request(
        self,
    ) -> TemplateInstallRequest | None:
        dialog = InstallModelDialog(self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None
        return dialog.selected_request()

    def _refresh_template_selectors(
        self,
        *,
        selected_name: str | None = None,
        active_name: str | None = None,
        include_deprecated: bool | None = None,
    ) -> None:
        include_deprecated = (
            self._show_deprecated_templates
            if include_deprecated is None
            else bool(include_deprecated)
        )
        active_names = [
            name
            for name in [
                active_name,
                self.project_setup_tab.active_template_name(),
                self.prefit_tab.active_template_name(),
                (
                    self.prefit_workflow.template_spec.name
                    if self.prefit_workflow is not None
                    else None
                ),
                (
                    self.current_settings.selected_model_template
                    if self.current_settings is not None
                    else None
                ),
            ]
            if name
        ]
        selected_names = [
            name
            for name in [
                selected_name,
                self.project_setup_tab.selected_template_name(),
                self.prefit_tab.selected_template_name(),
                *active_names,
            ]
            if name
        ]
        template_specs = self._template_specs_for_dropdown(
            include_deprecated=include_deprecated,
            selected_names=selected_names,
        )
        self._set_show_deprecated_templates(include_deprecated)
        project_selected_name = (
            selected_name or self.project_setup_tab.selected_template_name()
        )
        project_active_name = (
            active_name or self.project_setup_tab.active_template_name()
        )
        self.project_setup_tab.set_available_templates(
            template_specs,
            project_selected_name,
            active_name=project_active_name,
        )
        prefit_selected_name = (
            selected_name or self.prefit_tab.selected_template_name()
        )
        prefit_active_name = (
            active_name or self.prefit_tab.active_template_name()
        )
        self.prefit_tab.set_templates(
            template_specs,
            prefit_selected_name,
            active_name=prefit_active_name,
        )

    def _sync_template_selection_controls(
        self,
        template_name: str | None,
        *,
        source: str | None = None,
    ) -> None:
        normalized_name = str(template_name or "").strip() or None
        self._syncing_template_controls = True
        try:
            if source != "project_setup" and normalized_name is not None:
                self.project_setup_tab.set_selected_template(normalized_name)
            if source != "prefit" and normalized_name is not None:
                self.prefit_tab.set_selected_template(normalized_name)
        finally:
            self._syncing_template_controls = False

    def _sync_active_template_controls(
        self,
        template_name: str | None,
        *,
        sync_selected: bool = False,
    ) -> None:
        normalized_name = str(template_name or "").strip() or None
        self._syncing_template_controls = True
        try:
            self.project_setup_tab.set_active_template(
                normalized_name,
                sync_selected=False,
            )
            self.prefit_tab.set_active_template(
                normalized_name,
                sync_selected=False,
            )
            if sync_selected and normalized_name is not None:
                self.project_setup_tab.set_selected_template(normalized_name)
                self.prefit_tab.set_selected_template(normalized_name)
        finally:
            self._syncing_template_controls = False

    def _active_template_name(self) -> str | None:
        candidates = [
            self.project_setup_tab.active_template_name(),
            self.prefit_tab.active_template_name(),
            (
                self.prefit_workflow.template_spec.name
                if self.prefit_workflow is not None
                else None
            ),
            (
                self.current_settings.selected_model_template
                if self.current_settings is not None
                else None
            ),
        ]
        for candidate in candidates:
            normalized = str(candidate or "").strip()
            if normalized:
                return normalized
        return None

    def _template_specs_for_dropdown(
        self,
        *,
        include_deprecated: bool,
        selected_names: list[str] | tuple[str, ...],
    ) -> list:
        template_specs = list_template_specs(
            include_deprecated=include_deprecated
        )
        specs_by_name = {spec.name: spec for spec in template_specs}
        if not include_deprecated:
            for template_name in selected_names:
                normalized_name = str(template_name).strip()
                if not normalized_name or normalized_name in specs_by_name:
                    continue
                try:
                    spec = load_template_spec(normalized_name)
                except Exception:
                    continue
                specs_by_name[spec.name] = spec
                template_specs.append(spec)
        return sorted(
            specs_by_name.values(),
            key=lambda spec: (
                bool(spec.deprecated),
                spec.display_name.lower(),
                spec.name.lower(),
            ),
        )

    def _set_show_deprecated_templates(self, enabled: bool) -> None:
        self._show_deprecated_templates = bool(enabled)
        self._updating_deprecated_template_visibility = True
        try:
            self.project_setup_tab.set_show_deprecated_templates(enabled)
            self.prefit_tab.set_show_deprecated_templates(enabled)
        finally:
            self._updating_deprecated_template_visibility = False

    @Slot(bool)
    def _on_show_deprecated_templates_changed(self, enabled: bool) -> None:
        if self._updating_deprecated_template_visibility:
            return
        self._refresh_template_selectors(include_deprecated=enabled)

    def save_project_state(self) -> None:
        try:
            settings = self._settings_from_project_tab()
            saved_path = self._save_settings(settings)
            self._sync_live_project_settings_after_save(settings)
            self.project_setup_tab.append_summary(
                f"Saved project state to {saved_path}"
            )
            rebuild_message = self._component_q_range_rebuild_message(settings)
            if rebuild_message is not None:
                self._show_error(
                    "Expanded q-range requires rebuilding SAXS components",
                    rebuild_message,
                )
                self.statusBar().showMessage(
                    "Project state saved; rebuild SAXS components to apply the expanded q-range"
                )
            else:
                self.statusBar().showMessage("Project state saved")
        except Exception as exc:
            self._show_error("Save project state failed", str(exc))

    def _sync_live_project_settings_after_save(
        self,
        settings: ProjectSettings,
    ) -> None:
        active_settings = ProjectSettings.from_dict(settings.to_dict())
        self.current_settings = active_settings
        self.project_setup_tab.set_project_selected(True)
        self._refresh_component_plot()
        self._refresh_prior_plot()

        if self.prefit_workflow is not None:
            self.prefit_workflow.parameter_entries = (
                self.prefit_tab.parameter_entries()
            )
            self.prefit_workflow.apply_project_settings(active_settings)
            self.current_settings = self.prefit_workflow.settings
            self._load_prefit_preview()
            self._refresh_saved_prefit_states()

        if self.dream_workflow is not None:
            self.dream_workflow.apply_project_settings(self.current_settings)
            self._dream_workflow_project_dir = str(
                Path(self.current_settings.project_dir).resolve()
            )
            self._invalidate_written_dream_bundle()
            self._last_dream_constraint_warning_signature = None
            self._refresh_saved_dream_runs()

        self._refresh_model_only_mode_state()
        self._update_active_contrast_distribution_view_state(
            self.current_settings
        )
        self._update_file_menu_state()

    @Slot(str)
    def _autosave_project_from_tab(self, reason: str) -> None:
        if self.project_setup_tab.project_dir() is None:
            return
        try:
            settings = self._settings_from_project_tab()
            self._save_settings(
                settings,
                status_message=f"Project auto-saved after {reason}",
            )
        except Exception:
            # Autosave should never interrupt the user's file-selection flow.
            return

    def _save_settings(
        self,
        settings: ProjectSettings,
        *,
        status_message: str | None = None,
    ) -> Path:
        self._clear_registered_path_snapshots_for_changed_references(settings)
        self.current_settings = settings
        saved_path = self._save_project_without_registered_path_refresh(
            settings
        )
        self._remember_recent_project(settings.project_dir)
        if status_message is not None:
            self.statusBar().showMessage(status_message)
        self._update_file_menu_state()
        return saved_path

    def _save_project_without_registered_path_refresh(
        self,
        settings: ProjectSettings,
    ) -> Path:
        save_project = self.project_manager.save_project
        try:
            save_signature = signature(save_project)
        except (TypeError, ValueError):
            save_signature = None
        if save_signature is not None:
            supports_refresh_flag = (
                "refresh_registered_paths" in save_signature.parameters
                or any(
                    parameter.kind == Parameter.VAR_KEYWORD
                    for parameter in save_signature.parameters.values()
                )
            )
            if not supports_refresh_flag:
                return save_project(settings)
        return save_project(
            settings,
            refresh_registered_paths=False,
        )

    def _clear_registered_path_snapshots_for_changed_references(
        self,
        settings: ProjectSettings,
    ) -> None:
        current_settings = self.current_settings
        if current_settings is None:
            return
        for field_name, snapshot_field in (
            ("frames_dir", "frames_dir_snapshot"),
            ("pdb_frames_dir", "pdb_frames_dir_snapshot"),
            ("clusters_dir", "clusters_dir_snapshot"),
            ("trajectory_file", "trajectory_file_snapshot"),
            ("topology_file", "topology_file_snapshot"),
            ("energy_file", "energy_file_snapshot"),
        ):
            current_value = self._normalized_registered_path_value(
                getattr(current_settings, field_name)
            )
            updated_value = self._normalized_registered_path_value(
                getattr(settings, field_name)
            )
            if current_value != updated_value:
                setattr(settings, snapshot_field, None)

    @staticmethod
    def _normalized_registered_path_value(value: object) -> str | None:
        text = str(value or "").strip()
        return text or None

    def _component_q_range_rebuild_message(
        self,
        settings: ProjectSettings,
    ) -> str | None:
        artifact_paths = project_artifact_paths(settings)
        supported_range = load_built_component_q_range(
            settings.project_dir,
            include_predicted_structures=(
                settings.use_predicted_structure_weights
            ),
            component_dir=artifact_paths.component_dir,
        )
        if supported_range is None:
            return None
        supported_min, supported_max = supported_range
        requested_min = (
            float(settings.q_min)
            if settings.q_min is not None
            else float(supported_min)
        )
        requested_max = (
            float(settings.q_max)
            if settings.q_max is not None
            else float(supported_max)
        )
        if (
            not settings.model_only_mode
            and artifact_paths.distribution_metadata_file is not None
            and artifact_paths.distribution_metadata_file.is_file()
        ):
            try:
                experimental_data = (
                    self.project_manager.load_experimental_data(settings)
                )
            except Exception:
                experimental_data = None
            if experimental_data is not None:
                try:
                    requested_min, requested_max = (
                        effective_q_range_for_settings(
                            settings,
                            np.asarray(
                                experimental_data.q_values,
                                dtype=float,
                            ),
                        )
                    )
                except Exception:
                    requested_min = (
                        float(settings.q_min)
                        if settings.q_min is not None
                        else float(supported_min)
                    )
                    requested_max = (
                        float(settings.q_max)
                        if settings.q_max is not None
                        else float(supported_max)
                    )
        requested_min, requested_max = (
            normalize_requested_q_range_to_supported(
                requested_min,
                requested_max,
                supported_min,
                supported_max,
            )
        )
        tolerance = q_range_boundary_tolerance(
            supported_min,
            supported_max,
        )
        if requested_min >= (supported_min - tolerance) and requested_max <= (
            supported_max + tolerance
        ):
            return None
        return (
            "The requested q-range "
            f"{requested_min:.6g} to {requested_max:.6g} extends beyond the "
            "q-range currently covered by the built SAXS model components "
            f"({supported_min:.6g} to {supported_max:.6g}). Recompute the "
            "SAXS model components in Project Setup for the updated q-range "
            "to be applied in Prefit and DREAM."
        )

    def save_project_as(self) -> None:
        if self.current_settings is None:
            self._show_error(
                "Save Project As failed",
                "Load or create a project first.",
            )
            return
        try:
            current_settings = self._settings_from_project_tab()
            source_dir = (
                Path(current_settings.project_dir).expanduser().resolve()
            )
            if not source_dir.is_dir():
                raise ValueError(
                    "The active project folder could not be found on disk."
                )
            self._save_settings(
                current_settings,
                status_message="Project saved before Save Project As",
            )
            parent_dir = QFileDialog.getExistingDirectory(
                self,
                "Select destination parent folder",
                str(source_dir.parent),
            )
            if not parent_dir:
                self.statusBar().showMessage("Save Project As canceled")
                return
            project_name, accepted = QInputDialog.getText(
                self,
                "Save Project As",
                "New project folder name:",
                text=f"{source_dir.name}_copy",
            )
            project_name = str(project_name).strip()
            if not accepted or not project_name:
                self.statusBar().showMessage("Save Project As canceled")
                return
            destination_dir = (
                Path(parent_dir).expanduser().resolve() / project_name
            )
            if destination_dir.exists():
                raise ValueError(
                    "The selected Save Project As destination already exists. "
                    "Choose a new folder name."
                )
            shutil.copytree(source_dir, destination_dir)
            new_settings = ProjectSettings.from_dict(
                current_settings.to_dict()
            )
            new_settings.project_name = destination_dir.name
            new_settings.project_dir = str(destination_dir)
            self._remap_copied_project_paths(
                new_settings,
                old_project_dir=source_dir,
                new_project_dir=destination_dir,
            )
            saved_path = self._save_settings(
                new_settings,
                status_message="Project saved to new folder",
            )
            self.current_settings = new_settings
            self._apply_project_settings(new_settings)
            self.tabs.setCurrentWidget(self.project_setup_tab)
            self.project_setup_tab.append_summary(
                f"Saved project as {destination_dir}\nProject file: {saved_path}"
            )
            self.statusBar().showMessage("Project saved to new folder")
        except Exception as exc:
            self._show_error("Save Project As failed", str(exc))

    def _start_project_task(
        self,
        task_name: str,
        task_fn: Callable[[Callable[[int, int, str], None]], object],
        *,
        start_message: str,
        settings: ProjectSettings | None = None,
    ) -> None:
        if self._task_thread is not None:
            self.statusBar().showMessage(
                "A SAXS project task is already running."
            )
            return

        self._active_task_name = task_name
        self._active_task_settings = settings
        self._task_thread = QThread(self)
        self._task_worker = SAXSProjectTaskWorker(task_name, task_fn)
        self._task_worker.moveToThread(self._task_thread)
        self._task_thread.started.connect(self._task_worker.run)
        self._task_worker.progress.connect(self._on_task_progress)
        self._task_worker.finished.connect(self._on_task_finished)
        self._task_worker.failed.connect(self._on_task_failed)
        self._task_worker.finished.connect(self._task_thread.quit)
        self._task_worker.failed.connect(self._task_thread.quit)
        self._task_thread.finished.connect(self._cleanup_task_thread)

        self.project_setup_tab.start_activity_progress(1, start_message)
        self._show_progress_dialog(1, start_message)
        self.statusBar().showMessage(start_message)
        self._task_thread.start()

    @Slot(int, int, str)
    def _on_task_progress(
        self,
        processed: int,
        total: int,
        message: str,
    ) -> None:
        self.project_setup_tab.update_activity_progress(
            processed,
            total,
            message,
        )
        if self._progress_dialog is not None:
            self._progress_dialog.update_progress(
                processed,
                total,
                message,
            )
        self.statusBar().showMessage(message)

    @Slot(str, object)
    def _on_task_finished(self, task_name: str, result: object) -> None:
        if task_name == "scan_clusters":
            cluster_result = result
            if isinstance(cluster_result, ClusterImportResult):
                if self.current_settings is not None:
                    self.current_settings.available_elements = (
                        cluster_result.available_elements
                    )
                    self.current_settings.cluster_inventory_rows = (
                        cluster_result.cluster_rows
                    )
                    active_elements, active_rows = (
                        self._active_cluster_import_view(
                            self.current_settings,
                            fallback_available_elements=(
                                cluster_result.available_elements
                            ),
                            fallback_rows=cluster_result.cluster_rows,
                        )
                    )
                else:
                    active_elements = cluster_result.available_elements
                    active_rows = cluster_result.cluster_rows
                self.project_setup_tab.apply_cluster_import_data(
                    active_elements,
                    active_rows,
                )
                self.project_setup_tab.append_summary(
                    "Imported cluster files for project setup.\n"
                    f"Files scanned: {cluster_result.total_files}\n"
                    f"Recognized cluster bins: {len(cluster_result.cluster_rows)}"
                )
                self.project_setup_tab.finish_activity_progress(
                    "Cluster import complete."
                )
                self.statusBar().showMessage("Cluster import complete")
        elif task_name == "build_components":
            settings, build_result = result
            reloaded_settings = self.project_manager.load_project(
                settings.project_dir
            )
            reloaded_settings.cluster_inventory_rows = (
                build_result.cluster_rows
            )
            self.current_settings = reloaded_settings
            self._apply_project_settings(reloaded_settings)
            active_elements, active_rows = self._active_cluster_import_view(
                reloaded_settings,
                fallback_available_elements=reloaded_settings.available_elements,
                fallback_rows=build_result.cluster_rows,
            )
            self.project_setup_tab.apply_cluster_import_data(
                active_elements,
                active_rows,
            )
            if build_result.used_predicted_structure_weights:
                self.project_setup_tab.append_summary(
                    "Built observed + Predicted Structures SAXS components for "
                    f"{len(build_result.component_entries)} components.\n"
                    f"Predicted Structures included: {build_result.predicted_component_count}\n"
                    f"Saved component map: {build_result.model_map_path}\n"
                    "Generate prior weights separately when you are ready."
                )
            else:
                self.project_setup_tab.append_summary(
                    "Built SAXS components for "
                    f"{len(build_result.component_entries)} cluster bins.\n"
                    f"Saved component map: {build_result.model_map_path}\n"
                    "Generate prior weights separately when you are ready."
                )
            self._refresh_component_plot()
            try:
                self._load_prefit_workflow()
                self._load_dream_workflow()
            except Exception:
                pass
            self.project_setup_tab.finish_activity_progress(
                "SAXS component build complete."
            )
            self.statusBar().showMessage("Project components built")
        elif task_name == "build_prior_weights":
            settings, build_result = result
            reloaded_settings = self.project_manager.load_project(
                settings.project_dir
            )
            reloaded_settings.cluster_inventory_rows = (
                build_result.cluster_rows
            )
            self.current_settings = reloaded_settings
            self._apply_project_settings(reloaded_settings)
            active_elements, active_rows = self._active_cluster_import_view(
                reloaded_settings,
                fallback_available_elements=reloaded_settings.available_elements,
                fallback_rows=build_result.cluster_rows,
            )
            self.project_setup_tab.apply_cluster_import_data(
                active_elements,
                active_rows,
            )
            if build_result.used_predicted_structure_weights:
                self.project_setup_tab.append_summary(
                    "Generated observed + Predicted Structures prior weights.\n"
                    f"Predicted Structures included: {build_result.predicted_component_count}\n"
                    f"Saved prior weights: {build_result.md_prior_weights_path}\n"
                    f"Saved prior plot data: {build_result.prior_plot_data_path}"
                )
            else:
                self.project_setup_tab.append_summary(
                    "Generated prior weights for "
                    f"{len(build_result.component_entries)} cluster bins.\n"
                    f"Saved prior weights: {build_result.md_prior_weights_path}\n"
                    f"Saved prior plot data: {build_result.prior_plot_data_path}"
                )
            self._refresh_prior_plot()
            self.project_setup_tab.finish_activity_progress(
                "Prior-weight generation complete."
            )
            self.statusBar().showMessage("Prior weights generated")
        self._close_progress_dialog()

    @Slot(str, str)
    def _on_task_failed(self, task_name: str, message: str) -> None:
        del task_name
        self.project_setup_tab.finish_activity_progress("Progress: failed")
        self._close_progress_dialog()
        self._show_error("SAXS project task failed", message)

    def _cleanup_task_thread(self) -> None:
        if self._task_worker is not None:
            self._task_worker.deleteLater()
            self._task_worker = None
        if self._task_thread is not None:
            self._task_thread.deleteLater()
            self._task_thread = None
        self._active_task_name = None
        self._active_task_settings = None

    def _ensure_progress_dialog(self) -> SAXSProgressDialog:
        if self._progress_dialog is None:
            self._progress_dialog = SAXSProgressDialog(self)
        return self._progress_dialog

    def _show_progress_dialog(
        self,
        total: int,
        message: str,
        *,
        unit_label: str = "items",
        title: str | None = None,
    ) -> None:
        dialog = self._ensure_progress_dialog()
        dialog.begin(total, message, unit_label=unit_label, title=title)

    def _begin_project_load_progress(
        self,
        total_steps: int,
        message: str,
    ) -> None:
        self.project_setup_tab.start_activity_progress(
            total_steps,
            message,
            unit_label="steps",
        )
        self._show_progress_dialog(
            total_steps,
            message,
            unit_label="steps",
            title="Loading SAXS Project",
        )
        self.statusBar().showMessage(message)
        QApplication.processEvents()

    def _update_project_load_progress(
        self,
        processed: int,
        total_steps: int,
        message: str,
        *,
        log_message: str | None = None,
    ) -> None:
        self.project_setup_tab.update_activity_progress(
            processed,
            total_steps,
            message,
            unit_label="steps",
        )
        if self._progress_dialog is not None:
            self._progress_dialog.update_progress(
                processed,
                total_steps,
                message,
                unit_label="steps",
            )
        if log_message:
            self._append_project_load_output(log_message)
        self.statusBar().showMessage(message)
        QApplication.processEvents()

    def _append_project_load_output(self, message: str) -> None:
        stripped = str(message).strip()
        if not stripped:
            return
        self.project_setup_tab.append_summary(stripped)
        if self._progress_dialog is not None:
            self._progress_dialog.append_output(stripped)

    def _close_progress_dialog(self) -> None:
        if self._progress_dialog is not None:
            self._progress_dialog.close()

    def _update_prefit_cluster_geometry_progress(
        self,
        processed: int,
        total: int,
        message: str,
    ) -> None:
        self.prefit_tab.update_cluster_geometry_progress(
            processed,
            total,
            message,
            unit_label="files",
        )
        if self._progress_dialog is not None:
            self._progress_dialog.update_progress(
                processed,
                total,
                message,
                unit_label="files",
            )
        self.statusBar().showMessage(message)
        QApplication.processEvents()

    def _ensure_dream_progress_dialog(self) -> SAXSProgressDialog:
        if self._dream_progress_dialog is None:
            self._dream_progress_dialog = SAXSProgressDialog(self)
        return self._dream_progress_dialog

    def _show_dream_progress_dialog(
        self,
        message: str,
        *,
        total: int | None = None,
        unit_label: str = "runs",
        title: str = "SAXS DREAM Progress",
    ) -> None:
        dialog = self._ensure_dream_progress_dialog()
        if total is None:
            dialog.begin_busy(message, title=title)
            return
        dialog.begin(
            total,
            message,
            unit_label=unit_label,
            title=title,
        )

    def _close_dream_progress_dialog(self) -> None:
        if self._dream_progress_dialog is not None:
            self._dream_progress_dialog.close()

    def _start_dream_run_task(
        self,
        bundle: DreamRunBundle,
        settings: DreamRunSettings,
    ) -> None:
        if self._dream_task_thread is not None:
            self.statusBar().showMessage(
                "A DREAM refinement is already running."
            )
            return
        if self.current_settings is None:
            raise ValueError("No project is currently loaded.")

        self._active_dream_run_settings = self._copy_dream_settings(settings)
        self._dream_task_thread = QThread(self)
        self._dream_task_worker = SAXSDreamRunWorker(
            self.current_settings.project_dir,
            bundle,
            verbose_output_interval_seconds=(
                settings.verbose_output_interval_seconds
            ),
        )
        self._dream_task_worker.moveToThread(self._dream_task_thread)
        self._dream_task_thread.started.connect(self._dream_task_worker.run)
        self._dream_task_worker.status.connect(self._on_dream_run_status)
        self._dream_task_worker.output.connect(self._on_dream_run_output)
        self._dream_task_worker.finished.connect(self._on_dream_run_finished)
        self._dream_task_worker.failed.connect(self._on_dream_run_failed)
        self._dream_task_worker.finished.connect(self._dream_task_thread.quit)
        self._dream_task_worker.failed.connect(self._dream_task_thread.quit)
        self._dream_task_thread.finished.connect(
            self._cleanup_dream_task_thread
        )

        start_message = "DREAM refinement in progress..."
        self.dream_tab.start_progress(start_message)
        self._show_dream_progress_dialog(start_message)
        self.dream_tab.run_button.setEnabled(False)
        self.statusBar().showMessage(start_message)
        self._dream_task_thread.start()

    @Slot(str)
    def _on_dream_run_status(self, message: str) -> None:
        self.dream_tab.append_log(message)
        self.dream_tab.start_progress(message)
        if (
            self._dream_progress_dialog is not None
            and self._dream_progress_dialog.isVisible()
        ):
            self._dream_progress_dialog.message_label.setText(message)
        self.statusBar().showMessage(message)

    @Slot(str, object)
    def _on_dream_run_finished(
        self,
        run_dir: str,
        result: object,
    ) -> None:
        settings = (
            self._active_dream_run_settings
            or self.dream_tab.settings_payload()
        )
        self._last_results_loader = SAXSDreamResultsLoader(
            run_dir,
            burnin_percent=settings.burnin_percent,
        )
        self._loaded_dream_run_dir = Path(run_dir).resolve()
        assessment_message: str | None = None
        assessment_error: str | None = None
        self._last_dream_filter_assessments = []
        self._last_dream_filter_recommendation = None
        try:
            (
                settings,
                assessment_message,
            ) = self._assess_and_apply_dream_filter_recommendation(settings)
            self._active_dream_run_settings = self._copy_dream_settings(
                settings
            )
            self.dream_tab.set_settings(settings, preset_name=None)
        except Exception as exc:  # pragma: no cover - defensive UI logging
            assessment_error = str(exc)
        self._refresh_saved_dream_runs(selected_run_dir=run_dir)
        self._refresh_loaded_dream_results()
        auto_export_paths: list[Path] = []
        auto_export_error: str | None = None
        try:
            auto_export_paths = self._auto_export_dream_condensed_outputs(
                settings
            )
        except Exception as exc:  # pragma: no cover - defensive UI logging
            auto_export_error = str(exc)
        result_payload = dict(result) if isinstance(result, dict) else {}
        self.dream_tab.append_log(
            "DREAM run complete.\n"
            f"Run directory: {run_dir}\n"
            f"Samples: {result_payload.get('sampled_params_path', 'unknown')}\n"
            f"Log-posteriors: {result_payload.get('log_ps_path', 'unknown')}"
        )
        if assessment_message:
            self.dream_tab.append_log(assessment_message)
        if assessment_error:
            self.dream_tab.append_log(
                "Automatic DREAM filter assessment warning.\n"
                "The refinement completed, but the post-run filter "
                f"assessment could not be completed.\n{assessment_error}"
            )
        if auto_export_paths:
            self.dream_tab.append_log(
                "Saved condensed DREAM exports for easy access in "
                "exported_results/data:\n"
                + "\n".join(str(path) for path in auto_export_paths)
                + "\nFull DREAM run artifacts remain in the DREAM run "
                "folder."
            )
        if auto_export_error:
            self.dream_tab.append_log(
                "Automatic DREAM export warning.\n"
                "The refinement completed, but the condensed export copy "
                "could not be written to exported_results/data.\n"
                f"{auto_export_error}"
            )
        self.dream_tab.finish_runtime_output()
        self.dream_tab.finish_progress("DREAM refinement complete.")
        self._close_dream_progress_dialog()
        self.dream_tab.run_button.setEnabled(True)
        if auto_export_paths:
            self.statusBar().showMessage(
                "DREAM run complete; condensed exports saved"
            )
        else:
            self.statusBar().showMessage("DREAM run complete")

    @Slot(str)
    def _on_dream_run_failed(self, message: str) -> None:
        self.dream_tab.finish_runtime_output()
        self.dream_tab.append_log("DREAM run failed.\n" + message)
        self.dream_tab.finish_progress("DREAM refinement failed.")
        self._close_dream_progress_dialog()
        self.dream_tab.run_button.setEnabled(True)
        self._show_error("Run DREAM failed", message)

    @Slot(str)
    def _on_dream_run_output(self, message: str) -> None:
        stripped = message.strip()
        if not stripped:
            return
        self.dream_tab.append_runtime_output(stripped)
        latest_line = next(
            (line for line in reversed(stripped.splitlines()) if line.strip()),
            stripped,
        )
        self.dream_tab.progress_label.setText(latest_line)
        if (
            self._dream_progress_dialog is not None
            and self._dream_progress_dialog.isVisible()
        ):
            self._dream_progress_dialog.message_label.setText(latest_line)
        self.statusBar().showMessage(latest_line)

    def _cleanup_dream_task_thread(self) -> None:
        if self._dream_task_worker is not None:
            self._dream_task_worker.deleteLater()
            self._dream_task_worker = None
        if self._dream_task_thread is not None:
            self._dream_task_thread.deleteLater()
            self._dream_task_thread = None
        self._active_dream_run_settings = None

    def closeEvent(self, event) -> None:
        if not self._close_child_tool_windows():
            self.statusBar().showMessage(
                "A linked tool is still busy. Finish its current task "
                "before closing SAXSShell.",
                5000,
            )
            event.ignore()
            return
        self._shutdown_background_threads()
        super().closeEvent(event)

    def _shutdown_background_threads(self) -> None:
        self._close_progress_dialog()
        self._close_dream_progress_dialog()
        self._shutdown_thread(
            thread=self._task_thread,
            cleanup_callback=self._cleanup_task_thread,
        )
        self._shutdown_thread(
            thread=self._dream_task_thread,
            cleanup_callback=self._cleanup_dream_task_thread,
        )

    @staticmethod
    def _shutdown_thread(
        *,
        thread,
        cleanup_callback,
    ) -> None:
        if thread is None:
            cleanup_callback()
            return
        if thread.isRunning():
            thread.quit()
            if not thread.wait(1500):
                thread.terminate()
                thread.wait(1500)
        cleanup_callback()

    def _refresh_prefit_cluster_geometry_section(self) -> None:
        if self.prefit_workflow is None:
            self.prefit_tab.set_cluster_geometry_visible(False)
            self._clear_prefit_cluster_geometry_sync_snapshot()
            return
        visible = self.prefit_workflow.supports_cluster_geometry_metadata()
        self.prefit_tab.set_cluster_geometry_visible(visible)
        if not visible:
            self._clear_prefit_cluster_geometry_sync_snapshot()
            return
        self.prefit_tab.set_cluster_geometry_active_radii_type(
            self.prefit_workflow.cluster_geometry_active_radii_type(),
            emit_signal=False,
        )
        self.prefit_tab.set_cluster_geometry_active_ionic_radius_type(
            self.prefit_workflow.cluster_geometry_active_ionic_radius_type(),
            emit_signal=False,
        )
        self.prefit_tab.populate_cluster_geometry_table(
            self.prefit_workflow.cluster_geometry_rows(),
            mapping_options=(
                self.prefit_workflow.cluster_geometry_mapping_options()
            ),
            active_radii_type=(
                self.prefit_workflow.cluster_geometry_active_radii_type()
            ),
            active_ionic_radius_type=(
                self.prefit_workflow.cluster_geometry_active_ionic_radius_type()
            ),
            allowed_sf_approximations=(
                self.prefit_workflow.allowed_cluster_geometry_approximations()
            ),
        )
        self.prefit_tab.set_cluster_geometry_status_text(
            self.prefit_workflow.cluster_geometry_status_text()
        )
        self._capture_prefit_cluster_geometry_sync_snapshot()

    def _clear_prefit_cluster_geometry_sync_snapshot(self) -> None:
        self._prefit_cluster_geometry_sync_rows_snapshot = None
        self._prefit_cluster_geometry_sync_radii_type = None
        self._prefit_cluster_geometry_sync_ionic_radius_type = None

    def _capture_prefit_cluster_geometry_sync_snapshot(
        self,
        *,
        rows: list[ClusterGeometryMetadataRow] | None = None,
        active_radii_type: str | None = None,
        active_ionic_radius_type: str | None = None,
    ) -> None:
        if (
            self.prefit_workflow is None
            or not self.prefit_workflow.supports_cluster_geometry_metadata()
        ):
            self._clear_prefit_cluster_geometry_sync_snapshot()
            return
        snapshot_rows = (
            rows
            if rows is not None
            else self.prefit_tab.cluster_geometry_rows()
        )
        snapshot_radii_type = (
            active_radii_type
            if active_radii_type is not None
            else self.prefit_tab.cluster_geometry_active_radii_type()
        )
        snapshot_ionic_radius_type = (
            active_ionic_radius_type
            if active_ionic_radius_type is not None
            else self.prefit_tab.cluster_geometry_active_ionic_radius_type()
        )
        self._prefit_cluster_geometry_sync_rows_snapshot = [
            row.to_dict() for row in snapshot_rows
        ]
        self._prefit_cluster_geometry_sync_radii_type = snapshot_radii_type
        self._prefit_cluster_geometry_sync_ionic_radius_type = (
            snapshot_ionic_radius_type
        )

    def _prefit_cluster_geometry_matches_sync_snapshot(
        self,
        rows: list[ClusterGeometryMetadataRow],
        *,
        active_radii_type: str,
        active_ionic_radius_type: str,
    ) -> bool:
        if self._prefit_cluster_geometry_sync_rows_snapshot is None:
            return False
        return (
            self._prefit_cluster_geometry_sync_radii_type == active_radii_type
            and self._prefit_cluster_geometry_sync_ionic_radius_type
            == active_ionic_radius_type
            and [row.to_dict() for row in rows]
            == self._prefit_cluster_geometry_sync_rows_snapshot
        )

    def _refresh_prefit_volume_fraction_section(self) -> None:
        target = self._current_volume_fraction_target()
        solvent_weight_target = self._current_solvent_weight_target()
        visible = self.prefit_workflow is not None
        self.prefit_tab.set_solute_volume_fraction_visible(visible)
        parameter_name = None
        fraction_kind = None
        if target is not None:
            parameter_name, fraction_kind = target
        self.prefit_tab.set_solute_volume_fraction_target(
            parameter_name,
            fraction_kind,
            solvent_weight_target,
        )
        self._sync_solution_scattering_tool_targets()

    def _current_volume_fraction_target(self) -> tuple[str, str] | None:
        if self.prefit_workflow is None:
            return None
        return self.prefit_workflow.volume_fraction_estimator_target()

    def _current_solvent_weight_target(self) -> str | None:
        if self.prefit_workflow is None:
            return None
        return self.prefit_workflow.solvent_weight_estimator_target()

    def _sync_solution_scattering_tool_targets(self) -> None:
        target = self._current_volume_fraction_target()
        if target is None:
            parameter_name = None
            fraction_kind = None
        else:
            parameter_name, fraction_kind = target
        solvent_weight_parameter = self._current_solvent_weight_target()
        for window in (
            self._solute_volume_fraction_tool_window,
            self._number_density_tool_window,
            self._attenuation_tool_window,
            self._fluorescence_tool_window,
        ):
            if window is None:
                continue
            window.estimator_widget.set_target_parameter(
                parameter_name,
                fraction_kind,
                solvent_weight_parameter,
            )

    def _apply_estimator_parameter_to_prefit(
        self,
        parameter_name: str,
        parameter_value: float,
    ) -> None:
        current_entry = next(
            (
                entry
                for entry in self.prefit_tab.parameter_entries()
                if entry.name == parameter_name
            ),
            None,
        )
        minimum = 0.0
        maximum = max(parameter_value, 1.0)
        if current_entry is not None:
            minimum = min(float(current_entry.minimum), 0.0)
            maximum = max(
                float(current_entry.maximum),
                parameter_value,
                1.0,
            )
        self.prefit_tab.set_parameter_row(
            parameter_name,
            value=parameter_value,
            minimum=minimum,
            maximum=maximum,
            vary=False,
        )

    @Slot(object)
    def _on_solution_scattering_estimate_calculated(
        self,
        estimate_payload: object,
    ) -> None:
        if not isinstance(estimate_payload, SolutionScatteringEstimate):
            return
        self._last_solution_scattering_estimate = estimate_payload
        widget = self.sender()
        applied_notes: list[str] = []
        log_lines = ["Applied solution-scattering estimates."]
        if self.prefit_workflow is None:
            if hasattr(widget, "append_application_note"):
                cast(object, widget).append_application_note(
                    "The calculations completed, but there is no active "
                    "Prefit workflow to apply them to."
                )
            self.statusBar().showMessage(
                "Solution-scattering estimate calculated"
            )
            return

        preview_changed = False
        try:
            volume_target = self._current_volume_fraction_target()
            interaction_estimate = (
                estimate_payload.interaction_contrast_estimate
            )
            if interaction_estimate is not None and volume_target is not None:
                parameter_name, fraction_kind = volume_target
                parameter_value = (
                    float(
                        interaction_estimate.saxs_effective_solute_interaction_ratio
                    )
                    if fraction_kind == "solute"
                    else float(
                        interaction_estimate.saxs_effective_solvent_background_ratio
                    )
                )
                self._apply_estimator_parameter_to_prefit(
                    parameter_name,
                    parameter_value,
                )
                applied_notes.append(
                    f"Applied {parameter_name} = "
                    f"{parameter_value:.{DISPLAY_FRACTION_DECIMALS}f} "
                    "from the SAXS-effective interaction ratio."
                )
                log_lines.extend(
                    [
                        f"Volume-fraction target: {parameter_name}",
                        f"Model fraction kind: {fraction_kind}",
                        (
                            "Physical solute-associated volume fraction: "
                            f"{interaction_estimate.physical_solute_associated_volume_fraction:.{DISPLAY_FRACTION_DECIMALS}f}"
                        ),
                        (
                            "Physical solvent-associated volume fraction: "
                            f"{interaction_estimate.physical_solvent_associated_volume_fraction:.{DISPLAY_FRACTION_DECIMALS}f}"
                        ),
                        (
                            "SAXS-effective solute interaction ratio: "
                            f"{interaction_estimate.saxs_effective_solute_interaction_ratio:.{DISPLAY_FRACTION_DECIMALS}f}"
                        ),
                        (
                            "SAXS-effective solvent background ratio: "
                            f"{interaction_estimate.saxs_effective_solvent_background_ratio:.{DISPLAY_FRACTION_DECIMALS}f}"
                        ),
                        (
                            "Contrast weight factor: "
                            f"{interaction_estimate.contrast_weight_factor:.6g}"
                        ),
                    ]
                )
                preview_changed = True
            elif (
                estimate_payload.volume_fraction_estimate is not None
                and hasattr(widget, "append_application_note")
            ):
                cast(object, widget).append_application_note(
                    "Calculated the physical bulk fraction and the "
                    "SAXS-effective interaction ratio, but the active "
                    "template does not expose a model-facing solute/solvent "
                    "interaction fraction parameter."
                )

            solvent_weight_target = self._current_solvent_weight_target()
            if (
                estimate_payload.attenuation_estimate is not None
                and solvent_weight_target is not None
            ):
                attenuation_scale = float(
                    estimate_payload.attenuation_estimate.solvent_scattering_scale_factor
                )
                uses_split_fraction_parameter = volume_target is not None
                solvent_scale = attenuation_scale
                if (
                    not uses_split_fraction_parameter
                    and interaction_estimate is not None
                ):
                    solvent_scale = (
                        attenuation_scale
                        * interaction_estimate.saxs_effective_solvent_background_ratio
                    )
                self._apply_estimator_parameter_to_prefit(
                    solvent_weight_target,
                    solvent_scale,
                )
                if (
                    not uses_split_fraction_parameter
                    and interaction_estimate is not None
                ):
                    applied_notes.append(
                        f"Applied {solvent_weight_target} = "
                        f"{solvent_scale:.{DISPLAY_FRACTION_DECIMALS}f} "
                        "from attenuation x the SAXS-effective solvent "
                        "background ratio."
                    )
                elif not uses_split_fraction_parameter:
                    applied_notes.append(
                        f"Applied {solvent_weight_target} = "
                        f"{solvent_scale:.{DISPLAY_FRACTION_DECIMALS}f} "
                        "from attenuation only because the SAXS-effective "
                        "interaction ratio was not available."
                    )
                else:
                    applied_notes.append(
                        f"Applied {solvent_weight_target} = "
                        f"{solvent_scale:.{DISPLAY_FRACTION_DECIMALS}f} "
                        "as the attenuation solvent scale."
                    )
                log_lines.extend(
                    [
                        f"Solvent-weight target: {solvent_weight_target}",
                        (
                            "Attenuation solvent scale factor: "
                            f"{attenuation_scale:.{DISPLAY_FRACTION_DECIMALS}f}"
                        ),
                        (
                            "Sample transmission: "
                            f"{estimate_payload.attenuation_estimate.sample_transmission:.{DISPLAY_FRACTION_DECIMALS}f}"
                        ),
                        (
                            "Neat-solvent transmission: "
                            f"{estimate_payload.attenuation_estimate.neat_solvent_transmission:.{DISPLAY_FRACTION_DECIMALS}f}"
                        ),
                    ]
                )
                if (
                    not uses_split_fraction_parameter
                    and interaction_estimate is not None
                ):
                    log_lines.append(
                        "Combined solvent background multiplier: "
                        f"{solvent_scale:.{DISPLAY_FRACTION_DECIMALS}f}"
                    )
                elif (
                    uses_split_fraction_parameter
                    and interaction_estimate is not None
                ):
                    log_lines.append(
                        "SAXS-effective solvent background ratio is carried by "
                        f"{volume_target[0]} = "
                        f"{interaction_estimate.saxs_effective_solvent_background_ratio:.{DISPLAY_FRACTION_DECIMALS}f}"
                    )
                preview_changed = True
            elif estimate_payload.attenuation_estimate is not None and hasattr(
                widget, "append_application_note"
            ):
                cast(object, widget).append_application_note(
                    "Calculated the attenuation-based solvent contribution, "
                    "but the active template does not expose a solvent "
                    "background parameter such as solv_w or solvent_scale."
                )

            if applied_notes and hasattr(widget, "append_application_note"):
                cast(object, widget).append_application_note(
                    "\n".join(applied_notes)
                )

            if not preview_changed:
                if estimate_payload.fluorescence_estimate is not None:
                    self.statusBar().showMessage(
                        "Solution-scattering estimate calculated"
                    )
                return

            self.prefit_workflow.parameter_entries = (
                self.prefit_tab.parameter_entries()
            )
            self.prefit_tab.append_log("\n".join(log_lines))
            evaluation = self._load_prefit_preview()
            if evaluation is None:
                self.statusBar().showMessage(
                    "Estimator values applied; preview is waiting on the "
                    "remaining template metadata"
                )
            else:
                self.statusBar().showMessage("Estimator values applied")
        except Exception as exc:
            self._show_error("Apply estimator values failed", str(exc))

    def _sync_prefit_cluster_geometry_rows(self) -> None:
        if (
            self.prefit_workflow is None
            or not self.prefit_workflow.supports_cluster_geometry_metadata()
        ):
            return
        cluster_geometry_rows = self.prefit_tab.cluster_geometry_rows()
        active_radii_type = (
            self.prefit_tab.cluster_geometry_active_radii_type()
        )
        active_ionic_radius_type = (
            self.prefit_tab.cluster_geometry_active_ionic_radius_type()
        )
        preserve_geometry_entries = (
            self._prefit_cluster_geometry_matches_sync_snapshot(
                cluster_geometry_rows,
                active_radii_type=active_radii_type,
                active_ionic_radius_type=active_ionic_radius_type,
            )
        )
        self.prefit_workflow.parameter_entries = (
            self.prefit_tab.parameter_entries()
        )
        self.prefit_workflow.set_cluster_geometry_state(
            rows=cluster_geometry_rows,
            active_radii_type=active_radii_type,
            active_ionic_radius_type=active_ionic_radius_type,
            preserve_geometry_entries=preserve_geometry_entries,
        )
        self._capture_prefit_cluster_geometry_sync_snapshot(
            rows=cluster_geometry_rows,
            active_radii_type=active_radii_type,
            active_ionic_radius_type=active_ionic_radius_type,
        )
        self.prefit_tab.set_cluster_geometry_status_text(
            self.prefit_workflow.cluster_geometry_status_text()
        )

    def _restore_prefit_cluster_geometry_view_from_workflow(self) -> None:
        if (
            self.prefit_workflow is None
            or not self.prefit_workflow.supports_cluster_geometry_metadata()
        ):
            return
        self._pending_prefit_sf_approximation_change = None
        self._refresh_prefit_cluster_geometry_section()
        self.prefit_tab.populate_parameter_table(
            self.prefit_workflow.parameter_entries
        )
        self._set_prefit_sequence_baseline(
            self.prefit_workflow.parameter_entries
        )

    @Slot(str, str)
    def _on_prefit_cluster_geometry_sf_approximation_changed(
        self,
        previous_approximation: str,
        current_approximation: str,
    ) -> None:
        previous_value = str(previous_approximation).strip()
        current_value = str(current_approximation).strip()
        if (
            not previous_value
            or not current_value
            or previous_value == current_value
        ):
            self._pending_prefit_sf_approximation_change = None
            return
        self._pending_prefit_sf_approximation_change = (
            previous_value,
            current_value,
        )

    def _on_prefit_cluster_geometry_changed(self) -> None:
        previous_evaluation = self.prefit_tab.current_evaluation()
        try:
            self._sync_prefit_cluster_geometry_rows()
            self.prefit_tab.populate_parameter_table(
                self.prefit_workflow.parameter_entries
            )
            self._set_prefit_sequence_baseline(
                self.prefit_workflow.parameter_entries
            )
            self._invalidate_dream_workflow_cache()
            evaluation = self._load_prefit_preview()
            self._maybe_note_equivalent_sphere_shape_switch(
                previous_evaluation,
                evaluation,
            )
        except Exception as exc:
            self._restore_prefit_cluster_geometry_view_from_workflow()
            self._load_prefit_preview()
            self._show_error("Invalid cluster geometry radii", str(exc))

    def _on_prefit_cluster_geometry_radii_type_changed(
        self,
        radii_type: str,
    ) -> None:
        if (
            self.prefit_workflow is None
            or not self.prefit_workflow.supports_cluster_geometry_metadata()
        ):
            return
        try:
            self.prefit_workflow.parameter_entries = (
                self.prefit_tab.parameter_entries()
            )
            self.prefit_workflow.set_cluster_geometry_state(
                rows=self.prefit_tab.cluster_geometry_rows(),
                active_radii_type=radii_type,
                active_ionic_radius_type=(
                    self.prefit_tab.cluster_geometry_active_ionic_radius_type()
                ),
            )
            self.prefit_tab.populate_cluster_geometry_table(
                self.prefit_workflow.cluster_geometry_rows(),
                mapping_options=(
                    self.prefit_workflow.cluster_geometry_mapping_options()
                ),
                active_radii_type=(
                    self.prefit_workflow.cluster_geometry_active_radii_type()
                ),
                active_ionic_radius_type=(
                    self.prefit_workflow.cluster_geometry_active_ionic_radius_type()
                ),
                allowed_sf_approximations=(
                    self.prefit_workflow.allowed_cluster_geometry_approximations()
                ),
            )
            self.prefit_tab.populate_parameter_table(
                self.prefit_workflow.parameter_entries
            )
            self._capture_prefit_cluster_geometry_sync_snapshot()
            self._set_prefit_sequence_baseline(
                self.prefit_workflow.parameter_entries
            )
            self.prefit_tab.set_cluster_geometry_status_text(
                self.prefit_workflow.cluster_geometry_status_text()
            )
            self._invalidate_dream_workflow_cache()
            self._load_prefit_preview()
        except Exception as exc:
            self._restore_prefit_cluster_geometry_view_from_workflow()
            self._load_prefit_preview()
            self._show_error("Invalid cluster geometry radii", str(exc))

    def _on_prefit_cluster_geometry_ionic_radius_type_changed(
        self,
        ionic_radius_type: str,
    ) -> None:
        if (
            self.prefit_workflow is None
            or not self.prefit_workflow.supports_cluster_geometry_metadata()
        ):
            return
        try:
            self.prefit_workflow.parameter_entries = (
                self.prefit_tab.parameter_entries()
            )
            self.prefit_workflow.set_cluster_geometry_state(
                rows=self.prefit_tab.cluster_geometry_rows(),
                active_radii_type=(
                    self.prefit_tab.cluster_geometry_active_radii_type()
                ),
                active_ionic_radius_type=ionic_radius_type,
            )
            self.prefit_tab.populate_cluster_geometry_table(
                self.prefit_workflow.cluster_geometry_rows(),
                mapping_options=(
                    self.prefit_workflow.cluster_geometry_mapping_options()
                ),
                active_radii_type=(
                    self.prefit_workflow.cluster_geometry_active_radii_type()
                ),
                active_ionic_radius_type=(
                    self.prefit_workflow.cluster_geometry_active_ionic_radius_type()
                ),
                allowed_sf_approximations=(
                    self.prefit_workflow.allowed_cluster_geometry_approximations()
                ),
            )
            self.prefit_tab.populate_parameter_table(
                self.prefit_workflow.parameter_entries
            )
            self._capture_prefit_cluster_geometry_sync_snapshot()
            self._set_prefit_sequence_baseline(
                self.prefit_workflow.parameter_entries
            )
            self.prefit_tab.set_cluster_geometry_status_text(
                self.prefit_workflow.cluster_geometry_status_text()
            )
            self._invalidate_dream_workflow_cache()
            self._load_prefit_preview()
        except Exception as exc:
            self._restore_prefit_cluster_geometry_view_from_workflow()
            self._load_prefit_preview()
            self._show_error("Invalid cluster geometry radii", str(exc))

    def _load_prefit_preview(
        self,
        *,
        append_blocked_log: bool = False,
    ):
        if self.prefit_workflow is None:
            self.prefit_tab.plot_evaluation(None)
            self.prefit_tab.set_summary_text(
                "Build a SAXS project to preview the prefit model."
            )
            return None
        try:
            evaluation = self.prefit_workflow.evaluate()
        except Exception as exc:
            self.prefit_tab.plot_evaluation(None)
            self.prefit_tab.set_summary_text(
                "Prefit preview is waiting on template metadata.\n\n" f"{exc}"
            )
            if append_blocked_log:
                self.prefit_tab.append_log(
                    "Prefit preview is waiting on additional template "
                    "metadata before the model can be evaluated.\n"
                    f"{exc}"
                )
            return None
        self.prefit_tab.plot_evaluation(evaluation)
        self.prefit_tab.set_summary_text(
            self._format_prefit_summary(evaluation)
        )
        self._update_prefit_stoichiometry_status()
        self._maybe_append_scale_recommendation(
            self.prefit_workflow.parameter_entries
        )
        return evaluation

    def compute_prefit_cluster_geometry(self) -> None:
        if self.prefit_workflow is None:
            self._show_error(
                "Prefit unavailable",
                "Build a project and load its SAXS components first.",
            )
            return
        start_message = "Computing cluster geometry metadata..."
        self.prefit_tab.start_cluster_geometry_progress(start_message)
        self._show_progress_dialog(
            1,
            start_message,
            unit_label="files",
            title="SAXS Cluster Geometry Progress",
        )
        QApplication.processEvents()
        try:
            self.prefit_workflow.parameter_entries = (
                self.prefit_tab.parameter_entries()
            )
            self.prefit_workflow.set_cluster_geometry_active_radii_type(
                self.prefit_tab.cluster_geometry_active_radii_type()
            )
            self.prefit_workflow.set_cluster_geometry_active_ionic_radius_type(
                self.prefit_tab.cluster_geometry_active_ionic_radius_type()
            )
            table = self.prefit_workflow.compute_cluster_geometry_table_with_progress(
                progress_callback=self._update_prefit_cluster_geometry_progress
            )
            self.prefit_tab.populate_cluster_geometry_table(
                self.prefit_workflow.cluster_geometry_rows(),
                mapping_options=(
                    self.prefit_workflow.cluster_geometry_mapping_options()
                ),
                active_radii_type=(
                    self.prefit_workflow.cluster_geometry_active_radii_type()
                ),
                active_ionic_radius_type=(
                    self.prefit_workflow.cluster_geometry_active_ionic_radius_type()
                ),
                allowed_sf_approximations=(
                    self.prefit_workflow.allowed_cluster_geometry_approximations()
                ),
            )
            self.prefit_tab.populate_parameter_table(
                self.prefit_workflow.parameter_entries
            )
            self._capture_prefit_cluster_geometry_sync_snapshot()
            self._set_prefit_sequence_baseline(
                self.prefit_workflow.parameter_entries
            )
            self.prefit_tab.set_cluster_geometry_status_text(
                self.prefit_workflow.cluster_geometry_status_text()
            )
            self.prefit_tab.append_log(
                "Computed cluster geometry metadata.\n"
                f"Template: {self.prefit_workflow.template_spec.name}\n"
                f"Clusters: {len(table.rows)}\n"
                "Active radii mode: "
                f"{self.prefit_workflow.cluster_geometry_active_radii_type()}"
            )
            self._append_prefit_sequence_event(
                "cluster_geometry_computed",
                "Computed Prefit cluster geometry metadata.",
                details={
                    "clusters": len(table.rows),
                    "active_radii_mode": (
                        self.prefit_workflow.cluster_geometry_active_radii_type()
                    ),
                    "active_ionic_radius_type": (
                        self.prefit_workflow.cluster_geometry_active_ionic_radius_type()
                    ),
                },
                parameter_entries=self.prefit_workflow.parameter_entries,
            )
            self.prefit_tab.finish_cluster_geometry_progress(
                "Cluster geometry metadata ready."
            )
            self._invalidate_dream_workflow_cache()
            self._load_prefit_preview()
            self.statusBar().showMessage("Cluster geometry metadata computed")
        except Exception as exc:
            self.prefit_tab.finish_cluster_geometry_progress(
                "Cluster geometry metadata failed."
            )
            self._show_error(
                "Compute cluster geometry failed",
                str(exc),
            )
        finally:
            self._close_progress_dialog()

    def update_prefit_cluster_geometry(self) -> None:
        if self.prefit_workflow is None:
            self._show_error(
                "Prefit unavailable",
                "Build a project and load its SAXS components first.",
            )
            return
        try:
            self._sync_prefit_cluster_geometry_rows()
            self.prefit_tab.populate_cluster_geometry_table(
                self.prefit_workflow.cluster_geometry_rows(),
                mapping_options=(
                    self.prefit_workflow.cluster_geometry_mapping_options()
                ),
                active_radii_type=(
                    self.prefit_workflow.cluster_geometry_active_radii_type()
                ),
                active_ionic_radius_type=(
                    self.prefit_workflow.cluster_geometry_active_ionic_radius_type()
                ),
                allowed_sf_approximations=(
                    self.prefit_workflow.allowed_cluster_geometry_approximations()
                ),
            )
            self.prefit_tab.populate_parameter_table(
                self.prefit_workflow.parameter_entries
            )
            self._set_prefit_sequence_baseline(
                self.prefit_workflow.parameter_entries
            )
            self.prefit_tab.set_cluster_geometry_status_text(
                self.prefit_workflow.cluster_geometry_status_text()
            )
            self.prefit_tab.append_log(
                "Updated cluster geometry metadata.\n"
                f"Template: {self.prefit_workflow.template_spec.name}\n"
                f"Active radii mode: "
                f"{self.prefit_workflow.cluster_geometry_active_radii_type()}"
            )
            self._append_prefit_sequence_event(
                "cluster_geometry_updated",
                "Updated Prefit cluster geometry metadata.",
                details={
                    "active_radii_mode": (
                        self.prefit_workflow.cluster_geometry_active_radii_type()
                    ),
                    "active_ionic_radius_type": (
                        self.prefit_workflow.cluster_geometry_active_ionic_radius_type()
                    ),
                },
                parameter_entries=self.prefit_workflow.parameter_entries,
            )
            self._invalidate_dream_workflow_cache()
            self._load_prefit_preview()
            self.statusBar().showMessage("Cluster geometry metadata updated")
        except Exception as exc:
            self._restore_prefit_cluster_geometry_view_from_workflow()
            self._load_prefit_preview()
            self._show_error("Update cluster geometry failed", str(exc))

    def update_prefit_model(self) -> None:
        if self.prefit_workflow is None:
            self._show_error(
                "Prefit unavailable",
                "Build a project and load its SAXS components first.",
            )
            return
        try:
            self._flush_prefit_sequence_pending_manual_updates(
                trigger="update_model_preview",
            )
            previous_evaluation = self.prefit_tab.current_evaluation()
            self._sync_prefit_cluster_geometry_rows()
            if self.prefit_workflow.supports_cluster_geometry_metadata():
                entries = self.prefit_workflow.parameter_entries
                self.prefit_tab.populate_parameter_table(entries)
            else:
                entries = self.prefit_tab.parameter_entries()
            self.prefit_workflow.parameter_entries = entries
            evaluation = self.prefit_workflow.evaluate(entries)
            self.prefit_tab.plot_evaluation(evaluation)
            run_config = self.prefit_tab.run_config()
            self.prefit_tab.append_log(
                "Updated model preview.\n"
                f"Template: {self.prefit_workflow.template_spec.name}\n"
                f"Minimizer: {run_config.method}\n"
                f"Max nfev: {run_config.max_nfev}"
            )
            self.prefit_tab.set_summary_text(
                self._format_prefit_summary(evaluation)
            )
            self._update_prefit_stoichiometry_status()
            self._maybe_append_scale_recommendation(entries)
            self._maybe_note_equivalent_sphere_shape_switch(
                previous_evaluation,
                evaluation,
            )
        except Exception as exc:
            self._show_error("Update model failed", str(exc))

    def _maybe_note_equivalent_sphere_shape_switch(
        self,
        previous_evaluation,
        current_evaluation,
    ) -> None:
        pending_change = self._pending_prefit_sf_approximation_change
        self._pending_prefit_sf_approximation_change = None
        if pending_change is None or self.prefit_workflow is None:
            return
        if (
            self.prefit_workflow.template_spec.name
            not in EQUIVALENT_SPHERE_MIX_TEMPLATE_NAMES
        ):
            return
        if previous_evaluation is None or current_evaluation is None:
            return
        previous_model = np.asarray(
            previous_evaluation.model_intensities,
            dtype=float,
        )
        current_model = np.asarray(
            current_evaluation.model_intensities,
            dtype=float,
        )
        if previous_model.shape != current_model.shape:
            return
        if not np.allclose(
            previous_model,
            current_model,
            rtol=1e-5,
            atol=1e-8,
        ):
            return
        previous_shape, current_shape = pending_change
        self.prefit_tab.append_log(
            "Cluster-geometry shape switch left the Prefit model unchanged "
            "to plotting precision.\n"
            f"Changed approximation: {previous_shape} -> {current_shape}\n"
            "Reason: the active mixed hard-sphere/ellipsoid template is an "
            "equivalent-sphere approximation. Ellipsoid semiaxes are reduced "
            "to an equivalent-volume sphere radius before S(Q) is evaluated, "
            "so changing only the S.F. Approx. dropdown may leave the model "
            "curve unchanged when the effective interaction radius does not "
            "change."
        )
        self.statusBar().showMessage(
            "Shape switch preserved the equivalent-sphere interaction radius; "
            "the model curve stayed the same to plotting precision"
        )

    def run_prefit(self) -> None:
        if self.prefit_workflow is None:
            self._show_error(
                "Prefit unavailable",
                "Build a project and load its SAXS components first.",
            )
            return
        if not self.prefit_workflow.can_run_prefit():
            self._show_error(
                "Prefit unavailable",
                "Disable Model Only Mode and load experimental SAXS data to run a prefit.",
            )
            return
        try:
            self._sync_prefit_cluster_geometry_rows()
            config = self.prefit_tab.run_config()
            self._flush_prefit_sequence_pending_manual_updates(
                trigger="run_prefit",
            )
            entries = self.prefit_tab.parameter_entries()
            starting_entries = self._copy_prefit_entries(entries)
            varying_parameter_names = (
                self.prefit_workflow.grid_searchable_parameter_names(entries)
            )
            if (
                len(varying_parameter_names) > 3
                and self._warn_on_large_prefit_parameter_count
            ):
                continue_anyway, suppress_warning = (
                    self._confirm_large_prefit_parameter_count(
                        varying_parameter_names
                    )
                )
                if suppress_warning:
                    self._warn_on_large_prefit_parameter_count = False
                if not continue_anyway:
                    self.prefit_tab.append_log(
                        "Prefit canceled.\n"
                        "Reason: more than 3 independent parameters were "
                        "selected to vary, and the run was returned to the "
                        "parameter editor."
                    )
                    self.statusBar().showMessage("Prefit canceled")
                    return
            self.prefit_workflow.parameter_entries = entries
            self.prefit_tab.append_log(
                "Running prefit.\n"
                f"Template: {self.prefit_workflow.template_spec.name}\n"
                f"Minimizer: {config.method}\n"
                f"Max nfev: {config.max_nfev}\n"
                "Optimization strategy: "
                + self.prefit_workflow.optimization_strategy_preview(
                    entries,
                    method=config.method,
                )
                + "\n"
                "Autosave fit results: "
                + (
                    "enabled"
                    if self.prefit_workflow.settings.autosave_prefits
                    else "disabled"
                )
            )
            result = self.prefit_workflow.run_fit(
                entries,
                method=config.method,
                max_nfev=config.max_nfev,
            )
            self.prefit_tab.populate_parameter_table(result.parameter_entries)
            self._set_prefit_sequence_baseline(result.parameter_entries)
            self.prefit_tab.plot_evaluation(result.evaluation)
            self.prefit_tab.append_log(
                "Fit complete.\n"
                f"Minimizer: {result.method}\n"
                f"Optimization strategy: {result.optimization_strategy}\n"
                f"Grid evaluations: {result.grid_evaluations}\n"
                f"Max nfev request: {config.max_nfev}\n"
                f"R^2: {result.r_squared:.6g}\n"
                f"Reduced chi^2: {result.reduced_chi_square:.6g}\n"
                f"Function evals: {result.nfev}"
            )
            self.prefit_tab.set_summary_text(
                self._format_prefit_summary(
                    result.evaluation,
                    fit_result=result,
                    report_path=result.report_path,
                )
            )
            self._update_prefit_stoichiometry_status()
            self._refresh_saved_prefit_states(
                selected_name=(
                    result.report_path.parent.name
                    if result.report_path is not None
                    else None
                )
            )
            self._append_prefit_sequence_event(
                "prefit_run_complete",
                "Completed a Prefit refinement.",
                details={
                    "method": config.method,
                    "max_nfev": int(config.max_nfev),
                    "autosave_prefits": bool(
                        self.prefit_workflow.settings.autosave_prefits
                    ),
                    "optimization_strategy": result.optimization_strategy,
                    "grid_evaluations": int(result.grid_evaluations),
                    "varying_parameters": [
                        {
                            "structure": entry.structure,
                            "motif": entry.motif,
                            "parameter_name": entry.name,
                            "value": float(entry.value),
                            "minimum": float(entry.minimum),
                            "maximum": float(entry.maximum),
                        }
                        for entry in starting_entries
                        if bool(entry.vary)
                    ],
                    "statistics": {
                        "nfev": int(result.nfev),
                        "chi_square": float(result.chi_square),
                        "reduced_chi_square": float(result.reduced_chi_square),
                        "r_squared": float(result.r_squared),
                    },
                    "report_path": (
                        None
                        if result.report_path is None
                        else str(result.report_path)
                    ),
                },
                parameter_entries=result.parameter_entries,
            )
            self._load_dream_workflow()
            self.statusBar().showMessage("Prefit complete")
        except Exception as exc:
            self._show_error("Prefit failed", str(exc))

    def save_prefit(self) -> None:
        if self.prefit_workflow is None:
            self._show_error(
                "Prefit unavailable",
                "Build a project and load its SAXS components first.",
            )
            return
        if not self.prefit_workflow.can_run_prefit():
            self._show_error(
                "Save Prefit unavailable",
                "Disable Model Only Mode and load experimental SAXS data before saving a prefit report.",
            )
            return
        try:
            self._sync_prefit_cluster_geometry_rows()
            self._flush_prefit_sequence_pending_manual_updates(
                trigger="save_prefit",
            )
            entries = self.prefit_tab.parameter_entries()
            evaluation = self.prefit_workflow.evaluate(entries)
            config = self.prefit_tab.run_config()
            report_path = self.prefit_workflow.save_fit(
                entries,
                evaluation=evaluation,
                method=config.method,
                max_nfev=config.max_nfev,
                autosave_prefits=self.prefit_tab.autosave_checkbox.isChecked(),
            )
            self.prefit_tab.plot_evaluation(evaluation)
            self.prefit_tab.append_log(
                "Saved prefit state.\n"
                f"Template: {self.prefit_workflow.template_spec.name}\n"
                f"Minimizer: {config.method}\n"
                f"Max nfev: {config.max_nfev}\n"
                f"Saved report: {report_path}"
            )
            self.prefit_tab.set_summary_text(
                self._format_prefit_summary(
                    evaluation,
                    report_path=report_path,
                )
            )
            self._update_prefit_stoichiometry_status()
            self._refresh_saved_prefit_states(
                selected_name=report_path.parent.name
            )
            self._append_prefit_sequence_event(
                "prefit_state_saved",
                "Saved a Prefit snapshot.",
                details={
                    "method": config.method,
                    "max_nfev": int(config.max_nfev),
                    "autosave_prefits": bool(
                        self.prefit_tab.autosave_checkbox.isChecked()
                    ),
                    "report_path": str(report_path),
                    "snapshot_name": report_path.parent.name,
                },
                parameter_entries=entries,
            )
            self._set_prefit_sequence_baseline(entries)
            self._load_dream_workflow()
            self.statusBar().showMessage("Prefit saved")
        except Exception as exc:
            self._show_error("Save fit failed", str(exc))

    def save_prefit_plot_data(self) -> None:
        if self.prefit_workflow is None:
            self._show_error(
                "Save prefit plot data failed",
                "Build a project and load its SAXS components first.",
            )
            return
        if self.current_settings is None:
            self._show_error(
                "Save prefit plot data failed",
                "Load or build a project first.",
            )
            return
        try:
            entries = self.prefit_tab.parameter_entries()
            evaluation = self.prefit_workflow.evaluate(entries)
            config = self.prefit_tab.run_config()
            metadata = self._build_prefit_plot_export_metadata(
                entries=entries,
                evaluation=evaluation,
                method=config.method,
                max_nfev=config.max_nfev,
            )
            destination = self._prompt_project_plot_export_path(
                dialog_title="Export prefit plot data",
                default_filename=(
                    "prefit_plot_data_"
                    f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                ),
                export_kind="data",
            )
            if destination is None:
                return
            columns = ["q"]
            matrix_columns = [np.asarray(evaluation.q_values, dtype=float)]
            if evaluation.experimental_intensities is not None:
                columns.append("experimental_intensity")
                matrix_columns.append(
                    np.asarray(
                        evaluation.experimental_intensities,
                        dtype=float,
                    )
                )
            columns.append("model_intensity")
            matrix_columns.append(
                np.asarray(evaluation.model_intensities, dtype=float)
            )
            if evaluation.residuals is not None:
                columns.append("residual")
                matrix_columns.append(
                    np.asarray(evaluation.residuals, dtype=float)
                )
            columns.extend(["solvent_intensity", "solvent_contribution"])
            matrix_columns.extend(
                [
                    (
                        np.asarray(evaluation.solvent_intensities, dtype=float)
                        if evaluation.solvent_intensities is not None
                        else np.full_like(evaluation.q_values, np.nan)
                    ),
                    (
                        np.asarray(
                            evaluation.solvent_contribution,
                            dtype=float,
                        )
                        if evaluation.solvent_contribution is not None
                        else np.full_like(evaluation.q_values, np.nan)
                    ),
                ]
            )
            columns.append("structure_factor")
            matrix_columns.append(
                (
                    np.asarray(
                        evaluation.structure_factor_trace,
                        dtype=float,
                    )
                    if evaluation.structure_factor_trace is not None
                    else np.full_like(evaluation.q_values, np.nan)
                )
            )
            matrix = np.column_stack(matrix_columns)
            if destination.suffix.lower() == ".csv":
                self._write_prefit_plot_csv(
                    destination,
                    metadata=metadata,
                    columns=columns,
                    matrix=matrix,
                )
            else:
                np.save(destination, matrix)
                metadata_path = destination.with_suffix(".metadata.json")
                metadata_with_columns = dict(metadata)
                metadata_with_columns["columns"] = columns
                metadata_path.write_text(
                    json.dumps(metadata_with_columns, indent=2) + "\n",
                    encoding="utf-8",
                )
            self.prefit_tab.append_log(
                f"Saved prefit plot data to {destination}"
            )
            self.statusBar().showMessage("Prefit plot data exported")
        except Exception as exc:
            self._show_error("Save prefit plot data failed", str(exc))

    def reset_prefit_entries(self) -> None:
        if self.prefit_workflow is None:
            return
        self._flush_prefit_sequence_pending_manual_updates(
            trigger="reset_to_template",
        )
        self.prefit_workflow.parameter_entries = (
            self.prefit_workflow.load_template_reset_entries()
        )
        self.prefit_tab.populate_parameter_table(
            self.prefit_workflow.parameter_entries
        )
        self._set_prefit_sequence_baseline(
            self.prefit_workflow.parameter_entries
        )
        self.prefit_tab.append_log(
            "Reset parameter table to the template-default prefit preset "
            "saved in the project."
        )
        self._append_prefit_sequence_event(
            "prefit_parameters_reset_to_template",
            "Reset the Prefit parameter table to the template preset.",
            parameter_entries=self.prefit_workflow.parameter_entries,
        )
        self.update_prefit_model()

    def reset_single_prefit_parameter(
        self,
        structure: str,
        motif: str,
        parameter_name: str,
    ) -> None:
        if self.prefit_workflow is None:
            return
        try:
            self._flush_prefit_sequence_pending_manual_updates(
                trigger="reset_single_parameter",
            )
            reset_source_label = "Best Prefit / recycled preset"
            reset_entries = self.prefit_workflow.load_best_prefit_entries()
            if reset_entries is None:
                reset_entries = (
                    self.prefit_workflow.load_template_reset_entries()
                )
                reset_source_label = "template-default prefit preset"
            default_entry = next(
                (
                    entry
                    for entry in reset_entries
                    if (
                        entry.structure == structure
                        and entry.motif == motif
                        and entry.name == parameter_name
                    )
                ),
                None,
            )
            if default_entry is None:
                raise ValueError(
                    f"No template-default entry is available for {parameter_name}."
                )
            self.prefit_tab.set_parameter_row(
                parameter_name,
                structure=structure,
                motif=motif,
                value=default_entry.value,
                minimum=default_entry.minimum,
                maximum=default_entry.maximum,
                vary=default_entry.vary,
            )
            self.prefit_workflow.parameter_entries = (
                self.prefit_tab.parameter_entries()
            )
            self._set_prefit_sequence_baseline(
                self.prefit_workflow.parameter_entries
            )
            self.prefit_tab.append_log(
                "Reset individual parameter.\n"
                f"Parameter: {parameter_name}\n"
                f"Source: {reset_source_label}"
            )
            self._append_prefit_sequence_event(
                "prefit_parameter_reset",
                "Reset an individual Prefit parameter.",
                details={
                    "structure": structure,
                    "motif": motif,
                    "parameter_name": parameter_name,
                    "source": reset_source_label,
                },
                parameter_entries=self.prefit_workflow.parameter_entries,
            )
            self.update_prefit_model()
            self.statusBar().showMessage(f"Reset {parameter_name}")
        except Exception as exc:
            self._show_error("Reset parameter failed", str(exc))

    def set_best_prefit_parameters(self) -> None:
        if self.prefit_workflow is None:
            self._show_error(
                "Prefit unavailable",
                "Build a project and load its SAXS components first.",
            )
            return
        try:
            self._flush_prefit_sequence_pending_manual_updates(
                trigger="set_best_prefit",
            )
            entries = self.prefit_tab.parameter_entries()
            self.prefit_workflow.parameter_entries = entries
            self.prefit_workflow.save_best_prefit_entries(entries)
            if self.current_settings is not None:
                self.current_settings = self.prefit_workflow.settings
            synced_entries = self._sync_dream_parameter_map_from_prefit(
                source_label="Best Prefit preset",
            )
            self.prefit_tab.append_log(
                "Saved the current parameter table as the Best Prefit preset "
                "in the project file."
                + (
                    "\nUpdated the DREAM parameter map centers from the Best Prefit preset."
                    if synced_entries is not None
                    else ""
                )
            )
            self._append_prefit_sequence_event(
                "best_prefit_preset_saved",
                "Saved the current table as the Best Prefit preset.",
                details={
                    "dream_parameter_map_updated": synced_entries is not None,
                },
                parameter_entries=entries,
            )
            self._set_prefit_sequence_baseline(entries)
            self.statusBar().showMessage(
                "Best prefit preset saved"
                + (
                    " and DREAM parameter map updated"
                    if synced_entries is not None
                    else ""
                )
            )
        except Exception as exc:
            self._show_error("Save Best Prefit failed", str(exc))

    def reset_parameters_to_best_prefit(self) -> None:
        if self.prefit_workflow is None:
            self._show_error(
                "Prefit unavailable",
                "Build a project and load its SAXS components first.",
            )
            return
        try:
            self._flush_prefit_sequence_pending_manual_updates(
                trigger="reset_to_best_prefit",
            )
            entries = self.prefit_workflow.load_best_prefit_entries()
            if entries is None:
                raise ValueError(
                    "No Best Prefit preset is saved for the active template."
                )
            self.prefit_workflow.parameter_entries = entries
            self.prefit_tab.populate_parameter_table(entries)
            self._set_prefit_sequence_baseline(entries)
            self.prefit_tab.append_log(
                "Reset parameter table to the Best Prefit preset saved in "
                "the project."
            )
            self._append_prefit_sequence_event(
                "prefit_parameters_reset_to_best",
                "Reset the Prefit parameter table to the Best Prefit preset.",
                parameter_entries=entries,
            )
            self.update_prefit_model()
            self.statusBar().showMessage("Best prefit preset applied")
        except Exception as exc:
            self._show_error("Reset to Best Prefit failed", str(exc))

    def restore_prefit_state(self) -> None:
        if self.prefit_workflow is None:
            self._show_error(
                "Prefit unavailable",
                "Build a project and load its SAXS components first.",
            )
            return
        state_name = self.prefit_tab.selected_saved_state_name()
        if not state_name:
            self._show_error(
                "Restore Prefit State failed",
                "Select a saved prefit snapshot folder first.",
            )
            return
        try:
            self._flush_prefit_sequence_pending_manual_updates(
                trigger="restore_prefit_state",
            )
            saved_state = self.prefit_workflow.load_saved_state(state_name)
            if (
                saved_state.template_name
                != self.prefit_workflow.template_spec.name
            ):
                raise ValueError(
                    "The selected prefit snapshot was saved with template "
                    f"{saved_state.template_name}, but the active prefit "
                    f"template is {self.prefit_workflow.template_spec.name}. "
                    "Switch to the matching template before restoring this "
                    "snapshot so the Best Prefit preset remains untouched."
                )
            self.prefit_workflow.restore_cluster_geometry_table(
                saved_state.cluster_geometry_table
            )
            self.prefit_workflow.parameter_entries = (
                saved_state.parameter_entries
            )
            self.prefit_tab.populate_parameter_table(
                saved_state.parameter_entries
            )
            self._set_prefit_sequence_baseline(saved_state.parameter_entries)
            self._refresh_prefit_cluster_geometry_section()
            if saved_state.method and saved_state.max_nfev is not None:
                self.prefit_tab.set_run_config(
                    method=saved_state.method,
                    max_nfev=saved_state.max_nfev,
                )
            if saved_state.autosave_prefits is not None:
                self.prefit_workflow.settings.autosave_prefits = bool(
                    saved_state.autosave_prefits
                )
                if self.current_settings is not None:
                    self.current_settings.autosave_prefits = bool(
                        saved_state.autosave_prefits
                    )
                self.prefit_tab.set_autosave(saved_state.autosave_prefits)
            self.prefit_tab.append_log(
                "Restored prefit snapshot.\n"
                f"Snapshot: {saved_state.name}\n"
                f"Template: {saved_state.template_name}\n"
                f"Minimizer: {saved_state.method or self.prefit_tab.run_config().method}\n"
                "Max nfev: "
                f"{saved_state.max_nfev or self.prefit_tab.run_config().max_nfev}\n"
                "Best Prefit preset was not modified."
            )
            self._append_prefit_sequence_event(
                "prefit_state_restored",
                "Restored a saved Prefit snapshot.",
                details={
                    "snapshot_name": saved_state.name,
                    "template_name": saved_state.template_name,
                    "method": saved_state.method,
                    "max_nfev": saved_state.max_nfev,
                },
                parameter_entries=saved_state.parameter_entries,
            )
            self._invalidate_dream_workflow_cache()
            self.update_prefit_model()
            self.statusBar().showMessage("Prefit state restored")
        except Exception as exc:
            self._show_error("Restore Prefit State failed", str(exc))

    def save_dream_settings(self) -> None:
        try:
            workflow = self._load_dream_workflow()
            settings = self.dream_tab.settings_payload()
            if not settings.model_name:
                settings.model_name = self.prefit_workflow.template_spec.name
            active_path = workflow.save_settings(settings)
            self._active_dream_settings_snapshot = self._copy_dream_settings(
                settings
            )
            preset_name = self._prompt_dream_settings_preset_name(
                suggested_name=(
                    self.dream_tab.selected_settings_preset_name()
                    or settings.run_label
                    or f"dream_settings_{datetime.now():%Y%m%d_%H%M%S}"
                )
            )
            preset_path = None
            if preset_name:
                preset_path = workflow.save_settings_preset(
                    settings,
                    preset_name,
                )
            self._invalidate_written_dream_bundle()
            preset_names = workflow.list_settings_presets()
            self.dream_tab.set_available_settings_presets(
                preset_names,
                DreamTab.ACTIVE_SETTINGS_LABEL,
            )
            self.dream_tab.set_settings(settings, preset_name=None)
            self._current_dream_preset_name = None
            self.dream_tab.append_log(
                "Saved DREAM settings.\n"
                f"Active settings: {active_path}\n"
                "Preset: "
                + (
                    f"{preset_name} ({preset_path})"
                    if preset_name and preset_path is not None
                    else "no named preset created"
                )
            )
            self.statusBar().showMessage("DREAM settings saved")
        except Exception as exc:
            self._show_error("Save DREAM settings failed", str(exc))

    def open_distribution_editor(self) -> None:
        try:
            workflow = self._load_dream_workflow()
            has_saved_parameter_map = workflow.parameter_map_path.is_file()
            entries = workflow.load_parameter_map(persist_if_missing=False)
            if self.distribution_window is None:
                self.distribution_window = DistributionSetupWindow(
                    entries, self
                )
                self.distribution_window.saved.connect(
                    self._save_distribution_entries
                )
            self.distribution_window.load_entries(
                entries,
                has_existing_parameter_map=has_saved_parameter_map,
            )
            self.distribution_window.show()
            self.distribution_window.raise_()
            self.distribution_window.activateWindow()
        except Exception as exc:
            self._show_error("Open priors failed", str(exc))

    def write_dream_bundle(self) -> None:
        try:
            workflow = self._load_dream_workflow()
            settings = self.dream_tab.settings_payload()
            if not settings.model_name:
                settings.model_name = self.prefit_workflow.template_spec.name
            entries = workflow.load_parameter_map()
            self._append_dream_vary_recommendation(entries)
            bundle = workflow.create_runtime_bundle(
                settings=settings,
                entries=entries,
            )
            self._last_written_dream_bundle = bundle
            self.dream_tab.append_log(
                "Wrote DREAM runtime bundle.\n"
                f"Runtime script: {bundle.runtime_script_path}\n"
                f"Best-fit method: {settings.bestfit_method}\n"
                "Posterior filter: "
                f"{self._describe_posterior_filter(settings)}\n"
                f"Violin data mode: {settings.violin_parameter_mode}\n"
                f"Violin sample source: {settings.violin_sample_source}"
            )
            self.statusBar().showMessage("DREAM runtime bundle written")
        except Exception as exc:
            self._show_error("Write DREAM bundle failed", str(exc))

    def preview_dream_runtime_bundle(self) -> None:
        try:
            script_path = self._resolve_runtime_script_preview_path()
            opener = self._get_or_prompt_runtime_bundle_opener()
            if opener is None:
                self.statusBar().showMessage("Runtime bundle preview canceled")
                return
            self._launch_runtime_bundle_with_opener(script_path, opener)
            self.dream_tab.append_log(
                "Opened DREAM runtime bundle preview: "
                f"{script_path}\nApplication: {opener.label}"
            )
            self.statusBar().showMessage("Opened DREAM runtime bundle")
        except Exception as exc:
            self._show_error("Preview Runtime Bundle failed", str(exc))

    def _get_or_prompt_runtime_bundle_opener(
        self,
    ) -> RuntimeBundleOpener | None:
        if self.current_settings is None:
            raise ValueError(
                "Load a project before previewing a runtime bundle."
            )
        stored_value = (
            self.current_settings.runtime_bundle_opener or ""
        ).strip()
        if stored_value:
            opener = self._runtime_bundle_opener_from_stored_value(
                stored_value
            )
            if opener is not None:
                return opener
        opener = self._prompt_runtime_bundle_opener()
        if opener is None:
            return None
        self.current_settings.runtime_bundle_opener = opener.stored_value
        self._save_settings(self.current_settings)
        return opener

    def _prompt_runtime_bundle_opener(self) -> RuntimeBundleOpener | None:
        openers = self._available_runtime_bundle_openers()
        labels = [opener.label for opener in openers]
        custom_label = "Choose another application..."
        labels.append(custom_label)
        current_index = 0
        if self.current_settings is not None:
            current_value = (
                self.current_settings.runtime_bundle_opener or ""
            ).strip()
            for index, opener in enumerate(openers):
                if opener.stored_value == current_value:
                    current_index = index
                    break
        selected_label, accepted = QInputDialog.getItem(
            self,
            "Choose Runtime Bundle Opener",
            (
                "Select the application to use when previewing the DREAM "
                "runtime bundle for this project."
            ),
            labels,
            current=current_index,
            editable=False,
        )
        if not accepted or not str(selected_label).strip():
            return None
        if str(selected_label) == custom_label:
            return self._prompt_custom_runtime_bundle_opener()
        for opener in openers:
            if opener.label == str(selected_label):
                return opener
        return None

    def _prompt_custom_runtime_bundle_opener(
        self,
    ) -> RuntimeBundleOpener | None:
        if sys.platform == "darwin":
            selected = QFileDialog.getExistingDirectory(
                self,
                "Select application to open the DREAM runtime bundle",
                "/Applications",
            )
            if not selected:
                return None
            selected_path = Path(selected).expanduser().resolve()
            if selected_path.suffix != ".app":
                raise ValueError(
                    "Choose a macOS .app bundle when selecting a custom "
                    "runtime bundle opener."
                )
            return RuntimeBundleOpener(
                label=selected_path.stem,
                stored_value=str(selected_path),
                launch_target=str(selected_path),
                launch_mode="mac_app",
            )
        selected, _file_filter = QFileDialog.getOpenFileName(
            self,
            "Select application to open the DREAM runtime bundle",
            str(Path.home()),
            "Applications (*)",
        )
        if not selected:
            return None
        selected_path = Path(selected).expanduser().resolve()
        return RuntimeBundleOpener(
            label=selected_path.stem or selected_path.name,
            stored_value=str(selected_path),
            launch_target=str(selected_path),
            launch_mode="executable",
        )

    def _available_runtime_bundle_openers(self) -> list[RuntimeBundleOpener]:
        openers: list[RuntimeBundleOpener] = []
        if sys.platform == "darwin":
            candidate_paths = [
                ("TextEdit", "/System/Applications/TextEdit.app"),
                ("Visual Studio Code", "/Applications/Visual Studio Code.app"),
                ("Cursor", "/Applications/Cursor.app"),
                ("Sublime Text", "/Applications/Sublime Text.app"),
                ("BBEdit", "/Applications/BBEdit.app"),
                ("CotEditor", "/Applications/CotEditor.app"),
                ("Xcode", "/Applications/Xcode.app"),
                ("PyCharm CE", "/Applications/PyCharm CE.app"),
                ("PyCharm", "/Applications/PyCharm.app"),
                (
                    "PyCharm Community Edition",
                    "/Applications/PyCharm Community Edition.app",
                ),
            ]
            for label, raw_path in candidate_paths:
                app_path = Path(raw_path).expanduser()
                if not app_path.exists():
                    continue
                openers.append(
                    RuntimeBundleOpener(
                        label=label,
                        stored_value=str(app_path.resolve()),
                        launch_target=str(app_path.resolve()),
                        launch_mode="mac_app",
                    )
                )
            return openers

        command_candidates = [
            ("Visual Studio Code", "code"),
            ("Cursor", "cursor"),
            ("Sublime Text", "subl"),
            ("Kate", "kate"),
            ("Gedit", "gedit"),
            ("Mousepad", "mousepad"),
            ("Geany", "geany"),
            ("Xed", "xed"),
            ("Pluma", "pluma"),
            ("Notepad", "notepad"),
            ("Notepad++", "notepad++"),
        ]
        for label, command in command_candidates:
            executable = shutil.which(command)
            if executable is None:
                continue
            openers.append(
                RuntimeBundleOpener(
                    label=label,
                    stored_value=str(Path(executable).resolve()),
                    launch_target=str(Path(executable).resolve()),
                    launch_mode="executable",
                )
            )
        return openers

    def _runtime_bundle_opener_from_stored_value(
        self,
        stored_value: str,
    ) -> RuntimeBundleOpener | None:
        normalized = str(stored_value).strip()
        if not normalized:
            return None
        for opener in self._available_runtime_bundle_openers():
            if opener.stored_value == normalized:
                return opener
        opener_path = Path(normalized).expanduser()
        if opener_path.exists():
            launch_mode = (
                "mac_app"
                if sys.platform == "darwin" and opener_path.suffix == ".app"
                else "executable"
            )
            return RuntimeBundleOpener(
                label=opener_path.stem or opener_path.name,
                stored_value=str(opener_path.resolve()),
                launch_target=str(opener_path.resolve()),
                launch_mode=launch_mode,
            )
        executable = shutil.which(normalized)
        if executable is None:
            return None
        executable_path = Path(executable).resolve()
        return RuntimeBundleOpener(
            label=executable_path.stem or executable_path.name,
            stored_value=str(executable_path),
            launch_target=str(executable_path),
            launch_mode="executable",
        )

    def _launch_runtime_bundle_with_opener(
        self,
        script_path: Path,
        opener: RuntimeBundleOpener,
    ) -> None:
        if opener.launch_mode == "mac_app":
            subprocess.Popen(
                ["open", "-a", opener.launch_target, str(script_path)]
            )
            return
        subprocess.Popen([opener.launch_target, str(script_path)])

    def run_dream_bundle(self) -> None:
        try:
            if self._dream_task_thread is not None:
                self.statusBar().showMessage(
                    "A DREAM refinement is already running."
                )
                return
            workflow = self._load_dream_workflow()
            settings = self.dream_tab.settings_payload()
            if not settings.model_name:
                settings.model_name = self.prefit_workflow.template_spec.name
            entries = workflow.load_parameter_map()
            if not self._dream_parameter_map_saved_in_session:
                self.dream_tab.blink_edit_priors_button()
                self.dream_tab.append_log(
                    "Run DREAM blocked.\n"
                    "Review the priors in Edit Priors and click Save "
                    "Parameter Map before starting a DREAM refinement."
                )
                self.statusBar().showMessage(
                    "Edit and save the DREAM parameter map first"
                )
                self._append_dream_vary_recommendation(entries)
                return
            if (
                self._last_written_dream_bundle is None
                or not self._last_written_dream_bundle.run_dir.exists()
            ):
                self.dream_tab.blink_write_bundle_button()
                self.dream_tab.append_log(
                    "Run DREAM blocked.\n"
                    "Runtime Bundle not generated. Click Write Runtime "
                    "Bundle before running DREAM."
                )
                self._show_error(
                    "Runtime Bundle not generated",
                    "Runtime Bundle not generated. Click Write Runtime Bundle before running DREAM.",
                )
                return
            self._append_dream_vary_recommendation(entries)
            bundle = self._last_written_dream_bundle
            self.dream_tab.append_log(
                "Running DREAM.\n"
                f"Model name: {settings.model_name}\n"
                f"Chains: {settings.nchains}\n"
                f"Iterations: {settings.niterations}\n"
                f"Burn-in: {settings.burnin_percent}%\n"
                f"Best-fit method: {settings.bestfit_method}\n"
                "Posterior filter: "
                f"{self._describe_posterior_filter(settings)}\n"
                f"Violin data mode: {settings.violin_parameter_mode}\n"
                f"Violin sample source: {settings.violin_sample_source}"
            )
            self._start_dream_run_task(bundle, settings)
        except Exception as exc:
            self._show_error("Run DREAM failed", str(exc))

    def load_latest_results(self) -> None:
        try:
            self._load_dream_workflow()
            saved_runs = self._list_saved_dream_runs()
            latest_dir = saved_runs[0].run_dir
            self._load_dream_results_from_run_dir(latest_dir)
        except StopIteration:
            self._show_error(
                "Load results failed",
                "No DREAM sample files were found in the runtime bundle folder.",
            )
        except IndexError:
            self._show_error(
                "Load results failed",
                "No completed DREAM runs were found for the active project.",
            )
        except Exception as exc:
            self._show_error("Load results failed", str(exc))

    def load_selected_results(self) -> None:
        try:
            self._load_dream_workflow()
            selected_run_dir = self.dream_tab.selected_saved_run_dir()
            if selected_run_dir is None:
                raise FileNotFoundError(
                    "No completed DREAM runs were found for the active project."
                )
            self._load_dream_results_from_run_dir(selected_run_dir)
        except Exception as exc:
            self._show_error("Load results failed", str(exc))

    def preview_selected_dream_run(self) -> None:
        try:
            selected_run_dir = self.dream_tab.selected_saved_run_dir()
            if selected_run_dir is None:
                raise FileNotFoundError(
                    "No completed DREAM runs were found for the active project."
                )
            resolved_run_dir = Path(selected_run_dir).expanduser().resolve()
            run_settings = self._load_saved_dream_run_settings(
                resolved_run_dir
            )
            try:
                parameter_map_entries = self._load_saved_dream_parameter_map(
                    resolved_run_dir
                )
            except Exception:
                parameter_map_entries = []
            display_label = self.dream_tab.selected_saved_run_label()
            summary_text = self._format_saved_dream_run_preview(
                run_dir=resolved_run_dir,
                display_label=display_label,
                settings=run_settings,
                parameter_map_entries=parameter_map_entries,
            )
            if self._dream_saved_run_preview_dialog is None:
                self._dream_saved_run_preview_dialog = (
                    DreamSavedRunPreviewDialog(self)
                )
            self._dream_saved_run_preview_dialog.set_preview(
                display_label=display_label,
                summary_text=summary_text,
                parameter_map_entries=parameter_map_entries,
            )
            self._dream_saved_run_preview_dialog.show()
            self._dream_saved_run_preview_dialog.raise_()
            self._dream_saved_run_preview_dialog.activateWindow()
            self.statusBar().showMessage("Saved DREAM run preview opened")
        except Exception as exc:
            self._show_error("Preview saved run failed", str(exc))

    def _load_dream_results_from_run_dir(
        self,
        run_dir: str | Path,
    ) -> None:
        selected_run_dir = Path(run_dir).expanduser().resolve()
        settings_path = selected_run_dir / "pd_settings.json"
        parameter_map_path = selected_run_dir / "pd_param_map.json"
        run_settings = self._load_saved_dream_run_settings(selected_run_dir)
        display_settings = self._merge_dream_analysis_settings(
            run_settings,
            self.dream_tab.settings_payload(),
        )
        try:
            parameter_map_entries = self._load_saved_dream_parameter_map(
                selected_run_dir
            )
        except Exception:
            parameter_map_entries = []
        self._dream_refresh_timer.stop()
        self._pending_dream_refresh_scope = 0
        self._last_results_loader = None
        self._applied_dream_analysis_settings = None
        self._last_dream_summary = None
        self.dream_tab.set_settings(display_settings, preset_name=None)
        self._last_results_loader = SAXSDreamResultsLoader(
            selected_run_dir,
            burnin_percent=display_settings.burnin_percent,
        )
        if not parameter_map_entries:
            parameter_map_entries = [
                DreamParameterEntry.from_dict(dict(entry))
                for entry in self._last_results_loader.parameter_map_entries
                if isinstance(entry, dict)
            ]
        self.dream_tab.set_parameter_map_entries(parameter_map_entries)
        self._loaded_dream_run_dir = selected_run_dir
        self._refresh_saved_dream_runs(selected_run_dir=selected_run_dir)
        self._refresh_loaded_dream_results()
        self.dream_tab.append_log(
            "Loaded DREAM results for inspection.\n"
            f"Run directory: {selected_run_dir}\n"
            f"Saved settings: {settings_path if settings_path.is_file() else 'metadata fallback'}\n"
            "Saved prior parameter map: "
            + (
                str(parameter_map_path)
                if parameter_map_path.is_file()
                else "not found in run folder"
            )
            + "\n"
            "These settings and priors are shown for inspection only until "
            "you explicitly save them back to the active project."
        )
        self.statusBar().showMessage("DREAM results loaded")

    @staticmethod
    def _load_saved_dream_run_metadata(
        run_dir: str | Path,
    ) -> dict[str, object]:
        resolved_run_dir = Path(run_dir).expanduser().resolve()
        metadata_path = resolved_run_dir / "dream_runtime_metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(
                "The selected saved DREAM run does not contain "
                "dream_runtime_metadata.json."
            )
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(
                "The saved DREAM runtime metadata did not contain a JSON object."
            )
        return payload

    def _load_saved_dream_run_settings(
        self,
        run_dir: str | Path,
    ) -> DreamRunSettings:
        resolved_run_dir = Path(run_dir).expanduser().resolve()
        settings_path = resolved_run_dir / "pd_settings.json"
        if settings_path.is_file():
            settings = load_dream_settings(settings_path)
        else:
            metadata = self._load_saved_dream_run_metadata(resolved_run_dir)
            settings = DreamRunSettings.from_dict(
                dict(metadata.get("settings", {}))
            )
            if settings.model_name is None:
                template_name = str(metadata.get("template_name", "")).strip()
                settings.model_name = template_name or None
        return settings

    def _load_saved_dream_parameter_map(
        self,
        run_dir: str | Path,
    ) -> list[DreamParameterEntry]:
        resolved_run_dir = Path(run_dir).expanduser().resolve()
        parameter_map_path = resolved_run_dir / "pd_param_map.json"
        if parameter_map_path.is_file():
            return load_parameter_map(parameter_map_path)
        metadata = self._load_saved_dream_run_metadata(resolved_run_dir)
        payload = metadata.get("parameter_map")
        if payload is None:
            raise FileNotFoundError(
                "The selected saved DREAM run does not contain a saved prior "
                "parameter map."
            )
        return [
            DreamParameterEntry.from_dict(dict(entry))
            for entry in payload
            if isinstance(entry, dict)
        ]

    def _resolve_runtime_script_preview_path(self) -> Path:
        if (
            self._last_written_dream_bundle is not None
            and self._last_written_dream_bundle.runtime_script_path.is_file()
        ):
            return self._last_written_dream_bundle.runtime_script_path
        workflow = self._load_dream_workflow()
        run_dirs = sorted(
            workflow.dream_runtime_dir.glob("dream_*"),
            key=lambda path: path.name,
        )
        for run_dir in reversed(run_dirs):
            scripts = sorted(run_dir.glob("*.py"))
            if scripts:
                return scripts[0]
        raise FileNotFoundError(
            "No DREAM runtime bundle script was found. Write a runtime "
            "bundle before previewing it."
        )

    def save_dream_report(self) -> None:
        if self._last_results_loader is None:
            self._show_error(
                "Save report failed",
                "Load DREAM results first.",
            )
            return
        try:
            settings = self.dream_tab.settings_payload()
            summary, model_plot, _violin_plot, _plot_payload = (
                self._build_dream_export_context(settings)
            )
            export_dir = build_project_paths(
                self.current_settings.project_dir
            ).exported_data_dir
            export_dir.mkdir(parents=True, exist_ok=True)
            output_path = export_dir / (
                f"dream_statistics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            )
            output_path.write_text(
                self._build_dream_export_report_text(
                    export_kind="dream_statistics",
                    data_paths=[output_path],
                    settings=settings,
                    summary=summary,
                    model_plot=model_plot,
                    auto_generated=False,
                ),
                encoding="utf-8",
            )
            self.dream_tab.append_log(
                f"Saved DREAM statistics to {output_path}"
            )
            self.statusBar().showMessage("DREAM statistics saved")
        except Exception as exc:
            self._show_error("Save report failed", str(exc))

    def recycle_dream_output_to_prefit(self) -> None:
        if self._last_results_loader is None:
            self._show_error(
                "Recycle DREAM output failed",
                "Load DREAM results first.",
            )
            return
        if self.prefit_workflow is None:
            self._show_error(
                "Recycle DREAM output failed",
                "Load or build a project first so the Prefit workflow is available.",
            )
            return
        try:
            self._flush_prefit_sequence_pending_manual_updates(
                trigger="recycle_dream_output",
            )
            settings = self.dream_tab.settings_payload()
            filter_kwargs = self._dream_filter_kwargs(settings)
            summary = self._last_results_loader.get_summary(
                bestfit_method=settings.bestfit_method,
                **filter_kwargs,
            )
            current_entries = self.prefit_tab.parameter_entries()
            updated_entries = []
            summary_lookup = {
                str(name): float(summary.bestfit_params[index])
                for index, name in enumerate(summary.full_parameter_names)
            }
            matched_names: list[str] = []
            unmatched_prefit_names: list[str] = []
            for entry in current_entries:
                copied_entry = PrefitParameterEntry.from_dict(entry.to_dict())
                if copied_entry.name in summary_lookup:
                    copied_entry.value = float(
                        summary_lookup[copied_entry.name]
                    )
                    matched_names.append(copied_entry.name)
                else:
                    unmatched_prefit_names.append(copied_entry.name)
                updated_entries.append(copied_entry)
            if not matched_names:
                raise ValueError(
                    "The loaded DREAM result did not contain any parameters "
                    "that match the active Prefit table."
                )

            self.prefit_tab.populate_parameter_table(updated_entries)
            self.prefit_workflow.parameter_entries = list(updated_entries)
            self.prefit_workflow.save_best_prefit_entries(updated_entries)
            self._set_prefit_sequence_baseline(updated_entries)
            if self.current_settings is not None:
                self.current_settings = self.prefit_workflow.settings
            synced_entries = self._sync_dream_parameter_map_from_prefit(
                source_label="recycled DREAM best fit",
            )
            self._invalidate_dream_workflow_cache()
            self.tabs.setCurrentWidget(self.prefit_tab)
            self.prefit_tab.update_button.setFocus(
                Qt.FocusReason.OtherFocusReason
            )

            unmatched_dream_names = [
                str(name)
                for name in summary.full_parameter_names
                if str(name) not in {entry.name for entry in current_entries}
            ]
            log_lines = [
                "Recycled DREAM output into Prefit.",
                f"DREAM run: {summary.run_dir}",
                (
                    "Best-fit selection: "
                    f"{settings.bestfit_method} with "
                    f"{self._describe_posterior_filter(settings)}"
                ),
                (
                    "Matched Prefit parameters: "
                    f"{len(matched_names)} / {len(current_entries)}"
                ),
                (
                    "Prefit preview refresh: deferred to keep recycle "
                    "responsive. Click Update Model when you're ready to "
                    "rerender the Prefit plot."
                ),
                (
                    "The recycled values are now the active Best Prefit "
                    "preset and the single-parameter reset baseline."
                ),
            ]
            if synced_entries is not None:
                log_lines.append(
                    "Updated the DREAM parameter map centers from the "
                    "recycled Prefit values."
                )
            if (
                self.prefit_workflow.template_spec.name
                != self._last_results_loader.template_name
            ):
                log_lines.append(
                    "Template mismatch note: active Prefit template is "
                    f"{self.prefit_workflow.template_spec.name}, while the "
                    f"loaded DREAM run used {self._last_results_loader.template_name}. "
                    "Only overlapping parameter names were recycled."
                )
            if unmatched_prefit_names:
                preview = ", ".join(unmatched_prefit_names[:8])
                if len(unmatched_prefit_names) > 8:
                    preview += ", ..."
                log_lines.append(
                    "Prefit-only parameters left unchanged: " + preview
                )
            if unmatched_dream_names:
                preview = ", ".join(unmatched_dream_names[:8])
                if len(unmatched_dream_names) > 8:
                    preview += ", ..."
                log_lines.append(
                    "DREAM-only parameters not copied: " + preview
                )
            self.prefit_tab.append_log("\n".join(log_lines))
            self._append_prefit_sequence_event(
                "dream_recycled_to_prefit",
                "Recycled DREAM best-fit values into Prefit.",
                details={
                    "dream_run_dir": str(summary.run_dir),
                    "bestfit_method": settings.bestfit_method,
                    "posterior_filter": self._describe_posterior_filter(
                        settings
                    ),
                    "matched_parameter_names": list(matched_names),
                    "unmatched_prefit_names": list(unmatched_prefit_names),
                    "unmatched_dream_names": list(unmatched_dream_names),
                    "dream_parameter_map_updated": synced_entries is not None,
                },
                parameter_entries=updated_entries,
            )
            self.dream_tab.append_log(
                "Recycled the current DREAM best fit into the Prefit tab.\n"
                + "\n".join(log_lines[1:])
            )
            self.statusBar().showMessage(
                "DREAM output copied into Prefit; preview refresh deferred"
            )
        except Exception as exc:
            self._show_error("Recycle DREAM output failed", str(exc))

    def export_dream_model_report(self) -> None:
        if self._last_results_loader is None:
            self._show_error(
                "Export model report failed",
                "Load DREAM results first.",
            )
            return
        if self.current_settings is None:
            self._show_error(
                "Export model report failed",
                "Load or build a project first.",
            )
            return
        progress_started = False
        progress_total = 1
        wait_message = (
            "Generating DREAM model report PowerPoint. Please wait..."
        )
        try:
            settings = self.dream_tab.settings_payload()
            paths = build_project_paths(self.current_settings.project_dir)
            paths.reports_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = f"dream_model_report_{timestamp}"
            output_path = paths.reports_dir / f"{base_name}.pptx"
            asset_dir = paths.reports_dir / f"{base_name}_assets"
            self.dream_tab.append_log(wait_message)
            self.dream_tab.begin_progress(
                progress_total,
                wait_message,
                unit_label="steps",
            )
            self._show_dream_progress_dialog(
                wait_message,
                total=progress_total,
                unit_label="steps",
                title="DREAM Report Export",
            )
            self.statusBar().showMessage(wait_message)
            QApplication.processEvents()
            progress_started = True
            context = self._build_dream_model_report_context(
                settings=settings,
                output_path=output_path,
                asset_dir=asset_dir,
            )

            def _on_export_progress(
                processed: int,
                total: int,
                message: str,
            ) -> None:
                nonlocal progress_total
                progress_total = max(int(total), 1)
                self.dream_tab.update_progress(
                    processed,
                    progress_total,
                    message,
                    unit_label="steps",
                )
                if (
                    self._dream_progress_dialog is not None
                    and self._dream_progress_dialog.isVisible()
                ):
                    self._dream_progress_dialog.update_progress(
                        processed,
                        progress_total,
                        message,
                        unit_label="steps",
                    )
                self.statusBar().showMessage(message)
                QApplication.processEvents()

            result = export_dream_model_report_pptx(
                context,
                progress_callback=_on_export_progress,
            )
            self.dream_tab.finish_progress(
                "DREAM model report exported.",
                total=progress_total,
                unit_label="steps",
            )
            log_lines = [
                "Exported DREAM model report to:",
                f"{result.report_path}",
            ]
            if result.manifest_path is not None:
                log_lines.append(f"Manifest: {result.manifest_path}")
            if result.figure_paths:
                log_lines.append(
                    "The report assets folder contains the rendered figures "
                    "used to assemble the PowerPoint."
                )
            else:
                log_lines.append(
                    "Supplemental rendered figure assets were disabled for "
                    "this export."
                )
            self.dream_tab.append_log("\n".join(log_lines))
            self.statusBar().showMessage("DREAM model report exported")
        except Exception as exc:
            if progress_started:
                self.dream_tab.finish_progress(
                    "DREAM model report export failed.",
                    total=progress_total,
                    unit_label="steps",
                )
            self._show_error("Export model report failed", str(exc))
        finally:
            if progress_started:
                self._close_dream_progress_dialog()

    def _build_dream_export_context(
        self,
        settings: DreamRunSettings,
    ) -> tuple[object, object, object, dict[str, object]]:
        if self._last_results_loader is None:
            raise RuntimeError("Load DREAM results first.")
        filter_kwargs = self._dream_filter_kwargs(settings)
        summary = self._last_results_loader.get_summary(
            bestfit_method=settings.bestfit_method,
            **filter_kwargs,
        )
        model_plot = self._last_results_loader.build_model_fit_data(
            bestfit_method=settings.bestfit_method,
            **filter_kwargs,
        )
        violin_plot = self._last_results_loader.build_violin_data(
            mode=self._effective_dream_violin_mode(settings),
            sample_source=settings.violin_sample_source,
            weight_order=settings.violin_weight_order,
            **filter_kwargs,
        )
        plot_payload = self.dream_tab.prepare_violin_plot_payload(
            summary,
            violin_plot,
        )
        return summary, model_plot, violin_plot, plot_payload

    def _dream_parameter_summary_rows(
        self, summary: object
    ) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for index, name in enumerate(summary.full_parameter_names):
            rows.append(
                {
                    "name": str(name),
                    "selected_value": float(summary.bestfit_params[index]),
                    "map_value": float(summary.map_params[index]),
                    "chain_mean_value": float(
                        summary.chain_mean_params[index]
                    ),
                    "median_value": float(summary.median_params[index]),
                    "interval_low_value": float(
                        summary.interval_low_values[index]
                    ),
                    "interval_high_value": float(
                        summary.interval_high_values[index]
                    ),
                }
            )
        return rows

    def _dream_screening_metrics_payload(
        self,
        settings: DreamRunSettings,
        summary: object,
    ) -> dict[str, object]:
        payload = {
            "bestfit_method": str(settings.bestfit_method),
            "burnin_percent": int(settings.burnin_percent),
            "posterior_filter_mode": str(settings.posterior_filter_mode),
            "posterior_filter_description": self._describe_posterior_filter(
                settings
            ),
            "posterior_top_percent": float(settings.posterior_top_percent),
            "posterior_top_n": int(settings.posterior_top_n),
            "auto_select_best_posterior_filter": bool(
                settings.auto_select_best_posterior_filter
            ),
            "stoichiometry_target_elements_text": str(
                settings.stoichiometry_target_elements_text
            ),
            "stoichiometry_target_ratio_text": str(
                settings.stoichiometry_target_ratio_text
            ),
            "stoichiometry_filter_enabled": bool(
                settings.stoichiometry_filter_enabled
            ),
            "stoichiometry_tolerance_percent": float(
                settings.stoichiometry_tolerance_percent
            ),
            "posterior_candidate_sample_count": int(
                summary.posterior_candidate_sample_count
            ),
            "posterior_sample_count": int(summary.posterior_sample_count),
            "credible_interval_low": float(summary.credible_interval_low),
            "credible_interval_high": float(summary.credible_interval_high),
            "map_chain": int(summary.map_chain),
            "map_step": int(summary.map_step),
            "violin_parameter_mode": str(settings.violin_parameter_mode),
            "violin_sample_source": str(settings.violin_sample_source),
            "violin_weight_order": str(settings.violin_weight_order),
            "violin_value_scale_mode": str(settings.violin_value_scale_mode),
        }
        if self._last_dream_filter_assessments:
            payload["filter_assessments"] = [
                dict(assessment)
                for assessment in self._last_dream_filter_assessments
            ]
        if self._last_dream_filter_recommendation is not None:
            payload["recommended_posterior_filter"] = dict(
                self._last_dream_filter_recommendation
            )
        return payload

    def _dream_summary_payload(self, summary: object) -> dict[str, object]:
        return {
            "run_dir": str(summary.run_dir),
            "bestfit_method": str(summary.bestfit_method),
            "posterior_filter_mode": str(summary.posterior_filter_mode),
            "posterior_candidate_sample_count": int(
                summary.posterior_candidate_sample_count
            ),
            "posterior_sample_count": int(summary.posterior_sample_count),
            "credible_interval_low": float(summary.credible_interval_low),
            "credible_interval_high": float(summary.credible_interval_high),
            "stoichiometry_target": (
                None
                if summary.stoichiometry_target is None
                else {
                    "elements": list(summary.stoichiometry_target.elements),
                    "ratio": list(summary.stoichiometry_target.ratio),
                    "normalized_ratio": list(
                        summary.stoichiometry_target.normalized_ratio
                    ),
                }
            ),
            "stoichiometry_filter_enabled": bool(
                summary.stoichiometry_filter_enabled
            ),
            "stoichiometry_tolerance_percent": (
                None
                if summary.stoichiometry_tolerance_percent is None
                else float(summary.stoichiometry_tolerance_percent)
            ),
            "stoichiometry_evaluation": (
                None
                if summary.stoichiometry_evaluation is None
                else {
                    "element_totals": dict(
                        summary.stoichiometry_evaluation.element_totals
                    ),
                    "observed_ratio": (
                        None
                        if summary.stoichiometry_evaluation.observed_ratio
                        is None
                        else list(
                            summary.stoichiometry_evaluation.observed_ratio
                        )
                    ),
                    "deviation_percent_by_element": dict(
                        summary.stoichiometry_evaluation.deviation_percent_by_element
                    ),
                    "max_deviation_percent": (
                        None
                        if summary.stoichiometry_evaluation.max_deviation_percent
                        is None
                        else float(
                            summary.stoichiometry_evaluation.max_deviation_percent
                        )
                    ),
                    "is_valid": bool(
                        summary.stoichiometry_evaluation.is_valid
                    ),
                }
            ),
            "full_parameter_names": [
                str(name) for name in summary.full_parameter_names
            ],
            "active_parameter_names": [
                str(name) for name in summary.active_parameter_names
            ],
            "map_chain": int(summary.map_chain),
            "map_step": int(summary.map_step),
            "parameter_summary": self._dream_parameter_summary_rows(summary),
        }

    def _dream_export_metadata_payload(
        self,
        *,
        export_kind: str,
        data_paths: list[Path],
        settings: DreamRunSettings,
        summary: object,
        model_plot: object | None = None,
        violin_plot: object | None = None,
        plot_payload: dict[str, object] | None = None,
        auto_generated: bool,
    ) -> dict[str, object]:
        metadata: dict[str, object] = {
            "export_kind": export_kind,
            "exported_at": datetime.now().isoformat(),
            "auto_generated": bool(auto_generated),
            "purpose": (
                "Condensed user-facing DREAM export saved in "
                "exported_results/data for easy access. Full DREAM run "
                "artifacts and intermediate metadata remain in the DREAM "
                "run folder."
            ),
            "project_dir": (
                str(self.current_settings.project_dir)
                if self.current_settings is not None
                else ""
            ),
            "data_files": [str(path) for path in data_paths],
            "dream_settings": settings.to_dict(),
            "screening_metrics": self._dream_screening_metrics_payload(
                settings,
                summary,
            ),
            "summary": self._dream_summary_payload(summary),
        }
        if model_plot is not None:
            metadata["model_fit"] = {
                "template_name": str(model_plot.template_name),
                "bestfit_method": str(model_plot.bestfit_method),
                "point_count": int(np.asarray(model_plot.q_values).size),
                "includes_solvent_contribution": bool(
                    model_plot.solvent_contribution is not None
                ),
                "includes_structure_factor": bool(
                    model_plot.structure_factor_trace is not None
                ),
                "fit_metrics": {
                    "rmse": float(model_plot.rmse),
                    "mean_abs_residual": float(model_plot.mean_abs_residual),
                    "r_squared": float(model_plot.r_squared),
                },
            }
        if violin_plot is not None:
            metadata["violin_plot"] = {
                "parameter_names": [
                    str(name) for name in violin_plot.parameter_names
                ],
                "display_names": [
                    str(name) for name in violin_plot.display_names
                ],
                "mode": str(violin_plot.mode),
                "sample_source": str(violin_plot.sample_source),
                "sample_count": int(violin_plot.sample_count),
                "weight_order": str(violin_plot.weight_order),
                "sample_matrix_shape": list(
                    np.asarray(violin_plot.samples).shape
                ),
            }
        if plot_payload is not None:
            y_limits = plot_payload.get("y_limits")
            metadata["plot_view"] = {
                "display_names": [
                    str(name) for name in plot_payload.get("display_names", [])
                ],
                "selected_values": [
                    float(value)
                    for value in np.asarray(
                        plot_payload.get("selected_values", []),
                        dtype=float,
                    )
                ],
                "interval_low_values": [
                    float(value)
                    for value in np.asarray(
                        plot_payload.get("interval_low_values", []),
                        dtype=float,
                    )
                ],
                "interval_high_values": [
                    float(value)
                    for value in np.asarray(
                        plot_payload.get("interval_high_values", []),
                        dtype=float,
                    )
                ],
                "ylabel": str(plot_payload.get("ylabel", "")),
                "title": str(plot_payload.get("title", "")),
                "y_limits": (
                    None
                    if y_limits is None
                    else [float(y_limits[0]), float(y_limits[1])]
                ),
            }
        return metadata

    def _build_dream_export_report_text(
        self,
        *,
        export_kind: str,
        data_paths: list[Path],
        settings: DreamRunSettings,
        summary: object,
        model_plot: object | None = None,
        violin_plot: object | None = None,
        auto_generated: bool,
    ) -> str:
        lines = [
            "DREAM condensed export",
            (
                "This export is an easy-to-find copy saved in "
                "exported_results/data."
            ),
            (
                "Full DREAM run artifacts and intermediate metadata remain "
                "in the DREAM run folder."
            ),
            f"Export kind: {export_kind}",
            f"Auto-generated after refinement: {'yes' if auto_generated else 'no'}",
            "Exported data files:",
        ]
        lines.extend(f"  {path}" for path in data_paths)
        lines.extend(
            [
                "",
                self._format_dream_summary(summary, settings=settings),
            ]
        )
        if model_plot is not None:
            lines.extend(
                [
                    "",
                    "Model fit metrics:",
                    f"  RMSE: {model_plot.rmse:.6g}",
                    (
                        "  Mean absolute residual: "
                        f"{model_plot.mean_abs_residual:.6g}"
                    ),
                    f"  R^2: {model_plot.r_squared:.6g}",
                ]
            )
        if violin_plot is not None:
            lines.extend(
                [
                    "",
                    "Posterior violin data:",
                    f"  Mode: {violin_plot.mode}",
                    f"  Sample source: {violin_plot.sample_source}",
                    f"  Weight order: {violin_plot.weight_order}",
                    f"  Sample count: {violin_plot.sample_count}",
                ]
            )
        if self._last_dream_filter_assessments:
            lines.extend(["", "Posterior filter assessment:"])
            for assessment in self._last_dream_filter_assessments:
                lines.append(
                    "  "
                    f"{assessment['description']}: "
                    f"RMSE={assessment['rmse']:.6g}, "
                    f"Mean |res|={assessment['mean_abs_residual']:.6g}, "
                    f"R^2={assessment['r_squared']:.6g}, "
                    f"samples={assessment['posterior_sample_count']}"
                )
            if self._last_dream_filter_recommendation is not None:
                lines.append(
                    "  Recommended: "
                    f"{self._last_dream_filter_recommendation['description']}"
                )
        return "\n".join(lines) + "\n"

    def _write_dream_export_sidecars(
        self,
        *,
        base_path: Path,
        data_paths: list[Path],
        settings: DreamRunSettings,
        summary: object,
        model_plot: object | None = None,
        violin_plot: object | None = None,
        plot_payload: dict[str, object] | None = None,
        auto_generated: bool,
        export_kind: str,
    ) -> list[Path]:
        metadata_path = base_path.parent / f"{base_path.name}.metadata.json"
        report_path = base_path.parent / f"{base_path.name}.report.txt"
        metadata_path.write_text(
            json.dumps(
                self._dream_export_metadata_payload(
                    export_kind=export_kind,
                    data_paths=data_paths,
                    settings=settings,
                    summary=summary,
                    model_plot=model_plot,
                    violin_plot=violin_plot,
                    plot_payload=plot_payload,
                    auto_generated=auto_generated,
                ),
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        report_path.write_text(
            self._build_dream_export_report_text(
                export_kind=export_kind,
                data_paths=data_paths,
                settings=settings,
                summary=summary,
                model_plot=model_plot,
                violin_plot=violin_plot,
                auto_generated=auto_generated,
            ),
            encoding="utf-8",
        )
        return [metadata_path, report_path]

    def _export_dream_model_fit_bundle(
        self,
        *,
        base_path: Path,
        settings: DreamRunSettings,
        summary: object,
        model_plot: object,
        auto_generated: bool,
    ) -> list[Path]:
        output_path = base_path.parent / f"{base_path.name}.csv"
        solvent_contribution = (
            np.asarray(model_plot.solvent_contribution, dtype=float)
            if model_plot.solvent_contribution is not None
            else np.full_like(
                np.asarray(model_plot.q_values, dtype=float),
                np.nan,
                dtype=float,
            )
        )
        structure_factor = (
            np.asarray(model_plot.structure_factor_trace, dtype=float)
            if model_plot.structure_factor_trace is not None
            else np.full_like(
                np.asarray(model_plot.q_values, dtype=float),
                np.nan,
                dtype=float,
            )
        )
        np.savetxt(
            output_path,
            np.column_stack(
                [
                    model_plot.q_values,
                    model_plot.experimental_intensities,
                    model_plot.model_intensities,
                    solvent_contribution,
                    structure_factor,
                ]
            ),
            delimiter=",",
            header=(
                "q,experimental_intensity,model_intensity,"
                "solvent_contribution,structure_factor"
            ),
            comments="",
        )
        return [
            output_path,
            *self._write_dream_export_sidecars(
                base_path=base_path,
                data_paths=[output_path],
                settings=settings,
                summary=summary,
                model_plot=model_plot,
                auto_generated=auto_generated,
                export_kind="dream_model_fit",
            ),
        ]

    def _export_dream_violin_bundle(
        self,
        *,
        base_path: Path,
        settings: DreamRunSettings,
        summary: object,
        violin_plot: object,
        plot_payload: dict[str, object],
        save_csv: bool,
        save_pkl: bool,
        auto_generated: bool,
    ) -> list[Path]:
        saved_paths: list[Path] = []
        if save_csv:
            csv_output_path = base_path.parent / f"{base_path.name}.csv"
            with csv_output_path.open(
                "w",
                encoding="utf-8",
                newline="",
            ) as handle:
                writer = csv.writer(handle)
                writer.writerow(plot_payload["display_names"])
                writer.writerows(
                    np.asarray(plot_payload["samples"], dtype=float)
                )
            saved_paths.append(csv_output_path)
        if save_pkl:
            pkl_output_path = base_path.parent / f"{base_path.name}.pkl"
            payload = {
                "exported_at": datetime.now().isoformat(),
                "project_dir": str(self.current_settings.project_dir),
                "settings": settings.to_dict(),
                "screening_metrics": self._dream_screening_metrics_payload(
                    settings,
                    summary,
                ),
                "summary": self._dream_summary_payload(summary),
                "violin_plot": {
                    "parameter_names": list(violin_plot.parameter_names),
                    "display_names": list(violin_plot.display_names),
                    "mode": violin_plot.mode,
                    "sample_source": violin_plot.sample_source,
                    "sample_count": violin_plot.sample_count,
                    "weight_order": violin_plot.weight_order,
                },
                "plot_payload": {
                    "display_names": list(plot_payload["display_names"]),
                    "samples": np.asarray(
                        plot_payload["samples"],
                        dtype=float,
                    ),
                    "selected_values": np.asarray(
                        plot_payload["selected_values"],
                        dtype=float,
                    ),
                    "interval_low_values": np.asarray(
                        plot_payload["interval_low_values"],
                        dtype=float,
                    ),
                    "interval_high_values": np.asarray(
                        plot_payload["interval_high_values"],
                        dtype=float,
                    ),
                    "ylabel": str(plot_payload["ylabel"]),
                    "title": str(plot_payload["title"]),
                    "y_limits": plot_payload["y_limits"],
                },
            }
            with pkl_output_path.open("wb") as handle:
                pickle.dump(payload, handle)
            saved_paths.append(pkl_output_path)
        if not saved_paths:
            return []
        return [
            *saved_paths,
            *self._write_dream_export_sidecars(
                base_path=base_path,
                data_paths=saved_paths,
                settings=settings,
                summary=summary,
                violin_plot=violin_plot,
                plot_payload=plot_payload,
                auto_generated=auto_generated,
                export_kind="dream_violin",
            ),
        ]

    def _auto_export_dream_condensed_outputs(
        self,
        settings: DreamRunSettings,
    ) -> list[Path]:
        if self.current_settings is None:
            return []
        export_dir = build_project_paths(
            self.current_settings.project_dir
        ).exported_data_dir
        export_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary, model_plot, violin_plot, plot_payload = (
            self._build_dream_export_context(settings)
        )
        model_base = export_dir / (
            f"dream_model_fit_auto_{settings.bestfit_method}_{timestamp}"
        )
        violin_base = export_dir / (
            "dream_violin_auto_"
            f"{self._effective_dream_violin_mode(settings)}_"
            f"{settings.violin_weight_order}_{timestamp}"
        )
        saved_paths = self._export_dream_model_fit_bundle(
            base_path=model_base,
            settings=settings,
            summary=summary,
            model_plot=model_plot,
            auto_generated=True,
        )
        saved_paths.extend(
            self._export_dream_violin_bundle(
                base_path=violin_base,
                settings=settings,
                summary=summary,
                violin_plot=violin_plot,
                plot_payload=plot_payload,
                save_csv=True,
                save_pkl=False,
                auto_generated=True,
            )
        )
        return saved_paths

    def _evaluate_dream_posterior_filters(
        self,
        settings: DreamRunSettings,
    ) -> tuple[list[dict[str, object]], dict[str, object] | None]:
        if self._last_results_loader is None:
            return [], None
        assessments: list[dict[str, object]] = []
        for mode in (
            "all_post_burnin",
            "top_percent_logp",
            "top_n_logp",
        ):
            candidate_settings = self._copy_dream_settings(settings)
            candidate_settings.posterior_filter_mode = mode
            filter_kwargs = self._dream_filter_kwargs(candidate_settings)
            summary = self._last_results_loader.get_summary(
                bestfit_method=candidate_settings.bestfit_method,
                **filter_kwargs,
            )
            model_plot = self._last_results_loader.build_model_fit_data(
                bestfit_method=candidate_settings.bestfit_method,
                **filter_kwargs,
            )
            assessments.append(
                {
                    "mode": mode,
                    "description": self._describe_posterior_filter(
                        candidate_settings
                    ),
                    "rmse": float(model_plot.rmse),
                    "mean_abs_residual": float(model_plot.mean_abs_residual),
                    "r_squared": float(model_plot.r_squared),
                    "posterior_sample_count": int(
                        summary.posterior_sample_count
                    ),
                }
            )
        if not assessments:
            return [], None
        recommended = min(
            assessments,
            key=lambda assessment: (
                float(assessment["rmse"]),
                float(assessment["mean_abs_residual"]),
                -float(assessment["r_squared"]),
            ),
        )
        return assessments, dict(recommended)

    def _assess_and_apply_dream_filter_recommendation(
        self,
        settings: DreamRunSettings,
    ) -> tuple[DreamRunSettings, str | None]:
        assessments, recommendation = self._evaluate_dream_posterior_filters(
            settings
        )
        self._last_dream_filter_assessments = assessments
        self._last_dream_filter_recommendation = recommendation
        if not assessments or recommendation is None:
            return settings, None

        lines = [
            "DREAM posterior filter assessment.",
            (
                "Evaluated All Post-burnin, Top % by Log-posterior, and "
                "Top N by Log-posterior using the current best-fit method "
                f"and the default thresholds Top % = "
                f"{settings.posterior_top_percent:g}, Top N = "
                f"{settings.posterior_top_n}."
            ),
        ]
        for assessment in assessments:
            lines.append(
                f"{assessment['description']}: "
                f"RMSE={assessment['rmse']:.6g}, "
                f"Mean |res|={assessment['mean_abs_residual']:.6g}, "
                f"R^2={assessment['r_squared']:.6g}, "
                f"samples={assessment['posterior_sample_count']}"
            )
        lines.append(
            "Recommended posterior filter by fit quality: "
            f"{recommendation['description']}"
        )

        updated_settings = self._copy_dream_settings(settings)
        if bool(settings.auto_select_best_posterior_filter):
            if (
                str(recommendation["mode"])
                != updated_settings.posterior_filter_mode
            ):
                updated_settings.search_filter_preset = "custom"
            updated_settings.posterior_filter_mode = str(
                recommendation["mode"]
            )
            lines.append("Applied recommended posterior filter automatically.")
        else:
            lines.append(
                "Automatic posterior filter selection is off, so the "
                "current filter setting was left unchanged."
            )
        return updated_settings, "\n".join(lines)

    def save_dream_model_fit(self) -> None:
        if self._last_results_loader is None:
            self._show_error(
                "Save model fit failed",
                "Load DREAM results first.",
            )
            return
        if self.current_settings is None:
            self._show_error(
                "Save model fit failed",
                "Load or build a project first.",
            )
            return
        try:
            settings = self.dream_tab.settings_payload()
            paths = build_project_paths(self.current_settings.project_dir)
            paths.exported_data_dir.mkdir(parents=True, exist_ok=True)
            base_path = paths.exported_data_dir / (
                "dream_model_fit_"
                f"{settings.bestfit_method}_"
                f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )
            base_path = base_path.with_suffix("")
            summary, model_plot, _violin_plot, _plot_payload = (
                self._build_dream_export_context(settings)
            )
            saved_paths = self._export_dream_model_fit_bundle(
                base_path=base_path,
                settings=settings,
                summary=summary,
                model_plot=model_plot,
                auto_generated=False,
            )
            self.dream_tab.append_log(
                "Exported DREAM model fit bundle to:\n"
                + "\n".join(str(path) for path in saved_paths)
                + "\nThis condensed export lives in exported_results/data. "
                "Full DREAM run artifacts remain in the DREAM run folder."
            )
            self.statusBar().showMessage("DREAM model fit exported")
        except Exception as exc:
            self._show_error("Save model fit failed", str(exc))

    def save_dream_violin_data(self) -> None:
        if self._last_results_loader is None:
            self._show_error(
                "Save violin data failed",
                "Load DREAM results first.",
            )
            return
        if self.current_settings is None:
            self._show_error(
                "Save violin data failed",
                "Load or build a project first.",
            )
            return
        try:
            settings = self.dream_tab.settings_payload()
            paths = build_project_paths(self.current_settings.project_dir)
            paths.exported_data_dir.mkdir(parents=True, exist_ok=True)
            base_name = (
                "dream_violin_"
                f"{self._effective_dream_violin_mode(settings)}_"
                f"{settings.violin_weight_order}_"
                f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            dialog = DreamViolinExportDialog(
                default_output_dir=paths.exported_data_dir,
                default_base_name=base_name,
                parent=self,
            )
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return
            export_options = dialog.selected_options
            if export_options is None:
                return
            export_options.output_dir.mkdir(parents=True, exist_ok=True)
            summary, _model_plot, violin_plot, plot_payload = (
                self._build_dream_export_context(settings)
            )
            saved_paths = self._export_dream_violin_bundle(
                base_path=(
                    export_options.output_dir / export_options.base_name
                ),
                settings=settings,
                summary=summary,
                violin_plot=violin_plot,
                plot_payload=plot_payload,
                save_csv=bool(export_options.save_csv),
                save_pkl=bool(export_options.save_pkl),
                auto_generated=False,
            )
            self.dream_tab.append_log(
                "Exported DREAM violin data bundle to:\n"
                + "\n".join(str(path) for path in saved_paths)
                + "\nThis condensed export lives in exported_results/data. "
                "Full DREAM run artifacts remain in the DREAM run folder."
            )
            self.statusBar().showMessage("DREAM violin data exported")
        except Exception as exc:
            self._show_error("Save violin data failed", str(exc))

    def _build_dream_model_report_context(
        self,
        *,
        settings: DreamRunSettings,
        output_path: Path,
        asset_dir: Path,
    ) -> DreamModelReportContext:
        if self.current_settings is None or self._last_results_loader is None:
            raise RuntimeError("Load DREAM results first.")
        powerpoint_settings = self._effective_powerpoint_export_settings()
        summary, model_plot, violin_plot, plot_payload = (
            self._build_dream_export_context(settings)
        )
        project_paths = build_project_paths(self.current_settings.project_dir)
        prefit_entries = tuple(
            self.prefit_tab.parameter_entries()
            if self.prefit_workflow is not None
            else []
        )
        prefit_evaluation = None
        if self.prefit_workflow is not None:
            try:
                prefit_evaluation = self.prefit_workflow.evaluate(
                    list(prefit_entries)
                )
            except Exception:
                prefit_evaluation = None
        prefit_statistics = self._latest_prefit_statistics_payload()
        assessments = tuple(self._report_dream_filter_assessments(settings))
        filter_views = self._build_dream_report_filter_views(settings)
        prior_requests = tuple(
            self._build_report_prior_requests(powerpoint_settings)
        )
        q_values = np.asarray(model_plot.q_values, dtype=float)
        q_range_text = self._format_selected_q_range_text(q_values)
        supported_q_range = load_built_component_q_range(
            self.current_settings.project_dir,
            include_predicted_structures=(
                self.current_settings.use_predicted_structure_weights
            ),
            component_dir=project_artifact_paths(
                self.current_settings
            ).component_dir,
        )
        supported_q_range_text = (
            None
            if supported_q_range is None
            else (
                f"{float(supported_q_range[0]):.6g} to "
                f"{float(supported_q_range[1]):.6g}"
            )
        )
        (
            template_display_name,
            template_module_path,
            model_equation_text,
            model_context_lines,
            model_definition_lines,
            model_reference_lines,
        ) = self._build_report_template_details(
            template_name=str(model_plot.template_name),
            q_range_text=q_range_text,
            supported_q_range_text=supported_q_range_text,
            q_sampling_text=self._report_q_sampling_text(),
            prefit_parameter_count=len(prefit_entries),
            dream_active_parameter_names=tuple(
                str(name).strip()
                for name in summary.active_parameter_names
                if str(name).strip()
            ),
            includes_solvent=bool(model_plot.solvent_contribution is not None),
            includes_structure_factor=bool(
                model_plot.structure_factor_trace is not None
            ),
        )
        output_summary_lines = [
            f"Report file: {output_path}",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Template: {model_plot.template_name}",
            f"Best-fit method: {settings.bestfit_method}",
            (
                "Posterior filter: "
                f"{self._describe_posterior_filter(settings)}"
            ),
            f"Posterior samples kept: {summary.posterior_sample_count}",
            f"DREAM RMSE: {model_plot.rmse:.6g}",
            f"DREAM Mean |res|: {model_plot.mean_abs_residual:.6g}",
            f"DREAM R^2: {model_plot.r_squared:.6g}",
        ]
        if (
            prefit_evaluation is not None
            and prefit_evaluation.experimental_intensities is not None
        ):
            prefit_metrics = self._fit_quality_metrics_from_curves(
                np.asarray(
                    prefit_evaluation.experimental_intensities,
                    dtype=float,
                ),
                np.asarray(prefit_evaluation.model_intensities, dtype=float),
            )
            output_summary_lines.extend(
                [
                    f"Prefit RMSE: {prefit_metrics.rmse:.6g}",
                    f"Prefit Mean |res|: {prefit_metrics.mean_abs_residual:.6g}",
                    f"Prefit R^2: {prefit_metrics.r_squared:.6g}",
                ]
            )
        directory_lines = [
            f"Project directory: {project_paths.project_dir}",
            f"Exported data directory: {project_paths.exported_data_dir}",
            f"Exported plots directory: {project_paths.exported_plots_dir}",
            (
                "Prefit directory: "
                f"{project_artifact_paths(self.current_settings).prefit_dir}"
            ),
            f"DREAM run directory: {summary.run_dir}",
            (
                "Prior weights source: "
                f"{self._report_prior_json_path() or 'Not available'}"
            ),
        ]
        if (
            powerpoint_settings.export_figure_assets
            or powerpoint_settings.generate_manifest
        ):
            directory_lines.insert(0, f"Report assets: {asset_dir}")
        if powerpoint_settings.generate_manifest:
            directory_lines.insert(
                1,
                f"Report manifest: {asset_dir / 'report_manifest.json'}",
            )
        return DreamModelReportContext(
            output_path=output_path,
            asset_dir=asset_dir,
            project_name=self.current_settings.project_name,
            project_dir=Path(self.current_settings.project_dir).resolve(),
            generated_at=datetime.now(),
            powerpoint_settings=powerpoint_settings,
            user_q_range_text=q_range_text,
            supported_q_range_text=supported_q_range_text,
            q_sampling_text=self._report_q_sampling_text(),
            template_name=str(model_plot.template_name),
            template_display_name=template_display_name,
            template_module_path=template_module_path,
            model_equation_text=model_equation_text,
            model_context_lines=tuple(model_context_lines),
            model_definition_lines=tuple(model_definition_lines),
            model_reference_lines=tuple(model_reference_lines),
            prior_histograms=prior_requests,
            component_plot_without_solvent=self._build_report_component_plot_data(
                include_solvent=False,
                powerpoint_settings=powerpoint_settings,
            ),
            component_plot_with_solvent=self._build_report_component_plot_data(
                include_solvent=True,
                powerpoint_settings=powerpoint_settings,
            ),
            prefit_evaluation=prefit_evaluation,
            prefit_parameter_entries=prefit_entries,
            prefit_statistics=prefit_statistics,
            cluster_geometry_rows=tuple(self._report_cluster_geometry_rows()),
            solution_scattering_estimate=(
                self._current_report_solution_scattering_estimate()
            ),
            dream_settings=self._copy_dream_settings(settings),
            dream_summary=summary,
            dream_model_plot=model_plot,
            dream_violin_plot=violin_plot,
            dream_violin_payload=plot_payload,
            dream_parameter_map_entries=tuple(
                DreamParameterEntry.from_dict(dict(entry))
                for entry in self._last_results_loader.parameter_map_entries
                if isinstance(entry, dict)
            ),
            dream_filter_assessments=assessments,
            dream_filter_views=filter_views,
            output_summary_lines=tuple(output_summary_lines),
            directory_lines=tuple(directory_lines),
        )

    def _build_report_template_details(
        self,
        *,
        template_name: str,
        q_range_text: str,
        supported_q_range_text: str | None,
        q_sampling_text: str,
        prefit_parameter_count: int,
        dream_active_parameter_names: tuple[str, ...],
        includes_solvent: bool,
        includes_structure_factor: bool,
    ) -> tuple[str, Path | None, str | None, list[str], list[str], list[str]]:
        try:
            spec = load_template_spec(template_name)
        except Exception:
            return (
                template_name,
                None,
                None,
                [
                    f"Template name: {template_name}",
                    "Template metadata could not be loaded for this report.",
                    f"User selected q-range: {q_range_text}",
                    (
                        "Supported component q-range: "
                        f"{supported_q_range_text or 'Unavailable'}"
                    ),
                    f"q-grid: {q_sampling_text}",
                ],
                [],
                [],
            )

        description_sections = self._report_named_sections_from_text(
            spec.description
        )
        source_sections = self._report_named_sections_from_template_source(
            spec.module_path
        )
        model_equation_lines = self._report_template_section_lines(
            source_sections,
            description_sections,
            "model equation",
        )
        model_equation_text = (
            " ".join(
                line.strip() for line in model_equation_lines if line.strip()
            )
            or None
        )

        context_lines = [
            f"Template display name: {spec.display_name}",
            f"Template key: {spec.name}",
            f"Template module: {spec.module_path}",
            f"LMFit entrypoint: {spec.lmfit_model_name}",
            f"pyDREAM entrypoint: {spec.dream_model_name}",
            f"LMFit inputs: {', '.join(spec.lmfit_inputs) or 'None'}",
            f"pyDREAM inputs: {', '.join(spec.dream_inputs) or 'None'}",
            (
                "Parameter columns: "
                f"{', '.join(spec.param_columns) or 'None'}"
            ),
            f"Declared static template parameters: {len(spec.parameters)}",
            f"Prefit parameter rows in this report: {prefit_parameter_count}",
            (
                "DREAM active parameters: "
                f"{', '.join(dream_active_parameter_names) or 'None'}"
            ),
            f"User selected q-range: {q_range_text}",
            (
                "Supported component q-range: "
                f"{supported_q_range_text or 'Unavailable'}"
            ),
            f"q-grid: {q_sampling_text}",
            (
                "Best-fit plot includes solvent contribution: "
                f"{'yes' if includes_solvent else 'no'}"
            ),
            (
                "Best-fit plot includes structure factor trace: "
                f"{'yes' if includes_structure_factor else 'no'}"
            ),
            (
                "Cluster geometry metadata support: "
                f"{'enabled' if spec.cluster_geometry_support.supported else 'disabled'}"
            ),
        ]
        if spec.cluster_geometry_support.supported:
            context_lines.extend(
                [
                    (
                        "Allowed structure-factor approximations: "
                        f"{', '.join(spec.cluster_geometry_support.allowed_sf_approximations)}"
                    ),
                    (
                        "Cluster metadata fields: "
                        f"{', '.join(spec.cluster_geometry_support.metadata_fields)}"
                    ),
                    (
                        "Dynamic geometry parameters: "
                        f"{'yes' if spec.cluster_geometry_support.dynamic_parameters else 'no'}"
                    ),
                    (
                        "Runtime bindings: "
                        + ", ".join(
                            (
                                f"{binding.runtime_name} <- "
                                f"{binding.metadata_field}"
                            )
                            for binding in (
                                spec.cluster_geometry_support.runtime_bindings
                            )
                        )
                    ),
                ]
            )

        definition_lines: list[str] = []
        for heading in (
            "purpose",
            "scientific scope",
            "structure factor",
            "form factor",
            "model organization",
            "internal abundance normalization",
            "model parameters",
            "parameter definitions",
            "fitting guidance",
            "likelihood convention",
            "required pydream globals",
            "practical notes",
            "cluster geometry metadata",
        ):
            section_lines = self._report_template_section_lines(
                source_sections,
                description_sections,
                heading,
            )
            if not section_lines:
                continue
            if definition_lines:
                definition_lines.append("")
            definition_lines.append(f"{heading.title()}:")
            definition_lines.extend(section_lines)

        reference_lines = self._report_template_section_lines(
            source_sections,
            description_sections,
            "relevant resources",
            "references",
        )
        if not reference_lines and spec.metadata_path is not None:
            reference_lines = [
                "No explicit literature references were declared in the "
                f"template metadata or source comments for {spec.name}.",
            ]

        return (
            spec.display_name,
            spec.module_path,
            model_equation_text,
            context_lines,
            definition_lines,
            reference_lines,
        )

    @staticmethod
    def _report_named_sections_from_template_source(
        module_path: Path,
    ) -> dict[str, list[str]]:
        try:
            raw_lines = module_path.read_text(encoding="utf-8").splitlines()
        except Exception:
            return {}
        comment_lines: list[str] = []
        for raw_line in raw_lines:
            stripped = raw_line.strip()
            if stripped.startswith("def ") or stripped.startswith("class "):
                break
            if stripped.startswith("#"):
                comment_lines.append(stripped[1:].strip())
            elif not stripped:
                comment_lines.append("")
        return SAXSMainWindow._report_named_sections_from_lines(comment_lines)

    @staticmethod
    def _report_named_sections_from_text(
        text: str,
    ) -> dict[str, list[str]]:
        return SAXSMainWindow._report_named_sections_from_lines(
            text.splitlines()
        )

    @staticmethod
    def _report_named_sections_from_lines(
        raw_lines: list[str],
    ) -> dict[str, list[str]]:
        recognized_headings = {
            "purpose",
            "scientific scope",
            "structure factor",
            "form factor",
            "model organization",
            "model equation",
            "internal abundance normalization",
            "model parameters",
            "parameter definitions",
            "fitting guidance",
            "likelihood convention",
            "required pydream globals",
            "practical notes",
            "cluster geometry metadata",
            "relevant resources",
            "references",
        }
        sections: dict[str, list[str]] = {}
        current_heading: str | None = None
        current_lines: list[str] = []
        for raw_line in raw_lines:
            stripped = str(raw_line).strip()
            normalized_heading = stripped[:-1].strip().lower()
            if (
                stripped.endswith(":")
                and ":" not in stripped[:-1]
                and not stripped.startswith("-")
                and normalized_heading in recognized_headings
            ):
                if current_heading is not None:
                    sections[current_heading] = [
                        line for line in current_lines if line or line == ""
                    ]
                current_heading = normalized_heading
                current_lines = []
                continue
            if current_heading is None:
                continue
            current_lines.append(stripped)
        if current_heading is not None:
            sections[current_heading] = [
                line for line in current_lines if line or line == ""
            ]
        normalized: dict[str, list[str]] = {}
        for heading, lines in sections.items():
            cleaned_lines: list[str] = []
            previous_blank = True
            for line in lines:
                if not line:
                    if not previous_blank:
                        cleaned_lines.append("")
                    previous_blank = True
                    continue
                cleaned_lines.append(line)
                previous_blank = False
            while cleaned_lines and not cleaned_lines[0]:
                cleaned_lines.pop(0)
            while cleaned_lines and not cleaned_lines[-1]:
                cleaned_lines.pop()
            normalized[heading] = cleaned_lines
        return normalized

    @staticmethod
    def _report_template_section_lines(
        source_sections: dict[str, list[str]],
        description_sections: dict[str, list[str]],
        *section_names: str,
    ) -> list[str]:
        for section_name in section_names:
            normalized = str(section_name).strip().lower()
            lines = source_sections.get(normalized)
            if lines:
                return list(lines)
            lines = description_sections.get(normalized)
            if lines:
                return list(lines)
        return []

    def _report_prior_json_path(self) -> Path | None:
        current_prior_path = self.project_setup_tab.current_prior_json_path()
        if current_prior_path is not None and current_prior_path.is_file():
            return current_prior_path
        if self.current_settings is None:
            return None
        candidate = project_artifact_paths(
            self.current_settings
        ).prior_weights_file
        return candidate if candidate.is_file() else None

    def _report_secondary_element(self) -> str | None:
        return self.project_setup_tab.selected_prior_secondary_element()

    def _build_report_prior_requests(
        self,
        powerpoint_settings: PowerPointExportSettings,
    ) -> list[PriorHistogramRequest]:
        prior_json_path = self._report_prior_json_path()
        if prior_json_path is None:
            return []
        secondary_element = self._report_secondary_element()
        return [
            PriorHistogramRequest(
                title="Structure Fraction Histogram",
                json_path=prior_json_path,
                mode="structure_fraction",
                cmap=powerpoint_settings.prior_histogram_color_map,
            ),
            PriorHistogramRequest(
                title="Atom Fraction Histogram",
                json_path=prior_json_path,
                mode="atom_fraction",
                cmap=powerpoint_settings.prior_histogram_color_map,
            ),
            PriorHistogramRequest(
                title="Solvent Sort Structure Fraction Histogram",
                json_path=prior_json_path,
                mode="solvent_sort_structure_fraction",
                cmap=powerpoint_settings.solvent_sort_histogram_color_map,
                secondary_element=secondary_element,
            ),
            PriorHistogramRequest(
                title="Solvent Sort Atom Fraction Histogram",
                json_path=prior_json_path,
                mode="solvent_sort_atom_fraction",
                cmap=powerpoint_settings.solvent_sort_histogram_color_map,
                secondary_element=secondary_element,
            ),
        ]

    def _report_data_summary(
        self,
        *,
        solvent: bool,
    ) -> ExperimentalDataSummary | None:
        if self.current_settings is None:
            return None
        settings = self.current_settings
        if solvent:
            preferred_paths = [
                settings.copied_solvent_data_file,
                settings.solvent_data_path,
            ]
            skiprows = int(settings.solvent_header_rows)
            q_column = settings.solvent_q_column
            intensity_column = settings.solvent_intensity_column
            error_column = settings.solvent_error_column
        else:
            preferred_paths = [
                settings.copied_experimental_data_file,
                settings.experimental_data_path,
            ]
            skiprows = int(settings.experimental_header_rows)
            q_column = settings.experimental_q_column
            intensity_column = settings.experimental_intensity_column
            error_column = settings.experimental_error_column
        resolved_path = next(
            (
                Path(candidate).expanduser().resolve()
                for candidate in preferred_paths
                if candidate
                and Path(candidate).expanduser().resolve().is_file()
            ),
            None,
        )
        if resolved_path is None:
            return None
        return load_experimental_data_file(
            resolved_path,
            skiprows=skiprows,
            q_column=q_column,
            intensity_column=intensity_column,
            error_column=error_column,
        )

    def _build_report_component_plot_data(
        self,
        *,
        include_solvent: bool,
        powerpoint_settings: PowerPointExportSettings,
    ) -> ReportComponentPlotData | None:
        if self.current_settings is None:
            return None
        component_paths = sorted(
            project_artifact_paths(self.current_settings).component_dir.glob(
                "*.txt"
            )
        )
        experimental_summary = self._report_data_summary(solvent=False)
        solvent_summary = (
            self._report_data_summary(solvent=True)
            if include_solvent
            else None
        )
        if not component_paths and experimental_summary is None:
            return None
        try:
            cmap = colormaps[powerpoint_settings.component_color_map]
        except Exception:
            cmap = colormaps["viridis"]
        if len(component_paths) <= 1:
            positions = np.asarray([0.68], dtype=float)
        else:
            positions = np.linspace(0.15, 0.9, len(component_paths))
        component_series: list[ReportComponentSeries] = []
        for component_path, position in zip(
            component_paths,
            positions,
            strict=False,
        ):
            raw_data = np.loadtxt(component_path, comments="#")
            if raw_data.ndim == 1:
                raw_data = raw_data.reshape(1, -1)
            component_series.append(
                ReportComponentSeries(
                    label=component_path.stem,
                    q_values=np.asarray(raw_data[:, 0], dtype=float),
                    intensities=np.asarray(raw_data[:, 1], dtype=float),
                    color=to_hex(cmap(float(position)), keep_alpha=False),
                )
            )
        title = (
            "Initial SAXS traces without solvent"
            if not include_solvent
            else "Initial SAXS traces with solvent"
        )
        return ReportComponentPlotData(
            title=title,
            selected_q_min=(
                float(self.current_settings.q_min)
                if self.current_settings.q_min is not None
                else None
            ),
            selected_q_max=(
                float(self.current_settings.q_max)
                if self.current_settings.q_max is not None
                else None
            ),
            use_experimental_grid=bool(
                self.current_settings.use_experimental_grid
                and not self.current_settings.model_only_mode
            ),
            log_x=bool(
                self.project_setup_tab.component_log_x_checkbox.isChecked()
            ),
            log_y=bool(
                self.project_setup_tab.component_log_y_checkbox.isChecked()
            ),
            experimental_q_values=(
                None
                if experimental_summary is None
                else np.asarray(experimental_summary.q_values, dtype=float)
            ),
            experimental_intensities=(
                None
                if experimental_summary is None
                else np.asarray(experimental_summary.intensities, dtype=float)
            ),
            solvent_q_values=(
                None
                if solvent_summary is None
                else np.asarray(solvent_summary.q_values, dtype=float)
            ),
            solvent_intensities=(
                None
                if solvent_summary is None
                else np.asarray(solvent_summary.intensities, dtype=float)
            ),
            component_series=tuple(component_series),
        )

    def _latest_prefit_statistics_payload(self) -> dict[str, object]:
        if self.current_settings is None:
            return {}
        prefit_dir = project_artifact_paths(self.current_settings).prefit_dir
        state_path = prefit_dir / "prefit_state.json"
        if not state_path.is_file():
            return {}
        payload = json.loads(state_path.read_text(encoding="utf-8"))
        statistics = dict(payload.get("statistics", {}))
        statistics["saved_at"] = str(payload.get("saved_at", ""))
        latest_snapshot = next(
            iter(sorted(prefit_dir.glob("prefit_*"), reverse=True)),
            None,
        )
        if latest_snapshot is not None:
            statistics["snapshot_dir"] = str(latest_snapshot)
            latest_report_path = latest_snapshot / "prefit_report.txt"
            if latest_report_path.is_file():
                statistics["report_path"] = str(latest_report_path)
        return statistics

    def _report_cluster_geometry_rows(self) -> list[object]:
        if self.prefit_workflow is None:
            return []
        try:
            return list(self.prefit_workflow.cluster_geometry_rows())
        except Exception:
            return []

    @staticmethod
    def _estimate_section_count(
        estimate: SolutionScatteringEstimate,
    ) -> int:
        return sum(
            section is not None
            for section in (
                estimate.number_density_estimate,
                estimate.volume_fraction_estimate,
                estimate.attenuation_estimate,
                estimate.fluorescence_estimate,
            )
        )

    def _current_report_solution_scattering_estimate(
        self,
    ) -> SolutionScatteringEstimate | None:
        candidates: list[SolutionScatteringEstimate] = []
        embedded_estimate = (
            self.prefit_tab.solute_volume_fraction_widget.current_estimate()
        )
        if embedded_estimate is not None:
            candidates.append(embedded_estimate)
        for window in (
            self._solute_volume_fraction_tool_window,
            self._number_density_tool_window,
            self._attenuation_tool_window,
            self._fluorescence_tool_window,
        ):
            if window is None:
                continue
            estimate = window.estimator_widget.current_estimate()
            if estimate is not None:
                candidates.append(estimate)
        if self._last_solution_scattering_estimate is not None:
            candidates.append(self._last_solution_scattering_estimate)
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda estimate: (
                self._estimate_section_count(estimate),
                estimate is self._last_solution_scattering_estimate,
            ),
        )

    def _report_dream_filter_assessments(
        self,
        settings: DreamRunSettings,
    ) -> list[dict[str, object]]:
        if self._last_dream_filter_assessments:
            return [dict(item) for item in self._last_dream_filter_assessments]
        assessments, recommendation = self._evaluate_dream_posterior_filters(
            settings
        )
        if recommendation is not None:
            self._last_dream_filter_recommendation = dict(recommendation)
        self._last_dream_filter_assessments = [
            dict(item) for item in assessments
        ]
        return [dict(item) for item in assessments]

    def _build_dream_report_filter_views(
        self,
        settings: DreamRunSettings,
    ) -> tuple[DreamFilterReportView, ...]:
        if self._last_results_loader is None:
            return ()
        views: list[DreamFilterReportView] = []
        for mode, label in (
            ("all_post_burnin", "All Post-burnin"),
            ("top_percent_logp", "Top % by Log-posterior"),
            ("top_n_logp", "Top N by Log-posterior"),
        ):
            candidate_settings = self._copy_dream_settings(settings)
            candidate_settings.posterior_filter_mode = mode
            filter_kwargs = self._dream_filter_kwargs(candidate_settings)
            summary = self._last_results_loader.get_summary(
                bestfit_method=candidate_settings.bestfit_method,
                **filter_kwargs,
            )
            model_plot = self._last_results_loader.build_model_fit_data(
                bestfit_method=candidate_settings.bestfit_method,
                **filter_kwargs,
            )
            violin_plot = self._last_results_loader.build_violin_data(
                mode=self._effective_dream_violin_mode(candidate_settings),
                sample_source=candidate_settings.violin_sample_source,
                weight_order=candidate_settings.violin_weight_order,
                **filter_kwargs,
            )
            weights_violin_plot = self._last_results_loader.build_violin_data(
                mode="weights_only",
                sample_source=candidate_settings.violin_sample_source,
                weight_order=candidate_settings.violin_weight_order,
                **filter_kwargs,
            )
            effective_radii_violin_plot = (
                self._last_results_loader.build_violin_data(
                    mode="effective_radii_only",
                    sample_source=candidate_settings.violin_sample_source,
                    weight_order=candidate_settings.violin_weight_order,
                    **filter_kwargs,
                )
            )
            title = label
            is_active = mode == settings.posterior_filter_mode
            if is_active:
                title += " [Active]"
            views.append(
                DreamFilterReportView(
                    title=title,
                    description=self._describe_posterior_filter(
                        candidate_settings
                    ),
                    filter_mode=mode,
                    is_active=is_active,
                    summary=summary,
                    model_plot=model_plot,
                    violin_plot=violin_plot,
                    violin_payload=self.dream_tab.prepare_violin_plot_payload(
                        summary,
                        violin_plot,
                    ),
                    weights_violin_payload=(
                        self.dream_tab.prepare_violin_plot_payload(
                            summary,
                            weights_violin_plot,
                        )
                    ),
                    effective_radii_violin_payload=(
                        self.dream_tab.prepare_violin_plot_payload(
                            summary,
                            effective_radii_violin_plot,
                        )
                    ),
                )
            )
        return tuple(views)

    def _report_q_sampling_text(self) -> str:
        if self.current_settings is None:
            return "Unavailable"
        if self.current_settings.model_only_mode:
            return f"Model-only resampled grid ({self.current_settings.q_points or 0} points)"
        if self.current_settings.use_experimental_grid:
            return "Experimental grid"
        if self.current_settings.q_points is not None:
            return f"Resampled grid ({self.current_settings.q_points} points)"
        return "Project q-grid"

    def _format_selected_q_range_text(self, q_values: np.ndarray) -> str:
        lower = (
            float(self.current_settings.q_min)
            if self.current_settings is not None
            and self.current_settings.q_min is not None
            else float(np.min(q_values))
        )
        upper = (
            float(self.current_settings.q_max)
            if self.current_settings is not None
            and self.current_settings.q_max is not None
            else float(np.max(q_values))
        )
        return f"{lower:.6g} to {upper:.6g}"

    @staticmethod
    def _effective_dream_violin_mode(settings: DreamRunSettings) -> str:
        if settings.violin_value_scale_mode == "weights_unit_interval":
            return "weights_only"
        if settings.violin_value_scale_mode == "normalized_all":
            return "all_parameters"
        if settings.violin_value_scale_mode == "effective_radii_only":
            return "effective_radii_only"
        if settings.violin_value_scale_mode == "additional_parameters_only":
            return "additional_parameters_only"
        return settings.violin_parameter_mode

    def save_prior_plot_data_as(self) -> None:
        if self.current_settings is None:
            self._show_error(
                "Save prior plot data failed",
                "Load or build a project first.",
            )
            return
        try:
            paths = build_project_paths(self.current_settings.project_dir)
            source = paths.plots_dir / "prior_histogram_data.json"
            if not source.is_file():
                raise FileNotFoundError(
                    "No prior_histogram_data.json file was found. Generate "
                    "prior weights first."
                )
            destination, _ = QFileDialog.getSaveFileName(
                self,
                "Save prior histogram data",
                str(paths.exported_data_dir / source.name),
                "JSON files (*.json)",
            )
            if destination:
                Path(destination).write_text(
                    source.read_text(encoding="utf-8"),
                    encoding="utf-8",
                )
                self.project_setup_tab.append_summary(
                    f"Saved prior histogram data to {destination}"
                )
        except Exception as exc:
            self._show_error("Save prior data failed", str(exc))

    def save_component_plot_data(self) -> None:
        if self.current_settings is None:
            self._show_error(
                "Save component plot data failed",
                "Load or build a project first.",
            )
            return
        try:
            payload = self.project_setup_tab.component_plot_export_payload()
            traces = list(payload.get("traces", []))
            if not traces:
                raise ValueError(
                    "No experimental or component traces are currently "
                    "available to export."
                )
            destination = self._prompt_project_plot_export_path(
                dialog_title="Export component plot data",
                default_filename=(
                    "component_plot_data_"
                    f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                ),
                export_kind="data",
            )
            if destination is None:
                return
            if destination.suffix.lower() == ".csv":
                self._write_component_plot_csv(destination, payload)
            else:
                np.save(destination, payload, allow_pickle=True)
            self.project_setup_tab.append_summary(
                f"Saved component plot data to {destination}"
            )
            self.statusBar().showMessage("Component plot data exported")
        except Exception as exc:
            self._show_error("Save component plot data failed", str(exc))

    def save_prior_plot_data(self) -> None:
        if self.current_settings is None:
            self._show_error(
                "Save prior plot data failed",
                "Load or build a project first.",
            )
            return
        prior_json = self.project_setup_tab.current_prior_json_path()
        if prior_json is None or not prior_json.is_file():
            self._show_error(
                "Save prior plot data failed",
                "Generate prior weights first.",
            )
            return
        mode = self.project_setup_tab.prior_mode()
        secondary = self.project_setup_tab.prior_secondary_element()
        if mode.startswith("solvent_sort") and secondary is None:
            self._show_error(
                "Save prior plot data failed",
                "Select a secondary atom filter before exporting a solvent-sort prior histogram.",
            )
            return
        try:
            destination = self._prompt_project_plot_export_path(
                dialog_title="Export prior histogram data",
                default_filename=(
                    "prior_histogram_data_"
                    f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                ),
                export_kind="data",
            )
            if destination is None:
                return
            if destination.suffix.lower() == ".csv":
                export_prior_histogram_table(
                    prior_json,
                    destination,
                    mode=mode,
                    value_mode="percent",
                    secondary_element=secondary,
                )
            else:
                export_prior_histogram_npy(
                    prior_json,
                    destination,
                    mode=mode,
                    value_mode="percent",
                    secondary_element=secondary,
                )
            self.project_setup_tab.append_summary(
                f"Saved prior histogram plot data to {destination}"
            )
            self.statusBar().showMessage("Prior histogram plot data exported")
        except Exception as exc:
            self._show_error("Save prior plot data failed", str(exc))

    def _prompt_project_plot_export_path(
        self,
        *,
        dialog_title: str,
        default_filename: str,
        export_kind: str = "data",
    ) -> Path | None:
        if self.current_settings is None:
            return None
        paths = build_project_paths(self.current_settings.project_dir)
        if export_kind == "plots":
            target_dir = paths.exported_plots_dir
        else:
            target_dir = paths.exported_data_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        destination, selected_filter = QFileDialog.getSaveFileName(
            self,
            dialog_title,
            str(target_dir / default_filename),
            "CSV files (*.csv);;NumPy files (*.npy)",
        )
        if not destination:
            return None
        output_path = Path(destination)
        if output_path.suffix.lower() not in {".csv", ".npy"}:
            if "npy" in selected_filter.lower():
                output_path = output_path.with_suffix(".npy")
            else:
                output_path = output_path.with_suffix(".csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return output_path

    @staticmethod
    def _write_component_plot_csv(
        output_path: Path,
        payload: dict[str, object],
    ) -> None:
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "series",
                    "component_key",
                    "axis_index",
                    "axis_ylabel",
                    "color",
                    "visible",
                    "x",
                    "y",
                ]
            )
            for trace in payload.get("traces", []):
                x_values = np.asarray(trace.get("x", []), dtype=float)
                y_values = np.asarray(trace.get("y", []), dtype=float)
                count = min(len(x_values), len(y_values))
                for index in range(count):
                    writer.writerow(
                        [
                            str(trace.get("series", "")),
                            str(trace.get("component_key", "")),
                            int(trace.get("axis_index", 0)),
                            str(trace.get("axis_ylabel", "")),
                            str(trace.get("color", "")),
                            bool(trace.get("visible", True)),
                            f"{x_values[index]:.10g}",
                            f"{y_values[index]:.10g}",
                        ]
                    )

    @staticmethod
    def _write_prefit_plot_csv(
        output_path: Path,
        *,
        metadata: dict[str, object],
        columns: list[str],
        matrix: np.ndarray,
    ) -> None:
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            for key, value in metadata.items():
                handle.write(f"# {key}: {json.dumps(value, sort_keys=True)}\n")
            writer = csv.writer(handle)
            writer.writerow(columns)
            writer.writerows(matrix.tolist())

    def _build_prefit_plot_export_metadata(
        self,
        *,
        entries,
        evaluation,
        method: str,
        max_nfev: int,
    ) -> dict[str, object]:
        q_values = np.asarray(evaluation.q_values, dtype=float)
        fit_metrics: dict[str, object]
        if (
            evaluation.residuals is None
            or evaluation.experimental_intensities is None
        ):
            fit_metrics = {
                "mode": "model_only",
                "chi_square": None,
                "reduced_chi_square": None,
                "r_squared": None,
                "residual_rms": None,
                "mean_absolute_residual": None,
            }
        else:
            residuals = np.asarray(evaluation.residuals, dtype=float)
            chi_square = float(np.sum(residuals**2))
            dof = max(
                len(q_values) - sum(1 for entry in entries if entry.vary),
                1,
            )
            reduced_chi_square = chi_square / dof
            experimental = np.asarray(
                evaluation.experimental_intensities,
                dtype=float,
            )
            ss_total = float(
                np.sum((experimental - np.mean(experimental)) ** 2)
            )
            fit_metrics = {
                "mode": "fit",
                "chi_square": chi_square,
                "reduced_chi_square": reduced_chi_square,
                "r_squared": (
                    1.0 - chi_square / ss_total
                    if ss_total > 0.0
                    else float("nan")
                ),
                "residual_rms": float(np.sqrt(np.mean(residuals**2))),
                "mean_absolute_residual": float(np.mean(np.abs(residuals))),
            }
        return {
            "exported_at": datetime.now().isoformat(),
            "project_dir": str(self.current_settings.project_dir),
            "template_name": (
                self.prefit_workflow.template_spec.name
                if self.prefit_workflow is not None
                else None
            ),
            "fit_conditions": {
                "method": method,
                "max_nfev": int(max_nfev),
                "autosave_prefits": bool(
                    self.prefit_tab.autosave_checkbox.isChecked()
                ),
                "log_x": bool(self.prefit_tab.log_x_checkbox.isChecked()),
                "log_y": bool(self.prefit_tab.log_y_checkbox.isChecked()),
                "point_count": int(len(q_values)),
                "q_min": float(np.min(q_values)) if len(q_values) else None,
                "q_max": float(np.max(q_values)) if len(q_values) else None,
                "includes_structure_factor": bool(
                    evaluation.structure_factor_trace is not None
                ),
            },
            "fit_metrics": fit_metrics,
            "parameter_entries": [entry.to_dict() for entry in entries],
        }

    def _apply_project_settings(
        self,
        settings: ProjectSettings,
        *,
        progress_callback: Callable[[str], None] | None = None,
        log_callback: Callable[[str], None] | None = None,
        prefit_payload: ProjectLoadPrefitPayload | None = None,
        dream_payload: ProjectLoadDreamPayload | None = None,
    ) -> None:
        def start_load_step(
            message: str,
            *,
            log_message: str | None = None,
        ) -> None:
            if log_message and log_callback is not None:
                log_callback(log_message)
            if progress_callback is not None:
                progress_callback(message)

        self._prefit_missing_components_warning_shown = False
        template_specs = self._template_specs_for_dropdown(
            include_deprecated=self._show_deprecated_templates,
            selected_names=[settings.selected_model_template or ""],
        )
        self._last_results_loader = None
        self._loaded_dream_run_dir = None
        self._dream_workflow_project_dir = None
        self._dream_parameter_map_saved_in_session = False
        self._active_dream_settings_snapshot = None
        self._current_dream_preset_name = None
        self.dream_tab.reset_progress()
        project_setup_lines = [
            "Loading Project Setup controls.",
            "Selected template: "
            + (settings.selected_model_template or "not selected"),
            "Component build mode: "
            + component_build_mode_label(settings.component_build_mode),
        ]
        experimental_source = (
            settings.copied_experimental_data_file
            or settings.experimental_data_path
        )
        solvent_source = (
            settings.copied_solvent_data_file or settings.solvent_data_path
        )
        project_setup_lines.append(
            "Experimental data preview: " + (experimental_source or "none")
        )
        project_setup_lines.append(
            "Solvent data preview: " + (solvent_source or "none")
        )
        with self.project_setup_tab.batch_preview_updates():
            start_load_step(
                "Loading project controls...",
                log_message="\n".join(project_setup_lines),
            )
            self.project_setup_tab.set_project_selected(True)
            self.project_setup_tab.set_project_settings(
                settings,
                template_specs,
            )
            start_load_step(
                "Loading saved distributions...",
                log_message="Loading saved parameter distributions.",
            )
            distribution_count = self._refresh_saved_distributions(settings)
            if log_callback is not None:
                if distribution_count:
                    log_callback(
                        "Loaded "
                        f"{distribution_count} saved parameter distribution"
                        f"{'' if distribution_count == 1 else 's'}."
                    )
                else:
                    log_callback("No saved parameter distributions found.")
            start_load_step(
                "Loading cluster inventory...",
                log_message=(
                    "Loading recognized cluster rows and available elements."
                ),
            )
            available_elements, cluster_rows = (
                self._active_cluster_import_view(
                    settings,
                    fallback_available_elements=settings.available_elements,
                    fallback_rows=settings.cluster_inventory_rows,
                )
            )
            self.project_setup_tab.apply_cluster_import_data(
                available_elements,
                cluster_rows,
            )
            if log_callback is not None:
                log_callback(
                    "Loaded "
                    f"{len(cluster_rows)} cluster row"
                    f"{'' if len(cluster_rows) == 1 else 's'} across "
                    f"{len(available_elements)} available element"
                    f"{'' if len(available_elements) == 1 else 's'}."
                )
            start_load_step(
                "Loading predicted structure status...",
                log_message="Checking Predicted Structures readiness.",
            )
            predicted_status = self._refresh_predicted_structure_status(
                settings
            )
            if log_callback is not None:
                log_callback(predicted_status)
            start_load_step(
                "Loading component preview...",
                log_message="Loading SAXS component traces for the preview.",
            )
            component_paths = self._refresh_component_plot()
            if log_callback is not None:
                if component_paths:
                    log_callback(
                        "Loaded "
                        f"{len(component_paths)} SAXS component trace"
                        f"{'' if len(component_paths) == 1 else 's'}."
                    )
                else:
                    log_callback("No SAXS component traces were found yet.")
            start_load_step(
                "Loading prior histogram preview...",
                log_message="Loading prior histogram preview data.",
            )
            prior_path = self._refresh_prior_plot()
            if log_callback is not None:
                log_callback(
                    f"Loaded prior histogram data from {prior_path}"
                    if prior_path is not None
                    else "No prior histogram data was found yet."
                )
        start_load_step(
            "Loading Prefit workflow...",
            log_message="Loading Prefit workflow and preview.",
        )
        try:
            if prefit_payload is None:
                self._load_prefit_workflow()
            else:
                self._apply_loaded_prefit_payload(
                    template_specs,
                    prefit_payload,
                )
        except Exception as exc:
            self.prefit_workflow = None
            self._apply_prefit_template_fallback(settings, template_specs)
            self.prefit_tab.plot_evaluation(None)
            self.prefit_tab.set_log_text(
                "Prefit workflow is not ready yet.\n" f"{exc}"
            )
            self.prefit_tab.set_summary_text(
                "Prefit summary is not available yet.\n" f"{exc}"
            )
            self.prefit_tab.set_saved_states([], None)
            if log_callback is not None:
                log_callback(f"Prefit workflow is not ready yet: {exc}")
        else:
            if log_callback is not None and self.prefit_workflow is not None:
                log_callback(
                    "Loaded Prefit workflow with "
                    f"{len(self.prefit_workflow.parameter_entries)} "
                    "parameter entries."
                )
        self._refresh_model_only_mode_state()
        start_load_step(
            "Loading DREAM workflow...",
            log_message="Loading DREAM workflow settings and parameter map.",
        )
        try:
            if dream_payload is not None:
                self._apply_loaded_dream_payload(dream_payload)
            elif not settings.model_only_mode:
                self._load_dream_workflow()
            else:
                raise ValueError(
                    "DREAM is disabled in Model Only Mode. Disable Model "
                    "Only Mode and add experimental SAXS data to enable DREAM."
                )
        except Exception as exc:
            self.dream_workflow = None
            self.dream_tab.set_available_saved_runs([])
            self.dream_tab.set_log_text(
                "DREAM workflow is not ready yet.\n" f"{exc}"
            )
            self.dream_tab.set_summary_text(
                "DREAM summary is not available yet.\n" f"{exc}"
            )
            self.dream_tab.clear_plots()
            if log_callback is not None:
                log_callback(f"DREAM workflow is not ready yet: {exc}")
        else:
            if log_callback is not None and self.dream_workflow is not None:
                preset_count = len(self.dream_workflow.list_settings_presets())
                log_callback(
                    "Loaded DREAM workflow with "
                    f"{preset_count} settings preset"
                    f"{'' if preset_count == 1 else 's'}."
                )
        self._refresh_model_only_mode_state()
        self._update_file_menu_state()

    def _apply_loaded_prefit_payload(
        self,
        template_specs: list,
        payload: ProjectLoadPrefitPayload,
    ) -> None:
        if payload.workflow is None:
            raise ValueError(
                payload.workflow_error or "Prefit workflow is not ready yet."
            )
        self.prefit_workflow = payload.workflow
        self.current_settings = self.prefit_workflow.settings
        self._sync_active_template_controls(
            self.prefit_workflow.template_spec.name,
            sync_selected=True,
        )
        self.prefit_tab.set_templates(
            template_specs,
            self.prefit_workflow.template_spec.name,
            active_name=self.prefit_workflow.template_spec.name,
        )
        self.prefit_tab.set_autosave(
            self.prefit_workflow.settings.autosave_prefits
        )
        self.prefit_tab.set_sequence_history_enabled(
            self.prefit_workflow.settings.prefit_sequence_history_enabled
        )
        self.prefit_tab.populate_parameter_table(
            self.prefit_workflow.parameter_entries
        )
        self._set_prefit_sequence_baseline(
            self.prefit_workflow.parameter_entries
        )
        self._refresh_prefit_volume_fraction_section()
        self._refresh_prefit_cluster_geometry_section()
        if payload.evaluation is None:
            self.prefit_tab.plot_evaluation(None)
            self.prefit_tab.set_summary_text(
                "Prefit preview is waiting on template metadata.\n\n"
                + (payload.preview_error or "Preview data is not available.")
            )
        else:
            self.prefit_tab.plot_evaluation(payload.evaluation)
            self.prefit_tab.set_summary_text(
                self._format_prefit_summary(payload.evaluation)
            )
            self._update_prefit_stoichiometry_status()
            if payload.scale_recommendation is not None:
                self._append_scale_recommendation_log(
                    payload.scale_recommendation
                )
        self.prefit_tab.set_log_text(
            self._format_prefit_console_intro(
                evaluation=payload.evaluation,
                preview_block_reason=payload.preview_error,
            )
        )
        if self.prefit_workflow.has_best_prefit_entries():
            self.prefit_tab.append_log(
                "Loaded the Best Prefit preset from the project file."
            )
        self._refresh_saved_prefit_states()

    def _apply_loaded_dream_payload(
        self,
        payload: ProjectLoadDreamPayload,
    ) -> None:
        if payload.workflow is None or payload.settings is None:
            raise ValueError(
                payload.error or "DREAM workflow is not ready yet."
            )
        self.dream_workflow = payload.workflow
        self._dream_workflow_project_dir = str(
            Path(self.dream_workflow.settings.project_dir).resolve()
        )
        self._invalidate_written_dream_bundle()
        self._last_dream_constraint_warning_signature = None
        self.dream_tab.set_available_settings_presets(
            payload.preset_names,
            payload.selected_preset,
        )
        self.dream_tab.set_settings(
            payload.settings,
            preset_name=payload.selected_preset,
        )
        self._active_dream_settings_snapshot = self._copy_dream_settings(
            payload.settings
        )
        self._current_dream_preset_name = payload.selected_preset
        self.dream_tab.set_parameter_map_entries(payload.parameter_map_entries)
        self._dream_parameter_map_saved_in_session = False
        self._applied_dream_analysis_settings = None
        self._last_dream_summary = None
        self.dream_tab.set_log_text(self._format_dream_console_intro())
        self.dream_tab.set_summary_text("DREAM results are not loaded yet.")
        self.dream_tab.set_filter_status_text(
            "No DREAM dataset is loaded yet. Load a run, then apply "
            "posterior or stoichiometry filters."
        )
        self.dream_tab.set_filter_dirty(False)
        self.dream_tab.reset_progress()
        self.dream_tab.clear_plots()
        self._refresh_saved_dream_runs()

    def _refresh_saved_distributions(
        self,
        settings: ProjectSettings | None = None,
    ) -> int:
        active_settings = settings or self.current_settings
        if active_settings is None:
            self.project_setup_tab.set_available_distributions([])
            self._update_active_contrast_distribution_view_state(None)
            return 0
        records = self.project_manager.list_saved_distributions(
            active_settings.project_dir
        )
        labels = []
        for record in records:
            readiness = []
            if record.component_artifacts_ready:
                readiness.append("components")
            if record.prior_artifacts_ready:
                readiness.append("prior")
            readiness_text = (
                " | " + ", ".join(readiness)
                if readiness
                else " | metadata only"
            )
            labels.append(
                (
                    f"{record.label}{readiness_text}",
                    record.distribution_id,
                )
            )
        selected_id = None
        if labels:
            selected_id = project_artifact_paths(
                active_settings
            ).distribution_id
        self.project_setup_tab.set_available_distributions(
            labels,
            selected_distribution_id=selected_id,
        )
        self._update_active_contrast_distribution_view_state(active_settings)
        return len(records)

    def _set_dream_tab_enabled(self, enabled: bool) -> None:
        dream_index = self.tabs.indexOf(self.dream_tab)
        if dream_index < 0:
            return
        self.tabs.setTabEnabled(dream_index, enabled)
        if not enabled and self.tabs.currentIndex() == dream_index:
            self.tabs.setCurrentWidget(self.prefit_tab)

    def _refresh_model_only_mode_state(self) -> None:
        model_only = bool(
            self.current_settings is not None
            and self.current_settings.model_only_mode
        )
        self.project_setup_tab.set_model_only_mode(model_only)
        self.prefit_tab.set_model_only_mode(model_only)
        prefit_enabled = (
            self.prefit_workflow is not None
            and self.prefit_workflow.can_run_prefit()
        )
        self.prefit_tab.set_prefit_execution_enabled(prefit_enabled)
        dream_enabled = (
            self.current_settings is not None
            and not model_only
            and self.prefit_workflow is not None
            and self.prefit_workflow.can_run_prefit()
        )
        self._set_dream_tab_enabled(dream_enabled)

    @Slot(bool)
    def _on_model_only_mode_changed(self, enabled: bool) -> None:
        if self.current_settings is None:
            return
        self.current_settings.model_only_mode = bool(enabled)
        if enabled:
            self.current_settings.use_experimental_grid = False
            if self.current_settings.q_points is None:
                self.current_settings.q_points = 500
        if self.prefit_workflow is not None:
            self.prefit_workflow.parameter_entries = (
                self.prefit_tab.parameter_entries()
            )
            self.prefit_workflow.set_model_only_mode(enabled)
            self.current_settings = self.prefit_workflow.settings
            self._load_prefit_preview()
            self.prefit_tab.set_log_text(self._format_prefit_console_intro())
        self._invalidate_dream_workflow_cache()
        self._refresh_model_only_mode_state()
        self.statusBar().showMessage(
            "Model Only Mode enabled"
            if enabled
            else "Model Only Mode disabled"
        )

    @Slot(bool)
    def _on_predicted_structure_weights_changed(self, enabled: bool) -> None:
        if self.current_settings is None:
            return
        try:
            settings = self._settings_from_project_tab()
            settings.use_predicted_structure_weights = bool(enabled)
            self._save_settings(settings)
            self.current_settings = settings
            self._apply_project_settings(settings)
            self.statusBar().showMessage(
                "Use Predicted Structure Weights enabled"
                if enabled
                else "Use Predicted Structure Weights disabled"
            )
        except Exception as exc:
            self._show_error(
                "Update Predicted Structures mode failed",
                str(exc),
            )

    @Slot(str)
    def _load_saved_distribution(self, distribution_id: str) -> None:
        if self.current_settings is None:
            return
        try:
            record = self.project_manager.load_saved_distribution(
                self.current_settings.project_dir,
                distribution_id,
            )
            settings = self.project_manager.settings_for_saved_distribution(
                self.current_settings.project_dir,
                distribution_id,
                base_settings=self._settings_from_project_tab(),
            )
            self._save_settings(settings)
            self.current_settings = settings
            self._apply_project_settings(settings)
            self.project_setup_tab.append_summary(
                "Loaded computed distribution.\n"
                f"Distribution: {record.label}\n"
                f"Folder: {record.distribution_dir}"
            )
            self.statusBar().showMessage("Computed distribution loaded")
        except Exception as exc:
            self._show_error("Load distribution failed", str(exc))

    def _settings_from_project_tab(self) -> ProjectSettings:
        project_dir = self.project_setup_tab.project_dir()
        if project_dir is None:
            raise ValueError("Select a project directory.")
        base = (
            ProjectSettings.from_dict(self.current_settings.to_dict())
            if self.current_settings is not None
            else ProjectSettings(
                project_name=project_dir.name,
                project_dir=str(project_dir),
            )
        )
        base.project_name = project_dir.name
        base.project_dir = str(project_dir)
        base.model_only_mode = self.project_setup_tab.model_only_mode()
        base.use_predicted_structure_weights = (
            self.project_setup_tab.use_predicted_structure_weights()
        )
        base.frames_dir = (
            str(self.project_setup_tab.frames_dir())
            if self.project_setup_tab.frames_dir() is not None
            else None
        )
        base.pdb_frames_dir = (
            str(self.project_setup_tab.pdb_frames_dir())
            if self.project_setup_tab.pdb_frames_dir() is not None
            else None
        )
        base.clusters_dir = (
            str(self.project_setup_tab.clusters_dir())
            if self.project_setup_tab.clusters_dir() is not None
            else None
        )
        base.experimental_data_path = (
            str(self.project_setup_tab.experimental_data_path())
            if self.project_setup_tab.experimental_data_path() is not None
            else None
        )
        base.copied_experimental_data_file = None
        base.solvent_data_path = (
            str(self.project_setup_tab.solvent_data_path())
            if self.project_setup_tab.solvent_data_path() is not None
            else None
        )
        base.copied_solvent_data_file = None
        base.experimental_header_rows = (
            self.project_setup_tab.experimental_header_rows()
        )
        base.experimental_q_column = (
            self.project_setup_tab.experimental_q_column()
        )
        base.experimental_intensity_column = (
            self.project_setup_tab.experimental_intensity_column()
        )
        base.experimental_error_column = (
            self.project_setup_tab.experimental_error_column()
        )
        base.solvent_header_rows = self.project_setup_tab.solvent_header_rows()
        base.solvent_q_column = self.project_setup_tab.solvent_q_column()
        base.solvent_intensity_column = (
            self.project_setup_tab.solvent_intensity_column()
        )
        base.solvent_error_column = (
            self.project_setup_tab.solvent_error_column()
        )
        base.q_min = self.project_setup_tab.q_min()
        base.q_max = self.project_setup_tab.q_max()
        base.use_experimental_grid = (
            self.project_setup_tab.use_experimental_grid()
        )
        base.q_points = self.project_setup_tab.q_points()
        base.available_elements = self.project_setup_tab.available_elements()
        base.cluster_inventory_rows = (
            self.project_setup_tab.recognized_cluster_rows()
        )
        base.include_elements = []
        base.exclude_elements = self.project_setup_tab.exclude_elements()
        base.component_trace_colors = (
            self.project_setup_tab.component_trace_colors()
        )
        base.component_trace_color_scheme = (
            self.project_setup_tab.component_trace_color_scheme()
        )
        base.experimental_trace_visible = (
            self.project_setup_tab.experimental_trace_visible()
        )
        base.experimental_trace_color = (
            self.project_setup_tab.experimental_trace_color()
        )
        base.solvent_trace_visible = (
            self.project_setup_tab.solvent_trace_visible()
        )
        base.solvent_trace_color = self.project_setup_tab.solvent_trace_color()
        base.selected_model_template = (
            self.project_setup_tab.active_template_name()
            or self.project_setup_tab.selected_template_name()
        )
        base.component_build_mode = (
            self.project_setup_tab.component_build_mode()
        )
        return base

    def _load_prefit_workflow(self) -> SAXSPrefitWorkflow:
        if self.current_settings is None:
            raise ValueError("No project is currently loaded.")
        self.prefit_workflow = SAXSPrefitWorkflow(
            self.current_settings.project_dir,
            template_name=self.current_settings.selected_model_template,
        )
        self.current_settings = self.prefit_workflow.settings
        self._sync_active_template_controls(
            self.prefit_workflow.template_spec.name,
            sync_selected=True,
        )
        template_specs = self._template_specs_for_dropdown(
            include_deprecated=self._show_deprecated_templates,
            selected_names=[self.prefit_workflow.template_spec.name],
        )
        self.prefit_tab.set_templates(
            template_specs,
            self.prefit_workflow.template_spec.name,
            active_name=self.prefit_workflow.template_spec.name,
        )
        self.prefit_tab.set_autosave(
            self.prefit_workflow.settings.autosave_prefits
        )
        self.prefit_tab.set_sequence_history_enabled(
            self.prefit_workflow.settings.prefit_sequence_history_enabled
        )
        self.prefit_tab.populate_parameter_table(
            self.prefit_workflow.parameter_entries
        )
        self._set_prefit_sequence_baseline(
            self.prefit_workflow.parameter_entries
        )
        self._refresh_prefit_volume_fraction_section()
        self._refresh_prefit_cluster_geometry_section()
        self._load_prefit_preview()
        self.prefit_tab.set_log_text(self._format_prefit_console_intro())
        if self.prefit_workflow.has_best_prefit_entries():
            self.prefit_tab.append_log(
                "Loaded the Best Prefit preset from the project file."
            )
        self._refresh_saved_prefit_states()
        self._refresh_model_only_mode_state()
        return self.prefit_workflow

    def _apply_prefit_template_fallback(
        self,
        settings: ProjectSettings,
        template_specs: list,
    ) -> None:
        selected_template = (
            str(settings.selected_model_template or "").strip() or None
        )
        self._sync_active_template_controls(
            selected_template,
            sync_selected=True,
        )
        self.prefit_tab.set_templates(
            template_specs,
            selected_template,
            active_name=selected_template,
        )
        self.prefit_tab.set_autosave(settings.autosave_prefits)
        self.prefit_tab.set_sequence_history_enabled(
            settings.prefit_sequence_history_enabled
        )
        self.prefit_tab.set_model_only_mode(settings.model_only_mode)
        self.prefit_tab.set_prefit_execution_enabled(False)
        self.prefit_tab.populate_parameter_table([])
        self._set_prefit_sequence_baseline([])
        template_spec = None
        if selected_template:
            try:
                template_spec = load_template_spec(selected_template)
            except Exception:
                template_spec = None
        target = self._volume_fraction_target_for_template_spec(template_spec)
        solvent_weight_target = self._solvent_weight_target_for_template_spec(
            template_spec
        )
        self.prefit_tab.set_solute_volume_fraction_visible(
            template_spec is not None
        )
        if target is None:
            self.prefit_tab.set_solute_volume_fraction_target(
                None,
                None,
                solvent_weight_target,
            )
        else:
            self.prefit_tab.set_solute_volume_fraction_target(
                *target,
                solvent_weight_target,
            )
        self._sync_solution_scattering_tool_targets()
        supports_cluster_geometry = bool(
            template_spec is not None
            and template_spec.cluster_geometry_support.supported
        )
        self.prefit_tab.set_cluster_geometry_visible(supports_cluster_geometry)
        if supports_cluster_geometry and template_spec is not None:
            self.prefit_tab.populate_cluster_geometry_table(
                [],
                mapping_options=[],
                allowed_sf_approximations=(
                    template_spec.cluster_geometry_support.allowed_sf_approximations
                ),
            )
            self.prefit_tab.set_cluster_geometry_status_text(
                "The active template expects per-cluster geometry metadata. "
                "Build prior weights and load the Prefit workflow to map "
                "component rows, then compute cluster geometry metadata."
            )
        else:
            self.prefit_tab.set_cluster_geometry_status_text(
                "This template does not use per-cluster geometry metadata."
            )

    @staticmethod
    def _volume_fraction_target_for_template_spec(
        template_spec,
    ) -> tuple[str, str] | None:
        if template_spec is None:
            return None
        parameter_names = {
            str(parameter.name).strip()
            for parameter in template_spec.parameters
            if str(parameter.name).strip()
        }
        for candidate in SOLUTE_VOLUME_FRACTION_PARAMETER_NAMES:
            if candidate in parameter_names:
                return candidate, "solute"
        for candidate in SOLVENT_VOLUME_FRACTION_PARAMETER_NAMES:
            if candidate in parameter_names:
                return candidate, "solvent"
        return None

    @staticmethod
    def _solvent_weight_target_for_template_spec(
        template_spec,
    ) -> str | None:
        if template_spec is None:
            return None
        parameter_names = {
            str(parameter.name).strip()
            for parameter in template_spec.parameters
            if str(parameter.name).strip()
        }
        for candidate in SOLVENT_WEIGHT_PARAMETER_NAMES:
            if candidate in parameter_names:
                return candidate
        return None

    def _load_dream_workflow(self) -> SAXSDreamWorkflow:
        if self.current_settings is None:
            raise ValueError("No project is currently loaded.")
        if self.current_settings.model_only_mode:
            raise ValueError(
                "DREAM is disabled in Model Only Mode. Disable Model Only "
                "Mode and add experimental SAXS data to enable DREAM."
            )
        if (
            self.prefit_workflow is not None
            and not self.prefit_workflow.can_run_prefit()
        ):
            raise ValueError(
                "DREAM requires experimental SAXS data and an enabled prefit workflow."
            )
        project_dir = str(Path(self.current_settings.project_dir).resolve())
        is_new_project = self._dream_workflow_project_dir != project_dir
        if self.dream_workflow is None or is_new_project:
            self.dream_workflow = SAXSDreamWorkflow(project_dir)
        self._dream_workflow_project_dir = project_dir
        selected_preset = self.dream_tab.selected_settings_preset_name()
        preset_names = self.dream_workflow.list_settings_presets()
        if selected_preset not in preset_names:
            selected_preset = None
        self.dream_tab.set_available_settings_presets(
            preset_names,
            selected_preset,
        )
        if not is_new_project:
            self._current_dream_preset_name = selected_preset
        if is_new_project:
            self._invalidate_written_dream_bundle()
            self._last_dream_constraint_warning_signature = None
            settings = self.dream_workflow.load_settings_preset(
                selected_preset
            )
            self.dream_tab.set_settings(settings, preset_name=selected_preset)
            self._active_dream_settings_snapshot = self._copy_dream_settings(
                settings
            )
            self._current_dream_preset_name = selected_preset
            try:
                parameter_map_entries = self.dream_workflow.load_parameter_map(
                    persist_if_missing=False
                )
            except Exception:
                parameter_map_entries = []
            self.dream_tab.set_parameter_map_entries(parameter_map_entries)
            self._dream_parameter_map_saved_in_session = False
            self._applied_dream_analysis_settings = None
            self._last_dream_summary = None
            self.dream_tab.set_log_text(self._format_dream_console_intro())
            self.dream_tab.set_summary_text(
                "DREAM results are not loaded yet."
            )
            self.dream_tab.set_filter_status_text(
                "No DREAM dataset is loaded yet. Load a run, then apply "
                "posterior or stoichiometry filters."
            )
            self.dream_tab.set_filter_dirty(False)
            self.dream_tab.reset_progress()
            self.dream_tab.clear_plots()
        self._refresh_saved_dream_runs()
        return self.dream_workflow

    def _list_saved_dream_runs(self) -> list[DreamSavedRunRecord]:
        if self.current_settings is None:
            return []
        runtime_dir = project_artifact_paths(
            self.current_settings
        ).dream_runtime_dir
        if not runtime_dir.is_dir():
            return []
        records: list[DreamSavedRunRecord] = []
        run_dirs = sorted(
            runtime_dir.glob("dream_*"), key=lambda path: path.name
        )
        for run_dir in reversed(run_dirs):
            if not (
                (run_dir / "dream_sampled_params.npy").is_file()
                and (run_dir / "dream_log_ps.npy").is_file()
            ):
                continue
            try:
                settings = self._load_saved_dream_run_settings(run_dir)
            except Exception:
                settings = DreamRunSettings()
            label_parts = [run_dir.name]
            if (
                settings.run_label.strip()
                and settings.run_label.strip() != "dream"
            ):
                label_parts.append(settings.run_label.strip())
            if settings.model_name:
                label_parts.append(settings.model_name)
            label_parts.append(
                f"{settings.nchains} chains x {settings.niterations} iter"
            )
            records.append(
                DreamSavedRunRecord(
                    run_dir=run_dir,
                    display_label=" | ".join(label_parts),
                )
            )
        return records

    def _refresh_saved_dream_runs(
        self,
        *,
        selected_run_dir: str | Path | None = None,
    ) -> None:
        records = self._list_saved_dream_runs()
        selected_value: str | None = None
        if selected_run_dir is not None:
            selected_value = str(Path(selected_run_dir).resolve())
        elif self._loaded_dream_run_dir is not None:
            selected_value = str(self._loaded_dream_run_dir.resolve())
        self.dream_tab.set_available_saved_runs(
            [
                (record.display_label, str(record.run_dir))
                for record in records
            ],
            selected_run_dir=selected_value,
        )

    def _refresh_prior_plot(self) -> Path | None:
        if self.current_settings is None:
            self.project_setup_tab.draw_prior_plot(None)
            return None
        artifact_paths = project_artifact_paths(self.current_settings)
        prior_json = artifact_paths.prior_weights_file
        resolved_prior = prior_json if prior_json.is_file() else None
        self.project_setup_tab.draw_prior_plot(resolved_prior)
        return resolved_prior

    def _refresh_component_plot(self) -> list[Path]:
        if self.current_settings is None:
            self.project_setup_tab.draw_component_plot(None)
            return []
        artifact_paths = project_artifact_paths(self.current_settings)
        component_paths = sorted(artifact_paths.component_dir.glob("*.txt"))
        self.project_setup_tab.draw_component_plot(component_paths or None)
        return component_paths

    def _active_cluster_import_view(
        self,
        settings: ProjectSettings,
        *,
        fallback_available_elements: list[str] | None = None,
        fallback_rows: list[dict[str, object]] | None = None,
    ) -> tuple[list[str], list[dict[str, object]]]:
        artifact_paths = project_artifact_paths(settings)
        prior_json = artifact_paths.prior_weights_file
        if not prior_json.is_file():
            filtered_rows = [dict(row) for row in (fallback_rows or [])]
            if not settings.use_predicted_structure_weights:
                filtered_rows = [
                    row
                    for row in filtered_rows
                    if str(row.get("source_kind", "cluster_dir"))
                    != "predicted_structure"
                ]
            return (
                list(fallback_available_elements or []),
                filtered_rows,
            )
        try:
            payload = json.loads(prior_json.read_text(encoding="utf-8"))
        except Exception:
            filtered_rows = [dict(row) for row in (fallback_rows or [])]
            if not settings.use_predicted_structure_weights:
                filtered_rows = [
                    row
                    for row in filtered_rows
                    if str(row.get("source_kind", "cluster_dir"))
                    != "predicted_structure"
                ]
            return (
                list(fallback_available_elements or []),
                filtered_rows,
            )
        return (
            [
                str(element)
                for element in payload.get("available_elements", [])
                if str(element).strip()
            ],
            self._cluster_rows_from_prior_payload(payload),
        )

    def _cluster_rows_from_prior_payload(
        self,
        payload: dict[str, object],
    ) -> list[dict[str, object]]:
        structures = payload.get("structures", {})
        rows: list[dict[str, object]] = []
        atom_weight_total = sum(
            max(
                sum(
                    int(token)
                    for token in re.findall(r"(\d+)", str(structure))
                ),
                1,
            )
            * float(
                motif_payload.get(
                    "normalized_weight",
                    motif_payload.get("weight", 0.0),
                )
                or 0.0
            )
            for structure, motif_map in dict(structures).items()
            for motif_payload in dict(motif_map).values()
            if isinstance(motif_payload, dict)
        )
        for structure, motif_map in dict(structures).items():
            for motif, motif_payload in dict(motif_map).items():
                if not isinstance(motif_payload, dict):
                    continue
                weight = float(
                    motif_payload.get(
                        "normalized_weight",
                        motif_payload.get("weight", 0.0),
                    )
                    or 0.0
                )
                rows.append(
                    {
                        "structure": str(structure),
                        "motif": str(motif),
                        "count": int(motif_payload.get("count", 0) or 0),
                        "weight": weight,
                        "atom_fraction_percent": (
                            weight
                            * max(
                                sum(
                                    int(token)
                                    for token in re.findall(
                                        r"(\d+)",
                                        str(structure),
                                    )
                                ),
                                1,
                            )
                            / atom_weight_total
                            * 100.0
                            if atom_weight_total > 0.0
                            else 0.0
                        ),
                        "structure_fraction_percent": weight * 100.0,
                        "source_kind": str(
                            motif_payload.get("source_kind", "cluster_dir")
                        ),
                        "source_dir": str(motif_payload.get("source_dir", "")),
                        "source_file": str(
                            motif_payload.get("source_file", "")
                        ),
                        "source_file_name": str(
                            motif_payload.get("source_file_name", "")
                        ),
                        "representative": str(
                            motif_payload.get("representative", "")
                        ),
                        "profile_file": str(
                            motif_payload.get("profile_file", "")
                        ),
                    }
                )
        rows.sort(
            key=lambda row: (
                str(row["structure"]).lower(),
                str(row["motif"]).lower(),
            )
        )
        return rows

    def _refresh_predicted_structure_status(
        self,
        settings: ProjectSettings | None = None,
    ) -> str:
        active_settings = settings or self.current_settings
        if active_settings is None:
            status_text = (
                self.project_setup_tab._default_predicted_structure_status_text()
            )
            self.project_setup_tab.set_predicted_structure_status_text(
                status_text
            )
            return status_text
        if not active_settings.use_predicted_structure_weights:
            status_text = (
                self.project_setup_tab._default_predicted_structure_status_text()
            )
            self.project_setup_tab.set_predicted_structure_status_text(
                status_text
            )
            return status_text
        artifact_paths = project_artifact_paths(active_settings)
        component_artifacts_ready = bool(
            artifact_paths.component_dir.is_dir()
            and any(artifact_paths.component_dir.glob("*.txt"))
            and artifact_paths.component_map_file.is_file()
        )
        prior_artifacts_ready = bool(
            artifact_paths.prior_weights_file.is_file()
        )
        predicted_state = self.project_manager.inspect_predicted_structures(
            active_settings.project_dir
        )
        if component_artifacts_ready and prior_artifacts_ready:
            status_text = (
                "Predicted Structures mode is on.\n"
                "Project Setup, Prefit, and DREAM are using the observed + "
                "Predicted Structures SAXS components and prior weights."
            )
            self.project_setup_tab.set_predicted_structure_status_text(
                status_text
            )
            return status_text
        if predicted_state.dataset_file is None:
            status_text = (
                "Predicted Structures mode is on, but no Cluster Dynamics ML "
                "prediction bundle was found in this project.\n"
                "Open Tools > Cluster Dynamics > Open Cluster Dynamics (ML), "
                "run a prediction, "
                "then rebuild SAXS components and prior weights."
            )
            self.project_setup_tab.set_predicted_structure_status_text(
                status_text
            )
            return status_text
        status_text = (
            "Predicted Structures mode is on.\n"
            f"Found {predicted_state.prediction_count} predicted structure"
            f"{'' if predicted_state.prediction_count == 1 else 's'} in "
            f"{predicted_state.dataset_file.name}. Rebuild SAXS components "
            "and prior weights to bring them into Project Setup, Prefit, and DREAM."
        )
        self.project_setup_tab.set_predicted_structure_status_text(status_text)
        return status_text

    def _refresh_saved_prefit_states(
        self,
        *,
        selected_name: str | None = None,
    ) -> None:
        if self.prefit_workflow is None:
            self.prefit_tab.set_saved_states([], None)
            return
        self.prefit_tab.set_saved_states(
            self.prefit_workflow.list_saved_state_options(),
            selected_name=selected_name,
        )

    @staticmethod
    def _copy_prefit_entries(
        entries: list[PrefitParameterEntry],
    ) -> list[PrefitParameterEntry]:
        return [
            PrefitParameterEntry.from_dict(entry.to_dict())
            for entry in entries
        ]

    def _set_prefit_sequence_baseline(
        self,
        entries: list[PrefitParameterEntry] | None,
    ) -> None:
        self._prefit_sequence_baseline_entries = self._copy_prefit_entries(
            entries or []
        )
        self._prefit_sequence_pending_manual_edits = False

    def _update_prefit_stoichiometry_status(self) -> None:
        if self.prefit_workflow is None:
            self.prefit_tab.set_stoichiometry_status_text(
                "Stoichiometry monitor: build a project to evaluate cluster-weight stoichiometry."
            )
            return
        settings = self.dream_tab.settings_payload()
        try:
            target = build_stoichiometry_target(
                settings.stoichiometry_target_elements_text,
                settings.stoichiometry_target_ratio_text,
            )
        except Exception as exc:
            self.prefit_tab.set_stoichiometry_status_text(
                f"Stoichiometry monitor: invalid target ({exc})"
            )
            return
        if target is None:
            self.prefit_tab.set_stoichiometry_status_text(
                "Stoichiometry monitor: configure target elements and ratio in DREAM > Posterior Filtering."
            )
            return
        try:
            entries = self.prefit_tab.parameter_entries()
        except Exception as exc:
            self.prefit_tab.set_stoichiometry_status_text(
                f"Stoichiometry monitor: waiting on valid Prefit parameter values ({exc})"
            )
            return
        evaluation = evaluate_weighted_stoichiometry(
            [
                (entry.structure, float(entry.value))
                for entry in entries
                if str(entry.name).startswith("w")
            ],
            target,
        )
        if not evaluation.is_valid:
            self.prefit_tab.set_stoichiometry_status_text(
                "Stoichiometry monitor:\n"
                f"Target: {stoichiometry_target_text(target)}\n"
                "Observed ratio: unavailable\n"
                "Deviation from target: unavailable"
            )
            return
        self.prefit_tab.set_stoichiometry_status_text(
            "Stoichiometry monitor:\n"
            f"Target: {stoichiometry_target_text(target)}\n"
            "Observed ratio: "
            f"{stoichiometry_ratio_text(target, evaluation.observed_ratio)}\n"
            f"{stoichiometry_deviation_text(evaluation)}"
        )

    def _on_prefit_parameter_table_edited(self) -> None:
        self._prefit_sequence_pending_manual_edits = True
        self._update_prefit_stoichiometry_status()

    @staticmethod
    def _prefit_entry_change_records(
        previous_entries: list[PrefitParameterEntry],
        current_entries: list[PrefitParameterEntry],
    ) -> list[dict[str, object]]:
        def _entry_key(entry: PrefitParameterEntry) -> tuple[str, str, str]:
            return (
                str(entry.structure).strip(),
                str(entry.motif).strip(),
                str(entry.name).strip(),
            )

        previous_lookup = {
            _entry_key(entry): entry for entry in previous_entries
        }
        current_lookup = {
            _entry_key(entry): entry for entry in current_entries
        }
        change_records: list[dict[str, object]] = []
        for key in sorted(set(previous_lookup) | set(current_lookup)):
            previous_entry = previous_lookup.get(key)
            current_entry = current_lookup.get(key)
            previous_payload = (
                None if previous_entry is None else previous_entry.to_dict()
            )
            current_payload = (
                None if current_entry is None else current_entry.to_dict()
            )
            if previous_payload == current_payload:
                continue
            structure, motif, parameter_name = key
            change_records.append(
                {
                    "structure": structure,
                    "motif": motif,
                    "parameter_name": parameter_name,
                    "previous": previous_payload,
                    "current": current_payload,
                }
            )
        return change_records

    def _append_prefit_sequence_event(
        self,
        event_type: str,
        summary: str,
        *,
        details: dict[str, object] | None = None,
        parameter_entries: list[PrefitParameterEntry] | None = None,
        force: bool = False,
    ) -> Path | None:
        if self.prefit_workflow is None:
            return None
        try:
            return self.prefit_workflow.append_sequence_history_event(
                event_type,
                summary,
                details=details,
                parameter_entries=parameter_entries,
                force=force,
            )
        except Exception:
            return None

    def _flush_prefit_sequence_pending_manual_updates(
        self,
        *,
        trigger: str,
    ) -> list[dict[str, object]]:
        if self.prefit_workflow is None:
            return []
        if not self._prefit_sequence_pending_manual_edits:
            return []
        try:
            current_entries = self.prefit_tab.parameter_entries()
        except Exception:
            return []
        change_records = self._prefit_entry_change_records(
            self._prefit_sequence_baseline_entries,
            current_entries,
        )
        if change_records:
            self._append_prefit_sequence_event(
                "manual_parameter_update",
                "Recorded manual Prefit parameter edits.",
                details={
                    "trigger": trigger,
                    "changed_parameters": change_records,
                },
                parameter_entries=current_entries,
            )
        self._set_prefit_sequence_baseline(current_entries)
        return change_records

    def _save_distribution_entries(self, entries: list) -> None:
        try:
            workflow = self._load_dream_workflow()
            workflow.save_parameter_map(entries)
            self._dream_parameter_map_saved_in_session = True
            self._invalidate_written_dream_bundle()
            self._last_dream_constraint_warning_signature = None
            self.dream_tab.set_parameter_map_entries(entries)
            self.dream_tab.append_log(
                "Updated DREAM parameter map.\n"
                "The current DREAM session is now allowed to write or run a "
                "bundle with these priors."
            )
            self._append_dream_vary_recommendation(entries)
        except Exception as exc:
            self._show_error("Save priors failed", str(exc))

    def _sync_dream_parameter_map_from_prefit(
        self,
        *,
        source_label: str,
    ) -> list[DreamParameterEntry] | None:
        if self.prefit_workflow is None:
            return None
        try:
            workflow = self._load_dream_workflow()
        except Exception:
            return None
        entries = workflow.sync_parameter_map_to_prefit(persist=True)
        self._dream_parameter_map_saved_in_session = True
        self._invalidate_written_dream_bundle()
        self._last_dream_constraint_warning_signature = None
        self.dream_tab.set_parameter_map_entries(entries)
        self.dream_tab.append_log(
            "Updated DREAM parameter map from Prefit.\n"
            f"Source: {source_label}\n"
            "The current DREAM priors keep their saved vary flags and "
            "distribution families, but their centers now follow the latest "
            "Prefit baseline."
        )
        return entries

    def _on_project_setup_template_selected(
        self,
        template_name: str,
    ) -> None:
        if not template_name or self._syncing_template_controls:
            return
        self._sync_template_selection_controls(
            template_name,
            source="project_setup",
        )

    def _on_prefit_template_changed(self, template_name: str) -> None:
        if not template_name or self._syncing_template_controls:
            return
        self._sync_template_selection_controls(
            template_name,
            source="prefit",
        )

    def _on_change_template_requested(self, template_name: str) -> None:
        normalized_template = str(template_name or "").strip()
        if not normalized_template:
            return
        if self.current_settings is None:
            self._show_error(
                "Template change failed",
                "Create or open a SAXS project before changing the active template.",
            )
            return
        current_template = self._active_template_name()
        if normalized_template == current_template:
            self._sync_active_template_controls(
                current_template,
                sync_selected=True,
            )
            return
        should_continue, disable_future_warnings = (
            self._confirm_prefit_template_change(
                current_template or "not selected",
                normalized_template,
            )
        )
        if not should_continue:
            self._sync_active_template_controls(
                current_template,
                sync_selected=True,
            )
            return
        if disable_future_warnings:
            self._warn_on_prefit_template_change = False
        try:
            settings = self._settings_from_project_tab()
            if current_template:
                settings.selected_model_template = current_template
            current_distribution_id = project_artifact_paths(
                settings
            ).distribution_id
            cloned_record = None
            if current_distribution_id:
                try:
                    cloned_record = (
                        self.project_manager.clone_distribution_for_template(
                            settings.project_dir,
                            current_distribution_id,
                            normalized_template,
                            base_settings=settings,
                        )
                    )
                except FileNotFoundError:
                    cloned_record = None
            if cloned_record is not None:
                settings = (
                    self.project_manager.settings_for_saved_distribution(
                        settings.project_dir,
                        cloned_record.distribution_id,
                        base_settings=settings,
                    )
                )
            else:
                settings.selected_model_template = normalized_template
            self._save_settings(settings)
            self.current_settings = settings
            self._apply_project_settings(settings)
            summary_lines = [
                "Changed active template for the current computed distribution.",
                f"Previous template: {current_template or 'not selected'}",
                f"Current template: {normalized_template}",
            ]
            if cloned_record is not None:
                summary_lines.extend(
                    [
                        f"Distribution: {cloned_record.label}",
                        f"Folder: {cloned_record.distribution_dir}",
                    ]
                )
            else:
                summary_lines.append(
                    "No saved computed distribution metadata was available to clone, "
                    "so the active project template was updated directly."
                )
            summary_lines.append(
                "Recompute cluster geometry metadata for this template-specific "
                "distribution before using geometry-dependent Prefit models."
            )
            self.project_setup_tab.append_summary("\n".join(summary_lines))
            self.prefit_tab.append_log(
                "Active template changed.\n"
                f"Previous template: {current_template or 'not selected'}\n"
                f"Current template: {normalized_template}\n"
                "This template now points to its own computed distribution. "
                "Recompute cluster geometry metadata before using "
                "geometry-dependent Prefit models."
            )
            self.statusBar().showMessage("Active template updated")
        except Exception as exc:
            self._sync_active_template_controls(
                current_template,
                sync_selected=True,
            )
            self._show_error("Template change failed", str(exc))

    def _on_autosave_changed(self, enabled: bool) -> None:
        if self.prefit_workflow is None:
            return
        self.prefit_workflow.set_autosave(enabled)
        if self.current_settings is not None:
            self.current_settings = self.prefit_workflow.settings
        self._append_prefit_sequence_event(
            "autosave_toggled",
            "Changed Prefit autosave setting.",
            details={"enabled": bool(enabled)},
            parameter_entries=self.prefit_workflow.parameter_entries,
        )
        self.prefit_tab.append_log(
            "Autosave fit results " + ("enabled." if enabled else "disabled.")
        )

    def _on_prefit_sequence_history_changed(self, enabled: bool) -> None:
        if self.prefit_workflow is None:
            return
        self.prefit_workflow.set_sequence_history_enabled(enabled)
        if self.current_settings is not None:
            self.current_settings = self.prefit_workflow.settings
        try:
            current_entries = self.prefit_tab.parameter_entries()
        except Exception:
            current_entries = self.prefit_workflow.parameter_entries
        self._set_prefit_sequence_baseline(current_entries)
        history_path = self._append_prefit_sequence_event(
            "sequence_history_logger_toggled",
            "Changed Prefit sequence history logging.",
            details={"enabled": bool(enabled)},
            parameter_entries=current_entries,
            force=True,
        )
        if enabled and history_path is not None:
            self.prefit_tab.append_log(
                "Sequence history logger enabled.\n"
                f"History file: {history_path}"
            )
            self.statusBar().showMessage("Prefit sequence history enabled")
            return
        self.prefit_tab.append_log("Sequence history logger disabled.")
        self.statusBar().showMessage("Prefit sequence history disabled")

    def _on_dream_settings_preset_changed(self, _text: str) -> None:
        if self.dream_workflow is None:
            return
        try:
            preset_name = self.dream_tab.selected_settings_preset_name()
            previous_preset_name = self._current_dream_preset_name
            if (
                previous_preset_name is None
                and preset_name is not None
                and self._loaded_dream_run_dir is None
            ):
                self._active_dream_settings_snapshot = (
                    self._copy_dream_settings(
                        self.dream_tab.settings_payload()
                    )
                )
            if preset_name is None:
                settings = (
                    self._copy_dream_settings(
                        self._active_dream_settings_snapshot
                    )
                    if self._active_dream_settings_snapshot is not None
                    else self.dream_workflow.load_settings()
                )
            else:
                settings = self.dream_workflow.load_settings_preset(
                    preset_name
                )
            self.dream_tab.set_settings(settings, preset_name=preset_name)
            self._current_dream_preset_name = preset_name
            self._invalidate_written_dream_bundle()
            self.dream_tab.append_log(
                "Loaded DREAM settings preset: "
                f"{preset_name or 'active project settings'}"
            )
            self._on_dream_analysis_settings_changed()
        except Exception as exc:
            self._show_error("Load DREAM settings failed", str(exc))

    def _on_dream_analysis_settings_changed(self) -> None:
        self._update_prefit_stoichiometry_status()
        if self._last_results_loader is None:
            self.dream_tab.set_filter_dirty(False)
            return
        self.dream_tab.set_filter_dirty(True)
        self._refresh_dream_filter_status(dirty=True)

    def _apply_dream_filter_settings(self) -> None:
        if self._last_results_loader is None:
            self.dream_tab.set_filter_status_text(
                "No DREAM dataset is loaded yet. Load a run before applying "
                "posterior or stoichiometry filters."
            )
            self.dream_tab.set_filter_dirty(False)
            return
        self._refresh_loaded_dream_results(scope=self.DREAM_REFRESH_FULL)

    @staticmethod
    def _copy_dream_settings(
        settings: DreamRunSettings,
    ) -> DreamRunSettings:
        return DreamRunSettings.from_dict(settings.to_dict())

    def _load_console_autoscroll_setting(self) -> bool:
        raw_value = self._recent_projects_settings().value(
            CONSOLE_AUTOSCROLL_KEY,
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

    def _toggle_console_autoscroll(self, enabled: bool) -> None:
        self._set_console_autoscroll_enabled(enabled, persist=True)

    def _set_console_autoscroll_enabled(
        self,
        enabled: bool,
        *,
        persist: bool,
    ) -> None:
        self._console_autoscroll_enabled = bool(enabled)
        if hasattr(self, "console_autoscroll_action"):
            self.console_autoscroll_action.blockSignals(True)
            self.console_autoscroll_action.setChecked(
                self._console_autoscroll_enabled
            )
            self.console_autoscroll_action.blockSignals(False)
        if hasattr(self, "project_setup_tab"):
            self.project_setup_tab.set_console_autoscroll_enabled(
                self._console_autoscroll_enabled
            )
        if hasattr(self, "prefit_tab"):
            self.prefit_tab.set_console_autoscroll_enabled(
                self._console_autoscroll_enabled
            )
        if hasattr(self, "dream_tab"):
            self.dream_tab.set_console_autoscroll_enabled(
                self._console_autoscroll_enabled
            )
        if persist:
            self._recent_projects_settings().setValue(
                CONSOLE_AUTOSCROLL_KEY,
                self._console_autoscroll_enabled,
            )
            self.statusBar().showMessage(
                "Console autoscroll "
                + (
                    "enabled"
                    if self._console_autoscroll_enabled
                    else "disabled"
                )
            )

    def _recent_projects_settings(self) -> QSettings:
        return QSettings("SAXShell", "SAXS")

    def _recent_project_paths(self) -> list[str]:
        raw_value = self._recent_projects_settings().value(
            RECENT_PROJECTS_KEY,
            [],
        )
        if isinstance(raw_value, str):
            candidates = [raw_value]
        elif isinstance(raw_value, (list, tuple)):
            candidates = [str(item) for item in raw_value]
        else:
            candidates = []
        existing_paths: list[str] = []
        for candidate in candidates:
            normalized = candidate.strip()
            if not normalized:
                continue
            if Path(normalized).expanduser().exists():
                existing_paths.append(normalized)
        if existing_paths != candidates:
            self._recent_projects_settings().setValue(
                RECENT_PROJECTS_KEY,
                existing_paths,
            )
        return existing_paths[:MAX_RECENT_PROJECTS]

    def _remember_recent_project(self, project_dir: str | Path) -> None:
        normalized = str(Path(project_dir).expanduser().resolve())
        recent = [
            path for path in self._recent_project_paths() if path != normalized
        ]
        recent.insert(0, normalized)
        self._recent_projects_settings().setValue(
            RECENT_PROJECTS_KEY,
            recent[:MAX_RECENT_PROJECTS],
        )
        self._refresh_recent_projects_menu()

    def _refresh_recent_projects_menu(self) -> None:
        self.open_recent_menu.clear()
        recent_paths = self._recent_project_paths()
        if not recent_paths:
            empty_action = self.open_recent_menu.addAction(
                "No recent projects"
            )
            empty_action.setEnabled(False)
            return
        for project_path in recent_paths:
            action = self.open_recent_menu.addAction(project_path)
            action.triggered.connect(
                lambda checked=False, path=project_path: self._open_recent_project(
                    path
                )
            )

    def _open_recent_project(self, project_path: str | Path) -> None:
        try:
            self.tabs.setCurrentWidget(self.project_setup_tab)
            self.load_project(self._validated_project_dir(project_path))
        except Exception as exc:
            self._show_error("Open recent project failed", str(exc))

    def _update_file_menu_state(self) -> None:
        has_project = self.current_settings is not None
        self.save_project_action.setEnabled(has_project)
        self.save_project_as_action.setEnabled(has_project)

    def _remap_copied_project_paths(
        self,
        settings: ProjectSettings,
        *,
        old_project_dir: Path,
        new_project_dir: Path,
    ) -> None:
        for attribute in (
            "experimental_data_path",
            "copied_experimental_data_file",
            "solvent_data_path",
            "copied_solvent_data_file",
            "frames_dir",
            "pdb_frames_dir",
            "clusters_dir",
            "trajectory_file",
            "topology_file",
            "energy_file",
        ):
            current_value = getattr(settings, attribute)
            if not current_value:
                continue
            remapped = self._remap_if_within_project(
                current_value,
                old_project_dir=old_project_dir,
                new_project_dir=new_project_dir,
            )
            setattr(settings, attribute, remapped)
        self._restore_internal_staged_paths(settings, new_project_dir)

    @staticmethod
    def _restore_internal_staged_paths(
        settings: ProjectSettings,
        project_dir: Path,
    ) -> None:
        experimental_dir = (project_dir / "experimental_data").resolve()
        if (
            not settings.copied_experimental_data_file
            and settings.experimental_data_path
        ):
            experimental_path = Path(settings.experimental_data_path).resolve()
            if experimental_dir in experimental_path.parents:
                settings.copied_experimental_data_file = str(experimental_path)
        if (
            not settings.copied_solvent_data_file
            and settings.solvent_data_path
        ):
            solvent_path = Path(settings.solvent_data_path).resolve()
            if experimental_dir in solvent_path.parents:
                settings.copied_solvent_data_file = str(solvent_path)

    @staticmethod
    def _remap_if_within_project(
        path_text: str,
        *,
        old_project_dir: Path,
        new_project_dir: Path,
    ) -> str:
        try:
            resolved_path = Path(path_text).expanduser().resolve()
            relative = resolved_path.relative_to(old_project_dir.resolve())
        except Exception:
            return path_text
        return str((new_project_dir / relative).resolve())

    def _active_project_launch_settings(self) -> ProjectSettings | None:
        if self.current_settings is None:
            return None
        try:
            return self.project_manager.load_project(
                self.current_settings.project_dir
            )
        except Exception:
            return self.current_settings

    def _active_contrast_distribution_artifact_context(
        self,
        settings: ProjectSettings | None = None,
    ) -> tuple[str | None, Path | None, Path | None]:
        active_settings = settings or self.current_settings
        if active_settings is None:
            return None, None, None
        if (
            normalize_component_build_mode(
                active_settings.component_build_mode
            )
            != COMPONENT_BUILD_MODE_CONTRAST
        ):
            return None, None, None
        artifact_paths = project_artifact_paths(active_settings)
        contrast_dir = artifact_paths.contrast_dir
        if (
            not artifact_paths.uses_distribution_storage
            or artifact_paths.distribution_id is None
            or not contrast_dir.is_dir()
        ):
            return None, None, None
        if not (
            (contrast_dir / "selection_summary.json").is_file()
            or (contrast_dir / "debye" / "component_summary.json").is_file()
        ):
            return None, None, None
        return (
            artifact_paths.distribution_id,
            artifact_paths.root_dir,
            contrast_dir,
        )

    def _update_active_contrast_distribution_view_state(
        self,
        settings: ProjectSettings | None = None,
    ) -> None:
        distribution_id, _root_dir, contrast_dir = (
            self._active_contrast_distribution_artifact_context(settings)
        )
        self.project_setup_tab.set_active_contrast_distribution_view_available(
            bool(distribution_id and contrast_dir is not None)
        )

    @Slot()
    def _open_active_contrast_distribution_view(self) -> None:
        distribution_id, _root_dir, contrast_dir = (
            self._active_contrast_distribution_artifact_context()
        )
        if distribution_id is None or contrast_dir is None:
            self._show_error(
                "Contrast distribution unavailable",
                "The active computed distribution does not have saved "
                "contrast-mode representative outputs to reopen.",
            )
            return
        self._open_contrast_mode_tool(load_saved_distribution_view=True)

    def _connect_project_path_updates(self, window: object) -> None:
        signal = getattr(window, "project_paths_registered", None)
        if signal is None or not hasattr(signal, "connect"):
            return
        signal.connect(self._sync_project_folder_references_from_child)

    def _connect_contrast_mode_updates(self, window: object) -> None:
        signal = getattr(window, "contrast_components_built", None)
        if signal is None or not hasattr(signal, "connect"):
            return
        signal.connect(self._on_contrast_components_built)

    def _track_child_tool_window(self, window: object) -> None:
        if window in self._child_tool_windows:
            return
        if isinstance(window, QWidget):
            window.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        destroyed_signal = getattr(window, "destroyed", None)
        if destroyed_signal is not None and hasattr(
            destroyed_signal, "connect"
        ):
            destroyed_signal.connect(
                lambda _obj=None, win=window: self._forget_child_tool_window(
                    win
                )
            )
        self._child_tool_windows.append(window)

    def _forget_child_tool_window(self, window: object) -> None:
        self._child_tool_windows = [
            existing
            for existing in self._child_tool_windows
            if existing is not window
        ]

    def _close_child_tool_windows(self) -> bool:
        for window in list(self._child_tool_windows):
            close_method = getattr(window, "close", None)
            if not callable(close_method):
                continue
            result = close_method()
            if result is False:
                return False
        return True

    @Slot(object)
    def _sync_project_folder_references_from_child(
        self,
        payload: object,
    ) -> None:
        if self.current_settings is None or not isinstance(payload, dict):
            return
        project_dir_value = payload.get("project_dir")
        if project_dir_value is None:
            return
        project_dir = Path(project_dir_value).expanduser().resolve()
        active_project_dir = (
            Path(self.current_settings.project_dir).expanduser().resolve()
        )
        if project_dir != active_project_dir:
            return

        try:
            reloaded_settings = self.project_manager.load_project(project_dir)
        except Exception:
            reloaded_settings = None

        updated_labels: list[str] = []
        with self.project_setup_tab.batch_preview_updates():
            if "frames_dir" in payload:
                resolved_frames_dir = (
                    reloaded_settings.frames_dir
                    if reloaded_settings is not None
                    else (
                        None
                        if payload["frames_dir"] is None
                        else str(
                            Path(payload["frames_dir"]).expanduser().resolve()
                        )
                    )
                )
                self.project_setup_tab.frames_dir_edit.setText(
                    resolved_frames_dir or ""
                )
                self.current_settings.frames_dir = resolved_frames_dir
                self.current_settings.frames_dir_snapshot = None
                updated_labels.append("frames folder")
            if "pdb_frames_dir" in payload:
                resolved_pdb_frames_dir = (
                    reloaded_settings.pdb_frames_dir
                    if reloaded_settings is not None
                    else (
                        None
                        if payload["pdb_frames_dir"] is None
                        else str(
                            Path(payload["pdb_frames_dir"])
                            .expanduser()
                            .resolve()
                        )
                    )
                )
                self.project_setup_tab.pdb_frames_dir_edit.setText(
                    resolved_pdb_frames_dir or ""
                )
                self.current_settings.pdb_frames_dir = resolved_pdb_frames_dir
                self.current_settings.pdb_frames_dir_snapshot = None
                updated_labels.append("PDB structure folder")
            if "clusters_dir" in payload:
                resolved_clusters_dir = (
                    reloaded_settings.clusters_dir
                    if reloaded_settings is not None
                    else (
                        None
                        if payload["clusters_dir"] is None
                        else str(
                            Path(payload["clusters_dir"])
                            .expanduser()
                            .resolve()
                        )
                    )
                )
                self.project_setup_tab.clusters_dir_edit.setText(
                    resolved_clusters_dir or ""
                )
                self.current_settings.clusters_dir = resolved_clusters_dir
                self.current_settings.clusters_dir_snapshot = None
                updated_labels.append("clusters folder")

        if "clusters_dir" in payload:
            self.project_setup_tab.request_cluster_scan()

        if updated_labels:
            summary = ", ".join(updated_labels)
            self.project_setup_tab.append_summary(
                "Updated project folder references from a linked tool: "
                f"{summary}."
            )
            self.statusBar().showMessage(
                "Updated project folder references from linked tool"
            )

    def _open_mdtrajectory_tool(self) -> None:
        from saxshell.mdtrajectory.ui.main_window import (
            launch_mdtrajectory_app,
        )

        settings = self._active_project_launch_settings()
        project_dir = None
        trajectory_file = None
        topology_file = None
        energy_file = None
        if settings is not None:
            project_dir = Path(settings.project_dir).resolve()
            trajectory_file = settings.resolved_trajectory_file
            topology_file = settings.resolved_topology_file
            energy_file = settings.resolved_energy_file
        window = launch_mdtrajectory_app(
            project_dir=project_dir,
            trajectory_file=trajectory_file,
            topology_file=topology_file,
            energy_file=energy_file,
        )
        self._connect_project_path_updates(window)
        self._track_child_tool_window(window)
        if project_dir is not None:
            self.statusBar().showMessage(
                f"Opened MD trajectory extraction for {project_dir}"
            )
        else:
            self.statusBar().showMessage("Opened MD trajectory extraction")

    def _open_cluster_tool(self) -> None:
        from saxshell.cluster.ui.main_window import ClusterMainWindow

        settings = self._active_project_launch_settings()
        frames_dir = None
        project_dir = None
        if settings is not None:
            frames_dir = settings.resolved_frames_dir
            project_dir = Path(settings.project_dir).resolve()
        window = ClusterMainWindow(
            initial_frames_dir=frames_dir,
            initial_project_dir=project_dir,
        )
        self._connect_project_path_updates(window)
        window.show()
        window.raise_()
        self._track_child_tool_window(window)
        if frames_dir is not None:
            self.statusBar().showMessage(
                f"Opened cluster extraction for {frames_dir}"
            )
        else:
            self.statusBar().showMessage("Opened cluster extraction")

    def _open_xyz2pdb_tool(self) -> None:
        from saxshell.xyz2pdb.ui.main_window import launch_xyz2pdb_ui

        settings = self._active_project_launch_settings()
        input_path = None
        project_dir = None
        if settings is not None:
            input_path = settings.resolved_frames_dir
            project_dir = Path(settings.project_dir).resolve()
        window = launch_xyz2pdb_ui(
            input_path=input_path,
            project_dir=project_dir,
        )
        self._connect_project_path_updates(window)
        self._track_child_tool_window(window)
        if input_path is not None:
            self.statusBar().showMessage(
                f"Opened XYZ -> PDB conversion for {input_path}"
            )
        else:
            self.statusBar().showMessage("Opened XYZ -> PDB conversion")

    def _open_bondanalysis_tool(self) -> None:
        from saxshell.bondanalysis.ui.main_window import BondAnalysisMainWindow

        settings = self._active_project_launch_settings()
        clusters_dir = None
        project_dir = None
        if settings is not None:
            clusters_dir = settings.clusters_dir
            project_dir = Path(settings.project_dir).resolve()
        window = BondAnalysisMainWindow(
            initial_clusters_dir=clusters_dir,
            initial_project_dir=project_dir,
        )
        window.show()
        window.raise_()
        self._track_child_tool_window(window)
        if clusters_dir:
            self.statusBar().showMessage(
                f"Opened bond analysis for {clusters_dir}"
            )
        else:
            self.statusBar().showMessage("Opened bond analysis")

    def _open_clusterdynamics_tool(self) -> None:
        from saxshell.clusterdynamics.ui.main_window import (
            ClusterDynamicsMainWindow,
        )

        settings = self._active_project_launch_settings()
        project_dir = None
        frames_dir = None
        energy_file = None
        if settings is not None:
            project_dir = Path(settings.project_dir).resolve()
            frames_dir = settings.resolved_frames_dir
            energy_file = settings.resolved_energy_file
        window = ClusterDynamicsMainWindow(
            initial_frames_dir=frames_dir,
            initial_energy_file=energy_file,
            initial_project_dir=project_dir,
        )
        window.show()
        window.raise_()
        self._track_child_tool_window(window)
        if project_dir is not None:
            self.statusBar().showMessage(
                f"Opened cluster dynamics for {project_dir}"
            )
        else:
            self.statusBar().showMessage("Opened cluster dynamics")

    def _open_clusterdynamicsml_tool(self) -> None:
        from saxshell.clusterdynamicsml.ui.main_window import (
            ClusterDynamicsMLMainWindow,
        )

        settings = self._active_project_launch_settings()
        project_dir = None
        frames_dir = None
        clusters_dir = None
        energy_file = None
        experimental_data_file = None
        if settings is not None:
            project_dir = Path(settings.project_dir).resolve()
            frames_dir = settings.resolved_frames_dir
            clusters_dir = settings.resolved_clusters_dir
            energy_file = settings.resolved_energy_file
            experimental_data_file = settings.resolved_experimental_data_path
        window = ClusterDynamicsMLMainWindow(
            initial_frames_dir=frames_dir,
            initial_energy_file=energy_file,
            initial_project_dir=project_dir,
            initial_clusters_dir=clusters_dir,
            initial_experimental_data_file=experimental_data_file,
        )
        window.show()
        window.raise_()
        self._track_child_tool_window(window)
        if project_dir is not None:
            self.statusBar().showMessage(
                f"Opened cluster dynamics (ML) for {project_dir}"
            )
        else:
            self.statusBar().showMessage("Opened cluster dynamics (ML)")

    def _open_fullrmc_tool(self) -> None:
        from saxshell.fullrmc.ui.main_window import RMCSetupMainWindow

        project_dir = None
        if self.current_settings is not None:
            project_dir = self.current_settings.project_dir
        window = RMCSetupMainWindow(initial_project_dir=project_dir)
        window.show()
        window.raise_()
        self._track_child_tool_window(window)
        if project_dir:
            self.statusBar().showMessage(
                f"Opened fullrmc setup for {project_dir}"
            )
        else:
            self.statusBar().showMessage("Opened fullrmc setup")

    def _open_pdfsetup_tool(self) -> None:
        from saxshell.pdf.debyer.ui.main_window import DebyerPDFMainWindow

        settings = self._active_project_launch_settings()
        project_dir = None
        frames_dir = None
        if settings is not None:
            project_dir = Path(settings.project_dir).resolve()
            frames_dir = settings.resolved_frames_dir
        window = DebyerPDFMainWindow(
            initial_project_dir=project_dir,
            initial_frames_dir=frames_dir,
        )
        window.show()
        window.raise_()
        self._track_child_tool_window(window)
        if project_dir is not None:
            self.statusBar().showMessage(
                f"Opened PDF calculation for {project_dir}"
            )
        else:
            self.statusBar().showMessage("Opened PDF calculation")

    def _open_blenderxyz_tool(self) -> None:
        from saxshell.toolbox.blender.ui.main_window import (
            launch_blender_xyz_renderer_ui,
        )

        window = launch_blender_xyz_renderer_ui()
        self._track_child_tool_window(window)
        self.statusBar().showMessage("Opened Blender XYZ renderer")

    def _open_contrast_mode_tool(
        self,
        *,
        load_saved_distribution_view: bool = False,
    ) -> None:
        existing_window = self._contrast_mode_tool_window
        if isinstance(existing_window, QWidget):
            settings = self._active_project_launch_settings()
            distribution_id = None
            distribution_root_dir = None
            contrast_artifact_dir = None
            if settings is not None and load_saved_distribution_view:
                (
                    distribution_id,
                    distribution_root_dir,
                    contrast_artifact_dir,
                ) = self._active_contrast_distribution_artifact_context(
                    settings
                )
            if settings is not None and hasattr(
                existing_window, "apply_launch_context"
            ):
                from saxshell.saxs.contrast.settings import (
                    ContrastModeLaunchContext,
                )

                existing_window.apply_launch_context(
                    ContrastModeLaunchContext.from_values(
                        project_dir=settings.project_dir,
                        clusters_dir=settings.resolved_clusters_dir,
                        experimental_data_file=(
                            settings.resolved_experimental_data_path
                        ),
                        q_min=settings.q_min,
                        q_max=settings.q_max,
                        active_template_name=settings.selected_model_template,
                        active_distribution_id=distribution_id,
                        distribution_root_dir=distribution_root_dir,
                        contrast_artifact_dir=contrast_artifact_dir,
                    )
                )
            existing_window.show()
            existing_window.raise_()
            existing_window.activateWindow()
            self.statusBar().showMessage("Opened SAXS contrast-mode workflow")
            return

        from saxshell.saxs.contrast.ui.main_window import (
            launch_contrast_mode_ui,
        )

        settings = self._active_project_launch_settings()
        project_dir = None
        clusters_dir = None
        experimental_data_file = None
        q_min = None
        q_max = None
        template_name = None
        distribution_id = None
        distribution_root_dir = None
        contrast_artifact_dir = None
        if settings is not None:
            project_dir = Path(settings.project_dir).resolve()
            clusters_dir = settings.resolved_clusters_dir
            experimental_data_file = settings.resolved_experimental_data_path
            q_min = settings.q_min
            q_max = settings.q_max
            template_name = settings.selected_model_template
            if load_saved_distribution_view:
                (
                    distribution_id,
                    distribution_root_dir,
                    contrast_artifact_dir,
                ) = self._active_contrast_distribution_artifact_context(
                    settings
                )
        window = launch_contrast_mode_ui(
            initial_project_dir=project_dir,
            initial_clusters_dir=clusters_dir,
            initial_experimental_data_file=experimental_data_file,
            initial_q_min=q_min,
            initial_q_max=q_max,
            initial_template_name=template_name,
            initial_distribution_id=distribution_id,
            initial_distribution_root_dir=distribution_root_dir,
            initial_contrast_artifact_dir=contrast_artifact_dir,
        )
        self._connect_project_path_updates(window)
        self._connect_contrast_mode_updates(window)
        self._contrast_mode_tool_window = window
        destroyed_signal = getattr(window, "destroyed", None)
        if destroyed_signal is not None and hasattr(
            destroyed_signal, "connect"
        ):
            destroyed_signal.connect(
                lambda *_args: setattr(
                    self,
                    "_contrast_mode_tool_window",
                    None,
                )
            )
        self._track_child_tool_window(window)
        if project_dir is not None:
            self.statusBar().showMessage(
                f"Opened SAXS contrast-mode workflow for {project_dir}"
            )
        else:
            self.statusBar().showMessage("Opened SAXS contrast-mode workflow")

    @Slot(object)
    def _on_contrast_components_built(self, payload: object) -> None:
        if self.current_settings is None or not isinstance(payload, dict):
            return
        distribution_id = str(payload.get("distribution_id") or "").strip()
        project_dir_value = str(payload.get("project_dir") or "").strip()
        if not distribution_id or not project_dir_value:
            return
        active_project_dir = (
            Path(self.current_settings.project_dir).expanduser().resolve()
        )
        payload_project_dir = Path(project_dir_value).expanduser().resolve()
        if payload_project_dir != active_project_dir:
            return
        self._load_saved_distribution(distribution_id)
        self.statusBar().showMessage(
            "Contrast SAXS components built and loaded"
        )

    def _open_solution_scattering_tool_window(
        self,
        *,
        attribute_name: str,
        factory,
        status_message: str,
    ) -> None:
        existing_window = getattr(self, attribute_name)
        if existing_window is not None:
            existing_window.show()
            existing_window.raise_()
            existing_window.activateWindow()
            self.statusBar().showMessage(status_message)
            return

        window = factory()
        window.estimator_widget.estimate_calculated.connect(
            self._on_solution_scattering_estimate_calculated
        )
        setattr(self, attribute_name, window)
        self._sync_solution_scattering_tool_targets()
        window.destroyed.connect(
            lambda *_args, name=attribute_name: setattr(self, name, None)
        )
        window.show()
        window.raise_()
        self._track_child_tool_window(window)
        self.statusBar().showMessage(status_message)

    def _open_solute_volume_fraction_tool(self) -> None:
        self._open_solution_scattering_tool_window(
            attribute_name="_solute_volume_fraction_tool_window",
            factory=SoluteVolumeFractionToolWindow,
            status_message="Opened volume fraction estimate",
        )

    def _open_number_density_tool(self) -> None:
        self._open_solution_scattering_tool_window(
            attribute_name="_number_density_tool_window",
            factory=NumberDensityEstimateToolWindow,
            status_message="Opened number density estimate",
        )

    def _open_attenuation_tool(self) -> None:
        self._open_solution_scattering_tool_window(
            attribute_name="_attenuation_tool_window",
            factory=AttenuationEstimateToolWindow,
            status_message="Opened attenuation estimate",
        )

    def _open_fluorescence_tool(self) -> None:
        self._open_solution_scattering_tool_window(
            attribute_name="_fluorescence_tool_window",
            factory=FluorescenceEstimateToolWindow,
            status_message="Opened fluorescence estimate",
        )

    def _show_placeholder_tool_message(self, tool_name: str) -> None:
        QMessageBox.information(
            self,
            "Coming soon",
            (
                f"{tool_name} is listed in the Tools menu as a placeholder "
                "for future SAXSShell integration."
            ),
        )
        self.statusBar().showMessage(f"{tool_name} is not available yet", 5000)

    def _effective_powerpoint_export_settings(
        self,
    ) -> PowerPointExportSettings:
        if self.current_settings is None:
            return PowerPointExportSettings()
        return PowerPointExportSettings.from_dict(
            self.current_settings.powerpoint_export_settings.to_dict()
        )

    def _open_main_ui_settings_dialog(self) -> None:
        dialog = MainUISettingsDialog(
            dream_settings=self.dream_tab.settings_payload(),
            powerpoint_settings=self._effective_powerpoint_export_settings(),
            powerpoint_enabled=self.current_settings is not None,
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        verbose, interval_seconds = dialog.dream_output_values()
        self._apply_dream_output_settings(
            verbose=verbose,
            interval_seconds=interval_seconds,
        )
        if self.current_settings is not None:
            self._apply_powerpoint_export_settings(
                dialog.powerpoint_settings_value()
            )

    def _open_dream_output_settings_dialog(self) -> None:
        self._open_main_ui_settings_dialog()

    def _apply_dream_output_settings(
        self,
        *,
        verbose: bool,
        interval_seconds: float,
    ) -> None:
        settings = self.dream_tab.settings_payload()
        settings.verbose = bool(verbose)
        settings.verbose_output_interval_seconds = max(
            float(interval_seconds),
            0.1,
        )
        self.dream_tab.set_settings(
            settings,
            preset_name=self.dream_tab.selected_settings_preset_name(),
        )
        self.dream_tab.append_log(
            "Updated DREAM output settings.\n"
            f"Verbose sampler output: {'on' if settings.verbose else 'off'}\n"
            "Runtime output interval: "
            f"{settings.verbose_output_interval_seconds:.1f} s\n"
            "Save DREAM settings if you want to persist this change."
        )
        self.statusBar().showMessage("DREAM output settings updated")

    def _apply_powerpoint_export_settings(
        self,
        settings: PowerPointExportSettings,
    ) -> None:
        if self.current_settings is None:
            return
        normalized = PowerPointExportSettings.from_dict(settings.to_dict())
        enabled_section_count = sum(
            int(value)
            for value in (
                normalized.include_prior_histograms,
                normalized.include_initial_traces,
                normalized.include_prefit_model,
                normalized.include_prefit_parameters,
                normalized.include_geometry_table,
                normalized.include_estimator_metrics,
                normalized.include_dream_settings,
                normalized.include_dream_prior_table,
                normalized.include_dream_output_model,
                normalized.include_posterior_comparisons,
                normalized.include_output_summary,
                normalized.include_directory_summary,
            )
        )
        self.current_settings.powerpoint_export_settings = normalized
        self.dream_tab.append_log(
            "Updated PowerPoint export settings.\n"
            f"Font: {normalized.font_family}\n"
            f"Component palette: {normalized.component_color_map}\n"
            "Prior palettes: "
            f"{normalized.prior_histogram_color_map} / "
            f"{normalized.solvent_sort_histogram_color_map}\n"
            f"Slides enabled: {enabled_section_count}/12\n"
            f"Manifest export: {'on' if normalized.generate_manifest else 'off'}\n"
            "Rendered figure assets: "
            f"{'kept' if normalized.export_figure_assets else 'temporary only'}\n"
            "Save the project if you want to persist this change."
        )
        self.statusBar().showMessage("PowerPoint export settings updated")

    def _show_version_information(self) -> None:
        QMessageBox.information(
            self,
            "Version Information",
            self._version_information_text(),
        )

    def _version_information_text(self) -> str:
        branch = self._git_output("rev-parse", "--abbrev-ref", "HEAD")
        commit = self._git_output("rev-parse", "--short", "HEAD")
        origin_url = self._normalize_repository_url(
            self._git_output("remote", "get-url", "origin")
            or GITHUB_REPOSITORY_URL
        )
        upstream_url = self._normalize_repository_url(
            self._git_output("remote", "get-url", "upstream") or ""
        )
        lines = [
            "SAXSShell Version Information",
            "",
            f"Package version: {__version__}",
            f"Git branch: {branch or 'unavailable'}",
            f"Git commit: {commit or 'unavailable'}",
            f"GitHub repository: {origin_url or GITHUB_REPOSITORY_URL}",
        ]
        if upstream_url:
            lines.append(f"Upstream repository: {upstream_url}")
        lines.extend(
            [
                "",
                "This information is read from the local Git checkout so it "
                "stays aligned with the GitHub-backed repository state for "
                "the branch you have open.",
                f"Developer contact: {CONTACT_EMAIL}",
            ]
        )
        return "\n".join(lines)

    def _open_github_repository(self) -> None:
        repository_url = self._normalize_repository_url(
            self._git_output("remote", "get-url", "origin")
            or GITHUB_REPOSITORY_URL
        )
        QDesktopServices.openUrl(QUrl(repository_url))
        self.statusBar().showMessage("Opened SAXSShell GitHub repository")

    def _show_contact_information(self) -> None:
        QMessageBox.information(
            self,
            "Developer Contact",
            (
                "For SAXSShell questions, template requests, or bug reports, "
                "contact the developer at:\n\n"
                f"{CONTACT_EMAIL}"
            ),
        )
        self.statusBar().showMessage("Opened developer contact information")

    @staticmethod
    def _normalize_repository_url(url: str) -> str:
        normalized = str(url or "").strip()
        if not normalized:
            return ""
        if normalized.startswith("git@github.com:"):
            normalized = normalized.replace(
                "git@github.com:",
                "https://github.com/",
                1,
            )
        if normalized.endswith(".git"):
            normalized = normalized[:-4]
        return normalized

    @staticmethod
    def _git_output(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", "-C", str(REPO_ROOT), *args],
                capture_output=True,
                text=True,
                check=True,
            )
        except Exception:
            return None
        output = result.stdout.strip()
        return output or None

    def _prompt_dream_settings_preset_name(
        self,
        *,
        suggested_name: str,
    ) -> str | None:
        preset_name, accepted = QInputDialog.getText(
            self,
            "Save DREAM Settings",
            (
                "Enter a name for this DREAM settings preset.\n"
                "Leave it blank to save only the active project settings."
            ),
            text=suggested_name.strip(),
        )
        if not accepted:
            return None
        return preset_name.strip() or None

    def show_prior_histogram_window(self) -> None:
        if self.current_settings is None:
            self._show_error(
                "Prior histogram unavailable",
                "Load or build a project first.",
            )
            return
        prior_json = project_artifact_paths(
            self.current_settings
        ).prior_weights_file
        if not prior_json.is_file():
            self._show_error(
                "Prior histogram unavailable",
                "Generate prior weights before opening prior histograms.",
            )
            return
        if (
            self.project_setup_tab.prior_mode().startswith("solvent_sort")
            and self.project_setup_tab.prior_secondary_element() is None
        ):
            self._show_error(
                "Prior histogram unavailable",
                "Select a secondary atom filter before opening a solvent-sort prior histogram.",
            )
            return
        window = PriorHistogramWindow(
            prior_json,
            mode=self.project_setup_tab.prior_mode(),
            secondary_element=self.project_setup_tab.prior_secondary_element(),
            cmap=self.project_setup_tab.prior_cmap(),
            structure_motif_colors=(
                self.project_setup_tab.prior_structure_motif_colors()
            ),
            parent=None,
        )
        self._prior_histogram_windows.append(window)
        window.destroyed.connect(self._on_prior_histogram_window_destroyed)
        window.show()
        window.raise_()
        window.activateWindow()

    def save_prior_plot_png(self) -> None:
        if self.current_settings is None:
            self._show_error(
                "Save prior histogram failed",
                "Load or build a project first.",
            )
            return
        prior_json = project_artifact_paths(
            self.current_settings
        ).prior_weights_file
        if not prior_json.is_file():
            self._show_error(
                "Save prior histogram failed",
                "Generate prior weights before saving a prior histogram image.",
            )
            return
        if (mode := self.project_setup_tab.prior_mode()).startswith(
            "solvent_sort"
        ) and (self.project_setup_tab.prior_secondary_element() is None):
            self._show_error(
                "Save prior histogram failed",
                "Select a secondary atom filter before saving a solvent-sort prior histogram image.",
            )
            return
        paths = build_project_paths(self.current_settings.project_dir)
        paths.exported_plots_dir.mkdir(parents=True, exist_ok=True)
        secondary = self.project_setup_tab.prior_secondary_element()
        suffix = f"_{secondary}" if secondary else ""
        output_path = paths.exported_plots_dir / (
            f"prior_histogram_{mode}{suffix}_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        )
        try:
            self.project_setup_tab.draw_prior_plot(prior_json)
            self.project_setup_tab.prior_figure.savefig(
                output_path,
                dpi=300,
                bbox_inches="tight",
            )
        except Exception as exc:
            self._show_error("Save prior histogram failed", str(exc))
            return
        self.project_setup_tab.append_summary(
            f"Saved prior histogram image to {output_path}"
        )
        self.statusBar().showMessage("Prior histogram image saved")

    @Slot(QObject)
    def _on_prior_histogram_window_destroyed(
        self,
        window: QObject | None,
    ) -> None:
        self._prior_histogram_windows = [
            open_window
            for open_window in self._prior_histogram_windows
            if open_window is not window
        ]

    def _schedule_dream_results_refresh(self, scope: int) -> None:
        if self._last_results_loader is None:
            return
        self._pending_dream_refresh_scope = max(
            int(self._pending_dream_refresh_scope),
            int(scope),
        )
        self._dream_refresh_timer.start()

    def _flush_pending_dream_refresh(self) -> None:
        scope = int(self._pending_dream_refresh_scope)
        self._pending_dream_refresh_scope = 0
        if scope <= 0:
            return
        self._refresh_loaded_dream_results(scope=scope)

    def _refresh_loaded_dream_results(
        self,
        *,
        scope: int | None = None,
    ) -> None:
        if self._last_results_loader is None:
            return
        refresh_scope = (
            int(scope) if scope is not None else self.DREAM_REFRESH_FULL
        )
        try:
            settings = self.dream_tab.settings_payload()
            loader = self._last_results_loader
            filter_kwargs = self._dream_filter_kwargs(settings)
            summary = loader.get_summary(
                bestfit_method=settings.bestfit_method,
                **filter_kwargs,
            )
            self._last_dream_summary = summary
            self._applied_dream_analysis_settings = self._copy_dream_settings(
                settings
            )
            self.dream_tab.set_summary_text(
                self._format_dream_summary(summary, settings=settings)
            )
            self._refresh_dream_filter_status()
            self.dream_tab.set_filter_dirty(False)
            if refresh_scope <= self.DREAM_REFRESH_STYLE:
                if self.dream_tab.current_violin_plot_data() is None:
                    violin_plot = loader.build_violin_data(
                        mode=self._effective_dream_violin_mode(settings),
                        sample_source=settings.violin_sample_source,
                        weight_order=settings.violin_weight_order,
                        **filter_kwargs,
                    )
                    self.dream_tab.plot_violin_plot(summary, violin_plot)
                else:
                    self.dream_tab.redraw_current_violin_plot()
                return
            if refresh_scope == self.DREAM_REFRESH_SUMMARY:
                violin_plot = self.dream_tab.current_violin_plot_data()
                if violin_plot is None:
                    violin_plot = loader.build_violin_data(
                        mode=self._effective_dream_violin_mode(settings),
                        sample_source=settings.violin_sample_source,
                        weight_order=settings.violin_weight_order,
                        **filter_kwargs,
                    )
                self.dream_tab.plot_violin_plot(summary, violin_plot)
                return
            if refresh_scope == self.DREAM_REFRESH_VIOLIN:
                violin_plot = loader.build_violin_data(
                    mode=self._effective_dream_violin_mode(settings),
                    sample_source=settings.violin_sample_source,
                    weight_order=settings.violin_weight_order,
                    **filter_kwargs,
                )
                self.dream_tab.plot_violin_plot(summary, violin_plot)
                return
            model_plot = loader.build_model_fit_data(
                bestfit_method=settings.bestfit_method,
                **filter_kwargs,
            )
            violin_plot = loader.build_violin_data(
                mode=self._effective_dream_violin_mode(settings),
                sample_source=settings.violin_sample_source,
                weight_order=settings.violin_weight_order,
                **filter_kwargs,
            )
            self.dream_tab.plot_model_fit(model_plot)
            self.dream_tab.plot_violin_plot(summary, violin_plot)
        except Exception as exc:
            self._show_error("Render DREAM results failed", str(exc))

    @staticmethod
    def _fit_quality_metrics_from_curves(
        experimental: np.ndarray,
        model: np.ndarray,
    ) -> FitQualityMetrics:
        experimental_values = np.asarray(experimental, dtype=float)
        model_values = np.asarray(model, dtype=float)
        residuals = np.asarray(
            model_values - experimental_values,
            dtype=float,
        )
        rmse = float(np.sqrt(np.mean(residuals**2)))
        mean_abs_residual = float(np.mean(np.abs(residuals)))
        experimental_mean = float(np.mean(experimental_values))
        total_sum_squares = float(
            np.sum((experimental_values - experimental_mean) ** 2)
        )
        residual_sum_squares = float(np.sum(residuals**2))
        r_squared = (
            float(1.0 - (residual_sum_squares / total_sum_squares))
            if total_sum_squares > 0.0
            else 1.0
        )
        return FitQualityMetrics(
            rmse=rmse,
            mean_abs_residual=mean_abs_residual,
            r_squared=r_squared,
        )

    @staticmethod
    def _is_weight_parameter_name(name: str) -> bool:
        return name.startswith("w") and name[1:].isdigit()

    @staticmethod
    def _is_prefit_better_than_dream(
        prefit_metrics: FitQualityMetrics,
        dream_metrics: FitQualityMetrics,
    ) -> bool:
        if prefit_metrics.rmse < dream_metrics.rmse - 1e-9:
            return True
        if np.isclose(prefit_metrics.rmse, dream_metrics.rmse):
            return prefit_metrics.r_squared > dream_metrics.r_squared + 1e-9
        return False

    def _build_dream_constraint_comparison(
        self,
        *,
        model_plot,
        entries: list,
    ) -> DreamFitConstraintComparison | None:
        if self.prefit_workflow is None:
            return None
        prefit_evaluation = self.prefit_workflow.evaluate()
        prefit_metrics = self._fit_quality_metrics_from_curves(
            prefit_evaluation.experimental_intensities,
            prefit_evaluation.model_intensities,
        )
        dream_metrics = FitQualityMetrics(
            rmse=float(model_plot.rmse),
            mean_abs_residual=float(model_plot.mean_abs_residual),
            r_squared=float(model_plot.r_squared),
        )
        fixed_parameters = tuple(
            str(entry.param) for entry in entries if not bool(entry.vary)
        )
        fixed_non_weight_parameters = tuple(
            name
            for name in fixed_parameters
            if not self._is_weight_parameter_name(name)
        )
        fixed_weight_parameters = tuple(
            name
            for name in fixed_parameters
            if self._is_weight_parameter_name(name)
        )
        return DreamFitConstraintComparison(
            prefit_metrics=prefit_metrics,
            dream_metrics=dream_metrics,
            fixed_parameters=fixed_parameters,
            fixed_non_weight_parameters=fixed_non_weight_parameters,
            fixed_weight_parameters=fixed_weight_parameters,
        )

    def _format_dream_fit_comparison_log(
        self,
        comparison: DreamFitConstraintComparison,
    ) -> str:
        lines = [
            "DREAM vs Prefit fit-quality comparison:",
            f"  Prefit RMSE: {comparison.prefit_metrics.rmse:.6g}",
            (
                "  Prefit Mean |res|: "
                f"{comparison.prefit_metrics.mean_abs_residual:.6g}"
            ),
            f"  Prefit R^2: {comparison.prefit_metrics.r_squared:.6g}",
            f"  DREAM RMSE: {comparison.dream_metrics.rmse:.6g}",
            (
                "  DREAM Mean |res|: "
                f"{comparison.dream_metrics.mean_abs_residual:.6g}"
            ),
            f"  DREAM R^2: {comparison.dream_metrics.r_squared:.6g}",
        ]
        if comparison.fixed_parameters:
            lines.append(
                "  DREAM vary=off parameters: "
                + ", ".join(comparison.fixed_parameters)
            )
        else:
            lines.append("  DREAM vary=off parameters: none")
        return "\n".join(lines)

    def _prompt_dream_constraint_update(
        self,
        comparison: DreamFitConstraintComparison,
    ) -> bool:
        dialog = QMessageBox(self)
        dialog.setIcon(QMessageBox.Icon.Warning)
        dialog.setWindowTitle("DREAM fit appears overconstrained")
        dialog.setText(
            "The saved prefit currently fits the experimental SAXS data "
            "better than the current DREAM best fit."
        )
        fixed_non_weight_text = (
            ", ".join(comparison.fixed_non_weight_parameters)
            if comparison.fixed_non_weight_parameters
            else "none"
        )
        fixed_weight_text = (
            ", ".join(comparison.fixed_weight_parameters)
            if comparison.fixed_weight_parameters
            else "none"
        )
        dialog.setInformativeText(
            "This usually means the DREAM prior distributions are "
            "overconstrained.\n\n"
            f"Prefit RMSE: {comparison.prefit_metrics.rmse:.6g}\n"
            f"DREAM RMSE: {comparison.dream_metrics.rmse:.6g}\n"
            f"Prefit R^2: {comparison.prefit_metrics.r_squared:.6g}\n"
            f"DREAM R^2: {comparison.dream_metrics.r_squared:.6g}\n\n"
            f"Fixed non-weight parameters: {fixed_non_weight_text}\n"
            f"Fixed weight parameters: {fixed_weight_text}\n\n"
            "Recommended next step: retry DREAM without constraining the "
            "non-weight parameters listed above, and allow all model "
            "weight parameters w<##> to vary.\n\n"
            "Do you want to update the current DREAM parameter map now and "
            "set vary=yes for all parameters?"
        )
        update_button = dialog.addButton(
            "Set All Vary = Yes",
            QMessageBox.ButtonRole.AcceptRole,
        )
        dialog.addButton(
            "Keep Current Settings",
            QMessageBox.ButtonRole.RejectRole,
        )
        dialog.setDefaultButton(update_button)
        dialog.exec()
        return dialog.clickedButton() is update_button

    def _maybe_warn_about_dream_overconstraints(
        self,
        settings: DreamRunSettings,
    ) -> None:
        if self._last_results_loader is None:
            return
        workflow = self._load_dream_workflow()
        try:
            entries = workflow.load_parameter_map(persist_if_missing=False)
        except Exception:
            return
        if not entries:
            return
        filter_kwargs = self._dream_filter_kwargs(settings)
        model_plot = self._last_results_loader.build_model_fit_data(
            bestfit_method=settings.bestfit_method,
            **filter_kwargs,
        )
        comparison = self._build_dream_constraint_comparison(
            model_plot=model_plot,
            entries=entries,
        )
        if comparison is None:
            return
        self.dream_tab.append_log(
            self._format_dream_fit_comparison_log(comparison)
        )
        if (
            not comparison.fixed_parameters
            or not self._is_prefit_better_than_dream(
                comparison.prefit_metrics,
                comparison.dream_metrics,
            )
        ):
            return
        signature = (
            str(self._last_results_loader.run_dir),
            str(settings.bestfit_method),
            str(settings.posterior_filter_mode),
            float(settings.posterior_top_percent),
            int(settings.posterior_top_n),
            tuple(comparison.fixed_parameters),
            round(comparison.prefit_metrics.rmse, 12),
            round(comparison.dream_metrics.rmse, 12),
        )
        if signature == self._last_dream_constraint_warning_signature:
            return
        self._last_dream_constraint_warning_signature = signature
        if not self._prompt_dream_constraint_update(comparison):
            return
        updated_entries = [
            entry.__class__(
                structure=entry.structure,
                motif=entry.motif,
                param_type=entry.param_type,
                param=entry.param,
                value=entry.value,
                vary=True,
                distribution=entry.distribution,
                dist_params=dict(entry.dist_params),
            )
            for entry in entries
        ]
        self._save_distribution_entries(updated_entries)
        self.dream_tab.append_log(
            "Updated DREAM parameter map from the overconstraint warning.\n"
            "All DREAM parameters now have vary=yes.\n"
            "Rewrite the Runtime Bundle before rerunning DREAM."
        )
        self.statusBar().showMessage(
            "Updated DREAM parameter map; rewrite the runtime bundle"
        )

    def _format_saved_dream_run_preview(
        self,
        *,
        run_dir: Path,
        display_label: str | None,
        settings: DreamRunSettings,
        parameter_map_entries: list[DreamParameterEntry],
    ) -> str:
        varying_count = sum(1 for entry in parameter_map_entries if entry.vary)
        fixed_count = len(parameter_map_entries) - varying_count
        varying_names = [
            entry.param.strip()
            for entry in parameter_map_entries
            if entry.vary and entry.param.strip()
        ]
        lines = [
            f"Saved run: {display_label or run_dir.name}",
            f"Run directory: {run_dir}",
            f"Run label: {settings.run_label.strip() or 'dream'}",
            f"Model name: {settings.model_name or 'Unknown'}",
            "",
            "Saved DREAM runtime settings:",
            (
                "Search/filter preset: "
                f"{self._format_dream_setting_label(settings.search_filter_preset)}"
            ),
            f"Chains: {settings.nchains}",
            f"Iterations: {settings.niterations}",
            f"Burn-in: {settings.burnin_percent}%",
            f"History thinning: every {settings.history_thin} samples",
            f"Seed chains: {settings.nseedchains}",
            f"Crossover burn-in: {settings.crossover_burnin}",
            f"Parallel chains: {'on' if settings.parallel else 'off'}",
            f"Adaptive crossover: {'on' if settings.adapt_crossover else 'off'}",
            f"Restart from history: {'on' if settings.restart else 'off'}",
            f"History file: {settings.history_file or 'None'}",
            f"Verbose sampler output: {'on' if settings.verbose else 'off'}",
            (
                "Verbose output interval (s): "
                f"{settings.verbose_output_interval_seconds:g}"
            ),
            f"Lambda: {settings.lamb:.6g}",
            f"Zeta: {settings.zeta:.6g}",
            f"Snooker probability: {settings.snooker:.6g}",
            f"P(gamma = 1): {settings.p_gamma_unity:.6g}",
            "",
            "Saved analysis defaults:",
            (
                "Best-fit method: "
                f"{self._format_dream_setting_label(settings.bestfit_method)}"
            ),
            f"Posterior filter: {self._describe_posterior_filter(settings)}",
            (
                "Auto-select best filter after run: "
                f"{'on' if settings.auto_select_best_posterior_filter else 'off'}"
            ),
            (
                "Filter defaults: "
                f"Top % = {settings.posterior_top_percent:g}, "
                f"Top N = {settings.posterior_top_n}"
            ),
            (
                "Credible interval (%): "
                f"{settings.credible_interval_low:g} - "
                f"{settings.credible_interval_high:g}"
            ),
            (
                "Violin data mode: "
                f"{self._format_dream_setting_label(settings.violin_parameter_mode)}"
            ),
            (
                "Violin sample source: "
                f"{self._format_dream_setting_label(settings.violin_sample_source)}"
            ),
            (
                "Weight order: "
                f"{self._format_dream_setting_label(settings.violin_weight_order)}"
            ),
            (
                "Y-axis scale: "
                f"{self._format_dream_setting_label(settings.violin_value_scale_mode)}"
            ),
        ]
        stoichiometry_lines = self._dream_stoichiometry_summary_lines(settings)
        if stoichiometry_lines:
            lines.append("")
            lines.extend(stoichiometry_lines)
        lines.extend(
            [
                "",
                "Saved prior parameter map:",
                (
                    "Entries: "
                    f"{len(parameter_map_entries)} total, {varying_count} varying, "
                    f"{fixed_count} fixed"
                ),
                (
                    "Varying parameters: "
                    f"{self._format_saved_dream_parameter_list(varying_names)}"
                ),
            ]
        )
        return "\n".join(lines)

    @staticmethod
    def _format_saved_dream_parameter_list(names: list[str]) -> str:
        if not names:
            return "none"
        preview = ", ".join(names[:12])
        if len(names) > 12:
            preview += ", ..."
        return preview

    @staticmethod
    def _format_dream_setting_label(value: str | None) -> str:
        normalized = str(value or "").strip()
        if not normalized:
            return "None"
        labels = {
            "map": "MAP",
            "chain_mean": "Chain Mean MAP",
            "median": "Median",
            "less_aggressive": "Less Aggressive",
            "medium": "Medium",
            "more_aggressive": "More Aggressive",
            "custom": "Custom",
            "all_post_burnin": "All Post-burnin Samples",
            "top_percent_logp": "Top % by Log-posterior",
            "top_n_logp": "Top N by Log-posterior",
            "varying_parameters": "Varying Parameters",
            "all_parameters": "All Parameters",
            "weights_only": "Weights Only",
            "fit_parameters": "Fit Parameters",
            "effective_radii_only": "Effective Radii Only",
            "additional_parameters_only": "Additional Parameters Only",
            "filtered_posterior": "Filtered Posterior",
            "map_chain_only": "MAP Chain Only",
            "weight_index": "Weight Index",
            "structure_order": "Structure Order",
            "parameter_value": "Parameter Value",
            "weights_unit_interval": "Weights 0-1 Only",
            "normalized_all": "Normalized 0-1 (All)",
        }
        return labels.get(normalized, normalized.replace("_", " ").title())

    def _format_dream_summary(
        self,
        summary,
        *,
        settings: DreamRunSettings,
    ) -> str:
        lines = [
            f"Run directory: {summary.run_dir}",
            f"Best-fit method: {summary.bestfit_method}",
            (
                "Posterior filter: "
                f"{self._describe_posterior_filter(settings)}"
            ),
            (
                "Auto-select best filter after run: "
                f"{'on' if settings.auto_select_best_posterior_filter else 'off'}"
            ),
            (
                "Filter defaults: "
                f"Top % = {settings.posterior_top_percent:g}, "
                f"Top N = {settings.posterior_top_n}"
            ),
            (
                "Posterior candidates before ranking: "
                f"{summary.posterior_candidate_sample_count}"
            ),
            f"Posterior samples kept: {summary.posterior_sample_count}",
            (
                "Credible interval (%): "
                f"{summary.credible_interval_low:g} - "
                f"{summary.credible_interval_high:g}"
            ),
            f"Violin data mode: {settings.violin_parameter_mode}",
            f"Violin sample source: {settings.violin_sample_source}",
            f"Weight order: {settings.violin_weight_order}",
            f"Y-axis scale: {settings.violin_value_scale_mode}",
            f"Violin palette: {settings.violin_palette}",
            f"Point color: {settings.violin_point_color}",
            (
                "MAP location: "
                f"chain {summary.map_chain + 1}, step {summary.map_step + 1}"
            ),
            "",
        ]
        lines.extend(
            self._dream_stoichiometry_summary_lines(
                settings,
                evaluation=summary.stoichiometry_evaluation,
            )
        )
        if lines[-1] != "":
            lines.append("")
        lines.extend(
            [
                "Posterior summary:",
            ]
        )
        for index, name in enumerate(summary.full_parameter_names):
            lines.append(
                f"{name}: selected={summary.bestfit_params[index]:.6g}, "
                f"MAP={summary.map_params[index]:.6g}, "
                f"chain_mean={summary.chain_mean_params[index]:.6g}, "
                f"median={summary.median_params[index]:.6g}, "
                f"p{summary.credible_interval_low:g}="
                f"{summary.interval_low_values[index]:.6g}, "
                f"p{summary.credible_interval_high:g}="
                f"{summary.interval_high_values[index]:.6g}"
            )
        return "\n".join(lines)

    def _format_dream_console_intro(self) -> str:
        if self.prefit_workflow is None:
            return "DREAM workflow is not loaded."
        settings = self.dream_tab.settings_payload()
        stoichiometry_lines = self._dream_stoichiometry_summary_lines(settings)
        lines = [
            "DREAM workflow loaded.",
            f"Template: {self.prefit_workflow.template_spec.name}",
            "Review the priors with Edit Priors and click Save Parameter "
            "Map before running DREAM.",
            f"Best-fit method: {settings.bestfit_method}",
            f"Posterior filter: {self._describe_posterior_filter(settings)}",
            "Auto-select best filter after run: "
            f"{'on' if settings.auto_select_best_posterior_filter else 'off'}",
            f"Filter defaults: Top % = {settings.posterior_top_percent:g}, "
            f"Top N = {settings.posterior_top_n}",
            f"Violin data mode: {settings.violin_parameter_mode}",
            f"Violin sample source: {settings.violin_sample_source}",
            f"Weight order: {settings.violin_weight_order}",
            f"Y-axis scale: {settings.violin_value_scale_mode}",
            f"Violin palette: {settings.violin_palette}",
            f"Point color: {settings.violin_point_color}",
        ]
        lines.extend(stoichiometry_lines)
        lines.append(
            "Recommendation: all refinable parameters are usually allowed "
            "to vary during DREAM refinement."
        )
        return "\n".join(lines)

    @staticmethod
    def _dream_filter_kwargs(
        settings: DreamRunSettings,
    ) -> dict[str, object]:
        return {
            "posterior_filter_mode": settings.posterior_filter_mode,
            "posterior_top_percent": settings.posterior_top_percent,
            "posterior_top_n": settings.posterior_top_n,
            "credible_interval_low": settings.credible_interval_low,
            "credible_interval_high": settings.credible_interval_high,
            "stoichiometry_target_elements_text": (
                settings.stoichiometry_target_elements_text
            ),
            "stoichiometry_target_ratio_text": (
                settings.stoichiometry_target_ratio_text
            ),
            "stoichiometry_filter_enabled": (
                settings.stoichiometry_filter_enabled
            ),
            "stoichiometry_tolerance_percent": (
                settings.stoichiometry_tolerance_percent
            ),
        }

    def _dream_stoichiometry_summary_lines(
        self,
        settings: DreamRunSettings,
        *,
        evaluation=None,
        candidate_sample_count: int | None = None,
        posterior_sample_count: int | None = None,
    ) -> list[str]:
        try:
            target = build_stoichiometry_target(
                settings.stoichiometry_target_elements_text,
                settings.stoichiometry_target_ratio_text,
            )
        except Exception as exc:
            if settings.stoichiometry_filter_enabled:
                return [f"Stoichiometry target: invalid ({exc})"]
            return []
        if target is None:
            return []
        lines = [f"Stoichiometry target: {stoichiometry_target_text(target)}"]
        lines.append(
            "Stoichiometry filter: "
            + (
                f"on ({settings.stoichiometry_tolerance_percent:g}% tolerance)"
                if settings.stoichiometry_filter_enabled
                else "off"
            )
        )
        if candidate_sample_count is not None:
            lines.append(
                f"Samples considered before posterior ranking: {candidate_sample_count}"
            )
        if posterior_sample_count is not None:
            lines.append(
                f"Posterior samples kept after filtering: {posterior_sample_count}"
            )
        if evaluation is not None:
            lines.append(
                "Selected stoichiometry: "
                + stoichiometry_ratio_text(target, evaluation.observed_ratio)
            )
            lines.append(stoichiometry_deviation_text(evaluation))
        return lines

    def _refresh_dream_filter_status(
        self, *, dirty: bool | None = None
    ) -> None:
        if (
            self._last_results_loader is None
            or self._last_dream_summary is None
        ):
            self.dream_tab.set_filter_status_text(
                "No DREAM dataset is loaded yet. Load a run, then apply posterior "
                "filter settings from this panel."
            )
            if dirty is not None:
                self.dream_tab.set_filter_dirty(bool(dirty))
            return
        settings = self._applied_dream_analysis_settings
        if settings is None:
            settings = self.dream_tab.settings_payload()
        summary = self._last_dream_summary
        lines = [
            f"Active run: {summary.run_dir}",
            f"Best-fit method: {settings.bestfit_method}",
            f"Posterior filter: {self._describe_posterior_filter(settings)}",
            (
                "Credible interval (%): "
                f"{summary.credible_interval_low:g} - "
                f"{summary.credible_interval_high:g}"
            ),
            f"Posterior samples kept: {summary.posterior_sample_count}",
        ]
        lines.extend(
            self._dream_stoichiometry_summary_lines(
                settings,
                evaluation=summary.stoichiometry_evaluation,
                candidate_sample_count=summary.posterior_candidate_sample_count,
                posterior_sample_count=summary.posterior_sample_count,
            )
        )
        pending = (
            self.dream_tab.filter_settings_dirty()
            if dirty is None
            else bool(dirty)
        )
        if pending:
            lines.extend(
                [
                    "",
                    "Pending changes are not applied yet. Press Apply Filter "
                    "to redraw the active DREAM dataset with the new settings.",
                ]
            )
        self.dream_tab.set_filter_status_text("\n".join(lines))

    @staticmethod
    def _describe_posterior_filter(settings: DreamRunSettings) -> str:
        if settings.posterior_filter_mode == "top_percent_logp":
            return (
                f"top_percent_logp "
                f"(top {settings.posterior_top_percent:g}% by log-posterior)"
            )
        if settings.posterior_filter_mode == "top_n_logp":
            return (
                f"top_n_logp "
                f"(top {settings.posterior_top_n} samples by log-posterior)"
            )
        return "all_post_burnin"

    @staticmethod
    def _merge_dream_analysis_settings(
        run_settings: DreamRunSettings,
        current_settings: DreamRunSettings,
    ) -> DreamRunSettings:
        merged = DreamRunSettings.from_dict(run_settings.to_dict())
        merged.bestfit_method = current_settings.bestfit_method
        merged.posterior_filter_mode = current_settings.posterior_filter_mode
        merged.posterior_top_percent = current_settings.posterior_top_percent
        merged.posterior_top_n = current_settings.posterior_top_n
        merged.credible_interval_low = current_settings.credible_interval_low
        merged.credible_interval_high = current_settings.credible_interval_high
        merged.violin_parameter_mode = current_settings.violin_parameter_mode
        merged.violin_sample_source = current_settings.violin_sample_source
        merged.violin_weight_order = current_settings.violin_weight_order
        merged.violin_value_scale_mode = (
            current_settings.violin_value_scale_mode
        )
        merged.stoichiometry_target_elements_text = (
            current_settings.stoichiometry_target_elements_text
        )
        merged.stoichiometry_target_ratio_text = (
            current_settings.stoichiometry_target_ratio_text
        )
        merged.stoichiometry_filter_enabled = (
            current_settings.stoichiometry_filter_enabled
        )
        merged.stoichiometry_tolerance_percent = (
            current_settings.stoichiometry_tolerance_percent
        )
        merged.violin_palette = current_settings.violin_palette
        merged.violin_custom_color = current_settings.violin_custom_color
        merged.violin_point_color = current_settings.violin_point_color
        merged.violin_interval_color = current_settings.violin_interval_color
        merged.violin_median_color = current_settings.violin_median_color
        merged.violin_outline_color = current_settings.violin_outline_color
        merged.violin_outline_width = current_settings.violin_outline_width
        return merged

    def _append_dream_vary_recommendation(self, entries: list) -> None:
        fixed = [entry.param for entry in entries if not entry.vary]
        if not fixed:
            return
        names = ", ".join(fixed[:10])
        if len(fixed) > 10:
            names += ", ..."
        self.dream_tab.append_log(
            "Recommendation: allow all refinable parameters to vary during "
            "the DREAM refinement. The current parameter map has vary=off "
            f"for: {names}"
        )

    def _invalidate_written_dream_bundle(self) -> None:
        self._last_written_dream_bundle = None

    def _invalidate_dream_workflow_cache(self) -> None:
        self.dream_workflow = None
        self._invalidate_written_dream_bundle()
        self._last_dream_constraint_warning_signature = None

    def _format_prefit_console_intro(
        self,
        *,
        evaluation: PrefitEvaluation | None = None,
        preview_block_reason: str | None = None,
    ) -> str:
        if self.prefit_workflow is None:
            return "Prefit workflow is not loaded."
        settings = self.prefit_workflow.settings
        if evaluation is None and preview_block_reason is None:
            try:
                evaluation = self.prefit_workflow.evaluate()
            except Exception as exc:
                preview_block_reason = str(exc).strip() or None
        if evaluation is not None:
            q_values = np.asarray(evaluation.q_values, dtype=float)
        else:
            try:
                q_values = self.prefit_workflow._component_q_values()
            except Exception:
                if self.prefit_workflow.experimental_data is not None:
                    q_values = np.asarray(
                        self.prefit_workflow.experimental_data.q_values,
                        dtype=float,
                    )
                else:
                    q_values = np.asarray([], dtype=float)
        run_config = self.prefit_tab.run_config()
        if q_values.size == 0:
            grid_text = "The active q-grid is not available yet."
        elif settings.model_only_mode:
            grid_text = (
                "Model Only Mode is active. Using a forward-model q-grid with "
                f"{len(q_values)} points from {float(q_values.min()):.6g} to "
                f"{float(q_values.max()):.6g}."
            )
        elif settings.use_experimental_grid:
            grid_text = (
                "Using the experimental q-grid cropped to the nearest "
                "available q-points inside the requested range "
                f"({len(q_values)} points from {float(q_values.min()):.6g} "
                f"to {float(q_values.max()):.6g})."
            )
        else:
            grid_text = (
                "Resampling the experimental data onto "
                f"{len(q_values)} evenly spaced q-points between "
                f"{float(q_values.min()):.6g} and {float(q_values.max()):.6g}."
            )
        excluded = ", ".join(settings.exclude_elements) or "None"
        return (
            "Prefit workflow loaded.\n"
            f"Template: {self.prefit_workflow.template_spec.name}\n"
            f"{grid_text}\n"
            f"Excluded elements: {excluded}\n"
            "Project presets: template-default reset is available"
            + (
                "; Best Prefit preset will be applied on project reload."
                if self.prefit_workflow.has_best_prefit_entries()
                else "; no Best Prefit preset is saved yet."
            )
            + "\n"
            + (
                "Prefit preview is currently waiting on template metadata.\n"
                f"{preview_block_reason}\n"
                if preview_block_reason is not None
                else ""
            )
            + (
                "Prefit fitting is disabled while Model Only Mode is active.\n"
                if settings.model_only_mode
                else ""
            )
            + "Recommended order: refine scale first, then scale + offset. "
            "Component weights w<##> are not recommended for prefit refinement.\n"
            f"Default minimizer: {run_config.method}\n"
            f"Default max nfev: {run_config.max_nfev}\n"
            "Autosave fit results: "
            + ("enabled" if settings.autosave_prefits else "disabled")
        )

    def _format_prefit_summary(
        self,
        evaluation,
        *,
        fit_result=None,
        report_path: Path | None = None,
    ) -> str:
        if self.prefit_workflow is None:
            return "Prefit summary is not available."
        q_values = np.asarray(evaluation.q_values, dtype=float)
        lines = [
            "Prefit summary:",
            f"Template: {self.prefit_workflow.template_spec.name}",
            f"Points: {len(q_values)}",
            (
                f"q-range: {float(q_values.min()):.6g} to "
                f"{float(q_values.max()):.6g}"
            ),
            f"Configured minimizer: {self.prefit_tab.run_config().method}",
            f"Configured max nfev: {self.prefit_tab.run_config().max_nfev}",
            (
                "Autosave fits: "
                + (
                    "enabled"
                    if self.prefit_workflow.settings.autosave_prefits
                    else "disabled"
                )
            ),
        ]
        if (
            evaluation.residuals is None
            or evaluation.experimental_intensities is None
        ):
            lines.extend(
                [
                    "Mode: Model Only",
                    "Experimental fit metrics: unavailable",
                ]
            )
        else:
            residuals = np.asarray(evaluation.residuals, dtype=float)
            rms_residual = float(np.sqrt(np.mean(residuals**2)))
            mean_abs_residual = float(np.mean(np.abs(residuals)))
            lines[4:4] = [
                f"Residual RMS: {rms_residual:.6g}",
                f"Mean |residual|: {mean_abs_residual:.6g}",
            ]
        if fit_result is not None:
            lines.extend(
                [
                    f"Method: {fit_result.method}",
                    f"Optimization strategy: {fit_result.optimization_strategy}",
                    f"Grid evaluations: {fit_result.grid_evaluations}",
                    f"Function evals: {fit_result.nfev}",
                    f"Chi^2: {fit_result.chi_square:.6g}",
                    f"Reduced chi^2: {fit_result.reduced_chi_square:.6g}",
                    f"R^2: {fit_result.r_squared:.6g}",
                ]
            )
        if report_path is not None:
            lines.append(f"Saved report: {report_path}")
        return "\n".join(lines)

    def _confirm_large_prefit_parameter_count(
        self,
        varying_parameter_names: list[str],
    ) -> tuple[bool, bool]:
        dialog = QMessageBox(self)
        dialog.setIcon(QMessageBox.Icon.Warning)
        dialog.setWindowTitle("Too many prefit parameters selected")
        dialog.setText(
            "More than 3 independent prefit parameters are selected to vary."
        )
        dialog.setInformativeText(
            "The coarse-to-fine grid prefit is only used when 1-3 "
            "parameters vary at once.\n\n"
            f"Selected varying parameters ({len(varying_parameter_names)}): "
            + ", ".join(varying_parameter_names)
            + "\n\n"
            "Continue anyway to use the current lmfit-only prefit, or go "
            "back and edit the parameter table."
        )
        continue_button = dialog.addButton(
            "Continue Anyway",
            QMessageBox.ButtonRole.AcceptRole,
        )
        dialog.addButton(
            "Go Back and Edit Parameters",
            QMessageBox.ButtonRole.RejectRole,
        )
        suppress_checkbox = QCheckBox(
            "Don't show this warning again during this session"
        )
        dialog.setCheckBox(suppress_checkbox)
        dialog.setDefaultButton(continue_button)
        dialog.exec()
        return (
            dialog.clickedButton() is continue_button,
            suppress_checkbox.isChecked(),
        )

    def _confirm_prefit_template_change(
        self,
        current_name: str,
        new_name: str,
    ) -> tuple[bool, bool]:
        if not self._warn_on_prefit_template_change:
            return True, False
        dialog = QMessageBox(self)
        dialog.setIcon(QMessageBox.Icon.Warning)
        dialog.setWindowTitle("Change active template?")
        dialog.setText(
            "Changing the active template creates or loads a separate "
            "computed distribution for that template."
        )
        dialog.setInformativeText(
            f"Current template: {current_name}\n"
            f"New template: {new_name}\n\n"
            "The new distribution keeps the same files and q-range, but "
            "template-specific metadata such as cluster geometry may need "
            "to be recomputed."
        )
        dialog.setStandardButtons(
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        dialog.setDefaultButton(QMessageBox.StandardButton.No)
        suppress_checkbox = QCheckBox(
            "Don't warn again for template changes during this session"
        )
        dialog.setCheckBox(suppress_checkbox)
        response = dialog.exec()
        return (
            response == QMessageBox.StandardButton.Yes,
            suppress_checkbox.isChecked(),
        )

    def apply_recommended_scale_settings(self) -> None:
        if self.prefit_workflow is None:
            self._show_error(
                "Prefit unavailable",
                "Build a project and load its SAXS components first.",
            )
            return
        if not self.prefit_workflow.can_run_prefit():
            self._show_error(
                "Scale recommendation failed",
                "Disable Model Only Mode and load experimental SAXS data before applying autoscale.",
            )
            return
        try:
            self._flush_prefit_sequence_pending_manual_updates(
                trigger="apply_autoscale",
            )
            entries = self.prefit_tab.parameter_entries()
            recommendation = self.prefit_workflow.recommend_scale_settings(
                entries
            )
            self.prefit_tab.set_parameter_row(
                "scale",
                value=recommendation.recommended_scale,
                minimum=recommendation.recommended_minimum,
                maximum=recommendation.recommended_maximum,
                vary=True,
            )
            if recommendation.recommended_offset is not None:
                offset_kwargs: dict[str, object] = {
                    "value": recommendation.recommended_offset,
                }
                if recommendation.recommended_offset_minimum is not None:
                    offset_kwargs["minimum"] = (
                        recommendation.recommended_offset_minimum
                    )
                if recommendation.recommended_offset_maximum is not None:
                    offset_kwargs["maximum"] = (
                        recommendation.recommended_offset_maximum
                    )
                self.prefit_tab.set_parameter_row(
                    "offset",
                    **offset_kwargs,
                )
            self.prefit_workflow.parameter_entries = (
                self.prefit_tab.parameter_entries()
            )
            offset_text = (
                ""
                if recommendation.recommended_offset is None
                else (
                    f"Current offset: "
                    f"{(recommendation.current_offset or 0.0):.6g}\n"
                    f"Recommended offset: "
                    f"{recommendation.recommended_offset:.6g}\n"
                )
            )
            message = (
                "Applied autoscale settings.\n"
                + f"Current scale: {recommendation.current_scale:.6g}\n"
                + f"Recommended scale: {recommendation.recommended_scale:.6g}\n"
                + f"Scale min: {recommendation.recommended_minimum:.6g}\n"
                + f"Scale max: {recommendation.recommended_maximum:.6g}\n"
                + offset_text
                + f"Adjustment factor: {recommendation.adjustment_factor:.6g}\n"
                + f"Points used: {recommendation.points_used}"
            )
            self.prefit_tab.append_log(message)
            current_entries = self.prefit_tab.parameter_entries()
            self._set_prefit_sequence_baseline(current_entries)
            self._append_prefit_sequence_event(
                "autoscale_applied",
                "Applied Prefit autoscale settings.",
                details={
                    "current_scale": float(recommendation.current_scale),
                    "recommended_scale": float(
                        recommendation.recommended_scale
                    ),
                    "recommended_minimum": float(
                        recommendation.recommended_minimum
                    ),
                    "recommended_maximum": float(
                        recommendation.recommended_maximum
                    ),
                    "adjustment_factor": float(
                        recommendation.adjustment_factor
                    ),
                    "points_used": int(recommendation.points_used),
                    "current_offset": (
                        None
                        if recommendation.current_offset is None
                        else float(recommendation.current_offset)
                    ),
                    "recommended_offset": (
                        None
                        if recommendation.recommended_offset is None
                        else float(recommendation.recommended_offset)
                    ),
                    "recommended_offset_minimum": (
                        None
                        if recommendation.recommended_offset_minimum is None
                        else float(recommendation.recommended_offset_minimum)
                    ),
                    "recommended_offset_maximum": (
                        None
                        if recommendation.recommended_offset_maximum is None
                        else float(recommendation.recommended_offset_maximum)
                    ),
                },
                parameter_entries=current_entries,
            )
            self.update_prefit_model()
            self.statusBar().showMessage("Autoscale applied")
        except Exception as exc:
            self._show_error("Scale recommendation failed", str(exc))

    def _append_scale_recommendation_log(
        self,
        recommendation: PrefitScaleRecommendation,
    ) -> None:
        offset_text = (
            ""
            if recommendation.recommended_offset is None
            else (
                f"Current offset: "
                f"{(recommendation.current_offset or 0.0):.6g}\n"
                f"Recommended offset: "
                f"{recommendation.recommended_offset:.6g}\n"
            )
        )
        message = (
            "Recommended scale estimate available.\n"
            + f"Current scale: {recommendation.current_scale:.6g}\n"
            + f"Recommended scale: {recommendation.recommended_scale:.6g}\n"
            + f"Suggested range: {recommendation.recommended_minimum:.6g} "
            + f"to {recommendation.recommended_maximum:.6g}\n"
            + offset_text
            + f"Adjustment factor: {recommendation.adjustment_factor:.6g}\n"
            + f"Points used: {recommendation.points_used}"
        )
        self.prefit_tab.append_log(message)

    def _maybe_append_scale_recommendation(
        self,
        entries=None,
    ) -> None:
        if self.prefit_workflow is None:
            return
        try:
            recommendation = self.prefit_workflow.recommend_scale_settings(
                entries
            )
        except Exception:
            return
        self._append_scale_recommendation_log(recommendation)

    def _validated_project_dir(self, project_dir: str | Path) -> Path:
        resolved_dir = Path(project_dir).expanduser().resolve()
        project_file = build_project_paths(resolved_dir).project_file
        if not project_file.is_file():
            raise ValueError(
                "Select a complete SAXS project folder that contains "
                "saxs_project.json, not a parent directory of multiple projects."
            )
        return resolved_dir

    def _show_error(self, title: str, message: str) -> None:
        QMessageBox.critical(self, title, message)
        self.statusBar().showMessage(message, 5000)


def launch_saxs_ui(
    initial_project_dir: str | Path | None = None,
) -> int:
    app = QApplication.instance()
    owns_app = app is None
    if app is None:
        prepare_saxshell_application_identity()
        app = QApplication([])
    configure_saxshell_application(app)
    splash = create_saxshell_startup_splash()
    splash.show()
    app.processEvents()
    window: SAXSMainWindow | None = None
    try:
        window = SAXSMainWindow(initial_project_dir=initial_project_dir)
        window.show()
        splash.finish(window)
    except Exception:
        splash.close()
        raise
    if owns_app:
        return int(app.exec())
    return 0
