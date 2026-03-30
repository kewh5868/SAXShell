from __future__ import annotations

import numpy as np
from matplotlib import colormaps
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg,
    NavigationToolbar2QT,
)
from matplotlib.colors import to_hex
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor, QTextCursor, QTextOption, QValidator
from PySide6.QtWidgets import (
    QAbstractItemView,
    QAbstractScrollArea,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGraphicsColorizeEffect,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from saxshell.saxs.dream import (
    DreamModelPlotData,
    DreamRunSettings,
    DreamSummary,
    DreamViolinPlotData,
)

DREAM_SEARCH_FILTER_PRESETS: dict[str, dict[str, object]] = {
    "less_aggressive": {
        "nchains": 4,
        "niterations": 5000,
        "burnin_percent": 15,
        "nseedchains": 24,
        "crossover_burnin": 500,
        "posterior_filter_mode": "all_post_burnin",
        "posterior_top_percent": 20.0,
        "posterior_top_n": 1000,
        "violin_sample_source": "filtered_posterior",
    },
    "medium": {
        "nchains": 4,
        "niterations": 10000,
        "burnin_percent": 20,
        "nseedchains": 40,
        "crossover_burnin": 1000,
        "posterior_filter_mode": "all_post_burnin",
        "posterior_top_percent": 10.0,
        "posterior_top_n": 500,
        "violin_sample_source": "filtered_posterior",
    },
    "more_aggressive": {
        "nchains": 8,
        "niterations": 20000,
        "burnin_percent": 25,
        "nseedchains": 80,
        "crossover_burnin": 2000,
        "posterior_filter_mode": "top_percent_logp",
        "posterior_top_percent": 5.0,
        "posterior_top_n": 250,
        "violin_sample_source": "filtered_posterior",
    },
}


class ScientificDoubleSpinBox(QDoubleSpinBox):
    """QDoubleSpinBox that displays and accepts scientific notation."""

    def textFromValue(self, value: float) -> str:
        if abs(float(value)) < 1e-300:
            return "0.0e+00"
        return f"{float(value):.3e}"

    def valueFromText(self, text: str) -> float:
        cleaned = text.strip()
        if not cleaned:
            return 0.0
        return float(cleaned)

    def validate(
        self,
        text: str,
        position: int,
    ) -> tuple[QValidator.State, str, int]:
        cleaned = text.strip()
        if not cleaned:
            return (QValidator.State.Intermediate, text, position)
        if cleaned in {"-", "+", ".", "-.", "+.", "e", "E"}:
            return (QValidator.State.Intermediate, text, position)
        if cleaned.lower().endswith(("e", "e+", "e-")):
            return (QValidator.State.Intermediate, text, position)
        try:
            value = float(cleaned)
        except ValueError:
            return (QValidator.State.Invalid, text, position)
        if self.minimum() <= value <= self.maximum():
            return (QValidator.State.Acceptable, text, position)
        return (QValidator.State.Invalid, text, position)


class DreamTab(QWidget):
    ACTIVE_SETTINGS_LABEL = "Active project settings"
    NO_SAVED_RUNS_LABEL = "No saved DREAM runs"
    RUNTIME_OUTPUT_FLUSH_INTERVAL_MS = 300
    MAX_VIOLIN_PLOT_SAMPLES = 4096

    edit_parameter_map_requested = Signal()
    save_settings_requested = Signal()
    write_runtime_requested = Signal()
    preview_runtime_requested = Signal()
    run_dream_requested = Signal()
    load_results_requested = Signal()
    save_report_requested = Signal()
    recycle_output_requested = Signal()
    export_model_report_requested = Signal()
    save_model_fit_requested = Signal()
    save_violin_data_requested = Signal()
    settings_preset_changed = Signal(str)
    visualization_settings_changed = Signal()
    results_settings_changed = Signal()
    summary_settings_changed = Signal()
    violin_data_settings_changed = Signal()
    violin_style_settings_changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._console_autoscroll_enabled = True
        self._summary_text = ""
        self._base_log_text = ""
        self._history_messages: list[str] = []
        self._live_output_history_index: int | None = None
        self._pending_runtime_output_lines: list[str] = []
        self._applying_search_filter_preset = False
        self._current_model_plot_data: DreamModelPlotData | None = None
        self._current_summary: DreamSummary | None = None
        self._current_violin_plot_data: DreamViolinPlotData | None = None
        self._model_legend_line_map: dict[object, object] = {}
        self._model_legend_handle_lookup: dict[str, object] = {}
        self._suspend_visualization_notifications = False
        self._blink_timer = QTimer(self)
        self._blink_timer.setInterval(180)
        self._blink_timer.timeout.connect(self._advance_button_blink)
        self._blink_remaining = 0
        self._blink_target_button: QPushButton | None = None
        self._runtime_output_flush_timer = QTimer(self)
        self._runtime_output_flush_timer.setSingleShot(True)
        self._runtime_output_flush_timer.setInterval(
            self.RUNTIME_OUTPUT_FLUSH_INTERVAL_MS
        )
        self._runtime_output_flush_timer.timeout.connect(
            self._flush_pending_runtime_output
        )
        self._build_ui()
        self.set_settings(DreamRunSettings(), preset_name=None)

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._outer_scroll_area = QScrollArea()
        self._outer_scroll_area.setWidgetResizable(True)
        self._outer_scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self._outer_scroll_area.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(12)

        self._output_group = self._build_output_group()
        self._top_splitter = QSplitter()
        self._top_splitter.setOrientation(Qt.Orientation.Horizontal)
        self._top_splitter.setChildrenCollapsible(False)
        self._top_splitter.setHandleWidth(10)
        self._main_splitter = self._top_splitter
        self._settings_scroll_area = self._build_settings_scroll_area()
        self._plot_scroll_area = QScrollArea()
        self._plot_scroll_area.setWidgetResizable(True)
        self._plot_scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self._plot_scroll_area.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self._plot_panel = self._build_plot_panel()
        self._plot_scroll_area.setWidget(self._plot_panel)
        self._top_splitter.addWidget(self._settings_scroll_area)
        self._top_splitter.addWidget(self._plot_scroll_area)
        self._top_splitter.setStretchFactor(0, 4)
        self._top_splitter.setStretchFactor(1, 7)
        self._top_splitter.setSizes([420, 760])
        content_layout.addWidget(self._top_splitter)

        self._outer_scroll_area.setWidget(content)
        root.addWidget(self._outer_scroll_area)

    def _build_settings_scroll_area(self) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(12)
        content_layout.addWidget(self._build_settings_group())
        content_layout.addWidget(self._build_parameter_map_group())
        content_layout.addWidget(self._output_group)
        content_layout.addStretch(1)
        scroll.setWidget(content)
        return scroll

    def _build_settings_group(self) -> QGroupBox:
        group = QGroupBox("DREAM Settings")
        layout = QGridLayout(group)
        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(3, 1)

        row = 0
        self.settings_preset_combo = QComboBox()
        self.settings_preset_combo.currentTextChanged.connect(
            self.settings_preset_changed.emit
        )
        settings_tip = (
            "Load or reference a saved DREAM settings preset from the "
            "project's dream/settings_presets folder."
        )
        settings_label = QLabel("Saved settings")
        self._set_widget_tooltip(
            settings_label,
            self.settings_preset_combo,
            settings_tip,
        )
        layout.addWidget(settings_label, row, 0)
        layout.addWidget(self.settings_preset_combo, row, 1, 1, 3)

        row += 1
        self.search_filter_preset_combo = QComboBox()
        self.search_filter_preset_combo.addItem("Custom", "custom")
        self.search_filter_preset_combo.addItem(
            "Less Aggressive",
            "less_aggressive",
        )
        self.search_filter_preset_combo.addItem("Medium", "medium")
        self.search_filter_preset_combo.addItem(
            "More Aggressive",
            "more_aggressive",
        )
        self.search_filter_preset_combo.currentIndexChanged.connect(
            self._on_search_filter_preset_changed
        )
        search_filter_tip = (
            "Apply a built-in DREAM search and posterior-filtering profile. "
            "Less Aggressive keeps broader posterior filtering and lighter "
            "sampling, Medium matches the default balanced setup, and More "
            "Aggressive increases search depth while tightening posterior "
            "filtering."
        )
        search_filter_label = QLabel("Search/filter preset")
        self._set_widget_tooltip(
            search_filter_label,
            self.search_filter_preset_combo,
            search_filter_tip,
        )
        layout.addWidget(search_filter_label, row, 0)
        layout.addWidget(self.search_filter_preset_combo, row, 1, 1, 3)

        row += 1
        self.model_name_edit = QLineEdit()
        self.model_name_edit.setPlaceholderText("dream")
        model_name_tip = (
            "Unique identifier passed through to the DREAM runtime so the "
            "run can be labeled consistently in saved outputs."
        )
        model_name_label = QLabel("Model name")
        self._set_widget_tooltip(
            model_name_label,
            self.model_name_edit,
            model_name_tip,
        )
        layout.addWidget(model_name_label, row, 0)
        layout.addWidget(self.model_name_edit, row, 1, 1, 3)

        row += 1
        self.chains_spin = QSpinBox()
        self.chains_spin.setRange(1, 512)
        self.chains_spin.valueChanged.connect(
            lambda _value: self._mark_search_filter_preset_custom()
        )
        self.iterations_spin = QSpinBox()
        self.iterations_spin.setRange(100, 10_000_000)
        self.iterations_spin.setSingleStep(100)
        self.iterations_spin.valueChanged.connect(
            lambda _value: self._mark_search_filter_preset_custom()
        )
        chains_tip = (
            "Number of DREAM MCMC chains to run. The runtime bundle will "
            "increase this if needed so there is at least one chain per "
            "varying parameter."
        )
        chains_label = QLabel("Chains")
        self._set_widget_tooltip(
            chains_label,
            self.chains_spin,
            chains_tip,
        )
        iterations_tip = "Number of DREAM iterations to execute per chain."
        iterations_label = QLabel("Iterations")
        self._set_widget_tooltip(
            iterations_label,
            self.iterations_spin,
            iterations_tip,
        )
        layout.addWidget(chains_label, row, 0)
        layout.addWidget(self.chains_spin, row, 1)
        layout.addWidget(iterations_label, row, 2)
        layout.addWidget(self.iterations_spin, row, 3)

        row += 1
        self.burnin_spin = QSpinBox()
        self.burnin_spin.setRange(0, 95)
        self.burnin_spin.valueChanged.connect(
            lambda _value: self._mark_search_filter_preset_custom()
        )
        self.history_thin_spin = QSpinBox()
        self.history_thin_spin.setRange(1, 1000)
        burnin_tip = (
            "Percent of the earliest DREAM samples to discard before MAP "
            "filtering, summary statistics, and violin plotting."
        )
        burnin_label = QLabel("Burn-in (%)")
        self._set_widget_tooltip(
            burnin_label,
            self.burnin_spin,
            burnin_tip,
        )
        history_thin_tip = (
            "Keep every Nth point in DREAM's saved chain-history output."
        )
        history_thin_label = QLabel("History thin")
        self._set_widget_tooltip(
            history_thin_label,
            self.history_thin_spin,
            history_thin_tip,
        )
        layout.addWidget(burnin_label, row, 0)
        layout.addWidget(self.burnin_spin, row, 1)
        layout.addWidget(history_thin_label, row, 2)
        layout.addWidget(self.history_thin_spin, row, 3)

        row += 1
        self.nseedchains_spin = QSpinBox()
        self.nseedchains_spin.setRange(0, 100_000)
        self.nseedchains_spin.valueChanged.connect(
            lambda _value: self._mark_search_filter_preset_custom()
        )
        self.crossover_burnin_spin = QSpinBox()
        self.crossover_burnin_spin.setRange(0, 10_000_000)
        self.crossover_burnin_spin.setSingleStep(100)
        self.crossover_burnin_spin.valueChanged.connect(
            lambda _value: self._mark_search_filter_preset_custom()
        )
        nseedchains_tip = (
            "Number of seed chains DREAM uses to initialize proposals. If "
            "this is too small, the runtime bundle will raise it to at "
            "least 2 x nChains."
        )
        nseedchains_label = QLabel("nSeedChains")
        self._set_widget_tooltip(
            nseedchains_label,
            self.nseedchains_spin,
            nseedchains_tip,
        )
        crossover_tip = (
            "Number of iterations collected before DREAM begins adapting "
            "the crossover probabilities."
        )
        crossover_label = QLabel("Crossover burn-in")
        self._set_widget_tooltip(
            crossover_label,
            self.crossover_burnin_spin,
            crossover_tip,
        )
        layout.addWidget(nseedchains_label, row, 0)
        layout.addWidget(self.nseedchains_spin, row, 1)
        layout.addWidget(crossover_label, row, 2)
        layout.addWidget(self.crossover_burnin_spin, row, 3)

        row += 1
        self.lambda_spin = QDoubleSpinBox()
        self.lambda_spin.setRange(0.0, 100.0)
        self.lambda_spin.setDecimals(6)
        self.lambda_spin.setSingleStep(0.01)
        self.zeta_spin = ScientificDoubleSpinBox()
        self.zeta_spin.setRange(0.0, 1.0)
        self.zeta_spin.setDecimals(16)
        self.zeta_spin.setSingleStep(1e-12)
        lambda_tip = "DREAM proposal step-size scaling factor (lambda)."
        lambda_label = QLabel("Lambda")
        self._set_widget_tooltip(
            lambda_label,
            self.lambda_spin,
            lambda_tip,
        )
        zeta_tip = (
            "Small numerical jitter added to DREAM proposals to avoid "
            "degenerate moves."
        )
        zeta_label = QLabel("Zeta")
        self._set_widget_tooltip(
            zeta_label,
            self.zeta_spin,
            zeta_tip,
        )
        layout.addWidget(lambda_label, row, 0)
        layout.addWidget(self.lambda_spin, row, 1)
        layout.addWidget(zeta_label, row, 2)
        layout.addWidget(self.zeta_spin, row, 3)

        row += 1
        self.snooker_spin = QDoubleSpinBox()
        self.snooker_spin.setRange(0.0, 1.0)
        self.snooker_spin.setDecimals(6)
        self.snooker_spin.setSingleStep(0.01)
        self.p_gamma_unity_spin = QDoubleSpinBox()
        self.p_gamma_unity_spin.setRange(0.0, 1.0)
        self.p_gamma_unity_spin.setDecimals(6)
        self.p_gamma_unity_spin.setSingleStep(0.01)
        snooker_tip = "Probability of using DREAM's snooker update move."
        snooker_label = QLabel("Snooker")
        self._set_widget_tooltip(
            snooker_label,
            self.snooker_spin,
            snooker_tip,
        )
        p_gamma_tip = (
            "Probability of using gamma = 1 proposal scaling in DREAM."
        )
        p_gamma_label = QLabel("p_gamma_unity")
        self._set_widget_tooltip(
            p_gamma_label,
            self.p_gamma_unity_spin,
            p_gamma_tip,
        )
        layout.addWidget(snooker_label, row, 0)
        layout.addWidget(self.snooker_spin, row, 1)
        layout.addWidget(p_gamma_label, row, 2)
        layout.addWidget(self.p_gamma_unity_spin, row, 3)

        row += 1
        self.verbose_checkbox = QCheckBox("Verbose sampler output")
        self.verbose_interval_spin = QDoubleSpinBox()
        self.verbose_interval_spin.setRange(0.1, 30.0)
        self.verbose_interval_spin.setDecimals(1)
        self.verbose_interval_spin.setSingleStep(0.1)
        self.verbose_checkbox.setChecked(True)
        self.verbose_interval_spin.setValue(1.0)
        self.parallel_checkbox = QCheckBox("Run chains in parallel")
        self.adapt_checkbox = QCheckBox("Adapt crossover")
        self.restart_checkbox = QCheckBox("Restart previous run")
        self.verbose_checkbox.setToolTip(
            "Print DREAM sampler progress and status updates into the run output."
        )
        self.verbose_checkbox.toggled.connect(
            lambda _checked: self._update_verbose_output_controls()
        )
        self.verbose_interval_spin.setToolTip(
            "Minimum number of seconds between DREAM runtime output updates "
            "shown in the UI while verbose sampler output is enabled."
        )
        self.parallel_checkbox.setToolTip(
            "Allow DREAM to execute its chains in parallel."
        )
        self.adapt_checkbox.setToolTip(
            "Enable DREAM's adaptive crossover probability updates."
        )
        self.restart_checkbox.setToolTip(
            "Continue from a previous DREAM run instead of starting a fresh one."
        )
        layout.addWidget(self.verbose_checkbox, row, 0, 1, 2)
        layout.addWidget(self.parallel_checkbox, row, 2, 1, 2)

        row += 1
        verbose_interval_label = QLabel("Verbose interval (s)")
        verbose_interval_label.setToolTip(self.verbose_interval_spin.toolTip())
        layout.addWidget(verbose_interval_label, row, 0)
        layout.addWidget(self.verbose_interval_spin, row, 1)
        layout.addWidget(self.adapt_checkbox, row, 2, 1, 2)

        row += 1
        layout.addWidget(self.restart_checkbox, row, 2, 1, 2)

        row += 1
        self.history_file_edit = QLineEdit()
        self.history_file_edit.setPlaceholderText(
            "Optional chain history .npy"
        )
        self.history_browse_button = QPushButton("Browse...")
        self.history_browse_button.clicked.connect(self._browse_history_file)
        history_file_tip = (
            "Optional existing DREAM chain-history .npy file to reuse when "
            "continuing or comparing runs."
        )
        self.history_file_edit.setToolTip(history_file_tip)
        self.history_browse_button.setToolTip(history_file_tip)
        history_row = QWidget()
        history_layout = QHBoxLayout(history_row)
        history_layout.setContentsMargins(0, 0, 0, 0)
        history_layout.setSpacing(6)
        history_layout.addWidget(self.history_file_edit, stretch=1)
        history_layout.addWidget(self.history_browse_button)
        history_label = QLabel("History file")
        history_label.setToolTip(history_file_tip)
        layout.addWidget(history_label, row, 0)
        layout.addWidget(history_row, row, 1, 1, 3)

        row += 1
        self.bestfit_method_combo = QComboBox()
        self.bestfit_method_combo.addItem("MAP", "map")
        self.bestfit_method_combo.addItem("Chain Mean MAP", "chain_mean")
        self.bestfit_method_combo.addItem("Median", "median")
        self.bestfit_method_combo.currentIndexChanged.connect(
            self._on_bestfit_method_changed
        )
        self.violin_mode_combo = QComboBox()
        self.violin_mode_combo.addItem(
            "Varying Parameters", "varying_parameters"
        )
        self.violin_mode_combo.addItem("All Parameters", "all_parameters")
        self.violin_mode_combo.addItem("Weights Only", "weights_only")
        self.violin_mode_combo.addItem("Fit Parameters", "fit_parameters")
        self.violin_mode_combo.addItem(
            "Effective Radii Only",
            "effective_radii_only",
        )
        self.violin_mode_combo.addItem(
            "Additional Parameters Only",
            "additional_parameters_only",
        )
        self.violin_mode_combo.currentIndexChanged.connect(
            self._on_violin_mode_changed
        )
        bestfit_tip = (
            "Choose how posterior samples are reduced to one best-fit "
            "parameter set for the DREAM summary and model-vs-experimental plot."
        )
        bestfit_label = QLabel("Best-fit method")
        self._set_widget_tooltip(
            bestfit_label,
            self.bestfit_method_combo,
            bestfit_tip,
        )
        violin_tip = (
            "Choose which subset of posterior parameters is shown in the "
            "violin plot, including weight-only, effective-radius-only, "
            "and additional-parameter-only views."
        )
        violin_label = QLabel("Violin data")
        self._set_widget_tooltip(
            violin_label,
            self.violin_mode_combo,
            violin_tip,
        )
        layout.addWidget(bestfit_label, row, 0)
        layout.addWidget(self.bestfit_method_combo, row, 1)
        layout.addWidget(violin_label, row, 2)
        layout.addWidget(self.violin_mode_combo, row, 3)

        row += 1
        self.weight_order_combo = QComboBox()
        self.weight_order_combo.addItem("Weight Index", "weight_index")
        self.weight_order_combo.addItem("Structure Order", "structure_order")
        self.weight_order_combo.currentIndexChanged.connect(
            self._on_weight_order_changed
        )
        weight_order_tip = (
            "Choose whether weight parameters stay in their original w-index "
            "order or are reordered by the same cluster-structure ordering "
            "used in the Project Setup prior histograms."
        )
        weight_order_label = QLabel("Weight order")
        self._set_widget_tooltip(
            weight_order_label,
            self.weight_order_combo,
            weight_order_tip,
        )
        layout.addWidget(weight_order_label, row, 0)
        layout.addWidget(self.weight_order_combo, row, 1, 1, 3)

        row += 1
        self.violin_value_scale_combo = QComboBox()
        self.violin_value_scale_combo.addItem(
            "Parameter Value", "parameter_value"
        )
        self.violin_value_scale_combo.addItem(
            "Weights 0-1 Only", "weights_unit_interval"
        )
        self.violin_value_scale_combo.addItem(
            "Normalized 0-1 (All)", "normalized_all"
        )
        self.violin_value_scale_combo.addItem(
            "Effective Radii Only",
            "effective_radii_only",
        )
        self.violin_value_scale_combo.addItem(
            "Additional Parameters Only",
            "additional_parameters_only",
        )
        self.violin_value_scale_combo.currentIndexChanged.connect(
            self._on_violin_value_scale_changed
        )
        value_scale_tip = (
            "Choose whether the posterior violin plot uses native parameter "
            "values, only the weight parameters on a 0-1 fraction scale, "
            "all parameters normalized independently onto a 0-1 axis, or "
            "effective-radius-only / additional-parameter-only views."
        )
        value_scale_label = QLabel("Y-axis scale")
        self._set_widget_tooltip(
            value_scale_label,
            self.violin_value_scale_combo,
            value_scale_tip,
        )
        layout.addWidget(value_scale_label, row, 0)
        layout.addWidget(self.violin_value_scale_combo, row, 1, 1, 3)

        row += 1
        self.violin_palette_combo = QComboBox()
        for label, palette in [
            ("Blues", "Blues"),
            ("Viridis", "viridis"),
            ("Plasma", "plasma"),
            ("Magma", "magma"),
            ("Inferno", "inferno"),
            ("Cividis", "cividis"),
            ("Turbo", "turbo"),
            ("Cubehelix", "cubehelix"),
            ("Coolwarm", "coolwarm"),
            ("Set2", "Set2"),
            ("Set3", "Set3"),
            ("tab10", "tab10"),
            ("tab20", "tab20"),
            ("Custom Solid", "custom_solid"),
        ]:
            self.violin_palette_combo.addItem(label, palette)
        self.violin_palette_combo.currentIndexChanged.connect(
            self._on_violin_palette_changed
        )
        violin_palette_tip = (
            "Choose the color palette used to fill the posterior violin "
            "distributions."
        )
        violin_palette_label = QLabel("Violin palette")
        self._set_widget_tooltip(
            violin_palette_label,
            self.violin_palette_combo,
            violin_palette_tip,
        )
        point_color_tip = (
            "Choose the color used for the selected best-fit scatter points "
            "overlaid on the violin plot."
        )
        self.violin_point_color_button = QPushButton()
        self.violin_point_color_button.clicked.connect(
            self._choose_violin_point_color
        )
        self._configure_plot_color_button(
            self.violin_point_color_button,
            "tab:red",
            label="Point",
        )
        point_color_label = QLabel("Point color")
        self._set_widget_tooltip(
            point_color_label,
            self.violin_point_color_button,
            point_color_tip,
        )
        layout.addWidget(violin_palette_label, row, 0)
        layout.addWidget(self.violin_palette_combo, row, 1)
        layout.addWidget(point_color_label, row, 2)
        layout.addWidget(self.violin_point_color_button, row, 3)

        row += 1
        self.violin_custom_color_button = QPushButton()
        self.violin_custom_color_button.clicked.connect(
            self._choose_violin_custom_color
        )
        self._configure_plot_color_button(
            self.violin_custom_color_button,
            "#4c72b0",
            label="Custom violin",
        )
        violin_custom_tip = (
            "Choose the solid fill color used when the violin palette is "
            "set to Custom Solid."
        )
        violin_custom_label = QLabel("Custom violin")
        self._set_widget_tooltip(
            violin_custom_label,
            self.violin_custom_color_button,
            violin_custom_tip,
        )
        self.interval_color_button = QPushButton()
        self.interval_color_button.clicked.connect(self._choose_interval_color)
        self._configure_plot_color_button(
            self.interval_color_button,
            "#8c8c8c",
            label="Interval",
        )
        interval_color_tip = (
            "Choose the color used for the posterior interval bars and "
            "the violin whisker lines."
        )
        interval_color_label = QLabel("Interval color")
        self._set_widget_tooltip(
            interval_color_label,
            self.interval_color_button,
            interval_color_tip,
        )
        layout.addWidget(violin_custom_label, row, 0)
        layout.addWidget(self.violin_custom_color_button, row, 1)
        layout.addWidget(interval_color_label, row, 2)
        layout.addWidget(self.interval_color_button, row, 3)

        row += 1
        self.median_color_button = QPushButton()
        self.median_color_button.clicked.connect(self._choose_median_color)
        self._configure_plot_color_button(
            self.median_color_button,
            "#4d4d4d",
            label="Median",
        )
        median_color_tip = "Choose the color used for the violin median lines."
        median_color_label = QLabel("Median color")
        self._set_widget_tooltip(
            median_color_label,
            self.median_color_button,
            median_color_tip,
        )
        outline_color_tip = (
            "Choose the outline color drawn around each violin body."
        )
        self.violin_outline_color_button = QPushButton()
        self.violin_outline_color_button.clicked.connect(
            self._choose_outline_color
        )
        self._configure_plot_color_button(
            self.violin_outline_color_button,
            "#000000",
            label="Outline",
        )
        outline_color_label = QLabel("Outline")
        self._set_widget_tooltip(
            outline_color_label,
            self.violin_outline_color_button,
            outline_color_tip,
        )
        outline_width_tip = (
            "Set the thickness of the violin outline stroke in points."
        )
        self.violin_outline_width_spin = QDoubleSpinBox()
        self.violin_outline_width_spin.setDecimals(1)
        self.violin_outline_width_spin.setRange(0.1, 5.0)
        self.violin_outline_width_spin.setSingleStep(0.1)
        self.violin_outline_width_spin.setValue(0.8)
        self.violin_outline_width_spin.valueChanged.connect(
            self._on_violin_outline_width_changed
        )
        outline_width_label = QLabel("Width")
        self._set_widget_tooltip(
            outline_width_label,
            self.violin_outline_width_spin,
            outline_width_tip,
        )
        outline_widget = QWidget()
        outline_layout = QHBoxLayout(outline_widget)
        outline_layout.setContentsMargins(0, 0, 0, 0)
        outline_layout.setSpacing(6)
        outline_layout.addWidget(outline_color_label)
        outline_layout.addWidget(self.violin_outline_color_button)
        outline_layout.addWidget(outline_width_label)
        outline_layout.addWidget(self.violin_outline_width_spin)
        layout.addWidget(median_color_label, row, 0)
        layout.addWidget(self.median_color_button, row, 1)
        layout.addWidget(outline_widget, row, 2, 1, 2)

        row += 1
        layout.addWidget(
            self._build_posterior_filter_group(),
            row,
            0,
            1,
            4,
        )

        self.edit_button = QPushButton("Edit Priors")
        self.edit_button.setToolTip(
            "Open the DREAM prior editor and save the parameter map before "
            "running a refinement."
        )
        self.edit_button.clicked.connect(
            self.edit_parameter_map_requested.emit
        )
        self.save_settings_button = QPushButton("Save Settings")
        self.save_settings_button.setToolTip(
            "Save the current DREAM settings to the active project state "
            "and optionally create a named reusable settings preset."
        )
        self.save_settings_button.clicked.connect(
            self.save_settings_requested.emit
        )
        self.write_button = QPushButton("Write Runtime Bundle")
        self.write_button.setToolTip(
            "Generate the DREAM runtime script bundle using the current "
            "prefit state, settings, and saved parameter map."
        )
        self.write_button.clicked.connect(self.write_runtime_requested.emit)
        self.preview_button = QPushButton("Preview Runtime Bundle")
        self.preview_button.setToolTip(
            "Open the most recently written DREAM runtime script in the "
            "system's default text editor."
        )
        self.preview_button.clicked.connect(
            self.preview_runtime_requested.emit
        )
        self.run_button = QPushButton("Run DREAM")
        self.run_button.setToolTip(
            "Execute the DREAM refinement using the current settings and "
            "saved parameter map."
        )
        self.run_button.clicked.connect(self.run_dream_requested.emit)
        self.saved_runs_combo = QComboBox()
        self.saved_runs_combo.setToolTip(
            "Choose a completed DREAM run from the active project to "
            "reload its saved settings, prior parameter map, and analysis "
            "results for inspection."
        )
        self.saved_runs_combo.setEnabled(False)
        self.load_button = QPushButton("Load Selected Run")
        self.load_button.setToolTip(
            "Load the selected DREAM run from the active project runtime "
            "folder and redraw the summary plots using that run's saved "
            "settings and priors."
        )
        self.load_button.clicked.connect(self.load_results_requested.emit)
        self.load_button.setEnabled(False)
        self.report_button = QPushButton("Save Statistics")
        self.report_button.setToolTip(
            "Save a text report of the currently loaded DREAM posterior summary."
        )
        self.report_button.clicked.connect(self.save_report_requested.emit)
        self.recycle_button = QPushButton("Recycle")
        self.recycle_button.setToolTip(
            "Copy the currently selected DREAM best-fit parameter values into "
            "the Prefit parameter table so you can manually refine them "
            "before running DREAM again."
        )
        self.recycle_button.clicked.connect(self.recycle_output_requested.emit)
        self.export_model_report_button = QPushButton(
            "Export Model Report (PPTX)"
        )
        self.export_model_report_button.setToolTip(
            "Create a multi-slide PowerPoint report for the current DREAM "
            "fit, including project information, prior histograms, prefit "
            "state, posterior filtering comparisons, and output paths."
        )
        self.export_model_report_button.clicked.connect(
            self.export_model_report_requested.emit
        )

        row += 1
        action_sections = QWidget()
        action_sections_layout = QVBoxLayout(action_sections)
        action_sections_layout.setContentsMargins(0, 0, 0, 0)
        action_sections_layout.setSpacing(10)
        self.setup_actions_group = self._build_action_group(
            "DREAM Setup",
            [
                self.edit_button,
                self.save_settings_button,
                self.write_button,
                self.preview_button,
                self.run_button,
            ],
        )
        self.analysis_actions_group = QGroupBox("DREAM Analysis")
        analysis_layout = QGridLayout(self.analysis_actions_group)
        analysis_layout.addWidget(QLabel("Saved run"), 0, 0)
        analysis_layout.addWidget(self.saved_runs_combo, 0, 1, 1, 2)
        analysis_layout.addWidget(self.load_button, 1, 0)
        analysis_layout.addWidget(self.report_button, 1, 1)
        analysis_layout.addWidget(self.recycle_button, 1, 2)
        analysis_layout.addWidget(
            self.export_model_report_button,
            2,
            0,
            1,
            3,
        )
        action_sections_layout.addWidget(self.setup_actions_group)
        action_sections_layout.addWidget(self.analysis_actions_group)
        layout.addWidget(action_sections, row, 0, 1, 4)
        self._set_combo_data(self.search_filter_preset_combo, "medium")
        self._update_verbose_output_controls()
        self._update_violin_style_controls()
        return group

    def _build_posterior_filter_group(self) -> QGroupBox:
        group = QGroupBox("Posterior Filtering")
        layout = QGridLayout(group)
        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(3, 1)

        self.posterior_filter_combo = QComboBox()
        self.posterior_filter_combo.addItem(
            "All Post-burnin Samples", "all_post_burnin"
        )
        self.posterior_filter_combo.addItem(
            "Top % by Log-posterior", "top_percent_logp"
        )
        self.posterior_filter_combo.addItem(
            "Top N by Log-posterior", "top_n_logp"
        )
        self.posterior_filter_combo.currentIndexChanged.connect(
            self._on_posterior_filter_mode_changed
        )
        filter_tip = (
            "Choose which posterior samples are kept after burn-in before "
            "the best-fit summary, violin plot, and statistics report are "
            "computed."
        )
        filter_label = QLabel("Posterior filter")
        self._set_widget_tooltip(
            filter_label,
            self.posterior_filter_combo,
            filter_tip,
        )

        self.violin_sample_source_combo = QComboBox()
        self.violin_sample_source_combo.addItem(
            "Filtered Posterior", "filtered_posterior"
        )
        self.violin_sample_source_combo.addItem(
            "MAP Chain Only", "map_chain_only"
        )
        self.violin_sample_source_combo.currentIndexChanged.connect(
            lambda _index: self._mark_search_filter_preset_custom()
        )
        self.violin_sample_source_combo.currentIndexChanged.connect(
            self._on_violin_sample_source_changed
        )
        violin_source_tip = (
            "Choose whether the violin plot shows the full filtered "
            "posterior sample set or only the filtered samples from the "
            "chain containing the global MAP point."
        )
        violin_source_label = QLabel("Violin samples")
        self._set_widget_tooltip(
            violin_source_label,
            self.violin_sample_source_combo,
            violin_source_tip,
        )
        layout.addWidget(filter_label, 0, 0)
        layout.addWidget(self.posterior_filter_combo, 0, 1)
        layout.addWidget(violin_source_label, 0, 2)
        layout.addWidget(self.violin_sample_source_combo, 0, 3)

        self.posterior_top_percent_spin = QDoubleSpinBox()
        self.posterior_top_percent_spin.setRange(0.1, 100.0)
        self.posterior_top_percent_spin.setDecimals(2)
        self.posterior_top_percent_spin.setSingleStep(1.0)
        self.posterior_top_percent_spin.setValue(10.0)
        self.posterior_top_percent_spin.valueChanged.connect(
            lambda _value: self._mark_search_filter_preset_custom()
        )
        self.posterior_top_percent_spin.valueChanged.connect(
            self._on_posterior_top_percent_changed
        )
        top_percent_tip = (
            "Default percent used whenever Top % log-posterior screening is "
            "evaluated or selected. The automatic post-run filter "
            "assessment also uses this value."
        )
        top_percent_label = QLabel("Top % default")
        self._set_widget_tooltip(
            top_percent_label,
            self.posterior_top_percent_spin,
            top_percent_tip,
        )

        self.posterior_top_n_spin = QSpinBox()
        self.posterior_top_n_spin.setRange(1, 10_000_000)
        self.posterior_top_n_spin.setValue(500)
        self.posterior_top_n_spin.valueChanged.connect(
            lambda _value: self._mark_search_filter_preset_custom()
        )
        self.posterior_top_n_spin.valueChanged.connect(
            self._on_posterior_top_n_changed
        )
        top_n_tip = (
            "Default count used whenever Top N log-posterior screening is "
            "evaluated or selected. The automatic post-run filter "
            "assessment also uses this value."
        )
        top_n_label = QLabel("Top N default")
        self._set_widget_tooltip(
            top_n_label,
            self.posterior_top_n_spin,
            top_n_tip,
        )
        layout.addWidget(top_percent_label, 1, 0)
        layout.addWidget(self.posterior_top_percent_spin, 1, 1)
        layout.addWidget(top_n_label, 1, 2)
        layout.addWidget(self.posterior_top_n_spin, 1, 3)

        self.auto_filter_assessment_checkbox = QCheckBox(
            "Auto-select best filter after run"
        )
        self.auto_filter_assessment_checkbox.setChecked(True)
        self.auto_filter_assessment_checkbox.setToolTip(
            "After a DREAM refinement completes, evaluate All Post-burnin, "
            "Top % by log-posterior, and Top N by log-posterior using the "
            "default Top % / Top N values above, then automatically apply "
            "the filter with the best fit quality."
        )
        layout.addWidget(self.auto_filter_assessment_checkbox, 2, 0, 1, 4)

        self.credible_interval_low_spin = QDoubleSpinBox()
        self.credible_interval_low_spin.setRange(0.0, 99.9)
        self.credible_interval_low_spin.setDecimals(1)
        self.credible_interval_low_spin.setSingleStep(1.0)
        self.credible_interval_low_spin.setValue(16.0)
        self.credible_interval_low_spin.valueChanged.connect(
            self._on_credible_interval_low_changed
        )
        interval_low_tip = (
            "Lower percentile used for the posterior interval bars and "
            "reported statistics."
        )
        interval_low_label = QLabel("Interval low (%)")
        self._set_widget_tooltip(
            interval_low_label,
            self.credible_interval_low_spin,
            interval_low_tip,
        )

        self.credible_interval_high_spin = QDoubleSpinBox()
        self.credible_interval_high_spin.setRange(0.1, 100.0)
        self.credible_interval_high_spin.setDecimals(1)
        self.credible_interval_high_spin.setSingleStep(1.0)
        self.credible_interval_high_spin.setValue(84.0)
        self.credible_interval_high_spin.valueChanged.connect(
            self._on_credible_interval_high_changed
        )
        interval_high_tip = (
            "Upper percentile used for the posterior interval bars and "
            "reported statistics."
        )
        interval_high_label = QLabel("Interval high (%)")
        self._set_widget_tooltip(
            interval_high_label,
            self.credible_interval_high_spin,
            interval_high_tip,
        )
        layout.addWidget(interval_low_label, 3, 0)
        layout.addWidget(self.credible_interval_low_spin, 3, 1)
        layout.addWidget(interval_high_label, 3, 2)
        layout.addWidget(self.credible_interval_high_spin, 3, 3)

        self._update_posterior_filter_controls()
        return group

    def _build_parameter_map_group(self) -> QGroupBox:
        group = QGroupBox("Current Prior Map")
        layout = QVBoxLayout(group)
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
        self.parameter_map_table.setMinimumHeight(240)
        header = self.parameter_map_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setStretchLastSection(True)
        layout.addWidget(self.parameter_map_table)
        return group

    def _build_plot_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        panel.setMinimumHeight(860)
        self._plot_splitter = QSplitter(Qt.Orientation.Vertical)
        self._plot_splitter.setChildrenCollapsible(False)
        self._plot_splitter.setHandleWidth(10)
        self._plot_splitter.addWidget(self._build_model_plot_group())
        self._plot_splitter.addWidget(self._build_violin_plot_group())
        self._plot_splitter.setStretchFactor(0, 1)
        self._plot_splitter.setStretchFactor(1, 1)
        self._plot_splitter.setSizes([430, 430])
        layout.addWidget(self._plot_splitter, stretch=1)
        return panel

    def _build_model_plot_group(self) -> QGroupBox:
        group = QGroupBox("DREAM Model vs Experimental")
        layout = QVBoxLayout(group)
        controls = QHBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        self.show_experimental_trace_checkbox = QCheckBox("Experimental")
        self.show_experimental_trace_checkbox.setChecked(True)
        self.show_experimental_trace_checkbox.toggled.connect(
            self._redraw_current_model_plot
        )
        self.show_model_trace_checkbox = QCheckBox("Model")
        self.show_model_trace_checkbox.setChecked(True)
        self.show_model_trace_checkbox.toggled.connect(
            self._redraw_current_model_plot
        )
        self.show_solvent_trace_checkbox = QCheckBox("Solvent")
        self.show_solvent_trace_checkbox.setChecked(False)
        self.show_solvent_trace_checkbox.toggled.connect(
            self._redraw_current_model_plot
        )
        self.show_structure_factor_trace_checkbox = QCheckBox(
            "Structure factor"
        )
        self.show_structure_factor_trace_checkbox.setChecked(False)
        self.show_structure_factor_trace_checkbox.toggled.connect(
            self._redraw_current_model_plot
        )
        self.model_log_x_checkbox = QCheckBox("Log X")
        self.model_log_x_checkbox.setChecked(True)
        self.model_log_x_checkbox.toggled.connect(
            self._redraw_current_model_plot
        )
        self.model_log_y_checkbox = QCheckBox("Log Y")
        self.model_log_y_checkbox.setChecked(True)
        self.model_log_y_checkbox.toggled.connect(
            self._redraw_current_model_plot
        )
        controls.addWidget(self.show_experimental_trace_checkbox)
        controls.addWidget(self.show_model_trace_checkbox)
        controls.addWidget(self.show_solvent_trace_checkbox)
        controls.addWidget(self.show_structure_factor_trace_checkbox)
        controls.addWidget(self.model_log_x_checkbox)
        controls.addWidget(self.model_log_y_checkbox)
        controls.addStretch(1)
        self.save_model_button = QPushButton("Export Data")
        self.save_model_button.setToolTip(
            "Export a condensed copy of the displayed DREAM model-fit "
            "data to exported_results/data, alongside companion metadata "
            "and a fit report. Full DREAM run artifacts remain in the "
            "DREAM run folder."
        )
        self.save_model_button.clicked.connect(
            self.save_model_fit_requested.emit
        )
        controls.addWidget(self.save_model_button)
        layout.addLayout(controls)
        self.model_figure = Figure(figsize=(8.0, 4.2))
        self.model_canvas = FigureCanvasQTAgg(self.model_figure)
        self.model_canvas.mpl_connect(
            "pick_event", self._handle_model_legend_pick
        )
        self.model_toolbar = NavigationToolbar2QT(self.model_canvas, self)
        self.model_canvas.setMinimumHeight(320)
        layout.addWidget(self.model_toolbar)
        layout.addWidget(self.model_canvas)
        return group

    def _build_violin_plot_group(self) -> QGroupBox:
        group = QGroupBox("Posterior Violin Plot")
        layout = QVBoxLayout(group)
        controls = QHBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.addStretch(1)
        self.save_violin_button = QPushButton("Export Data")
        self.save_violin_button.setToolTip(
            "Export a condensed copy of the displayed posterior violin "
            "data to exported_results/data, alongside companion metadata "
            "and a fit report. Full DREAM run artifacts remain in the "
            "DREAM run folder."
        )
        self.save_violin_button.clicked.connect(
            self.save_violin_data_requested.emit
        )
        controls.addWidget(self.save_violin_button)
        layout.addLayout(controls)
        self.violin_figure = Figure(figsize=(8.0, 4.2))
        self.violin_canvas = FigureCanvasQTAgg(self.violin_figure)
        self.violin_toolbar = NavigationToolbar2QT(self.violin_canvas, self)
        self.violin_canvas.setMinimumHeight(320)
        layout.addWidget(self.violin_toolbar)
        layout.addWidget(self.violin_canvas)
        return group

    def _build_output_group(self) -> QGroupBox:
        group = QGroupBox("DREAM Output")
        layout = QVBoxLayout(group)
        self.progress_label = QLabel("Progress: idle")
        self.progress_label.setWordWrap(True)
        self.progress_label.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Preferred,
        )
        layout.addWidget(self.progress_label)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%v / %m runs")
        layout.addWidget(self.progress_bar)
        self.output_box = QTextEdit()
        self.output_box.setReadOnly(True)
        self.output_box.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self.output_box.setWordWrapMode(QTextOption.WrapMode.WrapAnywhere)
        self.output_box.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.output_box.setSizeAdjustPolicy(
            QAbstractScrollArea.SizeAdjustPolicy.AdjustIgnored
        )
        self.output_box.setMinimumHeight(220)
        layout.addWidget(self.output_box)
        self.log_box = self.output_box
        self.summary_box = self.output_box
        return group

    def set_available_settings_presets(
        self,
        preset_names: list[str],
        selected_name: str | None = None,
    ) -> None:
        active_text = (
            selected_name
            if selected_name is not None
            else self.settings_preset_combo.currentText().strip()
        )
        self.settings_preset_combo.blockSignals(True)
        self.settings_preset_combo.clear()
        self.settings_preset_combo.addItem(self.ACTIVE_SETTINGS_LABEL, None)
        for preset_name in preset_names:
            self.settings_preset_combo.addItem(preset_name, preset_name)
        if active_text:
            index = self.settings_preset_combo.findText(active_text)
            if index >= 0:
                self.settings_preset_combo.setCurrentIndex(index)
            else:
                self.settings_preset_combo.setCurrentIndex(0)
        else:
            self.settings_preset_combo.setCurrentIndex(0)
        self.settings_preset_combo.blockSignals(False)

    def selected_settings_preset_name(self) -> str | None:
        text = self.settings_preset_combo.currentText().strip()
        if not text or text == self.ACTIVE_SETTINGS_LABEL:
            return None
        return text

    def set_available_saved_runs(
        self,
        runs: list[tuple[str, str]],
        selected_run_dir: str | None = None,
    ) -> None:
        selected_value = (
            selected_run_dir
            if selected_run_dir is not None
            else self.selected_saved_run_dir()
        )
        self.saved_runs_combo.blockSignals(True)
        self.saved_runs_combo.clear()
        if not runs:
            self.saved_runs_combo.addItem(self.NO_SAVED_RUNS_LABEL, None)
            self.saved_runs_combo.setCurrentIndex(0)
            self.saved_runs_combo.setEnabled(False)
            self.load_button.setEnabled(False)
            self.saved_runs_combo.blockSignals(False)
            return
        for label, run_dir in runs:
            self.saved_runs_combo.addItem(label, run_dir)
        if selected_value:
            index = self.saved_runs_combo.findData(selected_value)
            if index >= 0:
                self.saved_runs_combo.setCurrentIndex(index)
            else:
                self.saved_runs_combo.setCurrentIndex(0)
        else:
            self.saved_runs_combo.setCurrentIndex(0)
        self.saved_runs_combo.setEnabled(True)
        self.load_button.setEnabled(True)
        self.saved_runs_combo.blockSignals(False)

    def selected_saved_run_dir(self) -> str | None:
        value = self.saved_runs_combo.currentData()
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def set_settings(
        self,
        settings: DreamRunSettings,
        *,
        preset_name: str | None = None,
    ) -> None:
        self._suspend_visualization_notifications = True
        self._applying_search_filter_preset = True
        try:
            self.settings_preset_combo.blockSignals(True)
            self.settings_preset_combo.setCurrentText(
                preset_name or self.ACTIVE_SETTINGS_LABEL
            )
            self.settings_preset_combo.blockSignals(False)
            self.model_name_edit.setText(settings.model_name or "")
            self.chains_spin.setValue(settings.nchains)
            self.iterations_spin.setValue(settings.niterations)
            self.burnin_spin.setValue(settings.burnin_percent)
            self.history_thin_spin.setValue(settings.history_thin)
            self.nseedchains_spin.setValue(settings.nseedchains)
            self.crossover_burnin_spin.setValue(settings.crossover_burnin)
            self.lambda_spin.setValue(settings.lamb)
            self.zeta_spin.setValue(settings.zeta)
            self.snooker_spin.setValue(settings.snooker)
            self.p_gamma_unity_spin.setValue(settings.p_gamma_unity)
            self.history_file_edit.setText(settings.history_file or "")
            self.verbose_checkbox.setChecked(settings.verbose)
            self.parallel_checkbox.setChecked(settings.parallel)
            self.adapt_checkbox.setChecked(settings.adapt_crossover)
            self.restart_checkbox.setChecked(settings.restart)
            self._set_combo_data(
                self.search_filter_preset_combo,
                settings.search_filter_preset or "custom",
            )
            self._set_combo_data(
                self.bestfit_method_combo, settings.bestfit_method
            )
            self.verbose_interval_spin.setValue(
                settings.verbose_output_interval_seconds
            )
            self._set_combo_data(
                self.posterior_filter_combo,
                settings.posterior_filter_mode,
            )
            self.posterior_top_percent_spin.setValue(
                settings.posterior_top_percent
            )
            self.posterior_top_n_spin.setValue(settings.posterior_top_n)
            self.auto_filter_assessment_checkbox.setChecked(
                settings.auto_select_best_posterior_filter
            )
            self.credible_interval_low_spin.setValue(
                settings.credible_interval_low
            )
            self.credible_interval_high_spin.setValue(
                settings.credible_interval_high
            )
            self._set_combo_data(
                self.violin_mode_combo,
                settings.violin_parameter_mode,
            )
            self._set_combo_data(
                self.violin_sample_source_combo,
                settings.violin_sample_source,
            )
            self._set_combo_data(
                self.weight_order_combo,
                settings.violin_weight_order,
            )
            self._set_combo_data(
                self.violin_value_scale_combo,
                settings.violin_value_scale_mode,
            )
            self._configure_plot_color_button(
                self.violin_custom_color_button,
                settings.violin_custom_color,
                label="Custom violin",
            )
            if not self._set_combo_data(
                self.violin_palette_combo,
                settings.violin_palette,
            ):
                self._set_combo_data(
                    self.violin_palette_combo,
                    "custom_solid",
                )
            self._configure_plot_color_button(
                self.violin_point_color_button,
                settings.violin_point_color,
                label="Point",
            )
            self._configure_plot_color_button(
                self.interval_color_button,
                settings.violin_interval_color,
                label="Interval",
            )
            self._configure_plot_color_button(
                self.median_color_button,
                settings.violin_median_color,
                label="Median",
            )
            self._configure_plot_color_button(
                self.violin_outline_color_button,
                settings.violin_outline_color,
                label="Outline",
            )
            self.violin_outline_width_spin.setValue(
                float(settings.violin_outline_width)
            )
        finally:
            self._applying_search_filter_preset = False
            self._suspend_visualization_notifications = False
        self._update_violin_style_controls()
        self._update_posterior_filter_controls()
        self._update_verbose_output_controls()

    def settings_payload(self) -> DreamRunSettings:
        return DreamRunSettings(
            nchains=int(self.chains_spin.value()),
            niterations=int(self.iterations_spin.value()),
            burnin_percent=int(self.burnin_spin.value()),
            restart=bool(self.restart_checkbox.isChecked()),
            verbose=bool(self.verbose_checkbox.isChecked()),
            verbose_output_interval_seconds=float(
                self.verbose_interval_spin.value()
            ),
            parallel=bool(self.parallel_checkbox.isChecked()),
            nseedchains=int(self.nseedchains_spin.value()),
            adapt_crossover=bool(self.adapt_checkbox.isChecked()),
            crossover_burnin=int(self.crossover_burnin_spin.value()),
            lamb=float(self.lambda_spin.value()),
            zeta=float(self.zeta_spin.value()),
            snooker=float(self.snooker_spin.value()),
            p_gamma_unity=float(self.p_gamma_unity_spin.value()),
            history_thin=int(self.history_thin_spin.value()),
            history_file=self.history_file_edit.text().strip() or None,
            model_name=self.model_name_edit.text().strip() or None,
            search_filter_preset=self.selected_search_filter_preset(),
            bestfit_method=self.selected_bestfit_method(),
            posterior_filter_mode=self.selected_posterior_filter_mode(),
            posterior_top_percent=float(
                self.posterior_top_percent_spin.value()
            ),
            posterior_top_n=int(self.posterior_top_n_spin.value()),
            auto_select_best_posterior_filter=bool(
                self.auto_filter_assessment_checkbox.isChecked()
            ),
            credible_interval_low=float(
                self.credible_interval_low_spin.value()
            ),
            credible_interval_high=float(
                self.credible_interval_high_spin.value()
            ),
            violin_parameter_mode=self.selected_violin_mode(),
            violin_sample_source=self.selected_violin_sample_source(),
            violin_weight_order=self.selected_weight_order(),
            violin_value_scale_mode=self.selected_violin_value_scale_mode(),
            violin_palette=self.selected_violin_palette(),
            violin_custom_color=self.selected_violin_custom_color(),
            violin_point_color=self.selected_violin_point_color(),
            violin_interval_color=self.selected_violin_interval_color(),
            violin_median_color=self.selected_violin_median_color(),
            violin_outline_color=self.selected_violin_outline_color(),
            violin_outline_width=float(self.violin_outline_width_spin.value()),
        )

    def set_parameter_map_entries(self, entries) -> None:
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
                QTableWidgetItem(str(entry.dist_params)),
            )
        self.parameter_map_table.resizeRowsToContents()

    def selected_bestfit_method(self) -> str:
        return str(self.bestfit_method_combo.currentData() or "map")

    def selected_search_filter_preset(self) -> str:
        return str(self.search_filter_preset_combo.currentData() or "custom")

    def selected_posterior_filter_mode(self) -> str:
        return str(
            self.posterior_filter_combo.currentData() or "all_post_burnin"
        )

    def selected_violin_mode(self) -> str:
        return str(
            self.violin_mode_combo.currentData() or "varying_parameters"
        )

    def selected_violin_sample_source(self) -> str:
        return str(
            self.violin_sample_source_combo.currentData()
            or "filtered_posterior"
        )

    def selected_weight_order(self) -> str:
        return str(self.weight_order_combo.currentData() or "weight_index")

    def selected_violin_value_scale_mode(self) -> str:
        return str(
            self.violin_value_scale_combo.currentData() or "parameter_value"
        )

    def selected_violin_palette(self) -> str:
        return str(self.violin_palette_combo.currentData() or "Blues")

    def selected_violin_custom_color(self) -> str:
        return self._plot_color_button_value(
            self.violin_custom_color_button,
            default="#4c72b0",
        )

    def selected_violin_point_color(self) -> str:
        return self._plot_color_button_value(
            self.violin_point_color_button,
            default="tab:red",
        )

    def selected_violin_interval_color(self) -> str:
        return self._plot_color_button_value(
            self.interval_color_button,
            default="#8c8c8c",
        )

    def selected_violin_median_color(self) -> str:
        return self._plot_color_button_value(
            self.median_color_button,
            default="#4d4d4d",
        )

    def selected_violin_outline_color(self) -> str:
        return self._plot_color_button_value(
            self.violin_outline_color_button,
            default="#000000",
        )

    def _update_verbose_output_controls(self) -> None:
        self.verbose_interval_spin.setEnabled(
            self.verbose_checkbox.isChecked()
        )

    def append_log(self, message: str) -> None:
        self._flush_pending_runtime_output()
        stripped = message.strip()
        if stripped:
            self._history_messages.append(stripped)
            self._live_output_history_index = None
        self._render_output(scroll_to_end=True)

    def set_log_text(self, text: str) -> None:
        self._runtime_output_flush_timer.stop()
        self._pending_runtime_output_lines = []
        self._base_log_text = text.strip()
        self._history_messages = []
        self._live_output_history_index = None
        self._render_output()

    def set_summary_text(self, text: str) -> None:
        self._summary_text = text.strip()
        self._render_output()

    def set_console_autoscroll_enabled(self, enabled: bool) -> None:
        self._console_autoscroll_enabled = bool(enabled)
        if self._console_autoscroll_enabled:
            self._scroll_output_to_end()

    def append_runtime_output(self, message: str) -> None:
        stripped = message.rstrip()
        if not stripped:
            return
        self._pending_runtime_output_lines.append(stripped)
        if not self._runtime_output_flush_timer.isActive():
            self._runtime_output_flush_timer.start()

    def _flush_pending_runtime_output(self) -> None:
        if not self._pending_runtime_output_lines:
            return
        chunk = "\n".join(self._pending_runtime_output_lines)
        self._pending_runtime_output_lines = []
        if self._live_output_history_index is None:
            self._history_messages.append("DREAM Runtime Output\n" + chunk)
            self._live_output_history_index = len(self._history_messages) - 1
        else:
            self._history_messages[self._live_output_history_index] += (
                "\n" + chunk
            )
        self._render_output(scroll_to_end=True)

    def finish_runtime_output(self) -> None:
        self._runtime_output_flush_timer.stop()
        self._flush_pending_runtime_output()
        self._live_output_history_index = None

    def clear_plots(self) -> None:
        self.plot_model_fit(None)
        self.plot_violin_plot(None, None)

    def start_progress(self, message: str) -> None:
        self.progress_label.setText(message)
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("")

    def begin_progress(
        self,
        total: int,
        message: str,
        *,
        unit_label: str = "steps",
    ) -> None:
        total = max(int(total), 1)
        self.progress_label.setText(message)
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat(f"%v / %m {unit_label}")

    def update_progress(
        self,
        processed: int,
        total: int,
        message: str,
        *,
        unit_label: str = "steps",
    ) -> None:
        total = max(int(total), 1)
        processed = max(0, min(int(processed), total))
        self.progress_label.setText(message)
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(processed)
        self.progress_bar.setFormat(f"%v / %m {unit_label}")

    def finish_progress(
        self,
        message: str,
        *,
        total: int | None = None,
        unit_label: str = "runs",
    ) -> None:
        if total is None:
            total = 1
        self.progress_label.setText(message)
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(total)
        self.progress_bar.setFormat(f"%v / %m {unit_label}")

    def reset_progress(self) -> None:
        self.progress_label.setText("Progress: idle")
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%v / %m runs")

    def plot_model_fit(self, plot_data: DreamModelPlotData | None) -> None:
        self._current_model_plot_data = plot_data
        self._model_legend_line_map.clear()
        self._model_legend_handle_lookup.clear()
        for axis in self.model_figure.axes:
            axis.set_xscale("linear")
            axis.set_yscale("linear")
        self.model_figure.clear()
        self._update_model_trace_toggle_state(plot_data)
        if plot_data is None:
            axis = self.model_figure.add_subplot(111)
            axis.text(
                0.5,
                0.5,
                "No DREAM results have been loaded yet.",
                ha="center",
                va="center",
            )
            axis.set_axis_off()
            self.model_canvas.draw_idle()
            return

        grid = self.model_figure.add_gridspec(2, 1, height_ratios=[3, 1])
        top_axis = self.model_figure.add_subplot(grid[0, 0])
        bottom_axis = self.model_figure.add_subplot(
            grid[1, 0], sharex=top_axis
        )

        plotted_lines: list[object] = []
        structure_axis = None
        if self.show_experimental_trace_checkbox.isChecked():
            experimental_artist = top_axis.scatter(
                plot_data.q_values,
                plot_data.experimental_intensities,
                color="black",
                s=14,
                label="Experimental",
                zorder=3,
            )
            plotted_lines.append(experimental_artist)
        if (
            self.show_solvent_trace_checkbox.isChecked()
            and plot_data.solvent_contribution is not None
        ):
            solvent_values = np.asarray(
                plot_data.solvent_contribution,
                dtype=float,
            )
            solvent_mask = np.isfinite(solvent_values)
            if self.model_log_y_checkbox.isChecked():
                solvent_mask &= solvent_values > 0.0
            if np.any(solvent_mask):
                (solvent_line,) = top_axis.plot(
                    np.asarray(plot_data.q_values, dtype=float)[solvent_mask],
                    solvent_values[solvent_mask],
                    color="green",
                    linewidth=1.5,
                    label="Solvent contribution",
                )
                plotted_lines.append(solvent_line)
        if (
            self.show_structure_factor_trace_checkbox.isChecked()
            and plot_data.structure_factor_trace is not None
        ):
            structure_values = np.asarray(
                plot_data.structure_factor_trace,
                dtype=float,
            )
            structure_mask = np.isfinite(structure_values)
            if np.any(structure_mask):
                structure_axis = top_axis.twinx()
                structure_axis.set_xscale(
                    "log"
                    if self.model_log_x_checkbox.isChecked()
                    else "linear"
                )
                (structure_line,) = structure_axis.plot(
                    np.asarray(plot_data.q_values, dtype=float)[
                        structure_mask
                    ],
                    structure_values[structure_mask],
                    color="tab:purple",
                    linestyle="--",
                    linewidth=1.5,
                    label="Structure factor S(q)",
                )
                structure_axis.set_ylabel("S(q)", color="tab:purple")
                structure_axis.tick_params(axis="y", colors="tab:purple")
                structure_axis.spines["right"].set_color("tab:purple")
                plotted_lines.append(structure_line)
        if self.show_model_trace_checkbox.isChecked():
            (model_line,) = top_axis.plot(
                plot_data.q_values,
                plot_data.model_intensities,
                color="tab:red",
                linewidth=2,
                label=f"Model ({plot_data.bestfit_method})",
            )
            plotted_lines.append(model_line)
        top_axis.set_xscale(
            "log" if self.model_log_x_checkbox.isChecked() else "linear"
        )
        top_axis.set_yscale(
            "log" if self.model_log_y_checkbox.isChecked() else "linear"
        )
        top_axis.set_ylabel("Intensity (arb. units)")
        top_axis.set_title(f"DREAM refinement: {plot_data.template_name}")
        metric_lines = [
            f"RMSE: {plot_data.rmse:.4g}",
            f"Mean |res|: {plot_data.mean_abs_residual:.4g}",
            f"R²: {plot_data.r_squared:.4g}",
        ]
        top_axis.text(
            0.02,
            0.02,
            "\n".join(metric_lines),
            transform=top_axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "white",
                "edgecolor": "0.6",
                "alpha": 0.85,
            },
        )
        residuals = np.asarray(
            plot_data.model_intensities - plot_data.experimental_intensities,
            dtype=float,
        )
        bottom_axis.axhline(0.0, color="0.5", linewidth=1.0)
        bottom_axis.plot(
            plot_data.q_values,
            residuals,
            color="tab:blue",
        )
        bottom_axis.set_xscale(
            "log" if self.model_log_x_checkbox.isChecked() else "linear"
        )
        bottom_axis.set_xlabel("q (Å⁻¹)")
        bottom_axis.set_ylabel("Residual")
        if plotted_lines:
            self._build_interactive_model_legend(top_axis, plotted_lines)
        self.model_figure.tight_layout()
        self.model_canvas.draw_idle()

    def _redraw_current_model_plot(self) -> None:
        self.plot_model_fit(self._current_model_plot_data)

    def _update_model_trace_toggle_state(
        self,
        plot_data: DreamModelPlotData | None,
    ) -> None:
        has_plot_data = plot_data is not None
        has_solvent = (
            has_plot_data and plot_data.solvent_contribution is not None
        )
        has_structure_factor = (
            has_plot_data and plot_data.structure_factor_trace is not None
        )
        self.show_experimental_trace_checkbox.setEnabled(has_plot_data)
        self.show_model_trace_checkbox.setEnabled(has_plot_data)
        self.show_solvent_trace_checkbox.setEnabled(bool(has_solvent))
        self.show_structure_factor_trace_checkbox.setEnabled(
            bool(has_structure_factor)
        )

    def _build_interactive_model_legend(
        self, axis, lines: list[object]
    ) -> None:
        legend = axis.legend(handles=lines, loc="best")
        if legend is None:
            return
        legend_handles = getattr(legend, "legend_handles", None)
        if legend_handles is None:
            legend_handles = getattr(legend, "legendHandles", [])
        for legend_handle, original_line in zip(legend_handles, lines):
            if hasattr(legend_handle, "set_picker"):
                legend_handle.set_picker(True)
                legend_handle.set_pickradius(6)
            legend_handle.set_alpha(
                1.0 if original_line.get_visible() else 0.25
            )
            self._model_legend_line_map[legend_handle] = original_line
            label = str(original_line.get_label()).strip()
            if label:
                self._model_legend_handle_lookup[label] = legend_handle

    def _handle_model_legend_pick(self, event) -> None:
        original_line = self._model_legend_line_map.get(event.artist)
        if original_line is None:
            return
        is_visible = not original_line.get_visible()
        original_line.set_visible(is_visible)
        if hasattr(event.artist, "set_alpha"):
            event.artist.set_alpha(1.0 if is_visible else 0.25)
        for axis in self.model_figure.axes:
            try:
                axis.relim(visible_only=True)
                axis.autoscale_view()
            except Exception:
                continue
        self.model_canvas.draw_idle()

    def plot_violin_plot(
        self,
        summary: DreamSummary | None,
        violin_data: DreamViolinPlotData | None,
    ) -> None:
        self._current_summary = summary
        self._current_violin_plot_data = violin_data
        self.violin_figure.clear()
        axis = self.violin_figure.add_subplot(111)
        if (
            summary is None
            or violin_data is None
            or not violin_data.parameter_names
            or violin_data.samples.size == 0
        ):
            axis.text(
                0.5,
                0.5,
                "No posterior violin data are available yet.",
                ha="center",
                va="center",
            )
            axis.set_axis_off()
            self.violin_canvas.draw_idle()
            return

        payload = self.prepare_violin_plot_payload(summary, violin_data)
        display_samples = self._display_violin_samples(payload["samples"])
        positions = np.arange(1, len(payload["display_names"]) + 1)
        violin_parts = axis.violinplot(
            display_samples,
            positions=positions,
            showmedians=True,
        )
        colors = self._violin_body_colors(len(payload["display_names"]))
        outline_color = self.selected_violin_outline_color()
        outline_width = float(self.violin_outline_width_spin.value())
        for body, color in zip(violin_parts["bodies"], colors, strict=True):
            red, green, blue, _alpha = color
            body.set_facecolor((red, green, blue, 0.62))
            body.set_edgecolor(outline_color)
            body.set_linewidth(outline_width)
        interval_color = self.selected_violin_interval_color()
        median_color = self.selected_violin_median_color()
        for key in ("cbars", "cmins", "cmaxes"):
            artist = violin_parts.get(key)
            if artist is not None:
                artist.set_color(interval_color)
                artist.set_linewidth(1.2)
        median_artist = violin_parts.get("cmedians")
        if median_artist is not None:
            median_artist.set_color(median_color)
            median_artist.set_linewidth(1.4)
        axis.vlines(
            positions,
            payload["interval_low_values"],
            payload["interval_high_values"],
            color=interval_color,
            linewidth=1.8,
            label=(
                f"p{summary.credible_interval_low:g}-"
                f"p{summary.credible_interval_high:g} interval"
            ),
        )
        axis.scatter(
            positions,
            payload["selected_values"],
            color=self.selected_violin_point_color(),
            s=18,
            zorder=3,
            label=f"Selected {summary.bestfit_method}",
        )
        axis.set_xticks(positions)
        axis.set_xticklabels(
            payload["display_names"],
            rotation=45,
            ha="right",
        )
        axis.set_ylabel(str(payload["ylabel"]))
        axis.set_title(str(payload["title"]))
        if payload["y_limits"] is not None:
            axis.set_ylim(*payload["y_limits"])
        axis.legend(loc="best")
        self.violin_figure.tight_layout()
        self.violin_canvas.draw_idle()

    def redraw_current_violin_plot(self) -> None:
        self.plot_violin_plot(
            self._current_summary,
            self._current_violin_plot_data,
        )

    def current_summary(self) -> DreamSummary | None:
        return self._current_summary

    def current_violin_plot_data(self) -> DreamViolinPlotData | None:
        return self._current_violin_plot_data

    def prepare_violin_plot_payload(
        self,
        summary: DreamSummary,
        violin_data: DreamViolinPlotData,
    ) -> dict[str, object]:
        samples = np.asarray(violin_data.samples, dtype=float)
        if samples.ndim == 1:
            samples = samples.reshape(1, -1)
        summary_lookup = {
            name: index
            for index, name in enumerate(summary.full_parameter_names)
        }
        selected_values = np.asarray(
            [
                summary.bestfit_params[summary_lookup[name]]
                for name in violin_data.parameter_names
            ],
            dtype=float,
        )
        interval_low_values = np.asarray(
            [
                summary.interval_low_values[summary_lookup[name]]
                for name in violin_data.parameter_names
            ],
            dtype=float,
        )
        interval_high_values = np.asarray(
            [
                summary.interval_high_values[summary_lookup[name]]
                for name in violin_data.parameter_names
            ],
            dtype=float,
        )

        value_scale_mode = self.selected_violin_value_scale_mode()
        ylabel = "Parameter value"
        title = "Posterior parameter distributions"
        y_limits: tuple[float, float] | None = None
        if value_scale_mode == "weights_unit_interval":
            ylabel = "Weight fraction"
            title = "Posterior weight distributions"
            y_limits = (0.0, 1.0)
        elif value_scale_mode == "normalized_all":
            (
                samples,
                selected_values,
                interval_low_values,
                interval_high_values,
            ) = self._normalize_violin_series(
                samples,
                selected_values,
                interval_low_values,
                interval_high_values,
            )
            ylabel = "Normalized parameter value"
            title = "Posterior parameter distributions (normalized)"
            y_limits = (0.0, 1.0)

        return {
            "display_names": list(violin_data.display_names),
            "samples": samples,
            "selected_values": selected_values,
            "interval_low_values": interval_low_values,
            "interval_high_values": interval_high_values,
            "ylabel": ylabel,
            "title": title,
            "y_limits": y_limits,
        }

    def _violin_body_colors(self, count: int) -> list[tuple[float, ...]]:
        palette = self.selected_violin_palette()
        if palette == "custom_solid":
            color = QColor(self.selected_violin_custom_color()).name()
            return [
                tuple(QColor(color).getRgbF()) for _ in range(max(count, 1))
            ]
        cmap = colormaps.get_cmap(palette)
        if count <= 1:
            return [tuple(cmap(0.72))]
        positions = np.linspace(0.35, 0.9, count)
        return [tuple(cmap(position)) for position in positions]

    @staticmethod
    def _normalize_violin_series(
        samples: np.ndarray,
        selected_values: np.ndarray,
        interval_low_values: np.ndarray,
        interval_high_values: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        mins = np.min(samples, axis=0)
        maxs = np.max(samples, axis=0)
        spans = maxs - mins
        normalized_samples = np.empty_like(samples, dtype=float)
        normalized_selected = np.empty_like(selected_values, dtype=float)
        normalized_low = np.empty_like(interval_low_values, dtype=float)
        normalized_high = np.empty_like(interval_high_values, dtype=float)
        varying = spans > 0
        if np.any(varying):
            normalized_samples[:, varying] = (
                samples[:, varying] - mins[varying]
            ) / spans[varying]
            normalized_selected[varying] = (
                selected_values[varying] - mins[varying]
            ) / spans[varying]
            normalized_low[varying] = (
                interval_low_values[varying] - mins[varying]
            ) / spans[varying]
            normalized_high[varying] = (
                interval_high_values[varying] - mins[varying]
            ) / spans[varying]
        if np.any(~varying):
            normalized_samples[:, ~varying] = 0.5
            normalized_selected[~varying] = 0.5
            normalized_low[~varying] = 0.5
            normalized_high[~varying] = 0.5
        return (
            np.clip(normalized_samples, 0.0, 1.0),
            np.clip(normalized_selected, 0.0, 1.0),
            np.clip(normalized_low, 0.0, 1.0),
            np.clip(normalized_high, 0.0, 1.0),
        )

    def blink_edit_priors_button(self) -> None:
        self._start_button_blink(self.edit_button)

    def blink_write_bundle_button(self) -> None:
        self._start_button_blink(self.write_button)

    def _start_button_blink(self, button: QPushButton) -> None:
        self._blink_timer.stop()
        self._blink_remaining = 8
        if self._blink_target_button is not None:
            self._set_button_blink_highlight(
                self._blink_target_button,
                enabled=False,
            )
        self._blink_target_button = button
        self._set_button_blink_highlight(
            self._blink_target_button,
            enabled=True,
        )
        self._blink_timer.start()

    def _advance_button_blink(self) -> None:
        if self._blink_target_button is None:
            self._blink_timer.stop()
            return
        if self._blink_remaining <= 0:
            self._blink_timer.stop()
            self._set_button_blink_highlight(
                self._blink_target_button,
                enabled=False,
            )
            self._blink_target_button = None
            return
        try:
            self._set_button_blink_highlight(
                self._blink_target_button,
                enabled=bool(self._blink_remaining % 2),
            )
        except RuntimeError:
            self._blink_timer.stop()
            self._blink_target_button = None
            return
        self._blink_remaining -= 1

    def _set_button_blink_highlight(
        self,
        button: QPushButton,
        *,
        enabled: bool,
    ) -> None:
        if enabled:
            effect = button.graphicsEffect()
            if not isinstance(effect, QGraphicsColorizeEffect):
                effect = QGraphicsColorizeEffect(button)
                effect.setColor(QColor("#f6d365"))
                effect.setStrength(0.75)
                button.setGraphicsEffect(effect)
            return
        if button.graphicsEffect() is not None:
            button.setGraphicsEffect(None)

    def _render_output(self, *, scroll_to_end: bool = False) -> None:
        del scroll_to_end
        sections: list[str] = []
        if self._summary_text:
            sections.append("DREAM Summary\n" + self._summary_text)
        history_parts = [
            part
            for part in [self._base_log_text, *self._history_messages]
            if part
        ]
        if history_parts:
            sections.append("DREAM Console\n" + "\n\n".join(history_parts))
        scrollbar = self.output_box.verticalScrollBar()
        previous_value = scrollbar.value()
        previous_maximum = max(scrollbar.maximum(), 1)
        self.output_box.setPlainText("\n\n".join(sections).strip())
        if self._console_autoscroll_enabled:
            self._scroll_output_to_end()
            return
        updated_scrollbar = self.output_box.verticalScrollBar()
        if updated_scrollbar.maximum() > 0:
            position_fraction = previous_value / previous_maximum
            updated_scrollbar.setValue(
                int(round(position_fraction * updated_scrollbar.maximum()))
            )

    def _scroll_output_to_end(self) -> None:
        cursor = self.output_box.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.output_box.setTextCursor(cursor)
        self.output_box.ensureCursorVisible()
        scrollbar = self.output_box.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        QTimer.singleShot(
            0,
            self._scroll_output_to_end_once,
        )

    def _scroll_output_to_end_once(self) -> None:
        cursor = self.output_box.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.output_box.setTextCursor(cursor)
        self.output_box.ensureCursorVisible()
        scrollbar = self.output_box.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _browse_history_file(self) -> None:
        selected, _filter = QFileDialog.getOpenFileName(
            self,
            "Select DREAM chain history file",
            self.history_file_edit.text().strip(),
            "NumPy arrays (*.npy);;All files (*)",
        )
        if selected:
            self.history_file_edit.setText(selected)

    def _on_violin_palette_changed(self, _index: int) -> None:
        self._update_violin_style_controls()
        self._notify_violin_style_settings_changed()

    def _update_violin_style_controls(self) -> None:
        self.violin_custom_color_button.setEnabled(True)

    def _choose_violin_custom_color(self) -> None:
        self._set_combo_data(self.violin_palette_combo, "custom_solid")
        self._update_violin_style_controls()
        self._choose_plot_color(
            self.violin_custom_color_button,
            title="Choose custom violin color",
            label="Custom violin",
            default="#4c72b0",
        )

    def _choose_violin_point_color(self) -> None:
        self._choose_plot_color(
            self.violin_point_color_button,
            title="Choose point color",
            label="Point",
            default="tab:red",
        )

    def _choose_interval_color(self) -> None:
        self._choose_plot_color(
            self.interval_color_button,
            title="Choose interval color",
            label="Interval",
            default="#8c8c8c",
        )

    def _choose_median_color(self) -> None:
        self._choose_plot_color(
            self.median_color_button,
            title="Choose median color",
            label="Median",
            default="#4d4d4d",
        )

    def _choose_outline_color(self) -> None:
        self._choose_plot_color(
            self.violin_outline_color_button,
            title="Choose violin outline color",
            label="Outline",
            default="#000000",
        )

    def _choose_plot_color(
        self,
        button: QPushButton,
        *,
        title: str,
        label: str,
        default: str,
    ) -> None:
        current_color = self._plot_color_button_value(
            button,
            default=default,
        )
        chosen = QColorDialog.getColor(
            QColor(current_color),
            self,
            title,
        )
        if not chosen.isValid():
            return
        self._configure_plot_color_button(
            button,
            chosen.name(),
            label=label,
        )
        self._notify_violin_style_settings_changed()

    def _on_posterior_filter_mode_changed(self, _index: int) -> None:
        self._mark_search_filter_preset_custom()
        self._update_posterior_filter_controls()
        self._notify_results_settings_changed()

    def _on_search_filter_preset_changed(self, _index: int) -> None:
        if self._applying_search_filter_preset:
            return
        preset_name = self.selected_search_filter_preset()
        if preset_name == "custom":
            return
        self._apply_search_filter_preset(preset_name)

    def _apply_search_filter_preset(self, preset_name: str) -> None:
        preset = DREAM_SEARCH_FILTER_PRESETS.get(preset_name)
        if preset is None:
            return
        self._applying_search_filter_preset = True
        self._suspend_visualization_notifications = True
        try:
            self.chains_spin.setValue(int(preset["nchains"]))
            self.iterations_spin.setValue(int(preset["niterations"]))
            self.burnin_spin.setValue(int(preset["burnin_percent"]))
            self.nseedchains_spin.setValue(int(preset["nseedchains"]))
            self.crossover_burnin_spin.setValue(
                int(preset["crossover_burnin"])
            )
            self._set_combo_data(
                self.posterior_filter_combo,
                str(preset["posterior_filter_mode"]),
            )
            self.posterior_top_percent_spin.setValue(
                float(preset["posterior_top_percent"])
            )
            self.posterior_top_n_spin.setValue(int(preset["posterior_top_n"]))
            self._set_combo_data(
                self.violin_sample_source_combo,
                str(preset["violin_sample_source"]),
            )
            self._set_combo_data(self.search_filter_preset_combo, preset_name)
            self._update_posterior_filter_controls()
        finally:
            self._applying_search_filter_preset = False
            self._suspend_visualization_notifications = False
        self._notify_results_settings_changed()

    def _mark_search_filter_preset_custom(self) -> None:
        if self._applying_search_filter_preset:
            return
        self._applying_search_filter_preset = True
        try:
            self._set_combo_data(self.search_filter_preset_combo, "custom")
        finally:
            self._applying_search_filter_preset = False

    def _update_posterior_filter_controls(self) -> None:
        self.posterior_top_percent_spin.setEnabled(True)
        self.posterior_top_n_spin.setEnabled(True)

    def _notify_results_settings_changed(self) -> None:
        if self._suspend_visualization_notifications:
            return
        self.results_settings_changed.emit()
        self.visualization_settings_changed.emit()

    def _notify_summary_settings_changed(self) -> None:
        if self._suspend_visualization_notifications:
            return
        self.summary_settings_changed.emit()
        self.visualization_settings_changed.emit()

    def _notify_violin_data_settings_changed(self) -> None:
        if self._suspend_visualization_notifications:
            return
        self.violin_data_settings_changed.emit()
        self.visualization_settings_changed.emit()

    def _notify_violin_style_settings_changed(self) -> None:
        if self._suspend_visualization_notifications:
            return
        self.violin_style_settings_changed.emit()
        self.visualization_settings_changed.emit()

    def _on_bestfit_method_changed(self, _index: int) -> None:
        self._notify_results_settings_changed()

    def _on_violin_mode_changed(self, _index: int) -> None:
        self._notify_violin_data_settings_changed()

    def _on_violin_sample_source_changed(self, _index: int) -> None:
        self._notify_violin_data_settings_changed()

    def _on_weight_order_changed(self, _index: int) -> None:
        self._notify_violin_data_settings_changed()

    def _on_violin_value_scale_changed(self, _index: int) -> None:
        self._notify_violin_data_settings_changed()

    def _on_violin_outline_width_changed(self, _value: float) -> None:
        self._notify_violin_style_settings_changed()

    def _on_posterior_top_percent_changed(self, _value: float) -> None:
        self._notify_results_settings_changed()

    def _on_posterior_top_n_changed(self, _value: int) -> None:
        self._notify_results_settings_changed()

    def _on_credible_interval_low_changed(self, _value: float) -> None:
        self._notify_summary_settings_changed()

    def _on_credible_interval_high_changed(self, _value: float) -> None:
        self._notify_summary_settings_changed()

    @classmethod
    def _display_violin_samples(cls, samples: object) -> np.ndarray:
        display_samples = np.asarray(samples, dtype=float)
        if display_samples.ndim == 1:
            display_samples = display_samples.reshape(-1, 1)
        max_samples = max(int(cls.MAX_VIOLIN_PLOT_SAMPLES), 1)
        if display_samples.shape[0] <= max_samples:
            return display_samples
        sample_indices = np.linspace(
            0,
            display_samples.shape[0] - 1,
            max_samples,
            dtype=int,
        )
        return display_samples[sample_indices]

    @staticmethod
    def _build_action_group(
        title: str,
        buttons: list[QPushButton],
    ) -> QGroupBox:
        group = QGroupBox(title)
        layout = QGridLayout(group)
        for index, button in enumerate(buttons):
            layout.addWidget(button, index // 3, index % 3)
        return group

    @staticmethod
    def _set_widget_tooltip(
        label: QLabel,
        widget: QWidget,
        tooltip: str,
    ) -> None:
        label.setToolTip(tooltip)
        widget.setToolTip(tooltip)

    @staticmethod
    def _set_combo_data(combo: QComboBox, value: str) -> bool:
        for index in range(combo.count()):
            if str(combo.itemData(index) or "") == value:
                combo.setCurrentIndex(index)
                return True
        return False

    @staticmethod
    def _configure_plot_color_button(
        button: QPushButton,
        color: str,
        *,
        label: str,
    ) -> None:
        normalized = DreamTab._normalize_plot_color(color)
        qcolor = QColor(normalized)
        foreground = "#ffffff"
        if qcolor.isValid() and qcolor.lightness() > 128:
            foreground = "#000000"
        button.setText(normalized)
        button.setToolTip(f"{label} color: {normalized}")
        button.setMinimumWidth(88)
        button.setStyleSheet(
            "QPushButton {"
            f"background-color: {normalized};"
            f"color: {foreground};"
            "border: 1px solid #666666;"
            "padding: 2px 8px;"
            "}"
        )

    @staticmethod
    def _plot_color_button_value(
        button: QPushButton,
        *,
        default: str,
    ) -> str:
        text = button.text().strip()
        return DreamTab._normalize_plot_color(text or default)

    @staticmethod
    def _normalize_plot_color(color: str) -> str:
        normalized = str(color).strip() or "#000000"
        qcolor = QColor(normalized)
        if qcolor.isValid():
            return qcolor.name()
        try:
            return str(to_hex(normalized))
        except ValueError:
            return "#000000"
