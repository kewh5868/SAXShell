from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg,
    NavigationToolbar2QT,
)
from matplotlib.figure import Figure
from PySide6.QtCore import QTimer, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
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
    QSpinBox,
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


class DreamTab(QWidget):
    ACTIVE_SETTINGS_LABEL = "Active project settings"

    edit_parameter_map_requested = Signal()
    save_settings_requested = Signal()
    write_runtime_requested = Signal()
    preview_runtime_requested = Signal()
    run_dream_requested = Signal()
    load_results_requested = Signal()
    save_report_requested = Signal()
    settings_preset_changed = Signal(str)
    visualization_settings_changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._summary_text = ""
        self._base_log_text = ""
        self._history_messages: list[str] = []
        self._blink_timer = QTimer(self)
        self._blink_timer.setInterval(180)
        self._blink_timer.timeout.connect(self._advance_button_blink)
        self._blink_remaining = 0
        self._blink_target_button: QPushButton | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(12)

        top_row = QHBoxLayout()
        top_row.setSpacing(12)
        top_row.addWidget(self._build_settings_scroll_area(), stretch=4)
        top_row.addWidget(self._build_plot_panel(), stretch=7)
        root.addLayout(top_row, stretch=1)
        root.addWidget(self._build_output_group())

    def _build_settings_scroll_area(self) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(12)
        content_layout.addWidget(self._build_settings_group())
        content_layout.addWidget(self._build_parameter_map_group())
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
        self.iterations_spin = QSpinBox()
        self.iterations_spin.setRange(100, 10_000_000)
        self.iterations_spin.setSingleStep(100)
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
        self.crossover_burnin_spin = QSpinBox()
        self.crossover_burnin_spin.setRange(0, 10_000_000)
        self.crossover_burnin_spin.setSingleStep(100)
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
        self.zeta_spin = QDoubleSpinBox()
        self.zeta_spin.setRange(0.0, 1.0)
        self.zeta_spin.setDecimals(12)
        self.zeta_spin.setSingleStep(0.000001)
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
        self.parallel_checkbox = QCheckBox("Run chains in parallel")
        self.adapt_checkbox = QCheckBox("Adapt crossover")
        self.restart_checkbox = QCheckBox("Restart previous run")
        self.verbose_checkbox.setToolTip(
            "Print DREAM sampler progress and status updates into the run output."
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
        layout.addWidget(self.adapt_checkbox, row, 0, 1, 2)
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
            lambda _index: self.visualization_settings_changed.emit()
        )
        self.violin_mode_combo = QComboBox()
        self.violin_mode_combo.addItem(
            "Varying Parameters", "varying_parameters"
        )
        self.violin_mode_combo.addItem("All Parameters", "all_parameters")
        self.violin_mode_combo.addItem("Weights Only", "weights_only")
        self.violin_mode_combo.addItem("Fit Parameters", "fit_parameters")
        self.violin_mode_combo.currentIndexChanged.connect(
            lambda _index: self.visualization_settings_changed.emit()
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
            "violin plot."
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
        self.load_button = QPushButton("Load Results")
        self.load_button.setToolTip(
            "Load the most recent DREAM run from the project runtime folder "
            "and redraw the summary plots."
        )
        self.load_button.clicked.connect(self.load_results_requested.emit)
        self.report_button = QPushButton("Save Statistics")
        self.report_button.setToolTip(
            "Save a text report of the currently loaded DREAM posterior summary."
        )
        self.report_button.clicked.connect(self.save_report_requested.emit)

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
        self.analysis_actions_group = self._build_action_group(
            "DREAM Analysis",
            [
                self.load_button,
                self.report_button,
            ],
        )
        action_sections_layout.addWidget(self.setup_actions_group)
        action_sections_layout.addWidget(self.analysis_actions_group)
        layout.addWidget(action_sections, row, 0, 1, 4)
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
            lambda _index: self.visualization_settings_changed.emit()
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
            lambda _value: self.visualization_settings_changed.emit()
        )
        top_percent_tip = (
            "When 'Top % by Log-posterior' is selected, keep this percent "
            "of the highest-log-posterior samples across all chains."
        )
        top_percent_label = QLabel("Top %")
        self._set_widget_tooltip(
            top_percent_label,
            self.posterior_top_percent_spin,
            top_percent_tip,
        )

        self.posterior_top_n_spin = QSpinBox()
        self.posterior_top_n_spin.setRange(1, 10_000_000)
        self.posterior_top_n_spin.setValue(500)
        self.posterior_top_n_spin.valueChanged.connect(
            lambda _value: self.visualization_settings_changed.emit()
        )
        top_n_tip = (
            "When 'Top N by Log-posterior' is selected, keep this many of "
            "the highest-log-posterior samples across all chains."
        )
        top_n_label = QLabel("Top N")
        self._set_widget_tooltip(
            top_n_label,
            self.posterior_top_n_spin,
            top_n_tip,
        )
        layout.addWidget(top_percent_label, 1, 0)
        layout.addWidget(self.posterior_top_percent_spin, 1, 1)
        layout.addWidget(top_n_label, 1, 2)
        layout.addWidget(self.posterior_top_n_spin, 1, 3)

        self.credible_interval_low_spin = QDoubleSpinBox()
        self.credible_interval_low_spin.setRange(0.0, 99.9)
        self.credible_interval_low_spin.setDecimals(1)
        self.credible_interval_low_spin.setSingleStep(1.0)
        self.credible_interval_low_spin.setValue(16.0)
        self.credible_interval_low_spin.valueChanged.connect(
            lambda _value: self.visualization_settings_changed.emit()
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
            lambda _value: self.visualization_settings_changed.emit()
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
        layout.addWidget(interval_low_label, 2, 0)
        layout.addWidget(self.credible_interval_low_spin, 2, 1)
        layout.addWidget(interval_high_label, 2, 2)
        layout.addWidget(self.credible_interval_high_spin, 2, 3)

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
        layout.addWidget(self._build_model_plot_group(), stretch=1)
        layout.addWidget(self._build_violin_plot_group(), stretch=1)
        return panel

    def _build_model_plot_group(self) -> QGroupBox:
        group = QGroupBox("DREAM Model vs Experimental")
        layout = QVBoxLayout(group)
        self.model_figure = Figure(figsize=(8.0, 4.2))
        self.model_canvas = FigureCanvasQTAgg(self.model_figure)
        self.model_toolbar = NavigationToolbar2QT(self.model_canvas, self)
        self.model_canvas.setMinimumHeight(240)
        layout.addWidget(self.model_toolbar)
        layout.addWidget(self.model_canvas)
        return group

    def _build_violin_plot_group(self) -> QGroupBox:
        group = QGroupBox("Posterior Violin Plot")
        layout = QVBoxLayout(group)
        self.violin_figure = Figure(figsize=(8.0, 4.2))
        self.violin_canvas = FigureCanvasQTAgg(self.violin_figure)
        self.violin_toolbar = NavigationToolbar2QT(self.violin_canvas, self)
        self.violin_canvas.setMinimumHeight(240)
        layout.addWidget(self.violin_toolbar)
        layout.addWidget(self.violin_canvas)
        return group

    def _build_output_group(self) -> QGroupBox:
        group = QGroupBox("DREAM Output")
        layout = QVBoxLayout(group)
        self.progress_label = QLabel("Progress: idle")
        layout.addWidget(self.progress_label)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%v / %m runs")
        layout.addWidget(self.progress_bar)
        self.output_box = QTextEdit()
        self.output_box.setReadOnly(True)
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

    def set_settings(
        self,
        settings: DreamRunSettings,
        *,
        preset_name: str | None = None,
    ) -> None:
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
            self.bestfit_method_combo, settings.bestfit_method
        )
        self._set_combo_data(
            self.posterior_filter_combo,
            settings.posterior_filter_mode,
        )
        self.posterior_top_percent_spin.setValue(
            settings.posterior_top_percent
        )
        self.posterior_top_n_spin.setValue(settings.posterior_top_n)
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
        self._update_posterior_filter_controls()

    def settings_payload(self) -> DreamRunSettings:
        return DreamRunSettings(
            nchains=int(self.chains_spin.value()),
            niterations=int(self.iterations_spin.value()),
            burnin_percent=int(self.burnin_spin.value()),
            restart=bool(self.restart_checkbox.isChecked()),
            verbose=bool(self.verbose_checkbox.isChecked()),
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
            bestfit_method=self.selected_bestfit_method(),
            posterior_filter_mode=self.selected_posterior_filter_mode(),
            posterior_top_percent=float(
                self.posterior_top_percent_spin.value()
            ),
            posterior_top_n=int(self.posterior_top_n_spin.value()),
            credible_interval_low=float(
                self.credible_interval_low_spin.value()
            ),
            credible_interval_high=float(
                self.credible_interval_high_spin.value()
            ),
            violin_parameter_mode=self.selected_violin_mode(),
            violin_sample_source=self.selected_violin_sample_source(),
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

    def append_log(self, message: str) -> None:
        stripped = message.strip()
        if stripped:
            self._history_messages.append(stripped)
        self._render_output(scroll_to_end=True)

    def set_log_text(self, text: str) -> None:
        self._base_log_text = text.strip()
        self._history_messages = []
        self._render_output()

    def set_summary_text(self, text: str) -> None:
        self._summary_text = text.strip()
        self._render_output()

    def clear_plots(self) -> None:
        self.plot_model_fit(None)
        self.plot_violin_plot(None, None)

    def start_progress(self, message: str) -> None:
        self.progress_label.setText(message)
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("")

    def finish_progress(self, message: str) -> None:
        self.progress_label.setText(message)
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(1)
        self.progress_bar.setFormat("%v / %m runs")

    def reset_progress(self) -> None:
        self.progress_label.setText("Progress: idle")
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%v / %m runs")

    def plot_model_fit(self, plot_data: DreamModelPlotData | None) -> None:
        self.model_figure.clear()
        axis = self.model_figure.add_subplot(111)
        if plot_data is None:
            axis.text(
                0.5,
                0.5,
                "No DREAM results have been loaded yet.",
                ha="center",
                va="center",
            )
            axis.set_axis_off()
            self.model_canvas.draw()
            return

        axis.scatter(
            plot_data.q_values,
            plot_data.experimental_intensities,
            color="black",
            s=14,
            label="Experimental",
            zorder=3,
        )
        axis.plot(
            plot_data.q_values,
            plot_data.model_intensities,
            color="tab:red",
            linewidth=2,
            label=f"Model ({plot_data.bestfit_method})",
        )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("q (Å⁻¹)")
        axis.set_ylabel("Intensity (arb. units)")
        axis.set_title(f"DREAM refinement: {plot_data.template_name}")
        axis.legend(loc="best")
        self.model_figure.tight_layout()
        self.model_canvas.draw()

    def plot_violin_plot(
        self,
        summary: DreamSummary | None,
        violin_data: DreamViolinPlotData | None,
    ) -> None:
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
            self.violin_canvas.draw()
            return

        positions = np.arange(1, len(violin_data.parameter_names) + 1)
        samples = np.asarray(violin_data.samples, dtype=float)
        axis.violinplot(samples, positions=positions, showmedians=True)

        summary_lookup = {
            name: index
            for index, name in enumerate(summary.full_parameter_names)
        }
        selected_values = []
        interval_low_values = []
        interval_high_values = []
        for name in violin_data.parameter_names:
            index = summary_lookup[name]
            selected_values.append(summary.bestfit_params[index])
            interval_low_values.append(summary.interval_low_values[index])
            interval_high_values.append(summary.interval_high_values[index])
        axis.vlines(
            positions,
            interval_low_values,
            interval_high_values,
            color="0.55",
            linewidth=1.8,
            label=(
                f"p{summary.credible_interval_low:g}-"
                f"p{summary.credible_interval_high:g} interval"
            ),
        )
        axis.scatter(
            positions,
            selected_values,
            color="tab:red",
            s=18,
            zorder=3,
            label=f"Selected {summary.bestfit_method}",
        )
        axis.set_xticks(positions)
        axis.set_xticklabels(
            violin_data.parameter_names,
            rotation=45,
            ha="right",
        )
        axis.set_ylabel("Parameter value")
        axis.set_title("Posterior parameter distributions")
        axis.legend(loc="best")
        self.violin_figure.tight_layout()
        self.violin_canvas.draw()

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
        self.output_box.setPlainText("\n\n".join(sections).strip())
        if scroll_to_end:
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

    def _on_posterior_filter_mode_changed(self, _index: int) -> None:
        self._update_posterior_filter_controls()
        self.visualization_settings_changed.emit()

    def _update_posterior_filter_controls(self) -> None:
        mode = self.selected_posterior_filter_mode()
        self.posterior_top_percent_spin.setEnabled(mode == "top_percent_logp")
        self.posterior_top_n_spin.setEnabled(mode == "top_n_logp")

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
    def _set_combo_data(combo: QComboBox, value: str) -> None:
        for index in range(combo.count()):
            if str(combo.itemData(index) or "") == value:
                combo.setCurrentIndex(index)
                return
