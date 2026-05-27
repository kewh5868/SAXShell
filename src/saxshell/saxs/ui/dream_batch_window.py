from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg,
    NavigationToolbar2QT,
)
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from saxshell.plotting import Q_A_INVERSE_LABEL
from saxshell.saxs.dream import (
    DEFAULT_DREAM_BATCH_CONDA_ENV,
    DREAM_SAMPLER_SETTING_NAMES,
    DREAM_SEARCH_FILTER_PRESET_LABELS,
    DREAM_SEARCH_FILTER_PRESETS,
    DreamBatchFilterSet,
    DreamBatchRunSetManager,
    DreamParameterEntry,
    DreamRunSettings,
    PosteriorFilterSettings,
    command_text_for_run_set,
    dream_run_settings_to_dict,
    load_dream_settings,
)
from saxshell.saxs.dream.batch import (
    load_parameter_map_for_queue_item,
    queue_item_weight_state_summary,
    queue_item_weight_states,
)
from saxshell.saxs.dream.distributions import (
    build_default_parameter_map_from_prefit_entries,
)
from saxshell.saxs.prefit import PrefitEvaluation, PrefitParameterEntry
from saxshell.saxs.ui.branding import (
    configure_saxshell_application,
    load_saxshell_icon,
    prepare_saxshell_application_identity,
)
from saxshell.saxs.ui.distribution_window import (
    LEGACY_MD_WEIGHT_SMART_PRIOR_MODE,
    SMART_PRIOR_SPREAD_FACTORS,
    DistributionSetupWindow,
    distribution_guide_bounds,
    format_distribution_guide_value,
)

DREAM_BATCH_QUEUE_PRIOR_MODES = ("strict", "proportional", "lenient")
DREAM_BATCH_FILTER_PRESET_DIR_NAME = "backend_filter_presets"
DREAM_BATCH_FILTER_LIST_PRESET_DIR_NAME = "backend_filter_list_presets"

DREAM_BATCH_QUEUE_PRESETS: tuple[dict[str, object], ...] = (
    {
        "id": "scale-offset-toggle_fixed-solv_w",
        "label": "Scale/Offset Toggle, Fixed solv_w",
        "description": (
            "Monosq sweep: solv_w fixed off, weights/radii/vol_frac on, "
            "scale+offset toggled on/off, strict/proportional/lenient priors "
            "plus rounded guide and legacy MD weight priors, all using the "
            "more aggressive DREAM search settings. Creates 10 queue items."
        ),
        "requires_template_contains": "monosq",
        "search_preset": "more_aggressive",
        "scale_offset_modes": ("on", "off"),
        "prior_modes": (
            "strict",
            "proportional",
            "lenient",
            "rounded_guides",
            LEGACY_MD_WEIGHT_SMART_PRIOR_MODE,
        ),
        "vary_flags": {
            "solv_w": False,
            "weights": True,
            "effective_radius": True,
            "vol_frac": True,
        },
    },
    {
        "id": "scale-offset-on_fixed-solv_w",
        "label": "Scale/Offset On, Fixed solv_w",
        "description": (
            "Keeps solv_w fixed while refining weights, effective radius, "
            "vol_frac, scale, and offset across strict/proportional/lenient "
            "prior widths. Creates 3 queue items."
        ),
        "search_preset": "more_aggressive",
        "scale_offset_modes": ("on",),
        "prior_modes": DREAM_BATCH_QUEUE_PRIOR_MODES,
        "vary_flags": {
            "solv_w": False,
            "weights": True,
            "effective_radius": True,
            "vol_frac": True,
        },
    },
    {
        "id": "fixed-scale-offset_fixed-solv_w",
        "label": "Fixed Scale/Offset and solv_w",
        "description": (
            "Tests the prefit normalization directly: solv_w, scale, and "
            "offset fixed while weights, effective radius, and vol_frac vary "
            "across strict/proportional/lenient priors. Creates 3 queue items."
        ),
        "search_preset": "medium",
        "scale_offset_modes": ("off",),
        "prior_modes": DREAM_BATCH_QUEUE_PRIOR_MODES,
        "vary_flags": {
            "solv_w": False,
            "weights": True,
            "effective_radius": True,
            "vol_frac": True,
        },
    },
    {
        "id": "weights-radius-focus",
        "label": "Weights + Radius Focus",
        "description": (
            "Focused structure-distribution sweep: weights and effective "
            "radius vary while solv_w, scale, offset, and vol_frac remain "
            "fixed. Useful when the prefit background is already trusted. "
            "Creates 3 queue items."
        ),
        "search_preset": "medium",
        "scale_offset_modes": ("off",),
        "prior_modes": DREAM_BATCH_QUEUE_PRIOR_MODES,
        "vary_flags": {
            "solv_w": False,
            "weights": True,
            "effective_radius": True,
            "vol_frac": False,
        },
    },
    {
        "id": "normalization-flex-sweep",
        "label": "Normalization Flex Sweep",
        "description": (
            "Explores normalization freedom: scale and offset vary, solv_w "
            "and vol_frac vary when present, weights and radius vary, and "
            "strict/proportional/lenient priors are paired with less, medium, "
            "and more aggressive DREAM settings. Creates 9 queue items."
        ),
        "search_presets": ("less_aggressive", "medium", "more_aggressive"),
        "scale_offset_modes": ("on",),
        "prior_modes": DREAM_BATCH_QUEUE_PRIOR_MODES,
        "vary_flags": {
            "solv_w": True,
            "weights": True,
            "effective_radius": True,
            "vol_frac": True,
        },
    },
)


class _WheelGuardedDoubleSpinBox(QDoubleSpinBox):
    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt override
        event.ignore()


class _WheelGuardedSpinBox(QSpinBox):
    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt override
        event.ignore()


class DreamBatchRunFileWindow(QMainWindow):
    QUEUE_TABLE_HEADERS = (
        "Label",
        "Status",
        "Fit q min",
        "Fit q max",
        "Weights",
        "Run directory",
        "Created",
    )
    QUEUE_COL_LABEL = 0
    QUEUE_COL_STATUS = 1
    QUEUE_COL_FIT_Q_MIN = 2
    QUEUE_COL_FIT_Q_MAX = 3
    QUEUE_COL_WEIGHTS = 4
    QUEUE_COL_RUN_DIR = 5
    QUEUE_COL_CREATED = 6

    FILTER_TABLE_HEADERS = (
        "Label",
        "Posterior",
        "Top %",
        "Top N",
        "Best-fit",
        "Violin data",
        "Violin samples",
        "Y-axis scale",
        "Stoichiometry",
        "Created",
    )
    FILTER_COL_LABEL = 0
    FILTER_COL_POSTERIOR_MODE = 1
    FILTER_COL_TOP_PERCENT = 2
    FILTER_COL_TOP_N = 3
    FILTER_COL_BESTFIT = 4
    FILTER_COL_VIOLIN_MODE = 5
    FILTER_COL_VIOLIN_SAMPLE_SOURCE = 6
    FILTER_COL_VIOLIN_VALUE_SCALE = 7
    FILTER_COL_STOICHIOMETRY = 8
    FILTER_COL_CREATED = 9

    FILTER_POSTERIOR_MODE_OPTIONS = (
        ("All Post-burnin Samples", "all_post_burnin"),
        ("Top % by Log-posterior", "top_percent_logp"),
        ("Top N by Log-posterior", "top_n_logp"),
    )
    FILTER_BESTFIT_OPTIONS = (
        ("MAP", "map"),
        ("Chain Mean MAP", "chain_mean"),
        ("Median", "median"),
    )
    FILTER_VIOLIN_MODE_OPTIONS = (
        ("Varying Parameters", "varying_parameters"),
        ("All Parameters", "all_parameters"),
        ("Weights Only", "weights_only"),
        ("Fit Parameters", "fit_parameters"),
        ("Effective Radii Only", "effective_radii_only"),
        ("Additional Parameters Only", "additional_parameters_only"),
        ("Selected Additional Parameter", "selected_additional_parameter"),
    )
    FILTER_VIOLIN_SAMPLE_SOURCE_OPTIONS = (
        ("Filtered Posterior", "filtered_posterior"),
        ("MAP Chain Only", "map_chain_only"),
    )
    FILTER_VIOLIN_VALUE_SCALE_OPTIONS = (
        ("Parameter Value", "parameter_value"),
        ("Weights 0-1 Only", "weights_unit_interval"),
        ("Structure Fraction (%)", "structure_fraction_percent"),
        ("Total Atom Fraction (%)", "atom_fraction_percent"),
        ("Normalized 0-1 (All)", "normalized_all"),
        ("Effective Radii Only", "effective_radii_only"),
        ("Additional Parameters Only", "additional_parameters_only"),
    )

    def __init__(
        self,
        *,
        initial_project_dir: str | Path | None = None,
        initial_prefit_parameter_entries: (
            list[PrefitParameterEntry] | None
        ) = None,
        initial_fit_q_range: (
            tuple[
                float | None,
                float | None,
                float | None,
                float | None,
            ]
            | None
        ) = None,
    ) -> None:
        super().__init__()
        self._manager: DreamBatchRunSetManager | None = None
        self._draft_settings = DreamRunSettings()
        self._draft_parameter_entries: list[DreamParameterEntry] = []
        self._default_parameter_entries: list[DreamParameterEntry] = []
        self._draft_prefit_entries: list[PrefitParameterEntry] = []
        self._pending_prefit_parameter_entries = (
            self._copy_prefit_parameter_entries(
                initial_prefit_parameter_entries or []
            )
        )
        self._pending_fit_q_range = initial_fit_q_range
        self._updating_fit_range_controls = False
        self._distribution_window: DistributionSetupWindow | None = None
        self._browse_start_dir = Path.home()
        self._applying_search_filter_preset = False
        self._updating_queue_table = False
        self._updating_filter_table = False
        self._filter_preset_items: dict[str, dict[str, object]] = {}
        self._filter_list_preset_items: dict[str, dict[str, object]] = {}

        self.setWindowTitle("SAXS DREAM Backend Batch Setup (Beta)")
        self.setWindowIcon(load_saxshell_icon())
        self.resize(1180, 820)
        self._build_ui()
        self._set_settings_json(self._draft_settings)
        self._populate_filter_controls(
            PosteriorFilterSettings.from_run_settings(self._draft_settings)
        )
        self._refresh_filter_preset_combo()
        self._refresh_filter_list_preset_combo()

        if initial_project_dir is not None:
            project_dir = Path(initial_project_dir).expanduser().resolve()
            self.project_dir_edit.setText(str(project_dir))
            self._browse_start_dir = project_dir
            self._initialize_run_set()
        else:
            self._set_status("Choose a SAXSShell project folder.")
            self._refresh_command_box()
            self._clear_prefit_preview(
                "Choose a ready SAXSShell project first."
            )

    def _build_ui(self) -> None:
        central = QWidget(self)
        root = QVBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)
        self.setCentralWidget(central)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.setChildrenCollapsible(False)
        root.addWidget(splitter, stretch=1)

        left_scroll = QScrollArea(self)
        left_scroll.setWidgetResizable(True)
        left_panel = QWidget()
        self.left_layout = QVBoxLayout(left_panel)
        self.left_layout.setContentsMargins(10, 10, 10, 10)
        self.left_layout.setSpacing(10)
        left_scroll.setWidget(left_panel)

        right_scroll = QScrollArea(self)
        right_scroll.setWidgetResizable(True)
        right_panel = QWidget()
        self.right_layout = QVBoxLayout(right_panel)
        self.right_layout.setContentsMargins(10, 10, 10, 10)
        self.right_layout.setSpacing(10)
        right_scroll.setWidget(right_panel)

        splitter.addWidget(left_scroll)
        splitter.addWidget(right_scroll)
        splitter.setSizes([560, 620])

        self.left_layout.addWidget(self._build_project_group())
        self.left_layout.addWidget(self._build_settings_group())
        self.left_layout.addWidget(self._build_priors_group())
        self.left_layout.addWidget(self._build_queue_action_group())
        self.left_layout.addStretch(1)

        self.right_layout.addWidget(self._build_prefit_preview_group())
        self.right_layout.addWidget(self._build_queue_group())
        self.right_layout.addWidget(self._build_filter_group())
        self.right_layout.addWidget(self._build_command_group())
        self.right_layout.addStretch(1)
        self.statusBar().showMessage("Ready")

    def _build_project_group(self) -> QGroupBox:
        group = QGroupBox("Project / Run Set")
        form = QFormLayout(group)

        project_row = QHBoxLayout()
        self.project_dir_edit = QLineEdit()
        self.project_dir_edit.editingFinished.connect(self._initialize_run_set)
        project_row.addWidget(self.project_dir_edit, stretch=1)
        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self._browse_project_dir)
        project_row.addWidget(browse_button)
        project_widget = QWidget()
        project_widget.setLayout(project_row)
        form.addRow("Project folder", project_widget)

        self.run_set_edit = QLineEdit()
        self.run_set_edit.setReadOnly(True)
        form.addRow("Run set folder", self.run_set_edit)

        self.conda_env_edit = QLineEdit(DEFAULT_DREAM_BATCH_CONDA_ENV)
        self.conda_env_edit.editingFinished.connect(self._sync_conda_env)
        form.addRow("Conda env", self.conda_env_edit)

        self.prefit_status_label = QLabel()
        self.prefit_status_label.setWordWrap(True)
        form.addRow("Active Prefit", self.prefit_status_label)
        return group

    def _build_settings_group(self) -> QGroupBox:
        group = QGroupBox("DREAM Settings")
        layout = QVBoxLayout(group)

        run_label_row = QHBoxLayout()
        run_label_row.addWidget(QLabel("Queue item label"))
        self.queue_label_edit = QLineEdit()
        self.queue_label_edit.setPlaceholderText(
            "Example: strict priors / 8 chains"
        )
        run_label_row.addWidget(self.queue_label_edit, stretch=1)
        layout.addLayout(run_label_row)

        grid = QGridLayout()
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)
        layout.addLayout(grid)

        row = 0
        self.model_name_edit = QLineEdit()
        self.model_name_edit.setPlaceholderText("Active Prefit model")
        self.search_filter_preset_combo = QComboBox()
        self.search_filter_preset_combo.addItem("Custom", "custom")
        for preset_name in DREAM_SEARCH_FILTER_PRESETS:
            self.search_filter_preset_combo.addItem(
                DREAM_SEARCH_FILTER_PRESET_LABELS.get(
                    preset_name,
                    preset_name.replace("_", " ").title(),
                ),
                preset_name,
            )
        self.search_filter_preset_combo.currentIndexChanged.connect(
            self._on_search_filter_preset_changed
        )
        grid.addWidget(QLabel("Model name"), row, 0)
        grid.addWidget(self.model_name_edit, row, 1)
        grid.addWidget(QLabel("Search preset"), row, 2)
        grid.addWidget(self.search_filter_preset_combo, row, 3)

        row += 1
        self.fit_q_min_spin = _WheelGuardedDoubleSpinBox()
        self.fit_q_min_spin.setDecimals(8)
        self.fit_q_min_spin.setKeyboardTracking(False)
        self.fit_q_min_spin.setEnabled(False)
        self.fit_q_min_spin.valueChanged.connect(
            self._on_fit_range_spin_changed
        )
        self.fit_q_max_spin = _WheelGuardedDoubleSpinBox()
        self.fit_q_max_spin.setDecimals(8)
        self.fit_q_max_spin.setKeyboardTracking(False)
        self.fit_q_max_spin.setEnabled(False)
        self.fit_q_max_spin.valueChanged.connect(
            self._on_fit_range_spin_changed
        )
        grid.addWidget(QLabel("DREAM fit q min"), row, 0)
        grid.addWidget(self.fit_q_min_spin, row, 1)
        grid.addWidget(QLabel("DREAM fit q max"), row, 2)
        grid.addWidget(self.fit_q_max_spin, row, 3)

        row += 1
        self.fit_range_status_label = QLabel("DREAM fit range unavailable")
        self.fit_range_status_label.setWordWrap(True)
        grid.addWidget(self.fit_range_status_label, row, 0, 1, 4)

        row += 1
        self.chains_spin = _WheelGuardedSpinBox()
        self.chains_spin.setRange(1, 512)
        self.chains_spin.valueChanged.connect(
            self._mark_search_filter_preset_custom
        )
        self.iterations_spin = _WheelGuardedSpinBox()
        self.iterations_spin.setRange(1, 10_000_000)
        self.iterations_spin.setSingleStep(100)
        self.iterations_spin.valueChanged.connect(
            self._mark_search_filter_preset_custom
        )
        grid.addWidget(QLabel("Chains"), row, 0)
        grid.addWidget(self.chains_spin, row, 1)
        grid.addWidget(QLabel("Iterations"), row, 2)
        grid.addWidget(self.iterations_spin, row, 3)

        row += 1
        self.burnin_spin = _WheelGuardedSpinBox()
        self.burnin_spin.setRange(0, 95)
        self.burnin_spin.valueChanged.connect(
            self._mark_search_filter_preset_custom
        )
        self.history_thin_spin = _WheelGuardedSpinBox()
        self.history_thin_spin.setRange(1, 1000)
        grid.addWidget(QLabel("Burn-in (%)"), row, 0)
        grid.addWidget(self.burnin_spin, row, 1)
        grid.addWidget(QLabel("History thin"), row, 2)
        grid.addWidget(self.history_thin_spin, row, 3)

        row += 1
        self.nseedchains_spin = _WheelGuardedSpinBox()
        self.nseedchains_spin.setRange(0, 100_000)
        self.nseedchains_spin.valueChanged.connect(
            self._mark_search_filter_preset_custom
        )
        self.crossover_burnin_spin = _WheelGuardedSpinBox()
        self.crossover_burnin_spin.setRange(0, 10_000_000)
        self.crossover_burnin_spin.setSingleStep(100)
        self.crossover_burnin_spin.valueChanged.connect(
            self._mark_search_filter_preset_custom
        )
        grid.addWidget(QLabel("nSeedChains"), row, 0)
        grid.addWidget(self.nseedchains_spin, row, 1)
        grid.addWidget(QLabel("Crossover burn-in"), row, 2)
        grid.addWidget(self.crossover_burnin_spin, row, 3)

        row += 1
        self.lambda_spin = _WheelGuardedDoubleSpinBox()
        self.lambda_spin.setRange(0.0, 100.0)
        self.lambda_spin.setDecimals(6)
        self.lambda_spin.setSingleStep(0.01)
        self.zeta_spin = _WheelGuardedDoubleSpinBox()
        self.zeta_spin.setRange(0.0, 1.0)
        self.zeta_spin.setDecimals(16)
        self.zeta_spin.setSingleStep(1e-12)
        grid.addWidget(QLabel("Lambda"), row, 0)
        grid.addWidget(self.lambda_spin, row, 1)
        grid.addWidget(QLabel("Zeta"), row, 2)
        grid.addWidget(self.zeta_spin, row, 3)

        row += 1
        self.snooker_spin = _WheelGuardedDoubleSpinBox()
        self.snooker_spin.setRange(0.0, 1.0)
        self.snooker_spin.setDecimals(6)
        self.snooker_spin.setSingleStep(0.01)
        self.p_gamma_unity_spin = _WheelGuardedDoubleSpinBox()
        self.p_gamma_unity_spin.setRange(0.0, 1.0)
        self.p_gamma_unity_spin.setDecimals(6)
        self.p_gamma_unity_spin.setSingleStep(0.01)
        grid.addWidget(QLabel("Snooker"), row, 0)
        grid.addWidget(self.snooker_spin, row, 1)
        grid.addWidget(QLabel("p_gamma_unity"), row, 2)
        grid.addWidget(self.p_gamma_unity_spin, row, 3)

        row += 1
        self.verbose_checkbox = QCheckBox("Verbose sampler output")
        self.verbose_checkbox.toggled.connect(self._update_verbose_controls)
        self.parallel_checkbox = QCheckBox("Run chains in parallel")
        self.adapt_checkbox = QCheckBox("Adapt crossover")
        self.restart_checkbox = QCheckBox("Restart previous run")
        grid.addWidget(self.verbose_checkbox, row, 0, 1, 2)
        grid.addWidget(self.parallel_checkbox, row, 2, 1, 2)

        row += 1
        self.verbose_interval_spin = _WheelGuardedDoubleSpinBox()
        self.verbose_interval_spin.setRange(0.1, 30.0)
        self.verbose_interval_spin.setDecimals(1)
        self.verbose_interval_spin.setSingleStep(0.1)
        grid.addWidget(QLabel("Verbose interval (s)"), row, 0)
        grid.addWidget(self.verbose_interval_spin, row, 1)
        grid.addWidget(self.adapt_checkbox, row, 2, 1, 2)

        row += 1
        grid.addWidget(self.restart_checkbox, row, 2, 1, 2)

        row += 1
        self.history_file_edit = QLineEdit()
        self.history_file_edit.setPlaceholderText(
            "Optional chain history .npy"
        )
        grid.addWidget(QLabel("History file"), row, 0)
        grid.addWidget(self.history_file_edit, row, 1, 1, 3)

        layout.addWidget(QLabel("Advanced DREAM settings JSON"))
        self.settings_json_edit = QPlainTextEdit()
        self.settings_json_edit.setMinimumHeight(150)
        layout.addWidget(self.settings_json_edit)

        button_row = QHBoxLayout()
        reload_button = QPushButton("Reload Active DREAM Settings")
        reload_button.clicked.connect(self._reload_project_state)
        button_row.addWidget(reload_button)
        sync_button = QPushButton("Refresh JSON From Controls")
        sync_button.clicked.connect(self._refresh_settings_json_preview)
        button_row.addWidget(sync_button)
        apply_button = QPushButton("Apply JSON to Controls")
        apply_button.clicked.connect(self._validate_settings_json)
        button_row.addWidget(apply_button)
        button_row.addStretch(1)
        layout.addLayout(button_row)
        return group

    def _build_priors_group(self) -> QGroupBox:
        group = QGroupBox("DREAM Priors")
        layout = QVBoxLayout(group)

        self.prior_summary_label = QLabel()
        self.prior_summary_label.setWordWrap(True)
        layout.addWidget(self.prior_summary_label)

        self.prior_table = QTableWidget(0, 7)
        self.prior_table.setHorizontalHeaderLabels(
            [
                "Param",
                "Structure",
                "Vary",
                "Distribution",
                "Params",
                "Guide Low",
                "Guide High",
            ]
        )
        self.prior_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.prior_table.setMinimumHeight(160)
        layout.addWidget(self.prior_table)

        button_row = QHBoxLayout()
        edit_button = QPushButton("Edit Priors")
        edit_button.clicked.connect(self._open_prior_editor)
        button_row.addWidget(edit_button)
        reset_button = QPushButton("Reset to Project Priors")
        reset_button.clicked.connect(self._reload_project_priors)
        button_row.addWidget(reset_button)
        button_row.addStretch(1)
        layout.addLayout(button_row)
        return group

    def _build_queue_action_group(self) -> QGroupBox:
        group = QGroupBox("Queue Items")
        layout = QVBoxLayout(group)

        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Queue preset"))
        self.queue_preset_combo = QComboBox()
        for preset in DREAM_BATCH_QUEUE_PRESETS:
            self.queue_preset_combo.addItem(
                str(preset["label"]),
                str(preset["id"]),
            )
        self.queue_preset_combo.currentIndexChanged.connect(
            self._on_queue_preset_changed
        )
        preset_row.addWidget(self.queue_preset_combo, stretch=1)
        apply_preset_button = QPushButton("Add Preset Queue Items")
        apply_preset_button.clicked.connect(self._apply_selected_queue_preset)
        preset_row.addWidget(apply_preset_button)
        layout.addLayout(preset_row)

        self.queue_preset_description_label = QLabel()
        self.queue_preset_description_label.setWordWrap(True)
        layout.addWidget(self.queue_preset_description_label)
        self._on_queue_preset_changed(self.queue_preset_combo.currentIndex())

        create_row = QHBoxLayout()
        create_button = QPushButton("Create Queue Item + Runtime Bundle")
        create_button.clicked.connect(self._create_queue_item)
        create_row.addWidget(create_button)
        create_row.addStretch(1)
        layout.addLayout(create_row)
        return group

    def _build_prefit_preview_group(self) -> QGroupBox:
        group = QGroupBox("Active Prefit Preview")
        layout = QVBoxLayout(group)

        self.prefit_preview_status_label = QLabel()
        self.prefit_preview_status_label.setWordWrap(True)
        layout.addWidget(self.prefit_preview_status_label)

        self.prefit_preview_figure = Figure(figsize=(6.4, 3.2))
        self.prefit_preview_canvas = FigureCanvasQTAgg(
            self.prefit_preview_figure
        )
        self.prefit_preview_canvas.setMinimumHeight(260)
        self.prefit_preview_toolbar = NavigationToolbar2QT(
            self.prefit_preview_canvas,
            self,
        )
        layout.addWidget(self.prefit_preview_toolbar)
        layout.addWidget(self.prefit_preview_canvas)

        self.prefit_parameter_table = QTableWidget(0, 9)
        self.prefit_parameter_table.setHorizontalHeaderLabels(
            [
                "Param",
                "Structure",
                "Motif",
                "Value",
                "Vary",
                "Min",
                "Max",
                "Category",
                "Use",
            ]
        )
        self.prefit_parameter_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.prefit_parameter_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.prefit_parameter_table.setMinimumHeight(180)
        self.prefit_parameter_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        self.prefit_parameter_table.horizontalHeader().setStretchLastSection(
            True
        )
        layout.addWidget(self.prefit_parameter_table)
        return group

    def _build_queue_group(self) -> QGroupBox:
        group = QGroupBox("Queued DREAM Runs")
        layout = QVBoxLayout(group)
        self.queue_table = QTableWidget(0, len(self.QUEUE_TABLE_HEADERS))
        self.queue_table.setHorizontalHeaderLabels(
            list(self.QUEUE_TABLE_HEADERS)
        )
        self.queue_table.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked
            | QAbstractItemView.EditTrigger.EditKeyPressed
        )
        self.queue_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.queue_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.queue_table.itemSelectionChanged.connect(
            self._refresh_selected_queue_item_view
        )
        self.queue_table.itemChanged.connect(self._on_queue_table_item_changed)
        self.queue_table.setMinimumHeight(210)
        layout.addWidget(self.queue_table)

        queue_button_row = QHBoxLayout()
        remove_queue_item_button = QPushButton("Remove Selected Queue Item")
        remove_queue_item_button.clicked.connect(
            self._remove_selected_queue_item
        )
        queue_button_row.addWidget(remove_queue_item_button)
        queue_button_row.addStretch(1)
        layout.addLayout(queue_button_row)

        self.queue_detail_label = QLabel(
            "Select a queued DREAM run to inspect its settings and priors."
        )
        self.queue_detail_label.setWordWrap(True)
        layout.addWidget(self.queue_detail_label)

        self.queue_settings_table = QTableWidget(0, 2)
        self.queue_settings_table.setHorizontalHeaderLabels(
            ["DREAM Setting", "Value"]
        )
        self.queue_settings_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.queue_settings_table.setMinimumHeight(145)
        layout.addWidget(self.queue_settings_table)

        self.queue_prior_table = QTableWidget(0, 7)
        self.queue_prior_table.setHorizontalHeaderLabels(
            [
                "Param",
                "Structure",
                "Vary",
                "Distribution",
                "Params",
                "Guide Low",
                "Guide High",
            ]
        )
        self.queue_prior_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.queue_prior_table.setMinimumHeight(170)
        layout.addWidget(self.queue_prior_table)
        return group

    def _build_filter_group(self) -> QGroupBox:
        group = QGroupBox("Posterior Filter Sets")
        layout = QVBoxLayout(group)

        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Filter preset"))
        self.filter_preset_combo = QComboBox()
        self.filter_preset_combo.currentIndexChanged.connect(
            self._on_filter_preset_changed
        )
        preset_row.addWidget(self.filter_preset_combo, stretch=1)
        load_preset_button = QPushButton("Load Filter Preset")
        load_preset_button.clicked.connect(self._load_selected_filter_preset)
        preset_row.addWidget(load_preset_button)
        save_preset_button = QPushButton("Save Current Filter Preset")
        save_preset_button.clicked.connect(self._save_current_filter_preset)
        preset_row.addWidget(save_preset_button)
        layout.addLayout(preset_row)

        self.filter_preset_description_label = QLabel()
        self.filter_preset_description_label.setWordWrap(True)
        layout.addWidget(self.filter_preset_description_label)

        filter_list_label_row = QHBoxLayout()
        filter_list_label_row.addWidget(QLabel("Filter list label"))
        self.filter_list_label_edit = QLineEdit()
        self.filter_list_label_edit.setPlaceholderText(
            "Example: all / top 10% / top 500 comparison"
        )
        filter_list_label_row.addWidget(
            self.filter_list_label_edit,
            stretch=1,
        )
        layout.addLayout(filter_list_label_row)

        list_preset_row = QHBoxLayout()
        list_preset_row.addWidget(QLabel("Filter list preset"))
        self.filter_list_preset_combo = QComboBox()
        self.filter_list_preset_combo.currentIndexChanged.connect(
            self._on_filter_list_preset_changed
        )
        list_preset_row.addWidget(self.filter_list_preset_combo, stretch=1)
        load_list_preset_button = QPushButton("Load Filter List Preset")
        load_list_preset_button.clicked.connect(
            self._load_selected_filter_list_preset
        )
        list_preset_row.addWidget(load_list_preset_button)
        save_list_preset_button = QPushButton("Save Current Filter List")
        save_list_preset_button.clicked.connect(
            self._save_current_filter_list_preset
        )
        list_preset_row.addWidget(save_list_preset_button)
        layout.addLayout(list_preset_row)

        self.filter_list_preset_description_label = QLabel()
        self.filter_list_preset_description_label.setWordWrap(True)
        layout.addWidget(self.filter_list_preset_description_label)

        grid = QGridLayout()
        self.filter_controls_grid = grid
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)
        layout.addLayout(grid)

        row = 0
        self.filter_label_edit = QLineEdit("default_filter")
        grid.addWidget(QLabel("Filter label"), row, 0)
        grid.addWidget(self.filter_label_edit, row, 1)

        row += 1
        self.bestfit_method_combo = QComboBox()
        self.bestfit_method_combo.addItem("MAP", "map")
        self.bestfit_method_combo.addItem("Chain Mean MAP", "chain_mean")
        self.bestfit_method_combo.addItem("Median", "median")
        self.violin_mode_combo = QComboBox()
        self.violin_mode_combo.addItem(
            "Varying Parameters",
            "varying_parameters",
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
        self.violin_mode_combo.addItem(
            "Selected Additional Parameter",
            "selected_additional_parameter",
        )
        grid.addWidget(QLabel("Best-fit method"), row, 0)
        grid.addWidget(self.bestfit_method_combo, row, 1)
        grid.addWidget(QLabel("Violin data"), row, 2)
        grid.addWidget(self.violin_mode_combo, row, 3)

        row += 1
        self.violin_parameter_edit = QLineEdit()
        self.violin_parameter_edit.setPlaceholderText(
            "Parameter for selected-parameter violin mode"
        )
        grid.addWidget(QLabel("Parameter"), row, 0)
        grid.addWidget(self.violin_parameter_edit, row, 1, 1, 3)

        row += 1
        self.weight_order_combo = QComboBox()
        self.weight_order_combo.addItem("Weight Index", "weight_index")
        self.weight_order_combo.addItem("Structure Order", "structure_order")
        self.violin_value_scale_combo = QComboBox()
        self.violin_value_scale_combo.addItem(
            "Parameter Value",
            "parameter_value",
        )
        self.violin_value_scale_combo.addItem(
            "Weights 0-1 Only",
            "weights_unit_interval",
        )
        self.violin_value_scale_combo.addItem(
            "Structure Fraction (%)",
            "structure_fraction_percent",
        )
        self.violin_value_scale_combo.addItem(
            "Total Atom Fraction (%)",
            "atom_fraction_percent",
        )
        self.violin_value_scale_combo.addItem(
            "Normalized 0-1 (All)",
            "normalized_all",
        )
        self.violin_value_scale_combo.addItem(
            "Effective Radii Only",
            "effective_radii_only",
        )
        self.violin_value_scale_combo.addItem(
            "Additional Parameters Only",
            "additional_parameters_only",
        )
        grid.addWidget(QLabel("Weight order"), row, 0)
        grid.addWidget(self.weight_order_combo, row, 1)
        grid.addWidget(QLabel("Y-axis scale"), row, 2)
        grid.addWidget(self.violin_value_scale_combo, row, 3)

        row += 1
        self.stoichiometry_elements_edit = QLineEdit()
        self.stoichiometry_elements_edit.setPlaceholderText("e.g. Pb, I")
        self.stoichiometry_ratio_edit = QLineEdit()
        self.stoichiometry_ratio_edit.setPlaceholderText("e.g. 1:2")
        grid.addWidget(QLabel("Target elements"), row, 0)
        grid.addWidget(self.stoichiometry_elements_edit, row, 1)
        grid.addWidget(QLabel("Target ratio"), row, 2)
        grid.addWidget(self.stoichiometry_ratio_edit, row, 3)

        row += 1
        self.stoichiometry_filter_checkbox = QCheckBox(
            "Enable stoichiometry filter"
        )
        self.stoichiometry_filter_checkbox.toggled.connect(
            self._update_posterior_filter_controls
        )
        self.stoichiometry_tolerance_spin = _WheelGuardedDoubleSpinBox()
        self.stoichiometry_tolerance_spin.setRange(0.0, 1000.0)
        self.stoichiometry_tolerance_spin.setDecimals(2)
        self.stoichiometry_tolerance_spin.setSingleStep(0.5)
        grid.addWidget(self.stoichiometry_filter_checkbox, row, 0, 1, 2)
        grid.addWidget(QLabel("Tolerance (%)"), row, 2)
        grid.addWidget(self.stoichiometry_tolerance_spin, row, 3)

        row += 1
        self.posterior_filter_combo = QComboBox()
        self.posterior_filter_combo.addItem(
            "All Post-burnin Samples",
            "all_post_burnin",
        )
        self.posterior_filter_combo.addItem(
            "Top % by Log-posterior",
            "top_percent_logp",
        )
        self.posterior_filter_combo.addItem(
            "Top N by Log-posterior",
            "top_n_logp",
        )
        self.posterior_filter_combo.currentIndexChanged.connect(
            self._on_posterior_filter_mode_changed
        )
        self.violin_sample_source_combo = QComboBox()
        self.violin_sample_source_combo.addItem(
            "Filtered Posterior",
            "filtered_posterior",
        )
        self.violin_sample_source_combo.addItem(
            "MAP Chain Only",
            "map_chain_only",
        )
        self.violin_sample_source_combo.currentIndexChanged.connect(
            self._refresh_filter_json_preview
        )
        grid.addWidget(QLabel("Posterior filter"), row, 0)
        grid.addWidget(self.posterior_filter_combo, row, 1)
        grid.addWidget(QLabel("Violin samples"), row, 2)
        grid.addWidget(self.violin_sample_source_combo, row, 3)

        row += 1
        self.posterior_top_percent_spin = _WheelGuardedDoubleSpinBox()
        self.posterior_top_percent_spin.setRange(0.1, 100.0)
        self.posterior_top_percent_spin.setDecimals(2)
        self.posterior_top_percent_spin.setSingleStep(1.0)
        self.posterior_top_percent_spin.valueChanged.connect(
            self._refresh_filter_json_preview
        )
        self.posterior_top_n_spin = _WheelGuardedSpinBox()
        self.posterior_top_n_spin.setRange(1, 10_000_000)
        self.posterior_top_n_spin.valueChanged.connect(
            self._refresh_filter_json_preview
        )
        grid.addWidget(QLabel("Top % default"), row, 0)
        grid.addWidget(self.posterior_top_percent_spin, row, 1)
        grid.addWidget(QLabel("Top N default"), row, 2)
        grid.addWidget(self.posterior_top_n_spin, row, 3)

        row += 1
        self.auto_filter_assessment_checkbox = QCheckBox(
            "Auto-select best filter after run"
        )
        grid.addWidget(self.auto_filter_assessment_checkbox, row, 0, 1, 4)

        row += 1
        self.credible_interval_low_spin = _WheelGuardedDoubleSpinBox()
        self.credible_interval_low_spin.setRange(0.0, 99.9)
        self.credible_interval_low_spin.setDecimals(1)
        self.credible_interval_low_spin.setSingleStep(1.0)
        self.credible_interval_high_spin = _WheelGuardedDoubleSpinBox()
        self.credible_interval_high_spin.setRange(0.1, 100.0)
        self.credible_interval_high_spin.setDecimals(1)
        self.credible_interval_high_spin.setSingleStep(1.0)
        grid.addWidget(QLabel("Interval low (%)"), row, 0)
        grid.addWidget(self.credible_interval_low_spin, row, 1)
        grid.addWidget(QLabel("Interval high (%)"), row, 2)
        grid.addWidget(self.credible_interval_high_spin, row, 3)

        layout.addWidget(QLabel("Posterior filter JSON preview"))
        self.filter_json_edit = QPlainTextEdit()
        self.filter_json_edit.setMinimumHeight(120)
        layout.addWidget(self.filter_json_edit)

        button_row = QHBoxLayout()
        load_button = QPushButton("Load From DREAM Settings")
        load_button.clicked.connect(self._load_filter_from_settings)
        button_row.addWidget(load_button)
        apply_button = QPushButton("Apply JSON to Controls")
        apply_button.clicked.connect(self._validate_filter_json)
        button_row.addWidget(apply_button)
        preview_button = QPushButton("Refresh JSON From Controls")
        preview_button.clicked.connect(self._refresh_filter_json_preview)
        button_row.addWidget(preview_button)
        add_button = QPushButton("Add Filter Set")
        add_button.clicked.connect(self._add_filter_set)
        button_row.addWidget(add_button)
        remove_button = QPushButton("Remove Selected Filter Set")
        remove_button.clicked.connect(self._remove_selected_filter_set)
        button_row.addWidget(remove_button)
        button_row.addStretch(1)
        layout.addLayout(button_row)

        self.filter_table = QTableWidget(0, len(self.FILTER_TABLE_HEADERS))
        self.filter_table.setHorizontalHeaderLabels(
            list(self.FILTER_TABLE_HEADERS)
        )
        self.filter_table.setEditTriggers(
            QAbstractItemView.EditTrigger.AllEditTriggers
        )
        self.filter_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.filter_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.filter_table.itemChanged.connect(
            self._on_filter_table_item_changed
        )
        self.filter_table.setMinimumHeight(140)
        layout.addWidget(self.filter_table)
        return group

    def _build_command_group(self) -> QGroupBox:
        group = QGroupBox("Command Output")
        layout = QVBoxLayout(group)
        button_row = QHBoxLayout()
        generate_button = QPushButton("Generate Shell Script")
        generate_button.clicked.connect(self._generate_shell_script)
        button_row.addWidget(generate_button)
        self.open_commands_button = QPushButton(
            "Reveal Command Set TXT in Finder"
        )
        self.open_commands_button.clicked.connect(
            self._reveal_command_set_file
        )
        button_row.addWidget(self.open_commands_button)
        button_row.addStretch(1)
        layout.addLayout(button_row)
        self.command_box = QPlainTextEdit()
        self.command_box.setReadOnly(True)
        self.command_box.setMinimumHeight(260)
        layout.addWidget(self.command_box)
        return group

    def _browse_project_dir(self, *_args: object) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "Select SAXSShell project folder",
            str(self._browse_start_dir),
        )
        if not selected:
            return
        self.project_dir_edit.setText(selected)
        self._browse_start_dir = Path(selected).expanduser().resolve()
        self._initialize_run_set()

    def _initialize_run_set(self, *_args: object) -> None:
        project_dir = self._project_dir()
        if project_dir is None:
            return
        try:
            self._manager = DreamBatchRunSetManager(
                project_dir,
                conda_env=self.conda_env_edit.text().strip()
                or DEFAULT_DREAM_BATCH_CONDA_ENV,
                reuse_latest=True,
            )
        except Exception as exc:
            self._manager = None
            self.run_set_edit.clear()
            self.prefit_status_label.setText(str(exc))
            self._clear_prefit_preview(str(exc))
            self._set_status("DREAM backend setup is not ready.")
            self._refresh_command_box()
            self._refresh_filter_preset_combo()
            self._refresh_filter_list_preset_combo()
            return
        self.run_set_edit.setText(
            str(self._manager.run_set.resolved_run_set_dir)
        )
        self.conda_env_edit.setText(self._manager.run_set.conda_env)
        self._reload_project_state()
        self._apply_pending_prefit_parameter_entries()
        self._apply_pending_fit_q_range()
        self._refresh_filter_preset_combo()
        self._refresh_filter_list_preset_combo()
        self._refresh_tables()
        self._refresh_command_box()
        self._set_status("DREAM backend run set ready.")

    def _reload_project_state(self, *_args: object) -> None:
        manager = self._require_manager()
        workflow = manager.workflow
        self._draft_settings = workflow.load_settings()
        if not self._draft_settings.model_name:
            self._draft_settings.model_name = (
                workflow.prefit_workflow.template_spec.name
            )
        self._set_settings_json(self._draft_settings)
        self._draft_prefit_entries = [
            PrefitParameterEntry.from_dict(entry.to_dict())
            for entry in workflow.prefit_workflow.parameter_entries
        ]
        self._reload_project_priors()
        self._load_filter_from_settings()
        state_path = workflow.prefit_dir / "prefit_state.json"
        self.prefit_status_label.setText(
            f"{workflow.prefit_workflow.template_spec.name}\n{state_path}"
        )
        self._refresh_prefit_preview()
        self._apply_pending_fit_q_range()

    def set_active_prefit_fit_q_range(
        self,
        fit_q_min: float | None,
        fit_q_max: float | None,
        model_q_min: float | None = None,
        model_q_max: float | None = None,
    ) -> None:
        self._pending_fit_q_range = (
            fit_q_min,
            fit_q_max,
            model_q_min,
            model_q_max,
        )
        self._apply_pending_fit_q_range()

    def set_active_prefit_parameter_entries(
        self,
        entries: list[PrefitParameterEntry],
    ) -> None:
        self._pending_prefit_parameter_entries = (
            self._copy_prefit_parameter_entries(entries)
        )
        self._apply_pending_prefit_parameter_entries()

    def _apply_pending_prefit_parameter_entries(self) -> None:
        if not self._pending_prefit_parameter_entries:
            return
        entries = self._copy_prefit_parameter_entries(
            self._pending_prefit_parameter_entries
        )
        self._draft_prefit_entries = entries
        if self._manager is not None:
            self._manager.workflow.prefit_workflow.parameter_entries = (
                self._copy_prefit_parameter_entries(entries)
            )
        self._refresh_prefit_preview()
        if hasattr(self, "prefit_status_label"):
            base_text = self.prefit_status_label.text().strip()
            note = "Using active Prefit parameters from the main UI."
            if note not in base_text:
                self.prefit_status_label.setText(
                    f"{base_text}\n{note}" if base_text else note
                )
        self._set_status(
            "DREAM backend setup updated from the active Prefit parameters."
        )

    def _apply_pending_fit_q_range(self) -> None:
        if self._pending_fit_q_range is None:
            return
        fit_q_min, fit_q_max, model_q_min, model_q_max = (
            self._pending_fit_q_range
        )
        self._set_fit_q_range_controls(
            fit_q_min=fit_q_min,
            fit_q_max=fit_q_max,
            model_q_min=model_q_min,
            model_q_max=model_q_max,
        )
        settings = self._settings_from_controls()
        self._set_settings_json(settings, update_controls=False)

    def _set_fit_q_range_controls(
        self,
        *,
        fit_q_min: float | None,
        fit_q_max: float | None,
        model_q_min: float | None = None,
        model_q_max: float | None = None,
    ) -> None:
        if (
            fit_q_min is None
            or fit_q_max is None
            or not np.isfinite(float(fit_q_min))
            or not np.isfinite(float(fit_q_max))
        ):
            self._updating_fit_range_controls = True
            try:
                self.fit_q_min_spin.setEnabled(False)
                self.fit_q_max_spin.setEnabled(False)
                self.fit_range_status_label.setText(
                    "DREAM fit range unavailable"
                )
            finally:
                self._updating_fit_range_controls = False
            return
        selected_lower = float(fit_q_min)
        selected_upper = float(fit_q_max)
        if selected_lower > selected_upper:
            selected_lower, selected_upper = selected_upper, selected_lower
        lower = (
            selected_lower
            if model_q_min is None or not np.isfinite(float(model_q_min))
            else float(model_q_min)
        )
        upper = (
            selected_upper
            if model_q_max is None or not np.isfinite(float(model_q_max))
            else float(model_q_max)
        )
        if lower > upper:
            lower, upper = selected_lower, selected_upper
        selected_lower = min(max(selected_lower, lower), upper)
        selected_upper = min(max(selected_upper, lower), upper)
        if selected_lower > selected_upper:
            selected_lower, selected_upper = selected_upper, selected_lower
        step = max((upper - lower) / 100.0, 1.0e-6)
        self._updating_fit_range_controls = True
        try:
            for spin in (self.fit_q_min_spin, self.fit_q_max_spin):
                spin.blockSignals(True)
                spin.setRange(lower, upper)
                spin.setSingleStep(step)
                spin.setEnabled(True)
            self.fit_q_min_spin.setValue(selected_lower)
            self.fit_q_max_spin.setValue(selected_upper)
            for spin in (self.fit_q_min_spin, self.fit_q_max_spin):
                spin.blockSignals(False)
            self.fit_range_status_label.setText(
                "DREAM fit: "
                f"{selected_lower:.6g} to {selected_upper:.6g} A^-1 "
                f"(model {lower:.6g} to {upper:.6g})"
            )
        finally:
            self._updating_fit_range_controls = False

    def _fit_q_range_from_controls(self) -> tuple[float | None, float | None]:
        if not self.fit_q_min_spin.isEnabled():
            return (None, None)
        return (
            float(self.fit_q_min_spin.value()),
            float(self.fit_q_max_spin.value()),
        )

    def _on_fit_range_spin_changed(self, _value: float) -> None:
        if self._updating_fit_range_controls:
            return
        q_min = float(self.fit_q_min_spin.value())
        q_max = float(self.fit_q_max_spin.value())
        sender = self.sender()
        if q_min > q_max:
            if sender is self.fit_q_min_spin:
                q_max = q_min
            else:
                q_min = q_max
            self._updating_fit_range_controls = True
            try:
                self.fit_q_min_spin.blockSignals(True)
                self.fit_q_max_spin.blockSignals(True)
                self.fit_q_min_spin.setValue(q_min)
                self.fit_q_max_spin.setValue(q_max)
                self.fit_q_min_spin.blockSignals(False)
                self.fit_q_max_spin.blockSignals(False)
            finally:
                self._updating_fit_range_controls = False
        self.fit_range_status_label.setText(
            f"DREAM fit: {q_min:.6g} to {q_max:.6g} A^-1"
        )
        self._refresh_settings_json_preview()

    def _reload_project_priors(self, *_args: object) -> None:
        manager = self._require_manager()
        workflow = manager.workflow
        try:
            entries = workflow.load_parameter_map(persist_if_missing=False)
        except Exception:
            entries = []
        if not entries:
            entries = workflow.create_default_parameter_map(persist=False)
        self._draft_parameter_entries = [
            DreamParameterEntry.from_dict(entry.to_dict()) for entry in entries
        ]
        self._default_parameter_entries = (
            workflow.create_default_parameter_map(persist=False)
        )
        self._refresh_prior_table()

    def _rebuild_draft_priors_from_prefit(self) -> None:
        defaults = build_default_parameter_map_from_prefit_entries(
            self._draft_prefit_entries
        )
        existing_by_param = {
            entry.param: entry for entry in self._draft_parameter_entries
        }
        rebuilt: list[DreamParameterEntry] = []
        for default_entry in defaults:
            existing_entry = existing_by_param.get(default_entry.param)
            if existing_entry is None:
                rebuilt.append(default_entry)
                continue
            rebuilt.append(
                DreamParameterEntry(
                    structure=default_entry.structure,
                    motif=default_entry.motif,
                    param_type=default_entry.param_type,
                    param=default_entry.param,
                    value=existing_entry.value,
                    vary=existing_entry.vary,
                    distribution=existing_entry.distribution,
                    dist_params=dict(existing_entry.dist_params),
                    smart_preset_status=existing_entry.smart_preset_status,
                )
            )
        self._draft_parameter_entries = rebuilt
        self._default_parameter_entries = defaults
        self._refresh_prior_table()
        if self._distribution_window is not None:
            self._distribution_window.load_entries(
                self._draft_parameter_entries,
                has_existing_parameter_map=bool(self._draft_parameter_entries),
                default_entries=self._default_parameter_entries,
            )

    def _open_prior_editor(self, *_args: object) -> None:
        self._require_manager()
        if self._distribution_window is None:
            self._distribution_window = DistributionSetupWindow(
                self._draft_parameter_entries,
                self,
                default_entries=self._default_parameter_entries,
            )
            self._distribution_window.saved.connect(self._save_prior_draft)
        self._distribution_window.load_entries(
            self._draft_parameter_entries,
            has_existing_parameter_map=bool(self._draft_parameter_entries),
            default_entries=self._default_parameter_entries,
        )
        self._distribution_window.show()
        self._distribution_window.raise_()
        self._distribution_window.activateWindow()

    def _save_prior_draft(self, entries: list[DreamParameterEntry]) -> None:
        self._draft_parameter_entries = [
            DreamParameterEntry.from_dict(entry.to_dict()) for entry in entries
        ]
        self._refresh_prior_table()
        self._set_status("Updated DREAM prior draft for the next queue item.")

    def _create_queue_item(self, *_args: object) -> None:
        try:
            manager = self._require_manager()
            settings = self._settings_from_controls()
            label = (
                self.queue_label_edit.text().strip()
                or manager.next_queue_item_label()
            )
            settings.run_label = label
            self._draft_settings = settings
            self._set_settings_json(settings, update_controls=False)
            item = manager.add_queue_item(
                label=label,
                settings=settings,
                entries=self._draft_parameter_entries,
                prefit_parameter_entries=self._draft_prefit_entries or None,
            )
            self.queue_label_edit.clear()
            self._refresh_tables(selected_queue_item_id=item.item_id)
            self._refresh_command_box()
            self._set_status(f"Created runtime bundle: {item.run_dir}")
        except Exception as exc:
            QMessageBox.warning(self, "DREAM Backend Batch Setup", str(exc))
            self._set_status("Queue item creation failed.")

    def _remove_selected_queue_item(self, *_args: object) -> None:
        try:
            manager = self._require_manager()
            item_id = self._selected_queue_item_id()
            if not item_id:
                raise ValueError("Select a DREAM queue item to remove.")
            queue_item = next(
                (
                    item
                    for item in manager.run_set.queue_items
                    if item.item_id == item_id
                ),
                None,
            )
            if queue_item is None:
                raise ValueError("Selected DREAM queue item is unavailable.")
            response = QMessageBox.question(
                self,
                "Remove DREAM Queue Item",
                (
                    "Remove this queued DREAM run and delete its generated "
                    "runtime bundle?\n\n"
                    f"{queue_item.label}\n{queue_item.run_dir}"
                ),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if response != QMessageBox.StandardButton.Yes:
                self._set_status("Queue item removal cancelled.")
                return
            removed = manager.remove_queue_item(queue_item.item_id)
            self._refresh_tables()
            self._refresh_command_box()
            self._set_status(f"Removed queue item: {removed.label}")
        except Exception as exc:
            QMessageBox.warning(self, "Remove DREAM Queue Item", str(exc))
            self._set_status("Queue item removal failed.")

    def _on_queue_preset_changed(self, _index: int) -> None:
        if not hasattr(self, "queue_preset_combo"):
            return
        preset = self._selected_queue_preset()
        description = ""
        if preset is not None:
            description = str(preset.get("description", ""))
        self.queue_preset_combo.setToolTip(description)
        if hasattr(self, "queue_preset_description_label"):
            self.queue_preset_description_label.setText(description)

    def _selected_queue_preset(self) -> dict[str, object] | None:
        preset_id = self._combo_data(self.queue_preset_combo, "")
        for preset in DREAM_BATCH_QUEUE_PRESETS:
            if str(preset.get("id", "")) == preset_id:
                return preset
        return None

    def _apply_selected_queue_preset(self, *_args: object) -> None:
        try:
            manager = self._require_manager()
            preset = self._selected_queue_preset()
            if preset is None:
                raise ValueError("Choose a DREAM queue preset first.")
            self._validate_queue_preset_template(preset)
            settings_template = self._settings_from_controls()
            entries_template = self._copy_parameter_entries(
                self._draft_parameter_entries
            )
            prefix = self.queue_label_edit.text().strip()
            created_count = 0
            last_item_id = ""
            for variant in self._queue_preset_variants(preset):
                label = self._queue_preset_label(
                    preset,
                    variant,
                    prefix=prefix,
                )
                settings = self._queue_preset_settings(
                    settings_template,
                    label=label,
                    search_preset=str(variant["search_preset"]),
                )
                entries = self._queue_preset_entries(
                    entries_template,
                    prior_mode=str(variant["prior_mode"]),
                    scale_offset_mode=str(variant["scale_offset_mode"]),
                    vary_flags=dict(preset.get("vary_flags", {})),
                )
                item = manager.add_queue_item(
                    label=label,
                    settings=settings,
                    entries=entries,
                    prefit_parameter_entries=self._draft_prefit_entries
                    or None,
                )
                last_item_id = item.item_id
                created_count += 1
            self.queue_label_edit.clear()
            self._refresh_tables(selected_queue_item_id=last_item_id or None)
            self._refresh_command_box()
            self._set_status(
                f"Added {created_count} queue preset item"
                f"{'' if created_count == 1 else 's'}."
            )
        except Exception as exc:
            QMessageBox.warning(self, "DREAM Queue Preset", str(exc))
            self._set_status("Queue preset creation failed.")

    def _validate_queue_preset_template(
        self,
        preset: dict[str, object],
    ) -> None:
        required_text = str(
            preset.get("requires_template_contains", "")
        ).strip()
        if not required_text:
            return
        manager = self._require_manager()
        template_name = manager.workflow.prefit_workflow.template_spec.name
        if required_text.lower() not in template_name.lower():
            raise ValueError(
                f"The {preset['label']} queue preset is intended for "
                f"{required_text} templates. Active template: {template_name}."
            )

    def _queue_preset_variants(
        self,
        preset: dict[str, object],
    ) -> list[dict[str, str]]:
        search_presets = tuple(
            str(value)
            for value in preset.get(
                "search_presets",
                (preset.get("search_preset", "medium"),),
            )
        )
        scale_offset_modes = tuple(
            str(value) for value in preset.get("scale_offset_modes", ("on",))
        )
        prior_modes = tuple(
            str(value)
            for value in preset.get("prior_modes", ("proportional",))
        )
        variants: list[dict[str, str]] = []
        for search_preset in search_presets:
            for scale_offset_mode in scale_offset_modes:
                for prior_mode in prior_modes:
                    variants.append(
                        {
                            "search_preset": search_preset,
                            "scale_offset_mode": scale_offset_mode,
                            "prior_mode": prior_mode,
                        }
                    )
        return variants

    def _queue_preset_label(
        self,
        preset: dict[str, object],
        variant: dict[str, str],
        *,
        prefix: str,
    ) -> str:
        base_label = prefix or str(preset["label"])
        parts = [base_label]
        scale_offset_modes = tuple(
            str(value) for value in preset.get("scale_offset_modes", ())
        )
        if len(scale_offset_modes) > 1:
            parts.append(
                self._queue_preset_scale_offset_label(
                    variant["scale_offset_mode"]
                )
            )
        search_presets = preset.get("search_presets")
        if search_presets:
            parts.append(
                self._queue_preset_search_label(variant["search_preset"])
            )
        parts.append(self._queue_preset_prior_label(variant["prior_mode"]))
        label = " - ".join(part for part in parts if part)
        return self._unique_queue_label(label)

    def _unique_queue_label(self, label: str) -> str:
        manager = self._require_manager()
        existing = {item.label for item in manager.run_set.queue_items}
        if label not in existing:
            return label
        index = 2
        while f"{label}_{index}" in existing:
            index += 1
        return f"{label}_{index}"

    @staticmethod
    def _queue_preset_scale_offset_label(scale_offset_mode: str) -> str:
        if scale_offset_mode == "on":
            return "Scale+Offset On"
        if scale_offset_mode == "off":
            return "Scale+Offset Off"
        return scale_offset_mode.replace("_", " ").title()

    @staticmethod
    def _queue_preset_prior_label(prior_mode: str) -> str:
        labels = {
            "strict": "Strict",
            "proportional": "Proportional",
            "lenient": "Lenient",
            "rounded_guides": "Rounded Guides",
            LEGACY_MD_WEIGHT_SMART_PRIOR_MODE: "Legacy MD Weights",
        }
        return labels.get(prior_mode, prior_mode.replace("_", " ").title())

    @staticmethod
    def _queue_preset_search_label(search_preset: str) -> str:
        label = DREAM_SEARCH_FILTER_PRESET_LABELS.get(
            search_preset,
            search_preset.replace("_", " ").title(),
        )
        return f"{label} Search"

    def _queue_preset_settings(
        self,
        settings_template: DreamRunSettings,
        *,
        label: str,
        search_preset: str,
    ) -> DreamRunSettings:
        settings = DreamRunSettings.from_dict(settings_template.to_dict())
        settings.run_label = label
        self._apply_search_preset_to_settings(settings, search_preset)
        return settings

    @staticmethod
    def _apply_search_preset_to_settings(
        settings: DreamRunSettings,
        search_preset: str,
    ) -> None:
        preset = DREAM_SEARCH_FILTER_PRESETS.get(search_preset)
        if preset is None:
            settings.search_filter_preset = "custom"
            return
        settings.search_filter_preset = search_preset
        settings.nchains = int(preset["nchains"])
        settings.niterations = int(preset["niterations"])
        settings.burnin_percent = int(preset["burnin_percent"])
        settings.nseedchains = int(preset["nseedchains"])
        settings.crossover_burnin = int(preset["crossover_burnin"])
        if "history_thin" in preset:
            settings.history_thin = int(preset["history_thin"])
        if "lamb" in preset:
            settings.lamb = float(preset["lamb"])
        if "zeta" in preset:
            settings.zeta = float(preset["zeta"])
        if "snooker" in preset:
            settings.snooker = float(preset["snooker"])
        if "p_gamma_unity" in preset:
            settings.p_gamma_unity = float(preset["p_gamma_unity"])
        if "verbose" in preset:
            settings.verbose = bool(preset["verbose"])
        if "parallel" in preset:
            settings.parallel = bool(preset["parallel"])
        if "adapt_crossover" in preset:
            settings.adapt_crossover = bool(preset["adapt_crossover"])

    def _queue_preset_entries(
        self,
        entries_template: list[DreamParameterEntry],
        *,
        prior_mode: str,
        scale_offset_mode: str,
        vary_flags: dict[str, object],
    ) -> list[DreamParameterEntry]:
        if prior_mode == "rounded_guides":
            entries = self._rounded_guide_entries(entries_template)
        else:
            entries = self._smart_prior_entries(entries_template, prior_mode)
        self._apply_queue_vary_flags(
            entries,
            scale_offset_mode=scale_offset_mode,
            vary_flags=vary_flags,
        )
        return entries

    @staticmethod
    def _copy_parameter_entries(
        entries: list[DreamParameterEntry],
    ) -> list[DreamParameterEntry]:
        return [
            DreamParameterEntry.from_dict(entry.to_dict()) for entry in entries
        ]

    @staticmethod
    def _copy_prefit_parameter_entries(
        entries: list[PrefitParameterEntry],
    ) -> list[PrefitParameterEntry]:
        return [
            PrefitParameterEntry.from_dict(entry.to_dict())
            for entry in entries
        ]

    @staticmethod
    def _smart_prior_entries(
        entries: list[DreamParameterEntry],
        prior_mode: str,
    ) -> list[DreamParameterEntry]:
        factor = SMART_PRIOR_SPREAD_FACTORS.get(prior_mode)
        copied_entries = [
            DreamParameterEntry.from_dict(entry.to_dict()) for entry in entries
        ]
        if prior_mode == LEGACY_MD_WEIGHT_SMART_PRIOR_MODE:
            return [
                DistributionSetupWindow._legacy_md_weight_adjusted_entry(entry)
                for entry in copied_entries
            ]
        if factor is None:
            return copied_entries
        for entry in copied_entries:
            entry.dist_params = (
                DistributionSetupWindow._adjust_distribution_params(
                    entry,
                    factor=factor,
                )
            )
            entry.smart_preset_status = prior_mode
        return copied_entries

    @staticmethod
    def _rounded_guide_entries(
        entries: list[DreamParameterEntry],
    ) -> list[DreamParameterEntry]:
        return [
            DistributionSetupWindow._rounded_guide_adjusted_entry(entry)
            for entry in entries
        ]

    @classmethod
    def _apply_queue_vary_flags(
        cls,
        entries: list[DreamParameterEntry],
        *,
        scale_offset_mode: str,
        vary_flags: dict[str, object],
    ) -> None:
        scale_offset_enabled = scale_offset_mode == "on"
        for entry in entries:
            param = entry.param.strip()
            if param in {"scale", "offset"}:
                entry.vary = scale_offset_enabled
            elif param == "solv_w":
                entry.vary = bool(vary_flags.get("solv_w", entry.vary))
            elif cls._is_weight_parameter(param):
                entry.vary = bool(vary_flags.get("weights", entry.vary))
            elif cls._is_effective_radius_parameter(param):
                entry.vary = bool(
                    vary_flags.get("effective_radius", entry.vary)
                )
            elif param == "vol_frac":
                entry.vary = bool(vary_flags.get("vol_frac", entry.vary))

    @staticmethod
    def _is_weight_parameter(param: str) -> bool:
        return re.fullmatch(r"w\d+", param.strip()) is not None

    @staticmethod
    def _is_effective_radius_parameter(param: str) -> bool:
        normalized = param.strip().lower()
        return normalized in {"eff_r", "effective_radius"} or (
            "effective_radius" in normalized
        )

    def _load_filter_from_settings(self, *_args: object) -> None:
        try:
            settings = self._settings_from_controls()
        except Exception:
            settings = self._draft_settings
        self._populate_filter_controls(
            PosteriorFilterSettings.from_run_settings(settings)
        )

    def _refresh_filter_preset_combo(
        self,
        *,
        selected_id: str | None = None,
    ) -> None:
        if not hasattr(self, "filter_preset_combo"):
            return
        previous_id = selected_id or self._combo_data(
            self.filter_preset_combo,
            "",
        )
        self._filter_preset_items = {}
        was_blocked = self.filter_preset_combo.blockSignals(True)
        try:
            self.filter_preset_combo.clear()
            for preset in self._built_in_filter_presets():
                preset_id = f"builtin:{preset['id']}"
                self._filter_preset_items[preset_id] = preset
                self.filter_preset_combo.addItem(
                    str(preset["label"]),
                    preset_id,
                )
            for preset in self._saved_filter_presets():
                preset_id = str(preset["id"])
                self._filter_preset_items[preset_id] = preset
                self.filter_preset_combo.addItem(
                    f"{preset['label']} (saved)",
                    preset_id,
                )
            if previous_id:
                self._set_combo_data(self.filter_preset_combo, previous_id)
        finally:
            self.filter_preset_combo.blockSignals(was_blocked)
        self._on_filter_preset_changed(self.filter_preset_combo.currentIndex())

    @staticmethod
    def _built_in_filter_presets() -> list[dict[str, object]]:
        return [
            {
                "id": "all_post_burnin_map",
                "label": "All Post-burnin MAP",
                "description": (
                    "Uses every post-burnin posterior sample and reports the "
                    "MAP best fit. Good as a broad baseline."
                ),
                "settings": PosteriorFilterSettings(
                    bestfit_method="map",
                    posterior_filter_mode="all_post_burnin",
                    violin_parameter_mode="varying_parameters",
                    violin_sample_source="filtered_posterior",
                ),
            },
            {
                "id": "top_10_percent_map",
                "label": "Top 10% MAP",
                "description": (
                    "Keeps the top 10% of samples by log-posterior and uses "
                    "the MAP best fit. Good for comparing sharper posterior "
                    "tails without becoming too sparse."
                ),
                "settings": PosteriorFilterSettings(
                    bestfit_method="map",
                    posterior_filter_mode="top_percent_logp",
                    posterior_top_percent=10.0,
                    posterior_top_n=500,
                    violin_parameter_mode="varying_parameters",
                    violin_sample_source="filtered_posterior",
                ),
            },
            {
                "id": "top_5_percent_map",
                "label": "Top 5% MAP",
                "description": (
                    "Keeps the top 5% of samples by log-posterior and uses "
                    "the MAP best fit. Useful for aggressive fit-quality "
                    "screening."
                ),
                "settings": PosteriorFilterSettings(
                    bestfit_method="map",
                    posterior_filter_mode="top_percent_logp",
                    posterior_top_percent=5.0,
                    posterior_top_n=250,
                    violin_parameter_mode="varying_parameters",
                    violin_sample_source="filtered_posterior",
                ),
            },
            {
                "id": "top_500_chain_mean",
                "label": "Top 500 Chain Mean",
                "description": (
                    "Keeps the top 500 samples by log-posterior and uses the "
                    "chain-mean best fit. Useful when percent-based filters "
                    "leave too many samples."
                ),
                "settings": PosteriorFilterSettings(
                    bestfit_method="chain_mean",
                    posterior_filter_mode="top_n_logp",
                    posterior_top_percent=10.0,
                    posterior_top_n=500,
                    violin_parameter_mode="varying_parameters",
                    violin_sample_source="filtered_posterior",
                ),
            },
            {
                "id": "top_10_percent_median_all_parameters",
                "label": "Top 10% Median All Params",
                "description": (
                    "Keeps the top 10% by log-posterior, reports median "
                    "best-fit parameters, and shows all parameters in violin "
                    "views."
                ),
                "settings": PosteriorFilterSettings(
                    bestfit_method="median",
                    posterior_filter_mode="top_percent_logp",
                    posterior_top_percent=10.0,
                    posterior_top_n=500,
                    violin_parameter_mode="all_parameters",
                    violin_sample_source="filtered_posterior",
                ),
            },
        ]

    def _saved_filter_presets(self) -> list[dict[str, object]]:
        if self._manager is None:
            return []
        preset_dir = self._filter_preset_dir()
        if not preset_dir.is_dir():
            return []
        presets: list[dict[str, object]] = []
        for path in sorted(preset_dir.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                settings_payload = payload.get("posterior_filter_settings")
                if not isinstance(settings_payload, dict):
                    settings_payload = payload.get("settings", {})
                if not isinstance(settings_payload, dict):
                    settings_payload = {}
                settings = PosteriorFilterSettings.from_dict(settings_payload)
            except Exception:
                continue
            label = str(payload.get("label", path.stem)).strip() or path.stem
            description = str(
                payload.get(
                    "description",
                    self._filter_preset_description(settings),
                )
            ).strip()
            presets.append(
                {
                    "id": f"saved:{path.name}",
                    "label": label,
                    "description": description,
                    "settings": settings,
                    "path": path,
                }
            )
        return presets

    def _on_filter_preset_changed(self, _index: int) -> None:
        if not hasattr(self, "filter_preset_combo"):
            return
        preset = self._filter_preset_items.get(
            self._combo_data(self.filter_preset_combo, "")
        )
        description = ""
        if preset is not None:
            description = str(preset.get("description", ""))
        self.filter_preset_combo.setToolTip(description)
        if hasattr(self, "filter_preset_description_label"):
            self.filter_preset_description_label.setText(description)

    def _load_selected_filter_preset(self, *_args: object) -> None:
        try:
            preset = self._filter_preset_items.get(
                self._combo_data(self.filter_preset_combo, "")
            )
            if preset is None:
                raise ValueError("Choose a posterior filter preset first.")
            settings = preset.get("settings")
            if not isinstance(settings, PosteriorFilterSettings):
                raise ValueError("The selected filter preset is invalid.")
            self._populate_filter_controls(settings)
            self.filter_label_edit.setText(
                str(preset.get("label", "")).strip()
            )
            self._set_status(f"Loaded filter preset: {preset['label']}.")
        except Exception as exc:
            QMessageBox.warning(self, "Posterior Filter Preset", str(exc))
            self._set_status("Filter preset load failed.")

    def _save_current_filter_preset(self, *_args: object) -> None:
        try:
            self._require_manager()
            settings = self._filter_settings_from_controls()
            label = (
                self.filter_label_edit.text().strip()
                or self._next_filter_preset_label()
            )
            preset_dir = self._filter_preset_dir()
            preset_dir.mkdir(parents=True, exist_ok=True)
            path = preset_dir / f"{self._safe_preset_filename(label)}.json"
            payload = {
                "format_version": 1,
                "label": label,
                "description": self._filter_preset_description(settings),
                "posterior_filter_settings": settings.to_dict(),
            }
            path.write_text(
                json.dumps(payload, indent=2) + "\n",
                encoding="utf-8",
            )
            self._refresh_filter_preset_combo(selected_id=f"saved:{path.name}")
            self._set_status(f"Saved filter preset: {path}")
        except Exception as exc:
            QMessageBox.warning(self, "Posterior Filter Preset", str(exc))
            self._set_status("Filter preset save failed.")

    def _refresh_filter_list_preset_combo(
        self,
        *,
        selected_id: str | None = None,
    ) -> None:
        if not hasattr(self, "filter_list_preset_combo"):
            return
        previous_id = selected_id or self._combo_data(
            self.filter_list_preset_combo,
            "",
        )
        self._filter_list_preset_items = {}
        was_blocked = self.filter_list_preset_combo.blockSignals(True)
        try:
            self.filter_list_preset_combo.clear()
            presets = self._saved_filter_list_presets()
            if not presets:
                self.filter_list_preset_combo.addItem(
                    "No saved filter lists",
                    "",
                )
            for preset in presets:
                preset_id = str(preset["id"])
                self._filter_list_preset_items[preset_id] = preset
                self.filter_list_preset_combo.addItem(
                    f"{preset['label']} (saved)",
                    preset_id,
                )
            if previous_id:
                self._set_combo_data(
                    self.filter_list_preset_combo,
                    previous_id,
                )
        finally:
            self.filter_list_preset_combo.blockSignals(was_blocked)
        self._on_filter_list_preset_changed(
            self.filter_list_preset_combo.currentIndex()
        )

    def _saved_filter_list_presets(self) -> list[dict[str, object]]:
        if self._manager is None:
            return []
        preset_dir = self._filter_list_preset_dir()
        if not preset_dir.is_dir():
            return []
        presets: list[dict[str, object]] = []
        for path in sorted(preset_dir.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                filter_sets_payload = payload.get("filter_sets", [])
                if not isinstance(filter_sets_payload, list):
                    filter_sets_payload = []
                filter_sets: list[dict[str, object]] = []
                for filter_payload in filter_sets_payload:
                    if not isinstance(filter_payload, dict):
                        continue
                    settings_payload = filter_payload.get(
                        "posterior_filter_settings"
                    )
                    if not isinstance(settings_payload, dict):
                        settings_payload = filter_payload.get("settings", {})
                    if not isinstance(settings_payload, dict):
                        settings_payload = {}
                    settings = PosteriorFilterSettings.from_dict(
                        settings_payload
                    )
                    label = str(filter_payload.get("label", "")).strip()
                    filter_sets.append(
                        {
                            "label": label,
                            "settings": settings,
                        }
                    )
            except Exception:
                continue
            if not filter_sets:
                continue
            label = str(payload.get("label", path.stem)).strip() or path.stem
            description = str(
                payload.get(
                    "description",
                    self._filter_list_preset_description(filter_sets),
                )
            ).strip()
            presets.append(
                {
                    "id": f"saved:{path.name}",
                    "label": label,
                    "description": description,
                    "filter_sets": filter_sets,
                    "path": path,
                }
            )
        return presets

    def _on_filter_list_preset_changed(self, _index: int) -> None:
        if not hasattr(self, "filter_list_preset_combo"):
            return
        preset = self._filter_list_preset_items.get(
            self._combo_data(self.filter_list_preset_combo, "")
        )
        description = ""
        if preset is not None:
            description = str(preset.get("description", ""))
        self.filter_list_preset_combo.setToolTip(description)
        if hasattr(self, "filter_list_preset_description_label"):
            self.filter_list_preset_description_label.setText(description)

    def _load_selected_filter_list_preset(self, *_args: object) -> None:
        try:
            manager = self._require_manager()
            preset = self._filter_list_preset_items.get(
                self._combo_data(self.filter_list_preset_combo, "")
            )
            if preset is None:
                raise ValueError(
                    "Choose a posterior filter list preset first."
                )
            filter_sets = preset.get("filter_sets")
            if not isinstance(filter_sets, list) or not filter_sets:
                raise ValueError("The selected filter list preset is invalid.")
            valid_filter_sets = [
                (
                    str(filter_set.get("label", "")).strip(),
                    filter_set["settings"],
                )
                for filter_set in filter_sets
                if isinstance(
                    filter_set.get("settings"),
                    PosteriorFilterSettings,
                )
            ]
            if not valid_filter_sets:
                raise ValueError("The selected filter list preset is invalid.")
            manager.replace_filter_sets(valid_filter_sets)
            self.filter_list_label_edit.setText(
                str(preset.get("label", "")).strip()
            )
            self._refresh_tables()
            self._refresh_command_box()
            self._set_status(f"Loaded filter list preset: {preset['label']}.")
        except Exception as exc:
            QMessageBox.warning(self, "Posterior Filter List Preset", str(exc))
            self._set_status("Filter list preset load failed.")

    def _save_current_filter_list_preset(self, *_args: object) -> None:
        try:
            manager = self._require_manager()
            if not manager.run_set.filter_sets:
                raise ValueError(
                    "Add at least one posterior filter set before saving a "
                    "filter list preset."
                )
            label = (
                self.filter_list_label_edit.text().strip()
                or self._next_filter_list_preset_label()
            )
            preset_dir = self._filter_list_preset_dir()
            preset_dir.mkdir(parents=True, exist_ok=True)
            path = preset_dir / f"{self._safe_preset_filename(label)}.json"
            filter_sets_payload = [
                {
                    "label": filter_set.label,
                    "posterior_filter_settings": filter_set.settings.to_dict(),
                }
                for filter_set in manager.run_set.filter_sets
            ]
            payload = {
                "format_version": 1,
                "label": label,
                "description": self._filter_list_preset_description(
                    [
                        {
                            "label": filter_set.label,
                            "settings": filter_set.settings,
                        }
                        for filter_set in manager.run_set.filter_sets
                    ]
                ),
                "filter_sets": filter_sets_payload,
            }
            path.write_text(
                json.dumps(payload, indent=2) + "\n",
                encoding="utf-8",
            )
            self._refresh_filter_list_preset_combo(
                selected_id=f"saved:{path.name}"
            )
            self._set_status(f"Saved filter list preset: {path}")
        except Exception as exc:
            QMessageBox.warning(self, "Posterior Filter List Preset", str(exc))
            self._set_status("Filter list preset save failed.")

    def _filter_preset_dir(self) -> Path:
        manager = self._require_manager()
        return (
            Path(manager.workflow.dream_dir)
            / DREAM_BATCH_FILTER_PRESET_DIR_NAME
        )

    def _filter_list_preset_dir(self) -> Path:
        manager = self._require_manager()
        return (
            Path(manager.workflow.dream_dir)
            / DREAM_BATCH_FILTER_LIST_PRESET_DIR_NAME
        )

    def _next_filter_preset_label(self) -> str:
        if self._manager is None:
            return "filter_preset_1"
        existing = {
            str(preset.get("label", ""))
            for preset in self._saved_filter_presets()
        }
        index = 1
        while f"filter_preset_{index}" in existing:
            index += 1
        return f"filter_preset_{index}"

    def _next_filter_list_preset_label(self) -> str:
        if self._manager is None:
            return "filter_list_preset_1"
        existing = {
            str(preset.get("label", ""))
            for preset in self._saved_filter_list_presets()
        }
        index = 1
        while f"filter_list_preset_{index}" in existing:
            index += 1
        return f"filter_list_preset_{index}"

    @staticmethod
    def _safe_preset_filename(label: str) -> str:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", label.strip()).strip("._")
        return safe or "filter_preset"

    @staticmethod
    def _filter_preset_description(settings: PosteriorFilterSettings) -> str:
        mode = settings.posterior_filter_mode
        if mode == "top_percent_logp":
            filter_text = (
                f"top {settings.posterior_top_percent:.3g}% by log-posterior"
            )
        elif mode == "top_n_logp":
            filter_text = f"top {settings.posterior_top_n} by log-posterior"
        else:
            filter_text = "all post-burnin samples"
        return (
            f"Uses {filter_text}, {settings.bestfit_method} best-fit "
            f"selection, {settings.violin_parameter_mode} violin data, and "
            f"{settings.credible_interval_low:.3g}-"
            f"{settings.credible_interval_high:.3g}% credible intervals."
        )

    @staticmethod
    def _filter_stoichiometry_status_text(
        settings: PosteriorFilterSettings,
    ) -> str:
        elements = settings.stoichiometry_target_elements_text.strip()
        ratio = settings.stoichiometry_target_ratio_text.strip()
        target = ""
        if elements or ratio:
            target = f"{elements or '?'} = {ratio or '?'}"
        if settings.stoichiometry_filter_enabled:
            status = "On"
            if target:
                status = f"{status}: {target}"
            status = (
                f"{status} (+/- {settings.stoichiometry_tolerance_percent:g}%)"
            )
            return status
        if target:
            return f"Target only: {target}"
        return "Off"

    @staticmethod
    def _filter_violin_data_text(settings: PosteriorFilterSettings) -> str:
        mode_labels = {
            "varying_parameters": "Varying Parameters",
            "all_parameters": "All Parameters",
            "weights_only": "Weights Only",
            "fit_parameters": "Fit Parameters",
            "effective_radii_only": "Effective Radii Only",
            "additional_parameters_only": "Additional Parameters Only",
            "selected_additional_parameter": "Selected Additional Parameter",
        }
        source_labels = {
            "filtered_posterior": "Filtered Posterior",
            "map_chain_only": "MAP Chain Only",
        }
        mode = mode_labels.get(
            settings.violin_parameter_mode,
            settings.violin_parameter_mode,
        )
        if (
            settings.violin_parameter_mode == "selected_additional_parameter"
            and settings.violin_selected_parameter.strip()
        ):
            mode = f"{mode}: {settings.violin_selected_parameter.strip()}"
        source = source_labels.get(
            settings.violin_sample_source,
            settings.violin_sample_source,
        )
        return f"{mode} / {source}"

    @classmethod
    def _filter_list_preset_description(
        cls,
        filter_sets: list[dict[str, object]],
    ) -> str:
        labels: list[str] = []
        for index, filter_set in enumerate(filter_sets, start=1):
            settings = filter_set.get("settings")
            if not isinstance(settings, PosteriorFilterSettings):
                continue
            label = str(filter_set.get("label", "")).strip() or (
                f"Filter {index}"
            )
            labels.append(
                f"{label}: {cls._filter_preset_description(settings)}"
            )
        if not labels:
            return "No posterior filter sets."
        return f"{len(labels)} posterior filter set(s): " + " | ".join(labels)

    def _add_filter_set(self, *_args: object) -> None:
        try:
            manager = self._require_manager()
            settings = self._filter_settings_from_controls()
            self._set_filter_json(settings)
            filter_set = manager.add_filter_set(
                label=self.filter_label_edit.text().strip(),
                settings=settings,
            )
            self.filter_label_edit.setText(
                f"filter_{len(manager.run_set.filter_sets) + 1}"
            )
            self._refresh_tables()
            self._refresh_command_box()
            self._set_status(f"Added posterior filter set: {filter_set.label}")
        except Exception as exc:
            QMessageBox.warning(self, "DREAM Backend Batch Setup", str(exc))
            self._set_status("Posterior filter set creation failed.")

    def _remove_selected_filter_set(self, *_args: object) -> None:
        try:
            manager = self._require_manager()
            filter_id = self._selected_filter_set_id()
            if not filter_id:
                raise ValueError("Select a posterior filter set to remove.")
            removed = manager.remove_filter_set(filter_id)
            self._refresh_tables()
            self._refresh_command_box()
            self._set_status(f"Removed posterior filter set: {removed.label}")
        except Exception as exc:
            QMessageBox.warning(self, "Remove Posterior Filter Set", str(exc))
            self._set_status("Posterior filter set removal failed.")

    def _generate_shell_script(self, *_args: object) -> None:
        try:
            manager = self._require_manager()
            if not manager.run_set.queue_items:
                raise ValueError("Create at least one DREAM queue item first.")
            if not manager.run_set.filter_sets:
                settings = self._filter_settings_from_controls()
                manager.add_filter_set(
                    label=self.filter_label_edit.text().strip()
                    or "default_filter",
                    settings=settings,
                )
            script_path, commands_path = manager.generate_shell_script(
                new_run_set_dir=True
            )
            self.run_set_edit.setText(
                str(manager.run_set.resolved_run_set_dir)
            )
            self._refresh_tables()
            self._refresh_command_box()
            self._set_status(f"Generated shell script: {script_path}")
            QMessageBox.information(
                self,
                "DREAM Backend Batch Setup",
                (
                    "Generated DREAM backend shell script:\n"
                    f"{script_path}\n\nCommands:\n{commands_path}"
                ),
            )
        except Exception as exc:
            QMessageBox.warning(self, "DREAM Backend Batch Setup", str(exc))
            self._set_status("Shell script generation failed.")

    def _reveal_command_set_file(self, *_args: object) -> None:
        try:
            manager = self._require_manager()
            commands_path = manager.run_set.commands_path
            if not commands_path.is_file():
                raise ValueError(
                    "Generate the shell script before opening the command "
                    "set TXT file."
                )
            self._reveal_path_in_file_browser(commands_path)
            self._set_status(f"Opened command set TXT: {commands_path}")
        except Exception as exc:
            QMessageBox.warning(self, "DREAM Backend Batch Setup", str(exc))
            self._set_status("Could not open command set TXT.")

    def _validate_settings_json(self, *_args: object) -> None:
        try:
            settings = self._settings_from_json(self.settings_json_edit)
        except Exception as exc:
            QMessageBox.warning(self, "DREAM Settings JSON", str(exc))
            return
        self._draft_settings = settings
        self._populate_settings_controls(settings)
        QMessageBox.information(
            self,
            "DREAM Settings JSON",
            "The DREAM settings JSON was applied to the controls.",
        )

    def _validate_filter_json(self, *_args: object) -> None:
        try:
            settings = self._filter_settings_from_json(self.filter_json_edit)
        except Exception as exc:
            QMessageBox.warning(self, "Posterior Filter JSON", str(exc))
            return
        self._populate_filter_controls(settings)
        QMessageBox.information(
            self,
            "Posterior Filter JSON",
            "The posterior filter JSON was applied to the controls.",
        )

    def _sync_conda_env(self, *_args: object) -> None:
        if self._manager is None:
            return
        self._manager.run_set.conda_env = (
            self.conda_env_edit.text().strip() or DEFAULT_DREAM_BATCH_CONDA_ENV
        )
        self._manager.save_manifest()
        self._refresh_command_box()

    def _refresh_prior_table(self) -> None:
        self._populate_prior_table(self._draft_parameter_entries)

    def _populate_prior_table(
        self,
        entries: list[DreamParameterEntry],
        *,
        summary_prefix: str | None = None,
    ) -> None:
        varying_count = sum(1 for entry in entries if entry.vary)
        summary = f"{len(entries)} prior entries; {varying_count} set to vary."
        if summary_prefix:
            summary = f"{summary_prefix}: {summary}"
        self.prior_summary_label.setText(summary)
        self.prior_table.setRowCount(len(entries))
        for row, entry in enumerate(entries):
            guide_low, guide_high, guide_tooltip = self._prior_guide_text(
                entry
            )
            values = [
                entry.param,
                entry.structure,
                "Yes" if entry.vary else "No",
                entry.distribution,
                json.dumps(entry.dist_params, sort_keys=True),
                guide_low,
                guide_high,
            ]
            for column, value in enumerate(values):
                table_item = QTableWidgetItem(value)
                if column in {5, 6}:
                    table_item.setToolTip(guide_tooltip)
                self.prior_table.setItem(row, column, table_item)
        self.prior_table.resizeColumnsToContents()

    def _refresh_prefit_preview(self) -> None:
        if self._manager is None:
            self._clear_prefit_preview(
                "Choose a ready SAXSShell project first."
            )
            return
        workflow = self._manager.workflow.prefit_workflow
        entries = self._draft_prefit_entries or workflow.parameter_entries
        self._refresh_prefit_parameter_table(entries)
        try:
            evaluation = workflow.evaluate(entries)
        except Exception as exc:
            self._clear_prefit_plot()
            self.prefit_preview_status_label.setText(
                f"Unable to evaluate active Prefit: {exc}"
            )
            return

        self._clear_prefit_plot()
        axis = self.prefit_preview_figure.add_subplot(111)
        plotted_lines = False
        q_values = np.asarray(evaluation.q_values, dtype=float)
        if q_values.size:
            model_q_min = float(np.min(q_values))
            model_q_max = float(np.max(q_values))
            if (
                self._pending_fit_q_range is None
                and not self.fit_q_min_spin.isEnabled()
            ):
                self._set_fit_q_range_controls(
                    fit_q_min=evaluation.fit_q_min,
                    fit_q_max=evaluation.fit_q_max,
                    model_q_min=model_q_min,
                    model_q_max=model_q_max,
                )
                settings = self._settings_from_controls()
                self._set_settings_json(settings, update_controls=False)
            elif self.fit_q_min_spin.isEnabled():
                current_q_min, current_q_max = (
                    self._fit_q_range_from_controls()
                )
                self._set_fit_q_range_controls(
                    fit_q_min=current_q_min,
                    fit_q_max=current_q_max,
                    model_q_min=model_q_min,
                    model_q_max=model_q_max,
                )
                settings = self._settings_from_controls()
                self._set_settings_json(settings, update_controls=False)
        model_values = np.asarray(evaluation.model_intensities, dtype=float)
        model_mask = np.isfinite(q_values) & np.isfinite(model_values)
        if np.any(model_mask):
            axis.plot(
                q_values[model_mask],
                model_values[model_mask],
                color="tab:red",
                linewidth=1.7,
                label="Active Prefit",
            )
            plotted_lines = True
        experimental = evaluation.experimental_intensities
        if experimental is not None:
            experimental_values = np.asarray(experimental, dtype=float)
            experimental_mask = np.isfinite(q_values) & np.isfinite(
                experimental_values
            )
            if np.any(experimental_mask):
                axis.plot(
                    q_values[experimental_mask],
                    experimental_values[experimental_mask],
                    linestyle="",
                    marker="o",
                    markersize=2.4,
                    color="black",
                    alpha=0.7,
                    label="Experimental",
                )
                plotted_lines = True

        axis.set_xlabel(Q_A_INVERSE_LABEL)
        axis.set_ylabel("Intensity (arb. units)")
        axis.grid(True, which="both", alpha=0.25)
        if self._prefit_preview_has_positive_q_values(evaluation):
            axis.set_xscale("log")
        if self._prefit_preview_has_positive_intensities(evaluation):
            axis.set_yscale("log")
        if plotted_lines:
            axis.legend(loc="best")
        self.prefit_preview_figure.tight_layout()
        self.prefit_preview_canvas.draw_idle()

        varying_count = sum(
            1
            for entry in entries
            if entry.vary and self._prefit_entry_active(entry)
        )
        active_weight_count = sum(
            1
            for entry in entries
            if self._is_prefit_weight_entry(entry)
            and self._prefit_entry_active(entry)
        )
        status = (
            f"{workflow.template_spec.name}; {len(entries)} parameters; "
            f"{varying_count} varying; {active_weight_count} active weights; "
            f"{len(q_values)} q points."
        )
        if evaluation.fitted_stoichiometry_text:
            status = f"{status}\n{evaluation.fitted_stoichiometry_text}"
        self.prefit_preview_status_label.setText(status)

    def _refresh_prefit_parameter_table(
        self,
        entries: list[PrefitParameterEntry],
    ) -> None:
        entry_list = list(entries)
        self.prefit_parameter_table.setRowCount(len(entry_list))
        for row, entry in enumerate(entry_list):
            values = [
                entry.name,
                entry.structure,
                entry.motif,
                self._format_table_number(entry.value),
                "Yes" if entry.vary else "No",
                self._format_table_number(entry.minimum),
                self._format_table_number(entry.maximum),
                entry.category,
                "",
            ]
            for column, value in enumerate(values):
                if column == 8:
                    continue
                self.prefit_parameter_table.setItem(
                    row,
                    column,
                    QTableWidgetItem(str(value)),
                )
            self._set_prefit_active_widget(row, entry)
        self.prefit_parameter_table.resizeColumnsToContents()

    def _set_prefit_active_widget(
        self,
        row: int,
        entry: PrefitParameterEntry,
    ) -> None:
        if not self._is_prefit_weight_entry(entry):
            item = QTableWidgetItem("")
            item.setToolTip("Only component weights w<NN> can be toggled.")
            self.prefit_parameter_table.setItem(row, 8, item)
            return
        button = QPushButton()
        button.setCheckable(True)
        button.setChecked(self._prefit_entry_active(entry))
        self._update_prefit_active_button(button)
        button.clicked.connect(
            lambda checked=False, row_index=row, control=button: (
                self._on_prefit_active_toggled(row_index, control)
            )
        )
        self.prefit_parameter_table.setCellWidget(row, 8, button)

    def _on_prefit_active_toggled(
        self,
        row: int,
        button: QPushButton,
    ) -> None:
        if row < 0 or row >= len(self._draft_prefit_entries):
            return
        entry = self._draft_prefit_entries[row]
        if not self._is_prefit_weight_entry(entry):
            return
        entry.active = bool(button.isChecked())
        self._update_prefit_active_button(button)
        self._rebuild_draft_priors_from_prefit()
        self._refresh_prefit_preview()
        self._set_status(
            f"{entry.name} {'enabled' if entry.active else 'disabled'} "
            "for the next DREAM queue item."
        )

    @staticmethod
    def _update_prefit_active_button(button: QPushButton) -> None:
        enabled = bool(button.isChecked())
        button.setText("On" if enabled else "Off")
        button.setToolTip(
            "This component weight is included in this queue draft."
            if enabled
            else "This component weight is excluded from this queue draft."
        )

    @staticmethod
    def _is_prefit_weight_entry(entry: PrefitParameterEntry) -> bool:
        name = str(entry.name).strip()
        return (
            str(entry.category).strip() == "weight"
            and name.startswith("w")
            and name[1:].isdigit()
        )

    @classmethod
    def _prefit_entry_active(cls, entry: PrefitParameterEntry) -> bool:
        return (not cls._is_prefit_weight_entry(entry)) or bool(
            getattr(entry, "active", True)
        )

    def _clear_prefit_preview(self, message: str) -> None:
        if hasattr(self, "prefit_parameter_table"):
            self.prefit_parameter_table.setRowCount(0)
        if hasattr(self, "prefit_preview_status_label"):
            self.prefit_preview_status_label.setText(message)
        self._clear_prefit_plot()

    def _clear_prefit_plot(self) -> None:
        if not hasattr(self, "prefit_preview_figure"):
            return
        self.prefit_preview_figure.clear()
        self.prefit_preview_canvas.draw_idle()

    @staticmethod
    def _prefit_preview_has_positive_intensities(
        evaluation: PrefitEvaluation,
    ) -> bool:
        arrays = [np.asarray(evaluation.model_intensities, dtype=float)]
        if evaluation.experimental_intensities is not None:
            arrays.append(
                np.asarray(evaluation.experimental_intensities, dtype=float)
            )
        finite_values: list[np.ndarray] = []
        for values in arrays:
            mask = np.isfinite(values)
            if np.any(mask):
                finite_values.append(values[mask])
        if not finite_values:
            return False
        plotted_values = np.concatenate(finite_values)
        return bool(np.all(plotted_values > 0.0))

    @staticmethod
    def _prefit_preview_has_positive_q_values(
        evaluation: PrefitEvaluation,
    ) -> bool:
        q_values = np.asarray(evaluation.q_values, dtype=float)
        mask = np.isfinite(q_values)
        return bool(np.any(mask) and np.all(q_values[mask] > 0.0))

    @staticmethod
    def _format_table_number(value: object) -> str:
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return str(value)
        if not np.isfinite(numeric_value):
            return str(numeric_value)
        return f"{numeric_value:.6g}"

    @staticmethod
    def _prior_guide_text(
        entry: DreamParameterEntry,
    ) -> tuple[str, str, str]:
        guide_tooltip = "Guide bounds are unavailable for this prior."
        try:
            guide_low, guide_high, guide_kind = distribution_guide_bounds(
                entry
            )
        except Exception:
            return "n/a", "n/a", guide_tooltip
        if guide_low is None or guide_high is None:
            return "n/a", "n/a", guide_tooltip
        return (
            format_distribution_guide_value(guide_low),
            format_distribution_guide_value(guide_high),
            f"{guide_kind} for the current {entry.distribution} prior.",
        )

    def _refresh_tables(
        self,
        *,
        selected_queue_item_id: str | None = None,
        selected_filter_set_id: str | None = None,
    ) -> None:
        if self._manager is None:
            self.queue_table.setRowCount(0)
            self.filter_table.setRowCount(0)
            self._clear_queue_item_view(
                "Choose a ready SAXSShell project first."
            )
            return
        run_set = self._manager.run_set
        selected_item_id = (
            selected_queue_item_id or self._selected_queue_item_id()
        )
        was_updating_queue = self._updating_queue_table
        self._updating_queue_table = True
        try:
            self.queue_table.setRowCount(len(run_set.queue_items))
            for row, item in enumerate(run_set.queue_items):
                fit_q_min, fit_q_max = self._queue_item_fit_q_range(item)
                values = [
                    item.label,
                    item.status,
                    self._format_queue_fit_q_value(fit_q_min),
                    self._format_queue_fit_q_value(fit_q_max),
                    queue_item_weight_state_summary(item),
                    item.run_dir,
                    item.created_at,
                ]
                for column, value in enumerate(values):
                    table_item = QTableWidgetItem(value)
                    table_item.setData(Qt.ItemDataRole.UserRole, item.item_id)
                    table_item.setToolTip(
                        self._queue_table_cell_tooltip(item, column)
                    )
                    if not self._queue_table_cell_is_editable(item, column):
                        table_item.setFlags(
                            table_item.flags() & ~Qt.ItemFlag.ItemIsEditable
                        )
                    self.queue_table.setItem(row, column, table_item)
        finally:
            self._updating_queue_table = was_updating_queue
        self.queue_table.resizeColumnsToContents()
        if run_set.queue_items:
            row_to_select = 0
            if selected_item_id:
                for row, item in enumerate(run_set.queue_items):
                    if item.item_id == selected_item_id:
                        row_to_select = row
                        break
            self.queue_table.selectRow(row_to_select)
            self._refresh_selected_queue_item_view()
        else:
            self._clear_queue_item_view("No queued DREAM runs yet.")

        selected_filter_id = (
            selected_filter_set_id or self._selected_filter_set_id()
        )
        was_updating = self._updating_filter_table
        self._updating_filter_table = True
        try:
            self.filter_table.setRowCount(0)
            self.filter_table.setRowCount(len(run_set.filter_sets))
            for row, filter_set in enumerate(run_set.filter_sets):
                self._populate_filter_table_row(row, filter_set)
        finally:
            self._updating_filter_table = was_updating
        self.filter_table.resizeColumnsToContents()
        if run_set.filter_sets:
            row_to_select = 0
            if selected_filter_id:
                for row, filter_set in enumerate(run_set.filter_sets):
                    if filter_set.filter_id == selected_filter_id:
                        row_to_select = row
                        break
            self.filter_table.selectRow(row_to_select)

    def _queue_item_fit_q_range(
        self,
        item: object,
    ) -> tuple[float | None, float | None]:
        try:
            settings = load_dream_settings(str(getattr(item, "settings_path")))
        except Exception:
            return (None, None)
        return settings.fit_q_min, settings.fit_q_max

    @staticmethod
    def _format_queue_fit_q_value(value: object) -> str:
        if value is None:
            return "full"
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return str(value)
        if not np.isfinite(numeric_value):
            return str(value)
        return f"{numeric_value:.6g}"

    def _queue_table_cell_tooltip(self, item: object, column: int) -> str:
        if column in {self.QUEUE_COL_FIT_Q_MIN, self.QUEUE_COL_FIT_Q_MAX}:
            if str(getattr(item, "status", "")).strip() == "queued":
                return (
                    "Edit this queued item q-bound directly. Use 'full' or "
                    "leave blank to inherit the computed model limit."
                )
            return "Completed or running queue items cannot be edited."
        if column == self.QUEUE_COL_WEIGHTS:
            if str(getattr(item, "status", "")).strip() == "queued":
                return (
                    "Edit per-run weight use directly, e.g. "
                    "'w0=on; w1=off'. Inactive weights remain listed here."
                )
            return "Completed or running queue items cannot be edited."
        return ""

    def _queue_table_cell_is_editable(self, item: object, column: int) -> bool:
        return str(
            getattr(item, "status", "")
        ).strip() == "queued" and column in {
            self.QUEUE_COL_FIT_Q_MIN,
            self.QUEUE_COL_FIT_Q_MAX,
            self.QUEUE_COL_WEIGHTS,
        }

    def _on_queue_table_item_changed(
        self,
        item: QTableWidgetItem,
    ) -> None:
        editable_columns = {
            self.QUEUE_COL_FIT_Q_MIN,
            self.QUEUE_COL_FIT_Q_MAX,
            self.QUEUE_COL_WEIGHTS,
        }
        if self._updating_queue_table or item.column() not in editable_columns:
            return
        item_id = str(item.data(Qt.ItemDataRole.UserRole) or "").strip()
        if not item_id:
            return
        try:
            manager = self._require_manager()
            queue_item = next(
                (
                    queued
                    for queued in manager.run_set.queue_items
                    if queued.item_id == item_id
                ),
                None,
            )
            if queue_item is None:
                raise ValueError(
                    "The edited queue item is no longer available."
                )
            if item.column() == self.QUEUE_COL_WEIGHTS:
                active_weight_names = self._parse_queue_weight_state_text(
                    item.text(),
                    queue_item,
                )
                updated = manager.update_queue_item_active_weights(
                    item_id,
                    active_weight_names=active_weight_names,
                )
                self._refresh_tables(selected_queue_item_id=updated.item_id)
                self._refresh_command_box()
                self._set_status(
                    "Updated queued DREAM weight use for item: "
                    f"{updated.label}"
                )
                return
            current_min, current_max = self._queue_item_fit_q_range(queue_item)
            fit_q_min = (
                self._parse_queue_fit_q_value(item.text())
                if item.column() == self.QUEUE_COL_FIT_Q_MIN
                else current_min
            )
            fit_q_max = (
                self._parse_queue_fit_q_value(item.text())
                if item.column() == self.QUEUE_COL_FIT_Q_MAX
                else current_max
            )
            updated = manager.update_queue_item_fit_range(
                item_id,
                fit_q_min=fit_q_min,
                fit_q_max=fit_q_max,
            )
            self._refresh_tables(selected_queue_item_id=updated.item_id)
            self._refresh_command_box()
            self._set_status(
                "Updated DREAM fit q-range for queued item: "
                f"{updated.label}"
            )
        except Exception as exc:
            QMessageBox.warning(self, "Queued DREAM Run", str(exc))
            self._refresh_tables(selected_queue_item_id=item_id)
            self._refresh_command_box()
            self._set_status("Queued DREAM fit q-range update failed.")

    @staticmethod
    def _parse_queue_fit_q_value(value: object) -> float | None:
        text = str(value or "").strip()
        if not text or text.lower() in {"full", "none", "auto", "inherit"}:
            return None
        numeric_value = float(text)
        if not np.isfinite(numeric_value):
            raise ValueError("DREAM fit q bounds must be finite.")
        return numeric_value

    @staticmethod
    def _parse_queue_weight_state_text(
        value: object,
        item: object,
    ) -> set[str]:
        text = str(value or "").strip()
        states = queue_item_weight_states(item)
        known_names = {str(state.get("name", "")).strip() for state in states}
        known_names.discard("")
        if not known_names:
            raise ValueError("This queued DREAM run has no editable weights.")
        active_names: set[str] = set()
        explicit_state_seen = False
        for match in re.finditer(
            r"\b(w\d+)\b\s*(?:=|:)?\s*" r"\b(on|off|yes|no|true|false|1|0)\b",
            text,
            flags=re.IGNORECASE,
        ):
            explicit_state_seen = True
            name = match.group(1)
            state_text = match.group(2).lower()
            if name not in known_names:
                raise ValueError(f"Unknown queued DREAM weight: {name}")
            if state_text in {"on", "yes", "true", "1"}:
                active_names.add(name)
        if not explicit_state_seen:
            for match in re.finditer(r"\b(w\d+)\b", text):
                name = match.group(1)
                if name not in known_names:
                    raise ValueError(f"Unknown queued DREAM weight: {name}")
                active_names.add(name)
        if not active_names:
            raise ValueError(
                "At least one queued DREAM weight must remain on."
            )
        return active_names

    def _populate_filter_table_row(
        self,
        row: int,
        filter_set: DreamBatchFilterSet,
    ) -> None:
        settings = filter_set.settings
        label_item = QTableWidgetItem(filter_set.label)
        label_item.setData(Qt.ItemDataRole.UserRole, filter_set.filter_id)
        label_item.setToolTip("Edit this label directly in the table.")
        self.filter_table.setItem(row, self.FILTER_COL_LABEL, label_item)

        self.filter_table.setCellWidget(
            row,
            self.FILTER_COL_POSTERIOR_MODE,
            self._make_filter_table_combo(
                filter_set.filter_id,
                self.FILTER_POSTERIOR_MODE_OPTIONS,
                settings.posterior_filter_mode,
            ),
        )
        self.filter_table.setCellWidget(
            row,
            self.FILTER_COL_TOP_PERCENT,
            self._make_filter_table_top_percent_spin(
                filter_set.filter_id,
                settings.posterior_top_percent,
            ),
        )
        self.filter_table.setCellWidget(
            row,
            self.FILTER_COL_TOP_N,
            self._make_filter_table_top_n_spin(
                filter_set.filter_id,
                settings.posterior_top_n,
            ),
        )
        self.filter_table.setCellWidget(
            row,
            self.FILTER_COL_BESTFIT,
            self._make_filter_table_combo(
                filter_set.filter_id,
                self.FILTER_BESTFIT_OPTIONS,
                settings.bestfit_method,
            ),
        )
        self.filter_table.setCellWidget(
            row,
            self.FILTER_COL_VIOLIN_MODE,
            self._make_filter_table_combo(
                filter_set.filter_id,
                self.FILTER_VIOLIN_MODE_OPTIONS,
                settings.violin_parameter_mode,
            ),
        )
        self.filter_table.setCellWidget(
            row,
            self.FILTER_COL_VIOLIN_SAMPLE_SOURCE,
            self._make_filter_table_combo(
                filter_set.filter_id,
                self.FILTER_VIOLIN_SAMPLE_SOURCE_OPTIONS,
                settings.violin_sample_source,
            ),
        )
        self.filter_table.setCellWidget(
            row,
            self.FILTER_COL_VIOLIN_VALUE_SCALE,
            self._make_filter_table_combo(
                filter_set.filter_id,
                self.FILTER_VIOLIN_VALUE_SCALE_OPTIONS,
                settings.violin_value_scale_mode,
            ),
        )
        self._set_filter_table_read_only_item(
            row,
            self.FILTER_COL_STOICHIOMETRY,
            self._filter_stoichiometry_status_text(settings),
            filter_set.filter_id,
        )
        self._set_filter_table_read_only_item(
            row,
            self.FILTER_COL_CREATED,
            filter_set.created_at,
            filter_set.filter_id,
        )
        self._refresh_filter_table_row_state(row)

    def _set_filter_table_read_only_item(
        self,
        row: int,
        column: int,
        text: str,
        filter_id: str,
    ) -> None:
        table_item = QTableWidgetItem(text)
        table_item.setData(Qt.ItemDataRole.UserRole, filter_id)
        table_item.setFlags(table_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.filter_table.setItem(row, column, table_item)

    def _make_filter_table_combo(
        self,
        filter_id: str,
        options: tuple[tuple[str, str], ...],
        value: str,
    ) -> QComboBox:
        combo = QComboBox(self.filter_table)
        for label, data in options:
            combo.addItem(label, data)
        self._set_combo_data(combo, value)
        combo.currentIndexChanged.connect(
            lambda _index, filter_id=filter_id: (
                self._on_filter_table_widget_changed(filter_id)
            )
        )
        return combo

    def _make_filter_table_top_percent_spin(
        self,
        filter_id: str,
        value: float,
    ) -> QDoubleSpinBox:
        spin = _WheelGuardedDoubleSpinBox(self.filter_table)
        spin.setRange(0.1, 100.0)
        spin.setDecimals(2)
        spin.setSingleStep(1.0)
        spin.setValue(float(value))
        spin.valueChanged.connect(
            lambda _value, filter_id=filter_id: (
                self._on_filter_table_widget_changed(filter_id)
            )
        )
        return spin

    def _make_filter_table_top_n_spin(
        self,
        filter_id: str,
        value: int,
    ) -> QSpinBox:
        spin = _WheelGuardedSpinBox(self.filter_table)
        spin.setRange(1, 10_000_000)
        spin.setValue(int(value))
        spin.valueChanged.connect(
            lambda _value, filter_id=filter_id: (
                self._on_filter_table_widget_changed(filter_id)
            )
        )
        return spin

    def _on_filter_table_item_changed(
        self,
        item: QTableWidgetItem,
    ) -> None:
        if (
            self._updating_filter_table
            or item.column() != self.FILTER_COL_LABEL
        ):
            return
        filter_id = str(item.data(Qt.ItemDataRole.UserRole) or "").strip()
        if not filter_id:
            return
        try:
            manager = self._require_manager()
            filter_set = self._filter_set_by_id(filter_id)
            label = item.text().strip()
            if not label:
                label = filter_set.label
                was_updating = self._updating_filter_table
                self._updating_filter_table = True
                try:
                    item.setText(label)
                finally:
                    self._updating_filter_table = was_updating
            updated = manager.update_filter_set(filter_id, label=label)
            self._refresh_command_box()
            self._set_status(f"Updated posterior filter set: {updated.label}")
        except Exception as exc:
            QMessageBox.warning(self, "Posterior Filter Set", str(exc))
            self._set_status("Posterior filter set update failed.")

    def _on_filter_table_widget_changed(self, filter_id: str) -> None:
        if self._updating_filter_table:
            return
        try:
            manager = self._require_manager()
            row = self._filter_table_row_for_filter_id(filter_id)
            if row < 0:
                raise ValueError(
                    "The edited posterior filter set is no longer available."
                )
            filter_set = self._filter_set_by_id(filter_id)
            settings = self._filter_table_settings_from_row(
                row,
                filter_set.settings,
            )
            label = self._filter_table_label(row, filter_set.label)
            updated = manager.update_filter_set(
                filter_id,
                label=label,
                settings=settings,
            )
            self.filter_table.selectRow(row)
            self._refresh_filter_table_row_state(row)
            self._set_filter_table_read_only_item(
                row,
                self.FILTER_COL_STOICHIOMETRY,
                self._filter_stoichiometry_status_text(updated.settings),
                filter_id,
            )
            self._refresh_command_box()
            self._set_status(f"Updated posterior filter set: {updated.label}")
        except Exception as exc:
            QMessageBox.warning(self, "Posterior Filter Set", str(exc))
            self._set_status("Posterior filter set update failed.")

    def _filter_table_settings_from_row(
        self,
        row: int,
        current_settings: PosteriorFilterSettings,
    ) -> PosteriorFilterSettings:
        settings = PosteriorFilterSettings.from_dict(
            current_settings.to_dict()
        )
        settings.posterior_filter_mode = self._filter_table_combo_data(
            row,
            self.FILTER_COL_POSTERIOR_MODE,
            settings.posterior_filter_mode,
        )
        settings.posterior_top_percent = float(
            self._filter_table_spin_value(
                row,
                self.FILTER_COL_TOP_PERCENT,
                settings.posterior_top_percent,
            )
        )
        settings.posterior_top_n = int(
            self._filter_table_spin_value(
                row,
                self.FILTER_COL_TOP_N,
                settings.posterior_top_n,
            )
        )
        settings.bestfit_method = self._filter_table_combo_data(
            row,
            self.FILTER_COL_BESTFIT,
            settings.bestfit_method,
        )
        settings.violin_parameter_mode = self._filter_table_combo_data(
            row,
            self.FILTER_COL_VIOLIN_MODE,
            settings.violin_parameter_mode,
        )
        settings.violin_sample_source = self._filter_table_combo_data(
            row,
            self.FILTER_COL_VIOLIN_SAMPLE_SOURCE,
            settings.violin_sample_source,
        )
        settings.violin_value_scale_mode = self._filter_table_combo_data(
            row,
            self.FILTER_COL_VIOLIN_VALUE_SCALE,
            settings.violin_value_scale_mode,
        )
        return settings

    def _filter_table_combo_data(
        self,
        row: int,
        column: int,
        default: str,
    ) -> str:
        widget = self.filter_table.cellWidget(row, column)
        if isinstance(widget, QComboBox):
            return self._combo_data(widget, default)
        return default

    def _filter_table_spin_value(
        self,
        row: int,
        column: int,
        default: float,
    ) -> float:
        widget = self.filter_table.cellWidget(row, column)
        if isinstance(widget, (QDoubleSpinBox, QSpinBox)):
            return float(widget.value())
        return float(default)

    def _refresh_filter_table_row_state(self, row: int) -> None:
        mode = self._filter_table_combo_data(
            row,
            self.FILTER_COL_POSTERIOR_MODE,
            "all_post_burnin",
        )
        top_percent_widget = self.filter_table.cellWidget(
            row,
            self.FILTER_COL_TOP_PERCENT,
        )
        if top_percent_widget is not None:
            top_percent_widget.setEnabled(mode == "top_percent_logp")
        top_n_widget = self.filter_table.cellWidget(
            row,
            self.FILTER_COL_TOP_N,
        )
        if top_n_widget is not None:
            top_n_widget.setEnabled(mode == "top_n_logp")

    def _filter_table_label(self, row: int, fallback: str) -> str:
        item = self.filter_table.item(row, self.FILTER_COL_LABEL)
        if item is None:
            return fallback
        return item.text().strip() or fallback

    def _filter_table_row_for_filter_id(self, filter_id: str) -> int:
        filter_id = str(filter_id or "").strip()
        for row in range(self.filter_table.rowCount()):
            item = self.filter_table.item(row, self.FILTER_COL_LABEL)
            if item is None:
                continue
            item_filter_id = str(
                item.data(Qt.ItemDataRole.UserRole) or ""
            ).strip()
            if item_filter_id == filter_id:
                return row
        return -1

    def _filter_set_by_id(self, filter_id: str) -> DreamBatchFilterSet:
        manager = self._require_manager()
        filter_id = str(filter_id or "").strip()
        for filter_set in manager.run_set.filter_sets:
            if filter_set.filter_id == filter_id:
                return filter_set
        raise ValueError(f"DREAM posterior filter set not found: {filter_id}")

    def _selected_filter_set_id(self) -> str | None:
        if not hasattr(self, "filter_table"):
            return None
        selected_rows = self.filter_table.selectionModel().selectedRows()
        if selected_rows:
            row = selected_rows[0].row()
        else:
            row = self.filter_table.currentRow()
        if row < 0:
            return None
        item = self.filter_table.item(row, 0)
        if item is None:
            return None
        value = item.data(Qt.ItemDataRole.UserRole)
        text = str(value or "").strip()
        return text or None

    def _selected_queue_item_id(self) -> str | None:
        if not hasattr(self, "queue_table"):
            return None
        selected_rows = self.queue_table.selectionModel().selectedRows()
        if selected_rows:
            row = selected_rows[0].row()
        else:
            row = self.queue_table.currentRow()
        if row < 0:
            return None
        item = self.queue_table.item(row, 0)
        if item is None:
            return None
        value = item.data(Qt.ItemDataRole.UserRole)
        text = str(value or "").strip()
        return text or None

    def _refresh_selected_queue_item_view(self) -> None:
        if self._manager is None:
            self._clear_queue_item_view(
                "Choose a ready SAXSShell project first."
            )
            return
        item_id = self._selected_queue_item_id()
        if not item_id:
            self._clear_queue_item_view("Select a queued DREAM run.")
            return
        queue_item = next(
            (
                item
                for item in self._manager.run_set.queue_items
                if item.item_id == item_id
            ),
            None,
        )
        if queue_item is None:
            self._clear_queue_item_view("Selected queue item is unavailable.")
            return
        fit_q_min, fit_q_max = self._queue_item_fit_q_range(queue_item)
        self.queue_detail_label.setText(
            (
                f"{queue_item.label}\n{queue_item.run_dir}\n"
                "Fit q-range: "
                f"{self._format_queue_fit_q_value(fit_q_min)} to "
                f"{self._format_queue_fit_q_value(fit_q_max)}\n"
                f"Weights: {queue_item_weight_state_summary(queue_item)}"
            )
        )
        try:
            settings = load_dream_settings(queue_item.settings_path)
            self._populate_queue_settings_table(settings)
        except Exception as exc:
            self.queue_settings_table.setRowCount(0)
            self.queue_detail_label.setText(
                f"{queue_item.label}\nCould not load DREAM settings: {exc}"
            )
        try:
            entries = load_parameter_map_for_queue_item(queue_item)
            self._populate_queue_prior_table(entries)
            self._populate_prior_table(
                entries,
                summary_prefix=(
                    f"Selected queue item {queue_item.label!r} priors"
                ),
            )
        except Exception as exc:
            self.queue_prior_table.setRowCount(0)
            self.queue_detail_label.setText(
                self.queue_detail_label.text()
                + f"\nCould not load DREAM priors: {exc}"
            )

    def _clear_queue_item_view(self, message: str) -> None:
        if hasattr(self, "queue_detail_label"):
            self.queue_detail_label.setText(message)
        if hasattr(self, "queue_settings_table"):
            self.queue_settings_table.setRowCount(0)
        if hasattr(self, "queue_prior_table"):
            self.queue_prior_table.setRowCount(0)
        if hasattr(self, "prior_table"):
            self._refresh_prior_table()

    def _populate_queue_settings_table(
        self, settings: DreamRunSettings
    ) -> None:
        rows = [
            (name, getattr(settings, name))
            for name in DREAM_SAMPLER_SETTING_NAMES
        ]
        self.queue_settings_table.setRowCount(len(rows))
        for row, (name, value) in enumerate(rows):
            self.queue_settings_table.setItem(
                row,
                0,
                QTableWidgetItem(str(name)),
            )
            self.queue_settings_table.setItem(
                row,
                1,
                QTableWidgetItem(self._format_settings_value(value)),
            )
        self.queue_settings_table.resizeColumnsToContents()

    def _populate_queue_prior_table(
        self,
        entries: list[DreamParameterEntry],
    ) -> None:
        self.queue_prior_table.setRowCount(len(entries))
        for row, entry in enumerate(entries):
            guide_low, guide_high, guide_tooltip = self._prior_guide_text(
                entry
            )
            values = [
                entry.param,
                entry.structure,
                "Yes" if entry.vary else "No",
                entry.distribution,
                json.dumps(entry.dist_params, sort_keys=True),
                guide_low,
                guide_high,
            ]
            for column, value in enumerate(values):
                table_item = QTableWidgetItem(value)
                if column in {5, 6}:
                    table_item.setToolTip(guide_tooltip)
                self.queue_prior_table.setItem(row, column, table_item)
        self.queue_prior_table.resizeColumnsToContents()

    @staticmethod
    def _format_settings_value(value: object) -> str:
        if value is None:
            return ""
        if isinstance(value, bool):
            return "Yes" if value else "No"
        if isinstance(value, (dict, list, tuple)):
            return json.dumps(value, sort_keys=True)
        return str(value)

    def _refresh_command_box(self) -> None:
        if self._manager is None:
            self.command_box.setPlainText(
                "Create a DREAM backend run set before generating commands."
            )
            if hasattr(self, "open_commands_button"):
                self.open_commands_button.setEnabled(False)
            return
        self.command_box.setPlainText(
            command_text_for_run_set(self._manager.run_set)
        )
        if hasattr(self, "open_commands_button"):
            self.open_commands_button.setEnabled(
                self._manager.run_set.commands_path.is_file()
            )

    @staticmethod
    def _reveal_path_in_file_browser(path: Path) -> None:
        resolved_path = Path(path).expanduser().resolve()
        if sys.platform == "darwin":
            subprocess.run(["open", "-R", str(resolved_path)], check=False)
        elif sys.platform.startswith("win"):
            subprocess.run(
                ["explorer", f"/select,{resolved_path}"],
                check=False,
            )
        else:
            subprocess.run(
                ["xdg-open", str(resolved_path.parent)],
                check=False,
            )

    def _set_settings_json(
        self,
        settings: DreamRunSettings,
        *,
        update_controls: bool = True,
    ) -> None:
        self._draft_settings = DreamRunSettings.from_dict(settings.to_dict())
        if update_controls:
            self._populate_settings_controls(self._draft_settings)
        self.settings_json_edit.setPlainText(
            json.dumps(
                dream_run_settings_to_dict(
                    self._draft_settings,
                    include_posterior_filter_settings=False,
                ),
                indent=2,
            )
        )

    def _refresh_settings_json_preview(self, *_args: object) -> None:
        try:
            settings = self._settings_from_controls()
        except Exception:
            settings = self._draft_settings
        self._set_settings_json(settings, update_controls=False)

    def _settings_from_json(
        self,
        editor: QPlainTextEdit,
    ) -> DreamRunSettings:
        payload = json.loads(editor.toPlainText())
        if not isinstance(payload, dict):
            raise ValueError("DREAM settings JSON must contain an object.")
        return DreamRunSettings.from_dict(payload)

    def _populate_settings_controls(self, settings: DreamRunSettings) -> None:
        was_applying = self._applying_search_filter_preset
        self._applying_search_filter_preset = True
        try:
            self.model_name_edit.setText(settings.model_name or "")
            self._set_combo_data(
                self.search_filter_preset_combo,
                settings.search_filter_preset or "custom",
            )
            self._set_fit_q_range_controls(
                fit_q_min=settings.fit_q_min,
                fit_q_max=settings.fit_q_max,
            )
            self.chains_spin.setValue(int(settings.nchains))
            self.iterations_spin.setValue(int(settings.niterations))
            self.burnin_spin.setValue(int(settings.burnin_percent))
            self.history_thin_spin.setValue(int(settings.history_thin))
            self.nseedchains_spin.setValue(int(settings.nseedchains))
            self.crossover_burnin_spin.setValue(int(settings.crossover_burnin))
            self.lambda_spin.setValue(float(settings.lamb))
            self.zeta_spin.setValue(float(settings.zeta))
            self.snooker_spin.setValue(float(settings.snooker))
            self.p_gamma_unity_spin.setValue(float(settings.p_gamma_unity))
            self.verbose_checkbox.setChecked(bool(settings.verbose))
            self.verbose_interval_spin.setValue(
                float(settings.verbose_output_interval_seconds)
            )
            self.parallel_checkbox.setChecked(bool(settings.parallel))
            self.adapt_checkbox.setChecked(bool(settings.adapt_crossover))
            self.restart_checkbox.setChecked(bool(settings.restart))
            self.history_file_edit.setText(settings.history_file or "")
            self._update_verbose_controls()
        finally:
            self._applying_search_filter_preset = was_applying

    def _settings_from_controls(self) -> DreamRunSettings:
        settings = DreamRunSettings.from_dict(self._draft_settings.to_dict())
        settings.run_label = self.queue_label_edit.text().strip() or (
            settings.run_label or "dream"
        )
        settings.model_name = self.model_name_edit.text().strip() or None
        settings.search_filter_preset = self._combo_data(
            self.search_filter_preset_combo,
            "custom",
        )
        settings.fit_q_min, settings.fit_q_max = (
            self._fit_q_range_from_controls()
        )
        settings.nchains = int(self.chains_spin.value())
        settings.niterations = int(self.iterations_spin.value())
        settings.burnin_percent = int(self.burnin_spin.value())
        settings.history_thin = int(self.history_thin_spin.value())
        settings.nseedchains = int(self.nseedchains_spin.value())
        settings.crossover_burnin = int(self.crossover_burnin_spin.value())
        settings.lamb = float(self.lambda_spin.value())
        settings.zeta = float(self.zeta_spin.value())
        settings.snooker = float(self.snooker_spin.value())
        settings.p_gamma_unity = float(self.p_gamma_unity_spin.value())
        settings.verbose = bool(self.verbose_checkbox.isChecked())
        settings.verbose_output_interval_seconds = float(
            self.verbose_interval_spin.value()
        )
        settings.parallel = bool(self.parallel_checkbox.isChecked())
        settings.adapt_crossover = bool(self.adapt_checkbox.isChecked())
        settings.restart = bool(self.restart_checkbox.isChecked())
        settings.history_file = self.history_file_edit.text().strip() or None
        return settings

    def _populate_filter_controls(
        self,
        settings: PosteriorFilterSettings,
    ) -> None:
        was_applying = self._applying_search_filter_preset
        self._applying_search_filter_preset = True
        try:
            self._set_combo_data(
                self.bestfit_method_combo,
                settings.bestfit_method,
            )
            self._set_combo_data(
                self.violin_mode_combo,
                settings.violin_parameter_mode,
            )
            self.violin_parameter_edit.setText(
                settings.violin_selected_parameter or ""
            )
            self._set_combo_data(
                self.weight_order_combo,
                settings.violin_weight_order,
            )
            self._set_combo_data(
                self.violin_value_scale_combo,
                settings.violin_value_scale_mode,
            )
            self.stoichiometry_elements_edit.setText(
                settings.stoichiometry_target_elements_text
            )
            self.stoichiometry_ratio_edit.setText(
                settings.stoichiometry_target_ratio_text
            )
            self.stoichiometry_filter_checkbox.setChecked(
                bool(settings.stoichiometry_filter_enabled)
            )
            self.stoichiometry_tolerance_spin.setValue(
                float(settings.stoichiometry_tolerance_percent)
            )
            self._set_combo_data(
                self.posterior_filter_combo,
                settings.posterior_filter_mode,
            )
            self._set_combo_data(
                self.violin_sample_source_combo,
                settings.violin_sample_source,
            )
            self.posterior_top_percent_spin.setValue(
                float(settings.posterior_top_percent)
            )
            self.posterior_top_n_spin.setValue(int(settings.posterior_top_n))
            self.auto_filter_assessment_checkbox.setChecked(
                bool(settings.auto_select_best_posterior_filter)
            )
            self.credible_interval_low_spin.setValue(
                float(settings.credible_interval_low)
            )
            self.credible_interval_high_spin.setValue(
                float(settings.credible_interval_high)
            )
            self._update_posterior_filter_controls()
        finally:
            self._applying_search_filter_preset = was_applying
        self._set_filter_json(settings)

    def _filter_settings_from_controls(self) -> PosteriorFilterSettings:
        return PosteriorFilterSettings(
            bestfit_method=self._combo_data(self.bestfit_method_combo, "map"),
            posterior_filter_mode=self._combo_data(
                self.posterior_filter_combo,
                "all_post_burnin",
            ),
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
            violin_parameter_mode=self._combo_data(
                self.violin_mode_combo,
                "varying_parameters",
            ),
            violin_sample_source=self._combo_data(
                self.violin_sample_source_combo,
                "filtered_posterior",
            ),
            violin_weight_order=self._combo_data(
                self.weight_order_combo,
                "weight_index",
            ),
            violin_value_scale_mode=self._combo_data(
                self.violin_value_scale_combo,
                "parameter_value",
            ),
            violin_selected_parameter=(
                self.violin_parameter_edit.text().strip()
            ),
            stoichiometry_target_elements_text=(
                self.stoichiometry_elements_edit.text().strip()
            ),
            stoichiometry_target_ratio_text=(
                self.stoichiometry_ratio_edit.text().strip()
            ),
            stoichiometry_filter_enabled=bool(
                self.stoichiometry_filter_checkbox.isChecked()
            ),
            stoichiometry_tolerance_percent=float(
                self.stoichiometry_tolerance_spin.value()
            ),
        )

    def _set_filter_json(self, settings: PosteriorFilterSettings) -> None:
        self.filter_json_edit.setPlainText(
            json.dumps(settings.to_dict(), indent=2)
        )

    def _refresh_filter_json_preview(self, *_args: object) -> None:
        self._set_filter_json(self._filter_settings_from_controls())

    def _filter_settings_from_json(
        self,
        editor: QPlainTextEdit,
    ) -> PosteriorFilterSettings:
        payload = json.loads(editor.toPlainText())
        if not isinstance(payload, dict):
            raise ValueError("Posterior filter JSON must contain an object.")
        return PosteriorFilterSettings.from_dict(payload)

    def _on_search_filter_preset_changed(self, _index: int) -> None:
        if self._applying_search_filter_preset:
            return
        preset_name = self._combo_data(
            self.search_filter_preset_combo,
            "custom",
        )
        if preset_name == "custom":
            return
        self._apply_search_filter_preset(preset_name)

    def _apply_search_filter_preset(self, preset_name: str) -> None:
        preset = DREAM_SEARCH_FILTER_PRESETS.get(preset_name)
        if preset is None:
            return
        self._applying_search_filter_preset = True
        try:
            self.chains_spin.setValue(int(preset["nchains"]))
            self.iterations_spin.setValue(int(preset["niterations"]))
            self.burnin_spin.setValue(int(preset["burnin_percent"]))
            self.nseedchains_spin.setValue(int(preset["nseedchains"]))
            self.crossover_burnin_spin.setValue(
                int(preset["crossover_burnin"])
            )
            if "history_thin" in preset:
                self.history_thin_spin.setValue(int(preset["history_thin"]))
            if "lamb" in preset:
                self.lambda_spin.setValue(float(preset["lamb"]))
            if "zeta" in preset:
                self.zeta_spin.setValue(float(preset["zeta"]))
            if "snooker" in preset:
                self.snooker_spin.setValue(float(preset["snooker"]))
            if "p_gamma_unity" in preset:
                self.p_gamma_unity_spin.setValue(
                    float(preset["p_gamma_unity"])
                )
            if "verbose" in preset:
                self.verbose_checkbox.setChecked(bool(preset["verbose"]))
            if "parallel" in preset:
                self.parallel_checkbox.setChecked(bool(preset["parallel"]))
            if "adapt_crossover" in preset:
                self.adapt_checkbox.setChecked(bool(preset["adapt_crossover"]))
            self._set_combo_data(self.search_filter_preset_combo, preset_name)
        finally:
            self._applying_search_filter_preset = False
        self._refresh_settings_json_preview()
        self._set_status(f"Applied DREAM search preset: {preset_name}.")

    def _on_posterior_filter_mode_changed(self, _index: int) -> None:
        self._update_posterior_filter_controls()
        self._refresh_filter_json_preview()

    def _mark_search_filter_preset_custom(self, *_args: object) -> None:
        if self._applying_search_filter_preset:
            return
        self._applying_search_filter_preset = True
        try:
            self._set_combo_data(self.search_filter_preset_combo, "custom")
        finally:
            self._applying_search_filter_preset = False

    def _update_posterior_filter_controls(self, *_args: object) -> None:
        self.posterior_top_percent_spin.setEnabled(True)
        self.posterior_top_n_spin.setEnabled(True)
        self.stoichiometry_tolerance_spin.setEnabled(
            self.stoichiometry_filter_checkbox.isChecked()
        )

    def _update_verbose_controls(self, *_args: object) -> None:
        self.verbose_interval_spin.setEnabled(
            self.verbose_checkbox.isChecked()
        )

    @staticmethod
    def _set_combo_data(combo: QComboBox, value: str) -> bool:
        index = combo.findData(value)
        if index < 0:
            index = combo.findText(value)
        if index < 0:
            return False
        combo.setCurrentIndex(index)
        return True

    @staticmethod
    def _combo_data(combo: QComboBox, default: str) -> str:
        data = combo.currentData()
        if data is None:
            text = combo.currentText().strip()
            return text or default
        return str(data)

    def _require_manager(self) -> DreamBatchRunSetManager:
        if self._manager is None:
            raise ValueError("Choose a ready SAXSShell project first.")
        return self._manager

    def _project_dir(self) -> Path | None:
        text = self.project_dir_edit.text().strip()
        if not text:
            return None
        return Path(text).expanduser().resolve()

    def _set_status(self, message: str) -> None:
        self.statusBar().showMessage(message)


def launch_dream_batch_run_file_ui(
    *,
    initial_project_dir: str | Path | None = None,
    initial_prefit_parameter_entries: list[PrefitParameterEntry] | None = None,
    initial_fit_q_range: (
        tuple[
            float | None,
            float | None,
            float | None,
            float | None,
        ]
        | None
    ) = None,
) -> DreamBatchRunFileWindow:
    app = QApplication.instance()
    if app is None:
        prepare_saxshell_application_identity()
        app = QApplication(sys.argv)
    configure_saxshell_application(app)
    window = DreamBatchRunFileWindow(
        initial_project_dir=initial_project_dir,
        initial_prefit_parameter_entries=initial_prefit_parameter_entries,
        initial_fit_q_range=initial_fit_q_range,
    )
    window.show()
    window.raise_()
    return window


__all__ = [
    "DreamBatchRunFileWindow",
    "launch_dream_batch_run_file_ui",
]
