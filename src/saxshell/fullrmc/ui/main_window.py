from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.backends.backend_qtagg import (
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure
from PySide6.QtCore import QObject, QSettings, Qt, QThread, QUrl, Signal, Slot
from PySide6.QtGui import QAction, QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from saxshell.bondanalysis import (
    AngleTripletDefinition,
    BondAnalysisPreset,
    BondPairDefinition,
    load_presets,
    ordered_preset_names,
    save_custom_preset,
)
from saxshell.fullrmc.constraint_generation import (
    ConstraintGenerationMetadata,
    ConstraintGenerationSettings,
    build_constraint_generation,
)
from saxshell.fullrmc.packmol_docker import (
    DEFAULT_PACKMOL_CONTAINER_ROOT,
    PackmolDockerClient,
    PackmolDockerLink,
    PackmolDockerSyncResult,
    save_packmol_docker_link_metadata,
)
from saxshell.fullrmc.packmol_planning import (
    PackmolPlanningMetadata,
    PackmolPlanningSettings,
    build_packmol_plan,
)
from saxshell.fullrmc.packmol_setup import (
    PackmolSetupMetadata,
    PackmolSetupSettings,
    build_packmol_setup,
)
from saxshell.fullrmc.project_loader import (
    RMCDreamProjectSource,
    RMCDreamRunRecord,
    load_rmc_project_source,
)
from saxshell.fullrmc.representatives import (
    RepresentativeSelectionMetadata,
    RepresentativeSelectionSettings,
    build_representative_preview_clusters,
    representative_source_solvent_mode_to_variant,
    select_distribution_representatives,
    select_first_file_representatives,
)
from saxshell.fullrmc.solution_properties import (
    SolutionPropertiesMetadata,
    SolutionPropertiesResult,
    SolutionPropertiesSettings,
    calculate_solution_properties,
    save_solution_properties_metadata,
    solution_properties_mode_hint_text,
)
from saxshell.fullrmc.solution_property_presets import (
    SolutionPropertiesPreset,
    load_solution_property_presets,
    ordered_solution_property_preset_names,
    save_custom_solution_property_preset,
    solution_property_presets_path,
)
from saxshell.fullrmc.solvent_handling import (
    GeneratedPDBInspection,
    RepresentativeSolventDistributionAnalysis,
    SoluteAtomBuildSetting,
    SolventHandlingMetadata,
    SolventHandlingSettings,
    analyze_representative_solvent_distribution,
    available_representative_structure_modes,
    build_generated_pdb_inspections,
    build_representative_solvent_outputs,
    list_solvent_reference_presets,
    representative_structure_mode_label,
    representative_structure_path_for_mode,
    resolved_representative_structure_mode,
    save_solvent_handling_metadata,
    solvent_entry_lookup_for_representatives,
)
from saxshell.fullrmc.solvent_shell_builder import (
    DEFAULT_REFERENCE_MATCH_TOLERANCE_A,
    default_director_atom_name,
    reference_atom_choices,
)
from saxshell.fullrmc.ui.constraints_preview_window import (
    ConstraintsPreviewWindow,
)
from saxshell.fullrmc.ui.generated_pdb_preview_window import (
    GeneratedPDBPreviewWindow,
)
from saxshell.fullrmc.ui.packmol_docker_dialog import PackmolDockerLinkDialog
from saxshell.fullrmc.ui.representative_preview_window import (
    RepresentativePreviewWindow,
)
from saxshell.plotting import Q_A_INVERSE_LABEL
from saxshell.saxs.dream import (
    DreamModelPlotData,
    DreamSummary,
    DreamViolinPlotData,
    SAXSDreamResultsLoader,
)
from saxshell.saxs.electron_density_mapping.ui.viewer import (
    ElectronDensityStructureViewer,
)
from saxshell.saxs.electron_density_mapping.workflow import (
    load_electron_density_structure,
)
from saxshell.saxs.project_manager import (
    DreamBestFitSelection,
    SAXSProjectManager,
)
from saxshell.saxs.stoichiometry import (
    format_stoich_for_axis,
    sort_stoich_labels,
)
from saxshell.saxs.ui.branding import (
    configure_saxshell_application,
    load_saxshell_icon,
    prepare_saxshell_application_identity,
)

_OPEN_WINDOWS: list["RMCSetupMainWindow"] = []

_BESTFIT_METHOD_ITEMS = [
    ("MAP", "map"),
    ("Chain Mean MAP", "chain_mean"),
    ("Median", "median"),
]
_POSTERIOR_FILTER_ITEMS = [
    ("All Post-burnin Samples", "all_post_burnin"),
    ("Top % by Log-posterior", "top_percent_logp"),
    ("Top N by Log-posterior", "top_n_logp"),
]
_SOLUTION_MODE_ITEMS = [
    ("Masses", "mass"),
    ("Mass Percent", "mass_percent"),
    ("Molarity per Liter", "molarity_per_liter"),
]
_REPRESENTATIVE_MODE_ITEMS = [
    ("First File (Fast)", "first_file"),
    (
        "Representative by Bond/Angle Distributions",
        "bond_angle_distribution",
    ),
]
_COORDINATED_SOLVENT_MODE_ITEMS = [
    ("No Coordinated Solvent", "no_coordinated_solvent"),
    ("Partial Coordinated Solvent", "partial_coordinated_solvent"),
    ("Full Coordinated Solvent", "full_coordinated_solvent"),
]
_SOLVENT_REFERENCE_SOURCE_ITEMS = [
    ("Bundled Preset", "preset"),
    ("Custom PDB", "custom"),
]
_PACKMOL_PLANNING_MODE_ITEMS = [
    ("Per Element (Recommended)", "per_element"),
    ("Total Number Density", "total"),
]
_REPRESENTATIVE_ALGORITHM_ITEMS = [
    (
        "Quantile Distance (Recommended)",
        "target_distribution_quantile_distance",
    ),
    ("Mean/Std Distance", "target_distribution_moment_distance"),
]
_PACKMOL_PREVIEW_COLORS = (
    "#4f7d5c",
    "#7aa6c2",
    "#d08c60",
    "#9a6ea8",
    "#d0b060",
    "#5f9e8f",
)
_PACKMOL_DOCKER_PRESETS_KEY = "packmol_docker_presets"
_READINESS_TASK_DETAILS = {
    "project_source": {
        "title": "Project Source",
        "purpose": (
            "Load the SAXS project, saved DREAM runs, cluster metadata, "
            "and previously generated rmcsetup outputs."
        ),
        "needed": (
            "A valid SAXS project directory containing DREAM output files "
            "and cluster metadata."
        ),
        "prerequisites": (
            "No prior rmcsetup step is required. Choose or reload a project "
            "directory first."
        ),
    },
    "dream_selection": {
        "title": "DREAM Model Selection",
        "purpose": (
            "Choose the DREAM run and posterior-summary settings that define "
            "the active model weights used downstream."
        ),
        "needed": (
            "A completed DREAM run plus the desired best-fit method, "
            "posterior filter, and interval settings."
        ),
        "prerequisites": (
            "The SAXS project source must be loaded so valid DREAM runs can "
            "be scanned."
        ),
    },
    "solution_properties": {
        "title": "Solution Properties",
        "purpose": (
            "Convert the solution description into number-density and box-"
            "composition targets for Packmol planning."
        ),
        "needed": (
            "Solute and solvent stoichiometry, solution density, molar "
            "masses, and one complete composition input mode."
        ),
        "prerequisites": (
            "The SAXS project must be loaded. No representative structures "
            "are required yet."
        ),
    },
    "representative_selection": {
        "title": "Representative Structures",
        "purpose": (
            "Load the saved representative structures that will be combined "
            "with the active DREAM weights for the Solvent Shell Builder, "
            "Packmol planning, Packmol setup, and cluster-specific "
            "constraints."
        ),
        "needed": (
            "A saved representative-structure set in the SAXS project plus "
            "an active DREAM selection so the current model weights remain "
            "available downstream."
        ),
        "prerequisites": (
            "The SAXS project must be loaded. Use the dedicated "
            "Representative Structures tool to create or update the saved "
            "structure set for this project."
        ),
    },
    "solvent_outputs": {
        "title": "Solvent Shell Builder",
        "purpose": (
            "Use the active representative structure set directly when it "
            "already has full solvent, or build the missing solvent shell "
            "for no-solvent and partial-solvent representatives."
        ),
        "needed": (
            "Either imported representative structures that already provide "
            "the Full solvent set, or a solvent reference plus the saved "
            "representative solvent outputs with Full solvent selected as "
            "active."
        ),
        "prerequisites": (
            "Representative structures must already be saved for the project."
        ),
    },
    "packmol_plan": {
        "title": "Packmol Plan",
        "purpose": (
            "Translate the selected DREAM-weighted representatives and "
            "solution targets into planned cluster counts for the box."
        ),
        "needed": (
            "Packmol planning mode, box side length, saved solution "
            "properties, and saved representative structures."
        ),
        "prerequisites": (
            "Solution properties and representative structures must both be "
            "available first."
        ),
    },
    "packmol_setup": {
        "title": "Packmol Setup",
        "purpose": (
            "Write the Packmol input files and audit outputs needed to build "
            "the simulation box."
        ),
        "needed": (
            "A saved Packmol plan and the Full solvent representative "
            "structure set."
        ),
        "prerequisites": (
            "Packmol planning and representative PDB solvent outputs must "
            "already be complete."
        ),
    },
}


def _natural_sort_key(value: str) -> list[object]:
    return [
        int(token) if token.isdigit() else token.lower()
        for token in re.split(r"(\d+)", value)
        if token
    ]


def _packmol_motif_label(motif: str) -> str:
    return "No motif" if motif == "no_motif" else motif


def _readiness_task_tooltip(key: str) -> str:
    details = _READINESS_TASK_DETAILS.get(key)
    if details is None:
        return ""
    return (
        f"<b>{details['title']}</b><br>"
        f"<b>Purpose:</b> {details['purpose']}<br>"
        f"<b>Information needed:</b> {details['needed']}<br>"
        f"<b>Processes or saved results required first:</b> "
        f"{details['prerequisites']}"
    )


def _workflow_readiness_help_tooltip() -> str:
    parts = [
        "<b>Project Readiness Tasks</b><br>"
        "Each readiness checkbox tracks one saved workflow milestone."
    ]
    for key in _READINESS_TASK_DETAILS:
        parts.append(_readiness_task_tooltip(key))
    parts.append(
        "<b>Note:</b> Constraint generation is a downstream task and is not "
        "currently counted in the project-readiness total."
    )
    return "<br><br>".join(parts)


class RepresentativeSelectionWorker(QObject):
    log = Signal(str)
    progress = Signal(int, int, str)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        project_source: RMCDreamProjectSource,
        selection: DreamBestFitSelection,
        settings: RepresentativeSelectionSettings,
    ) -> None:
        super().__init__()
        self.project_source = project_source
        self.selection = selection
        self.settings = settings

    @Slot()
    def run(self) -> None:
        try:
            if self.settings.selection_mode == "first_file":
                metadata = select_first_file_representatives(
                    self.project_source,
                    self.selection,
                    settings=self.settings,
                    progress_callback=self._emit_progress,
                    log_callback=self.log.emit,
                )
            else:
                metadata = select_distribution_representatives(
                    self.project_source,
                    self.selection,
                    settings=self.settings,
                    progress_callback=self._emit_progress,
                    log_callback=self.log.emit,
                )
            self.finished.emit(metadata)
        except Exception as exc:
            self.failed.emit(str(exc))

    def _emit_progress(
        self,
        processed: int,
        total: int,
        message: str,
    ) -> None:
        self.progress.emit(processed, total, message)


class RMCSetupMainWindow(QMainWindow):
    """Initial fullrmc source-selection and favorite-management
    window."""

    def __init__(
        self,
        initial_project_dir: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.setWindowTitle("SAXSShell (rmcsetup)")
        self.setWindowIcon(load_saxshell_icon())
        self.resize(1080, 860)
        self._build_menu_bar()

        self.project_manager = SAXSProjectManager()
        self._project_source_state: RMCDreamProjectSource | None = None
        self._child_tool_windows: list[object] = []
        self._representative_preview_window: (
            RepresentativePreviewWindow | None
        ) = None
        self._generated_pdb_preview_window: (
            GeneratedPDBPreviewWindow | None
        ) = None
        self._constraints_preview_window: ConstraintsPreviewWindow | None = (
            None
        )
        self._solvent_distribution_analysis: (
            RepresentativeSolventDistributionAnalysis | None
        ) = None
        self._updating_dream_controls = False
        self._representative_presets: dict[str, BondAnalysisPreset] = {}
        self._solution_presets: dict[str, SolutionPropertiesPreset] = {}
        self._generated_pdb_inspections: list[
            GeneratedPDBInspection | None
        ] = []
        self._solvent_table_preview_paths: list[Path | None] = []
        self._solvent_table_details: list[str] = []
        self._updating_generated_pdb_mode_combo = False
        self._solvent_cutoff_spins: dict[str, QDoubleSpinBox] = {}
        self._solvent_coordination_center_items: dict[
            str, QTableWidgetItem
        ] = {}
        self._solvent_coordination_target_spins: dict[str, QDoubleSpinBox] = {}
        self._updating_solvent_table = False
        self._current_dream_model_plot_data: DreamModelPlotData | None = None
        self._representative_thread: QThread | None = None
        self._representative_worker: RepresentativeSelectionWorker | None = (
            None
        )
        self._representative_job_state: RMCDreamProjectSource | None = None
        self._updating_solution_preset_selection = False
        self._dream_results_loader_cache: dict[
            Path, SAXSDreamResultsLoader
        ] = {}
        self._section_toggle_buttons: dict[str, QToolButton] = {}
        self._section_content_widgets: dict[str, QWidget] = {}
        self._readiness_checkboxes: dict[str, QCheckBox] = {}
        self._available_solvent_presets = list_solvent_reference_presets()
        self._dream_model_preview_figure = Figure(
            figsize=(6.2, 3.2), tight_layout=True
        )
        self._dream_model_preview_canvas = FigureCanvasQTAgg(
            self._dream_model_preview_figure
        )
        self._dream_model_preview_toolbar = NavigationToolbar(
            self._dream_model_preview_canvas,
            self,
        )
        self._dream_violin_preview_figure = Figure(
            figsize=(6.2, 3.2), tight_layout=True
        )
        self._dream_violin_preview_canvas = FigureCanvasQTAgg(
            self._dream_violin_preview_figure
        )
        self._dream_violin_preview_toolbar = NavigationToolbar(
            self._dream_violin_preview_canvas,
            self,
        )
        self._packmol_plan_figure = Figure(
            figsize=(6.2, 3.0), tight_layout=True
        )
        self._packmol_plan_canvas = FigureCanvasQTAgg(
            self._packmol_plan_figure
        )
        self._packmol_plan_toolbar = NavigationToolbar(
            self._packmol_plan_canvas,
            self,
        )

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        root_layout = QVBoxLayout(central_widget)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(12)

        self._main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self._left_column = QWidget()
        self._left_column_layout = QVBoxLayout(self._left_column)
        self._left_column_layout.setContentsMargins(0, 0, 0, 0)
        self._left_column_layout.setSpacing(12)
        self._left_scroll_area = QScrollArea()
        self._left_scroll_area.setWidgetResizable(True)
        self._right_scroll_area = QScrollArea()
        self._right_scroll_area.setWidgetResizable(True)
        self._left_panel = QWidget()
        self._right_panel = QWidget()
        self._left_layout = QVBoxLayout(self._left_panel)
        self._left_layout.setContentsMargins(0, 0, 0, 0)
        self._left_layout.setSpacing(12)
        self._right_layout = QVBoxLayout(self._right_panel)
        self._right_layout.setContentsMargins(0, 0, 0, 0)
        self._right_layout.setSpacing(12)
        self._left_scroll_area.setWidget(self._left_panel)
        self._right_scroll_area.setWidget(self._right_panel)
        self._main_splitter.addWidget(self._left_column)
        self._main_splitter.addWidget(self._right_scroll_area)
        self._main_splitter.setChildrenCollapsible(False)
        self._main_splitter.setStretchFactor(0, 1)
        self._main_splitter.setStretchFactor(1, 1)
        root_layout.addWidget(self._main_splitter, stretch=1)

        self.software_details_button = QPushButton("Software Details")
        self.software_details_button.setCheckable(True)
        self.software_details_button.setChecked(False)
        self.software_details_button.toggled.connect(
            self._toggle_software_details
        )
        self._left_layout.addWidget(self.software_details_button)

        self.software_details_panel = QWidget()
        self.software_details_panel.setStyleSheet(
            "QWidget {"
            "background-color: #f7f4ee;"
            "border: 1px solid #d8cfbf;"
            "border-radius: 8px;"
            "}"
        )
        software_details_layout = QVBoxLayout(self.software_details_panel)
        software_details_layout.setContentsMargins(12, 10, 12, 10)
        software_details_layout.setSpacing(8)
        self.software_details_label = QLabel(
            (
                "<p><b>rmcsetup</b> is designed to prepare reverse "
                "Monte-Carlo workflows for pair distribution function "
                "(PDF) datasets using <a href='https://www.fullrmc.com/'>"
                "fullrmc</a>. The outputs generated here are intended to be "
                "used downstream in <a href='https://github.com/m3g/packmol'>"
                "Packmol</a> for box construction and then in fullrmc for "
                "reverse Monte-Carlo refinement.</p>"
                "<p><b>References</b><br>"
                "fullrmc website: "
                "<a href='https://www.fullrmc.com/'>https://www.fullrmc.com/</a><br>"
                "fullrmc seminal paper: "
                "<a href='https://onlinelibrary.wiley.com/doi/10.1002/jcc.24304'>"
                "https://onlinelibrary.wiley.com/doi/10.1002/jcc.24304</a><br>"
                "Packmol project page: "
                "<a href='https://github.com/m3g/packmol'>"
                "https://github.com/m3g/packmol</a><br>"
                "Packmol citation: "
                "<a href='https://onlinelibrary.wiley.com/doi/10.1002/jcc.21224'>"
                "https://onlinelibrary.wiley.com/doi/10.1002/jcc.21224</a></p>"
            )
        )
        self.software_details_label.setWordWrap(True)
        self.software_details_label.setTextFormat(Qt.TextFormat.RichText)
        self.software_details_label.setOpenExternalLinks(True)
        self.software_details_label.setStyleSheet(
            "QLabel {"
            "background: transparent;"
            "border: none;"
            "color: #2c241a;"
            "line-height: 1.3;"
            "}"
        )
        software_details_layout.addWidget(self.software_details_label)
        self.software_details_panel.setVisible(False)
        self._left_layout.addWidget(self.software_details_panel)

        self.status_group = QGroupBox("Workflow Status")
        status_layout = QVBoxLayout(self.status_group)
        status_header_row = QHBoxLayout()
        self.readiness_label = QLabel("Project readiness: 0 / 7 complete")
        status_header_row.addWidget(self.readiness_label)
        status_header_row.addStretch(1)
        self.readiness_help_button = QToolButton()
        self.readiness_help_button.setText("?")
        self.readiness_help_button.setAutoRaise(True)
        self.readiness_help_button.setToolTip(
            _workflow_readiness_help_tooltip()
        )
        self.readiness_help_button.setStyleSheet(
            "QToolButton { font-weight: bold; padding: 2px 6px; }"
        )
        status_header_row.addWidget(self.readiness_help_button)
        status_layout.addLayout(status_header_row)
        self.readiness_progress_bar = QProgressBar()
        self.readiness_progress_bar.setRange(0, 7)
        self.readiness_progress_bar.setValue(0)
        status_layout.addWidget(self.readiness_progress_bar)
        self.task_status_label = QLabel("Current task: idle")
        status_layout.addWidget(self.task_status_label)
        self.task_progress_bar = QProgressBar()
        self.task_progress_bar.setRange(0, 100)
        self.task_progress_bar.setValue(0)
        status_layout.addWidget(self.task_progress_bar)
        self._left_column_layout.addWidget(self.status_group)
        self._left_column_layout.addWidget(self._left_scroll_area, stretch=1)

        self.project_group = QGroupBox("SAXS Project Source")
        project_layout = QVBoxLayout(self.project_group)
        self._add_group_readiness_row(
            project_layout,
            (
                "project_source",
                "Ready",
                "Checked after the SAXS project source loads successfully.",
            ),
            section_key="project_source",
        )
        project_content_layout = self._create_collapsible_section_layout(
            project_layout,
            "project_source",
        )
        project_row = QHBoxLayout()
        self.project_dir_edit = QLineEdit()
        self.project_dir_edit.setPlaceholderText(
            "Choose a SAXS project directory with DREAM outputs"
        )
        self.project_dir_edit.editingFinished.connect(
            self._refresh_project_source
        )
        project_row.addWidget(self.project_dir_edit, stretch=1)

        self.browse_button = QPushButton("Browse...")
        self.browse_button.clicked.connect(self._browse_project_dir)
        project_row.addWidget(self.browse_button)

        self.refresh_button = QPushButton("Reload Project")
        self.refresh_button.clicked.connect(self._refresh_project_source)
        project_row.addWidget(self.refresh_button)
        project_content_layout.addLayout(project_row)

        self.project_summary_box = QPlainTextEdit()
        self.project_summary_box.setReadOnly(True)
        self.project_summary_box.setMinimumHeight(170)
        project_content_layout.addWidget(self.project_summary_box)
        self._left_layout.addWidget(self.project_group)

        self.output_group = QGroupBox("RMCSetup Output Structure")
        output_layout = QVBoxLayout(self.output_group)
        self.output_summary_box = QPlainTextEdit()
        self.output_summary_box.setReadOnly(True)
        self.output_summary_box.setMinimumHeight(160)
        output_layout.addWidget(self.output_summary_box)
        self._left_layout.addWidget(self.output_group)

        self.dream_group = QGroupBox("Select DREAM Model")
        dream_layout = QVBoxLayout(self.dream_group)
        self._add_group_readiness_row(
            dream_layout,
            (
                "dream_selection",
                "Ready",
                "Checked after a DREAM run is selected for rmcsetup.",
            ),
            section_key="dream_selection",
        )
        dream_content_layout = self._create_collapsible_section_layout(
            dream_layout,
            "dream_selection",
        )
        dream_form = QFormLayout()

        run_row = QHBoxLayout()
        self.dream_run_combo = QComboBox()
        self.dream_run_combo.currentIndexChanged.connect(
            self._on_dream_run_changed
        )
        run_row.addWidget(self.dream_run_combo, stretch=1)
        self.reload_runs_button = QPushButton("Rescan Runs")
        self.reload_runs_button.clicked.connect(self._refresh_project_source)
        run_row.addWidget(self.reload_runs_button)
        run_container = QWidget()
        run_container.setLayout(run_row)
        dream_form.addRow("DREAM run", run_container)

        self.bestfit_method_combo = QComboBox()
        for label, value in _BESTFIT_METHOD_ITEMS:
            self.bestfit_method_combo.addItem(label, value)
        self.bestfit_method_combo.currentIndexChanged.connect(
            self._refresh_dream_source_summary
        )
        dream_form.addRow("Best-fit method", self.bestfit_method_combo)

        self.posterior_filter_combo = QComboBox()
        for label, value in _POSTERIOR_FILTER_ITEMS:
            self.posterior_filter_combo.addItem(label, value)
        self.posterior_filter_combo.currentIndexChanged.connect(
            self._on_posterior_filter_changed
        )
        dream_form.addRow("Posterior filter", self.posterior_filter_combo)

        self.posterior_top_percent_spin = QDoubleSpinBox()
        self.posterior_top_percent_spin.setRange(0.1, 100.0)
        self.posterior_top_percent_spin.setDecimals(1)
        self.posterior_top_percent_spin.setSingleStep(0.5)
        self.posterior_top_percent_spin.valueChanged.connect(
            self._refresh_dream_source_summary
        )
        dream_form.addRow("Top percent (%)", self.posterior_top_percent_spin)

        self.posterior_top_n_spin = QSpinBox()
        self.posterior_top_n_spin.setRange(1, 1_000_000)
        self.posterior_top_n_spin.valueChanged.connect(
            self._refresh_dream_source_summary
        )
        dream_form.addRow("Top N samples", self.posterior_top_n_spin)

        interval_row = QHBoxLayout()
        self.credible_interval_low_spin = QDoubleSpinBox()
        self.credible_interval_low_spin.setRange(0.0, 100.0)
        self.credible_interval_low_spin.setDecimals(1)
        self.credible_interval_low_spin.setSingleStep(1.0)
        self.credible_interval_low_spin.valueChanged.connect(
            self._refresh_dream_source_summary
        )
        interval_row.addWidget(self.credible_interval_low_spin)
        interval_row.addWidget(QLabel("to"))
        self.credible_interval_high_spin = QDoubleSpinBox()
        self.credible_interval_high_spin.setRange(0.0, 100.0)
        self.credible_interval_high_spin.setDecimals(1)
        self.credible_interval_high_spin.setSingleStep(1.0)
        self.credible_interval_high_spin.valueChanged.connect(
            self._refresh_dream_source_summary
        )
        interval_row.addWidget(self.credible_interval_high_spin)
        interval_container = QWidget()
        interval_container.setLayout(interval_row)
        dream_form.addRow("Credible interval (%)", interval_container)
        dream_content_layout.addLayout(dream_form)

        self.dream_source_summary_box = QPlainTextEdit()
        self.dream_source_summary_box.setReadOnly(True)
        self.dream_source_summary_box.setMinimumHeight(170)
        dream_content_layout.addWidget(self.dream_source_summary_box)
        self._left_layout.addWidget(self.dream_group)

        self.favorite_group = QGroupBox("Saved DREAM Model")
        favorite_layout = QVBoxLayout(self.favorite_group)
        favorite_button_row = QHBoxLayout()
        self.set_favorite_button = QPushButton("Save Current Selection")
        self.set_favorite_button.clicked.connect(
            self._save_current_selection_as_favorite
        )
        favorite_button_row.addWidget(self.set_favorite_button)

        self.use_project_favorite_button = QPushButton(
            "Load Saved DREAM Model"
        )
        self.use_project_favorite_button.clicked.connect(
            self._apply_project_favorite
        )
        favorite_button_row.addWidget(self.use_project_favorite_button)
        favorite_layout.addLayout(favorite_button_row)

        history_row = QHBoxLayout()
        self.favorite_history_combo = QComboBox()
        history_row.addWidget(self.favorite_history_combo, stretch=1)
        self.load_history_button = QPushButton("Load History Entry")
        self.load_history_button.clicked.connect(self._load_history_entry)
        history_row.addWidget(self.load_history_button)
        favorite_layout.addLayout(history_row)

        self.favorite_summary_box = QPlainTextEdit()
        self.favorite_summary_box.setReadOnly(True)
        self.favorite_summary_box.setMinimumHeight(160)
        favorite_layout.addWidget(self.favorite_summary_box)
        self._left_layout.addWidget(self.favorite_group)

        self.solution_group = QGroupBox("Solution Properties")
        solution_layout = QVBoxLayout(self.solution_group)
        self._add_group_readiness_row(
            solution_layout,
            (
                "solution_properties",
                "Ready",
                "Checked after solution properties are calculated.",
            ),
            section_key="solution_properties",
        )
        solution_content_layout = self._create_collapsible_section_layout(
            solution_layout,
            "solution_properties",
        )
        self.solution_preset_group = QGroupBox("Solution Presets")
        solution_preset_layout = QVBoxLayout(self.solution_preset_group)
        solution_preset_row = QHBoxLayout()
        self.solution_preset_combo = QComboBox()
        solution_preset_row.addWidget(self.solution_preset_combo, stretch=1)
        self.load_solution_preset_button = QPushButton("Load")
        self.load_solution_preset_button.clicked.connect(
            self._load_selected_solution_preset
        )
        solution_preset_row.addWidget(self.load_solution_preset_button)
        self.save_solution_preset_button = QPushButton("Save Current As...")
        self.save_solution_preset_button.clicked.connect(
            self._save_current_solution_as_preset
        )
        solution_preset_row.addWidget(self.save_solution_preset_button)
        solution_preset_layout.addLayout(solution_preset_row)
        self.solution_preset_hint_label = QLabel(
            "Load bundled solution-composition presets or save the current "
            "fields into the editable package preset file at "
            f"{solution_property_presets_path()}."
        )
        self.solution_preset_hint_label.setWordWrap(True)
        solution_preset_layout.addWidget(self.solution_preset_hint_label)
        solution_content_layout.addWidget(self.solution_preset_group)
        solution_form = QFormLayout()

        self.solution_mode_combo = QComboBox()
        for label, value in _SOLUTION_MODE_ITEMS:
            self.solution_mode_combo.addItem(label, value)
        self.solution_mode_combo.currentIndexChanged.connect(
            self._on_solution_mode_changed
        )
        self.solution_mode_combo.currentIndexChanged.connect(
            self._on_solution_settings_changed
        )
        solution_form.addRow("Input mode", self.solution_mode_combo)

        self.solution_density_spin = self._new_float_spin(
            maximum=100.0,
            step=0.01,
            decimals=6,
            value=1.0,
        )
        self.solution_density_spin.valueChanged.connect(
            self._on_solution_settings_changed
        )
        solution_form.addRow(
            "Solution density (g/mL)",
            self.solution_density_spin,
        )

        self.solute_stoich_edit = QLineEdit()
        self.solute_stoich_edit.setPlaceholderText("e.g. Cs1Pb1I3")
        self.solute_stoich_edit.textChanged.connect(
            self._on_solution_settings_changed
        )
        solution_form.addRow("Solute stoichiometry", self.solute_stoich_edit)

        self.solvent_stoich_edit = QLineEdit()
        self.solvent_stoich_edit.setPlaceholderText("e.g. H2O or C3H7NO")
        self.solvent_stoich_edit.textChanged.connect(
            self._on_solution_settings_changed
        )
        solution_form.addRow("Solvent stoichiometry", self.solvent_stoich_edit)

        self.molar_mass_solute_spin = self._new_float_spin(
            maximum=1_000_000.0,
            step=1.0,
            decimals=6,
            value=0.0,
        )
        self.molar_mass_solute_spin.valueChanged.connect(
            self._on_solution_settings_changed
        )
        solution_form.addRow(
            "Solute molar mass (g/mol)",
            self.molar_mass_solute_spin,
        )

        self.molar_mass_solvent_spin = self._new_float_spin(
            maximum=1_000_000.0,
            step=1.0,
            decimals=6,
            value=0.0,
        )
        self.molar_mass_solvent_spin.valueChanged.connect(
            self._on_solution_settings_changed
        )
        solution_form.addRow(
            "Solvent molar mass (g/mol)",
            self.molar_mass_solvent_spin,
        )
        solution_content_layout.addLayout(solution_form)
        self.solution_mode_hint_label = QLabel()
        self.solution_mode_hint_label.setWordWrap(True)
        self.solution_mode_hint_label.setText(
            solution_properties_mode_hint_text("mass")
        )
        solution_content_layout.addWidget(self.solution_mode_hint_label)

        self.solution_mode_stack = QStackedWidget()

        mass_page = QWidget()
        mass_form = QFormLayout(mass_page)
        self.mass_solute_spin = self._new_float_spin(
            maximum=1_000_000.0,
            step=0.1,
            decimals=6,
            value=0.0,
        )
        self.mass_solute_spin.valueChanged.connect(
            self._on_solution_settings_changed
        )
        mass_form.addRow("Mass solute (g)", self.mass_solute_spin)
        self.mass_solvent_spin = self._new_float_spin(
            maximum=1_000_000.0,
            step=0.1,
            decimals=6,
            value=0.0,
        )
        self.mass_solvent_spin.valueChanged.connect(
            self._on_solution_settings_changed
        )
        mass_form.addRow("Mass solvent (g)", self.mass_solvent_spin)
        self.solution_mode_stack.addWidget(mass_page)

        mass_percent_page = QWidget()
        mass_percent_form = QFormLayout(mass_percent_page)
        self.mass_percent_solute_spin = self._new_float_spin(
            maximum=100.0,
            step=0.1,
            decimals=4,
            value=0.0,
        )
        self.mass_percent_solute_spin.valueChanged.connect(
            self._on_solution_settings_changed
        )
        mass_percent_form.addRow(
            "Mass percent solute (%)",
            self.mass_percent_solute_spin,
        )
        self.total_mass_solution_spin = self._new_float_spin(
            maximum=1_000_000.0,
            step=0.1,
            decimals=6,
            value=0.0,
        )
        self.total_mass_solution_spin.valueChanged.connect(
            self._on_solution_settings_changed
        )
        mass_percent_form.addRow(
            "Total solution mass (g)",
            self.total_mass_solution_spin,
        )
        self.solution_mode_stack.addWidget(mass_percent_page)

        molarity_page = QWidget()
        molarity_form = QFormLayout(molarity_page)
        self.molarity_spin = self._new_float_spin(
            maximum=100_000.0,
            step=0.01,
            decimals=6,
            value=0.0,
        )
        self.molarity_spin.valueChanged.connect(
            self._on_solution_settings_changed
        )
        molarity_form.addRow("Molarity (mol/L)", self.molarity_spin)
        self.molarity_element_edit = QLineEdit()
        self.molarity_element_edit.setPlaceholderText("e.g. Pb")
        self.molarity_element_edit.textChanged.connect(
            self._on_solution_settings_changed
        )
        molarity_form.addRow(
            "Molarity element",
            self.molarity_element_edit,
        )
        self.solution_mode_stack.addWidget(molarity_page)

        solution_content_layout.addWidget(self.solution_mode_stack)

        solution_button_row = QHBoxLayout()
        self.calculate_solution_button = QPushButton("Calculate")
        self.calculate_solution_button.clicked.connect(
            self._calculate_solution_properties
        )
        solution_button_row.addWidget(self.calculate_solution_button)
        solution_button_row.addStretch(1)
        solution_content_layout.addLayout(solution_button_row)

        self.solution_output_box = QPlainTextEdit()
        self.solution_output_box.setReadOnly(True)
        self.solution_output_box.setMinimumHeight(220)
        solution_content_layout.addWidget(self.solution_output_box)
        self._left_layout.addWidget(self.solution_group)

        self.dream_preview_group = QGroupBox("Selected DREAM Model Preview")
        dream_preview_layout = QVBoxLayout(self.dream_preview_group)
        self.dream_preview_intro_label = QLabel(
            "Preview the selected DREAM model fit and posterior weight "
            "distributions before loading representative structures or "
            "Packmol inputs."
        )
        self.dream_preview_intro_label.setWordWrap(True)
        dream_preview_layout.addWidget(self.dream_preview_intro_label)
        self.dream_preview_status_label = QLabel("")
        self.dream_preview_status_label.setWordWrap(True)
        dream_preview_layout.addWidget(self.dream_preview_status_label)

        self._dream_preview_splitter = QSplitter(Qt.Orientation.Horizontal)
        self._dream_preview_splitter.setChildrenCollapsible(False)
        self._dream_preview_splitter.setHandleWidth(10)

        self.dream_model_preview_group = QGroupBox("Model Fit")
        dream_model_preview_layout = QVBoxLayout(
            self.dream_model_preview_group
        )
        dream_model_trace_row = QHBoxLayout()
        dream_model_trace_row.addWidget(QLabel("Visible traces:"))
        self.show_experimental_trace_checkbox = QCheckBox("Experimental")
        self.show_experimental_trace_checkbox.setChecked(True)
        dream_model_trace_row.addWidget(self.show_experimental_trace_checkbox)
        self.show_model_trace_checkbox = QCheckBox("Model")
        self.show_model_trace_checkbox.setChecked(True)
        dream_model_trace_row.addWidget(self.show_model_trace_checkbox)
        self.show_solvent_trace_checkbox = QCheckBox("Solvent")
        self.show_solvent_trace_checkbox.setChecked(False)
        dream_model_trace_row.addWidget(self.show_solvent_trace_checkbox)
        dream_model_trace_row.addStretch(1)
        dream_model_preview_layout.addLayout(dream_model_trace_row)
        self.show_experimental_trace_checkbox.toggled.connect(
            self._refresh_dream_model_preview_from_cache
        )
        self.show_model_trace_checkbox.toggled.connect(
            self._refresh_dream_model_preview_from_cache
        )
        self.show_solvent_trace_checkbox.toggled.connect(
            self._refresh_dream_model_preview_from_cache
        )
        self._dream_model_preview_canvas.setMinimumHeight(280)
        dream_model_preview_layout.addWidget(self._dream_model_preview_toolbar)
        dream_model_preview_layout.addWidget(self._dream_model_preview_canvas)
        self._dream_preview_splitter.addWidget(self.dream_model_preview_group)

        self.dream_violin_preview_group = QGroupBox(
            "Posterior Weight Violin Plot"
        )
        dream_violin_preview_layout = QVBoxLayout(
            self.dream_violin_preview_group
        )
        self._dream_violin_preview_canvas.setMinimumHeight(280)
        dream_violin_preview_layout.addWidget(
            self._dream_violin_preview_toolbar
        )
        dream_violin_preview_layout.addWidget(
            self._dream_violin_preview_canvas
        )
        self._dream_preview_splitter.addWidget(self.dream_violin_preview_group)
        self._dream_preview_splitter.setStretchFactor(0, 1)
        self._dream_preview_splitter.setStretchFactor(1, 1)
        self._dream_preview_splitter.setSizes([500, 500])
        dream_preview_layout.addWidget(self._dream_preview_splitter)
        self._right_layout.addWidget(self.dream_preview_group)

        self.representative_group = QGroupBox("Representative Structures")
        representative_layout = QVBoxLayout(self.representative_group)
        self._add_group_readiness_row(
            representative_layout,
            (
                "representative_selection",
                "Ready",
                "Checked after representative structures are saved and "
                "loaded for the active project.",
            ),
            section_key="representative_selection",
        )
        representative_content_layout = (
            self._create_collapsible_section_layout(
                representative_layout,
                "representative_selection",
            )
        )
        self.representative_intro_label = QLabel(
            "rmcsetup consumes saved representative structures from the "
            "dedicated Representative Structures tool. Those saved files are "
            "combined here with the selected DREAM distribution weights, the "
            "solution density targets, Solvent Shell Builder, Packmol planning, "
            "and cluster-specific constraint generation."
        )
        self.representative_intro_label.setWordWrap(True)
        representative_content_layout.addWidget(
            self.representative_intro_label
        )
        self.representative_workflow_label = QLabel(
            "Use Open Representative Structures to create or update the "
            "saved project set, then reload it here. The active DREAM model "
            "selection in rmcsetup remains the source of the fitted weights "
            "used for downstream box planning and constraint generation."
        )
        self.representative_workflow_label.setWordWrap(True)
        representative_content_layout.addWidget(
            self.representative_workflow_label
        )

        self.representative_mode_widget = QWidget()
        representative_form = QFormLayout(self.representative_mode_widget)
        representative_form.setContentsMargins(0, 0, 0, 0)
        self.representative_mode_combo = QComboBox()
        for label, value in _REPRESENTATIVE_MODE_ITEMS:
            self.representative_mode_combo.addItem(label, value)
        self.representative_mode_combo.currentIndexChanged.connect(
            self._on_representative_mode_changed
        )
        representative_form.addRow(
            "Selection mode",
            self.representative_mode_combo,
        )
        self.representative_mode_widget.setVisible(False)
        representative_content_layout.addWidget(
            self.representative_mode_widget
        )

        self.representative_preset_group = QGroupBox("Bondanalysis Presets")
        representative_preset_layout = QVBoxLayout(
            self.representative_preset_group
        )
        representative_preset_row = QHBoxLayout()
        self.representative_preset_combo = QComboBox()
        representative_preset_row.addWidget(
            self.representative_preset_combo,
            stretch=1,
        )
        self.load_representative_preset_button = QPushButton("Load")
        self.load_representative_preset_button.clicked.connect(
            self._load_selected_representative_preset
        )
        representative_preset_row.addWidget(
            self.load_representative_preset_button
        )
        self.save_representative_preset_button = QPushButton(
            "Save Current As..."
        )
        self.save_representative_preset_button.clicked.connect(
            self._save_current_representative_as_preset
        )
        representative_preset_row.addWidget(
            self.save_representative_preset_button
        )
        representative_preset_layout.addLayout(representative_preset_row)
        self.representative_preset_hint_label = QLabel(
            "Uses the same built-in and custom presets as bondanalysis. "
            "Cached bondanalysis distributions are reused for representative "
            "selection when matching results are already available."
        )
        self.representative_preset_hint_label.setWordWrap(True)
        representative_preset_layout.addWidget(
            self.representative_preset_hint_label
        )
        self.representative_preset_group.setVisible(False)
        representative_content_layout.addWidget(
            self.representative_preset_group
        )

        self.representative_bond_pairs_row = QGroupBox("Bond Pairs")
        representative_bond_pairs_layout = QVBoxLayout(
            self.representative_bond_pairs_row
        )
        representative_bond_pair_controls = QHBoxLayout()
        self.add_representative_bond_pair_button = QPushButton("Add Bond Pair")
        self.add_representative_bond_pair_button.clicked.connect(
            self._add_representative_bond_pair_row
        )
        representative_bond_pair_controls.addWidget(
            self.add_representative_bond_pair_button
        )
        self.remove_representative_bond_pair_button = QPushButton(
            "Remove Selected"
        )
        self.remove_representative_bond_pair_button.clicked.connect(
            self._remove_selected_representative_bond_pair_rows
        )
        representative_bond_pair_controls.addWidget(
            self.remove_representative_bond_pair_button
        )
        representative_bond_pair_controls.addStretch(1)
        representative_bond_pairs_layout.addLayout(
            representative_bond_pair_controls
        )
        self.representative_bond_pair_table = QTableWidget(0, 3)
        self.representative_bond_pair_table.setHorizontalHeaderLabels(
            ["Atom 1", "Atom 2", "Cutoff (A)"]
        )
        self.representative_bond_pair_table.horizontalHeader().setStretchLastSection(
            True
        )
        representative_bond_pairs_layout.addWidget(
            self.representative_bond_pair_table
        )
        self._add_empty_representative_bond_pair_row(blocked=True)
        self.representative_bond_pairs_row.setVisible(False)
        representative_content_layout.addWidget(
            self.representative_bond_pairs_row
        )

        self.representative_angle_triplets_row = QGroupBox("Angle Triplets")
        representative_angle_triplets_layout = QVBoxLayout(
            self.representative_angle_triplets_row
        )
        representative_angle_triplet_controls = QHBoxLayout()
        self.add_representative_angle_triplet_button = QPushButton(
            "Add Angle Triplet"
        )
        self.add_representative_angle_triplet_button.clicked.connect(
            self._add_representative_angle_triplet_row
        )
        representative_angle_triplet_controls.addWidget(
            self.add_representative_angle_triplet_button
        )
        self.remove_representative_angle_triplet_button = QPushButton(
            "Remove Selected"
        )
        self.remove_representative_angle_triplet_button.clicked.connect(
            self._remove_selected_representative_angle_triplet_rows
        )
        representative_angle_triplet_controls.addWidget(
            self.remove_representative_angle_triplet_button
        )
        representative_angle_triplet_controls.addStretch(1)
        representative_angle_triplets_layout.addLayout(
            representative_angle_triplet_controls
        )
        self.representative_angle_triplet_table = QTableWidget(0, 5)
        self.representative_angle_triplet_table.setHorizontalHeaderLabels(
            [
                "Vertex",
                "Arm 1",
                "Arm 2",
                "Vertex-Arm 1 Cutoff (A)",
                "Vertex-Arm 2 Cutoff (A)",
            ]
        )
        self.representative_angle_triplet_table.horizontalHeader().setStretchLastSection(
            True
        )
        representative_angle_triplets_layout.addWidget(
            self.representative_angle_triplet_table
        )
        self._add_empty_representative_angle_triplet_row(blocked=True)
        self.representative_angle_triplets_row.setVisible(False)
        representative_content_layout.addWidget(
            self.representative_angle_triplets_row
        )

        self.representative_advanced_toggle = QPushButton(
            "Show Advanced Settings"
        )
        self.representative_advanced_toggle.setCheckable(True)
        self.representative_advanced_toggle.toggled.connect(
            self._toggle_representative_advanced_settings
        )
        self.representative_advanced_toggle.setVisible(False)
        representative_content_layout.addWidget(
            self.representative_advanced_toggle
        )

        self.representative_advanced_widget = QWidget()
        representative_advanced_layout = QFormLayout(
            self.representative_advanced_widget
        )
        representative_advanced_layout.setContentsMargins(0, 0, 0, 0)
        self.representative_algorithm_combo = QComboBox()
        for label, value in _REPRESENTATIVE_ALGORITHM_ITEMS:
            self.representative_algorithm_combo.addItem(label, value)
        representative_advanced_layout.addRow(
            "Selection algorithm",
            self.representative_algorithm_combo,
        )
        self.representative_count_cutoff_spin = QSpinBox()
        self.representative_count_cutoff_spin.setRange(1, 1_000_000)
        self.representative_count_cutoff_spin.setValue(1)
        representative_advanced_layout.addRow(
            "Min cluster count",
            self.representative_count_cutoff_spin,
        )
        self.representative_bond_weight_spin = self._new_float_spin(
            maximum=100.0,
            step=0.1,
            decimals=3,
            value=1.0,
        )
        representative_advanced_layout.addRow(
            "Bond weight",
            self.representative_bond_weight_spin,
        )
        self.representative_angle_weight_spin = self._new_float_spin(
            maximum=100.0,
            step=0.1,
            decimals=3,
            value=1.0,
        )
        representative_advanced_layout.addRow(
            "Angle weight",
            self.representative_angle_weight_spin,
        )
        self.representative_advanced_widget.setVisible(False)
        representative_content_layout.addWidget(
            self.representative_advanced_widget
        )

        representative_button_row = QHBoxLayout()
        self.compute_representatives_button = QPushButton(
            "Open Representative Structures"
        )
        self.compute_representatives_button.clicked.connect(
            self._open_representative_structures_tool
        )
        representative_button_row.addWidget(
            self.compute_representatives_button
        )
        self.preview_representatives_button = QPushButton(
            "Reload Saved Representative Structures"
        )
        self.preview_representatives_button.clicked.connect(
            self._reload_saved_representative_structures
        )
        representative_button_row.addWidget(
            self.preview_representatives_button
        )
        representative_button_row.addStretch(1)
        representative_content_layout.addLayout(representative_button_row)

        self.representative_status_label = QLabel(
            "Representative structures: waiting for saved project data."
        )
        self.representative_status_label.setWordWrap(True)
        representative_content_layout.addWidget(
            self.representative_status_label
        )
        self.representative_progress_bar = QProgressBar()
        self.representative_progress_bar.setRange(0, 1)
        self.representative_progress_bar.setValue(0)
        self.representative_progress_bar.setVisible(False)
        representative_content_layout.addWidget(
            self.representative_progress_bar
        )

        self.representative_summary_box = QPlainTextEdit()
        self.representative_summary_box.setReadOnly(True)
        self.representative_summary_box.setMinimumHeight(210)
        representative_content_layout.addWidget(
            self.representative_summary_box
        )
        self._right_layout.addWidget(self.representative_group)

        self.solvent_group = QGroupBox("Solvent Shell Builder")
        solvent_layout = QVBoxLayout(self.solvent_group)
        self._add_group_readiness_row(
            solvent_layout,
            (
                "solvent_outputs",
                "Ready",
                "Checked when the active representative structure set already "
                "has full solvent.",
            ),
            section_key="solvent_outputs",
        )
        solvent_content_layout = self._create_collapsible_section_layout(
            solvent_layout,
            "solvent_outputs",
        )
        self.solvent_intro_label = QLabel(
            "This subsection is active for no-solvent and partial-solvent "
            "representative structure sets. The build action analyzes the "
            "selected representatives, then writes the completed full-solvent "
            "PDBs for previewing and Packmol."
        )
        self.solvent_intro_label.setWordWrap(True)
        solvent_content_layout.addWidget(self.solvent_intro_label)

        solvent_form = QFormLayout()

        self.solvent_reference_source_combo = QComboBox()
        for label, value in _SOLVENT_REFERENCE_SOURCE_ITEMS:
            self.solvent_reference_source_combo.addItem(label, value)
        self.solvent_reference_source_combo.currentIndexChanged.connect(
            self._handle_solvent_reference_source_changed
        )
        solvent_form.addRow(
            "Reference source",
            self.solvent_reference_source_combo,
        )

        self.solvent_preset_combo = QComboBox()
        for preset in self._available_solvent_presets:
            self.solvent_preset_combo.addItem(preset.name, preset.name)
        self.solvent_preset_combo.currentIndexChanged.connect(
            self._handle_solvent_reference_changed
        )
        solvent_form.addRow("Preset reference", self.solvent_preset_combo)

        solvent_path_row = QHBoxLayout()
        self.solvent_reference_edit = QLineEdit()
        self.solvent_reference_edit.setPlaceholderText(
            "Choose a solvent reference PDB"
        )
        self.solvent_reference_edit.editingFinished.connect(
            self._handle_solvent_reference_changed
        )
        solvent_path_row.addWidget(self.solvent_reference_edit, stretch=1)
        self.browse_solvent_reference_button = QPushButton("Browse...")
        self.browse_solvent_reference_button.clicked.connect(
            self._browse_solvent_reference_pdb
        )
        solvent_path_row.addWidget(self.browse_solvent_reference_button)
        solvent_path_widget = QWidget()
        solvent_path_widget.setLayout(solvent_path_row)
        solvent_form.addRow("Custom reference PDB", solvent_path_widget)

        self.solvent_reference_match_tolerance_spin = self._new_float_spin(
            maximum=5.0,
            step=0.05,
            decimals=3,
            value=DEFAULT_REFERENCE_MATCH_TOLERANCE_A,
        )
        self.solvent_reference_match_tolerance_spin.valueChanged.connect(
            self._handle_solvent_analysis_setting_changed
        )
        solvent_form.addRow(
            "Reference match tolerance (A)",
            self.solvent_reference_match_tolerance_spin,
        )

        self.solvent_director_atom_combo = QComboBox()
        solvent_form.addRow(
            "Director atom",
            self.solvent_director_atom_combo,
        )

        self.solvent_minimum_separation_spin = self._new_float_spin(
            maximum=10.0,
            step=0.1,
            decimals=2,
            value=1.2,
        )
        self.solvent_minimum_separation_spin.valueChanged.connect(
            self._update_solvent_build_panel_state
        )
        solvent_form.addRow(
            "Minimum solvent atom separation (A)",
            self.solvent_minimum_separation_spin,
        )
        solvent_content_layout.addLayout(solvent_form)

        self.solvent_reference_details_box = QPlainTextEdit()
        self.solvent_reference_details_box.setReadOnly(True)
        self.solvent_reference_details_box.setMinimumHeight(90)
        solvent_content_layout.addWidget(self.solvent_reference_details_box)

        self.solvent_status_group = QGroupBox(
            "Detected Representative Solvent State"
        )
        solvent_status_layout = QVBoxLayout(self.solvent_status_group)
        self.solvent_status_headline_label = QLabel(
            "No representative solvent state has been determined yet."
        )
        self.solvent_status_headline_label.setWordWrap(True)
        solvent_status_layout.addWidget(self.solvent_status_headline_label)
        self.solvent_status_stats_label = QLabel(
            "Save representative structures, choose a solvent reference, "
            "and press Build Solvent-Decorated Representative PDBs."
        )
        self.solvent_status_stats_label.setWordWrap(True)
        self.solvent_status_stats_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        solvent_status_layout.addWidget(self.solvent_status_stats_label)
        solvent_content_layout.addWidget(self.solvent_status_group)

        self.solvent_cutoff_group = QGroupBox("Solute Coordination Settings")
        solvent_cutoff_layout = QVBoxLayout(self.solvent_cutoff_group)
        self.solvent_cutoff_status_label = QLabel(
            "Analyze or build the representative structures to populate the "
            "solute atom types used for solvent-shell building."
        )
        self.solvent_cutoff_status_label.setWordWrap(True)
        solvent_cutoff_layout.addWidget(self.solvent_cutoff_status_label)
        self.solvent_cutoff_table = QTableWidget(0, 5)
        self.solvent_cutoff_table.setHorizontalHeaderLabels(
            [
                "Element",
                "Count",
                "Coordination Center",
                "Avg Coord #",
                "Director Distance (A)",
            ]
        )
        self.solvent_cutoff_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers
        )
        self.solvent_cutoff_table.itemChanged.connect(
            self._handle_solvent_coordination_table_item_changed
        )
        self.solvent_cutoff_table.horizontalHeader().setStretchLastSection(
            True
        )
        solvent_cutoff_layout.addWidget(self.solvent_cutoff_table)
        solvent_content_layout.addWidget(self.solvent_cutoff_group)

        solvent_button_row = QHBoxLayout()
        self.analyze_solvent_outputs_button = QPushButton(
            "Analyze / Refresh Solvent State"
        )
        self.analyze_solvent_outputs_button.clicked.connect(
            self._analyze_representative_solvent_states
        )
        solvent_button_row.addWidget(self.analyze_solvent_outputs_button)
        self.build_solvent_outputs_button = QPushButton(
            "Build Solvent-Decorated Representative PDBs"
        )
        self.build_solvent_outputs_button.clicked.connect(
            self._build_representative_solvent_outputs
        )
        solvent_button_row.addWidget(self.build_solvent_outputs_button)
        solvent_button_row.addStretch(1)
        solvent_content_layout.addLayout(solvent_button_row)

        self.solvent_summary_box = QPlainTextEdit()
        self.solvent_summary_box.setReadOnly(True)
        self.solvent_summary_box.setMinimumHeight(190)
        solvent_content_layout.addWidget(self.solvent_summary_box)

        self.generated_pdb_group = QGroupBox(
            "Active Representative Structures"
        )
        generated_pdb_layout = QVBoxLayout(self.generated_pdb_group)
        self.generated_pdb_intro_label = QLabel(
            "Choose the active representative structure set, then select a "
            "row to inspect the currently selected representative structure "
            "and preview it directly below. Solvent-aware variants appear "
            "here after the Solvent Shell Builder has run."
        )
        self.generated_pdb_intro_label.setWordWrap(True)
        generated_pdb_layout.addWidget(self.generated_pdb_intro_label)

        generated_pdb_mode_row = QHBoxLayout()
        generated_pdb_mode_row.addWidget(QLabel("Active structure set"))
        self.generated_pdb_mode_combo = QComboBox()
        self.generated_pdb_mode_combo.currentIndexChanged.connect(
            self._handle_generated_pdb_mode_changed
        )
        generated_pdb_mode_row.addWidget(
            self.generated_pdb_mode_combo,
            stretch=1,
        )
        generated_pdb_layout.addLayout(generated_pdb_mode_row)

        generated_pdb_button_row = QHBoxLayout()
        self.open_generated_pdb_preview_button = QPushButton(
            "Open Selected Preview"
        )
        self.open_generated_pdb_preview_button.clicked.connect(
            self._open_selected_generated_pdb_preview
        )
        generated_pdb_button_row.addWidget(
            self.open_generated_pdb_preview_button
        )
        generated_pdb_button_row.addStretch(1)
        generated_pdb_layout.addLayout(generated_pdb_button_row)

        self.generated_pdb_table = QTableWidget(0, 6)
        self.generated_pdb_table.setHorizontalHeaderLabels(
            [
                "Representative",
                "Detected State",
                "Active Set",
                "Structure File",
                "Atoms",
                "Source",
            ]
        )
        self.generated_pdb_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self.generated_pdb_table.setSelectionMode(
            QTableWidget.SelectionMode.SingleSelection
        )
        self.generated_pdb_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers
        )
        self.generated_pdb_table.horizontalHeader().setStretchLastSection(True)
        self.generated_pdb_table.itemSelectionChanged.connect(
            self._on_generated_pdb_selection_changed
        )
        self.generated_pdb_table.itemDoubleClicked.connect(
            lambda _item: self._open_selected_generated_pdb_preview()
        )
        generated_pdb_layout.addWidget(self.generated_pdb_table)

        self.generated_pdb_details_box = QPlainTextEdit()
        self.generated_pdb_details_box.setReadOnly(True)
        self.generated_pdb_details_box.setMinimumHeight(150)
        generated_pdb_layout.addWidget(self.generated_pdb_details_box)

        self.generated_pdb_viewer_status_label = QLabel(
            "Select a representative row to preview the active structure."
        )
        self.generated_pdb_viewer_status_label.setWordWrap(True)
        generated_pdb_layout.addWidget(self.generated_pdb_viewer_status_label)
        self.generated_pdb_viewer = ElectronDensityStructureViewer(
            self.generated_pdb_group
        )
        self.generated_pdb_viewer.setMinimumHeight(360)
        generated_pdb_layout.addWidget(self.generated_pdb_viewer, stretch=1)
        representative_content_layout.addWidget(self.generated_pdb_group)
        self._right_layout.addWidget(self.solvent_group)

        self.packmol_group = QGroupBox("Packmol Planning")
        packmol_layout = QVBoxLayout(self.packmol_group)
        self._add_group_readiness_row(
            packmol_layout,
            (
                "packmol_plan",
                "Plan Ready",
                "Checked after Packmol cluster counts are computed.",
            ),
            (
                "packmol_setup",
                "Setup Ready",
                "Checked after Packmol setup inputs are built.",
            ),
            section_key="packmol",
        )
        packmol_content_layout = self._create_collapsible_section_layout(
            packmol_layout,
            "packmol",
        )
        self.packmol_docker_group = QGroupBox("Linked Packmol Docker")
        packmol_docker_layout = QVBoxLayout(self.packmol_docker_group)
        self.packmol_docker_hint_label = QLabel(
            "Use Tools > Link Packmol Docker Container to validate a "
            "container, confirm Packmol is installed, and select the "
            f"container-side project folder inside {DEFAULT_PACKMOL_CONTAINER_ROOT}."
        )
        self.packmol_docker_hint_label.setWordWrap(True)
        packmol_docker_layout.addWidget(self.packmol_docker_hint_label)
        self.packmol_docker_summary_box = QPlainTextEdit()
        self.packmol_docker_summary_box.setReadOnly(True)
        self.packmol_docker_summary_box.setMinimumHeight(150)
        packmol_docker_layout.addWidget(self.packmol_docker_summary_box)
        packmol_content_layout.addWidget(self.packmol_docker_group)
        packmol_form = QFormLayout()

        self.packmol_planning_mode_combo = QComboBox()
        for label, value in _PACKMOL_PLANNING_MODE_ITEMS:
            self.packmol_planning_mode_combo.addItem(label, value)
        packmol_form.addRow(
            "Planning mode",
            self.packmol_planning_mode_combo,
        )

        self.packmol_box_side_spin = self._new_float_spin(
            maximum=10_000.0,
            step=1.0,
            decimals=3,
            value=100.0,
        )
        packmol_form.addRow(
            "Box side length (A)",
            self.packmol_box_side_spin,
        )
        self.packmol_free_solvent_combo = QComboBox()
        packmol_form.addRow(
            "Free solvent structure",
            self.packmol_free_solvent_combo,
        )
        packmol_content_layout.addLayout(packmol_form)

        packmol_button_row = QHBoxLayout()
        self.compute_packmol_plan_button = QPushButton(
            "Compute Cluster Counts"
        )
        self.compute_packmol_plan_button.clicked.connect(
            self._compute_packmol_plan
        )
        packmol_button_row.addWidget(self.compute_packmol_plan_button)
        packmol_button_row.addWidget(QLabel("Tolerance (A)"))
        self.packmol_tolerance_spin = QDoubleSpinBox()
        self.packmol_tolerance_spin.setDecimals(3)
        self.packmol_tolerance_spin.setRange(0.1, 100.0)
        self.packmol_tolerance_spin.setSingleStep(0.1)
        self.packmol_tolerance_spin.setValue(2.0)
        self.packmol_tolerance_spin.setSuffix(" A")
        packmol_button_row.addWidget(self.packmol_tolerance_spin)
        self.build_packmol_setup_button = QPushButton("Build Packmol Setup")
        self.build_packmol_setup_button.clicked.connect(
            self._build_packmol_setup
        )
        packmol_button_row.addWidget(self.build_packmol_setup_button)
        self.open_packmol_setup_folder_button = QPushButton(
            "Open Packmol Setup Folder"
        )
        self.open_packmol_setup_folder_button.clicked.connect(
            self._open_packmol_setup_folder
        )
        self.open_packmol_setup_folder_button.setEnabled(False)
        packmol_button_row.addWidget(self.open_packmol_setup_folder_button)
        packmol_button_row.addStretch(1)
        packmol_content_layout.addLayout(packmol_button_row)

        self.packmol_plan_summary_box = QPlainTextEdit()
        self.packmol_plan_summary_box.setReadOnly(True)
        self.packmol_plan_summary_box.setMinimumHeight(180)
        packmol_content_layout.addWidget(self.packmol_plan_summary_box)
        packmol_content_layout.addWidget(self._packmol_plan_toolbar)
        self._packmol_plan_canvas.setMinimumHeight(260)
        packmol_content_layout.addWidget(self._packmol_plan_canvas)
        self.packmol_build_summary_box = QPlainTextEdit()
        self.packmol_build_summary_box.setReadOnly(True)
        self.packmol_build_summary_box.setMinimumHeight(150)
        packmol_content_layout.addWidget(self.packmol_build_summary_box)
        self._right_layout.addWidget(self.packmol_group)

        self.constraints_group = QGroupBox("Constraint Generation")
        constraints_layout = QVBoxLayout(self.constraints_group)
        constraints_form = QFormLayout()
        self.constraint_length_tolerance_spin = self._new_float_spin(
            maximum=10.0,
            step=0.01,
            decimals=3,
            value=0.05,
        )
        constraints_form.addRow(
            "Bond-length tolerance (A)",
            self.constraint_length_tolerance_spin,
        )
        self.constraint_angle_tolerance_spin = self._new_float_spin(
            maximum=180.0,
            step=0.5,
            decimals=3,
            value=5.0,
        )
        constraints_form.addRow(
            "Bond-angle tolerance (deg)",
            self.constraint_angle_tolerance_spin,
        )
        constraints_layout.addLayout(constraints_form)

        constraints_button_row = QHBoxLayout()
        self.generate_constraints_button = QPushButton("Generate Constraints")
        self.generate_constraints_button.clicked.connect(
            self._generate_constraints
        )
        constraints_button_row.addWidget(self.generate_constraints_button)
        self.open_constraints_folder_button = QPushButton(
            "Open Constraints Folder"
        )
        self.open_constraints_folder_button.clicked.connect(
            self._open_constraints_folder
        )
        self.open_constraints_folder_button.setEnabled(False)
        constraints_button_row.addWidget(self.open_constraints_folder_button)
        self.preview_constraints_button = QPushButton(
            "Show Merged Constraints"
        )
        self.preview_constraints_button.clicked.connect(
            self._open_constraints_preview
        )
        self.preview_constraints_button.setEnabled(False)
        constraints_button_row.addWidget(self.preview_constraints_button)
        constraints_button_row.addStretch(1)
        constraints_layout.addLayout(constraints_button_row)

        self.constraints_summary_box = QPlainTextEdit()
        self.constraints_summary_box.setReadOnly(True)
        self.constraints_summary_box.setMinimumHeight(170)
        constraints_layout.addWidget(self.constraints_summary_box)
        self._right_layout.addWidget(self.constraints_group)

        self.workflow_group = QGroupBox("Planned Workflow")
        workflow_layout = QVBoxLayout(self.workflow_group)
        self.workflow_box = QPlainTextEdit()
        self.workflow_box.setReadOnly(True)
        self.workflow_box.setPlainText(
            "\n".join(
                [
                    "1. Load a SAXS project and choose a DREAM result source.",
                    "2. Open Representative Structures and save the representative project set.",
                    "3. Enter solution properties for the target box density and composition.",
                    "4. Build solvent shells as needed and convert DREAM "
                    "weights into Packmol cluster counts.",
                    "5. Build the Packmol box inputs and cluster-specific fullrmc constraints.",
                ]
            )
        )
        workflow_layout.addWidget(self.workflow_box)
        self._right_layout.addWidget(self.workflow_group)

        self.run_log_group = QGroupBox("Run Log")
        run_log_layout = QVBoxLayout(self.run_log_group)
        self.run_log_box = QPlainTextEdit()
        self.run_log_box.setReadOnly(True)
        self.run_log_box.setPlainText(
            "RMC setup scaffold ready.\n"
            "No full workflow has been executed yet."
        )
        run_log_layout.addWidget(self.run_log_box)
        self._right_layout.addWidget(self.run_log_group)
        self._reload_solution_presets()
        self._reload_representative_presets()
        self._configure_tooltips()
        self._left_layout.addStretch(1)
        self._right_layout.addStretch(1)
        self._main_splitter.setSizes([540, 540])

        if initial_project_dir is not None:
            self.set_project_dir(initial_project_dir)
        else:
            self._refresh_project_source()

    def _build_menu_bar(self) -> None:
        menu_bar = self.menuBar()
        self.tools_menu = menu_bar.addMenu("Tools")
        self.open_representative_structures_action = QAction(
            "Open Representative Structures",
            self,
        )
        self.open_representative_structures_action.triggered.connect(
            self._open_representative_structures_tool
        )
        self.tools_menu.addAction(self.open_representative_structures_action)
        self.link_packmol_docker_action = QAction(
            "Link Packmol Docker Container",
            self,
        )
        self.link_packmol_docker_action.triggered.connect(
            self._open_packmol_docker_link_dialog
        )
        self.tools_menu.addSeparator()
        self.tools_menu.addAction(self.link_packmol_docker_action)

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

    def _reload_saved_representative_structures(self) -> None:
        if self.project_dir() is None:
            QMessageBox.information(
                self,
                "No SAXS project loaded",
                "Load a SAXS project before reloading representative structures.",
            )
            return
        self._append_run_log("Reloading saved representative structures.")
        self._refresh_project_source()

    def _handle_representative_structure_results_changed(
        self,
        project_dir_text: str,
    ) -> None:
        current_project_dir = self.project_dir()
        if current_project_dir is None:
            return
        try:
            changed_project_dir = Path(project_dir_text).expanduser().resolve()
        except Exception:
            return
        if changed_project_dir != current_project_dir:
            return
        self._append_run_log(
            "Representative structures were updated in the dedicated tool; "
            "reloading the saved project set."
        )
        self._refresh_project_source()

    def _open_representative_structures_tool(self) -> None:
        from saxshell.representativefinder.ui.main_window import (
            launch_representativefinder_ui,
        )

        state = self._project_source_state
        if state is None:
            QMessageBox.information(
                self,
                "No SAXS project loaded",
                "Load a SAXS project before opening Representative Structures.",
            )
            return
        project_dir = Path(state.settings.project_dir).resolve()
        initial_input_path = state.settings.resolved_clusters_dir
        window = launch_representativefinder_ui(
            initial_project_dir=project_dir,
            initial_input_path=initial_input_path,
        )
        project_results_changed = getattr(
            window, "project_results_changed", None
        )
        if project_results_changed is not None and hasattr(
            project_results_changed, "connect"
        ):
            project_results_changed.connect(
                self._handle_representative_structure_results_changed
            )
        self._track_child_tool_window(window)
        self.statusBar().showMessage(
            f"Opened representative structures for {project_dir}"
        )
        self._append_run_log(
            "Opened the Representative Structures tool for this project."
        )

    def project_dir(self) -> Path | None:
        text = self.project_dir_edit.text().strip()
        if not text:
            return None
        return Path(text).expanduser().resolve()

    def set_project_dir(self, project_dir: str | Path | None) -> None:
        if project_dir is None:
            self.project_dir_edit.clear()
        else:
            self.project_dir_edit.setText(
                str(Path(project_dir).expanduser().resolve())
            )
        self._refresh_project_source()

    def _browse_project_dir(self) -> None:
        start_dir = self.project_dir_edit.text().strip() or str(Path.home())
        selected = QFileDialog.getExistingDirectory(
            self,
            "Choose SAXS Project Directory",
            start_dir,
        )
        if not selected:
            return
        self.set_project_dir(selected)

    def _toggle_software_details(self, checked: bool) -> None:
        self.software_details_panel.setVisible(checked)

    def _packmol_docker_settings(self) -> QSettings:
        return QSettings("SAXShell", "RMCSetup")

    def _create_packmol_docker_client(self) -> PackmolDockerClient:
        return PackmolDockerClient()

    def _recent_packmol_docker_presets(self) -> list[PackmolDockerLink]:
        raw_value = self._packmol_docker_settings().value(
            _PACKMOL_DOCKER_PRESETS_KEY,
            "[]",
        )
        if isinstance(raw_value, str):
            try:
                payload = json.loads(raw_value)
            except Exception:
                payload = []
        elif isinstance(raw_value, (list, tuple)):
            payload = list(raw_value)
        else:
            payload = []
        presets: list[PackmolDockerLink] = []
        for entry in payload:
            preset = PackmolDockerLink.from_dict(
                dict(entry) if isinstance(entry, dict) else None
            )
            if preset is not None:
                presets.append(preset)
        return presets

    def _remember_packmol_docker_preset(
        self,
        link: PackmolDockerLink,
    ) -> None:
        preset = PackmolDockerLink.from_dict(link.to_preset_dict())
        if preset is None:
            return
        signature = (
            preset.container_name,
            preset.packmol_command,
            preset.shell_command,
            preset.container_project_root,
        )
        kept = [
            existing
            for existing in self._recent_packmol_docker_presets()
            if (
                existing.container_name,
                existing.packmol_command,
                existing.shell_command,
                existing.container_project_root,
            )
            != signature
        ]
        payload = [preset.to_preset_dict()] + [
            item.to_preset_dict() for item in kept[:7]
        ]
        self._packmol_docker_settings().setValue(
            _PACKMOL_DOCKER_PRESETS_KEY,
            json.dumps(payload),
        )

    def _save_packmol_docker_link(
        self,
        link: PackmolDockerLink | None,
    ) -> None:
        state = self._project_source_state
        if state is None:
            return
        save_packmol_docker_link_metadata(
            state.rmcsetup_paths.packmol_docker_link_path,
            link,
        )
        state.packmol_docker_link = link

    def _open_packmol_docker_link_dialog(self) -> None:
        state = self._project_source_state
        if state is None:
            QMessageBox.information(
                self,
                "No SAXS project loaded",
                "Load a SAXS project before linking a Packmol Docker container.",
            )
            return
        dialog = PackmolDockerLinkDialog(
            current_link=state.packmol_docker_link,
            recent_presets=self._recent_packmol_docker_presets(),
            docker_client=self._create_packmol_docker_client(),
            parent=self,
        )
        if not dialog.exec():
            return
        link = dialog.selected_link()
        if link is None:
            return
        link.linked_at = datetime.now().isoformat(timespec="seconds")
        self._remember_packmol_docker_preset(link)
        self._save_packmol_docker_link(link)
        self.packmol_docker_summary_box.setPlainText(
            self._packmol_docker_summary_text(state.packmol_setup)
        )
        self.output_summary_box.setPlainText(self._output_structure_text())
        self._append_run_log(
            "Linked Packmol Docker container "
            f"{link.container_name} at {link.container_project_root}."
        )

    def _refresh_project_source(self) -> None:
        self._set_task_progress("Loading project source...", 10)
        project_dir = self.project_dir()
        self._project_source_state = None
        self._dream_results_loader_cache.clear()
        if project_dir is not None and project_dir.exists():
            try:
                self._project_source_state = load_rmc_project_source(
                    project_dir
                )
                self.project_dir_edit.setText(
                    self._project_source_state.settings.project_dir
                )
            except Exception as exc:
                self._append_run_log(f"Unable to load project source: {exc}")
        self._populate_dream_controls()
        self._populate_favorite_controls()
        self._populate_solution_properties_controls()
        self._populate_representative_controls()
        self._populate_solvent_controls()
        self._populate_packmol_planning_controls()
        self._populate_constraint_controls()
        self.project_summary_box.setPlainText(self._project_summary_text())
        self.output_summary_box.setPlainText(self._output_structure_text())
        self.favorite_summary_box.setPlainText(self._favorite_summary_text())
        self._initialize_selection_from_state()
        self._refresh_dream_source_summary()
        self._update_readiness_progress()
        self._set_task_progress("Project source loaded.", 100)

    def _populate_dream_controls(self) -> None:
        self._updating_dream_controls = True
        try:
            self.dream_run_combo.clear()
            state = self._project_source_state
            if state is not None:
                for run in state.valid_runs:
                    self.dream_run_combo.addItem(run.run_name, run)
                    index = self.dream_run_combo.count() - 1
                    self.dream_run_combo.setItemData(
                        index,
                        run.relative_path,
                        Qt.ItemDataRole.ToolTipRole,
                    )
        finally:
            self._updating_dream_controls = False
        self._update_posterior_filter_widgets()

    def _populate_favorite_controls(self) -> None:
        self.favorite_history_combo.clear()
        state = self._project_source_state
        if state is None:
            self._set_favorite_controls_enabled(False)
            return
        self._set_favorite_controls_enabled(True)
        for entry in state.favorite_history:
            label = self._selection_label(entry)
            self.favorite_history_combo.addItem(label, entry)
            index = self.favorite_history_combo.count() - 1
            self.favorite_history_combo.setItemData(
                index,
                self._selection_metadata_text(entry),
                Qt.ItemDataRole.ToolTipRole,
            )
        has_history = self.favorite_history_combo.count() > 0
        self.load_history_button.setEnabled(has_history)
        self.use_project_favorite_button.setEnabled(
            state.favorite_selection is not None
        )

    def _initialize_selection_from_state(self) -> None:
        state = self._project_source_state
        if state is None:
            return
        selection = state.favorite_selection
        if selection is not None and state.find_run_for_selection(selection):
            self._apply_selection(selection, announce=False)
            return
        if state.valid_runs:
            self._apply_run_defaults(state.valid_runs[0], announce=False)

    def _project_summary_text(self) -> str:
        project_dir = self.project_dir()
        if project_dir is None:
            return (
                "No project directory selected.\n\n"
                "Choose an existing SAXS project folder. rmcsetup will use "
                "this project to discover DREAM outputs, cluster sources, "
                "representative structures, and later Packmol/fullrmc "
                "setup outputs."
            )

        project_file = project_dir / "saxs_project.json"
        state = self._project_source_state
        valid_runs = state.valid_runs if state is not None else []
        lines = [
            f"Project directory: {project_dir}",
            f"Directory exists: {'yes' if project_dir.exists() else 'no'}",
            (
                "SAXS project file detected: "
                f"{'yes' if project_file.exists() else 'no'}"
            ),
            f"Valid DREAM runs discovered: {len(valid_runs)}",
        ]
        if valid_runs:
            preview = ", ".join(run.run_name for run in valid_runs[:5])
            if len(valid_runs) > 5:
                preview += ", ..."
            lines.append(f"Run preview: {preview}")
        if state is not None:
            lines.append(
                "Saved DREAM model configured: "
                f"{'yes' if state.favorite_selection is not None else 'no'}"
            )
            lines.append(
                (
                    "Saved selection history entries: "
                    f"{len(state.favorite_history)}"
                )
            )
        lines.append("")
        if project_file.exists() and valid_runs:
            lines.append(
                "This project has valid DREAM outputs ready for source "
                "selection in rmcsetup."
            )
        elif project_file.exists():
            lines.append(
                "This project is valid, but no completed DREAM runs were "
                "discovered yet."
            )
        else:
            lines.append(
                "This directory is not a complete SAXS project source."
            )
        return "\n".join(lines)

    def _output_structure_text(self) -> str:
        state = self._project_source_state
        if state is None:
            return (
                "Load a SAXS project to create and inspect the /rmcsetup/ "
                "output structure."
            )
        paths = state.rmcsetup_paths
        lines = [
            f"Root: {paths.rmcsetup_dir}",
            (
                "Representative structures root: "
                f"{paths.representative_clusters_dir}"
            ),
            (
                "Representative metadata: "
                f"{paths.representative_selection_path}"
            ),
            (
                "Partial-solvent representatives: "
                f"{paths.representative_partial_solvent_dir}"
            ),
            f"PDBs without solvent: {paths.pdb_no_solvent_dir}",
            f"PDBs with solvent: {paths.pdb_with_solvent_dir}",
            f"Packmol inputs: {paths.packmol_inputs_dir}",
            f"Constraints: {paths.constraints_dir}",
            f"Reports: {paths.reports_dir}",
            ("Distribution metadata: " f"{paths.distribution_selection_path}"),
            ("Solution metadata: " f"{paths.solution_properties_path}"),
            ("Solvent metadata: " f"{paths.solvent_handling_path}"),
            ("Packmol Docker link: " f"{paths.packmol_docker_link_path}"),
            ("Packmol plan metadata: " f"{paths.packmol_plan_path}"),
            ("Packmol setup metadata: " f"{paths.packmol_setup_path}"),
            ("Constraint metadata: " f"{paths.constraint_generation_path}"),
            ("Packmol plan report: " f"{paths.packmol_plan_report_path}"),
            ("Packmol audit report: " f"{paths.packmol_audit_report_path}"),
            ("Cluster counts report: " f"{paths.cluster_counts_csv_path}"),
            ("Packmol input file: " f"{paths.packmol_input_path}"),
            ("Merged constraints file: " f"{paths.merged_constraints_path}"),
        ]
        return "\n".join(lines)

    def _favorite_summary_text(self) -> str:
        state = self._project_source_state
        if state is None:
            return (
                "Load a SAXS project to review the saved DREAM model and "
                "its saved history."
            )
        lines = [
            "Active saved DREAM model:",
        ]
        if state.favorite_selection is None:
            lines.append("  none")
        else:
            lines.append(
                self._selection_metadata_text(
                    state.favorite_selection, indent="  "
                )
            )
        lines.append("")
        lines.append(
            f"History entries available: {len(state.favorite_history)}"
        )
        if state.favorite_history:
            lines.append("Most recent saved entry:")
            lines.append(
                self._selection_metadata_text(
                    state.favorite_history[-1],
                    indent="  ",
                )
            )
        return "\n".join(lines)

    def _update_readiness_progress(self) -> None:
        readiness_states = self._readiness_states()
        completed = sum(
            1 for is_complete in readiness_states.values() if is_complete
        )
        for key, checkbox in self._readiness_checkboxes.items():
            checkbox.setChecked(readiness_states.get(key, False))
        self.readiness_progress_bar.setValue(completed)
        self.readiness_label.setText(
            "Project readiness: "
            f"{completed} / {len(readiness_states)} complete"
        )

    def _set_task_progress(self, message: str, value: int) -> None:
        bounded = max(0, min(int(value), 100))
        self.task_status_label.setText(f"Current task: {message}")
        self.task_progress_bar.setValue(bounded)

    def _warn_if_cluster_validation_needs_attention(
        self,
        action_label: str,
    ) -> None:
        state = self._project_source_state
        if state is None or state.cluster_validation.is_valid:
            return
        warning_text = (
            f"{action_label} is using a cluster source that differs from the "
            "saved project snapshot. Continue carefully and revalidate the "
            "cluster folder if results look inconsistent."
        )
        self._append_run_log(f"WARNING: {warning_text}")
        QMessageBox.warning(
            self,
            "Cluster validation warning",
            warning_text,
        )

    def _on_dream_run_changed(self) -> None:
        if self._updating_dream_controls:
            return
        run = self.selected_run_record()
        if run is not None:
            self._apply_run_defaults(run)
        else:
            self._refresh_dream_source_summary()

    def _apply_run_defaults(
        self,
        run: RMCDreamRunRecord,
        *,
        announce: bool = True,
    ) -> None:
        selection = DreamBestFitSelection(
            run_name=run.run_name,
            run_relative_path=run.relative_path,
            bestfit_method=run.settings.bestfit_method,
            posterior_filter_mode=run.settings.posterior_filter_mode,
            posterior_top_percent=run.settings.posterior_top_percent,
            posterior_top_n=run.settings.posterior_top_n,
            credible_interval_low=run.settings.credible_interval_low,
            credible_interval_high=run.settings.credible_interval_high,
            template_name=run.template_name,
            model_name=run.model_name,
            selection_source="rmcsetup_default",
        )
        self._apply_selection(selection, announce=announce)

    def _apply_selection(
        self,
        selection: DreamBestFitSelection,
        *,
        announce: bool = True,
    ) -> None:
        state = self._project_source_state
        if state is None:
            return
        run = state.find_run_for_selection(selection)
        if run is None:
            if announce:
                QMessageBox.warning(
                    self,
                    "Missing DREAM run",
                    (
                        "The selected DREAM source could not be found in the "
                        "current project. Reload the project or choose a "
                        "different run."
                    ),
                )
            return
        self._updating_dream_controls = True
        try:
            self._set_combo_value(
                self.dream_run_combo,
                run,
                compare=lambda current, target: (
                    isinstance(current, RMCDreamRunRecord)
                    and current.relative_path == target.relative_path
                ),
            )
            self._set_combo_value(
                self.bestfit_method_combo,
                selection.bestfit_method,
            )
            self._set_combo_value(
                self.posterior_filter_combo,
                selection.posterior_filter_mode,
            )
            self.posterior_top_percent_spin.setValue(
                selection.posterior_top_percent
            )
            self.posterior_top_n_spin.setValue(selection.posterior_top_n)
            self.credible_interval_low_spin.setValue(
                selection.credible_interval_low
            )
            self.credible_interval_high_spin.setValue(
                selection.credible_interval_high
            )
        finally:
            self._updating_dream_controls = False
        self._update_posterior_filter_widgets()
        self._refresh_dream_source_summary()
        if announce:
            self._append_run_log(
                f"Applied DREAM selection: {self._selection_label(selection)}"
            )

    def _set_combo_value(
        self,
        combo: QComboBox,
        target: object,
        *,
        compare: Callable[[object, object], bool] | None = None,
    ) -> None:
        for index in range(combo.count()):
            current = combo.itemData(index)
            matches = (
                compare(current, target)
                if compare is not None
                else current == target
            )
            if matches:
                was_blocked = combo.blockSignals(True)
                combo.setCurrentIndex(index)
                combo.blockSignals(was_blocked)
                return

    @staticmethod
    def _new_float_spin(
        *,
        maximum: float,
        step: float,
        decimals: int = 6,
        value: float = 0.0,
    ) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(0.0, maximum)
        spin.setDecimals(decimals)
        spin.setSingleStep(step)
        spin.setValue(value)
        return spin

    def _add_group_readiness_row(
        self,
        layout: QVBoxLayout,
        *definitions: tuple[str, str, str],
        section_key: str | None = None,
    ) -> None:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        if section_key is not None:
            toggle = QToolButton()
            toggle.setCheckable(True)
            toggle.setChecked(True)
            toggle.setArrowType(Qt.ArrowType.DownArrow)
            toggle.setToolButtonStyle(
                Qt.ToolButtonStyle.ToolButtonTextBesideIcon
            )
            toggle.setText("Collapse")
            toggle.setToolTip(
                "Collapse or expand this section while keeping its "
                "readiness status visible."
            )
            toggle.toggled.connect(
                lambda checked, key=section_key: self._toggle_section_content(
                    key,
                    checked,
                )
            )
            self._section_toggle_buttons[section_key] = toggle
            row.addWidget(toggle)
        row.addStretch(1)
        for key, label, tooltip in definitions:
            checkbox = QCheckBox(label)
            checkbox.setChecked(False)
            checkbox.setEnabled(False)
            checkbox.setStyleSheet("QCheckBox:disabled { color: #3c3c3c; }")
            checkbox.setToolTip(_readiness_task_tooltip(key) or tooltip)
            self._readiness_checkboxes[key] = checkbox
            row.addWidget(checkbox)
        layout.addLayout(row)

    def _create_collapsible_section_layout(
        self,
        layout: QVBoxLayout,
        section_key: str,
    ) -> QVBoxLayout:
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(layout.spacing())
        layout.addWidget(content_widget)
        self._section_content_widgets[section_key] = content_widget
        self._toggle_section_content(section_key, True)
        return content_layout

    def _toggle_section_content(
        self,
        section_key: str,
        expanded: bool,
    ) -> None:
        content_widget = self._section_content_widgets.get(section_key)
        if content_widget is not None:
            content_widget.setVisible(expanded)
        toggle = self._section_toggle_buttons.get(section_key)
        if toggle is not None:
            toggle.blockSignals(True)
            toggle.setChecked(expanded)
            toggle.setArrowType(
                Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
            )
            toggle.setText("Collapse" if expanded else "Expand")
            toggle.blockSignals(False)

    def _readiness_states(self) -> dict[str, bool]:
        state = self._project_source_state
        has_project = state is not None
        return {
            "project_source": has_project,
            "dream_selection": (
                has_project and self.selected_run_record() is not None
            ),
            "solution_properties": (
                has_project and state.solution_properties.result is not None
            ),
            "representative_selection": (
                has_project and state.representative_selection is not None
            ),
            "solvent_outputs": (
                has_project
                and self._active_representative_structure_set_is_ready()
            ),
            "packmol_plan": (
                has_project and state.packmol_planning is not None
            ),
            "packmol_setup": (has_project and state.packmol_setup is not None),
        }

    @staticmethod
    def _set_widget_tooltip(widget: QWidget, tooltip: str) -> None:
        widget.setToolTip(tooltip)

    def _configure_tooltips(self) -> None:
        self._set_widget_tooltip(
            self.software_details_button,
            "Show or hide the software references that rmcsetup builds on.",
        )
        self._set_widget_tooltip(
            self.project_dir_edit,
            "Path to the SAXS project folder that contains the saved DREAM "
            "runs and cluster metadata used by rmcsetup.",
        )
        self._set_widget_tooltip(
            self.browse_button,
            "Browse to a SAXS project folder with DREAM outputs.",
        )
        self._set_widget_tooltip(
            self.refresh_button,
            "Reload the selected project folder and rescan its saved data.",
        )
        self._set_widget_tooltip(
            self.dream_run_combo,
            "Choose the DREAM run whose model fit and weights should drive "
            "the downstream RMC setup workflow.",
        )
        self._set_widget_tooltip(
            self.reload_runs_button,
            "Rescan the current project for completed DREAM runs.",
        )
        self._set_widget_tooltip(
            self.bestfit_method_combo,
            "Choose which posterior summary is used as the selected DREAM "
            "model parameters.",
        )
        self._set_widget_tooltip(
            self.posterior_filter_combo,
            "Choose how posterior samples are filtered before computing the "
            "selected model, intervals, and weight preview.",
        )
        self._set_widget_tooltip(
            self.posterior_top_percent_spin,
            "When Top % is selected, keep this percentage of the highest "
            "log-posterior samples.",
        )
        self._set_widget_tooltip(
            self.posterior_top_n_spin,
            "When Top N is selected, keep this many highest "
            "log-posterior samples.",
        )
        credible_interval_tip = (
            "Set the lower and upper percentiles used for the displayed "
            "credible interval on the weight preview."
        )
        self._set_widget_tooltip(
            self.credible_interval_low_spin,
            credible_interval_tip,
        )
        self._set_widget_tooltip(
            self.credible_interval_high_spin,
            credible_interval_tip,
        )
        self._set_widget_tooltip(
            self.set_favorite_button,
            "Save the current DREAM model selection so the project can "
            "reopen with the same run and posterior settings.",
        )
        self._set_widget_tooltip(
            self.use_project_favorite_button,
            "Reload the project's saved DREAM model selection into the "
            "active controls.",
        )
        self._set_widget_tooltip(
            self.favorite_history_combo,
            "Review previously saved DREAM model selections for this "
            "project.",
        )
        self._set_widget_tooltip(
            self.load_history_button,
            "Load the highlighted saved DREAM model history entry.",
        )
        self._set_widget_tooltip(
            self.show_experimental_trace_checkbox,
            "Show or hide the experimental scattering points in the DREAM "
            "model-fit preview.",
        )
        self._set_widget_tooltip(
            self.show_model_trace_checkbox,
            "Show or hide the selected DREAM model-fit trace.",
        )
        self._set_widget_tooltip(
            self.show_solvent_trace_checkbox,
            "Show or hide the solvent-contribution trace. This starts off "
            "hidden when DREAM data is loaded.",
        )
        self._set_widget_tooltip(
            self.solution_mode_combo,
            "Choose how solution properties are entered before converting "
            "them to number density and Packmol planning inputs.",
        )
        self._set_widget_tooltip(
            self.solution_preset_combo,
            "Load a bundled solution preset or one saved to the editable "
            "solution-property preset JSON.",
        )
        self._set_widget_tooltip(
            self.load_solution_preset_button,
            "Apply the selected solution preset to the current fields.",
        )
        self._set_widget_tooltip(
            self.save_solution_preset_button,
            "Save the current solution-property inputs as a named preset in "
            "the editable package preset JSON.",
        )
        self._set_widget_tooltip(
            self.solution_density_spin,
            "Solution density used to convert the chosen composition into "
            "number density.",
        )
        self._set_widget_tooltip(
            self.solute_stoich_edit,
            "Empirical formula for the solute, for example Cs1Pb1I3.",
        )
        self._set_widget_tooltip(
            self.solvent_stoich_edit,
            "Empirical formula for the solvent, for example H2O or C3H7NO.",
        )
        self._set_widget_tooltip(
            self.molar_mass_solute_spin,
            "Molar mass of the solute used in the solution-property "
            "calculation.",
        )
        self._set_widget_tooltip(
            self.molar_mass_solvent_spin,
            "Molar mass of the solvent used in the solution-property "
            "calculation.",
        )
        self._set_widget_tooltip(
            self.mass_solute_spin,
            "Mass of solute present in the mixture.",
        )
        self._set_widget_tooltip(
            self.mass_solvent_spin,
            "Mass of solvent present in the mixture.",
        )
        self._set_widget_tooltip(
            self.mass_percent_solute_spin,
            "Solute mass percent when entering the mixture by composition.",
        )
        self._set_widget_tooltip(
            self.total_mass_solution_spin,
            "Total solution mass used with the solute mass percent.",
        )
        self._set_widget_tooltip(
            self.molarity_spin,
            "Target molarity of the chosen element in one liter of solution.",
        )
        self._set_widget_tooltip(
            self.molarity_element_edit,
            "Element symbol used to interpret the entered molarity.",
        )
        self._set_widget_tooltip(
            self.calculate_solution_button,
            "Calculate solution properties and save them into the rmcsetup "
            "metadata folder.",
        )
        self._set_widget_tooltip(
            self.representative_mode_combo,
            "Choose how representative cluster files are selected from the "
            "saved cluster bins.",
        )
        self._set_widget_tooltip(
            self.representative_preset_combo,
            "Load the same built-in and custom definition presets used by "
            "the bondanalysis application.",
        )
        self._set_widget_tooltip(
            self.load_representative_preset_button,
            "Load the selected bondanalysis preset into the representative "
            "selection tables.",
        )
        self._set_widget_tooltip(
            self.save_representative_preset_button,
            "Save the current representative bond-pair and angle-triplet "
            "tables as a bondanalysis preset for later reuse.",
        )
        self._set_widget_tooltip(
            self.representative_bond_pair_table,
            "Bond-pair definitions in the same column format used by "
            "bondanalysis.",
        )
        self._set_widget_tooltip(
            self.representative_angle_triplet_table,
            "Angle-triplet definitions in the same column format used by "
            "bondanalysis.",
        )
        self._set_widget_tooltip(
            self.representative_advanced_toggle,
            "Show or hide advanced controls for representative selection.",
        )
        self._set_widget_tooltip(
            self.representative_algorithm_combo,
            "Distance metric used to compare each structure against the "
            "target bond and angle distributions.",
        )
        self._set_widget_tooltip(
            self.representative_count_cutoff_spin,
            "Ignore cluster bins with fewer members than this cutoff.",
        )
        self._set_widget_tooltip(
            self.representative_bond_weight_spin,
            "Relative weight given to bond distributions during "
            "representative selection.",
        )
        self._set_widget_tooltip(
            self.representative_angle_weight_spin,
            "Relative weight given to angle distributions during "
            "representative selection.",
        )
        self._set_widget_tooltip(
            self.compute_representatives_button,
            "Open the dedicated Representative Structures tool for this "
            "project.",
        )
        self._set_widget_tooltip(
            self.representative_progress_bar,
            "Progress for the legacy in-panel representative-selection job.",
        )
        self._set_widget_tooltip(
            self.preview_representatives_button,
            "Reload the saved representative structures from this project.",
        )
        self._set_widget_tooltip(
            self.solvent_reference_source_combo,
            "Choose whether the solvent reference structure comes from a "
            "bundled preset or a custom PDB file.",
        )
        self._set_widget_tooltip(
            self.solvent_preset_combo,
            "Select a bundled solvent-reference preset.",
        )
        self._set_widget_tooltip(
            self.solvent_reference_edit,
            "Path to the custom solvent-reference PDB used for coordinated "
            "solvent-shell building.",
        )
        self._set_widget_tooltip(
            self.browse_solvent_reference_button,
            "Browse for a custom solvent-reference PDB file.",
        )
        self._set_widget_tooltip(
            self.solvent_reference_match_tolerance_spin,
            "Tolerance used while matching the selected solvent reference "
            "against each representative structure.",
        )
        self._set_widget_tooltip(
            self.solvent_director_atom_combo,
            "Reference atom that should point toward the solute cluster "
            "during solvent-shell building.",
        )
        self._set_widget_tooltip(
            self.solvent_minimum_separation_spin,
            "Minimum allowed distance between placed solvent atoms and the "
            "surrounding coordination sphere during solvent-shell "
            "placement and refinement.",
        )
        self._set_widget_tooltip(
            self.analyze_solvent_outputs_button,
            "Analyze every saved representative structure to determine the "
            "current coordinated-solvent state of the representative set.",
        )
        self._set_widget_tooltip(
            self.solvent_cutoff_table,
            "Choose which solute elements are coordination centers, set "
            "their target average coordination numbers, and define the "
            "director-atom cutoff distance used for solvent placement.",
        )
        self._set_widget_tooltip(
            self.build_solvent_outputs_button,
            "Build the stripped and solvent-decorated representative PDB "
            "outputs using the current automatic solvent analysis and "
            "coordination settings.",
        )
        self._set_widget_tooltip(
            self.generated_pdb_mode_combo,
            "Choose which saved representative structure set is active for "
            "the table, viewer, and Packmol setup: source, no solvent, "
            "partial solvent, or full solvent when available.",
        )
        self._set_widget_tooltip(
            self.generated_pdb_table,
            "Review the currently active representative structures and "
            "select one row to inspect it in the embedded viewer.",
        )
        self._set_widget_tooltip(
            self.open_generated_pdb_preview_button,
            "Open a 3D preview window for the currently selected saved "
            "no-solvent or full-solvent representative PDB file.",
        )
        self._set_widget_tooltip(
            self.packmol_planning_mode_combo,
            "Choose whether Packmol counts are planned per element or from "
            "a total number density.",
        )
        self._set_widget_tooltip(
            self.packmol_box_side_spin,
            "Side length of the cubic Packmol box used for count planning.",
        )
        self._set_widget_tooltip(
            self.packmol_free_solvent_combo,
            "Choose the solvent structure file used for the free bulk "
            "solvent population in the Packmol box.",
        )
        self._set_widget_tooltip(
            self.compute_packmol_plan_button,
            "Compute cluster counts and target weights for the current "
            "Packmol plan.",
        )
        self._set_widget_tooltip(
            self.build_packmol_setup_button,
            "Build Packmol input files and audit outputs from the saved "
            "plan using the active Full solvent representative structure set.",
        )
        self._set_widget_tooltip(
            self.packmol_tolerance_spin,
            "Tolerance written into the Packmol input file for the setup "
            "build.",
        )
        self._set_widget_tooltip(
            self.constraint_length_tolerance_spin,
            "Bond-length tolerance used while generating fullrmc "
            "constraints.",
        )
        self._set_widget_tooltip(
            self.constraint_angle_tolerance_spin,
            "Bond-angle tolerance used while generating fullrmc "
            "constraints.",
        )
        self._set_widget_tooltip(
            self.generate_constraints_button,
            "Generate per-structure and merged fullrmc constraints from the "
            "selected representative structures.",
        )
        self._set_widget_tooltip(
            self.open_constraints_folder_button,
            "Open the folder that contains the generated merged fullrmc "
            "constraints file and per-structure constraint files.",
        )
        self._set_widget_tooltip(
            self.preview_constraints_button,
            "Open a copy-friendly window that shows the merged fullrmc "
            "constraints file contents.",
        )

    def _on_posterior_filter_changed(self) -> None:
        self._update_posterior_filter_widgets()
        self._refresh_dream_source_summary()

    def _update_posterior_filter_widgets(self) -> None:
        mode = self.selected_posterior_filter_mode()
        self.posterior_top_percent_spin.setEnabled(mode == "top_percent_logp")
        self.posterior_top_n_spin.setEnabled(mode == "top_n_logp")

    def _set_favorite_controls_enabled(self, enabled: bool) -> None:
        self.set_favorite_button.setEnabled(enabled)
        self.favorite_history_combo.setEnabled(enabled)
        if not enabled:
            self.load_history_button.setEnabled(False)
            self.use_project_favorite_button.setEnabled(False)

    def _populate_solution_properties_controls(self) -> None:
        state = self._project_source_state
        self.solution_group.setEnabled(state is not None)
        if state is None:
            self._apply_solution_metadata(
                SolutionPropertiesMetadata(
                    settings=SolutionPropertiesSettings(),
                    result=None,
                    updated_at=None,
                )
            )
            self.solution_output_box.setPlainText(
                "Load a SAXS project to edit and calculate solution "
                "properties for the future Packmol box setup."
            )
            return
        self._apply_solution_metadata(state.solution_properties)

    def _populate_representative_controls(self) -> None:
        state = self._project_source_state
        self.representative_group.setEnabled(state is not None)
        if state is None:
            self._apply_representative_metadata(None)
            self.compute_representatives_button.setEnabled(False)
            self.preview_representatives_button.setEnabled(False)
            self.representative_status_label.setText(
                "Representative structures: no SAXS project loaded."
            )
            self.representative_summary_box.setPlainText(
                "Load a SAXS project and choose a DREAM source before "
                "loading saved representative structures."
            )
            return
        self._apply_representative_metadata(state.representative_selection)
        self.compute_representatives_button.setEnabled(True)
        self.preview_representatives_button.setEnabled(True)
        if state.representative_selection is None:
            self.representative_status_label.setText(
                "Representative structures: no saved project set loaded."
            )
        else:
            selection_mode = (
                state.representative_selection.selection_mode or "unknown"
            )
            self.representative_status_label.setText(
                "Representative structures: saved project set loaded "
                f"({selection_mode})."
            )
        self.representative_summary_box.setPlainText(
            self._representative_summary_text(
                state.representative_selection,
            )
        )

    def _populate_solvent_controls(self) -> None:
        state = self._project_source_state
        self.solvent_group.setEnabled(state is not None)
        if state is None:
            self._apply_solvent_metadata(None)
            self.analyze_solvent_outputs_button.setEnabled(False)
            self.build_solvent_outputs_button.setEnabled(False)
            self.solvent_summary_box.setPlainText(
                "Load a SAXS project and save representative structures before "
                "running the Solvent Shell Builder."
            )
            return
        self._apply_solvent_metadata(state.solvent_handling)
        self.solvent_summary_box.setPlainText(
            self._solvent_summary_text(state.solvent_handling)
        )
        self._update_solvent_build_panel_state()

    def _populate_packmol_planning_controls(self) -> None:
        state = self._project_source_state
        self.packmol_group.setEnabled(state is not None)
        if state is None:
            self._apply_packmol_planning_metadata(None)
            self.compute_packmol_plan_button.setEnabled(False)
            self.build_packmol_setup_button.setEnabled(False)
            self.packmol_free_solvent_combo.setEnabled(False)
            self.packmol_plan_summary_box.setPlainText(
                "Load a SAXS project, calculate solution properties, and "
                "save representative structures before planning Packmol "
                "cluster counts."
            )
            self.packmol_build_summary_box.setPlainText(
                "Build Packmol setup after computing cluster counts, "
                "choosing a free-solvent structure, and preparing the "
                "active full-solvent representative files."
            )
            self.open_packmol_setup_folder_button.setEnabled(False)
            self.packmol_docker_summary_box.setPlainText(
                self._packmol_docker_summary_text()
            )
            return
        self._apply_packmol_planning_metadata(state.packmol_planning)
        self.compute_packmol_plan_button.setEnabled(
            state.solution_properties.result is not None
            and state.representative_selection is not None
        )
        self.build_packmol_setup_button.setEnabled(
            state.packmol_planning is not None
            and self._active_representative_structure_set_is_ready()
        )
        self.packmol_free_solvent_combo.setEnabled(
            self.packmol_free_solvent_combo.count() > 0
        )
        self.packmol_plan_summary_box.setPlainText(
            self._packmol_plan_summary_text(state.packmol_planning)
        )
        self._apply_packmol_setup_metadata(state.packmol_setup)
        self.packmol_docker_summary_box.setPlainText(
            self._packmol_docker_summary_text(state.packmol_setup)
        )

    def _populate_constraint_controls(self) -> None:
        state = self._project_source_state
        self.constraints_group.setEnabled(state is not None)
        if state is None:
            self._apply_constraint_metadata(None)
            self.generate_constraints_button.setEnabled(False)
            self.open_constraints_folder_button.setEnabled(False)
            self.preview_constraints_button.setEnabled(False)
            self.constraints_summary_box.setPlainText(
                "Build Packmol setup inputs before generating per-structure "
                "constraint files and the merged fullrmc constraints file."
            )
            return
        self._apply_constraint_metadata(state.constraint_generation)
        self.generate_constraints_button.setEnabled(
            state.packmol_setup is not None
        )
        has_constraints = state.constraint_generation is not None
        self.open_constraints_folder_button.setEnabled(has_constraints)
        self.preview_constraints_button.setEnabled(has_constraints)

    def _selected_representative_mode(self) -> str:
        return str(
            self.representative_mode_combo.currentData() or "first_file"
        )

    def _apply_representative_metadata(
        self,
        metadata: RepresentativeSelectionMetadata | None,
    ) -> None:
        settings = (
            metadata.settings
            if metadata is not None
            else RepresentativeSelectionSettings()
        )
        self._set_combo_value(
            self.representative_mode_combo,
            settings.selection_mode,
        )
        self._set_representative_bond_pair_rows(tuple(settings.bond_pairs))
        self._set_representative_angle_triplet_rows(
            tuple(settings.angle_triplets)
        )
        self._reload_representative_presets(
            selected_name=self._matching_representative_preset_name(settings)
        )
        self._set_combo_value(
            self.representative_algorithm_combo,
            settings.selection_algorithm,
        )
        self.representative_count_cutoff_spin.setValue(
            settings.minimum_cluster_count_for_analysis
        )
        self.representative_bond_weight_spin.setValue(settings.bond_weight)
        self.representative_angle_weight_spin.setValue(settings.angle_weight)
        self._update_representative_mode_widgets()
        self._refresh_generated_pdb_mode_combo()
        state = self._project_source_state
        self._refresh_generated_pdb_browser(
            state.solvent_handling if state is not None else None
        )

    def _selected_representative_preset_name(self) -> str | None:
        payload = self.representative_preset_combo.currentData()
        if payload is None:
            return None
        return str(payload)

    def _reload_representative_presets(
        self,
        *,
        selected_name: str | None = None,
    ) -> None:
        previous_name = (
            selected_name or self._selected_representative_preset_name()
        )
        self._representative_presets = load_presets()
        self.representative_preset_combo.blockSignals(True)
        self.representative_preset_combo.clear()
        self.representative_preset_combo.addItem("Select preset...", None)
        selected_index = 0
        for index, name in enumerate(
            ordered_preset_names(self._representative_presets),
            start=1,
        ):
            preset = self._representative_presets[name]
            label = f"{name} (Built-in)" if preset.builtin else name
            self.representative_preset_combo.addItem(label, name)
            if name == previous_name:
                selected_index = index
        self.representative_preset_combo.setCurrentIndex(selected_index)
        self.representative_preset_combo.blockSignals(False)

    def _matching_representative_preset_name(
        self,
        settings: RepresentativeSelectionSettings,
    ) -> str | None:
        for name in ordered_preset_names(self._representative_presets):
            preset = self._representative_presets.get(name)
            if preset is None:
                continue
            if (
                tuple(settings.bond_pairs) == preset.bond_pairs
                and tuple(settings.angle_triplets) == preset.angle_triplets
            ):
                return name
        return None

    def _load_selected_representative_preset(self) -> None:
        preset_name = self._selected_representative_preset_name()
        if preset_name is None:
            QMessageBox.information(
                self,
                "Representative Presets",
                "Select a preset to load.",
            )
            return
        preset = self._representative_presets.get(preset_name)
        if preset is None:
            QMessageBox.warning(
                self,
                "Representative Presets",
                f"Unknown preset: {preset_name}",
            )
            return
        self._apply_representative_preset(preset)
        self._append_run_log(f"Loaded representative preset: {preset_name}")

    def _save_current_representative_as_preset(self) -> None:
        try:
            bond_pairs = self._read_representative_bond_pairs()
            angle_triplets = self._read_representative_angle_triplets()
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Representative Presets",
                str(exc),
            )
            return

        suggested_name = self._selected_representative_preset_name() or ""
        name, accepted = QInputDialog.getText(
            self,
            "Save Representative Preset",
            "Preset name:",
            text=suggested_name,
        )
        if not accepted:
            return
        name = name.strip()
        if not name:
            return

        if name in self._representative_presets:
            response = QMessageBox.question(
                self,
                "Overwrite Preset?",
                f"A preset named '{name}' already exists. Overwrite it?",
            )
            if response != QMessageBox.StandardButton.Yes:
                return

        save_custom_preset(
            BondAnalysisPreset(
                name=name,
                bond_pairs=tuple(bond_pairs),
                angle_triplets=tuple(angle_triplets),
            )
        )
        self._reload_representative_presets(selected_name=name)
        self._append_run_log(f"Saved representative preset: {name}")

    def _apply_representative_preset(
        self,
        preset: BondAnalysisPreset,
    ) -> None:
        self._set_representative_bond_pair_rows(preset.bond_pairs)
        self._set_representative_angle_triplet_rows(preset.angle_triplets)
        self._select_representative_preset_name(preset.name)

    def _select_representative_preset_name(self, preset_name: str) -> None:
        for index in range(self.representative_preset_combo.count()):
            if self.representative_preset_combo.itemData(index) == preset_name:
                self.representative_preset_combo.setCurrentIndex(index)
                return

    def _set_representative_bond_pair_rows(
        self,
        definitions: tuple[BondPairDefinition, ...],
    ) -> None:
        self.representative_bond_pair_table.blockSignals(True)
        self.representative_bond_pair_table.setRowCount(0)
        if not definitions:
            self._add_empty_representative_bond_pair_row(blocked=True)
        else:
            for definition in definitions:
                row = self.representative_bond_pair_table.rowCount()
                self.representative_bond_pair_table.insertRow(row)
                self.representative_bond_pair_table.setItem(
                    row,
                    0,
                    QTableWidgetItem(definition.atom1),
                )
                self.representative_bond_pair_table.setItem(
                    row,
                    1,
                    QTableWidgetItem(definition.atom2),
                )
                self.representative_bond_pair_table.setItem(
                    row,
                    2,
                    QTableWidgetItem(f"{definition.cutoff_angstrom:g}"),
                )
        self.representative_bond_pair_table.blockSignals(False)

    def _set_representative_angle_triplet_rows(
        self,
        definitions: tuple[AngleTripletDefinition, ...],
    ) -> None:
        self.representative_angle_triplet_table.blockSignals(True)
        self.representative_angle_triplet_table.setRowCount(0)
        if not definitions:
            self._add_empty_representative_angle_triplet_row(blocked=True)
        else:
            for definition in definitions:
                row = self.representative_angle_triplet_table.rowCount()
                self.representative_angle_triplet_table.insertRow(row)
                self.representative_angle_triplet_table.setItem(
                    row,
                    0,
                    QTableWidgetItem(definition.vertex),
                )
                self.representative_angle_triplet_table.setItem(
                    row,
                    1,
                    QTableWidgetItem(definition.arm1),
                )
                self.representative_angle_triplet_table.setItem(
                    row,
                    2,
                    QTableWidgetItem(definition.arm2),
                )
                self.representative_angle_triplet_table.setItem(
                    row,
                    3,
                    QTableWidgetItem(f"{definition.cutoff1_angstrom:g}"),
                )
                self.representative_angle_triplet_table.setItem(
                    row,
                    4,
                    QTableWidgetItem(f"{definition.cutoff2_angstrom:g}"),
                )
        self.representative_angle_triplet_table.blockSignals(False)

    def _add_empty_representative_bond_pair_row(
        self,
        *,
        blocked: bool = False,
    ) -> None:
        previous_blocked = self.representative_bond_pair_table.blockSignals(
            blocked
        )
        row = self.representative_bond_pair_table.rowCount()
        self.representative_bond_pair_table.insertRow(row)
        for column in range(self.representative_bond_pair_table.columnCount()):
            self.representative_bond_pair_table.setItem(
                row,
                column,
                QTableWidgetItem(""),
            )
        self.representative_bond_pair_table.blockSignals(previous_blocked)

    def _add_empty_representative_angle_triplet_row(
        self,
        *,
        blocked: bool = False,
    ) -> None:
        previous_blocked = (
            self.representative_angle_triplet_table.blockSignals(blocked)
        )
        row = self.representative_angle_triplet_table.rowCount()
        self.representative_angle_triplet_table.insertRow(row)
        for column in range(
            self.representative_angle_triplet_table.columnCount()
        ):
            self.representative_angle_triplet_table.setItem(
                row,
                column,
                QTableWidgetItem(""),
            )
        self.representative_angle_triplet_table.blockSignals(previous_blocked)

    def _add_representative_bond_pair_row(self) -> None:
        self._add_empty_representative_bond_pair_row(blocked=True)

    def _remove_selected_representative_bond_pair_rows(self) -> None:
        selected_rows = sorted(
            {
                index.row()
                for index in self.representative_bond_pair_table.selectedIndexes()
            },
            reverse=True,
        )
        for row in selected_rows:
            self.representative_bond_pair_table.removeRow(row)
        if self.representative_bond_pair_table.rowCount() == 0:
            self._add_empty_representative_bond_pair_row(blocked=True)

    def _add_representative_angle_triplet_row(self) -> None:
        self._add_empty_representative_angle_triplet_row(blocked=True)

    def _remove_selected_representative_angle_triplet_rows(self) -> None:
        selected_rows = sorted(
            {
                index.row()
                for index in self.representative_angle_triplet_table.selectedIndexes()
            },
            reverse=True,
        )
        for row in selected_rows:
            self.representative_angle_triplet_table.removeRow(row)
        if self.representative_angle_triplet_table.rowCount() == 0:
            self._add_empty_representative_angle_triplet_row(blocked=True)

    def _read_representative_bond_pairs(
        self,
    ) -> tuple[BondPairDefinition, ...]:
        definitions: list[BondPairDefinition] = []
        for row in range(self.representative_bond_pair_table.rowCount()):
            atom1 = self._table_text(
                self.representative_bond_pair_table, row, 0
            )
            atom2 = self._table_text(
                self.representative_bond_pair_table, row, 1
            )
            cutoff_text = self._table_text(
                self.representative_bond_pair_table, row, 2
            )
            if not atom1 and not atom2 and not cutoff_text:
                continue
            if not atom1 or not atom2 or not cutoff_text:
                raise ValueError(
                    "Every populated bond-pair row needs atom 1, atom 2, "
                    "and a cutoff."
                )
            definitions.append(
                BondPairDefinition(
                    atom1,
                    atom2,
                    float(cutoff_text),
                )
            )
        return tuple(definitions)

    def _read_representative_angle_triplets(
        self,
    ) -> tuple[AngleTripletDefinition, ...]:
        definitions: list[AngleTripletDefinition] = []
        for row in range(self.representative_angle_triplet_table.rowCount()):
            vertex = self._table_text(
                self.representative_angle_triplet_table, row, 0
            )
            arm1 = self._table_text(
                self.representative_angle_triplet_table, row, 1
            )
            arm2 = self._table_text(
                self.representative_angle_triplet_table, row, 2
            )
            cutoff1_text = self._table_text(
                self.representative_angle_triplet_table, row, 3
            )
            cutoff2_text = self._table_text(
                self.representative_angle_triplet_table, row, 4
            )
            if (
                not vertex
                and not arm1
                and not arm2
                and not cutoff1_text
                and not cutoff2_text
            ):
                continue
            if not all((vertex, arm1, arm2, cutoff1_text, cutoff2_text)):
                raise ValueError(
                    "Every populated angle-triplet row needs the vertex, "
                    "both arms, and both cutoffs."
                )
            definitions.append(
                AngleTripletDefinition(
                    vertex,
                    arm1,
                    arm2,
                    float(cutoff1_text),
                    float(cutoff2_text),
                )
            )
        return tuple(definitions)

    @staticmethod
    def _table_text(table: QTableWidget, row: int, column: int) -> str:
        item = table.item(row, column)
        return item.text().strip() if item is not None else ""

    def _available_generated_pdb_mode_items(
        self,
    ) -> list[tuple[str, str]]:
        state = self._project_source_state
        if state is None or state.representative_selection is None:
            return []
        return [
            (representative_structure_mode_label(mode), mode)
            for mode in available_representative_structure_modes(
                state.representative_selection,
                state.solvent_handling,
            )
        ]

    def _active_generated_pdb_mode(self) -> str:
        state = self._project_source_state
        if state is None or state.representative_selection is None:
            return "source"
        preferred = self.generated_pdb_mode_combo.currentData()
        preferred_mode = (
            str(preferred).strip() if preferred is not None else None
        )
        return resolved_representative_structure_mode(
            state.representative_selection,
            state.solvent_handling,
            preferred_mode=preferred_mode,
        )

    def _active_representative_structure_set_is_ready(self) -> bool:
        state = self._project_source_state
        return bool(
            state is not None
            and state.representative_selection is not None
            and self._active_generated_pdb_mode() == "full_solvent"
        )

    def _solvent_shell_builder_required(self) -> bool:
        state = self._project_source_state
        return bool(
            state is not None
            and state.representative_selection is not None
            and self._active_generated_pdb_mode() != "full_solvent"
        )

    def _set_solvent_shell_builder_controls_enabled(
        self,
        enabled: bool,
    ) -> None:
        for widget in (
            self.solvent_reference_source_combo,
            self.solvent_preset_combo,
            self.solvent_reference_edit,
            self.browse_solvent_reference_button,
            self.solvent_reference_match_tolerance_spin,
            self.solvent_director_atom_combo,
            self.solvent_minimum_separation_spin,
            self.solvent_reference_details_box,
            self.solvent_status_group,
            self.solvent_cutoff_group,
        ):
            widget.setEnabled(enabled)

    def _solvent_build_has_required_coordination_settings(
        self,
        analysis: RepresentativeSolventDistributionAnalysis,
    ) -> bool:
        if analysis.distribution_status in {
            "complete_solvent",
            "partial_solvent",
            "mixed_complete_and_partial",
        }:
            return True
        selected_settings = self._selected_solvent_coordination_settings()
        return any(
            setting.coordination_center
            and float(setting.target_coordination_number) > 0.0
            and float(setting.director_distance_cutoff_a) > 0.0
            for setting in selected_settings.values()
        )

    def _representative_entry_selected_state_text(
        self,
        representative_entry: object,
        active_mode: str,
    ) -> str:
        source_variant = representative_source_solvent_mode_to_variant(
            getattr(representative_entry, "source_solvent_mode", None)
        )
        if active_mode == "full_solvent":
            return "Full solvent analyzed"
        if source_variant == active_mode:
            return f"{representative_structure_mode_label(active_mode)} source"
        if active_mode in {
            "no_solvent",
            "partial_solvent",
            "full_solvent",
        }:
            return (
                f"{representative_structure_mode_label(active_mode)} selected"
            )
        return "Not analyzed"

    def _refresh_generated_pdb_mode_combo(self) -> None:
        state = self._project_source_state
        items = self._available_generated_pdb_mode_items()
        preferred_mode: str | None = None
        if state is not None and state.solvent_handling is not None:
            preferred_mode = str(
                state.solvent_handling.settings.coordinated_solvent_mode
            ).strip()
        current = self.generated_pdb_mode_combo.currentData()
        if current is not None and str(current).strip():
            preferred_mode = str(current).strip()
        self._updating_generated_pdb_mode_combo = True
        try:
            self.generated_pdb_mode_combo.clear()
            for label, mode in items:
                self.generated_pdb_mode_combo.addItem(label, mode)
            self.generated_pdb_mode_combo.setEnabled(len(items) > 1)
            if not items:
                return
            resolved_mode = resolved_representative_structure_mode(
                state.representative_selection if state is not None else None,
                state.solvent_handling if state is not None else None,
                preferred_mode=preferred_mode,
            )
            self._set_combo_value(self.generated_pdb_mode_combo, resolved_mode)
        finally:
            self._updating_generated_pdb_mode_combo = False

    def _handle_generated_pdb_mode_changed(self) -> None:
        if self._updating_generated_pdb_mode_combo:
            return
        state = self._project_source_state
        if state is None or state.representative_selection is None:
            return
        resolved_mode = self._active_generated_pdb_mode()
        if state.solvent_handling is not None:
            state.solvent_handling.settings.coordinated_solvent_mode = (
                resolved_mode
            )
            save_solvent_handling_metadata(
                state.rmcsetup_paths.solvent_handling_path,
                state.solvent_handling,
            )
            self.solvent_summary_box.setPlainText(
                self._solvent_summary_text(state.solvent_handling)
            )
        self._refresh_generated_pdb_browser(
            state.solvent_handling if state is not None else None
        )
        self.solvent_summary_box.setPlainText(
            self._solvent_summary_text(state.solvent_handling)
        )
        self._update_solvent_status_panel(state.solvent_handling)
        self._update_solvent_build_panel_state()
        self._populate_packmol_planning_controls()
        self._update_readiness_progress()

    def _apply_solvent_metadata(
        self,
        metadata: SolventHandlingMetadata | None,
    ) -> None:
        settings = (
            metadata.settings
            if metadata is not None
            else SolventHandlingSettings()
        )
        self._set_combo_value(
            self.solvent_reference_source_combo,
            settings.reference_source,
        )
        self._set_combo_value(self.solvent_preset_combo, settings.preset_name)
        self.solvent_reference_edit.setText(
            settings.custom_reference_path or ""
        )
        self.solvent_reference_match_tolerance_spin.setValue(
            settings.reference_match_tolerance_a
        )
        self.solvent_minimum_separation_spin.setValue(
            settings.minimum_solvent_atom_separation_a
        )
        self._solvent_distribution_analysis = None
        self._update_solvent_reference_widgets()
        self._populate_solvent_director_atom_choices(
            selected_name=settings.director_atom_name
        )
        self._update_solvent_reference_details()
        counts = (
            metadata.aggregate_solute_element_counts
            if metadata is not None
            else {}
        )
        self._populate_solvent_cutoff_table(
            counts,
            settings.solute_atom_settings,
        )
        self._refresh_generated_pdb_mode_combo()
        self._refresh_generated_pdb_browser(metadata)
        self._update_solvent_status_panel(metadata)
        self._update_solvent_build_panel_state()

    def _refresh_generated_pdb_browser(
        self,
        metadata: SolventHandlingMetadata | None,
    ) -> None:
        state = self._project_source_state
        self._generated_pdb_inspections = []
        self._solvent_table_preview_paths = []
        self._solvent_table_details = []
        self.generated_pdb_table.setRowCount(0)
        self.generated_pdb_viewer.draw_placeholder()
        self.generated_pdb_group.setEnabled(
            state is not None and state.representative_selection is not None
        )
        if state is None or state.representative_selection is None:
            self.generated_pdb_details_box.setPlainText(
                "Open Representative Structures, save the project set, and "
                "reload it here to browse the active representative "
                "structures."
            )
            self.generated_pdb_viewer_status_label.setText(
                "Select a representative row to preview the active structure."
            )
            self.open_generated_pdb_preview_button.setEnabled(False)
            return

        active_mode = self._active_generated_pdb_mode()
        active_mode_label = representative_structure_mode_label(active_mode)
        inspection_lookup: dict[
            tuple[str, str, str, str], GeneratedPDBInspection
        ] = {}
        inspection_error: str | None = None
        if metadata is not None:
            try:
                inspections = build_generated_pdb_inspections(metadata)
            except Exception as exc:
                inspection_error = str(exc)
            else:
                inspection_lookup = {
                    (
                        inspection.structure,
                        inspection.motif,
                        inspection.param,
                        inspection.file_role,
                    ): inspection
                    for inspection in inspections
                }

        solvent_lookup = solvent_entry_lookup_for_representatives(
            state.representative_selection,
            metadata,
        )
        self.generated_pdb_table.setRowCount(
            len(state.representative_selection.representative_entries)
        )
        for row, representative_entry in enumerate(
            state.representative_selection.representative_entries
        ):
            key = (
                representative_entry.structure,
                representative_entry.motif,
                representative_entry.param,
            )
            solvent_entry = solvent_lookup.get(key)
            file_role: str | None = None
            if active_mode == "full_solvent":
                file_role = "completed"
            elif active_mode == "no_solvent":
                file_role = "no_solvent"
            inspection = (
                inspection_lookup.get((*key, file_role))
                if file_role is not None
                else None
            )
            preview_path = representative_structure_path_for_mode(
                representative_entry,
                solvent_entry,
                active_mode,
            )
            source_text = representative_entry.analysis_source or "n/a"
            atom_count = representative_entry.atom_count
            if solvent_entry is not None:
                if active_mode == "full_solvent":
                    atom_count = solvent_entry.atom_count_completed
                    source_text = (
                        solvent_entry.completion_strategy
                        or "saved Solvent Shell Builder output"
                    )
                elif active_mode == "no_solvent":
                    atom_count = solvent_entry.atom_count_no_solvent
                    source_text = "saved no-solvent export"
                elif active_mode == "partial_solvent":
                    source_text = "representative selection source"
            if inspection is not None and inspection.exists:
                atom_count = inspection.atom_count

            detected_state = (
                solvent_entry.detected_source_status_text
                if solvent_entry is not None
                else self._representative_entry_selected_state_text(
                    representative_entry,
                    active_mode,
                )
            )
            values = [
                (
                    representative_entry.structure
                    if representative_entry.motif == "no_motif"
                    else (
                        f"{representative_entry.structure}/"
                        f"{representative_entry.motif}"
                    )
                ),
                detected_state,
                active_mode_label,
                preview_path.name,
                str(atom_count),
                source_text,
            ]
            details_lines = [
                f"Representative: {values[0]}",
                f"Detected source solvent state: {detected_state}",
                f"Active structure set: {active_mode_label}",
                f"Structure file: {preview_path}",
                (
                    "Selected weight: "
                    f"{representative_entry.selected_weight:.6g}"
                ),
                f"Cluster count: {representative_entry.cluster_count}",
                (
                    "Representative atom count: "
                    f"{representative_entry.atom_count}"
                ),
                (
                    "Representative selection source: "
                    f"{representative_entry.analysis_source}"
                ),
            ]
            if representative_entry.element_counts:
                details_lines.append(
                    "Representative elements: "
                    + ", ".join(
                        f"{element}:{count}"
                        for element, count in sorted(
                            representative_entry.element_counts.items()
                        )
                    )
                )
            if solvent_entry is not None:
                details_lines.extend(
                    [
                        f"No-solvent PDB: {solvent_entry.no_solvent_pdb}",
                        f"Full-solvent PDB: {solvent_entry.completed_pdb}",
                        (
                            "Saved solvent-handling strategy: "
                            f"{solvent_entry.completion_strategy or 'n/a'}"
                        ),
                    ]
                )
            detail_sections = ["\n".join(details_lines)]
            if inspection is not None:
                detail_sections.append(inspection.details_text())
            if solvent_entry is not None and solvent_entry.analysis_summary:
                detail_sections.append(solvent_entry.analysis_summary)
            if solvent_entry is not None and solvent_entry.build_summary:
                detail_sections.append(solvent_entry.build_summary)
            if inspection_error:
                detail_sections.append(
                    "Generated structure inspection warning:\n"
                    + inspection_error
                )
            self._generated_pdb_inspections.append(inspection)
            self._solvent_table_preview_paths.append(
                preview_path if preview_path.is_file() else None
            )
            self._solvent_table_details.append(
                "\n\n".join(section for section in detail_sections if section)
            )
            for column, value in enumerate(values):
                self.generated_pdb_table.setItem(
                    row,
                    column,
                    QTableWidgetItem(value),
                )

        if state.representative_selection.representative_entries:
            self.generated_pdb_table.selectRow(0)
            return

        self.generated_pdb_details_box.setPlainText(
            "No representative structures are available for browsing."
        )
        self.generated_pdb_viewer_status_label.setText(
            "No representative structures are available for preview."
        )
        self.open_generated_pdb_preview_button.setEnabled(False)

    def _selected_generated_pdb_inspection(
        self,
    ) -> GeneratedPDBInspection | None:
        row = self.generated_pdb_table.currentRow()
        if row < 0 or row >= len(self._generated_pdb_inspections):
            return None
        return self._generated_pdb_inspections[row]

    def _on_generated_pdb_selection_changed(self) -> None:
        row = self.generated_pdb_table.currentRow()
        if row < 0 or row >= len(self._solvent_table_details):
            self.generated_pdb_details_box.setPlainText(
                "Select a representative row to review the active structure details."
            )
            self.generated_pdb_viewer.draw_placeholder()
            self.generated_pdb_viewer_status_label.setText(
                "Select a representative row to preview the active structure."
            )
            self.open_generated_pdb_preview_button.setEnabled(False)
            return
        self.generated_pdb_details_box.setPlainText(
            self._solvent_table_details[row]
        )
        preview_path = self._solvent_table_preview_paths[row]
        self._refresh_generated_pdb_viewer(preview_path)
        inspection = self._selected_generated_pdb_inspection()
        self.open_generated_pdb_preview_button.setEnabled(
            inspection is not None
            and inspection.exists
            and inspection.load_error is None
        )

    def _open_selected_generated_pdb_preview(self) -> None:
        inspection = self._selected_generated_pdb_inspection()
        if inspection is None:
            QMessageBox.information(
                self,
                "No previewable PDB selected",
                "Switch to a saved no-solvent or full-solvent representative "
                "structure set before opening the separate PDB preview window.",
            )
            return
        if not inspection.exists or inspection.load_error is not None:
            QMessageBox.warning(
                self,
                "Generated PDB preview unavailable",
                inspection.load_error
                or "The selected generated PDB file could not be opened.",
            )
            return
        self._generated_pdb_preview_window = GeneratedPDBPreviewWindow(
            inspection,
            parent=self,
        )
        self._generated_pdb_preview_window.show()
        self._generated_pdb_preview_window.raise_()
        self._generated_pdb_preview_window.activateWindow()
        self._append_run_log(
            f"Opened generated PDB preview: {inspection.file_name}"
        )

    def _refresh_generated_pdb_viewer(
        self,
        preview_path: Path | None,
    ) -> None:
        if preview_path is None or not preview_path.is_file():
            self.generated_pdb_viewer.draw_placeholder()
            self.generated_pdb_viewer_status_label.setText(
                "No previewable representative structure is available for the selected row."
            )
            return
        try:
            structure = load_electron_density_structure(
                preview_path,
                center_mode="center_of_mass",
                include_bonds=True,
                include_comment=True,
            )
        except Exception as exc:
            self.generated_pdb_viewer.draw_placeholder()
            self.generated_pdb_viewer_status_label.setText(
                f"Unable to preview {preview_path.name}: {exc}"
            )
            return
        self.generated_pdb_viewer.set_structure(
            structure,
            scene_key=f"rmcsetup-solvent:{preview_path}",
        )
        self.generated_pdb_viewer_status_label.setText(
            f"Previewing {preview_path.name} with {structure.atom_count} atom(s)."
        )

    def _update_solvent_status_panel(
        self,
        metadata: SolventHandlingMetadata | None,
    ) -> None:
        analysis = self._solvent_distribution_analysis
        state = self._project_source_state
        status_lookup = {
            "complete_solvent": "Complete solvent molecules detected",
            "partial_solvent": "Partial solvent molecules detected",
            "no_solvent": "No solvent molecules detected",
            "mixed_complete_and_partial": (
                "Complete and partial solvent molecules detected"
            ),
            "unknown": "Unknown solvent state",
        }
        if (
            state is not None
            and self._active_representative_structure_set_is_ready()
            and (metadata is None or not metadata.entries)
        ):
            active_mode = self._active_generated_pdb_mode()
            self.solvent_status_headline_label.setText(
                "Full-solvent representative structures are selected."
            )
            self.solvent_status_stats_label.setText(
                "\n".join(
                    [
                        "The active representative source files already "
                        "provide the Full solvent structure set.",
                        (
                            "Active representative structure set: "
                            + representative_structure_mode_label(active_mode)
                        ),
                        "Solvent Shell Builder readiness: Ready for Packmol",
                        (
                            "Solvent state analysis and solvent-shell "
                            "building are not required for this selection."
                        ),
                    ]
                )
            )
            return
        if metadata is not None:
            active_mode = self._active_generated_pdb_mode()
            self.solvent_status_headline_label.setText(
                status_lookup.get(
                    metadata.detected_distribution_status,
                    metadata.detected_distribution_status.replace("_", " "),
                )
            )
            status_lines = [
                metadata.detected_distribution_note
                or "Representative solvent outputs have been generated.",
                (
                    "Recognized solute elements: "
                    + ", ".join(
                        f"{element}:{count}"
                        for element, count in sorted(
                            metadata.aggregate_solute_element_counts.items()
                        )
                    )
                    if metadata.aggregate_solute_element_counts
                    else "Recognized solute elements: none"
                ),
                (
                    "Active representative structure set: "
                    + representative_structure_mode_label(active_mode)
                ),
                (
                    "Solvent Shell Builder readiness: Ready for Packmol"
                    if active_mode == "full_solvent"
                    else "Solvent Shell Builder readiness: Select Full solvent "
                    "to mark this step ready for Packmol"
                ),
            ]
            self.solvent_status_stats_label.setText("\n".join(status_lines))
            return
        if analysis is None:
            self.solvent_status_headline_label.setText(
                "No representative solvent state has been determined yet."
            )
            if self._solvent_shell_builder_required():
                self.solvent_status_stats_label.setText(
                    "Choose a solvent reference, set coordination options as "
                    "needed, and press Build Solvent-Decorated Representative "
                    "PDBs. The required solvent-state analysis will run first."
                )
            else:
                self.solvent_status_stats_label.setText(
                    "Save representative structures before running the "
                    "Solvent Shell Builder."
                )
            return
        self.solvent_status_headline_label.setText(
            status_lookup.get(
                analysis.distribution_status,
                analysis.distribution_status.replace("_", " "),
            )
        )
        status_lines = [
            f"Representative entries analyzed: {len(analysis.entries)}",
            (
                "Saved full-solvent representatives are not available yet. "
                "Build solvent-decorated representative PDBs to store the "
                "Full solvent representative structure set."
            ),
        ]
        if analysis.distribution_note:
            status_lines.append(analysis.distribution_note)
        status_lines.append(
            (
                "Recognized solute elements: "
                + ", ".join(
                    f"{element}:{count}"
                    for element, count in sorted(
                        analysis.aggregate_solute_element_counts.items()
                    )
                )
            )
            if analysis.aggregate_solute_element_counts
            else "Recognized solute elements: none"
        )
        self.solvent_status_stats_label.setText("\n".join(status_lines))

    def _populate_solvent_cutoff_table(
        self,
        element_counts: dict[str, int],
        element_settings: dict[str, SoluteAtomBuildSetting] | None = None,
    ) -> None:
        self._updating_solvent_table = True
        try:
            self._solvent_cutoff_spins = {}
            self._solvent_coordination_center_items = {}
            self._solvent_coordination_target_spins = {}
            self.solvent_cutoff_table.setRowCount(0)
            settings_lookup = element_settings or {}
            if not element_counts:
                self.solvent_cutoff_status_label.setText(
                    "Analyze or build the representative structures to "
                    "populate the solute atom types used for solvent-shell "
                    "building."
                )
                return
            self.solvent_cutoff_status_label.setText(
                "Choose which solute elements should coordinate the solvent, "
                "set their average coordination targets, and review the "
                "director-atom cutoffs that will be used across the "
                "representative set."
            )
            sorted_counts = sorted(element_counts.items())
            self.solvent_cutoff_table.setRowCount(len(sorted_counts))
            for row, (element, count) in enumerate(sorted_counts):
                setting = settings_lookup.get(
                    str(element), SoluteAtomBuildSetting()
                )
                self.solvent_cutoff_table.setItem(
                    row, 0, QTableWidgetItem(str(element))
                )
                self.solvent_cutoff_table.setItem(
                    row, 1, QTableWidgetItem(str(count))
                )
                center_item = QTableWidgetItem("")
                center_item.setFlags(
                    Qt.ItemFlag.ItemIsEnabled
                    | Qt.ItemFlag.ItemIsSelectable
                    | Qt.ItemFlag.ItemIsUserCheckable
                )
                center_item.setCheckState(
                    Qt.CheckState.Checked
                    if setting.coordination_center
                    else Qt.CheckState.Unchecked
                )
                self.solvent_cutoff_table.setItem(row, 2, center_item)
                self._solvent_coordination_center_items[str(element)] = (
                    center_item
                )

                coordination_spin = QDoubleSpinBox(self.solvent_cutoff_table)
                coordination_spin.setDecimals(2)
                coordination_spin.setRange(0.0, 12.0)
                coordination_spin.setSingleStep(0.25)
                coordination_spin.setValue(
                    float(setting.target_coordination_number)
                )
                coordination_spin.valueChanged.connect(
                    self._update_solvent_build_panel_state
                )
                self.solvent_cutoff_table.setCellWidget(
                    row, 3, coordination_spin
                )
                self._solvent_coordination_target_spins[str(element)] = (
                    coordination_spin
                )

                cutoff_spin = QDoubleSpinBox(self.solvent_cutoff_table)
                cutoff_spin.setDecimals(3)
                cutoff_spin.setRange(0.0, 20.0)
                cutoff_spin.setSingleStep(0.1)
                cutoff_spin.setSuffix(" A")
                cutoff_spin.setValue(float(setting.director_distance_cutoff_a))
                cutoff_spin.valueChanged.connect(
                    self._update_solvent_build_panel_state
                )
                self.solvent_cutoff_table.setCellWidget(row, 4, cutoff_spin)
                self._solvent_cutoff_spins[str(element)] = cutoff_spin
        finally:
            self._updating_solvent_table = False

    def _selected_solvent_coordination_settings(
        self,
    ) -> dict[str, SoluteAtomBuildSetting]:
        settings: dict[str, SoluteAtomBuildSetting] = {}
        for element, cutoff_spin in self._solvent_cutoff_spins.items():
            center_item = self._solvent_coordination_center_items.get(element)
            coordination_spin = self._solvent_coordination_target_spins.get(
                element
            )
            settings[str(element)] = SoluteAtomBuildSetting(
                coordination_center=(
                    center_item is not None
                    and center_item.checkState() == Qt.CheckState.Checked
                ),
                target_coordination_number=(
                    float(coordination_spin.value())
                    if coordination_spin is not None
                    else 0.0
                ),
                director_distance_cutoff_a=float(cutoff_spin.value()),
            )
        return settings

    def _apply_packmol_planning_metadata(
        self,
        metadata: PackmolPlanningMetadata | None,
    ) -> None:
        state = self._project_source_state
        settings = (
            metadata.settings
            if metadata is not None
            else PackmolPlanningSettings()
        )
        self._set_combo_value(
            self.packmol_planning_mode_combo,
            settings.planning_mode,
        )
        self.packmol_box_side_spin.setValue(settings.box_side_length_a)
        selected_reference = settings.free_solvent_reference
        if (
            selected_reference is None
            and metadata is not None
            and metadata.solvent_allocation is not None
        ):
            selected_reference = metadata.solvent_allocation.reference_path
        if (
            selected_reference is None
            and state is not None
            and state.packmol_setup is not None
        ):
            selected_reference = (
                state.packmol_setup.free_solvent_reference_path
            )
        if (
            selected_reference is None
            and state is not None
            and state.solvent_handling is not None
        ):
            selected_reference = state.solvent_handling.reference_path
        self._populate_packmol_free_solvent_choices(
            selected_identifier=selected_reference
        )
        self._refresh_packmol_plan_plot(metadata)

    def _apply_packmol_setup_metadata(
        self,
        metadata: PackmolSetupMetadata | None,
    ) -> None:
        settings = (
            metadata.settings
            if metadata is not None
            else PackmolSetupSettings()
        )
        self.packmol_tolerance_spin.setValue(settings.tolerance_angstrom)
        self.packmol_build_summary_box.setPlainText(
            self._packmol_setup_summary_text(metadata)
        )
        self.open_packmol_setup_folder_button.setEnabled(metadata is not None)

    def _apply_constraint_metadata(
        self,
        metadata: ConstraintGenerationMetadata | None,
    ) -> None:
        settings = (
            metadata.settings
            if metadata is not None
            else ConstraintGenerationSettings()
        )
        self.constraint_length_tolerance_spin.setValue(
            settings.bond_length_tolerance_angstrom
        )
        self.constraint_angle_tolerance_spin.setValue(
            settings.bond_angle_tolerance_degrees
        )
        self.constraints_summary_box.setPlainText(
            self._constraint_summary_text(metadata)
        )
        self.open_constraints_folder_button.setEnabled(metadata is not None)
        self.preview_constraints_button.setEnabled(metadata is not None)

    def _on_representative_mode_changed(self) -> None:
        self._update_representative_mode_widgets()

    def _toggle_representative_advanced_settings(
        self,
        checked: bool,
    ) -> None:
        self.representative_advanced_widget.setVisible(checked)
        self.representative_advanced_toggle.setText(
            "Hide Advanced Settings" if checked else "Show Advanced Settings"
        )

    def _update_representative_mode_widgets(self) -> None:
        is_distribution = (
            self._selected_representative_mode() == "bond_angle_distribution"
        )
        self.representative_preset_group.setVisible(is_distribution)
        self.representative_bond_pairs_row.setVisible(is_distribution)
        self.representative_angle_triplets_row.setVisible(is_distribution)

    def _update_solvent_reference_widgets(self) -> None:
        use_custom = (
            str(self.solvent_reference_source_combo.currentData() or "preset")
            == "custom"
        )
        self.solvent_preset_combo.setEnabled(not use_custom)
        self.solvent_reference_edit.setEnabled(use_custom)
        self.browse_solvent_reference_button.setEnabled(use_custom)
        self._populate_solvent_director_atom_choices()
        self._update_solvent_reference_details()

    def _handle_solvent_reference_source_changed(self) -> None:
        self._update_solvent_reference_widgets()
        self._clear_solvent_analysis_outputs()

    def _handle_solvent_reference_changed(self) -> None:
        self._populate_solvent_director_atom_choices()
        self._update_solvent_reference_details()
        self._clear_solvent_analysis_outputs()

    def _handle_solvent_analysis_setting_changed(self) -> None:
        self._clear_solvent_analysis_outputs()

    def _handle_solvent_coordination_table_item_changed(
        self,
        _item: QTableWidgetItem,
    ) -> None:
        if self._updating_solvent_table:
            return
        self._update_solvent_build_panel_state()

    def _selected_solvent_reference_identifier(self) -> str | None:
        source = str(
            self.solvent_reference_source_combo.currentData() or "preset"
        )
        if source == "custom":
            reference_path = self.solvent_reference_edit.text().strip()
            return reference_path or None
        selected_name = self.solvent_preset_combo.currentData()
        if selected_name is None:
            return None
        return str(selected_name)

    def _populate_solvent_director_atom_choices(
        self,
        *,
        selected_name: str | None = None,
    ) -> None:
        self.solvent_director_atom_combo.blockSignals(True)
        self.solvent_director_atom_combo.clear()
        reference_identifier = self._selected_solvent_reference_identifier()
        if reference_identifier is None:
            self.solvent_director_atom_combo.blockSignals(False)
            return
        try:
            atom_names = reference_atom_choices(reference_identifier)
            suggested_name = selected_name or default_director_atom_name(
                reference_identifier
            )
        except Exception:
            self.solvent_director_atom_combo.blockSignals(False)
            return
        for atom_name in atom_names:
            self.solvent_director_atom_combo.addItem(atom_name, atom_name)
        if suggested_name is not None:
            suggested_index = self.solvent_director_atom_combo.findData(
                suggested_name
            )
            if suggested_index >= 0:
                self.solvent_director_atom_combo.setCurrentIndex(
                    suggested_index
                )
        elif atom_names:
            self.solvent_director_atom_combo.setCurrentIndex(0)
        self.solvent_director_atom_combo.blockSignals(False)

    def _update_solvent_reference_details(self) -> None:
        reference_identifier = self._selected_solvent_reference_identifier()
        if reference_identifier is None:
            self.solvent_reference_details_box.setPlainText(
                "Choose a solvent reference before analyzing representative structures."
            )
            return
        try:
            atom_names = reference_atom_choices(reference_identifier)
            suggested_director = default_director_atom_name(
                reference_identifier
            )
            reference_path = Path(reference_identifier).expanduser()
            reference_name = reference_path.stem
            source_text = (
                str(reference_path.resolve())
                if reference_path.is_file()
                else reference_name
            )
        except Exception as exc:
            self.solvent_reference_details_box.setPlainText(
                f"Unable to inspect the selected solvent reference: {exc}"
            )
            return
        director_text = suggested_director or "n/a"
        self.solvent_reference_details_box.setPlainText(
            f"Reference molecule: {reference_name}\n"
            f"Atom count: {len(atom_names)}\n"
            f"Suggested director atom: {director_text}\n"
            f"Reference source: {source_text}"
        )

    def _clear_solvent_analysis_outputs(self) -> None:
        self._solvent_distribution_analysis = None
        state = self._project_source_state
        self.solvent_summary_box.setPlainText(
            self._solvent_summary_text(
                state.solvent_handling if state is not None else None
            )
        )
        self._populate_solvent_cutoff_table(
            {},
            self._selected_solvent_coordination_settings(),
        )
        self._refresh_generated_pdb_browser(
            state.solvent_handling if state is not None else None
        )
        self._update_solvent_status_panel(
            state.solvent_handling if state is not None else None
        )
        self._update_solvent_build_panel_state()

    def _current_representative_settings(
        self,
    ) -> RepresentativeSelectionSettings:
        mode = self._selected_representative_mode()
        if mode == "bond_angle_distribution":
            bond_pairs = self._read_representative_bond_pairs()
            angle_triplets = self._read_representative_angle_triplets()
        else:
            bond_pairs = ()
            angle_triplets = ()
        return RepresentativeSelectionSettings(
            selection_mode=mode,
            selection_algorithm=str(
                self.representative_algorithm_combo.currentData()
                or "target_distribution_quantile_distance"
            ),
            minimum_cluster_count_for_analysis=int(
                self.representative_count_cutoff_spin.value()
            ),
            bond_weight=float(self.representative_bond_weight_spin.value()),
            angle_weight=float(self.representative_angle_weight_spin.value()),
            bond_pairs=bond_pairs,
            angle_triplets=angle_triplets,
        )

    def _compute_representative_clusters(self) -> None:
        if self._representative_thread is not None:
            QMessageBox.information(
                self,
                "Representative selection running",
                "Representative-cluster selection is already running in the background.",
            )
            return
        state = self._project_source_state
        if state is None:
            QMessageBox.information(
                self,
                "No SAXS project loaded",
                "Load a SAXS project before computing representatives.",
            )
            return
        self._warn_if_cluster_validation_needs_attention(
            "Representative selection"
        )
        selection = self.current_selection()
        if selection is None:
            QMessageBox.information(
                self,
                "No DREAM source selected",
                "Choose a DREAM run and best-fit selection first.",
            )
            return
        try:
            representative_settings = self._current_representative_settings()
        except Exception as exc:
            message = (
                "Unable to compute representative clusters in "
                f"{self._selected_representative_mode()} mode: {exc}"
            )
            self.representative_summary_box.setPlainText(message)
            self._append_run_log(message)
            self._set_task_progress("Representative selection failed.", 0)
            QMessageBox.warning(
                self,
                "Representative selection failed",
                str(exc),
            )
            return

        self.compute_representatives_button.setEnabled(False)
        self.representative_status_label.setText(
            "Representative selection: starting background task..."
        )
        self.representative_progress_bar.setRange(0, 1)
        self.representative_progress_bar.setValue(0)
        self.representative_summary_box.setPlainText(
            "Representative selection is running in the background.\n\n"
            "You can keep using the rest of the window while the active "
            "cluster bins are analyzed. Progress updates will appear above "
            "and short status messages will be appended to the run log."
        )
        self._set_task_progress(
            "Representative selection running in background...",
            0,
        )
        self._append_run_log(
            "Starting representative-cluster computation in background."
        )
        self._representative_job_state = state
        self._representative_thread = QThread(self)
        self._representative_worker = RepresentativeSelectionWorker(
            state,
            selection,
            representative_settings,
        )
        self._representative_worker.moveToThread(self._representative_thread)
        self._representative_thread.started.connect(
            self._representative_worker.run
        )
        self._representative_worker.log.connect(self._append_run_log)
        self._representative_worker.progress.connect(
            self._update_representative_selection_progress
        )
        self._representative_worker.finished.connect(
            self._finish_representative_selection
        )
        self._representative_worker.failed.connect(
            self._fail_representative_selection
        )
        self._representative_worker.finished.connect(
            self._representative_thread.quit
        )
        self._representative_worker.failed.connect(
            self._representative_thread.quit
        )
        self._representative_thread.finished.connect(
            self._cleanup_representative_thread
        )
        self._representative_thread.finished.connect(
            self._representative_thread.deleteLater
        )
        self._representative_worker.finished.connect(
            self._representative_worker.deleteLater
        )
        self._representative_worker.failed.connect(
            self._representative_worker.deleteLater
        )
        self._representative_thread.start()

    def _update_representative_selection_progress(
        self,
        processed: int,
        total: int,
        message: str,
    ) -> None:
        total = max(int(total), 1)
        processed = max(0, min(int(processed), total))
        self.representative_progress_bar.setRange(0, total)
        self.representative_progress_bar.setValue(processed)
        self.representative_status_label.setText(
            f"Representative selection: {message}"
        )
        percent_complete = int(round((100.0 * processed) / total))
        self._set_task_progress(message, percent_complete)

    def _finish_representative_selection(
        self,
        metadata: RepresentativeSelectionMetadata,
    ) -> None:
        self.compute_representatives_button.setEnabled(True)
        self.representative_status_label.setText(
            "Representative selection: complete"
        )
        self.representative_progress_bar.setValue(
            self.representative_progress_bar.maximum()
        )
        state = self._project_source_state
        if (
            state is None
            or self._representative_job_state is None
            or state is not self._representative_job_state
        ):
            self._append_run_log(
                "Representative selection finished for an older project "
                "state, so the current window was not overwritten."
            )
            self._set_task_progress(
                "Representative clusters finished for a previous state.",
                100,
            )
            return

        state.representative_selection = metadata
        state.packmol_planning = None
        state.packmol_setup = None
        state.constraint_generation = None
        self.preview_representatives_button.setEnabled(True)
        self.representative_summary_box.setPlainText(
            self._representative_summary_text(metadata)
        )
        self._populate_packmol_planning_controls()
        self._populate_constraint_controls()
        self._update_readiness_progress()
        self._set_task_progress(
            "Representative clusters ready.",
            100,
        )
        self._append_run_log(
            "Computed representative clusters in "
            f"{metadata.selection_mode} mode."
        )
        self._populate_solvent_controls()

    def _fail_representative_selection(self, message: str) -> None:
        self.compute_representatives_button.setEnabled(True)
        self.representative_status_label.setText(
            "Representative selection: failed"
        )
        self.representative_progress_bar.setRange(0, 1)
        self.representative_progress_bar.setValue(0)
        summary_message = (
            "Unable to compute representative clusters: " f"{message}"
        )
        self.representative_summary_box.setPlainText(summary_message)
        self._append_run_log(summary_message)
        self._set_task_progress("Representative selection failed.", 0)
        QMessageBox.warning(
            self,
            "Representative selection failed",
            message,
        )

    def _cleanup_representative_thread(self) -> None:
        self._representative_job_state = None
        self._representative_worker = None
        self._representative_thread = None

    def closeEvent(self, event) -> None:
        if self._representative_thread is not None:
            QMessageBox.information(
                self,
                "Representative selection running",
                "Representative-cluster selection is still running. Wait "
                "for it to finish before closing this window.",
            )
            event.ignore()
            return
        super().closeEvent(event)

    def _preview_representative_clusters(self) -> None:
        state = self._project_source_state
        if state is None or state.representative_selection is None:
            QMessageBox.information(
                self,
                "No representative selection",
                (
                    "Save representative structures before opening the "
                    "preview window."
                ),
            )
            return
        try:
            preview_clusters = build_representative_preview_clusters(
                state,
                state.representative_selection,
            )
        except Exception as exc:
            self._append_run_log(
                f"Unable to build representative preview: {exc}"
            )
            QMessageBox.warning(
                self,
                "Representative preview failed",
                str(exc),
            )
            return
        if not preview_clusters:
            QMessageBox.information(
                self,
                "No preview data available",
                (
                    "The representative selection did not produce any bond "
                    "or angle distributions to preview."
                ),
            )
            return
        self._representative_preview_window = RepresentativePreviewWindow(
            preview_clusters,
            parent=self,
        )
        self._representative_preview_window.show()
        self._representative_preview_window.raise_()
        self._representative_preview_window.activateWindow()
        self._append_run_log("Opened representative preview window.")

    def _browse_solvent_reference_pdb(self) -> None:
        selected_path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "Choose Solvent Reference PDB",
            self.solvent_reference_edit.text().strip() or str(Path.home()),
            "PDB Files (*.pdb)",
        )
        if not selected_path:
            return
        self.solvent_reference_edit.setText(selected_path)
        self._handle_solvent_reference_changed()

    def _current_solvent_settings(self) -> SolventHandlingSettings:
        selected_mode = self.generated_pdb_mode_combo.currentData()
        coordinated_solvent_mode = (
            str(selected_mode).strip()
            if selected_mode is not None
            and str(selected_mode).strip()
            in {"no_solvent", "partial_solvent", "full_solvent"}
            else "automatic_detection"
        )
        return SolventHandlingSettings(
            coordinated_solvent_mode=coordinated_solvent_mode,
            reference_source=str(
                self.solvent_reference_source_combo.currentData() or "preset"
            ),
            preset_name=str(self.solvent_preset_combo.currentData() or "dmf"),
            custom_reference_path=(
                self.solvent_reference_edit.text().strip() or None
            ),
            reference_match_tolerance_a=float(
                self.solvent_reference_match_tolerance_spin.value()
            ),
            director_atom_name=(
                str(self.solvent_director_atom_combo.currentData())
                if self.solvent_director_atom_combo.currentData() is not None
                else None
            ),
            minimum_solvent_atom_separation_a=float(
                self.solvent_minimum_separation_spin.value()
            ),
            solute_atom_settings=self._selected_solvent_coordination_settings(),
        )

    def _update_solvent_build_panel_state(self) -> None:
        state = self._project_source_state
        reference_identifier = self._selected_solvent_reference_identifier()
        has_reference = reference_identifier is not None
        has_representatives = (
            state is not None and state.representative_selection is not None
        )
        builder_required = self._solvent_shell_builder_required()
        self._set_solvent_shell_builder_controls_enabled(builder_required)
        if not builder_required:
            self.analyze_solvent_outputs_button.setEnabled(False)
            self.build_solvent_outputs_button.setEnabled(False)
            return
        self.analyze_solvent_outputs_button.setEnabled(
            bool(has_reference and has_representatives)
        )
        if not has_reference or not has_representatives:
            self.build_solvent_outputs_button.setEnabled(False)
            return
        analysis = self._solvent_distribution_analysis
        if analysis is None:
            self.build_solvent_outputs_button.setEnabled(True)
            return
        if self._solvent_build_has_required_coordination_settings(analysis):
            self.build_solvent_outputs_button.setEnabled(True)
            return
        self.build_solvent_outputs_button.setEnabled(False)

    def _analyze_representative_solvent_states(self) -> None:
        self._run_representative_solvent_analysis(
            progress_message="Analyzing representative solvent states...",
            completion_message="Representative solvent analysis ready.",
            log_completion="Analyzed representative solvent states.",
        )

    def _run_representative_solvent_analysis(
        self,
        *,
        progress_message: str,
        completion_message: str | None = None,
        log_completion: str | None = None,
    ) -> RepresentativeSolventDistributionAnalysis | None:
        state = self._project_source_state
        if state is None:
            QMessageBox.information(
                self,
                "No SAXS project loaded",
                "Load a SAXS project before analyzing representative solvent states.",
            )
            return None
        if state.representative_selection is None:
            QMessageBox.information(
                self,
                "No representative selection",
                "Save representative structures before analyzing the representative solvent states.",
            )
            return None
        if self._selected_solvent_reference_identifier() is None:
            QMessageBox.information(
                self,
                "No solvent reference selected",
                "Choose a bundled preset or custom solvent reference PDB before analyzing the representative structures.",
            )
            return None
        try:
            self._set_task_progress(
                progress_message,
                35,
            )
            self._append_run_log("Analyzing representative solvent states.")
            analysis = analyze_representative_solvent_distribution(
                state,
                self._current_solvent_settings(),
                representative_metadata=state.representative_selection,
            )
        except Exception as exc:
            self._solvent_distribution_analysis = None
            message = (
                "Unable to analyze representative solvent states: " f"{exc}"
            )
            self.solvent_summary_box.setPlainText(message)
            self._append_run_log(message)
            self._set_task_progress("Solvent analysis failed.", 0)
            QMessageBox.warning(
                self,
                "Representative solvent analysis failed",
                str(exc),
            )
            self._populate_solvent_cutoff_table(
                {},
                self._selected_solvent_coordination_settings(),
            )
            self._refresh_generated_pdb_browser(None)
            self._update_solvent_status_panel(None)
            self._update_solvent_build_panel_state()
            return None
        self._solvent_distribution_analysis = analysis
        self.solvent_summary_box.setPlainText(analysis.summary_text())
        self._populate_solvent_cutoff_table(
            analysis.aggregate_solute_element_counts,
            self._selected_solvent_coordination_settings(),
        )
        self._refresh_generated_pdb_browser(None)
        self._update_solvent_status_panel(None)
        self._update_solvent_build_panel_state()
        if completion_message is not None:
            self._set_task_progress(completion_message, 100)
        if log_completion is not None:
            self._append_run_log(log_completion)
        return analysis

    def _current_packmol_planning_settings(self) -> PackmolPlanningSettings:
        return PackmolPlanningSettings(
            planning_mode=str(
                self.packmol_planning_mode_combo.currentData() or "per_element"
            ),
            box_side_length_a=float(self.packmol_box_side_spin.value()),
            free_solvent_reference=self._selected_packmol_free_solvent_reference(),
        )

    def _current_packmol_setup_settings(self) -> PackmolSetupSettings:
        return PackmolSetupSettings(
            tolerance_angstrom=float(self.packmol_tolerance_spin.value()),
            free_solvent_reference=self._selected_packmol_free_solvent_reference(),
        )

    def _selected_packmol_free_solvent_reference(self) -> str | None:
        current_data = self.packmol_free_solvent_combo.currentData()
        if current_data is None:
            return None
        text = str(current_data).strip()
        return text or None

    def _normalize_packmol_free_solvent_identifier(
        self,
        identifier: str | None,
    ) -> str | None:
        if identifier is None:
            return None
        text = str(identifier).strip()
        if not text:
            return None
        candidate = Path(text).expanduser()
        if candidate.is_file():
            return str(candidate.resolve())
        for preset in self._available_solvent_presets:
            preset_path = str(Path(preset.path).expanduser().resolve())
            if text == preset.name or text == preset_path:
                return preset_path
        return None

    def _populate_packmol_free_solvent_choices(
        self,
        *,
        selected_identifier: str | None = None,
    ) -> None:
        state = self._project_source_state
        current_identifier = (
            selected_identifier
            or self._selected_packmol_free_solvent_reference()
        )
        combo = self.packmol_free_solvent_combo
        combo.blockSignals(True)
        combo.clear()
        seen_paths: set[str] = set()
        for preset in self._available_solvent_presets:
            preset_path = str(Path(preset.path).expanduser().resolve())
            combo.addItem(preset.name, preset_path)
            seen_paths.add(preset_path)

        extra_identifiers = [
            current_identifier,
            (
                None
                if state is None or state.solvent_handling is None
                else state.solvent_handling.reference_path
            ),
            (
                None
                if state is None or state.packmol_planning is None
                else state.packmol_planning.settings.free_solvent_reference
            ),
            (
                None
                if state is None or state.packmol_setup is None
                else state.packmol_setup.free_solvent_reference_path
            ),
        ]
        for identifier in extra_identifiers:
            normalized = self._normalize_packmol_free_solvent_identifier(
                identifier
            )
            if normalized is None or normalized in seen_paths:
                continue
            combo.addItem(Path(normalized).stem, normalized)
            seen_paths.add(normalized)

        target_identifier = self._normalize_packmol_free_solvent_identifier(
            current_identifier
        )
        if target_identifier is not None:
            index = combo.findData(target_identifier)
            if index >= 0:
                combo.setCurrentIndex(index)
        elif combo.count() > 0:
            combo.setCurrentIndex(0)
        combo.setEnabled(combo.count() > 0 and state is not None)
        combo.blockSignals(False)

    def _current_constraint_settings(self) -> ConstraintGenerationSettings:
        return ConstraintGenerationSettings(
            bond_length_tolerance_angstrom=float(
                self.constraint_length_tolerance_spin.value()
            ),
            bond_angle_tolerance_degrees=float(
                self.constraint_angle_tolerance_spin.value()
            ),
        )

    def _sync_packmol_inputs_to_linked_container(
        self,
        setup_metadata: PackmolSetupMetadata,
    ) -> PackmolDockerSyncResult | None:
        state = self._project_source_state
        if state is None or state.packmol_docker_link is None:
            return None
        link = state.packmol_docker_link
        try:
            result = self._create_packmol_docker_client().sync_packmol_inputs(
                link,
                state.rmcsetup_paths.packmol_inputs_dir,
                packmol_setup_metadata=setup_metadata,
            )
        except Exception as exc:
            link.last_sync_at = datetime.now().isoformat(timespec="seconds")
            link.last_sync_status = "error"
            link.last_sync_message = str(exc)
            self._save_packmol_docker_link(link)
            self.packmol_docker_summary_box.setPlainText(
                self._packmol_docker_summary_text(setup_metadata)
            )
            self._append_run_log(
                "Packmol Docker sync failed after local build: " f"{exc}"
            )
            QMessageBox.warning(
                self,
                "Packmol Docker sync failed",
                "Packmol inputs were built locally, but syncing them to the "
                "linked Docker container failed.\n\n"
                f"{exc}",
            )
            return None
        link.last_sync_at = result.synced_at
        link.last_sync_status = "success"
        link.last_sync_message = result.summary_text()
        self._save_packmol_docker_link(link)
        self.packmol_docker_summary_box.setPlainText(
            self._packmol_docker_summary_text(setup_metadata)
        )
        self._append_run_log(result.summary_text())
        return result

    def _build_representative_solvent_outputs(self) -> None:
        state = self._project_source_state
        if state is None:
            QMessageBox.information(
                self,
                "No SAXS project loaded",
                "Load a SAXS project before building representative PDB outputs.",
            )
            return
        if state.representative_selection is None:
            QMessageBox.information(
                self,
                "No representative selection",
                "Save representative structures before building representative PDB outputs.",
            )
            return
        if not self._solvent_shell_builder_required():
            QMessageBox.information(
                self,
                "Full-solvent representatives selected",
                "The active representative structure set already has full "
                "solvent, so solvent-state analysis and solvent-shell "
                "building are not required.",
            )
            return
        analysis = self._solvent_distribution_analysis
        if analysis is None:
            analysis = self._run_representative_solvent_analysis(
                progress_message=(
                    "Analyzing representative solvent states before build..."
                ),
            )
            if analysis is None:
                return
        if not self._solvent_build_has_required_coordination_settings(
            analysis
        ):
            self._update_solvent_build_panel_state()
            QMessageBox.information(
                self,
                "Coordination settings required",
                "Select at least one coordination-center element and set "
                "its average coordination number and director distance "
                "before building full-solvent representatives from "
                "no-solvent structures.",
            )
            return
        try:
            self._set_task_progress(
                "Building solvent-decorated representative PDBs...",
                35,
            )
            self._append_run_log(
                "Starting Solvent Shell Builder representative PDB build."
            )
            solvent_metadata = build_representative_solvent_outputs(
                state,
                self._current_solvent_settings(),
                representative_metadata=state.representative_selection,
                distribution_analysis=analysis,
            )
        except Exception as exc:
            message = f"Unable to build representative PDB outputs: {exc}"
            self.solvent_summary_box.setPlainText(message)
            self._append_run_log(message)
            self._set_task_progress("Solvent build failed.", 0)
            QMessageBox.warning(
                self,
                "Solvent Shell Builder failed",
                str(exc),
            )
            return
        state.solvent_handling = solvent_metadata
        state.packmol_planning = None
        state.packmol_setup = None
        state.constraint_generation = None
        self._apply_solvent_metadata(solvent_metadata)
        self.solvent_summary_box.setPlainText(
            self._solvent_summary_text(solvent_metadata)
        )
        self._populate_packmol_planning_controls()
        self._populate_constraint_controls()
        self._update_readiness_progress()
        self._set_task_progress(
            "Solvent-decorated representative PDBs ready.",
            100,
        )
        self._append_run_log(
            "Built solvent-decorated representative PDB outputs."
        )

    def _compute_packmol_plan(self) -> None:
        state = self._project_source_state
        if state is None:
            QMessageBox.information(
                self,
                "No SAXS project loaded",
                "Load a SAXS project before planning Packmol cluster counts.",
            )
            return
        if state.representative_selection is None:
            QMessageBox.information(
                self,
                "No representative selection",
                "Save representative structures before planning Packmol counts.",
            )
            return
        if state.solution_properties.result is None:
            QMessageBox.information(
                self,
                "No solution properties",
                "Calculate solution properties before planning Packmol counts.",
            )
            return
        try:
            self._set_task_progress(
                "Computing Packmol cluster counts...",
                50,
            )
            self._append_run_log("Starting Packmol planning.")
            planning_metadata = build_packmol_plan(
                state,
                self._current_packmol_planning_settings(),
                representative_metadata=state.representative_selection,
                solution_metadata=state.solution_properties,
                solvent_metadata=state.solvent_handling,
            )
        except Exception as exc:
            message = f"Unable to compute Packmol planning counts: {exc}"
            self.packmol_plan_summary_box.setPlainText(message)
            self._append_run_log(message)
            self._set_task_progress("Packmol planning failed.", 0)
            QMessageBox.warning(
                self,
                "Packmol planning failed",
                str(exc),
            )
            return
        state.packmol_planning = planning_metadata
        state.packmol_setup = None
        state.constraint_generation = None
        self.compute_packmol_plan_button.setEnabled(True)
        self.build_packmol_setup_button.setEnabled(
            self._active_representative_structure_set_is_ready()
        )
        self.packmol_plan_summary_box.setPlainText(
            self._packmol_plan_summary_text(planning_metadata)
        )
        self._apply_packmol_setup_metadata(None)
        self._populate_constraint_controls()
        self._refresh_packmol_plan_plot(planning_metadata)
        self.output_summary_box.setPlainText(self._output_structure_text())
        self._update_readiness_progress()
        self._set_task_progress("Packmol planning ready.", 100)
        self._append_run_log("Computed Packmol planning counts.")

    def _build_packmol_setup(self) -> None:
        state = self._project_source_state
        if state is None:
            QMessageBox.information(
                self,
                "No SAXS project loaded",
                "Load a SAXS project before generating Packmol inputs.",
            )
            return
        if state.packmol_planning is None:
            QMessageBox.information(
                self,
                "No Packmol plan",
                "Compute Packmol cluster counts before generating Packmol inputs.",
            )
            return
        if not self._active_representative_structure_set_is_ready():
            QMessageBox.information(
                self,
                "Full-solvent representatives not selected",
                "Select the Full solvent representative structure set in "
                "Representative Structures before generating Packmol inputs.",
            )
            return
        if self._selected_packmol_free_solvent_reference() is None:
            QMessageBox.information(
                self,
                "No free-solvent structure selected",
                "Choose the free-solvent structure used for the Packmol "
                "bulk-solvent population before generating Packmol inputs.",
            )
            return
        try:
            self._set_task_progress(
                "Building Packmol setup inputs...",
                70,
            )
            self._append_run_log("Starting Packmol setup build.")
            setup_metadata = build_packmol_setup(
                state,
                self._current_packmol_setup_settings(),
                plan_metadata=state.packmol_planning,
                representative_metadata=state.representative_selection,
                solvent_metadata=state.solvent_handling,
            )
        except Exception as exc:
            message = f"Unable to build Packmol setup: {exc}"
            self.packmol_build_summary_box.setPlainText(message)
            self._append_run_log(message)
            self._set_task_progress("Packmol setup failed.", 0)
            QMessageBox.warning(
                self,
                "Packmol setup failed",
                str(exc),
            )
            return
        state.packmol_setup = setup_metadata
        state.constraint_generation = None
        self._sync_packmol_inputs_to_linked_container(setup_metadata)
        self._apply_packmol_setup_metadata(setup_metadata)
        self._populate_constraint_controls()
        self.output_summary_box.setPlainText(self._output_structure_text())
        self._update_readiness_progress()
        self._set_task_progress("Packmol setup ready.", 100)
        self._append_run_log("Built Packmol setup inputs and audit report.")

    def _open_packmol_setup_folder(self) -> None:
        state = self._project_source_state
        if state is None or state.packmol_setup is None:
            QMessageBox.information(
                self,
                "No Packmol setup",
                "Build Packmol setup inputs before opening the setup folder.",
            )
            return
        folder_path = (
            state.rmcsetup_paths.packmol_inputs_dir.expanduser().resolve()
        )
        if not folder_path.is_dir():
            QMessageBox.warning(
                self,
                "Packmol setup folder missing",
                f"Could not find the Packmol setup folder:\n{folder_path}",
            )
            return
        try:
            self._open_path_in_file_manager(folder_path)
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Packmol setup folder",
                f"Could not open the Packmol setup folder:\n{exc}",
            )
            return
        self.statusBar().showMessage(
            f"Opened Packmol setup folder: {folder_path.name}"
        )
        self._append_run_log(
            f"Opened Packmol setup folder in Finder/file manager: {folder_path}"
        )

    def _open_constraints_folder(self) -> None:
        state = self._project_source_state
        if state is None or state.constraint_generation is None:
            QMessageBox.information(
                self,
                "No constraints generated",
                "Generate constraints before opening the constraints folder.",
            )
            return
        merged_path = (
            Path(state.constraint_generation.merged_constraints_path)
            .expanduser()
            .resolve()
        )
        if not merged_path.is_file():
            QMessageBox.warning(
                self,
                "Constraints file missing",
                f"Could not find the merged constraints file:\n{merged_path}",
            )
            return
        try:
            self._open_path_in_file_manager(merged_path)
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Constraints folder",
                f"Could not open the constraints folder:\n{exc}",
            )
            return
        self.statusBar().showMessage(
            f"Opened constraints file location: {merged_path.name}"
        )
        self._append_run_log(
            "Opened constraints file location in Finder/file manager: "
            f"{merged_path}"
        )

    def _open_constraints_preview(self) -> None:
        state = self._project_source_state
        if state is None or state.constraint_generation is None:
            QMessageBox.information(
                self,
                "No constraints generated",
                "Generate constraints before opening the merged constraints preview.",
            )
            return
        merged_path = (
            Path(state.constraint_generation.merged_constraints_path)
            .expanduser()
            .resolve()
        )
        if not merged_path.is_file():
            QMessageBox.warning(
                self,
                "Constraints preview unavailable",
                f"Could not find the merged constraints file:\n{merged_path}",
            )
            return
        try:
            self._constraints_preview_window = ConstraintsPreviewWindow(
                merged_path,
                parent=self,
            )
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Constraints preview unavailable",
                f"Could not open the merged constraints file:\n{exc}",
            )
            return
        self._track_child_tool_window(self._constraints_preview_window)
        self._constraints_preview_window.show()
        self._constraints_preview_window.raise_()
        self._constraints_preview_window.activateWindow()
        self._append_run_log(
            f"Opened merged constraints preview: {merged_path.name}"
        )

    def _generate_constraints(self) -> None:
        state = self._project_source_state
        if state is None:
            QMessageBox.information(
                self,
                "No SAXS project loaded",
                "Load a SAXS project before generating constraints.",
            )
            return
        if state.packmol_setup is None:
            QMessageBox.information(
                self,
                "No Packmol setup",
                "Build Packmol setup inputs before generating constraints.",
            )
            return
        try:
            self._set_task_progress(
                "Generating constraints...",
                85,
            )
            self._append_run_log("Starting constraint generation.")
            metadata = build_constraint_generation(
                state,
                self._current_constraint_settings(),
                packmol_setup_metadata=state.packmol_setup,
            )
        except Exception as exc:
            message = f"Unable to generate constraints: {exc}"
            self.constraints_summary_box.setPlainText(message)
            self._append_run_log(message)
            self._set_task_progress("Constraint generation failed.", 0)
            QMessageBox.warning(
                self,
                "Constraint generation failed",
                str(exc),
            )
            return
        state.constraint_generation = metadata
        self._apply_constraint_metadata(metadata)
        self.output_summary_box.setPlainText(self._output_structure_text())
        self._update_readiness_progress()
        self._set_task_progress("Constraint generation complete.", 100)
        self._append_run_log(
            "Generated per-structure constraints and merged fullrmc constraints."
        )

    def _apply_solution_metadata(
        self,
        metadata: SolutionPropertiesMetadata,
    ) -> None:
        self._apply_solution_settings(metadata.settings)
        self._select_solution_preset_name(
            self._matching_solution_preset_name(metadata.settings)
        )
        self.solution_output_box.setPlainText(
            self._solution_output_text(
                metadata.result,
                updated_at=metadata.updated_at,
            )
        )

    def _apply_solution_settings(
        self,
        settings: SolutionPropertiesSettings,
    ) -> None:
        previous_updating = self._updating_solution_preset_selection
        self._updating_solution_preset_selection = True
        try:
            self._set_combo_value(self.solution_mode_combo, settings.mode)
            self.solution_density_spin.setValue(settings.solution_density)
            self.solute_stoich_edit.setText(settings.solute_stoich)
            self.solvent_stoich_edit.setText(settings.solvent_stoich)
            self.molar_mass_solute_spin.setValue(settings.molar_mass_solute)
            self.molar_mass_solvent_spin.setValue(settings.molar_mass_solvent)
            self.mass_solute_spin.setValue(settings.mass_solute)
            self.mass_solvent_spin.setValue(settings.mass_solvent)
            self.mass_percent_solute_spin.setValue(
                settings.mass_percent_solute
            )
            self.total_mass_solution_spin.setValue(
                settings.total_mass_solution
            )
            self.molarity_spin.setValue(settings.molarity)
            self.molarity_element_edit.setText(settings.molarity_element)
            self._update_solution_mode_widgets()
        finally:
            self._updating_solution_preset_selection = previous_updating

    def _selected_solution_preset_name(self) -> str | None:
        payload = self.solution_preset_combo.currentData()
        if payload is None:
            return None
        return str(payload)

    def _reload_solution_presets(
        self,
        *,
        selected_name: str | None = None,
    ) -> None:
        previous_name = selected_name or self._selected_solution_preset_name()
        self._solution_presets = load_solution_property_presets()
        self.solution_preset_combo.blockSignals(True)
        self.solution_preset_combo.clear()
        self.solution_preset_combo.addItem("Current values", None)
        selected_index = 0
        for index, name in enumerate(
            ordered_solution_property_preset_names(self._solution_presets),
            start=1,
        ):
            preset = self._solution_presets[name]
            label = f"{name} (Built-in)" if preset.builtin else name
            self.solution_preset_combo.addItem(label, name)
            if name == previous_name:
                selected_index = index
        self.solution_preset_combo.setCurrentIndex(selected_index)
        self.solution_preset_combo.blockSignals(False)

    def _solution_settings_match(
        self,
        left: SolutionPropertiesSettings,
        right: SolutionPropertiesSettings,
    ) -> bool:
        float_fields = (
            "solution_density",
            "molar_mass_solute",
            "molar_mass_solvent",
            "mass_solute",
            "mass_solvent",
            "mass_percent_solute",
            "total_mass_solution",
            "molarity",
        )
        text_fields = (
            "mode",
            "solute_stoich",
            "solvent_stoich",
            "molarity_element",
        )
        for field_name in float_fields:
            if (
                abs(
                    float(getattr(left, field_name))
                    - float(getattr(right, field_name))
                )
                > 1e-9
            ):
                return False
        for field_name in text_fields:
            if str(getattr(left, field_name)) != str(
                getattr(right, field_name)
            ):
                return False
        return True

    def _matching_solution_preset_name(
        self,
        settings: SolutionPropertiesSettings,
    ) -> str | None:
        for name in ordered_solution_property_preset_names(
            self._solution_presets
        ):
            preset = self._solution_presets.get(name)
            if preset is None:
                continue
            if self._solution_settings_match(settings, preset.settings):
                return name
        return None

    def _select_solution_preset_name(self, preset_name: str | None) -> None:
        target_index = 0
        if preset_name is not None:
            for index in range(self.solution_preset_combo.count()):
                if self.solution_preset_combo.itemData(index) == preset_name:
                    target_index = index
                    break
        previous_updating = self._updating_solution_preset_selection
        self._updating_solution_preset_selection = True
        try:
            self.solution_preset_combo.setCurrentIndex(target_index)
        finally:
            self._updating_solution_preset_selection = previous_updating

    def _load_selected_solution_preset(self) -> None:
        preset_name = self._selected_solution_preset_name()
        if preset_name is None:
            QMessageBox.information(
                self,
                "No solution preset selected",
                "Select a solution preset to load.",
            )
            return
        preset = self._solution_presets.get(preset_name)
        if preset is None:
            QMessageBox.warning(
                self,
                "Solution preset unavailable",
                f"Unknown solution preset: {preset_name}",
            )
            return
        self._apply_solution_settings(preset.settings)
        self._select_solution_preset_name(preset.name)
        self._append_run_log(f"Loaded solution preset: {preset_name}")

    def _save_current_solution_as_preset(self) -> None:
        suggested_name = self._selected_solution_preset_name() or ""
        name, accepted = QInputDialog.getText(
            self,
            "Save Solution Preset",
            (
                "Enter a name for this solution-properties preset.\n"
                "It will be written to the editable package preset JSON."
            ),
            text=suggested_name,
        )
        if not accepted:
            return
        name = name.strip()
        if not name:
            QMessageBox.information(
                self,
                "Invalid preset name",
                "Enter a non-empty preset name.",
            )
            return
        if name in self._solution_presets:
            overwrite = QMessageBox.question(
                self,
                "Overwrite solution preset?",
                f"A solution preset named '{name}' already exists. Overwrite it?",
            )
            if overwrite != QMessageBox.StandardButton.Yes:
                return
        existing = self._solution_presets.get(name)
        try:
            save_custom_solution_property_preset(
                SolutionPropertiesPreset(
                    name=name,
                    settings=self._current_solution_settings(),
                    solute_molecule_count=(
                        existing.solute_molecule_count
                        if existing is not None
                        else 1
                    ),
                    notes=existing.notes if existing is not None else "",
                )
            )
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Unable to save solution preset",
                str(exc),
            )
            return
        self._reload_solution_presets(selected_name=name)
        self._append_run_log(f"Saved solution preset: {name}")

    def _on_solution_settings_changed(self, *_args: object) -> None:
        if self._updating_solution_preset_selection:
            return
        self._select_solution_preset_name(
            self._matching_solution_preset_name(
                self._current_solution_settings()
            )
        )

    def _selected_solution_mode(self) -> str:
        return str(self.solution_mode_combo.currentData() or "mass")

    def _current_solution_settings(self) -> SolutionPropertiesSettings:
        return SolutionPropertiesSettings(
            mode=self._selected_solution_mode(),
            solution_density=float(self.solution_density_spin.value()),
            solute_stoich=self.solute_stoich_edit.text().strip(),
            solvent_stoich=self.solvent_stoich_edit.text().strip(),
            molar_mass_solute=float(self.molar_mass_solute_spin.value()),
            molar_mass_solvent=float(self.molar_mass_solvent_spin.value()),
            mass_solute=float(self.mass_solute_spin.value()),
            mass_solvent=float(self.mass_solvent_spin.value()),
            mass_percent_solute=float(self.mass_percent_solute_spin.value()),
            total_mass_solution=float(self.total_mass_solution_spin.value()),
            molarity=float(self.molarity_spin.value()),
            molarity_element=self.molarity_element_edit.text().strip(),
        )

    def _on_solution_mode_changed(self) -> None:
        self._update_solution_mode_widgets()

    def _update_solution_mode_widgets(self) -> None:
        selected_mode = self._selected_solution_mode()
        mode_to_index = {
            "mass": 0,
            "mass_percent": 1,
            "molarity_per_liter": 2,
        }
        self.solution_mode_stack.setCurrentIndex(
            mode_to_index.get(selected_mode, 0)
        )
        self.solution_mode_hint_label.setText(
            solution_properties_mode_hint_text(selected_mode)
        )

    def _calculate_solution_properties(self) -> None:
        state = self._project_source_state
        if state is None:
            QMessageBox.information(
                self,
                "No SAXS project loaded",
                "Load a SAXS project before calculating solution properties.",
            )
            return
        settings = self._current_solution_settings()
        try:
            self._set_task_progress(
                "Calculating solution properties...",
                15,
            )
            self._append_run_log("Starting solution-properties calculation.")
            result = calculate_solution_properties(settings)
            metadata = save_solution_properties_metadata(
                state.rmcsetup_paths.solution_properties_path,
                settings=settings,
                result=result,
            )
        except Exception as exc:
            message = f"Unable to calculate solution properties: {exc}"
            self.solution_output_box.setPlainText(message)
            self._append_run_log(message)
            self._set_task_progress(
                "Solution-properties calculation failed.", 0
            )
            QMessageBox.warning(self, "Calculation failed", str(exc))
            return

        state.solution_properties = metadata
        state.packmol_planning = None
        state.packmol_setup = None
        state.constraint_generation = None
        self.solution_output_box.setPlainText(
            self._solution_output_text(
                result,
                updated_at=metadata.updated_at,
            )
        )
        self._populate_packmol_planning_controls()
        self._populate_constraint_controls()
        self._update_readiness_progress()
        self._set_task_progress("Solution properties ready.", 100)
        self._append_run_log(
            "Calculated and saved solution properties metadata."
        )

    def _solution_output_text(
        self,
        result: SolutionPropertiesResult | None,
        *,
        updated_at: str | None = None,
    ) -> str:
        state = self._project_source_state
        if result is None:
            if state is None:
                return (
                    "Load a SAXS project to edit and calculate solution "
                    "properties."
                )
            return (
                "No saved solution-properties calculation yet.\n\n"
                "Choose an input mode, fill in the solution metadata, and "
                "press Calculate to persist the result to:\n"
                f"{state.rmcsetup_paths.solution_properties_path}"
            )
        return result.summary_text(updated_at=updated_at)

    def _solvent_summary_text(
        self,
        metadata: SolventHandlingMetadata | None,
    ) -> str:
        state = self._project_source_state
        if state is None:
            return (
                "Load a SAXS project and save representative structures "
                "before running the Solvent Shell Builder."
            )
        if metadata is None:
            active_mode = self._active_generated_pdb_mode()
            if active_mode == "full_solvent":
                return (
                    "Imported representative structures already include the "
                    "Full solvent structure set, so this step is ready for "
                    "Packmol without rebuilding solvent outputs.\n\n"
                    "Solvent state analysis and solvent-shell building are "
                    "not required for this selection.\n\n"
                    "Metadata will be written to:\n"
                    f"{state.rmcsetup_paths.solvent_handling_path}"
                )
            return (
                "No full-solvent representative export has been built yet "
                f"for the active {representative_structure_mode_label(active_mode)} "
                "set.\n\n"
                "Choose a solvent reference, review the coordination "
                "settings, and press Build Solvent-Decorated Representative "
                "PDBs. The required solvent-state analysis will run first.\n\n"
                "Metadata will be written to:\n"
                f"{state.rmcsetup_paths.solvent_handling_path}"
            )
        return (
            metadata.summary_text()
            + "\n\nMetadata path:\n"
            + str(state.rmcsetup_paths.solvent_handling_path)
        )

    def _packmol_plan_summary_text(
        self,
        metadata: PackmolPlanningMetadata | None,
    ) -> str:
        state = self._project_source_state
        if state is None:
            return (
                "Load a SAXS project, calculate solution properties, and "
                "save representative structures before planning Packmol "
                "cluster counts."
            )
        if metadata is None:
            return (
                "No Packmol plan has been saved yet.\n\n"
                "Press Compute Cluster Counts to convert the selected DREAM "
                "distribution and representative structures into planned box "
                "counts, solvent-allocation totals, and output reports.\n\n"
                "Choose the free-solvent structure above before planning if "
                "the representative source files already contain solvent.\n\n"
                "Metadata will be written to:\n"
                f"{state.rmcsetup_paths.packmol_plan_path}"
            )
        return (
            metadata.summary_text()
            + "\n\nMetadata path:\n"
            + str(state.rmcsetup_paths.packmol_plan_path)
            + "\nReport path:\n"
            + str(state.rmcsetup_paths.packmol_plan_report_path)
        )

    def _packmol_setup_summary_text(
        self,
        metadata: PackmolSetupMetadata | None,
    ) -> str:
        state = self._project_source_state
        if state is None:
            return (
                "Load a SAXS project, save representative structures, "
                "build solvent-decorated representative PDBs, and plan counts "
                "before generating Packmol inputs."
            )
        if metadata is None:
            if not self._active_representative_structure_set_is_ready():
                text = (
                    "No Packmol setup has been built yet.\n\n"
                    "Select the Full solvent representative structure set in "
                    "Representative Structures before generating Packmol "
                    "inputs.\n\n"
                    "Metadata will be written to:\n"
                    f"{state.rmcsetup_paths.packmol_setup_path}"
                )
                if state.packmol_docker_link is not None:
                    text += (
                        "\n\nLinked Docker target:\n"
                        + state.packmol_docker_link.summary_text()
                    )
                return text
            text = (
                "No Packmol setup has been built yet.\n\n"
                "Press Build Packmol Setup to generate representative input "
                "PDBs, the Packmol .inp file, the selected free-solvent "
                "single-molecule PDB, and the audit report.\n\n"
                "Metadata will be written to:\n"
                f"{state.rmcsetup_paths.packmol_setup_path}"
            )
            if state.packmol_docker_link is not None:
                text += (
                    "\n\nLinked Docker target:\n"
                    + state.packmol_docker_link.summary_text()
                )
            return text
        text = (
            metadata.summary_text()
            + "\n\nMetadata path:\n"
            + str(state.rmcsetup_paths.packmol_setup_path)
            + "\nAudit report:\n"
            + str(state.rmcsetup_paths.packmol_audit_report_path)
        )
        if state.packmol_docker_link is not None:
            text += (
                "\n\nLinked Docker target:\n"
                + state.packmol_docker_link.summary_text(
                    packmol_setup_metadata=metadata
                )
            )
        return text

    def _packmol_docker_summary_text(
        self,
        packmol_setup_metadata: PackmolSetupMetadata | None = None,
    ) -> str:
        state = self._project_source_state
        if state is None:
            return (
                "Load a SAXS project before linking a Packmol Docker "
                "container."
            )
        if state.packmol_docker_link is None:
            return (
                "No Packmol Docker container is linked yet.\n\n"
                "Use Tools > Link Packmol Docker Container to validate a "
                "container, confirm Packmol is installed, and choose the "
                "container-side project folder. The required bind-mounted "
                f"root inside the container is {DEFAULT_PACKMOL_CONTAINER_ROOT}.\n\n"
                "Project link metadata will be written to:\n"
                f"{state.rmcsetup_paths.packmol_docker_link_path}"
            )
        return (
            state.packmol_docker_link.summary_text(
                packmol_setup_metadata=packmol_setup_metadata
            )
            + "\n\nProject link metadata path:\n"
            + str(state.rmcsetup_paths.packmol_docker_link_path)
        )

    def _constraint_summary_text(
        self,
        metadata: ConstraintGenerationMetadata | None,
    ) -> str:
        state = self._project_source_state
        if state is None:
            return (
                "Load a SAXS project and build Packmol setup inputs before "
                "generating constraints."
            )
        if metadata is None:
            return (
                "No constraints have been generated yet.\n\n"
                "Set the bond-length and bond-angle tolerances, then press "
                "Generate Constraints to write one file per Packmol structure "
                "plus a merged fullrmc constraints file.\n\n"
                "Metadata will be written to:\n"
                f"{state.rmcsetup_paths.constraint_generation_path}"
            )
        return (
            metadata.summary_text()
            + "\n\nMetadata path:\n"
            + str(state.rmcsetup_paths.constraint_generation_path)
            + "\nMerged file:\n"
            + str(metadata.merged_constraints_path)
        )

    def _refresh_packmol_plan_plot(
        self,
        metadata: PackmolPlanningMetadata | None,
    ) -> None:
        figure = self._packmol_plan_figure
        figure.clear()
        if metadata is None or not metadata.entries:
            axis = figure.add_subplot(111)
            axis.text(
                0.5,
                0.5,
                "Compute Packmol cluster counts to preview the\n"
                "active histogram/count distribution.",
                ha="center",
                va="center",
                transform=axis.transAxes,
                wrap=True,
            )
            axis.set_axis_off()
            self._packmol_plan_canvas.draw_idle()
            return

        state = self._project_source_state
        distribution_entries = ()
        if state is not None and state.representative_selection is not None:
            distribution_entries = (
                state.representative_selection.distribution_selection.entries
            )

        original_lookup: dict[tuple[str, str], float] = {}
        structures: set[str] = set()
        motifs: set[str] = set()
        for entry in distribution_entries:
            if not entry.structure:
                continue
            key = (entry.structure, entry.motif)
            original_lookup[key] = original_lookup.get(key, 0.0) + max(
                float(entry.cluster_count),
                0.0,
            )
            structures.add(entry.structure)
            motifs.add(entry.motif)

        dream_lookup: dict[tuple[str, str], float] = {}
        packmol_lookup: dict[tuple[str, str], float] = {}
        for entry in metadata.entries:
            if not entry.structure:
                continue
            key = (entry.structure, entry.motif)
            dream_lookup[key] = dream_lookup.get(key, 0.0) + max(
                entry.selected_weight,
                0.0,
            )
            packmol_lookup[key] = packmol_lookup.get(key, 0.0) + max(
                entry.planned_count_weight,
                0.0,
            )
            structures.add(entry.structure)
            motifs.add(entry.motif)

        ordered_structures = list(sort_stoich_labels(structures))
        ordered_motifs = sorted(
            motifs,
            key=lambda motif: (motif != "no_motif", _natural_sort_key(motif)),
        )
        if not ordered_structures or not ordered_motifs:
            axis = figure.add_subplot(111)
            axis.text(
                0.5,
                0.5,
                "No Packmol histogram data is available for preview.",
                ha="center",
                va="center",
                transform=axis.transAxes,
                wrap=True,
            )
            axis.set_axis_off()
            self._packmol_plan_canvas.draw_idle()
            return

        structure_index = {
            structure: index
            for index, structure in enumerate(ordered_structures)
        }
        motif_index = {
            motif: index for index, motif in enumerate(ordered_motifs)
        }

        original_matrix = np.zeros(
            (len(ordered_structures), len(ordered_motifs)),
            dtype=float,
        )
        dream_matrix = np.zeros_like(original_matrix)
        packmol_matrix = np.zeros_like(original_matrix)
        for (structure, motif), value in original_lookup.items():
            original_matrix[
                structure_index[structure],
                motif_index[motif],
            ] = value
        for (structure, motif), value in dream_lookup.items():
            dream_matrix[
                structure_index[structure],
                motif_index[motif],
            ] = value
        for (structure, motif), value in packmol_lookup.items():
            packmol_matrix[
                structure_index[structure],
                motif_index[motif],
            ] = value

        def _to_percent(matrix: np.ndarray) -> np.ndarray:
            total = float(matrix.sum())
            if total <= 0.0:
                return np.zeros_like(matrix)
            return matrix * (100.0 / total)

        original_percent = _to_percent(original_matrix)
        dream_percent = _to_percent(dream_matrix)
        packmol_percent = _to_percent(packmol_matrix)
        max_total = max(
            float(original_percent.sum(axis=1).max(initial=0.0)),
            float(dream_percent.sum(axis=1).max(initial=0.0)),
            float(packmol_percent.sum(axis=1).max(initial=0.0)),
        )
        y_limit = max(max_total + 6.0, 10.0)
        x_positions = np.arange(len(ordered_structures), dtype=float)
        axes = figure.subplots(3, 1, sharex=True, sharey=True)
        if not isinstance(axes, np.ndarray):
            axes = np.asarray([axes], dtype=object)

        panel_data = (
            (
                (
                    "Original cluster distribution"
                    if distribution_entries
                    else "Original cluster distribution unavailable"
                ),
                original_percent,
            ),
            ("DREAM fit model distribution used", dream_percent),
            ("Packmol planned distribution", packmol_percent),
        )

        for axis_index, (axis, (title, matrix)) in enumerate(
            zip(axes, panel_data)
        ):
            bottoms = np.zeros(len(ordered_structures), dtype=float)
            for motif_position, motif in enumerate(ordered_motifs):
                heights = matrix[:, motif_position]
                axis.bar(
                    x_positions,
                    heights,
                    bottom=bottoms,
                    label=(
                        _packmol_motif_label(motif)
                        if axis_index == 0
                        else None
                    ),
                    color=_PACKMOL_PREVIEW_COLORS[
                        motif_position % len(_PACKMOL_PREVIEW_COLORS)
                    ],
                    edgecolor="white",
                    width=0.8,
                )
                bottoms += heights
            for index, total in enumerate(bottoms):
                if total >= 1.0:
                    axis.text(
                        x_positions[index],
                        total + 0.8,
                        f"{total:.1f}%",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )
            axis.set_ylim(0.0, y_limit)
            axis.set_ylabel("Share (%)")
            axis.set_title(title, fontsize=10)
            axis.grid(axis="y", alpha=0.25)
            axis.set_axisbelow(True)

        axes[0].legend(title="Motif")
        axes[-1].set_xlabel("Structure")
        axes[-1].set_xticks(x_positions)
        axes[-1].set_xticklabels(
            [format_stoich_for_axis(label) for label in ordered_structures],
            rotation=45,
            ha="right",
        )
        figure.tight_layout()
        self._packmol_plan_canvas.draw_idle()

    def _representative_summary_text(
        self,
        metadata: RepresentativeSelectionMetadata | None,
    ) -> str:
        state = self._project_source_state
        if state is None:
            return (
                "Load a SAXS project and choose a DREAM source before "
                "loading saved representative structures."
            )
        if metadata is None:
            return (
                "No representative structures have been saved yet.\n\n"
                "Use Open Representative Structures to create or update the "
                "saved project set, then reload it here. rmcsetup will "
                "combine those structure files with the selected DREAM "
                "distribution weights, solution-density targets, solvent "
                "handling, Packmol planning, and cluster-specific "
                "constraints.\n\n"
                "Metadata will be written to:\n"
                f"{state.rmcsetup_paths.representative_selection_path}"
            )
        return (
            metadata.summary_text()
            + "\n\nMetadata path:\n"
            + str(state.rmcsetup_paths.representative_selection_path)
        )

    def _save_current_selection_as_favorite(self) -> None:
        state = self._project_source_state
        selection = self.current_selection()
        if state is None or selection is None:
            QMessageBox.information(
                self,
                "No DREAM model selected",
                "Choose a valid project and DREAM run before saving the "
                "current DREAM model selection.",
            )
            return
        selection.selection_source = "rmcsetup"
        selection.selected_at = datetime.now().isoformat(timespec="seconds")
        selection.label = self._selection_label(selection)

        settings = self.project_manager.load_project(
            state.settings.project_dir
        )
        settings.dream_favorite_selection = selection
        settings.dream_favorite_history = list(settings.dream_favorite_history)
        settings.dream_favorite_history.append(selection)
        self.project_manager.save_project(settings)

        self._append_run_log(
            f"Saved DREAM model: {self._selection_label(selection)}"
        )
        self._refresh_project_source()
        self._apply_selection(selection, announce=False)

    def _apply_project_favorite(self) -> None:
        state = self._project_source_state
        if state is None or state.favorite_selection is None:
            QMessageBox.information(
                self,
                "No saved DREAM model",
                "This project does not have a saved DREAM model selection "
                "yet.",
            )
            return
        self._apply_selection(state.favorite_selection)

    def _load_history_entry(self) -> None:
        selection = self.selected_history_entry()
        if selection is None:
            return
        self._apply_selection(selection)

    def _refresh_dream_source_summary(self) -> None:
        state = self._project_source_state
        run = self.selected_run_record()
        if state is None:
            self.dream_source_summary_box.setPlainText(
                "Load a SAXS project to discover saved DREAM runs and select "
                "the source for RMC setup."
            )
            self._refresh_dream_preview()
            return
        if run is None:
            self.dream_source_summary_box.setPlainText(
                "No valid DREAM runs were discovered in the current project."
            )
            self._refresh_dream_preview()
            return
        selection = self.current_selection()
        lines = [
            f"Selected run: {run.run_name}",
            f"Run location: {run.relative_path}",
            f"Template: {run.template_name or 'unknown'}",
            f"Model name: {run.model_name or 'unknown'}",
            (
                "Saved run defaults: "
                f"{run.settings.bestfit_method}, "
                f"{self._posterior_filter_label(run.settings.posterior_filter_mode)}"
            ),
            "",
            "Current DREAM model selection:",
        ]
        if selection is not None:
            lines.append(self._selection_metadata_text(selection, indent="  "))
            lines.append("")
            lines.append(
                "Matches saved DREAM model: "
                f"{'yes' if self._selection_matches_saved_model(selection) else 'no'}"
            )
        self.dream_source_summary_box.setPlainText("\n".join(lines))
        self.favorite_summary_box.setPlainText(self._favorite_summary_text())
        self._refresh_dream_preview()
        self._update_readiness_progress()

    def _selection_matches_saved_model(
        self,
        selection: DreamBestFitSelection | None,
    ) -> bool:
        state = self._project_source_state
        if (
            state is None
            or selection is None
            or state.favorite_selection is None
        ):
            return False
        saved_selection = state.favorite_selection
        return (
            saved_selection.run_relative_path == selection.run_relative_path
            and saved_selection.bestfit_method == selection.bestfit_method
            and (
                saved_selection.posterior_filter_mode
                == selection.posterior_filter_mode
            )
            and abs(
                saved_selection.posterior_top_percent
                - selection.posterior_top_percent
            )
            < 1e-9
            and saved_selection.posterior_top_n == selection.posterior_top_n
            and abs(
                saved_selection.credible_interval_low
                - selection.credible_interval_low
            )
            < 1e-9
            and abs(
                saved_selection.credible_interval_high
                - selection.credible_interval_high
            )
            < 1e-9
        )

    def _refresh_dream_preview(self) -> None:
        state = self._project_source_state
        run = self.selected_run_record()
        selection = self.current_selection()
        self._current_dream_model_plot_data = None
        if state is None:
            self.dream_preview_status_label.setText(
                "Load a SAXS project to preview a DREAM model fit and its "
                "posterior weight distributions."
            )
            self._draw_empty_preview(
                self._dream_model_preview_figure,
                self._dream_model_preview_canvas,
                "No DREAM model preview is available yet.",
            )
            self._draw_empty_preview(
                self._dream_violin_preview_figure,
                self._dream_violin_preview_canvas,
                "No posterior weight preview is available yet.",
            )
            return
        if run is None or selection is None:
            self.dream_preview_status_label.setText(
                "Choose a valid DREAM run to preview the selected model."
            )
            self._draw_empty_preview(
                self._dream_model_preview_figure,
                self._dream_model_preview_canvas,
                "Choose a DREAM run to render the model-fit preview.",
            )
            self._draw_empty_preview(
                self._dream_violin_preview_figure,
                self._dream_violin_preview_canvas,
                "Choose a DREAM run to render the posterior weight preview.",
            )
            return

        try:
            loader = self._dream_results_loader(run)
            selection_kwargs = self._dream_selection_kwargs(selection)
            summary = loader.get_summary(**selection_kwargs)
            model_plot = loader.build_model_fit_data(**selection_kwargs)
            violin_plot = loader.build_violin_data(
                mode="weights_only",
                posterior_filter_mode=selection.posterior_filter_mode,
                posterior_top_percent=selection.posterior_top_percent,
                posterior_top_n=selection.posterior_top_n,
                credible_interval_low=selection.credible_interval_low,
                credible_interval_high=selection.credible_interval_high,
                sample_source="filtered_posterior",
                weight_order="weight_index",
            )
        except Exception as exc:
            self.dream_preview_status_label.setText(
                f"Selected run: {run.run_name}\n"
                f"Preview could not be rendered: {exc}"
            )
            self._draw_empty_preview(
                self._dream_model_preview_figure,
                self._dream_model_preview_canvas,
                "The selected DREAM model fit could not be rendered.",
            )
            self._draw_empty_preview(
                self._dream_violin_preview_figure,
                self._dream_violin_preview_canvas,
                "The posterior weight preview could not be rendered.",
            )
            return

        self.dream_preview_status_label.setText(
            self._dream_preview_status_text(run, selection, summary)
        )
        self._current_dream_model_plot_data = model_plot
        self._plot_dream_model_preview(model_plot)
        self._plot_dream_weight_violin_preview(summary, violin_plot)

    def _refresh_dream_model_preview_from_cache(self) -> None:
        plot_data = self._current_dream_model_plot_data
        if plot_data is None:
            return
        self._plot_dream_model_preview(plot_data)

    def _dream_results_loader(
        self,
        run: RMCDreamRunRecord,
    ) -> SAXSDreamResultsLoader:
        loader = self._dream_results_loader_cache.get(run.run_dir)
        if loader is None:
            loader = SAXSDreamResultsLoader(
                run.run_dir,
                burnin_percent=run.settings.burnin_percent,
            )
            self._dream_results_loader_cache[run.run_dir] = loader
        return loader

    @staticmethod
    def _dream_selection_kwargs(
        selection: DreamBestFitSelection,
    ) -> dict[str, float | int | str]:
        return {
            "bestfit_method": selection.bestfit_method,
            "posterior_filter_mode": selection.posterior_filter_mode,
            "posterior_top_percent": selection.posterior_top_percent,
            "posterior_top_n": selection.posterior_top_n,
            "credible_interval_low": selection.credible_interval_low,
            "credible_interval_high": selection.credible_interval_high,
        }

    def _dream_preview_status_text(
        self,
        run: RMCDreamRunRecord,
        selection: DreamBestFitSelection,
        summary: DreamSummary,
    ) -> str:
        detail = self._posterior_filter_detail(selection)
        return "\n".join(
            [
                f"Selected run: {run.run_name} ({run.relative_path})",
                (
                    "Best-fit method: "
                    f"{selection.bestfit_method}; Posterior filter: {detail}"
                ),
                (
                    "Posterior samples used: "
                    f"{summary.posterior_sample_count}; "
                    "Matches saved DREAM model: "
                    f"{'yes' if self._selection_matches_saved_model(selection) else 'no'}"
                ),
            ]
        )

    def _posterior_filter_detail(
        self,
        selection: DreamBestFitSelection,
    ) -> str:
        label = self._posterior_filter_label(selection.posterior_filter_mode)
        if selection.posterior_filter_mode == "top_percent_logp":
            return f"{label} ({selection.posterior_top_percent:g}%)"
        if selection.posterior_filter_mode == "top_n_logp":
            return f"{label} ({selection.posterior_top_n} samples)"
        return label

    def _draw_empty_preview(
        self,
        figure: Figure,
        canvas: FigureCanvasQTAgg,
        message: str,
    ) -> None:
        figure.clear()
        axis = figure.add_subplot(111)
        axis.text(
            0.5,
            0.5,
            message,
            ha="center",
            va="center",
            wrap=True,
        )
        axis.set_axis_off()
        figure.tight_layout()
        canvas.draw()

    def _plot_dream_model_preview(
        self,
        plot_data: DreamModelPlotData,
    ) -> None:
        self._dream_model_preview_figure.clear()
        axis = self._dream_model_preview_figure.add_subplot(111)
        if self.show_experimental_trace_checkbox.isChecked():
            axis.scatter(
                plot_data.q_values,
                plot_data.experimental_intensities,
                color="black",
                s=14,
                label="Experimental",
                zorder=3,
            )
        if (
            self.show_solvent_trace_checkbox.isChecked()
            and plot_data.solvent_contribution is not None
        ):
            solvent_values = np.asarray(
                plot_data.solvent_contribution,
                dtype=float,
            )
            solvent_mask = np.isfinite(solvent_values) & (solvent_values > 0.0)
            if np.any(solvent_mask):
                axis.plot(
                    np.asarray(plot_data.q_values, dtype=float)[solvent_mask],
                    solvent_values[solvent_mask],
                    color="green",
                    linewidth=1.5,
                    label="Solvent contribution",
                )
        if self.show_model_trace_checkbox.isChecked():
            axis.plot(
                plot_data.q_values,
                plot_data.model_intensities,
                color="tab:red",
                linewidth=2,
                label=f"Model ({plot_data.bestfit_method})",
            )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel(Q_A_INVERSE_LABEL)
        axis.set_ylabel("Intensity (arb. units)")
        axis.set_title(f"DREAM refinement: {plot_data.template_name}")
        axis.text(
            0.02,
            0.02,
            "\n".join(
                [
                    f"RMSE: {plot_data.rmse:.4g}",
                    f"Mean |res|: {plot_data.mean_abs_residual:.4g}",
                    f"R²: {plot_data.r_squared:.4g}",
                ]
            ),
            transform=axis.transAxes,
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
        handles, labels = axis.get_legend_handles_labels()
        if handles and labels:
            axis.legend(loc="best")
        self._dream_model_preview_figure.tight_layout()
        self._dream_model_preview_canvas.draw()

    def _plot_dream_weight_violin_preview(
        self,
        summary: DreamSummary,
        violin_data: DreamViolinPlotData,
    ) -> None:
        self._dream_violin_preview_figure.clear()
        axis = self._dream_violin_preview_figure.add_subplot(111)
        if not violin_data.parameter_names or violin_data.samples.size == 0:
            axis.text(
                0.5,
                0.5,
                "The selected DREAM run does not expose posterior weight "
                "parameters to preview.",
                ha="center",
                va="center",
                wrap=True,
            )
            axis.set_axis_off()
            self._dream_violin_preview_figure.tight_layout()
            self._dream_violin_preview_canvas.draw()
            return

        payload = self._weight_violin_payload(summary, violin_data)
        positions = np.arange(1, len(payload["display_names"]) + 1)
        violin_parts = axis.violinplot(
            payload["samples"],
            positions=positions,
            showmedians=True,
        )
        for body in violin_parts["bodies"]:
            body.set_facecolor((0.235, 0.447, 0.741, 0.62))
            body.set_edgecolor("#1f2a36")
            body.set_linewidth(0.8)
        for key in ("cbars", "cmins", "cmaxes"):
            artist = violin_parts.get(key)
            if artist is not None:
                artist.set_color("#6b6b6b")
                artist.set_linewidth(1.1)
        median_artist = violin_parts.get("cmedians")
        if median_artist is not None:
            median_artist.set_color("#303030")
            median_artist.set_linewidth(1.3)
        axis.vlines(
            positions,
            payload["interval_low_values"],
            payload["interval_high_values"],
            color="#6b6b6b",
            linewidth=1.7,
            label=(
                f"p{summary.credible_interval_low:g}-"
                f"p{summary.credible_interval_high:g} interval"
            ),
        )
        axis.scatter(
            positions,
            payload["selected_values"],
            color="#c0392b",
            s=20,
            zorder=3,
            label=f"Selected {summary.bestfit_method}",
        )
        axis.set_xticks(positions)
        axis.set_xticklabels(
            payload["display_names"],
            rotation=45,
            ha="right",
        )
        axis.set_ylim(0.0, 1.0)
        axis.set_ylabel("Weight fraction")
        axis.set_title("Posterior weight distributions")
        axis.legend(loc="best")
        self._dream_violin_preview_figure.tight_layout()
        self._dream_violin_preview_canvas.draw()

    @staticmethod
    def _weight_violin_payload(
        summary: DreamSummary,
        violin_data: DreamViolinPlotData,
    ) -> dict[str, object]:
        summary_lookup = {
            name: index
            for index, name in enumerate(summary.full_parameter_names)
        }
        samples = np.asarray(violin_data.samples, dtype=float)
        if samples.ndim == 1:
            samples = samples.reshape(-1, 1)
        return {
            "display_names": list(violin_data.display_names),
            "samples": samples,
            "selected_values": np.asarray(
                [
                    summary.bestfit_params[summary_lookup[name]]
                    for name in violin_data.parameter_names
                ],
                dtype=float,
            ),
            "interval_low_values": np.asarray(
                [
                    summary.interval_low_values[summary_lookup[name]]
                    for name in violin_data.parameter_names
                ],
                dtype=float,
            ),
            "interval_high_values": np.asarray(
                [
                    summary.interval_high_values[summary_lookup[name]]
                    for name in violin_data.parameter_names
                ],
                dtype=float,
            ),
        }

    def selected_run_record(self) -> RMCDreamRunRecord | None:
        payload = self.dream_run_combo.currentData()
        if isinstance(payload, RMCDreamRunRecord):
            return payload
        return None

    def selected_posterior_filter_mode(self) -> str:
        return str(
            self.posterior_filter_combo.currentData() or "all_post_burnin"
        )

    def selected_history_entry(self) -> DreamBestFitSelection | None:
        payload = self.favorite_history_combo.currentData()
        if isinstance(payload, DreamBestFitSelection):
            return payload
        return None

    def current_selection(self) -> DreamBestFitSelection | None:
        run = self.selected_run_record()
        if run is None:
            return None
        return DreamBestFitSelection(
            run_name=run.run_name,
            run_relative_path=run.relative_path,
            bestfit_method=str(
                self.bestfit_method_combo.currentData() or "map"
            ),
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
            template_name=run.template_name,
            model_name=run.model_name,
            selection_source="rmcsetup",
        )

    @staticmethod
    def _posterior_filter_label(mode: str) -> str:
        labels = {
            "all_post_burnin": "all post-burnin samples",
            "top_percent_logp": "top % by log-posterior",
            "top_n_logp": "top N by log-posterior",
        }
        return labels.get(mode, mode)

    def _selection_label(self, selection: DreamBestFitSelection) -> str:
        timestamp = selection.selected_at or "unsaved"
        return (
            f"{timestamp} • {selection.run_name} • "
            f"{selection.bestfit_method}"
        )

    def _selection_metadata_text(
        self,
        selection: DreamBestFitSelection,
        *,
        indent: str = "",
    ) -> str:
        lines = [
            f"{indent}Run: {selection.run_name}",
            f"{indent}Relative path: {selection.run_relative_path}",
            f"{indent}Best-fit method: {selection.bestfit_method}",
            (
                f"{indent}Posterior filter: "
                f"{self._posterior_filter_label(selection.posterior_filter_mode)}"
            ),
        ]
        if selection.posterior_filter_mode == "top_percent_logp":
            lines.append(
                f"{indent}Top percent: {selection.posterior_top_percent:g}"
            )
        elif selection.posterior_filter_mode == "top_n_logp":
            lines.append(f"{indent}Top N: {selection.posterior_top_n}")
        lines.append(
            (
                f"{indent}Credible interval: "
                f"{selection.credible_interval_low:g} - "
                f"{selection.credible_interval_high:g}"
            )
        )
        if selection.template_name:
            lines.append(f"{indent}Template: {selection.template_name}")
        if selection.model_name:
            lines.append(f"{indent}Model name: {selection.model_name}")
        if selection.selection_source:
            lines.append(
                f"{indent}Selection source: {selection.selection_source}"
            )
        if selection.selected_at:
            lines.append(f"{indent}Saved at: {selection.selected_at}")
        return "\n".join(lines)

    def _append_run_log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.run_log_box.appendPlainText(f"[{timestamp}] {message}")

    @staticmethod
    def _open_path_in_file_manager(path: Path) -> None:
        resolved_path = path.expanduser().resolve()
        target_path = (
            resolved_path if resolved_path.is_dir() else resolved_path.parent
        )
        if not target_path.exists():
            raise FileNotFoundError(target_path)
        if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(target_path))):
            raise RuntimeError(
                "Qt could not open the requested folder in the file manager."
            )


def launch_rmcsetup_ui(
    project_dir: str | Path | None = None,
) -> int:
    """Launch the Qt6 rmcsetup UI."""
    app = QApplication.instance()
    owns_app = app is None
    if app is None:
        prepare_saxshell_application_identity()
        app = QApplication(sys.argv)
    configure_saxshell_application(app)

    window = RMCSetupMainWindow(initial_project_dir=project_dir)
    _OPEN_WINDOWS.append(window)
    window.show()
    if owns_app:
        return app.exec()
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point for launching the Qt6 rmcsetup UI."""
    parser = argparse.ArgumentParser(
        prog="rmcsetup-ui",
        description="Launch the SAXSShell rmcsetup UI scaffold.",
    )
    parser.add_argument(
        "project_dir",
        nargs="?",
        help="Optional SAXS project directory to prefill in the UI.",
    )
    args = parser.parse_args(argv)
    return launch_rmcsetup_ui(args.project_dir)


__all__ = ["RMCSetupMainWindow", "launch_rmcsetup_ui", "main"]
