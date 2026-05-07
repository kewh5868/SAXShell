from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
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
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from saxshell.fullrmc.solvent_shell_builder import (
    DEFAULT_REFERENCE_MATCH_TOLERANCE_A,
    SolventShellAnalysisResult,
    analyze_solvent_shell,
    build_solvent_shell_output,
    default_director_atom_name,
    reference_atom_choices,
)
from saxshell.saxs.electron_density_mapping.ui.viewer import (
    ElectronDensityStructureViewer,
)
from saxshell.saxs.electron_density_mapping.workflow import (
    load_electron_density_structure,
)
from saxshell.saxs.ui.branding import (
    configure_saxshell_application,
    load_saxshell_icon,
    prepare_saxshell_application_identity,
)
from saxshell.xyz2pdb import (
    ReferenceLibraryEntry,
    default_reference_library_dir,
    list_reference_library,
)


class SolventShellBuilderMainWindow(QMainWindow):
    """Small beta utility for isolated solvent-shell detection tests."""

    _DEFAULT_MINIMUM_SOLVENT_SEPARATION_A = 1.2
    _DEFAULT_SOLUTE_DISTANCE_CUTOFF_A = 2.5

    def __init__(
        self,
        *,
        initial_project_dir: Path | None = None,
        initial_input_path: Path | None = None,
        reference_library_dir: Path | None = None,
    ) -> None:
        super().__init__()
        self._initial_project_dir = (
            None
            if initial_project_dir is None
            else Path(initial_project_dir).expanduser().resolve()
        )
        self.reference_library_dir = (
            default_reference_library_dir()
            if reference_library_dir is None
            else Path(reference_library_dir).expanduser().resolve()
        )
        self._available_references = list_reference_library(
            self.reference_library_dir
        )
        self._analysis_result: SolventShellAnalysisResult | None = None
        self._build_result_text: str | None = None
        self._last_suggested_output_path: str | None = None
        self._solute_cutoff_spins: dict[str, QDoubleSpinBox] = {}
        self._coordination_center_items: dict[str, QTableWidgetItem] = {}
        self._coordination_target_spins: dict[str, QDoubleSpinBox] = {}
        self._updating_solute_table = False
        self._browse_start_path = (
            self._initial_project_dir
            if self._initial_project_dir is not None
            else Path.home()
        )

        self.setWindowTitle("Solvent Shell Builder (Beta)")
        self.setWindowIcon(load_saxshell_icon())
        self.resize(920, 700)

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        intro_label = QLabel(
            "Beta utility for detecting whether an input PDB or XYZ contains "
            "no, partial, or complete solvent molecules that match one "
            "selected reference preset, then building a solvated output PDB "
            "for no-solvent or partial-solvent cases."
        )
        intro_label.setWordWrap(True)
        layout.addWidget(intro_label)

        self._pane_splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self._pane_splitter.setChildrenCollapsible(False)
        self._pane_splitter.setStretchFactor(0, 0)
        self._pane_splitter.setStretchFactor(1, 1)
        layout.addWidget(self._pane_splitter, stretch=1)

        self._left_scroll_area = QScrollArea(self)
        self._left_scroll_area.setWidgetResizable(True)
        self._left_scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._right_scroll_area = QScrollArea(self)
        self._right_scroll_area.setWidgetResizable(True)
        self._right_scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._left_panel = QWidget()
        self._right_panel = QWidget()
        self._left_layout = QVBoxLayout(self._left_panel)
        self._left_layout.setContentsMargins(10, 10, 10, 10)
        self._left_layout.setSpacing(10)
        self._right_layout = QVBoxLayout(self._right_panel)
        self._right_layout.setContentsMargins(10, 10, 10, 10)
        self._right_layout.setSpacing(10)
        self._left_scroll_area.setWidget(self._left_panel)
        self._right_scroll_area.setWidget(self._right_panel)
        self._pane_splitter.addWidget(self._left_scroll_area)
        self._pane_splitter.addWidget(self._right_scroll_area)
        self._pane_splitter.setSizes([380, 620])

        self.input_group = QGroupBox("Input Settings")
        input_layout = QVBoxLayout(self.input_group)
        form = QFormLayout()
        self.reference_preset_combo = QComboBox()
        for preset in self._available_references:
            self.reference_preset_combo.addItem(preset.name, preset.name)
        self.reference_preset_combo.currentIndexChanged.connect(
            self._handle_reference_selection_changed
        )
        form.addRow("Reference molecule", self.reference_preset_combo)

        self.reference_details_box = QPlainTextEdit()
        self.reference_details_box.setReadOnly(True)
        self.reference_details_box.setPlaceholderText(
            "Reference preset details will appear here."
        )
        self.reference_details_box.setMinimumHeight(72)
        self.reference_details_box.setMaximumBlockCount(8)
        form.addRow("Preset details", self.reference_details_box)

        self.reference_match_tolerance_spin = QDoubleSpinBox()
        self.reference_match_tolerance_spin.setDecimals(3)
        self.reference_match_tolerance_spin.setRange(0.001, 10.0)
        self.reference_match_tolerance_spin.setSingleStep(0.025)
        self.reference_match_tolerance_spin.setSuffix(" A")
        self.reference_match_tolerance_spin.setValue(
            DEFAULT_REFERENCE_MATCH_TOLERANCE_A
        )
        self.reference_match_tolerance_spin.setToolTip(
            "Maximum allowed anchor-pair distance deviation when matching "
            "the selected solvent reference."
        )
        self.reference_match_tolerance_spin.valueChanged.connect(
            self._handle_reference_tolerance_changed
        )
        form.addRow(
            "Reference match tolerance",
            self.reference_match_tolerance_spin,
        )

        input_row = QHBoxLayout()
        self.input_path_edit = QLineEdit()
        self.input_path_edit.setPlaceholderText(
            "Choose an input PDB or XYZ structure file..."
        )
        self.input_path_edit.editingFinished.connect(
            self._handle_input_path_edited
        )
        input_row.addWidget(self.input_path_edit, stretch=1)
        self.browse_input_button = QPushButton("Browse...")
        self.browse_input_button.clicked.connect(self._browse_input_file)
        input_row.addWidget(self.browse_input_button)
        form.addRow("Input file", input_row)
        input_layout.addLayout(form)

        action_row = QHBoxLayout()
        action_row.addStretch(1)
        self.analyze_button = QPushButton("Analyze Solvent Shell")
        self.analyze_button.clicked.connect(self._analyze_input_structure)
        self.analyze_button.setEnabled(bool(self._available_references))
        action_row.addWidget(self.analyze_button)
        input_layout.addLayout(action_row)
        self._left_layout.addWidget(self.input_group)

        self.cluster_status_group = QGroupBox("Detected Cluster Status")
        status_layout = QVBoxLayout(self.cluster_status_group)
        self.cluster_status_headline_label = QLabel()
        self.cluster_status_headline_label.setWordWrap(True)
        self.cluster_status_stats_label = QLabel()
        self.cluster_status_stats_label.setWordWrap(True)
        self.cluster_status_stats_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        status_layout.addWidget(self.cluster_status_headline_label)
        status_layout.addWidget(self.cluster_status_stats_label)
        self._left_layout.addWidget(self.cluster_status_group)

        self.build_group = QGroupBox("Build Solvation Shell")
        build_layout = QVBoxLayout(self.build_group)
        self.build_intro_label = QLabel(
            "Use the analyzed solvent status to complete partial solvent "
            "molecules or build a new shell for a cluster with no "
            "coordinated solvent."
        )
        self.build_intro_label.setWordWrap(True)
        build_layout.addWidget(self.build_intro_label)

        build_form = QFormLayout()
        self.director_atom_combo = QComboBox()
        self.director_atom_combo.setToolTip(
            "Reference atom that should point toward the solute cluster "
            "during solvent placement."
        )
        build_form.addRow("Director atom", self.director_atom_combo)

        self.minimum_solvent_separation_spin = QDoubleSpinBox()
        self.minimum_solvent_separation_spin.setDecimals(3)
        self.minimum_solvent_separation_spin.setRange(0.0, 20.0)
        self.minimum_solvent_separation_spin.setSingleStep(0.1)
        self.minimum_solvent_separation_spin.setSuffix(" A")
        self.minimum_solvent_separation_spin.setValue(
            self._DEFAULT_MINIMUM_SOLVENT_SEPARATION_A
        )
        self.minimum_solvent_separation_spin.setToolTip(
            "Minimum allowed atom-to-atom separation between placed solvent "
            "molecules and already placed neighbors."
        )
        build_form.addRow(
            "Solvent-solvent separation",
            self.minimum_solvent_separation_spin,
        )

        output_row = QHBoxLayout()
        self.output_path_edit = QLineEdit()
        self.output_path_edit.setPlaceholderText(
            "Choose where the solvated output PDB should be written..."
        )
        self.output_path_edit.editingFinished.connect(
            self._handle_output_path_edited
        )
        output_row.addWidget(self.output_path_edit, stretch=1)
        self.browse_output_button = QPushButton("Browse...")
        self.browse_output_button.clicked.connect(self._browse_output_file)
        output_row.addWidget(self.browse_output_button)
        build_form.addRow("Output PDB", output_row)
        build_layout.addLayout(build_form)

        self.solute_cutoff_status_label = QLabel(
            "Analyze the input structure to populate the recognized solute "
            "atom types and their solvent-placement distances."
        )
        self.solute_cutoff_status_label.setWordWrap(True)
        build_layout.addWidget(self.solute_cutoff_status_label)

        self.solute_cutoff_table = QTableWidget(0, 5)
        self.solute_cutoff_table.setHorizontalHeaderLabels(
            [
                "Solute Element",
                "Atom Count",
                "Coordination Center",
                "Avg Coord #",
                "Director Distance (A)",
            ]
        )
        self.solute_cutoff_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.solute_cutoff_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.solute_cutoff_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.solute_cutoff_table.verticalHeader().setVisible(False)
        cutoff_header = self.solute_cutoff_table.horizontalHeader()
        cutoff_header.setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        cutoff_header.setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents
        )
        cutoff_header.setSectionResizeMode(
            2, QHeaderView.ResizeMode.ResizeToContents
        )
        cutoff_header.setSectionResizeMode(
            3, QHeaderView.ResizeMode.ResizeToContents
        )
        cutoff_header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        self.solute_cutoff_table.itemChanged.connect(
            self._handle_solute_table_item_changed
        )
        build_layout.addWidget(self.solute_cutoff_table)

        self.build_status_label = QLabel(
            "Analyze the input structure to enable solvent-shell building."
        )
        self.build_status_label.setWordWrap(True)
        self.build_status_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        build_layout.addWidget(self.build_status_label)

        build_action_row = QHBoxLayout()
        build_action_row.addStretch(1)
        self.build_output_button = QPushButton("Build Solvated Output PDB")
        self.build_output_button.clicked.connect(self._build_solvated_output)
        build_action_row.addWidget(self.build_output_button)
        build_layout.addLayout(build_action_row)
        self._left_layout.addWidget(self.build_group)

        left_notes_group = QGroupBox("Notes")
        left_notes_layout = QVBoxLayout(left_notes_group)
        self.input_notes_label = QLabel(
            "Run the analysis after changing the reference or input file. "
            "Adjust the reference match tolerance if the solvent geometry "
            "needs a looser or tighter match. Then review the director atom, "
            "solute cutoffs, and solvent separation before building the "
            "solvated output PDB."
        )
        self.input_notes_label.setWordWrap(True)
        left_notes_layout.addWidget(self.input_notes_label)
        self._left_layout.addWidget(left_notes_group)
        self._left_layout.addStretch(1)

        visualizer_group = QGroupBox("Structure Visualizer")
        visualizer_layout = QVBoxLayout(visualizer_group)
        self.visualizer_status_label = QLabel(
            "Choose a PDB or XYZ input file to preview the structure."
        )
        self.visualizer_status_label.setWordWrap(True)
        visualizer_layout.addWidget(self.visualizer_status_label)
        self.structure_viewer = ElectronDensityStructureViewer(
            self._right_panel
        )
        self.structure_viewer.setMinimumHeight(460)
        visualizer_layout.addWidget(self.structure_viewer, stretch=1)
        self._right_layout.addWidget(visualizer_group, stretch=1)

        summary_group = QGroupBox("Generated Outputs")
        summary_layout = QVBoxLayout(summary_group)
        self.summary_box = QPlainTextEdit()
        self.summary_box.setReadOnly(True)
        self.summary_box.setPlaceholderText(
            "Analysis results will appear here."
        )
        summary_layout.addWidget(self.summary_box)
        self._right_layout.addWidget(summary_group)

        residue_group = QGroupBox("Matched PDB Residues")
        residue_layout = QVBoxLayout(residue_group)
        self.residue_status_label = QLabel(
            "Residue-level solvent types are reported only for PDB inputs."
        )
        self.residue_status_label.setWordWrap(True)
        residue_layout.addWidget(self.residue_status_label)
        self.residue_table = QTableWidget(0, 5)
        self.residue_table.setHorizontalHeaderLabels(
            [
                "Residue",
                "Matched Molecules",
                "Residue Numbers",
                "Atoms / Molecule",
                "Elements",
            ]
        )
        self.residue_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.residue_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.residue_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        header = self.residue_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        residue_layout.addWidget(self.residue_table)
        self._right_layout.addWidget(residue_group)

        mismatch_group = QGroupBox("Incomplete / Partial Solvent Candidates")
        mismatch_layout = QVBoxLayout(mismatch_group)
        self.mismatch_status_label = QLabel(
            "Incomplete solvent-like candidates are reported after analysis."
        )
        self.mismatch_status_label.setWordWrap(True)
        mismatch_layout.addWidget(self.mismatch_status_label)
        self.mismatch_table = QTableWidget(0, 7)
        self.mismatch_table.setHorizontalHeaderLabels(
            [
                "Residue",
                "Residue Number",
                "Observed Atoms",
                "Matched / Ref",
                "Missing Atoms",
                "Extra Atoms",
                "Reason",
            ]
        )
        self.mismatch_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.mismatch_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.mismatch_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        mismatch_header = self.mismatch_table.horizontalHeader()
        mismatch_header.setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        mismatch_header.setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents
        )
        mismatch_header.setSectionResizeMode(
            2, QHeaderView.ResizeMode.ResizeToContents
        )
        mismatch_header.setSectionResizeMode(
            3, QHeaderView.ResizeMode.ResizeToContents
        )
        mismatch_header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        mismatch_header.setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        mismatch_header.setSectionResizeMode(6, QHeaderView.ResizeMode.Stretch)
        mismatch_layout.addWidget(self.mismatch_table)
        self._right_layout.addWidget(mismatch_group)
        self._right_layout.addStretch(1)

        if initial_input_path is not None:
            resolved_input = Path(initial_input_path).expanduser().resolve()
            if resolved_input.is_dir():
                self._browse_start_path = resolved_input
            else:
                self._browse_start_path = resolved_input.parent
                self.input_path_edit.setText(str(resolved_input))

        self._populate_director_atom_choices()
        self._update_reference_details()
        self._update_suggested_output_path(force=True)
        self._update_cluster_status_panel(None)
        self._populate_solute_cutoff_table(None)
        self._update_build_panel_state()
        if not self._available_references:
            self._set_cluster_status_panel_text(
                "No solvent status is available yet.",
                "No reference presets were found. Add a reference molecule "
                "to the library before using this beta tool.",
            )
            self._set_build_status_text(
                "Solvation-shell building is unavailable.",
                "No reference presets were found. Add a solvent reference "
                "molecule to the library before using this beta tool.",
            )
            self.summary_box.setPlainText(
                "No reference presets were found. Add a reference molecule "
                "to the library before using this beta tool."
            )
        elif self.input_path_edit.text().strip():
            self._refresh_structure_preview()

    def _selected_reference(self) -> ReferenceLibraryEntry | None:
        selected_name = self.reference_preset_combo.currentData()
        if selected_name is None:
            return None
        for preset in self._available_references:
            if preset.name == selected_name:
                return preset
        return None

    def _update_reference_details(self) -> None:
        preset = self._selected_reference()
        if preset is None:
            self.reference_details_box.setPlainText(
                "No solvent reference preset is selected."
            )
            return
        suggested_director = default_director_atom_name(
            preset.name,
            reference_library_dir=self.reference_library_dir,
        )
        director_text = (
            suggested_director if suggested_director is not None else "n/a"
        )
        self.reference_details_box.setPlainText(
            f"Residue {preset.residue_name}\n"
            f"Atom count: {preset.atom_count}\n"
            f"Suggested director atom: {director_text}\n"
            f"Reference file: {preset.path.name}"
        )

    def _browse_input_file(self) -> None:
        start_path = self.input_path_edit.text().strip() or str(
            self._browse_start_path
        )
        selected_path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "Choose Input Structure",
            start_path,
            "Structure files (*.pdb *.xyz);;PDB files (*.pdb);;XYZ files (*.xyz)",
        )
        if not selected_path:
            return
        resolved_path = Path(selected_path).expanduser().resolve()
        self._browse_start_path = resolved_path.parent
        self.input_path_edit.setText(str(resolved_path))
        self._handle_input_path_edited()

    def _handle_reference_selection_changed(self) -> None:
        self._populate_director_atom_choices()
        self._update_reference_details()
        self._update_suggested_output_path()
        self._clear_analysis_outputs()

    def _handle_reference_tolerance_changed(self) -> None:
        self._clear_analysis_outputs()

    def _handle_input_path_edited(self) -> None:
        self._update_suggested_output_path()
        self._clear_analysis_outputs()
        self._refresh_structure_preview()

    def _handle_output_path_edited(self) -> None:
        output_text = self.output_path_edit.text().strip()
        if output_text and not output_text.lower().endswith(".pdb"):
            self.output_path_edit.setText(f"{output_text}.pdb")

    def _analyze_input_structure(self) -> None:
        preset = self._selected_reference()
        if preset is None:
            QMessageBox.information(
                self,
                "No reference selected",
                "Choose a solvent reference molecule before analyzing the input file.",
            )
            return
        input_text = self.input_path_edit.text().strip()
        if not input_text:
            QMessageBox.information(
                self,
                "No input file selected",
                "Choose a PDB or XYZ file to inspect.",
            )
            return
        input_path = Path(input_text).expanduser().resolve()
        self._refresh_structure_preview()
        try:
            result = analyze_solvent_shell(
                input_path,
                preset.name,
                reference_library_dir=self.reference_library_dir,
                reference_match_tolerance_a=float(
                    self.reference_match_tolerance_spin.value()
                ),
            )
        except Exception as exc:
            self._analysis_result = None
            self.summary_box.setPlainText(str(exc))
            self._set_cluster_status_panel_text(
                "Solvent-shell analysis failed.",
                str(exc),
            )
            self._populate_residue_table(None)
            self._populate_mismatch_table(None)
            self.statusBar().showMessage("Solvent-shell analysis failed", 5000)
            QMessageBox.warning(
                self,
                "Solvent-shell analysis failed",
                str(exc),
            )
            return
        self._analysis_result = result
        self._build_result_text = None
        self.summary_box.setPlainText(self._combined_summary_text())
        self._update_cluster_status_panel(result)
        self._populate_solute_cutoff_table(result)
        self._update_build_panel_state()
        self._populate_residue_table(result)
        self._populate_mismatch_table(result)
        self._refresh_structure_preview(preserve_display=True)
        self.statusBar().showMessage(
            result.cluster_solvent_status_text,
            5000,
        )

    def _clear_analysis_outputs(self) -> None:
        self._analysis_result = None
        self._build_result_text = None
        self.summary_box.clear()
        self._update_cluster_status_panel(None)
        self._populate_solute_cutoff_table(None)
        self._update_build_panel_state()
        self._populate_residue_table(None)
        self._populate_mismatch_table(None)

    def _set_cluster_status_panel_text(
        self,
        headline: str,
        details: str,
    ) -> None:
        self.cluster_status_headline_label.setText(headline)
        self.cluster_status_stats_label.setText(details)

    def _update_cluster_status_panel(
        self,
        result: SolventShellAnalysisResult | None,
    ) -> None:
        if result is None:
            self._set_cluster_status_panel_text(
                "No solvent status has been determined yet.",
                "Choose a reference molecule and input structure, then run "
                "Analyze Solvent Shell to identify whether the cluster "
                "contains no, partial, or complete solvent molecules.",
            )
            return
        self._set_cluster_status_panel_text(
            result.cluster_solvent_status_text,
            result.status_statistics_text(),
        )

    def _previewable_input_path(self) -> Path | None:
        input_text = self.input_path_edit.text().strip()
        if not input_text:
            return None
        input_path = Path(input_text).expanduser().resolve()
        if not input_path.is_file() or input_path.suffix.lower() not in {
            ".pdb",
            ".xyz",
        }:
            return None
        return input_path

    def _refresh_structure_preview(
        self,
        *,
        preview_path: Path | None = None,
        preserve_display: bool = False,
    ) -> None:
        structure_path = (
            preview_path
            if preview_path is not None
            else self._previewable_input_path()
        )
        if (
            structure_path is None
            or not structure_path.is_file()
            or structure_path.suffix.lower() not in {".pdb", ".xyz"}
        ):
            self.structure_viewer.draw_placeholder()
            self.visualizer_status_label.setText(
                "Choose a valid PDB or XYZ input file to preview the structure."
            )
            return
        try:
            structure = load_electron_density_structure(
                structure_path,
                center_mode="center_of_mass",
                include_bonds=True,
                include_comment=True,
            )
        except Exception as exc:
            self.structure_viewer.draw_placeholder()
            self.visualizer_status_label.setText(
                f"Unable to preview {structure_path.name}: {exc}"
            )
            return
        scene_key = f"solvent-shell-builder:{structure_path}"
        if (
            preserve_display
            and self.structure_viewer.current_structure is not None
        ):
            self.structure_viewer.set_structure_preserving_display(structure)
        else:
            self.structure_viewer.set_structure(
                structure,
                scene_key=scene_key,
            )
        is_generated_output = preview_path is not None and (
            self._previewable_input_path() != structure_path
        )
        file_role = (
            "generated output" if is_generated_output else "input structure"
        )
        self.visualizer_status_label.setText(
            f"Previewing {file_role} {structure_path.name} with "
            f"{structure.atom_count} atom(s)."
        )

    def _populate_residue_table(
        self,
        result: SolventShellAnalysisResult | None,
    ) -> None:
        self.residue_table.setRowCount(0)
        if result is None:
            self.residue_status_label.setText(
                "Residue-level solvent types are reported only for PDB inputs."
            )
            return
        if result.input_format != "pdb":
            self.residue_status_label.setText(
                "Residue-level solvent types are not available for XYZ inputs."
            )
            return
        if not result.matched_residue_summaries:
            self.residue_status_label.setText(
                "No complete solvent residues matching the selected reference were detected."
            )
            return
        self.residue_status_label.setText(
            "Residue names below matched the selected solvent geometry."
        )
        self.residue_table.setRowCount(len(result.matched_residue_summaries))
        for row, summary in enumerate(result.matched_residue_summaries):
            self._set_table_item(row, 0, summary.residue_name)
            self._set_table_item(row, 1, str(summary.molecule_count))
            self._set_table_item(row, 2, summary.residue_numbers_text)
            self._set_table_item(row, 3, str(summary.atom_count))
            self._set_table_item(row, 4, summary.element_counts_text)

    def _populate_mismatch_table(
        self,
        result: SolventShellAnalysisResult | None,
    ) -> None:
        self.mismatch_table.setRowCount(0)
        if result is None:
            self.mismatch_status_label.setText(
                "Incomplete solvent-like candidates are reported after analysis."
            )
            return
        if not result.residue_mismatch_summaries:
            if result.input_format == "pdb":
                self.mismatch_status_label.setText(
                    "No incomplete or mismatched solvent-like PDB residues were preserved."
                )
            else:
                self.mismatch_status_label.setText(
                    "No partial solvent candidates were inferred from the XYZ input."
                )
            return
        if result.input_format == "pdb":
            self.mismatch_status_label.setText(
                "Incomplete or mismatched solvent-like residues were preserved with missing-atom details."
            )
        else:
            self.mismatch_status_label.setText(
                "Partial solvent candidates were inferred heuristically from XYZ atom sets and preserved with missing-atom details."
            )
        self.mismatch_table.setRowCount(len(result.residue_mismatch_summaries))
        for row, summary in enumerate(result.residue_mismatch_summaries):
            self._set_mismatch_table_item(row, 0, summary.residue_name)
            self._set_mismatch_table_item(row, 1, str(summary.residue_number))
            self._set_mismatch_table_item(
                row,
                2,
                str(summary.observed_atom_count),
            )
            self._set_mismatch_table_item(
                row,
                3,
                summary.matched_atom_ratio_text,
            )
            self._set_mismatch_table_item(
                row,
                4,
                summary.missing_atom_names_text,
            )
            self._set_mismatch_table_item(
                row,
                5,
                summary.extra_atom_names_text,
            )
            self._set_mismatch_table_item(
                row,
                6,
                summary.mismatch_reason,
            )

    def _set_table_item(self, row: int, column: int, text: str) -> None:
        self.residue_table.setItem(row, column, QTableWidgetItem(text))

    def _set_mismatch_table_item(
        self,
        row: int,
        column: int,
        text: str,
    ) -> None:
        self.mismatch_table.setItem(row, column, QTableWidgetItem(text))

    def _combined_summary_text(self) -> str:
        sections: list[str] = []
        if self._analysis_result is not None:
            sections.append(self._analysis_result.summary_text())
        if self._build_result_text:
            sections.extend(
                [
                    "",
                    "Generated solvent shell output:",
                    self._build_result_text,
                ]
            )
        return "\n".join(sections)

    def _populate_director_atom_choices(self) -> None:
        self.director_atom_combo.blockSignals(True)
        self.director_atom_combo.clear()
        preset = self._selected_reference()
        if preset is None:
            self.director_atom_combo.blockSignals(False)
            return
        try:
            atom_names = reference_atom_choices(
                preset.name,
                reference_library_dir=self.reference_library_dir,
            )
            suggested_name = default_director_atom_name(
                preset.name,
                reference_library_dir=self.reference_library_dir,
            )
        except Exception:
            self.director_atom_combo.blockSignals(False)
            return
        for atom_name in atom_names:
            self.director_atom_combo.addItem(atom_name, atom_name)
        if suggested_name is not None:
            suggested_index = self.director_atom_combo.findData(suggested_name)
            if suggested_index >= 0:
                self.director_atom_combo.setCurrentIndex(suggested_index)
        elif atom_names:
            self.director_atom_combo.setCurrentIndex(0)
        self.director_atom_combo.blockSignals(False)

    def _suggested_output_path(self) -> str:
        input_text = self.input_path_edit.text().strip()
        preset = self._selected_reference()
        if not input_text:
            base_dir = self._browse_start_path
            reference_suffix = (
                preset.name.casefold() if preset is not None else "solvent"
            )
            return str(
                (base_dir / f"solvent_shell_builder_{reference_suffix}.pdb")
                .expanduser()
                .resolve()
            )
        input_path = Path(input_text).expanduser().resolve()
        reference_suffix = (
            preset.name.casefold() if preset is not None else "solvent"
        )
        return str(
            input_path.with_name(
                f"{input_path.stem}__solvated_{reference_suffix}.pdb"
            )
        )

    def _update_suggested_output_path(self, *, force: bool = False) -> None:
        suggested_path = self._suggested_output_path()
        current_text = self.output_path_edit.text().strip()
        if (
            force
            or not current_text
            or current_text == self._last_suggested_output_path
        ):
            self.output_path_edit.setText(suggested_path)
        self._last_suggested_output_path = suggested_path

    def _browse_output_file(self) -> None:
        start_path = self.output_path_edit.text().strip() or str(
            self._browse_start_path
        )
        selected_path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "Choose Solvated Output PDB",
            start_path,
            "PDB files (*.pdb);;All files (*)",
        )
        if not selected_path:
            return
        resolved_path = Path(selected_path).expanduser().resolve()
        if resolved_path.suffix.lower() != ".pdb":
            resolved_path = resolved_path.with_suffix(".pdb")
        self._browse_start_path = resolved_path.parent
        self.output_path_edit.setText(str(resolved_path))

    def _populate_solute_cutoff_table(
        self,
        result: SolventShellAnalysisResult | None,
    ) -> None:
        self._updating_solute_table = True
        previous_values = {
            element: spin.value()
            for element, spin in self._solute_cutoff_spins.items()
        }
        previous_coordination_targets = {
            element: spin.value()
            for element, spin in self._coordination_target_spins.items()
        }
        previous_center_states = {
            element: item.checkState() == Qt.CheckState.Checked
            for element, item in self._coordination_center_items.items()
        }
        self._solute_cutoff_spins = {}
        self._coordination_center_items = {}
        self._coordination_target_spins = {}
        self.solute_cutoff_table.setRowCount(0)
        if result is None:
            self.solute_cutoff_status_label.setText(
                "Analyze the input structure to populate the recognized "
                "solute atom types and their solvent-placement distances."
            )
            self._updating_solute_table = False
            return
        if not result.solute_element_counts:
            if result.partial_solvent_molecule_count > 0:
                self.solute_cutoff_status_label.setText(
                    "Partial solvent anchors were found. No additional "
                    "solute atom types remain after excluding those solvent "
                    "candidates, so coordination-center selection is not "
                    "needed unless you want to extend this workflow later."
                )
            else:
                self.solute_cutoff_status_label.setText(
                    "No remaining solute atom types were recognized after "
                    "excluding solvent-like atoms."
                )
            self._updating_solute_table = False
            return
        if result.partial_solvent_molecule_count > 0:
            self.solute_cutoff_status_label.setText(
                "Mark any solute elements that should act as coordinating "
                "centers, set their average target coordination numbers, and "
                "review the director-atom distance cutoffs if you want this "
                "partial-solvent build to add new molecules beyond the "
                "reconstructed anchors."
            )
        else:
            self.solute_cutoff_status_label.setText(
                "Choose which solute elements should coordinate the solvent, "
                "set their average target coordination numbers, and review "
                "the director-atom distance cutoffs before building a new "
                "solvent shell from scratch."
            )
        self.solute_cutoff_table.setRowCount(len(result.solute_element_counts))
        for row, (element, atom_count) in enumerate(
            sorted(result.solute_element_counts.items())
        ):
            self.solute_cutoff_table.setItem(
                row,
                0,
                QTableWidgetItem(str(element)),
            )
            self.solute_cutoff_table.setItem(
                row,
                1,
                QTableWidgetItem(str(atom_count)),
            )
            center_item = QTableWidgetItem("")
            center_item.setFlags(
                Qt.ItemFlag.ItemIsEnabled
                | Qt.ItemFlag.ItemIsSelectable
                | Qt.ItemFlag.ItemIsUserCheckable
            )
            center_item.setCheckState(
                Qt.CheckState.Checked
                if previous_center_states.get(str(element), False)
                else Qt.CheckState.Unchecked
            )
            self.solute_cutoff_table.setItem(row, 2, center_item)
            self._coordination_center_items[str(element)] = center_item

            coordination_spin = QDoubleSpinBox(self.solute_cutoff_table)
            coordination_spin.setDecimals(2)
            coordination_spin.setRange(0.0, 12.0)
            coordination_spin.setSingleStep(0.25)
            coordination_spin.setValue(
                previous_coordination_targets.get(str(element), 0.0)
            )
            coordination_spin.valueChanged.connect(
                self._handle_coordination_settings_changed
            )
            self.solute_cutoff_table.setCellWidget(row, 3, coordination_spin)
            self._coordination_target_spins[str(element)] = coordination_spin

            spin = QDoubleSpinBox(self.solute_cutoff_table)
            spin.setDecimals(3)
            spin.setRange(0.0, 20.0)
            spin.setSingleStep(0.1)
            spin.setSuffix(" A")
            spin.setValue(
                previous_values.get(
                    str(element),
                    self._DEFAULT_SOLUTE_DISTANCE_CUTOFF_A,
                )
            )
            spin.valueChanged.connect(
                self._handle_coordination_settings_changed
            )
            self.solute_cutoff_table.setCellWidget(row, 4, spin)
            self._solute_cutoff_spins[str(element)] = spin
        self._updating_solute_table = False

    def _set_build_status_text(self, headline: str, details: str) -> None:
        self.build_status_label.setText(f"{headline}\n{details}")

    def _update_build_panel_state(self) -> None:
        result = self._analysis_result
        can_choose_reference = bool(self.director_atom_combo.count())
        self.director_atom_combo.setEnabled(can_choose_reference)
        self.minimum_solvent_separation_spin.setEnabled(
            bool(self._available_references)
        )
        self.output_path_edit.setEnabled(bool(self._available_references))
        self.browse_output_button.setEnabled(bool(self._available_references))
        self.solute_cutoff_table.setEnabled(result is not None)
        if result is None:
            self.build_output_button.setEnabled(False)
            self._set_build_status_text(
                "Solvation-shell build is waiting for analysis.",
                "Analyze the input structure first so the beta tool can "
                "identify the solvent status, populate the solute cutoffs, "
                "and determine whether a build is needed.",
            )
            return
        selected_centers = self._selected_coordination_center_elements()
        selected_coordination_targets = (
            self._selected_target_average_coordination_numbers()
        )
        solute_cutoffs = self._collect_solute_distance_cutoffs()
        selected_centers_missing_cutoffs = [
            element
            for element in selected_centers
            if solute_cutoffs.get(element, 0.0) <= 0.0
        ]
        if (
            result.complete_solvent_molecule_count > 0
            and result.partial_solvent_molecule_count == 0
        ):
            self.build_output_button.setEnabled(False)
            self._set_build_status_text(
                "Solvation-shell build is disabled for this input.",
                "The analyzed structure already contains complete solvent "
                "molecules and does not expose partial solvent candidates "
                "to rebuild.",
            )
            return
        if result.partial_solvent_molecule_count == 0 and not selected_centers:
            self.build_output_button.setEnabled(False)
            self._set_build_status_text(
                "Solvation-shell build needs coordination targets.",
                "Select at least one solute element as a coordinating "
                "center and provide its average coordination number before "
                "building a shell from scratch.",
            )
            return
        if (
            result.partial_solvent_molecule_count == 0
            and not selected_coordination_targets
        ):
            self.build_output_button.setEnabled(False)
            self._set_build_status_text(
                "Solvation-shell build needs coordination targets.",
                "Set a positive average coordination number for at least "
                "one selected coordinating center element.",
            )
            return
        if (
            result.partial_solvent_molecule_count == 0
            and selected_centers_missing_cutoffs
        ):
            self.build_output_button.setEnabled(False)
            self._set_build_status_text(
                "Solvation-shell build needs coordination cutoffs.",
                "Each selected coordinating center element needs a positive "
                "director-distance cutoff before the shell can be built.",
            )
            return
        if (
            result.partial_solvent_molecule_count == 0
            and not result.solute_element_counts
        ):
            self.build_output_button.setEnabled(False)
            self._set_build_status_text(
                "Solvation-shell build cannot start yet.",
                "No partial solvent anchors or remaining solute atom types "
                "were recognized for solvent placement.",
            )
            return
        self.build_output_button.setEnabled(True)
        if result.partial_solvent_molecule_count > 0:
            if selected_coordination_targets:
                self._set_build_status_text(
                    "Solvation-shell build is ready.",
                    "The build will complete the partial solvent anchors and "
                    "then add more solvent molecules until the selected "
                    "average coordination targets are met or no valid "
                    "placements remain.",
                )
                return
            self._set_build_status_text(
                "Solvation-shell build is ready.",
                "The selected director atom will be used to complete the "
                "partial solvent candidates that were detected in the input.",
            )
            return
        self._set_build_status_text(
            "Solvation-shell build is ready.",
            "Review the per-solute director distances and build a solvated "
            "output PDB for this no-solvent cluster.",
        )

    def _collect_solute_distance_cutoffs(self) -> dict[str, float]:
        return {
            element: float(spin.value())
            for element, spin in sorted(self._solute_cutoff_spins.items())
        }

    def _selected_coordination_center_elements(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                element
                for element, item in self._coordination_center_items.items()
                if item.checkState() == Qt.CheckState.Checked
            )
        )

    def _selected_target_average_coordination_numbers(
        self,
    ) -> dict[str, float]:
        selected_elements = set(self._selected_coordination_center_elements())
        return {
            element: float(spin.value())
            for element, spin in sorted(
                self._coordination_target_spins.items()
            )
            if element in selected_elements and float(spin.value()) > 0.0
        }

    def _handle_solute_table_item_changed(
        self,
        item: QTableWidgetItem,
    ) -> None:
        if self._updating_solute_table:
            return
        if item.column() != 2:
            return
        self._update_build_panel_state()

    def _handle_coordination_settings_changed(self) -> None:
        if self._updating_solute_table:
            return
        self._update_build_panel_state()

    def _build_solvated_output(self) -> None:
        preset = self._selected_reference()
        result = self._analysis_result
        if preset is None:
            QMessageBox.information(
                self,
                "No reference selected",
                "Choose a solvent reference molecule before building an output PDB.",
            )
            return
        if result is None:
            QMessageBox.information(
                self,
                "No analysis available",
                "Analyze the input structure before building a solvated output PDB.",
            )
            return
        output_text = self.output_path_edit.text().strip()
        if not output_text:
            QMessageBox.information(
                self,
                "No output path selected",
                "Choose where the solvated output PDB should be written.",
            )
            return
        director_atom_name = str(self.director_atom_combo.currentData() or "")
        if not director_atom_name:
            QMessageBox.information(
                self,
                "No director atom selected",
                "Choose the solvent reference atom that should point toward the solute cluster.",
            )
            return
        solute_cutoffs = self._collect_solute_distance_cutoffs()
        coordinating_center_elements = (
            self._selected_coordination_center_elements()
        )
        target_average_coordination_numbers = (
            self._selected_target_average_coordination_numbers()
        )
        output_path = Path(output_text).expanduser().resolve()
        if output_path.suffix.lower() != ".pdb":
            output_path = output_path.with_suffix(".pdb")
            self.output_path_edit.setText(str(output_path))
        try:
            build_result = build_solvent_shell_output(
                result.input_path,
                preset.name,
                output_path=output_path,
                director_atom_name=director_atom_name,
                minimum_solvent_atom_separation_a=float(
                    self.minimum_solvent_separation_spin.value()
                ),
                solute_distance_cutoffs_a=solute_cutoffs,
                coordinating_center_elements=coordinating_center_elements,
                target_average_coordination_numbers=(
                    target_average_coordination_numbers
                ),
                reference_library_dir=self.reference_library_dir,
                reference_match_tolerance_a=float(
                    self.reference_match_tolerance_spin.value()
                ),
                analysis_result=result,
            )
        except Exception as exc:
            self._build_result_text = None
            self.summary_box.setPlainText(self._combined_summary_text())
            self._set_build_status_text(
                "Solvation-shell build failed.",
                str(exc),
            )
            self.statusBar().showMessage(
                "Solvation-shell build failed",
                5000,
            )
            QMessageBox.warning(
                self,
                "Solvation-shell build failed",
                str(exc),
            )
            return
        self._build_result_text = build_result.summary_text()
        self.summary_box.setPlainText(self._combined_summary_text())
        self._refresh_structure_preview(
            preview_path=build_result.output_path,
            preserve_display=True,
        )
        self._set_build_status_text(
            "Solvated output PDB generated.",
            (
                f"Wrote {build_result.output_path.name} with "
                f"{build_result.solvent_molecules_added} solvent molecule(s) "
                "added."
            ),
        )
        self.statusBar().showMessage(
            f"Built solvated output: {build_result.output_path.name}",
            5000,
        )


def launch_solvent_shell_builder_ui(
    *,
    initial_project_dir: str | Path | None = None,
    initial_input_path: str | Path | None = None,
    reference_library_dir: str | Path | None = None,
) -> SolventShellBuilderMainWindow:
    app = QApplication.instance()
    if app is None:
        prepare_saxshell_application_identity()
        app = QApplication(sys.argv)
    configure_saxshell_application(app)
    window = SolventShellBuilderMainWindow(
        initial_project_dir=(
            None
            if initial_project_dir is None
            else Path(initial_project_dir).expanduser().resolve()
        ),
        initial_input_path=(
            None
            if initial_input_path is None
            else Path(initial_input_path).expanduser().resolve()
        ),
        reference_library_dir=(
            None
            if reference_library_dir is None
            else Path(reference_library_dir).expanduser().resolve()
        ),
    )
    window.show()
    window.raise_()
    return window


__all__ = [
    "SolventShellBuilderMainWindow",
    "launch_solvent_shell_builder_ui",
]
