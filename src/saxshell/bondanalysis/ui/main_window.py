from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from saxshell.bondanalysis import (
    AngleTripletDefinition,
    BondAnalysisBatchResult,
    BondAnalysisPreset,
    BondAnalysisWorkflow,
    BondPairDefinition,
    load_presets,
    ordered_preset_names,
    save_custom_preset,
    suggest_bondanalysis_output_dir,
)
from saxshell.bondanalysis.results import (
    BondAnalysisPlotRequest,
    BondAnalysisResultIndex,
    BondAnalysisResultLeaf,
    build_plot_request,
    load_result_index,
)
from saxshell.bondanalysis.ui.plot_window import BondAnalysisPlotWindow
from saxshell.saxs.ui.branding import (
    configure_saxshell_application,
    load_saxshell_icon,
    prepare_saxshell_application_identity,
)

_OPEN_WINDOWS: list["BondAnalysisMainWindow"] = []


class BondAnalysisWorker(QObject):
    """Background worker that runs one bond-analysis workflow."""

    log = Signal(str)
    progress = Signal(int, int)
    status = Signal(str)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, workflow: BondAnalysisWorkflow) -> None:
        super().__init__()
        self.workflow = workflow

    @Slot()
    def run(self) -> None:
        try:
            result = self.workflow.run(
                progress_callback=self._emit_progress,
                log_callback=self.log.emit,
            )
            self.finished.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))

    def _emit_progress(
        self,
        processed: int,
        total: int,
        message: str,
    ) -> None:
        self.progress.emit(processed, total)
        self.status.emit(message)


class BondAnalysisMainWindow(QMainWindow):
    """Main Qt window for bond-pair and angle-distribution analysis."""

    def __init__(
        self,
        initial_clusters_dir: str | Path | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._run_thread: QThread | None = None
        self._run_worker: BondAnalysisWorker | None = None
        self._presets: dict[str, BondAnalysisPreset] = {}
        self._results_index: BondAnalysisResultIndex | None = None
        self._plot_windows: list[BondAnalysisPlotWindow] = []
        self._build_ui()
        if initial_clusters_dir is not None:
            self.set_clusters_dir(initial_clusters_dir)
        else:
            self._update_selection_summary()

    def _build_ui(self) -> None:
        self.setWindowTitle("SAXSShell (bondanalysis)")
        self.setWindowIcon(load_saxshell_icon())
        self.resize(1320, 840)

        central = QWidget()
        root = QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setSizes([700, 620])

        root.addWidget(splitter)
        self.setCentralWidget(central)
        self.statusBar().showMessage("Ready")
        self._reload_presets()

    def _build_left_panel(self) -> QWidget:
        left = QWidget()
        layout = QVBoxLayout(left)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        layout.addWidget(self._build_paths_group())
        layout.addWidget(self._build_presets_group())
        layout.addWidget(self._build_cluster_types_group())
        layout.addWidget(self._build_bond_pairs_group())
        layout.addWidget(self._build_angle_triplets_group())
        layout.addStretch(1)

        return self._wrap_scroll_area(left)

    def _build_right_panel(self) -> QWidget:
        right = QWidget()
        layout = QVBoxLayout(right)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        preview_group = QGroupBox("Selection Preview")
        preview_layout = QVBoxLayout(preview_group)
        self.selection_box = QTextEdit()
        self.selection_box.setReadOnly(True)
        self.selection_box.setMinimumHeight(150)
        preview_layout.addWidget(self.selection_box)
        layout.addWidget(preview_group)

        run_group = QGroupBox("Run")
        run_layout = QVBoxLayout(run_group)
        self.run_button = QPushButton(
            "Analyze Bond Pairs and Angle Distributions"
        )
        self.run_button.clicked.connect(self._start_run)
        run_layout.addWidget(self.run_button)

        self.progress_label = QLabel("Progress: idle")
        run_layout.addWidget(self.progress_label)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%v / %m files")
        run_layout.addWidget(self.progress_bar)

        self.legacy_label = QLabel(
            "Legacy note: displacement analysis is deprecated and is not "
            "part of this interface until that workflow is updated."
        )
        self.legacy_label.setWordWrap(True)
        self.legacy_label.setFrameShape(QFrame.Shape.StyledPanel)
        run_layout.addWidget(self.legacy_label)
        layout.addWidget(run_group)

        browser_log_panel = QWidget()
        browser_log_layout = QVBoxLayout(browser_log_panel)
        browser_log_layout.setContentsMargins(0, 0, 0, 0)
        browser_log_layout.setSpacing(12)
        browser_log_layout.addWidget(self._build_results_browser_group())

        log_group = QGroupBox("Run Log")
        log_layout = QVBoxLayout(log_group)
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMinimumHeight(180)
        log_layout.addWidget(self.log_box)
        browser_log_layout.addWidget(log_group)

        layout.addWidget(browser_log_panel, stretch=1)

        layout.addStretch(1)
        return self._wrap_scroll_area(right)

    def _build_results_browser_group(self) -> QGroupBox:
        group = QGroupBox("Computed Distributions")
        layout = QVBoxLayout(group)

        controls = QHBoxLayout()
        self.refresh_results_button = QPushButton("Refresh Results")
        self.refresh_results_button.clicked.connect(self._refresh_results_tree)
        controls.addWidget(self.refresh_results_button)

        self.open_selected_window_button = QPushButton(
            "Open Selected in Window"
        )
        self.open_selected_window_button.clicked.connect(
            self._open_selected_plot_window
        )
        controls.addWidget(self.open_selected_window_button)
        controls.addStretch(1)
        layout.addLayout(controls)

        self.results_tree = QTreeWidget()
        self.results_tree.setColumnCount(3)
        self.results_tree.setHeaderLabels(["Distribution", "Scope", "Values"])
        self.results_tree.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.results_tree.itemSelectionChanged.connect(
            self._on_results_tree_selection_changed
        )
        self.results_tree.header().setStretchLastSection(False)
        self.results_tree.header().setSectionResizeMode(
            0,
            QHeaderView.ResizeMode.Stretch,
        )
        self.results_tree.header().setSectionResizeMode(
            1,
            QHeaderView.ResizeMode.ResizeToContents,
        )
        self.results_tree.header().setSectionResizeMode(
            2,
            QHeaderView.ResizeMode.ResizeToContents,
        )
        layout.addWidget(self.results_tree, stretch=1)

        self.results_hint_label = QLabel(
            "Select one computed bond pair or angle distribution and use "
            "'Open Selected in Window' to view it. Select multiple leaves "
            "of the same type across different cluster types to overlay "
            "them together in a separate window. The 'all' entry opens that "
            "bond pair or angle across all cluster types."
        )
        self.results_hint_label.setWordWrap(True)
        layout.addWidget(self.results_hint_label)

        self.results_status_label = QLabel(
            "Run bondanalysis or refresh an existing output directory to "
            "browse computed distributions."
        )
        self.results_status_label.setWordWrap(True)
        self.results_status_label.setFrameShape(QFrame.Shape.StyledPanel)
        layout.addWidget(self.results_status_label)
        return group

    def _build_paths_group(self) -> QGroupBox:
        group = QGroupBox("Directories")
        layout = QFormLayout(group)

        self.clusters_dir_edit = QLineEdit()
        self.clusters_dir_edit.textChanged.connect(
            self._on_clusters_dir_changed
        )
        layout.addRow(
            "Clusters directory",
            self._make_dir_row(
                self.clusters_dir_edit,
                "Select clusters directory",
            ),
        )

        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.textChanged.connect(
            lambda _text: self._update_selection_summary()
        )
        layout.addRow(
            "Output directory",
            self._make_dir_row(
                self.output_dir_edit,
                "Select bondanalysis output directory",
            ),
        )

        refresh_button = QPushButton("Refresh Cluster Types")
        refresh_button.clicked.connect(self._refresh_cluster_types)
        layout.addRow("", refresh_button)

        load_existing_button = QPushButton("Load Existing Bondanalysis Folder")
        load_existing_button.clicked.connect(self._choose_existing_results_dir)
        layout.addRow("", load_existing_button)
        return group

    def _build_presets_group(self) -> QGroupBox:
        group = QGroupBox("Presets")
        layout = QVBoxLayout(group)

        row = QHBoxLayout()
        self.preset_combo = QComboBox()
        self.preset_combo.currentIndexChanged.connect(
            lambda _index: self._update_selection_summary()
        )
        self.preset_combo.setToolTip(
            "Load a built-in bondanalysis preset or a custom preset saved "
            "from a previous session."
        )
        row.addWidget(self.preset_combo, stretch=1)

        load_button = QPushButton("Load")
        load_button.clicked.connect(self._load_selected_preset)
        row.addWidget(load_button)

        save_button = QPushButton("Save Current As...")
        save_button.clicked.connect(self._save_current_as_preset)
        row.addWidget(save_button)
        layout.addLayout(row)

        self.preset_hint_label = QLabel(
            "Built-in presets: DMSO and DMF. Custom presets are saved for "
            "later sessions."
        )
        self.preset_hint_label.setWordWrap(True)
        layout.addWidget(self.preset_hint_label)
        return group

    def _build_cluster_types_group(self) -> QGroupBox:
        group = QGroupBox("Cluster Types")
        layout = QVBoxLayout(group)

        self.use_checked_cluster_types_box = QCheckBox(
            "Analyze only checked cluster types"
        )
        self.use_checked_cluster_types_box.setChecked(True)
        self.use_checked_cluster_types_box.toggled.connect(
            lambda _checked: self._update_selection_summary()
        )
        layout.addWidget(self.use_checked_cluster_types_box)

        self.cluster_type_list = QListWidget()
        self.cluster_type_list.setSelectionMode(
            QAbstractItemView.SelectionMode.NoSelection
        )
        self.cluster_type_list.itemChanged.connect(
            lambda _item: self._update_selection_summary()
        )
        layout.addWidget(self.cluster_type_list)
        hint_label = QLabel(
            "Tick or clear individual cluster types in the list below."
        )
        hint_label.setWordWrap(True)
        layout.addWidget(hint_label)
        return group

    def _build_bond_pairs_group(self) -> QGroupBox:
        group = QGroupBox("Bond Pairs")
        layout = QVBoxLayout(group)

        controls = QHBoxLayout()
        add_button = QPushButton("Add Bond Pair")
        add_button.clicked.connect(self._add_bond_pair_row)
        remove_button = QPushButton("Remove Selected")
        remove_button.clicked.connect(self._remove_selected_bond_pair_rows)
        controls.addWidget(add_button)
        controls.addWidget(remove_button)
        controls.addStretch(1)
        layout.addLayout(controls)

        self.bond_pair_table = QTableWidget(0, 3)
        self.bond_pair_table.setHorizontalHeaderLabels(
            ["Atom 1", "Atom 2", "Cutoff (A)"]
        )
        self.bond_pair_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.bond_pair_table)
        self._add_bond_pair_row()
        self.bond_pair_table.itemChanged.connect(
            lambda _item: self._update_selection_summary()
        )
        return group

    def _build_angle_triplets_group(self) -> QGroupBox:
        group = QGroupBox("Angle Triplets")
        layout = QVBoxLayout(group)

        controls = QHBoxLayout()
        add_button = QPushButton("Add Angle Triplet")
        add_button.clicked.connect(self._add_angle_triplet_row)
        remove_button = QPushButton("Remove Selected")
        remove_button.clicked.connect(self._remove_selected_angle_triplet_rows)
        controls.addWidget(add_button)
        controls.addWidget(remove_button)
        controls.addStretch(1)
        layout.addLayout(controls)

        self.angle_triplet_table = QTableWidget(0, 5)
        self.angle_triplet_table.setHorizontalHeaderLabels(
            [
                "Vertex",
                "Arm 1",
                "Arm 2",
                "Vertex-Arm 1 Cutoff (A)",
                "Vertex-Arm 2 Cutoff (A)",
            ]
        )
        self.angle_triplet_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.angle_triplet_table)
        self._add_angle_triplet_row()
        self.angle_triplet_table.itemChanged.connect(
            lambda _item: self._update_selection_summary()
        )
        return group

    def _make_dir_row(
        self,
        line_edit: QLineEdit,
        title: str,
    ) -> QWidget:
        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)

        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(
            lambda _checked=False: self._choose_dir(line_edit, title)
        )

        row_layout.addWidget(line_edit)
        row_layout.addWidget(browse_button)
        return row_widget

    @staticmethod
    def _wrap_scroll_area(widget: QWidget) -> QScrollArea:
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setFrameShape(QFrame.Shape.NoFrame)
        area.setWidget(widget)
        return area

    def _choose_dir(self, line_edit: QLineEdit, title: str) -> None:
        path = QFileDialog.getExistingDirectory(self, title)
        if path:
            line_edit.setText(path)

    def _choose_existing_results_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            "Select existing bondanalysis output directory",
        )
        if not path:
            return
        try:
            self.load_existing_results_dir(path)
        except Exception as exc:
            QMessageBox.warning(self, "Bond Analysis", str(exc))

    def set_clusters_dir(self, clusters_dir: str | Path) -> None:
        self.clusters_dir_edit.setText(str(clusters_dir))

    def load_existing_results_dir(self, output_dir: str | Path) -> None:
        result_index = load_result_index(output_dir)
        self._results_index = result_index
        self.output_dir_edit.blockSignals(True)
        self.output_dir_edit.setText(str(result_index.output_dir))
        self.output_dir_edit.blockSignals(False)
        self.clusters_dir_edit.blockSignals(True)
        self.clusters_dir_edit.setText(str(result_index.clusters_dir))
        self.clusters_dir_edit.blockSignals(False)
        self._set_bond_pair_rows(result_index.bond_pairs)
        self._set_angle_triplet_rows(result_index.angle_triplets)
        self._restore_cluster_type_list(result_index)
        self._refresh_results_tree()
        self._append_log(
            "Loaded existing bondanalysis folder: "
            f"{result_index.output_dir}"
        )
        self._append_log(
            f"Results index file: {result_index.results_index_path}"
        )
        self.statusBar().showMessage(
            f"Loaded existing bondanalysis results: {result_index.output_dir}"
        )
        self._update_selection_summary()

    def _restore_cluster_type_list(
        self,
        result_index: BondAnalysisResultIndex,
    ) -> None:
        cluster_type_names = list(result_index.cluster_type_names)
        selected_names = set(result_index.selected_cluster_types)
        use_checked_filter = bool(selected_names) and (
            len(selected_names) != len(cluster_type_names)
        )

        self.cluster_type_list.blockSignals(True)
        self.cluster_type_list.clear()
        for cluster_type in cluster_type_names:
            item = QListWidgetItem(cluster_type)
            item.setFlags(
                (item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                & ~Qt.ItemFlag.ItemIsSelectable
            )
            check_state = (
                Qt.CheckState.Checked
                if not selected_names or cluster_type in selected_names
                else Qt.CheckState.Unchecked
            )
            item.setCheckState(check_state)
            self.cluster_type_list.addItem(item)
        self.cluster_type_list.blockSignals(False)
        self.use_checked_cluster_types_box.blockSignals(True)
        self.use_checked_cluster_types_box.setChecked(use_checked_filter)
        self.use_checked_cluster_types_box.blockSignals(False)

    def _on_clusters_dir_changed(self, _text: str) -> None:
        self._refresh_cluster_types()

    def _refresh_cluster_types(self) -> None:
        clusters_dir = self._clusters_dir_path()
        previous_states = self._cluster_type_check_states()
        self.cluster_type_list.blockSignals(True)
        self.cluster_type_list.clear()
        if clusters_dir is None:
            self.cluster_type_list.blockSignals(False)
            self._update_selection_summary()
            return

        try:
            workflow = BondAnalysisWorkflow(clusters_dir)
            summary = workflow.inspect()
        except Exception as exc:
            self.cluster_type_list.blockSignals(False)
            self._append_log(f"Unable to inspect clusters directory: {exc}")
            self._update_selection_summary(error_text=str(exc))
            return

        for cluster_type in summary["cluster_types"]:
            item = QListWidgetItem(cluster_type)
            item.setFlags(
                (item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                & ~Qt.ItemFlag.ItemIsSelectable
            )
            item.setCheckState(
                previous_states.get(cluster_type, Qt.CheckState.Checked)
            )
            self.cluster_type_list.addItem(item)
        self.cluster_type_list.blockSignals(False)

        current_output_dir = self.output_dir_edit.text().strip()
        suggested_output_dir = str(
            suggest_bondanalysis_output_dir(clusters_dir)
        )
        if not current_output_dir:
            self.output_dir_edit.setText(suggested_output_dir)

        self._update_selection_summary()

    def _clusters_dir_path(self) -> Path | None:
        text = self.clusters_dir_edit.text().strip()
        return Path(text) if text else None

    def _output_dir_path(self) -> Path | None:
        text = self.output_dir_edit.text().strip()
        return Path(text) if text else None

    def _selected_cluster_types(self) -> list[str] | None:
        if not self.use_checked_cluster_types_box.isChecked():
            return None
        return self._checked_cluster_types()

    def _checked_cluster_types(self) -> list[str]:
        return [
            self.cluster_type_list.item(index).text()
            for index in range(self.cluster_type_list.count())
            if self.cluster_type_list.item(index).checkState()
            == Qt.CheckState.Checked
        ]

    def _cluster_type_check_states(self) -> dict[str, Qt.CheckState]:
        return {
            self.cluster_type_list.item(index)
            .text(): self.cluster_type_list.item(index)
            .checkState()
            for index in range(self.cluster_type_list.count())
        }

    def _selected_preset_name(self) -> str | None:
        return self.preset_combo.currentData()

    def _reload_presets(self, *, selected_name: str | None = None) -> None:
        previous_name = selected_name or self._selected_preset_name()
        self._presets = load_presets()
        self.preset_combo.blockSignals(True)
        self.preset_combo.clear()
        self.preset_combo.addItem("Select preset...", None)
        selected_index = 0
        for index, name in enumerate(
            ordered_preset_names(self._presets),
            start=1,
        ):
            preset = self._presets[name]
            label = name
            if preset.builtin:
                label = f"{name} (Built-in)"
            self.preset_combo.addItem(label, name)
            if name == previous_name:
                selected_index = index
        self.preset_combo.setCurrentIndex(selected_index)
        self.preset_combo.blockSignals(False)
        self._update_selection_summary()

    def load_preset(self, preset_name: str) -> None:
        preset = self._presets.get(preset_name)
        if preset is None:
            raise ValueError(f"Unknown preset: {preset_name}")
        self._apply_preset(preset)
        self._select_preset_name(preset_name)
        self._append_log(f"Loaded preset: {preset_name}")

    def save_current_preset(self, preset_name: str) -> None:
        name = preset_name.strip()
        if not name:
            raise ValueError("Preset names cannot be empty.")
        preset = BondAnalysisPreset(
            name=name,
            bond_pairs=tuple(self._read_bond_pairs()),
            angle_triplets=tuple(self._read_angle_triplets()),
        )
        save_custom_preset(preset)
        self._reload_presets(selected_name=name)
        self._append_log(f"Saved preset: {name}")

    def _select_preset_name(self, preset_name: str) -> None:
        for index in range(self.preset_combo.count()):
            if self.preset_combo.itemData(index) == preset_name:
                self.preset_combo.setCurrentIndex(index)
                return

    def _load_selected_preset(self) -> None:
        preset_name = self._selected_preset_name()
        if preset_name is None:
            QMessageBox.information(
                self,
                "Bond Analysis Presets",
                "Select a preset to load.",
            )
            return
        try:
            self.load_preset(preset_name)
        except Exception as exc:
            QMessageBox.warning(self, "Bond Analysis Presets", str(exc))

    def _save_current_as_preset(self) -> None:
        try:
            self._read_bond_pairs()
            self._read_angle_triplets()
        except Exception as exc:
            QMessageBox.warning(self, "Bond Analysis Presets", str(exc))
            return

        suggested_name = self._selected_preset_name() or ""
        name, accepted = QInputDialog.getText(
            self,
            "Save Bondanalysis Preset",
            "Preset name:",
            text=suggested_name,
        )
        if not accepted:
            return
        name = name.strip()
        if not name:
            return

        if name in self._presets:
            response = QMessageBox.question(
                self,
                "Overwrite Preset?",
                f"A preset named '{name}' already exists. Overwrite it?",
            )
            if response != QMessageBox.StandardButton.Yes:
                return

        try:
            self.save_current_preset(name)
        except Exception as exc:
            QMessageBox.warning(self, "Bond Analysis Presets", str(exc))

    def _apply_preset(self, preset: BondAnalysisPreset) -> None:
        self._set_bond_pair_rows(preset.bond_pairs)
        self._set_angle_triplet_rows(preset.angle_triplets)
        self._update_selection_summary()

    def _set_bond_pair_rows(
        self,
        definitions: tuple[BondPairDefinition, ...],
    ) -> None:
        self.bond_pair_table.blockSignals(True)
        self.bond_pair_table.setRowCount(0)
        if not definitions:
            self._add_empty_bond_pair_row(blocked=True)
        else:
            for definition in definitions:
                row = self.bond_pair_table.rowCount()
                self.bond_pair_table.insertRow(row)
                self.bond_pair_table.setItem(
                    row,
                    0,
                    QTableWidgetItem(definition.atom1),
                )
                self.bond_pair_table.setItem(
                    row,
                    1,
                    QTableWidgetItem(definition.atom2),
                )
                self.bond_pair_table.setItem(
                    row,
                    2,
                    QTableWidgetItem(f"{definition.cutoff_angstrom:g}"),
                )
        self.bond_pair_table.blockSignals(False)

    def _set_angle_triplet_rows(
        self,
        definitions: tuple[AngleTripletDefinition, ...],
    ) -> None:
        self.angle_triplet_table.blockSignals(True)
        self.angle_triplet_table.setRowCount(0)
        if not definitions:
            self._add_empty_angle_triplet_row(blocked=True)
        else:
            for definition in definitions:
                row = self.angle_triplet_table.rowCount()
                self.angle_triplet_table.insertRow(row)
                self.angle_triplet_table.setItem(
                    row,
                    0,
                    QTableWidgetItem(definition.vertex),
                )
                self.angle_triplet_table.setItem(
                    row,
                    1,
                    QTableWidgetItem(definition.arm1),
                )
                self.angle_triplet_table.setItem(
                    row,
                    2,
                    QTableWidgetItem(definition.arm2),
                )
                self.angle_triplet_table.setItem(
                    row,
                    3,
                    QTableWidgetItem(f"{definition.cutoff1_angstrom:g}"),
                )
                self.angle_triplet_table.setItem(
                    row,
                    4,
                    QTableWidgetItem(f"{definition.cutoff2_angstrom:g}"),
                )
        self.angle_triplet_table.blockSignals(False)

    def _add_empty_angle_triplet_row(self, *, blocked: bool = False) -> None:
        previous_blocked = self.angle_triplet_table.blockSignals(blocked)
        row = self.angle_triplet_table.rowCount()
        self.angle_triplet_table.insertRow(row)
        for column in range(self.angle_triplet_table.columnCount()):
            self.angle_triplet_table.setItem(
                row,
                column,
                QTableWidgetItem(""),
            )
        self.angle_triplet_table.blockSignals(previous_blocked)

    def _add_empty_bond_pair_row(self, *, blocked: bool = False) -> None:
        previous_blocked = self.bond_pair_table.blockSignals(blocked)
        row = self.bond_pair_table.rowCount()
        self.bond_pair_table.insertRow(row)
        for column in range(self.bond_pair_table.columnCount()):
            self.bond_pair_table.setItem(row, column, QTableWidgetItem(""))
        self.bond_pair_table.blockSignals(previous_blocked)

    def _add_bond_pair_row(self) -> None:
        self._add_empty_bond_pair_row(blocked=True)
        if hasattr(self, "selection_box"):
            self._update_selection_summary()

    def _remove_selected_bond_pair_rows(self) -> None:
        selected_rows = sorted(
            {index.row() for index in self.bond_pair_table.selectedIndexes()},
            reverse=True,
        )
        for row in selected_rows:
            self.bond_pair_table.removeRow(row)
        self._update_selection_summary()

    def _add_angle_triplet_row(self) -> None:
        self._add_empty_angle_triplet_row(blocked=True)
        if hasattr(self, "selection_box"):
            self._update_selection_summary()

    def _remove_selected_angle_triplet_rows(self) -> None:
        selected_rows = sorted(
            {
                index.row()
                for index in self.angle_triplet_table.selectedIndexes()
            },
            reverse=True,
        )
        for row in selected_rows:
            self.angle_triplet_table.removeRow(row)
        self._update_selection_summary()

    def _read_bond_pairs(self) -> list[BondPairDefinition]:
        definitions: list[BondPairDefinition] = []
        for row in range(self.bond_pair_table.rowCount()):
            atom1 = self._table_text(self.bond_pair_table, row, 0)
            atom2 = self._table_text(self.bond_pair_table, row, 1)
            cutoff_text = self._table_text(self.bond_pair_table, row, 2)
            if not atom1 and not atom2 and not cutoff_text:
                continue
            if not atom1 or not atom2 or not cutoff_text:
                raise ValueError(
                    "Every populated bond-pair row needs atom 1, atom 2, "
                    "and a cutoff."
                )
            definitions.append(
                BondPairDefinition(atom1, atom2, float(cutoff_text))
            )
        return definitions

    def _read_angle_triplets(self) -> list[AngleTripletDefinition]:
        definitions: list[AngleTripletDefinition] = []
        for row in range(self.angle_triplet_table.rowCount()):
            vertex = self._table_text(self.angle_triplet_table, row, 0)
            arm1 = self._table_text(self.angle_triplet_table, row, 1)
            arm2 = self._table_text(self.angle_triplet_table, row, 2)
            cutoff1_text = self._table_text(self.angle_triplet_table, row, 3)
            cutoff2_text = self._table_text(self.angle_triplet_table, row, 4)
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
        return definitions

    @staticmethod
    def _table_text(table: QTableWidget, row: int, column: int) -> str:
        item = table.item(row, column)
        return item.text().strip() if item is not None else ""

    def _update_selection_summary(
        self,
        *,
        error_text: str | None = None,
    ) -> None:
        clusters_dir = self._clusters_dir_path()
        output_dir = self._output_dir_path()
        selected_cluster_types = self._selected_cluster_types()

        lines: list[str] = []
        if error_text is not None:
            lines.append(f"Inspection error: {error_text}")
        if clusters_dir is None:
            lines.append(
                "Select a stoichiometry-level clusters directory to preview "
                "the bond-analysis run."
            )
            self.selection_box.setPlainText("\n".join(lines))
            return

        lines.append(f"Clusters directory: {clusters_dir}")
        if output_dir is not None:
            lines.append(f"Output directory: {output_dir}")

        cluster_types = [
            self.cluster_type_list.item(index).text()
            for index in range(self.cluster_type_list.count())
        ]
        lines.append(f"Cluster types detected: {len(cluster_types)}")
        checked_cluster_types = self._checked_cluster_types()
        lines.append(f"Checked cluster types: {len(checked_cluster_types)}")
        if self.use_checked_cluster_types_box.isChecked():
            if selected_cluster_types:
                lines.append(
                    "Analyzing checked cluster types: "
                    + ", ".join(selected_cluster_types)
                )
            else:
                lines.append(
                    "Analyzing checked cluster types: none checked yet"
                )
        elif cluster_types:
            lines.append("Analyzing cluster types: all detected types")

        preset_name = self._selected_preset_name()
        if preset_name is not None:
            lines.append(f"Selected preset: {preset_name}")

        try:
            bond_pairs = self._read_bond_pairs()
            lines.append(f"Bond pairs configured: {len(bond_pairs)}")
        except Exception as exc:
            lines.append(f"Bond pairs configured: invalid ({exc})")

        try:
            angle_triplets = self._read_angle_triplets()
            lines.append(f"Angle triplets configured: {len(angle_triplets)}")
        except Exception as exc:
            lines.append(f"Angle triplets configured: invalid ({exc})")

        lines.append(
            "Displacement analysis: deprecated and not part of this window"
        )
        self.selection_box.setPlainText("\n".join(lines))

    def _refresh_results_tree(self) -> None:
        output_dir = self._output_dir_path()
        if output_dir is None:
            self._clear_results_tree(
                "Choose a bondanalysis output directory first."
            )
            return

        try:
            self._results_index = load_result_index(output_dir)
        except Exception as exc:
            self._results_index = None
            self._clear_results_tree(str(exc))
            return

        self.results_tree.clear()
        self._populate_results_category(
            "Bond Pairs",
            self._results_index.bond_groups,
        )
        self._populate_results_category(
            "Bond Angles",
            self._results_index.angle_groups,
        )
        self.results_tree.expandAll()
        self.results_status_label.setText(
            "Browse computed bond-pair and angle distributions from the "
            f"current output directory: {self._results_index.output_dir}"
        )

    def _populate_results_category(
        self,
        title: str,
        groups: tuple[object, ...],
    ) -> None:
        if not groups:
            return
        category_item = QTreeWidgetItem([title, "", ""])
        category_item.setFlags(
            category_item.flags() & ~Qt.ItemFlag.ItemIsSelectable
        )
        self.results_tree.addTopLevelItem(category_item)
        for group in groups:
            group_item = QTreeWidgetItem([group.display_label, "", ""])
            group_item.setFlags(
                group_item.flags() & ~Qt.ItemFlag.ItemIsSelectable
            )
            category_item.addChild(group_item)
            group_item.addChild(self._make_results_leaf_item(group.all_leaf))
            for leaf in group.cluster_leaves:
                group_item.addChild(self._make_results_leaf_item(leaf))

    def _make_results_leaf_item(
        self,
        leaf: BondAnalysisResultLeaf,
    ) -> QTreeWidgetItem:
        scope_label = "all clusters" if leaf.is_all else "cluster type"
        item = QTreeWidgetItem(
            [leaf.scope_name, scope_label, str(leaf.point_count)]
        )
        item.setData(0, Qt.ItemDataRole.UserRole, leaf)
        item.setToolTip(
            0,
            f"{leaf.display_label} • {leaf.scope_name} "
            f"({leaf.point_count} values)",
        )
        return item

    def _selected_result_leaves(self) -> list[BondAnalysisResultLeaf]:
        leaves: list[BondAnalysisResultLeaf] = []
        for item in self.results_tree.selectedItems():
            payload = item.data(0, Qt.ItemDataRole.UserRole)
            if isinstance(payload, BondAnalysisResultLeaf):
                leaves.append(payload)
        return leaves

    def _on_results_tree_selection_changed(self) -> None:
        leaves = self._selected_result_leaves()
        if not leaves:
            self.results_status_label.setText(
                "Select one computed distribution and use 'Open Selected in "
                "Window' to view it, or select multiple leaves of the same "
                "type and open them together as an overlay."
            )
            return

        if len(leaves) == 1:
            leaf = leaves[0]
            self.results_status_label.setText(
                f"Ready to open {leaf.display_label} for {leaf.scope_name} "
                "in a separate plot window."
            )
            return

        try:
            self._validate_multi_leaf_selection(leaves)
        except ValueError as exc:
            self.results_status_label.setText(str(exc))
            return

        cluster_names = ", ".join(leaf.scope_name for leaf in leaves)
        self.results_status_label.setText(
            "Ready to overlay "
            f"{leaves[0].display_label} across: {cluster_names}"
        )

    def _validate_multi_leaf_selection(
        self,
        leaves: list[BondAnalysisResultLeaf],
    ) -> None:
        first_leaf = leaves[0]
        if any(leaf.is_all for leaf in leaves):
            raise ValueError(
                "Select either the 'all' entry or multiple individual "
                "cluster leaves, not both together."
            )
        if any(
            leaf.category != first_leaf.category
            or leaf.display_label != first_leaf.display_label
            for leaf in leaves[1:]
        ):
            raise ValueError(
                "To overlay distributions, select bond pairs or angles of "
                "the same type across different cluster types."
            )

    def _open_selected_plot_window(self) -> None:
        leaves = self._selected_result_leaves()
        if not leaves:
            QMessageBox.information(
                self,
                "Computed Distributions",
                "Select one or more computed bond pair or angle "
                "distributions first.",
            )
            return
        self._open_plot_window_for_leaves(leaves)

    def _open_plot_window_for_leaves(
        self,
        leaves: list[BondAnalysisResultLeaf],
    ) -> None:
        if self._results_index is None:
            QMessageBox.warning(
                self,
                "Computed Distributions",
                "Run bondanalysis or refresh an existing output directory "
                "before opening standalone plot windows.",
            )
            return
        try:
            if len(leaves) > 1:
                self._validate_multi_leaf_selection(leaves)
            plot_request = build_plot_request(self._results_index, leaves)
        except Exception as exc:
            QMessageBox.warning(self, "Computed Distributions", str(exc))
            return
        self._open_plot_window_for_request(plot_request)

    def _open_plot_window_for_request(
        self,
        plot_request: BondAnalysisPlotRequest,
    ) -> None:
        default_output_dir = (
            self._results_index.output_dir
            if self._results_index is not None
            else (self._output_dir_path() or Path.cwd())
        )
        if self._plot_windows:
            window = self._plot_windows[0]
            window.add_plot_request(plot_request)
        else:
            window = BondAnalysisPlotWindow(
                plot_request,
                default_output_dir=default_output_dir,
                parent=self,
            )
            window.destroyed.connect(
                lambda _obj=None, win=window: self._remove_plot_window(win)
            )
            self._plot_windows.append(window)
        window.show()
        window.raise_()
        window.activateWindow()

    def _remove_plot_window(self, window: BondAnalysisPlotWindow) -> None:
        self._plot_windows = [
            existing
            for existing in self._plot_windows
            if existing is not window
        ]

    def _clear_results_tree(self, message: str) -> None:
        self.results_tree.clear()
        self.results_status_label.setText(message)

    def _append_log(self, text: str) -> None:
        current = self.log_box.toPlainText().strip()
        if current:
            self.log_box.append(text)
        else:
            self.log_box.setPlainText(text)

    def _start_run(self) -> None:
        if self._run_thread is not None:
            return

        try:
            clusters_dir = self._clusters_dir_path()
            if clusters_dir is None or not clusters_dir.is_dir():
                raise ValueError(
                    "Choose a valid clusters directory before running."
                )
            output_dir = self._output_dir_path()
            selected_cluster_types = self._selected_cluster_types()
            if (
                self.use_checked_cluster_types_box.isChecked()
                and not selected_cluster_types
            ):
                raise ValueError(
                    "Check at least one cluster type, or turn off the "
                    "checked-cluster-type filter."
                )

            workflow = BondAnalysisWorkflow(
                clusters_dir,
                bond_pairs=self._read_bond_pairs(),
                angle_triplets=self._read_angle_triplets(),
                output_dir=output_dir,
                selected_cluster_types=selected_cluster_types,
            )
        except Exception as exc:
            QMessageBox.warning(self, "Bond Analysis", str(exc))
            return

        self.log_box.clear()
        self._append_log("Bond-analysis run started.")
        self.run_button.setEnabled(False)
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_label.setText("Progress: starting")
        self.statusBar().showMessage("Preparing bond analysis...")

        self._run_thread = QThread(self)
        self._run_worker = BondAnalysisWorker(workflow)
        self._run_worker.moveToThread(self._run_thread)
        self._run_thread.started.connect(self._run_worker.run)
        self._run_worker.log.connect(self._append_log)
        self._run_worker.progress.connect(self._update_progress)
        self._run_worker.status.connect(self.statusBar().showMessage)
        self._run_worker.finished.connect(self._finish_run)
        self._run_worker.failed.connect(self._fail_run)
        self._run_worker.finished.connect(self._cleanup_run_thread)
        self._run_worker.failed.connect(self._cleanup_run_thread)
        self._run_thread.start()

    def _update_progress(self, processed: int, total: int) -> None:
        total = max(int(total), 1)
        processed = max(0, min(int(processed), total))
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(processed)
        self.progress_bar.setFormat("%v / %m files")
        self.progress_label.setText(
            f"Progress: {processed} processed, {total - processed} remaining"
        )

    def _finish_run(self, result: BondAnalysisBatchResult) -> None:
        self.run_button.setEnabled(True)
        self.progress_label.setText(
            f"Progress: complete ({result.total_structure_files} files)"
        )
        self.statusBar().showMessage(
            f"Bond analysis complete: {result.output_dir}"
        )
        self.output_dir_edit.setText(str(result.output_dir))
        self._append_log(f"Output directory: {result.output_dir}")
        self._append_log(f"Results index file: {result.results_index_path}")
        for cluster_result in result.cluster_results:
            self._append_log(
                f"{cluster_result.cluster_type}: "
                f"{cluster_result.structure_count} file(s)"
            )
        self._refresh_results_tree()
        self._update_selection_summary()

    def _fail_run(self, message: str) -> None:
        self.run_button.setEnabled(True)
        self.progress_label.setText("Progress: failed")
        self.statusBar().showMessage("Bond analysis failed")
        self._append_log(f"Run failed: {message}")
        QMessageBox.critical(self, "Bond Analysis", message)

    def _cleanup_run_thread(self, _payload: object) -> None:
        if self._run_thread is None:
            return
        self._run_thread.quit()
        self._run_thread.wait()
        if self._run_worker is not None:
            self._run_worker.deleteLater()
        self._run_thread.deleteLater()
        self._run_worker = None
        self._run_thread = None


def launch_bondanalysis_ui(
    clusters_dir: str | Path | None = None,
) -> int:
    """Launch the Qt6 bond-analysis UI."""
    app = QApplication.instance()
    owns_app = app is None
    if app is None:
        prepare_saxshell_application_identity()
        app = QApplication(sys.argv)
    configure_saxshell_application(app)

    window = BondAnalysisMainWindow(initial_clusters_dir=clusters_dir)
    _OPEN_WINDOWS.append(window)
    window.show()
    if owns_app:
        return app.exec()
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point for launching the Qt6 bond-analysis UI."""
    parser = argparse.ArgumentParser(
        prog="bondanalysis-ui",
        description=(
            "Launch the SAXSShell bondanalysis UI for stoichiometry-level "
            "cluster folders."
        ),
    )
    parser.add_argument(
        "clusters_dir",
        nargs="?",
        help="Optional clusters directory to prefill in the UI.",
    )
    args = parser.parse_args(argv)
    return launch_bondanalysis_ui(args.clusters_dir)


__all__ = ["BondAnalysisMainWindow", "launch_bondanalysis_ui", "main"]
