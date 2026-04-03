from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from saxshell.cluster import (
    DEFAULT_CLUSTER_EXTRACTION_PRESET_NAME,
    DEFAULT_SAVE_STATE_FREQUENCY,
    SEARCH_MODE_BRUTEFORCE,
    SEARCH_MODE_KDTREE,
    SEARCH_MODE_VECTORIZED,
    ClusterExtractionPreset,
    PairCutoffDefinitions,
    PDBShellReferenceDefinition,
    format_search_mode_label,
    load_cluster_extraction_presets,
    ordered_cluster_extraction_preset_names,
    save_custom_cluster_extraction_preset,
)
from saxshell.structure import AtomTypeDefinitions
from saxshell.xyz2pdb import ReferenceLibraryEntry


class ClusterDefinitionsPanel(QGroupBox):
    """Editable cluster rules and analysis options."""

    settings_changed = Signal()

    def __init__(self) -> None:
        super().__init__("Cluster Definitions")
        self._presets: dict[str, ClusterExtractionPreset] = {}
        self._applying_preset = False
        self._frame_format: str | None = None
        self._shell_reference_editing_enabled = False
        self._reference_entries_by_name: dict[str, ReferenceLibraryEntry] = {}
        self._build_ui()
        self.settings_changed.connect(self._sync_preset_selection)

    def _build_ui(self) -> None:
        layout = QVBoxLayout()

        layout.addWidget(self._build_presets_group())

        self.mode_hint_label = QLabel()
        self.mode_hint_label.setWordWrap(True)
        self.mode_hint_label.setToolTip(
            "Mode-specific guidance for the currently detected extracted "
            "frame format."
        )
        layout.addWidget(self.mode_hint_label)

        atom_group = QGroupBox("Atom Type Definitions")
        atom_group.setToolTip(
            "Map element and residue combinations to node, linker, or shell "
            "roles for cluster analysis."
        )
        atom_layout = QVBoxLayout(atom_group)
        self.atom_table = QTableWidget(0, 3)
        self.atom_table.setHorizontalHeaderLabels(
            ["Type", "Element", "Residue"]
        )
        self.atom_table.verticalHeader().setVisible(False)
        self.atom_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.atom_table.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked
            | QAbstractItemView.EditTrigger.EditKeyPressed
            | QAbstractItemView.EditTrigger.AnyKeyPressed
        )
        atom_header = self.atom_table.horizontalHeader()
        atom_header.setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        atom_header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        atom_header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.atom_table.itemChanged.connect(self._on_atom_table_changed)
        atom_layout.addWidget(self.atom_table)

        atom_buttons = QHBoxLayout()
        add_atom_button = QPushButton("+")
        add_atom_button.setToolTip("Add another atom-type definition row.")
        add_atom_button.clicked.connect(
            lambda _checked=False: self._add_atom_row()
        )
        remove_atom_button = QPushButton("-")
        remove_atom_button.setToolTip("Remove the selected atom-type row.")
        remove_atom_button.clicked.connect(
            lambda _checked=False: self._remove_atom_row()
        )
        atom_buttons.addWidget(add_atom_button)
        atom_buttons.addWidget(remove_atom_button)
        atom_buttons.addStretch(1)
        atom_layout.addLayout(atom_buttons)
        layout.addWidget(atom_group)

        self.shell_reference_group = self._build_shell_reference_group()
        layout.addWidget(self.shell_reference_group)

        pair_group = QGroupBox("Pair Cutoff Definitions")
        pair_group.setToolTip(
            "Cutoff distances used to connect atom pairs for shell 0, shell "
            "1, and shell 2 growth."
        )
        pair_layout = QVBoxLayout(pair_group)
        self.pair_table = QTableWidget(0, 5)
        self.pair_table.setHorizontalHeaderLabels(
            ["Atom 1", "Atom 2", "Shell 0", "Shell 1", "Shell 2"]
        )
        self.pair_table.verticalHeader().setVisible(False)
        self.pair_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        pair_header = self.pair_table.horizontalHeader()
        pair_header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        pair_header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        for column in (2, 3, 4):
            pair_header.setSectionResizeMode(
                column, QHeaderView.ResizeMode.ResizeToContents
            )
        self.pair_table.itemChanged.connect(
            lambda _item: self.settings_changed.emit()
        )
        pair_layout.addWidget(self.pair_table)

        pair_buttons = QHBoxLayout()
        add_pair_button = QPushButton("+")
        add_pair_button.setToolTip("Add another pair-cutoff row.")
        add_pair_button.clicked.connect(
            lambda _checked=False: self._add_pair_row()
        )
        remove_pair_button = QPushButton("-")
        remove_pair_button.setToolTip("Remove the selected pair-cutoff row.")
        remove_pair_button.clicked.connect(
            lambda _checked=False: self._remove_pair_row()
        )
        pair_buttons.addWidget(add_pair_button)
        pair_buttons.addWidget(remove_pair_button)
        pair_buttons.addStretch(1)
        pair_layout.addLayout(pair_buttons)
        layout.addWidget(pair_group)

        options_group = QGroupBox("Options")
        options_layout = QFormLayout(options_group)

        box_widget = QWidget()
        box_row = QHBoxLayout(box_widget)
        box_row.setContentsMargins(0, 0, 0, 0)
        self.box_x_spin = self._make_box_spin(
            "Periodic box length along X in angstrom. Leave at Auto to use "
            "the detected value from the selected frames folder when PBC is "
            "enabled. Split XYZ folders can also auto-fill this from a "
            "sibling source filename containing _pbc_."
        )
        self.box_y_spin = self._make_box_spin(
            "Periodic box length along Y in angstrom. Leave at Auto to use "
            "the detected value from the selected frames folder when PBC is "
            "enabled. Split XYZ folders can also auto-fill this from a "
            "sibling source filename containing _pbc_."
        )
        self.box_z_spin = self._make_box_spin(
            "Periodic box length along Z in angstrom. Leave at Auto to use "
            "the detected value from the selected frames folder when PBC is "
            "enabled. Split XYZ folders can also auto-fill this from a "
            "sibling source filename containing _pbc_."
        )
        for label, spin in (
            ("X", self.box_x_spin),
            ("Y", self.box_y_spin),
            ("Z", self.box_z_spin),
        ):
            box_row.addWidget(QLabel(f"{label}:"))
            box_row.addWidget(spin)
        box_row.addStretch(1)
        options_layout.addRow("Box dimensions (A)", box_widget)

        self.use_pbc_box = QCheckBox("Use periodic boundary conditions")
        self.use_pbc_box.setChecked(False)
        self.use_pbc_box.setToolTip(
            "Enable minimum-image periodic wrapping when connecting atoms "
            "into clusters. Leave off to use direct Cartesian distances."
        )
        self.use_pbc_box.toggled.connect(
            lambda _checked: self.settings_changed.emit()
        )
        options_layout.addRow("", self.use_pbc_box)

        self.search_mode_combo = QComboBox()
        self.search_mode_combo.addItem(
            f"{format_search_mode_label(SEARCH_MODE_KDTREE)} " "(Recommended)",
            SEARCH_MODE_KDTREE,
        )
        self.search_mode_combo.addItem(
            f"{format_search_mode_label(SEARCH_MODE_VECTORIZED)}",
            SEARCH_MODE_VECTORIZED,
        )
        self.search_mode_combo.addItem(
            f"{format_search_mode_label(SEARCH_MODE_BRUTEFORCE)} (Legacy)",
            SEARCH_MODE_BRUTEFORCE,
        )
        self.search_mode_combo.setToolTip(
            "Choose how neighbors are searched while building clusters. "
            "KDTree is much faster for larger frames, while brute force "
            "keeps the legacy all-pairs loop for debugging or parity "
            "checks."
        )
        self.search_mode_combo.currentIndexChanged.connect(
            lambda _index: self.settings_changed.emit()
        )
        options_layout.addRow("Search mode", self.search_mode_combo)

        self.save_state_frequency_spin = QSpinBox()
        self.save_state_frequency_spin.setRange(1, 10**9)
        self.save_state_frequency_spin.setValue(DEFAULT_SAVE_STATE_FREQUENCY)
        self.save_state_frequency_spin.setSingleStep(100)
        self.save_state_frequency_spin.setToolTip(
            "Write resumable extraction metadata after this many processed "
            "frames. Lower values improve resume granularity but add more "
            "disk I/O during long runs."
        )
        self.save_state_frequency_spin.valueChanged.connect(
            lambda _value: self.settings_changed.emit()
        )
        options_layout.addRow(
            "Save-state frequency (frames)",
            self.save_state_frequency_spin,
        )

        self.default_cutoff_spin = QDoubleSpinBox()
        self.default_cutoff_spin.setDecimals(3)
        self.default_cutoff_spin.setRange(0.0, 10**6)
        self.default_cutoff_spin.setSingleStep(0.1)
        self.default_cutoff_spin.setSpecialValueText("None")
        self.default_cutoff_spin.setToolTip(
            "Fallback cutoff used when a specific atom-pair cutoff is not "
            "defined. Leave at 0.0 to disable."
        )
        self.default_cutoff_spin.valueChanged.connect(
            lambda _value: self.settings_changed.emit()
        )
        options_layout.addRow("Default cutoff (A)", self.default_cutoff_spin)

        shell_widget = QWidget()
        shell_row = QHBoxLayout(shell_widget)
        shell_row.setContentsMargins(0, 0, 0, 0)
        self.shell0_box = QCheckBox("0")
        self.shell0_box.setChecked(True)
        self.shell0_box.setEnabled(False)
        self.shell0_box.setToolTip(
            "Core cluster atoms are always included in export output."
        )
        self.shell1_box = QCheckBox("1")
        self.shell1_box.setToolTip(
            "Expand one shell of solvent around each cluster."
        )
        self.shell1_box.toggled.connect(
            lambda _checked: self.settings_changed.emit()
        )
        self.shell2_box = QCheckBox("2")
        self.shell2_box.setToolTip(
            "Expand a second shell of solvent around each cluster."
        )
        self.shell2_box.toggled.connect(
            lambda _checked: self.settings_changed.emit()
        )
        shell_row.addWidget(QLabel("Include shell levels:"))
        shell_row.addWidget(self.shell0_box)
        shell_row.addWidget(self.shell1_box)
        shell_row.addWidget(self.shell2_box)
        shell_row.addStretch(1)
        options_layout.addRow("", shell_widget)

        self.shared_shells_box = QCheckBox(
            "Allow shell atoms to be shared between clusters"
        )
        self.shared_shells_box.setToolTip(
            "If enabled, solvent atoms can belong to more than one cluster "
            "shell instead of being consumed by the first cluster that finds "
            "them."
        )
        self.shared_shells_box.toggled.connect(
            lambda _checked: self.settings_changed.emit()
        )
        options_layout.addRow("", self.shared_shells_box)

        self.include_shell_stoichiometry_box = QCheckBox(
            "Include shell atoms in stoichiometry bins"
        )
        self.include_shell_stoichiometry_box.setChecked(False)
        self.include_shell_stoichiometry_box.setToolTip(
            "If enabled, detected shell atoms are added to the "
            "stoichiometry count that controls the output folder bins."
        )
        self.include_shell_stoichiometry_box.toggled.connect(
            lambda _checked: self.settings_changed.emit()
        )
        options_layout.addRow("", self.include_shell_stoichiometry_box)

        layout.addWidget(options_group)
        layout.addStretch(1)
        self.setLayout(layout)

        self._reload_presets()
        self.set_frame_mode(None)

    def _build_presets_group(self) -> QGroupBox:
        group = QGroupBox("Presets")
        layout = QVBoxLayout(group)

        row = QHBoxLayout()
        self.preset_combo = QComboBox()
        self.preset_combo.setToolTip(
            "Load a built-in cluster extraction preset or a custom preset "
            "saved from a previous session."
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
            "Built-in preset: "
            f"{DEFAULT_CLUSTER_EXTRACTION_PRESET_NAME}. Custom presets are "
            "saved for later sessions for this install."
        )
        self.preset_hint_label.setWordWrap(True)
        layout.addWidget(self.preset_hint_label)
        return group

    def _make_box_spin(self, tooltip: str) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setDecimals(3)
        spin.setRange(0.0, 10**6)
        spin.setSingleStep(0.5)
        spin.setSpecialValueText("Auto")
        spin.setToolTip(tooltip)
        spin.valueChanged.connect(lambda _value: self.settings_changed.emit())
        return spin

    def _build_shell_reference_group(self) -> QGroupBox:
        group = QGroupBox("PDB Shell References")
        group.setToolTip(
            "PDB-only solvent references used by clusterdynamicsml to "
            "rebuild full solvent molecules from predicted shell-anchor "
            "atoms in the final PDB files."
        )
        layout = QVBoxLayout(group)
        helper = QLabel(
            "When a shell atom rule is defined in PDB mode, choose the "
            "reference molecule to inject at each predicted shell-anchor "
            "atom. The solvent anchor and orientation are resolved "
            "automatically from the reference molecule."
        )
        helper.setWordWrap(True)
        layout.addWidget(helper)

        self.shell_reference_table = QTableWidget(0, 3)
        self.shell_reference_table.setHorizontalHeaderLabels(
            [
                "Shell Element",
                "Shell Residue",
                "Reference",
            ]
        )
        self.shell_reference_table.verticalHeader().setVisible(False)
        self.shell_reference_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.shell_reference_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        shell_header = self.shell_reference_table.horizontalHeader()
        shell_header.setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        shell_header.setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents
        )
        shell_header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.shell_reference_table)
        group.setVisible(False)
        return group

    def _add_atom_row(
        self,
        atom_type: str = "node",
        *,
        element: str = "",
        residue: str = "",
    ) -> None:
        row = self.atom_table.rowCount()
        self.atom_table.insertRow(row)

        type_combo = self._make_atom_type_combo(atom_type)
        self.atom_table.setCellWidget(row, 0, type_combo)

        self.atom_table.setItem(row, 1, QTableWidgetItem(element))
        self.atom_table.setItem(row, 2, QTableWidgetItem(residue))
        self._sync_shell_reference_rows()
        self.settings_changed.emit()

    def _remove_atom_row(self) -> None:
        row = self.atom_table.currentRow()
        if row < 0:
            row = self.atom_table.rowCount() - 1
        if row >= 0:
            self.atom_table.removeRow(row)
            self._sync_pair_element_choices()
            self._sync_shell_reference_rows()
            self.settings_changed.emit()

    def _add_pair_row(
        self,
        *,
        atom1: str = "",
        atom2: str = "",
        shell0: str = "",
        shell1: str = "",
        shell2: str = "",
    ) -> None:
        row = self.pair_table.rowCount()
        self.pair_table.insertRow(row)

        atom1_combo = self._make_pair_combo()
        atom2_combo = self._make_pair_combo()
        self.pair_table.setCellWidget(row, 0, atom1_combo)
        self.pair_table.setCellWidget(row, 1, atom2_combo)

        for column, text in enumerate((shell0, shell1, shell2), start=2):
            self.pair_table.setItem(row, column, QTableWidgetItem(text))

        self._sync_pair_element_choices()
        self._set_pair_combo_text(atom1_combo, atom1)
        self._set_pair_combo_text(atom2_combo, atom2)
        self.settings_changed.emit()

    def _remove_pair_row(self) -> None:
        row = self.pair_table.currentRow()
        if row < 0:
            row = self.pair_table.rowCount() - 1
        if row >= 0:
            self.pair_table.removeRow(row)
            self.settings_changed.emit()

    def _make_pair_combo(self) -> QComboBox:
        combo = QComboBox()
        combo.setEditable(False)
        combo.setToolTip(
            "Element symbol for one side of the atom-pair cutoff rule."
        )
        combo.currentTextChanged.connect(
            lambda _text: self.settings_changed.emit()
        )
        return combo

    def _make_atom_type_combo(self, atom_type: str) -> QComboBox:
        combo = QComboBox()
        combo.addItems(["node", "linker", "shell"])
        if atom_type not in {"node", "linker", "shell"}:
            combo.addItem(atom_type)
        combo.blockSignals(True)
        combo.setCurrentText(atom_type)
        combo.blockSignals(False)
        combo.setToolTip("Cluster role assigned to this element.")
        combo.currentTextChanged.connect(self._on_atom_type_combo_changed)
        return combo

    def _make_shell_reference_combo(self) -> QComboBox:
        combo = QComboBox()
        combo.addItem("", "")
        for name in sorted(self._reference_entries_by_name):
            combo.addItem(name, name)
        combo.currentTextChanged.connect(
            lambda _text: self.settings_changed.emit()
        )
        return combo

    def _set_pair_combo_text(self, combo: QComboBox, text: str) -> None:
        value = text.strip().title()
        combo.blockSignals(True)
        if value and combo.findText(value) < 0:
            combo.addItem(value)
        combo.setCurrentText(value)
        combo.blockSignals(False)

    def _on_atom_type_combo_changed(self, _text: str) -> None:
        self._sync_shell_reference_rows()
        self.settings_changed.emit()

    def _on_atom_table_changed(self) -> None:
        self._sync_pair_element_choices()
        self._sync_shell_reference_rows()
        self.settings_changed.emit()

    def _on_shell_reference_selection_changed(self) -> None:
        self.settings_changed.emit()

    def _shell_rule_keys(self) -> list[tuple[str, str | None]]:
        keys: list[tuple[str, str | None]] = []
        seen: set[tuple[str, str | None]] = set()
        for row in range(self.atom_table.rowCount()):
            combo = self.atom_table.cellWidget(row, 0)
            if not isinstance(combo, QComboBox):
                continue
            if combo.currentText().strip() != "shell":
                continue
            element = self._table_text(self.atom_table, row, 1).title()
            residue_text = self._table_text(self.atom_table, row, 2)
            residue = residue_text or None
            if not element:
                continue
            key = (element, residue)
            if key in seen:
                continue
            seen.add(key)
            keys.append(key)
        return keys

    def _shell_reference_selection_map(
        self,
    ) -> dict[tuple[str, str | None], str]:
        selections: dict[tuple[str, str | None], str] = {}
        table = self.shell_reference_table
        for row in range(table.rowCount()):
            key = self._shell_reference_row_key(row)
            if key is None:
                continue
            reference_combo = table.cellWidget(row, 2)
            if not isinstance(reference_combo, QComboBox):
                continue
            selections[key] = str(
                reference_combo.currentData() or reference_combo.currentText()
            ).strip()
        return selections

    def _shell_reference_row_key(
        self,
        row: int,
    ) -> tuple[str, str | None] | None:
        element_item = self.shell_reference_table.item(row, 0)
        residue_item = self.shell_reference_table.item(row, 1)
        if element_item is None:
            return None
        element = element_item.text().strip().title()
        residue_text = (
            "" if residue_item is None else residue_item.text().strip()
        )
        if not element:
            return None
        return element, (residue_text or None)

    def _sync_shell_reference_rows(self) -> None:
        selections = self._shell_reference_selection_map()
        rule_keys = self._shell_rule_keys()
        table = self.shell_reference_table
        table.blockSignals(True)
        table.setRowCount(0)
        for row, (element, residue) in enumerate(rule_keys):
            table.insertRow(row)
            element_item = QTableWidgetItem(element)
            element_item.setFlags(
                element_item.flags() & ~Qt.ItemFlag.ItemIsEditable
            )
            residue_item = QTableWidgetItem("" if residue is None else residue)
            residue_item.setFlags(
                residue_item.flags() & ~Qt.ItemFlag.ItemIsEditable
            )
            table.setItem(row, 0, element_item)
            table.setItem(row, 1, residue_item)
            reference_combo = self._make_shell_reference_combo()
            table.setCellWidget(row, 2, reference_combo)
            self._set_combo_data(
                reference_combo,
                selections.get((element, residue), ""),
            )
        table.blockSignals(False)
        self._update_shell_reference_visibility()

    def _set_combo_data(self, combo: QComboBox, data: str) -> None:
        combo.blockSignals(True)
        index = combo.findData(data)
        if index >= 0:
            combo.setCurrentIndex(index)
        else:
            combo.setCurrentText(data)
        combo.blockSignals(False)

    def _update_shell_reference_visibility(self) -> None:
        show_group = (
            self._shell_reference_editing_enabled
            and self._frame_format == "pdb"
            and bool(self._shell_rule_keys())
        )
        self.shell_reference_group.setVisible(show_group)

    def _sync_pair_element_choices(self) -> None:
        elements = sorted(
            {
                self._table_text(self.atom_table, row, 1).title()
                for row in range(self.atom_table.rowCount())
                if self._table_text(self.atom_table, row, 1)
            }
        )
        for row in range(self.pair_table.rowCount()):
            for column in (0, 1):
                combo = self.pair_table.cellWidget(row, column)
                if not isinstance(combo, QComboBox):
                    continue
                current_text = combo.currentText()
                combo.blockSignals(True)
                combo.clear()
                combo.addItem("")
                combo.addItems(elements)
                if current_text in elements:
                    combo.setCurrentText(current_text)
                combo.blockSignals(False)

    def atom_type_definitions(self) -> AtomTypeDefinitions:
        definitions: AtomTypeDefinitions = {}
        for row in range(self.atom_table.rowCount()):
            combo = self.atom_table.cellWidget(row, 0)
            if not isinstance(combo, QComboBox):
                continue
            atom_type = combo.currentText().strip()
            element = self._table_text(self.atom_table, row, 1)
            residue = self._table_text(self.atom_table, row, 2)
            if not atom_type or not element:
                continue
            definitions.setdefault(atom_type, []).append(
                (element.title(), residue or None)
            )
        return definitions

    def set_shell_reference_editor_enabled(
        self,
        enabled: bool,
    ) -> None:
        self._shell_reference_editing_enabled = bool(enabled)
        self._update_shell_reference_visibility()

    def set_shell_reference_library_entries(
        self,
        entries: list[ReferenceLibraryEntry],
        *,
        emit_signal: bool = True,
    ) -> None:
        self._reference_entries_by_name = {
            entry.name: entry for entry in entries
        }
        self._sync_shell_reference_rows()
        if emit_signal:
            self.settings_changed.emit()

    def shell_reference_definitions(
        self,
    ) -> tuple[PDBShellReferenceDefinition, ...]:
        if (
            not self._shell_reference_editing_enabled
            or self._frame_format != "pdb"
        ):
            return ()

        definitions: list[PDBShellReferenceDefinition] = []
        for row in range(self.shell_reference_table.rowCount()):
            key = self._shell_reference_row_key(row)
            if key is None:
                continue
            reference_combo = self.shell_reference_table.cellWidget(row, 2)
            if not isinstance(reference_combo, QComboBox):
                continue
            reference_name = str(
                reference_combo.currentData() or reference_combo.currentText()
            ).strip()
            if not reference_name:
                raise ValueError(
                    "Choose a reference molecule for PDB shell rule "
                    f"{key[0]}"
                    + ("" if key[1] is None else f" ({key[1]})")
                    + "."
                )
            definitions.append(
                PDBShellReferenceDefinition(
                    shell_element=key[0],
                    shell_residue=key[1],
                    reference_name=reference_name,
                )
            )
        return tuple(definitions)

    def load_shell_reference_definitions(
        self,
        definitions: (
            tuple[PDBShellReferenceDefinition, ...]
            | list[PDBShellReferenceDefinition]
        ),
        *,
        emit_signal: bool = True,
    ) -> None:
        selection_map = {
            (definition.shell_element.title(), definition.shell_residue): (
                definition.reference_name
            )
            for definition in definitions
        }
        self._sync_shell_reference_rows()
        for row in range(self.shell_reference_table.rowCount()):
            key = self._shell_reference_row_key(row)
            if key is None or key not in selection_map:
                continue
            reference_name = selection_map[key]
            reference_combo = self.shell_reference_table.cellWidget(row, 2)
            if isinstance(reference_combo, QComboBox):
                self._set_combo_data(reference_combo, reference_name)
        if emit_signal:
            self.settings_changed.emit()

    def pair_cutoff_definitions(self) -> PairCutoffDefinitions:
        definitions: PairCutoffDefinitions = {}
        for row in range(self.pair_table.rowCount()):
            atom1_combo = self.pair_table.cellWidget(row, 0)
            atom2_combo = self.pair_table.cellWidget(row, 1)
            if not isinstance(atom1_combo, QComboBox) or not isinstance(
                atom2_combo, QComboBox
            ):
                continue
            atom1 = atom1_combo.currentText().strip()
            atom2 = atom2_combo.currentText().strip()
            if not atom1 or not atom2:
                continue

            shell_cutoffs: dict[int, float] = {}
            for level, column in enumerate((2, 3, 4)):
                text = self._table_text(self.pair_table, row, column)
                if not text:
                    continue
                try:
                    value = float(text)
                except ValueError as exc:
                    raise ValueError(
                        "Pair cutoff row "
                        f"{row + 1}, shell {level} is not a valid number."
                    ) from exc
                if value > 0.0:
                    shell_cutoffs[level] = value
            if shell_cutoffs:
                definitions[(atom1.title(), atom2.title())] = shell_cutoffs
        return definitions

    def box_dimensions(self) -> tuple[float, float, float] | None:
        values = (
            self.box_x_spin.value(),
            self.box_y_spin.value(),
            self.box_z_spin.value(),
        )
        if all(value == 0.0 for value in values):
            return None
        if any(value == 0.0 for value in values):
            raise ValueError(
                "Specify all three box dimensions or leave all of them at "
                "0.0 to disable periodic wrapping."
            )
        return values

    def set_box_dimensions(
        self,
        box_dimensions: tuple[float, float, float] | None,
        *,
        emit_signal: bool = True,
    ) -> None:
        values = (
            (0.0, 0.0, 0.0)
            if box_dimensions is None
            else tuple(float(component) for component in box_dimensions)
        )
        for spin, value in zip(
            (self.box_x_spin, self.box_y_spin, self.box_z_spin),
            values,
        ):
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)
        if emit_signal:
            self.settings_changed.emit()

    def default_cutoff(self) -> float | None:
        value = self.default_cutoff_spin.value()
        return None if value <= 0.0 else value

    def load_atom_type_definitions(
        self,
        definitions: AtomTypeDefinitions,
        *,
        emit_signal: bool = True,
    ) -> None:
        self.atom_table.blockSignals(True)
        self.atom_table.setRowCount(0)
        ordered_atom_types = ["node", "linker", "shell"]
        seen_atom_types: set[str] = set()
        for atom_type in ordered_atom_types + sorted(definitions):
            if atom_type in seen_atom_types:
                continue
            seen_atom_types.add(atom_type)
            for element, residue in definitions.get(atom_type, []):
                row = self.atom_table.rowCount()
                self.atom_table.insertRow(row)
                type_combo = self._make_atom_type_combo(atom_type)
                self.atom_table.setCellWidget(row, 0, type_combo)
                self.atom_table.setItem(
                    row,
                    1,
                    QTableWidgetItem("" if element is None else str(element)),
                )
                self.atom_table.setItem(
                    row,
                    2,
                    QTableWidgetItem("" if residue is None else str(residue)),
                )
        self.atom_table.blockSignals(False)
        self._sync_pair_element_choices()
        self._sync_shell_reference_rows()
        if emit_signal:
            self.settings_changed.emit()

    def load_pair_cutoff_definitions(
        self,
        definitions: PairCutoffDefinitions,
        *,
        emit_signal: bool = True,
    ) -> None:
        self.pair_table.blockSignals(True)
        self.pair_table.setRowCount(0)
        self._sync_pair_element_choices()
        for atom1, atom2 in sorted(definitions):
            row = self.pair_table.rowCount()
            self.pair_table.insertRow(row)
            atom1_combo = self._make_pair_combo()
            atom2_combo = self._make_pair_combo()
            self.pair_table.setCellWidget(row, 0, atom1_combo)
            self.pair_table.setCellWidget(row, 1, atom2_combo)
            self._sync_pair_element_choices()
            self._set_pair_combo_text(atom1_combo, atom1)
            self._set_pair_combo_text(atom2_combo, atom2)
            shell_cutoffs = definitions[(atom1, atom2)]
            for level, column in enumerate((2, 3, 4)):
                cutoff = shell_cutoffs.get(level)
                self.pair_table.setItem(
                    row,
                    column,
                    QTableWidgetItem("" if cutoff is None else str(cutoff)),
                )
        self.pair_table.blockSignals(False)
        if emit_signal:
            self.settings_changed.emit()

    def set_default_cutoff(
        self,
        value: float | None,
        *,
        emit_signal: bool = True,
    ) -> None:
        self.default_cutoff_spin.blockSignals(True)
        self.default_cutoff_spin.setValue(
            0.0 if value is None else float(value)
        )
        self.default_cutoff_spin.blockSignals(False)
        if emit_signal:
            self.settings_changed.emit()

    def use_pbc(self) -> bool:
        return self.use_pbc_box.isChecked()

    def set_use_pbc(self, value: bool, *, emit_signal: bool = True) -> None:
        self.use_pbc_box.blockSignals(True)
        self.use_pbc_box.setChecked(bool(value))
        self.use_pbc_box.blockSignals(False)
        if emit_signal:
            self.settings_changed.emit()

    def search_mode(self) -> str:
        data = self.search_mode_combo.currentData()
        if data is None:
            return SEARCH_MODE_KDTREE
        return str(data)

    def set_search_mode(self, value: str, *, emit_signal: bool = True) -> None:
        for index in range(self.search_mode_combo.count()):
            if self.search_mode_combo.itemData(index) == value:
                self.search_mode_combo.blockSignals(True)
                self.search_mode_combo.setCurrentIndex(index)
                self.search_mode_combo.blockSignals(False)
                break
        if emit_signal:
            self.settings_changed.emit()

    def save_state_frequency(self) -> int:
        return int(self.save_state_frequency_spin.value())

    def set_save_state_frequency(
        self,
        value: int,
        *,
        emit_signal: bool = True,
    ) -> None:
        self.save_state_frequency_spin.blockSignals(True)
        self.save_state_frequency_spin.setValue(int(value))
        self.save_state_frequency_spin.blockSignals(False)
        if emit_signal:
            self.settings_changed.emit()

    def include_shell_levels(self) -> tuple[int, ...]:
        levels = [0]
        if self.shell1_box.isChecked():
            levels.append(1)
        if self.shell2_box.isChecked():
            levels.append(2)
        return tuple(levels)

    def shell_growth_levels(self) -> tuple[int, ...]:
        return tuple(
            level for level in self.include_shell_levels() if level > 0
        )

    def set_shell_growth_levels(
        self,
        levels: tuple[int, ...] | list[int],
        *,
        emit_signal: bool = True,
    ) -> None:
        normalized = {int(level) for level in levels}
        self.shell1_box.blockSignals(True)
        self.shell2_box.blockSignals(True)
        self.shell1_box.setChecked(1 in normalized)
        self.shell2_box.setChecked(2 in normalized)
        self.shell1_box.blockSignals(False)
        self.shell2_box.blockSignals(False)
        if emit_signal:
            self.settings_changed.emit()

    def shared_shells(self) -> bool:
        return self.shared_shells_box.isChecked()

    def set_shared_shells(
        self,
        value: bool,
        *,
        emit_signal: bool = True,
    ) -> None:
        self.shared_shells_box.blockSignals(True)
        self.shared_shells_box.setChecked(bool(value))
        self.shared_shells_box.blockSignals(False)
        if emit_signal:
            self.settings_changed.emit()

    def include_shell_atoms_in_stoichiometry(self) -> bool:
        return self.include_shell_stoichiometry_box.isChecked()

    def set_include_shell_atoms_in_stoichiometry(
        self,
        value: bool,
        *,
        emit_signal: bool = True,
    ) -> None:
        self.include_shell_stoichiometry_box.blockSignals(True)
        self.include_shell_stoichiometry_box.setChecked(bool(value))
        self.include_shell_stoichiometry_box.blockSignals(False)
        if emit_signal:
            self.settings_changed.emit()

    def rule_counts(self) -> tuple[int, int]:
        atom_rules = sum(
            len(criteria) for criteria in self.atom_type_definitions().values()
        )
        pair_rules = len(self.pair_cutoff_definitions())
        return atom_rules, pair_rules

    def _selected_preset_name(self) -> str | None:
        return self.preset_combo.currentData()

    def _reload_presets(self, *, selected_name: str | None = None) -> None:
        previous_name = selected_name or self._selected_preset_name()
        self._presets = load_cluster_extraction_presets()
        self.preset_combo.blockSignals(True)
        self.preset_combo.clear()
        self.preset_combo.addItem("Current values", None)
        selected_index = 0
        for index, name in enumerate(
            ordered_cluster_extraction_preset_names(self._presets),
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

    def load_preset(self, preset_name: str) -> None:
        preset = self._presets.get(preset_name)
        if preset is None:
            raise ValueError(f"Unknown preset: {preset_name}")
        self._applying_preset = True
        try:
            self._apply_preset(preset)
        finally:
            self._applying_preset = False
        self._select_preset_name(preset_name)

    def save_current_preset(self, preset_name: str) -> None:
        name = preset_name.strip()
        if not name:
            raise ValueError("Preset names cannot be empty.")
        preset = self._build_current_preset(name)
        save_custom_cluster_extraction_preset(preset)
        self._reload_presets(selected_name=name)

    def _select_preset_name(self, preset_name: str | None) -> None:
        target_index = 0
        if preset_name is not None:
            for index in range(self.preset_combo.count()):
                if self.preset_combo.itemData(index) == preset_name:
                    target_index = index
                    break
        self.preset_combo.blockSignals(True)
        self.preset_combo.setCurrentIndex(target_index)
        self.preset_combo.blockSignals(False)

    def _build_current_preset(
        self,
        preset_name: str,
    ) -> ClusterExtractionPreset:
        return ClusterExtractionPreset(
            name=preset_name,
            atom_type_definitions=self.atom_type_definitions(),
            pair_cutoff_definitions=self.pair_cutoff_definitions(),
            use_pbc=self.use_pbc(),
            search_mode=self.search_mode(),
            save_state_frequency=self.save_state_frequency(),
            default_cutoff=self.default_cutoff(),
            shell_growth_levels=self.shell_growth_levels(),
            shared_shells=self.shared_shells(),
            include_shell_atoms_in_stoichiometry=(
                self.include_shell_atoms_in_stoichiometry()
            ),
        )

    def _load_selected_preset(self) -> None:
        preset_name = self._selected_preset_name()
        if preset_name is None:
            QMessageBox.information(
                self,
                "Cluster Extraction Presets",
                "Select a preset to load.",
            )
            return
        try:
            self.load_preset(preset_name)
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Cluster Extraction Presets",
                str(exc),
            )

    def _save_current_as_preset(self) -> None:
        try:
            self.pair_cutoff_definitions()
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Cluster Extraction Presets",
                str(exc),
            )
            return

        suggested_name = self._selected_preset_name() or ""
        preset_name, accepted = QInputDialog.getText(
            self,
            "Save Cluster Extraction Preset",
            "Preset name:",
            text=suggested_name,
        )
        if not accepted:
            return
        preset_name = preset_name.strip()
        if not preset_name:
            return

        if preset_name in self._presets:
            response = QMessageBox.question(
                self,
                "Overwrite Preset?",
                f"A preset named '{preset_name}' already exists. Overwrite it?",
            )
            if response != QMessageBox.StandardButton.Yes:
                return

        try:
            self.save_current_preset(preset_name)
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Cluster Extraction Presets",
                str(exc),
            )

    def _apply_preset(self, preset: ClusterExtractionPreset) -> None:
        self.load_atom_type_definitions(
            preset.atom_type_definitions,
            emit_signal=False,
        )
        self.load_pair_cutoff_definitions(
            preset.pair_cutoff_definitions,
            emit_signal=False,
        )
        self.set_use_pbc(preset.use_pbc, emit_signal=False)
        self.set_search_mode(preset.search_mode, emit_signal=False)
        self.set_save_state_frequency(
            preset.save_state_frequency,
            emit_signal=False,
        )
        self.set_default_cutoff(preset.default_cutoff, emit_signal=False)
        self.set_shell_growth_levels(
            preset.shell_growth_levels,
            emit_signal=False,
        )
        self.set_shared_shells(preset.shared_shells, emit_signal=False)
        self.set_include_shell_atoms_in_stoichiometry(
            preset.include_shell_atoms_in_stoichiometry,
            emit_signal=False,
        )
        self.settings_changed.emit()

    def _sync_preset_selection(self) -> None:
        if self._applying_preset:
            return
        if self._selected_preset_name() is None:
            return
        self._select_preset_name(None)

    def set_frame_mode(self, frame_format: str | None) -> None:
        self._frame_format = frame_format
        if frame_format == "xyz":
            self.setTitle("Cluster Definitions (XYZ mode)")
            self.atom_table.setColumnHidden(2, True)
            self.mode_hint_label.setText(
                "XYZ mode uses element-only atom matching. Residue names are "
                "not available, and shell extraction cannot rebuild full "
                "molecules the way PDB mode can."
            )
            self._update_shell_reference_visibility()
            return

        self.atom_table.setColumnHidden(2, False)
        if frame_format == "pdb":
            self.setTitle("Cluster Definitions (PDB mode)")
            self.mode_hint_label.setText(
                "PDB mode can match atom types by element and residue, and "
                "shell export can keep complete residue molecules together."
            )
            self._update_shell_reference_visibility()
            return

        self.setTitle("Cluster Definitions")
        self.mode_hint_label.setText(
            "Select an extracted PDB or XYZ frames folder. The UI will adapt "
            "these rules automatically once the frame-set mode is detected."
        )
        self._update_shell_reference_visibility()

    @staticmethod
    def _table_text(table: QTableWidget, row: int, column: int) -> str:
        item = table.item(row, column)
        if item is None:
            return ""
        return item.text().strip()
