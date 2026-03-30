from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from saxshell.cluster import (
    DEFAULT_SAVE_STATE_FREQUENCY,
    SEARCH_MODE_BRUTEFORCE,
    SEARCH_MODE_KDTREE,
    SEARCH_MODE_VECTORIZED,
    PairCutoffDefinitions,
    example_atom_type_definitions,
    example_pair_cutoff_definitions,
    format_search_mode_label,
)
from saxshell.structure import AtomTypeDefinitions


class ClusterDefinitionsPanel(QGroupBox):
    """Editable cluster rules and analysis options."""

    settings_changed = Signal()

    def __init__(self) -> None:
        super().__init__("Cluster Definitions")
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout()

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

        example_atoms = example_atom_type_definitions()
        self._add_atom_row("node", element=example_atoms["node"][0][0])
        self._add_atom_row("linker", element=example_atoms["linker"][0][0])
        self._add_atom_row("shell", element=example_atoms["shell"][0][0])
        example_pairs = example_pair_cutoff_definitions()
        self._add_pair_row(
            atom1="Pb",
            atom2="I",
            shell0=str(example_pairs[("Pb", "I")][0]),
        )
        self._add_pair_row(
            atom1="Pb",
            atom2="O",
            shell0=str(example_pairs[("Pb", "O")][0]),
        )
        self._sync_pair_element_choices()
        self.set_frame_mode(None)

    def _make_box_spin(self, tooltip: str) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setDecimals(3)
        spin.setRange(0.0, 10**6)
        spin.setSingleStep(0.5)
        spin.setSpecialValueText("Auto")
        spin.setToolTip(tooltip)
        spin.valueChanged.connect(lambda _value: self.settings_changed.emit())
        return spin

    def _add_atom_row(
        self,
        atom_type: str = "node",
        *,
        element: str = "",
        residue: str = "",
    ) -> None:
        row = self.atom_table.rowCount()
        self.atom_table.insertRow(row)

        type_combo = QComboBox()
        type_combo.addItems(["node", "linker", "shell"])
        type_combo.setCurrentText(atom_type)
        type_combo.setToolTip("Cluster role assigned to this element.")
        type_combo.currentTextChanged.connect(
            lambda _text: self.settings_changed.emit()
        )
        self.atom_table.setCellWidget(row, 0, type_combo)

        self.atom_table.setItem(row, 1, QTableWidgetItem(element))
        self.atom_table.setItem(row, 2, QTableWidgetItem(residue))
        self.settings_changed.emit()

    def _remove_atom_row(self) -> None:
        row = self.atom_table.currentRow()
        if row < 0:
            row = self.atom_table.rowCount() - 1
        if row >= 0:
            self.atom_table.removeRow(row)
            self._sync_pair_element_choices()
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
        atom1_combo.setCurrentText(atom1)
        atom2_combo.setCurrentText(atom2)
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

    def _on_atom_table_changed(self) -> None:
        self._sync_pair_element_choices()
        self.settings_changed.emit()

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
                type_combo = QComboBox()
                type_combo.addItems(["node", "linker", "shell"])
                if atom_type not in {"node", "linker", "shell"}:
                    type_combo.addItem(atom_type)
                type_combo.setCurrentText(atom_type)
                type_combo.setToolTip("Cluster role assigned to this element.")
                type_combo.currentTextChanged.connect(
                    lambda _text: self.settings_changed.emit()
                )
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
        self._sync_pair_element_choices()
        if emit_signal:
            self.settings_changed.emit()

    def load_pair_cutoff_definitions(
        self,
        definitions: PairCutoffDefinitions,
        *,
        emit_signal: bool = True,
    ) -> None:
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
            atom1_combo.setCurrentText(atom1)
            atom2_combo.setCurrentText(atom2)
            shell_cutoffs = definitions[(atom1, atom2)]
            for level, column in enumerate((2, 3, 4)):
                cutoff = shell_cutoffs.get(level)
                self.pair_table.setItem(
                    row,
                    column,
                    QTableWidgetItem("" if cutoff is None else str(cutoff)),
                )
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

    def set_frame_mode(self, frame_format: str | None) -> None:
        if frame_format == "xyz":
            self.setTitle("Cluster Definitions (XYZ mode)")
            self.atom_table.setColumnHidden(2, True)
            self.mode_hint_label.setText(
                "XYZ mode uses element-only atom matching. Residue names are "
                "not available, and shell extraction cannot rebuild full "
                "molecules the way PDB mode can."
            )
            return

        self.atom_table.setColumnHidden(2, False)
        if frame_format == "pdb":
            self.setTitle("Cluster Definitions (PDB mode)")
            self.mode_hint_label.setText(
                "PDB mode can match atom types by element and residue, and "
                "shell export can keep complete residue molecules together."
            )
            return

        self.setTitle("Cluster Definitions")
        self.mode_hint_label.setText(
            "Select an extracted PDB or XYZ frames folder. The UI will adapt "
            "these rules automatically once the frame-set mode is detected."
        )

    @staticmethod
    def _table_text(table: QTableWidget, row: int, column: int) -> str:
        item = table.item(row, column)
        if item is None:
            return ""
        return item.text().strip()
