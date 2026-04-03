from __future__ import annotations

import re
from math import dist

from PySide6.QtCore import QRegularExpression, Qt, Signal
from PySide6.QtGui import QRegularExpressionValidator
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from saxshell.structure import PDBStructure
from saxshell.xyz2pdb import ReferenceLibraryEntry
from saxshell.xyz2pdb.mapping_workflow import (
    FreeAtomMappingInput,
    MoleculeMappingInput,
    ReferenceBondToleranceInput,
)

_FALLBACK_BOND_TOLERANCE_PERCENT = 12.0


class XYZToPDBMappingPanel(QGroupBox):
    """Editable mapping definitions for the native xyz2pdb workflow."""

    settings_changed = Signal()

    def __init__(self) -> None:
        super().__init__("PDB Mapping Definitions")
        self._molecule_inputs: list[MoleculeMappingInput] = []
        self._reference_bond_defaults: dict[
            str, tuple[ReferenceBondToleranceInput, ...]
        ] = {}
        self._reference_residue_defaults: dict[str, str] = {}
        self._reference_paths: dict[str, str] = {}
        self._reference_atom_coordinates: dict[
            str, dict[str, tuple[float, float, float]]
        ] = {}
        self._last_autofilled_residue_name: str | None = None
        self._populating_bond_table = False
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout()
        layout.addWidget(self._build_free_atoms_group(), stretch=2)
        layout.addWidget(self._build_molecules_group(), stretch=5)

        hydrogen_group = QGroupBox("Hydrogen Handling")
        hydrogen_layout = QFormLayout(hydrogen_group)
        self.hydrogen_mode_combo = QComboBox()
        self.hydrogen_mode_combo.addItem(
            "Leave unassigned (Recommended)",
            "leave_unassigned",
        )
        self.hydrogen_mode_combo.addItem(
            "Assign orphaned hydrogen",
            "assign_orphaned",
        )
        self.hydrogen_mode_combo.addItem(
            "Restore missing hydrogen",
            "restore_missing",
        )
        self.hydrogen_mode_combo.currentIndexChanged.connect(
            lambda _index: self.settings_changed.emit()
        )
        hydrogen_layout.addRow(
            "Missing hydrogen mode", self.hydrogen_mode_combo
        )
        layout.addWidget(hydrogen_group)
        self.setLayout(layout)

    def _build_free_atoms_group(self) -> QWidget:
        group = QGroupBox("Free Atoms")
        layout = QVBoxLayout(group)

        controls = QGridLayout()
        controls.setHorizontalSpacing(8)
        controls.setVerticalSpacing(6)
        self.free_element_combo = QComboBox()
        self.free_element_combo.setToolTip(
            "Elements detected in the analyzed sample frame that may be treated as free atoms."
        )
        self.free_residue_edit = QLineEdit()
        self.free_residue_edit.setPlaceholderText("SOL")
        self.free_residue_edit.setToolTip(
            "Residue code written into the exported PDB for this free atom. "
            "Must be exactly three capital letters."
        )
        _configure_residue_line_edit(self.free_residue_edit)
        add_button = QPushButton("Add Free Atom")
        add_button.clicked.connect(
            lambda _checked=False: self._add_free_atom()
        )
        remove_button = QPushButton("Remove Selected")
        remove_button.clicked.connect(
            lambda _checked=False: self._remove_selected_free_atom()
        )
        controls.addWidget(QLabel("Element"), 0, 0)
        controls.addWidget(self.free_element_combo, 0, 1)
        controls.addWidget(QLabel("Residue"), 0, 2)
        controls.addWidget(self.free_residue_edit, 0, 3)
        controls.addWidget(add_button, 1, 0)
        controls.addWidget(remove_button, 1, 1)
        controls.setColumnStretch(1, 1)
        controls.setColumnStretch(3, 1)
        layout.addLayout(controls)

        self.free_atom_table = QTableWidget(0, 2)
        self.free_atom_table.setHorizontalHeaderLabels(["Element", "Residue"])
        self.free_atom_table.verticalHeader().setVisible(False)
        self.free_atom_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.free_atom_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.free_atom_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        header = self.free_atom_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.free_atom_table.setMinimumHeight(150)
        layout.addWidget(self.free_atom_table, stretch=1)
        return group

    def _build_molecules_group(self) -> QWidget:
        group = QGroupBox("Reference Molecules")
        layout = QVBoxLayout(group)

        controls = QGridLayout()
        controls.setHorizontalSpacing(8)
        controls.setVerticalSpacing(6)
        self.reference_combo = QComboBox()
        self.reference_combo.currentIndexChanged.connect(
            self._on_reference_selection_changed
        )
        self.reference_combo.setToolTip(
            "Choose the reference molecule definition to match against the "
            "XYZ frame. Its registered residue name is used to prefill the "
            "residue field below."
        )
        self.molecule_residue_edit = QLineEdit()
        self.molecule_residue_edit.setPlaceholderText("DMF")
        self.molecule_residue_edit.setToolTip(
            "Residue name written into the exported PDB for this reference. "
            "It is auto-filled from the reference library entry, but you can "
            "override it with another three-letter residue code if needed. "
            "Residue codes must be exactly three capital letters."
        )
        _configure_residue_line_edit(self.molecule_residue_edit)
        self.tight_scale_spin = QDoubleSpinBox()
        self.tight_scale_spin.setDecimals(1)
        self.tight_scale_spin.setRange(1.0, 500.0)
        self.tight_scale_spin.setSingleStep(5.0)
        self.tight_scale_spin.setSuffix(" %")
        self.tight_scale_spin.setValue(85.0)
        self.tight_scale_spin.setToolTip(
            "Percentage multiplier applied to each bond's base tolerance "
            "percentage in the first matching pass. 100% keeps the bond-table "
            "value unchanged; lower values make the first pass stricter."
        )
        self.tight_scale_spin.valueChanged.connect(
            self._on_bond_range_controls_changed
        )
        self.relaxed_scale_spin = QDoubleSpinBox()
        self.relaxed_scale_spin.setDecimals(1)
        self.relaxed_scale_spin.setRange(1.0, 500.0)
        self.relaxed_scale_spin.setSingleStep(5.0)
        self.relaxed_scale_spin.setSuffix(" %")
        self.relaxed_scale_spin.setValue(135.0)
        self.relaxed_scale_spin.setToolTip(
            "Fallback percentage multiplier used only after the tight "
            "full-hydrogen pass fails. Values above 100% widen the bond-table "
            "percent tolerances for the relaxed pass."
        )
        self.relaxed_scale_spin.valueChanged.connect(
            self._on_bond_range_controls_changed
        )
        self.max_missing_h_spin = QSpinBox()
        self.max_missing_h_spin.setRange(0, 8)
        self.max_missing_h_spin.setValue(0)
        self.max_missing_h_spin.setToolTip(
            "Maximum number of reference hydrogens that may be omitted after "
            "both full-hydrogen passes fail. Leave this at 0 to avoid "
            "assuming deprotonation unless you want to test for it explicitly."
        )

        add_button = QPushButton("Add Molecule")
        add_button.clicked.connect(lambda _checked=False: self._add_molecule())
        update_button = QPushButton("Update Selected")
        update_button.clicked.connect(
            lambda _checked=False: self._update_selected_molecule()
        )
        remove_button = QPushButton("Remove Selected")
        remove_button.clicked.connect(
            lambda _checked=False: self._remove_selected_molecule()
        )

        reference_label = QLabel("Reference")
        reference_label.setToolTip(self.reference_combo.toolTip())
        controls.addWidget(reference_label, 0, 0)
        controls.addWidget(self.reference_combo, 0, 1)
        residue_label = QLabel("Residue")
        residue_label.setToolTip(self.molecule_residue_edit.toolTip())
        controls.addWidget(residue_label, 0, 2)
        controls.addWidget(self.molecule_residue_edit, 0, 3)
        missing_h_label = QLabel("Missing H")
        missing_h_label.setToolTip(self.max_missing_h_spin.toolTip())
        controls.addWidget(missing_h_label, 0, 4)
        controls.addWidget(self.max_missing_h_spin, 0, 5)
        tight_label = QLabel("Tight")
        tight_label.setToolTip(self.tight_scale_spin.toolTip())
        controls.addWidget(tight_label, 1, 0)
        controls.addWidget(self.tight_scale_spin, 1, 1)
        relaxed_label = QLabel("Relaxed")
        relaxed_label.setToolTip(self.relaxed_scale_spin.toolTip())
        controls.addWidget(relaxed_label, 1, 2)
        controls.addWidget(self.relaxed_scale_spin, 1, 3)
        controls.addWidget(add_button, 1, 4)
        controls.addWidget(update_button, 1, 5)
        controls.addWidget(remove_button, 1, 6)
        controls.setColumnStretch(1, 1)
        controls.setColumnStretch(3, 1)
        layout.addLayout(controls)

        self.molecule_table = QTableWidget(0, 6)
        self.molecule_table.setHorizontalHeaderLabels(
            [
                "Reference",
                "Residue",
                "Bonds",
                "Tight %",
                "Relaxed %",
                "Missing H",
            ]
        )
        self.molecule_table.verticalHeader().setVisible(False)
        self.molecule_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.molecule_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.molecule_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.molecule_table.itemSelectionChanged.connect(
            self._on_selected_molecule_changed
        )
        header = self.molecule_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
        self.molecule_table.setMinimumHeight(220)
        layout.addWidget(self.molecule_table, stretch=2)

        bond_label = QLabel("Direct Bond Tolerances")
        bond_label.setToolTip(
            "Per-bond tolerance percentages. Each percentage is applied to "
            "that bond's reference length to produce the actual absolute "
            "tolerance used during matching. The table also shows the "
            "reference bond length and the tight/relaxed min-max search "
            "windows that will be used for each bond."
        )
        layout.addWidget(bond_label)
        self.bond_table = QTableWidget(0, 8)
        self.bond_table.setHorizontalHeaderLabels(
            [
                "Atom 1",
                "Atom 2",
                "Ref (A)",
                "Tolerance (%)",
                "Tight Min (A)",
                "Tight Max (A)",
                "Relaxed Min (A)",
                "Relaxed Max (A)",
            ]
        )
        self.bond_table.setToolTip(bond_label.toolTip())
        self.bond_table.verticalHeader().setVisible(False)
        self.bond_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.bond_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.bond_table.itemChanged.connect(self._on_bond_table_changed)
        bond_header = self.bond_table.horizontalHeader()
        for column in range(self.bond_table.columnCount()):
            bond_header.setSectionResizeMode(
                column,
                QHeaderView.ResizeMode.ResizeToContents,
            )
        self.bond_table.setMinimumHeight(220)
        layout.addWidget(self.bond_table, stretch=2)
        return group

    def set_available_elements(self, elements: Sequence[str]) -> None:
        current = self.free_element_combo.currentData()
        self.free_element_combo.blockSignals(True)
        self.free_element_combo.clear()
        for element in elements:
            self.free_element_combo.addItem(str(element), str(element))
        if current is not None:
            index = self.free_element_combo.findData(current)
            if index >= 0:
                self.free_element_combo.setCurrentIndex(index)
        self.free_element_combo.blockSignals(False)

    def set_reference_entries(
        self,
        entries: Sequence[ReferenceLibraryEntry],
        *,
        bond_defaults_by_name: (
            dict[str, tuple[ReferenceBondToleranceInput, ...]] | None
        ) = None,
    ) -> None:
        if bond_defaults_by_name is not None:
            self._reference_bond_defaults = dict(bond_defaults_by_name)
        self._reference_residue_defaults = {
            entry.name: entry.residue_name for entry in entries
        }
        self._reference_paths = {
            entry.name: str(entry.path) for entry in entries
        }
        self._reference_atom_coordinates.clear()
        current = self.reference_combo.currentData()
        self.reference_combo.blockSignals(True)
        self.reference_combo.clear()
        for entry in entries:
            self.reference_combo.addItem(entry.name, entry.name)
        if current is not None:
            index = self.reference_combo.findData(current)
            if index >= 0:
                self.reference_combo.setCurrentIndex(index)
        self.reference_combo.blockSignals(False)
        self._apply_selected_reference_defaults(force=False)
        self._load_default_bonds_for_reference()

    def get_free_atom_inputs(self) -> list[FreeAtomMappingInput]:
        result: list[FreeAtomMappingInput] = []
        for row in range(self.free_atom_table.rowCount()):
            element_item = self.free_atom_table.item(row, 0)
            residue_item = self.free_atom_table.item(row, 1)
            if element_item is None or residue_item is None:
                continue
            result.append(
                FreeAtomMappingInput(
                    element=element_item.text().strip(),
                    residue_name=residue_item.text().strip(),
                )
            )
        return result

    def get_molecule_inputs(self) -> list[MoleculeMappingInput]:
        return list(self._molecule_inputs)

    def hydrogen_mode(self) -> str:
        return str(
            self.hydrogen_mode_combo.currentData() or "leave_unassigned"
        )

    def _add_free_atom(self) -> None:
        element = str(self.free_element_combo.currentData() or "").strip()
        residue = self.free_residue_edit.text().strip()
        if not element:
            self._warn("Choose an element to add as a free atom.")
            return
        if not residue:
            self._warn("Enter a three-letter residue for the free atom.")
            return
        if not _is_valid_residue_code(residue):
            self._warn(
                "Free-atom residues must be exactly three capital letters."
            )
            return
        for row in range(self.free_atom_table.rowCount()):
            item = self.free_atom_table.item(row, 0)
            if item is not None and item.text().strip() == element:
                self._warn(f"{element} is already listed as a free atom.")
                return
        row = self.free_atom_table.rowCount()
        self.free_atom_table.insertRow(row)
        self.free_atom_table.setItem(row, 0, QTableWidgetItem(element))
        self.free_atom_table.setItem(row, 1, QTableWidgetItem(residue))
        self.settings_changed.emit()

    def _remove_selected_free_atom(self) -> None:
        row = self.free_atom_table.currentRow()
        if row < 0:
            return
        self.free_atom_table.removeRow(row)
        self.settings_changed.emit()

    def _add_molecule(self) -> None:
        molecule = self._molecule_from_controls(use_selected_bonds=False)
        if molecule is None:
            return
        self._molecule_inputs.append(molecule)
        self._refresh_molecule_table()
        self.molecule_table.selectRow(
            max(self.molecule_table.rowCount() - 1, 0)
        )
        self.settings_changed.emit()

    def _update_selected_molecule(self) -> None:
        row = self.molecule_table.currentRow()
        if row < 0:
            self._warn("Select a molecule row to update it.")
            return
        molecule = self._molecule_from_controls(use_selected_bonds=True)
        if molecule is None:
            return
        self._molecule_inputs[row] = molecule
        self._refresh_molecule_table()
        self.molecule_table.selectRow(row)
        self.settings_changed.emit()

    def _remove_selected_molecule(self) -> None:
        row = self.molecule_table.currentRow()
        if row < 0:
            return
        del self._molecule_inputs[row]
        self._refresh_molecule_table()
        if self.molecule_table.rowCount():
            self.molecule_table.selectRow(
                min(row, self.molecule_table.rowCount() - 1)
            )
        else:
            self._populate_bond_table(())
        self.settings_changed.emit()

    def _molecule_from_controls(
        self,
        *,
        use_selected_bonds: bool,
    ) -> MoleculeMappingInput | None:
        reference_name = str(self.reference_combo.currentData() or "").strip()
        residue_name = self.molecule_residue_edit.text().strip()
        if not reference_name:
            self._warn("Choose a reference molecule first.")
            return None
        if not residue_name:
            self._warn("Enter a three-letter residue for the molecule.")
            return None
        if not _is_valid_residue_code(residue_name):
            self._warn(
                "Reference-molecule residues must be exactly three capital letters."
            )
            return None
        bond_tolerances = (
            self._bond_inputs_from_table()
            if use_selected_bonds and self.bond_table.rowCount()
            else tuple(self._reference_bond_defaults.get(reference_name, ()))
        )
        return MoleculeMappingInput(
            reference_name=reference_name,
            residue_name=residue_name,
            bond_tolerances=bond_tolerances,
            tight_pass_scale=float(self.tight_scale_spin.value()) / 100.0,
            relaxed_pass_scale=float(self.relaxed_scale_spin.value()) / 100.0,
            max_missing_hydrogens=int(self.max_missing_h_spin.value()),
        )

    def _refresh_molecule_table(self) -> None:
        self.molecule_table.setRowCount(0)
        for row, molecule in enumerate(self._molecule_inputs):
            self.molecule_table.insertRow(row)
            values = (
                molecule.reference_name,
                molecule.residue_name,
                str(len(molecule.bond_tolerances)),
                f"{molecule.tight_pass_scale * 100.0:.1f}%",
                f"{molecule.relaxed_pass_scale * 100.0:.1f}%",
                str(int(molecule.max_missing_hydrogens)),
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column != 0:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.molecule_table.setItem(row, column, item)

    def _on_selected_molecule_changed(self) -> None:
        row = self.molecule_table.currentRow()
        if row < 0 or row >= len(self._molecule_inputs):
            self._populate_bond_table(())
            return
        molecule = self._molecule_inputs[row]
        self._set_combo_value(self.reference_combo, molecule.reference_name)
        self.molecule_residue_edit.setText(molecule.residue_name)
        self.tight_scale_spin.setValue(
            float(molecule.tight_pass_scale) * 100.0
        )
        self.relaxed_scale_spin.setValue(
            float(molecule.relaxed_pass_scale) * 100.0
        )
        self.max_missing_h_spin.setValue(int(molecule.max_missing_hydrogens))
        self._populate_bond_table(molecule.bond_tolerances)

    def _on_reference_selection_changed(self, _index: int) -> None:
        self._apply_selected_reference_defaults(force=True)
        self._load_default_bonds_for_reference()

    def _apply_selected_reference_defaults(self, *, force: bool) -> None:
        reference_name = str(self.reference_combo.currentData() or "").strip()
        default_residue_name = self._reference_residue_defaults.get(
            reference_name
        )
        if not default_residue_name:
            return
        current_residue_name = self.molecule_residue_edit.text().strip()
        should_replace = force or not current_residue_name
        if (
            not should_replace
            and self._last_autofilled_residue_name is not None
            and current_residue_name == self._last_autofilled_residue_name
        ):
            should_replace = True
        if not should_replace:
            return
        self.molecule_residue_edit.setText(default_residue_name)
        self._last_autofilled_residue_name = default_residue_name

    def _load_default_bonds_for_reference(self) -> None:
        if self.molecule_table.currentRow() >= 0:
            return
        reference_name = str(self.reference_combo.currentData() or "").strip()
        self._populate_bond_table(
            self._reference_bond_defaults.get(reference_name, ())
        )

    def _populate_bond_table(
        self,
        bond_inputs: Sequence[ReferenceBondToleranceInput],
    ) -> None:
        reference_name = str(self.reference_combo.currentData() or "").strip()
        self._populating_bond_table = True
        self.bond_table.setRowCount(0)
        for row, bond in enumerate(bond_inputs):
            reference_length = self._reference_bond_length(
                reference_name,
                bond.atom1_name,
                bond.atom2_name,
            )
            self.bond_table.insertRow(row)
            self.bond_table.setItem(
                row,
                0,
                self._readonly_table_item(bond.atom1_name),
            )
            self.bond_table.setItem(
                row,
                1,
                self._readonly_table_item(bond.atom2_name),
            )
            reference_item = self._readonly_table_item(
                ""
                if reference_length is None
                else f"{float(reference_length):.3f}"
            )
            reference_item.setData(
                Qt.ItemDataRole.UserRole,
                reference_length,
            )
            self.bond_table.setItem(row, 2, reference_item)
            tolerance_item = QTableWidgetItem(f"{float(bond.tolerance):.2f}")
            tolerance_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.bond_table.setItem(row, 3, tolerance_item)
            for column in range(4, self.bond_table.columnCount()):
                self.bond_table.setItem(
                    row, column, self._readonly_table_item("")
                )
        self._populating_bond_table = False
        self._refresh_bond_table_range_columns()

    def _bond_inputs_from_table(
        self,
    ) -> tuple[ReferenceBondToleranceInput, ...]:
        result: list[ReferenceBondToleranceInput] = []
        for row in range(self.bond_table.rowCount()):
            atom1_item = self.bond_table.item(row, 0)
            atom2_item = self.bond_table.item(row, 1)
            tolerance_item = self.bond_table.item(row, 3)
            if (
                atom1_item is None
                or atom2_item is None
                or tolerance_item is None
            ):
                continue
            try:
                tolerance = float(tolerance_item.text().strip())
            except ValueError:
                tolerance = _FALLBACK_BOND_TOLERANCE_PERCENT
            result.append(
                ReferenceBondToleranceInput(
                    atom1_name=atom1_item.text().strip(),
                    atom2_name=atom2_item.text().strip(),
                    tolerance=tolerance,
                )
            )
        return tuple(result)

    def _on_bond_table_changed(self, _item: QTableWidgetItem) -> None:
        if self._populating_bond_table:
            return
        self._refresh_bond_table_range_columns()
        row = self.molecule_table.currentRow()
        if row < 0 or row >= len(self._molecule_inputs):
            return
        molecule = self._molecule_inputs[row]
        self._molecule_inputs[row] = MoleculeMappingInput(
            reference_name=molecule.reference_name,
            residue_name=molecule.residue_name,
            bond_tolerances=self._bond_inputs_from_table(),
            tight_pass_scale=molecule.tight_pass_scale,
            relaxed_pass_scale=molecule.relaxed_pass_scale,
            max_assignment_distance=molecule.max_assignment_distance,
            max_missing_hydrogens=molecule.max_missing_hydrogens,
        )
        self._refresh_molecule_table()
        self.molecule_table.selectRow(row)
        self.settings_changed.emit()

    def _on_bond_range_controls_changed(self, _value: float) -> None:
        self._refresh_bond_table_range_columns()

    def _refresh_bond_table_range_columns(self) -> None:
        if not self.bond_table.rowCount():
            return
        tight_scale = float(self.tight_scale_spin.value()) / 100.0
        relaxed_scale = float(self.relaxed_scale_spin.value()) / 100.0
        self._populating_bond_table = True
        for row in range(self.bond_table.rowCount()):
            reference_item = self.bond_table.item(row, 2)
            tolerance_item = self.bond_table.item(row, 3)
            if reference_item is None or tolerance_item is None:
                continue
            reference_length = reference_item.data(Qt.ItemDataRole.UserRole)
            try:
                resolved_reference_length = float(reference_length)
            except (TypeError, ValueError):
                resolved_reference_length = None
            try:
                tolerance_percent = float(tolerance_item.text().strip())
            except ValueError:
                tolerance_percent = _FALLBACK_BOND_TOLERANCE_PERCENT
            if resolved_reference_length is None:
                values = ("", "", "", "")
            else:
                tight_min, tight_max = self._bond_search_bounds(
                    resolved_reference_length,
                    tolerance_percent,
                    tight_scale,
                )
                relaxed_min, relaxed_max = self._bond_search_bounds(
                    resolved_reference_length,
                    tolerance_percent,
                    relaxed_scale,
                )
                values = (
                    f"{tight_min:.3f}",
                    f"{tight_max:.3f}",
                    f"{relaxed_min:.3f}",
                    f"{relaxed_max:.3f}",
                )
            for column, value in enumerate(values, start=4):
                item = self.bond_table.item(row, column)
                if item is None:
                    item = self._readonly_table_item(value)
                    self.bond_table.setItem(row, column, item)
                else:
                    item.setText(value)
        self._populating_bond_table = False

    def _reference_bond_length(
        self,
        reference_name: str,
        atom1_name: str,
        atom2_name: str,
    ) -> float | None:
        atom_coordinates = self._reference_atom_coordinates.get(reference_name)
        if atom_coordinates is None:
            reference_path = self._reference_paths.get(reference_name)
            if not reference_path:
                return None
            structure = PDBStructure.from_file(reference_path)
            atom_coordinates = {}
            for index, atom in enumerate(structure.atoms, start=1):
                fallback = f"{atom.element}{index}"
                atom_coordinates[
                    _normalized_table_atom_name(
                        atom.atom_name,
                        fallback=fallback,
                    )
                ] = tuple(float(value) for value in atom.coordinates)
            self._reference_atom_coordinates[reference_name] = atom_coordinates
        coord1 = atom_coordinates.get(
            _normalized_table_atom_name(atom1_name, fallback="A1")
        )
        coord2 = atom_coordinates.get(
            _normalized_table_atom_name(atom2_name, fallback="A2")
        )
        if coord1 is None or coord2 is None:
            return None
        return float(dist(coord1, coord2))

    def _bond_search_bounds(
        self,
        reference_length: float,
        tolerance_percent: float,
        pass_scale: float,
    ) -> tuple[float, float]:
        absolute_tolerance = (
            max(float(reference_length), 0.0)
            * max(float(tolerance_percent), 0.0)
            / 100.0
        )
        scaled_tolerance = absolute_tolerance * max(float(pass_scale), 0.0)
        minimum = max(float(reference_length) - scaled_tolerance, 0.0)
        maximum = float(reference_length) + scaled_tolerance
        return minimum, maximum

    def _readonly_table_item(self, text: str) -> QTableWidgetItem:
        item = QTableWidgetItem(text)
        item.setFlags(Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        return item

    def _set_combo_value(self, combo: QComboBox, value: str) -> None:
        index = combo.findData(value)
        if index < 0:
            return
        combo.blockSignals(True)
        combo.setCurrentIndex(index)
        combo.blockSignals(False)

    def _warn(self, message: str) -> None:
        QMessageBox.warning(self, "Mapping Definitions", message)


def _normalized_table_atom_name(value: str, *, fallback: str) -> str:
    text = re.sub(r"[^A-Za-z0-9]", "", str(value or "")).strip().upper()
    if not text:
        text = fallback.upper()
    return text[:4]


def _configure_residue_line_edit(line_edit: QLineEdit) -> None:
    line_edit.setMaxLength(3)
    line_edit.setValidator(
        QRegularExpressionValidator(
            QRegularExpression(r"[A-Z]{0,3}"),
            line_edit,
        )
    )
    line_edit.textChanged.connect(
        lambda text, edit=line_edit: _normalize_residue_line_edit(edit, text)
    )


def _normalize_residue_line_edit(line_edit: QLineEdit, text: str) -> None:
    normalized = re.sub(r"[^A-Za-z]", "", str(text or "")).upper()[:3]
    if normalized == text:
        return
    cursor_position = min(line_edit.cursorPosition(), len(normalized))
    line_edit.blockSignals(True)
    line_edit.setText(normalized)
    line_edit.setCursorPosition(cursor_position)
    line_edit.blockSignals(False)


def _is_valid_residue_code(value: str) -> bool:
    return re.fullmatch(r"[A-Z]{3}", str(value or "").strip()) is not None
