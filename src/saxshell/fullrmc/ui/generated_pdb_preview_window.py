from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg,
    NavigationToolbar2QT,
)
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)

from saxshell.fullrmc.solvent_handling import GeneratedPDBInspection
from saxshell.structure import PDBAtom, PDBStructure

_ELEMENT_PAIR_CUTOFFS = {
    ("Pb", "Pb"): 5.0,
    ("Pb", "I"): 4.0,
    ("Pb", "O"): 4.0,
    ("Pb", "C"): 5.0,
    ("Pb", "N"): 5.0,
    ("I", "I"): 4.0,
    ("I", "O"): 3.5,
    ("I", "C"): 3.8,
    ("I", "N"): 3.8,
}
_ELEMENT_CUTOFFS = {
    "H": 1.3,
    "C": 1.8,
    "N": 1.7,
    "O": 1.6,
    "S": 2.0,
    "P": 2.0,
    "Pb": 3.5,
    "I": 3.2,
}
_ELEMENT_COLORS = {
    "H": "#d9d9d9",
    "C": "#4f7d5c",
    "N": "#386cb0",
    "O": "#d94841",
    "S": "#b38f00",
    "P": "#7b5ea7",
    "I": "#7f7f7f",
    "Pb": "#8c564b",
}


class GeneratedPDBPreviewWindow(QMainWindow):
    def __init__(
        self,
        inspection: GeneratedPDBInspection,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.inspection = inspection
        self.structure = self._load_structure()
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.setWindowFlag(Qt.WindowType.Window, True)
        self.setWindowTitle(
            f"Generated PDB Preview - {self.inspection.file_name}"
        )
        self.resize(1100, 860)
        self._build_ui()
        self.refresh_plot()

    def _build_ui(self) -> None:
        central = QWidget(self)
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        header = QLabel(
            " | ".join(
                [
                    f"Representative: {self.inspection.representative_label}",
                    f"Variant: {self.inspection.variant_label}",
                    f"Atoms: {self.inspection.atom_count}",
                    (
                        "Solvent molecules: "
                        f"{self.inspection.solvent_molecule_count}"
                    ),
                ]
            )
        )
        header.setWordWrap(True)
        root.addWidget(header)

        controls_row = QHBoxLayout()
        controls_row.addWidget(QLabel("Visible structure layers:"))
        self.show_solute_atoms_checkbox = QCheckBox("Solute atoms")
        self.show_solute_atoms_checkbox.setChecked(True)
        controls_row.addWidget(self.show_solute_atoms_checkbox)
        self.show_solute_network_checkbox = QCheckBox("Solute network")
        self.show_solute_network_checkbox.setChecked(True)
        controls_row.addWidget(self.show_solute_network_checkbox)
        self.show_solvent_atoms_checkbox = QCheckBox("Solvent atoms")
        self.show_solvent_atoms_checkbox.setChecked(False)
        controls_row.addWidget(self.show_solvent_atoms_checkbox)
        self.show_solvent_network_checkbox = QCheckBox("Solvent network")
        self.show_solvent_network_checkbox.setChecked(False)
        controls_row.addWidget(self.show_solvent_network_checkbox)
        controls_row.addStretch(1)
        root.addLayout(controls_row)
        self.show_solute_atoms_checkbox.toggled.connect(self.refresh_plot)
        self.show_solute_network_checkbox.toggled.connect(self.refresh_plot)
        self.show_solvent_atoms_checkbox.toggled.connect(self.refresh_plot)
        self.show_solvent_network_checkbox.toggled.connect(self.refresh_plot)

        self.figure = Figure(figsize=(9.5, 7.2))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        root.addWidget(self.toolbar)
        root.addWidget(self.canvas, stretch=1)

        self.details_box = QPlainTextEdit()
        self.details_box.setReadOnly(True)
        self.details_box.setMinimumHeight(220)
        self.details_box.setPlainText(self.inspection.details_text())
        root.addWidget(self.details_box)

        self.setCentralWidget(central)

    def refresh_plot(self) -> None:
        self.figure.clear()
        if self.structure is None or not self.structure.atoms:
            axis = self.figure.add_subplot(111)
            axis.text(
                0.5,
                0.5,
                "No PDB coordinates are available for preview.",
                ha="center",
                va="center",
                wrap=True,
                transform=axis.transAxes,
            )
            axis.set_axis_off()
            self.canvas.draw_idle()
            return

        axis = self.figure.add_subplot(111, projection="3d")
        solvent_atom_ids = _solvent_atom_ids(
            self.structure,
            self.inspection.reference_residue_name,
        )
        solvent_bonds = _bond_pairs(
            [
                atom
                for atom in self.structure.atoms
                if atom.atom_id in solvent_atom_ids
            ],
            restrict_to_same_residue=True,
        )
        solute_bonds = _bond_pairs(
            [
                atom
                for atom in self.structure.atoms
                if atom.atom_id not in solvent_atom_ids
            ],
            restrict_to_same_residue=False,
        )

        atom_lookup = {atom.atom_id: atom for atom in self.structure.atoms}
        if self.show_solute_network_checkbox.isChecked():
            for atom_id_a, atom_id_b in solute_bonds:
                atom_a = atom_lookup[atom_id_a]
                atom_b = atom_lookup[atom_id_b]
                axis.plot(
                    [atom_a.coordinates[0], atom_b.coordinates[0]],
                    [atom_a.coordinates[1], atom_b.coordinates[1]],
                    [atom_a.coordinates[2], atom_b.coordinates[2]],
                    color="#355070",
                    linewidth=1.3,
                    alpha=0.7,
                )
        if self.show_solvent_network_checkbox.isChecked():
            for atom_id_a, atom_id_b in solvent_bonds:
                atom_a = atom_lookup[atom_id_a]
                atom_b = atom_lookup[atom_id_b]
                axis.plot(
                    [atom_a.coordinates[0], atom_b.coordinates[0]],
                    [atom_a.coordinates[1], atom_b.coordinates[1]],
                    [atom_a.coordinates[2], atom_b.coordinates[2]],
                    color="#bc6c25",
                    linewidth=1.1,
                    alpha=0.8,
                    linestyle="--",
                )

        unique_elements = sorted(
            {atom.element for atom in self.structure.atoms}
        )
        for element in unique_elements:
            solute_atoms = [
                atom
                for atom in self.structure.atoms
                if atom.element == element
                and atom.atom_id not in solvent_atom_ids
            ]
            solvent_atoms = [
                atom
                for atom in self.structure.atoms
                if atom.element == element and atom.atom_id in solvent_atom_ids
            ]
            color = _ELEMENT_COLORS.get(element, "#444444")
            if solute_atoms and self.show_solute_atoms_checkbox.isChecked():
                coords = np.asarray(
                    [atom.coordinates for atom in solute_atoms],
                    dtype=float,
                )
                axis.scatter(
                    coords[:, 0],
                    coords[:, 1],
                    coords[:, 2],
                    color=color,
                    s=52,
                    edgecolor="black",
                    linewidths=0.4,
                    marker="o",
                )
            if solvent_atoms and self.show_solvent_atoms_checkbox.isChecked():
                coords = np.asarray(
                    [atom.coordinates for atom in solvent_atoms],
                    dtype=float,
                )
                axis.scatter(
                    coords[:, 0],
                    coords[:, 1],
                    coords[:, 2],
                    color=color,
                    s=68,
                    edgecolor="black",
                    linewidths=0.5,
                    marker="^",
                )

        all_coords = np.asarray(
            [atom.coordinates for atom in self.structure.atoms],
            dtype=float,
        )
        _set_equal_3d_limits(axis, all_coords)
        axis.set_xlabel("X (A)")
        axis.set_ylabel("Y (A)")
        axis.set_zlabel("Z (A)")
        axis.set_title(
            f"{self.inspection.representative_label} - {self.inspection.variant_label}"
        )

        legend_handles = []
        if self.show_solute_network_checkbox.isChecked() and solute_bonds:
            legend_handles.append(
                Line2D(
                    [],
                    [],
                    color="#355070",
                    linewidth=1.4,
                    label="Solute coordination network",
                )
            )
        if self.show_solvent_network_checkbox.isChecked() and solvent_bonds:
            legend_handles.append(
                Line2D(
                    [],
                    [],
                    color="#bc6c25",
                    linewidth=1.2,
                    linestyle="--",
                    label="Solvent-molecule connectivity",
                )
            )
        if self.show_solute_atoms_checkbox.isChecked():
            legend_handles.append(
                Line2D(
                    [],
                    [],
                    marker="o",
                    color="black",
                    linestyle="None",
                    markerfacecolor="white",
                    label="Solute atoms",
                )
            )
        if self.show_solvent_atoms_checkbox.isChecked():
            legend_handles.append(
                Line2D(
                    [],
                    [],
                    marker="^",
                    color="black",
                    linestyle="None",
                    markerfacecolor="white",
                    label="Solvent atoms",
                )
            )
        visible_elements = set()
        if self.show_solute_atoms_checkbox.isChecked():
            visible_elements.update(
                atom.element
                for atom in self.structure.atoms
                if atom.atom_id not in solvent_atom_ids
            )
        if self.show_solvent_atoms_checkbox.isChecked():
            visible_elements.update(
                atom.element
                for atom in self.structure.atoms
                if atom.atom_id in solvent_atom_ids
            )
        for element in unique_elements:
            if element not in visible_elements:
                continue
            legend_handles.append(
                Line2D(
                    [],
                    [],
                    marker="o",
                    color="black",
                    linestyle="None",
                    markerfacecolor=_ELEMENT_COLORS.get(element, "#444444"),
                    label=element,
                )
            )
        if legend_handles:
            axis.legend(
                handles=legend_handles,
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                borderaxespad=0.0,
                frameon=False,
            )
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _load_structure(self) -> PDBStructure | None:
        if (
            not self.inspection.exists
            or self.inspection.load_error is not None
        ):
            return None
        path = Path(self.inspection.file_path)
        if not path.is_file():
            return None
        try:
            return PDBStructure.from_file(path)
        except Exception:
            return None


def _solvent_atom_ids(
    structure: PDBStructure,
    reference_residue_name: str,
) -> set[int]:
    residue_name = reference_residue_name.upper().strip()
    if not residue_name:
        return set()
    return {
        atom.atom_id
        for atom in structure.atoms
        if atom.residue_name.upper() == residue_name
    }


def _bond_pairs(
    atoms: list[PDBAtom],
    *,
    restrict_to_same_residue: bool,
) -> list[tuple[int, int]]:
    bonds: list[tuple[int, int]] = []
    for index, atom_a in enumerate(atoms):
        for atom_b in atoms[index + 1 :]:
            if restrict_to_same_residue and (
                atom_a.residue_name != atom_b.residue_name
                or atom_a.residue_number != atom_b.residue_number
            ):
                continue
            if np.linalg.norm(
                atom_a.coordinates - atom_b.coordinates
            ) <= _cutoff_for(
                atom_a.element,
                atom_b.element,
            ):
                bonds.append((atom_a.atom_id, atom_b.atom_id))
    return bonds


def _cutoff_for(element_a: str, element_b: str) -> float:
    pair = (element_a.title(), element_b.title())
    if pair in _ELEMENT_PAIR_CUTOFFS:
        return float(_ELEMENT_PAIR_CUTOFFS[pair])
    reverse_pair = (pair[1], pair[0])
    if reverse_pair in _ELEMENT_PAIR_CUTOFFS:
        return float(_ELEMENT_PAIR_CUTOFFS[reverse_pair])
    cutoff_a = _ELEMENT_CUTOFFS.get(pair[0])
    cutoff_b = _ELEMENT_CUTOFFS.get(pair[1])
    if cutoff_a is None and cutoff_b is None:
        return 2.0
    if cutoff_a is None:
        return float(cutoff_b)
    if cutoff_b is None:
        return float(cutoff_a)
    return float(max(cutoff_a, cutoff_b))


def _set_equal_3d_limits(axis, coordinates: np.ndarray) -> None:
    mins = np.min(coordinates, axis=0)
    maxs = np.max(coordinates, axis=0)
    centers = (mins + maxs) / 2.0
    radius = max(float(np.max(maxs - mins)) / 2.0, 1.0)
    axis.set_xlim(centers[0] - radius, centers[0] + radius)
    axis.set_ylim(centers[1] - radius, centers[1] + radius)
    axis.set_zlim(centers[2] - radius, centers[2] + radius)


__all__ = ["GeneratedPDBPreviewWindow"]
