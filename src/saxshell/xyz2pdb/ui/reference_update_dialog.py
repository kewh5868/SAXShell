from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from saxshell.structure import PDBAtom, PDBStructure
from saxshell.xyz2pdb.mapping_workflow import _infer_direct_reference_bonds
from saxshell.xyz2pdb.workflow import XYZToPDBReferenceUpdateCandidate

_ELEMENT_COLORS = {
    "H": "#f5f5f5",
    "C": "#30333a",
    "N": "#4b7bec",
    "O": "#e74c3c",
    "S": "#f4b942",
    "P": "#f39c12",
    "Cl": "#28b463",
    "Br": "#8e5a3c",
    "I": "#7d3c98",
    "Pb": "#566573",
    "Na": "#1f8dd6",
}


class MoleculePreviewWidget(QWidget):
    """Small projected ball-and-stick preview for one molecule."""

    def __init__(
        self,
        atoms: Sequence[PDBAtom],
        *,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._atoms = tuple(atom.copy() for atom in atoms)
        self.setMinimumSize(220, 220)

    def minimumSizeHint(self) -> QSize:
        return QSize(220, 220)

    def sizeHint(self) -> QSize:
        return QSize(260, 260)

    def paintEvent(self, event) -> None:  # noqa: N802
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QColor("#f8f6ef"))
        preview_rect = self.rect().adjusted(12, 12, -12, -12)
        painter.setPen(QPen(QColor("#d8d2c4"), 1.0))
        painter.drawRoundedRect(preview_rect, 10, 10)
        if not self._atoms:
            painter.setPen(QColor("#6c6a63"))
            painter.drawText(
                preview_rect,
                Qt.AlignmentFlag.AlignCenter,
                "No structure",
            )
            return

        projected_atoms = _project_atoms(self._atoms, preview_rect)
        bond_items = []
        for bond in _infer_direct_reference_bonds(self._atoms):
            point1 = projected_atoms[bond.atom1_index]
            point2 = projected_atoms[bond.atom2_index]
            average_depth = (point1[2] + point2[2]) / 2.0
            bond_items.append((average_depth, point1, point2))
        for _depth, point1, point2 in sorted(
            bond_items, key=lambda item: item[0]
        ):
            painter.setPen(QPen(QColor("#9ea7b3"), 3.0, Qt.PenStyle.SolidLine))
            painter.drawLine(
                int(round(point1[0])),
                int(round(point1[1])),
                int(round(point2[0])),
                int(round(point2[1])),
            )

        atom_items = []
        for index, atom in enumerate(self._atoms):
            x_coord, y_coord, depth = projected_atoms[index]
            radius = _preview_atom_radius(atom.element)
            atom_items.append((depth, x_coord, y_coord, radius, atom))
        for _depth, x_coord, y_coord, radius, atom in sorted(
            atom_items,
            key=lambda item: item[0],
        ):
            fill_color = QColor(_ELEMENT_COLORS.get(atom.element, "#7f8c8d"))
            painter.setPen(QPen(QColor("#39434d"), 1.2))
            painter.setBrush(fill_color)
            painter.drawEllipse(
                int(round(x_coord - radius)),
                int(round(y_coord - radius)),
                int(round(radius * 2.0)),
                int(round(radius * 2.0)),
            )
            if radius >= 10:
                painter.setPen(
                    QColor("#f7f7f7" if atom.element != "H" else "#3c3c3c")
                )
                painter.drawText(
                    int(round(x_coord - radius)),
                    int(round(y_coord - radius)),
                    int(round(radius * 2.0)),
                    int(round(radius * 2.0)),
                    int(Qt.AlignmentFlag.AlignCenter),
                    atom.element,
                )


class AssertionReferenceUpdateDialog(QDialog):
    """Dialog that previews a passed assertion candidate before
    saving."""

    def __init__(
        self,
        candidate: XYZToPDBReferenceUpdateCandidate,
        *,
        versioned_reference_name: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._candidate = candidate
        self._versioned_reference_name = versioned_reference_name
        self._decision = "skip"
        self._build_ui()

    @property
    def decision(self) -> str:
        return self._decision

    def _build_ui(self) -> None:
        self.setWindowTitle(
            f"Assertion Reference Update: {self._candidate.reference_name}"
        )
        self.resize(780, 560)

        current_atoms = PDBStructure.from_file(
            self._candidate.reference_path
        ).atoms
        average_atoms = PDBStructure.from_file(
            self._candidate.average_structure_file
        ).atoms

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        intro = QLabel(
            "This residue passed assertion mode, so you can promote its "
            "average simulation geometry into the reference library."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        summary = QLabel(
            "\n".join(
                [
                    f"Reference: {self._candidate.reference_name}",
                    f"Mapped residue: {self._candidate.residue_name}",
                    f"Library residue: {self._candidate.reference_residue_name}",
                    f"Molecules averaged: {self._candidate.molecule_count}",
                    (
                        "Assertion spread: median RMSD "
                        f"{self._candidate.median_distribution_rmsd:.3f} A, "
                        f"max RMSD {self._candidate.max_distribution_rmsd:.3f} A"
                    ),
                    f"New version name: {self._versioned_reference_name}",
                ]
            )
        )
        summary.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout.addWidget(summary)

        preview_row = QHBoxLayout()
        preview_row.setSpacing(12)
        preview_row.addWidget(
            self._build_preview_group(
                "Current Reference",
                current_atoms,
                str(self._candidate.reference_path.name),
            ),
            stretch=1,
        )
        preview_row.addWidget(
            self._build_preview_group(
                "Average From Assertion",
                average_atoms,
                str(self._candidate.average_structure_file.name),
            ),
            stretch=1,
        )
        layout.addLayout(preview_row, stretch=1)

        actions = QLabel(
            "Choose one action for this residue before moving to the next candidate."
        )
        actions.setWordWrap(True)
        layout.addWidget(actions)

        button_row = QHBoxLayout()
        button_row.addStretch(1)

        skip_button = QPushButton("Skip")
        skip_button.clicked.connect(self._skip_candidate)
        button_row.addWidget(skip_button)

        version_button = QPushButton("Save New Version")
        version_button.clicked.connect(self._save_new_version)
        button_row.addWidget(version_button)

        replace_button = QPushButton("Replace Existing Reference")
        replace_button.clicked.connect(self._replace_existing_reference)
        button_row.addWidget(replace_button)

        layout.addLayout(button_row)

    def _build_preview_group(
        self,
        title: str,
        atoms: Sequence[PDBAtom],
        footer: str,
    ) -> QGroupBox:
        group = QGroupBox(title)
        group_layout = QVBoxLayout(group)
        group_layout.setContentsMargins(10, 12, 10, 10)
        group_layout.setSpacing(8)
        group_layout.addWidget(MoleculePreviewWidget(atoms), stretch=1)
        footer_label = QLabel(footer)
        footer_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        footer_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        group_layout.addWidget(footer_label)
        return group

    def _skip_candidate(self) -> None:
        self._decision = "skip"
        self.reject()

    def _save_new_version(self) -> None:
        self._decision = "save_new_version"
        self.accept()

    def _replace_existing_reference(self) -> None:
        self._decision = "replace_existing"
        self.accept()


def _project_atoms(
    atoms: Sequence[PDBAtom],
    preview_rect,
) -> list[tuple[float, float, float]]:
    coordinates = np.asarray([atom.coordinates for atom in atoms], dtype=float)
    centered = coordinates - np.mean(coordinates, axis=0)
    rotation = _preview_rotation_matrix()
    rotated = centered @ rotation.T
    xy = rotated[:, :2]
    spans = np.ptp(xy, axis=0)
    max_span = max(float(np.max(spans)), 1.0)
    scale = min(preview_rect.width(), preview_rect.height()) * 0.58 / max_span
    center_x = preview_rect.center().x()
    center_y = preview_rect.center().y()
    projected: list[tuple[float, float, float]] = []
    for x_coord, y_coord, z_coord in rotated:
        projected.append(
            (
                center_x + (float(x_coord) * scale),
                center_y - (float(y_coord) * scale),
                float(z_coord),
            )
        )
    return projected


def _preview_rotation_matrix() -> np.ndarray:
    x_angle = np.deg2rad(35.0)
    y_angle = np.deg2rad(-25.0)
    z_angle = np.deg2rad(38.0)
    rotate_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(x_angle), -np.sin(x_angle)],
            [0.0, np.sin(x_angle), np.cos(x_angle)],
        ],
        dtype=float,
    )
    rotate_y = np.array(
        [
            [np.cos(y_angle), 0.0, np.sin(y_angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(y_angle), 0.0, np.cos(y_angle)],
        ],
        dtype=float,
    )
    rotate_z = np.array(
        [
            [np.cos(z_angle), -np.sin(z_angle), 0.0],
            [np.sin(z_angle), np.cos(z_angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    return rotate_z @ rotate_y @ rotate_x


def _preview_atom_radius(element: str) -> float:
    if element == "H":
        return 6.0
    if element in {"C", "N", "O"}:
        return 10.0
    if element in {"S", "P", "Cl"}:
        return 12.0
    if element in {"I", "Pb"}:
        return 14.0
    return 9.0
