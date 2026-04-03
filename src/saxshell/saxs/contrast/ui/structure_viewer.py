from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import (
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PySide6.QtWidgets import QVBoxLayout, QWidget

from saxshell.saxs.contrast.descriptors import load_parsed_contrast_structure

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
_DEFAULT_ELEMENT_COLOR = "#4b5563"
_MESH_FACE_COLOR = "#7fd3f7"
_MESH_EDGE_COLOR = "#1d4ed8"
_MESH_ALPHA = 0.24


@dataclass(slots=True, frozen=True)
class ContrastRepresentativePreview:
    file_path: Path
    display_label: str
    coordinates: np.ndarray
    elements: tuple[str, ...]
    element_counts: dict[str, int]
    mesh_vertices: tuple[tuple[float, float, float], ...] | None = None
    mesh_faces: tuple[tuple[int, int, int], ...] | None = None
    mesh_volume_a3: float | None = None
    mesh_surface_area_a2: float | None = None
    mesh_json_path: Path | None = None
    density_json_path: Path | None = None
    cluster_density_e_per_a3: float | None = None
    solvent_density_e_per_a3: float | None = None
    contrast_density_e_per_a3: float | None = None
    notes: str | None = None

    @property
    def atom_count(self) -> int:
        return int(len(self.elements))

    @property
    def has_mesh(self) -> bool:
        return bool(self.mesh_vertices and self.mesh_faces)

    def details_text(self) -> str:
        lines = [
            "Viewer status",
            f"  Source path: {self.file_path}",
            f"  Display label: {self.display_label}",
            f"  Atom count: {self.atom_count}",
            "  Elements: "
            + (
                ", ".join(
                    f"{element} x{count}"
                    for element, count in sorted(self.element_counts.items())
                )
                or "Unavailable"
            ),
        ]
        if self.mesh_json_path is not None:
            lines.append(f"  Mesh file: {self.mesh_json_path}")
        if self.mesh_volume_a3 is not None:
            lines.append(f"  Mesh volume: {self.mesh_volume_a3:.4f} A^3")
        if self.mesh_surface_area_a2 is not None:
            lines.append(
                f"  Mesh surface area: {self.mesh_surface_area_a2:.4f} A^2"
            )
        if self.density_json_path is not None:
            lines.append(f"  Density file: {self.density_json_path}")
        if self.cluster_density_e_per_a3 is not None:
            lines.append(
                "  Cluster electron density: "
                f"{self.cluster_density_e_per_a3:.6f} e/A^3"
            )
        if self.solvent_density_e_per_a3 is not None:
            lines.append(
                "  Solvent electron density: "
                f"{self.solvent_density_e_per_a3:.6f} e/A^3"
            )
        if self.contrast_density_e_per_a3 is not None:
            lines.append(
                "  Contrast density term: "
                f"{self.contrast_density_e_per_a3:.6f} e/A^3"
            )
        if self.notes:
            lines.extend(["", "Notes", f"  {self.notes}"])
        return "\n".join(lines)


def load_contrast_representative_preview(
    file_path: str | Path,
    *,
    display_label: str,
    mesh_json_path: str | Path | None = None,
    density_json_path: str | Path | None = None,
    notes: str | None = None,
) -> ContrastRepresentativePreview:
    resolved_file_path = Path(file_path).expanduser().resolve()
    parsed = load_parsed_contrast_structure(resolved_file_path)
    resolved_mesh_path = (
        None
        if mesh_json_path is None
        else Path(mesh_json_path).expanduser().resolve()
    )
    resolved_density_path = (
        None
        if density_json_path is None
        else Path(density_json_path).expanduser().resolve()
    )

    mesh_vertices: tuple[tuple[float, float, float], ...] | None = None
    mesh_faces: tuple[tuple[int, int, int], ...] | None = None
    mesh_volume_a3: float | None = None
    mesh_surface_area_a2: float | None = None
    if resolved_mesh_path is not None and resolved_mesh_path.is_file():
        mesh_payload = json.loads(
            resolved_mesh_path.read_text(encoding="utf-8")
        )
        mesh_vertices = tuple(
            tuple(float(component) for component in vertex[:3])
            for vertex in mesh_payload.get("vertices", [])
        )
        mesh_faces = tuple(
            tuple(int(index) for index in face[:3])
            for face in mesh_payload.get("faces", [])
            if len(face) >= 3
        )
        mesh_volume_a3 = (
            None
            if mesh_payload.get("volume_a3") is None
            else float(mesh_payload["volume_a3"])
        )
        mesh_surface_area_a2 = (
            None
            if mesh_payload.get("surface_area_a2") is None
            else float(mesh_payload["surface_area_a2"])
        )

    cluster_density_e_per_a3: float | None = None
    solvent_density_e_per_a3: float | None = None
    contrast_density_e_per_a3: float | None = None
    if resolved_density_path is not None and resolved_density_path.is_file():
        density_payload = json.loads(
            resolved_density_path.read_text(encoding="utf-8")
        )
        cluster_payload = density_payload.get("cluster_electron_density", {})
        solvent_payload = density_payload.get("solvent_electron_density", {})
        if cluster_payload.get("electron_density_e_per_a3") is not None:
            cluster_density_e_per_a3 = float(
                cluster_payload["electron_density_e_per_a3"]
            )
        if solvent_payload.get("electron_density_e_per_a3") is not None:
            solvent_density_e_per_a3 = float(
                solvent_payload["electron_density_e_per_a3"]
            )
        if (
            density_payload.get("contrast_electron_density_e_per_a3")
            is not None
        ):
            contrast_density_e_per_a3 = float(
                density_payload["contrast_electron_density_e_per_a3"]
            )

    return ContrastRepresentativePreview(
        file_path=resolved_file_path,
        display_label=str(display_label).strip() or resolved_file_path.name,
        coordinates=np.asarray(parsed.coordinates, dtype=float),
        elements=tuple(parsed.elements),
        element_counts=dict(sorted(parsed.element_counts.items())),
        mesh_vertices=mesh_vertices,
        mesh_faces=mesh_faces,
        mesh_volume_a3=mesh_volume_a3,
        mesh_surface_area_a2=mesh_surface_area_a2,
        mesh_json_path=(
            resolved_mesh_path
            if resolved_mesh_path and resolved_mesh_path.is_file()
            else None
        ),
        density_json_path=(
            resolved_density_path
            if resolved_density_path and resolved_density_path.is_file()
            else None
        ),
        cluster_density_e_per_a3=cluster_density_e_per_a3,
        solvent_density_e_per_a3=solvent_density_e_per_a3,
        contrast_density_e_per_a3=contrast_density_e_per_a3,
        notes=str(notes).strip() or None,
    )


class ContrastRepresentativeViewer(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.current_preview: ContrastRepresentativePreview | None = None
        self.figure = Figure(figsize=(8.8, 6.8))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self._axis = None
        self._initial_view: dict[str, object] | None = None
        self._build_ui()
        self.draw_placeholder()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self.toolbar)
        self.canvas.setMinimumHeight(420)
        layout.addWidget(self.canvas, stretch=1)

    def draw_placeholder(self) -> None:
        self.current_preview = None
        self._axis = None
        self._initial_view = None
        self.figure.clear()
        axis = self.figure.add_subplot(111)
        axis.text(
            0.5,
            0.58,
            "Select a representative structure to preview it here.",
            ha="center",
            va="center",
            wrap=True,
            transform=axis.transAxes,
        )
        axis.text(
            0.5,
            0.43,
            "Saved retained meshes and electron-density outputs will appear as "
            "a transparent overlay when those artifacts are available.",
            ha="center",
            va="center",
            wrap=True,
            transform=axis.transAxes,
            alpha=0.75,
        )
        axis.set_axis_off()
        self.canvas.draw_idle()

    def set_preview(
        self,
        preview: ContrastRepresentativePreview | None,
        *,
        show_mesh: bool,
        show_legend: bool,
    ) -> None:
        if preview is None:
            self.draw_placeholder()
            return
        self.current_preview = preview
        self.figure.clear()
        axis = self.figure.add_subplot(111, projection="3d")
        self._axis = axis

        unique_elements = sorted(set(preview.elements))
        for element in unique_elements:
            indices = [
                index
                for index, candidate in enumerate(preview.elements)
                if candidate == element
            ]
            if not indices:
                continue
            coords = np.asarray(preview.coordinates[indices], dtype=float)
            axis.scatter(
                coords[:, 0],
                coords[:, 1],
                coords[:, 2],
                color=_ELEMENT_COLORS.get(element, _DEFAULT_ELEMENT_COLOR),
                s=96,
                edgecolor="black",
                linewidths=0.55,
                marker="o",
                depthshade=True,
            )

        if show_mesh and preview.has_mesh:
            assert preview.mesh_vertices is not None
            assert preview.mesh_faces is not None
            vertex_array = np.asarray(preview.mesh_vertices, dtype=float)
            face_vertices = [
                vertex_array[np.asarray(face, dtype=int)]
                for face in preview.mesh_faces
            ]
            axis.add_collection3d(
                Poly3DCollection(
                    face_vertices,
                    alpha=_MESH_ALPHA,
                    facecolor=_MESH_FACE_COLOR,
                    edgecolor=_MESH_EDGE_COLOR,
                    linewidths=0.7,
                )
            )

        all_coordinates = [np.asarray(preview.coordinates, dtype=float)]
        if preview.mesh_vertices:
            all_coordinates.append(
                np.asarray(preview.mesh_vertices, dtype=float)
            )
        _set_equal_3d_limits(axis, np.vstack(all_coordinates))
        axis.set_xlabel("X (A)")
        axis.set_ylabel("Y (A)")
        axis.set_zlabel("Z (A)")
        axis.set_title(preview.display_label)
        if hasattr(axis, "set_box_aspect"):
            axis.set_box_aspect((1.0, 1.0, 1.0))

        legend_handles: list[Line2D] = []
        if show_legend:
            for element in unique_elements:
                if preview.element_counts.get(element, 0) <= 0:
                    continue
                legend_handles.append(
                    Line2D(
                        [],
                        [],
                        marker="o",
                        color="black",
                        linestyle="None",
                        markerfacecolor=_ELEMENT_COLORS.get(
                            element, _DEFAULT_ELEMENT_COLOR
                        ),
                        label=f"{element} ({preview.element_counts[element]})",
                    )
                )
            if show_mesh and preview.has_mesh:
                legend_handles.append(
                    Line2D(
                        [],
                        [],
                        color=_MESH_EDGE_COLOR,
                        linewidth=1.2,
                        label="Retained volume mesh",
                    )
                )
        if legend_handles:
            axis.legend(
                handles=legend_handles,
                loc="upper right",
                frameon=True,
                facecolor="white",
                framealpha=0.88,
                fontsize="small",
            )

        self.figure.subplots_adjust(
            left=0.03,
            right=0.98,
            bottom=0.04,
            top=0.92,
        )
        self._initial_view = {
            "elev": float(axis.elev),
            "azim": float(axis.azim),
            "xlim": tuple(float(value) for value in axis.get_xlim3d()),
            "ylim": tuple(float(value) for value in axis.get_ylim3d()),
            "zlim": tuple(float(value) for value in axis.get_zlim3d()),
        }
        self.canvas.draw_idle()

    def rotate(self, delta_azim: float) -> None:
        if self._axis is None:
            return
        self._axis.view_init(
            elev=float(self._axis.elev),
            azim=float(self._axis.azim) + float(delta_azim),
        )
        self.canvas.draw_idle()

    def toggle_pan_mode(self) -> None:
        self.toolbar.pan()

    def zoom_by(self, factor: float) -> None:
        if self._axis is None or factor <= 0.0:
            return
        self._scale_axis_limits(
            self._axis.get_xlim3d(), self._axis.set_xlim3d, factor
        )
        self._scale_axis_limits(
            self._axis.get_ylim3d(), self._axis.set_ylim3d, factor
        )
        self._scale_axis_limits(
            self._axis.get_zlim3d(), self._axis.set_zlim3d, factor
        )
        self.canvas.draw_idle()

    def reset_view(self) -> None:
        if self._axis is None or self._initial_view is None:
            return
        self._axis.view_init(
            elev=float(self._initial_view["elev"]),
            azim=float(self._initial_view["azim"]),
        )
        self._axis.set_xlim3d(*self._initial_view["xlim"])
        self._axis.set_ylim3d(*self._initial_view["ylim"])
        self._axis.set_zlim3d(*self._initial_view["zlim"])
        self.canvas.draw_idle()

    @staticmethod
    def _scale_axis_limits(current_limits, setter, factor: float) -> None:
        minimum, maximum = [float(value) for value in current_limits]
        center = (minimum + maximum) / 2.0
        half_span = max((maximum - minimum) / 2.0, 0.5) * float(factor)
        setter(center - half_span, center + half_span)


def _set_equal_3d_limits(axis, coordinates: np.ndarray) -> None:
    coords = np.asarray(coordinates, dtype=float)
    if coords.size == 0:
        axis.set_xlim(-1.0, 1.0)
        axis.set_ylim(-1.0, 1.0)
        axis.set_zlim(-1.0, 1.0)
        return
    minimums = np.min(coords, axis=0)
    maximums = np.max(coords, axis=0)
    centers = (minimums + maximums) / 2.0
    radius = max(float(np.max(maximums - minimums)) / 2.0, 1.0)
    axis.set_xlim(centers[0] - radius, centers[0] + radius)
    axis.set_ylim(centers[1] - radius, centers[1] + radius)
    axis.set_zlim(centers[2] - radius, centers[2] + radius)


__all__ = [
    "ContrastRepresentativePreview",
    "ContrastRepresentativeViewer",
    "load_contrast_representative_preview",
]
