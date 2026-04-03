from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from scipy.spatial import ConvexHull
except ImportError:  # pragma: no cover - optional until runtime
    ConvexHull = None

DEFAULT_CONTRAST_MESH_PADDING_ANGSTROM = 1.5
_MINIMUM_BOX_SPAN_ANGSTROM = 1.0


def _require_convex_hull() -> None:
    if ConvexHull is None:
        raise RuntimeError(
            "scipy is required for contrast-mode mesh generation because the "
            "retained polygon volume uses scipy.spatial.ConvexHull."
        )


def _vector3(values: np.ndarray) -> tuple[float, float, float]:
    vector = np.asarray(values, dtype=float).reshape(3)
    return (float(vector[0]), float(vector[1]), float(vector[2]))


def _matrix_rows(
    values: np.ndarray,
    *,
    width: int,
) -> tuple[tuple[float, ...], ...]:
    matrix = np.asarray(values, dtype=float)
    return tuple(
        tuple(float(component) for component in row[:width]) for row in matrix
    )


@dataclass(slots=True, frozen=True)
class ContrastVolumeMesh:
    source_file: Path | None
    source_atom_count: int
    construction_method: str
    padding_angstrom: float
    centroid: tuple[float, float, float]
    source_bounds_min: tuple[float, float, float]
    source_bounds_max: tuple[float, float, float]
    source_spans: tuple[float, float, float]
    bounds_min: tuple[float, float, float]
    bounds_max: tuple[float, float, float]
    spans: tuple[float, float, float]
    volume_a3: float
    surface_area_a2: float
    vertices: tuple[tuple[float, float, float], ...]
    faces: tuple[tuple[int, int, int], ...]
    hull_equations: tuple[tuple[float, float, float, float], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "source_file": (
                None if self.source_file is None else str(self.source_file)
            ),
            "source_atom_count": self.source_atom_count,
            "construction_method": self.construction_method,
            "padding_angstrom": float(self.padding_angstrom),
            "centroid": list(self.centroid),
            "source_bounds_min": list(self.source_bounds_min),
            "source_bounds_max": list(self.source_bounds_max),
            "source_spans": list(self.source_spans),
            "bounds_min": list(self.bounds_min),
            "bounds_max": list(self.bounds_max),
            "spans": list(self.spans),
            "volume_a3": float(self.volume_a3),
            "surface_area_a2": float(self.surface_area_a2),
            "vertices": [list(vertex) for vertex in self.vertices],
            "faces": [list(face) for face in self.faces],
            "hull_equations": [list(row) for row in self.hull_equations],
        }


def _convex_hull_from_points(points: np.ndarray) -> ConvexHull:
    _require_convex_hull()
    return ConvexHull(np.asarray(points, dtype=float), qhull_options="QJ")


def _bounds(
    coordinates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coords = np.asarray(coordinates, dtype=float)
    bounds_min = np.min(coords, axis=0)
    bounds_max = np.max(coords, axis=0)
    spans = bounds_max - bounds_min
    return bounds_min, bounds_max, spans


def _mesh_from_vertices(
    vertices: np.ndarray,
    *,
    source_file: Path | None,
    source_atom_count: int,
    construction_method: str,
    padding_angstrom: float,
    centroid: np.ndarray,
    source_bounds_min: np.ndarray,
    source_bounds_max: np.ndarray,
    source_spans: np.ndarray,
) -> ContrastVolumeMesh:
    hull = _convex_hull_from_points(vertices)
    hull_vertices = np.asarray(vertices, dtype=float)
    bounds_min, bounds_max, spans = _bounds(hull_vertices)
    return ContrastVolumeMesh(
        source_file=source_file,
        source_atom_count=int(source_atom_count),
        construction_method=construction_method,
        padding_angstrom=float(padding_angstrom),
        centroid=_vector3(centroid),
        source_bounds_min=_vector3(source_bounds_min),
        source_bounds_max=_vector3(source_bounds_max),
        source_spans=_vector3(source_spans),
        bounds_min=_vector3(bounds_min),
        bounds_max=_vector3(bounds_max),
        spans=_vector3(spans),
        volume_a3=float(hull.volume),
        surface_area_a2=float(hull.area),
        vertices=_matrix_rows(hull_vertices, width=3),
        faces=tuple(
            tuple(int(index) for index in face[:3])
            for face in np.asarray(hull.simplices, dtype=int)
        ),
        hull_equations=_matrix_rows(
            np.asarray(hull.equations, dtype=float), width=4
        ),
    )


def _build_padded_box_vertices(
    coordinates: np.ndarray,
    *,
    padding_angstrom: float,
) -> np.ndarray:
    bounds_min, bounds_max, spans = _bounds(coordinates)
    half_padding = max(
        float(padding_angstrom), _MINIMUM_BOX_SPAN_ANGSTROM / 2.0
    )
    lower = np.asarray(bounds_min, dtype=float) - half_padding
    upper = np.asarray(bounds_max, dtype=float) + half_padding
    adjusted_spans = upper - lower
    for axis in range(3):
        if adjusted_spans[axis] >= _MINIMUM_BOX_SPAN_ANGSTROM:
            continue
        center = float((lower[axis] + upper[axis]) / 2.0)
        lower[axis] = center - _MINIMUM_BOX_SPAN_ANGSTROM / 2.0
        upper[axis] = center + _MINIMUM_BOX_SPAN_ANGSTROM / 2.0
    return np.asarray(
        [
            [lower[0], lower[1], lower[2]],
            [upper[0], lower[1], lower[2]],
            [upper[0], upper[1], lower[2]],
            [lower[0], upper[1], lower[2]],
            [lower[0], lower[1], upper[2]],
            [upper[0], lower[1], upper[2]],
            [upper[0], upper[1], upper[2]],
            [lower[0], upper[1], upper[2]],
        ],
        dtype=float,
    )


def build_contrast_volume_mesh(
    coordinates: np.ndarray,
    *,
    source_file: str | Path | None = None,
    padding_angstrom: float = DEFAULT_CONTRAST_MESH_PADDING_ANGSTROM,
) -> ContrastVolumeMesh:
    coords = np.asarray(coordinates, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(
            "Mesh construction requires an (N, 3) coordinate array."
        )
    if coords.shape[0] == 0:
        raise ValueError("Mesh construction requires at least one atom.")
    source_path = (
        None
        if source_file is None
        else Path(source_file).expanduser().resolve()
    )
    source_bounds_min, source_bounds_max, source_spans = _bounds(coords)
    centroid = np.mean(coords, axis=0)

    centered = coords - centroid
    if coords.shape[0] >= 4 and int(np.linalg.matrix_rank(centered)) >= 3:
        hull = _convex_hull_from_points(coords)
        hull_vertices = np.asarray(
            coords[np.asarray(hull.vertices, dtype=int)], dtype=float
        )
        expanded_vertices: list[np.ndarray] = []
        for vertex in hull_vertices:
            displacement = np.asarray(vertex - centroid, dtype=float)
            norm = float(np.linalg.norm(displacement))
            if norm <= 1e-12:
                expanded_vertices.append(
                    np.asarray(vertex, dtype=float)
                    + np.asarray(
                        [float(padding_angstrom), 0.0, 0.0], dtype=float
                    )
                )
                continue
            expanded_vertices.append(
                np.asarray(vertex, dtype=float)
                + displacement / norm * float(padding_angstrom)
            )
        expanded_array = np.asarray(expanded_vertices, dtype=float)
        if (
            expanded_array.shape[0] >= 4
            and int(
                np.linalg.matrix_rank(
                    expanded_array - np.mean(expanded_array, axis=0)
                )
            )
            >= 3
        ):
            return _mesh_from_vertices(
                expanded_array,
                source_file=source_path,
                source_atom_count=coords.shape[0],
                construction_method="expanded_convex_hull",
                padding_angstrom=float(padding_angstrom),
                centroid=centroid,
                source_bounds_min=source_bounds_min,
                source_bounds_max=source_bounds_max,
                source_spans=source_spans,
            )

    return _mesh_from_vertices(
        _build_padded_box_vertices(
            coords, padding_angstrom=float(padding_angstrom)
        ),
        source_file=source_path,
        source_atom_count=coords.shape[0],
        construction_method="padded_bounding_box",
        padding_angstrom=float(padding_angstrom),
        centroid=centroid,
        source_bounds_min=source_bounds_min,
        source_bounds_max=source_bounds_max,
        source_spans=source_spans,
    )


def translated_mesh_vertices(
    mesh: ContrastVolumeMesh,
    center: tuple[float, float, float] | np.ndarray,
) -> np.ndarray:
    target_center = np.asarray(center, dtype=float).reshape(3)
    source_center = np.asarray(mesh.centroid, dtype=float)
    vertices = np.asarray(mesh.vertices, dtype=float)
    return vertices - source_center + target_center


def points_inside_contrast_volume(
    points: np.ndarray,
    mesh: ContrastVolumeMesh,
    *,
    translated_center: tuple[float, float, float] | np.ndarray | None = None,
    tolerance: float = 1e-7,
) -> np.ndarray:
    coordinates = np.asarray(points, dtype=float)
    if coordinates.size == 0:
        return np.zeros((0,), dtype=bool)
    if translated_center is None:
        equations = np.asarray(mesh.hull_equations, dtype=float)
    else:
        translated_vertices = translated_mesh_vertices(mesh, translated_center)
        equations = np.asarray(
            _convex_hull_from_points(translated_vertices).equations,
            dtype=float,
        )
    left_side = coordinates @ equations[:, :3].T + equations[:, 3][None, :]
    return np.all(left_side <= float(tolerance), axis=1)
