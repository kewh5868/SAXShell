from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from saxshell.saxs.debye import discover_cluster_bins, load_structure_file

DEFAULT_ANISOTROPY_THRESHOLD = 1.25
STRUCTURE_FACTOR_RECOMMENDATIONS = ("sphere", "ellipsoid")


def cluster_identifier(structure: str, motif: str) -> str:
    normalized_motif = str(motif).strip() or "no_motif"
    if normalized_motif == "no_motif":
        return str(structure).strip()
    return f"{str(structure).strip()}/{normalized_motif}"


@dataclass(slots=True)
class ClusterGeometryMetadataRow:
    cluster_id: str
    structure: str
    motif: str
    cluster_path: str
    avg_size_metric: float
    effective_radius: float
    structure_factor_recommendation: str
    anisotropy_metric: float
    notes: str
    mapped_parameter: str | None = None
    mean_semiaxis_a: float = 0.0
    mean_semiaxis_b: float = 0.0
    mean_semiaxis_c: float = 0.0
    mean_radius_of_gyration: float = 0.0
    mean_max_radius: float = 0.0
    mean_atom_count: float = 0.0
    file_count: int = 0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, object],
    ) -> "ClusterGeometryMetadataRow":
        return cls(
            cluster_id=str(payload.get("cluster_id", "")).strip(),
            structure=str(payload.get("structure", "")).strip(),
            motif=str(payload.get("motif", "no_motif")).strip() or "no_motif",
            cluster_path=str(payload.get("cluster_path", "")).strip(),
            avg_size_metric=float(payload.get("avg_size_metric", 0.0)),
            effective_radius=float(payload.get("effective_radius", 0.0)),
            structure_factor_recommendation=str(
                payload.get(
                    "structure_factor_recommendation",
                    STRUCTURE_FACTOR_RECOMMENDATIONS[0],
                )
            ).strip()
            or STRUCTURE_FACTOR_RECOMMENDATIONS[0],
            anisotropy_metric=float(payload.get("anisotropy_metric", 0.0)),
            notes=str(payload.get("notes", "")).strip(),
            mapped_parameter=_optional_str(payload.get("mapped_parameter")),
            mean_semiaxis_a=float(payload.get("mean_semiaxis_a", 0.0)),
            mean_semiaxis_b=float(payload.get("mean_semiaxis_b", 0.0)),
            mean_semiaxis_c=float(payload.get("mean_semiaxis_c", 0.0)),
            mean_radius_of_gyration=float(
                payload.get("mean_radius_of_gyration", 0.0)
            ),
            mean_max_radius=float(payload.get("mean_max_radius", 0.0)),
            mean_atom_count=float(payload.get("mean_atom_count", 0.0)),
            file_count=int(payload.get("file_count", 0) or 0),
        )


@dataclass(slots=True)
class ClusterGeometryMetadataTable:
    rows: list[ClusterGeometryMetadataRow] = field(default_factory=list)
    source_clusters_dir: str | None = None
    computed_at: str | None = None
    anisotropy_threshold: float = DEFAULT_ANISOTROPY_THRESHOLD
    template_name: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "rows": [row.to_dict() for row in self.rows],
            "source_clusters_dir": self.source_clusters_dir,
            "computed_at": self.computed_at,
            "anisotropy_threshold": self.anisotropy_threshold,
            "template_name": self.template_name,
        }

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, object],
    ) -> "ClusterGeometryMetadataTable":
        return cls(
            rows=[
                ClusterGeometryMetadataRow.from_dict(dict(row))
                for row in payload.get("rows", [])
                if isinstance(row, dict)
            ],
            source_clusters_dir=_optional_str(
                payload.get("source_clusters_dir")
            ),
            computed_at=_optional_str(payload.get("computed_at")),
            anisotropy_threshold=float(
                payload.get(
                    "anisotropy_threshold",
                    DEFAULT_ANISOTROPY_THRESHOLD,
                )
            ),
            template_name=_optional_str(payload.get("template_name")),
        )


def compute_cluster_geometry_metadata(
    clusters_dir: str | Path,
    *,
    anisotropy_threshold: float = DEFAULT_ANISOTROPY_THRESHOLD,
    template_name: str | None = None,
) -> ClusterGeometryMetadataTable:
    resolved_clusters_dir = Path(clusters_dir).expanduser().resolve()
    cluster_bins = discover_cluster_bins(resolved_clusters_dir)
    rows = [
        _compute_cluster_geometry_row(
            cluster_bin,
            anisotropy_threshold=anisotropy_threshold,
        )
        for cluster_bin in cluster_bins
    ]
    return ClusterGeometryMetadataTable(
        rows=rows,
        source_clusters_dir=str(resolved_clusters_dir),
        computed_at=datetime.now().isoformat(timespec="seconds"),
        anisotropy_threshold=float(anisotropy_threshold),
        template_name=_optional_str(template_name),
    )


def load_cluster_geometry_metadata(
    path: str | Path,
) -> ClusterGeometryMetadataTable:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return ClusterGeometryMetadataTable.from_dict(payload)


def save_cluster_geometry_metadata(
    path: str | Path,
    table: ClusterGeometryMetadataTable,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(table.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path


def copy_cluster_geometry_rows(
    rows: list[ClusterGeometryMetadataRow],
) -> list[ClusterGeometryMetadataRow]:
    return [
        ClusterGeometryMetadataRow.from_dict(row.to_dict()) for row in rows
    ]


def apply_default_component_mapping(
    rows: list[ClusterGeometryMetadataRow],
    components: list[Any],
) -> bool:
    component_lookup = {
        (
            str(getattr(component, "structure", "")).strip(),
            str(getattr(component, "motif", "no_motif")).strip() or "no_motif",
        ): str(getattr(component, "param_name", "")).strip()
        for component in components
        if str(getattr(component, "param_name", "")).strip()
    }
    valid_parameters = {
        str(getattr(component, "param_name", "")).strip()
        for component in components
        if str(getattr(component, "param_name", "")).strip()
    }
    dirty = False
    for row in rows:
        mapped_parameter = (
            str(row.mapped_parameter).strip()
            if row.mapped_parameter is not None
            else ""
        )
        if mapped_parameter in valid_parameters:
            continue
        default_parameter = component_lookup.get((row.structure, row.motif))
        normalized_default = (
            default_parameter.strip() if default_parameter else None
        )
        if row.mapped_parameter != normalized_default:
            row.mapped_parameter = normalized_default
            dirty = True
    return dirty


def _compute_cluster_geometry_row(
    cluster_bin,
    *,
    anisotropy_threshold: float,
) -> ClusterGeometryMetadataRow:
    semiaxes_samples: list[np.ndarray] = []
    radius_of_gyration_samples: list[float] = []
    max_radius_samples: list[float] = []
    atom_counts: list[float] = []

    for file_path in cluster_bin.files:
        coordinates, _elements = load_structure_file(file_path)
        descriptors = _describe_cluster_geometry(coordinates)
        semiaxes_samples.append(descriptors["semiaxes"])
        radius_of_gyration_samples.append(descriptors["radius_of_gyration"])
        max_radius_samples.append(descriptors["max_radius"])
        atom_counts.append(float(coordinates.shape[0]))

    mean_semiaxes = np.mean(np.vstack(semiaxes_samples), axis=0)
    mean_semiaxes = np.asarray(np.sort(mean_semiaxes)[::-1], dtype=float)
    effective_radius = _equivalent_volume_radius(mean_semiaxes)
    anisotropy_metric = _anisotropy_metric(mean_semiaxes)
    recommendation = (
        "ellipsoid"
        if anisotropy_metric > float(anisotropy_threshold)
        else "sphere"
    )
    avg_size_metric = effective_radius * 2.0
    notes = (
        "Equivalent-volume sphere radius from centroid-aligned average "
        f"semiaxes ({mean_semiaxes[0]:.2f}, {mean_semiaxes[1]:.2f}, "
        f"{mean_semiaxes[2]:.2f}) A. Recommended {recommendation} "
        f"approximation from anisotropy ratio {anisotropy_metric:.3f}."
    )

    return ClusterGeometryMetadataRow(
        cluster_id=cluster_identifier(
            cluster_bin.structure, cluster_bin.motif
        ),
        structure=str(cluster_bin.structure).strip(),
        motif=str(cluster_bin.motif).strip() or "no_motif",
        cluster_path=str(Path(cluster_bin.source_dir).resolve()),
        avg_size_metric=float(avg_size_metric),
        effective_radius=float(effective_radius),
        structure_factor_recommendation=recommendation,
        anisotropy_metric=float(anisotropy_metric),
        notes=notes,
        mean_semiaxis_a=float(mean_semiaxes[0]),
        mean_semiaxis_b=float(mean_semiaxes[1]),
        mean_semiaxis_c=float(mean_semiaxes[2]),
        mean_radius_of_gyration=float(np.mean(radius_of_gyration_samples)),
        mean_max_radius=float(np.mean(max_radius_samples)),
        mean_atom_count=float(np.mean(atom_counts)),
        file_count=len(cluster_bin.files),
    )


def _describe_cluster_geometry(
    coordinates: np.ndarray,
) -> dict[str, float | np.ndarray]:
    coordinates = np.asarray(coordinates, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError(
            "Cluster geometry requires coordinate arrays with shape (N, 3)."
        )

    centroid = np.mean(coordinates, axis=0, keepdims=True)
    centered = coordinates - centroid
    radii = np.linalg.norm(centered, axis=1)
    radius_of_gyration = float(np.sqrt(np.mean(np.sum(centered**2, axis=1))))
    max_radius = float(np.max(radii)) if radii.size else 0.0

    if coordinates.shape[0] <= 1 or np.allclose(centered, 0.0):
        semiaxes = np.zeros(3, dtype=float)
    else:
        covariance = np.cov(centered, rowvar=False, bias=True)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        eigenvectors = np.asarray(eigenvectors[:, order], dtype=float)
        projected = centered @ eigenvectors
        semiaxes = 0.5 * (
            np.max(projected, axis=0) - np.min(projected, axis=0)
        )
        semiaxes = np.asarray(np.sort(semiaxes)[::-1], dtype=float)

    return {
        "semiaxes": semiaxes,
        "radius_of_gyration": radius_of_gyration,
        "max_radius": max_radius,
    }


def _equivalent_volume_radius(semiaxes: np.ndarray) -> float:
    safe_semiaxes = np.maximum(np.asarray(semiaxes, dtype=float), 0.0)
    if np.allclose(safe_semiaxes, 0.0):
        return 0.0
    return float(np.cbrt(np.prod(safe_semiaxes)))


def _anisotropy_metric(semiaxes: np.ndarray) -> float:
    safe_semiaxes = np.maximum(np.asarray(semiaxes, dtype=float), 0.0)
    largest = float(np.max(safe_semiaxes)) if safe_semiaxes.size else 0.0
    smallest = float(np.min(safe_semiaxes)) if safe_semiaxes.size else 0.0
    if largest <= 0.0:
        return 1.0
    if smallest <= 0.0:
        return largest / max(float(np.mean(safe_semiaxes)), 1e-12)
    return largest / smallest


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


__all__ = [
    "DEFAULT_ANISOTROPY_THRESHOLD",
    "STRUCTURE_FACTOR_RECOMMENDATIONS",
    "ClusterGeometryMetadataRow",
    "ClusterGeometryMetadataTable",
    "apply_default_component_mapping",
    "cluster_identifier",
    "compute_cluster_geometry_metadata",
    "copy_cluster_geometry_rows",
    "load_cluster_geometry_metadata",
    "save_cluster_geometry_metadata",
]
