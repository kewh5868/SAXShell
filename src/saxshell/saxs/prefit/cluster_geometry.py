from __future__ import annotations

import json
import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from saxshell.saxs.debye import discover_cluster_bins, load_structure_file

DEFAULT_ANISOTROPY_THRESHOLD = 1.25
STRUCTURE_FACTOR_RECOMMENDATIONS = ("sphere", "ellipsoid")
RADIUS_TYPE_OPTIONS = ("ionic", "bond_length")
DEFAULT_RADIUS_TYPE = RADIUS_TYPE_OPTIONS[0]
IONIC_RADIUS_TYPE_OPTIONS = ("effective", "crystal")
DEFAULT_IONIC_RADIUS_TYPE = IONIC_RADIUS_TYPE_OPTIONS[0]
CLUSTER_GEOMETRY_SCHEMA_VERSION = 3
IONIC_RADIUS_FALLBACK_ANGSTROM = 1.5
CRYSTAL_IONIC_RADIUS_OFFSET_ANGSTROM = 0.14
COVALENT_RADIUS_FALLBACK_ANGSTROM = 1.2
TELLURIUM_SYMBOL = "T" "e"

# Approximate single-radius ionic-size heuristics in angstroms.
# These are intentionally simple element-level defaults so the Prefit
# geometry workflow can operate without charge-state annotations.
APPROXIMATE_EFFECTIVE_IONIC_RADII_ANGSTROM = {
    "Ag": 1.15,
    "Al": 0.54,
    "As": 0.58,
    "Au": 1.37,
    "B": 0.23,
    "Ba": 1.35,
    "Be": 0.45,
    "Bi": 1.03,
    "Br": 1.96,
    "C": 0.30,
    "Ca": 1.00,
    "Cd": 0.95,
    "Cl": 1.81,
    "Co": 0.75,
    "Cr": 0.76,
    "Cs": 1.67,
    "Cu": 0.73,
    "F": 1.33,
    "Fe": 0.78,
    "Ga": 0.62,
    "Ge": 0.53,
    "H": 0.25,
    "Hf": 0.71,
    "Hg": 1.02,
    "I": 2.20,
    "In": 0.80,
    "K": 1.38,
    "Li": 0.76,
    "Mg": 0.72,
    "Mn": 0.83,
    "Mo": 0.65,
    "N": 1.46,
    "Na": 1.02,
    "Ni": 0.69,
    "O": 1.40,
    "P": 0.44,
    "Pb": 1.19,
    "Rb": 1.52,
    "S": 1.84,
    "Sb": 0.76,
    "Se": 1.98,
    "Si": 0.40,
    "Sn": 0.69,
    "Sr": 1.18,
    TELLURIUM_SYMBOL: 2.21,
    "Ti": 0.61,
    "Tl": 1.50,
    "V": 0.79,
    "Y": 0.90,
    "Zn": 0.74,
    "Zr": 0.72,
}
APPROXIMATE_CRYSTAL_IONIC_RADII_ANGSTROM = {
    element: float(radius) + CRYSTAL_IONIC_RADIUS_OFFSET_ANGSTROM
    for element, radius in APPROXIMATE_EFFECTIVE_IONIC_RADII_ANGSTROM.items()
}
APPROXIMATE_IONIC_RADII_ANGSTROM = APPROXIMATE_EFFECTIVE_IONIC_RADII_ANGSTROM

# Approximate single-bond covalent radii in angstroms, following the
# Cordero et al. 2008 style picture closely enough for lone-atom
# bond-length fallbacks in the Prefit geometry workflow.
APPROXIMATE_COVALENT_RADII_ANGSTROM = {
    "Ag": 1.45,
    "Al": 1.21,
    "As": 1.19,
    "Au": 1.36,
    "B": 0.84,
    "Ba": 2.15,
    "Be": 0.96,
    "Bi": 1.48,
    "Br": 1.20,
    "C": 0.76,
    "Ca": 1.76,
    "Cd": 1.44,
    "Cl": 1.02,
    "Co": 1.26,
    "Cr": 1.39,
    "Cs": 2.44,
    "Cu": 1.32,
    "F": 0.57,
    "Fe": 1.32,
    "Ga": 1.22,
    "Ge": 1.20,
    "H": 0.31,
    "Hf": 1.75,
    "Hg": 1.32,
    "I": 1.39,
    "In": 1.42,
    "K": 2.03,
    "Li": 1.28,
    "Mg": 1.41,
    "Mn": 1.39,
    "Mo": 1.54,
    "N": 0.71,
    "Na": 1.66,
    "Ni": 1.24,
    "O": 0.66,
    "P": 1.07,
    "Pb": 1.46,
    "Rb": 2.20,
    "S": 1.05,
    "Sb": 1.39,
    "Se": 1.20,
    "Si": 1.11,
    "Sn": 1.39,
    "Sr": 1.95,
    TELLURIUM_SYMBOL: 1.38,
    "Ti": 1.60,
    "Tl": 1.45,
    "V": 1.53,
    "Y": 1.90,
    "Zn": 1.22,
    "Zr": 1.75,
}


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
    sf_approximation: str = STRUCTURE_FACTOR_RECOMMENDATIONS[0]
    radii_type_used: str = DEFAULT_RADIUS_TYPE
    ionic_radius_type_used: str = DEFAULT_IONIC_RADIUS_TYPE
    ionic_sphere_effective_radius: float = 0.0
    crystal_ionic_sphere_effective_radius: float = 0.0
    bond_length_sphere_effective_radius: float = 0.0
    ionic_ellipsoid_semiaxis_a: float = 0.0
    ionic_ellipsoid_semiaxis_b: float = 0.0
    ionic_ellipsoid_semiaxis_c: float = 0.0
    crystal_ionic_ellipsoid_semiaxis_a: float = 0.0
    crystal_ionic_ellipsoid_semiaxis_b: float = 0.0
    crystal_ionic_ellipsoid_semiaxis_c: float = 0.0
    bond_length_ellipsoid_semiaxis_a: float = 0.0
    bond_length_ellipsoid_semiaxis_b: float = 0.0
    bond_length_ellipsoid_semiaxis_c: float = 0.0
    active_semiaxis_a: float = 0.0
    active_semiaxis_b: float = 0.0
    active_semiaxis_c: float = 0.0
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
        legacy_payload = "sf_approximation" not in payload
        structure_factor_recommendation = _normalize_sf_approximation(
            payload.get(
                "structure_factor_recommendation",
                STRUCTURE_FACTOR_RECOMMENDATIONS[0],
            ),
            fallback=STRUCTURE_FACTOR_RECOMMENDATIONS[0],
        )
        bond_length_semiaxes = np.asarray(
            [
                float(
                    payload.get(
                        "bond_length_ellipsoid_semiaxis_a",
                        payload.get("mean_semiaxis_a", 0.0),
                    )
                ),
                float(
                    payload.get(
                        "bond_length_ellipsoid_semiaxis_b",
                        payload.get("mean_semiaxis_b", 0.0),
                    )
                ),
                float(
                    payload.get(
                        "bond_length_ellipsoid_semiaxis_c",
                        payload.get("mean_semiaxis_c", 0.0),
                    )
                ),
            ],
            dtype=float,
        )
        legacy_effective_radius = float(payload.get("effective_radius", 0.0))
        bond_length_sphere_effective_radius = float(
            payload.get(
                "bond_length_sphere_effective_radius",
                legacy_effective_radius,
            )
        )
        ionic_semiaxes = np.asarray(
            [
                float(
                    payload.get(
                        "ionic_ellipsoid_semiaxis_a",
                        bond_length_semiaxes[0],
                    )
                ),
                float(
                    payload.get(
                        "ionic_ellipsoid_semiaxis_b",
                        bond_length_semiaxes[1],
                    )
                ),
                float(
                    payload.get(
                        "ionic_ellipsoid_semiaxis_c",
                        bond_length_semiaxes[2],
                    )
                ),
            ],
            dtype=float,
        )
        row = cls(
            cluster_id=str(payload.get("cluster_id", "")).strip(),
            structure=str(payload.get("structure", "")).strip(),
            motif=str(payload.get("motif", "no_motif")).strip() or "no_motif",
            cluster_path=str(payload.get("cluster_path", "")).strip(),
            avg_size_metric=float(payload.get("avg_size_metric", 0.0)),
            effective_radius=legacy_effective_radius,
            structure_factor_recommendation=structure_factor_recommendation,
            anisotropy_metric=float(payload.get("anisotropy_metric", 0.0)),
            notes=str(payload.get("notes", "")).strip(),
            mapped_parameter=_optional_str(payload.get("mapped_parameter")),
            sf_approximation=_normalize_sf_approximation(
                payload.get("sf_approximation"),
                fallback=(
                    STRUCTURE_FACTOR_RECOMMENDATIONS[0]
                    if legacy_payload
                    else structure_factor_recommendation
                ),
            ),
            radii_type_used=_normalize_radii_type(
                payload.get("radii_type_used"),
                default=DEFAULT_RADIUS_TYPE,
            ),
            ionic_radius_type_used=_normalize_ionic_radius_type(
                payload.get("ionic_radius_type_used"),
                default=DEFAULT_IONIC_RADIUS_TYPE,
            ),
            ionic_sphere_effective_radius=float(
                payload.get(
                    "ionic_sphere_effective_radius",
                    bond_length_sphere_effective_radius,
                )
            ),
            crystal_ionic_sphere_effective_radius=float(
                payload.get(
                    "crystal_ionic_sphere_effective_radius",
                    _upgrade_crystal_ionic_value(
                        float(
                            payload.get(
                                "ionic_sphere_effective_radius",
                                bond_length_sphere_effective_radius,
                            )
                        )
                    ),
                )
            ),
            bond_length_sphere_effective_radius=(
                bond_length_sphere_effective_radius
            ),
            ionic_ellipsoid_semiaxis_a=float(ionic_semiaxes[0]),
            ionic_ellipsoid_semiaxis_b=float(ionic_semiaxes[1]),
            ionic_ellipsoid_semiaxis_c=float(ionic_semiaxes[2]),
            crystal_ionic_ellipsoid_semiaxis_a=float(
                payload.get(
                    "crystal_ionic_ellipsoid_semiaxis_a",
                    _upgrade_crystal_ionic_value(float(ionic_semiaxes[0])),
                )
            ),
            crystal_ionic_ellipsoid_semiaxis_b=float(
                payload.get(
                    "crystal_ionic_ellipsoid_semiaxis_b",
                    _upgrade_crystal_ionic_value(float(ionic_semiaxes[1])),
                )
            ),
            crystal_ionic_ellipsoid_semiaxis_c=float(
                payload.get(
                    "crystal_ionic_ellipsoid_semiaxis_c",
                    _upgrade_crystal_ionic_value(float(ionic_semiaxes[2])),
                )
            ),
            bond_length_ellipsoid_semiaxis_a=float(bond_length_semiaxes[0]),
            bond_length_ellipsoid_semiaxis_b=float(bond_length_semiaxes[1]),
            bond_length_ellipsoid_semiaxis_c=float(bond_length_semiaxes[2]),
            active_semiaxis_a=float(
                payload.get("active_semiaxis_a", bond_length_semiaxes[0])
            ),
            active_semiaxis_b=float(
                payload.get("active_semiaxis_b", bond_length_semiaxes[1])
            ),
            active_semiaxis_c=float(
                payload.get("active_semiaxis_c", bond_length_semiaxes[2])
            ),
            mean_semiaxis_a=float(
                payload.get("mean_semiaxis_a", bond_length_semiaxes[0])
            ),
            mean_semiaxis_b=float(
                payload.get("mean_semiaxis_b", bond_length_semiaxes[1])
            ),
            mean_semiaxis_c=float(
                payload.get("mean_semiaxis_c", bond_length_semiaxes[2])
            ),
            mean_radius_of_gyration=float(
                payload.get("mean_radius_of_gyration", 0.0)
            ),
            mean_max_radius=float(payload.get("mean_max_radius", 0.0)),
            mean_atom_count=float(payload.get("mean_atom_count", 0.0)),
            file_count=int(payload.get("file_count", 0) or 0),
        )
        synchronize_cluster_geometry_row(
            row,
            active_radii_type=row.radii_type_used,
            active_ionic_radius_type=row.ionic_radius_type_used,
        )
        return row


@dataclass(slots=True)
class ClusterGeometryMetadataTable:
    rows: list[ClusterGeometryMetadataRow] = field(default_factory=list)
    source_clusters_dir: str | None = None
    computed_at: str | None = None
    anisotropy_threshold: float = DEFAULT_ANISOTROPY_THRESHOLD
    template_name: str | None = None
    schema_version: int = CLUSTER_GEOMETRY_SCHEMA_VERSION
    active_radii_type: str = DEFAULT_RADIUS_TYPE
    active_ionic_radius_type: str = DEFAULT_IONIC_RADIUS_TYPE

    def to_dict(self) -> dict[str, object]:
        return {
            "rows": [row.to_dict() for row in self.rows],
            "source_clusters_dir": self.source_clusters_dir,
            "computed_at": self.computed_at,
            "anisotropy_threshold": self.anisotropy_threshold,
            "template_name": self.template_name,
            "schema_version": self.schema_version,
            "active_radii_type": self.active_radii_type,
            "active_ionic_radius_type": self.active_ionic_radius_type,
        }

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, object],
    ) -> "ClusterGeometryMetadataTable":
        table = cls(
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
            schema_version=int(
                payload.get("schema_version", CLUSTER_GEOMETRY_SCHEMA_VERSION)
                or CLUSTER_GEOMETRY_SCHEMA_VERSION
            ),
            active_radii_type=_normalize_radii_type(
                payload.get("active_radii_type"),
                default=DEFAULT_RADIUS_TYPE,
            ),
            active_ionic_radius_type=_normalize_ionic_radius_type(
                payload.get("active_ionic_radius_type"),
                default=DEFAULT_IONIC_RADIUS_TYPE,
            ),
        )
        synchronize_cluster_geometry_table(table)
        return table


def compute_cluster_geometry_metadata(
    clusters_dir: str | Path,
    *,
    anisotropy_threshold: float = DEFAULT_ANISOTROPY_THRESHOLD,
    template_name: str | None = None,
    active_radii_type: str = DEFAULT_RADIUS_TYPE,
    active_ionic_radius_type: str = DEFAULT_IONIC_RADIUS_TYPE,
    allowed_sf_approximations: tuple[str, ...] = (
        "sphere",
        "ellipsoid",
    ),
    progress_callback: Callable[[int, int, str], None] | None = None,
    max_workers: int | None = None,
) -> ClusterGeometryMetadataTable:
    resolved_clusters_dir = Path(clusters_dir).expanduser().resolve()
    normalized_radii_type = _normalize_radii_type(
        active_radii_type,
        default=DEFAULT_RADIUS_TYPE,
    )
    normalized_ionic_radius_type = _normalize_ionic_radius_type(
        active_ionic_radius_type,
        default=DEFAULT_IONIC_RADIUS_TYPE,
    )
    cluster_bins = discover_cluster_bins(resolved_clusters_dir)
    total_files = sum(len(cluster_bin.files) for cluster_bin in cluster_bins)
    if progress_callback is not None:
        progress_callback(
            0,
            max(total_files, 1),
            "Preparing cluster geometry computation...",
        )
    total_files = max(total_files, 1)
    worker_count = _resolve_cluster_geometry_max_workers(
        cluster_bins,
        max_workers=max_workers,
    )
    rows: list[ClusterGeometryMetadataRow]
    if worker_count <= 1:
        rows = []
        processed_files = 0
        for cluster_bin in cluster_bins:
            rows.append(
                _compute_cluster_geometry_row(
                    cluster_bin,
                    anisotropy_threshold=anisotropy_threshold,
                    active_radii_type=normalized_radii_type,
                    active_ionic_radius_type=normalized_ionic_radius_type,
                    total_files=total_files,
                    processed_files=processed_files,
                    progress_callback=progress_callback,
                )
            )
            processed_files += len(cluster_bin.files)
    else:
        rows_by_index: list[ClusterGeometryMetadataRow | None] = [None] * len(
            cluster_bins
        )
        processed_files = 0
        completed_bins = 0
        with ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="cluster-geometry",
        ) as executor:
            future_map = {
                executor.submit(
                    _compute_cluster_geometry_row,
                    cluster_bin,
                    anisotropy_threshold=anisotropy_threshold,
                    active_radii_type=normalized_radii_type,
                    active_ionic_radius_type=normalized_ionic_radius_type,
                ): (index, cluster_bin)
                for index, cluster_bin in enumerate(cluster_bins)
            }
            for future in as_completed(future_map):
                index, cluster_bin = future_map[future]
                rows_by_index[index] = future.result()
                processed_files += len(cluster_bin.files)
                completed_bins += 1
                if progress_callback is not None:
                    progress_callback(
                        min(processed_files, total_files),
                        total_files,
                        "Computing cluster geometry for "
                        f"{cluster_identifier(cluster_bin.structure, cluster_bin.motif)} "
                        f"(completed {completed_bins}/{len(cluster_bins)} clusters)...",
                    )
        rows = [row for row in rows_by_index if row is not None]
    if progress_callback is not None:
        progress_callback(
            total_files,
            total_files,
            "Cluster geometry metadata ready.",
        )
    table = ClusterGeometryMetadataTable(
        rows=rows,
        source_clusters_dir=str(resolved_clusters_dir),
        computed_at=datetime.now().isoformat(timespec="seconds"),
        anisotropy_threshold=float(anisotropy_threshold),
        template_name=_optional_str(template_name),
        schema_version=CLUSTER_GEOMETRY_SCHEMA_VERSION,
        active_radii_type=normalized_radii_type,
        active_ionic_radius_type=normalized_ionic_radius_type,
    )
    synchronize_cluster_geometry_table(
        table,
        allowed_sf_approximations=allowed_sf_approximations,
    )
    return table


def load_cluster_geometry_metadata(
    path: str | Path,
) -> ClusterGeometryMetadataTable:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return ClusterGeometryMetadataTable.from_dict(payload)


def save_cluster_geometry_metadata(
    path: str | Path,
    table: ClusterGeometryMetadataTable,
) -> Path:
    synchronize_cluster_geometry_table(table)
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


def synchronize_cluster_geometry_table(
    table: ClusterGeometryMetadataTable,
    *,
    allowed_sf_approximations: tuple[str, ...] = (
        "sphere",
        "ellipsoid",
    ),
) -> bool:
    dirty = False
    normalized_allowed_sf = _normalize_allowed_sf_approximations(
        allowed_sf_approximations
    )
    normalized_radii_type = _normalize_radii_type(
        table.active_radii_type,
        default=DEFAULT_RADIUS_TYPE,
    )
    normalized_ionic_radius_type = _normalize_ionic_radius_type(
        table.active_ionic_radius_type,
        default=DEFAULT_IONIC_RADIUS_TYPE,
    )
    if table.active_radii_type != normalized_radii_type:
        table.active_radii_type = normalized_radii_type
        dirty = True
    if table.active_ionic_radius_type != normalized_ionic_radius_type:
        table.active_ionic_radius_type = normalized_ionic_radius_type
        dirty = True
    if table.schema_version != CLUSTER_GEOMETRY_SCHEMA_VERSION:
        table.schema_version = CLUSTER_GEOMETRY_SCHEMA_VERSION
        dirty = True
    for row in table.rows:
        if synchronize_cluster_geometry_row(
            row,
            active_radii_type=table.active_radii_type,
            active_ionic_radius_type=table.active_ionic_radius_type,
            allowed_sf_approximations=normalized_allowed_sf,
        ):
            dirty = True
    return dirty


def validate_positive_cluster_geometry_table(
    table: ClusterGeometryMetadataTable | None,
) -> None:
    if table is None:
        return
    for row in table.rows:
        validate_positive_cluster_geometry_row(row)


def validate_positive_cluster_geometry_row(
    row: ClusterGeometryMetadataRow,
) -> None:
    invalid_fields: list[str] = []
    for label, value in (
        ("effective radius", row.effective_radius),
        ("semiaxis x", row.active_semiaxis_a),
        ("semiaxis y", row.active_semiaxis_b),
        ("semiaxis z", row.active_semiaxis_c),
    ):
        numeric_value = float(value)
        if not np.isfinite(numeric_value) or numeric_value <= 0.0:
            invalid_fields.append(f"{label}={numeric_value:.6g}")
    if invalid_fields:
        raise ValueError(
            "Cluster geometry metadata for "
            f"{row.cluster_id} must provide positive active radii for the "
            f"{row.sf_approximation} approximation in {row.radii_type_used} "
            "mode. Invalid values: " + ", ".join(invalid_fields) + "."
        )


def synchronize_cluster_geometry_row(
    row: ClusterGeometryMetadataRow,
    *,
    active_radii_type: str,
    active_ionic_radius_type: str = DEFAULT_IONIC_RADIUS_TYPE,
    allowed_sf_approximations: tuple[str, ...] = (
        "sphere",
        "ellipsoid",
    ),
) -> bool:
    dirty = False
    normalized_allowed_sf = _normalize_allowed_sf_approximations(
        allowed_sf_approximations
    )

    if _apply_single_atom_bond_length_fallback(row):
        dirty = True

    normalized_radii_type = _normalize_radii_type(
        active_radii_type,
        default=DEFAULT_RADIUS_TYPE,
    )
    if row.radii_type_used != normalized_radii_type:
        row.radii_type_used = normalized_radii_type
        dirty = True
    normalized_ionic_radius_type = _normalize_ionic_radius_type(
        active_ionic_radius_type,
        default=DEFAULT_IONIC_RADIUS_TYPE,
    )
    if row.ionic_radius_type_used != normalized_ionic_radius_type:
        row.ionic_radius_type_used = normalized_ionic_radius_type
        dirty = True

    recommendation = _normalize_sf_approximation(
        row.structure_factor_recommendation,
        fallback=normalized_allowed_sf[0],
        allowed=normalized_allowed_sf,
    )
    if row.structure_factor_recommendation != recommendation:
        row.structure_factor_recommendation = recommendation
        dirty = True

    sf_approximation = _normalize_sf_approximation(
        row.sf_approximation,
        fallback=recommendation,
        allowed=normalized_allowed_sf,
    )
    if row.sf_approximation != sf_approximation:
        row.sf_approximation = sf_approximation
        dirty = True

    bond_length_semiaxes = np.asarray(
        [
            row.bond_length_ellipsoid_semiaxis_a,
            row.bond_length_ellipsoid_semiaxis_b,
            row.bond_length_ellipsoid_semiaxis_c,
        ],
        dtype=float,
    )
    ionic_semiaxes = np.asarray(
        [
            row.ionic_ellipsoid_semiaxis_a,
            row.ionic_ellipsoid_semiaxis_b,
            row.ionic_ellipsoid_semiaxis_c,
        ],
        dtype=float,
    )
    crystal_ionic_semiaxes = np.asarray(
        [
            row.crystal_ionic_ellipsoid_semiaxis_a,
            row.crystal_ionic_ellipsoid_semiaxis_b,
            row.crystal_ionic_ellipsoid_semiaxis_c,
        ],
        dtype=float,
    )
    if not np.allclose(
        [
            row.mean_semiaxis_a,
            row.mean_semiaxis_b,
            row.mean_semiaxis_c,
        ],
        bond_length_semiaxes,
    ):
        row.mean_semiaxis_a = float(bond_length_semiaxes[0])
        row.mean_semiaxis_b = float(bond_length_semiaxes[1])
        row.mean_semiaxis_c = float(bond_length_semiaxes[2])
        dirty = True

    if normalized_radii_type == "ionic":
        if normalized_ionic_radius_type == "crystal":
            selected_sphere_radius = row.crystal_ionic_sphere_effective_radius
            selected_ellipsoid_semiaxes = crystal_ionic_semiaxes
        else:
            selected_sphere_radius = row.ionic_sphere_effective_radius
            selected_ellipsoid_semiaxes = ionic_semiaxes
    else:
        selected_sphere_radius = row.bond_length_sphere_effective_radius
        selected_ellipsoid_semiaxes = bond_length_semiaxes

    if sf_approximation == "sphere":
        active_radius = float(max(selected_sphere_radius, 0.0))
        active_semiaxes = np.full(3, active_radius, dtype=float)
        anisotropy_metric = 1.0
    else:
        active_semiaxes = np.maximum(
            np.asarray(selected_ellipsoid_semiaxes, dtype=float),
            0.0,
        )
        active_radius = _equivalent_volume_radius(active_semiaxes)
        anisotropy_metric = _anisotropy_metric(active_semiaxes)

    if row.effective_radius != active_radius:
        row.effective_radius = active_radius
        dirty = True
    if row.avg_size_metric != active_radius * 2.0:
        row.avg_size_metric = active_radius * 2.0
        dirty = True
    if row.anisotropy_metric != anisotropy_metric:
        row.anisotropy_metric = anisotropy_metric
        dirty = True

    active_semiaxes_tuple = (
        float(active_semiaxes[0]),
        float(active_semiaxes[1]),
        float(active_semiaxes[2]),
    )
    if (
        row.active_semiaxis_a,
        row.active_semiaxis_b,
        row.active_semiaxis_c,
    ) != active_semiaxes_tuple:
        row.active_semiaxis_a = active_semiaxes_tuple[0]
        row.active_semiaxis_b = active_semiaxes_tuple[1]
        row.active_semiaxis_c = active_semiaxes_tuple[2]
        dirty = True

    return dirty


def _compute_cluster_geometry_row(
    cluster_bin,
    *,
    anisotropy_threshold: float,
    active_radii_type: str,
    active_ionic_radius_type: str,
    total_files: int = 1,
    processed_files: int = 0,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> ClusterGeometryMetadataRow:
    bond_length_semiaxes_sum = np.zeros(3, dtype=float)
    ionic_semiaxes_sum = np.zeros(3, dtype=float)
    crystal_ionic_semiaxes_sum = np.zeros(3, dtype=float)
    radius_of_gyration_sum = 0.0
    max_radius_sum = 0.0
    atom_count_sum = 0.0
    sample_count = 0
    missing_ionic_elements: set[str] = set()
    missing_covalent_elements: set[str] = set()

    cluster_id = cluster_identifier(
        cluster_bin.structure,
        cluster_bin.motif,
    )

    def accumulate_structure(
        coordinates: np.ndarray,
        elements: list[str],
    ) -> None:
        nonlocal sample_count
        nonlocal bond_length_semiaxes_sum
        nonlocal ionic_semiaxes_sum
        nonlocal crystal_ionic_semiaxes_sum
        nonlocal radius_of_gyration_sum
        nonlocal max_radius_sum
        nonlocal atom_count_sum
        nonlocal missing_covalent_elements
        bond_length_descriptors = _describe_cluster_geometry(coordinates)
        if _bond_length_geometry_needs_covalent_padding(
            coordinates,
            bond_length_descriptors["semiaxes"],
        ):
            bond_length_atomic_radii, missing_bond_length_elements = (
                _lookup_covalent_radii(elements)
            )
            missing_covalent_elements.update(missing_bond_length_elements)
            bond_length_descriptors = _describe_cluster_geometry(
                coordinates,
                atomic_radii=bond_length_atomic_radii,
            )
        effective_ionic_radii, missing_elements = _lookup_ionic_radii(
            elements,
            ionic_radius_type="effective",
        )
        crystal_ionic_radii, _ = _lookup_ionic_radii(
            elements,
            ionic_radius_type="crystal",
        )
        ionic_descriptors = _describe_cluster_geometry(
            coordinates,
            atomic_radii=effective_ionic_radii,
        )
        crystal_ionic_descriptors = _describe_cluster_geometry(
            coordinates,
            atomic_radii=crystal_ionic_radii,
        )
        bond_length_semiaxes_sum += np.asarray(
            bond_length_descriptors["semiaxes"],
            dtype=float,
        )
        ionic_semiaxes_sum += np.asarray(
            ionic_descriptors["semiaxes"],
            dtype=float,
        )
        crystal_ionic_semiaxes_sum += np.asarray(
            crystal_ionic_descriptors["semiaxes"],
            dtype=float,
        )
        radius_of_gyration_sum += float(
            bond_length_descriptors["radius_of_gyration"]
        )
        max_radius_sum += float(bond_length_descriptors["max_radius"])
        atom_count_sum += float(coordinates.shape[0])
        sample_count += 1
        missing_ionic_elements.update(missing_elements)

    if cluster_bin.files:
        first_coordinates, first_elements = load_structure_file(
            cluster_bin.files[0]
        )
        if (
            np.asarray(first_coordinates, dtype=float).shape == (1, 3)
            and len(first_elements) == 1
        ):
            accumulate_structure(first_coordinates, first_elements)
            if progress_callback is not None:
                progress_callback(
                    min(processed_files + len(cluster_bin.files), total_files),
                    total_files,
                    "Computing cluster geometry for "
                    f"{cluster_id} (single-atom shortcut, "
                    f"{len(cluster_bin.files)} files)...",
                )
        else:
            accumulate_structure(first_coordinates, first_elements)
            if progress_callback is not None:
                progress_callback(
                    min(processed_files + 1, total_files),
                    total_files,
                    "Computing cluster geometry for "
                    f"{cluster_id} (1/{len(cluster_bin.files)} files)...",
                )
            for file_index, file_path in enumerate(
                cluster_bin.files[1:],
                start=2,
            ):
                coordinates, elements = load_structure_file(file_path)
                accumulate_structure(coordinates, elements)
                if progress_callback is not None:
                    progress_callback(
                        min(processed_files + file_index, total_files),
                        total_files,
                        "Computing cluster geometry for "
                        f"{cluster_id} ({file_index}/{len(cluster_bin.files)} files)...",
                    )

    if sample_count <= 0:
        raise ValueError(
            f"No structure files were available for cluster bin {cluster_id}."
        )

    mean_bond_length_semiaxes = bond_length_semiaxes_sum / float(sample_count)
    mean_bond_length_semiaxes = np.asarray(
        np.sort(mean_bond_length_semiaxes)[::-1],
        dtype=float,
    )
    mean_ionic_semiaxes = ionic_semiaxes_sum / float(sample_count)
    mean_ionic_semiaxes = np.asarray(
        np.sort(mean_ionic_semiaxes)[::-1],
        dtype=float,
    )
    mean_crystal_ionic_semiaxes = crystal_ionic_semiaxes_sum / float(
        sample_count
    )
    mean_crystal_ionic_semiaxes = np.asarray(
        np.sort(mean_crystal_ionic_semiaxes)[::-1],
        dtype=float,
    )
    bond_length_sphere_effective_radius = _equivalent_volume_radius(
        mean_bond_length_semiaxes
    )
    ionic_sphere_effective_radius = _equivalent_volume_radius(
        mean_ionic_semiaxes
    )
    crystal_ionic_sphere_effective_radius = _equivalent_volume_radius(
        mean_crystal_ionic_semiaxes
    )
    recommendation = (
        "ellipsoid"
        if _anisotropy_metric(mean_ionic_semiaxes)
        > float(anisotropy_threshold)
        else "sphere"
    )
    notes = _build_cluster_geometry_notes(
        mean_ionic_semiaxes=mean_ionic_semiaxes,
        mean_crystal_ionic_semiaxes=mean_crystal_ionic_semiaxes,
        mean_bond_length_semiaxes=mean_bond_length_semiaxes,
        ionic_sphere_effective_radius=ionic_sphere_effective_radius,
        crystal_ionic_sphere_effective_radius=(
            crystal_ionic_sphere_effective_radius
        ),
        bond_length_sphere_effective_radius=(
            bond_length_sphere_effective_radius
        ),
        recommendation=recommendation,
        missing_ionic_elements=missing_ionic_elements,
        missing_covalent_elements=missing_covalent_elements,
        mean_atom_count=atom_count_sum / float(sample_count),
        bond_length_single_atom_fallback_used=(
            atom_count_sum / float(sample_count)
        )
        <= 1.0,
    )
    row = ClusterGeometryMetadataRow(
        cluster_id=cluster_id,
        structure=str(cluster_bin.structure).strip(),
        motif=str(cluster_bin.motif).strip() or "no_motif",
        cluster_path=str(Path(cluster_bin.source_dir).resolve()),
        avg_size_metric=0.0,
        effective_radius=0.0,
        structure_factor_recommendation=recommendation,
        anisotropy_metric=0.0,
        notes=notes,
        sf_approximation=recommendation,
        radii_type_used=_normalize_radii_type(
            active_radii_type,
            default=DEFAULT_RADIUS_TYPE,
        ),
        ionic_radius_type_used=_normalize_ionic_radius_type(
            active_ionic_radius_type,
            default=DEFAULT_IONIC_RADIUS_TYPE,
        ),
        ionic_sphere_effective_radius=float(ionic_sphere_effective_radius),
        crystal_ionic_sphere_effective_radius=float(
            crystal_ionic_sphere_effective_radius
        ),
        bond_length_sphere_effective_radius=float(
            bond_length_sphere_effective_radius
        ),
        ionic_ellipsoid_semiaxis_a=float(mean_ionic_semiaxes[0]),
        ionic_ellipsoid_semiaxis_b=float(mean_ionic_semiaxes[1]),
        ionic_ellipsoid_semiaxis_c=float(mean_ionic_semiaxes[2]),
        crystal_ionic_ellipsoid_semiaxis_a=float(
            mean_crystal_ionic_semiaxes[0]
        ),
        crystal_ionic_ellipsoid_semiaxis_b=float(
            mean_crystal_ionic_semiaxes[1]
        ),
        crystal_ionic_ellipsoid_semiaxis_c=float(
            mean_crystal_ionic_semiaxes[2]
        ),
        bond_length_ellipsoid_semiaxis_a=float(mean_bond_length_semiaxes[0]),
        bond_length_ellipsoid_semiaxis_b=float(mean_bond_length_semiaxes[1]),
        bond_length_ellipsoid_semiaxis_c=float(mean_bond_length_semiaxes[2]),
        mean_semiaxis_a=float(mean_bond_length_semiaxes[0]),
        mean_semiaxis_b=float(mean_bond_length_semiaxes[1]),
        mean_semiaxis_c=float(mean_bond_length_semiaxes[2]),
        mean_radius_of_gyration=radius_of_gyration_sum / float(sample_count),
        mean_max_radius=max_radius_sum / float(sample_count),
        mean_atom_count=atom_count_sum / float(sample_count),
        file_count=len(cluster_bin.files),
    )
    synchronize_cluster_geometry_row(
        row,
        active_radii_type=active_radii_type,
        active_ionic_radius_type=active_ionic_radius_type,
    )
    return row


def _resolve_cluster_geometry_max_workers(
    cluster_bins: list[Any],
    *,
    max_workers: int | None = None,
) -> int:
    bin_count = len(cluster_bins)
    if bin_count <= 1:
        return 1
    if max_workers is not None:
        return max(1, min(int(max_workers), bin_count))
    available_cpus = os.cpu_count() or 1
    return max(1, min(bin_count, available_cpus, 8))


def _describe_cluster_geometry(
    coordinates: np.ndarray,
    *,
    atomic_radii: np.ndarray | None = None,
) -> dict[str, float | np.ndarray]:
    coordinates = np.asarray(coordinates, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError(
            "Cluster geometry requires coordinate arrays with shape (N, 3)."
        )

    if atomic_radii is None:
        radii_padding = np.zeros(coordinates.shape[0], dtype=float)
    else:
        radii_padding = np.maximum(np.asarray(atomic_radii, dtype=float), 0.0)
        if radii_padding.shape != (coordinates.shape[0],):
            raise ValueError(
                "atomic_radii must contain one entry per coordinate row."
            )

    centroid = np.mean(coordinates, axis=0, keepdims=True)
    centered = coordinates - centroid
    radii = np.linalg.norm(centered, axis=1) + radii_padding
    radius_of_gyration = float(np.sqrt(np.mean(np.sum(centered**2, axis=1))))
    max_radius = float(np.max(radii)) if radii.size else 0.0

    if coordinates.shape[0] <= 1 or np.allclose(centered, 0.0):
        if radii_padding.size:
            semiaxes = np.full(
                3,
                float(np.max(radii_padding)),
                dtype=float,
            )
        else:
            semiaxes = np.zeros(3, dtype=float)
    else:
        covariance = np.cov(centered, rowvar=False, bias=True)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        eigenvectors = np.asarray(eigenvectors[:, order], dtype=float)
        projected = centered @ eigenvectors
        semiaxes = 0.5 * (
            np.max(projected + radii_padding[:, None], axis=0)
            - np.min(projected - radii_padding[:, None], axis=0)
        )
        semiaxes = np.asarray(np.sort(semiaxes)[::-1], dtype=float)

    return {
        "semiaxes": semiaxes,
        "radius_of_gyration": radius_of_gyration,
        "max_radius": max_radius,
    }


def _lookup_ionic_radii(
    elements: list[str],
    *,
    ionic_radius_type: str = DEFAULT_IONIC_RADIUS_TYPE,
) -> tuple[np.ndarray, set[str]]:
    normalized_type = _normalize_ionic_radius_type(
        ionic_radius_type,
        default=DEFAULT_IONIC_RADIUS_TYPE,
    )
    normalized_elements = tuple(
        _normalize_element_symbol(raw_element) for raw_element in elements
    )
    radii, missing_elements = _cached_ionic_radii(
        normalized_elements,
        normalized_type,
    )
    return (
        np.asarray(radii, dtype=float),
        set(missing_elements),
    )


def _lookup_covalent_radii(elements: list[str]) -> tuple[np.ndarray, set[str]]:
    normalized_elements = tuple(
        _normalize_element_symbol(raw_element) for raw_element in elements
    )
    radii, missing_elements = _cached_covalent_radii(normalized_elements)
    return (
        np.asarray(radii, dtype=float),
        set(missing_elements),
    )


@lru_cache(maxsize=None)
def _normalize_element_symbol(raw_value: str) -> str:
    text = str(raw_value).strip()
    if not text:
        return ""
    return text[:1].upper() + text[1:].lower()


@lru_cache(maxsize=2048)
def _cached_ionic_radii(
    normalized_elements: tuple[str, ...],
    ionic_radius_type: str,
) -> tuple[tuple[float, ...], tuple[str, ...]]:
    if ionic_radius_type == "crystal":
        radii_lookup = APPROXIMATE_CRYSTAL_IONIC_RADII_ANGSTROM
        fallback_radius = (
            IONIC_RADIUS_FALLBACK_ANGSTROM
            + CRYSTAL_IONIC_RADIUS_OFFSET_ANGSTROM
        )
    else:
        radii_lookup = APPROXIMATE_EFFECTIVE_IONIC_RADII_ANGSTROM
        fallback_radius = IONIC_RADIUS_FALLBACK_ANGSTROM
    radii: list[float] = []
    missing_elements: list[str] = []
    for element in normalized_elements:
        radius = radii_lookup.get(element)
        if radius is None:
            missing_elements.append(element or "Unknown")
            radius = fallback_radius
        radii.append(float(radius))
    return tuple(radii), tuple(sorted(set(missing_elements)))


@lru_cache(maxsize=2048)
def _cached_covalent_radii(
    normalized_elements: tuple[str, ...],
) -> tuple[tuple[float, ...], tuple[str, ...]]:
    radii: list[float] = []
    missing_elements: list[str] = []
    for element in normalized_elements:
        radius = APPROXIMATE_COVALENT_RADII_ANGSTROM.get(element)
        if radius is None:
            missing_elements.append(element or "Unknown")
            radius = COVALENT_RADIUS_FALLBACK_ANGSTROM
        radii.append(float(radius))
    return tuple(radii), tuple(sorted(set(missing_elements)))


def _build_cluster_geometry_notes(
    *,
    mean_ionic_semiaxes: np.ndarray,
    mean_crystal_ionic_semiaxes: np.ndarray,
    mean_bond_length_semiaxes: np.ndarray,
    ionic_sphere_effective_radius: float,
    crystal_ionic_sphere_effective_radius: float,
    bond_length_sphere_effective_radius: float,
    recommendation: str,
    missing_ionic_elements: set[str],
    missing_covalent_elements: set[str],
    mean_atom_count: float,
    bond_length_single_atom_fallback_used: bool = False,
) -> str:
    notes = (
        "Computed ionic-radii and bond-length geometry descriptors. "
        f"Effective ionic sphere R_eff={ionic_sphere_effective_radius:.2f} A; "
        f"crystal ionic sphere R_eff={crystal_ionic_sphere_effective_radius:.2f} A; "
        f"bond-length sphere R_eff={bond_length_sphere_effective_radius:.2f} A. "
        "Mean effective ionic semiaxes "
        f"({mean_ionic_semiaxes[0]:.2f}, {mean_ionic_semiaxes[1]:.2f}, "
        f"{mean_ionic_semiaxes[2]:.2f}) A; "
        "mean crystal ionic semiaxes "
        f"({mean_crystal_ionic_semiaxes[0]:.2f}, "
        f"{mean_crystal_ionic_semiaxes[1]:.2f}, "
        f"{mean_crystal_ionic_semiaxes[2]:.2f}) A; "
        "mean bond-length semiaxes "
        f"({mean_bond_length_semiaxes[0]:.2f}, "
        f"{mean_bond_length_semiaxes[1]:.2f}, "
        f"{mean_bond_length_semiaxes[2]:.2f}) A. "
        f"Recommended {recommendation} approximation."
    )
    if mean_atom_count <= 1.0:
        notes += " Single-atom ionic mode uses the atom's ionic radius."
    if bond_length_single_atom_fallback_used:
        notes += (
            " Single-atom bond-length mode uses a covalent-radius proxy "
            "because no bond-length extent exists for a lone atom."
        )
    if missing_ionic_elements:
        notes += (
            " Missing ionic-radius entries used fallback "
            f"{IONIC_RADIUS_FALLBACK_ANGSTROM:.2f} A for: "
            + ", ".join(sorted(missing_ionic_elements))
            + "."
        )
    if missing_covalent_elements:
        notes += (
            " Missing covalent-radius entries used fallback "
            f"{COVALENT_RADIUS_FALLBACK_ANGSTROM:.2f} A for: "
            + ", ".join(sorted(missing_covalent_elements))
            + "."
        )
    return notes


def _equivalent_volume_radius(semiaxes: np.ndarray) -> float:
    safe_semiaxes = np.maximum(np.asarray(semiaxes, dtype=float), 0.0)
    if np.allclose(safe_semiaxes, 0.0):
        return 0.0
    return float(np.cbrt(np.prod(safe_semiaxes)))


def _bond_length_geometry_needs_covalent_padding(
    coordinates: np.ndarray,
    semiaxes: np.ndarray,
) -> bool:
    coordinates = np.asarray(coordinates, dtype=float)
    safe_semiaxes = np.asarray(semiaxes, dtype=float)
    if coordinates.shape[0] <= 1:
        return True
    return bool(np.any(safe_semiaxes <= 0.0))


def _apply_single_atom_bond_length_fallback(
    row: ClusterGeometryMetadataRow,
) -> bool:
    if float(row.mean_atom_count) > 1.0:
        return False

    fallback_radius = _single_atom_bond_length_radius_from_structure(
        row.structure
    )
    if fallback_radius <= 0.0:
        return False
    fallback_semiaxes = np.full(3, fallback_radius, dtype=float)

    dirty = False
    current_radius = float(row.bond_length_sphere_effective_radius)
    current_semiaxes = np.asarray(
        [
            row.bond_length_ellipsoid_semiaxis_a,
            row.bond_length_ellipsoid_semiaxis_b,
            row.bond_length_ellipsoid_semiaxis_c,
        ],
        dtype=float,
    )
    resembles_legacy_ionic_fallback = np.isclose(
        current_radius, float(row.ionic_sphere_effective_radius)
    ) and np.allclose(
        current_semiaxes,
        np.asarray(
            [
                row.ionic_ellipsoid_semiaxis_a,
                row.ionic_ellipsoid_semiaxis_b,
                row.ionic_ellipsoid_semiaxis_c,
            ],
            dtype=float,
        ),
    )
    if current_radius <= 0.0 or resembles_legacy_ionic_fallback:
        row.bond_length_sphere_effective_radius = fallback_radius
        dirty = True

    if np.any(current_semiaxes <= 0.0) or resembles_legacy_ionic_fallback:
        (
            row.bond_length_ellipsoid_semiaxis_a,
            row.bond_length_ellipsoid_semiaxis_b,
            row.bond_length_ellipsoid_semiaxis_c,
        ) = tuple(float(value) for value in fallback_semiaxes)
        dirty = True
    return dirty


def _single_atom_bond_length_radius_from_structure(structure: str) -> float:
    element = _normalize_element_symbol(structure)
    radius = APPROXIMATE_COVALENT_RADII_ANGSTROM.get(element)
    if radius is None:
        radius = COVALENT_RADIUS_FALLBACK_ANGSTROM
    return float(radius)


def _anisotropy_metric(semiaxes: np.ndarray) -> float:
    safe_semiaxes = np.maximum(np.asarray(semiaxes, dtype=float), 0.0)
    largest = float(np.max(safe_semiaxes)) if safe_semiaxes.size else 0.0
    smallest = float(np.min(safe_semiaxes)) if safe_semiaxes.size else 0.0
    if largest <= 0.0:
        return 1.0
    if smallest <= 0.0:
        return largest / max(float(np.mean(safe_semiaxes)), 1e-12)
    return largest / smallest


def _normalize_radii_type(
    value: object,
    *,
    default: str,
) -> str:
    text = str(value).strip().lower() if value is not None else ""
    return text if text in RADIUS_TYPE_OPTIONS else default


def _normalize_ionic_radius_type(
    value: object,
    *,
    default: str,
) -> str:
    text = str(value).strip().lower() if value is not None else ""
    return text if text in IONIC_RADIUS_TYPE_OPTIONS else default


def _upgrade_crystal_ionic_value(value: float) -> float:
    numeric_value = float(value)
    if numeric_value <= 0.0:
        return 0.0
    return numeric_value + CRYSTAL_IONIC_RADIUS_OFFSET_ANGSTROM


def _normalize_sf_approximation(
    value: object,
    *,
    fallback: str,
    allowed: tuple[str, ...] = STRUCTURE_FACTOR_RECOMMENDATIONS,
) -> str:
    normalized_allowed = _normalize_allowed_sf_approximations(allowed)
    text = str(value).strip().lower() if value is not None else ""
    if text in normalized_allowed:
        return text
    normalized_fallback = str(fallback).strip().lower()
    if normalized_fallback in normalized_allowed:
        return normalized_fallback
    return normalized_allowed[0]


def _normalize_allowed_sf_approximations(
    values: tuple[str, ...] | list[str] | None,
) -> tuple[str, ...]:
    if not values:
        return STRUCTURE_FACTOR_RECOMMENDATIONS
    normalized = tuple(
        dict.fromkeys(
            str(value).strip().lower()
            for value in values
            if str(value).strip().lower() in STRUCTURE_FACTOR_RECOMMENDATIONS
        )
    )
    return normalized or STRUCTURE_FACTOR_RECOMMENDATIONS


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


__all__ = [
    "APPROXIMATE_IONIC_RADII_ANGSTROM",
    "APPROXIMATE_CRYSTAL_IONIC_RADII_ANGSTROM",
    "APPROXIMATE_EFFECTIVE_IONIC_RADII_ANGSTROM",
    "CLUSTER_GEOMETRY_SCHEMA_VERSION",
    "CRYSTAL_IONIC_RADIUS_OFFSET_ANGSTROM",
    "DEFAULT_ANISOTROPY_THRESHOLD",
    "DEFAULT_IONIC_RADIUS_TYPE",
    "DEFAULT_RADIUS_TYPE",
    "IONIC_RADIUS_TYPE_OPTIONS",
    "RADIUS_TYPE_OPTIONS",
    "STRUCTURE_FACTOR_RECOMMENDATIONS",
    "ClusterGeometryMetadataRow",
    "ClusterGeometryMetadataTable",
    "apply_default_component_mapping",
    "cluster_identifier",
    "compute_cluster_geometry_metadata",
    "copy_cluster_geometry_rows",
    "load_cluster_geometry_metadata",
    "save_cluster_geometry_metadata",
    "synchronize_cluster_geometry_row",
    "synchronize_cluster_geometry_table",
]
