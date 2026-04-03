from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np

try:
    import xraydb  # type: ignore
except ImportError:  # pragma: no cover - optional until runtime
    xraydb = None

from saxshell.saxs.contrast.descriptors import load_parsed_contrast_structure
from saxshell.saxs.contrast.mesh import (
    DEFAULT_CONTRAST_MESH_PADDING_ANGSTROM,
    ContrastVolumeMesh,
    build_contrast_volume_mesh,
    points_inside_contrast_volume,
)
from saxshell.saxs.contrast.representatives import (
    ContrastRepresentativeSelectionResult,
)

AVOGADRO_NUMBER = 6.02214076e23
ANGSTROM3_PER_CM3 = 1.0e24

CONTRAST_SOLVENT_METHOD_NEAT = "neat_solvent_estimate"
CONTRAST_SOLVENT_METHOD_REFERENCE = "reference_structure"
CONTRAST_SOLVENT_METHOD_DIRECT = "direct_electron_density"
_VALID_SOLVENT_METHODS = {
    CONTRAST_SOLVENT_METHOD_NEAT,
    CONTRAST_SOLVENT_METHOD_REFERENCE,
    CONTRAST_SOLVENT_METHOD_DIRECT,
}

ContrastDensityProgressCallback = Callable[[int, int, str], None]
ContrastDensityLogCallback = Callable[[str], None]


def _require_xraydb() -> None:
    if xraydb is None:
        raise RuntimeError(
            "xraydb is required for contrast-mode electron-density estimation."
        )


def _now_text() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _normalize_method(value: object) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_")
    if normalized in _VALID_SOLVENT_METHODS:
        return normalized
    if normalized in {"direct", "direct_density", "manual_density"}:
        return CONTRAST_SOLVENT_METHOD_DIRECT
    if normalized in {
        "reference",
        "reference_file",
        "reference_structure_file",
    }:
        return CONTRAST_SOLVENT_METHOD_REFERENCE
    return CONTRAST_SOLVENT_METHOD_NEAT


def _normalized_exclude_elements(
    values: list[str] | tuple[str, ...] | set[str] | None,
) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                str(value).strip().upper()
                for value in (values or [])
                if str(value).strip()
            }
        )
    )


def _emit_progress(
    callback: ContrastDensityProgressCallback | None,
    processed: int,
    total: int,
    message: str,
) -> None:
    if callback is None:
        return
    callback(max(int(processed), 0), max(int(total), 1), str(message).strip())


def _emit_log(
    callback: ContrastDensityLogCallback | None,
    message: str,
) -> None:
    if callback is None:
        return
    text = str(message).strip()
    if text:
        callback(text)


def _sorted_counts(counts: Counter[str] | dict[str, int]) -> dict[str, int]:
    return {
        str(element): int(count)
        for element, count in sorted(dict(counts).items())
        if int(count) > 0
    }


def _electron_total_for_element_counts(counts: dict[str, int]) -> float:
    _require_xraydb()
    total = 0.0
    for element, count in counts.items():
        total += float(xraydb.atomic_number(str(element))) * float(count)
    return float(total)


def _formula_electron_total_and_mass(
    formula: str,
) -> tuple[dict[str, int], float, float]:
    _require_xraydb()
    parsed = xraydb.chemparse(str(formula).strip())
    if not parsed:
        raise ValueError(
            f"Unable to parse solvent stoichiometry formula {formula!r}."
        )
    counts = {
        str(element): int(float(amount))
        for element, amount in parsed.items()
        if float(amount) > 0.0
    }
    if not counts:
        raise ValueError(
            f"Solvent stoichiometry formula {formula!r} produced no elements."
        )
    total_electrons = 0.0
    formula_mass = 0.0
    for element, count in counts.items():
        total_electrons += float(xraydb.atomic_number(element)) * float(count)
        formula_mass += float(xraydb.atomic_mass(element)) * float(count)
    if formula_mass <= 0.0:
        raise ValueError(
            f"Solvent stoichiometry formula {formula!r} produced a non-positive molar mass."
        )
    return counts, float(total_electrons), float(formula_mass)


def _density_per_cm3_to_per_a3(value: float) -> float:
    return float(value) / ANGSTROM3_PER_CM3


@dataclass(slots=True, frozen=True)
class ContrastSolventDensitySettings:
    method: str = CONTRAST_SOLVENT_METHOD_NEAT
    solvent_formula: str | None = None
    solvent_density_g_per_ml: float | None = None
    reference_structure_file: Path | None = None
    direct_electron_density_e_per_a3: float | None = None

    @classmethod
    def from_values(
        cls,
        *,
        method: object = CONTRAST_SOLVENT_METHOD_NEAT,
        solvent_formula: object = None,
        solvent_density_g_per_ml: object = None,
        reference_structure_file: str | Path | None = None,
        direct_electron_density_e_per_a3: object = None,
    ) -> "ContrastSolventDensitySettings":
        density_value: float | None
        if (
            solvent_density_g_per_ml is None
            or str(solvent_density_g_per_ml).strip() == ""
        ):
            density_value = None
        else:
            density_value = float(solvent_density_g_per_ml)
        direct_density_value: float | None
        if (
            direct_electron_density_e_per_a3 is None
            or str(direct_electron_density_e_per_a3).strip() == ""
        ):
            direct_density_value = None
        else:
            direct_density_value = float(direct_electron_density_e_per_a3)
        return cls(
            method=_normalize_method(method),
            solvent_formula=(str(solvent_formula).strip() or None),
            solvent_density_g_per_ml=density_value,
            reference_structure_file=(
                None
                if reference_structure_file is None
                else Path(reference_structure_file).expanduser().resolve()
            ),
            direct_electron_density_e_per_a3=direct_density_value,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "solvent_formula": self.solvent_formula,
            "solvent_density_g_per_ml": self.solvent_density_g_per_ml,
            "reference_structure_file": (
                None
                if self.reference_structure_file is None
                else str(self.reference_structure_file)
            ),
            "direct_electron_density_e_per_a3": (
                self.direct_electron_density_e_per_a3
            ),
        }


@dataclass(slots=True, frozen=True)
class ContrastGeometryDensitySettings:
    solvent: ContrastSolventDensitySettings
    mesh_padding_angstrom: float = DEFAULT_CONTRAST_MESH_PADDING_ANGSTROM
    exclude_elements: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "solvent": self.solvent.to_dict(),
            "mesh_padding_angstrom": float(self.mesh_padding_angstrom),
            "exclude_elements": list(self.exclude_elements),
        }


@dataclass(slots=True, frozen=True)
class ContrastElectronDensityEstimate:
    method: str
    label: str
    volume_a3: float
    total_electrons: float
    electron_density_e_per_a3: float
    electron_density_e_per_cm3: float
    atom_count: int | None
    element_counts: dict[str, int]
    formula: str | None = None
    source_density_g_per_cm3: float | None = None
    reference_structure_file: Path | None = None
    reference_box_spans: tuple[float, float, float] | None = None
    translated_volume_center: tuple[float, float, float] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "label": self.label,
            "volume_a3": float(self.volume_a3),
            "total_electrons": float(self.total_electrons),
            "electron_density_e_per_a3": float(self.electron_density_e_per_a3),
            "electron_density_e_per_cm3": float(
                self.electron_density_e_per_cm3
            ),
            "atom_count": self.atom_count,
            "element_counts": dict(sorted(self.element_counts.items())),
            "formula": self.formula,
            "source_density_g_per_cm3": self.source_density_g_per_cm3,
            "reference_structure_file": (
                None
                if self.reference_structure_file is None
                else str(self.reference_structure_file)
            ),
            "reference_box_spans": (
                None
                if self.reference_box_spans is None
                else list(self.reference_box_spans)
            ),
            "translated_volume_center": (
                None
                if self.translated_volume_center is None
                else list(self.translated_volume_center)
            ),
        }


@dataclass(slots=True, frozen=True)
class ContrastGeometryDensityBinResult:
    structure: str
    motif: str
    representative_file: Path
    mesh: ContrastVolumeMesh
    mesh_json_path: Path
    density_json_path: Path
    cluster_electron_density: ContrastElectronDensityEstimate
    solvent_electron_density: ContrastElectronDensityEstimate
    contrast_electron_density_e_per_a3: float
    contrast_electron_density_e_per_cm3: float

    @property
    def display_label(self) -> str:
        if self.motif == "no_motif":
            return self.structure
        return f"{self.structure}/{self.motif}"

    def to_dict(self) -> dict[str, object]:
        return {
            "structure": self.structure,
            "motif": self.motif,
            "display_label": self.display_label,
            "representative_file": str(self.representative_file),
            "mesh": self.mesh.to_dict(),
            "mesh_json_path": str(self.mesh_json_path),
            "density_json_path": str(self.density_json_path),
            "cluster_electron_density": self.cluster_electron_density.to_dict(),
            "solvent_electron_density": self.solvent_electron_density.to_dict(),
            "contrast_electron_density_e_per_a3": float(
                self.contrast_electron_density_e_per_a3
            ),
            "contrast_electron_density_e_per_cm3": float(
                self.contrast_electron_density_e_per_cm3
            ),
        }


@dataclass(slots=True, frozen=True)
class ContrastGeometryDensityResult:
    output_dir: Path
    geometry_dir: Path
    density_dir: Path
    generated_at: str
    settings: ContrastGeometryDensitySettings
    bin_results: tuple[ContrastGeometryDensityBinResult, ...]
    summary_json_path: Path
    summary_table_path: Path
    summary_text_path: Path

    def summary_text(self) -> str:
        lines = [
            "Contrast geometry and electron-density calculation complete",
            f"Generated at: {self.generated_at}",
            f"Output folder: {self.output_dir}",
            f"Mesh folder: {self.geometry_dir}",
            f"Density folder: {self.density_dir}",
            f"Solvent method: {self.settings.solvent.method}",
            f"Processed bins: {len(self.bin_results)}",
        ]
        if self.settings.solvent.method == CONTRAST_SOLVENT_METHOD_NEAT:
            lines.append(
                "Neat-solvent estimate: "
                f"{self.settings.solvent.solvent_formula or 'unavailable'} at "
                f"{float(self.settings.solvent.solvent_density_g_per_ml or 0.0):.6g} g/mL"
            )
        elif self.settings.solvent.method == CONTRAST_SOLVENT_METHOD_DIRECT:
            lines.append(
                "Direct solvent electron density: "
                f"{float(self.settings.solvent.direct_electron_density_e_per_a3 or 0.0):.6f} e/A^3"
            )
        elif self.settings.solvent.reference_structure_file is not None:
            lines.append(
                "Reference solvent structure: "
                f"{self.settings.solvent.reference_structure_file}"
            )
        if self.bin_results:
            lines.extend(["", "Per-bin results"])
            for bin_result in self.bin_results:
                lines.append(
                    "  "
                    + f"{bin_result.display_label}: V={bin_result.mesh.volume_a3:.4f} A^3, "
                    + f"rho_cluster={bin_result.cluster_electron_density.electron_density_e_per_a3:.6f} e/A^3, "
                    + f"rho_solvent={bin_result.solvent_electron_density.electron_density_e_per_a3:.6f} e/A^3"
                )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        return {
            "version": 1,
            "generated_at": self.generated_at,
            "output_dir": str(self.output_dir),
            "geometry_dir": str(self.geometry_dir),
            "density_dir": str(self.density_dir),
            "settings": self.settings.to_dict(),
            "summary_json_path": str(self.summary_json_path),
            "summary_table_path": str(self.summary_table_path),
            "summary_text_path": str(self.summary_text_path),
            "bin_results": [result.to_dict() for result in self.bin_results],
        }


def _density_output_dirs(
    representative_output_dir: Path,
) -> tuple[Path, Path]:
    geometry_dir = representative_output_dir / "geometry"
    density_dir = representative_output_dir / "electron_density"
    geometry_dir.mkdir(parents=True, exist_ok=True)
    density_dir.mkdir(parents=True, exist_ok=True)
    return geometry_dir, density_dir


def _electron_density_from_counts(
    counts: dict[str, int],
    *,
    volume_a3: float,
    method: str,
    label: str,
    atom_count: int | None,
    formula: str | None = None,
    source_density_g_per_cm3: float | None = None,
    reference_structure_file: Path | None = None,
    reference_box_spans: tuple[float, float, float] | None = None,
    translated_volume_center: tuple[float, float, float] | None = None,
) -> ContrastElectronDensityEstimate:
    if volume_a3 <= 0.0:
        raise ValueError(
            "Electron-density estimation requires a positive volume."
        )
    total_electrons = _electron_total_for_element_counts(counts)
    density_e_per_a3 = float(total_electrons / float(volume_a3))
    return ContrastElectronDensityEstimate(
        method=method,
        label=label,
        volume_a3=float(volume_a3),
        total_electrons=float(total_electrons),
        electron_density_e_per_a3=density_e_per_a3,
        electron_density_e_per_cm3=density_e_per_a3 * ANGSTROM3_PER_CM3,
        atom_count=atom_count,
        element_counts=dict(sorted(counts.items())),
        formula=formula,
        source_density_g_per_cm3=source_density_g_per_cm3,
        reference_structure_file=reference_structure_file,
        reference_box_spans=reference_box_spans,
        translated_volume_center=translated_volume_center,
    )


def _cluster_electron_density(
    representative_file: Path,
    mesh: ContrastVolumeMesh,
    *,
    exclude_elements: list[str] | tuple[str, ...] | set[str] | None = None,
) -> ContrastElectronDensityEstimate:
    parsed = load_parsed_contrast_structure(
        representative_file,
        exclude_elements=exclude_elements,
    )
    return _electron_density_from_counts(
        parsed.element_counts,
        volume_a3=mesh.volume_a3,
        method="representative_mesh",
        label="Cluster representative mesh",
        atom_count=parsed.atom_count,
    )


def _neat_solvent_electron_density(
    settings: ContrastSolventDensitySettings,
    *,
    volume_a3: float,
) -> ContrastElectronDensityEstimate:
    formula = str(settings.solvent_formula or "").strip()
    if not formula:
        raise ValueError(
            "Neat-solvent density estimation requires a solvent stoichiometry formula."
        )
    density_g_per_cm3 = float(settings.solvent_density_g_per_ml or 0.0)
    if formula.lower() in {"vacuum", "none"}:
        return ContrastElectronDensityEstimate(
            method=CONTRAST_SOLVENT_METHOD_NEAT,
            label="Vacuum solvent",
            volume_a3=float(volume_a3),
            total_electrons=0.0,
            electron_density_e_per_a3=0.0,
            electron_density_e_per_cm3=0.0,
            atom_count=None,
            element_counts={},
            formula=formula,
            source_density_g_per_cm3=0.0,
        )
    if density_g_per_cm3 < 0.0:
        raise ValueError(
            "Neat-solvent density estimation requires a non-negative solvent density."
        )
    if density_g_per_cm3 == 0.0:
        return ContrastElectronDensityEstimate(
            method=CONTRAST_SOLVENT_METHOD_NEAT,
            label="Zero-density solvent estimate",
            volume_a3=float(volume_a3),
            total_electrons=0.0,
            electron_density_e_per_a3=0.0,
            electron_density_e_per_cm3=0.0,
            atom_count=None,
            element_counts={},
            formula=formula,
            source_density_g_per_cm3=0.0,
        )
    counts, total_electrons, formula_mass = _formula_electron_total_and_mass(
        formula
    )
    density_e_per_cm3 = float(
        density_g_per_cm3 * AVOGADRO_NUMBER * total_electrons / formula_mass
    )
    density_e_per_a3 = _density_per_cm3_to_per_a3(density_e_per_cm3)
    return ContrastElectronDensityEstimate(
        method=CONTRAST_SOLVENT_METHOD_NEAT,
        label="Neat-solvent estimate",
        volume_a3=float(volume_a3),
        total_electrons=float(density_e_per_a3 * float(volume_a3)),
        electron_density_e_per_a3=density_e_per_a3,
        electron_density_e_per_cm3=density_e_per_cm3,
        atom_count=None,
        element_counts=counts,
        formula=formula,
        source_density_g_per_cm3=density_g_per_cm3,
    )


def _direct_solvent_electron_density(
    settings: ContrastSolventDensitySettings,
    *,
    volume_a3: float,
) -> ContrastElectronDensityEstimate:
    direct_density = float(settings.direct_electron_density_e_per_a3 or 0.0)
    if direct_density < 0.0:
        raise ValueError(
            "Direct solvent electron-density mode requires a non-negative electron density in e/A^3."
        )
    return ContrastElectronDensityEstimate(
        method=CONTRAST_SOLVENT_METHOD_DIRECT,
        label="Direct electron-density input",
        volume_a3=float(volume_a3),
        total_electrons=float(direct_density * float(volume_a3)),
        electron_density_e_per_a3=direct_density,
        electron_density_e_per_cm3=direct_density * ANGSTROM3_PER_CM3,
        atom_count=None,
        element_counts={},
    )


def _reference_box_center(coordinates: np.ndarray) -> np.ndarray:
    coords = np.asarray(coordinates, dtype=float)
    return (np.min(coords, axis=0) + np.max(coords, axis=0)) / 2.0


def _reference_box_spans(
    coordinates: np.ndarray,
) -> tuple[float, float, float]:
    coords = np.asarray(coordinates, dtype=float)
    spans = np.max(coords, axis=0) - np.min(coords, axis=0)
    return (
        float(spans[0]),
        float(spans[1]),
        float(spans[2]),
    )


def _validate_reference_structure_size(
    *,
    reference_spans: tuple[float, float, float],
    largest_mesh_spans: tuple[float, float, float],
    reference_file: Path,
) -> None:
    if any(
        float(reference_spans[index]) <= float(largest_mesh_spans[index])
        for index in range(3)
    ):
        raise ValueError(
            "The reference solvent structure "
            f"{reference_file} is not larger than the largest retained "
            "representative mesh in every Cartesian direction. "
            f"Reference spans: {reference_spans}; "
            f"largest mesh spans: {largest_mesh_spans}."
        )


def _reference_solvent_electron_density(
    settings: ContrastSolventDensitySettings,
    *,
    mesh: ContrastVolumeMesh,
    reference_coordinates: np.ndarray,
    reference_elements: tuple[str, ...],
    reference_file: Path,
    reference_spans: tuple[float, float, float],
) -> ContrastElectronDensityEstimate:
    center = _reference_box_center(reference_coordinates)
    inside_mask = points_inside_contrast_volume(
        reference_coordinates,
        mesh,
        translated_center=center,
    )
    enclosed_elements = [
        str(reference_elements[index])
        for index, inside in enumerate(inside_mask.tolist())
        if bool(inside)
    ]
    enclosed_counts = _sorted_counts(Counter(enclosed_elements))
    if not enclosed_counts:
        raise ValueError(
            "No atoms from the reference solvent structure fell inside the "
            "translated retained volume."
        )
    return _electron_density_from_counts(
        enclosed_counts,
        volume_a3=mesh.volume_a3,
        method=CONTRAST_SOLVENT_METHOD_REFERENCE,
        label="Reference solvent structure",
        atom_count=len(enclosed_elements),
        reference_structure_file=reference_file,
        reference_box_spans=reference_spans,
        translated_volume_center=(
            float(center[0]),
            float(center[1]),
            float(center[2]),
        ),
    )


def _mesh_json_path(
    geometry_dir: Path,
    *,
    structure: str,
    motif: str,
) -> Path:
    suffix = structure if motif == "no_motif" else f"{structure}__{motif}"
    return geometry_dir / f"{suffix}_mesh.json"


def _density_json_path(
    density_dir: Path,
    *,
    structure: str,
    motif: str,
) -> Path:
    suffix = structure if motif == "no_motif" else f"{structure}__{motif}"
    return density_dir / f"{suffix}_density.json"


def compute_contrast_geometry_and_electron_density(
    representative_selection: ContrastRepresentativeSelectionResult,
    settings: ContrastGeometryDensitySettings,
    *,
    progress_callback: ContrastDensityProgressCallback | None = None,
    log_callback: ContrastDensityLogCallback | None = None,
) -> ContrastGeometryDensityResult:
    normalized_exclude_elements = _normalized_exclude_elements(
        settings.exclude_elements
    )
    geometry_dir, density_dir = _density_output_dirs(
        representative_selection.output_dir
    )
    total_bins = len(representative_selection.bin_results)
    if total_bins <= 0:
        raise ValueError(
            "Representative selection is empty, so no contrast geometry or "
            "electron-density calculation can be performed."
        )

    mesh_rows: list[
        tuple[
            Path, ContrastVolumeMesh, ContrastElectronDensityEstimate, str, str
        ]
    ] = []
    largest_mesh_spans = np.zeros(3, dtype=float)
    for index, representative_bin in enumerate(
        representative_selection.bin_results,
        start=1,
    ):
        representative_file = representative_bin.copied_representative_file
        parsed = load_parsed_contrast_structure(
            representative_file,
            exclude_elements=normalized_exclude_elements,
        )
        if parsed.atom_count <= 0:
            raise ValueError(
                "All atoms were excluded from "
                f"{representative_file.name} while building the contrast mesh."
            )
        mesh = build_contrast_volume_mesh(
            parsed.coordinates,
            source_file=representative_file,
            padding_angstrom=float(settings.mesh_padding_angstrom),
        )
        cluster_density = _cluster_electron_density(
            representative_file,
            mesh,
            exclude_elements=normalized_exclude_elements,
        )
        mesh_path = _mesh_json_path(
            geometry_dir,
            structure=representative_bin.structure,
            motif=representative_bin.motif,
        )
        mesh_path.write_text(
            json.dumps(mesh.to_dict(), indent=2) + "\n",
            encoding="utf-8",
        )
        largest_mesh_spans = np.maximum(
            largest_mesh_spans,
            np.asarray(mesh.spans, dtype=float),
        )
        mesh_rows.append(
            (
                mesh_path,
                mesh,
                cluster_density,
                representative_bin.structure,
                representative_bin.motif,
            )
        )
        _emit_progress(
            progress_callback,
            index - 1,
            total_bins,
            f"Built retained volume mesh for {representative_bin.display_label} ({index}/{total_bins})",
        )
        _emit_log(
            log_callback,
            f"Built {mesh.construction_method} mesh for {representative_bin.display_label} "
            f"with volume {mesh.volume_a3:.4f} A^3.",
        )

    reference_coordinates: np.ndarray | None = None
    reference_elements: tuple[str, ...] | None = None
    reference_file: Path | None = None
    reference_spans: tuple[float, float, float] | None = None
    if settings.solvent.method == CONTRAST_SOLVENT_METHOD_REFERENCE:
        reference_file = settings.solvent.reference_structure_file
        if reference_file is None:
            raise ValueError(
                "Reference-solvent electron-density estimation requires a reference XYZ/PDB file."
            )
        parsed_reference = load_parsed_contrast_structure(
            reference_file,
            exclude_elements=normalized_exclude_elements,
        )
        if parsed_reference.atom_count <= 0:
            raise ValueError(
                "All atoms were excluded from the reference solvent structure "
                f"{reference_file}."
            )
        reference_coordinates = np.asarray(
            parsed_reference.coordinates, dtype=float
        )
        reference_elements = parsed_reference.elements
        reference_spans = _reference_box_spans(reference_coordinates)
        _validate_reference_structure_size(
            reference_spans=reference_spans,
            largest_mesh_spans=(
                float(largest_mesh_spans[0]),
                float(largest_mesh_spans[1]),
                float(largest_mesh_spans[2]),
            ),
            reference_file=reference_file,
        )
        _emit_log(
            log_callback,
            f"Validated reference solvent structure {reference_file.name} against "
            f"largest retained mesh spans {tuple(float(value) for value in largest_mesh_spans)}.",
        )

    bin_results: list[ContrastGeometryDensityBinResult] = []
    for index, (
        mesh_path,
        mesh,
        cluster_density,
        structure,
        motif,
    ) in enumerate(mesh_rows, start=1):
        if settings.solvent.method == CONTRAST_SOLVENT_METHOD_REFERENCE:
            assert reference_coordinates is not None
            assert reference_elements is not None
            assert reference_file is not None
            assert reference_spans is not None
            solvent_density = _reference_solvent_electron_density(
                settings.solvent,
                mesh=mesh,
                reference_coordinates=reference_coordinates,
                reference_elements=reference_elements,
                reference_file=reference_file,
                reference_spans=reference_spans,
            )
        elif settings.solvent.method == CONTRAST_SOLVENT_METHOD_DIRECT:
            solvent_density = _direct_solvent_electron_density(
                settings.solvent,
                volume_a3=mesh.volume_a3,
            )
        else:
            solvent_density = _neat_solvent_electron_density(
                settings.solvent,
                volume_a3=mesh.volume_a3,
            )
        density_json_path = _density_json_path(
            density_dir,
            structure=structure,
            motif=motif,
        )
        contrast_e_per_a3 = float(
            cluster_density.electron_density_e_per_a3
            - solvent_density.electron_density_e_per_a3
        )
        contrast_e_per_cm3 = float(
            cluster_density.electron_density_e_per_cm3
            - solvent_density.electron_density_e_per_cm3
        )
        density_payload = {
            "structure": structure,
            "motif": motif,
            "mesh_json_path": str(mesh_path),
            "cluster_electron_density": cluster_density.to_dict(),
            "solvent_electron_density": solvent_density.to_dict(),
            "contrast_electron_density_e_per_a3": contrast_e_per_a3,
            "contrast_electron_density_e_per_cm3": contrast_e_per_cm3,
        }
        density_json_path.write_text(
            json.dumps(density_payload, indent=2) + "\n",
            encoding="utf-8",
        )
        bin_result = ContrastGeometryDensityBinResult(
            structure=structure,
            motif=motif,
            representative_file=Path(mesh.source_file or ""),
            mesh=mesh,
            mesh_json_path=mesh_path,
            density_json_path=density_json_path,
            cluster_electron_density=cluster_density,
            solvent_electron_density=solvent_density,
            contrast_electron_density_e_per_a3=contrast_e_per_a3,
            contrast_electron_density_e_per_cm3=contrast_e_per_cm3,
        )
        bin_results.append(bin_result)
        display_label = bin_result.display_label
        _emit_progress(
            progress_callback,
            index,
            total_bins,
            f"Computed electron-density terms for {display_label} ({index}/{total_bins})",
        )
        _emit_log(
            log_callback,
            f"Computed cluster/solvent electron densities for {display_label}: "
            f"{cluster_density.electron_density_e_per_a3:.6f} vs "
            f"{solvent_density.electron_density_e_per_a3:.6f} e/A^3.",
        )

    result = ContrastGeometryDensityResult(
        output_dir=representative_selection.output_dir,
        geometry_dir=geometry_dir,
        density_dir=density_dir,
        generated_at=_now_text(),
        settings=settings,
        bin_results=tuple(bin_results),
        summary_json_path=density_dir / "density_summary.json",
        summary_table_path=density_dir / "density_summary.tsv",
        summary_text_path=density_dir / "density_summary.txt",
    )
    result.summary_json_path.write_text(
        json.dumps(result.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )
    with result.summary_table_path.open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "structure",
                "motif",
                "representative_file",
                "mesh_json_path",
                "density_json_path",
                "mesh_volume_a3",
                "mesh_surface_area_a2",
                "cluster_electron_density_e_per_a3",
                "solvent_method",
                "solvent_electron_density_e_per_a3",
                "contrast_electron_density_e_per_a3",
            ]
        )
        for bin_result in result.bin_results:
            writer.writerow(
                [
                    bin_result.structure,
                    bin_result.motif,
                    str(bin_result.representative_file),
                    str(bin_result.mesh_json_path),
                    str(bin_result.density_json_path),
                    f"{bin_result.mesh.volume_a3:.8f}",
                    f"{bin_result.mesh.surface_area_a2:.8f}",
                    f"{bin_result.cluster_electron_density.electron_density_e_per_a3:.8f}",
                    bin_result.solvent_electron_density.method,
                    f"{bin_result.solvent_electron_density.electron_density_e_per_a3:.8f}",
                    f"{bin_result.contrast_electron_density_e_per_a3:.8f}",
                ]
            )
    result.summary_text_path.write_text(
        result.summary_text() + "\n",
        encoding="utf-8",
    )
    _emit_log(
        log_callback,
        f"Wrote contrast geometry/electron-density outputs to {result.density_dir}.",
    )
    return result
