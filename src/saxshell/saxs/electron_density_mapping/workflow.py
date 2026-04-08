from __future__ import annotations

import csv
import json
import re
from collections import Counter
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np

try:
    import xraydb  # type: ignore
except ImportError:  # pragma: no cover - optional import until runtime
    xraydb = None

from saxshell.saxs.contrast.electron_density import (
    CONTRAST_SOLVENT_METHOD_DIRECT,
    CONTRAST_SOLVENT_METHOD_NEAT,
    CONTRAST_SOLVENT_METHOD_REFERENCE,
    ContrastElectronDensityEstimate,
    ContrastSolventDensitySettings,
)
from saxshell.saxs.debye import load_structure_file
from saxshell.toolbox.blender.workflow import (
    PreviewAtomRecord,
    detect_bonds_records,
    read_structure_comment,
)

_SUPPORTED_STRUCTURE_SUFFIXES = {".pdb", ".xyz"}
_SUPPORTED_TRANSFORM_WINDOWS = {
    "none",
    "lorch",
    "cosine",
    "hanning",
    "parzen",
    "welch",
    "gaussian",
    "sine",
    "kaiser_bessel",
}
_AVOGADRO_NUMBER = 6.02214076e23
_ANGSTROM3_PER_CM3 = 1.0e24
ElectronDensityProgressCallback = Callable[[int, int, str], None]


def _require_xraydb() -> None:
    if xraydb is None:
        raise RuntimeError(
            "xraydb is required for electron-density mapping because the "
            "center-of-mass and electron-count calculations use element "
            "masses and atomic numbers."
        )


def _natural_sort_key(value: str) -> list[object]:
    return [
        int(token) if token.isdigit() else token.lower()
        for token in re.split(r"(\d+)", str(value))
        if token
    ]


def _normalized_element_symbol(raw_value: str) -> str:
    text = str(raw_value or "").strip()
    if not text:
        return ""
    return text[:1].upper() + text[1:].lower()


@lru_cache(maxsize=None)
def _atomic_mass(symbol: str) -> float:
    _require_xraydb()
    normalized = _normalized_element_symbol(symbol)
    if not normalized:
        raise ValueError("Encountered an empty element symbol.")
    try:
        return float(xraydb.atomic_mass(normalized))
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(
            f"Could not resolve an atomic mass for element {normalized!r}."
        ) from exc


@lru_cache(maxsize=None)
def _atomic_number(symbol: str) -> int:
    _require_xraydb()
    normalized = _normalized_element_symbol(symbol)
    if not normalized:
        raise ValueError("Encountered an empty element symbol.")
    try:
        return int(xraydb.atomic_number(normalized))
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(
            f"Could not resolve an atomic number for element {normalized!r}."
        ) from exc


def _sorted_count_dict(elements: tuple[str, ...]) -> dict[str, int]:
    counts = Counter(elements)
    return {
        element: int(counts[element])
        for element in sorted(counts)
        if int(counts[element]) > 0
    }


def _nearest_atom_to_point(
    coordinates: np.ndarray,
    point: np.ndarray,
) -> tuple[int, float]:
    coordinate_array = np.asarray(coordinates, dtype=float)
    point_array = np.asarray(point, dtype=float)
    distances = np.linalg.norm(coordinate_array - point_array, axis=1)
    index = int(np.argmin(distances))
    return index, float(distances[index])


def _geometric_center(coordinates: np.ndarray) -> np.ndarray:
    coordinate_array = np.asarray(coordinates, dtype=float)
    if coordinate_array.size == 0:
        return np.zeros(3, dtype=float)
    return np.asarray(np.mean(coordinate_array, axis=0), dtype=float)


def _default_reference_element(elements: tuple[str, ...]) -> str:
    available = {
        _normalized_element_symbol(element)
        for element in elements
        if _normalized_element_symbol(element)
    }
    if not available:
        raise ValueError("Could not determine a reference element.")
    return max(
        available,
        key=lambda element: (
            float(_atomic_mass(element)),
            int(_atomic_number(element)),
            str(element),
        ),
    )


def _reference_element_center_details(
    coordinates: np.ndarray,
    elements: tuple[str, ...],
    *,
    reference_element: str | None = None,
) -> tuple[str, np.ndarray, np.ndarray, float]:
    coordinate_array = np.asarray(coordinates, dtype=float)
    total_geometric_center = _geometric_center(coordinate_array)
    resolved_reference_element = _normalized_element_symbol(
        reference_element or ""
    )
    if not resolved_reference_element:
        resolved_reference_element = _default_reference_element(elements)
    element_mask = np.asarray(
        [element == resolved_reference_element for element in elements],
        dtype=bool,
    )
    if not np.any(element_mask):
        raise ValueError(
            "Reference element "
            f"{resolved_reference_element!r} is not present in the structure."
        )
    reference_coordinates = coordinate_array[element_mask]
    reference_geometric_center = _geometric_center(reference_coordinates)
    offset = float(
        np.linalg.norm(reference_geometric_center - total_geometric_center)
    )
    return (
        resolved_reference_element,
        total_geometric_center,
        reference_geometric_center,
        offset,
    )


def _centered_coordinates_from_origin(
    coordinates: np.ndarray,
    origin: np.ndarray,
) -> tuple[np.ndarray, float]:
    centered_coordinates = np.asarray(coordinates, dtype=float) - np.asarray(
        origin,
        dtype=float,
    )
    radial_distances = np.linalg.norm(centered_coordinates, axis=1)
    rmax = float(np.max(radial_distances)) if radial_distances.size else 0.0
    return centered_coordinates, rmax


def _gaussian_smooth_profile(
    radial_centers: np.ndarray,
    values: np.ndarray,
    *,
    sigma_a: float,
) -> np.ndarray:
    centers = np.asarray(radial_centers, dtype=float)
    profile = np.asarray(values, dtype=float)
    sigma_value = max(float(sigma_a), 0.0)
    if profile.size <= 1 or sigma_value <= 0.0:
        return profile.copy()
    distance_matrix = centers[:, np.newaxis] - centers[np.newaxis, :]
    kernel = np.exp(
        -0.5 * np.square(distance_matrix / max(sigma_value, 1.0e-12))
    )
    normalizers = np.sum(kernel, axis=1, keepdims=True)
    kernel = np.divide(
        kernel,
        normalizers,
        out=np.zeros_like(kernel, dtype=float),
        where=normalizers > 0.0,
    )
    return np.asarray(kernel @ profile, dtype=float)


def _trapezoid_integral(
    values: np.ndarray,
    *,
    x: np.ndarray,
    axis: int = -1,
) -> np.ndarray:
    if hasattr(np, "trapezoid"):
        return np.asarray(
            np.trapezoid(values, x=x, axis=axis),
            dtype=float,
        )
    return np.asarray(
        np.trapz(values, x=x, axis=axis),
        dtype=float,
    )


def _normalized_transform_window_name(value: object) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_")
    if normalized in {"hann", "hanning"}:
        return "hanning"
    if normalized in {"kaiser", "kaiser_bessel", "kaiserbessel"}:
        return "kaiser_bessel"
    return normalized


def _extend_profile_to_origin(
    radial_values: np.ndarray,
    density_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    radii = np.asarray(radial_values, dtype=float)
    densities = np.asarray(density_values, dtype=float)
    if radii.size == 0 or densities.size == 0:
        return radii.copy(), densities.copy()
    if abs(float(radii[0])) <= 1.0e-12:
        return radii.copy(), densities.copy()
    return (
        np.asarray(np.concatenate(([0.0], radii)), dtype=float),
        np.asarray(
            np.concatenate(([float(densities[0])], densities)),
            dtype=float,
        ),
    )


def _emit_progress(
    callback: ElectronDensityProgressCallback | None,
    current: int,
    total: int,
    message: str,
) -> None:
    if callback is None:
        return
    callback(int(current), int(total), str(message))


def _validate_supported_structure_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Structure file not found: {resolved}")
    if resolved.suffix.lower() not in _SUPPORTED_STRUCTURE_SUFFIXES:
        raise ValueError(
            f"Unsupported structure file format for {resolved}. Choose an "
            "XYZ or PDB file."
        )
    return resolved


@dataclass(slots=True, frozen=True)
class ElectronDensityMeshSettings:
    rstep: float = 0.05
    theta_divisions: int = 120
    phi_divisions: int = 60
    rmax: float = 8.0

    def normalized(self) -> "ElectronDensityMeshSettings":
        rstep = float(self.rstep)
        theta_divisions = int(self.theta_divisions)
        phi_divisions = int(self.phi_divisions)
        rmax = float(self.rmax)
        if rstep <= 0.0:
            raise ValueError("rstep must be greater than zero.")
        if theta_divisions < 2:
            raise ValueError("Theta divisions must be at least 2.")
        if phi_divisions < 2:
            raise ValueError("Phi divisions must be at least 2.")
        if rmax <= 0.0:
            raise ValueError("rmax must be greater than zero.")
        return ElectronDensityMeshSettings(
            rstep=rstep,
            theta_divisions=theta_divisions,
            phi_divisions=phi_divisions,
            rmax=rmax,
        )

    def to_dict(self) -> dict[str, object]:
        normalized = self.normalized()
        return {
            "rstep_a": float(normalized.rstep),
            "theta_divisions": int(normalized.theta_divisions),
            "phi_divisions": int(normalized.phi_divisions),
            "rmax_a": float(normalized.rmax),
        }


@dataclass(slots=True, frozen=True)
class ElectronDensitySmearingSettings:
    debye_waller_factor: float = 0.006

    def normalized(self) -> "ElectronDensitySmearingSettings":
        factor = max(float(self.debye_waller_factor), 0.0)
        return ElectronDensitySmearingSettings(
            debye_waller_factor=factor,
        )

    @property
    def gaussian_sigma_a(self) -> float:
        return float(np.sqrt(self.normalized().debye_waller_factor))

    def to_dict(self) -> dict[str, float]:
        normalized = self.normalized()
        return {
            "debye_waller_factor_a2": float(normalized.debye_waller_factor),
            "gaussian_sigma_a": float(normalized.gaussian_sigma_a),
        }


@dataclass(slots=True, frozen=True)
class ElectronDensityFourierTransformSettings:
    r_min: float = 0.0
    r_max: float = 1.0
    window_function: str = "none"
    resampling_points: int = 1024
    q_min: float = 0.02
    q_max: float = 10.0
    q_step: float = 0.02
    use_solvent_subtracted_profile: bool = True
    log_q_axis: bool = True
    log_intensity_axis: bool = True

    def normalized(self) -> "ElectronDensityFourierTransformSettings":
        r_min = max(float(self.r_min), 0.0)
        r_max = float(self.r_max)
        resampling_points = int(self.resampling_points)
        q_min = max(float(self.q_min), 0.0)
        q_max = float(self.q_max)
        q_step = float(self.q_step)
        window_function = (
            _normalized_transform_window_name(self.window_function) or "none"
        )
        if r_max <= r_min:
            raise ValueError("Transform rmax must be greater than r min.")
        if resampling_points < 8:
            raise ValueError(
                "Fourier-transform resampling points must be at least 8."
            )
        if q_max <= q_min:
            raise ValueError("Transform q max must be greater than q min.")
        if q_step <= 0.0:
            raise ValueError("Transform q step must be greater than zero.")
        if window_function not in _SUPPORTED_TRANSFORM_WINDOWS:
            raise ValueError(
                "Unsupported transform window. Choose from: "
                + ", ".join(sorted(_SUPPORTED_TRANSFORM_WINDOWS))
                + "."
            )
        return ElectronDensityFourierTransformSettings(
            r_min=r_min,
            r_max=r_max,
            window_function=window_function,
            resampling_points=resampling_points,
            q_min=q_min,
            q_max=q_max,
            q_step=q_step,
            use_solvent_subtracted_profile=bool(
                self.use_solvent_subtracted_profile
            ),
            log_q_axis=bool(self.log_q_axis),
            log_intensity_axis=bool(self.log_intensity_axis),
        )

    def to_dict(self) -> dict[str, object]:
        normalized = self.normalized()
        return {
            "r_min_a": float(normalized.r_min),
            "r_max_a": float(normalized.r_max),
            "window_function": str(normalized.window_function),
            "resampling_points": int(normalized.resampling_points),
            "q_min_a_inverse": float(normalized.q_min),
            "q_max_a_inverse": float(normalized.q_max),
            "q_step_a_inverse": float(normalized.q_step),
            "use_solvent_subtracted_profile": bool(
                normalized.use_solvent_subtracted_profile
            ),
            "log_q_axis": bool(normalized.log_q_axis),
            "log_intensity_axis": bool(normalized.log_intensity_axis),
        }


@dataclass(slots=True, frozen=True)
class ElectronDensityInputInspection:
    selection_path: Path
    input_mode: str
    structure_files: tuple[Path, ...]
    reference_file: Path
    format_counts: dict[str, int]

    @property
    def total_files(self) -> int:
        return int(len(self.structure_files))

    def notes(self) -> tuple[str, ...]:
        if self.input_mode == "folder":
            return (
                "Folder mode previews the first valid structure file as the "
                "visualization reference.",
                "Running the calculation averages the electron-density "
                "profiles across every valid structure in the folder and "
                "tracks the variance across the ensemble.",
            )
        return (
            "Single-structure mode is the primary validation path in this "
            "first implementation.",
        )


@dataclass(slots=True, frozen=True)
class ElectronDensityStructure:
    file_path: Path
    display_label: str
    structure_comment: str
    coordinates: np.ndarray
    centered_coordinates: np.ndarray
    elements: tuple[str, ...]
    element_counts: dict[str, int]
    atomic_numbers: np.ndarray
    atomic_masses: np.ndarray
    center_of_mass: np.ndarray
    geometric_center: np.ndarray
    reference_element: str
    reference_element_geometric_center: np.ndarray
    reference_element_offset_from_geometric_center: float
    active_center: np.ndarray
    center_mode: str
    nearest_atom_index: int
    nearest_atom_distance: float
    bonds: tuple[tuple[int, int], ...]
    rmax: float

    @property
    def atom_count(self) -> int:
        return int(len(self.elements))

    @property
    def domain_max_radius(self) -> float:
        return float(max(self.rmax, 0.0))

    @property
    def nearest_atom_element(self) -> str:
        return str(self.elements[int(self.nearest_atom_index)])

    @property
    def nearest_atom_coordinates(self) -> np.ndarray:
        return np.asarray(
            self.coordinates[int(self.nearest_atom_index)],
            dtype=float,
        )


@dataclass(slots=True, frozen=True)
class ElectronDensityMeshGeometry:
    settings: ElectronDensityMeshSettings
    domain_max_radius: float
    radial_edges: np.ndarray
    theta_edges: np.ndarray
    phi_edges: np.ndarray

    @property
    def shell_count(self) -> int:
        return int(max(len(self.radial_edges) - 1, 0))


@dataclass(slots=True, frozen=True)
class ElectronDensityMemberSummary:
    file_path: Path
    display_label: str
    center_of_mass: np.ndarray
    geometric_center: np.ndarray
    reference_element: str
    reference_element_geometric_center: np.ndarray
    reference_element_offset_from_geometric_center: float
    active_center: np.ndarray
    center_mode: str
    nearest_atom_index: int
    nearest_atom_element: str
    nearest_atom_distance: float
    rmax: float
    total_electrons: float
    excluded_atom_count: int
    excluded_electron_count: float


@dataclass(slots=True, frozen=True)
class ElectronDensityProfileResult:
    structure: ElectronDensityStructure
    input_mode: str
    source_files: tuple[Path, ...]
    source_structure_count: int
    member_summaries: tuple[ElectronDensityMemberSummary, ...]
    member_orientation_average_densities: tuple[np.ndarray, ...]
    mesh_geometry: ElectronDensityMeshGeometry
    smearing_settings: ElectronDensitySmearingSettings
    radial_centers: np.ndarray
    orientation_average_density: np.ndarray
    orientation_density_variance: np.ndarray
    orientation_density_stddev: np.ndarray
    smeared_orientation_average_density: np.ndarray
    smeared_orientation_density_variance: np.ndarray
    smeared_orientation_density_stddev: np.ndarray
    shell_volume_average_density: np.ndarray
    shell_electron_counts: np.ndarray
    shell_volumes: np.ndarray
    excluded_atom_count: int
    excluded_electron_count: float
    solvent_contrast: "ElectronDensitySolventContrastResult | None" = None


@dataclass(slots=True, frozen=True)
class ElectronDensitySolventContrastResult:
    settings: ContrastSolventDensitySettings
    solvent_name: str
    density_estimate: ContrastElectronDensityEstimate
    solvent_density_e_per_a3: float
    cutoff_radius_a: float | None
    solvent_subtracted_smeared_density: np.ndarray

    @property
    def legend_label(self) -> str:
        return (
            f"{self.solvent_name} "
            f"({self.solvent_density_e_per_a3:.5f} e-/ Å³)"
        )

    @property
    def cutoff_label(self) -> str | None:
        if self.cutoff_radius_a is None:
            return None
        return f"Cutoff = {self.cutoff_radius_a:.3f} Å"

    def to_dict(self) -> dict[str, object]:
        return {
            "settings": self.settings.to_dict(),
            "solvent_name": self.solvent_name,
            "density_estimate": self.density_estimate.to_dict(),
            "solvent_density_e_per_a3": float(self.solvent_density_e_per_a3),
            "cutoff_radius_a": (
                None
                if self.cutoff_radius_a is None
                else float(self.cutoff_radius_a)
            ),
            "solvent_subtracted_smeared_density_e_per_a3": [
                float(value)
                for value in np.asarray(
                    self.solvent_subtracted_smeared_density,
                    dtype=float,
                )
            ],
        }


@dataclass(slots=True, frozen=True)
class ElectronDensityFourierTransformPreview:
    settings: ElectronDensityFourierTransformSettings
    source_profile_label: str
    source_radial_values: np.ndarray
    source_density_values: np.ndarray
    resampled_r_values: np.ndarray
    resampled_density_values: np.ndarray
    window_values: np.ndarray
    windowed_density_values: np.ndarray
    available_r_min: float
    available_r_max: float
    resampling_step_a: float
    nyquist_q_max_a_inverse: float
    independent_q_step_a_inverse: float
    q_grid_is_oversampled: bool
    q_max_was_clamped: bool
    notes: tuple[str, ...]


@dataclass(slots=True, frozen=True)
class ElectronDensityScatteringTransformResult:
    preview: ElectronDensityFourierTransformPreview
    q_values: np.ndarray
    scattering_amplitude: np.ndarray
    intensity: np.ndarray


@dataclass(slots=True, frozen=True)
class ElectronDensityOutputArtifacts:
    csv_path: Path
    json_path: Path


def _electron_total_for_element_counts(counts: dict[str, int]) -> float:
    total = 0.0
    for element, count in counts.items():
        total += float(_atomic_number(str(element))) * float(count)
    return float(total)


def _parse_formula_electron_totals(
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
    electron_total = 0.0
    molar_mass = 0.0
    for element, count in counts.items():
        electron_total += float(_atomic_number(element)) * float(count)
        molar_mass += float(_atomic_mass(element)) * float(count)
    if molar_mass <= 0.0:
        raise ValueError(
            f"Solvent stoichiometry formula {formula!r} produced a non-positive molar mass."
        )
    return counts, float(electron_total), float(molar_mass)


def _estimate_solvent_density(
    settings: ContrastSolventDensitySettings,
    *,
    solvent_name: str | None,
    domain_volume_a3: float,
) -> tuple[str, ContrastElectronDensityEstimate]:
    method = str(settings.method or "").strip().lower()
    if method == CONTRAST_SOLVENT_METHOD_DIRECT:
        direct_density = float(
            settings.direct_electron_density_e_per_a3 or 0.0
        )
        if direct_density < 0.0:
            raise ValueError(
                "Direct solvent electron-density mode requires a non-negative value."
            )
        return (
            str(solvent_name or "Direct solvent"),
            ContrastElectronDensityEstimate(
                method=CONTRAST_SOLVENT_METHOD_DIRECT,
                label="Direct electron-density input",
                volume_a3=float(domain_volume_a3),
                total_electrons=float(
                    direct_density * float(domain_volume_a3)
                ),
                electron_density_e_per_a3=float(direct_density),
                electron_density_e_per_cm3=float(
                    direct_density * _ANGSTROM3_PER_CM3
                ),
                atom_count=None,
                element_counts={},
            ),
        )

    if method == CONTRAST_SOLVENT_METHOD_REFERENCE:
        reference_file = settings.reference_structure_file
        if reference_file is None:
            raise ValueError(
                "Choose a reference solvent XYZ or PDB file before computing the solvent electron density."
            )
        resolved_reference = _validate_supported_structure_path(reference_file)
        coordinates, raw_elements = load_structure_file(resolved_reference)
        coordinate_array = np.asarray(coordinates, dtype=float)
        if coordinate_array.ndim != 2 or coordinate_array.shape[1] != 3:
            raise ValueError(
                f"Expected Nx3 coordinates from reference solvent file {resolved_reference}."
            )
        elements = tuple(
            _normalized_element_symbol(element) for element in raw_elements
        )
        if not elements:
            raise ValueError(
                f"No atoms were parsed from the reference solvent file {resolved_reference}."
            )
        spans = np.max(coordinate_array, axis=0) - np.min(
            coordinate_array,
            axis=0,
        )
        if np.any(np.asarray(spans, dtype=float) <= 0.0):
            raise ValueError(
                "Reference solvent density estimation requires a solvent structure "
                "with a non-zero Cartesian span in x, y, and z so a finite box volume can be estimated."
            )
        volume_a3 = float(np.prod(spans))
        counts = _sorted_count_dict(elements)
        total_electrons = _electron_total_for_element_counts(counts)
        box_center = (
            np.min(coordinate_array, axis=0) + np.max(coordinate_array, axis=0)
        ) * 0.5
        return (
            str(solvent_name or resolved_reference.stem),
            ContrastElectronDensityEstimate(
                method=CONTRAST_SOLVENT_METHOD_REFERENCE,
                label="Reference solvent structure",
                volume_a3=volume_a3,
                total_electrons=float(total_electrons),
                electron_density_e_per_a3=float(total_electrons / volume_a3),
                electron_density_e_per_cm3=float(
                    total_electrons / volume_a3 * _ANGSTROM3_PER_CM3
                ),
                atom_count=len(elements),
                element_counts=counts,
                reference_structure_file=resolved_reference,
                reference_box_spans=(
                    float(spans[0]),
                    float(spans[1]),
                    float(spans[2]),
                ),
                translated_volume_center=(
                    float(box_center[0]),
                    float(box_center[1]),
                    float(box_center[2]),
                ),
            ),
        )

    formula = str(settings.solvent_formula or "").strip()
    if not formula:
        raise ValueError(
            "Enter a solvent stoichiometry formula before computing the solvent electron density."
        )
    density_g_per_cm3 = float(settings.solvent_density_g_per_ml or 0.0)
    display_name = str(solvent_name or formula)
    if formula.lower() in {"vacuum", "none"} or density_g_per_cm3 == 0.0:
        label = (
            "Vacuum solvent"
            if formula.lower() in {"vacuum", "none"}
            else "Zero-density solvent estimate"
        )
        return (
            display_name,
            ContrastElectronDensityEstimate(
                method=CONTRAST_SOLVENT_METHOD_NEAT,
                label=label,
                volume_a3=float(domain_volume_a3),
                total_electrons=0.0,
                electron_density_e_per_a3=0.0,
                electron_density_e_per_cm3=0.0,
                atom_count=None,
                element_counts={},
                formula=formula,
                source_density_g_per_cm3=max(density_g_per_cm3, 0.0),
            ),
        )
    if density_g_per_cm3 < 0.0:
        raise ValueError(
            "Neat-solvent density estimation requires a non-negative solvent density."
        )
    counts, formula_electrons, formula_mass = _parse_formula_electron_totals(
        formula
    )
    density_e_per_cm3 = float(
        density_g_per_cm3 * _AVOGADRO_NUMBER * formula_electrons / formula_mass
    )
    density_e_per_a3 = float(density_e_per_cm3 / _ANGSTROM3_PER_CM3)
    return (
        display_name,
        ContrastElectronDensityEstimate(
            method=CONTRAST_SOLVENT_METHOD_NEAT,
            label="Neat-solvent estimate",
            volume_a3=float(domain_volume_a3),
            total_electrons=float(density_e_per_a3 * float(domain_volume_a3)),
            electron_density_e_per_a3=density_e_per_a3,
            electron_density_e_per_cm3=density_e_per_cm3,
            atom_count=None,
            element_counts=counts,
            formula=formula,
            source_density_g_per_cm3=density_g_per_cm3,
        ),
    )


def _highest_r_profile_crossing(
    radial_values: np.ndarray,
    residual_values: np.ndarray,
) -> float | None:
    radii = np.asarray(radial_values, dtype=float)
    residual = np.asarray(residual_values, dtype=float)
    if radii.size != residual.size or radii.size == 0:
        return None
    tolerance = 1.0e-12
    for index in range(len(radii) - 1, 0, -1):
        left_value = float(residual[index - 1])
        right_value = float(residual[index])
        if abs(right_value) <= tolerance:
            return float(radii[index])
        if abs(left_value) <= tolerance:
            return float(radii[index - 1])
        if left_value * right_value > 0.0:
            continue
        left_radius = float(radii[index - 1])
        right_radius = float(radii[index])
        if abs(right_value - left_value) <= tolerance:
            return float(right_radius)
        fraction = -left_value / (right_value - left_value)
        fraction = float(np.clip(fraction, 0.0, 1.0))
        return left_radius + (right_radius - left_radius) * fraction
    if abs(float(residual[-1])) <= tolerance:
        return float(radii[-1])
    if abs(float(residual[0])) <= tolerance:
        return float(radii[0])
    return None


def _build_solvent_contrast_result(
    result: ElectronDensityProfileResult,
    *,
    settings: ContrastSolventDensitySettings,
    solvent_name: str,
    density_estimate: ContrastElectronDensityEstimate,
) -> ElectronDensitySolventContrastResult:
    solvent_density_e_per_a3 = float(
        density_estimate.electron_density_e_per_a3
    )
    solvent_subtracted = (
        np.asarray(
            result.smeared_orientation_average_density,
            dtype=float,
        )
        - solvent_density_e_per_a3
    )
    cutoff_radius_a = _highest_r_profile_crossing(
        np.asarray(result.radial_centers, dtype=float),
        solvent_subtracted,
    )
    return ElectronDensitySolventContrastResult(
        settings=settings,
        solvent_name=str(solvent_name).strip() or "Solvent",
        density_estimate=density_estimate,
        solvent_density_e_per_a3=solvent_density_e_per_a3,
        cutoff_radius_a=cutoff_radius_a,
        solvent_subtracted_smeared_density=np.asarray(
            solvent_subtracted,
            dtype=float,
        ),
    )


def apply_solvent_contrast_to_profile_result(
    result: ElectronDensityProfileResult,
    settings: ContrastSolventDensitySettings,
    *,
    solvent_name: str | None = None,
) -> ElectronDensityProfileResult:
    display_name, density_estimate = _estimate_solvent_density(
        settings,
        solvent_name=solvent_name,
        domain_volume_a3=float(
            np.sum(np.asarray(result.shell_volumes, dtype=float))
        ),
    )
    return replace(
        result,
        solvent_contrast=_build_solvent_contrast_result(
            result,
            settings=settings,
            solvent_name=display_name,
            density_estimate=density_estimate,
        ),
    )


def inspect_structure_input(
    selection_path: str | Path,
) -> ElectronDensityInputInspection:
    resolved = Path(selection_path).expanduser().resolve()
    if resolved.is_file():
        file_path = _validate_supported_structure_path(resolved)
        format_counts = {file_path.suffix.lower().lstrip("."): 1}
        return ElectronDensityInputInspection(
            selection_path=resolved,
            input_mode="file",
            structure_files=(file_path,),
            reference_file=file_path,
            format_counts=format_counts,
        )
    if not resolved.is_dir():
        raise FileNotFoundError(f"Input path was not found: {resolved}")

    structure_files = tuple(
        sorted(
            (
                child.resolve()
                for child in resolved.iterdir()
                if child.is_file()
                and child.suffix.lower() in _SUPPORTED_STRUCTURE_SUFFIXES
            ),
            key=lambda candidate: _natural_sort_key(candidate.name),
        )
    )
    if not structure_files:
        raise ValueError(
            f"No XYZ or PDB files were found in the folder {resolved}."
        )
    format_counts: dict[str, int] = {}
    for structure_file in structure_files:
        key = structure_file.suffix.lower().lstrip(".")
        format_counts[key] = format_counts.get(key, 0) + 1
    return ElectronDensityInputInspection(
        selection_path=resolved,
        input_mode="folder",
        structure_files=structure_files,
        reference_file=structure_files[0],
        format_counts=dict(sorted(format_counts.items())),
    )


def load_electron_density_structure(
    file_path: str | Path,
    *,
    center_mode: str = "center_of_mass",
    reference_element: str | None = None,
) -> ElectronDensityStructure:
    resolved = _validate_supported_structure_path(file_path)
    coordinates, raw_elements = load_structure_file(resolved)
    coordinates_array = np.asarray(coordinates, dtype=float)
    elements = tuple(
        _normalized_element_symbol(element) for element in raw_elements
    )
    if coordinates_array.ndim != 2 or coordinates_array.shape[1] != 3:
        raise ValueError(
            f"Expected Nx3 coordinates from structure file {resolved}."
        )
    if len(elements) != len(coordinates_array):
        raise ValueError(
            f"Element and coordinate counts differ in {resolved}: "
            f"{len(elements)} elements for {len(coordinates_array)} positions."
        )
    if not elements:
        raise ValueError(f"No atoms were parsed from {resolved}.")

    atomic_masses = np.asarray(
        [_atomic_mass(element) for element in elements],
        dtype=float,
    )
    atomic_numbers = np.asarray(
        [_atomic_number(element) for element in elements],
        dtype=float,
    )
    total_mass = float(np.sum(atomic_masses))
    if total_mass <= 0.0:
        raise ValueError(
            f"Could not compute a positive total mass for {resolved}."
        )
    center_of_mass = (
        np.sum(
            coordinates_array * atomic_masses[:, np.newaxis],
            axis=0,
        )
        / total_mass
    )
    (
        resolved_reference_element,
        geometric_center,
        reference_element_geometric_center,
        reference_element_offset,
    ) = _reference_element_center_details(
        coordinates_array,
        elements,
        reference_element=reference_element,
    )
    nearest_atom_index, nearest_atom_distance = _nearest_atom_to_point(
        coordinates_array,
        center_of_mass,
    )
    if center_mode == "center_of_mass":
        active_center = np.asarray(center_of_mass, dtype=float)
    elif center_mode == "nearest_atom":
        active_center = np.asarray(
            coordinates_array[int(nearest_atom_index)],
            dtype=float,
        )
    elif center_mode == "reference_element":
        active_center = np.asarray(
            reference_element_geometric_center,
            dtype=float,
        )
    else:
        raise ValueError(f"Unsupported center mode: {center_mode!r}")
    centered_coordinates, rmax = _centered_coordinates_from_origin(
        coordinates_array,
        active_center,
    )

    preview_atoms = tuple(
        PreviewAtomRecord(
            element=element,
            position=tuple(float(value) for value in coordinates_array[index]),
        )
        for index, element in enumerate(elements)
    )
    bonds = detect_bonds_records(preview_atoms)
    comment = read_structure_comment(resolved)
    display_label = comment.strip() or resolved.name
    return ElectronDensityStructure(
        file_path=resolved,
        display_label=display_label,
        structure_comment=comment.strip(),
        coordinates=coordinates_array,
        centered_coordinates=centered_coordinates,
        elements=elements,
        element_counts=_sorted_count_dict(elements),
        atomic_numbers=atomic_numbers,
        atomic_masses=atomic_masses,
        center_of_mass=np.asarray(center_of_mass, dtype=float),
        geometric_center=np.asarray(geometric_center, dtype=float),
        reference_element=str(resolved_reference_element),
        reference_element_geometric_center=np.asarray(
            reference_element_geometric_center,
            dtype=float,
        ),
        reference_element_offset_from_geometric_center=float(
            reference_element_offset
        ),
        active_center=np.asarray(active_center, dtype=float),
        center_mode=str(center_mode),
        nearest_atom_index=int(nearest_atom_index),
        nearest_atom_distance=float(nearest_atom_distance),
        bonds=bonds,
        rmax=rmax,
    )


def recenter_electron_density_structure(
    structure: ElectronDensityStructure,
    *,
    center_mode: str,
    reference_element: str | None = None,
) -> ElectronDensityStructure:
    (
        resolved_reference_element,
        geometric_center,
        reference_element_geometric_center,
        reference_element_offset,
    ) = _reference_element_center_details(
        structure.coordinates,
        structure.elements,
        reference_element=reference_element or structure.reference_element,
    )
    if center_mode == "center_of_mass":
        active_center = np.asarray(structure.center_of_mass, dtype=float)
    elif center_mode == "nearest_atom":
        active_center = np.asarray(
            structure.coordinates[int(structure.nearest_atom_index)],
            dtype=float,
        )
    elif center_mode == "reference_element":
        active_center = np.asarray(
            reference_element_geometric_center,
            dtype=float,
        )
    else:
        raise ValueError(f"Unsupported center mode: {center_mode!r}")
    centered_coordinates, rmax = _centered_coordinates_from_origin(
        structure.coordinates,
        active_center,
    )
    return replace(
        structure,
        centered_coordinates=np.asarray(centered_coordinates, dtype=float),
        geometric_center=np.asarray(geometric_center, dtype=float),
        reference_element=str(resolved_reference_element),
        reference_element_geometric_center=np.asarray(
            reference_element_geometric_center,
            dtype=float,
        ),
        reference_element_offset_from_geometric_center=float(
            reference_element_offset
        ),
        active_center=np.asarray(active_center, dtype=float),
        center_mode=str(center_mode),
        rmax=float(rmax),
    )


def build_electron_density_mesh(
    structure: ElectronDensityStructure,
    settings: ElectronDensityMeshSettings,
) -> ElectronDensityMeshGeometry:
    normalized = settings.normalized()
    del structure
    domain_max_radius = float(max(normalized.rmax, normalized.rstep))
    radial_edges = np.arange(
        0.0,
        domain_max_radius + normalized.rstep,
        normalized.rstep,
        dtype=float,
    )
    if radial_edges.size < 2 or radial_edges[-1] < domain_max_radius:
        radial_edges = np.append(radial_edges, domain_max_radius)
    else:
        radial_edges[-1] = domain_max_radius
    theta_edges = np.linspace(
        0.0,
        float(np.pi),
        normalized.theta_divisions + 1,
        dtype=float,
    )
    phi_edges = np.linspace(
        0.0,
        float(2.0 * np.pi),
        normalized.phi_divisions + 1,
        dtype=float,
    )
    return ElectronDensityMeshGeometry(
        settings=normalized,
        domain_max_radius=domain_max_radius,
        radial_edges=radial_edges,
        theta_edges=theta_edges,
        phi_edges=phi_edges,
    )


def _member_summary_from_profile_result(
    result: ElectronDensityProfileResult,
) -> ElectronDensityMemberSummary:
    structure = result.structure
    return ElectronDensityMemberSummary(
        file_path=structure.file_path,
        display_label=structure.display_label,
        center_of_mass=np.asarray(structure.center_of_mass, dtype=float),
        geometric_center=np.asarray(structure.geometric_center, dtype=float),
        reference_element=str(structure.reference_element),
        reference_element_geometric_center=np.asarray(
            structure.reference_element_geometric_center,
            dtype=float,
        ),
        reference_element_offset_from_geometric_center=float(
            structure.reference_element_offset_from_geometric_center
        ),
        active_center=np.asarray(structure.active_center, dtype=float),
        center_mode=str(structure.center_mode),
        nearest_atom_index=int(structure.nearest_atom_index),
        nearest_atom_element=str(structure.nearest_atom_element),
        nearest_atom_distance=float(structure.nearest_atom_distance),
        rmax=float(structure.rmax),
        total_electrons=float(np.sum(result.shell_electron_counts)),
        excluded_atom_count=int(result.excluded_atom_count),
        excluded_electron_count=float(result.excluded_electron_count),
    )


def _compute_smeared_profile_statistics(
    radial_centers: np.ndarray,
    member_profiles: tuple[np.ndarray, ...],
    *,
    smearing_settings: ElectronDensitySmearingSettings | None = None,
) -> tuple[
    ElectronDensitySmearingSettings,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    active_smearing = (
        ElectronDensitySmearingSettings()
        if smearing_settings is None
        else smearing_settings
    ).normalized()
    profile_members = tuple(
        np.asarray(profile, dtype=float).copy() for profile in member_profiles
    )
    if not profile_members:
        raise ValueError(
            "At least one member profile is required for smearing."
        )
    smeared_stack = np.vstack(
        [
            _gaussian_smooth_profile(
                radial_centers,
                profile,
                sigma_a=active_smearing.gaussian_sigma_a,
            )
            for profile in profile_members
        ]
    )
    smeared_mean = np.mean(smeared_stack, axis=0)
    if len(profile_members) > 1:
        smeared_variance = np.var(smeared_stack, axis=0)
    else:
        smeared_variance = np.zeros_like(smeared_mean, dtype=float)
    return (
        active_smearing,
        np.asarray(smeared_mean, dtype=float),
        np.asarray(smeared_variance, dtype=float),
        np.sqrt(np.asarray(smeared_variance, dtype=float)),
    )


def apply_smearing_to_profile_result(
    result: ElectronDensityProfileResult,
    smearing_settings: ElectronDensitySmearingSettings | None = None,
) -> ElectronDensityProfileResult:
    member_profiles = result.member_orientation_average_densities or (
        np.asarray(result.orientation_average_density, dtype=float).copy(),
    )
    (
        active_smearing,
        smeared_orientation_average_density,
        smeared_orientation_density_variance,
        smeared_orientation_density_stddev,
    ) = _compute_smeared_profile_statistics(
        np.asarray(result.radial_centers, dtype=float),
        member_profiles,
        smearing_settings=smearing_settings,
    )
    updated_result = replace(
        result,
        smearing_settings=active_smearing,
        smeared_orientation_average_density=smeared_orientation_average_density,
        smeared_orientation_density_variance=smeared_orientation_density_variance,
        smeared_orientation_density_stddev=smeared_orientation_density_stddev,
    )
    if result.solvent_contrast is None:
        return updated_result
    return replace(
        updated_result,
        solvent_contrast=_build_solvent_contrast_result(
            updated_result,
            settings=result.solvent_contrast.settings,
            solvent_name=result.solvent_contrast.solvent_name,
            density_estimate=result.solvent_contrast.density_estimate,
        ),
    )


def _transform_window_values(
    radial_values: np.ndarray,
    *,
    window_function: str,
    r_min: float,
    r_max: float,
) -> np.ndarray:
    radii = np.asarray(radial_values, dtype=float)
    normalized_window = _normalized_transform_window_name(window_function)
    if normalized_window == "none":
        return np.ones_like(radii, dtype=float)
    span = max(float(r_max) - float(r_min), 1.0e-12)
    position = np.clip((radii - float(r_min)) / span, 0.0, 1.0)
    midpoint = float(r_min) + span * 0.5
    normalized_centered = np.clip(
        (radii - midpoint) / max(span * 0.5, 1.0e-12),
        -1.0,
        1.0,
    )
    absolute_centered = np.abs(normalized_centered)
    if normalized_window == "lorch":
        return np.asarray(np.sinc(position), dtype=float)
    if normalized_window == "cosine":
        return np.asarray(
            np.cos(0.5 * np.pi * normalized_centered),
            dtype=float,
        )
    if normalized_window == "sine":
        return np.asarray(np.sin(np.pi * position), dtype=float)
    if normalized_window == "hanning":
        return np.asarray(0.5 * (1.0 + np.cos(np.pi * normalized_centered)))
    if normalized_window == "welch":
        return np.asarray(1.0 - np.square(normalized_centered), dtype=float)
    if normalized_window == "gaussian":
        sigma = 0.40
        return np.asarray(
            np.exp(-0.5 * np.square(normalized_centered / sigma)),
            dtype=float,
        )
    if normalized_window == "parzen":
        values = np.zeros_like(radii, dtype=float)
        inner_mask = absolute_centered <= 0.5
        outer_mask = (absolute_centered > 0.5) & (absolute_centered <= 1.0)
        values[inner_mask] = (
            1.0
            - 6.0 * np.square(absolute_centered[inner_mask])
            + 6.0 * np.power(absolute_centered[inner_mask], 3)
        )
        values[outer_mask] = 2.0 * np.power(
            1.0 - absolute_centered[outer_mask],
            3,
        )
        return np.asarray(values, dtype=float)
    if normalized_window == "kaiser_bessel":
        beta = 6.0
        argument = beta * np.sqrt(
            np.clip(1.0 - np.square(normalized_centered), 0.0, 1.0)
        )
        return np.asarray(np.i0(argument) / np.i0(beta), dtype=float)
    raise ValueError(
        f"Unsupported Fourier-transform window: {window_function!r}"
    )


def prepare_electron_density_fourier_transform(
    result: ElectronDensityProfileResult,
    settings: ElectronDensityFourierTransformSettings,
) -> ElectronDensityFourierTransformPreview:
    normalized = settings.normalized()
    source_profile_label = "Smeared ρ(r)"
    source_density_values = np.asarray(
        result.smeared_orientation_average_density,
        dtype=float,
    )
    source_radial_values = np.asarray(result.radial_centers, dtype=float)
    if source_radial_values.size < 2:
        raise ValueError(
            "At least two radial density samples are required for the Fourier transform."
        )
    available_r_min = float(source_radial_values[0])
    available_r_max = float(source_radial_values[-1])
    notes: list[str] = []
    if normalized.use_solvent_subtracted_profile:
        if result.solvent_contrast is None:
            notes.append(
                "Requested the solvent-subtracted density profile for the Fourier transform, "
                "but no solvent electron density has been computed yet. Using the smeared profile instead."
            )
        else:
            source_profile_label = "Solvent-subtracted smeared ρ(r)"
            source_density_values = np.asarray(
                result.solvent_contrast.solvent_subtracted_smeared_density,
                dtype=float,
            )
    source_radial_values, source_density_values = _extend_profile_to_origin(
        source_radial_values,
        source_density_values,
    )
    available_r_min = 0.0
    available_r_max = float(source_radial_values[-1])
    effective_r_min = max(float(normalized.r_min), available_r_min)
    effective_r_max = min(float(normalized.r_max), available_r_max)
    if effective_r_min > float(normalized.r_min) + 1.0e-12:
        notes.append(
            f"Raised transform r min to the available profile limit at {effective_r_min:.4f} Å."
        )
    if effective_r_max < float(normalized.r_max) - 1.0e-12:
        notes.append(
            f"Lowered transform r max to the available profile limit at {effective_r_max:.4f} Å."
        )
    if effective_r_max <= effective_r_min:
        raise ValueError(
            "The selected Fourier-transform bounds do not overlap the available density profile."
        )

    effective_settings = replace(
        normalized,
        r_min=effective_r_min,
        r_max=effective_r_max,
    )
    resampled_r_values = np.linspace(
        effective_r_min,
        effective_r_max,
        effective_settings.resampling_points,
        dtype=float,
    )
    resampled_density_values = np.interp(
        resampled_r_values,
        source_radial_values,
        source_density_values,
    )
    window_values = _transform_window_values(
        resampled_r_values,
        window_function=effective_settings.window_function,
        r_min=effective_r_min,
        r_max=effective_r_max,
    )
    windowed_density_values = np.asarray(
        resampled_density_values, dtype=float
    ) * np.asarray(window_values, dtype=float)
    resampling_step_a = float(resampled_r_values[1] - resampled_r_values[0])
    nyquist_q_max_a_inverse = float(np.pi / max(resampling_step_a, 1.0e-12))
    independent_q_step_a_inverse = float(
        np.pi / max(effective_r_max - effective_r_min, 1.0e-12)
    )
    effective_q_max = min(
        float(effective_settings.q_max),
        nyquist_q_max_a_inverse,
    )
    q_max_was_clamped = bool(
        effective_q_max < float(effective_settings.q_max) - 1.0e-12
    )
    if q_max_was_clamped:
        notes.append(
            f"Clamped q max to the Nyquist limit at {effective_q_max:.4f} Å⁻¹."
        )
    if effective_q_max <= float(effective_settings.q_min):
        raise ValueError(
            "The requested q-space range exceeds the Nyquist limit for the selected transform sampling."
        )
    q_grid_is_oversampled = bool(
        float(effective_settings.q_step)
        < independent_q_step_a_inverse - 1.0e-12
    )
    if q_grid_is_oversampled:
        notes.append(
            "Requested q-step is finer than the independent Nyquist spacing; "
            "the plotted transform will be oversampled but not contain more independent information."
        )

    return ElectronDensityFourierTransformPreview(
        settings=replace(
            effective_settings,
            q_max=effective_q_max,
        ),
        source_profile_label=source_profile_label,
        source_radial_values=np.asarray(source_radial_values, dtype=float),
        source_density_values=np.asarray(source_density_values, dtype=float),
        resampled_r_values=np.asarray(resampled_r_values, dtype=float),
        resampled_density_values=np.asarray(
            resampled_density_values, dtype=float
        ),
        window_values=np.asarray(window_values, dtype=float),
        windowed_density_values=np.asarray(
            windowed_density_values, dtype=float
        ),
        available_r_min=available_r_min,
        available_r_max=available_r_max,
        resampling_step_a=resampling_step_a,
        nyquist_q_max_a_inverse=nyquist_q_max_a_inverse,
        independent_q_step_a_inverse=independent_q_step_a_inverse,
        q_grid_is_oversampled=q_grid_is_oversampled,
        q_max_was_clamped=q_max_was_clamped,
        notes=tuple(notes),
    )


def compute_electron_density_scattering_profile(
    result: ElectronDensityProfileResult,
    settings: ElectronDensityFourierTransformSettings,
) -> ElectronDensityScatteringTransformResult:
    preview = prepare_electron_density_fourier_transform(result, settings)
    effective_settings = preview.settings
    q_values = np.arange(
        float(effective_settings.q_min),
        float(effective_settings.q_max)
        + float(effective_settings.q_step) * 0.5,
        float(effective_settings.q_step),
        dtype=float,
    )
    radial_values = np.asarray(preview.resampled_r_values, dtype=float)
    windowed_density = np.asarray(preview.windowed_density_values, dtype=float)
    radial_square = np.square(radial_values)
    amplitude = np.zeros_like(q_values, dtype=float)
    chunk_size = 256
    for start_index in range(0, q_values.size, chunk_size):
        stop_index = min(start_index + chunk_size, q_values.size)
        q_chunk = q_values[start_index:stop_index]
        kernel = np.sinc(
            (q_chunk[:, np.newaxis] * radial_values[np.newaxis, :]) / np.pi
        )
        integrand = (
            windowed_density[np.newaxis, :]
            * radial_square[np.newaxis, :]
            * kernel
        )
        amplitude[start_index:stop_index] = (
            4.0
            * np.pi
            * _trapezoid_integral(
                integrand,
                x=radial_values,
                axis=1,
            )
        )
    intensity = np.square(np.abs(amplitude))
    return ElectronDensityScatteringTransformResult(
        preview=preview,
        q_values=np.asarray(q_values, dtype=float),
        scattering_amplitude=np.asarray(amplitude, dtype=float),
        intensity=np.asarray(intensity, dtype=float),
    )


def compute_electron_density_profile(
    structure: ElectronDensityStructure,
    mesh_settings: ElectronDensityMeshSettings,
    *,
    smearing_settings: ElectronDensitySmearingSettings | None = None,
) -> ElectronDensityProfileResult:
    mesh_geometry = build_electron_density_mesh(structure, mesh_settings)
    centered_coordinates = np.asarray(
        structure.centered_coordinates, dtype=float
    )
    radial_distances = np.linalg.norm(centered_coordinates, axis=1)

    theta_values = np.zeros_like(radial_distances, dtype=float)
    nonzero_mask = radial_distances > 0.0
    theta_values[nonzero_mask] = np.arccos(
        np.clip(
            centered_coordinates[nonzero_mask, 2]
            / radial_distances[nonzero_mask],
            -1.0,
            1.0,
        )
    )
    phi_values = np.mod(
        np.arctan2(centered_coordinates[:, 1], centered_coordinates[:, 0]),
        2.0 * np.pi,
    )

    radial_edges = np.asarray(mesh_geometry.radial_edges, dtype=float)
    theta_edges = np.asarray(mesh_geometry.theta_edges, dtype=float)
    phi_edges = np.asarray(mesh_geometry.phi_edges, dtype=float)
    shell_count = int(len(radial_edges) - 1)
    theta_count = int(len(theta_edges) - 1)
    phi_count = int(len(phi_edges) - 1)
    domain_max_radius = float(mesh_geometry.domain_max_radius)

    in_domain_mask = radial_distances <= (domain_max_radius + 1.0e-12)
    excluded_atom_count = int(np.count_nonzero(~in_domain_mask))
    excluded_electron_count = float(
        np.sum(
            np.asarray(structure.atomic_numbers, dtype=float)[~in_domain_mask]
        )
    )
    radial_distances = radial_distances[in_domain_mask]
    theta_values = theta_values[in_domain_mask]
    phi_values = phi_values[in_domain_mask]
    atomic_numbers = np.asarray(structure.atomic_numbers, dtype=float)[
        in_domain_mask
    ]

    radial_indices = (
        np.searchsorted(radial_edges, radial_distances, side="right") - 1
    )
    radial_indices = np.clip(radial_indices, 0, shell_count - 1)
    theta_indices = (
        np.searchsorted(theta_edges, theta_values, side="right") - 1
    )
    theta_indices = np.clip(theta_indices, 0, theta_count - 1)
    phi_indices = np.searchsorted(phi_edges, phi_values, side="right") - 1
    phi_indices = np.clip(phi_indices, 0, phi_count - 1)

    flat_indices = (
        radial_indices * theta_count * phi_count
        + theta_indices * phi_count
        + phi_indices
    )
    cell_electron_counts = np.zeros(
        shell_count * theta_count * phi_count,
        dtype=float,
    )
    np.add.at(
        cell_electron_counts,
        flat_indices,
        atomic_numbers,
    )
    cell_electron_counts = cell_electron_counts.reshape(
        shell_count,
        theta_count,
        phi_count,
    )

    radial_shell_factors = (
        np.power(radial_edges[1:], 3.0) - np.power(radial_edges[:-1], 3.0)
    ) / 3.0
    theta_factors = (np.cos(theta_edges[:-1]) - np.cos(theta_edges[1:]))[
        :, np.newaxis
    ]
    phi_widths = np.diff(phi_edges)[np.newaxis, :]
    angular_factors = theta_factors * phi_widths
    cell_volumes = (
        radial_shell_factors[:, np.newaxis, np.newaxis]
        * angular_factors[np.newaxis, :, :]
    )
    cell_densities = np.divide(
        cell_electron_counts,
        cell_volumes,
        out=np.zeros_like(cell_electron_counts, dtype=float),
        where=cell_volumes > 0.0,
    )

    shell_electron_counts = np.sum(cell_electron_counts, axis=(1, 2))
    shell_volumes = np.sum(cell_volumes, axis=(1, 2))
    shell_volume_average_density = np.divide(
        shell_electron_counts,
        shell_volumes,
        out=np.zeros_like(shell_electron_counts, dtype=float),
        where=shell_volumes > 0.0,
    )
    angular_weight_sum = float(np.sum(angular_factors))
    solid_angle_weights = np.divide(
        angular_factors,
        angular_weight_sum,
        out=np.zeros_like(angular_factors, dtype=float),
        where=angular_weight_sum > 0.0,
    )
    orientation_average_density = np.sum(
        cell_densities * solid_angle_weights[np.newaxis, :, :],
        axis=(1, 2),
    )
    radial_centers = (radial_edges[:-1] + radial_edges[1:]) * 0.5
    active_smearing = (
        ElectronDensitySmearingSettings()
        if smearing_settings is None
        else smearing_settings
    ).normalized()
    orientation_density_variance = np.zeros_like(
        orientation_average_density,
        dtype=float,
    )
    orientation_density_stddev = np.zeros_like(
        orientation_average_density,
        dtype=float,
    )
    member_orientation_average_densities = (
        np.asarray(orientation_average_density, dtype=float).copy(),
    )
    (
        active_smearing,
        smeared_orientation_average_density,
        smeared_orientation_density_variance,
        smeared_orientation_density_stddev,
    ) = _compute_smeared_profile_statistics(
        radial_centers,
        member_orientation_average_densities,
        smearing_settings=active_smearing,
    )
    result = ElectronDensityProfileResult(
        structure=structure,
        input_mode="file",
        source_files=(structure.file_path,),
        source_structure_count=1,
        member_summaries=tuple(),
        member_orientation_average_densities=member_orientation_average_densities,
        mesh_geometry=mesh_geometry,
        smearing_settings=active_smearing,
        radial_centers=radial_centers,
        orientation_average_density=orientation_average_density,
        orientation_density_variance=orientation_density_variance,
        orientation_density_stddev=orientation_density_stddev,
        smeared_orientation_average_density=smeared_orientation_average_density,
        smeared_orientation_density_variance=smeared_orientation_density_variance,
        smeared_orientation_density_stddev=smeared_orientation_density_stddev,
        shell_volume_average_density=shell_volume_average_density,
        shell_electron_counts=shell_electron_counts,
        shell_volumes=shell_volumes,
        excluded_atom_count=excluded_atom_count,
        excluded_electron_count=excluded_electron_count,
    )
    return replace(
        result,
        member_summaries=(_member_summary_from_profile_result(result),),
    )


def compute_electron_density_profile_for_input(
    selection: ElectronDensityInputInspection | str | Path,
    mesh_settings: ElectronDensityMeshSettings,
    *,
    smearing_settings: ElectronDensitySmearingSettings | None = None,
    center_mode: str = "center_of_mass",
    reference_element: str | None = None,
    progress_callback: ElectronDensityProgressCallback | None = None,
) -> ElectronDensityProfileResult:
    inspection = (
        selection
        if isinstance(selection, ElectronDensityInputInspection)
        else inspect_structure_input(selection)
    )
    structure_files = tuple(inspection.structure_files)
    if not structure_files:
        raise ValueError("No structure files were available for calculation.")

    total_steps = len(structure_files) * 2 + 2
    step_index = 0
    if inspection.input_mode == "folder":
        _emit_progress(
            progress_callback,
            step_index,
            total_steps,
            f"Preparing ensemble electron-density calculation for {len(structure_files)} structures.",
        )
    else:
        _emit_progress(
            progress_callback,
            step_index,
            total_steps,
            "Preparing electron-density calculation.",
        )

    member_results: list[ElectronDensityProfileResult] = []
    for file_index, file_path in enumerate(structure_files, start=1):
        step_index += 1
        _emit_progress(
            progress_callback,
            step_index,
            total_steps,
            f"Loading structure {file_index}/{len(structure_files)}: {file_path.name}",
        )
        structure = load_electron_density_structure(
            file_path,
            center_mode=center_mode,
            reference_element=reference_element,
        )
        step_index += 1
        _emit_progress(
            progress_callback,
            step_index,
            total_steps,
            f"Computing electron density {file_index}/{len(structure_files)}: {file_path.name}",
        )
        member_results.append(
            compute_electron_density_profile(
                structure,
                mesh_settings,
                smearing_settings=smearing_settings,
            )
        )

    step_index += 1
    _emit_progress(
        progress_callback,
        step_index,
        total_steps,
        "Averaging electron density and variance across structures.",
    )

    reference_result = member_results[0]
    orientation_stack = np.vstack(
        [result.orientation_average_density for result in member_results]
    )
    shell_volume_density_stack = np.vstack(
        [result.shell_volume_average_density for result in member_results]
    )
    shell_electron_stack = np.vstack(
        [result.shell_electron_counts for result in member_results]
    )
    shell_volume_stack = np.vstack(
        [result.shell_volumes for result in member_results]
    )
    orientation_density_variance = np.var(orientation_stack, axis=0)
    member_orientation_average_densities = tuple(
        np.asarray(result.orientation_average_density, dtype=float).copy()
        for result in member_results
    )
    (
        active_smearing,
        smeared_orientation_average_density,
        smeared_orientation_density_variance,
        smeared_orientation_density_stddev,
    ) = _compute_smeared_profile_statistics(
        np.asarray(reference_result.radial_centers, dtype=float),
        member_orientation_average_densities,
        smearing_settings=smearing_settings,
    )
    aggregated_result = replace(
        reference_result,
        input_mode=str(inspection.input_mode),
        source_files=tuple(
            result.structure.file_path for result in member_results
        ),
        source_structure_count=len(member_results),
        member_summaries=tuple(
            _member_summary_from_profile_result(result)
            for result in member_results
        ),
        member_orientation_average_densities=member_orientation_average_densities,
        smearing_settings=active_smearing,
        orientation_average_density=np.mean(orientation_stack, axis=0),
        orientation_density_variance=orientation_density_variance,
        orientation_density_stddev=np.sqrt(orientation_density_variance),
        smeared_orientation_average_density=smeared_orientation_average_density,
        smeared_orientation_density_variance=smeared_orientation_density_variance,
        smeared_orientation_density_stddev=smeared_orientation_density_stddev,
        shell_volume_average_density=np.mean(
            shell_volume_density_stack,
            axis=0,
        ),
        shell_electron_counts=np.mean(shell_electron_stack, axis=0),
        shell_volumes=np.mean(shell_volume_stack, axis=0),
        excluded_atom_count=int(
            sum(result.excluded_atom_count for result in member_results)
        ),
        excluded_electron_count=float(
            sum(result.excluded_electron_count for result in member_results)
        ),
    )
    step_index += 1
    _emit_progress(
        progress_callback,
        step_index,
        total_steps,
        "Electron-density profile ready.",
    )
    return aggregated_result


def suggest_output_dir(
    selection_path: str | Path,
    *,
    project_dir: str | Path | None = None,
) -> Path:
    if project_dir is not None:
        return (
            Path(project_dir).expanduser().resolve()
            / "electron_density_mapping"
        )
    resolved = Path(selection_path).expanduser().resolve()
    base_dir = resolved if resolved.is_dir() else resolved.parent
    return base_dir / "electron_density_mapping"


def suggest_output_basename(
    inspection: ElectronDensityInputInspection,
) -> str:
    stem = inspection.reference_file.stem
    if inspection.input_mode == "folder":
        return f"{stem}_folder_average_density"
    return f"{stem}_electron_density"


def write_electron_density_profile_outputs(
    result: ElectronDensityProfileResult,
    output_dir: str | Path,
    basename: str,
) -> ElectronDensityOutputArtifacts:
    resolved_output_dir = Path(output_dir).expanduser().resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    safe_basename = re.sub(r"[^A-Za-z0-9._-]+", "_", str(basename).strip())
    safe_basename = safe_basename.strip("._") or "electron_density_profile"

    csv_path = resolved_output_dir / f"{safe_basename}.csv"
    json_path = resolved_output_dir / f"{safe_basename}.json"

    radial_edges = np.asarray(result.mesh_geometry.radial_edges, dtype=float)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "r_inner_a",
                "r_outer_a",
                "r_center_a",
                "orientation_average_density_e_per_a3",
                "orientation_density_variance_e2_per_a6",
                "orientation_density_stddev_e_per_a3",
                "smeared_orientation_average_density_e_per_a3",
                "smeared_orientation_density_variance_e2_per_a6",
                "smeared_orientation_density_stddev_e_per_a3",
                "shell_volume_average_density_e_per_a3",
                "electrons_in_shell",
                "shell_volume_a3",
            ]
        )
        for index, center in enumerate(result.radial_centers):
            writer.writerow(
                [
                    f"{float(radial_edges[index]):.8f}",
                    f"{float(radial_edges[index + 1]):.8f}",
                    f"{float(center):.8f}",
                    f"{float(result.orientation_average_density[index]):.8f}",
                    f"{float(result.orientation_density_variance[index]):.8f}",
                    f"{float(result.orientation_density_stddev[index]):.8f}",
                    f"{float(result.smeared_orientation_average_density[index]):.8f}",
                    f"{float(result.smeared_orientation_density_variance[index]):.8f}",
                    f"{float(result.smeared_orientation_density_stddev[index]):.8f}",
                    f"{float(result.shell_volume_average_density[index]):.8f}",
                    f"{float(result.shell_electron_counts[index]):.8f}",
                    f"{float(result.shell_volumes[index]):.8f}",
                ]
            )

    summary_payload = {
        "input_mode": str(result.input_mode),
        "source_file": str(result.structure.file_path),
        "source_files": [str(path) for path in result.source_files],
        "source_structure_count": int(result.source_structure_count),
        "display_label": result.structure.display_label,
        "structure_comment": result.structure.structure_comment,
        "atom_count": int(result.structure.atom_count),
        "element_counts": dict(result.structure.element_counts),
        "center_of_mass_a": [
            float(value) for value in result.structure.center_of_mass
        ],
        "geometric_center_a": [
            float(value) for value in result.structure.geometric_center
        ],
        "reference_element": str(result.structure.reference_element),
        "reference_element_geometric_center_a": [
            float(value)
            for value in result.structure.reference_element_geometric_center
        ],
        "reference_element_offset_from_geometric_center_a": float(
            result.structure.reference_element_offset_from_geometric_center
        ),
        "active_center_a": [
            float(value) for value in result.structure.active_center
        ],
        "center_mode": str(result.structure.center_mode),
        "nearest_atom_index": int(result.structure.nearest_atom_index),
        "nearest_atom_element": str(result.structure.nearest_atom_element),
        "nearest_atom_distance_to_center_of_mass_a": float(
            result.structure.nearest_atom_distance
        ),
        "structure_rmax_a": float(result.structure.rmax),
        "domain_max_radius_a": float(result.mesh_geometry.domain_max_radius),
        "mesh_settings": result.mesh_geometry.settings.to_dict(),
        "smearing_settings": result.smearing_settings.to_dict(),
        "shell_count": int(result.mesh_geometry.shell_count),
        "excluded_atom_count": int(result.excluded_atom_count),
        "excluded_electron_count": float(result.excluded_electron_count),
        "member_summaries": [
            {
                "file_path": str(entry.file_path),
                "display_label": str(entry.display_label),
                "center_of_mass_a": [
                    float(value) for value in entry.center_of_mass
                ],
                "geometric_center_a": [
                    float(value) for value in entry.geometric_center
                ],
                "reference_element": str(entry.reference_element),
                "reference_element_geometric_center_a": [
                    float(value)
                    for value in entry.reference_element_geometric_center
                ],
                "reference_element_offset_from_geometric_center_a": float(
                    entry.reference_element_offset_from_geometric_center
                ),
                "active_center_a": [
                    float(value) for value in entry.active_center
                ],
                "center_mode": str(entry.center_mode),
                "nearest_atom_index": int(entry.nearest_atom_index),
                "nearest_atom_element": str(entry.nearest_atom_element),
                "nearest_atom_distance_to_center_of_mass_a": float(
                    entry.nearest_atom_distance
                ),
                "rmax_a": float(entry.rmax),
                "total_electrons": float(entry.total_electrons),
                "excluded_atom_count": int(entry.excluded_atom_count),
                "excluded_electron_count": float(
                    entry.excluded_electron_count
                ),
            }
            for entry in result.member_summaries
        ],
        "profile_csv_path": str(csv_path),
        "radial_profile": [
            {
                "r_inner_a": float(radial_edges[index]),
                "r_outer_a": float(radial_edges[index + 1]),
                "r_center_a": float(result.radial_centers[index]),
                "orientation_average_density_e_per_a3": float(
                    result.orientation_average_density[index]
                ),
                "orientation_density_variance_e2_per_a6": float(
                    result.orientation_density_variance[index]
                ),
                "orientation_density_stddev_e_per_a3": float(
                    result.orientation_density_stddev[index]
                ),
                "smeared_orientation_average_density_e_per_a3": float(
                    result.smeared_orientation_average_density[index]
                ),
                "smeared_orientation_density_variance_e2_per_a6": float(
                    result.smeared_orientation_density_variance[index]
                ),
                "smeared_orientation_density_stddev_e_per_a3": float(
                    result.smeared_orientation_density_stddev[index]
                ),
                "shell_volume_average_density_e_per_a3": float(
                    result.shell_volume_average_density[index]
                ),
                "electrons_in_shell": float(
                    result.shell_electron_counts[index]
                ),
                "shell_volume_a3": float(result.shell_volumes[index]),
            }
            for index in range(len(result.radial_centers))
        ],
    }
    json_path.write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return ElectronDensityOutputArtifacts(
        csv_path=csv_path,
        json_path=json_path,
    )


__all__ = [
    "ElectronDensityFourierTransformPreview",
    "ElectronDensityFourierTransformSettings",
    "ElectronDensityInputInspection",
    "ElectronDensityMemberSummary",
    "ElectronDensityMeshGeometry",
    "ElectronDensityMeshSettings",
    "ElectronDensityOutputArtifacts",
    "ElectronDensityProfileResult",
    "ElectronDensityScatteringTransformResult",
    "ElectronDensitySmearingSettings",
    "ElectronDensitySolventContrastResult",
    "ElectronDensityStructure",
    "apply_solvent_contrast_to_profile_result",
    "apply_smearing_to_profile_result",
    "build_electron_density_mesh",
    "compute_electron_density_profile",
    "compute_electron_density_profile_for_input",
    "compute_electron_density_scattering_profile",
    "inspect_structure_input",
    "load_electron_density_structure",
    "prepare_electron_density_fourier_transform",
    "recenter_electron_density_structure",
    "suggest_output_basename",
    "suggest_output_dir",
    "write_electron_density_profile_outputs",
]
