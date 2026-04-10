from __future__ import annotations

import csv
import json
import os
import re
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
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
from saxshell.saxs.debye import (
    atomic_form_factor,
    compute_debye_intensity,
    load_structure_file,
)
from saxshell.saxs.debye.profiles import build_f0_dictionary
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
_SUPPORTED_TRANSFORM_DOMAIN_MODES = {
    "legacy",
    "mirrored",
}
_FRAME_ID_PATTERN = re.compile(r"frame_(\d+)", re.IGNORECASE)
_AVOGADRO_NUMBER = 6.02214076e23
_ANGSTROM3_PER_CM3 = 1.0e24
_DEBYE_WALLER_MODE_UNIVERSAL = "universal"
_DEBYE_WALLER_MODE_PAIR_SPECIFIC = "pair_specific"
_SUPPORTED_DEBYE_WALLER_MODES = {
    _DEBYE_WALLER_MODE_UNIVERSAL,
    _DEBYE_WALLER_MODE_PAIR_SPECIFIC,
}
ElectronDensityProgressCallback = Callable[[int, int, str], None]
ElectronDensityCancelCallback = Callable[[], bool]


class ElectronDensityCalculationCanceled(RuntimeError):
    """Raised when a UI-driven electron-density calculation is
    stopped."""


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


def _progress_center_mode_label(
    center_mode: str,
    *,
    reference_element: str | None = None,
) -> str:
    normalized_mode = str(center_mode or "").strip().lower()
    if normalized_mode == "nearest_atom":
        return "nearest-atom centering"
    if normalized_mode == "reference_element":
        element_text = str(reference_element or "").strip() or "auto"
        return f"reference-element centering ({element_text})"
    return "geometric-mass-center centering"


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


@lru_cache(maxsize=None)
def _waasmaier_parameters(
    symbol: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    _require_xraydb()
    normalized = _normalized_element_symbol(symbol)
    if not normalized:
        raise ValueError("Encountered an empty element symbol.")
    try:
        rows = [
            row
            for row in xraydb.get_xraydb().get_cache("Waasmaier")
            if normalized == str(row.ion)
        ]
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(
            "Could not load Waasmaier-Kirfel form-factor parameters for "
            f"{normalized!r}."
        ) from exc
    if not rows:
        raise ValueError(
            "No Waasmaier-Kirfel form-factor parameters were available for "
            f"{normalized!r}."
        )
    row = rows[0]
    scales = np.asarray(json.loads(row.scale), dtype=float)
    exponents = np.asarray(json.loads(row.exponents), dtype=float)
    offset = float(row.offset)
    valid_mask = (
        np.isfinite(scales)
        & np.isfinite(exponents)
        & (scales > 0.0)
        & (exponents > 0.0)
    )
    filtered_scales = np.asarray(scales[valid_mask], dtype=float)
    filtered_exponents = np.asarray(exponents[valid_mask], dtype=float)
    if filtered_scales.size == 0 or filtered_exponents.size == 0:
        raise ValueError(
            "Waasmaier-Kirfel form-factor parameters for "
            f"{normalized!r} did not include any usable Gaussian terms."
        )
    return filtered_scales, filtered_exponents, offset


@lru_cache(maxsize=None)
def _effective_atomic_overlap_radius(symbol: str) -> float:
    scales, exponents, offset = _waasmaier_parameters(symbol)
    electron_count = max(float(_atomic_number(symbol)), 1.0)
    low_q_second_moment = float(
        np.sum(scales * exponents) * (3.0 / (8.0 * np.pi**2 * electron_count))
    )
    sphere_radius = np.sqrt(max((5.0 / 3.0) * low_q_second_moment, 0.0))
    if sphere_radius > 0.0:
        return float(sphere_radius)
    gaussian_electron_count = float(np.sum(scales))
    if gaussian_electron_count > 0.0:
        fallback_radius = np.cbrt(
            max(gaussian_electron_count + max(offset, 0.0), 1.0e-12)
        ) / max(np.pi, 1.0)
        return float(max(fallback_radius, 1.0e-3))
    return 1.0e-3


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


def heaviest_reference_element(elements: tuple[str, ...]) -> str | None:
    """Return the heaviest element by atomic mass from *elements*, or
    None."""
    try:
        return _default_reference_element(elements)
    except ValueError:
        return None


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
        # Requested element absent in this structure — fall back to the heaviest
        # element available rather than raising, so per-cluster reference-element
        # mismatches are handled gracefully.
        resolved_reference_element = _default_reference_element(elements)
        element_mask = np.asarray(
            [element == resolved_reference_element for element in elements],
            dtype=bool,
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
    kernel = _gaussian_smoothing_kernel(
        radial_centers,
        sigma_a=sigma_a,
    )
    profile = np.asarray(values, dtype=float)
    if profile.size <= 1 or kernel.size == 0:
        return profile.copy()
    return np.asarray(kernel @ profile, dtype=float)


def _gaussian_smoothing_kernel(
    radial_centers: np.ndarray,
    *,
    sigma_a: float,
) -> np.ndarray:
    centers = np.asarray(radial_centers, dtype=float)
    sigma_value = max(float(sigma_a), 0.0)
    if centers.size <= 1 or sigma_value <= 0.0:
        return np.eye(int(centers.size), dtype=float)
    distance_matrix = centers[:, np.newaxis] - centers[np.newaxis, :]
    kernel = np.exp(
        -0.5 * np.square(distance_matrix / max(sigma_value, 1.0e-12))
    )
    normalizers = np.sum(kernel, axis=1, keepdims=True)
    return np.divide(
        kernel,
        normalizers,
        out=np.zeros_like(kernel, dtype=float),
        where=normalizers > 0.0,
    )


def _legacy_exact_count_conserving_gaussian_transfer_kernel(
    radial_centers: np.ndarray,
    shell_volumes: np.ndarray,
    *,
    sigma_a: float,
) -> np.ndarray:
    centers = np.asarray(radial_centers, dtype=float)
    volumes = np.asarray(shell_volumes, dtype=float)
    sigma_value = max(float(sigma_a), 0.0)
    if centers.size <= 1 or sigma_value <= 0.0 or volumes.size <= 1:
        return np.eye(int(centers.size), dtype=float)
    total_volume = float(np.sum(volumes))
    if total_volume <= 0.0:
        return np.eye(int(centers.size), dtype=float)
    distance_matrix = centers[:, np.newaxis] - centers[np.newaxis, :]
    affinity = np.exp(
        -0.5 * np.square(distance_matrix / max(sigma_value, 1.0e-12))
    )
    # Balance the Gaussian affinity to the shell-volume measure so a flat
    # density remains flat, the integrated electron count is preserved, and
    # each smeared shell stays a convex combination of the raw densities.
    target = volumes / total_volume
    left_scale = np.ones_like(target, dtype=float)
    right_scale = np.ones_like(target, dtype=float)
    for _ in range(512):
        affinity_right = affinity @ right_scale
        next_left = np.divide(
            target,
            affinity_right,
            out=np.zeros_like(left_scale, dtype=float),
            where=affinity_right > 0.0,
        )
        affinity_left = affinity.T @ next_left
        next_right = np.divide(
            target,
            affinity_left,
            out=np.zeros_like(right_scale, dtype=float),
            where=affinity_left > 0.0,
        )
        delta = max(
            float(np.max(np.abs(next_left - left_scale))),
            float(np.max(np.abs(next_right - right_scale))),
        )
        left_scale = next_left
        right_scale = next_right
        if delta <= 1.0e-12:
            break
    balanced_transport = (
        left_scale[:, np.newaxis] * affinity * right_scale[np.newaxis, :]
    )
    normalizers = np.sum(
        balanced_transport,
        axis=1,
        keepdims=True,
    )
    return np.divide(
        balanced_transport,
        normalizers,
        out=np.zeros_like(balanced_transport, dtype=float),
        where=normalizers > 0.0,
    )


def _count_conserving_gaussian_transfer_kernel(
    radial_centers: np.ndarray,
    shell_volumes: np.ndarray,
    *,
    sigma_a: float,
) -> np.ndarray:
    # Preserve the exact balanced transport kernel for legacy comparisons while
    # letting the active smearing path apply a non-amplifying clamp after the
    # matrix multiply.
    return _legacy_exact_count_conserving_gaussian_transfer_kernel(
        radial_centers,
        shell_volumes,
        sigma_a=sigma_a,
    )


def _cap_smeared_density_stack_to_input_totals(
    density_stack: np.ndarray,
    smeared_density_stack: np.ndarray,
    shell_volumes: np.ndarray,
) -> np.ndarray:
    raw_density_stack = np.asarray(density_stack, dtype=float)
    raw_smeared_density_stack = np.asarray(
        smeared_density_stack,
        dtype=float,
    )
    volumes = np.asarray(shell_volumes, dtype=float)
    if (
        raw_density_stack.ndim != 2
        or raw_smeared_density_stack.ndim != 2
        or raw_density_stack.shape != raw_smeared_density_stack.shape
        or volumes.ndim != 1
        or volumes.size != raw_density_stack.shape[1]
    ):
        raise ValueError(
            "Smearing totals can only be capped for aligned density/profile "
            "arrays with one shell-volume entry per shell."
        )
    input_totals = raw_density_stack @ volumes
    smeared_totals = raw_smeared_density_stack @ volumes
    scale_factors = np.ones_like(input_totals, dtype=float)
    overshoot_mask = smeared_totals > (input_totals + 1.0e-12)
    if np.any(overshoot_mask):
        scale_factors[overshoot_mask] = np.divide(
            input_totals[overshoot_mask],
            smeared_totals[overshoot_mask],
            out=np.zeros_like(input_totals[overshoot_mask], dtype=float),
            where=smeared_totals[overshoot_mask] > 0.0,
        )
    return np.asarray(
        raw_smeared_density_stack * scale_factors[:, np.newaxis],
        dtype=float,
    )


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


def _normalized_transform_domain_mode(value: object) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_")
    if normalized in {
        "mirror",
        "mirrored",
        "symmetric",
        "symmetric_about_zero",
        "reflected",
    }:
        return "mirrored"
    if normalized in {
        "",
        "legacy",
        "positive",
        "positive_only",
        "rmin_rmax",
    }:
        return "legacy"
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


def _mirror_profile_about_zero(
    radial_values: np.ndarray,
    density_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    radii = np.asarray(radial_values, dtype=float)
    densities = np.asarray(density_values, dtype=float)
    if radii.size == 0 or densities.size == 0:
        return radii.copy(), densities.copy()
    if radii.size != densities.size:
        raise ValueError(
            "Radial and density arrays must have matching lengths when mirroring the Fourier source profile."
        )
    if abs(float(radii[0])) > 1.0e-12:
        radii, densities = _extend_profile_to_origin(radii, densities)
    if radii.size <= 1:
        return radii.copy(), densities.copy()
    mirrored_radii = np.concatenate((-radii[:0:-1], radii))
    mirrored_densities = np.concatenate((densities[:0:-1], densities))
    return (
        np.asarray(mirrored_radii, dtype=float),
        np.asarray(mirrored_densities, dtype=float),
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


def _raise_if_canceled(
    cancel_callback: ElectronDensityCancelCallback | None,
) -> None:
    if cancel_callback is not None and bool(cancel_callback()):
        raise ElectronDensityCalculationCanceled(
            "Electron-density calculation was stopped by the user."
        )


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
    pin_contiguous_geometric_tracking: bool = True

    def normalized(self) -> "ElectronDensityMeshSettings":
        rstep = float(self.rstep)
        theta_divisions = int(self.theta_divisions)
        phi_divisions = int(self.phi_divisions)
        rmax = float(self.rmax)
        pin_contiguous_geometric_tracking = bool(
            self.pin_contiguous_geometric_tracking
        )
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
            pin_contiguous_geometric_tracking=pin_contiguous_geometric_tracking,
        )

    def to_dict(self) -> dict[str, object]:
        normalized = self.normalized()
        return {
            "rstep_a": float(normalized.rstep),
            "theta_divisions": int(normalized.theta_divisions),
            "phi_divisions": int(normalized.phi_divisions),
            "rmax_a": float(normalized.rmax),
            "pin_contiguous_geometric_tracking": bool(
                normalized.pin_contiguous_geometric_tracking
            ),
        }


@dataclass(slots=True, frozen=True)
class ElectronDensityDebyeWallerPairTerm:
    scope: str
    type_definition: str
    pair_label_a: str
    pair_label_b: str
    pair_label: str
    mean_distance_a: float
    sigma_a: float
    sigma_std_a: float
    sigma_squared_a2: float
    sigma_squared_std_a2: float
    b_factor_a2: float
    b_factor_std_a2: float

    @property
    def key(self) -> tuple[str, str]:
        normalized = self.normalized()
        return (
            str(normalized.scope),
            str(normalized.pair_label),
        )

    def normalized(self) -> "ElectronDensityDebyeWallerPairTerm":
        return ElectronDensityDebyeWallerPairTerm(
            scope=str(self.scope or "").strip(),
            type_definition=str(self.type_definition or "").strip(),
            pair_label_a=str(self.pair_label_a or "").strip(),
            pair_label_b=str(self.pair_label_b or "").strip(),
            pair_label=str(self.pair_label or "").strip(),
            mean_distance_a=max(float(self.mean_distance_a), 0.0),
            sigma_a=max(float(self.sigma_a), 0.0),
            sigma_std_a=max(float(self.sigma_std_a), 0.0),
            sigma_squared_a2=max(float(self.sigma_squared_a2), 0.0),
            sigma_squared_std_a2=max(float(self.sigma_squared_std_a2), 0.0),
            b_factor_a2=max(float(self.b_factor_a2), 0.0),
            b_factor_std_a2=max(float(self.b_factor_std_a2), 0.0),
        )

    def to_dict(self) -> dict[str, object]:
        normalized = self.normalized()
        return {
            "scope": str(normalized.scope),
            "type_definition": str(normalized.type_definition),
            "pair_label_a": str(normalized.pair_label_a),
            "pair_label_b": str(normalized.pair_label_b),
            "pair_label": str(normalized.pair_label),
            "mean_distance_a": float(normalized.mean_distance_a),
            "sigma_a": float(normalized.sigma_a),
            "sigma_std_a": float(normalized.sigma_std_a),
            "sigma_squared_a2": float(normalized.sigma_squared_a2),
            "sigma_squared_std_a2": float(normalized.sigma_squared_std_a2),
            "b_factor_a2": float(normalized.b_factor_a2),
            "b_factor_std_a2": float(normalized.b_factor_std_a2),
        }


def _debye_waller_pair_term_sort_key(
    term: ElectronDensityDebyeWallerPairTerm,
) -> tuple[str, str, str, str]:
    normalized = term.normalized()
    return (
        str(normalized.scope).lower(),
        str(normalized.pair_label).lower(),
        str(normalized.pair_label_a).lower(),
        str(normalized.pair_label_b).lower(),
    )


@dataclass(slots=True, frozen=True)
class ElectronDensitySmearingSettings:
    debye_waller_factor: float = 0.006
    debye_waller_mode: str = _DEBYE_WALLER_MODE_UNIVERSAL
    pair_specific_terms: tuple[ElectronDensityDebyeWallerPairTerm, ...] = ()
    imported_pair_specific_terms: tuple[
        ElectronDensityDebyeWallerPairTerm, ...
    ] = ()

    def normalized(self) -> "ElectronDensitySmearingSettings":
        factor = max(float(self.debye_waller_factor), 0.0)
        pair_specific_terms = tuple(
            sorted(
                (term.normalized() for term in self.pair_specific_terms),
                key=_debye_waller_pair_term_sort_key,
            )
        )
        imported_pair_specific_terms = tuple(
            sorted(
                (
                    term.normalized()
                    for term in self.imported_pair_specific_terms
                ),
                key=_debye_waller_pair_term_sort_key,
            )
        )
        if not imported_pair_specific_terms and pair_specific_terms:
            imported_pair_specific_terms = pair_specific_terms
        if not pair_specific_terms and imported_pair_specific_terms:
            pair_specific_terms = imported_pair_specific_terms
        debye_waller_mode = str(self.debye_waller_mode or "").strip().lower()
        if debye_waller_mode not in _SUPPORTED_DEBYE_WALLER_MODES:
            debye_waller_mode = (
                _DEBYE_WALLER_MODE_PAIR_SPECIFIC
                if pair_specific_terms
                else _DEBYE_WALLER_MODE_UNIVERSAL
            )
        if (
            debye_waller_mode == _DEBYE_WALLER_MODE_PAIR_SPECIFIC
            and not pair_specific_terms
        ):
            debye_waller_mode = _DEBYE_WALLER_MODE_UNIVERSAL
        return ElectronDensitySmearingSettings(
            debye_waller_factor=factor,
            debye_waller_mode=debye_waller_mode,
            pair_specific_terms=pair_specific_terms,
            imported_pair_specific_terms=imported_pair_specific_terms,
        )

    @property
    def gaussian_sigma_a(self) -> float:
        # TODO: When pair-specific Debye-Waller smearing is implemented,
        # derive the active broadening from the imported pair terms instead of
        # always falling back to the universal Gaussian factor.
        return float(np.sqrt(self.normalized().debye_waller_factor))

    @property
    def has_pair_specific_terms(self) -> bool:
        return bool(self.normalized().pair_specific_terms)

    @property
    def uses_pair_specific_terms(self) -> bool:
        normalized = self.normalized()
        return (
            normalized.debye_waller_mode == _DEBYE_WALLER_MODE_PAIR_SPECIFIC
            and bool(normalized.pair_specific_terms)
        )

    def with_pair_specific_terms(
        self,
        terms: (
            tuple[ElectronDensityDebyeWallerPairTerm, ...]
            | list[ElectronDensityDebyeWallerPairTerm]
        ),
        *,
        imported_terms: (
            tuple[ElectronDensityDebyeWallerPairTerm, ...]
            | list[ElectronDensityDebyeWallerPairTerm]
            | None
        ) = None,
        prefer_pair_specific: bool = True,
    ) -> "ElectronDensitySmearingSettings":
        return ElectronDensitySmearingSettings(
            debye_waller_factor=float(self.debye_waller_factor),
            debye_waller_mode=(
                _DEBYE_WALLER_MODE_PAIR_SPECIFIC
                if prefer_pair_specific and terms
                else _DEBYE_WALLER_MODE_UNIVERSAL
            ),
            pair_specific_terms=tuple(terms),
            imported_pair_specific_terms=(
                tuple(terms)
                if imported_terms is None
                else tuple(imported_terms)
            ),
        ).normalized()

    def reset_pair_specific_terms(self) -> "ElectronDensitySmearingSettings":
        normalized = self.normalized()
        return ElectronDensitySmearingSettings(
            debye_waller_factor=float(normalized.debye_waller_factor),
            debye_waller_mode=str(normalized.debye_waller_mode),
            pair_specific_terms=tuple(normalized.imported_pair_specific_terms),
            imported_pair_specific_terms=tuple(
                normalized.imported_pair_specific_terms
            ),
        ).normalized()

    def to_dict(self) -> dict[str, object]:
        normalized = self.normalized()
        return {
            "debye_waller_factor_a2": float(normalized.debye_waller_factor),
            "gaussian_sigma_a": float(normalized.gaussian_sigma_a),
            "debye_waller_mode": str(normalized.debye_waller_mode),
            "pair_specific_terms": [
                term.to_dict() for term in normalized.pair_specific_terms
            ],
            "imported_pair_specific_terms": [
                term.to_dict()
                for term in normalized.imported_pair_specific_terms
            ],
        }


@dataclass(slots=True, frozen=True)
class ElectronDensityFourierTransformSettings:
    r_min: float = 0.0
    r_max: float = 1.0
    domain_mode: str = "mirrored"
    window_function: str = "none"
    resampling_points: int = 2048
    q_min: float = 0.02
    q_max: float = 1.2
    q_step: float = 0.01
    use_solvent_subtracted_profile: bool = True
    log_q_axis: bool = True
    log_intensity_axis: bool = True

    def normalized(self) -> "ElectronDensityFourierTransformSettings":
        domain_mode = (
            _normalized_transform_domain_mode(self.domain_mode) or "legacy"
        )
        resampling_points = int(self.resampling_points)
        q_min = max(float(self.q_min), 0.0)
        q_max = float(self.q_max)
        q_step = float(self.q_step)
        window_function = (
            _normalized_transform_window_name(self.window_function) or "none"
        )
        if domain_mode not in _SUPPORTED_TRANSFORM_DOMAIN_MODES:
            raise ValueError(
                "Unsupported Fourier-transform domain mode. Choose from: "
                + ", ".join(sorted(_SUPPORTED_TRANSFORM_DOMAIN_MODES))
                + "."
            )
        if domain_mode == "mirrored":
            r_max = max(
                abs(float(self.r_max)),
                abs(float(self.r_min)),
            )
            r_min = -r_max
            if r_max <= 0.0:
                raise ValueError(
                    "Transform rmax must be greater than zero in mirrored mode."
                )
        else:
            r_min = max(float(self.r_min), 0.0)
            r_max = float(self.r_max)
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
            domain_mode=domain_mode,
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
            "domain_mode": str(normalized.domain_mode),
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
class ElectronDensityContiguousFrameSetSummary:
    series_label: str
    frame_ids: tuple[int, ...]
    frame_labels: tuple[str, ...]
    file_paths: tuple[Path, ...]

    @property
    def frame_count(self) -> int:
        return int(len(self.frame_ids))

    @property
    def frame_range_label(self) -> str:
        if not self.frame_labels:
            return "Unavailable"
        if len(self.frame_labels) == 1:
            return self.frame_labels[0]
        return f"{self.frame_labels[0]}-{self.frame_labels[-1]}"

    def to_dict(self) -> dict[str, object]:
        return {
            "series_label": str(self.series_label),
            "frame_ids": [int(value) for value in self.frame_ids],
            "frame_labels": [str(value) for value in self.frame_labels],
            "file_paths": [str(path) for path in self.file_paths],
        }


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
    averaging_mode: str = "complete_average"
    contiguous_frame_mode_requested: bool = False
    contiguous_frame_mode_applied: bool = False
    pinned_geometric_tracking_requested: bool = False
    pinned_geometric_tracking_applied: bool = False
    averaging_notes: tuple[str, ...] = ()
    contiguous_frame_sets: tuple[
        ElectronDensityContiguousFrameSetSummary, ...
    ] = ()
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
    source_mode: str = "density_fourier"


@dataclass(slots=True, frozen=True)
class ElectronDensityScatteringTransformResult:
    preview: ElectronDensityFourierTransformPreview
    q_values: np.ndarray
    scattering_amplitude: np.ndarray
    intensity: np.ndarray


@dataclass(slots=True, frozen=True)
class ElectronDensityDebyeScatteringAverageResult:
    q_values: np.ndarray
    mean_intensity: np.ndarray
    std_intensity: np.ndarray
    se_intensity: np.ndarray
    source_files: tuple[Path, ...]
    source_structure_count: int
    unique_elements: tuple[str, ...]
    notes: tuple[str, ...] = ()


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
    include_bonds: bool = True,
    include_comment: bool = True,
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

    bonds: tuple[tuple[int, int], ...] = ()
    if include_bonds:
        preview_atoms = tuple(
            PreviewAtomRecord(
                element=element,
                position=tuple(
                    float(value) for value in coordinates_array[index]
                ),
            )
            for index, element in enumerate(elements)
        )
        bonds = detect_bonds_records(preview_atoms)
    comment = read_structure_comment(resolved) if include_comment else ""
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


def _prepared_reference_structure_for_run(
    reference_structure: ElectronDensityStructure | None,
    *,
    reference_file: Path,
    center_mode: str,
    reference_element: str | None = None,
) -> ElectronDensityStructure | None:
    if reference_structure is None:
        return None
    resolved_reference = Path(reference_file).expanduser().resolve()
    if reference_structure.file_path != resolved_reference:
        return None
    requested_reference_element = _normalized_element_symbol(
        reference_element or reference_structure.reference_element
    )
    needs_recenter = reference_structure.center_mode != center_mode
    if (
        center_mode == "reference_element"
        and requested_reference_element
        and requested_reference_element
        != reference_structure.reference_element
    ):
        needs_recenter = True
    if not needs_recenter:
        return reference_structure
    return recenter_electron_density_structure(
        reference_structure,
        center_mode=center_mode,
        reference_element=requested_reference_element or None,
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


def _tag_atoms_by_radial_shell(
    radial_distances: np.ndarray,
    radial_edges: np.ndarray,
    *,
    domain_max_radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    distances = np.asarray(radial_distances, dtype=float)
    edges = np.asarray(radial_edges, dtype=float)
    shell_count = int(max(len(edges) - 1, 0))
    in_domain_mask = distances <= (float(domain_max_radius) + 1.0e-12)
    shell_tags = np.full(distances.shape, -1, dtype=np.intp)
    if shell_count <= 0 or not np.any(in_domain_mask):
        return in_domain_mask, shell_tags

    # Assign exactly one shell tag per in-domain atom, then sum by tag. This
    # keeps origin-centered atoms in the first shell instead of letting them
    # inflate every angular bin that touches r = 0.
    candidate_shell_indices = (
        np.searchsorted(edges, distances[in_domain_mask], side="right") - 1
    )
    shell_tags[in_domain_mask] = np.clip(
        candidate_shell_indices,
        0,
        shell_count - 1,
    )
    return in_domain_mask, shell_tags


def _sphere_overlap_volume(
    sphere_radius: float,
    radial_limits: np.ndarray,
    *,
    center_distance: float,
) -> np.ndarray:
    finite_radius = max(float(sphere_radius), 0.0)
    radial_array = np.clip(np.asarray(radial_limits, dtype=float), 0.0, None)
    if finite_radius <= 0.0 or radial_array.size == 0:
        return np.zeros_like(radial_array, dtype=float)

    separation = abs(float(center_distance))
    volume = np.zeros_like(radial_array, dtype=float)
    sphere_volume = (4.0 * np.pi / 3.0) * finite_radius**3

    contains_mask = separation <= np.abs(radial_array - finite_radius)
    if np.any(contains_mask):
        min_radii = np.minimum(radial_array[contains_mask], finite_radius)
        volume[contains_mask] = 4.0 * np.pi / 3.0 * np.power(min_radii, 3.0)

    partial_mask = (
        ~contains_mask
        & (separation < (radial_array + finite_radius))
        & (radial_array > 0.0)
    )
    if np.any(partial_mask):
        radial_subset = radial_array[partial_mask]
        numerator = (
            np.pi
            * np.square(radial_subset + finite_radius - separation)
            * (
                separation**2
                + (2.0 * separation * (radial_subset + finite_radius))
                - 3.0 * np.square(radial_subset - finite_radius)
            )
        )
        denominator = max(12.0 * separation, 1.0e-12)
        volume[partial_mask] = np.clip(
            numerator / denominator,
            0.0,
            sphere_volume,
        )

    return np.clip(volume, 0.0, sphere_volume)


def _finite_radius_shell_overlap_counts(
    radial_edges: np.ndarray,
    *,
    atom_distance: float,
    electron_count: float,
    overlap_radius: float,
) -> np.ndarray:
    edge_array = np.asarray(radial_edges, dtype=float)
    shell_count = int(max(len(edge_array) - 1, 0))
    if shell_count <= 0 or electron_count <= 0.0:
        return np.asarray([], dtype=float)
    cumulative_overlap = _sphere_overlap_volume(
        overlap_radius,
        edge_array,
        center_distance=atom_distance,
    )
    shell_overlap = np.diff(cumulative_overlap)
    shell_overlap = np.clip(shell_overlap, 0.0, None)
    overlap_total = float(np.sum(shell_overlap))
    if overlap_total <= 0.0:
        return np.zeros(shell_count, dtype=float)
    return np.asarray(shell_overlap, dtype=float) * (
        float(electron_count) / overlap_total
    )


def _resolve_electron_density_worker_count(
    structure_count: int,
) -> int:
    if structure_count <= 1:
        return 1
    available_cpus = os.cpu_count() or 1
    conservative_cpu_budget = max(1, (available_cpus + 1) // 2)
    return max(
        1,
        min(
            int(structure_count),
            int(conservative_cpu_budget),
            4,
        ),
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


@dataclass(slots=True, frozen=True)
class _ContiguousFrameRecord:
    file_path: Path
    frame_id: int
    frame_label: str
    series_key: str
    series_label: str


def _frame_series_label(prefix: str, suffix: str) -> str:
    parts = [part for part in (prefix.strip("_"), suffix.strip("_")) if part]
    if not parts:
        return "default"
    return " ".join(parts)


def _parse_contiguous_frame_record(
    file_path: str | Path,
) -> _ContiguousFrameRecord | None:
    resolved = Path(file_path).expanduser().resolve()
    stem = str(resolved.stem)
    match = _FRAME_ID_PATTERN.search(stem)
    if match is None:
        return None
    frame_label = str(match.group(1))
    prefix = stem[: match.start()]
    suffix = stem[match.end() :]
    return _ContiguousFrameRecord(
        file_path=resolved,
        frame_id=int(frame_label),
        frame_label=frame_label,
        series_key=f"{prefix}::{suffix}",
        series_label=_frame_series_label(prefix, suffix),
    )


def _detect_contiguous_frame_sets(
    structure_files: tuple[Path, ...],
) -> tuple[tuple[ElectronDensityContiguousFrameSetSummary, ...], str | None]:
    if not structure_files:
        return tuple(), None
    records: list[_ContiguousFrameRecord] = []
    for file_path in structure_files:
        parsed = _parse_contiguous_frame_record(file_path)
        if parsed is None:
            return (
                tuple(),
                "Contiguous-frame averaging requires every input file to "
                "contain a frame_<NNNN> identifier. Falling back to complete averaging.",
            )
        records.append(parsed)

    series_map: dict[str, list[_ContiguousFrameRecord]] = {}
    for record in sorted(
        records,
        key=lambda item: (item.series_key, item.frame_id, item.file_path.name),
    ):
        series_map.setdefault(record.series_key, []).append(record)

    contiguous_sets: list[ElectronDensityContiguousFrameSetSummary] = []
    for series_records in series_map.values():
        current_records: list[_ContiguousFrameRecord] = []
        for record in series_records:
            if (
                current_records
                and record.frame_id != current_records[-1].frame_id + 1
            ):
                contiguous_sets.append(
                    ElectronDensityContiguousFrameSetSummary(
                        series_label=str(current_records[0].series_label),
                        frame_ids=tuple(
                            int(entry.frame_id) for entry in current_records
                        ),
                        frame_labels=tuple(
                            str(entry.frame_label) for entry in current_records
                        ),
                        file_paths=tuple(
                            entry.file_path for entry in current_records
                        ),
                    )
                )
                current_records = []
            current_records.append(record)
        if current_records:
            contiguous_sets.append(
                ElectronDensityContiguousFrameSetSummary(
                    series_label=str(current_records[0].series_label),
                    frame_ids=tuple(
                        int(entry.frame_id) for entry in current_records
                    ),
                    frame_labels=tuple(
                        str(entry.frame_label) for entry in current_records
                    ),
                    file_paths=tuple(
                        entry.file_path for entry in current_records
                    ),
                )
            )
    return tuple(contiguous_sets), None


def _heaviest_element_geometric_center(
    structure: ElectronDensityStructure,
) -> np.ndarray:
    (
        _resolved_reference_element,
        _geometric_center,
        reference_element_geometric_center,
        _reference_element_offset,
    ) = _reference_element_center_details(
        structure.coordinates,
        structure.elements,
        reference_element=None,
    )
    return np.asarray(reference_element_geometric_center, dtype=float)


def _structure_with_active_center(
    structure: ElectronDensityStructure,
    active_center: np.ndarray,
) -> ElectronDensityStructure:
    active_center_array = np.asarray(active_center, dtype=float)
    centered_coordinates, rmax = _centered_coordinates_from_origin(
        structure.coordinates,
        active_center_array,
    )
    return replace(
        structure,
        centered_coordinates=np.asarray(centered_coordinates, dtype=float),
        active_center=active_center_array,
        rmax=float(rmax),
    )


def _apply_contiguous_frame_center_lock(
    structures: tuple[ElectronDensityStructure, ...],
    contiguous_frame_sets: tuple[
        ElectronDensityContiguousFrameSetSummary, ...
    ],
) -> tuple[ElectronDensityStructure, ...]:
    if not structures or not contiguous_frame_sets:
        return structures
    structure_map = {
        structure.file_path: structure for structure in structures
    }
    locked_structures: dict[Path, ElectronDensityStructure] = {}
    for frame_set in contiguous_frame_sets:
        set_structures = [
            structure_map[path]
            for path in frame_set.file_paths
            if path in structure_map
        ]
        if not set_structures:
            continue
        anchor_centers = [
            _heaviest_element_geometric_center(structure)
            for structure in set_structures
        ]
        average_offset = np.mean(
            np.vstack(
                [
                    np.asarray(structure.active_center, dtype=float) - anchor
                    for structure, anchor in zip(
                        set_structures,
                        anchor_centers,
                    )
                ]
            ),
            axis=0,
        )
        for structure, anchor_center in zip(set_structures, anchor_centers):
            locked_structures[structure.file_path] = (
                _structure_with_active_center(
                    structure,
                    np.asarray(anchor_center, dtype=float) + average_offset,
                )
            )
    return tuple(
        locked_structures.get(structure.file_path, structure)
        for structure in structures
    )


def _structure_files_are_pdb_only(
    structure_files: tuple[Path, ...],
) -> bool:
    return bool(structure_files) and all(
        Path(file_path).suffix.lower() == ".pdb"
        for file_path in structure_files
    )


def _apply_pinned_geometric_tracking_to_frame_sets(
    structures: tuple[ElectronDensityStructure, ...],
    contiguous_frame_sets: tuple[
        ElectronDensityContiguousFrameSetSummary, ...
    ],
) -> tuple[ElectronDensityStructure, ...]:
    if not structures or not contiguous_frame_sets:
        return structures
    structure_map = {
        structure.file_path: structure for structure in structures
    }
    locked_structures: dict[Path, ElectronDensityStructure] = {}
    for frame_set in contiguous_frame_sets:
        set_structures = [
            structure_map[path]
            for path in frame_set.file_paths
            if path in structure_map
        ]
        if not set_structures:
            continue
        pinned_center = np.asarray(
            set_structures[0].active_center,
            dtype=float,
        )
        for structure in set_structures:
            locked_structures[structure.file_path] = (
                _structure_with_active_center(
                    structure,
                    pinned_center,
                )
            )
    return tuple(
        locked_structures.get(structure.file_path, structure)
        for structure in structures
    )


def _compute_smeared_profile_statistics(
    radial_centers: np.ndarray,
    shell_volumes: np.ndarray,
    member_density_profiles: tuple[np.ndarray, ...],
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
    density_members = tuple(
        np.asarray(profile, dtype=float).copy()
        for profile in member_density_profiles
    )
    if not density_members:
        raise ValueError(
            "At least one member profile is required for smearing."
        )
    density_stack = np.vstack(density_members)
    if density_stack.shape[1] <= 1 or active_smearing.gaussian_sigma_a <= 0.0:
        smeared_density_stack = np.asarray(
            density_stack,
            dtype=float,
        ).copy()
    else:
        kernel = _count_conserving_gaussian_transfer_kernel(
            radial_centers,
            shell_volumes,
            sigma_a=active_smearing.gaussian_sigma_a,
        )
        smeared_density_stack = np.asarray(
            density_stack @ kernel.T,
            dtype=float,
        )
        # Keep the finite-precision balanced transport from ever increasing the
        # integrated electron density. Small losses remain acceptable if a
        # future smearing kernel intentionally leaks density outside the domain.
        smeared_density_stack = _cap_smeared_density_stack_to_input_totals(
            density_stack,
            smeared_density_stack,
            shell_volumes,
        )
    smeared_mean = np.mean(smeared_density_stack, axis=0)
    if len(density_members) > 1:
        smeared_variance = np.var(smeared_density_stack, axis=0)
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
    # TODO: Apply pair-specific Debye-Waller terms to the electron-density
    # smearing path once the pair-aware broadening model is implemented.
    # For now this continues to use the universal Gaussian factor only.
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
        np.asarray(result.shell_volumes, dtype=float),
        tuple(
            np.asarray(profile, dtype=float).copy()
            for profile in member_profiles
        ),
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
    symmetric_about_zero = bool(
        float(r_min) < 0.0 < float(r_max)
        and np.isclose(float(r_min), -float(r_max), atol=1.0e-12)
    )
    if normalized_window == "lorch":
        argument = normalized_centered if symmetric_about_zero else position
        return np.asarray(np.sinc(argument), dtype=float)
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


def _q_values_from_transform_settings(
    settings: ElectronDensityFourierTransformSettings,
) -> np.ndarray:
    normalized = settings.normalized()
    q_values = np.arange(
        float(normalized.q_min),
        float(normalized.q_max) + float(normalized.q_step) * 0.5,
        float(normalized.q_step),
        dtype=float,
    )
    if q_values.size > 0 and q_values[-1] > float(normalized.q_max) + 1.0e-12:
        q_values = q_values[:-1]
    return np.asarray(q_values, dtype=float)


def _validated_debye_q_values(
    *,
    settings: ElectronDensityFourierTransformSettings | None,
    q_values: np.ndarray | list[float] | tuple[float, ...] | None,
) -> np.ndarray:
    if q_values is None:
        if settings is None:
            raise ValueError(
                "Provide Fourier settings or an explicit q grid for the Debye scattering calculation."
            )
        return _q_values_from_transform_settings(settings)
    q_grid = np.asarray(q_values, dtype=float)
    if q_grid.ndim != 1 or q_grid.size == 0:
        raise ValueError(
            "Debye scattering q-values must be a non-empty 1D grid."
        )
    if not np.all(np.isfinite(q_grid)):
        raise ValueError("Debye scattering q-values must be finite.")
    if q_grid.size > 1 and np.any(np.diff(q_grid) <= 0.0):
        raise ValueError(
            "Debye scattering q-values must be strictly increasing."
        )
    return np.asarray(q_grid, dtype=float)


def _all_single_atom_structures_are_equivalent(
    structures: list[tuple[np.ndarray, tuple[str, ...]]],
) -> bool:
    if not structures:
        return False
    atom_signatures: set[tuple[str, ...]] = set()
    for coordinates, elements in structures:
        coordinate_array = np.asarray(coordinates, dtype=float)
        if (
            coordinate_array.ndim != 2
            or coordinate_array.shape[0] != 1
            or coordinate_array.shape[1] != 3
            or len(elements) != 1
        ):
            return False
        atom_signatures.add(tuple(str(element) for element in elements))
    return len(atom_signatures) == 1


def prepare_single_atom_debye_scattering_preview(
    settings: ElectronDensityFourierTransformSettings,
) -> ElectronDensityFourierTransformPreview:
    normalized = settings.normalized()
    return ElectronDensityFourierTransformPreview(
        settings=normalized,
        source_profile_label="Single-atom Debye scattering",
        source_radial_values=np.asarray([], dtype=float),
        source_density_values=np.asarray([], dtype=float),
        resampled_r_values=np.asarray([], dtype=float),
        resampled_density_values=np.asarray([], dtype=float),
        window_values=np.asarray([], dtype=float),
        windowed_density_values=np.asarray([], dtype=float),
        available_r_min=0.0,
        available_r_max=0.0,
        resampling_step_a=0.0,
        nyquist_q_max_a_inverse=0.0,
        independent_q_step_a_inverse=0.0,
        q_grid_is_oversampled=False,
        q_max_was_clamped=False,
        notes=(
            "Single-atom cluster bins use direct Debye scattering. "
            "Electron-density, solvent-contrast, and Fourier-profile "
            "preparation are skipped; only the q-grid controls apply.",
        ),
        source_mode="single_atom_debye",
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
    if normalized.domain_mode == "mirrored":
        source_radial_values, source_density_values = (
            _mirror_profile_about_zero(
                source_radial_values,
                source_density_values,
            )
        )
        available_r_min = -available_r_max
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
    ).normalized()
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
        source_mode="density_fourier",
    )


def compute_electron_density_scattering_profile(
    result: ElectronDensityProfileResult,
    settings: ElectronDensityFourierTransformSettings,
) -> ElectronDensityScatteringTransformResult:
    preview = prepare_electron_density_fourier_transform(result, settings)
    effective_settings = preview.settings
    q_values = _q_values_from_transform_settings(effective_settings)
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


def compute_average_debye_scattering_profile_for_input(
    selection: ElectronDensityInputInspection | str | Path,
    settings: ElectronDensityFourierTransformSettings | None = None,
    *,
    q_values: np.ndarray | list[float] | tuple[float, ...] | None = None,
    progress_callback: ElectronDensityProgressCallback | None = None,
    cancel_callback: ElectronDensityCancelCallback | None = None,
) -> ElectronDensityDebyeScatteringAverageResult:
    inspection = (
        selection
        if isinstance(selection, ElectronDensityInputInspection)
        else inspect_structure_input(selection)
    )
    structure_files = tuple(inspection.structure_files)
    if not structure_files:
        raise ValueError("No structure files were available for calculation.")

    q_grid = _validated_debye_q_values(settings=settings, q_values=q_values)
    total_steps = len(structure_files) * 2 + 2
    _raise_if_canceled(cancel_callback)
    _emit_progress(
        progress_callback,
        0,
        total_steps,
        "Preparing Debye scattering average calculation.",
    )

    loaded_structures: list[tuple[np.ndarray, tuple[str, ...]]] = []
    unique_elements: set[str] = set()
    for file_index, file_path in enumerate(structure_files, start=1):
        _raise_if_canceled(cancel_callback)
        _emit_progress(
            progress_callback,
            file_index,
            total_steps,
            "Loading structure "
            f"{file_index}/{len(structure_files)} for Debye scattering: "
            f"{file_path.name}",
        )
        coordinates, raw_elements = load_structure_file(file_path)
        coordinate_array = np.asarray(coordinates, dtype=float)
        elements = tuple(
            _normalized_element_symbol(element) for element in raw_elements
        )
        if (
            coordinate_array.ndim != 2
            or coordinate_array.shape[0] != len(elements)
            or coordinate_array.shape[1] != 3
        ):
            raise ValueError(
                f"Structure {file_path.name} could not be parsed into a valid coordinate array."
            )
        loaded_structures.append((coordinate_array, elements))
        unique_elements.update(
            element for element in elements if str(element).strip()
        )

    if not loaded_structures:
        raise ValueError(
            "No valid structures were available for Debye scattering."
        )

    _raise_if_canceled(cancel_callback)
    _emit_progress(
        progress_callback,
        len(structure_files) + 1,
        total_steps,
        "Building atomic form factors for "
        f"{len(unique_elements)} unique element"
        f"{'' if len(unique_elements) == 1 else 's'}.",
    )
    f0_dictionary = build_f0_dictionary(
        sorted(unique_elements),
        q_grid,
    )

    traces: list[np.ndarray] = []
    notes: list[str] = []
    if _all_single_atom_structures_are_equivalent(loaded_structures):
        coordinates, elements = loaded_structures[0]
        trace = compute_debye_intensity(
            coordinates,
            list(elements),
            q_grid,
            f0_dictionary=f0_dictionary,
        )
        traces = [np.asarray(trace, dtype=float)] * len(loaded_structures)
        notes.append(
            "All source structures contained the same single-atom type, so the Debye trace was evaluated once and reused across the average."
        )
        _emit_progress(
            progress_callback,
            total_steps - 1,
            total_steps,
            "Reused a single equivalent single-atom Debye trace across the average.",
        )
    else:
        for structure_index, (coordinates, elements) in enumerate(
            loaded_structures,
            start=1,
        ):
            _raise_if_canceled(cancel_callback)
            _emit_progress(
                progress_callback,
                len(structure_files) + 1 + structure_index,
                total_steps,
                "Computing Debye trace "
                f"{structure_index}/{len(loaded_structures)}.",
            )
            traces.append(
                np.asarray(
                    compute_debye_intensity(
                        coordinates,
                        list(elements),
                        q_grid,
                        f0_dictionary=f0_dictionary,
                    ),
                    dtype=float,
                )
            )

    stacked = np.asarray(traces, dtype=float)
    mean_trace = np.asarray(np.mean(stacked, axis=0), dtype=float)
    std_trace = np.asarray(np.std(stacked, axis=0), dtype=float)
    se_trace = np.asarray(
        std_trace / np.sqrt(float(max(stacked.shape[0], 1))),
        dtype=float,
    )
    _emit_progress(
        progress_callback,
        total_steps,
        total_steps,
        "Debye scattering average profile ready.",
    )
    return ElectronDensityDebyeScatteringAverageResult(
        q_values=np.asarray(q_grid, dtype=float),
        mean_intensity=mean_trace,
        std_intensity=std_trace,
        se_intensity=se_trace,
        source_files=structure_files,
        source_structure_count=len(structure_files),
        unique_elements=tuple(sorted(unique_elements)),
        notes=tuple(notes),
    )


def compute_single_atom_debye_scattering_profile_for_input(
    selection: ElectronDensityInputInspection | str | Path,
    settings: ElectronDensityFourierTransformSettings,
    *,
    progress_callback: ElectronDensityProgressCallback | None = None,
    cancel_callback: ElectronDensityCancelCallback | None = None,
) -> ElectronDensityScatteringTransformResult:
    inspection = (
        selection
        if isinstance(selection, ElectronDensityInputInspection)
        else inspect_structure_input(selection)
    )
    structure_files = tuple(inspection.structure_files)
    if not structure_files:
        raise ValueError("No structure files were available for calculation.")

    _raise_if_canceled(cancel_callback)
    total_steps = len(structure_files) + 2
    _emit_progress(
        progress_callback,
        0,
        total_steps,
        "Preparing single-atom Debye scattering calculation.",
    )
    preview = prepare_single_atom_debye_scattering_preview(settings)
    q_values = _q_values_from_transform_settings(preview.settings)
    element_counts: Counter[str] = Counter()

    for file_index, file_path in enumerate(structure_files, start=1):
        _raise_if_canceled(cancel_callback)
        _emit_progress(
            progress_callback,
            file_index,
            total_steps,
            "Validating single-atom structure "
            f"{file_index}/{len(structure_files)}: {file_path.name}",
        )
        coordinates, raw_elements = load_structure_file(file_path)
        coordinate_array = np.asarray(coordinates, dtype=float)
        elements = tuple(
            _normalized_element_symbol(element) for element in raw_elements
        )
        if (
            coordinate_array.ndim != 2
            or coordinate_array.shape[0] != 1
            or coordinate_array.shape[1] != 3
            or len(elements) != 1
        ):
            raise ValueError(
                "Single-atom Debye scattering requires exactly one atom in "
                f"every structure file; {file_path.name} did not match that requirement."
            )
        element_counts[str(elements[0])] += 1

    _raise_if_canceled(cancel_callback)
    _emit_progress(
        progress_callback,
        total_steps - 1,
        total_steps,
        "Computing single-atom Debye scattering from "
        f"{len(element_counts)} unique atom type"
        f"{'' if len(element_counts) == 1 else 's'}.",
    )
    weighted_intensity = np.zeros_like(q_values, dtype=float)
    total_structure_count = sum(
        int(count) for count in element_counts.values()
    )
    for element, count in sorted(element_counts.items()):
        _raise_if_canceled(cancel_callback)
        form_factor = atomic_form_factor(element, q_values)
        weighted_intensity += float(count) * np.square(
            np.asarray(form_factor, dtype=float)
        )
    intensity = weighted_intensity / float(max(total_structure_count, 1))
    amplitude = np.sqrt(np.clip(np.asarray(intensity, dtype=float), 0.0, None))
    _emit_progress(
        progress_callback,
        total_steps,
        total_steps,
        "Single-atom Debye scattering profile ready.",
    )
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

    radial_edges = np.asarray(mesh_geometry.radial_edges, dtype=float)
    shell_count = int(len(radial_edges) - 1)
    domain_max_radius = float(mesh_geometry.domain_max_radius)

    in_domain_mask, shell_tags = _tag_atoms_by_radial_shell(
        radial_distances,
        radial_edges,
        domain_max_radius=domain_max_radius,
    )
    excluded_atom_count = int(np.count_nonzero(~in_domain_mask))
    excluded_electron_count = float(
        np.sum(
            np.asarray(structure.atomic_numbers, dtype=float)[~in_domain_mask]
        )
    )
    atomic_numbers = np.asarray(structure.atomic_numbers, dtype=float)[
        in_domain_mask
    ]
    tagged_shells = np.asarray(shell_tags[in_domain_mask], dtype=np.intp)

    if shell_count > 0:
        shell_electron_counts = np.bincount(
            tagged_shells,
            weights=atomic_numbers,
            minlength=shell_count,
        ).astype(float, copy=False)
    else:
        shell_electron_counts = np.asarray([], dtype=float)

    shell_volumes = (4.0 * np.pi / 3.0) * (
        np.power(radial_edges[1:], 3.0) - np.power(radial_edges[:-1], 3.0)
    )
    shell_volume_average_density = np.divide(
        shell_electron_counts,
        shell_volumes,
        out=np.zeros_like(shell_electron_counts, dtype=float),
        where=shell_volumes > 0.0,
    )
    finite_radius_shell_electron_counts = np.zeros_like(
        shell_electron_counts,
        dtype=float,
    )
    in_domain_elements = structure.elements
    if np.any(in_domain_mask):
        in_domain_elements = tuple(
            structure.elements[index]
            for index, included in enumerate(in_domain_mask)
            if included
        )
    for atom_distance, atom_electrons, atom_element in zip(
        radial_distances[in_domain_mask],
        atomic_numbers,
        in_domain_elements,
        strict=False,
    ):
        overlap_radius = _effective_atomic_overlap_radius(str(atom_element))
        finite_radius_shell_electron_counts += (
            _finite_radius_shell_overlap_counts(
                radial_edges,
                atom_distance=float(atom_distance),
                electron_count=float(atom_electrons),
                overlap_radius=overlap_radius,
            )
        )

    orientation_average_density = np.asarray(
        np.divide(
            finite_radius_shell_electron_counts,
            shell_volumes,
            out=np.zeros_like(
                finite_radius_shell_electron_counts, dtype=float
            ),
            where=shell_volumes > 0.0,
        ),
        dtype=float,
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
        shell_volumes,
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
    reference_structure: ElectronDensityStructure | None = None,
    center_mode: str = "center_of_mass",
    reference_element: str | None = None,
    use_contiguous_frame_mode: bool = False,
    progress_callback: ElectronDensityProgressCallback | None = None,
    cancel_callback: ElectronDensityCancelCallback | None = None,
) -> ElectronDensityProfileResult:
    normalized_mesh_settings = mesh_settings.normalized()
    inspection = (
        selection
        if isinstance(selection, ElectronDensityInputInspection)
        else inspect_structure_input(selection)
    )
    structure_files = tuple(inspection.structure_files)
    if not structure_files:
        raise ValueError("No structure files were available for calculation.")

    should_try_contiguous_mode = bool(
        use_contiguous_frame_mode
        and inspection.input_mode == "folder"
        and len(structure_files) > 1
    )
    contiguous_frame_sets: tuple[
        ElectronDensityContiguousFrameSetSummary, ...
    ] = ()
    averaging_mode = "complete_average"
    contiguous_frame_mode_applied = False
    pinned_geometric_tracking_requested = bool(
        should_try_contiguous_mode
        and center_mode == "center_of_mass"
        and normalized_mesh_settings.pin_contiguous_geometric_tracking
    )
    pinned_geometric_tracking_applied = False
    averaging_notes: list[str] = []
    total_steps = (
        len(structure_files) * 2 + 2 + (1 if should_try_contiguous_mode else 0)
    )
    step_index = 0
    _raise_if_canceled(cancel_callback)
    if inspection.input_mode == "folder":
        _emit_progress(
            progress_callback,
            step_index,
            total_steps,
            "Preparing ensemble electron-density calculation for "
            f"{len(structure_files)} structures using "
            f"{_progress_center_mode_label(center_mode, reference_element=reference_element)}"
            + (
                " with contiguous-frame evaluation enabled."
                if should_try_contiguous_mode
                else "."
            ),
        )
    else:
        _emit_progress(
            progress_callback,
            step_index,
            total_steps,
            "Preparing electron-density calculation for "
            f"{structure_files[0].name} using "
            f"{_progress_center_mode_label(center_mode, reference_element=reference_element)}.",
        )

    if should_try_contiguous_mode:
        step_index += 1
        _raise_if_canceled(cancel_callback)
        contiguous_frame_sets, fallback_reason = _detect_contiguous_frame_sets(
            structure_files
        )
        if fallback_reason is None:
            contiguous_frame_mode_applied = True
            averaging_mode = "contiguous_frame_sets"
            if pinned_geometric_tracking_requested:
                if _structure_files_are_pdb_only(structure_files):
                    pinned_geometric_tracking_applied = True
                    averaging_notes.append(
                        "Pinned each contiguous PDB frame set to the first "
                        "frame's geometric mass center coordinates."
                    )
                else:
                    averaging_notes.append(
                        "Pinned geometric tracking requires contiguous PDB "
                        "frame sets. Reused the standard shared-center "
                        "contiguous evaluation instead."
                    )
            if not pinned_geometric_tracking_applied:
                averaging_notes.append(
                    "Locked each contiguous frame set to a shared active-center "
                    "offset relative to the heaviest-element geometric center."
                )
            contiguous_structure_count = sum(
                frame_set.frame_count for frame_set in contiguous_frame_sets
            )
            frame_set_count = len(contiguous_frame_sets)
            mode_message = (
                "pinning each set to its first frame's geometric mass center."
                if pinned_geometric_tracking_applied
                else "locking active centers within each set."
            )
            _emit_progress(
                progress_callback,
                step_index,
                total_steps,
                "Detected "
                f"{frame_set_count} contiguous frame set"
                f"{'' if frame_set_count == 1 else 's'} across "
                f"{contiguous_structure_count} structure"
                f"{'' if contiguous_structure_count == 1 else 's'}; "
                + mode_message,
            )
        else:
            averaging_notes.append(fallback_reason)
            _emit_progress(
                progress_callback,
                step_index,
                total_steps,
                fallback_reason,
            )

    worker_count = _resolve_electron_density_worker_count(len(structure_files))
    prepared_reference_structure = _prepared_reference_structure_for_run(
        reference_structure,
        reference_file=inspection.reference_file,
        center_mode=center_mode,
        reference_element=reference_element,
    )
    loaded_structures: list[ElectronDensityStructure] = []
    if worker_count <= 1:
        for file_index, file_path in enumerate(structure_files, start=1):
            step_index += 1
            _raise_if_canceled(cancel_callback)
            _emit_progress(
                progress_callback,
                step_index,
                total_steps,
                f"Loading structure {file_index}/{len(structure_files)}: {file_path.name}",
            )
            if file_index == 1 and prepared_reference_structure is not None:
                structure = prepared_reference_structure
            else:
                structure = load_electron_density_structure(
                    file_path,
                    center_mode=center_mode,
                    reference_element=reference_element,
                    include_bonds=file_index == 1,
                    include_comment=file_index == 1,
                )
            loaded_structures.append(structure)
    else:
        ordered_structures: list[ElectronDensityStructure | None] = [
            None
        ] * len(structure_files)
        with ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="electron-density-load",
        ) as executor:
            pending_futures = {}
            next_index = 0
            completed_count = 0
            if prepared_reference_structure is not None:
                ordered_structures[0] = prepared_reference_structure
                completed_count = 1
                step_index += 1
                _emit_progress(
                    progress_callback,
                    step_index,
                    total_steps,
                    "Loading structure "
                    f"1/{len(structure_files)}: {structure_files[0].name}",
                )
                next_index = 1
            while completed_count < len(structure_files):
                _raise_if_canceled(cancel_callback)
                while (
                    next_index < len(structure_files)
                    and len(pending_futures) < worker_count
                ):
                    file_path = structure_files[next_index]
                    _emit_progress(
                        progress_callback,
                        step_index,
                        total_steps,
                        "Loading structure "
                        f"{next_index + 1}/{len(structure_files)}: {file_path.name}",
                    )
                    future = executor.submit(
                        load_electron_density_structure,
                        file_path,
                        center_mode=center_mode,
                        reference_element=reference_element,
                        include_bonds=next_index == 0,
                        include_comment=next_index == 0,
                    )
                    pending_futures[future] = next_index
                    next_index += 1
                done_futures, _pending = wait(
                    tuple(pending_futures),
                    return_when=FIRST_COMPLETED,
                )
                for future in done_futures:
                    member_index = pending_futures.pop(future)
                    ordered_structures[member_index] = future.result()
                    completed_count += 1
                    step_index += 1
                    _emit_progress(
                        progress_callback,
                        step_index,
                        total_steps,
                        "Loaded structure "
                        f"{completed_count}/{len(structure_files)}: "
                        f"{structure_files[member_index].name}",
                    )
        loaded_structures = [
            structure
            for structure in ordered_structures
            if structure is not None
        ]
    structure_members = tuple(loaded_structures)
    if contiguous_frame_mode_applied:
        _raise_if_canceled(cancel_callback)
        if pinned_geometric_tracking_applied:
            structure_members = _apply_pinned_geometric_tracking_to_frame_sets(
                structure_members,
                contiguous_frame_sets,
            )
        else:
            structure_members = _apply_contiguous_frame_center_lock(
                structure_members,
                contiguous_frame_sets,
            )

    member_results: list[ElectronDensityProfileResult] = []
    member_compute_smearing = ElectronDensitySmearingSettings(
        debye_waller_factor=0.0
    )
    if worker_count <= 1:
        for file_index, structure in enumerate(structure_members, start=1):
            step_index += 1
            _raise_if_canceled(cancel_callback)
            _emit_progress(
                progress_callback,
                step_index,
                total_steps,
                "Computing electron density "
                f"{file_index}/{len(structure_files)}: {structure.file_path.name}",
            )
            member_results.append(
                compute_electron_density_profile(
                    structure,
                    normalized_mesh_settings,
                    smearing_settings=member_compute_smearing,
                )
            )
    else:
        ordered_results: list[ElectronDensityProfileResult | None] = [
            None
        ] * len(structure_members)
        with ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="electron-density",
        ) as executor:
            pending_futures = {}
            next_index = 0
            completed_count = 0
            while completed_count < len(structure_members):
                _raise_if_canceled(cancel_callback)
                while (
                    next_index < len(structure_members)
                    and len(pending_futures) < worker_count
                ):
                    future = executor.submit(
                        compute_electron_density_profile,
                        structure_members[next_index],
                        normalized_mesh_settings,
                        smearing_settings=member_compute_smearing,
                    )
                    pending_futures[future] = next_index
                    next_index += 1
                done_futures, _pending = wait(
                    tuple(pending_futures),
                    return_when=FIRST_COMPLETED,
                )
                for future in done_futures:
                    member_index = pending_futures.pop(future)
                    ordered_results[member_index] = future.result()
                    completed_count += 1
                    step_index += 1
                    _emit_progress(
                        progress_callback,
                        step_index,
                        total_steps,
                        "Computing electron density "
                        f"{completed_count}/{len(structure_files)}: "
                        f"{structure_members[member_index].file_path.name}",
                    )
        member_results = [
            result for result in ordered_results if result is not None
        ]

    step_index += 1
    _raise_if_canceled(cancel_callback)
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
        np.asarray(reference_result.shell_volumes, dtype=float),
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
        averaging_mode=str(averaging_mode),
        contiguous_frame_mode_requested=bool(should_try_contiguous_mode),
        contiguous_frame_mode_applied=bool(contiguous_frame_mode_applied),
        pinned_geometric_tracking_requested=bool(
            pinned_geometric_tracking_requested
        ),
        pinned_geometric_tracking_applied=bool(
            pinned_geometric_tracking_applied
        ),
        averaging_notes=tuple(str(note) for note in averaging_notes),
        contiguous_frame_sets=tuple(contiguous_frame_sets),
    )
    step_index += 1
    _raise_if_canceled(cancel_callback)
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
        "averaging_mode": str(result.averaging_mode),
        "contiguous_frame_mode_requested": bool(
            result.contiguous_frame_mode_requested
        ),
        "contiguous_frame_mode_applied": bool(
            result.contiguous_frame_mode_applied
        ),
        "pinned_geometric_tracking_requested": bool(
            result.pinned_geometric_tracking_requested
        ),
        "pinned_geometric_tracking_applied": bool(
            result.pinned_geometric_tracking_applied
        ),
        "averaging_notes": [str(note) for note in result.averaging_notes],
        "contiguous_frame_sets": [
            entry.to_dict() for entry in result.contiguous_frame_sets
        ],
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
        "density_profile_metadata": {
            "displayed_raw_density_method": "finite_radius_shell_overlap",
            "displayed_raw_density_field": (
                "orientation_average_density_e_per_a3"
            ),
            "shell_electron_count_field": "electrons_in_shell",
            "shell_electron_count_role": "point_tag_bookkeeping",
            "fourier_transform_density_source": (
                "smeared_orientation_average_density_e_per_a3 "
                "(or solvent-subtracted smeared density when enabled)"
            ),
            "interpretation_note": (
                "Displayed raw density is derived from finite-radius shell "
                "overlap using element-specific effective atomic radii, "
                "while shell electron counts remain point-tag bookkeeping "
                "and Gaussian smearing applies a shell-volume-balanced "
                "broadening kernel. Fourier/scattering outputs are built "
                "from the smeared profile."
            ),
        },
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
    "ElectronDensityCalculationCanceled",
    "ElectronDensityContiguousFrameSetSummary",
    "ElectronDensityDebyeScatteringAverageResult",
    "ElectronDensityDebyeWallerPairTerm",
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
    "heaviest_reference_element",
    "compute_average_debye_scattering_profile_for_input",
    "compute_electron_density_profile",
    "compute_electron_density_profile_for_input",
    "compute_electron_density_scattering_profile",
    "compute_single_atom_debye_scattering_profile_for_input",
    "inspect_structure_input",
    "load_electron_density_structure",
    "prepare_single_atom_debye_scattering_preview",
    "prepare_electron_density_fourier_transform",
    "recenter_electron_density_structure",
    "suggest_output_basename",
    "suggest_output_dir",
    "write_electron_density_profile_outputs",
]
