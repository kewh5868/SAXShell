from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np

from saxshell.saxs.contrast.descriptors import load_parsed_contrast_structure
from saxshell.saxs.contrast.electron_density import (
    ContrastGeometryDensityResult,
)
from saxshell.saxs.contrast.representatives import (
    ContrastRepresentativeSelectionResult,
)
from saxshell.saxs.debye.profiles import (
    build_f0_dictionary,
    compute_debye_intensity,
)
from saxshell.xyz2pdb.workflow import _covalent_radius

ContrastDebyeProgressCallback = Callable[[int, int, str], None]
ContrastDebyeLogCallback = Callable[[str], None]


def _emit_progress(
    callback: ContrastDebyeProgressCallback | None,
    processed: int,
    total: int,
    message: str,
) -> None:
    if callback is None:
        return
    callback(max(int(processed), 0), max(int(total), 1), str(message).strip())


def _emit_log(
    callback: ContrastDebyeLogCallback | None,
    message: str,
) -> None:
    if callback is None:
        return
    text = str(message).strip()
    if text:
        callback(text)


def _now_text() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _component_profile_filename(structure: str, motif: str) -> str:
    safe_name = f"{structure}_{motif}".replace("/", "_")
    return f"{safe_name}.txt"


def _component_map_payload(
    trace_results: tuple["ContrastDebyeTraceResult", ...],
) -> dict[str, dict[str, str]]:
    payload: dict[str, dict[str, str]] = {}
    for trace_result in trace_results:
        payload.setdefault(trace_result.structure, {})
        payload[trace_result.structure][
            trace_result.motif
        ] = trace_result.profile_file
    return payload


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


def _normalized_element_symbol(raw_value: str) -> str:
    text = str(raw_value).strip()
    if not text:
        return ""
    return text[:1].upper() + text[1:].lower()


def _element_excluded_volume_a3(element: str) -> float:
    radius_a = float(_covalent_radius(_normalized_element_symbol(element)))
    return float((4.0 / 3.0) * np.pi * radius_a**3)


def _sphere_form_factor(
    q_values: np.ndarray,
    *,
    radius_a: float,
) -> np.ndarray:
    q_grid = np.asarray(q_values, dtype=float)
    qr_values = q_grid * float(radius_a)
    form_factor = np.ones_like(q_grid, dtype=float)
    nonzero_mask = np.abs(qr_values) > 1.0e-12
    if np.any(nonzero_mask):
        qr_nonzero = qr_values[nonzero_mask]
        form_factor[nonzero_mask] = (
            3.0
            * (np.sin(qr_nonzero) - qr_nonzero * np.cos(qr_nonzero))
            / (qr_nonzero**3)
        )
    return form_factor


def _completed_f0_dictionary(
    elements: list[str] | tuple[str, ...],
    q_values: np.ndarray,
    f0_dictionary: dict[str, np.ndarray] | None,
) -> dict[str, np.ndarray]:
    normalized_elements = sorted(
        {
            _normalized_element_symbol(element)
            for element in elements
            if _normalized_element_symbol(element)
        }
    )
    if f0_dictionary is None:
        return build_f0_dictionary(normalized_elements, q_values)
    merged_dictionary = {
        _normalized_element_symbol(element): np.asarray(values, dtype=float)
        for element, values in f0_dictionary.items()
        if _normalized_element_symbol(element)
    }
    missing_elements = [
        element
        for element in normalized_elements
        if element not in merged_dictionary
    ]
    if missing_elements:
        merged_dictionary.update(
            build_f0_dictionary(missing_elements, q_values)
        )
    return merged_dictionary


def _effective_form_factor_dictionary(
    elements: list[str] | tuple[str, ...],
    q_values: np.ndarray,
    *,
    solvent_density_e_per_a3: float,
    f0_dictionary: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    # Approximate the displaced solvent as an element-sized sphere filled
    # with bulk solvent electron density, then subtract its scattering
    # amplitude from each atomic form factor.
    q_grid = np.asarray(q_values, dtype=float)
    base_f0_dictionary = _completed_f0_dictionary(
        elements,
        q_grid,
        f0_dictionary,
    )
    effective_dictionary: dict[str, np.ndarray] = {}
    for element, values in base_f0_dictionary.items():
        radius_a = float(_covalent_radius(element))
        displaced_electrons = (
            float(solvent_density_e_per_a3)
            * _element_excluded_volume_a3(element)
            * _sphere_form_factor(q_grid, radius_a=radius_a)
        )
        effective_dictionary[element] = (
            np.asarray(values, dtype=float) - displaced_electrons
        )
    return effective_dictionary


def _contrast_scale_factor(
    *,
    elements: list[str] | tuple[str, ...],
    solvent_density_e_per_a3: float,
) -> float:
    q_zero = np.asarray([0.0], dtype=float)
    base_f0_dictionary = _completed_f0_dictionary(
        elements,
        q_zero,
        None,
    )
    denominator = float(
        sum(
            float(base_f0_dictionary[_normalized_element_symbol(element)][0])
            for element in elements
            if _normalized_element_symbol(element)
        )
    )
    if abs(denominator) <= 1e-12:
        return 0.0
    effective_dictionary = _effective_form_factor_dictionary(
        elements,
        q_zero,
        solvent_density_e_per_a3=solvent_density_e_per_a3,
        f0_dictionary=base_f0_dictionary,
    )
    numerator = float(
        sum(
            float(effective_dictionary[_normalized_element_symbol(element)][0])
            for element in elements
            if _normalized_element_symbol(element)
        )
    )
    return float(numerator / denominator)


def compute_contrast_debye_intensity(
    coordinates: np.ndarray,
    elements: list[str] | tuple[str, ...],
    q_values: np.ndarray,
    *,
    cluster_density_e_per_a3: float,
    solvent_density_e_per_a3: float,
    exclude_elements: list[str] | tuple[str, ...] | None = None,
    f0_dictionary: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    q_grid = np.asarray(q_values, dtype=float)
    coordinates_array = np.asarray(coordinates, dtype=float)
    if coordinates_array.size == 0:
        return np.zeros_like(q_grid)
    del cluster_density_e_per_a3  # Retained for API compatibility/metadata.
    normalized_elements = [
        _normalized_element_symbol(element) for element in elements
    ]
    effective_f0_dictionary = _effective_form_factor_dictionary(
        normalized_elements,
        q_grid,
        solvent_density_e_per_a3=float(solvent_density_e_per_a3),
        f0_dictionary=f0_dictionary,
    )
    return np.asarray(
        compute_debye_intensity(
            coordinates_array,
            normalized_elements,
            q_grid,
            exclude_elements=exclude_elements,
            f0_dictionary=effective_f0_dictionary,
        ),
        dtype=float,
    )


@dataclass(slots=True, frozen=True)
class ContrastDebyeTraceResult:
    structure: str
    motif: str
    display_label: str
    file_count: int
    representative_file: Path
    source_dir: Path
    profile_file: str
    profile_path: Path
    mesh_json_path: Path
    density_json_path: Path
    q_values: np.ndarray
    intensity: np.ndarray
    contrast_scale_factor: float
    cluster_density_e_per_a3: float
    solvent_density_e_per_a3: float
    contrast_density_e_per_a3: float
    atom_count: int
    element_counts: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        return {
            "structure": self.structure,
            "motif": self.motif,
            "display_label": self.display_label,
            "file_count": int(self.file_count),
            "representative_file": str(self.representative_file),
            "source_dir": str(self.source_dir),
            "profile_file": self.profile_file,
            "profile_path": str(self.profile_path),
            "mesh_json_path": str(self.mesh_json_path),
            "density_json_path": str(self.density_json_path),
            "contrast_scale_factor": float(self.contrast_scale_factor),
            "cluster_density_e_per_a3": float(self.cluster_density_e_per_a3),
            "solvent_density_e_per_a3": float(self.solvent_density_e_per_a3),
            "contrast_density_e_per_a3": float(self.contrast_density_e_per_a3),
            "atom_count": int(self.atom_count),
            "element_counts": dict(sorted(self.element_counts.items())),
        }


@dataclass(slots=True, frozen=True)
class ContrastDebyeBuildResult:
    output_dir: Path
    metadata_dir: Path
    generated_at: str
    component_map_path: Path
    summary_json_path: Path
    summary_table_path: Path
    summary_text_path: Path
    q_values: np.ndarray
    trace_results: tuple[ContrastDebyeTraceResult, ...]

    def summary_text(self) -> str:
        lines = [
            "Contrast Debye scattering build complete",
            f"Generated at: {self.generated_at}",
            f"Output folder: {self.output_dir}",
            f"Metadata folder: {self.metadata_dir}",
            f"Component map: {self.component_map_path}",
            f"Components built: {len(self.trace_results)}",
        ]
        if self.trace_results:
            lines.extend(["", "Representative traces"])
            for result in self.trace_results:
                lines.append(
                    "  "
                    + f"{result.display_label}: scale={result.contrast_scale_factor:.6f}, "
                    + f"rho_cluster={result.cluster_density_e_per_a3:.6f}, "
                    + f"rho_solvent={result.solvent_density_e_per_a3:.6f}, "
                    + f"profile={result.profile_file}"
                )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        return {
            "version": 1,
            "generated_at": self.generated_at,
            "output_dir": str(self.output_dir),
            "metadata_dir": str(self.metadata_dir),
            "component_map_path": str(self.component_map_path),
            "summary_json_path": str(self.summary_json_path),
            "summary_table_path": str(self.summary_table_path),
            "summary_text_path": str(self.summary_text_path),
            "q_values": [float(value) for value in self.q_values.tolist()],
            "trace_results": [
                result.to_dict() for result in self.trace_results
            ],
        }


def _write_component_file(
    output_path: Path,
    q_values: np.ndarray,
    intensity: np.ndarray,
    *,
    file_count: int,
) -> None:
    header = (
        f"# Number of files: {max(int(file_count), 1)}\n"
        "# Columns: q, S(q)_avg, S(q)_std, S(q)_se\n"
    )
    data = np.column_stack(
        [
            np.asarray(q_values, dtype=float),
            np.asarray(intensity, dtype=float),
            np.zeros_like(q_values, dtype=float),
            np.zeros_like(q_values, dtype=float),
        ]
    )
    np.savetxt(
        output_path,
        data,
        comments="",
        header=header,
        fmt=["%.8f", "%.8f", "%.8f", "%.8f"],
    )


def build_contrast_component_profiles(
    representative_selection: ContrastRepresentativeSelectionResult,
    density_result: ContrastGeometryDensityResult,
    *,
    q_values: np.ndarray,
    output_dir: str | Path,
    metadata_dir: str | Path,
    component_map_path: str | Path,
    exclude_elements: list[str] | tuple[str, ...] | None = None,
    progress_callback: ContrastDebyeProgressCallback | None = None,
    log_callback: ContrastDebyeLogCallback | None = None,
) -> ContrastDebyeBuildResult:
    q_grid = np.asarray(q_values, dtype=float)
    normalized_exclude_elements = _normalized_exclude_elements(
        exclude_elements
    )
    if getattr(density_result.settings, "exclude_elements", ()):
        normalized_exclude_elements = tuple(
            sorted(
                set(normalized_exclude_elements)
                | set(density_result.settings.exclude_elements)
            )
        )
    resolved_output_dir = Path(output_dir).expanduser().resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    resolved_metadata_dir = Path(metadata_dir).expanduser().resolve()
    resolved_metadata_dir.mkdir(parents=True, exist_ok=True)
    resolved_component_map_path = (
        Path(component_map_path).expanduser().resolve()
    )
    resolved_component_map_path.parent.mkdir(parents=True, exist_ok=True)

    representative_by_key = {
        (bin_result.structure, bin_result.motif): bin_result
        for bin_result in representative_selection.bin_results
    }
    density_by_key = {
        (bin_result.structure, bin_result.motif): bin_result
        for bin_result in density_result.bin_results
    }
    missing_density = sorted(set(representative_by_key) - set(density_by_key))
    if missing_density:
        missing_text = ", ".join(
            f"{structure}/{motif}" for structure, motif in missing_density
        )
        raise ValueError(
            "Contrast Debye build requires electron-density outputs for every "
            f"representative bin. Missing: {missing_text}."
        )

    total_bins = len(representative_selection.bin_results)
    unique_elements: list[str] = []
    parsed_cache: dict[tuple[str, str], object] = {}
    for bin_result in representative_selection.bin_results:
        parsed = load_parsed_contrast_structure(
            bin_result.copied_representative_file,
            exclude_elements=normalized_exclude_elements,
        )
        if parsed.atom_count <= 0:
            raise ValueError(
                "All atoms were excluded from "
                f"{bin_result.copied_representative_file.name} before the "
                "contrast Debye build."
            )
        parsed_cache[(bin_result.structure, bin_result.motif)] = parsed
        unique_elements.extend(parsed.elements)
    base_f0_dictionary = build_f0_dictionary(unique_elements, q_grid)

    trace_results: list[ContrastDebyeTraceResult] = []
    for index, representative_bin in enumerate(
        representative_selection.bin_results,
        start=1,
    ):
        density_bin = density_by_key[
            (representative_bin.structure, representative_bin.motif)
        ]
        parsed = parsed_cache[
            (representative_bin.structure, representative_bin.motif)
        ]
        intensity = compute_contrast_debye_intensity(
            parsed.coordinates,
            list(parsed.elements),
            q_grid,
            cluster_density_e_per_a3=density_bin.cluster_electron_density.electron_density_e_per_a3,
            solvent_density_e_per_a3=density_bin.solvent_electron_density.electron_density_e_per_a3,
            exclude_elements=normalized_exclude_elements,
            f0_dictionary=base_f0_dictionary,
        )
        profile_file = _component_profile_filename(
            representative_bin.structure,
            representative_bin.motif,
        )
        profile_path = resolved_output_dir / profile_file
        _write_component_file(
            profile_path,
            q_grid,
            intensity,
            file_count=representative_bin.file_count,
        )
        trace_result = ContrastDebyeTraceResult(
            structure=representative_bin.structure,
            motif=representative_bin.motif,
            display_label=representative_bin.display_label,
            file_count=int(representative_bin.file_count),
            representative_file=representative_bin.copied_representative_file,
            source_dir=representative_bin.source_dir,
            profile_file=profile_file,
            profile_path=profile_path,
            mesh_json_path=density_bin.mesh_json_path,
            density_json_path=density_bin.density_json_path,
            q_values=q_grid.copy(),
            intensity=np.asarray(intensity, dtype=float),
            contrast_scale_factor=_contrast_scale_factor(
                elements=list(parsed.elements),
                solvent_density_e_per_a3=density_bin.solvent_electron_density.electron_density_e_per_a3,
            ),
            cluster_density_e_per_a3=density_bin.cluster_electron_density.electron_density_e_per_a3,
            solvent_density_e_per_a3=density_bin.solvent_electron_density.electron_density_e_per_a3,
            contrast_density_e_per_a3=density_bin.contrast_electron_density_e_per_a3,
            atom_count=parsed.atom_count,
            element_counts=dict(sorted(parsed.element_counts.items())),
        )
        trace_results.append(trace_result)
        _emit_progress(
            progress_callback,
            index,
            total_bins,
            f"Built contrast Debye profile for {trace_result.display_label} ({index}/{total_bins})",
        )
        _emit_log(
            log_callback,
            f"Built contrast Debye profile for {trace_result.display_label} "
            f"with scale {trace_result.contrast_scale_factor:.6f}.",
        )

    result = ContrastDebyeBuildResult(
        output_dir=resolved_output_dir,
        metadata_dir=resolved_metadata_dir,
        generated_at=_now_text(),
        component_map_path=resolved_component_map_path,
        summary_json_path=resolved_metadata_dir / "component_summary.json",
        summary_table_path=resolved_metadata_dir / "component_summary.tsv",
        summary_text_path=resolved_metadata_dir / "component_summary.txt",
        q_values=q_grid.copy(),
        trace_results=tuple(trace_results),
    )
    resolved_component_map_path.write_text(
        json.dumps(
            {"saxs_map": _component_map_payload(result.trace_results)},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
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
                "display_label",
                "file_count",
                "representative_file",
                "profile_file",
                "contrast_scale_factor",
                "cluster_density_e_per_a3",
                "solvent_density_e_per_a3",
                "contrast_density_e_per_a3",
            ]
        )
        for trace_result in result.trace_results:
            writer.writerow(
                [
                    trace_result.structure,
                    trace_result.motif,
                    trace_result.display_label,
                    int(trace_result.file_count),
                    str(trace_result.representative_file),
                    trace_result.profile_file,
                    f"{trace_result.contrast_scale_factor:.8f}",
                    f"{trace_result.cluster_density_e_per_a3:.8f}",
                    f"{trace_result.solvent_density_e_per_a3:.8f}",
                    f"{trace_result.contrast_density_e_per_a3:.8f}",
                ]
            )
    result.summary_text_path.write_text(
        result.summary_text() + "\n",
        encoding="utf-8",
    )
    _emit_log(
        log_callback,
        f"Wrote contrast Debye component outputs to {result.output_dir}.",
    )
    return result


__all__ = [
    "ContrastDebyeBuildResult",
    "ContrastDebyeTraceResult",
    "build_contrast_component_profiles",
    "compute_contrast_debye_intensity",
]
