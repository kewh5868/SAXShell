from __future__ import annotations

import json
import math
import shutil
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from saxshell.cluster.clusternetwork import (
    detect_frame_folder_mode,
    detect_source_box_dimensions,
    estimate_box_dimensions_from_coordinates,
)
from saxshell.saxs.debye.profiles import load_structure_file
from saxshell.saxs.project_manager import build_project_paths

DEBYER_DOCS_URL = "https://debyer.readthedocs.io/en/latest/"
DEBYER_GITHUB_URL = "https://github.com/wojdyr/debyer"
TOTAL_SCATTERING_PAPER_URL = (
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC7941302/"
)
SUPPORTED_DEBYER_MODES = ("PDF", "RDF", "rPDF")
SUPPORTED_PLOT_REPRESENTATIONS = ("g(r)", "G(r)", "R(r)")
DEFAULT_COLOR_SCHEMES = ("tab20", "tab10", "viridis", "plasma", "summer")
_COLUMN_PREFIX = "# columns:"
_TIME_PREDICTION_UPDATE_INTERVAL_FRAMES = 5


@dataclass(slots=True, frozen=True)
class DebyerRuntimeStatus:
    executable_path: Path | None
    available: bool
    runnable: bool
    permission_granted: bool
    message: str


@dataclass(slots=True, frozen=True)
class DebyerFrameInspection:
    frames_dir: Path
    frame_format: str
    frame_paths: tuple[Path, ...]
    detected_box_dimensions: tuple[float, float, float] | None
    detected_box_source: str | None
    detected_box_source_kind: str | None
    estimated_box_dimensions: tuple[float, float, float] | None
    atom_count: int
    element_counts: dict[str, int]


@dataclass(slots=True, frozen=True)
class DebyerPDFSettings:
    project_dir: Path
    frames_dir: Path
    filename_prefix: str
    mode: str = "PDF"
    from_value: float = 0.5
    to_value: float = 15.0
    step_value: float = 0.01
    box_dimensions: tuple[float, float, float] = (0.0, 0.0, 0.0)
    atom_count: int = 0
    store_frame_outputs: bool = False
    solute_elements: tuple[str, ...] = ()


@dataclass(slots=True, frozen=True)
class DebyerPeakFinderSettings:
    min_relative_height: float = 0.12
    min_spacing_angstrom: float = 0.35
    max_peak_count: int = 6


@dataclass(slots=True, frozen=True)
class DebyerPeakMarker:
    r_value: float
    label: str
    enabled: bool = True
    text_dx: float = 0.0
    text_dy: float = 0.0
    source: str = "auto"


@dataclass(slots=True, frozen=True)
class DebyerPDFCalculationSummary:
    calculation_id: str
    calculation_dir: Path
    created_at: str
    filename_prefix: str
    mode: str
    frame_count: int
    frames_dir: Path


@dataclass(slots=True, frozen=True)
class DebyerPDFCalculation:
    calculation_id: str
    calculation_dir: Path
    created_at: str
    project_dir: Path
    frames_dir: Path
    frame_format: str
    frame_count: int
    filename_prefix: str
    mode: str
    from_value: float
    to_value: float
    step_value: float
    box_dimensions: tuple[float, float, float]
    box_source: str | None
    box_source_kind: str | None
    atom_count: int
    rho0: float
    store_frame_outputs: bool
    frame_output_dir: Path | None
    averaged_output_file: Path
    solute_elements: tuple[str, ...]
    r_values: np.ndarray
    total_values: np.ndarray
    partial_values: dict[str, np.ndarray]
    processed_frame_count: int | None = None
    is_partial_average: bool = False
    elapsed_seconds: float | None = None
    estimated_remaining_seconds: float | None = None
    expected_total_seconds: float | None = None
    partial_peak_markers: dict[str, tuple[DebyerPeakMarker, ...]] = field(
        default_factory=dict
    )
    target_peak_markers: dict[str, dict[str, tuple[DebyerPeakMarker, ...]]] = (
        field(default_factory=dict)
    )
    peak_finder_settings: DebyerPeakFinderSettings = field(
        default_factory=DebyerPeakFinderSettings
    )


def _normalized_element(value: str) -> str:
    text = str(value).strip()
    if not text:
        return ""
    return text[:1].upper() + text[1:].lower()


def _normalize_solute_elements(
    values: list[str] | tuple[str, ...] | set[str] | None,
) -> tuple[str, ...]:
    if not values:
        return ()
    normalized = {
        _normalized_element(value)
        for value in values
        if _normalized_element(value)
    }
    return tuple(sorted(normalized))


def _sanitize_prefix(value: str) -> str:
    text = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in value.strip()
    ).strip("_")
    return text or "debyer_pdf"


def _coerce_peak_finder_settings(
    payload: dict[str, Any] | None,
) -> DebyerPeakFinderSettings:
    if not payload:
        return DebyerPeakFinderSettings()
    return DebyerPeakFinderSettings(
        min_relative_height=max(
            float(payload.get("min_relative_height", 0.12)),
            0.0,
        ),
        min_spacing_angstrom=max(
            float(payload.get("min_spacing_angstrom", 0.35)),
            0.0,
        ),
        max_peak_count=max(int(payload.get("max_peak_count", 6)), 0),
    )


def _serialize_peak_finder_settings(
    settings: DebyerPeakFinderSettings,
) -> dict[str, Any]:
    return {
        "min_relative_height": float(settings.min_relative_height),
        "min_spacing_angstrom": float(settings.min_spacing_angstrom),
        "max_peak_count": int(settings.max_peak_count),
    }


def _default_peak_label(pair_label: str, r_value: float) -> str:
    return f"{pair_label}: {float(r_value):.2f} A"


def _serialize_peak_markers(
    markers: dict[str, tuple[DebyerPeakMarker, ...]],
) -> dict[str, list[dict[str, Any]]]:
    return {
        pair_label: [
            {
                "r_value": float(marker.r_value),
                "label": str(marker.label),
                "enabled": bool(marker.enabled),
                "text_dx": float(marker.text_dx),
                "text_dy": float(marker.text_dy),
                "source": str(marker.source),
            }
            for marker in pair_markers
        ]
        for pair_label, pair_markers in sorted(markers.items())
    }


def _deserialize_peak_markers(
    payload: dict[str, Any] | None,
) -> dict[str, tuple[DebyerPeakMarker, ...]]:
    if not payload:
        return {}
    resolved: dict[str, tuple[DebyerPeakMarker, ...]] = {}
    for pair_label, entries in payload.items():
        if not isinstance(entries, list):
            continue
        markers: list[DebyerPeakMarker] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            try:
                r_value = float(entry.get("r_value"))
            except (TypeError, ValueError):
                continue
            markers.append(
                DebyerPeakMarker(
                    r_value=r_value,
                    label=str(
                        entry.get(
                            "label",
                            _default_peak_label(str(pair_label), r_value),
                        )
                    ),
                    enabled=bool(entry.get("enabled", True)),
                    text_dx=float(entry.get("text_dx", 0.0)),
                    text_dy=float(entry.get("text_dy", 0.0)),
                    source=str(entry.get("source", "manual")),
                )
            )
        resolved[str(pair_label)] = tuple(
            sorted(markers, key=lambda marker: marker.r_value)
        )
    return resolved


def _serialize_target_peak_markers(
    payload: dict[str, dict[str, tuple[DebyerPeakMarker, ...]]],
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    return {
        str(target_trace_key): _serialize_peak_markers(pair_markers)
        for target_trace_key, pair_markers in sorted(payload.items())
    }


def _deserialize_target_peak_markers(
    payload: dict[str, Any] | None,
) -> dict[str, dict[str, tuple[DebyerPeakMarker, ...]]]:
    if not payload:
        return {}
    resolved: dict[str, dict[str, tuple[DebyerPeakMarker, ...]]] = {}
    for target_trace_key, pair_payload in payload.items():
        if not isinstance(pair_payload, dict):
            continue
        resolved[str(target_trace_key)] = _deserialize_peak_markers(
            pair_payload
        )
    return resolved


def build_debyer_project_dir(project_dir: str | Path) -> Path:
    return (
        build_project_paths(project_dir).exported_data_dir
        / "debyer"
        / "saved_calculations"
    )


def _build_calculation_id(prefix: str) -> str:
    stamp = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_{_sanitize_prefix(prefix)}"


def calculate_number_density(
    atom_count: int,
    box_dimensions: tuple[float, float, float],
) -> float:
    volume = float(np.prod(np.asarray(box_dimensions, dtype=float)))
    if volume <= 0.0:
        raise ValueError(
            "The bounding-box volume must be positive to calculate rho0."
        )
    if int(atom_count) <= 0:
        raise ValueError("The atom count must be positive to calculate rho0.")
    return float(atom_count) / volume


def check_debyer_runtime(
    executable: str | Path | None = None,
    *,
    timeout_seconds: float = 3.0,
) -> DebyerRuntimeStatus:
    resolved = None if executable is None else Path(executable).expanduser()
    if resolved is None:
        discovered = shutil.which("debyer")
        if discovered:
            resolved = Path(discovered)

    if resolved is None:
        return DebyerRuntimeStatus(
            executable_path=None,
            available=False,
            runnable=False,
            permission_granted=False,
            message=(
                "Debyer was not found on PATH. Install Debyer and make sure "
                "the 'debyer' executable is available before running PDF "
                "calculations."
            ),
        )

    try:
        completed = subprocess.run(
            [str(resolved), "--help"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=max(float(timeout_seconds), 0.5),
        )
    except PermissionError:
        return DebyerRuntimeStatus(
            executable_path=resolved,
            available=True,
            runnable=False,
            permission_granted=False,
            message=(
                "Debyer was found but SAXSShell could not execute it. Check "
                "the executable permissions and any OS-level subprocess "
                "approval settings."
            ),
        )
    except OSError as exc:
        return DebyerRuntimeStatus(
            executable_path=resolved,
            available=True,
            runnable=False,
            permission_granted=False,
            message=f"Debyer was found but could not be started: {exc}",
        )
    except subprocess.TimeoutExpired:
        return DebyerRuntimeStatus(
            executable_path=resolved,
            available=True,
            runnable=False,
            permission_granted=False,
            message=(
                "Debyer was found, but the quick startup check timed out. "
                "Verify that Debyer can be launched manually from a terminal."
            ),
        )

    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        return DebyerRuntimeStatus(
            executable_path=resolved,
            available=True,
            runnable=False,
            permission_granted=False,
            message=(
                "Debyer was found, but the quick '--help' check failed"
                + (f": {detail}" if detail else ".")
            ),
        )

    return DebyerRuntimeStatus(
        executable_path=resolved,
        available=True,
        runnable=True,
        permission_granted=True,
        message=f"Debyer is available at {resolved}",
    )


def inspect_frames_dir(frames_dir: str | Path) -> DebyerFrameInspection:
    resolved_frames_dir = Path(frames_dir).expanduser().resolve()
    frame_format, frame_paths = detect_frame_folder_mode(resolved_frames_dir)
    first_frame = frame_paths[0]

    detected_box_dimensions: tuple[float, float, float] | None = None
    detected_box_source: str | None = None
    detected_box_source_kind: str | None = None
    if frame_format == "xyz":
        detected = detect_source_box_dimensions(resolved_frames_dir)
        if detected is not None:
            detected_box_dimensions, source_path = detected
            detected_box_source = source_path.name
            detected_box_source_kind = "source_filename"

    coordinates, elements = load_structure_file(first_frame)
    estimated_box_dimensions = estimate_box_dimensions_from_coordinates(
        coordinates
    )
    element_counts: dict[str, int] = {}
    for element in elements:
        normalized = _normalized_element(element)
        element_counts[normalized] = element_counts.get(normalized, 0) + 1

    return DebyerFrameInspection(
        frames_dir=resolved_frames_dir,
        frame_format=str(frame_format),
        frame_paths=tuple(frame_paths),
        detected_box_dimensions=detected_box_dimensions,
        detected_box_source=detected_box_source,
        detected_box_source_kind=detected_box_source_kind,
        estimated_box_dimensions=estimated_box_dimensions,
        atom_count=len(elements),
        element_counts=element_counts,
    )


def _parse_columns_from_comments(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped.startswith(_COLUMN_PREFIX):
                payload = stripped[len(_COLUMN_PREFIX) :].strip()
                return [token for token in payload.split() if token]
            if stripped.startswith("# sum"):
                return stripped[1:].split()
    return ["sum"]


def parse_debyer_output_file(
    path: str | Path,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    resolved = Path(path).expanduser().resolve()
    columns = _parse_columns_from_comments(resolved)
    raw = np.loadtxt(resolved, comments="#")
    if raw.size == 0:
        raise ValueError(f"Debyer output file is empty: {resolved}")
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    if raw.shape[1] < 2:
        raise ValueError(
            "Debyer output must contain the radial grid and at least one "
            f"distribution column: {resolved}"
        )
    r_values = np.asarray(raw[:, 0], dtype=float)
    value_columns = raw[:, 1:]
    if len(columns) != value_columns.shape[1]:
        columns = ["sum"] + [
            f"partial_{index:02d}"
            for index in range(1, value_columns.shape[1])
        ]
    values = {
        str(column): np.asarray(value_columns[:, index], dtype=float)
        for index, column in enumerate(columns)
    }
    if "sum" not in values and values:
        first_key = next(iter(values))
        values["sum"] = np.asarray(values[first_key], dtype=float)
    return r_values, values


def save_averaged_debyer_output(
    output_path: str | Path,
    *,
    r_values: np.ndarray,
    column_order: list[str],
    values: dict[str, np.ndarray],
    metadata: dict[str, object] | None = None,
) -> None:
    resolved = Path(output_path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    ordered_columns = [name for name in column_order if name in values]
    data = np.column_stack(
        [np.asarray(r_values, dtype=float)]
        + [np.asarray(values[name], dtype=float) for name in ordered_columns]
    )
    lines: list[str] = []
    if metadata:
        for key, value in metadata.items():
            lines.append(f"# {key}: {value}")
    lines.append(f"{_COLUMN_PREFIX} {' '.join(ordered_columns)}")
    np.savetxt(
        resolved,
        data,
        header="\n".join(lines),
        comments="",
    )


def _format_duration(seconds: float | None) -> str:
    if seconds is None or not np.isfinite(float(seconds)):
        return "unknown"
    rounded = max(int(round(float(seconds))), 0)
    hours, remainder = divmod(rounded, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _time_prediction_interval(total_frames: int) -> int:
    if total_frames <= 10:
        return 1
    return min(
        _TIME_PREDICTION_UPDATE_INTERVAL_FRAMES, max(total_frames // 10, 1)
    )


def _estimate_runtime(
    *,
    processed_frames: int,
    total_frames: int,
    elapsed_seconds: float,
) -> tuple[float | None, float | None]:
    if processed_frames <= 0:
        return None, None
    mean_seconds_per_frame = float(elapsed_seconds) / float(processed_frames)
    remaining_frames = max(int(total_frames) - int(processed_frames), 0)
    remaining_seconds = mean_seconds_per_frame * float(remaining_frames)
    expected_total_seconds = float(elapsed_seconds) + remaining_seconds
    return remaining_seconds, expected_total_seconds


def _build_averaged_output_metadata(
    *,
    calculation_id: str,
    created_at: str,
    settings: DebyerPDFSettings,
    inspection: DebyerFrameInspection,
    rho0: float,
    processed_frames: int,
    total_frames: int,
    elapsed_seconds: float | None,
    estimated_remaining_seconds: float | None,
    expected_total_seconds: float | None,
) -> dict[str, object]:
    return {
        "calculation_id": calculation_id,
        "created_at": created_at,
        "filename_prefix": settings.filename_prefix,
        "frames_dir": str(settings.frames_dir),
        "frame_format": inspection.frame_format,
        "processed_frames": int(processed_frames),
        "total_frames": int(total_frames),
        "mode": settings.mode,
        "from_value": settings.from_value,
        "to_value": settings.to_value,
        "step_value": settings.step_value,
        "box_dimensions": ", ".join(
            f"{component:.6g}" for component in settings.box_dimensions
        ),
        "box_source": inspection.detected_box_source or "estimated/manual",
        "box_source_kind": inspection.detected_box_source_kind or "estimate",
        "atom_count": settings.atom_count,
        "rho0": f"{rho0:.8g}",
        "store_frame_outputs": settings.store_frame_outputs,
        "solute_elements": ", ".join(settings.solute_elements) or "None",
        "elapsed_seconds": (
            None
            if elapsed_seconds is None
            else f"{float(elapsed_seconds):.6f}"
        ),
        "estimated_remaining_seconds": (
            None
            if estimated_remaining_seconds is None
            else f"{float(estimated_remaining_seconds):.6f}"
        ),
        "expected_total_seconds": (
            None
            if expected_total_seconds is None
            else f"{float(expected_total_seconds):.6f}"
        ),
        "elapsed_hms": _format_duration(elapsed_seconds),
        "remaining_hms": _format_duration(estimated_remaining_seconds),
        "expected_total_hms": _format_duration(expected_total_seconds),
    }


def _average_frame_outputs(
    outputs: list[tuple[np.ndarray, dict[str, np.ndarray]]],
) -> tuple[np.ndarray, list[str], dict[str, np.ndarray]]:
    if not outputs:
        raise ValueError(
            "No Debyer frame outputs were provided for averaging."
        )

    reference_r = np.asarray(outputs[0][0], dtype=float)
    union_columns: list[str] = ["sum"]
    seen = {"sum"}
    for _r_values, columns in outputs:
        for key in columns:
            if key not in seen:
                seen.add(key)
                union_columns.append(key)

    stacked: dict[str, list[np.ndarray]] = {key: [] for key in union_columns}
    for r_values, columns in outputs:
        if not np.allclose(reference_r, np.asarray(r_values, dtype=float)):
            raise ValueError(
                "Debyer frame outputs do not share the same radial grid."
            )
        for key in union_columns:
            if key in columns:
                stacked[key].append(np.asarray(columns[key], dtype=float))
            else:
                stacked[key].append(np.zeros_like(reference_r, dtype=float))

    averaged = {
        key: np.mean(np.vstack(series), axis=0)
        for key, series in stacked.items()
    }
    return reference_r, union_columns, averaged


def _candidate_peak_indices(values: np.ndarray) -> list[int]:
    array = np.asarray(values, dtype=float)
    count = int(array.size)
    if count == 0:
        return []
    if count == 1:
        return [0]
    candidate_indices: list[int] = []
    for index in range(count):
        center = float(array[index])
        if not np.isfinite(center):
            continue
        left = float(array[index - 1]) if index > 0 else -math.inf
        right = float(array[index + 1]) if index < (count - 1) else -math.inf
        if index == 0:
            is_peak = center > right
        elif index == (count - 1):
            is_peak = center > left
        else:
            is_peak = center >= left and center > right
        if is_peak:
            candidate_indices.append(index)
    return candidate_indices


def find_partial_peak_markers(
    *,
    pair_label: str,
    r_values: np.ndarray,
    values: np.ndarray,
    settings: DebyerPeakFinderSettings,
) -> tuple[DebyerPeakMarker, ...]:
    radial = np.asarray(r_values, dtype=float)
    signal = np.asarray(values, dtype=float)
    if radial.size == 0 or signal.size == 0:
        return ()
    max_value = float(np.nanmax(signal))
    if not np.isfinite(max_value) or max_value <= 0.0:
        return ()
    min_height = float(settings.min_relative_height) * max_value
    candidate_indices = [
        index
        for index in _candidate_peak_indices(signal)
        if float(signal[index]) >= min_height
    ]
    if not candidate_indices:
        return ()

    min_spacing = max(float(settings.min_spacing_angstrom), 0.0)
    max_peak_count = max(int(settings.max_peak_count), 0)
    selected: list[int] = []
    for index in sorted(
        candidate_indices,
        key=lambda candidate: float(signal[candidate]),
        reverse=True,
    ):
        peak_position = float(radial[index])
        if any(
            abs(peak_position - float(radial[chosen])) < min_spacing
            for chosen in selected
        ):
            continue
        selected.append(index)
        if max_peak_count and len(selected) >= max_peak_count:
            break

    selected.sort(key=lambda index: float(radial[index]))
    radial_span = (
        max(float(radial[-1]) - float(radial[0]), 0.0)
        if radial.size > 1
        else 0.0
    )
    default_dx = max(radial_span * 0.02, min_spacing * 0.5, 0.05)
    return tuple(
        DebyerPeakMarker(
            r_value=float(radial[index]),
            label=_default_peak_label(pair_label, float(radial[index])),
            enabled=True,
            text_dx=default_dx,
            text_dy=0.0,
            source="auto",
        )
        for index in selected
    )


def estimate_partial_peak_markers(
    *,
    r_values: np.ndarray,
    partial_values: dict[str, np.ndarray],
    settings: DebyerPeakFinderSettings,
) -> dict[str, tuple[DebyerPeakMarker, ...]]:
    return {
        pair_label: find_partial_peak_markers(
            pair_label=pair_label,
            r_values=r_values,
            values=np.asarray(values, dtype=float),
            settings=settings,
        )
        for pair_label, values in sorted(partial_values.items())
    }


def build_debyer_calculation_metadata(
    calculation: DebyerPDFCalculation,
) -> dict[str, Any]:
    return {
        "calculation_id": calculation.calculation_id,
        "created_at": calculation.created_at,
        "project_dir": str(calculation.project_dir),
        "frames_dir": str(calculation.frames_dir),
        "frame_format": calculation.frame_format,
        "frame_count": int(calculation.frame_count),
        "processed_frame_count": int(
            calculation.frame_count
            if calculation.processed_frame_count is None
            else calculation.processed_frame_count
        ),
        "is_partial_average": bool(calculation.is_partial_average),
        "filename_prefix": calculation.filename_prefix,
        "mode": calculation.mode,
        "from_value": float(calculation.from_value),
        "to_value": float(calculation.to_value),
        "step_value": float(calculation.step_value),
        "box_dimensions": [
            float(component) for component in calculation.box_dimensions
        ],
        "box_source": calculation.box_source,
        "box_source_kind": calculation.box_source_kind,
        "atom_count": int(calculation.atom_count),
        "rho0": float(calculation.rho0),
        "store_frame_outputs": bool(calculation.store_frame_outputs),
        "frame_output_dir": (
            None
            if calculation.frame_output_dir is None
            else str(calculation.frame_output_dir)
        ),
        "averaged_output_file": str(calculation.averaged_output_file),
        "solute_elements": list(calculation.solute_elements),
        "elapsed_seconds": calculation.elapsed_seconds,
        "estimated_remaining_seconds": calculation.estimated_remaining_seconds,
        "expected_total_seconds": calculation.expected_total_seconds,
        "peak_finder_settings": _serialize_peak_finder_settings(
            calculation.peak_finder_settings
        ),
        "partial_peak_markers": _serialize_peak_markers(
            calculation.partial_peak_markers
        ),
        "target_peak_markers": _serialize_target_peak_markers(
            calculation.target_peak_markers
        ),
    }


def write_debyer_calculation_metadata(
    calculation: DebyerPDFCalculation,
) -> None:
    (calculation.calculation_dir / "calculation.json").write_text(
        json.dumps(build_debyer_calculation_metadata(calculation), indent=2)
        + "\n",
        encoding="utf-8",
    )


def _safe_divide(
    numerator: np.ndarray,
    denominator: np.ndarray,
    *,
    fill_value: float = 0.0,
) -> np.ndarray:
    result = np.full_like(np.asarray(numerator, dtype=float), fill_value)
    valid = np.abs(np.asarray(denominator, dtype=float)) > 1.0e-12
    result[valid] = (
        np.asarray(numerator, dtype=float)[valid]
        / np.asarray(denominator, dtype=float)[valid]
    )
    return result


def convert_distribution_values(
    values: np.ndarray,
    *,
    r_values: np.ndarray,
    rho0: float,
    source_mode: str,
    target_representation: str,
    is_component: bool = False,
) -> np.ndarray:
    normalized_source = str(source_mode).strip()
    normalized_target = str(target_representation).strip()
    if normalized_source not in SUPPORTED_DEBYER_MODES:
        raise ValueError(f"Unsupported Debyer source mode: {source_mode}")
    if normalized_target not in SUPPORTED_PLOT_REPRESENTATIONS:
        raise ValueError(
            f"Unsupported PDF target representation: {target_representation}"
        )

    radial = np.asarray(r_values, dtype=float)
    values_array = np.asarray(values, dtype=float)
    prefactor_r = 4.0 * math.pi * float(rho0) * radial
    prefactor_r2 = prefactor_r * radial

    if normalized_source == "PDF":
        canonical_g = values_array
    elif normalized_source == "RDF":
        canonical_g = _safe_divide(values_array, prefactor_r2)
    else:
        canonical_g = _safe_divide(values_array, prefactor_r)
        if not is_component:
            canonical_g = canonical_g + 1.0

    if normalized_target == "g(r)":
        return canonical_g
    if normalized_target == "R(r)":
        return prefactor_r2 * canonical_g

    if is_component:
        return prefactor_r * canonical_g
    return prefactor_r * (canonical_g - 1.0)


def classify_partial_pair(
    pair_label: str,
    *,
    solute_elements: set[str] | None = None,
) -> str | None:
    if not solute_elements or "-" not in pair_label:
        return None
    left, right = pair_label.split("-", 1)
    first = _normalized_element(left)
    second = _normalized_element(right)
    first_is_solute = first in solute_elements
    second_is_solute = second in solute_elements
    if first_is_solute and second_is_solute:
        return "solute-solute"
    if not first_is_solute and not second_is_solute:
        return "solvent-solvent"
    return "solute-solvent"


def build_grouped_partial_values(
    partial_values: dict[str, np.ndarray],
    *,
    solute_elements: tuple[str, ...] = (),
) -> dict[str, np.ndarray]:
    if not solute_elements:
        return {}
    normalized_solutes = set(_normalize_solute_elements(solute_elements))
    grouped: dict[str, np.ndarray] = {}
    for pair_label, values in partial_values.items():
        family = classify_partial_pair(
            pair_label,
            solute_elements=normalized_solutes,
        )
        if family is None:
            continue
        current = grouped.get(family)
        if current is None:
            grouped[family] = np.asarray(values, dtype=float).copy()
        else:
            grouped[family] = current + np.asarray(values, dtype=float)
    return grouped


def build_display_traces(
    calculation: DebyerPDFCalculation,
    *,
    representation: str = "g(r)",
    include_grouped_partials: bool = True,
) -> list[dict[str, Any]]:
    traces: list[dict[str, Any]] = [
        {
            "key": "average",
            "label": "Average",
            "kind": "average",
            "values": convert_distribution_values(
                calculation.total_values,
                r_values=calculation.r_values,
                rho0=calculation.rho0,
                source_mode=calculation.mode,
                target_representation=representation,
                is_component=False,
            ),
        }
    ]
    for pair_label, values in sorted(calculation.partial_values.items()):
        traces.append(
            {
                "key": f"partial:{pair_label}",
                "label": pair_label,
                "kind": "partial",
                "values": convert_distribution_values(
                    values,
                    r_values=calculation.r_values,
                    rho0=calculation.rho0,
                    source_mode=calculation.mode,
                    target_representation=representation,
                    is_component=True,
                ),
            }
        )
    if include_grouped_partials:
        grouped = build_grouped_partial_values(
            calculation.partial_values,
            solute_elements=calculation.solute_elements,
        )
        for family, values in sorted(grouped.items()):
            traces.append(
                {
                    "key": f"group:{family}",
                    "label": family,
                    "kind": "group",
                    "values": convert_distribution_values(
                        values,
                        r_values=calculation.r_values,
                        rho0=calculation.rho0,
                        source_mode=calculation.mode,
                        target_representation=representation,
                        is_component=True,
                    ),
                }
            )
    return traces


def load_debyer_calculation(
    calculation_dir: str | Path,
) -> DebyerPDFCalculation:
    resolved_dir = Path(calculation_dir).expanduser().resolve()
    metadata_file = resolved_dir / "calculation.json"
    if not metadata_file.is_file():
        raise FileNotFoundError(
            f"The Debyer calculation metadata file is missing: {metadata_file}"
        )
    payload = json.loads(metadata_file.read_text(encoding="utf-8"))
    averaged_output_file = Path(payload["averaged_output_file"]).resolve()
    r_values, raw_values = parse_debyer_output_file(averaged_output_file)
    total_values = np.asarray(raw_values.pop("sum"), dtype=float)
    partial_values = {
        key: np.asarray(value, dtype=float)
        for key, value in raw_values.items()
    }
    peak_finder_settings = _coerce_peak_finder_settings(
        payload.get("peak_finder_settings")
    )
    stored_peak_markers = _deserialize_peak_markers(
        payload.get("partial_peak_markers")
    )
    stored_target_peak_markers = _deserialize_target_peak_markers(
        payload.get("target_peak_markers")
    )
    estimated_peak_markers = estimate_partial_peak_markers(
        r_values=r_values,
        partial_values=partial_values,
        settings=peak_finder_settings,
    )
    resolved_peak_markers: dict[str, tuple[DebyerPeakMarker, ...]] = {}
    needs_metadata_refresh = False
    for pair_label in sorted(partial_values):
        pair_markers = stored_peak_markers.get(pair_label)
        if pair_markers is None:
            pair_markers = estimated_peak_markers.get(pair_label, ())
            needs_metadata_refresh = True
        resolved_peak_markers[pair_label] = tuple(pair_markers)
    if "peak_finder_settings" not in payload:
        needs_metadata_refresh = True
    frame_output_dir_value = payload.get("frame_output_dir")
    calculation = DebyerPDFCalculation(
        calculation_id=str(payload["calculation_id"]),
        calculation_dir=resolved_dir,
        created_at=str(payload["created_at"]),
        project_dir=Path(payload["project_dir"]).resolve(),
        frames_dir=Path(payload["frames_dir"]).resolve(),
        frame_format=str(payload["frame_format"]),
        frame_count=int(payload["frame_count"]),
        filename_prefix=str(payload["filename_prefix"]),
        mode=str(payload["mode"]),
        from_value=float(payload["from_value"]),
        to_value=float(payload["to_value"]),
        step_value=float(payload["step_value"]),
        box_dimensions=tuple(
            float(component)
            for component in payload.get("box_dimensions", (0.0, 0.0, 0.0))
        ),
        box_source=payload.get("box_source"),
        box_source_kind=payload.get("box_source_kind"),
        atom_count=int(payload["atom_count"]),
        rho0=float(payload["rho0"]),
        store_frame_outputs=bool(payload.get("store_frame_outputs", False)),
        frame_output_dir=(
            None
            if not frame_output_dir_value
            else Path(frame_output_dir_value).resolve()
        ),
        averaged_output_file=averaged_output_file,
        solute_elements=_normalize_solute_elements(
            payload.get("solute_elements", [])
        ),
        r_values=r_values,
        total_values=total_values,
        partial_values=partial_values,
        processed_frame_count=int(
            payload.get("processed_frame_count", payload["frame_count"])
        ),
        is_partial_average=bool(payload.get("is_partial_average", False)),
        elapsed_seconds=(
            None
            if payload.get("elapsed_seconds") is None
            else float(payload["elapsed_seconds"])
        ),
        estimated_remaining_seconds=(
            None
            if payload.get("estimated_remaining_seconds") is None
            else float(payload["estimated_remaining_seconds"])
        ),
        expected_total_seconds=(
            None
            if payload.get("expected_total_seconds") is None
            else float(payload["expected_total_seconds"])
        ),
        partial_peak_markers=resolved_peak_markers,
        target_peak_markers=stored_target_peak_markers,
        peak_finder_settings=peak_finder_settings,
    )
    if needs_metadata_refresh:
        write_debyer_calculation_metadata(calculation)
    return calculation


def list_saved_debyer_calculations(
    project_dir: str | Path,
) -> list[DebyerPDFCalculationSummary]:
    root_dir = build_debyer_project_dir(project_dir)
    if not root_dir.is_dir():
        return []

    summaries: list[DebyerPDFCalculationSummary] = []
    for candidate in sorted(root_dir.iterdir()):
        metadata_file = candidate / "calculation.json"
        if not metadata_file.is_file():
            continue
        try:
            payload = json.loads(metadata_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        summaries.append(
            DebyerPDFCalculationSummary(
                calculation_id=str(payload["calculation_id"]),
                calculation_dir=candidate.resolve(),
                created_at=str(payload["created_at"]),
                filename_prefix=str(payload["filename_prefix"]),
                mode=str(payload["mode"]),
                frame_count=int(payload["frame_count"]),
                frames_dir=Path(payload["frames_dir"]).resolve(),
            )
        )
    summaries.sort(key=lambda entry: entry.created_at, reverse=True)
    return summaries


class DebyerPDFWorkflow:
    """Run Debyer PDF or partial-PDF calculations across a
    trajectory."""

    def __init__(
        self,
        settings: DebyerPDFSettings,
        *,
        debyer_executable: str | Path | None = None,
    ) -> None:
        if settings.mode not in SUPPORTED_DEBYER_MODES:
            raise ValueError(
                "Debyer mode must be one of "
                + ", ".join(SUPPORTED_DEBYER_MODES)
            )
        self.settings = DebyerPDFSettings(
            project_dir=Path(settings.project_dir).expanduser().resolve(),
            frames_dir=Path(settings.frames_dir).expanduser().resolve(),
            filename_prefix=_sanitize_prefix(settings.filename_prefix),
            mode=str(settings.mode),
            from_value=float(settings.from_value),
            to_value=float(settings.to_value),
            step_value=float(settings.step_value),
            box_dimensions=tuple(
                float(component) for component in settings.box_dimensions
            ),
            atom_count=int(settings.atom_count),
            store_frame_outputs=bool(settings.store_frame_outputs),
            solute_elements=_normalize_solute_elements(
                settings.solute_elements
            ),
        )
        self.debyer_executable = (
            None
            if debyer_executable is None
            else Path(debyer_executable).expanduser().resolve()
        )
        self._cached_runtime_status: DebyerRuntimeStatus | None = None
        self._cached_inspection: DebyerFrameInspection | None = None

    def check_runtime(self) -> DebyerRuntimeStatus:
        if self._cached_runtime_status is None:
            self._cached_runtime_status = check_debyer_runtime(
                self.debyer_executable
            )
        return self._cached_runtime_status

    def inspect_frames(self) -> DebyerFrameInspection:
        if self._cached_inspection is None:
            self._cached_inspection = inspect_frames_dir(
                self.settings.frames_dir
            )
        return self._cached_inspection

    def run(
        self,
        *,
        progress_callback: Callable[[int, int, str], None] | None = None,
        log_callback: Callable[[str], None] | None = None,
        status_callback: Callable[[str], None] | None = None,
        preview_callback: Callable[[DebyerPDFCalculation], None] | None = None,
    ) -> DebyerPDFCalculation:
        runtime_status = self.check_runtime()
        if not runtime_status.runnable:
            raise RuntimeError(runtime_status.message)

        inspection = self.inspect_frames()
        calculation_id = _build_calculation_id(self.settings.filename_prefix)
        created_at = (
            datetime.now(timezone.utc)
            .astimezone()
            .isoformat(timespec="seconds")
        )
        calculation_dir = (
            build_debyer_project_dir(self.settings.project_dir)
            / calculation_id
        )
        calculation_dir.mkdir(parents=True, exist_ok=True)
        frame_output_dir = calculation_dir / "frame_outputs"
        frame_output_dir.mkdir(parents=True, exist_ok=True)
        averaged_output_file = calculation_dir / "averaged_raw.txt"

        rho0 = calculate_number_density(
            self.settings.atom_count,
            self.settings.box_dimensions,
        )
        peak_finder_settings = DebyerPeakFinderSettings()
        total_frames = len(inspection.frame_paths)
        prediction_interval = _time_prediction_interval(total_frames)
        if status_callback is not None:
            status_callback("Running Debyer over trajectory frames")
        if log_callback is not None:
            log_callback(
                "Starting Debyer "
                f"{self.settings.mode} calculation on {total_frames} frames"
            )
            log_callback(
                "Bounding box: "
                + " x ".join(
                    f"{component:.4g}"
                    for component in self.settings.box_dimensions
                )
                + f" A; rho0={rho0:.6g} atoms/A^3"
            )

        averaged_inputs: list[tuple[np.ndarray, dict[str, np.ndarray]]] = []
        start_time = time.monotonic()
        last_verbose_log = time.monotonic()
        latest_preview: DebyerPDFCalculation | None = None
        for index, frame_path in enumerate(inspection.frame_paths, start=1):
            output_path = frame_output_dir / f"{frame_path.stem}.txt"
            command = self._build_command(
                input_file=frame_path,
                output_file=output_path,
                rho0=rho0,
                executable_path=runtime_status.executable_path,
            )
            completed = subprocess.run(
                command,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"Debyer failed on {frame_path.name}: "
                    + (completed.stderr.strip() or completed.stdout.strip())
                )
            averaged_inputs.append(parse_debyer_output_file(output_path))
            elapsed_seconds = time.monotonic() - start_time
            (
                estimated_remaining_seconds,
                expected_total_seconds,
            ) = _estimate_runtime(
                processed_frames=index,
                total_frames=total_frames,
                elapsed_seconds=elapsed_seconds,
            )

            if not self.settings.store_frame_outputs and output_path.exists():
                output_path.unlink()
            if progress_callback is not None:
                progress_message = (
                    f"Processed {index}/{total_frames} frames | "
                    f"elapsed {_format_duration(elapsed_seconds)} | "
                    f"remaining {_format_duration(estimated_remaining_seconds)}"
                )
                progress_callback(
                    index,
                    total_frames,
                    progress_message,
                )
            should_refresh_average = (
                index == 1
                or index == total_frames
                or index % prediction_interval == 0
            )
            if should_refresh_average:
                (
                    preview_r_values,
                    preview_column_order,
                    preview_values,
                ) = _average_frame_outputs(averaged_inputs)
                save_averaged_debyer_output(
                    averaged_output_file,
                    r_values=preview_r_values,
                    column_order=preview_column_order,
                    values=preview_values,
                    metadata=_build_averaged_output_metadata(
                        calculation_id=calculation_id,
                        created_at=created_at,
                        settings=self.settings,
                        inspection=inspection,
                        rho0=rho0,
                        processed_frames=index,
                        total_frames=total_frames,
                        elapsed_seconds=elapsed_seconds,
                        estimated_remaining_seconds=estimated_remaining_seconds,
                        expected_total_seconds=expected_total_seconds,
                    ),
                )
                latest_preview = DebyerPDFCalculation(
                    calculation_id=calculation_id,
                    calculation_dir=calculation_dir,
                    created_at=created_at,
                    project_dir=self.settings.project_dir,
                    frames_dir=self.settings.frames_dir,
                    frame_format=inspection.frame_format,
                    frame_count=total_frames,
                    filename_prefix=self.settings.filename_prefix,
                    mode=self.settings.mode,
                    from_value=self.settings.from_value,
                    to_value=self.settings.to_value,
                    step_value=self.settings.step_value,
                    box_dimensions=self.settings.box_dimensions,
                    box_source=inspection.detected_box_source,
                    box_source_kind=inspection.detected_box_source_kind,
                    atom_count=self.settings.atom_count,
                    rho0=rho0,
                    store_frame_outputs=self.settings.store_frame_outputs,
                    frame_output_dir=frame_output_dir,
                    averaged_output_file=averaged_output_file,
                    solute_elements=self.settings.solute_elements,
                    r_values=preview_r_values,
                    total_values=np.asarray(
                        preview_values["sum"],
                        dtype=float,
                    ),
                    partial_values={
                        key: np.asarray(value, dtype=float)
                        for key, value in preview_values.items()
                        if key != "sum"
                    },
                    processed_frame_count=index,
                    is_partial_average=index < total_frames,
                    elapsed_seconds=elapsed_seconds,
                    estimated_remaining_seconds=estimated_remaining_seconds,
                    expected_total_seconds=expected_total_seconds,
                    partial_peak_markers={},
                    target_peak_markers={},
                    peak_finder_settings=peak_finder_settings,
                )
                if preview_callback is not None:
                    preview_callback(latest_preview)
            if log_callback is not None:
                should_log = (
                    index == 1
                    or index == total_frames
                    or (time.monotonic() - last_verbose_log) >= 5.0
                )
                if should_log:
                    log_callback(
                        f"Processed {index}/{total_frames} frames "
                        f"({frame_path.name}) | elapsed "
                        f"{_format_duration(elapsed_seconds)} | remaining "
                        f"{_format_duration(estimated_remaining_seconds)}"
                    )
                    last_verbose_log = time.monotonic()

        if (
            not self.settings.store_frame_outputs
            and frame_output_dir.is_dir()
            and not any(frame_output_dir.iterdir())
        ):
            frame_output_dir.rmdir()
            stored_frame_output_dir: Path | None = None
        else:
            stored_frame_output_dir = frame_output_dir

        if (
            latest_preview is None
            or latest_preview.processed_frame_count != total_frames
        ):
            elapsed_seconds = time.monotonic() - start_time
            (
                estimated_remaining_seconds,
                expected_total_seconds,
            ) = _estimate_runtime(
                processed_frames=total_frames,
                total_frames=total_frames,
                elapsed_seconds=elapsed_seconds,
            )
            r_values, column_order, averaged_values = _average_frame_outputs(
                averaged_inputs
            )
            save_averaged_debyer_output(
                averaged_output_file,
                r_values=r_values,
                column_order=column_order,
                values=averaged_values,
                metadata=_build_averaged_output_metadata(
                    calculation_id=calculation_id,
                    created_at=created_at,
                    settings=self.settings,
                    inspection=inspection,
                    rho0=rho0,
                    processed_frames=total_frames,
                    total_frames=total_frames,
                    elapsed_seconds=elapsed_seconds,
                    estimated_remaining_seconds=estimated_remaining_seconds,
                    expected_total_seconds=expected_total_seconds,
                ),
            )
        else:
            elapsed_seconds = latest_preview.elapsed_seconds
            estimated_remaining_seconds = (
                latest_preview.estimated_remaining_seconds
            )
            expected_total_seconds = latest_preview.expected_total_seconds

        final_r_values, final_raw_values = parse_debyer_output_file(
            averaged_output_file
        )
        final_total_values = np.asarray(
            final_raw_values.pop("sum"), dtype=float
        )
        final_partial_values = {
            key: np.asarray(value, dtype=float)
            for key, value in final_raw_values.items()
        }
        final_calculation = DebyerPDFCalculation(
            calculation_id=calculation_id,
            calculation_dir=calculation_dir,
            created_at=created_at,
            project_dir=self.settings.project_dir,
            frames_dir=self.settings.frames_dir,
            frame_format=inspection.frame_format,
            frame_count=total_frames,
            filename_prefix=self.settings.filename_prefix,
            mode=self.settings.mode,
            from_value=self.settings.from_value,
            to_value=self.settings.to_value,
            step_value=self.settings.step_value,
            box_dimensions=self.settings.box_dimensions,
            box_source=inspection.detected_box_source,
            box_source_kind=inspection.detected_box_source_kind,
            atom_count=self.settings.atom_count,
            rho0=rho0,
            store_frame_outputs=self.settings.store_frame_outputs,
            frame_output_dir=stored_frame_output_dir,
            averaged_output_file=averaged_output_file,
            solute_elements=self.settings.solute_elements,
            r_values=final_r_values,
            total_values=final_total_values,
            partial_values=final_partial_values,
            processed_frame_count=total_frames,
            is_partial_average=False,
            elapsed_seconds=elapsed_seconds,
            estimated_remaining_seconds=estimated_remaining_seconds,
            expected_total_seconds=expected_total_seconds,
            partial_peak_markers=estimate_partial_peak_markers(
                r_values=final_r_values,
                partial_values=final_partial_values,
                settings=peak_finder_settings,
            ),
            target_peak_markers={},
            peak_finder_settings=peak_finder_settings,
        )
        write_debyer_calculation_metadata(final_calculation)
        if log_callback is not None:
            log_callback(
                f"Saved averaged Debyer output to {averaged_output_file}"
            )
        if status_callback is not None:
            status_callback("Debyer calculation complete")
        return final_calculation

    def _build_command(
        self,
        *,
        input_file: Path,
        output_file: Path,
        rho0: float,
        executable_path: Path | None,
    ) -> list[str]:
        executable = (
            str(executable_path)
            if executable_path is not None
            else str(self.debyer_executable)
        )
        box_a, box_b, box_c = self.settings.box_dimensions
        return [
            executable,
            f"--{self.settings.mode}",
            f"--pbc-a={box_a}",
            f"--pbc-b={box_b}",
            f"--pbc-c={box_c}",
            f"--from={self.settings.from_value}",
            f"--to={self.settings.to_value}",
            f"--step={self.settings.step_value}",
            "--weight=x",
            "--partials",
            f"--ro={rho0}",
            f"--output={output_file}",
            str(input_file),
        ]


__all__ = [
    "DEBYER_DOCS_URL",
    "DEBYER_GITHUB_URL",
    "TOTAL_SCATTERING_PAPER_URL",
    "DEFAULT_COLOR_SCHEMES",
    "DebyerFrameInspection",
    "DebyerPeakFinderSettings",
    "DebyerPeakMarker",
    "DebyerPDFCalculation",
    "DebyerPDFCalculationSummary",
    "DebyerPDFSettings",
    "DebyerPDFWorkflow",
    "DebyerRuntimeStatus",
    "SUPPORTED_DEBYER_MODES",
    "SUPPORTED_PLOT_REPRESENTATIONS",
    "build_debyer_project_dir",
    "build_display_traces",
    "build_grouped_partial_values",
    "calculate_number_density",
    "check_debyer_runtime",
    "classify_partial_pair",
    "convert_distribution_values",
    "estimate_partial_peak_markers",
    "find_partial_peak_markers",
    "inspect_frames_dir",
    "list_saved_debyer_calculations",
    "load_debyer_calculation",
    "parse_debyer_output_file",
    "save_averaged_debyer_output",
    "write_debyer_calculation_metadata",
]
