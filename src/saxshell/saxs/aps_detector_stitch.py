from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

APS_DETECTOR_ORDER: tuple[str, ...] = ("hs104", "hs103", "hs102")
APS_DETECTOR_PATTERN = re.compile(
    r"(?:^|[_\-.])hs(102|103|104)(?:[_\-.]|$)",
    re.I,
)


@dataclass(frozen=True)
class APSDetectorTrace:
    detector: str
    path: Path
    data: np.ndarray

    @property
    def q(self) -> np.ndarray:
        return self.data[:, 0]

    @property
    def intensity(self) -> np.ndarray:
        return self.data[:, 1]

    @property
    def error(self) -> np.ndarray:
        return self.data[:, 2]


@dataclass(frozen=True)
class APSDetectorJoinDiagnostic:
    left_detector: str
    right_detector: str
    q_min: float
    q_max: float
    scale_factor: float
    ratio_point_count: int
    blend_point_count: int
    median_fractional_residual: float | None
    method: str


@dataclass(frozen=True)
class APSDetectorStitchResult:
    stitched_data: np.ndarray
    detector_paths: dict[str, Path]
    scaled_traces: dict[str, np.ndarray]
    joins: tuple[APSDetectorJoinDiagnostic, ...]
    warnings: tuple[str, ...] = ()

    def summary_text(self) -> str:
        lines = [
            "APS Detector Stitch",
            f"Output points: {len(self.stitched_data)}",
        ]
        for detector in APS_DETECTOR_ORDER:
            path = self.detector_paths.get(detector)
            if path is not None:
                lines.append(f"{detector}: {path}")
        for join in self.joins:
            residual = (
                "n/a"
                if join.median_fractional_residual is None
                else f"{join.median_fractional_residual:.3g}"
            )
            lines.append(
                f"{join.left_detector}->{join.right_detector}: "
                f"scale={join.scale_factor:.6g}, "
                f"q={join.q_min:.6g}-{join.q_max:.6g}, "
                f"points={join.ratio_point_count}, residual={residual}, "
                f"method={join.method}"
            )
        if self.warnings:
            lines.append("Warnings:")
            lines.extend(f"- {warning}" for warning in self.warnings)
        return "\n".join(lines)


def detector_name_from_path(path: str | Path) -> str | None:
    match = APS_DETECTOR_PATTERN.search(Path(path).name)
    if match is None:
        return None
    return f"hs{match.group(1).lower()}"


def find_aps_detector_files(
    source: str | Path | Iterable[str | Path],
) -> dict[str, Path]:
    if isinstance(source, str | Path):
        source_path = Path(source).expanduser()
        if source_path.is_dir():
            candidates = [
                path for path in source_path.iterdir() if path.is_file()
            ]
            matched = _match_detector_candidates(candidates)
            if len(matched) == len(APS_DETECTOR_ORDER):
                return matched
            recursive_candidates = [
                path for path in source_path.rglob("*") if path.is_file()
            ]
            return _require_exact_detector_set(
                recursive_candidates,
                source_path,
            )
        if source_path.is_file():
            return _require_exact_detector_set(
                [source_path],
                source_path.parent,
            )
        raise FileNotFoundError(
            f"APS detector input does not exist: {source_path}"
        )

    return _require_exact_detector_set(
        [Path(path).expanduser() for path in source],
        Path.cwd(),
    )


def load_aps_detector_trace(
    path: str | Path,
    detector: str | None = None,
) -> APSDetectorTrace:
    resolved_path = Path(path).expanduser()
    detector_name = detector or detector_name_from_path(resolved_path)
    if detector_name not in APS_DETECTOR_ORDER:
        raise ValueError(
            f"Could not infer APS detector name from {resolved_path.name!r}."
        )
    try:
        raw = np.loadtxt(resolved_path, comments="#", ndmin=2)
    except Exception as exc:
        raise ValueError(f"Could not load {resolved_path}: {exc}") from exc
    if raw.ndim != 2 or raw.shape[1] < 2:
        raise ValueError(
            f"{resolved_path} must contain at least q and intensity columns."
        )
    if raw.shape[1] >= 3:
        data = raw[:, :3].astype(float, copy=True)
    else:
        data = np.column_stack(
            (raw[:, 0], raw[:, 1], np.full(raw.shape[0], np.nan))
        ).astype(float, copy=False)
    data = _clean_detector_data(data, resolved_path)
    return APSDetectorTrace(detector_name, resolved_path, data)


def stitch_aps_detector_files(
    source: str | Path | Iterable[str | Path],
    *,
    min_overlap_points: int = 5,
) -> APSDetectorStitchResult:
    detector_paths = find_aps_detector_files(source)
    traces = {
        detector: load_aps_detector_trace(detector_paths[detector], detector)
        for detector in APS_DETECTOR_ORDER
    }
    return stitch_aps_detector_traces(
        traces,
        min_overlap_points=min_overlap_points,
    )


def stitch_aps_detector_traces(
    traces: dict[str, APSDetectorTrace],
    *,
    min_overlap_points: int = 5,
) -> APSDetectorStitchResult:
    missing = [
        detector for detector in APS_DETECTOR_ORDER if detector not in traces
    ]
    if missing:
        raise ValueError(
            "Missing APS detector trace(s): " + ", ".join(missing)
        )
    scaled_traces: dict[str, np.ndarray] = {
        APS_DETECTOR_ORDER[0]: traces[APS_DETECTOR_ORDER[0]].data.copy()
    }
    detector_paths = {
        detector: traces[detector].path for detector in APS_DETECTOR_ORDER
    }
    stitched = scaled_traces[APS_DETECTOR_ORDER[0]].copy()
    joins: list[APSDetectorJoinDiagnostic] = []
    warnings: list[str] = []

    for left_detector, right_detector in zip(
        APS_DETECTOR_ORDER[:-1],
        APS_DETECTOR_ORDER[1:],
        strict=True,
    ):
        right_data = traces[right_detector].data.copy()
        (
            stitched,
            scaled_right,
            join,
            join_warnings,
        ) = _join_detector_segment(
            stitched,
            right_data,
            left_detector=left_detector,
            right_detector=right_detector,
            min_overlap_points=min_overlap_points,
        )
        scaled_traces[right_detector] = scaled_right
        joins.append(join)
        warnings.extend(join_warnings)

    return APSDetectorStitchResult(
        stitched_data=stitched,
        detector_paths=detector_paths,
        scaled_traces=scaled_traces,
        joins=tuple(joins),
        warnings=tuple(warnings),
    )


def save_aps_stitched_data(
    result: APSDetectorStitchResult,
    destination: str | Path,
) -> Path:
    path = Path(destination).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        path,
        result.stitched_data,
        header=_stitch_header(result),
        comments="# ",
        fmt="%.12g",
        delimiter="\t",
    )
    return path


def _match_detector_candidates(candidates: Iterable[Path]) -> dict[str, Path]:
    matched: dict[str, Path] = {}
    duplicates: dict[str, list[Path]] = {}
    for candidate in candidates:
        detector = detector_name_from_path(candidate)
        if detector is None:
            continue
        if detector in matched:
            duplicates.setdefault(detector, [matched[detector]]).append(
                candidate
            )
        else:
            matched[detector] = candidate
    if duplicates:
        details = "; ".join(
            f"{detector}: {', '.join(str(path) for path in paths)}"
            for detector, paths in sorted(duplicates.items())
        )
        raise ValueError(
            f"Found multiple files for APS detector(s): {details}"
        )
    return matched


def _require_exact_detector_set(
    candidates: Iterable[Path],
    source_label: Path,
) -> dict[str, Path]:
    matched = _match_detector_candidates(candidates)
    missing = [
        detector for detector in APS_DETECTOR_ORDER if detector not in matched
    ]
    if missing:
        raise ValueError(
            f"{source_label} must provide exactly one file for each detector "
            f"({', '.join(APS_DETECTOR_ORDER)}). Missing: {', '.join(missing)}"
        )
    return matched


def _clean_detector_data(data: np.ndarray, path: Path) -> np.ndarray:
    finite_mask = np.isfinite(data[:, 0]) & np.isfinite(data[:, 1])
    finite_mask &= data[:, 0] > 0.0
    cleaned = data[finite_mask]
    if cleaned.size == 0:
        raise ValueError(f"{path} does not contain finite positive-q data.")
    order = np.argsort(cleaned[:, 0], kind="mergesort")
    cleaned = cleaned[order]
    unique_q, first_indices = np.unique(cleaned[:, 0], return_index=True)
    if len(unique_q) != len(cleaned):
        cleaned = cleaned[np.sort(first_indices)]
    return cleaned


def _join_detector_segment(
    stitched: np.ndarray,
    right_data: np.ndarray,
    *,
    left_detector: str,
    right_detector: str,
    min_overlap_points: int,
) -> tuple[np.ndarray, np.ndarray, APSDetectorJoinDiagnostic, tuple[str, ...]]:
    left_q = stitched[:, 0]
    right_q = right_data[:, 0]
    overlap_min = max(float(left_q[0]), float(right_q[0]))
    overlap_max = min(float(left_q[-1]), float(right_q[-1]))
    warnings: list[str] = []

    if overlap_max > overlap_min:
        scale, ratio_count, residual = _overlap_scale_factor(
            stitched,
            right_data,
            overlap_min,
            overlap_max,
            min_overlap_points=min_overlap_points,
        )
        method = "overlap-median"
        q_min = overlap_min
        q_max = overlap_max
        if ratio_count < min_overlap_points:
            scale = _boundary_scale_factor(stitched, right_data)
            residual = None
            method = "boundary"
            warnings.append(
                f"{left_detector}->{right_detector} had only {ratio_count} "
                "usable positive overlap point(s); used endpoint scaling."
            )
    else:
        scale = _boundary_scale_factor(stitched, right_data)
        ratio_count = 0
        residual = None
        method = "boundary"
        q_min = float(left_q[-1])
        q_max = float(right_q[0])
        warnings.append(
            f"{left_detector}->{right_detector} has no q overlap; appended "
            "the scaled detector trace across the gap."
        )

    scaled_right = right_data.copy()
    scaled_right[:, 1] *= scale
    scaled_right[:, 2] = _scale_error_column(scaled_right[:, 2], scale)

    if overlap_max > overlap_min and method == "overlap-median":
        joined, blend_count = _blend_overlap(
            stitched,
            scaled_right,
            overlap_min,
            overlap_max,
        )
    else:
        joined = _append_without_overlap(stitched, scaled_right)
        blend_count = 0

    join = APSDetectorJoinDiagnostic(
        left_detector=left_detector,
        right_detector=right_detector,
        q_min=q_min,
        q_max=q_max,
        scale_factor=float(scale),
        ratio_point_count=int(ratio_count),
        blend_point_count=int(blend_count),
        median_fractional_residual=residual,
        method=method,
    )
    return joined, scaled_right, join, tuple(warnings)


def _overlap_scale_factor(
    left_data: np.ndarray,
    right_data: np.ndarray,
    overlap_min: float,
    overlap_max: float,
    *,
    min_overlap_points: int,
) -> tuple[float, int, float | None]:
    q_values = _overlap_sample_q(
        left_data[:, 0],
        right_data[:, 0],
        overlap_min,
        overlap_max,
        min_overlap_points=min_overlap_points,
    )
    left_i = np.interp(q_values, left_data[:, 0], left_data[:, 1])
    right_i = np.interp(q_values, right_data[:, 0], right_data[:, 1])
    mask = (
        np.isfinite(left_i)
        & np.isfinite(right_i)
        & (left_i > 0.0)
        & (right_i > 0.0)
    )
    if not np.any(mask):
        return _boundary_scale_factor(left_data, right_data), 0, None
    ratios = left_i[mask] / right_i[mask]
    ratios = ratios[np.isfinite(ratios) & (ratios > 0.0)]
    if len(ratios) == 0:
        return _boundary_scale_factor(left_data, right_data), 0, None
    scale = float(np.median(ratios))
    scaled_right = right_i[mask] * scale
    denominator = np.maximum(np.abs(left_i[mask]), np.finfo(float).eps)
    residual = float(
        np.median(np.abs(scaled_right - left_i[mask]) / denominator)
    )
    return scale, int(len(ratios)), residual


def _overlap_sample_q(
    left_q: np.ndarray,
    right_q: np.ndarray,
    overlap_min: float,
    overlap_max: float,
    *,
    min_overlap_points: int,
) -> np.ndarray:
    q_values = np.concatenate(
        (
            left_q[(left_q >= overlap_min) & (left_q <= overlap_max)],
            right_q[(right_q >= overlap_min) & (right_q <= overlap_max)],
            np.asarray([overlap_min, overlap_max], dtype=float),
        )
    )
    q_values = np.unique(q_values[np.isfinite(q_values)])
    if len(q_values) >= min_overlap_points:
        return q_values
    count = max(min_overlap_points, 25)
    if overlap_min > 0.0:
        return np.geomspace(overlap_min, overlap_max, count)
    return np.linspace(overlap_min, overlap_max, count)


def _boundary_scale_factor(
    left_data: np.ndarray,
    right_data: np.ndarray,
) -> float:
    left_value = float(left_data[-1, 1])
    right_value = float(right_data[0, 1])
    if not np.isfinite(left_value) or not np.isfinite(right_value):
        return 1.0
    if right_value == 0.0:
        return 1.0
    scale = left_value / right_value
    if not np.isfinite(scale) or scale <= 0.0:
        return 1.0
    return float(scale)


def _scale_error_column(error_values: np.ndarray, scale: float) -> np.ndarray:
    if np.all(~np.isfinite(error_values)):
        return error_values
    return error_values * abs(scale)


def _blend_overlap(
    left_data: np.ndarray,
    right_data: np.ndarray,
    overlap_min: float,
    overlap_max: float,
) -> tuple[np.ndarray, int]:
    q_blend = np.concatenate(
        (
            left_data[
                (left_data[:, 0] >= overlap_min)
                & (left_data[:, 0] <= overlap_max),
                0,
            ],
            right_data[
                (right_data[:, 0] >= overlap_min)
                & (right_data[:, 0] <= overlap_max),
                0,
            ],
            np.asarray([overlap_min, overlap_max], dtype=float),
        )
    )
    q_blend = np.unique(q_blend[np.isfinite(q_blend)])
    if len(q_blend) < 2:
        return _append_without_overlap(left_data, right_data), 0

    left_i = np.interp(q_blend, left_data[:, 0], left_data[:, 1])
    right_i = np.interp(q_blend, right_data[:, 0], right_data[:, 1])
    left_e = np.interp(
        q_blend,
        left_data[:, 0],
        _finite_error_source(left_data),
    )
    right_e = np.interp(
        q_blend,
        right_data[:, 0],
        _finite_error_source(right_data),
    )
    if overlap_min > 0.0 and overlap_max > overlap_min:
        t = (np.log(q_blend) - np.log(overlap_min)) / (
            np.log(overlap_max) - np.log(overlap_min)
        )
    else:
        t = (q_blend - overlap_min) / (overlap_max - overlap_min)
    t = np.clip(t, 0.0, 1.0)
    positive = (left_i > 0.0) & (right_i > 0.0)
    blended_i = (1.0 - t) * left_i + t * right_i
    blended_i[positive] = np.exp(
        (1.0 - t[positive]) * np.log(left_i[positive])
        + t[positive] * np.log(right_i[positive])
    )
    blended_e = np.sqrt(((1.0 - t) * left_e) ** 2 + (t * right_e) ** 2)
    blended = np.column_stack((q_blend, blended_i, blended_e))
    left_before = left_data[left_data[:, 0] < overlap_min]
    right_after = right_data[right_data[:, 0] > overlap_max]
    joined = np.vstack((left_before, blended, right_after))
    return joined, len(blended)


def _finite_error_source(data: np.ndarray) -> np.ndarray:
    errors = data[:, 2].copy()
    if np.all(~np.isfinite(errors)):
        return np.zeros_like(errors)
    finite_errors = errors[np.isfinite(errors)]
    fill_value = float(np.median(finite_errors)) if len(finite_errors) else 0.0
    errors[~np.isfinite(errors)] = fill_value
    return errors


def _append_without_overlap(
    left_data: np.ndarray,
    right_data: np.ndarray,
) -> np.ndarray:
    if right_data[0, 0] <= left_data[-1, 0]:
        right_data = right_data[right_data[:, 0] > left_data[-1, 0]]
    if len(right_data) == 0:
        return left_data.copy()
    return np.vstack((left_data, right_data))


def _stitch_header(result: APSDetectorStitchResult) -> str:
    lines = ["q\tI\terr", "Generated by SAXSShell APS Detector Stitch."]
    for detector in APS_DETECTOR_ORDER:
        path = result.detector_paths.get(detector)
        if path is not None:
            lines.append(f"{detector}: {path}")
    for join in result.joins:
        lines.append(
            f"{join.left_detector}->{join.right_detector}: "
            f"scale={join.scale_factor:.12g}, "
            f"q={join.q_min:.12g}-{join.q_max:.12g}, "
            f"method={join.method}"
        )
    return "\n".join(lines)


__all__ = [
    "APS_DETECTOR_ORDER",
    "APSDetectorJoinDiagnostic",
    "APSDetectorStitchResult",
    "APSDetectorTrace",
    "detector_name_from_path",
    "find_aps_detector_files",
    "load_aps_detector_trace",
    "save_aps_stitched_data",
    "stitch_aps_detector_files",
    "stitch_aps_detector_traces",
]
