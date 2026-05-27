from __future__ import annotations

import ast
import csv
import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
from lmfit import Parameters
from lmfit import minimize as lm_minimize

HC_EV_NM = 1239.841984
FWHM_GAUSS_FACTOR = 2.0 * math.sqrt(2.0 * math.log(2.0))
PEAK_COLORS = (
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#CC79A7",
    "#56B4E9",
    "#D55E00",
    "#000000",
    "#999999",
)
FIT_PARAMETER_NAMES = ("amplitude", "area", "center", "fwhm", "eta")
LMFIT_PARAMETER_NAMES = ("amplitude", "center", "fwhm", "eta")
DEFAULT_MONTE_CARLO_SEED = 42
MonteCarloProgressCallback = Callable[[int, int, str], None]


@dataclass(slots=True)
class PeakComponent:
    label: str
    amplitude: float
    center: float
    fwhm: float
    eta: float = 0.2
    color: str = PEAK_COLORS[0]
    locked: dict[str, bool] = field(default_factory=dict)
    constraints: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.label = str(self.label).strip() or "A"
        self.amplitude = max(float(self.amplitude), 0.0)
        self.center = float(self.center)
        self.fwhm = max(float(self.fwhm), 1e-9)
        self.eta = min(max(float(self.eta), 0.0), 1.0)
        self.locked = {
            name: bool(self.locked.get(name, False))
            for name in FIT_PARAMETER_NAMES
        }
        self.constraints = {
            name: str(self.constraints.get(name, "") or "")
            for name in FIT_PARAMETER_NAMES
        }

    @property
    def area(self) -> float:
        return component_area(self.amplitude, self.fwhm, self.eta)

    def copy(self) -> PeakComponent:
        return PeakComponent(
            label=self.label,
            amplitude=self.amplitude,
            center=self.center,
            fwhm=self.fwhm,
            eta=self.eta,
            color=self.color,
            locked=dict(self.locked),
            constraints=dict(self.constraints),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "amplitude": self.amplitude,
            "area": self.area,
            "center": self.center,
            "fwhm": self.fwhm,
            "eta": self.eta,
            "color": self.color,
            "locked": dict(self.locked),
            "constraints": dict(self.constraints),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> PeakComponent:
        fwhm = float(payload.get("fwhm", 20.0) or 20.0)
        eta = float(payload.get("eta", 0.2) or 0.2)
        amplitude = float(payload.get("amplitude", 0.0) or 0.0)
        if amplitude <= 0 and "area" in payload:
            amplitude = amplitude_from_area(
                float(payload.get("area", 0.0) or 0.0),
                fwhm,
                eta,
            )
        return cls(
            label=str(payload.get("label", "A") or "A"),
            amplitude=amplitude,
            center=float(payload.get("center", 0.0) or 0.0),
            fwhm=fwhm,
            eta=eta,
            color=str(payload.get("color", PEAK_COLORS[0]) or PEAK_COLORS[0]),
            locked=dict(payload.get("locked", {}) or {}),
            constraints=dict(payload.get("constraints", {}) or {}),
        )


@dataclass(slots=True)
class UVVisDataset:
    source_path: str
    x: np.ndarray
    y: np.ndarray
    x_label: str = "Wavelength (nm)"
    y_label: str = "Absorbance"

    def __post_init__(self) -> None:
        self.x = np.asarray(self.x, dtype=float)
        self.y = np.asarray(self.y, dtype=float)
        if self.x.shape != self.y.shape:
            raise ValueError("UV-Vis x and y arrays must have the same shape.")
        valid = np.isfinite(self.x) & np.isfinite(self.y)
        self.x = self.x[valid]
        self.y = self.y[valid]
        order = np.argsort(self.x)
        self.x = self.x[order]
        self.y = self.y[order]
        if self.x.size < 2:
            raise ValueError("UV-Vis data must contain at least two points.")

    @property
    def x_min(self) -> float:
        return float(np.nanmin(self.x))

    @property
    def x_max(self) -> float:
        return float(np.nanmax(self.x))

    @property
    def y_scale(self) -> float:
        scale = float(np.nanstd(self.y))
        if not np.isfinite(scale) or scale <= 0:
            scale = float(np.nanmax(np.abs(self.y)))
        return scale if np.isfinite(scale) and scale > 0 else 1.0

    def window_mask(
        self,
        fit_min: float | None = None,
        fit_max: float | None = None,
    ) -> np.ndarray:
        mask = np.ones_like(self.x, dtype=bool)
        if fit_min is not None:
            mask &= self.x >= float(fit_min)
        if fit_max is not None:
            mask &= self.x <= float(fit_max)
        return mask

    def to_dict(self, *, include_data: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "source_path": self.source_path,
            "x_label": self.x_label,
            "y_label": self.y_label,
        }
        if include_data:
            payload["x"] = self.x.tolist()
            payload["y"] = self.y.tolist()
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> UVVisDataset:
        return cls(
            source_path=str(payload.get("source_path", "") or ""),
            x=np.asarray(payload.get("x", []), dtype=float),
            y=np.asarray(payload.get("y", []), dtype=float),
            x_label=str(payload.get("x_label", "Wavelength (nm)") or ""),
            y_label=str(payload.get("y_label", "Absorbance") or ""),
        )


@dataclass(slots=True)
class FitResult:
    components: list[PeakComponent]
    x_fit: np.ndarray
    y_data: np.ndarray
    total: np.ndarray
    component_curves: list[np.ndarray]
    residual: np.ndarray
    chisq: float
    redchi: float
    success: bool
    message: str
    nfev: int

    def to_dict(self) -> dict[str, object]:
        return {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "model": "pseudo_voigt_height",
            "success": self.success,
            "message": self.message,
            "nfev": self.nfev,
            "chisq": self.chisq,
            "redchi": self.redchi,
            "peaks": [component.to_dict() for component in self.components],
        }


@dataclass(slots=True)
class MonteCarloSettings:
    iterations: int = 100
    keep_fraction: float = 0.8
    seed: int | None = DEFAULT_MONTE_CARLO_SEED
    noise_scale: float = 1.0
    noise_sigma: float | None = None
    amplitude_jitter: float = 0.25
    center_jitter: float = 5.0
    fwhm_jitter: float = 0.25
    eta_jitter: float = 0.1
    max_nfev: int = 1000
    percentile_low: float = 16.0
    percentile_high: float = 84.0

    def __post_init__(self) -> None:
        self.iterations = int(self.iterations)
        if self.iterations < 1:
            raise ValueError("Monte Carlo fitting requires at least one fit.")
        self.keep_fraction = float(self.keep_fraction)
        if not 0.0 < self.keep_fraction <= 1.0:
            raise ValueError("Monte Carlo keep fraction must be in (0, 1].")
        self.seed = None if self.seed is None else int(self.seed)
        self.noise_scale = max(float(self.noise_scale), 0.0)
        if self.noise_sigma is not None:
            self.noise_sigma = max(float(self.noise_sigma), 0.0)
        self.amplitude_jitter = max(float(self.amplitude_jitter), 0.0)
        self.center_jitter = max(float(self.center_jitter), 0.0)
        self.fwhm_jitter = max(float(self.fwhm_jitter), 0.0)
        self.eta_jitter = max(float(self.eta_jitter), 0.0)
        self.max_nfev = int(self.max_nfev)
        if self.max_nfev < 1:
            raise ValueError("Monte Carlo max_nfev must be positive.")
        self.percentile_low = float(self.percentile_low)
        self.percentile_high = float(self.percentile_high)
        if not 0.0 <= self.percentile_low <= self.percentile_high <= 100.0:
            raise ValueError(
                "Monte Carlo percentiles must be between 0 and 100."
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "iterations": self.iterations,
            "keep_fraction": self.keep_fraction,
            "seed": self.seed,
            "noise_scale": self.noise_scale,
            "noise_sigma": self.noise_sigma,
            "amplitude_jitter": self.amplitude_jitter,
            "center_jitter": self.center_jitter,
            "fwhm_jitter": self.fwhm_jitter,
            "eta_jitter": self.eta_jitter,
            "max_nfev": self.max_nfev,
            "percentile_low": self.percentile_low,
            "percentile_high": self.percentile_high,
        }


@dataclass(slots=True)
class MonteCarloFitRecord:
    index: int
    initial_components: list[PeakComponent]
    result: FitResult
    retained: bool = False

    @property
    def chisq(self) -> float:
        return self.result.chisq

    @property
    def redchi(self) -> float:
        return self.result.redchi


@dataclass(slots=True)
class MonteCarloParameterSummary:
    peak_label: str
    parameter: str
    best: float
    mean: float
    median: float
    std: float
    lower: float
    upper: float
    minimum: float
    maximum: float
    count: int

    @property
    def lower_error(self) -> float:
        return max(self.median - self.lower, 0.0)

    @property
    def upper_error(self) -> float:
        return max(self.upper - self.median, 0.0)

    def to_dict(self) -> dict[str, object]:
        return {
            "peak": self.peak_label,
            "parameter": self.parameter,
            "best": self.best,
            "mean": self.mean,
            "median": self.median,
            "std": self.std,
            "lower": self.lower,
            "upper": self.upper,
            "lower_error": self.lower_error,
            "upper_error": self.upper_error,
            "min": self.minimum,
            "max": self.maximum,
            "count": self.count,
        }


@dataclass(slots=True)
class MonteCarloResult:
    settings: MonteCarloSettings
    records: list[MonteCarloFitRecord]
    retained_records: list[MonteCarloFitRecord]
    summaries: list[MonteCarloParameterSummary]
    failures: list[str] = field(default_factory=list)
    output_dir: Path | None = None

    @property
    def best_record(self) -> MonteCarloFitRecord:
        if not self.records:
            raise ValueError("Monte Carlo result does not contain any fits.")
        return self.records[0]

    @property
    def attempted(self) -> int:
        return self.settings.iterations

    @property
    def completed(self) -> int:
        return len(self.records)

    @property
    def failed(self) -> int:
        return len(self.failures)

    def to_dict(self, *, include_records: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "settings": self.settings.to_dict(),
            "attempted": self.attempted,
            "completed": self.completed,
            "failed": self.failed,
            "retained": len(self.retained_records),
            "best_chisq": self.best_record.chisq if self.records else None,
            "best_redchi": self.best_record.redchi if self.records else None,
            "error_bars": [summary.to_dict() for summary in self.summaries],
            "failures": self.failures[:25],
        }
        if self.output_dir is not None:
            payload["output_dir"] = str(self.output_dir)
        if include_records:
            payload["records"] = [
                {
                    "rank": rank + 1,
                    "index": record.index,
                    "retained": record.retained,
                    "chisq": record.chisq,
                    "redchi": record.redchi,
                    "success": record.result.success,
                    "message": record.result.message,
                    "nfev": record.result.nfev,
                    "peaks": [
                        component.to_dict()
                        for component in record.result.components
                    ],
                }
                for rank, record in enumerate(self.records)
            ]
        return payload


@dataclass(slots=True)
class SweepFitRecord:
    value: float
    result: FitResult
    output_dir: Path

    @property
    def chisq(self) -> float:
        return self.result.chisq

    @property
    def redchi(self) -> float:
        return self.result.redchi

    @property
    def fit_dir(self) -> Path:
        return self.output_dir


def load_uvvis_file(path: str | Path) -> UVVisDataset:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"UV-Vis data file not found: {resolved}")

    try:
        data = np.genfromtxt(
            resolved,
            names=True,
            dtype=float,
            encoding="utf-8",
        )
    except ValueError:
        data = np.genfromtxt(resolved, dtype=float, encoding="utf-8")

    if getattr(data, "dtype", None) is not None and data.dtype.names:
        names = tuple(data.dtype.names)
        if len(names) < 2:
            raise ValueError("UV-Vis file must contain at least two columns.")
        x = np.asarray(data[names[0]], dtype=float)
        y = np.asarray(data[names[1]], dtype=float)
        x_label = _humanize_header(names[0])
        y_label = _humanize_header(names[1])
    else:
        array = np.asarray(data, dtype=float)
        if array.ndim != 2 or array.shape[1] < 2:
            raise ValueError("UV-Vis file must contain at least two columns.")
        x = array[:, 0]
        y = array[:, 1]
        x_label = "Wavelength (nm)"
        y_label = "Absorbance"

    return UVVisDataset(
        source_path=str(resolved),
        x=x,
        y=y,
        x_label=x_label,
        y_label=y_label,
    )


def _humanize_header(header: str) -> str:
    cleaned = str(header).replace("_", " ").strip()
    replacements = {
        "wavelength nm": "Wavelength (nm)",
        "apparent pb molar absorptivity": "Apparent Pb molar absorptivity",
    }
    return replacements.get(cleaned.lower(), cleaned)


def gaussian_from_fwhm(
    x: np.ndarray,
    amplitude: float,
    center: float,
    fwhm: float,
) -> np.ndarray:
    sigma = max(float(fwhm), 1e-9) / FWHM_GAUSS_FACTOR
    return float(amplitude) * np.exp(-0.5 * ((x - float(center)) / sigma) ** 2)


def lorentzian_from_fwhm(
    x: np.ndarray,
    amplitude: float,
    center: float,
    fwhm: float,
) -> np.ndarray:
    gamma = max(float(fwhm), 1e-9) / 2.0
    return float(amplitude) * gamma**2 / ((x - float(center)) ** 2 + gamma**2)


def pseudo_voigt_from_fwhm(
    x: np.ndarray,
    amplitude: float,
    center: float,
    fwhm: float,
    eta: float,
) -> np.ndarray:
    eta = min(max(float(eta), 0.0), 1.0)
    return (1.0 - eta) * gaussian_from_fwhm(
        x, amplitude, center, fwhm
    ) + eta * lorentzian_from_fwhm(x, amplitude, center, fwhm)


def component_area(amplitude: float, fwhm: float, eta: float) -> float:
    fwhm = max(float(fwhm), 1e-9)
    eta = min(max(float(eta), 0.0), 1.0)
    gaussian_area = fwhm / FWHM_GAUSS_FACTOR * math.sqrt(2.0 * math.pi)
    lorentzian_area = math.pi * fwhm / 2.0
    return float(amplitude) * (
        (1.0 - eta) * gaussian_area + eta * lorentzian_area
    )


def amplitude_from_area(area: float, fwhm: float, eta: float) -> float:
    unit_area = component_area(1.0, fwhm, eta)
    if not np.isfinite(unit_area) or abs(unit_area) < 1e-12:
        return 0.0
    return max(float(area) / unit_area, 0.0)


def evaluate_components(
    x: np.ndarray,
    components: list[PeakComponent],
) -> tuple[list[np.ndarray], np.ndarray]:
    curves = [
        pseudo_voigt_from_fwhm(
            x,
            component.amplitude,
            component.center,
            component.fwhm,
            component.eta,
        )
        for component in components
    ]
    total = np.sum(curves, axis=0) if curves else np.zeros_like(x)
    return curves, total


def guess_initial_components(
    dataset: UVVisDataset,
    *,
    count: int = 3,
) -> list[PeakComponent]:
    x = dataset.x
    y = dataset.y
    baseline = float(np.nanmin(y))
    shifted = y - baseline
    y_range = float(np.nanmax(shifted) - np.nanmin(shifted))
    if not np.isfinite(y_range) or y_range <= 0:
        y_range = max(dataset.y_scale, 1.0)
    indices = _find_peak_indices(shifted, count=count)
    if not indices:
        indices = [int(np.nanargmax(shifted))]
    dx = float(np.nanmedian(np.abs(np.diff(x))))
    if not np.isfinite(dx) or dx <= 0:
        dx = max((dataset.x_max - dataset.x_min) / max(len(x), 1), 1.0)
    components: list[PeakComponent] = []
    for idx, peak_idx in enumerate(indices[:count]):
        label = peak_label_for_index(idx)
        amplitude = max(float(shifted[peak_idx]), dataset.y_scale * 0.05)
        fwhm = _estimate_fwhm(x, shifted, peak_idx, fallback=20.0 * dx)
        components.append(
            PeakComponent(
                label=label,
                amplitude=amplitude,
                center=float(x[peak_idx]),
                fwhm=fwhm,
                eta=0.2,
                color=PEAK_COLORS[idx % len(PEAK_COLORS)],
            )
        )
    components.sort(key=lambda component: component.center)
    for idx, component in enumerate(components):
        component.label = peak_label_for_index(idx)
        component.color = PEAK_COLORS[idx % len(PEAK_COLORS)]
    return components


def _find_peak_indices(y: np.ndarray, *, count: int) -> list[int]:
    try:
        from scipy.signal import find_peaks
    except Exception:
        order = np.argsort(y)[::-1]
        return [int(idx) for idx in order[:count]]

    y_range = float(np.nanmax(y) - np.nanmin(y))
    prominence = max(y_range * 0.06, 1e-12)
    distance = max(3, int(len(y) / max(count * 4, 1)))
    peaks, props = find_peaks(y, prominence=prominence, distance=distance)
    if peaks.size == 0:
        return []
    prominences = np.asarray(props.get("prominences", np.zeros_like(peaks)))
    order = np.argsort(prominences)[::-1]
    return [int(peaks[idx]) for idx in order[:count]]


def _estimate_fwhm(
    x: np.ndarray,
    y: np.ndarray,
    peak_idx: int,
    *,
    fallback: float,
) -> float:
    peak_y = float(y[peak_idx])
    if not np.isfinite(peak_y) or peak_y <= 0:
        return max(float(fallback), 1e-9)
    half = peak_y / 2.0
    left = peak_idx
    while left > 0 and y[left] > half:
        left -= 1
    right = peak_idx
    while right < len(y) - 1 and y[right] > half:
        right += 1
    width = abs(float(x[right]) - float(x[left])) if right > left else fallback
    return max(float(width), float(fallback) * 0.25, 1e-9)


def peak_label_for_index(index: int) -> str:
    if index < 26:
        return chr(ord("A") + int(index))
    return f"P{index + 1}"


def next_peak_label(components: list[PeakComponent]) -> str:
    used = {component.label for component in components}
    idx = 0
    while True:
        label = peak_label_for_index(idx)
        if label not in used:
            return label
        idx += 1


def parameter_key(label: str, parameter: str) -> str:
    safe_label = re.sub(r"\W+", "_", label).strip("_") or "A"
    return f"p_{safe_label}_{parameter}"


def parse_interval_constraint(text: str) -> tuple[float, float] | None:
    stripped = str(text or "").strip()
    if not stripped:
        return None
    match = re.fullmatch(
        r"\[\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*,\s*"
        r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*\]",
        stripped,
    )
    if match is None:
        return None
    lower = float(match.group(1))
    upper = float(match.group(2))
    if lower > upper:
        lower, upper = upper, lower
    return lower, upper


def parse_sweep_range(text: str) -> tuple[float, float, int]:
    stripped = str(text or "").strip()
    match = re.fullmatch(
        r"\[\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*,\s*"
        r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*,\s*"
        r"(\d+)\s*\]",
        stripped,
    )
    if match is None:
        raise ValueError("Enter a range in the form [min, max, steps].")
    lower = float(match.group(1))
    upper = float(match.group(2))
    steps = int(match.group(3))
    if steps < 2:
        raise ValueError("Range fits require at least two steps.")
    return lower, upper, steps


def is_relation_constraint(text: str) -> bool:
    stripped = str(text or "").strip()
    return "@" in stripped


def relation_constraint_to_lmfit_expr(
    text: str,
    parameter: str,
    labels: set[str],
) -> str | None:
    stripped = str(text or "").strip()
    if not is_relation_constraint(stripped):
        return None

    def replace_reference(match: re.Match[str]) -> str:
        label = match.group(1)
        if label not in labels:
            raise ValueError(f"Unknown peak label in constraint: @{label}")
        return parameter_key(label, parameter)

    return re.sub(r"@([A-Za-z][A-Za-z0-9_]*)", replace_reference, stripped)


def relation_constraint_value(
    text: str,
    values_by_label: dict[str, float],
) -> float:
    stripped = str(text or "").strip()
    if not stripped:
        raise ValueError("Empty relation constraint.")
    expression = re.sub(
        r"@([A-Za-z][A-Za-z0-9_]*)",
        lambda match: match.group(1),
        stripped,
    )
    return float(_safe_arithmetic_eval(expression, values_by_label))


def _safe_arithmetic_eval(
    expression: str, variables: dict[str, float]
) -> float:
    tree = ast.parse(expression, mode="eval")
    return float(_eval_ast_node(tree.body, variables))


def _eval_ast_node(node: ast.AST, variables: dict[str, float]) -> float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError("Only numeric constants are allowed.")
    if isinstance(node, ast.Name):
        if node.id not in variables:
            raise ValueError(f"Unknown peak label in constraint: @{node.id}")
        return float(variables[node.id])
    if isinstance(node, ast.UnaryOp):
        value = _eval_ast_node(node.operand, variables)
        if isinstance(node.op, ast.UAdd):
            return value
        if isinstance(node.op, ast.USub):
            return -value
    if isinstance(node, ast.BinOp):
        left = _eval_ast_node(node.left, variables)
        right = _eval_ast_node(node.right, variables)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
    raise ValueError("Constraint expressions may only use +, -, *, and /.")


def fit_components(
    dataset: UVVisDataset,
    components: list[PeakComponent],
    *,
    fit_min: float | None = None,
    fit_max: float | None = None,
    max_nfev: int = 2000,
) -> FitResult:
    if not components:
        raise ValueError("Add at least one peak before fitting.")
    mask = dataset.window_mask(fit_min, fit_max)
    if int(np.count_nonzero(mask)) < 3:
        raise ValueError("Fit window must contain at least three data points.")
    x = dataset.x[mask]
    y = dataset.y[mask]
    starting_components = [component.copy() for component in components]
    params = _build_lmfit_parameters(
        dataset,
        starting_components,
        x_min=float(np.nanmin(x)),
        x_max=float(np.nanmax(x)),
    )
    area_targets = {
        component.label: component.area for component in starting_components
    }

    def residual(active_params: Parameters) -> np.ndarray:
        fitted_components = _params_to_components(
            starting_components,
            active_params,
        )
        _component_curves, total = evaluate_components(x, fitted_components)
        raw = total - y
        penalties = _area_constraint_penalties(
            fitted_components,
            starting_components,
            area_targets,
            dataset.y_scale,
            len(raw),
        )
        if penalties.size:
            return np.concatenate([raw, penalties])
        return raw

    result = lm_minimize(
        residual,
        params,
        method="least_squares",
        max_nfev=max_nfev,
    )
    fitted = _params_to_components(starting_components, result.params)
    component_curves, total = evaluate_components(x, fitted)
    raw_residual = total - y
    chisq = float(np.sum(raw_residual**2))
    nvarys = int(getattr(result, "nvarys", 0) or 0)
    redchi = chisq / max(int(raw_residual.size) - nvarys, 1)
    return FitResult(
        components=fitted,
        x_fit=x,
        y_data=y,
        total=total,
        component_curves=component_curves,
        residual=raw_residual,
        chisq=chisq,
        redchi=redchi,
        success=bool(getattr(result, "success", False)),
        message=str(getattr(result, "message", "")),
        nfev=int(getattr(result, "nfev", 0) or 0),
    )


def evaluate_fit_result(
    dataset: UVVisDataset,
    components: list[PeakComponent],
    *,
    fit_min: float | None = None,
    fit_max: float | None = None,
    message: str = "Evaluated peak model without optimization.",
) -> FitResult:
    if not components:
        raise ValueError("Add at least one peak before evaluating a fit.")
    mask = dataset.window_mask(fit_min, fit_max)
    if int(np.count_nonzero(mask)) < 3:
        raise ValueError("Fit window must contain at least three data points.")
    x = dataset.x[mask]
    y = dataset.y[mask]
    copied = [component.copy() for component in components]
    component_curves, total = evaluate_components(x, copied)
    residual = total - y
    chisq = float(np.sum(residual**2))
    nvarys = sum(
        1
        for component in copied
        for parameter in LMFIT_PARAMETER_NAMES
        if not component.locked.get(parameter, False)
        and not is_relation_constraint(
            component.constraints.get(parameter, "")
        )
    )
    redchi = chisq / max(int(residual.size) - nvarys, 1)
    return FitResult(
        components=copied,
        x_fit=x,
        y_data=y,
        total=total,
        component_curves=component_curves,
        residual=residual,
        chisq=chisq,
        redchi=redchi,
        success=True,
        message=message,
        nfev=0,
    )


def _default_lmfit_bounds(
    dataset: UVVisDataset,
    *,
    x_min: float | None = None,
    x_max: float | None = None,
) -> dict[str, tuple[float | None, float | None]]:
    lower_x = dataset.x_min if x_min is None else float(x_min)
    upper_x = dataset.x_max if x_max is None else float(x_max)
    if lower_x > upper_x:
        lower_x, upper_x = upper_x, lower_x
    x_span = max(upper_x - lower_x, 1.0)
    return {
        "amplitude": (0.0, None),
        "center": (None, None),
        "fwhm": (max(x_span / 10000.0, 1e-9), x_span * 4.0),
        "eta": (0.0, 1.0),
    }


def _build_lmfit_parameters(
    dataset: UVVisDataset,
    components: list[PeakComponent],
    *,
    x_min: float | None = None,
    x_max: float | None = None,
) -> Parameters:
    params = Parameters()
    labels = {component.label for component in components}
    default_bounds = _default_lmfit_bounds(dataset, x_min=x_min, x_max=x_max)
    # Pass 1: add every parameter as a plain (varying) value, so that
    # later expressions can reference any peak regardless of order.
    for component in components:
        for parameter in LMFIT_PARAMETER_NAMES:
            name = parameter_key(component.label, parameter)
            value = float(getattr(component, parameter))
            lower, upper = default_bounds[parameter]
            constraint = component.constraints.get(parameter, "")
            interval = parse_interval_constraint(constraint)
            if interval is not None:
                lower, upper = interval
            params.add(
                name,
                value=value,
                min=-np.inf if lower is None else float(lower),
                max=np.inf if upper is None else float(upper),
                vary=True,
            )
    # Pass 2: apply locks and relational expressions now that every name
    # referenced by `@X` exists in `params`.
    for component in components:
        for parameter in LMFIT_PARAMETER_NAMES:
            name = parameter_key(component.label, parameter)
            constraint = component.constraints.get(parameter, "")
            expr = relation_constraint_to_lmfit_expr(
                constraint,
                parameter,
                labels,
            )
            if expr is not None:
                params[name].set(expr=expr)
            elif bool(component.locked.get(parameter, False)):
                params[name].set(vary=False)
    return params


def _params_to_components(
    templates: list[PeakComponent],
    params: Parameters,
) -> list[PeakComponent]:
    fitted: list[PeakComponent] = []
    for template in templates:
        fitted.append(
            PeakComponent(
                label=template.label,
                amplitude=float(
                    params[parameter_key(template.label, "amplitude")].value
                ),
                center=float(
                    params[parameter_key(template.label, "center")].value
                ),
                fwhm=float(
                    params[parameter_key(template.label, "fwhm")].value
                ),
                eta=float(params[parameter_key(template.label, "eta")].value),
                color=template.color,
                locked=dict(template.locked),
                constraints=dict(template.constraints),
            )
        )
    return fitted


def _area_constraint_penalties(
    fitted: list[PeakComponent],
    templates: list[PeakComponent],
    area_targets: dict[str, float],
    data_scale: float,
    point_count: int,
) -> np.ndarray:
    areas = {component.label: component.area for component in fitted}
    fitted_by_label = {component.label: component for component in fitted}
    penalties: list[float] = []
    weight = max(float(data_scale), 1.0) * max(math.sqrt(point_count), 1.0)
    for template in templates:
        if template.label not in fitted_by_label:
            continue
        area = fitted_by_label[template.label].area
        constraint = template.constraints.get("area", "")
        target: float | None = None
        if template.locked.get("area", False):
            target = float(area_targets[template.label])
        elif is_relation_constraint(constraint):
            target = relation_constraint_value(constraint, areas)

        if target is not None:
            scale = max(abs(target), abs(area), 1.0)
            penalties.append(weight * (area - target) / scale)
            continue

        interval = parse_interval_constraint(constraint)
        if interval is None:
            continue
        lower, upper = interval
        if area < lower:
            scale = max(abs(lower), abs(area), 1.0)
            penalties.append(weight * (area - lower) / scale)
        elif area > upper:
            scale = max(abs(upper), abs(area), 1.0)
            penalties.append(weight * (area - upper) / scale)
        else:
            penalties.append(0.0)
    return np.asarray(penalties, dtype=float)


def set_component_parameter(
    component: PeakComponent,
    parameter: str,
    value: float,
) -> None:
    if parameter == "area":
        component.amplitude = amplitude_from_area(
            float(value),
            component.fwhm,
            component.eta,
        )
        return
    if parameter not in LMFIT_PARAMETER_NAMES:
        raise ValueError(f"Unsupported parameter: {parameter}")
    if parameter == "eta":
        value = min(max(float(value), 0.0), 1.0)
    elif parameter in {"amplitude", "fwhm"}:
        value = max(float(value), 1e-9 if parameter == "fwhm" else 0.0)
    setattr(component, parameter, float(value))


def component_parameter_value(
    component: PeakComponent,
    parameter: str,
) -> float:
    if parameter == "area":
        return component.area
    if parameter not in LMFIT_PARAMETER_NAMES:
        raise ValueError(f"Unsupported parameter: {parameter}")
    return float(getattr(component, parameter))


def run_monte_carlo_fit(
    dataset: UVVisDataset,
    components: list[PeakComponent],
    *,
    settings: MonteCarloSettings | None = None,
    fit_min: float | None = None,
    fit_max: float | None = None,
    reference_result: FitResult | None = None,
    output_root: str | Path | None = None,
    progress_callback: MonteCarloProgressCallback | None = None,
) -> MonteCarloResult:
    """Estimate peak-parameter uncertainty with CasaXPS-style
    simulations.

    A reference fit is used as the underlying envelope. Each Monte Carlo
    iteration adds synthetic Gaussian noise to that envelope, jitters
    the starting peak set, and refits the synthetic spectrum under the
    same locks, interval bounds, and relational constraints as the
    original model.
    """
    if not components:
        raise ValueError("Add at least one peak before Monte Carlo fitting.")
    active_settings = settings or MonteCarloSettings()
    if progress_callback is not None:
        progress_callback(
            0,
            active_settings.iterations,
            "Preparing Monte Carlo reference fit...",
        )
    if reference_result is None:
        reference_result = fit_components(
            dataset,
            components,
            fit_min=fit_min,
            fit_max=fit_max,
            max_nfev=active_settings.max_nfev,
        )
    reference_components = [
        component.copy() for component in reference_result.components
    ]
    noise_sigma = _monte_carlo_noise_sigma(
        dataset,
        reference_result,
        active_settings,
    )
    rng = np.random.default_rng(active_settings.seed)
    records: list[MonteCarloFitRecord] = []
    failures: list[str] = []
    for index in range(active_settings.iterations):
        try:
            synthetic_dataset = _synthetic_monte_carlo_dataset(
                dataset,
                reference_components,
                rng=rng,
                noise_sigma=noise_sigma,
                index=index,
            )
            initial_components = _randomized_monte_carlo_components(
                dataset,
                reference_components,
                active_settings,
                rng,
                fit_min=fit_min,
                fit_max=fit_max,
            )
            result = fit_components(
                synthetic_dataset,
                initial_components,
                fit_min=fit_min,
                fit_max=fit_max,
                max_nfev=active_settings.max_nfev,
            )
        except Exception as exc:
            failures.append(f"{index}: {exc}")
        else:
            if np.isfinite(result.chisq):
                records.append(
                    MonteCarloFitRecord(
                        index=index,
                        initial_components=initial_components,
                        result=result,
                    )
                )
            else:
                failures.append(f"{index}: non-finite chi-squared")
        if progress_callback is not None:
            progress_callback(
                index + 1,
                active_settings.iterations,
                f"Processed Monte Carlo fit {index + 1} of "
                f"{active_settings.iterations}.",
            )

    if not records:
        raise ValueError("No finite Monte Carlo fits completed.")
    records.sort(key=lambda record: record.chisq)
    keep_count = max(
        1,
        min(
            len(records),
            int(math.ceil(len(records) * active_settings.keep_fraction)),
        ),
    )
    for rank, record in enumerate(records):
        record.retained = rank < keep_count
    retained_records = records[:keep_count]
    summaries = _summarize_monte_carlo_records(
        retained_records,
        best_record=records[0],
        settings=active_settings,
    )
    resolved_output = (
        None
        if output_root is None
        else Path(output_root).expanduser().resolve()
    )
    monte_carlo_result = MonteCarloResult(
        settings=active_settings,
        records=records,
        retained_records=retained_records,
        summaries=summaries,
        failures=failures,
        output_dir=resolved_output,
    )
    if resolved_output is not None:
        save_monte_carlo_bundle(
            resolved_output,
            dataset,
            monte_carlo_result,
            reference_result=reference_result,
            fit_min=fit_min,
            fit_max=fit_max,
        )
    return monte_carlo_result


def _monte_carlo_noise_sigma(
    dataset: UVVisDataset,
    reference_result: FitResult,
    settings: MonteCarloSettings,
) -> float:
    if settings.noise_sigma is not None:
        return max(float(settings.noise_sigma) * settings.noise_scale, 0.0)
    residual = np.asarray(reference_result.residual, dtype=float)
    if residual.size:
        sigma = math.sqrt(float(np.mean(residual**2)))
    else:
        sigma = 0.0
    if not np.isfinite(sigma) or sigma <= 1e-15:
        sigma = max(float(dataset.y_scale) * 0.01, 1e-12)
    return max(sigma * settings.noise_scale, 0.0)


def _synthetic_monte_carlo_dataset(
    dataset: UVVisDataset,
    reference_components: list[PeakComponent],
    *,
    rng: np.random.Generator,
    noise_sigma: float,
    index: int,
) -> UVVisDataset:
    _curves, envelope = evaluate_components(dataset.x, reference_components)
    if noise_sigma > 0:
        noise = rng.normal(0.0, noise_sigma, size=dataset.x.shape)
    else:
        noise = np.zeros_like(dataset.x)
    return UVVisDataset(
        source_path=f"{dataset.source_path}#monte_carlo_{index}",
        x=dataset.x.copy(),
        y=envelope + noise,
        x_label=dataset.x_label,
        y_label=dataset.y_label,
    )


def _randomized_monte_carlo_components(
    dataset: UVVisDataset,
    components: list[PeakComponent],
    settings: MonteCarloSettings,
    rng: np.random.Generator,
    *,
    fit_min: float | None = None,
    fit_max: float | None = None,
) -> list[PeakComponent]:
    randomized = [component.copy() for component in components]
    for component in randomized:
        if not component.locked.get(
            "amplitude", False
        ) and not is_relation_constraint(
            component.constraints.get("amplitude", "")
        ):
            scale = 1.0 + rng.uniform(
                -settings.amplitude_jitter,
                settings.amplitude_jitter,
            )
            set_component_parameter(
                component,
                "amplitude",
                component.amplitude * max(scale, 0.0),
            )
        if not component.locked.get(
            "center", False
        ) and not is_relation_constraint(
            component.constraints.get("center", "")
        ):
            set_component_parameter(
                component,
                "center",
                component.center
                + rng.uniform(-settings.center_jitter, settings.center_jitter),
            )
        if not component.locked.get(
            "fwhm", False
        ) and not is_relation_constraint(
            component.constraints.get("fwhm", "")
        ):
            scale = 1.0 + rng.uniform(
                -settings.fwhm_jitter,
                settings.fwhm_jitter,
            )
            set_component_parameter(
                component,
                "fwhm",
                component.fwhm * max(scale, 0.0),
            )
        if not component.locked.get(
            "eta", False
        ) and not is_relation_constraint(component.constraints.get("eta", "")):
            set_component_parameter(
                component,
                "eta",
                component.eta
                + rng.uniform(-settings.eta_jitter, settings.eta_jitter),
            )
    _enforce_seed_constraints(
        dataset,
        randomized,
        components,
        fit_min=fit_min,
        fit_max=fit_max,
    )
    return randomized


def _enforce_seed_constraints(
    dataset: UVVisDataset,
    components: list[PeakComponent],
    originals: list[PeakComponent],
    *,
    fit_min: float | None = None,
    fit_max: float | None = None,
) -> None:
    original_by_label = {component.label: component for component in originals}
    bounds = _default_lmfit_bounds(dataset, x_min=fit_min, x_max=fit_max)
    for _ in range(16):
        changed = False
        for parameter in LMFIT_PARAMETER_NAMES:
            for component in components:
                current = component_parameter_value(component, parameter)
                target = current
                original = original_by_label.get(component.label)
                if (
                    component.locked.get(parameter, False)
                    and original is not None
                ):
                    target = component_parameter_value(original, parameter)
                elif is_relation_constraint(
                    component.constraints.get(parameter, "")
                ):
                    values_by_label = {
                        other.label: component_parameter_value(
                            other, parameter
                        )
                        for other in components
                        if other.label != component.label
                    }
                    try:
                        target = relation_constraint_value(
                            component.constraints[parameter],
                            values_by_label,
                        )
                    except Exception:
                        target = current
                else:
                    interval = parse_interval_constraint(
                        component.constraints.get(parameter, "")
                    )
                    lower, upper = (
                        interval if interval is not None else bounds[parameter]
                    )
                    target = _clamp_value(current, lower, upper)
                if abs(target - current) > 1e-10:
                    set_component_parameter(component, parameter, target)
                    changed = True

        for component in components:
            current_area = component.area
            target_area = current_area
            original = original_by_label.get(component.label)
            area_constraint = component.constraints.get("area", "")
            if component.locked.get("area", False) and original is not None:
                target_area = original.area
            elif is_relation_constraint(area_constraint):
                areas_by_label = {
                    other.label: other.area
                    for other in components
                    if other.label != component.label
                }
                try:
                    target_area = relation_constraint_value(
                        area_constraint,
                        areas_by_label,
                    )
                except Exception:
                    target_area = current_area
            else:
                interval = parse_interval_constraint(area_constraint)
                if interval is not None:
                    target_area = _clamp_value(current_area, *interval)
            if abs(target_area - current_area) > 1e-10:
                set_component_parameter(component, "area", target_area)
                changed = True
        if not changed:
            break


def _clamp_value(
    value: float,
    lower: float | None,
    upper: float | None,
) -> float:
    clipped = float(value)
    if lower is not None:
        clipped = max(clipped, float(lower))
    if upper is not None:
        clipped = min(clipped, float(upper))
    return clipped


def _summarize_monte_carlo_records(
    records: list[MonteCarloFitRecord],
    *,
    best_record: MonteCarloFitRecord,
    settings: MonteCarloSettings,
) -> list[MonteCarloParameterSummary]:
    best_by_label = {
        component.label: component
        for component in best_record.result.components
    }
    labels = [component.label for component in best_record.result.components]
    summaries: list[MonteCarloParameterSummary] = []
    for label in labels:
        best_component = best_by_label[label]
        for parameter in FIT_PARAMETER_NAMES:
            values = np.asarray(
                [
                    component_parameter_value(component, parameter)
                    for record in records
                    for component in record.result.components
                    if component.label == label
                ],
                dtype=float,
            )
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            summaries.append(
                MonteCarloParameterSummary(
                    peak_label=label,
                    parameter=parameter,
                    best=component_parameter_value(best_component, parameter),
                    mean=float(np.mean(values)),
                    median=float(np.median(values)),
                    std=(
                        float(np.std(values, ddof=1))
                        if values.size > 1
                        else 0.0
                    ),
                    lower=float(
                        np.percentile(values, settings.percentile_low)
                    ),
                    upper=float(
                        np.percentile(values, settings.percentile_high)
                    ),
                    minimum=float(np.min(values)),
                    maximum=float(np.max(values)),
                    count=int(values.size),
                )
            )
    return summaries


def run_parameter_sweep(
    dataset: UVVisDataset,
    components: list[PeakComponent],
    *,
    peak_label: str,
    parameter: str,
    lower: float,
    upper: float,
    steps: int,
    output_root: str | Path,
    fit_min: float | None = None,
    fit_max: float | None = None,
) -> list[SweepFitRecord]:
    if parameter not in FIT_PARAMETER_NAMES:
        raise ValueError(f"Unsupported sweep parameter: {parameter}")
    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    values = np.linspace(float(lower), float(upper), int(steps))
    records: list[SweepFitRecord] = []
    for index, value in enumerate(values):
        candidate_components = [component.copy() for component in components]
        target = next(
            (
                component
                for component in candidate_components
                if component.label == peak_label
            ),
            None,
        )
        if target is None:
            raise ValueError(f"Peak {peak_label!r} no longer exists.")
        set_component_parameter(target, parameter, float(value))
        target.locked[parameter] = True
        target.constraints[parameter] = ""
        result = fit_components(
            dataset,
            candidate_components,
            fit_min=fit_min,
            fit_max=fit_max,
        )
        run_dir = root / f"{index:03d}_{safe_name(peak_label)}_{parameter}"
        run_dir = run_dir.with_name(f"{run_dir.name}_{value:.8g}")
        save_fit_bundle(
            run_dir,
            dataset,
            result.components,
            result=result,
            fit_min=fit_min,
            fit_max=fit_max,
        )
        records.append(
            SweepFitRecord(
                value=float(value),
                result=result,
                output_dir=run_dir,
            )
        )
    return records


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_") or "fit"


def session_payload(
    dataset: UVVisDataset,
    components: list[PeakComponent],
    *,
    fit_min: float | None = None,
    fit_max: float | None = None,
    result: FitResult | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "format": "saxshell_uvvis_fit",
        "format_version": 1,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "dataset": dataset.to_dict(include_data=True),
        "fit_window": {"min": fit_min, "max": fit_max},
        "peaks": [component.to_dict() for component in components],
    }
    if result is not None:
        payload["last_fit"] = result.to_dict()
    return payload


def load_session_payload(
    payload: dict[str, object],
) -> tuple[UVVisDataset, list[PeakComponent], float | None, float | None]:
    dataset_payload = dict(payload.get("dataset", {}) or {})
    dataset = UVVisDataset.from_dict(dataset_payload)
    components = [
        PeakComponent.from_dict(dict(item))
        for item in list(payload.get("peaks", []) or [])
    ]
    fit_window = dict(payload.get("fit_window", {}) or {})
    fit_min_value = fit_window.get("min")
    fit_max_value = fit_window.get("max")
    fit_min = None if fit_min_value is None else float(fit_min_value)
    fit_max = None if fit_max_value is None else float(fit_max_value)
    return dataset, components, fit_min, fit_max


def load_fit_session(
    path: str | Path,
) -> tuple[UVVisDataset, list[PeakComponent], float | None, float | None]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return load_session_payload(dict(payload))


def save_fit_bundle(
    output_dir: str | Path,
    dataset: UVVisDataset,
    components: list[PeakComponent],
    *,
    result: FitResult | None = None,
    fit_min: float | None = None,
    fit_max: float | None = None,
) -> None:
    resolved = Path(output_dir).expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    payload = session_payload(
        dataset,
        components,
        fit_min=fit_min,
        fit_max=fit_max,
        result=result,
    )
    (resolved / "fit.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    _write_peak_csv(resolved / "peaks.csv", components)
    if result is not None:
        _write_curve_csv(resolved / "curve.csv", result)


def save_monte_carlo_bundle(
    output_dir: str | Path,
    dataset: UVVisDataset,
    result: MonteCarloResult,
    *,
    reference_result: FitResult | None = None,
    fit_min: float | None = None,
    fit_max: float | None = None,
) -> None:
    resolved = Path(output_dir).expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    result.output_dir = resolved
    (resolved / "monte_carlo_summary.json").write_text(
        json.dumps(result.to_dict(include_records=True), indent=2),
        encoding="utf-8",
    )
    _write_monte_carlo_population_csv(
        resolved / "monte_carlo_population.csv",
        result,
    )
    _write_monte_carlo_error_csv(
        resolved / "monte_carlo_error_bars.csv",
        result,
    )
    if reference_result is not None:
        save_fit_bundle(
            resolved / "reference_fit",
            dataset,
            reference_result.components,
            result=reference_result,
            fit_min=fit_min,
            fit_max=fit_max,
        )
    best = result.best_record.result
    best_original_result = evaluate_fit_result(
        dataset,
        best.components,
        fit_min=fit_min,
        fit_max=fit_max,
        message="Best Monte Carlo parameter set evaluated on original data.",
    )
    save_fit_bundle(
        resolved / "best_fit",
        dataset,
        best.components,
        result=best_original_result,
        fit_min=fit_min,
        fit_max=fit_max,
    )
    _write_monte_carlo_trace_error_csv(
        resolved / "monte_carlo_traces_with_errors.csv",
        dataset,
        result,
        best_original_result=best_original_result,
        fit_min=fit_min,
        fit_max=fit_max,
    )
    _write_monte_carlo_fit_report(
        resolved / "monte_carlo_fit_report.md",
        dataset,
        result,
        reference_result=reference_result,
        best_original_result=best_original_result,
        fit_min=fit_min,
        fit_max=fit_max,
    )


def _write_monte_carlo_fit_report(
    path: Path,
    dataset: UVVisDataset,
    result: MonteCarloResult,
    *,
    reference_result: FitResult | None,
    best_original_result: FitResult,
    fit_min: float | None,
    fit_max: float | None,
) -> None:
    settings = result.settings
    best_metrics = _fit_quality_metrics(best_original_result)
    lines = [
        "# Monte Carlo UV-Vis Fit Report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Source: {_markdown_escape(dataset.source_path or 'in-memory data')}",
        f"Fit window: {_format_optional_float(fit_min)} to "
        f"{_format_optional_float(fit_max)}",
        "",
        "## Run Summary",
        "",
        f"- Attempted fits: {result.attempted}",
        f"- Completed fits: {result.completed}",
        f"- Failed fits: {result.failed}",
        f"- Retained population: {len(result.retained_records)} fits "
        f"({settings.keep_fraction * 100.0:.3g}% of completed)",
        f"- Best Monte Carlo chi-squared: {result.best_record.chisq:.10g}",
        f"- Best fit chi-squared on original data: "
        f"{best_original_result.chisq:.10g}",
        f"- Best fit reduced chi-squared on original data: "
        f"{best_original_result.redchi:.10g}",
        f"- Best fit R-squared on original data: "
        f"{best_metrics['r_squared']:.10g}",
        "",
        "## Monte Carlo Settings",
        "",
        "| Setting | Value |",
        "| --- | ---: |",
    ]
    for key, value in settings.to_dict().items():
        lines.append(
            f"| {_markdown_escape(key)} | {_markdown_escape(value)} |"
        )
    lines.extend(
        [
            "",
            "## Output Files",
            "",
            "- `monte_carlo_summary.json`: full ranked population and settings.",
            "- `monte_carlo_population.csv`: ranked fit parameters for each run.",
            "- `monte_carlo_error_bars.csv`: parameter error bars.",
            "- `monte_carlo_traces_with_errors.csv`: experimental data, best "
            "total/peak traces, and retained-population curve bands.",
            "- `reference_fit/`: reference fit used to generate synthetic spectra.",
            "- `best_fit/`: best Monte Carlo parameter set evaluated on original "
            "data.",
            "",
            "## Peak Parameter Error Bars",
            "",
            "| Peak | Parameter | Best | Median | -err | +err | Std | "
            "Mean | Min | Max |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for summary in result.summaries:
        lines.append(
            f"| {_markdown_escape(summary.peak_label)} "
            f"| {_markdown_escape(summary.parameter)} "
            f"| {summary.best:.10g} "
            f"| {summary.median:.10g} "
            f"| {summary.lower_error:.10g} "
            f"| {summary.upper_error:.10g} "
            f"| {summary.std:.10g} "
            f"| {summary.mean:.10g} "
            f"| {summary.minimum:.10g} "
            f"| {summary.maximum:.10g} |"
        )
    lines.extend(
        [
            "",
            "## Peak Constraints",
            "",
            "| Peak | Locked | Constraints |",
            "| --- | --- | --- |",
        ]
    )
    for component in best_original_result.components:
        locked = ", ".join(
            name for name, value in component.locked.items() if value
        )
        constraints = "; ".join(
            f"{name}: {value}"
            for name, value in component.constraints.items()
            if value
        )
        lines.append(
            f"| {_markdown_escape(component.label)} "
            f"| {_markdown_escape(locked or 'none')} "
            f"| {_markdown_escape(constraints or 'none')} |"
        )
    if reference_result is not None:
        reference_metrics = _fit_quality_metrics(reference_result)
        lines.extend(
            [
                "",
                "## Reference Fit",
                "",
                f"- Chi-squared: {reference_result.chisq:.10g}",
                f"- Reduced chi-squared: {reference_result.redchi:.10g}",
                f"- R-squared: {reference_metrics['r_squared']:.10g}",
                f"- RMS residual: {reference_metrics['rms_residual']:.10g}",
                f"- Max absolute residual: "
                f"{reference_metrics['max_abs_residual']:.10g}",
            ]
        )
    if result.failures:
        lines.extend(["", "## Failed Fits", ""])
        for failure in result.failures[:25]:
            lines.append(f"- {_markdown_escape(failure)}")
        if len(result.failures) > 25:
            lines.append(f"- ... {len(result.failures) - 25} more")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "Error bars are percentile bands from the retained best-fit "
            "population. Synthetic spectra are generated by adding Gaussian "
            "noise to the reference fit envelope, then refitting under the "
            "same locks, interval bounds, area penalties, and relational ties.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_monte_carlo_trace_error_csv(
    path: Path,
    dataset: UVVisDataset,
    result: MonteCarloResult,
    *,
    best_original_result: FitResult,
    fit_min: float | None,
    fit_max: float | None,
) -> None:
    labels = [component.label for component in best_original_result.components]
    best_curves, best_total = evaluate_components(
        dataset.x,
        best_original_result.components,
    )
    retained_totals: list[np.ndarray] = []
    retained_peak_curves: dict[str, list[np.ndarray]] = {
        label: [] for label in labels
    }
    for record in result.retained_records:
        curves, total = evaluate_components(
            dataset.x, record.result.components
        )
        retained_totals.append(total)
        for component, curve in zip(
            record.result.components,
            curves,
            strict=False,
        ):
            if component.label in retained_peak_curves:
                retained_peak_curves[component.label].append(curve)

    total_stats = _array_distribution_stats(
        retained_totals,
        low=result.settings.percentile_low,
        high=result.settings.percentile_high,
        fallback=best_total,
    )
    peak_stats = {
        label: _array_distribution_stats(
            retained_peak_curves[label],
            low=result.settings.percentile_low,
            high=result.settings.percentile_high,
            fallback=best_curves[index],
        )
        for index, label in enumerate(labels)
    }
    fieldnames = [
        "x",
        "experimental",
        "in_fit_window",
        "best_total",
        "best_residual",
        "total_mean",
        "total_median",
        "total_std",
        "total_lower",
        "total_upper",
    ]
    for label in labels:
        safe_label = safe_name(label)
        fieldnames.extend(
            [
                f"peak_{safe_label}_best",
                f"peak_{safe_label}_mean",
                f"peak_{safe_label}_median",
                f"peak_{safe_label}_std",
                f"peak_{safe_label}_lower",
                f"peak_{safe_label}_upper",
            ]
        )

    fit_mask = dataset.window_mask(fit_min, fit_max)
    with path.open("w", newline="", encoding="utf-8") as handle:
        handle.write("# saxshell_uvvis_monte_carlo_trace_csv version=1\n")
        handle.write(
            f"# saved={datetime.now().isoformat(timespec='seconds')}\n"
        )
        handle.write(f"# source={dataset.source_path}\n")
        handle.write(f"# x_label={dataset.x_label}\n")
        handle.write(f"# y_label={dataset.y_label}\n")
        handle.write(f"# fit_min={_format_optional_float(fit_min)}\n")
        handle.write(f"# fit_max={_format_optional_float(fit_max)}\n")
        handle.write(f"# retained_fits={len(result.retained_records)}\n")
        handle.write(
            f"# percentile_low={result.settings.percentile_low:.10g}\n"
        )
        handle.write(
            f"# percentile_high={result.settings.percentile_high:.10g}\n"
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        best_curve_by_label = {
            label: best_curves[index] for index, label in enumerate(labels)
        }
        for index, x_value in enumerate(dataset.x):
            row: dict[str, object] = {
                "x": float(x_value),
                "experimental": float(dataset.y[index]),
                "in_fit_window": bool(fit_mask[index]),
                "best_total": float(best_total[index]),
                "best_residual": float(best_total[index] - dataset.y[index]),
                "total_mean": float(total_stats["mean"][index]),
                "total_median": float(total_stats["median"][index]),
                "total_std": float(total_stats["std"][index]),
                "total_lower": float(total_stats["lower"][index]),
                "total_upper": float(total_stats["upper"][index]),
            }
            for label in labels:
                safe_label = safe_name(label)
                stats = peak_stats[label]
                row[f"peak_{safe_label}_best"] = float(
                    best_curve_by_label[label][index]
                )
                row[f"peak_{safe_label}_mean"] = float(stats["mean"][index])
                row[f"peak_{safe_label}_median"] = float(
                    stats["median"][index]
                )
                row[f"peak_{safe_label}_std"] = float(stats["std"][index])
                row[f"peak_{safe_label}_lower"] = float(stats["lower"][index])
                row[f"peak_{safe_label}_upper"] = float(stats["upper"][index])
            writer.writerow(row)


def _array_distribution_stats(
    arrays: list[np.ndarray],
    *,
    low: float,
    high: float,
    fallback: np.ndarray,
) -> dict[str, np.ndarray]:
    if arrays:
        values = np.vstack(
            [np.asarray(array, dtype=float) for array in arrays]
        )
    else:
        values = np.asarray([fallback], dtype=float)
    return {
        "mean": np.mean(values, axis=0),
        "median": np.median(values, axis=0),
        "std": (
            np.std(values, axis=0, ddof=1)
            if values.shape[0] > 1
            else np.zeros(values.shape[1], dtype=float)
        ),
        "lower": np.percentile(values, low, axis=0),
        "upper": np.percentile(values, high, axis=0),
    }


def _fit_quality_metrics(result: FitResult) -> dict[str, float]:
    residual = np.asarray(result.residual, dtype=float)
    y_data = np.asarray(result.y_data, dtype=float)
    ss_res = float(np.sum(residual**2))
    y_mean = float(np.mean(y_data)) if y_data.size else 0.0
    ss_tot = float(np.sum((y_data - y_mean) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rms = math.sqrt(ss_res / max(residual.size, 1))
    max_abs = float(np.nanmax(np.abs(residual))) if residual.size else 0.0
    return {
        "r_squared": r2,
        "rms_residual": rms,
        "max_abs_residual": max_abs,
    }


def _format_optional_float(value: float | None) -> str:
    return "none" if value is None else f"{float(value):.10g}"


def _markdown_escape(value: object) -> str:
    return str(value).replace("|", "\\|")


def _write_monte_carlo_error_csv(
    path: Path,
    result: MonteCarloResult,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "peak",
                "parameter",
                "best",
                "mean",
                "median",
                "std",
                "lower",
                "upper",
                "lower_error",
                "upper_error",
                "min",
                "max",
                "count",
            ],
        )
        writer.writeheader()
        for summary in result.summaries:
            writer.writerow(summary.to_dict())


def _write_monte_carlo_population_csv(
    path: Path,
    result: MonteCarloResult,
) -> None:
    labels = [
        component.label for component in result.best_record.result.components
    ]
    parameter_columns = [
        f"{safe_name(label)}_{parameter}"
        for label in labels
        for parameter in FIT_PARAMETER_NAMES
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "rank",
                "iteration",
                "retained",
                "chisq",
                "redchi",
                "success",
                "nfev",
                *parameter_columns,
            ],
        )
        writer.writeheader()
        for rank, record in enumerate(result.records, start=1):
            row: dict[str, object] = {
                "rank": rank,
                "iteration": record.index,
                "retained": record.retained,
                "chisq": record.chisq,
                "redchi": record.redchi,
                "success": record.result.success,
                "nfev": record.result.nfev,
            }
            by_label = {
                component.label: component
                for component in record.result.components
            }
            for label in labels:
                component = by_label.get(label)
                if component is None:
                    continue
                for parameter in FIT_PARAMETER_NAMES:
                    row[f"{safe_name(label)}_{parameter}"] = (
                        component_parameter_value(component, parameter)
                    )
            writer.writerow(row)


def _write_peak_csv(path: Path, components: list[PeakComponent]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "peak",
                "amplitude",
                "area",
                "center",
                "fwhm",
                "eta",
                "amplitude_constraint",
                "area_constraint",
                "center_constraint",
                "fwhm_constraint",
                "eta_constraint",
            ],
        )
        writer.writeheader()
        for component in components:
            row = component.to_dict()
            writer.writerow(
                {
                    "peak": component.label,
                    "amplitude": row["amplitude"],
                    "area": row["area"],
                    "center": row["center"],
                    "fwhm": row["fwhm"],
                    "eta": row["eta"],
                    "amplitude_constraint": component.constraints["amplitude"],
                    "area_constraint": component.constraints["area"],
                    "center_constraint": component.constraints["center"],
                    "fwhm_constraint": component.constraints["fwhm"],
                    "eta_constraint": component.constraints["eta"],
                }
            )


def write_uvvis_fit_csv(
    path: str | Path,
    dataset: UVVisDataset,
    components: list[PeakComponent],
    *,
    result: FitResult | None = None,
    fit_min: float | None = None,
    fit_max: float | None = None,
) -> Path:
    """Write a single CSV that carries fit metadata in a `#`-prefixed
    header block followed by a standard
    x/data/total_fit/residual/peak_<label> table.

    Companion to the .json session file; parseable by pandas with
    ``comment='#'`` or by simple custom parsers.
    """
    resolved = Path(path).expanduser().resolve()
    if resolved.suffix.lower() != ".csv":
        resolved = resolved.with_suffix(".csv")
    lines: list[str] = []
    lines.append("# saxshell_uvvis_fit_csv version=1")
    lines.append(f"# saved={datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"# source={dataset.source_path}")
    lines.append(f"# x_label={dataset.x_label}")
    lines.append(f"# y_label={dataset.y_label}")
    if fit_min is not None:
        lines.append(f"# fit_min={float(fit_min):.10g}")
    if fit_max is not None:
        lines.append(f"# fit_max={float(fit_max):.10g}")
    if result is not None:
        residual = np.asarray(result.residual, dtype=float)
        y_data = np.asarray(result.y_data, dtype=float)
        ss_res = float(np.sum(residual**2))
        y_mean = float(np.mean(y_data)) if y_data.size else 0.0
        ss_tot = float(np.sum((y_data - y_mean) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        rms = math.sqrt(ss_res / max(residual.size, 1))
        max_abs = float(np.nanmax(np.abs(residual))) if residual.size else 0.0
        lines.append(f"# fit_success={bool(result.success)}")
        message = str(result.message or "").replace("\n", " ").strip()
        lines.append(f"# fit_message={message}")
        lines.append(f"# fit_nfev={int(result.nfev)}")
        lines.append(f"# fit_chisq={result.chisq:.10g}")
        lines.append(f"# fit_redchi={result.redchi:.10g}")
        lines.append(f"# fit_r_squared={r2:.10g}")
        lines.append(f"# fit_rms_residual={rms:.10g}")
        lines.append(f"# fit_max_abs_residual={max_abs:.10g}")
    for component in components:
        locks = ",".join(
            name for name, value in component.locked.items() if value
        )
        constraints = ";".join(
            f"{name}:{value}"
            for name, value in component.constraints.items()
            if value
        )
        lines.append(
            f"# peak label={component.label}"
            f" amplitude={component.amplitude:.10g}"
            f" area={component.area:.10g}"
            f" center={component.center:.10g}"
            f" fwhm={component.fwhm:.10g}"
            f" eta={component.eta:.10g}"
            f" color={component.color}"
            f" locks=[{locks}]"
            f" constraints=[{constraints}]"
        )
    lines.append("#")
    if result is not None:
        x_values = np.asarray(result.x_fit, dtype=float)
        y_values = np.asarray(result.y_data, dtype=float)
        total_values = np.asarray(result.total, dtype=float)
        residual_values = np.asarray(result.residual, dtype=float)
        component_curves = result.component_curves
        peak_columns = list(result.components)
    else:
        x_values = np.asarray(dataset.x, dtype=float)
        y_values = np.asarray(dataset.y, dtype=float)
        component_curves, total_values = evaluate_components(
            x_values,
            components,
        )
        residual_values = np.zeros_like(x_values)
        peak_columns = list(components)
    header = ["x", "data", "total_fit", "residual"] + [
        f"peak_{component.label}" for component in peak_columns
    ]
    lines.append(",".join(header))
    for index in range(int(x_values.size)):
        row = [
            f"{float(x_values[index]):.10g}",
            f"{float(y_values[index]):.10g}",
            f"{float(total_values[index]):.10g}",
            f"{float(residual_values[index]):.10g}",
        ]
        for curve in component_curves:
            row.append(f"{float(np.asarray(curve)[index]):.10g}")
        lines.append(",".join(row))
    resolved.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return resolved


def _write_curve_csv(path: Path, result: FitResult) -> None:
    fieldnames = [
        "x",
        "data",
        "total_fit",
        "residual",
        *[f"peak_{component.label}" for component in result.components],
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, x_value in enumerate(result.x_fit):
            row = {
                "x": float(x_value),
                "data": float(result.y_data[index]),
                "total_fit": float(result.total[index]),
                "residual": float(result.residual[index]),
            }
            for component, curve in zip(
                result.components,
                result.component_curves,
                strict=False,
            ):
                row[f"peak_{component.label}"] = float(curve[index])
            writer.writerow(row)


PARAMETER_NAMES = FIT_PARAMETER_NAMES
read_uvvis_file = load_uvvis_file
default_peak_components = guess_initial_components
fit_session_dict = session_payload
