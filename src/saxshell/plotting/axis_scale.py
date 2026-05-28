from __future__ import annotations

import numpy as np


def axis_has_positive_values(axis, *, dimension: str) -> bool:
    """Return True when the axis has a finite positive plotted value."""
    if dimension not in {"x", "y"}:
        raise ValueError(f"Unsupported axis dimension: {dimension!r}")

    for line in axis.get_lines():
        if not line.get_visible():
            continue
        if dimension == "x":
            values = np.asarray(line.get_xdata(orig=False), dtype=float)
        else:
            values = np.asarray(line.get_ydata(orig=False), dtype=float)
        if np.any(np.isfinite(values) & (values > 0.0)):
            return True
    return False


def resolve_axis_scale(requested_log: bool, axis, *, dimension: str) -> str:
    if not requested_log:
        return "linear"
    if axis_has_positive_values(axis, dimension=dimension):
        return "log"
    return "linear"


def safe_set_axis_scale(axis, dimension: str, scale: str) -> str:
    """Apply an axis scale, falling back to linear when log scaling is
    unsafe."""
    if scale not in {"linear", "log"}:
        scale = "linear"
    if scale == "log":
        scale = resolve_axis_scale(True, axis, dimension=dimension)
    setter = axis.set_xscale if dimension == "x" else axis.set_yscale
    try:
        setter(scale)
    except (OverflowError, ValueError):
        try:
            setter("linear")
        except Exception:
            pass
        return "linear"
    return scale


def apply_axis_scales(
    axis,
    *,
    log_x: bool = False,
    log_y: bool = False,
) -> None:
    safe_set_axis_scale(
        axis,
        "x",
        resolve_axis_scale(log_x, axis, dimension="x"),
    )
    safe_set_axis_scale(
        axis,
        "y",
        resolve_axis_scale(log_y, axis, dimension="y"),
    )
