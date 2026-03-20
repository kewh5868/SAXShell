from __future__ import annotations

from .base import FrameRecord


def select_frames(
    frames: list[FrameRecord],
    start: int | None = None,
    stop: int | None = None,
    stride: int = 1,
    min_time_fs: float | None = None,
    post_cutoff_stride: int = 1,
) -> list[FrameRecord]:
    """Apply index-based and optional time-based filtering to frame
    records.

    Parameters
    ----------
    frames
        Parsed frame records.
    start
        Starting frame index in the list slice.
    stop
        Stopping frame index in the list slice.
    stride
        Step size for frame selection.
    min_time_fs
        If provided, keep only frames with time_fs >= min_time_fs.
        Frames with time_fs=None are excluded when a cutoff is applied.
    post_cutoff_stride
        Optional step size applied after the time cutoff filter has been
        applied. This makes it possible to keep every Nth frame from the
        cutoff-trimmed selection.

    Returns
    -------
    list[FrameRecord]
        Selected frame records.
    """
    if stride <= 0:
        raise ValueError("stride must be a positive integer.")
    if post_cutoff_stride <= 0:
        raise ValueError("post_cutoff_stride must be a positive integer.")

    selected = frames[slice(start, stop, stride)]

    if min_time_fs is None:
        return selected

    filtered = [
        frame
        for frame in selected
        if frame.time_fs is not None and frame.time_fs >= min_time_fs
    ]
    return filtered[::post_cutoff_stride]
