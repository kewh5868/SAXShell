from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = ["dot_product"]


def dot_product(a: Sequence[float], b: Sequence[float]) -> float:
    """Return the scalar dot product of two 1D vectors."""

    vector_a = np.asarray(a, dtype=float)
    vector_b = np.asarray(b, dtype=float)
    if vector_a.ndim != 1 or vector_b.ndim != 1:
        raise ValueError("dot_product expects 1D vectors.")
    if vector_a.shape != vector_b.shape:
        raise ValueError("dot_product expects vectors of the same length.")
    return float(np.dot(vector_a, vector_b))
