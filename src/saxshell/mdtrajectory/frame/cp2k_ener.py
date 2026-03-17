from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(slots=True)
class CP2KEnergyData:
    """Container for CP2K .ener data."""

    filepath: Path
    step: np.ndarray
    time_fs: np.ndarray
    kinetic: np.ndarray
    temperature: np.ndarray
    potential: np.ndarray

    @classmethod
    def from_file(cls, filepath: str | Path) -> "CP2KEnergyData":
        path = Path(filepath)
        data = np.loadtxt(path, comments="#")

        if data.ndim != 2 or data.shape[1] < 5:
            raise ValueError(
                "CP2K .ener file must contain at least 5 numeric columns."
            )

        return cls(
            filepath=path,
            step=data[:, 0],
            time_fs=data[:, 1],
            kinetic=data[:, 2],
            temperature=data[:, 3],
            potential=data[:, 4],
        )

    @property
    def n_points(self) -> int:
        return int(len(self.time_fs))

    @property
    def time_min_fs(self) -> float:
        return float(np.min(self.time_fs))

    @property
    def time_max_fs(self) -> float:
        return float(np.max(self.time_fs))