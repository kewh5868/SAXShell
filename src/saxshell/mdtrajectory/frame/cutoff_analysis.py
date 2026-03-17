from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cp2k_ener import CP2KEnergyData


@dataclass(slots=True)
class SteadyStateResult:
    """Suggested steady-state cutoff result."""

    cutoff_time_fs: float | None
    temp_target_k: float
    temp_tol_k: float
    window: int


class CP2KEnergyAnalyzer:
    """Analysis helpers for CP2K energy trajectories."""

    def __init__(self, energy_data: CP2KEnergyData) -> None:
        self.energy_data = energy_data

    def suggest_steady_state_cutoff(
        self,
        temp_target_k: float,
        temp_tol_k: float = 1.0,
        window: int = 10,
        kinetic_rel_std_max: float = 1.0e-3,
        potential_rel_std_max: float = 1.0e-3,
    ) -> SteadyStateResult:
        """Find the first time where temperature stays near the target
        and kinetic/potential fluctuations remain small across a
        consecutive window."""
        time_fs = self.energy_data.time_fs
        kinetic = self.energy_data.kinetic
        temperature = self.energy_data.temperature
        potential = self.energy_data.potential

        n_points = len(time_fs)
        if window <= 0:
            raise ValueError("window must be a positive integer.")
        if n_points < window:
            return SteadyStateResult(
                cutoff_time_fs=None,
                temp_target_k=temp_target_k,
                temp_tol_k=temp_tol_k,
                window=window,
            )

        for i in range(n_points - window + 1):
            temp_window = temperature[i : i + window]
            kin_window = kinetic[i : i + window]
            pot_window = potential[i : i + window]

            temp_ok = np.all(np.abs(temp_window - temp_target_k) <= temp_tol_k)
            kin_ok = self._relative_std_ok(
                values=kin_window,
                rel_std_max=kinetic_rel_std_max,
            )
            pot_ok = self._relative_std_ok(
                values=pot_window,
                rel_std_max=potential_rel_std_max,
            )

            if temp_ok and kin_ok and pot_ok:
                return SteadyStateResult(
                    cutoff_time_fs=float(time_fs[i]),
                    temp_target_k=temp_target_k,
                    temp_tol_k=temp_tol_k,
                    window=window,
                )

        return SteadyStateResult(
            cutoff_time_fs=None,
            temp_target_k=temp_target_k,
            temp_tol_k=temp_tol_k,
            window=window,
        )

    def _relative_std_ok(
        self,
        values: np.ndarray,
        rel_std_max: float,
        eps: float = 1.0e-12,
    ) -> bool:
        mean_abs = max(float(np.abs(np.mean(values))), eps)
        rel_std = float(np.std(values)) / mean_abs
        return rel_std <= rel_std_max
