from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from saxshell.saxs._model_templates import (
    load_template_module,
    load_template_spec,
)
from saxshell.saxs.prefit import ClusterGeometryMetadataTable
from saxshell.saxs.stoichiometry import (
    StoichiometryEvaluation,
    StoichiometryTarget,
    build_stoichiometry_target,
    evaluate_weighted_stoichiometry,
    parse_stoich_label,
    stoich_sort_key,
)


@dataclass(slots=True)
class DreamSummary:
    bestfit_method: str
    bestfit_params: np.ndarray
    map_params: np.ndarray
    chain_mean_params: np.ndarray
    median_params: np.ndarray
    interval_low_values: np.ndarray
    interval_high_values: np.ndarray
    full_parameter_names: list[str]
    active_parameter_names: list[str]
    map_chain: int
    map_step: int
    chain_map_logps: np.ndarray
    posterior_filter_mode: str
    posterior_candidate_sample_count: int
    posterior_sample_count: int
    credible_interval_low: float
    credible_interval_high: float
    stoichiometry_target: StoichiometryTarget | None
    stoichiometry_evaluation: StoichiometryEvaluation | None
    stoichiometry_filter_enabled: bool
    stoichiometry_tolerance_percent: float | None
    run_dir: Path


@dataclass(slots=True)
class DreamModelPlotData:
    q_values: np.ndarray
    experimental_intensities: np.ndarray
    model_intensities: np.ndarray
    solvent_contribution: np.ndarray | None
    structure_factor_trace: np.ndarray | None
    bestfit_method: str
    template_name: str
    rmse: float
    mean_abs_residual: float
    r_squared: float


@dataclass(slots=True)
class DreamViolinPlotData:
    parameter_names: list[str]
    display_names: list[str]
    samples: np.ndarray
    mode: str
    sample_source: str
    sample_count: int
    weight_order: str


@dataclass(slots=True)
class _PosteriorView:
    filter_mode: str
    top_percent: float
    top_n: int
    sample_mask: np.ndarray
    samples_flat: np.ndarray
    log_ps_flat: np.ndarray
    map_params: np.ndarray
    chain_mean_params: np.ndarray
    median_params: np.ndarray
    interval_low_values: np.ndarray
    interval_high_values: np.ndarray
    map_chain: int
    map_step: int
    chain_map_logps: np.ndarray
    candidate_sample_count: int
    sample_count: int
    credible_interval_low: float
    credible_interval_high: float


class SAXSDreamResultsLoader:
    """Load DREAM posterior samples and expand them to full SAXS
    parameters."""

    def __init__(
        self,
        run_dir: str | Path,
        *,
        burnin_percent: int | None = None,
    ) -> None:
        self.run_dir = Path(run_dir).expanduser().resolve()
        metadata_path = self.run_dir / "dream_runtime_metadata.json"
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        settings = dict(self.metadata.get("settings", {}))
        self.burnin_percent = int(
            burnin_percent
            if burnin_percent is not None
            else settings.get("burnin_percent", 20)
        )
        self.sampled_params = self._normalize_sampled_params(
            np.load(self.run_dir / "dream_sampled_params.npy"),
            active_count=len(
                self.metadata.get("active_parameter_indices", [])
            ),
        )
        self.full_parameter_names = list(
            self.metadata.get("full_parameter_names", [])
        )
        self.active_indices = list(
            self.metadata.get("active_parameter_indices", [])
        )
        self.log_ps = self._normalize_log_ps(
            np.load(self.run_dir / "dream_log_ps.npy"),
            expected_shape=self.sampled_params.shape[:2],
        )
        self.fixed_parameter_values = np.asarray(
            self.metadata.get("fixed_parameter_values", []),
            dtype=float,
        )
        self.active_entries = list(
            self.metadata.get("active_parameter_entries", [])
        )
        self.parameter_map_entries = list(
            self.metadata.get("parameter_map", [])
        )
        self.active_parameter_names = [
            str(entry.get("param", "")) for entry in self.active_entries
        ]
        self.template_name = str(
            self.metadata.get("template_name", "")
        ).strip()
        self.q_values = np.asarray(
            self.metadata.get("q_values", []),
            dtype=float,
        )
        self.experimental_intensities = np.asarray(
            self.metadata.get("experimental_intensities", []),
            dtype=float,
        )
        self.theoretical_intensities = [
            np.asarray(values, dtype=float)
            for values in self.metadata.get("theoretical_intensities", [])
        ]
        self.solvent_intensities = np.asarray(
            self.metadata.get("solvent_intensities", []),
            dtype=float,
        )
        self.template_runtime_inputs = {
            str(name): np.asarray(values, dtype=float)
            for name, values in dict(
                self.metadata.get("template_runtime_inputs", {})
            ).items()
        }
        self.lmfit_extra_inputs = list(
            self.metadata.get("lmfit_extra_inputs", [])
        )
        raw_cluster_geometry_metadata = self.metadata.get(
            "cluster_geometry_metadata"
        )
        self.cluster_geometry_table = (
            ClusterGeometryMetadataTable.from_dict(
                raw_cluster_geometry_metadata
            )
            if isinstance(raw_cluster_geometry_metadata, dict)
            else None
        )
        self._expanded_samples_flat: np.ndarray | None = None
        self._posterior_view_cache: dict[
            tuple[object, ...], _PosteriorView
        ] = {}
        self._summary_cache: dict[tuple[object, ...], DreamSummary] = {}
        self._model_plot_cache: dict[
            tuple[object, ...], DreamModelPlotData
        ] = {}
        self._violin_data_cache: dict[
            tuple[object, ...], DreamViolinPlotData
        ] = {}
        self._parameter_entry_lookup = self._build_parameter_entry_lookup()
        self._stoichiometry_weight_map_cache: dict[
            tuple[str, ...], tuple[np.ndarray, np.ndarray, tuple[str, ...]]
        ] = {}
        self._stoichiometry_mask_cache: dict[
            tuple[object, ...], np.ndarray
        ] = {}
        self._apply_burnin()

    def _apply_burnin(self) -> None:
        n_iter = int(self.sampled_params.shape[1])
        burn_idx = int(n_iter * self.burnin_percent / 100)
        self.sampled_params = self.sampled_params[:, burn_idx:, :]
        self.log_ps = self.log_ps[:, burn_idx:]
        self.samples_flat = self.sampled_params.reshape(
            -1,
            self.sampled_params.shape[-1],
        )
        self.log_ps_flat = self.log_ps.reshape(-1)

    def expand_params(self, active_params: np.ndarray) -> np.ndarray:
        full = self.fixed_parameter_values.copy()
        full[np.asarray(self.active_indices, dtype=int)] = np.asarray(
            active_params,
            dtype=float,
        )
        return full

    def get_summary(
        self,
        *,
        bestfit_method: str = "map",
        posterior_filter_mode: str = "all_post_burnin",
        posterior_top_percent: float = 10.0,
        posterior_top_n: int = 500,
        credible_interval_low: float = 16.0,
        credible_interval_high: float = 84.0,
        stoichiometry_target_elements_text: str = "",
        stoichiometry_target_ratio_text: str = "",
        stoichiometry_filter_enabled: bool = False,
        stoichiometry_tolerance_percent: float = 5.0,
    ) -> DreamSummary:
        view = self._posterior_view(
            filter_mode=posterior_filter_mode,
            top_percent=posterior_top_percent,
            top_n=posterior_top_n,
            credible_interval_low=credible_interval_low,
            credible_interval_high=credible_interval_high,
            stoichiometry_target_elements_text=stoichiometry_target_elements_text,
            stoichiometry_target_ratio_text=stoichiometry_target_ratio_text,
            stoichiometry_filter_enabled=stoichiometry_filter_enabled,
            stoichiometry_tolerance_percent=stoichiometry_tolerance_percent,
        )
        stoichiometry_target = self._resolve_stoichiometry_target(
            stoichiometry_target_elements_text=stoichiometry_target_elements_text,
            stoichiometry_target_ratio_text=stoichiometry_target_ratio_text,
            required=False,
        )
        cache_key = (
            str(bestfit_method),
            str(view.filter_mode),
            round(float(view.top_percent), 6),
            int(view.top_n),
            round(float(view.credible_interval_low), 6),
            round(float(view.credible_interval_high), 6),
            (
                tuple(stoichiometry_target.elements)
                if stoichiometry_target is not None
                else ()
            ),
            (
                tuple(stoichiometry_target.ratio)
                if stoichiometry_target is not None
                else ()
            ),
            bool(stoichiometry_filter_enabled),
            round(float(stoichiometry_tolerance_percent), 6),
        )
        cached = self._summary_cache.get(cache_key)
        if cached is not None:
            return cached
        best_active = self._select_best_params(bestfit_method, view)
        best_full = self.expand_params(best_active)
        summary = DreamSummary(
            bestfit_method=bestfit_method,
            bestfit_params=best_full,
            map_params=self.expand_params(view.map_params),
            chain_mean_params=self.expand_params(view.chain_mean_params),
            median_params=self.expand_params(view.median_params),
            interval_low_values=self.expand_params(view.interval_low_values),
            interval_high_values=self.expand_params(view.interval_high_values),
            full_parameter_names=self.full_parameter_names,
            active_parameter_names=self.active_parameter_names,
            map_chain=int(view.map_chain),
            map_step=int(view.map_step),
            chain_map_logps=np.asarray(view.chain_map_logps, dtype=float),
            posterior_filter_mode=view.filter_mode,
            posterior_candidate_sample_count=int(view.candidate_sample_count),
            posterior_sample_count=int(view.sample_count),
            credible_interval_low=float(view.credible_interval_low),
            credible_interval_high=float(view.credible_interval_high),
            stoichiometry_target=stoichiometry_target,
            stoichiometry_evaluation=self._evaluate_parameter_stoichiometry(
                best_full,
                stoichiometry_target,
            ),
            stoichiometry_filter_enabled=bool(stoichiometry_filter_enabled),
            stoichiometry_tolerance_percent=(
                float(stoichiometry_tolerance_percent)
                if stoichiometry_target is not None
                else None
            ),
            run_dir=self.run_dir,
        )
        self._summary_cache[cache_key] = summary
        return summary

    def build_model_fit_data(
        self,
        *,
        bestfit_method: str = "map",
        posterior_filter_mode: str = "all_post_burnin",
        posterior_top_percent: float = 10.0,
        posterior_top_n: int = 500,
        credible_interval_low: float = 16.0,
        credible_interval_high: float = 84.0,
        stoichiometry_target_elements_text: str = "",
        stoichiometry_target_ratio_text: str = "",
        stoichiometry_filter_enabled: bool = False,
        stoichiometry_tolerance_percent: float = 5.0,
    ) -> DreamModelPlotData:
        cache_key = (
            str(bestfit_method),
            str(posterior_filter_mode),
            round(float(posterior_top_percent), 6),
            int(posterior_top_n),
            str(stoichiometry_target_elements_text).strip(),
            str(stoichiometry_target_ratio_text).strip(),
            bool(stoichiometry_filter_enabled),
            round(float(stoichiometry_tolerance_percent), 6),
        )
        cached = self._model_plot_cache.get(cache_key)
        if cached is not None:
            return cached
        summary = self.get_summary(
            bestfit_method=bestfit_method,
            posterior_filter_mode=posterior_filter_mode,
            posterior_top_percent=posterior_top_percent,
            posterior_top_n=posterior_top_n,
            credible_interval_low=credible_interval_low,
            credible_interval_high=credible_interval_high,
            stoichiometry_target_elements_text=stoichiometry_target_elements_text,
            stoichiometry_target_ratio_text=stoichiometry_target_ratio_text,
            stoichiometry_filter_enabled=stoichiometry_filter_enabled,
            stoichiometry_tolerance_percent=stoichiometry_tolerance_percent,
        )
        model_module = load_template_module(self.template_name)
        template_spec = load_template_spec(self.template_name)
        model_function = getattr(model_module, template_spec.lmfit_model_name)
        params = {
            name: float(summary.bestfit_params[index])
            for index, name in enumerate(self.full_parameter_names)
        }
        extra_inputs = [
            np.asarray(self.template_runtime_inputs[name], dtype=float)
            for name in self.lmfit_extra_inputs
        ]
        model_intensities = np.asarray(
            model_function(
                self.q_values,
                self.solvent_intensities,
                self.theoretical_intensities,
                *extra_inputs,
                **params,
            ),
            dtype=float,
        )
        solvent_contribution = self._evaluate_solvent_contribution(
            model_function,
            params=params,
            extra_inputs=extra_inputs,
        )
        structure_factor_trace = self._evaluate_structure_factor_trace(
            model_module,
            params=params,
            extra_inputs=extra_inputs,
        )
        residuals = np.asarray(
            model_intensities - self.experimental_intensities,
            dtype=float,
        )
        rmse = float(np.sqrt(np.mean(residuals**2)))
        mean_abs_residual = float(np.mean(np.abs(residuals)))
        experimental_mean = float(np.mean(self.experimental_intensities))
        total_sum_squares = float(
            np.sum((self.experimental_intensities - experimental_mean) ** 2)
        )
        residual_sum_squares = float(np.sum(residuals**2))
        r_squared = (
            float(1.0 - (residual_sum_squares / total_sum_squares))
            if total_sum_squares > 0.0
            else 1.0
        )
        plot_data = DreamModelPlotData(
            q_values=self.q_values,
            experimental_intensities=self.experimental_intensities,
            model_intensities=model_intensities,
            solvent_contribution=solvent_contribution,
            structure_factor_trace=structure_factor_trace,
            bestfit_method=bestfit_method,
            template_name=self.template_name,
            rmse=rmse,
            mean_abs_residual=mean_abs_residual,
            r_squared=r_squared,
        )
        self._model_plot_cache[cache_key] = plot_data
        return plot_data

    def build_violin_data(
        self,
        *,
        mode: str = "varying_parameters",
        posterior_filter_mode: str = "all_post_burnin",
        posterior_top_percent: float = 10.0,
        posterior_top_n: int = 500,
        credible_interval_low: float = 16.0,
        credible_interval_high: float = 84.0,
        sample_source: str = "filtered_posterior",
        weight_order: str = "weight_index",
        stoichiometry_target_elements_text: str = "",
        stoichiometry_target_ratio_text: str = "",
        stoichiometry_filter_enabled: bool = False,
        stoichiometry_tolerance_percent: float = 5.0,
    ) -> DreamViolinPlotData:
        cache_key = (
            str(mode),
            str(posterior_filter_mode),
            round(float(posterior_top_percent), 6),
            int(posterior_top_n),
            str(sample_source),
            str(weight_order),
            str(stoichiometry_target_elements_text).strip(),
            str(stoichiometry_target_ratio_text).strip(),
            bool(stoichiometry_filter_enabled),
            round(float(stoichiometry_tolerance_percent), 6),
        )
        cached = self._violin_data_cache.get(cache_key)
        if cached is not None:
            return cached
        view = self._posterior_view(
            filter_mode=posterior_filter_mode,
            top_percent=posterior_top_percent,
            top_n=posterior_top_n,
            credible_interval_low=credible_interval_low,
            credible_interval_high=credible_interval_high,
            stoichiometry_target_elements_text=stoichiometry_target_elements_text,
            stoichiometry_target_ratio_text=stoichiometry_target_ratio_text,
            stoichiometry_filter_enabled=stoichiometry_filter_enabled,
            stoichiometry_tolerance_percent=stoichiometry_tolerance_percent,
        )
        active_samples = self._active_samples_for_source(
            view,
            sample_source=sample_source,
        )
        if mode == "varying_parameters":
            samples = np.asarray(active_samples, dtype=float)
            names = list(self.active_parameter_names)
        else:
            full_samples = self._full_violin_samples(
                view,
                active_samples=active_samples,
                sample_source=sample_source,
            )
            if mode == "all_parameters":
                samples = full_samples
                names = list(self.full_parameter_names)
            elif mode == "weights_only":
                names, samples = self._select_columns(
                    full_samples,
                    include=lambda name: name.startswith("w"),
                )
            elif mode == "effective_radii_only":
                names, samples = self._select_columns(
                    full_samples,
                    include=self._is_effective_radius_parameter,
                )
            elif mode == "additional_parameters_only":
                names, samples = self._select_columns(
                    full_samples,
                    include=lambda name: (
                        not name.startswith("w")
                        and not self._is_effective_radius_parameter(name)
                    ),
                )
            elif mode == "fit_parameters":
                names, samples = self._select_columns(
                    full_samples,
                    include=lambda name: not name.startswith("w"),
                )
            else:
                raise ValueError(
                    "Unknown DREAM violin mode: "
                    f"{mode}. Expected one of "
                    "'varying_parameters', 'all_parameters', "
                    "'weights_only', 'effective_radii_only', "
                    "'additional_parameters_only', or 'fit_parameters'."
                )
        names, samples = self._ordered_violin_columns(
            names,
            samples,
            weight_order=weight_order,
        )
        violin_data = DreamViolinPlotData(
            parameter_names=names,
            display_names=[
                self._parameter_display_name(name) for name in names
            ],
            samples=np.asarray(samples, dtype=float),
            mode=mode,
            sample_source=sample_source,
            sample_count=int(np.asarray(samples).shape[0]),
            weight_order=weight_order,
        )
        self._violin_data_cache[cache_key] = violin_data
        return violin_data

    def save_statistics_report(
        self,
        output_path: str | Path,
        *,
        bestfit_method: str = "map",
        posterior_filter_mode: str = "all_post_burnin",
        posterior_top_percent: float = 10.0,
        posterior_top_n: int = 500,
        credible_interval_low: float = 16.0,
        credible_interval_high: float = 84.0,
        violin_parameter_mode: str = "varying_parameters",
        violin_sample_source: str = "filtered_posterior",
        stoichiometry_target_elements_text: str = "",
        stoichiometry_target_ratio_text: str = "",
        stoichiometry_filter_enabled: bool = False,
        stoichiometry_tolerance_percent: float = 5.0,
    ) -> Path:
        summary = self.get_summary(
            bestfit_method=bestfit_method,
            posterior_filter_mode=posterior_filter_mode,
            posterior_top_percent=posterior_top_percent,
            posterior_top_n=posterior_top_n,
            credible_interval_low=credible_interval_low,
            credible_interval_high=credible_interval_high,
            stoichiometry_target_elements_text=stoichiometry_target_elements_text,
            stoichiometry_target_ratio_text=stoichiometry_target_ratio_text,
            stoichiometry_filter_enabled=stoichiometry_filter_enabled,
            stoichiometry_tolerance_percent=stoichiometry_tolerance_percent,
        )
        lines = [
            f"Run directory: {self.run_dir}",
            f"Burn-in (%): {self.burnin_percent}",
            f"Best-fit method: {bestfit_method}",
            f"Posterior filter: {summary.posterior_filter_mode}",
            f"Posterior samples kept: {summary.posterior_sample_count}",
            (
                "Credible interval (%): "
                f"{summary.credible_interval_low:g} - "
                f"{summary.credible_interval_high:g}"
            ),
            f"Violin data mode: {violin_parameter_mode}",
            f"Violin sample source: {violin_sample_source}",
            "",
            "Posterior summary:",
        ]
        if summary.stoichiometry_target is not None:
            lines.insert(
                6,
                (
                    "Stoichiometry target: "
                    + ":".join(summary.stoichiometry_target.elements)
                    + " = "
                    + ":".join(
                        f"{value:g}"
                        for value in summary.stoichiometry_target.ratio
                    )
                ),
            )
            lines.insert(
                7,
                (
                    "Stoichiometry filter: "
                    + ("on" if summary.stoichiometry_filter_enabled else "off")
                ),
            )
            if summary.stoichiometry_filter_enabled:
                lines.insert(
                    8,
                    (
                        "Stoichiometry tolerance (%): "
                        f"{float(summary.stoichiometry_tolerance_percent or 0.0):g}"
                    ),
                )
        if posterior_filter_mode == "top_percent_logp":
            lines.insert(
                4,
                (
                    "Posterior filter detail: "
                    f"top {posterior_top_percent:g}% by log-posterior"
                ),
            )
        elif posterior_filter_mode == "top_n_logp":
            lines.insert(
                4,
                (
                    "Posterior filter detail: "
                    f"top {posterior_top_n} samples by log-posterior"
                ),
            )
        for index, name in enumerate(summary.full_parameter_names):
            lines.append(
                f"  {name}: "
                f"selected={summary.bestfit_params[index]:.6g}, "
                f"MAP={summary.map_params[index]:.6g}, "
                f"chain_mean={summary.chain_mean_params[index]:.6g}, "
                f"median={summary.median_params[index]:.6g}, "
                f"p{summary.credible_interval_low:g}="
                f"{summary.interval_low_values[index]:.6g}, "
                f"p{summary.credible_interval_high:g}="
                f"{summary.interval_high_values[index]:.6g}"
            )
        report_path = Path(output_path)
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return report_path

    def _evaluate_solvent_contribution(
        self,
        model_function,
        *,
        params: dict[str, float],
        extra_inputs: list[np.ndarray],
    ) -> np.ndarray | None:
        solvent_data = np.asarray(self.solvent_intensities, dtype=float)
        if solvent_data.size == 0 or not np.any(np.abs(solvent_data) > 0.0):
            return None
        isolated_params = dict(params)
        if "offset" in isolated_params:
            isolated_params["offset"] = 0.0
        zero_model_data = [
            np.zeros_like(np.asarray(component, dtype=float))
            for component in self.theoretical_intensities
        ]
        contribution = model_function(
            self.q_values,
            solvent_data,
            zero_model_data,
            *extra_inputs,
            **isolated_params,
        )
        return np.asarray(contribution, dtype=float)

    def _evaluate_structure_factor_trace(
        self,
        model_module,
        *,
        params: dict[str, float],
        extra_inputs: list[np.ndarray],
    ) -> np.ndarray | None:
        structure_factor_function = getattr(
            model_module,
            "structure_factor_profile",
            None,
        )
        if structure_factor_function is None:
            return None
        try:
            structure_factor = structure_factor_function(
                self.q_values,
                self.solvent_intensities,
                self.theoretical_intensities,
                *extra_inputs,
                **params,
            )
        except Exception:
            return None
        structure_factor_array = np.asarray(structure_factor, dtype=float)
        if structure_factor_array.shape != self.q_values.shape:
            return None
        if not np.all(np.isfinite(structure_factor_array)):
            return None
        return structure_factor_array

    def _select_best_params(
        self,
        bestfit_method: str,
        view: _PosteriorView,
    ) -> np.ndarray:
        if bestfit_method not in {"map", "chain_mean", "median"}:
            raise ValueError(
                "bestfit_method must be 'map', 'chain_mean', or 'median'."
            )
        if bestfit_method == "map":
            return view.map_params
        if bestfit_method == "chain_mean":
            return view.chain_mean_params
        return view.median_params

    def _posterior_view(
        self,
        *,
        filter_mode: str,
        top_percent: float,
        top_n: int,
        credible_interval_low: float,
        credible_interval_high: float,
        stoichiometry_target_elements_text: str = "",
        stoichiometry_target_ratio_text: str = "",
        stoichiometry_filter_enabled: bool = False,
        stoichiometry_tolerance_percent: float = 5.0,
    ) -> _PosteriorView:
        low, high = self._normalize_interval_bounds(
            credible_interval_low,
            credible_interval_high,
        )
        stoichiometry_target = self._resolve_stoichiometry_target(
            stoichiometry_target_elements_text=stoichiometry_target_elements_text,
            stoichiometry_target_ratio_text=stoichiometry_target_ratio_text,
            required=stoichiometry_filter_enabled,
        )
        key = (
            filter_mode,
            round(float(top_percent), 6),
            int(top_n),
            round(low, 6),
            round(high, 6),
            (
                tuple(stoichiometry_target.elements)
                if stoichiometry_target is not None
                else ()
            ),
            (
                tuple(stoichiometry_target.ratio)
                if stoichiometry_target is not None
                else ()
            ),
            bool(stoichiometry_filter_enabled),
            round(float(stoichiometry_tolerance_percent), 6),
        )
        cached = self._posterior_view_cache.get(key)
        if cached is not None:
            return cached

        available_mask: np.ndarray | None = None
        candidate_sample_count = int(self.log_ps.size)
        if stoichiometry_filter_enabled:
            if stoichiometry_target is None:
                raise ValueError(
                    "Enter stoichiometry target elements and a target ratio "
                    "before enabling stoichiometry filtering."
                )
            available_mask = self._stoichiometry_filter_mask(
                stoichiometry_target,
                tolerance_percent=stoichiometry_tolerance_percent,
            )
            candidate_sample_count = int(np.count_nonzero(available_mask))
            if candidate_sample_count == 0:
                raise ValueError(
                    "No DREAM samples satisfy the active stoichiometry filter."
                )

        sample_mask = self._posterior_mask(
            filter_mode=filter_mode,
            top_percent=top_percent,
            top_n=top_n,
            available_mask=available_mask,
        )
        samples_flat = np.asarray(
            self.sampled_params[sample_mask], dtype=float
        )
        if samples_flat.ndim == 1:
            samples_flat = samples_flat.reshape(1, -1)
        log_ps_flat = np.asarray(self.log_ps[sample_mask], dtype=float)
        if samples_flat.size == 0 or log_ps_flat.size == 0:
            raise ValueError(
                "The selected posterior filter removed all DREAM samples."
            )

        chain_indices, step_indices = np.nonzero(sample_mask)
        best_flat_index = int(np.argmax(log_ps_flat))
        map_chain = int(chain_indices[best_flat_index])
        map_step = int(step_indices[best_flat_index])
        map_params = np.asarray(
            self.sampled_params[map_chain, map_step, :],
            dtype=float,
        )

        chain_map_logps = np.full(self.sampled_params.shape[0], np.nan)
        chain_map_params: list[np.ndarray] = []
        for chain_index in range(self.sampled_params.shape[0]):
            chain_mask = sample_mask[chain_index]
            if not np.any(chain_mask):
                continue
            kept_steps = np.flatnonzero(chain_mask)
            kept_log_ps = self.log_ps[chain_index, chain_mask]
            best_step_index = int(np.argmax(kept_log_ps))
            chain_map_logps[chain_index] = float(kept_log_ps[best_step_index])
            chain_map_params.append(
                np.asarray(
                    self.sampled_params[
                        chain_index,
                        kept_steps[best_step_index],
                        :,
                    ],
                    dtype=float,
                )
            )
        if not chain_map_params:
            raise ValueError(
                "The selected posterior filter removed all chain-specific "
                "MAP samples."
            )
        chain_mean_params = np.mean(np.asarray(chain_map_params), axis=0)
        median_params = np.median(samples_flat, axis=0)
        interval_low_values, interval_high_values = np.percentile(
            samples_flat,
            [low, high],
            axis=0,
        )

        view = _PosteriorView(
            filter_mode=filter_mode,
            top_percent=float(top_percent),
            top_n=int(top_n),
            sample_mask=sample_mask,
            samples_flat=samples_flat,
            log_ps_flat=log_ps_flat,
            map_params=map_params,
            chain_mean_params=np.asarray(chain_mean_params, dtype=float),
            median_params=np.asarray(median_params, dtype=float),
            interval_low_values=np.asarray(interval_low_values, dtype=float),
            interval_high_values=np.asarray(interval_high_values, dtype=float),
            map_chain=map_chain,
            map_step=map_step,
            chain_map_logps=chain_map_logps,
            candidate_sample_count=candidate_sample_count,
            sample_count=int(samples_flat.shape[0]),
            credible_interval_low=float(low),
            credible_interval_high=float(high),
        )
        self._posterior_view_cache[key] = view
        return view

    def _expanded_flat_samples(self) -> np.ndarray:
        if self._expanded_samples_flat is None:
            self._expanded_samples_flat = np.asarray(
                [self.expand_params(sample) for sample in self.samples_flat],
                dtype=float,
            )
        return self._expanded_samples_flat

    def _expand_sample_matrix(self, samples: np.ndarray) -> np.ndarray:
        samples = np.asarray(samples, dtype=float)
        if samples.ndim == 1:
            samples = samples.reshape(1, -1)
        return np.asarray(
            [self.expand_params(sample) for sample in samples],
            dtype=float,
        )

    def _active_samples_for_source(
        self,
        view: _PosteriorView,
        *,
        sample_source: str,
    ) -> np.ndarray:
        if sample_source == "filtered_posterior":
            return np.asarray(view.samples_flat, dtype=float)
        if sample_source == "map_chain_only":
            chain_mask = view.sample_mask[view.map_chain]
            chain_samples = np.asarray(
                self.sampled_params[view.map_chain, chain_mask, :],
                dtype=float,
            )
            if chain_samples.ndim == 1:
                chain_samples = chain_samples.reshape(1, -1)
            return chain_samples
        raise ValueError(
            "Unknown DREAM violin sample source: "
            f"{sample_source}. Expected 'filtered_posterior' or "
            "'map_chain_only'."
        )

    def _full_violin_samples(
        self,
        view: _PosteriorView,
        *,
        active_samples: np.ndarray,
        sample_source: str,
    ) -> np.ndarray:
        if sample_source == "filtered_posterior":
            flat_mask = np.asarray(view.sample_mask, dtype=bool).reshape(-1)
            return np.asarray(
                self._expanded_flat_samples()[flat_mask],
                dtype=float,
            )
        return self._expand_sample_matrix(active_samples)

    def _posterior_mask(
        self,
        *,
        filter_mode: str,
        top_percent: float,
        top_n: int,
        available_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        if available_mask is None:
            available = np.ones_like(self.log_ps, dtype=bool)
        else:
            available = np.asarray(available_mask, dtype=bool)
            if available.shape != self.log_ps.shape:
                raise ValueError(
                    "Stoichiometry filter mask shape does not match the DREAM samples."
                )
        if filter_mode == "all_post_burnin":
            return available
        if filter_mode not in {"top_percent_logp", "top_n_logp"}:
            raise ValueError(
                "posterior_filter_mode must be 'all_post_burnin', "
                "'top_percent_logp', or 'top_n_logp'."
            )
        flat_log_ps = np.asarray(self.log_ps.reshape(-1), dtype=float)
        available_flat = np.asarray(available.reshape(-1), dtype=bool)
        if flat_log_ps.size == 0:
            raise ValueError("No DREAM samples were found after burn-in.")
        candidate_indices = np.flatnonzero(available_flat)
        if candidate_indices.size == 0:
            return np.zeros_like(self.log_ps, dtype=bool)
        candidate_log_ps = flat_log_ps[candidate_indices]
        sorted_positions = np.argsort(candidate_log_ps)[::-1]
        if filter_mode == "top_percent_logp":
            keep_count = int(
                np.ceil(
                    candidate_indices.size
                    * max(min(float(top_percent), 100.0), 0.1)
                    / 100.0
                )
            )
        else:
            keep_count = int(top_n)
        keep_count = max(1, min(keep_count, candidate_indices.size))
        mask = np.zeros(flat_log_ps.size, dtype=bool)
        selected_indices = candidate_indices[sorted_positions[:keep_count]]
        mask[selected_indices] = True
        return mask.reshape(self.log_ps.shape)

    def _resolve_stoichiometry_target(
        self,
        *,
        stoichiometry_target_elements_text: str,
        stoichiometry_target_ratio_text: str,
        required: bool = False,
    ) -> StoichiometryTarget | None:
        if (
            not str(stoichiometry_target_elements_text).strip()
            and not str(stoichiometry_target_ratio_text).strip()
        ):
            return None
        if not required and (
            not str(stoichiometry_target_elements_text).strip()
            or not str(stoichiometry_target_ratio_text).strip()
        ):
            return None
        return build_stoichiometry_target(
            stoichiometry_target_elements_text,
            stoichiometry_target_ratio_text,
        )

    def _stoichiometry_weight_mapping(
        self,
        target: StoichiometryTarget,
    ) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
        cached = self._stoichiometry_weight_map_cache.get(target.elements)
        if cached is not None:
            return cached
        parameter_indices: list[int] = []
        count_rows: list[list[float]] = []
        structures: list[str] = []
        for index, parameter_name in enumerate(self.full_parameter_names):
            if not str(parameter_name).startswith("w"):
                continue
            entry = self._parameter_entry_lookup.get(str(parameter_name), {})
            structure = str(entry.get("structure", "")).strip()
            if not structure:
                continue
            counts = parse_stoich_label(structure)
            row = [
                float(counts.get(element, 0)) for element in target.elements
            ]
            if not any(value > 0.0 for value in row):
                continue
            parameter_indices.append(index)
            count_rows.append(row)
            structures.append(structure)
        mapping = (
            np.asarray(parameter_indices, dtype=int),
            np.asarray(count_rows, dtype=float),
            tuple(structures),
        )
        self._stoichiometry_weight_map_cache[target.elements] = mapping
        return mapping

    def _evaluate_parameter_stoichiometry(
        self,
        params_full: np.ndarray,
        target: StoichiometryTarget | None,
    ) -> StoichiometryEvaluation | None:
        if target is None:
            return None
        parameter_indices, _count_rows, structures = (
            self._stoichiometry_weight_mapping(target)
        )
        if parameter_indices.size == 0 or not structures:
            return None
        full_values = np.asarray(params_full, dtype=float)
        weights = np.maximum(full_values[parameter_indices], 0.0)
        return evaluate_weighted_stoichiometry(
            zip(structures, weights.tolist(), strict=False),
            target,
        )

    def _stoichiometry_filter_mask(
        self,
        target: StoichiometryTarget,
        *,
        tolerance_percent: float,
    ) -> np.ndarray:
        key = (
            tuple(target.elements),
            tuple(target.ratio),
            round(float(tolerance_percent), 6),
        )
        cached = self._stoichiometry_mask_cache.get(key)
        if cached is not None:
            return cached

        parameter_indices, count_matrix, _structures = (
            self._stoichiometry_weight_mapping(target)
        )
        if parameter_indices.size == 0 or count_matrix.size == 0:
            mask = np.zeros_like(self.log_ps, dtype=bool)
            self._stoichiometry_mask_cache[key] = mask
            return mask

        expanded_samples = self._expanded_flat_samples()
        weights = np.maximum(expanded_samples[:, parameter_indices], 0.0)
        totals = np.asarray(weights @ count_matrix, dtype=float)
        if totals.ndim == 1:
            totals = totals.reshape(-1, 1)
        reference = np.asarray(totals[:, 0], dtype=float)
        valid = np.isfinite(reference) & (reference > 0.0)
        if totals.shape[1] <= 1:
            mask = valid.reshape(self.log_ps.shape)
            self._stoichiometry_mask_cache[key] = mask
            return mask

        normalized = np.zeros_like(totals, dtype=float)
        normalized[valid] = totals[valid] / reference[valid, np.newaxis]
        target_ratio = np.asarray(target.normalized_ratio, dtype=float)
        deviations = np.abs(normalized[:, 1:] - target_ratio[1:])
        deviations = deviations / target_ratio[1:] * 100.0
        max_deviation = np.max(deviations, axis=1)
        valid &= max_deviation <= float(max(tolerance_percent, 0.0))
        mask = valid.reshape(self.log_ps.shape)
        self._stoichiometry_mask_cache[key] = mask
        return mask

    @staticmethod
    def _normalize_interval_bounds(
        low: float,
        high: float,
    ) -> tuple[float, float]:
        low_value = float(max(0.0, min(low, 100.0)))
        high_value = float(max(0.0, min(high, 100.0)))
        if high_value <= low_value:
            if low_value >= 99.9:
                low_value = 98.9
                high_value = 99.9
            else:
                high_value = min(100.0, low_value + 1.0)
        return low_value, high_value

    def _select_columns(
        self,
        full_samples: np.ndarray,
        *,
        include,
    ) -> tuple[list[str], np.ndarray]:
        indices = [
            index
            for index, name in enumerate(self.full_parameter_names)
            if include(name)
        ]
        names = [self.full_parameter_names[index] for index in indices]
        if not indices:
            return names, np.zeros((full_samples.shape[0], 0), dtype=float)
        return names, full_samples[:, indices]

    def _ordered_violin_columns(
        self,
        parameter_names: list[str],
        sample_matrix: np.ndarray,
        *,
        weight_order: str,
    ) -> tuple[list[str], np.ndarray]:
        if weight_order != "structure_order":
            return parameter_names, np.asarray(sample_matrix, dtype=float)

        samples = np.asarray(sample_matrix, dtype=float)
        if samples.ndim == 1:
            samples = samples.reshape(1, -1)
        if not parameter_names or samples.shape[1] == 0:
            return parameter_names, samples

        weight_positions = [
            index
            for index, name in enumerate(parameter_names)
            if name.startswith("w")
        ]
        if len(weight_positions) <= 1:
            return parameter_names, samples

        ordered_weight_positions = sorted(
            weight_positions,
            key=lambda index: self._weight_parameter_sort_key(
                parameter_names[index],
                index,
            ),
        )
        reordered_indices = list(range(len(parameter_names)))
        for target_position, source_position in zip(
            weight_positions,
            ordered_weight_positions,
            strict=True,
        ):
            reordered_indices[target_position] = source_position
        ordered_names = [parameter_names[index] for index in reordered_indices]
        return ordered_names, samples[:, reordered_indices]

    def _weight_parameter_sort_key(
        self,
        parameter_name: str,
        original_index: int,
    ) -> tuple[object, ...]:
        entry = self._parameter_entry_lookup.get(parameter_name, {})
        structure = str(entry.get("structure", "")).strip()
        motif = str(entry.get("motif", "")).strip()
        return (
            stoich_sort_key(structure) if structure else (9, parameter_name),
            motif,
            parameter_name,
            int(original_index),
        )

    def _parameter_display_name(self, parameter_name: str) -> str:
        if not parameter_name.startswith("w"):
            return parameter_name
        entry = self._parameter_entry_lookup.get(parameter_name, {})
        structure = str(entry.get("structure", "")).strip()
        if not structure:
            return parameter_name
        return f"{parameter_name} ({structure})"

    def _build_parameter_entry_lookup(self) -> dict[str, dict[str, object]]:
        lookup: dict[str, dict[str, object]] = {}
        for entry in self.parameter_map_entries:
            name = str(entry.get("param", "")).strip()
            if name and name not in lookup:
                lookup[name] = dict(entry)
        for entry in self.active_entries:
            name = str(entry.get("param", "")).strip()
            if name and name not in lookup:
                lookup[name] = dict(entry)
        return lookup

    @staticmethod
    def _is_effective_radius_parameter(parameter_name: str) -> bool:
        name = str(parameter_name).strip()
        return bool(
            name == "eff_r"
            or name.startswith("r_eff_")
            or name.startswith("a_eff_")
            or name.startswith("b_eff_")
            or name.startswith("c_eff_")
        )

    @staticmethod
    def _normalize_sampled_params(
        raw_sampled_params: np.ndarray,
        *,
        active_count: int,
    ) -> np.ndarray:
        sampled_params = np.asarray(raw_sampled_params, dtype=float)
        if sampled_params.ndim == 3:
            return sampled_params
        if sampled_params.ndim == 2:
            if active_count <= 1:
                return sampled_params[..., np.newaxis]
            return sampled_params[np.newaxis, ...]
        if sampled_params.ndim == 1:
            return sampled_params.reshape(1, -1, 1)
        raise ValueError(
            "Unsupported DREAM sampled_params array shape: "
            f"{sampled_params.shape}"
        )

    @staticmethod
    def _normalize_log_ps(
        raw_log_ps: np.ndarray,
        *,
        expected_shape: tuple[int, int],
    ) -> np.ndarray:
        log_ps = np.asarray(raw_log_ps, dtype=float)
        if log_ps.ndim == 3 and log_ps.shape[-1] == 1:
            log_ps = log_ps[..., 0]
        if log_ps.ndim == 2:
            return log_ps
        if log_ps.ndim == 1 and expected_shape[0] == 1:
            return log_ps.reshape(1, -1)
        if log_ps.size == expected_shape[0] * expected_shape[1]:
            return log_ps.reshape(expected_shape)
        raise ValueError(
            "Unsupported DREAM log posterior array shape: " f"{log_ps.shape}"
        )


__all__ = [
    "DreamModelPlotData",
    "DreamSummary",
    "DreamViolinPlotData",
    "SAXSDreamResultsLoader",
]
