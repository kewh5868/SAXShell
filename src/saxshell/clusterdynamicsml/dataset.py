from __future__ import annotations

import csv
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from saxshell.cluster import FrameClusterResult
from saxshell.clusterdynamics.dataset import (
    export_cluster_dynamics_colormap_csv,
    export_cluster_dynamics_lifetime_csv,
)
from saxshell.clusterdynamics.workflow import (
    ClusterDynamicsResult,
    ClusterDynamicsSelectionPreview,
    ClusterLifetimeSummary,
    ClusterSizeLifetimeSummary,
    _resolve_colormap_timestep_settings,
)
from saxshell.mdtrajectory.frame.cp2k_ener import CP2KEnergyData

from .workflow import (
    ClusterDynamicsMLPreview,
    ClusterDynamicsMLResult,
    ClusterDynamicsMLSAXSComparison,
    ClusterDynamicsMLTrainingObservation,
    ClusterStructureObservation,
    DebyeWallerPairEstimate,
    PredictedClusterCandidate,
    SAXSComponentWeight,
    _resolved_population_weights,
)

DATASET_VERSION = 1


@dataclass(slots=True)
class SavedClusterDynamicsMLDataset:
    dataset_file: Path
    written_files: tuple[Path, ...]


@dataclass(slots=True)
class LoadedClusterDynamicsMLDataset:
    dataset_file: Path
    result: ClusterDynamicsMLResult
    analysis_settings: dict[str, object]


def save_cluster_dynamicsai_dataset(
    result: ClusterDynamicsMLResult,
    output_file: str | Path,
    *,
    analysis_settings: dict[str, object] | None = None,
) -> SavedClusterDynamicsMLDataset:
    dataset_file = Path(output_file).expanduser().resolve()
    if dataset_file.suffix.lower() != ".json":
        dataset_file = dataset_file.with_suffix(".json")
    dataset_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": DATASET_VERSION,
        "analysis_settings": analysis_settings or {},
        "dynamics_result": _serialize_cluster_dynamics_result(
            result.dynamics_result
        ),
        "preview": _serialize_preview(result.preview),
        "structure_observations": [
            _serialize_structure_observation(entry)
            for entry in result.structure_observations
        ],
        "training_observations": [
            _serialize_training_observation(entry)
            for entry in result.training_observations
        ],
        "predictions": [
            _serialize_prediction_candidate(entry)
            for entry in result.predictions
        ],
        "debye_waller_estimates": [
            _serialize_debye_waller_pair_estimate(entry)
            for entry in result.debye_waller_estimates
        ],
        "saxs_comparison": _serialize_saxs_comparison(result.saxs_comparison),
        "max_observed_node_count": int(result.max_observed_node_count),
        "max_predicted_node_count": (
            None
            if result.max_predicted_node_count is None
            else int(result.max_predicted_node_count)
        ),
        "prediction_population_share_threshold": float(
            result.prediction_population_share_threshold
        ),
    }
    dataset_file.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )

    written_files = [dataset_file]
    written_files.append(
        export_cluster_dynamics_colormap_csv(
            result.dynamics_result,
            dataset_file.with_name(
                f"{dataset_file.stem}_cluster_distribution.csv"
            ),
        )
    )
    written_files.append(
        export_cluster_dynamics_lifetime_csv(
            result.dynamics_result,
            dataset_file.with_name(f"{dataset_file.stem}_lifetime.csv"),
        )
    )
    written_files.append(_write_prediction_csv(result, dataset_file))
    written_files.extend(_write_histogram_csvs(result, dataset_file))
    if result.saxs_comparison is not None:
        written_files.append(
            _write_saxs_csv(result.saxs_comparison, dataset_file)
        )
        written_files.extend(
            _write_saxs_component_profiles(
                result.saxs_comparison, dataset_file
            )
        )
    written_files.extend(_write_prediction_structures(result, dataset_file))
    return SavedClusterDynamicsMLDataset(
        dataset_file=dataset_file,
        written_files=tuple(written_files),
    )


def load_cluster_dynamicsai_dataset(
    dataset_file: str | Path,
) -> LoadedClusterDynamicsMLDataset:
    resolved_file = Path(dataset_file).expanduser().resolve()
    payload = json.loads(resolved_file.read_text(encoding="utf-8"))
    if int(payload.get("version", 0)) != DATASET_VERSION:
        raise ValueError(
            "This clusterdynamicsml dataset uses an unsupported format version."
        )
    result = ClusterDynamicsMLResult(
        dynamics_result=_deserialize_cluster_dynamics_result(
            payload.get("dynamics_result", {}),
            fallback_path=resolved_file,
        ),
        preview=_deserialize_preview(payload.get("preview", {})),
        structure_observations=tuple(
            _deserialize_structure_observation(entry)
            for entry in payload.get("structure_observations", [])
            if isinstance(entry, dict)
        ),
        training_observations=tuple(
            _deserialize_training_observation(entry)
            for entry in payload.get("training_observations", [])
            if isinstance(entry, dict)
        ),
        predictions=tuple(
            _deserialize_prediction_candidate(entry)
            for entry in payload.get("predictions", [])
            if isinstance(entry, dict)
        ),
        debye_waller_estimates=tuple(
            _deserialize_debye_waller_pair_estimate(entry)
            for entry in payload.get("debye_waller_estimates", [])
            if isinstance(entry, dict)
        ),
        saxs_comparison=_deserialize_saxs_comparison(
            payload.get("saxs_comparison"),
            fallback_path=resolved_file,
        ),
        max_observed_node_count=int(payload.get("max_observed_node_count", 0)),
        max_predicted_node_count=_optional_int(
            payload.get("max_predicted_node_count")
        ),
        prediction_population_share_threshold=float(
            payload.get("prediction_population_share_threshold", 0.02)
        ),
    )
    return LoadedClusterDynamicsMLDataset(
        dataset_file=resolved_file,
        result=result,
        analysis_settings=dict(payload.get("analysis_settings", {})),
    )


def _serialize_preview(preview: ClusterDynamicsMLPreview) -> dict[str, object]:
    return {
        "dynamics_preview": preview.dynamics_preview.to_dict(),
        "preview_summary": dict(preview.dynamics_preview.summary),
        "selected_frame_indices": list(
            preview.dynamics_preview.selected_frame_indices
        ),
        "selected_frame_names": list(
            preview.dynamics_preview.selected_frame_names
        ),
        "selected_source_frame_indices": list(
            preview.dynamics_preview.selected_source_frame_indices
        ),
        "clusters_dir": (
            None if preview.clusters_dir is None else str(preview.clusters_dir)
        ),
        "project_dir": (
            None if preview.project_dir is None else str(preview.project_dir)
        ),
        "experimental_data_path": (
            None
            if preview.experimental_data_path is None
            else str(preview.experimental_data_path)
        ),
        "structure_label_count": int(preview.structure_label_count),
        "total_structure_files": int(preview.total_structure_files),
        "observed_node_counts": list(preview.observed_node_counts),
        "target_node_counts": list(preview.target_node_counts),
        "warnings": list(preview.warnings),
    }


def _deserialize_preview(payload: object) -> ClusterDynamicsMLPreview:
    preview_payload = dict(payload if isinstance(payload, dict) else {})
    dynamics_preview_payload = dict(
        preview_payload.get("dynamics_preview", {})
    )
    preview_summary = dict(preview_payload.get("preview_summary", {}))
    frame_timestep_fs = float(
        dynamics_preview_payload.get("frame_timestep_fs", 0.5)
    )
    (
        frames_per_colormap_timestep,
        colormap_timestep_fs,
    ) = _resolve_colormap_timestep_settings(
        frame_timestep_fs=frame_timestep_fs,
        frames_per_colormap_timestep=dynamics_preview_payload.get(
            "frames_per_colormap_timestep"
        ),
        colormap_timestep_fs=_optional_float(
            dynamics_preview_payload.get("colormap_timestep_fs")
        ),
        legacy_bin_size_fs=_optional_float(
            dynamics_preview_payload.get("bin_size_fs")
        ),
        require_integral_ratio=False,
    )
    dynamics_preview = ClusterDynamicsSelectionPreview(
        summary=preview_summary,
        frame_format=str(dynamics_preview_payload.get("frame_format", "xyz")),
        resolved_box_dimensions=_coerce_box_dimensions(
            dynamics_preview_payload.get("resolved_box_dimensions")
        ),
        use_pbc=bool(dynamics_preview_payload.get("use_pbc", False)),
        first_frame_time_fs=float(
            dynamics_preview_payload.get("first_frame_time_fs", 0.0)
        ),
        frame_timestep_fs=frame_timestep_fs,
        frames_per_colormap_timestep=frames_per_colormap_timestep,
        colormap_timestep_fs=colormap_timestep_fs,
        analysis_start_fs=float(
            dynamics_preview_payload.get("analysis_start_fs", 0.0)
        ),
        analysis_stop_fs=float(
            dynamics_preview_payload.get("analysis_stop_fs", 0.0)
        ),
        first_selected_time_fs=_optional_float(
            dynamics_preview_payload.get("first_selected_time_fs")
        ),
        last_selected_time_fs=_optional_float(
            dynamics_preview_payload.get("last_selected_time_fs")
        ),
        selected_frame_indices=tuple(
            int(value)
            for value in preview_payload.get("selected_frame_indices", [])
        ),
        selected_frame_names=tuple(
            str(value)
            for value in preview_payload.get("selected_frame_names", [])
        ),
        selected_source_frame_indices=tuple(
            None if value is None else int(value)
            for value in preview_payload.get(
                "selected_source_frame_indices",
                [],
            )
        ),
        energy_file=(
            None
            if dynamics_preview_payload.get("energy_file") is None
            else Path(str(dynamics_preview_payload.get("energy_file")))
        ),
        folder_start_time_fs=_optional_float(
            dynamics_preview_payload.get("folder_start_time_fs")
        ),
        folder_start_time_source=_optional_str(
            dynamics_preview_payload.get("folder_start_time_source")
        ),
        time_source_label=str(
            dynamics_preview_payload.get("time_source_label", "Saved dataset")
        ),
        time_warnings=tuple(
            str(value)
            for value in dynamics_preview_payload.get("time_warnings", [])
        ),
    )
    return ClusterDynamicsMLPreview(
        dynamics_preview=dynamics_preview,
        clusters_dir=_optional_path(preview_payload.get("clusters_dir")),
        project_dir=_optional_path(preview_payload.get("project_dir")),
        experimental_data_path=_optional_path(
            preview_payload.get("experimental_data_path")
        ),
        structure_label_count=int(
            preview_payload.get("structure_label_count", 0)
        ),
        total_structure_files=int(
            preview_payload.get("total_structure_files", 0)
        ),
        observed_node_counts=tuple(
            int(value)
            for value in preview_payload.get("observed_node_counts", [])
        ),
        target_node_counts=tuple(
            int(value)
            for value in preview_payload.get("target_node_counts", [])
        ),
        warnings=tuple(
            str(value) for value in preview_payload.get("warnings", [])
        ),
    )


def _serialize_structure_observation(
    observation: ClusterStructureObservation,
) -> dict[str, object]:
    return {
        "label": observation.label,
        "node_count": int(observation.node_count),
        "element_counts": dict(observation.element_counts),
        "file_count": int(observation.file_count),
        "representative_path": (
            None
            if observation.representative_path is None
            else str(observation.representative_path)
        ),
        "structure_dir": str(observation.structure_dir),
        "motifs": list(observation.motifs),
        "mean_atom_count": float(observation.mean_atom_count),
        "mean_radius_of_gyration": float(observation.mean_radius_of_gyration),
        "mean_max_radius": float(observation.mean_max_radius),
        "mean_semiaxis_a": float(observation.mean_semiaxis_a),
        "mean_semiaxis_b": float(observation.mean_semiaxis_b),
        "mean_semiaxis_c": float(observation.mean_semiaxis_c),
    }


def _deserialize_structure_observation(
    payload: dict[str, object],
) -> ClusterStructureObservation:
    return ClusterStructureObservation(
        label=str(payload.get("label", "")),
        node_count=int(payload.get("node_count", 0)),
        element_counts={
            str(element): int(count)
            for element, count in dict(
                payload.get("element_counts", {})
            ).items()
        },
        file_count=int(payload.get("file_count", 0)),
        representative_path=_optional_path(payload.get("representative_path")),
        structure_dir=Path(str(payload.get("structure_dir", "."))),
        motifs=tuple(str(value) for value in payload.get("motifs", [])),
        mean_atom_count=float(payload.get("mean_atom_count", 0.0)),
        mean_radius_of_gyration=float(
            payload.get("mean_radius_of_gyration", 0.0)
        ),
        mean_max_radius=float(payload.get("mean_max_radius", 0.0)),
        mean_semiaxis_a=float(payload.get("mean_semiaxis_a", 0.0)),
        mean_semiaxis_b=float(payload.get("mean_semiaxis_b", 0.0)),
        mean_semiaxis_c=float(payload.get("mean_semiaxis_c", 0.0)),
    )


def _serialize_training_observation(
    observation: ClusterDynamicsMLTrainingObservation,
) -> dict[str, object]:
    return {
        **_serialize_structure_observation(
            ClusterStructureObservation(
                label=observation.label,
                node_count=observation.node_count,
                element_counts=observation.element_counts,
                file_count=observation.file_count,
                representative_path=observation.representative_path,
                structure_dir=observation.structure_dir,
                motifs=observation.motifs,
                mean_atom_count=observation.mean_atom_count,
                mean_radius_of_gyration=observation.mean_radius_of_gyration,
                mean_max_radius=observation.mean_max_radius,
                mean_semiaxis_a=observation.mean_semiaxis_a,
                mean_semiaxis_b=observation.mean_semiaxis_b,
                mean_semiaxis_c=observation.mean_semiaxis_c,
            )
        ),
        "cluster_size": int(observation.cluster_size),
        "total_observations": int(observation.total_observations),
        "occupied_frames": int(observation.occupied_frames),
        "mean_count_per_frame": float(observation.mean_count_per_frame),
        "occupancy_fraction": float(observation.occupancy_fraction),
        "association_events": int(observation.association_events),
        "dissociation_events": int(observation.dissociation_events),
        "association_rate_per_ps": float(observation.association_rate_per_ps),
        "dissociation_rate_per_ps": float(
            observation.dissociation_rate_per_ps
        ),
        "completed_lifetime_count": int(observation.completed_lifetime_count),
        "window_truncated_lifetime_count": int(
            observation.window_truncated_lifetime_count
        ),
        "mean_lifetime_fs": _optional_float(observation.mean_lifetime_fs),
        "std_lifetime_fs": _optional_float(observation.std_lifetime_fs),
    }


def _deserialize_training_observation(
    payload: dict[str, object],
) -> ClusterDynamicsMLTrainingObservation:
    structure_observation = _deserialize_structure_observation(payload)
    return ClusterDynamicsMLTrainingObservation(
        label=structure_observation.label,
        node_count=structure_observation.node_count,
        cluster_size=int(payload.get("cluster_size", 0)),
        element_counts=structure_observation.element_counts,
        file_count=structure_observation.file_count,
        representative_path=structure_observation.representative_path,
        structure_dir=structure_observation.structure_dir,
        motifs=structure_observation.motifs,
        mean_atom_count=structure_observation.mean_atom_count,
        mean_radius_of_gyration=structure_observation.mean_radius_of_gyration,
        mean_max_radius=structure_observation.mean_max_radius,
        mean_semiaxis_a=structure_observation.mean_semiaxis_a,
        mean_semiaxis_b=structure_observation.mean_semiaxis_b,
        mean_semiaxis_c=structure_observation.mean_semiaxis_c,
        total_observations=int(payload.get("total_observations", 0)),
        occupied_frames=int(payload.get("occupied_frames", 0)),
        mean_count_per_frame=float(payload.get("mean_count_per_frame", 0.0)),
        occupancy_fraction=float(payload.get("occupancy_fraction", 0.0)),
        association_events=int(payload.get("association_events", 0)),
        dissociation_events=int(payload.get("dissociation_events", 0)),
        association_rate_per_ps=float(
            payload.get("association_rate_per_ps", 0.0)
        ),
        dissociation_rate_per_ps=float(
            payload.get("dissociation_rate_per_ps", 0.0)
        ),
        completed_lifetime_count=int(
            payload.get("completed_lifetime_count", 0)
        ),
        window_truncated_lifetime_count=int(
            payload.get("window_truncated_lifetime_count", 0)
        ),
        mean_lifetime_fs=_optional_float(payload.get("mean_lifetime_fs")),
        std_lifetime_fs=_optional_float(payload.get("std_lifetime_fs")),
    )


def _serialize_prediction_candidate(
    candidate: PredictedClusterCandidate,
) -> dict[str, object]:
    return {
        "target_node_count": int(candidate.target_node_count),
        "rank": int(candidate.rank),
        "label": candidate.label,
        "element_counts": dict(candidate.element_counts),
        "predicted_mean_count_per_frame": float(
            candidate.predicted_mean_count_per_frame
        ),
        "predicted_occupancy_fraction": float(
            candidate.predicted_occupancy_fraction
        ),
        "predicted_mean_lifetime_fs": float(
            candidate.predicted_mean_lifetime_fs
        ),
        "predicted_association_rate_per_ps": float(
            candidate.predicted_association_rate_per_ps
        ),
        "predicted_dissociation_rate_per_ps": float(
            candidate.predicted_dissociation_rate_per_ps
        ),
        "predicted_mean_radius_of_gyration": float(
            candidate.predicted_mean_radius_of_gyration
        ),
        "predicted_mean_max_radius": float(
            candidate.predicted_mean_max_radius
        ),
        "predicted_mean_semiaxis_a": float(
            candidate.predicted_mean_semiaxis_a
        ),
        "predicted_mean_semiaxis_b": float(
            candidate.predicted_mean_semiaxis_b
        ),
        "predicted_mean_semiaxis_c": float(
            candidate.predicted_mean_semiaxis_c
        ),
        "predicted_population_share": float(
            candidate.predicted_population_share
        ),
        "predicted_stability_score": float(
            candidate.predicted_stability_score
        ),
        "source_label": candidate.source_label,
        "notes": candidate.notes,
        "generated_elements": list(candidate.generated_elements),
        "generated_coordinates": candidate.generated_coordinates.tolist(),
    }


def _deserialize_prediction_candidate(
    payload: dict[str, object],
) -> PredictedClusterCandidate:
    return PredictedClusterCandidate(
        target_node_count=int(payload.get("target_node_count", 0)),
        rank=int(payload.get("rank", 0)),
        label=str(payload.get("label", "")),
        element_counts={
            str(element): int(count)
            for element, count in dict(
                payload.get("element_counts", {})
            ).items()
        },
        predicted_mean_count_per_frame=float(
            payload.get("predicted_mean_count_per_frame", 0.0)
        ),
        predicted_occupancy_fraction=float(
            payload.get("predicted_occupancy_fraction", 0.0)
        ),
        predicted_mean_lifetime_fs=float(
            payload.get("predicted_mean_lifetime_fs", 0.0)
        ),
        predicted_association_rate_per_ps=float(
            payload.get("predicted_association_rate_per_ps", 0.0)
        ),
        predicted_dissociation_rate_per_ps=float(
            payload.get("predicted_dissociation_rate_per_ps", 0.0)
        ),
        predicted_mean_radius_of_gyration=float(
            payload.get("predicted_mean_radius_of_gyration", 0.0)
        ),
        predicted_mean_max_radius=float(
            payload.get("predicted_mean_max_radius", 0.0)
        ),
        predicted_mean_semiaxis_a=float(
            payload.get("predicted_mean_semiaxis_a", 0.0)
        ),
        predicted_mean_semiaxis_b=float(
            payload.get("predicted_mean_semiaxis_b", 0.0)
        ),
        predicted_mean_semiaxis_c=float(
            payload.get("predicted_mean_semiaxis_c", 0.0)
        ),
        predicted_population_share=float(
            payload.get("predicted_population_share", 0.0)
        ),
        predicted_stability_score=float(
            payload.get("predicted_stability_score", 0.0)
        ),
        source_label=_optional_str(payload.get("source_label")),
        notes=str(payload.get("notes", "")),
        generated_elements=tuple(
            str(value) for value in payload.get("generated_elements", [])
        ),
        generated_coordinates=np.asarray(
            payload.get("generated_coordinates", []),
            dtype=float,
        ),
    )


def _serialize_debye_waller_pair_estimate(
    estimate: DebyeWallerPairEstimate,
) -> dict[str, object]:
    return {
        "source": estimate.source,
        "method": estimate.method,
        "label": estimate.label,
        "node_count": int(estimate.node_count),
        "candidate_rank": _optional_int(estimate.candidate_rank),
        "element_a": estimate.element_a,
        "element_b": estimate.element_b,
        "sigma": float(estimate.sigma),
        "b_factor": float(estimate.b_factor),
        "support_count": int(estimate.support_count),
        "aligned_pair_count": int(estimate.aligned_pair_count),
        "source_label": estimate.source_label,
    }


def _deserialize_debye_waller_pair_estimate(
    payload: dict[str, object],
) -> DebyeWallerPairEstimate:
    return DebyeWallerPairEstimate(
        source=str(payload.get("source", "")),
        method=str(payload.get("method", "")),
        label=str(payload.get("label", "")),
        node_count=int(payload.get("node_count", 0)),
        candidate_rank=_optional_int(payload.get("candidate_rank")),
        element_a=str(payload.get("element_a", "")),
        element_b=str(payload.get("element_b", "")),
        sigma=float(payload.get("sigma", 0.0)),
        b_factor=float(payload.get("b_factor", 0.0)),
        support_count=int(payload.get("support_count", 0)),
        aligned_pair_count=int(payload.get("aligned_pair_count", 0)),
        source_label=_optional_str(payload.get("source_label")),
    )


def _serialize_saxs_comparison(
    comparison: ClusterDynamicsMLSAXSComparison | None,
) -> dict[str, object] | None:
    if comparison is None:
        return None
    return {
        "q_values": comparison.q_values.tolist(),
        "observed_raw_model_intensity": (
            None
            if comparison.observed_raw_model_intensity is None
            else comparison.observed_raw_model_intensity.tolist()
        ),
        "observed_fitted_model_intensity": (
            None
            if comparison.observed_fitted_model_intensity is None
            else comparison.observed_fitted_model_intensity.tolist()
        ),
        "observed_rmse": _optional_float(comparison.observed_rmse),
        "raw_model_intensity": comparison.raw_model_intensity.tolist(),
        "fitted_model_intensity": comparison.fitted_model_intensity.tolist(),
        "experimental_intensity": (
            None
            if comparison.experimental_intensity is None
            else comparison.experimental_intensity.tolist()
        ),
        "residuals": (
            None
            if comparison.residuals is None
            else comparison.residuals.tolist()
        ),
        "scale_factor": float(comparison.scale_factor),
        "offset": float(comparison.offset),
        "rmse": _optional_float(comparison.rmse),
        "component_weights": [
            {
                "label": entry.label,
                "weight": float(entry.weight),
                "source": entry.source,
                "profile_path": (
                    None
                    if entry.profile_path is None
                    else str(entry.profile_path)
                ),
                "structure_path": (
                    None
                    if entry.structure_path is None
                    else str(entry.structure_path)
                ),
            }
            for entry in comparison.component_weights
        ],
        "experimental_data_path": (
            None
            if comparison.experimental_data_path is None
            else str(comparison.experimental_data_path)
        ),
        "component_output_dir": (
            None
            if comparison.component_output_dir is None
            else str(comparison.component_output_dir)
        ),
        "predicted_structure_dir": (
            None
            if comparison.predicted_structure_dir is None
            else str(comparison.predicted_structure_dir)
        ),
    }


def _deserialize_saxs_comparison(
    payload: object,
    *,
    fallback_path: Path,
) -> ClusterDynamicsMLSAXSComparison | None:
    if not isinstance(payload, dict):
        return None
    del fallback_path
    experimental_intensity = payload.get("experimental_intensity")
    observed_raw_model_intensity = payload.get("observed_raw_model_intensity")
    observed_fitted_model_intensity = payload.get(
        "observed_fitted_model_intensity"
    )
    residuals = payload.get("residuals")
    return ClusterDynamicsMLSAXSComparison(
        q_values=np.asarray(payload.get("q_values", []), dtype=float),
        observed_raw_model_intensity=(
            None
            if observed_raw_model_intensity is None
            else np.asarray(observed_raw_model_intensity, dtype=float)
        ),
        observed_fitted_model_intensity=(
            None
            if observed_fitted_model_intensity is None
            else np.asarray(observed_fitted_model_intensity, dtype=float)
        ),
        observed_rmse=_optional_float(payload.get("observed_rmse")),
        raw_model_intensity=np.asarray(
            payload.get("raw_model_intensity", []),
            dtype=float,
        ),
        fitted_model_intensity=np.asarray(
            payload.get("fitted_model_intensity", []),
            dtype=float,
        ),
        experimental_intensity=(
            None
            if experimental_intensity is None
            else np.asarray(experimental_intensity, dtype=float)
        ),
        residuals=(
            None if residuals is None else np.asarray(residuals, dtype=float)
        ),
        scale_factor=float(payload.get("scale_factor", 1.0)),
        offset=float(payload.get("offset", 0.0)),
        rmse=_optional_float(payload.get("rmse")),
        component_weights=tuple(
            SAXSComponentWeight(
                label=str(entry.get("label", "")),
                weight=float(entry.get("weight", 0.0)),
                source=str(entry.get("source", "")),
                profile_path=_optional_path(entry.get("profile_path")),
                structure_path=_optional_path(entry.get("structure_path")),
            )
            for entry in payload.get("component_weights", [])
            if isinstance(entry, dict)
        ),
        experimental_data_path=_optional_path(
            payload.get("experimental_data_path")
        ),
        component_output_dir=_optional_path(
            payload.get("component_output_dir")
        ),
        predicted_structure_dir=_optional_path(
            payload.get("predicted_structure_dir")
            or next(
                (
                    value
                    for key, value in payload.items()
                    if key != "component_output_dir"
                    and key.endswith("structure_dir")
                ),
                None,
            )
        ),
    )


def _serialize_cluster_dynamics_result(
    result: ClusterDynamicsResult,
) -> dict[str, object]:
    return {
        "preview_summary": dict(result.preview.summary),
        "preview": result.preview.to_dict(),
        "selected_frame_indices": list(result.preview.selected_frame_indices),
        "selected_frame_names": list(result.preview.selected_frame_names),
        "selected_source_frame_indices": list(
            result.preview.selected_source_frame_indices
        ),
        "frame_results": [
            {
                "frame_index": int(frame_result.frame_index),
                "time_fs": (
                    None
                    if frame_result.time_fs is None
                    else float(frame_result.time_fs)
                ),
            }
            for frame_result in result.frame_results
        ],
        "cluster_labels": list(result.cluster_labels),
        "cluster_sizes": {
            str(label): int(size)
            for label, size in result.cluster_sizes.items()
        },
        "bin_edges_fs": result.bin_edges_fs.tolist(),
        "frames_per_bin": result.frames_per_bin.tolist(),
        "total_clusters_per_bin": result.total_clusters_per_bin.tolist(),
        "raw_count_matrix": result.raw_count_matrix.tolist(),
        "fraction_matrix": result.fraction_matrix.tolist(),
        "mean_count_matrix": result.mean_count_matrix.tolist(),
        "frame_count_matrix": result.frame_count_matrix.tolist(),
        "total_clusters_per_frame": result.total_clusters_per_frame.tolist(),
        "lifetime_by_label": [
            asdict(entry) for entry in result.lifetime_by_label
        ],
        "lifetime_by_size": [
            asdict(entry) for entry in result.lifetime_by_size
        ],
        "energy_data": _serialize_energy_data(result.energy_data),
    }


def _deserialize_cluster_dynamics_result(
    payload: object,
    *,
    fallback_path: Path,
) -> ClusterDynamicsResult:
    result_payload = dict(payload if isinstance(payload, dict) else {})
    preview_payload = dict(result_payload.get("preview", {}))
    preview_summary = dict(result_payload.get("preview_summary", {}))
    frame_results_payload = result_payload.get("frame_results", [])
    frame_results = tuple(
        FrameClusterResult(
            frame_index=int(entry["frame_index"]),
            time_fs=(
                None
                if entry.get("time_fs") is None
                else float(entry.get("time_fs"))
            ),
            clusters=[],
        )
        for entry in frame_results_payload
        if isinstance(entry, dict) and "frame_index" in entry
    )
    frame_timestep_fs = float(preview_payload.get("frame_timestep_fs", 0.5))
    (
        frames_per_colormap_timestep,
        colormap_timestep_fs,
    ) = _resolve_colormap_timestep_settings(
        frame_timestep_fs=frame_timestep_fs,
        frames_per_colormap_timestep=preview_payload.get(
            "frames_per_colormap_timestep"
        ),
        colormap_timestep_fs=_optional_float(
            preview_payload.get("colormap_timestep_fs")
        ),
        legacy_bin_size_fs=_optional_float(preview_payload.get("bin_size_fs")),
        require_integral_ratio=False,
    )
    preview = ClusterDynamicsSelectionPreview(
        summary=preview_summary,
        frame_format=str(preview_payload.get("frame_format", "xyz")),
        resolved_box_dimensions=_coerce_box_dimensions(
            preview_payload.get("resolved_box_dimensions")
        ),
        use_pbc=bool(preview_payload.get("use_pbc", False)),
        first_frame_time_fs=float(
            preview_payload.get("first_frame_time_fs", 0.0)
        ),
        frame_timestep_fs=frame_timestep_fs,
        frames_per_colormap_timestep=frames_per_colormap_timestep,
        colormap_timestep_fs=colormap_timestep_fs,
        analysis_start_fs=float(preview_payload.get("analysis_start_fs", 0.0)),
        analysis_stop_fs=float(preview_payload.get("analysis_stop_fs", 0.0)),
        first_selected_time_fs=_optional_float(
            preview_payload.get("first_selected_time_fs")
        ),
        last_selected_time_fs=_optional_float(
            preview_payload.get("last_selected_time_fs")
        ),
        selected_frame_indices=tuple(
            int(value)
            for value in result_payload.get("selected_frame_indices", [])
        ),
        selected_frame_names=tuple(
            str(value)
            for value in result_payload.get("selected_frame_names", [])
        ),
        selected_source_frame_indices=tuple(
            None if value is None else int(value)
            for value in result_payload.get(
                "selected_source_frame_indices", []
            )
        ),
        energy_file=_optional_path(preview_payload.get("energy_file")),
        folder_start_time_fs=_optional_float(
            preview_payload.get("folder_start_time_fs")
        ),
        folder_start_time_source=_optional_str(
            preview_payload.get("folder_start_time_source")
        ),
        time_source_label=str(
            preview_payload.get("time_source_label", "Saved dataset")
        ),
        time_warnings=tuple(
            str(value) for value in preview_payload.get("time_warnings", [])
        ),
    )
    return ClusterDynamicsResult(
        preview=preview,
        frame_results=frame_results,
        bin_edges_fs=np.asarray(
            result_payload.get("bin_edges_fs", []), dtype=float
        ),
        frames_per_bin=np.asarray(
            result_payload.get("frames_per_bin", []),
            dtype=float,
        ),
        total_clusters_per_bin=np.asarray(
            result_payload.get("total_clusters_per_bin", []),
            dtype=float,
        ),
        cluster_labels=tuple(
            str(value) for value in result_payload.get("cluster_labels", [])
        ),
        cluster_sizes={
            str(label): int(size)
            for label, size in dict(
                result_payload.get("cluster_sizes", {})
            ).items()
        },
        raw_count_matrix=np.asarray(
            result_payload.get("raw_count_matrix", []),
            dtype=float,
        ),
        fraction_matrix=np.asarray(
            result_payload.get("fraction_matrix", []),
            dtype=float,
        ),
        mean_count_matrix=np.asarray(
            result_payload.get("mean_count_matrix", []),
            dtype=float,
        ),
        frame_count_matrix=np.asarray(
            result_payload.get("frame_count_matrix", []),
            dtype=float,
        ),
        total_clusters_per_frame=np.asarray(
            result_payload.get("total_clusters_per_frame", []),
            dtype=float,
        ),
        lifetime_by_label=tuple(
            ClusterLifetimeSummary(**_normalize_summary_payload(entry))
            for entry in result_payload.get("lifetime_by_label", [])
            if isinstance(entry, dict)
        ),
        lifetime_by_size=tuple(
            ClusterSizeLifetimeSummary(**_normalize_summary_payload(entry))
            for entry in result_payload.get("lifetime_by_size", [])
            if isinstance(entry, dict)
        ),
        energy_data=_deserialize_energy_data(
            result_payload.get("energy_data"),
            fallback_path=fallback_path,
        ),
    )


def _write_prediction_csv(
    result: ClusterDynamicsMLResult,
    dataset_file: Path,
) -> Path:
    output_path = dataset_file.with_name(
        f"{dataset_file.stem}_predictions.csv"
    )
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "target_node_count",
                "rank",
                "label",
                "predicted_population_share",
                "predicted_mean_count_per_frame",
                "predicted_occupancy_fraction",
                "predicted_mean_lifetime_fs",
                "predicted_association_rate_per_ps",
                "predicted_dissociation_rate_per_ps",
                "predicted_mean_radius_of_gyration",
                "predicted_mean_max_radius",
                "predicted_stability_score",
                "source_label",
                "notes",
            ]
        )
        for entry in result.predictions:
            writer.writerow(
                [
                    int(entry.target_node_count),
                    int(entry.rank),
                    entry.label,
                    float(entry.predicted_population_share),
                    float(entry.predicted_mean_count_per_frame),
                    float(entry.predicted_occupancy_fraction),
                    float(entry.predicted_mean_lifetime_fs),
                    float(entry.predicted_association_rate_per_ps),
                    float(entry.predicted_dissociation_rate_per_ps),
                    float(entry.predicted_mean_radius_of_gyration),
                    float(entry.predicted_mean_max_radius),
                    float(entry.predicted_stability_score),
                    "" if entry.source_label is None else entry.source_label,
                    entry.notes,
                ]
            )
    return output_path


def _write_histogram_csvs(
    result: ClusterDynamicsMLResult,
    dataset_file: Path,
) -> list[Path]:
    output_paths = [
        (
            dataset_file.with_name(
                f"{dataset_file.stem}_observed_histogram.csv"
            ),
            False,
        ),
        (
            dataset_file.with_name(
                f"{dataset_file.stem}_observed_plus_predicted_structures_histogram.csv"
            ),
            True,
        ),
    ]
    written_files: list[Path] = []
    for output_path, include_predictions in output_paths:
        written_files.append(
            _write_histogram_csv(
                result,
                output_path,
                include_predictions=include_predictions,
            )
        )
    return written_files


def _write_histogram_csv(
    result: ClusterDynamicsMLResult,
    output_path: Path,
    *,
    include_predictions: bool,
) -> Path:
    rows = _distribution_rows(
        result,
        include_predictions=include_predictions,
    )
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "label",
                "source",
                "node_count",
                "mean_lifetime_fs",
                "mean_count_per_frame",
                "mean_max_radius",
                "raw_weight",
                "normalized_weight",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["label"],
                    row["source"],
                    float(row["node_count"]),
                    (
                        ""
                        if row["mean_lifetime_fs"] is None
                        else float(row["mean_lifetime_fs"])
                    ),
                    float(row["mean_count_per_frame"]),
                    float(row["mean_max_radius"]),
                    float(row["raw_weight"]),
                    float(row["normalized_weight"]),
                ]
            )
    return output_path


def _write_saxs_csv(
    comparison: ClusterDynamicsMLSAXSComparison,
    dataset_file: Path,
) -> Path:
    output_path = dataset_file.with_name(f"{dataset_file.stem}_saxs.csv")
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "q",
                "raw_model_intensity",
                "fitted_model_intensity",
                "experimental_intensity",
                "residual",
            ]
        )
        experimental = comparison.experimental_intensity
        residuals = comparison.residuals
        for index, q_value in enumerate(comparison.q_values):
            writer.writerow(
                [
                    float(q_value),
                    float(comparison.raw_model_intensity[index]),
                    float(comparison.fitted_model_intensity[index]),
                    (
                        ""
                        if experimental is None
                        else float(experimental[index])
                    ),
                    "" if residuals is None else float(residuals[index]),
                ]
            )
    return output_path


def _write_saxs_component_profiles(
    comparison: ClusterDynamicsMLSAXSComparison,
    dataset_file: Path,
) -> list[Path]:
    output_dir = dataset_file.with_name(f"{dataset_file.stem}_saxs_components")
    output_dir.mkdir(parents=True, exist_ok=True)
    copied_files: list[Path] = []
    seen_sources: set[Path] = set()
    for entry in comparison.component_weights:
        source_path = entry.profile_path
        if source_path is None:
            continue
        resolved_source = source_path.expanduser().resolve()
        if resolved_source in seen_sources or not resolved_source.is_file():
            continue
        seen_sources.add(resolved_source)
        target_path = _unique_child_path(output_dir, resolved_source.name)
        shutil.copy2(resolved_source, target_path)
        copied_files.append(target_path)
    return copied_files


def _write_prediction_structures(
    result: ClusterDynamicsMLResult,
    dataset_file: Path,
) -> list[Path]:
    output_dir = dataset_file.with_name(
        f"{dataset_file.stem}_predicted_structures"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    if result.saxs_comparison is not None:
        copied_files: list[Path] = []
        seen_sources: set[Path] = set()
        for entry in result.saxs_comparison.component_weights:
            if entry.source != "predicted" or entry.structure_path is None:
                continue
            resolved_source = entry.structure_path.expanduser().resolve()
            if (
                resolved_source in seen_sources
                or not resolved_source.is_file()
            ):
                continue
            seen_sources.add(resolved_source)
            target_path = _unique_child_path(output_dir, resolved_source.name)
            shutil.copy2(resolved_source, target_path)
            copied_files.append(target_path)
        if copied_files:
            return copied_files

    written_files: list[Path] = []
    for entry in result.predictions:
        output_path = output_dir / (
            f"{entry.target_node_count:02d}_rank{entry.rank:02d}_{entry.label}.xyz"
        )
        lines = [
            f"{len(entry.generated_elements)}",
            (
                f"label={entry.label} target_node_count={entry.target_node_count} "
                f"rank={entry.rank}"
            ),
        ]
        for element, coords in zip(
            entry.generated_elements,
            entry.generated_coordinates,
            strict=False,
        ):
            lines.append(
                f"{element} {float(coords[0]):.8f} {float(coords[1]):.8f} "
                f"{float(coords[2]):.8f}"
            )
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        written_files.append(output_path)
    return written_files


def _distribution_rows(
    result: ClusterDynamicsMLResult,
    *,
    include_predictions: bool,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    observed_weights, predicted_weights = _resolved_population_weights(
        result.training_observations,
        result.predictions,
        frame_timestep_fs=float(
            result.dynamics_result.preview.frame_timestep_fs
        ),
    )
    for row, weight in zip(
        result.training_observations,
        observed_weights,
        strict=False,
    ):
        if weight <= 0.0:
            continue
        rows.append(
            {
                "label": row.label,
                "source": "observed",
                "node_count": float(row.node_count),
                "mean_lifetime_fs": row.mean_lifetime_fs,
                "mean_count_per_frame": float(row.mean_count_per_frame),
                "mean_max_radius": float(row.mean_max_radius),
                "raw_weight": weight,
            }
        )
    if include_predictions:
        for item, weight in zip(
            result.predictions,
            predicted_weights,
            strict=False,
        ):
            if weight <= 0.0:
                continue
            rows.append(
                {
                    "label": item.label,
                    "source": "predicted",
                    "node_count": float(item.target_node_count),
                    "mean_lifetime_fs": float(item.predicted_mean_lifetime_fs),
                    "mean_count_per_frame": float(
                        item.predicted_mean_count_per_frame
                    ),
                    "mean_max_radius": float(item.predicted_mean_max_radius),
                    "raw_weight": weight,
                }
            )
    total_weight = sum(float(row["raw_weight"]) for row in rows)
    if total_weight <= 0.0:
        return []
    for row in rows:
        row["normalized_weight"] = float(row["raw_weight"]) / total_weight
    return rows


def _unique_child_path(directory: Path, filename: str) -> Path:
    candidate = directory / filename
    if not candidate.exists():
        return candidate
    stem = candidate.stem
    suffix = candidate.suffix
    counter = 2
    while True:
        alternative = directory / f"{stem}_{counter}{suffix}"
        if not alternative.exists():
            return alternative
        counter += 1


def _serialize_energy_data(
    energy_data: CP2KEnergyData | None,
) -> dict[str, object] | None:
    if energy_data is None:
        return None
    return {
        "filepath": str(energy_data.filepath),
        "step": energy_data.step.tolist(),
        "time_fs": energy_data.time_fs.tolist(),
        "kinetic": energy_data.kinetic.tolist(),
        "temperature": energy_data.temperature.tolist(),
        "potential": energy_data.potential.tolist(),
    }


def _deserialize_energy_data(
    payload: object,
    *,
    fallback_path: Path,
) -> CP2KEnergyData | None:
    if not isinstance(payload, dict):
        return None
    filepath_value = payload.get("filepath")
    filepath = (
        fallback_path if filepath_value is None else Path(str(filepath_value))
    )
    return CP2KEnergyData(
        filepath=filepath,
        step=np.asarray(payload.get("step", []), dtype=float),
        time_fs=np.asarray(payload.get("time_fs", []), dtype=float),
        kinetic=np.asarray(payload.get("kinetic", []), dtype=float),
        temperature=np.asarray(payload.get("temperature", []), dtype=float),
        potential=np.asarray(payload.get("potential", []), dtype=float),
    )


def _normalize_summary_payload(
    payload: dict[str, object]
) -> dict[str, object]:
    normalized = dict(payload)
    if (
        "censored_lifetime_count" in normalized
        and "window_truncated_lifetime_count" not in normalized
    ):
        normalized["window_truncated_lifetime_count"] = normalized.pop(
            "censored_lifetime_count"
        )
    return normalized


def _coerce_box_dimensions(
    value: object,
) -> tuple[float, float, float] | None:
    if value is None:
        return None
    components = tuple(float(component) for component in value)
    if len(components) != 3:
        raise ValueError("Saved box dimensions must contain three values.")
    return components


def _optional_float(value: object) -> float | None:
    return None if value is None else float(value)


def _optional_int(value: object) -> int | None:
    return None if value is None else int(value)


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_path(value: object) -> Path | None:
    text = _optional_str(value)
    return None if text is None else Path(text)


__all__ = [
    "LoadedClusterDynamicsMLDataset",
    "SavedClusterDynamicsMLDataset",
    "load_cluster_dynamicsai_dataset",
    "save_cluster_dynamicsai_dataset",
]
