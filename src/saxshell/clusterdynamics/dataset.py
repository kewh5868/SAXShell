from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from saxshell.cluster import FrameClusterResult
from saxshell.mdtrajectory.frame.cp2k_ener import CP2KEnergyData

from .workflow import (
    ClusterDynamicsResult,
    ClusterDynamicsSelectionPreview,
    ClusterLifetimeSummary,
    ClusterSizeLifetimeSummary,
    DisplayMode,
    TimeUnit,
    _resolve_colormap_timestep_settings,
)

DATASET_VERSION = 1


@dataclass(slots=True)
class SavedClusterDynamicsDataset:
    dataset_file: Path
    written_files: tuple[Path, ...]


@dataclass(slots=True)
class LoadedClusterDynamicsDataset:
    dataset_file: Path
    result: ClusterDynamicsResult
    analysis_settings: dict[str, object]


def save_cluster_dynamics_dataset(
    result: ClusterDynamicsResult,
    output_file: str | Path,
    *,
    analysis_settings: dict[str, object] | None = None,
) -> SavedClusterDynamicsDataset:
    dataset_file = Path(output_file).expanduser().resolve()
    if dataset_file.suffix.lower() != ".json":
        dataset_file = dataset_file.with_suffix(".json")
    dataset_file.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "version": DATASET_VERSION,
        "analysis_settings": analysis_settings or {},
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
    dataset_file.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )

    written_files = [dataset_file]
    written_files.append(
        export_cluster_dynamics_colormap_csv(
            result,
            dataset_file.with_name(
                f"{dataset_file.stem}_cluster_distribution.csv"
            ),
        )
    )
    written_files.append(
        export_cluster_dynamics_lifetime_csv(
            result,
            dataset_file.with_name(f"{dataset_file.stem}_lifetime.csv"),
        )
    )
    if result.energy_data is not None:
        written_files.append(
            _write_energy_csv(result.energy_data, dataset_file)
        )

    return SavedClusterDynamicsDataset(
        dataset_file=dataset_file,
        written_files=tuple(written_files),
    )


def load_cluster_dynamics_dataset(
    dataset_file: str | Path,
) -> LoadedClusterDynamicsDataset:
    resolved_file = Path(dataset_file).expanduser().resolve()
    payload = json.loads(resolved_file.read_text(encoding="utf-8"))
    if int(payload.get("version", 0)) != DATASET_VERSION:
        raise ValueError(
            "This cluster-dynamics dataset uses an unsupported format version."
        )

    preview_payload = dict(payload.get("preview", {}))
    preview_summary = dict(payload.get("preview_summary", {}))
    frame_results_payload = payload.get("frame_results", [])
    frame_results = tuple(
        FrameClusterResult(
            frame_index=int(entry["frame_index"]),
            time_fs=(
                None
                if entry.get("time_fs") is None
                else float(entry["time_fs"])
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
            int(value) for value in payload.get("selected_frame_indices", [])
        ),
        selected_frame_names=tuple(
            str(value) for value in payload.get("selected_frame_names", [])
        ),
        selected_source_frame_indices=tuple(
            None if value is None else int(value)
            for value in payload.get("selected_source_frame_indices", [])
        ),
        energy_file=(
            None
            if preview_payload.get("energy_file") is None
            else Path(str(preview_payload.get("energy_file")))
        ),
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

    result = ClusterDynamicsResult(
        preview=preview,
        frame_results=frame_results,
        bin_edges_fs=np.asarray(payload.get("bin_edges_fs", []), dtype=float),
        frames_per_bin=np.asarray(
            payload.get("frames_per_bin", []),
            dtype=float,
        ),
        total_clusters_per_bin=np.asarray(
            payload.get("total_clusters_per_bin", []),
            dtype=float,
        ),
        cluster_labels=tuple(
            str(value) for value in payload.get("cluster_labels", [])
        ),
        cluster_sizes={
            str(label): int(size)
            for label, size in dict(payload.get("cluster_sizes", {})).items()
        },
        raw_count_matrix=np.asarray(
            payload.get("raw_count_matrix", []),
            dtype=float,
        ),
        fraction_matrix=np.asarray(
            payload.get("fraction_matrix", []),
            dtype=float,
        ),
        mean_count_matrix=np.asarray(
            payload.get("mean_count_matrix", []),
            dtype=float,
        ),
        frame_count_matrix=np.asarray(
            payload.get("frame_count_matrix", []),
            dtype=float,
        ),
        total_clusters_per_frame=np.asarray(
            payload.get("total_clusters_per_frame", []),
            dtype=float,
        ),
        lifetime_by_label=tuple(
            ClusterLifetimeSummary(**_normalize_summary_payload(entry))
            for entry in payload.get("lifetime_by_label", [])
            if isinstance(entry, dict)
        ),
        lifetime_by_size=tuple(
            ClusterSizeLifetimeSummary(**_normalize_summary_payload(entry))
            for entry in payload.get("lifetime_by_size", [])
            if isinstance(entry, dict)
        ),
        energy_data=_deserialize_energy_data(
            payload.get("energy_data"),
            fallback_path=resolved_file,
        ),
    )
    return LoadedClusterDynamicsDataset(
        dataset_file=resolved_file,
        result=result,
        analysis_settings=dict(payload.get("analysis_settings", {})),
    )


def export_cluster_dynamics_colormap_csv(
    result: ClusterDynamicsResult,
    output_file: str | Path,
    *,
    display_mode: DisplayMode = "fraction",
    time_unit: TimeUnit = "fs",
) -> Path:
    output_path = _resolve_csv_output_path(output_file)
    time_edges = result.time_edges(time_unit)
    time_centers = result.bin_centers(time_unit)
    displayed_matrix = result.matrix(display_mode)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "label",
                "cluster_size",
                "bin_index",
                "bin_start_fs",
                "bin_stop_fs",
                "bin_center_fs",
                "bin_start_time",
                "bin_stop_time",
                "bin_center_time",
                "time_unit",
                "display_mode",
                "colormap_value",
                "raw_count",
                "fraction",
                "mean_count_per_frame",
                "frames_in_bin",
                "total_clusters_in_bin",
            ]
        )
        for row_index, label in enumerate(result.cluster_labels):
            cluster_size = int(result.cluster_sizes.get(label, 0))
            for bin_index in range(result.bin_count):
                writer.writerow(
                    [
                        label,
                        cluster_size,
                        bin_index,
                        float(result.bin_edges_fs[bin_index]),
                        float(result.bin_edges_fs[bin_index + 1]),
                        float(result.bin_centers_fs[bin_index]),
                        float(time_edges[bin_index]),
                        float(time_edges[bin_index + 1]),
                        float(time_centers[bin_index]),
                        time_unit,
                        display_mode,
                        float(displayed_matrix[row_index, bin_index]),
                        float(result.raw_count_matrix[row_index, bin_index]),
                        float(result.fraction_matrix[row_index, bin_index]),
                        float(result.mean_count_matrix[row_index, bin_index]),
                        float(result.frames_per_bin[bin_index]),
                        float(result.total_clusters_per_bin[bin_index]),
                    ]
                )
    return output_path


def export_cluster_dynamics_lifetime_csv(
    result: ClusterDynamicsResult,
    output_file: str | Path,
) -> Path:
    output_path = _resolve_csv_output_path(output_file)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "label",
                "cluster_size",
                "mean_lifetime_fs",
                "std_lifetime_fs",
                "completed_lifetime_count",
                "window_truncated_lifetime_count",
                "association_rate_per_ps",
                "dissociation_rate_per_ps",
                "occupancy_fraction",
                "mean_count_per_frame",
                "total_observations",
                "occupied_frames",
                "association_events",
                "dissociation_events",
            ]
        )
        for entry in result.lifetime_by_label:
            writer.writerow(
                [
                    entry.label,
                    int(entry.cluster_size),
                    _csv_float(entry.mean_lifetime_fs),
                    _csv_float(entry.std_lifetime_fs),
                    int(entry.completed_lifetime_count),
                    int(entry.window_truncated_lifetime_count),
                    float(entry.association_rate_per_ps),
                    float(entry.dissociation_rate_per_ps),
                    float(entry.occupancy_fraction),
                    float(entry.mean_count_per_frame),
                    int(entry.total_observations),
                    int(entry.occupied_frames),
                    int(entry.association_events),
                    int(entry.dissociation_events),
                ]
            )
    return output_path


def _write_energy_csv(energy_data: CP2KEnergyData, dataset_file: Path) -> Path:
    output_path = dataset_file.with_name(f"{dataset_file.stem}_energy.csv")
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["step", "time_fs", "kinetic", "temperature", "potential"]
        )
        for index in range(len(energy_data.time_fs)):
            writer.writerow(
                [
                    float(energy_data.step[index]),
                    float(energy_data.time_fs[index]),
                    float(energy_data.kinetic[index]),
                    float(energy_data.temperature[index]),
                    float(energy_data.potential[index]),
                ]
            )
    return output_path


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


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _csv_float(value: float | None) -> str:
    return "" if value is None else f"{float(value):.12g}"


def _resolve_csv_output_path(output_file: str | Path) -> Path:
    output_path = Path(output_file).expanduser().resolve()
    if output_path.suffix.lower() != ".csv":
        output_path = output_path.with_suffix(".csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


__all__ = [
    "LoadedClusterDynamicsDataset",
    "SavedClusterDynamicsDataset",
    "export_cluster_dynamics_colormap_csv",
    "export_cluster_dynamics_lifetime_csv",
    "load_cluster_dynamics_dataset",
    "save_cluster_dynamics_dataset",
]
