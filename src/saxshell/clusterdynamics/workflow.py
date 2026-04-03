from __future__ import annotations

import json
import re
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from saxshell.cluster import (
    SEARCH_MODE_KDTREE,
    ClusterNetwork,
    FrameClusterResult,
    PairCutoffDefinitions,
    XYZClusterNetwork,
    XYZStructure,
    detect_frame_folder_mode,
    normalize_search_mode,
)
from saxshell.cluster.clusternetwork import stoichiometry_label
from saxshell.cluster.workflow import ClusterWorkflow
from saxshell.mdtrajectory.frame.cp2k_ener import CP2KEnergyData
from saxshell.mdtrajectory.workflow import EXPORT_METADATA_FILENAME
from saxshell.structure import AtomTypeDefinitions, PDBStructure

DisplayMode = Literal["count", "fraction", "mean_count"]
TimeUnit = Literal["fs", "ps"]
EnergySeriesName = Literal["kinetic", "temperature", "potential"]
_FRAME_FILENAME_PATTERN = re.compile(
    r"frame_(?P<index>\d+)\.(?:xyz|pdb)$",
    re.IGNORECASE,
)
_FOLDER_START_TIME_PATTERN = re.compile(
    r"(?:^|_)t(?P<value>\d+(?:[p_]\d+)?)fs(?:\d{4})?$",
    re.IGNORECASE,
)
_LEGACY_FOLDER_START_TIME_PATTERN = re.compile(
    r"(?:^|_)f(?P<value>\d+(?:_\d+)?)fs(?:\d{4})?$",
    re.IGNORECASE,
)


@dataclass(slots=True)
class ClusterLifetimeSummary:
    """Lifetime summary for one stoichiometry label."""

    label: str
    cluster_size: int
    total_observations: int
    occupied_frames: int
    mean_count_per_frame: float
    occupancy_fraction: float
    association_events: int
    dissociation_events: int
    association_rate_per_ps: float
    dissociation_rate_per_ps: float
    completed_lifetime_count: int
    window_truncated_lifetime_count: int
    mean_lifetime_fs: float | None
    std_lifetime_fs: float | None


@dataclass(slots=True)
class ClusterSizeLifetimeSummary:
    """Lifetime summary aggregated by cluster size."""

    cluster_size: int
    total_observations: int
    occupied_frames: int
    mean_count_per_frame: float
    occupancy_fraction: float
    association_events: int
    dissociation_events: int
    association_rate_per_ps: float
    dissociation_rate_per_ps: float
    completed_lifetime_count: int
    window_truncated_lifetime_count: int
    mean_lifetime_fs: float | None
    std_lifetime_fs: float | None


@dataclass(slots=True)
class ClusterDynamicsSelectionPreview:
    """Preview of the extracted frames selected for time-bin
    analysis."""

    summary: dict[str, object]
    frame_format: str
    resolved_box_dimensions: tuple[float, float, float] | None
    use_pbc: bool
    first_frame_time_fs: float
    frame_timestep_fs: float
    frames_per_colormap_timestep: int | None
    colormap_timestep_fs: float
    analysis_start_fs: float
    analysis_stop_fs: float
    first_selected_time_fs: float | None
    last_selected_time_fs: float | None
    selected_frame_indices: tuple[int, ...]
    selected_frame_names: tuple[str, ...]
    selected_source_frame_indices: tuple[int | None, ...]
    energy_file: Path | None
    folder_start_time_fs: float | None = None
    folder_start_time_source: str | None = None
    time_source_label: str = "Sequential frame order"
    time_warnings: tuple[str, ...] = ()

    @property
    def total_frames(self) -> int:
        return int(self.summary["n_frames"])

    @property
    def selected_frames(self) -> int:
        return len(self.selected_frame_indices)

    @property
    def first_selected_frame(self) -> str | None:
        return (
            self.selected_frame_names[0] if self.selected_frame_names else None
        )

    @property
    def last_selected_frame(self) -> str | None:
        return (
            self.selected_frame_names[-1]
            if self.selected_frame_names
            else None
        )

    @property
    def first_selected_source_frame_index(self) -> int | None:
        return (
            self.selected_source_frame_indices[0]
            if self.selected_source_frame_indices
            else None
        )

    @property
    def last_selected_source_frame_index(self) -> int | None:
        return (
            self.selected_source_frame_indices[-1]
            if self.selected_source_frame_indices
            else None
        )

    @property
    def bin_count(self) -> int:
        return max(
            len(
                _build_bin_edges(
                    self.analysis_start_fs,
                    self.analysis_stop_fs,
                    self.colormap_timestep_fs,
                )
            )
            - 1,
            0,
        )

    @property
    def bin_size_fs(self) -> float:
        return self.colormap_timestep_fs

    def to_dict(self) -> dict[str, object]:
        return {
            "frame_format": self.frame_format,
            "total_frames": self.total_frames,
            "selected_frames": self.selected_frames,
            "resolved_box_dimensions": self.resolved_box_dimensions,
            "use_pbc": self.use_pbc,
            "first_frame_time_fs": self.first_frame_time_fs,
            "frame_timestep_fs": self.frame_timestep_fs,
            "frames_per_colormap_timestep": self.frames_per_colormap_timestep,
            "colormap_timestep_fs": self.colormap_timestep_fs,
            "bin_size_fs": self.colormap_timestep_fs,
            "analysis_start_fs": self.analysis_start_fs,
            "analysis_stop_fs": self.analysis_stop_fs,
            "folder_start_time_fs": self.folder_start_time_fs,
            "folder_start_time_source": self.folder_start_time_source,
            "time_source_label": self.time_source_label,
            "time_warnings": list(self.time_warnings),
            "first_selected_time_fs": self.first_selected_time_fs,
            "last_selected_time_fs": self.last_selected_time_fs,
            "first_selected_frame": self.first_selected_frame,
            "last_selected_frame": self.last_selected_frame,
            "first_selected_source_frame_index": (
                self.first_selected_source_frame_index
            ),
            "last_selected_source_frame_index": (
                self.last_selected_source_frame_index
            ),
            "bin_count": self.bin_count,
            "energy_file": (
                None if self.energy_file is None else str(self.energy_file)
            ),
        }


@dataclass(slots=True)
class _SeriesLifetimeMetrics:
    total_observations: int
    occupied_frames: int
    mean_count_per_frame: float
    occupancy_fraction: float
    association_events: int
    dissociation_events: int
    association_rate_per_ps: float
    dissociation_rate_per_ps: float
    completed_lifetimes_fs: tuple[float, ...]
    window_truncated_lifetimes_fs: tuple[float, ...]
    mean_lifetime_fs: float | None
    std_lifetime_fs: float | None


@dataclass(slots=True)
class _FrameTimeAxis:
    frame_times_fs: np.ndarray
    source_frame_indices: tuple[int | None, ...]
    time_source_label: str
    folder_start_time_fs: float | None
    folder_start_time_source: str | None
    warnings: tuple[str, ...]


@dataclass(slots=True)
class ClusterDynamicsResult:
    """Computed time-binned cluster-distribution analysis."""

    preview: ClusterDynamicsSelectionPreview
    frame_results: tuple[FrameClusterResult, ...]
    bin_edges_fs: np.ndarray
    frames_per_bin: np.ndarray
    total_clusters_per_bin: np.ndarray
    cluster_labels: tuple[str, ...]
    cluster_sizes: dict[str, int]
    raw_count_matrix: np.ndarray
    fraction_matrix: np.ndarray
    mean_count_matrix: np.ndarray
    frame_count_matrix: np.ndarray
    total_clusters_per_frame: np.ndarray
    lifetime_by_label: tuple[ClusterLifetimeSummary, ...]
    lifetime_by_size: tuple[ClusterSizeLifetimeSummary, ...]
    energy_data: CP2KEnergyData | None = None

    @property
    def analyzed_frames(self) -> int:
        return len(self.frame_results)

    @property
    def bin_count(self) -> int:
        return int(len(self.bin_edges_fs) - 1)

    @property
    def bin_centers_fs(self) -> np.ndarray:
        return (self.bin_edges_fs[:-1] + self.bin_edges_fs[1:]) / 2.0

    @property
    def frame_times_fs(self) -> np.ndarray:
        if not self.frame_results:
            return np.zeros(0, dtype=float)
        return np.asarray(
            [float(frame.time_fs or 0.0) for frame in self.frame_results],
            dtype=float,
        )

    def matrix(self, mode: DisplayMode = "fraction") -> np.ndarray:
        if mode == "count":
            return np.asarray(self.raw_count_matrix, dtype=float)
        if mode == "mean_count":
            return np.asarray(self.mean_count_matrix, dtype=float)
        return np.asarray(self.fraction_matrix, dtype=float)

    def time_edges(self, unit: TimeUnit = "fs") -> np.ndarray:
        return self.bin_edges_fs / _time_scale(unit)

    def bin_centers(self, unit: TimeUnit = "fs") -> np.ndarray:
        return self.bin_centers_fs / _time_scale(unit)

    def energy_series(
        self,
        series_name: EnergySeriesName,
        *,
        unit: TimeUnit = "fs",
    ) -> tuple[np.ndarray, np.ndarray, str]:
        if self.energy_data is None:
            raise ValueError("No CP2K .ener file was loaded for this result.")

        time_fs = np.asarray(self.energy_data.time_fs, dtype=float)
        mask = (time_fs >= float(self.bin_edges_fs[0])) & (
            time_fs <= float(self.bin_edges_fs[-1])
        )
        label_map = {
            "kinetic": "Kinetic Energy",
            "temperature": "Temperature (K)",
            "potential": "Potential Energy",
        }
        value = np.asarray(getattr(self.energy_data, series_name), dtype=float)
        return (
            time_fs[mask] / _time_scale(unit),
            value[mask],
            label_map[series_name],
        )


class ClusterDynamicsWorkflow:
    """Headless workflow for time-binned cluster-distribution
    analysis."""

    def __init__(
        self,
        frames_dir: str | Path,
        *,
        atom_type_definitions: AtomTypeDefinitions,
        pair_cutoff_definitions: PairCutoffDefinitions,
        box_dimensions: tuple[float, float, float] | None = None,
        use_pbc: bool = False,
        default_cutoff: float | None = None,
        shell_levels: tuple[int, ...] = (),
        shared_shells: bool = False,
        include_shell_atoms_in_stoichiometry: bool = False,
        search_mode: str = SEARCH_MODE_KDTREE,
        folder_start_time_fs: float | None = None,
        first_frame_time_fs: float = 0.0,
        frame_timestep_fs: float = 0.5,
        frames_per_colormap_timestep: int | None = None,
        colormap_timestep_fs: float | None = None,
        bin_size_fs: float | None = None,
        analysis_start_fs: float | None = None,
        analysis_stop_fs: float | None = None,
        energy_file: str | Path | None = None,
    ) -> None:
        self.frames_dir = Path(frames_dir)
        self.atom_type_definitions = atom_type_definitions
        self.pair_cutoff_definitions = pair_cutoff_definitions
        self.box_dimensions = box_dimensions
        self.use_pbc = bool(use_pbc)
        self.default_cutoff = default_cutoff
        self.shell_levels = shell_levels
        self.shared_shells = bool(shared_shells)
        self.include_shell_atoms_in_stoichiometry = bool(
            include_shell_atoms_in_stoichiometry
        )
        self.search_mode = normalize_search_mode(search_mode)
        self.folder_start_time_fs = (
            None
            if folder_start_time_fs is None
            else float(folder_start_time_fs)
        )
        self.fallback_first_frame_time_fs = float(first_frame_time_fs)
        self.frame_timestep_fs = _validate_positive_number(
            frame_timestep_fs,
            label="Frame timestep",
        )
        (
            self.frames_per_colormap_timestep,
            self.colormap_timestep_fs,
        ) = _resolve_colormap_timestep_settings(
            frame_timestep_fs=self.frame_timestep_fs,
            frames_per_colormap_timestep=frames_per_colormap_timestep,
            colormap_timestep_fs=colormap_timestep_fs,
            legacy_bin_size_fs=bin_size_fs,
            require_integral_ratio=True,
        )
        self.bin_size_fs = self.colormap_timestep_fs
        self.analysis_start_fs = (
            None if analysis_start_fs is None else float(analysis_start_fs)
        )
        self.analysis_stop_fs = (
            None if analysis_stop_fs is None else float(analysis_stop_fs)
        )
        self.energy_file = None if energy_file is None else Path(energy_file)
        self._cluster_workflow = ClusterWorkflow(
            frames_dir=self.frames_dir,
            atom_type_definitions=self.atom_type_definitions,
            pair_cutoff_definitions=self.pair_cutoff_definitions,
            box_dimensions=self.box_dimensions,
            use_pbc=self.use_pbc,
            default_cutoff=self.default_cutoff,
            shell_levels=self.shell_levels,
            shared_shells=self.shared_shells,
            include_shell_atoms_in_stoichiometry=(
                self.include_shell_atoms_in_stoichiometry
            ),
            search_mode=self.search_mode,
        )
        self._cached_preview: ClusterDynamicsSelectionPreview | None = None
        self._cached_energy: CP2KEnergyData | None = None

    def inspect(self) -> dict[str, object]:
        return self._cluster_workflow.inspect()

    def load_energy(self) -> CP2KEnergyData:
        if self.energy_file is None:
            raise ValueError("No CP2K .ener file was provided.")
        if self._cached_energy is None:
            self._cached_energy = CP2KEnergyData.from_file(self.energy_file)
        return self._cached_energy

    def preview_selection(self) -> ClusterDynamicsSelectionPreview:
        if self._cached_preview is not None:
            return self._cached_preview

        summary = self.inspect()
        frame_format, frame_paths = detect_frame_folder_mode(self.frames_dir)
        time_axis = _infer_frame_time_axis(
            self.frames_dir,
            frame_paths,
            frame_timestep_fs=self.frame_timestep_fs,
            folder_start_time_fs=self.folder_start_time_fs,
            fallback_first_frame_time_fs=self.fallback_first_frame_time_fs,
        )
        frame_times_fs = time_axis.frame_times_fs
        resolved_box = self._cluster_workflow.resolve_box_dimensions(
            box_dimensions=self.box_dimensions,
            use_pbc=self.use_pbc,
        )
        (
            analysis_start_fs,
            analysis_stop_fs,
            selected_indices,
        ) = _resolve_time_window(
            frame_times_fs=frame_times_fs,
            analysis_start_fs=self.analysis_start_fs,
            analysis_stop_fs=self.analysis_stop_fs,
            frame_timestep_fs=self.frame_timestep_fs,
        )
        selected_names = tuple(
            frame_paths[index].name for index in selected_indices
        )
        selected_source_frame_indices = tuple(
            time_axis.source_frame_indices[index] for index in selected_indices
        )
        first_selected_time_fs = (
            None
            if not selected_indices
            else float(frame_times_fs[selected_indices[0]])
        )
        last_selected_time_fs = (
            None
            if not selected_indices
            else float(frame_times_fs[selected_indices[-1]])
        )
        self._cached_preview = ClusterDynamicsSelectionPreview(
            summary=summary,
            frame_format=str(frame_format),
            resolved_box_dimensions=resolved_box,
            use_pbc=self.use_pbc,
            first_frame_time_fs=(
                float(frame_times_fs[0])
                if frame_times_fs.size
                else (
                    time_axis.folder_start_time_fs
                    if time_axis.folder_start_time_fs is not None
                    else self.fallback_first_frame_time_fs
                )
            ),
            frame_timestep_fs=self.frame_timestep_fs,
            frames_per_colormap_timestep=self.frames_per_colormap_timestep,
            colormap_timestep_fs=self.colormap_timestep_fs,
            analysis_start_fs=analysis_start_fs,
            analysis_stop_fs=analysis_stop_fs,
            first_selected_time_fs=first_selected_time_fs,
            last_selected_time_fs=last_selected_time_fs,
            selected_frame_indices=selected_indices,
            selected_frame_names=selected_names,
            selected_source_frame_indices=selected_source_frame_indices,
            energy_file=self.energy_file,
            folder_start_time_fs=time_axis.folder_start_time_fs,
            folder_start_time_source=time_axis.folder_start_time_source,
            time_source_label=time_axis.time_source_label,
            time_warnings=time_axis.warnings,
        )
        return self._cached_preview

    def analyze(
        self,
        *,
        progress_callback: callable | None = None,
    ) -> ClusterDynamicsResult:
        preview = self.preview_selection()
        if preview.selected_frames == 0:
            raise ValueError(
                "No extracted frames fall within the selected time window."
            )

        frame_format, frame_paths = detect_frame_folder_mode(self.frames_dir)
        selected_paths = [
            frame_paths[index] for index in preview.selected_frame_indices
        ]
        frame_times_fs = _infer_frame_time_axis(
            self.frames_dir,
            frame_paths,
            frame_timestep_fs=preview.frame_timestep_fs,
            folder_start_time_fs=preview.folder_start_time_fs,
            fallback_first_frame_time_fs=self.fallback_first_frame_time_fs,
        ).frame_times_fs
        selected_times = frame_times_fs[list(preview.selected_frame_indices)]

        frame_results: list[FrameClusterResult] = []
        per_frame_counts: list[Counter[str]] = []
        label_sizes: dict[str, int] = {}
        total_frames = len(selected_paths)

        for processed, (frame_index, frame_path, time_fs) in enumerate(
            zip(
                preview.selected_frame_indices,
                selected_paths,
                selected_times,
                strict=False,
            ),
            start=1,
        ):
            network = self._build_network(
                frame_format=str(frame_format),
                frame_path=frame_path,
                resolved_box_dimensions=preview.resolved_box_dimensions,
            )
            clusters = network.find_clusters(
                shell_levels=self.shell_levels,
                shared_shells=self.shared_shells,
            )
            frame_results.append(
                FrameClusterResult(
                    frame_index=int(frame_index),
                    time_fs=float(time_fs),
                    clusters=clusters,
                )
            )
            counts = Counter[str]()
            for cluster in clusters:
                label = stoichiometry_label(cluster.stoichiometry)
                counts[label] += 1
                label_sizes[label] = max(
                    label_sizes.get(label, 0),
                    sum(
                        int(value) for value in cluster.stoichiometry.values()
                    ),
                )
            per_frame_counts.append(counts)
            if progress_callback is not None:
                progress_callback(processed, total_frames, frame_path.name)

        cluster_labels = tuple(
            sorted(label_sizes, key=lambda label: (label_sizes[label], label))
        )
        label_index = {
            label: index for index, label in enumerate(cluster_labels)
        }
        bin_edges_fs = _build_bin_edges(
            preview.analysis_start_fs,
            preview.analysis_stop_fs,
            preview.colormap_timestep_fs,
        )
        raw_count_matrix = np.zeros(
            (len(cluster_labels), len(bin_edges_fs) - 1),
            dtype=float,
        )
        frame_count_matrix = np.zeros(
            (len(cluster_labels), len(frame_results)),
            dtype=float,
        )
        frames_per_bin = np.zeros(len(bin_edges_fs) - 1, dtype=float)
        total_clusters_per_bin = np.zeros(len(bin_edges_fs) - 1, dtype=float)
        total_clusters_per_frame = np.zeros(len(frame_results), dtype=float)

        if frame_results:
            selected_frame_times_fs = np.asarray(
                [float(frame.time_fs or 0.0) for frame in frame_results],
                dtype=float,
            )
            bin_indices = _assign_bins(selected_frame_times_fs, bin_edges_fs)
        else:
            selected_frame_times_fs = np.zeros(0, dtype=float)
            bin_indices = np.zeros(0, dtype=int)

        for frame_position, counts in enumerate(per_frame_counts):
            bin_index = int(bin_indices[frame_position])
            frames_per_bin[bin_index] += 1.0
            total_clusters = float(sum(counts.values()))
            total_clusters_per_bin[bin_index] += total_clusters
            total_clusters_per_frame[frame_position] = total_clusters
            for label, count in counts.items():
                row = label_index[label]
                raw_count_matrix[row, bin_index] += float(count)
                frame_count_matrix[row, frame_position] = float(count)

        fraction_matrix = _safe_divide(
            raw_count_matrix,
            total_clusters_per_bin[np.newaxis, :],
        )
        mean_count_matrix = _safe_divide(
            raw_count_matrix,
            frames_per_bin[np.newaxis, :],
        )
        lifetime_by_label = tuple(
            self._summarize_label_series(
                label=label,
                cluster_size=label_sizes[label],
                count_series=frame_count_matrix[label_index[label], :],
                frame_times_fs=selected_frame_times_fs,
                observation_start_fs=preview.analysis_start_fs,
                observation_stop_fs=preview.analysis_stop_fs,
            )
            for label in cluster_labels
        )

        size_series: dict[int, np.ndarray] = defaultdict(
            lambda: np.zeros(len(frame_results), dtype=float)
        )
        for label, size in label_sizes.items():
            size_series[size] += frame_count_matrix[label_index[label], :]
        lifetime_by_size = tuple(
            self._summarize_size_series(
                cluster_size=size,
                count_series=size_series[size],
                frame_times_fs=selected_frame_times_fs,
                observation_start_fs=preview.analysis_start_fs,
                observation_stop_fs=preview.analysis_stop_fs,
            )
            for size in sorted(size_series)
        )

        energy_data = (
            self.load_energy() if self.energy_file is not None else None
        )
        return ClusterDynamicsResult(
            preview=preview,
            frame_results=tuple(frame_results),
            bin_edges_fs=bin_edges_fs,
            frames_per_bin=frames_per_bin,
            total_clusters_per_bin=total_clusters_per_bin,
            cluster_labels=cluster_labels,
            cluster_sizes=dict(label_sizes),
            raw_count_matrix=raw_count_matrix,
            fraction_matrix=fraction_matrix,
            mean_count_matrix=mean_count_matrix,
            frame_count_matrix=frame_count_matrix,
            total_clusters_per_frame=total_clusters_per_frame,
            lifetime_by_label=lifetime_by_label,
            lifetime_by_size=lifetime_by_size,
            energy_data=energy_data,
        )

    def _build_network(
        self,
        *,
        frame_format: str,
        frame_path: Path,
        resolved_box_dimensions: tuple[float, float, float] | None,
    ) -> ClusterNetwork | XYZClusterNetwork:
        if frame_format == "pdb":
            structure = PDBStructure(
                filepath=frame_path,
                atom_type_definitions=self.atom_type_definitions,
                source_name=frame_path.stem,
            )
            return ClusterNetwork(
                pdb_structure=structure,
                atom_type_definitions=self.atom_type_definitions,
                pair_cutoffs_def=self.pair_cutoff_definitions,
                box_dimensions=resolved_box_dimensions,
                default_cutoff=self.default_cutoff,
                use_pbc=self.use_pbc,
                include_shell_atoms_in_stoichiometry=(
                    self.include_shell_atoms_in_stoichiometry
                ),
                search_mode=self.search_mode,
            )

        structure = XYZStructure(
            filepath=frame_path,
            atom_type_definitions=self.atom_type_definitions,
            source_name=frame_path.stem,
        )
        return XYZClusterNetwork(
            xyz_structure=structure,
            atom_type_definitions=self.atom_type_definitions,
            pair_cutoffs_def=self.pair_cutoff_definitions,
            box_dimensions=resolved_box_dimensions,
            default_cutoff=self.default_cutoff,
            use_pbc=self.use_pbc,
            include_shell_atoms_in_stoichiometry=(
                self.include_shell_atoms_in_stoichiometry
            ),
            search_mode=self.search_mode,
        )

    def _summarize_label_series(
        self,
        *,
        label: str,
        cluster_size: int,
        count_series: np.ndarray,
        frame_times_fs: np.ndarray,
        observation_start_fs: float,
        observation_stop_fs: float,
    ) -> ClusterLifetimeSummary:
        metrics = _summarize_series_lifetimes(
            count_series,
            frame_times_fs=frame_times_fs,
            observation_start_fs=observation_start_fs,
            observation_stop_fs=observation_stop_fs,
        )
        return ClusterLifetimeSummary(
            label=label,
            cluster_size=cluster_size,
            total_observations=metrics.total_observations,
            occupied_frames=metrics.occupied_frames,
            mean_count_per_frame=metrics.mean_count_per_frame,
            occupancy_fraction=metrics.occupancy_fraction,
            association_events=metrics.association_events,
            dissociation_events=metrics.dissociation_events,
            association_rate_per_ps=metrics.association_rate_per_ps,
            dissociation_rate_per_ps=metrics.dissociation_rate_per_ps,
            completed_lifetime_count=len(metrics.completed_lifetimes_fs),
            window_truncated_lifetime_count=len(
                metrics.window_truncated_lifetimes_fs
            ),
            mean_lifetime_fs=metrics.mean_lifetime_fs,
            std_lifetime_fs=metrics.std_lifetime_fs,
        )

    def _summarize_size_series(
        self,
        *,
        cluster_size: int,
        count_series: np.ndarray,
        frame_times_fs: np.ndarray,
        observation_start_fs: float,
        observation_stop_fs: float,
    ) -> ClusterSizeLifetimeSummary:
        metrics = _summarize_series_lifetimes(
            count_series,
            frame_times_fs=frame_times_fs,
            observation_start_fs=observation_start_fs,
            observation_stop_fs=observation_stop_fs,
        )
        return ClusterSizeLifetimeSummary(
            cluster_size=cluster_size,
            total_observations=metrics.total_observations,
            occupied_frames=metrics.occupied_frames,
            mean_count_per_frame=metrics.mean_count_per_frame,
            occupancy_fraction=metrics.occupancy_fraction,
            association_events=metrics.association_events,
            dissociation_events=metrics.dissociation_events,
            association_rate_per_ps=metrics.association_rate_per_ps,
            dissociation_rate_per_ps=metrics.dissociation_rate_per_ps,
            completed_lifetime_count=len(metrics.completed_lifetimes_fs),
            window_truncated_lifetime_count=len(
                metrics.window_truncated_lifetimes_fs
            ),
            mean_lifetime_fs=metrics.mean_lifetime_fs,
            std_lifetime_fs=metrics.std_lifetime_fs,
        )


def _validate_positive_number(value: float, *, label: str) -> float:
    normalized = float(value)
    if normalized <= 0.0:
        raise ValueError(f"{label} must be greater than zero.")
    return normalized


def _validate_positive_integer(value: int | float, *, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a whole number greater than zero.")
    normalized = float(value)
    if normalized <= 0.0 or not normalized.is_integer():
        raise ValueError(f"{label} must be a whole number greater than zero.")
    return int(normalized)


def _resolve_colormap_timestep_settings(
    *,
    frame_timestep_fs: float,
    frames_per_colormap_timestep: int | float | None,
    colormap_timestep_fs: float | None,
    legacy_bin_size_fs: float | None,
    require_integral_ratio: bool,
) -> tuple[int | None, float]:
    resolved_colormap_timestep_fs = None
    if colormap_timestep_fs is not None:
        resolved_colormap_timestep_fs = _validate_positive_number(
            colormap_timestep_fs,
            label="Colormap timestep",
        )
    elif legacy_bin_size_fs is not None:
        resolved_colormap_timestep_fs = _validate_positive_number(
            legacy_bin_size_fs,
            label="Colormap timestep",
        )

    if frames_per_colormap_timestep is not None:
        resolved_frames = _validate_positive_integer(
            frames_per_colormap_timestep,
            label="Frames per colormap timestep",
        )
        expected_colormap_timestep_fs = float(frame_timestep_fs) * float(
            resolved_frames
        )
        if resolved_colormap_timestep_fs is None:
            return resolved_frames, expected_colormap_timestep_fs
        tolerance_fs = (
            max(
                abs(expected_colormap_timestep_fs),
                abs(resolved_colormap_timestep_fs),
                1.0,
            )
            * 1.0e-9
        )
        if (
            abs(expected_colormap_timestep_fs - resolved_colormap_timestep_fs)
            > tolerance_fs
        ):
            raise ValueError(
                "Frames per colormap timestep does not match the resolved "
                "colormap timestep."
            )
        return resolved_frames, expected_colormap_timestep_fs

    if resolved_colormap_timestep_fs is None:
        return 1, float(frame_timestep_fs)

    ratio = resolved_colormap_timestep_fs / float(frame_timestep_fs)
    nearest_ratio = int(round(ratio))
    expected_colormap_timestep_fs = float(frame_timestep_fs) * float(
        max(nearest_ratio, 1)
    )
    tolerance_fs = (
        max(
            abs(expected_colormap_timestep_fs),
            abs(resolved_colormap_timestep_fs),
            1.0,
        )
        * 1.0e-9
    )
    if (
        abs(expected_colormap_timestep_fs - resolved_colormap_timestep_fs)
        <= tolerance_fs
    ):
        return max(nearest_ratio, 1), expected_colormap_timestep_fs
    if require_integral_ratio:
        raise ValueError(
            "Colormap timestep must be an integer multiple of the frame "
            "timestep."
        )
    return None, resolved_colormap_timestep_fs


def _time_scale(unit: TimeUnit) -> float:
    if unit == "ps":
        return 1000.0
    return 1.0


def _frame_times(
    total_frames: int,
    *,
    first_frame_time_fs: float,
    frame_timestep_fs: float,
) -> np.ndarray:
    return first_frame_time_fs + (
        np.arange(max(int(total_frames), 0), dtype=float) * frame_timestep_fs
    )


def _infer_frame_time_axis(
    frames_dir: Path,
    frame_paths: tuple[Path, ...] | list[Path],
    *,
    frame_timestep_fs: float,
    folder_start_time_fs: float | None,
    fallback_first_frame_time_fs: float,
) -> _FrameTimeAxis:
    metadata_payload = _load_mdtrajectory_export_metadata(frames_dir)
    resolved_folder_start_time_fs, folder_start_source = (
        _resolve_folder_start_time(
            frames_dir=frames_dir,
            explicit_folder_start_time_fs=folder_start_time_fs,
            metadata_payload=metadata_payload,
        )
    )
    warnings: list[str] = []

    metadata_time_axis = _frame_times_from_mdtrajectory_metadata(
        metadata_payload,
        frame_paths,
    )
    if metadata_time_axis is not None:
        frame_times_fs, source_frame_indices = metadata_time_axis
        time_source_label = "mdtrajectory export metadata"
    else:
        if metadata_payload is not None:
            warnings.append(
                "Found mdtrajectory export metadata, but it did not provide "
                "usable times for every extracted frame. Falling back to the "
                "frame filenames and timestep."
            )
        source_frame_indices = tuple(
            _parse_frame_filename_index(path.name) for path in frame_paths
        )
        if source_frame_indices and all(
            index is not None for index in source_frame_indices
        ):
            frame_times_fs = np.asarray(
                [
                    float(index) * float(frame_timestep_fs)
                    for index in source_frame_indices
                ],
                dtype=float,
            )
            time_source_label = "Frame filenames x timestep"
        else:
            sequential_start_time_fs = (
                resolved_folder_start_time_fs
                if resolved_folder_start_time_fs is not None
                else float(fallback_first_frame_time_fs)
            )
            frame_times_fs = _frame_times(
                len(frame_paths),
                first_frame_time_fs=sequential_start_time_fs,
                frame_timestep_fs=frame_timestep_fs,
            )
            time_source_label = "Sequential frames from start time"
            if resolved_folder_start_time_fs is None:
                warnings.append(
                    "Start/cutoff time metadata was not found in the folder "
                    "name or mdtrajectory export metadata. Sequential frame "
                    "times are being generated from the fallback start time."
                )
            else:
                warnings.append(
                    "Using the folder/start time as the first extracted "
                    "frame time because the frame filenames do not expose "
                    "their original source-frame indices."
                )

    if frame_times_fs.size and resolved_folder_start_time_fs is not None:
        first_resolved_time_fs = float(frame_times_fs[0])
        mismatch_tolerance_fs = max(float(frame_timestep_fs) * 0.5, 1.0e-9)
        if (
            abs(first_resolved_time_fs - resolved_folder_start_time_fs)
            > mismatch_tolerance_fs
        ):
            warnings.append(
                "Folder cutoff/start time is "
                f"{resolved_folder_start_time_fs:.3f} fs, but the first "
                "resolved extracted-frame time is "
                f"{first_resolved_time_fs:.3f} fs. The analysis uses the "
                "resolved frame times for plotting and kinetics."
            )

    return _FrameTimeAxis(
        frame_times_fs=frame_times_fs,
        source_frame_indices=source_frame_indices,
        time_source_label=time_source_label,
        folder_start_time_fs=resolved_folder_start_time_fs,
        folder_start_time_source=folder_start_source,
        warnings=tuple(warnings),
    )


def _resolve_folder_start_time(
    *,
    frames_dir: Path,
    explicit_folder_start_time_fs: float | None,
    metadata_payload: dict[str, object] | None,
) -> tuple[float | None, str | None]:
    if explicit_folder_start_time_fs is not None:
        return float(explicit_folder_start_time_fs), "manual field"

    if metadata_payload is not None:
        selection_payload = metadata_payload.get("selection")
        if isinstance(selection_payload, dict):
            first_time_fs = selection_payload.get("first_time_fs")
            if first_time_fs is not None:
                return float(first_time_fs), "mdtrajectory export metadata"
            applied_cutoff_fs = selection_payload.get("applied_cutoff_fs")
            if applied_cutoff_fs is not None:
                return float(applied_cutoff_fs), "mdtrajectory export metadata"

    folder_start_time_fs = _parse_folder_start_time_from_name(frames_dir.name)
    if folder_start_time_fs is not None:
        return folder_start_time_fs, "folder name"
    return None, None


def _load_mdtrajectory_export_metadata(
    frames_dir: Path,
) -> dict[str, object] | None:
    metadata_path = frames_dir / EXPORT_METADATA_FILENAME
    if not metadata_path.is_file():
        return None
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _frame_times_from_mdtrajectory_metadata(
    metadata_payload: dict[str, object] | None,
    frame_paths: tuple[Path, ...] | list[Path],
) -> tuple[np.ndarray, tuple[int | None, ...]] | None:
    if metadata_payload is None:
        return None

    written_frames = metadata_payload.get("written_frames")
    if not isinstance(written_frames, list):
        return None

    mapping: dict[str, tuple[float, int | None]] = {}
    for entry in written_frames:
        if not isinstance(entry, dict):
            continue
        filename = entry.get("filename")
        time_fs = entry.get("time_fs")
        if not isinstance(filename, str) or time_fs is None:
            continue
        frame_index = entry.get("frame_index")
        mapping[filename] = (
            float(time_fs),
            None if frame_index is None else int(frame_index),
        )

    if not mapping:
        return None

    resolved_times: list[float] = []
    resolved_indices: list[int | None] = []
    for frame_path in frame_paths:
        payload = mapping.get(frame_path.name)
        if payload is None:
            return None
        time_fs, frame_index = payload
        resolved_times.append(float(time_fs))
        resolved_indices.append(frame_index)

    return np.asarray(resolved_times, dtype=float), tuple(resolved_indices)


def _parse_frame_filename_index(filename: str) -> int | None:
    match = _FRAME_FILENAME_PATTERN.match(filename.strip())
    if match is None:
        return None
    return int(match.group("index"))


def _parse_folder_start_time_from_name(folder_name: str) -> float | None:
    match = _FOLDER_START_TIME_PATTERN.search(folder_name.strip())
    if match is None:
        match = _LEGACY_FOLDER_START_TIME_PATTERN.search(
            folder_name.strip()
        )
    if match is None:
        return None
    return float(match.group("value").replace("_", ".").replace("p", "."))


def _resolve_time_window(
    *,
    frame_times_fs: np.ndarray,
    analysis_start_fs: float | None,
    analysis_stop_fs: float | None,
    frame_timestep_fs: float,
) -> tuple[float, float, tuple[int, ...]]:
    if frame_times_fs.size == 0:
        start = 0.0 if analysis_start_fs is None else float(analysis_start_fs)
        stop = start + frame_timestep_fs
        return start, stop, ()

    requested_start = (
        float(frame_times_fs[0])
        if analysis_start_fs is None
        else float(analysis_start_fs)
    )
    requested_stop = (
        float(frame_times_fs[-1]) + float(frame_timestep_fs)
        if analysis_stop_fs is None
        else float(analysis_stop_fs)
    )
    if requested_stop <= requested_start:
        requested_stop = requested_start + float(frame_timestep_fs)

    selected_indices = tuple(
        int(index)
        for index, time_fs in enumerate(frame_times_fs)
        if float(time_fs) >= requested_start
        and (
            float(time_fs) < requested_stop
            if analysis_stop_fs is None
            else float(time_fs) <= requested_stop
        )
    )
    return requested_start, requested_stop, selected_indices


def _build_bin_edges(
    analysis_start_fs: float,
    analysis_stop_fs: float,
    colormap_timestep_fs: float,
) -> np.ndarray:
    start = float(analysis_start_fs)
    stop = float(analysis_stop_fs)
    if stop <= start:
        stop = start + float(colormap_timestep_fs)
    n_bins = max(
        int(np.ceil((stop - start) / float(colormap_timestep_fs))),
        1,
    )
    edges = start + (
        np.arange(n_bins + 1, dtype=float) * float(colormap_timestep_fs)
    )
    if edges[-1] < stop:
        edges = np.append(edges, stop)
    else:
        edges[-1] = max(edges[-1], stop)
    return edges


def _assign_bins(
    frame_times_fs: np.ndarray, bin_edges_fs: np.ndarray
) -> np.ndarray:
    bin_indices = (
        np.searchsorted(bin_edges_fs, frame_times_fs, side="right") - 1
    )
    return np.clip(bin_indices, 0, len(bin_edges_fs) - 2)


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=denominator > 0.0,
    )


def _summarize_series_lifetimes(
    count_series: np.ndarray,
    *,
    frame_times_fs: np.ndarray,
    observation_start_fs: float,
    observation_stop_fs: float,
) -> _SeriesLifetimeMetrics:
    normalized = np.asarray(count_series, dtype=int)
    resolved_frame_times_fs = np.asarray(frame_times_fs, dtype=float)
    total_frames = len(normalized)
    if total_frames != len(resolved_frame_times_fs):
        raise ValueError(
            "Count-series length does not match the resolved frame-time axis."
        )
    total_observations = int(normalized.sum())
    occupied_frames = int(np.count_nonzero(normalized > 0))
    mean_count_per_frame = float(normalized.mean()) if total_frames else 0.0
    occupancy_fraction = (
        float(occupied_frames / total_frames) if total_frames else 0.0
    )
    association_events = 0
    dissociation_events = 0
    active_instances: deque[tuple[float, bool]] = deque()
    completed_lifetimes_fs: list[float] = []
    window_truncated_lifetimes_fs: list[float] = []

    if total_frames:
        for _ in range(int(normalized[0])):
            active_instances.append((float(resolved_frame_times_fs[0]), True))

    for frame_index in range(1, total_frames):
        previous_count = int(normalized[frame_index - 1])
        current_count = int(normalized[frame_index])
        current_time_fs = float(resolved_frame_times_fs[frame_index])
        if current_count > previous_count:
            births = current_count - previous_count
            association_events += births
            for _ in range(births):
                active_instances.append((current_time_fs, False))
        elif current_count < previous_count:
            deaths = previous_count - current_count
            dissociation_events += deaths
            for _ in range(deaths):
                if not active_instances:
                    break
                start_time_fs, left_censored = active_instances.popleft()
                lifetime_fs = max(current_time_fs - start_time_fs, 0.0)
                if left_censored:
                    window_truncated_lifetimes_fs.append(lifetime_fs)
                else:
                    completed_lifetimes_fs.append(lifetime_fs)

    if total_frames:
        observation_window_stop_fs = max(
            float(observation_stop_fs),
            float(resolved_frame_times_fs[-1]),
        )
    else:
        observation_window_stop_fs = float(observation_stop_fs)

    for start_time_fs, _left_censored in active_instances:
        lifetime_fs = max(observation_window_stop_fs - start_time_fs, 0.0)
        window_truncated_lifetimes_fs.append(lifetime_fs)

    completed_array = np.asarray(completed_lifetimes_fs, dtype=float)
    if completed_array.size:
        mean_lifetime_fs = float(completed_array.mean())
        std_lifetime_fs = float(completed_array.std(ddof=0))
    else:
        mean_lifetime_fs = None
        std_lifetime_fs = None

    observation_time_ps = max(
        (float(observation_window_stop_fs) - float(observation_start_fs))
        / 1000.0,
        1e-12,
    )
    return _SeriesLifetimeMetrics(
        total_observations=total_observations,
        occupied_frames=occupied_frames,
        mean_count_per_frame=mean_count_per_frame,
        occupancy_fraction=occupancy_fraction,
        association_events=association_events,
        dissociation_events=dissociation_events,
        association_rate_per_ps=float(
            association_events / observation_time_ps
        ),
        dissociation_rate_per_ps=float(
            dissociation_events / observation_time_ps
        ),
        completed_lifetimes_fs=tuple(completed_lifetimes_fs),
        window_truncated_lifetimes_fs=tuple(window_truncated_lifetimes_fs),
        mean_lifetime_fs=mean_lifetime_fs,
        std_lifetime_fs=std_lifetime_fs,
    )


__all__ = [
    "ClusterDynamicsResult",
    "ClusterDynamicsSelectionPreview",
    "ClusterDynamicsWorkflow",
    "ClusterLifetimeSummary",
    "ClusterSizeLifetimeSummary",
]
