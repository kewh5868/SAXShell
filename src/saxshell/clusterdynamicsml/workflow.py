from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from saxshell.cluster import PDBShellReferenceDefinition, PairCutoffDefinitions
from saxshell.cluster.clusternetwork import (
    detect_frame_folder_mode,
    stoichiometry_label,
)
from saxshell.clusterdynamics import (
    ClusterDynamicsResult,
    ClusterDynamicsSelectionPreview,
    ClusterDynamicsWorkflow,
    ClusterLifetimeSummary,
)
from saxshell.saxs.debye.profiles import (
    b_factor_from_sigma,
    compute_debye_intensity,
    compute_debye_intensity_with_debye_waller,
    load_structure_file,
)
from saxshell.saxs.project_manager import (
    ExperimentalDataSummary,
    SAXSProjectManager,
    build_project_paths,
    load_built_component_q_range,
    load_experimental_data_file,
)
from saxshell.structure import AtomTypeDefinitions, PDBAtom, PDBStructure
from saxshell.xyz2pdb import list_reference_library, resolve_reference_path
from saxshell.xyz2pdb.workflow import (
    rotation_matrix_about_axis,
    rotation_matrix_from_to,
)

PredictionProgressCallback = Callable[[str], None]
_STOICHIOMETRY_TOKEN_PATTERN = re.compile(r"([A-Z][a-z]*)(\d*)")
_DEFAULT_Q_MIN = 0.02
_DEFAULT_Q_MAX = 1.20
_DEFAULT_Q_POINTS = 250
_DEFAULT_SHARE_THRESHOLD = 0.02
_RIDGE_REGULARIZATION = 1e-6


@dataclass(slots=True)
class ClusterStructureObservation:
    """Aggregated structure information for one observed
    stoichiometry."""

    label: str
    node_count: int
    element_counts: dict[str, int]
    file_count: int
    representative_path: Path | None
    structure_dir: Path
    motifs: tuple[str, ...]
    mean_atom_count: float
    mean_radius_of_gyration: float
    mean_max_radius: float
    mean_semiaxis_a: float
    mean_semiaxis_b: float
    mean_semiaxis_c: float


@dataclass(slots=True)
class ClusterDynamicsMLTrainingObservation:
    """Joined kinetics and structure descriptors for one label."""

    label: str
    node_count: int
    cluster_size: int
    element_counts: dict[str, int]
    file_count: int
    representative_path: Path | None
    structure_dir: Path
    motifs: tuple[str, ...]
    mean_atom_count: float
    mean_radius_of_gyration: float
    mean_max_radius: float
    mean_semiaxis_a: float
    mean_semiaxis_b: float
    mean_semiaxis_c: float
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

    @property
    def stability_weight(self) -> float:
        finite_lifetime = (
            0.0 if self.mean_lifetime_fs is None else self.mean_lifetime_fs
        )
        return float(
            1.0
            + max(self.file_count, 0) / 10.0
            + max(self.completed_lifetime_count, 0) / 5.0
            + max(self.mean_count_per_frame, 0.0) * 5.0
            + max(self.occupancy_fraction, 0.0) * 5.0
            + finite_lifetime / 100.0
        )


@dataclass(slots=True)
class PredictedClusterCandidate:
    """Predicted larger-cluster candidate."""

    target_node_count: int
    rank: int
    label: str
    element_counts: dict[str, int]
    predicted_mean_count_per_frame: float
    predicted_occupancy_fraction: float
    predicted_mean_lifetime_fs: float
    predicted_association_rate_per_ps: float
    predicted_dissociation_rate_per_ps: float
    predicted_mean_radius_of_gyration: float
    predicted_mean_max_radius: float
    predicted_mean_semiaxis_a: float
    predicted_mean_semiaxis_b: float
    predicted_mean_semiaxis_c: float
    predicted_population_share: float
    predicted_stability_score: float
    source_label: str | None
    notes: str
    generated_elements: tuple[str, ...]
    generated_coordinates: np.ndarray


@dataclass(slots=True)
class SAXSComponentWeight:
    label: str
    weight: float
    source: str
    profile_path: Path | None = None
    structure_path: Path | None = None


@dataclass(slots=True)
class ClusterDynamicsMLSAXSComparison:
    """Cluster-only predicted-structures SAXS comparison trace."""

    q_values: np.ndarray
    observed_raw_model_intensity: np.ndarray | None
    observed_fitted_model_intensity: np.ndarray | None
    observed_rmse: float | None
    raw_model_intensity: np.ndarray
    fitted_model_intensity: np.ndarray
    experimental_intensity: np.ndarray | None
    residuals: np.ndarray | None
    scale_factor: float
    offset: float
    rmse: float | None
    component_weights: tuple[SAXSComponentWeight, ...]
    experimental_data_path: Path | None
    component_output_dir: Path | None = None
    predicted_structure_dir: Path | None = None


@dataclass(slots=True)
class DebyeWallerPairEstimate:
    source: str
    method: str
    label: str
    node_count: int
    candidate_rank: int | None
    element_a: str
    element_b: str
    sigma: float
    b_factor: float
    support_count: int
    aligned_pair_count: int
    source_label: str | None = None


@dataclass(slots=True)
class _ResolvedSAXSComponent:
    label: str
    weight: float
    source: str
    trace: np.ndarray
    profile_path: Path | None = None
    structure_path: Path | None = None


@dataclass(slots=True)
class ClusterDynamicsMLPreview:
    """Preview metadata for the separate prediction workflow."""

    dynamics_preview: ClusterDynamicsSelectionPreview
    clusters_dir: Path | None
    project_dir: Path | None
    experimental_data_path: Path | None
    structure_label_count: int
    total_structure_files: int
    observed_node_counts: tuple[int, ...]
    target_node_counts: tuple[int, ...]
    warnings: tuple[str, ...] = ()


@dataclass(slots=True)
class ClusterDynamicsMLResult:
    """End-to-end result for the experimental prediction workflow."""

    dynamics_result: ClusterDynamicsResult
    preview: ClusterDynamicsMLPreview
    structure_observations: tuple[ClusterStructureObservation, ...]
    training_observations: tuple[ClusterDynamicsMLTrainingObservation, ...]
    predictions: tuple[PredictedClusterCandidate, ...]
    debye_waller_estimates: tuple[DebyeWallerPairEstimate, ...]
    saxs_comparison: ClusterDynamicsMLSAXSComparison | None
    max_observed_node_count: int
    max_predicted_node_count: int | None
    prediction_population_share_threshold: float


@dataclass(slots=True)
class _PropertyModel:
    coefficients: np.ndarray | None
    constant_value: float | None
    transform: str
    default_value: float
    lower_bound: float | None = None
    upper_bound: float | None = None

    def predict(self, features: np.ndarray) -> float:
        if self.constant_value is not None:
            transformed = float(self.constant_value)
        elif self.coefficients is None:
            transformed = float(self.default_value)
        else:
            transformed = float(
                np.asarray(features, dtype=float) @ self.coefficients
            )
        if self.transform == "log1p":
            value = float(np.expm1(transformed))
        else:
            value = transformed
        if self.lower_bound is not None:
            value = max(self.lower_bound, value)
        if self.upper_bound is not None:
            value = min(self.upper_bound, value)
        return float(value)


@dataclass(slots=True)
class _TrainingGeometryStatistics:
    atom_type_by_element: dict[str, str]
    node_elements: tuple[str, ...]
    tracked_atom_types: tuple[str, ...]
    node_bond_length: float
    bond_length_medians: dict[tuple[str, str], float]
    contact_distance_medians: dict[tuple[str, str], float]
    geometry_contact_distance_medians: dict[tuple[str, str], float]
    node_angle_medians: dict[tuple[str, str], float]
    node_coordination_medians: dict[str, float]
    non_node_node_coordination_medians: dict[str, float]
    atom_coordination_medians: dict[tuple[str, str], float]


@dataclass(slots=True)
class _DebyeWallerPairModel:
    sigma_model: _PropertyModel
    support_count: int


@dataclass(slots=True)
class _LoadedMLStructure:
    coordinates: np.ndarray
    elements: list[str]
    pdb_atoms: tuple[PDBAtom, ...] = ()


@dataclass(slots=True)
class _ResolvedPDBShellReference:
    shell_element: str
    shell_residue: str | None
    reference_name: str
    reference_path: Path
    reference_residue_name: str
    reference_atoms: tuple[PDBAtom, ...]
    anchor_atom_name: str
    anchor_atom_element: str
    anchor_atom_index: int
    reference_outward_vector: np.ndarray
    backbone_atom1_name: str
    backbone_atom2_name: str
    backbone_atom1_element: str
    backbone_atom2_element: str
    backbone_atom1_index: int
    backbone_atom2_index: int
    backbone_distance: float


class ClusterDynamicsMLWorkflow:
    """Predict larger-cluster states from smaller-cluster data."""

    def __init__(
        self,
        frames_dir: str | Path,
        *,
        atom_type_definitions: AtomTypeDefinitions,
        pair_cutoff_definitions: PairCutoffDefinitions,
        clusters_dir: str | Path | None = None,
        project_dir: str | Path | None = None,
        experimental_data_file: str | Path | None = None,
        box_dimensions: tuple[float, float, float] | None = None,
        use_pbc: bool = False,
        default_cutoff: float | None = None,
        shell_levels: tuple[int, ...] = (),
        shared_shells: bool = False,
        include_shell_atoms_in_stoichiometry: bool = False,
        search_mode: str = "kdtree",
        pdb_shell_reference_definitions: Sequence[
            PDBShellReferenceDefinition
        ] = (),
        folder_start_time_fs: float | None = None,
        first_frame_time_fs: float = 0.0,
        frame_timestep_fs: float = 0.5,
        frames_per_colormap_timestep: int = 1,
        analysis_start_fs: float | None = None,
        analysis_stop_fs: float | None = None,
        energy_file: str | Path | None = None,
        target_node_counts: tuple[int, ...] | None = None,
        max_target_node_count: int | None = None,
        candidates_per_size: int = 3,
        prediction_population_share_threshold: float = _DEFAULT_SHARE_THRESHOLD,
        q_min: float | None = None,
        q_max: float | None = None,
        q_points: int = _DEFAULT_Q_POINTS,
    ) -> None:
        self.frames_dir = Path(frames_dir).expanduser().resolve()
        self.atom_type_definitions = atom_type_definitions
        self.pair_cutoff_definitions = pair_cutoff_definitions
        self.clusters_dir = (
            None
            if clusters_dir is None
            else Path(clusters_dir).expanduser().resolve()
        )
        self.project_dir = (
            None
            if project_dir is None
            else Path(project_dir).expanduser().resolve()
        )
        self.experimental_data_file = (
            None
            if experimental_data_file is None
            else Path(experimental_data_file).expanduser().resolve()
        )
        self.box_dimensions = box_dimensions
        self.use_pbc = bool(use_pbc)
        self.default_cutoff = default_cutoff
        self.shell_levels = tuple(int(level) for level in shell_levels)
        self.shared_shells = bool(shared_shells)
        self.include_shell_atoms_in_stoichiometry = bool(
            include_shell_atoms_in_stoichiometry
        )
        self.search_mode = str(search_mode)
        self.pdb_shell_reference_definitions = tuple(
            pdb_shell_reference_definitions
        )
        self.folder_start_time_fs = folder_start_time_fs
        self.first_frame_time_fs = float(first_frame_time_fs)
        self.frame_timestep_fs = float(frame_timestep_fs)
        self.frames_per_colormap_timestep = max(
            int(frames_per_colormap_timestep),
            1,
        )
        self.analysis_start_fs = analysis_start_fs
        self.analysis_stop_fs = analysis_stop_fs
        self.energy_file = (
            None
            if energy_file is None
            else Path(energy_file).expanduser().resolve()
        )
        self.target_node_counts = (
            None
            if target_node_counts is None
            else tuple(
                sorted(
                    {
                        int(value)
                        for value in target_node_counts
                        if int(value) > 0
                    }
                )
            )
        )
        self.max_target_node_count = (
            None
            if max_target_node_count is None
            else int(max_target_node_count)
        )
        self.candidates_per_size = max(int(candidates_per_size), 1)
        self.prediction_population_share_threshold = max(
            float(prediction_population_share_threshold),
            0.0,
        )
        self.q_min = None if q_min is None else float(q_min)
        self.q_max = None if q_max is None else float(q_max)
        self.q_points = max(int(q_points), 2)
        self._project_manager = SAXSProjectManager()
        self._cached_frames_input_format: str | None = None
        self._cached_resolved_pdb_shell_references: tuple[
            _ResolvedPDBShellReference, ...
        ] | None = None

    def preview_selection(self) -> ClusterDynamicsMLPreview:
        dynamics_preview = (
            self._build_cluster_dynamics_workflow().preview_selection()
        )
        resolved_clusters_dir, resolved_experimental, warnings = (
            self._resolve_project_inputs()
        )
        structure_label_count = 0
        total_structure_files = 0
        observed_node_counts: tuple[int, ...] = ()
        if (
            resolved_clusters_dir is not None
            and resolved_clusters_dir.is_dir()
        ):
            label_map = self._list_structure_labels(resolved_clusters_dir)
            structure_label_count = len(label_map)
            total_structure_files = sum(
                len(paths) for paths in label_map.values()
            )
            observed_node_counts = tuple(
                sorted(
                    {
                        self._node_count_from_counts(
                            _parse_stoichiometry_label(label)
                        )
                        for label in label_map
                        if self._node_count_from_counts(
                            _parse_stoichiometry_label(label)
                        )
                        > 0
                    }
                )
            )
        target_counts = self._resolve_target_node_counts(observed_node_counts)
        return ClusterDynamicsMLPreview(
            dynamics_preview=dynamics_preview,
            clusters_dir=resolved_clusters_dir,
            project_dir=self.project_dir,
            experimental_data_path=resolved_experimental,
            structure_label_count=structure_label_count,
            total_structure_files=total_structure_files,
            observed_node_counts=observed_node_counts,
            target_node_counts=target_counts,
            warnings=warnings,
        )

    def analyze(
        self,
        *,
        progress_callback: PredictionProgressCallback | None = None,
    ) -> ClusterDynamicsMLResult:
        total_steps = 7
        preview = self.preview_selection()
        resolved_clusters_dir = preview.clusters_dir
        if resolved_clusters_dir is None or not resolved_clusters_dir.is_dir():
            raise ValueError(
                "Select a cluster-structures directory, or provide a SAXSShell "
                "project with a saved clusters directory before running the "
                "prediction workflow."
            )
        self._emit(
            progress_callback,
            self._status_message(
                1,
                total_steps,
                "Extracting time-binned clusters and lifetime summaries.",
                "Reading the selected frames, assigning cluster labels, and "
                "building the observed population and lifetime summaries.",
            ),
        )
        dynamics_result = self._build_cluster_dynamics_workflow().analyze()
        self._emit(
            progress_callback,
            self._status_message(
                2,
                total_steps,
                "Loading observed reference structures.",
                "Scanning the smaller-cluster structure library grouped by "
                "stoichiometry label and selecting representative files.",
            ),
        )
        structure_observations = self._build_structure_observations(
            resolved_clusters_dir
        )
        self._emit(
            progress_callback,
            self._status_message(
                3,
                total_steps,
                "Joining kinetics and structure descriptors.",
                "Combining the cluster-dynamics lifetimes with the observed "
                "structure counts, radii, semiaxes, and motif information.",
            ),
        )
        training_observations = self._build_training_observations(
            dynamics_result,
            structure_observations,
        )
        if not training_observations:
            raise ValueError(
                "No overlapping structure labels could be matched to the "
                "observed cluster labels. The prediction workflow requires "
                "smaller-cluster structure files named by stoichiometry label."
            )
        observed_node_counts = tuple(
            sorted(
                {
                    entry.node_count
                    for entry in training_observations
                    if entry.node_count > 0
                }
            )
        )
        if not observed_node_counts:
            raise ValueError(
                "Could not infer any node-count observations from the structure labels."
            )
        if len(observed_node_counts) < 2:
            raise ValueError(
                "At least two observed node counts are required for extrapolation."
            )
        self._emit(
            progress_callback,
            self._status_message(
                4,
                total_steps,
                "Computing bond-length, bond-angle, and coordination distributions.",
                "Learning node, linker, shell, and non-node contact statistics "
                "from the observed reference structures.",
            ),
        )
        geometry_statistics = self._collect_training_geometry_statistics(
            training_observations
        )
        self._emit(
            progress_callback,
            self._status_message(
                5,
                total_steps,
                "Training ML regression models and scoring predicted structures.",
                "Fitting the weighted ridge models for stoichiometry trends, "
                "populations, lifetimes, and size descriptors, then ranking "
                "the predicted structure candidates.",
            ),
        )
        predictions = self._predict_candidates(
            training_observations,
            self._resolve_target_node_counts(observed_node_counts),
            geometry_statistics=geometry_statistics,
        )
        self._update_prediction_population_shares(
            training_observations, predictions
        )
        self._prune_prediction_population_tail(predictions)
        self._emit(
            progress_callback,
            self._status_message(
                6,
                total_steps,
                "Estimating Debye-Waller pair disorder.",
                "Measuring pairwise thermal-displacement statistics from the "
                "observed structure ensembles and predicting values for the "
                "larger structures.",
            ),
        )
        debye_waller_estimates = self._build_debye_waller_estimates(
            training_observations,
            predictions,
        )
        self._emit(
            progress_callback,
            self._status_message(
                7,
                total_steps,
                "Building predicted-structure SAXS traces and output tables.",
                "Composing the observed and predicted structure components, "
                "writing the structure/profile files, and fitting the cluster-only "
                "SAXS comparison model when experimental data are available.",
            ),
        )
        saxs_comparison = self._build_saxs_comparison(
            training_observations,
            predictions,
            preview.experimental_data_path,
            debye_waller_estimates=debye_waller_estimates,
        )
        max_predicted_node_count = self._resolve_max_predicted_node_count(
            predictions
        )
        return ClusterDynamicsMLResult(
            dynamics_result=dynamics_result,
            preview=ClusterDynamicsMLPreview(
                dynamics_preview=preview.dynamics_preview,
                clusters_dir=preview.clusters_dir,
                project_dir=preview.project_dir,
                experimental_data_path=preview.experimental_data_path,
                structure_label_count=preview.structure_label_count,
                total_structure_files=preview.total_structure_files,
                observed_node_counts=observed_node_counts,
                target_node_counts=self._resolve_target_node_counts(
                    observed_node_counts
                ),
                warnings=preview.warnings,
            ),
            structure_observations=tuple(
                sorted(
                    structure_observations,
                    key=lambda item: (item.node_count, item.label),
                )
            ),
            training_observations=tuple(
                sorted(
                    training_observations,
                    key=lambda item: (item.node_count, item.label),
                )
            ),
            predictions=tuple(predictions),
            debye_waller_estimates=tuple(debye_waller_estimates),
            saxs_comparison=saxs_comparison,
            max_observed_node_count=max(observed_node_counts),
            max_predicted_node_count=max_predicted_node_count,
            prediction_population_share_threshold=(
                self.prediction_population_share_threshold
            ),
        )

    def _build_cluster_dynamics_workflow(self) -> ClusterDynamicsWorkflow:
        return ClusterDynamicsWorkflow(
            self.frames_dir,
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
            folder_start_time_fs=self.folder_start_time_fs,
            first_frame_time_fs=self.first_frame_time_fs,
            frame_timestep_fs=self.frame_timestep_fs,
            frames_per_colormap_timestep=self.frames_per_colormap_timestep,
            analysis_start_fs=self.analysis_start_fs,
            analysis_stop_fs=self.analysis_stop_fs,
            energy_file=self.energy_file,
        )

    def _resolve_project_inputs(
        self,
    ) -> tuple[Path | None, Path | None, tuple[str, ...]]:
        warnings: list[str] = []
        resolved_clusters_dir = self.clusters_dir
        resolved_experimental = self.experimental_data_file
        if self.project_dir is None:
            return (
                resolved_clusters_dir,
                resolved_experimental,
                tuple(warnings),
            )
        project_file = build_project_paths(self.project_dir).project_file
        if not project_file.is_file():
            warnings.append(
                "The selected project directory does not contain a "
                "saxs_project.json file."
            )
            return (
                resolved_clusters_dir,
                resolved_experimental,
                tuple(warnings),
            )
        try:
            settings = self._project_manager.load_project(self.project_dir)
        except Exception as exc:
            warnings.append(f"Could not load SAXS project settings: {exc}")
            return (
                resolved_clusters_dir,
                resolved_experimental,
                tuple(warnings),
            )
        if (
            resolved_clusters_dir is None
            and settings.resolved_clusters_dir is not None
        ):
            resolved_clusters_dir = settings.resolved_clusters_dir
        if resolved_experimental is None:
            try:
                experimental = self._project_manager.load_experimental_data(
                    settings
                )
            except Exception:
                experimental = None
            if experimental is not None:
                resolved_experimental = experimental.path
        return resolved_clusters_dir, resolved_experimental, tuple(warnings)

    def _list_structure_labels(
        self,
        clusters_dir: Path,
    ) -> dict[str, list[Path]]:
        label_map: dict[str, list[Path]] = defaultdict(list)
        for structure_dir in sorted(
            path for path in clusters_dir.iterdir() if path.is_dir()
        ):
            if structure_dir.name.startswith("representative_"):
                continue
            motif_dirs = sorted(
                path
                for path in structure_dir.iterdir()
                if path.is_dir() and path.name.startswith("motif_")
            )
            if motif_dirs:
                for motif_dir in motif_dirs:
                    for file_path in sorted(motif_dir.iterdir()):
                        if file_path.suffix.lower() in {".xyz", ".pdb"}:
                            label_map[structure_dir.name].append(file_path)
                continue
            for file_path in sorted(structure_dir.iterdir()):
                if file_path.suffix.lower() in {".xyz", ".pdb"}:
                    label_map[structure_dir.name].append(file_path)
        return label_map

    def _build_structure_observations(
        self,
        clusters_dir: Path,
    ) -> list[ClusterStructureObservation]:
        grouped_paths = self._list_structure_labels(clusters_dir)
        observations: list[ClusterStructureObservation] = []
        tracked_elements = self._tracked_structure_elements()
        for label, paths in sorted(grouped_paths.items()):
            label_counts = _parse_stoichiometry_label(label)
            descriptor_rows: list[tuple[float, float, float, float, float]] = (
                []
            )
            count_rows: list[tuple[Path, dict[str, int]]] = []
            motifs = sorted(
                {
                    path.parent.name
                    for path in paths
                    if path.parent.name.startswith("motif_")
                }
            )
            for path in paths:
                loaded = self._load_structure_for_analysis(path)
                if loaded is None:
                    continue
                descriptor_rows.append(
                    _structure_descriptor_row(loaded.coordinates)
                )
                filtered_counts = _filtered_structure_counts(
                    loaded.elements,
                    tracked_elements=tracked_elements,
                )
                if filtered_counts:
                    count_rows.append((path, filtered_counts))
            if not descriptor_rows:
                continue
            representative_path, counts = (
                _select_representative_structure_counts(
                    count_rows,
                    fallback_path=(paths[0] if paths else None),
                    fallback_counts=label_counts,
                )
            )
            node_count = self._node_count_from_counts(counts)
            if node_count <= 0:
                continue
            descriptor_matrix = np.asarray(descriptor_rows, dtype=float)
            observations.append(
                ClusterStructureObservation(
                    label=label,
                    node_count=node_count,
                    element_counts=counts,
                    file_count=len(paths),
                    representative_path=representative_path,
                    structure_dir=clusters_dir / label,
                    motifs=tuple(motifs),
                    mean_atom_count=float(np.mean(descriptor_matrix[:, 0])),
                    mean_radius_of_gyration=float(
                        np.mean(descriptor_matrix[:, 1])
                    ),
                    mean_max_radius=float(np.mean(descriptor_matrix[:, 2])),
                    mean_semiaxis_a=float(np.mean(descriptor_matrix[:, 3])),
                    mean_semiaxis_b=float(np.mean(descriptor_matrix[:, 4])),
                    mean_semiaxis_c=float(np.mean(descriptor_matrix[:, 5])),
                )
            )
        return observations

    def _tracked_structure_elements(self) -> set[str]:
        tracked: set[str] = set()
        for definitions in self.atom_type_definitions.values():
            for element, _residue in definitions:
                normalized = _normalized_element_symbol(element)
                if normalized:
                    tracked.add(normalized)
        return tracked

    def _build_training_observations(
        self,
        dynamics_result: ClusterDynamicsResult,
        structure_observations: list[ClusterStructureObservation],
    ) -> list[ClusterDynamicsMLTrainingObservation]:
        lifetime_map = {
            entry.label: entry for entry in dynamics_result.lifetime_by_label
        }
        training_rows: list[ClusterDynamicsMLTrainingObservation] = []
        for structure in structure_observations:
            lifetime = lifetime_map.get(structure.label)
            if lifetime is None:
                lifetime = _empty_lifetime_summary(
                    structure.label,
                    cluster_size=sum(structure.element_counts.values()),
                )
            training_rows.append(
                ClusterDynamicsMLTrainingObservation(
                    label=structure.label,
                    node_count=structure.node_count,
                    cluster_size=sum(structure.element_counts.values()),
                    element_counts=dict(structure.element_counts),
                    file_count=structure.file_count,
                    representative_path=structure.representative_path,
                    structure_dir=structure.structure_dir,
                    motifs=structure.motifs,
                    mean_atom_count=structure.mean_atom_count,
                    mean_radius_of_gyration=structure.mean_radius_of_gyration,
                    mean_max_radius=structure.mean_max_radius,
                    mean_semiaxis_a=structure.mean_semiaxis_a,
                    mean_semiaxis_b=structure.mean_semiaxis_b,
                    mean_semiaxis_c=structure.mean_semiaxis_c,
                    total_observations=lifetime.total_observations,
                    occupied_frames=lifetime.occupied_frames,
                    mean_count_per_frame=lifetime.mean_count_per_frame,
                    occupancy_fraction=lifetime.occupancy_fraction,
                    association_events=lifetime.association_events,
                    dissociation_events=lifetime.dissociation_events,
                    association_rate_per_ps=lifetime.association_rate_per_ps,
                    dissociation_rate_per_ps=lifetime.dissociation_rate_per_ps,
                    completed_lifetime_count=lifetime.completed_lifetime_count,
                    window_truncated_lifetime_count=(
                        lifetime.window_truncated_lifetime_count
                    ),
                    mean_lifetime_fs=lifetime.mean_lifetime_fs,
                    std_lifetime_fs=lifetime.std_lifetime_fs,
                )
            )
        return training_rows

    def _resolve_target_node_counts(
        self,
        observed_node_counts: tuple[int, ...],
    ) -> tuple[int, ...]:
        if self.target_node_counts is not None:
            max_observed = (
                max(observed_node_counts) if observed_node_counts else 0
            )
            return tuple(
                value
                for value in self.target_node_counts
                if value > max_observed
            )
        if not observed_node_counts:
            return ()
        max_observed = max(observed_node_counts)
        max_target = (
            self.max_target_node_count
            if self.max_target_node_count is not None
            else max_observed + 2
        )
        max_target = max(max_target, max_observed + 1)
        return tuple(range(max_observed + 1, max_target + 1))

    def _predict_candidates(
        self,
        training_observations: list[ClusterDynamicsMLTrainingObservation],
        target_node_counts: tuple[int, ...],
        *,
        geometry_statistics: _TrainingGeometryStatistics | None = None,
    ) -> list[PredictedClusterCandidate]:
        if not target_node_counts:
            return []
        if geometry_statistics is None:
            geometry_statistics = self._collect_training_geometry_statistics(
                training_observations
            )
        atom_type_by_element = dict(geometry_statistics.atom_type_by_element)
        node_elements = tuple(sorted(self._atom_type_elements("node")))
        non_node_elements = tuple(
            sorted(
                {
                    element
                    for row in training_observations
                    for element in row.element_counts
                    if element not in node_elements
                }
            )
        )
        feature_matrix = np.asarray(
            [
                _candidate_feature_vector(
                    row.element_counts,
                    node_elements=node_elements,
                    non_node_elements=non_node_elements,
                )
                for row in training_observations
            ],
            dtype=float,
        )
        weights = np.asarray(
            [row.stability_weight for row in training_observations],
            dtype=float,
        )
        node_element_fractions = _weighted_node_element_fractions(
            training_observations,
            node_elements=node_elements,
        )
        element_count_models = {
            element: _fit_property_model(
                feature_matrix,
                np.asarray(
                    [
                        row.element_counts.get(element, 0)
                        for row in training_observations
                    ],
                    dtype=float,
                ),
                weights=weights,
                default_value=float(
                    np.average(
                        [
                            row.element_counts.get(element, 0)
                            for row in training_observations
                        ],
                        weights=weights,
                    )
                ),
                lower_bound=0.0,
            )
            for element in non_node_elements
        }
        property_models = self._fit_candidate_property_models(
            training_observations,
            feature_matrix,
            weights,
            non_node_elements=non_node_elements,
            node_elements=node_elements,
        )
        predictions: list[PredictedClusterCandidate] = []
        for target_node_count in target_node_counts:
            raw_candidates = self._build_raw_candidates(
                training_observations,
                target_node_count=target_node_count,
                node_elements=node_elements,
                non_node_elements=non_node_elements,
                atom_type_by_element=atom_type_by_element,
                node_element_fractions=node_element_fractions,
                element_count_models=element_count_models,
            )
            ranked_candidates: list[PredictedClusterCandidate] = []
            for counts, source_label, notes in raw_candidates:
                feature_vector = _candidate_feature_vector(
                    counts,
                    node_elements=node_elements,
                    non_node_elements=non_node_elements,
                )
                predicted_mean_count = property_models[
                    "mean_count_per_frame"
                ].predict(feature_vector)
                predicted_occupancy = property_models[
                    "occupancy_fraction"
                ].predict(feature_vector)
                predicted_lifetime = property_models[
                    "mean_lifetime_fs"
                ].predict(feature_vector)
                predicted_assoc = property_models[
                    "association_rate_per_ps"
                ].predict(feature_vector)
                predicted_dissoc = property_models[
                    "dissociation_rate_per_ps"
                ].predict(feature_vector)
                predicted_rg = property_models[
                    "mean_radius_of_gyration"
                ].predict(feature_vector)
                predicted_max_radius = property_models[
                    "mean_max_radius"
                ].predict(feature_vector)
                predicted_a = property_models["mean_semiaxis_a"].predict(
                    feature_vector
                )
                predicted_b = property_models["mean_semiaxis_b"].predict(
                    feature_vector
                )
                predicted_c = property_models["mean_semiaxis_c"].predict(
                    feature_vector
                )
                composition_distance = _composition_distance(
                    counts,
                    raw_candidates[0][0],
                    node_count=target_node_count,
                )
                predicted_score = float(
                    max(predicted_mean_count, 0.0)
                    * max(predicted_lifetime, self.frame_timestep_fs)
                    * max(predicted_occupancy, 0.05)
                    / (1.0 + composition_distance)
                )
                source_observation = self._best_source_observation(
                    training_observations,
                    counts=counts,
                    target_node_count=target_node_count,
                    preferred_label=source_label,
                )
                generated_elements, generated_coordinates = (
                    self._generate_predicted_structure(
                        source_observation,
                        target_counts=counts,
                        predicted_max_radius=predicted_max_radius,
                        geometry_statistics=geometry_statistics,
                    )
                )
                ranked_candidates.append(
                    PredictedClusterCandidate(
                        target_node_count=target_node_count,
                        rank=0,
                        label=stoichiometry_label(counts),
                        element_counts=counts,
                        predicted_mean_count_per_frame=float(
                            predicted_mean_count
                        ),
                        predicted_occupancy_fraction=float(
                            predicted_occupancy
                        ),
                        predicted_mean_lifetime_fs=float(predicted_lifetime),
                        predicted_association_rate_per_ps=float(
                            predicted_assoc
                        ),
                        predicted_dissociation_rate_per_ps=float(
                            predicted_dissoc
                        ),
                        predicted_mean_radius_of_gyration=float(predicted_rg),
                        predicted_mean_max_radius=float(predicted_max_radius),
                        predicted_mean_semiaxis_a=float(predicted_a),
                        predicted_mean_semiaxis_b=float(predicted_b),
                        predicted_mean_semiaxis_c=float(predicted_c),
                        predicted_population_share=0.0,
                        predicted_stability_score=float(predicted_score),
                        source_label=(
                            None
                            if source_observation is None
                            else source_observation.label
                        ),
                        notes=notes,
                        generated_elements=tuple(generated_elements),
                        generated_coordinates=np.asarray(
                            generated_coordinates,
                            dtype=float,
                        ),
                    )
                )
            ranked_candidates.sort(
                key=lambda item: (
                    -item.predicted_stability_score,
                    item.label,
                )
            )
            for rank, candidate in enumerate(
                ranked_candidates[: self.candidates_per_size],
                start=1,
            ):
                candidate.rank = rank
                predictions.append(candidate)
        return predictions

    def _fit_candidate_property_models(
        self,
        training_observations: list[ClusterDynamicsMLTrainingObservation],
        feature_matrix: np.ndarray,
        weights: np.ndarray,
        *,
        non_node_elements: tuple[str, ...],
        node_elements: tuple[str, ...],
    ) -> dict[str, _PropertyModel]:
        del non_node_elements
        del node_elements
        finite_lifetimes = [
            row.mean_lifetime_fs
            for row in training_observations
            if row.mean_lifetime_fs is not None
        ]
        default_lifetime = (
            float(np.mean(finite_lifetimes))
            if finite_lifetimes
            else self.frame_timestep_fs
        )
        return {
            "mean_count_per_frame": _fit_property_model(
                feature_matrix,
                np.asarray(
                    [
                        row.mean_count_per_frame
                        for row in training_observations
                    ],
                    dtype=float,
                ),
                weights=weights,
                default_value=max(
                    float(
                        np.average(
                            [
                                row.mean_count_per_frame
                                for row in training_observations
                            ],
                            weights=weights,
                        )
                    ),
                    0.0,
                ),
                transform="log1p",
                lower_bound=0.0,
            ),
            "occupancy_fraction": _fit_property_model(
                feature_matrix,
                np.asarray(
                    [row.occupancy_fraction for row in training_observations],
                    dtype=float,
                ),
                weights=weights,
                default_value=float(
                    np.average(
                        [
                            row.occupancy_fraction
                            for row in training_observations
                        ],
                        weights=weights,
                    )
                ),
                lower_bound=0.0,
                upper_bound=1.0,
            ),
            "mean_lifetime_fs": _fit_property_model(
                feature_matrix,
                np.asarray(
                    [
                        (
                            default_lifetime
                            if row.mean_lifetime_fs is None
                            else row.mean_lifetime_fs
                        )
                        for row in training_observations
                    ],
                    dtype=float,
                ),
                weights=weights,
                default_value=max(default_lifetime, self.frame_timestep_fs),
                transform="log1p",
                lower_bound=self.frame_timestep_fs,
            ),
            "association_rate_per_ps": _fit_property_model(
                feature_matrix,
                np.asarray(
                    [
                        row.association_rate_per_ps
                        for row in training_observations
                    ],
                    dtype=float,
                ),
                weights=weights,
                default_value=float(
                    np.average(
                        [
                            row.association_rate_per_ps
                            for row in training_observations
                        ],
                        weights=weights,
                    )
                ),
                transform="log1p",
                lower_bound=0.0,
            ),
            "dissociation_rate_per_ps": _fit_property_model(
                feature_matrix,
                np.asarray(
                    [
                        row.dissociation_rate_per_ps
                        for row in training_observations
                    ],
                    dtype=float,
                ),
                weights=weights,
                default_value=float(
                    np.average(
                        [
                            row.dissociation_rate_per_ps
                            for row in training_observations
                        ],
                        weights=weights,
                    )
                ),
                transform="log1p",
                lower_bound=0.0,
            ),
            "mean_radius_of_gyration": _fit_property_model(
                feature_matrix,
                np.asarray(
                    [
                        row.mean_radius_of_gyration
                        for row in training_observations
                    ],
                    dtype=float,
                ),
                weights=weights,
                default_value=float(
                    np.average(
                        [
                            row.mean_radius_of_gyration
                            for row in training_observations
                        ],
                        weights=weights,
                    )
                ),
                transform="log1p",
                lower_bound=0.1,
            ),
            "mean_max_radius": _fit_property_model(
                feature_matrix,
                np.asarray(
                    [row.mean_max_radius for row in training_observations],
                    dtype=float,
                ),
                weights=weights,
                default_value=float(
                    np.average(
                        [row.mean_max_radius for row in training_observations],
                        weights=weights,
                    )
                ),
                transform="log1p",
                lower_bound=0.1,
            ),
            "mean_semiaxis_a": _fit_property_model(
                feature_matrix,
                np.asarray(
                    [row.mean_semiaxis_a for row in training_observations],
                    dtype=float,
                ),
                weights=weights,
                default_value=float(
                    np.average(
                        [row.mean_semiaxis_a for row in training_observations],
                        weights=weights,
                    )
                ),
                transform="log1p",
                lower_bound=0.05,
            ),
            "mean_semiaxis_b": _fit_property_model(
                feature_matrix,
                np.asarray(
                    [row.mean_semiaxis_b for row in training_observations],
                    dtype=float,
                ),
                weights=weights,
                default_value=float(
                    np.average(
                        [row.mean_semiaxis_b for row in training_observations],
                        weights=weights,
                    )
                ),
                transform="log1p",
                lower_bound=0.05,
            ),
            "mean_semiaxis_c": _fit_property_model(
                feature_matrix,
                np.asarray(
                    [row.mean_semiaxis_c for row in training_observations],
                    dtype=float,
                ),
                weights=weights,
                default_value=float(
                    np.average(
                        [row.mean_semiaxis_c for row in training_observations],
                        weights=weights,
                    )
                ),
                transform="log1p",
                lower_bound=0.05,
            ),
        }

    def _build_raw_candidates(
        self,
        training_observations: list[ClusterDynamicsMLTrainingObservation],
        *,
        target_node_count: int,
        node_elements: tuple[str, ...],
        non_node_elements: tuple[str, ...],
        atom_type_by_element: dict[str, str],
        node_element_fractions: dict[str, float],
        element_count_models: dict[str, _PropertyModel],
    ) -> list[tuple[dict[str, int], str | None, str]]:
        candidates: list[tuple[dict[str, int], str | None, str]] = []
        required_non_node_counts = _required_non_node_count_floors(
            training_observations,
            target_node_count=target_node_count,
            non_node_elements=non_node_elements,
            atom_type_by_element=atom_type_by_element,
        )
        base_counts = _allocate_node_counts(
            target_node_count,
            node_element_fractions,
        )
        for element in non_node_elements:
            predicted = int(
                round(
                    element_count_models[element].predict(
                        _candidate_feature_vector(
                            base_counts,
                            node_elements=node_elements,
                            non_node_elements=non_node_elements,
                        )
                    )
                )
            )
            if predicted > 0:
                base_counts[element] = predicted
        candidates.append((base_counts, None, "Trend extrapolation"))
        ranked_observations = sorted(
            training_observations,
            key=lambda item: (
                -item.stability_weight,
                -item.node_count,
                item.label,
            ),
        )
        for row in ranked_observations:
            scaled = _allocate_node_counts(
                target_node_count, node_element_fractions
            )
            if row.node_count <= 0:
                continue
            scale_factor = float(target_node_count) / float(row.node_count)
            for element in non_node_elements:
                count = int(
                    round(row.element_counts.get(element, 0) * scale_factor)
                )
                if count > 0:
                    scaled[element] = count
            candidates.append(
                (
                    scaled,
                    row.label,
                    f"Composition scaled from observed {row.label}",
                )
            )
        deduplicated: list[tuple[dict[str, int], str | None, str]] = []
        seen_labels: set[str] = set()
        for counts, source_label, notes in candidates:
            normalized = self._apply_pdb_shell_backbone_count_constraints(
                counts
            )
            if any(
                normalized.get(element, 0) < minimum_count
                for element, minimum_count in required_non_node_counts.items()
            ):
                continue
            if not _candidate_has_support(
                training_observations,
                counts=normalized,
                target_node_count=target_node_count,
                node_elements=node_elements,
                non_node_elements=non_node_elements,
            ):
                continue
            label = stoichiometry_label(normalized)
            if label in seen_labels:
                continue
            seen_labels.add(label)
            deduplicated.append((normalized, source_label, notes))
        return deduplicated

    def _best_source_observation(
        self,
        training_observations: list[ClusterDynamicsMLTrainingObservation],
        *,
        counts: dict[str, int],
        target_node_count: int,
        preferred_label: str | None,
    ) -> ClusterDynamicsMLTrainingObservation | None:
        if preferred_label is not None:
            for row in training_observations:
                if row.label == preferred_label:
                    return row
        best_row: ClusterDynamicsMLTrainingObservation | None = None
        best_score: tuple[float, float, str] | None = None
        for row in training_observations:
            if row.representative_path is None:
                continue
            node_gap = abs(target_node_count - row.node_count)
            composition_gap = _composition_distance(
                counts,
                row.element_counts,
                node_count=max(target_node_count, 1),
            )
            score = (float(node_gap), float(composition_gap), row.label)
            if best_score is None or score < best_score:
                best_score = score
                best_row = row
        return best_row

    def _apply_pdb_shell_backbone_count_constraints(
        self,
        counts: dict[str, int],
    ) -> dict[str, int]:
        # Deprecated backbone-pair count coupling is intentionally disabled.
        # PDB solvent injection now uses only the predicted shell-anchor atom.
        return _normalized_counts(counts)

    def _generate_predicted_structure(
        self,
        source_observation: ClusterDynamicsMLTrainingObservation | None,
        *,
        target_counts: dict[str, int],
        predicted_max_radius: float,
        geometry_statistics: _TrainingGeometryStatistics,
    ) -> tuple[list[str], np.ndarray]:
        node_elements = self._atom_type_elements("node")
        seed_node_elements: list[str] = []
        seed_node_coordinates = np.zeros((0, 3), dtype=float)
        if (
            source_observation is not None
            and source_observation.representative_path is not None
        ):
            loaded = self._load_structure_for_analysis(
                source_observation.representative_path
            )
            if loaded is not None:
                source_coords_array = np.asarray(
                    loaded.coordinates,
                    dtype=float,
                )
                normalized_elements = [
                    _normalized_element_symbol(element)
                    for element in loaded.elements
                ]
                node_indices = [
                    index
                    for index, element in enumerate(normalized_elements)
                    if element in node_elements
                ]
                if node_indices:
                    seed_node_coordinates = np.asarray(
                        source_coords_array[node_indices],
                        dtype=float,
                    )
                    seed_node_elements = [
                        normalized_elements[index] for index in node_indices
                    ]
        generated_elements, generated = _build_geometry_guided_structure(
            target_counts,
            node_elements=tuple(sorted(node_elements)),
            pair_cutoff_definitions=self.pair_cutoff_definitions,
            geometry_statistics=geometry_statistics,
            predicted_max_radius=predicted_max_radius,
            seed_node_elements=seed_node_elements,
            seed_node_coordinates=seed_node_coordinates,
        )
        if generated.size == 0:
            return _build_fallback_structure(target_counts)
        return generated_elements, generated

    def _build_saxs_comparison(
        self,
        training_observations: list[ClusterDynamicsMLTrainingObservation],
        predictions: list[PredictedClusterCandidate],
        experimental_data_path: Path | None,
        *,
        debye_waller_estimates: list[DebyeWallerPairEstimate],
    ) -> ClusterDynamicsMLSAXSComparison | None:
        experimental_data = self._load_experimental_data(
            experimental_data_path
        )
        q_values = self._resolve_q_values(experimental_data)
        debye_waller_lookup = _debye_waller_sigma_lookup_by_label(
            debye_waller_estimates
        )
        component_dir, predicted_structure_dir = (
            self._resolve_saxs_artifact_dirs()
        )
        observed_weights, predicted_weights = _resolved_population_weights(
            training_observations,
            predictions,
            frame_timestep_fs=self.frame_timestep_fs,
        )
        source_observation_by_label = {
            row.label: row for row in training_observations
        }
        predicted_structure_paths = [
            self._write_predicted_structure_file(
                prediction,
                predicted_structure_dir=predicted_structure_dir,
                source_observation=(
                    None
                    if prediction.source_label is None
                    else source_observation_by_label.get(
                        prediction.source_label
                    )
                ),
            )
            for prediction in predictions
        ]
        observed_components = [
            component
            for row, weight in zip(
                training_observations,
                observed_weights,
                strict=False,
            )
            for component in [
                self._build_observed_saxs_component(
                    row,
                    weight=float(weight),
                    q_values=q_values,
                    component_dir=component_dir,
                    pair_sigma_by_element=debye_waller_lookup.get(row.label),
                )
            ]
            if component is not None
        ]
        predicted_components = [
            component
            for prediction, weight, structure_path in zip(
                predictions,
                predicted_weights,
                predicted_structure_paths,
                strict=False,
            )
            for component in [
                self._build_predicted_saxs_component(
                    prediction,
                    weight=float(weight),
                    structure_path=structure_path,
                    q_values=q_values,
                    component_dir=component_dir,
                    pair_sigma_by_element=debye_waller_lookup.get(
                        prediction.label
                    ),
                )
            ]
            if component is not None
        ]
        all_components = [*observed_components, *predicted_components]
        if not all_components:
            return None

        observed_model = self._compose_weighted_model(observed_components)
        combined_model = self._compose_weighted_model(all_components)
        if combined_model is None:
            return None

        observed_raw_model = (
            None if observed_model is None else observed_model[0]
        )
        raw_model, normalized_weights = combined_model
        experimental_intensity: np.ndarray | None = None
        observed_fitted_model = (
            None if observed_raw_model is None else observed_raw_model.copy()
        )
        observed_rmse: float | None = None
        scale_factor = 1.0
        offset = 0.0
        fitted_model = raw_model.copy()
        residuals: np.ndarray | None = None
        rmse: float | None = None
        if experimental_data is not None:
            experimental_intensity = np.asarray(
                experimental_data.intensities,
                dtype=float,
            )
            if observed_raw_model is not None:
                (
                    _observed_scale_factor,
                    _observed_offset,
                    observed_fitted_model,
                    _observed_residuals,
                    observed_rmse,
                ) = _fit_model_to_experimental(
                    observed_raw_model,
                    experimental_intensity,
                )
            (
                scale_factor,
                offset,
                fitted_model,
                residuals,
                rmse,
            ) = _fit_model_to_experimental(
                raw_model,
                experimental_intensity,
            )
        return ClusterDynamicsMLSAXSComparison(
            q_values=q_values,
            observed_raw_model_intensity=observed_raw_model,
            observed_fitted_model_intensity=observed_fitted_model,
            observed_rmse=observed_rmse,
            raw_model_intensity=raw_model,
            fitted_model_intensity=fitted_model,
            experimental_intensity=experimental_intensity,
            residuals=residuals,
            scale_factor=float(scale_factor),
            offset=float(offset),
            rmse=rmse,
            component_weights=tuple(
                SAXSComponentWeight(
                    label=component.label,
                    weight=float(weight),
                    source=component.source,
                    profile_path=component.profile_path,
                    structure_path=component.structure_path,
                )
                for component, weight in zip(
                    all_components,
                    normalized_weights,
                    strict=False,
                )
            ),
            experimental_data_path=(
                None if experimental_data is None else experimental_data.path
            ),
            component_output_dir=component_dir,
            predicted_structure_dir=predicted_structure_dir,
        )

    def _build_debye_waller_estimates(
        self,
        training_observations: list[ClusterDynamicsMLTrainingObservation],
        predictions: list[PredictedClusterCandidate],
    ) -> list[DebyeWallerPairEstimate]:
        node_elements = tuple(sorted(self._atom_type_elements("node")))
        non_node_elements = tuple(
            sorted(
                {
                    element
                    for row in training_observations
                    for element in row.element_counts
                    if element not in node_elements
                }
            )
        )
        observed_ensemble_rows = self._estimate_observed_debye_waller_pairs(
            training_observations
        )
        pair_models = self._fit_debye_waller_pair_models(
            training_observations,
            observed_ensemble_rows,
            node_elements=node_elements,
            non_node_elements=non_node_elements,
        )
        resolved_observed_rows = self._resolve_observed_debye_waller_pairs(
            training_observations,
            observed_ensemble_rows=observed_ensemble_rows,
            pair_models=pair_models,
            node_elements=node_elements,
            non_node_elements=non_node_elements,
        )
        predicted_rows = self._predict_debye_waller_pairs(
            predictions,
            pair_models=pair_models,
            node_elements=node_elements,
            non_node_elements=non_node_elements,
        )
        return sorted(
            [*resolved_observed_rows, *predicted_rows],
            key=lambda row: (
                row.source != "observed",
                row.node_count,
                row.candidate_rank is not None,
                0 if row.candidate_rank is None else row.candidate_rank,
                row.label,
                row.element_a,
                row.element_b,
            ),
        )

    def _estimate_observed_debye_waller_pairs(
        self,
        training_observations: list[ClusterDynamicsMLTrainingObservation],
    ) -> list[DebyeWallerPairEstimate]:
        tracked_elements = self._tracked_structure_elements()
        node_elements = set(self._atom_type_elements("node"))
        estimates: list[DebyeWallerPairEstimate] = []
        for observation in training_observations:
            pair_samples: dict[tuple[str, str], list[np.ndarray]] = (
                defaultdict(list)
            )
            for structure_path in _structure_files_for_cluster_dir(
                observation.structure_dir
            ):
                loaded = self._load_structure_for_analysis(structure_path)
                if loaded is None:
                    continue
                filtered = _filtered_coordinates_and_elements(
                    loaded.coordinates,
                    loaded.elements,
                    tracked_elements=tracked_elements,
                )
                if filtered is None:
                    continue
                filtered_coords, filtered_elements = filtered
                distance_map = _first_shell_pair_distances_by_element(
                    filtered_coords,
                    filtered_elements,
                    node_elements=node_elements,
                    pair_cutoff_definitions=self.pair_cutoff_definitions,
                )
                for pair_key, distances in distance_map.items():
                    if distances.size > 0:
                        pair_samples[pair_key].append(distances)
            for pair_key, samples in sorted(pair_samples.items()):
                if len(samples) < 2:
                    continue
                aligned_pair_count = min(sample.size for sample in samples)
                if aligned_pair_count <= 0:
                    continue
                aligned_matrix = np.asarray(
                    [sample[:aligned_pair_count] for sample in samples],
                    dtype=float,
                )
                per_rank_sigma = np.std(aligned_matrix, axis=0, ddof=0)
                sigma = float(
                    np.sqrt(np.mean(np.square(per_rank_sigma), dtype=float))
                )
                estimates.append(
                    DebyeWallerPairEstimate(
                        source="observed",
                        method="ensemble",
                        label=observation.label,
                        node_count=observation.node_count,
                        candidate_rank=None,
                        element_a=pair_key[0],
                        element_b=pair_key[1],
                        sigma=sigma,
                        b_factor=b_factor_from_sigma(sigma),
                        support_count=len(samples),
                        aligned_pair_count=int(aligned_pair_count),
                    )
                )
        return estimates

    def _fit_debye_waller_pair_models(
        self,
        training_observations: list[ClusterDynamicsMLTrainingObservation],
        observed_ensemble_rows: list[DebyeWallerPairEstimate],
        *,
        node_elements: tuple[str, ...],
        non_node_elements: tuple[str, ...],
    ) -> dict[tuple[str, str], _DebyeWallerPairModel]:
        training_by_label = {row.label: row for row in training_observations}
        grouped_rows: dict[tuple[str, str], list[DebyeWallerPairEstimate]] = (
            defaultdict(list)
        )
        for row in observed_ensemble_rows:
            grouped_rows[(row.element_a, row.element_b)].append(row)
        pair_models: dict[tuple[str, str], _DebyeWallerPairModel] = {}
        for pair_key, rows in grouped_rows.items():
            feature_rows: list[np.ndarray] = []
            target_rows: list[float] = []
            weight_rows: list[float] = []
            for row in rows:
                observation = training_by_label.get(row.label)
                if observation is None:
                    continue
                feature_rows.append(
                    _candidate_feature_vector(
                        observation.element_counts,
                        node_elements=node_elements,
                        non_node_elements=non_node_elements,
                    )
                )
                target_rows.append(float(row.sigma))
                weight_rows.append(
                    float(observation.stability_weight)
                    * max(float(row.support_count), 1.0)
                )
            if not feature_rows:
                continue
            weight_array = np.asarray(weight_rows, dtype=float)
            target_array = np.asarray(target_rows, dtype=float)
            pair_models[pair_key] = _DebyeWallerPairModel(
                sigma_model=_fit_property_model(
                    np.asarray(feature_rows, dtype=float),
                    target_array,
                    weights=weight_array,
                    default_value=float(
                        np.average(target_array, weights=weight_array)
                    ),
                    transform="log1p",
                    lower_bound=0.0,
                ),
                support_count=len(feature_rows),
            )
        return pair_models

    def _resolve_observed_debye_waller_pairs(
        self,
        training_observations: list[ClusterDynamicsMLTrainingObservation],
        *,
        observed_ensemble_rows: list[DebyeWallerPairEstimate],
        pair_models: dict[tuple[str, str], _DebyeWallerPairModel],
        node_elements: tuple[str, ...],
        non_node_elements: tuple[str, ...],
    ) -> list[DebyeWallerPairEstimate]:
        resolved_rows: list[DebyeWallerPairEstimate] = []
        ensemble_lookup = {
            (row.label, row.element_a, row.element_b): row
            for row in observed_ensemble_rows
        }
        for observation in training_observations:
            feature_vector = _candidate_feature_vector(
                observation.element_counts,
                node_elements=node_elements,
                non_node_elements=non_node_elements,
            )
            for pair_key in _element_pair_keys_from_counts(
                observation.element_counts
            ):
                ensemble_row = ensemble_lookup.get(
                    (observation.label, pair_key[0], pair_key[1])
                )
                if ensemble_row is not None:
                    resolved_rows.append(ensemble_row)
                    continue
                model = pair_models.get(pair_key)
                if model is None:
                    continue
                sigma = float(model.sigma_model.predict(feature_vector))
                resolved_rows.append(
                    DebyeWallerPairEstimate(
                        source="observed",
                        method="ridge",
                        label=observation.label,
                        node_count=observation.node_count,
                        candidate_rank=None,
                        element_a=pair_key[0],
                        element_b=pair_key[1],
                        sigma=sigma,
                        b_factor=b_factor_from_sigma(sigma),
                        support_count=model.support_count,
                        aligned_pair_count=0,
                    )
                )
        return resolved_rows

    def _predict_debye_waller_pairs(
        self,
        predictions: list[PredictedClusterCandidate],
        *,
        pair_models: dict[tuple[str, str], _DebyeWallerPairModel],
        node_elements: tuple[str, ...],
        non_node_elements: tuple[str, ...],
    ) -> list[DebyeWallerPairEstimate]:
        predicted_rows: list[DebyeWallerPairEstimate] = []
        for prediction in predictions:
            feature_vector = _candidate_feature_vector(
                prediction.element_counts,
                node_elements=node_elements,
                non_node_elements=non_node_elements,
            )
            for pair_key in _element_pair_keys_from_counts(
                prediction.element_counts
            ):
                model = pair_models.get(pair_key)
                if model is None:
                    continue
                sigma = float(model.sigma_model.predict(feature_vector))
                predicted_rows.append(
                    DebyeWallerPairEstimate(
                        source="predicted",
                        method="ridge",
                        label=prediction.label,
                        node_count=prediction.target_node_count,
                        candidate_rank=prediction.rank,
                        element_a=pair_key[0],
                        element_b=pair_key[1],
                        sigma=sigma,
                        b_factor=b_factor_from_sigma(sigma),
                        support_count=model.support_count,
                        aligned_pair_count=0,
                        source_label=prediction.source_label,
                    )
                )
        return predicted_rows

    def _resolve_q_values(
        self,
        experimental_data: ExperimentalDataSummary | None,
    ) -> np.ndarray:
        if experimental_data is not None:
            return np.asarray(experimental_data.q_values, dtype=float)
        supported_range = None
        if self.project_dir is not None:
            try:
                supported_range = load_built_component_q_range(
                    self.project_dir
                )
            except Exception:
                supported_range = None
        q_min = (
            self.q_min
            if self.q_min is not None
            else (
                float(supported_range[0])
                if supported_range is not None
                else _DEFAULT_Q_MIN
            )
        )
        q_max = (
            self.q_max
            if self.q_max is not None
            else (
                float(supported_range[1])
                if supported_range is not None
                else _DEFAULT_Q_MAX
            )
        )
        if q_min > q_max:
            raise ValueError("q min must be less than or equal to q max.")
        return np.linspace(float(q_min), float(q_max), int(self.q_points))

    def _resolve_saxs_artifact_dirs(self) -> tuple[Path, Path]:
        if self.project_dir is not None:
            base_dir = (
                build_project_paths(self.project_dir).exported_data_dir
                / "clusterdynamicsml"
            )
        else:
            base_dir = (
                self.frames_dir.parent
                / f"{self.frames_dir.name}_clusterdynamicsml"
            )
        component_dir = base_dir / "saxs_components"
        predicted_structure_dir = base_dir / "predicted_structures"
        component_dir.mkdir(parents=True, exist_ok=True)
        predicted_structure_dir.mkdir(parents=True, exist_ok=True)
        return component_dir, predicted_structure_dir

    def _write_predicted_structure_file(
        self,
        prediction: PredictedClusterCandidate,
        *,
        predicted_structure_dir: Path,
        source_observation: ClusterDynamicsMLTrainingObservation | None,
    ) -> Path:
        if self._frames_input_format() != "pdb":
            return _write_predicted_structure_file(
                prediction,
                predicted_structure_dir=predicted_structure_dir,
            )

        structure_path = (
            predicted_structure_dir
            / f"{_predicted_structure_file_stem(prediction)}.pdb"
        )
        atoms = self._build_predicted_pdb_atoms(
            prediction,
            source_observation=source_observation,
        )
        structure = PDBStructure(
            atoms=list(atoms),
            source_name=_predicted_structure_file_stem(prediction),
        )
        structure.write_pdb_file(structure_path, list(atoms))
        return structure_path

    def _build_predicted_pdb_atoms(
        self,
        prediction: PredictedClusterCandidate,
        *,
        source_observation: ClusterDynamicsMLTrainingObservation | None,
    ) -> list[PDBAtom]:
        predicted_coordinates = np.asarray(
            prediction.generated_coordinates,
            dtype=float,
        )
        predicted_elements = [
            _normalized_element_symbol(element)
            for element in prediction.generated_elements
        ]
        atoms: list[PDBAtom] = []
        consumed_indices: set[int] = set()
        next_atom_id = 1
        next_residue_number = 1

        atom_type_by_element = self._atom_type_by_element()
        solute_coordinates = np.asarray(
            [
                coordinate
                for element, coordinate in zip(
                    predicted_elements,
                    predicted_coordinates,
                    strict=False,
                )
                if atom_type_by_element.get(element) in {"node", "linker"}
            ],
            dtype=float,
        )
        if solute_coordinates.size == 0:
            solute_coordinates = np.asarray(predicted_coordinates, dtype=float)
        placed_solvent_coordinates = np.zeros((0, 3), dtype=float)

        for reference in self._active_pdb_shell_references():
            for anchor_index in self._shell_anchor_indices(
                predicted_coordinates,
                predicted_elements,
                reference=reference,
                consumed_indices=consumed_indices,
            ):
                reserved_anchor_coordinates = np.asarray(
                    [
                        predicted_coordinates[index]
                        for index, element in enumerate(predicted_elements)
                        if index not in consumed_indices
                        and index != anchor_index
                        and element == reference.shell_element
                    ],
                    dtype=float,
                )
                aligned_atoms = self._anchored_reference_atoms_to_prediction(
                    reference,
                    anchor_coordinate=predicted_coordinates[anchor_index],
                    solute_coordinates=solute_coordinates,
                    placed_solvent_coordinates=placed_solvent_coordinates,
                    reserved_anchor_coordinates=reserved_anchor_coordinates,
                    residue_number=next_residue_number,
                    starting_atom_id=next_atom_id,
                )
                atoms.extend(aligned_atoms)
                next_atom_id += len(aligned_atoms)
                next_residue_number += 1
                consumed_indices.add(anchor_index)
                new_solvent_coordinates = np.asarray(
                    [atom.coordinates for atom in aligned_atoms],
                    dtype=float,
                )
                if placed_solvent_coordinates.size == 0:
                    placed_solvent_coordinates = new_solvent_coordinates
                else:
                    placed_solvent_coordinates = np.vstack(
                        (
                            placed_solvent_coordinates,
                            new_solvent_coordinates,
                        )
                    )

        source_templates_by_element = self._source_pdb_atom_templates(
            source_observation
        )
        used_template_counts: defaultdict[str, int] = defaultdict(int)

        for atom_index, (element, coordinate) in enumerate(
            zip(predicted_elements, predicted_coordinates, strict=False)
        ):
            if atom_index in consumed_indices:
                continue
            template_list = source_templates_by_element.get(element, [])
            template_index = used_template_counts[element]
            template_atom = (
                None
                if template_index >= len(template_list)
                else template_list[template_index]
            )
            used_template_counts[element] += 1
            atom_type = atom_type_by_element.get(element, "shell")
            residue_name, atom_name = self._predicted_free_atom_identity(
                element,
                atom_type=atom_type,
                template_atom=template_atom,
                occurrence_index=used_template_counts[element],
            )
            atoms.append(
                PDBAtom(
                    atom_id=next_atom_id,
                    atom_name=atom_name,
                    residue_name=residue_name,
                    residue_number=next_residue_number,
                    coordinates=np.asarray(coordinate, dtype=float).copy(),
                    element=element,
                    atom_type=atom_type,
                )
            )
            next_atom_id += 1
            next_residue_number += 1

        return atoms

    def _source_pdb_atom_templates(
        self,
        source_observation: ClusterDynamicsMLTrainingObservation | None,
    ) -> dict[str, list[PDBAtom]]:
        if (
            source_observation is None
            or source_observation.representative_path is None
            or source_observation.representative_path.suffix.lower() != ".pdb"
        ):
            return {}
        loaded = self._load_filtered_pdb_structure(
            source_observation.representative_path
        )
        if loaded is None or not loaded.pdb_atoms:
            return {}
        grouped: dict[str, list[PDBAtom]] = defaultdict(list)
        for atom in loaded.pdb_atoms:
            if atom.atom_type not in {"node", "linker"}:
                continue
            grouped[_normalized_element_symbol(atom.element)].append(
                atom.copy()
            )
        return dict(grouped)

    def _predicted_free_atom_identity(
        self,
        element: str,
        *,
        atom_type: str,
        template_atom: PDBAtom | None,
        occurrence_index: int,
    ) -> tuple[str, str]:
        if template_atom is not None:
            return template_atom.residue_name, template_atom.atom_name
        for reference in self._active_pdb_shell_references():
            if element == reference.shell_element:
                return (
                    reference.reference_residue_name,
                    reference.anchor_atom_name,
                )
        residue_name = {
            "node": "NOD",
            "linker": "LNK",
            "shell": "SOL",
        }.get(atom_type, "UNK")
        atom_name = f"{element.upper()}{int(max(occurrence_index, 1))}"
        return residue_name, atom_name

    def _shell_anchor_indices(
        self,
        coordinates: np.ndarray,
        elements: list[str],
        *,
        reference: _ResolvedPDBShellReference,
        consumed_indices: set[int],
    ) -> list[int]:
        del coordinates
        return [
            index
            for index, element in enumerate(elements)
            if index not in consumed_indices
            and element == reference.shell_element
        ]

    def _anchored_reference_atoms_to_prediction(
        self,
        reference: _ResolvedPDBShellReference,
        *,
        anchor_coordinate: np.ndarray,
        solute_coordinates: np.ndarray,
        placed_solvent_coordinates: np.ndarray,
        reserved_anchor_coordinates: np.ndarray,
        residue_number: int,
        starting_atom_id: int,
    ) -> list[PDBAtom]:
        transformed_coordinates = self._anchored_reference_coordinates(
            reference,
            anchor_coordinate=anchor_coordinate,
            solute_coordinates=solute_coordinates,
            placed_solvent_coordinates=placed_solvent_coordinates,
            reserved_anchor_coordinates=reserved_anchor_coordinates,
        )
        transformed_atoms: list[PDBAtom] = []
        for offset, template_atom in enumerate(reference.reference_atoms):
            copied = template_atom.copy()
            copied.atom_id = starting_atom_id + offset
            copied.residue_number = residue_number
            copied.residue_name = reference.reference_residue_name
            copied.coordinates = transformed_coordinates[offset].copy()
            transformed_atoms.append(copied)
        return transformed_atoms

    def _anchored_reference_coordinates(
        self,
        reference: _ResolvedPDBShellReference,
        *,
        anchor_coordinate: np.ndarray,
        solute_coordinates: np.ndarray,
        placed_solvent_coordinates: np.ndarray,
        reserved_anchor_coordinates: np.ndarray,
    ) -> np.ndarray:
        reference_coordinates = np.asarray(
            [atom.coordinates for atom in reference.reference_atoms],
            dtype=float,
        )
        source_anchor = reference_coordinates[reference.anchor_atom_index]
        target_anchor = np.asarray(anchor_coordinate, dtype=float)
        if solute_coordinates.size == 0:
            solute_centroid = (
                target_anchor
                - _safe_unit_vector(reference.reference_outward_vector)
            )
        else:
            solute_centroid = np.mean(
                np.asarray(solute_coordinates, dtype=float),
                axis=0,
            )
        source_outward = _safe_unit_vector(
            reference.reference_outward_vector
        )
        target_outward = _safe_unit_vector(
            target_anchor - solute_centroid,
            fallback=source_outward,
        )
        base_rotation = rotation_matrix_from_to(
            source_outward,
            target_outward,
        )
        base_coordinates = (
            (reference_coordinates - source_anchor) @ base_rotation.T
        ) + target_anchor
        base_relative = base_coordinates - target_anchor
        non_anchor_mask = np.ones(len(reference.reference_atoms), dtype=bool)
        non_anchor_mask[reference.anchor_atom_index] = False
        rotation_axis = _safe_unit_vector(
            target_outward,
            fallback=source_outward,
        )
        best_coordinates = np.asarray(base_coordinates, dtype=float)
        best_score: float | None = None
        for sample_index in range(24):
            angle = (2.0 * math.pi * float(sample_index)) / 24.0
            axis_rotation = rotation_matrix_about_axis(rotation_axis, angle)
            candidate_coordinates = (
                base_relative @ axis_rotation.T
            ) + target_anchor
            candidate_non_anchor = candidate_coordinates[non_anchor_mask]
            overlap_penalty = 0.0
            min_clearance = 8.0
            for obstacle_set in (
                np.asarray(solute_coordinates, dtype=float),
                np.asarray(placed_solvent_coordinates, dtype=float),
                np.asarray(reserved_anchor_coordinates, dtype=float),
            ):
                if obstacle_set.size == 0 or candidate_non_anchor.size == 0:
                    continue
                distances = np.linalg.norm(
                    candidate_non_anchor[:, np.newaxis, :] - obstacle_set[np.newaxis, :, :],
                    axis=2,
                )
                min_clearance = min(
                    min_clearance,
                    float(np.min(distances)),
                )
                overlap_penalty += float(
                    np.sum(
                        np.square(np.clip(1.2 - distances, 0.0, None)),
                        dtype=float,
                    )
                    * 800.0
                )
                overlap_penalty += float(
                    np.sum(
                        np.square(np.clip(1.5 - distances, 0.0, None)),
                        dtype=float,
                    )
                    * 15.0
                )
            centroid_vector = (
                np.mean(candidate_non_anchor, axis=0) - target_anchor
                if candidate_non_anchor.size > 0
                else target_outward
            )
            outward_alignment = float(
                np.dot(
                    _safe_unit_vector(
                        centroid_vector,
                        fallback=target_outward,
                    ),
                    target_outward,
                )
            )
            score = (
                overlap_penalty
                - (min_clearance * 0.5)
                - (outward_alignment * 10.0)
            )
            if best_score is None or score < best_score:
                best_score = float(score)
                best_coordinates = np.asarray(
                    candidate_coordinates,
                    dtype=float,
                )
        return best_coordinates

    # Deprecated backbone-pair helpers are kept below for future reference.

    def _pair_shell_backbone_indices(
        self,
        coordinates: np.ndarray,
        elements: list[str],
        *,
        reference: _ResolvedPDBShellReference,
        consumed_indices: set[int],
    ) -> list[tuple[int, int]]:
        anchor_indices = [
            index
            for index, element in enumerate(elements)
            if index not in consumed_indices
            and element == reference.backbone_atom1_element
        ]
        axis_indices = [
            index
            for index, element in enumerate(elements)
            if index not in consumed_indices
            and element == reference.backbone_atom2_element
        ]
        if not anchor_indices or not axis_indices:
            return []

        cost_matrix = np.zeros(
            (len(anchor_indices), len(axis_indices)),
            dtype=float,
        )
        for row_index, anchor_index in enumerate(anchor_indices):
            for column_index, axis_index in enumerate(axis_indices):
                distance = float(
                    np.linalg.norm(
                        coordinates[anchor_index] - coordinates[axis_index]
                    )
                )
                cost_matrix[row_index, column_index] = abs(
                    distance - reference.backbone_distance
                )
        row_indices, column_indices = linear_sum_assignment(cost_matrix)
        return [
            (anchor_indices[int(row_index)], axis_indices[int(column_index)])
            for row_index, column_index in zip(
                row_indices,
                column_indices,
                strict=False,
            )
        ]

    def _aligned_reference_atoms_to_prediction(
        self,
        reference: _ResolvedPDBShellReference,
        *,
        anchor_coordinate: np.ndarray,
        axis_coordinate: np.ndarray,
        residue_number: int,
        starting_atom_id: int,
    ) -> list[PDBAtom]:
        reference_coordinates = np.asarray(
            [atom.coordinates for atom in reference.reference_atoms],
            dtype=float,
        )
        source_anchor = reference_coordinates[reference.backbone_atom1_index]
        source_axis = reference_coordinates[reference.backbone_atom2_index]
        source_vector = source_axis - source_anchor
        target_vector = np.asarray(axis_coordinate, dtype=float) - np.asarray(
            anchor_coordinate,
            dtype=float,
        )
        source_length = float(np.linalg.norm(source_vector))
        target_length = float(np.linalg.norm(target_vector))
        if source_length <= 1.0e-12 or target_length <= 1.0e-12:
            scale = 1.0
            rotation = np.eye(3, dtype=float)
        else:
            scale = target_length / source_length
            rotation = rotation_matrix_from_to(source_vector, target_vector)
        transformed_coordinates = (
            ((reference_coordinates - source_anchor) * scale) @ rotation.T
        ) + np.asarray(anchor_coordinate, dtype=float)
        transformed_atoms: list[PDBAtom] = []
        for offset, template_atom in enumerate(reference.reference_atoms):
            copied = template_atom.copy()
            copied.atom_id = starting_atom_id + offset
            copied.residue_number = residue_number
            copied.residue_name = reference.reference_residue_name
            copied.coordinates = transformed_coordinates[offset].copy()
            transformed_atoms.append(copied)
        return transformed_atoms

    def _build_observed_saxs_component(
        self,
        observation: ClusterDynamicsMLTrainingObservation,
        *,
        weight: float,
        q_values: np.ndarray,
        component_dir: Path,
        pair_sigma_by_element: dict[tuple[str, str], float] | None = None,
    ) -> _ResolvedSAXSComponent | None:
        if weight <= 0.0:
            return None

        profile_path = (
            component_dir
            / f"observed_{_safe_component_stem(observation.label)}.txt"
        )
        project_trace = self._load_project_component_trace(
            observation.label,
            q_values=q_values,
        )
        if project_trace is not None:
            _write_component_profile(
                profile_path,
                q_values=q_values,
                intensity=project_trace,
                source=f"project:{observation.label}",
            )
            return _ResolvedSAXSComponent(
                label=observation.label,
                weight=weight,
                source="observed_project",
                trace=project_trace,
                profile_path=profile_path,
                structure_path=observation.representative_path,
            )

        if observation.representative_path is None:
            return None
        try:
            coords, elements = load_structure_file(
                observation.representative_path
            )
            included_pair_indices = (
                _first_shell_pair_indices(
                    coords,
                    elements,
                    node_elements=set(self._atom_type_elements("node")),
                    pair_cutoff_definitions=self.pair_cutoff_definitions,
                )
                if pair_sigma_by_element
                else None
            )
            trace = np.asarray(
                (
                    compute_debye_intensity_with_debye_waller(
                        coords,
                        elements,
                        q_values,
                        pair_sigma_by_element=pair_sigma_by_element,
                        included_pair_indices=included_pair_indices,
                    )
                    if pair_sigma_by_element
                    else compute_debye_intensity(coords, elements, q_values)
                ),
                dtype=float,
            )
        except Exception:
            return None
        _write_component_profile(
            profile_path,
            q_values=q_values,
            intensity=trace,
            source=f"direct:{observation.representative_path.name}",
        )
        return _ResolvedSAXSComponent(
            label=observation.label,
            weight=weight,
            source="observed_direct",
            trace=trace,
            profile_path=profile_path,
            structure_path=observation.representative_path,
        )

    def _build_predicted_saxs_component(
        self,
        prediction: PredictedClusterCandidate,
        *,
        weight: float,
        structure_path: Path,
        q_values: np.ndarray,
        component_dir: Path,
        pair_sigma_by_element: dict[tuple[str, str], float] | None = None,
    ) -> _ResolvedSAXSComponent | None:
        if weight <= 0.0:
            return None
        try:
            structure_coordinates, structure_elements = load_structure_file(
                structure_path
            )
            included_pair_indices = (
                _first_shell_pair_indices(
                    structure_coordinates,
                    list(structure_elements),
                    node_elements=set(self._atom_type_elements("node")),
                    pair_cutoff_definitions=self.pair_cutoff_definitions,
                )
                if pair_sigma_by_element
                else None
            )
            trace = np.asarray(
                (
                    compute_debye_intensity_with_debye_waller(
                        structure_coordinates,
                        list(structure_elements),
                        q_values,
                        pair_sigma_by_element=pair_sigma_by_element,
                        included_pair_indices=included_pair_indices,
                    )
                    if pair_sigma_by_element
                    else compute_debye_intensity(
                        structure_coordinates,
                        list(structure_elements),
                        q_values,
                    )
                ),
                dtype=float,
            )
        except Exception:
            return None

        file_stem = _predicted_structure_file_stem(prediction)
        profile_path = component_dir / f"predicted_{file_stem}.txt"
        _write_component_profile(
            profile_path,
            q_values=q_values,
            intensity=trace,
            source="predicted_structure",
        )
        return _ResolvedSAXSComponent(
            label=prediction.label,
            weight=weight,
            source="predicted",
            trace=trace,
            profile_path=profile_path,
            structure_path=structure_path,
        )

    def _load_project_component_trace(
        self,
        label: str,
        *,
        q_values: np.ndarray,
    ) -> np.ndarray | None:
        if self.project_dir is None:
            return None
        project_dir = Path(self.project_dir).expanduser().resolve()
        map_path = project_dir / "md_saxs_map.json"
        if not map_path.is_file():
            return None

        try:
            map_payload = json.loads(map_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        saxs_map = map_payload.get("saxs_map", {})
        motif_map = saxs_map.get(label)
        if not isinstance(motif_map, dict) or not motif_map:
            return None

        prior_payload: dict[str, object] = {}
        prior_path = project_dir / "md_prior_weights.json"
        if prior_path.is_file():
            try:
                prior_payload = json.loads(
                    prior_path.read_text(encoding="utf-8")
                )
            except Exception:
                prior_payload = {}
        structures_payload = prior_payload.get("structures", {})
        structure_weights = (
            structures_payload.get(label, {})
            if isinstance(structures_payload, dict)
            else {}
        )

        component_dir = build_project_paths(
            project_dir
        ).scattering_components_dir
        motif_traces: list[np.ndarray] = []
        motif_weights: list[float] = []
        for motif in sorted(motif_map):
            profile_file = str(motif_map.get(motif, "")).strip()
            if not profile_file:
                continue
            profile_path = component_dir / profile_file
            if not profile_path.is_file():
                continue
            try:
                raw_data = np.loadtxt(profile_path, comments="#")
            except Exception:
                continue
            if raw_data.size == 0:
                continue
            if raw_data.ndim == 1:
                raw_data = raw_data.reshape(1, -1)
            component_q = np.asarray(raw_data[:, 0], dtype=float)
            component_i = np.asarray(raw_data[:, 1], dtype=float)
            if component_q.size == 0 or component_i.size == 0:
                continue
            trace = np.interp(q_values, component_q, component_i)
            motif_traces.append(np.asarray(trace, dtype=float))
            motif_info = (
                structure_weights.get(motif, {})
                if isinstance(structure_weights, dict)
                else {}
            )
            motif_weight = 0.0
            if isinstance(motif_info, dict):
                count_value = motif_info.get("count")
                weight_value = motif_info.get("weight")
                if count_value is not None:
                    motif_weight = max(float(count_value), 0.0)
                elif weight_value is not None:
                    motif_weight = max(float(weight_value), 0.0)
            motif_weights.append(motif_weight)
        if not motif_traces:
            return None
        weights = np.asarray(motif_weights, dtype=float)
        if np.sum(weights) <= 0.0:
            weights = np.ones(len(motif_traces), dtype=float)
        weights = weights / np.sum(weights)
        stacked = np.asarray(motif_traces, dtype=float)
        return np.asarray(np.einsum("i,ij->j", weights, stacked), dtype=float)

    @staticmethod
    def _compose_weighted_model(
        components: list[_ResolvedSAXSComponent],
    ) -> tuple[np.ndarray, np.ndarray] | None:
        if not components:
            return None
        weights = np.asarray(
            [component.weight for component in components], dtype=float
        )
        if np.sum(weights) <= 0.0:
            return None
        normalized_weights = weights / np.sum(weights)
        stacked = np.asarray(
            [component.trace for component in components], dtype=float
        )
        raw_model = np.einsum("i,ij->j", normalized_weights, stacked)
        return np.asarray(raw_model, dtype=float), normalized_weights

    def _load_experimental_data(
        self,
        experimental_data_path: Path | None,
    ) -> ExperimentalDataSummary | None:
        if experimental_data_path is None:
            return None
        try:
            return load_experimental_data_file(experimental_data_path)
        except Exception:
            return None

    def _update_prediction_population_shares(
        self,
        training_observations: list[ClusterDynamicsMLTrainingObservation],
        predictions: list[PredictedClusterCandidate],
    ) -> None:
        del training_observations
        total_weight = sum(
            _prediction_share_weight(
                item,
                frame_timestep_fs=self.frame_timestep_fs,
            )
            for item in predictions
        )
        total_weight = max(float(total_weight), 1e-12)
        for item in predictions:
            weight = _prediction_share_weight(
                item,
                frame_timestep_fs=self.frame_timestep_fs,
            )
            item.predicted_population_share = float(weight / total_weight)

    def _prune_prediction_population_tail(
        self,
        predictions: list[PredictedClusterCandidate],
    ) -> None:
        if not predictions:
            return
        threshold = max(float(self.prediction_population_share_threshold), 0.0)
        grouped: defaultdict[int, list[PredictedClusterCandidate]] = (
            defaultdict(list)
        )
        for item in predictions:
            grouped[int(item.target_node_count)].append(item)

        retained: list[PredictedClusterCandidate] = []
        removed_any = False
        for target_node_count in sorted(grouped):
            ordered = sorted(
                grouped[target_node_count],
                key=lambda item: (
                    -float(item.predicted_population_share),
                    -float(item.predicted_stability_score),
                    item.label,
                ),
            )
            for index, item in enumerate(ordered):
                if index == 0 or threshold <= 0.0:
                    retained.append(item)
                    continue
                if float(item.predicted_population_share) >= threshold:
                    retained.append(item)
                    continue
                removed_any = True
        if removed_any:
            predictions[:] = retained
            self._update_prediction_population_shares([], predictions)
        self._reassign_prediction_ranks(predictions)

    @staticmethod
    def _reassign_prediction_ranks(
        predictions: list[PredictedClusterCandidate],
    ) -> None:
        grouped: defaultdict[int, list[PredictedClusterCandidate]] = (
            defaultdict(list)
        )
        for item in predictions:
            grouped[int(item.target_node_count)].append(item)
        reordered: list[PredictedClusterCandidate] = []
        for target_node_count in sorted(grouped, reverse=True):
            ordered = sorted(
                grouped[target_node_count],
                key=lambda item: (
                    -float(item.predicted_population_share),
                    -float(item.predicted_stability_score),
                    item.label,
                ),
            )
            for rank, item in enumerate(ordered, start=1):
                item.rank = rank
                reordered.append(item)
        predictions[:] = reordered

    def _resolve_max_predicted_node_count(
        self,
        predictions: list[PredictedClusterCandidate],
    ) -> int | None:
        qualifying = [
            item.target_node_count
            for item in predictions
            if item.predicted_population_share
            >= self.prediction_population_share_threshold
            and item.predicted_mean_lifetime_fs >= self.frame_timestep_fs
        ]
        return None if not qualifying else max(qualifying)

    def _node_count_from_counts(self, counts: dict[str, int]) -> int:
        node_elements = self._atom_type_elements("node")
        if node_elements:
            return int(
                sum(counts.get(element, 0) for element in node_elements)
            )
        return int(sum(counts.values()))

    def _atom_type_elements(self, atom_type: str) -> set[str]:
        return {
            _normalized_element_symbol(element)
            for element, _residue in self.atom_type_definitions.get(
                atom_type, []
            )
            if _normalized_element_symbol(element)
        }

    def _atom_type_by_element(self) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for atom_type, definitions in self.atom_type_definitions.items():
            geometry_type = _geometry_atom_type_label(atom_type)
            for element, _residue in definitions:
                normalized = _normalized_element_symbol(element)
                if normalized and normalized not in mapping:
                    mapping[normalized] = geometry_type
        return mapping

    def _frames_input_format(self) -> str:
        if self._cached_frames_input_format is None:
            frame_format, _frame_paths = detect_frame_folder_mode(
                self.frames_dir
            )
            self._cached_frames_input_format = str(frame_format)
        return self._cached_frames_input_format

    def _active_pdb_shell_references(
        self,
    ) -> tuple[_ResolvedPDBShellReference, ...]:
        if self._frames_input_format() != "pdb":
            return ()
        if self._cached_resolved_pdb_shell_references is None:
            self._cached_resolved_pdb_shell_references = (
                self._resolve_pdb_shell_references()
            )
        return self._cached_resolved_pdb_shell_references

    def _resolve_pdb_shell_references(
        self,
    ) -> tuple[_ResolvedPDBShellReference, ...]:
        resolved: list[_ResolvedPDBShellReference] = []
        seen_shell_keys: set[tuple[str, str | None]] = set()
        seen_shell_elements: set[str] = set()
        for definition in self.pdb_shell_reference_definitions:
            shell_element = _normalized_element_symbol(
                definition.shell_element
            )
            if not shell_element:
                continue
            shell_residue = (
                None
                if definition.shell_residue is None
                else (str(definition.shell_residue).strip().upper() or None)
            )
            shell_key = (shell_element, shell_residue)
            if shell_key in seen_shell_keys:
                raise ValueError(
                    "PDB shell references must be unique per shell "
                    f"element/residue pair. Duplicate rule: {shell_element}"
                    + (
                        ""
                        if shell_residue is None
                        else f" ({shell_residue})"
                    )
                    + "."
                )
            seen_shell_keys.add(shell_key)
            if shell_element in seen_shell_elements:
                raise ValueError(
                    "PDB shell references currently need unique shell "
                    "elements so predicted shell anchors can be matched "
                    f"unambiguously. Duplicate shell element: {shell_element}."
                )
            seen_shell_elements.add(shell_element)
            reference_path = resolve_reference_path(definition.reference_name)
            reference_structure = PDBStructure.from_file(reference_path)
            reference_entry = next(
                (
                    entry
                    for entry in list_reference_library(reference_path.parent)
                    if entry.path.resolve() == reference_path.resolve()
                ),
                None,
            )
            preferred_pairs = (
                ()
                if reference_entry is None
                else tuple(reference_entry.backbone_pairs)
            )
            atom_lookup = {
                atom.atom_name.strip(): (index, atom)
                for index, atom in enumerate(reference_structure.atoms)
            }
            matching_shell_atoms = [
                (index, atom)
                for index, atom in enumerate(reference_structure.atoms)
                if _normalized_element_symbol(atom.element) == shell_element
            ]
            if not matching_shell_atoms:
                raise ValueError(
                    "Reference molecule "
                    f"{definition.reference_name!r} does not contain any "
                    f"{shell_element} atom for the shell anchor."
                )

            deprecated_atom1_name = (
                ""
                if definition.backbone_atom1_name in {None, ""}
                else str(definition.backbone_atom1_name).strip()
            )
            deprecated_atom2_name = (
                ""
                if definition.backbone_atom2_name in {None, ""}
                else str(definition.backbone_atom2_name).strip()
            )

            def _matching_shell_atom(
                atom_name: str,
            ) -> tuple[int, PDBAtom] | None:
                if not atom_name:
                    return None
                atom_data = atom_lookup.get(atom_name)
                if atom_data is None:
                    return None
                if (
                    _normalized_element_symbol(atom_data[1].element)
                    != shell_element
                ):
                    return None
                return atom_data

            preferred_anchor_name = ""
            preferred_partner_name = ""
            for atom1_name, atom2_name in preferred_pairs:
                atom1_data = atom_lookup.get(str(atom1_name).strip())
                atom2_data = atom_lookup.get(str(atom2_name).strip())
                if (
                    atom1_data is not None
                    and _normalized_element_symbol(atom1_data[1].element)
                    == shell_element
                ):
                    preferred_anchor_name = str(atom1_name).strip()
                    preferred_partner_name = str(atom2_name).strip()
                    break
                if (
                    atom2_data is not None
                    and _normalized_element_symbol(atom2_data[1].element)
                    == shell_element
                ):
                    preferred_anchor_name = str(atom2_name).strip()
                    preferred_partner_name = str(atom1_name).strip()
                    break

            anchor_data = (
                _matching_shell_atom(deprecated_atom1_name)
                or _matching_shell_atom(deprecated_atom2_name)
                or _matching_shell_atom(preferred_anchor_name)
            )
            if anchor_data is None and len(matching_shell_atoms) == 1:
                anchor_data = matching_shell_atoms[0]
            if anchor_data is None:
                raise ValueError(
                    "Reference molecule "
                    f"{definition.reference_name!r} contains multiple "
                    f"{shell_element} atoms. Add preferred backbone metadata "
                    "in the reference library so one shell anchor can be "
                    "chosen automatically."
                )

            anchor_index, anchor_atom = anchor_data
            anchor_name = anchor_atom.atom_name.strip()
            anchor_element = _normalized_element_symbol(anchor_atom.element)

            preferred_partner = atom_lookup.get(preferred_partner_name)
            deprecated_partner = (
                atom_lookup.get(deprecated_atom2_name)
                if deprecated_atom2_name != anchor_name
                else None
            )
            fallback_partner = next(
                (
                    (index, atom)
                    for index, atom in enumerate(reference_structure.atoms)
                    if index != anchor_index
                ),
                None,
            )
            partner_data = preferred_partner
            if (
                partner_data is None
                or int(partner_data[0]) == int(anchor_index)
            ):
                partner_data = deprecated_partner
            if (
                partner_data is None
                or int(partner_data[0]) == int(anchor_index)
            ):
                partner_data = fallback_partner
            if partner_data is None:
                raise ValueError(
                    "Reference molecule "
                    f"{definition.reference_name!r} must contain at least two "
                    "atoms so the deprecated backbone placement can be kept "
                    "for future reference."
                )

            partner_index, partner_atom = partner_data
            partner_name = partner_atom.atom_name.strip()
            partner_element = _normalized_element_symbol(partner_atom.element)
            reference_coordinates = np.asarray(
                [atom.coordinates for atom in reference_structure.atoms],
                dtype=float,
            )
            non_anchor_indices = [
                index
                for index in range(len(reference_structure.atoms))
                if index != anchor_index
            ]
            reference_outward_vector = (
                np.mean(reference_coordinates[non_anchor_indices], axis=0)
                - reference_coordinates[anchor_index]
            )
            if float(np.linalg.norm(reference_outward_vector)) <= 1.0e-12:
                reference_outward_vector = (
                    reference_coordinates[partner_index]
                    - reference_coordinates[anchor_index]
                )
            backbone_distance = float(
                np.linalg.norm(partner_atom.coordinates - anchor_atom.coordinates)
            )
            resolved.append(
                _ResolvedPDBShellReference(
                    shell_element=shell_element,
                    shell_residue=shell_residue,
                    reference_name=str(definition.reference_name).strip(),
                    reference_path=reference_path,
                    reference_residue_name=(
                        reference_structure.atoms[0].residue_name
                        if reference_structure.atoms
                        else "UNK"
                    ),
                    reference_atoms=tuple(
                        atom.copy() for atom in reference_structure.atoms
                    ),
                    anchor_atom_name=anchor_name,
                    anchor_atom_element=anchor_element,
                    anchor_atom_index=int(anchor_index),
                    reference_outward_vector=np.asarray(
                        reference_outward_vector,
                        dtype=float,
                    ),
                    backbone_atom1_name=anchor_name,
                    backbone_atom2_name=partner_name,
                    backbone_atom1_element=anchor_element,
                    backbone_atom2_element=partner_element,
                    backbone_atom1_index=int(anchor_index),
                    backbone_atom2_index=int(partner_index),
                    backbone_distance=backbone_distance,
                )
            )
        return tuple(resolved)

    def _load_structure_for_analysis(
        self,
        file_path: Path,
    ) -> _LoadedMLStructure | None:
        if (
            self._frames_input_format() == "pdb"
            and file_path.suffix.lower() == ".pdb"
        ):
            return self._load_filtered_pdb_structure(file_path)
        try:
            coordinates, elements = load_structure_file(file_path)
        except Exception:
            return None
        return _LoadedMLStructure(
            coordinates=np.asarray(coordinates, dtype=float),
            elements=[
                _normalized_element_symbol(element) for element in elements
            ],
        )

    def _load_filtered_pdb_structure(
        self,
        file_path: Path,
    ) -> _LoadedMLStructure | None:
        try:
            structure = PDBStructure.from_file(
                file_path,
                atom_type_definitions=self.atom_type_definitions,
            )
        except Exception:
            return None

        selected_atoms: dict[int, PDBAtom] = {
            atom.atom_id: atom
            for atom in structure.atoms
            if atom.atom_type in {"node", "linker", "shell"}
        }
        ordered_atoms = tuple(
            atom.copy() for atom in sorted(selected_atoms.values(), key=lambda atom: atom.atom_id)
        )
        if not ordered_atoms:
            return None
        return _LoadedMLStructure(
            coordinates=np.asarray(
                [atom.coordinates for atom in ordered_atoms],
                dtype=float,
            ),
            elements=[
                _normalized_element_symbol(atom.element)
                for atom in ordered_atoms
            ],
            pdb_atoms=ordered_atoms,
        )

    def _structure_files_for_observation(
        self,
        observation: ClusterDynamicsMLTrainingObservation,
    ) -> list[Path]:
        structure_dir = Path(observation.structure_dir).expanduser()
        if structure_dir.is_dir():
            motif_dirs = sorted(
                path
                for path in structure_dir.iterdir()
                if path.is_dir() and path.name.startswith("motif_")
            )
            if motif_dirs:
                file_paths = [
                    file_path
                    for motif_dir in motif_dirs
                    for file_path in _structure_files_in_directory(motif_dir)
                ]
                if file_paths:
                    return file_paths
            file_paths = _structure_files_in_directory(structure_dir)
            if file_paths:
                return file_paths
        representative_path = (
            None
            if observation.representative_path is None
            else Path(observation.representative_path).expanduser()
        )
        if representative_path is not None and representative_path.is_file():
            return [representative_path]
        return []

    def _collect_training_geometry_statistics(
        self,
        training_observations: list[ClusterDynamicsMLTrainingObservation],
    ) -> _TrainingGeometryStatistics:
        atom_type_by_element = self._atom_type_by_element()
        node_elements = tuple(sorted(self._atom_type_elements("node")))
        tracked_atom_types = tuple(
            sorted(
                {
                    "node",
                    *atom_type_by_element.values(),
                }
            )
        )
        node_bond_lengths: list[float] = []
        bond_lengths: defaultdict[tuple[str, str], list[float]] = defaultdict(
            list
        )
        seed_contact_distances: defaultdict[tuple[str, str], list[float]] = (
            defaultdict(list)
        )
        node_angles: defaultdict[tuple[str, str], list[float]] = defaultdict(
            list
        )
        node_coordination: defaultdict[str, list[int]] = defaultdict(list)
        non_node_node_coordination: defaultdict[str, list[int]] = defaultdict(
            list
        )
        parsed_structures: list[tuple[np.ndarray, list[str], list[str]]] = []

        for observation in training_observations:
            for file_path in self._structure_files_for_observation(
                observation
            ):
                loaded = self._load_structure_for_analysis(file_path)
                if loaded is None:
                    continue
                coordinates = np.asarray(loaded.coordinates, dtype=float)
                elements = [
                    _normalized_element_symbol(element)
                    for element in loaded.elements
                ]
                geometry_types = [
                    atom_type_by_element.get(element, "shell")
                    for element in elements
                ]
                node_indices = [
                    index
                    for index, element in enumerate(elements)
                    if element in node_elements
                ]
                if not node_indices:
                    continue
                parsed_structures.append(
                    (coordinates, elements, geometry_types)
                )
                nearest_pair_distances: dict[tuple[int, str, str], float] = {}
                for atom_index, element in enumerate(elements):
                    for other_index, other_element in enumerate(elements):
                        if atom_index == other_index:
                            continue
                        pair_key = _sorted_pair_key(element, other_element)
                        distance = float(
                            np.linalg.norm(
                                coordinates[atom_index]
                                - coordinates[other_index]
                            )
                        )
                        previous = nearest_pair_distances.get(
                            (atom_index, *pair_key)
                        )
                        if previous is None or distance < previous:
                            nearest_pair_distances[(atom_index, *pair_key)] = (
                                distance
                            )
                for key, distance in nearest_pair_distances.items():
                    _atom_index, pair_a, pair_b = key
                    seed_contact_distances[(pair_a, pair_b)].append(
                        float(distance)
                    )
                node_coordinates = coordinates[node_indices]
                node_edge_pairs = _node_scaffold_edges(
                    node_coordinates,
                    [elements[index] for index in node_indices],
                    pair_cutoff_definitions=self.pair_cutoff_definitions,
                )
                node_neighbors: dict[int, list[int]] = defaultdict(list)
                for local_index_a, local_index_b in node_edge_pairs:
                    global_index_a = node_indices[local_index_a]
                    global_index_b = node_indices[local_index_b]
                    node_neighbors[global_index_a].append(global_index_b)
                    node_neighbors[global_index_b].append(global_index_a)
                    distance = float(
                        np.linalg.norm(
                            coordinates[global_index_a]
                            - coordinates[global_index_b]
                        )
                    )
                    if distance > 0.0:
                        node_bond_lengths.append(distance)
                        bond_lengths[
                            _sorted_pair_key(
                                elements[global_index_a],
                                elements[global_index_b],
                            )
                        ].append(distance)

                attached_nodes_by_atom = _associate_non_node_atoms_to_nodes(
                    coordinates,
                    elements=elements,
                    node_indices=node_indices,
                    pair_cutoff_definitions=self.pair_cutoff_definitions,
                )
                attached_atoms_by_node: dict[int, list[int]] = defaultdict(
                    list
                )
                for (
                    atom_index,
                    attached_nodes,
                ) in attached_nodes_by_atom.items():
                    if not attached_nodes:
                        continue
                    non_node_node_coordination[elements[atom_index]].append(
                        len(attached_nodes)
                    )
                    for node_index in attached_nodes:
                        attached_atoms_by_node[node_index].append(atom_index)
                        distance = float(
                            np.linalg.norm(
                                coordinates[atom_index]
                                - coordinates[node_index]
                            )
                        )
                        if distance > 0.0:
                            bond_lengths[
                                _sorted_pair_key(
                                    elements[atom_index], elements[node_index]
                                )
                            ].append(distance)

                for node_index in node_indices:
                    neighbor_vectors: list[tuple[str, np.ndarray]] = []
                    coordination_counts = {
                        atom_type: 0 for atom_type in tracked_atom_types
                    }
                    coordination_counts["node"] = len(
                        node_neighbors[node_index]
                    )
                    for neighbor_index in node_neighbors[node_index]:
                        neighbor_vectors.append(
                            (
                                "node",
                                coordinates[neighbor_index]
                                - coordinates[node_index],
                            )
                        )
                    for atom_index in attached_atoms_by_node[node_index]:
                        geometry_type = atom_type_by_element.get(
                            elements[atom_index],
                            "shell",
                        )
                        coordination_counts[geometry_type] = (
                            coordination_counts.get(geometry_type, 0) + 1
                        )
                        neighbor_vectors.append(
                            (
                                geometry_type,
                                coordinates[atom_index]
                                - coordinates[node_index],
                            )
                        )
                    for atom_type, count in coordination_counts.items():
                        node_coordination[atom_type].append(int(count))
                    for (
                        (neighbor_type_a, vector_a),
                        (neighbor_type_b, vector_b),
                    ) in combinations(neighbor_vectors, 2):
                        angle = _angle_between_vectors(vector_a, vector_b)
                        if angle is None:
                            continue
                        node_angles[
                            _sorted_pair_key(neighbor_type_a, neighbor_type_b)
                        ].append(angle)

        default_node_bond_length = _fallback_node_bond_length(
            node_elements=node_elements,
            pair_cutoff_definitions=self.pair_cutoff_definitions,
        )
        preliminary_contact_medians = {
            pair: _median_or_default(
                values,
                default=_fallback_pair_distance(
                    pair[0],
                    pair[1],
                    pair_cutoff_definitions=self.pair_cutoff_definitions,
                    default=default_node_bond_length,
                ),
            )
            for pair, values in seed_contact_distances.items()
            if values
        }
        contact_distances: defaultdict[tuple[str, str], list[float]] = (
            defaultdict(list)
        )
        geometry_contact_distances: defaultdict[
            tuple[str, str], list[float]
        ] = defaultdict(list)
        atom_coordination: defaultdict[tuple[str, str], list[int]] = (
            defaultdict(list)
        )
        for coordinates, elements, geometry_types in parsed_structures:
            neighbor_counts = [Counter() for _ in range(len(elements))]
            for atom_index_a, atom_index_b in combinations(
                range(len(elements)), 2
            ):
                pair_key = _sorted_pair_key(
                    elements[atom_index_a],
                    elements[atom_index_b],
                )
                distance = float(
                    np.linalg.norm(
                        coordinates[atom_index_a] - coordinates[atom_index_b]
                    )
                )
                if distance > _contact_distance_cutoff(
                    elements[atom_index_a],
                    elements[atom_index_b],
                    preliminary_contact_medians=preliminary_contact_medians,
                    default_distance=default_node_bond_length,
                    pair_cutoff_definitions=self.pair_cutoff_definitions,
                ):
                    continue
                contact_distances[pair_key].append(distance)
                geometry_pair_key = _sorted_pair_key(
                    geometry_types[atom_index_a],
                    geometry_types[atom_index_b],
                )
                geometry_contact_distances[geometry_pair_key].append(distance)
                neighbor_counts[atom_index_a][
                    geometry_types[atom_index_b]
                ] += 1
                neighbor_counts[atom_index_b][
                    geometry_types[atom_index_a]
                ] += 1
            for atom_index, center_type in enumerate(geometry_types):
                for neighbor_type in tracked_atom_types:
                    atom_coordination[(center_type, neighbor_type)].append(
                        int(neighbor_counts[atom_index].get(neighbor_type, 0))
                    )
        return _TrainingGeometryStatistics(
            atom_type_by_element=atom_type_by_element,
            node_elements=node_elements,
            tracked_atom_types=tracked_atom_types,
            node_bond_length=_median_or_default(
                node_bond_lengths,
                default=default_node_bond_length,
            ),
            bond_length_medians={
                pair: _median_or_default(
                    values,
                    default=_fallback_pair_distance(
                        pair[0],
                        pair[1],
                        pair_cutoff_definitions=self.pair_cutoff_definitions,
                        default=default_node_bond_length,
                    ),
                )
                for pair, values in bond_lengths.items()
                if values
            },
            contact_distance_medians={
                pair: _median_or_default(
                    values,
                    default=_fallback_pair_distance(
                        pair[0],
                        pair[1],
                        pair_cutoff_definitions=self.pair_cutoff_definitions,
                        default=default_node_bond_length,
                    ),
                )
                for pair, values in contact_distances.items()
                if values
            },
            geometry_contact_distance_medians={
                pair: _median_or_default(
                    values,
                    default=default_node_bond_length,
                )
                for pair, values in geometry_contact_distances.items()
                if values
            },
            node_angle_medians={
                pair: _median_or_default(
                    values,
                    default=180.0 if pair == ("node", "node") else 120.0,
                )
                for pair, values in node_angles.items()
                if values
            },
            node_coordination_medians={
                atom_type: _median_or_default(values, default=0.0)
                for atom_type, values in node_coordination.items()
            },
            non_node_node_coordination_medians={
                element: _median_or_default(values, default=1.0)
                for element, values in non_node_node_coordination.items()
            },
            atom_coordination_medians={
                pair: _median_or_default(values, default=0.0)
                for pair, values in atom_coordination.items()
            },
        )

    @staticmethod
    def _status_message(
        step: int,
        total_steps: int,
        title: str,
        detail: str,
    ) -> str:
        header = f"Step {int(step)}/{int(total_steps)}: {str(title).strip()}"
        body = str(detail).strip()
        return header if not body else f"{header}\n{body}"

    @staticmethod
    def _emit(
        callback: PredictionProgressCallback | None,
        message: str,
    ) -> None:
        if callback is not None:
            callback(str(message))


def _fit_model_to_experimental(
    raw_model: np.ndarray,
    experimental_intensity: np.ndarray,
) -> tuple[float, float, np.ndarray, np.ndarray, float]:
    design = np.column_stack([raw_model, np.ones_like(raw_model)])
    try:
        scale_factor, offset = np.linalg.lstsq(
            design,
            experimental_intensity,
            rcond=None,
        )[0]
    except np.linalg.LinAlgError:
        scale_factor, offset = (1.0, 0.0)
    fitted_model = raw_model * float(scale_factor) + float(offset)
    residuals = experimental_intensity - fitted_model
    rmse = float(np.sqrt(np.mean(residuals**2)))
    return (
        float(scale_factor),
        float(offset),
        np.asarray(fitted_model, dtype=float),
        np.asarray(residuals, dtype=float),
        rmse,
    )


def _saxs_component_weight(
    mean_count_per_frame: float, occupancy_fraction: float
) -> float:
    return max(float(mean_count_per_frame), 0.0) * max(
        float(occupancy_fraction), 0.05
    )


def _resolved_population_weights(
    training_observations: (
        list[ClusterDynamicsMLTrainingObservation]
        | tuple[ClusterDynamicsMLTrainingObservation, ...]
    ),
    predictions: (
        list[PredictedClusterCandidate] | tuple[PredictedClusterCandidate, ...]
    ),
    *,
    frame_timestep_fs: float,
) -> tuple[np.ndarray, np.ndarray]:
    observed_weights = np.asarray(
        [
            _saxs_component_weight(
                observation.mean_count_per_frame,
                observation.occupancy_fraction,
            )
            for observation in training_observations
        ],
        dtype=float,
    )
    if not predictions:
        return observed_weights, np.zeros(0, dtype=float)

    prediction_base_weights = np.asarray(
        [
            _saxs_component_weight(
                prediction.predicted_mean_count_per_frame,
                prediction.predicted_occupancy_fraction,
            )
            for prediction in predictions
        ],
        dtype=float,
    )
    prediction_shares = np.asarray(
        [
            max(float(prediction.predicted_population_share), 0.0)
            for prediction in predictions
        ],
        dtype=float,
    )
    if np.sum(prediction_shares) <= 0.0:
        prediction_shares = np.asarray(
            [
                _prediction_share_weight(
                    prediction,
                    frame_timestep_fs=frame_timestep_fs,
                )
                for prediction in predictions
            ],
            dtype=float,
        )
    total_prediction_share = float(np.sum(prediction_shares))
    if total_prediction_share <= 0.0:
        return observed_weights, prediction_base_weights
    normalized_prediction_shares = prediction_shares / total_prediction_share

    predicted_total_weight = _predicted_total_weight_from_observed_tail(
        training_observations,
        observed_weights,
        predictions,
    )
    if predicted_total_weight <= 0.0:
        positive_observed = observed_weights[observed_weights > 0.0]
        positive_prediction_total = float(np.sum(prediction_base_weights))
        if positive_observed.size > 0:
            predicted_total_weight = float(np.min(positive_observed))
        elif positive_prediction_total > 0.0:
            predicted_total_weight = positive_prediction_total
        else:
            predicted_total_weight = 1.0
    if predicted_total_weight <= 0.0:
        return observed_weights, prediction_base_weights
    return (
        observed_weights,
        np.asarray(
            normalized_prediction_shares * predicted_total_weight,
            dtype=float,
        ),
    )


def _predicted_total_weight_from_observed_tail(
    training_observations: (
        list[ClusterDynamicsMLTrainingObservation]
        | tuple[ClusterDynamicsMLTrainingObservation, ...]
    ),
    observed_weights: np.ndarray,
    predictions: (
        list[PredictedClusterCandidate] | tuple[PredictedClusterCandidate, ...]
    ),
) -> float:
    positive_observed = observed_weights[observed_weights > 0.0]
    if positive_observed.size == 0:
        return 0.0

    observed_size_totals: defaultdict[int, float] = defaultdict(float)
    for observation, weight in zip(
        training_observations,
        observed_weights,
        strict=False,
    ):
        weight_value = max(float(weight), 0.0)
        if weight_value <= 0.0:
            continue
        observed_size_totals[int(observation.node_count)] += weight_value

    positive_sizes = sorted(
        size for size, total in observed_size_totals.items() if total > 0.0
    )
    if not positive_sizes:
        return float(np.min(positive_observed))

    last_observed_size = positive_sizes[-1]
    last_observed_total = float(observed_size_totals[last_observed_size])
    if last_observed_total <= 0.0:
        return float(np.min(positive_observed))

    step_ratios: list[float] = []
    for previous_size, current_size in zip(
        positive_sizes,
        positive_sizes[1:],
        strict=False,
    ):
        previous_total = float(observed_size_totals[previous_size])
        current_total = float(observed_size_totals[current_size])
        if previous_total <= 0.0 or current_total <= 0.0:
            continue
        gap = max(int(current_size - previous_size), 1)
        step_ratios.append((current_total / previous_total) ** (1.0 / gap))
    decay_per_node = (
        float(np.median(np.asarray(step_ratios, dtype=float)))
        if step_ratios
        else 0.5
    )
    decay_per_node = float(np.clip(decay_per_node, 0.05, 0.9))

    target_sizes = sorted(
        {
            int(prediction.target_node_count)
            for prediction in predictions
            if int(prediction.target_node_count) > 0
        }
    )
    if not target_sizes:
        return min(last_observed_total, float(np.min(positive_observed)))

    extrapolated_total = 0.0
    for target_size in target_sizes:
        if target_size <= last_observed_size:
            extrapolated_total += float(
                observed_size_totals.get(target_size, last_observed_total)
            )
            continue
        extrapolated_total += float(
            last_observed_total
            * (decay_per_node ** (target_size - last_observed_size))
        )
    return float(
        min(
            extrapolated_total,
            last_observed_total,
        )
    )


def _prediction_share_weight(
    prediction: PredictedClusterCandidate,
    *,
    frame_timestep_fs: float,
) -> float:
    saxs_weight = _saxs_component_weight(
        prediction.predicted_mean_count_per_frame,
        prediction.predicted_occupancy_fraction,
    )
    if saxs_weight > 0.0:
        return float(saxs_weight)

    occupancy_weight = max(float(prediction.predicted_occupancy_fraction), 0.0)
    if occupancy_weight > 0.0:
        return occupancy_weight

    timestep = max(float(frame_timestep_fs), 1e-12)
    return max(float(prediction.predicted_mean_lifetime_fs), 0.0) / timestep


def _safe_component_stem(label: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", str(label).strip())
    return sanitized.strip("._") or "component"


def _write_component_profile(
    output_path: Path,
    *,
    q_values: np.ndarray,
    intensity: np.ndarray,
    source: str,
) -> None:
    q_values = np.asarray(q_values, dtype=float)
    intensity = np.asarray(intensity, dtype=float)
    if q_values.shape != intensity.shape:
        raise ValueError("SAXS component q and intensity arrays must match.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        f"# Source: {source}\n" "# Columns: q, S(q)_avg, S(q)_std, S(q)_se\n"
    )
    data = np.column_stack(
        [
            q_values,
            intensity,
            np.zeros_like(intensity, dtype=float),
            np.zeros_like(intensity, dtype=float),
        ]
    )
    np.savetxt(
        output_path,
        data,
        comments="",
        header=header,
        fmt=["%.8f", "%.8f", "%.8f", "%.8f"],
    )


def _write_xyz_structure(
    output_path: Path,
    *,
    label: str,
    elements: tuple[str, ...] | list[str],
    coordinates: np.ndarray,
    comment: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    coords = np.asarray(coordinates, dtype=float)
    lines = [str(len(elements)), f"label={label} {comment}".strip()]
    for element, position in zip(elements, coords, strict=False):
        lines.append(
            f"{element} {float(position[0]):.8f} {float(position[1]):.8f} "
            f"{float(position[2]):.8f}"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _predicted_structure_file_stem(
    prediction: PredictedClusterCandidate,
) -> str:
    return (
        f"{prediction.target_node_count:02d}_rank{prediction.rank:02d}_"
        f"{_safe_component_stem(prediction.label)}"
    )


def _write_predicted_structure_file(
    prediction: PredictedClusterCandidate,
    *,
    predicted_structure_dir: Path,
) -> Path:
    structure_path = (
        predicted_structure_dir
        / f"{_predicted_structure_file_stem(prediction)}.xyz"
    )
    _write_xyz_structure(
        structure_path,
        label=prediction.label,
        elements=prediction.generated_elements,
        coordinates=prediction.generated_coordinates,
        comment=(
            f"target_node_count={prediction.target_node_count} "
            f"rank={prediction.rank} source_label="
            f"{'' if prediction.source_label is None else prediction.source_label}"
        ),
    )
    return structure_path


def _fit_property_model(
    feature_matrix: np.ndarray,
    targets: np.ndarray,
    *,
    weights: np.ndarray,
    default_value: float,
    transform: str = "identity",
    lower_bound: float | None = None,
    upper_bound: float | None = None,
) -> _PropertyModel:
    x_values = np.asarray(feature_matrix, dtype=float)
    y_values = np.asarray(targets, dtype=float)
    weight_values = np.asarray(weights, dtype=float)
    if y_values.size == 0:
        return _PropertyModel(
            coefficients=None,
            constant_value=None,
            transform=transform,
            default_value=default_value,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )
    transformed_targets = (
        np.log1p(np.clip(y_values, 0.0, None))
        if transform == "log1p"
        else y_values
    )
    if transformed_targets.size == 1:
        return _PropertyModel(
            coefficients=None,
            constant_value=float(transformed_targets[0]),
            transform=transform,
            default_value=default_value,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )
    clipped_weights = np.clip(weight_values, 1e-9, None)
    sqrt_weights = np.sqrt(clipped_weights)
    weighted_x = x_values * sqrt_weights[:, None]
    weighted_y = transformed_targets * sqrt_weights
    gram = weighted_x.T @ weighted_x
    ridge = np.eye(gram.shape[0], dtype=float) * _RIDGE_REGULARIZATION
    try:
        coefficients = np.linalg.solve(
            gram + ridge,
            weighted_x.T @ weighted_y,
        )
    except np.linalg.LinAlgError:
        coefficients = np.linalg.pinv(gram + ridge) @ (
            weighted_x.T @ weighted_y
        )
    return _PropertyModel(
        coefficients=np.asarray(coefficients, dtype=float),
        constant_value=None,
        transform=transform,
        default_value=default_value,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )


def _filtered_structure_counts(
    elements: list[str] | tuple[str, ...],
    *,
    tracked_elements: set[str],
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for raw_element in elements:
        normalized = _normalized_element_symbol(raw_element)
        if not normalized:
            continue
        if tracked_elements and normalized not in tracked_elements:
            continue
        counts[normalized] += 1
    return _normalized_counts(dict(counts))


def _select_representative_structure_counts(
    count_rows: list[tuple[Path, dict[str, int]]],
    *,
    fallback_path: Path | None,
    fallback_counts: dict[str, int],
) -> tuple[Path | None, dict[str, int]]:
    if not count_rows:
        return fallback_path, _normalized_counts(fallback_counts)

    signature_counts: Counter[tuple[tuple[str, int], ...]] = Counter(
        tuple(sorted(counts.items())) for _path, counts in count_rows if counts
    )
    if not signature_counts:
        return fallback_path, _normalized_counts(fallback_counts)

    best_signature = min(
        signature_counts,
        key=lambda signature: (
            -signature_counts[signature],
            -sum(count for _element, count in signature),
            signature,
        ),
    )
    matching_rows = [
        (path, counts)
        for path, counts in count_rows
        if tuple(sorted(counts.items())) == best_signature
    ]
    representative_path, representative_counts = min(
        matching_rows,
        key=lambda item: (
            -sum(item[1].values()),
            str(item[0]),
        ),
    )
    return representative_path, _normalized_counts(representative_counts)


def _parse_stoichiometry_label(label: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for element, count_text in _STOICHIOMETRY_TOKEN_PATTERN.findall(
        str(label)
    ):
        normalized = _normalized_element_symbol(element)
        if not normalized:
            continue
        counts[normalized] = counts.get(normalized, 0) + (
            int(count_text) if count_text else 1
        )
    return counts


def _normalized_counts(counts: dict[str, int]) -> dict[str, int]:
    return {
        _normalized_element_symbol(element): int(count)
        for element, count in counts.items()
        if int(count) > 0 and _normalized_element_symbol(element)
    }


def _normalized_element_symbol(raw_value: str) -> str:
    text = str(raw_value).strip()
    if not text:
        return ""
    return text[:1].upper() + text[1:].lower()


def _structure_files_for_cluster_dir(structure_dir: Path) -> tuple[Path, ...]:
    if not structure_dir.is_dir():
        return ()
    return tuple(
        sorted(
            (
                path
                for path in structure_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in {".xyz", ".pdb"}
            ),
            key=lambda path: str(path),
        )
    )


def _filtered_coordinates_and_elements(
    coordinates: np.ndarray,
    elements: list[str] | tuple[str, ...],
    *,
    tracked_elements: set[str],
) -> tuple[np.ndarray, list[str]] | None:
    normalized_elements = [
        _normalized_element_symbol(element) for element in elements
    ]
    keep_indices = [
        index
        for index, element in enumerate(normalized_elements)
        if element and (not tracked_elements or element in tracked_elements)
    ]
    if len(keep_indices) < 2:
        return None
    return (
        np.asarray(coordinates, dtype=float)[keep_indices],
        [normalized_elements[index] for index in keep_indices],
    )


def _sorted_pair_distances_by_element(
    coordinates: np.ndarray,
    elements: list[str] | tuple[str, ...],
) -> dict[tuple[str, str], np.ndarray]:
    pair_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    coords = np.asarray(coordinates, dtype=float)
    normalized_elements = [
        _normalized_element_symbol(element) for element in elements
    ]
    for index_a, index_b in combinations(range(len(normalized_elements)), 2):
        element_a = normalized_elements[index_a]
        element_b = normalized_elements[index_b]
        if not element_a or not element_b:
            continue
        pair_key = tuple(sorted((element_a, element_b)))
        pair_values[pair_key].append(
            float(np.linalg.norm(coords[index_a] - coords[index_b]))
        )
    return {
        pair_key: np.asarray(sorted(values), dtype=float)
        for pair_key, values in pair_values.items()
        if values
    }


def _first_shell_pair_indices(
    coordinates: np.ndarray,
    elements: list[str] | tuple[str, ...],
    *,
    node_elements: set[str],
    pair_cutoff_definitions: PairCutoffDefinitions,
) -> set[tuple[int, int]]:
    coords = np.asarray(coordinates, dtype=float)
    normalized_elements = [
        _normalized_element_symbol(element) for element in elements
    ]
    pair_indices: set[tuple[int, int]] = set()

    for index_a, index_b in combinations(range(len(normalized_elements)), 2):
        element_a = normalized_elements[index_a]
        element_b = normalized_elements[index_b]
        if not element_a or not element_b:
            continue
        cutoff = _pair_cutoff_distance(
            element_a,
            element_b,
            pair_cutoff_definitions=pair_cutoff_definitions,
        )
        if cutoff is None:
            continue
        distance = float(np.linalg.norm(coords[index_a] - coords[index_b]))
        if distance <= float(cutoff) * 1.15:
            pair_indices.add((min(index_a, index_b), max(index_a, index_b)))

    node_indices = [
        index
        for index, element in enumerate(normalized_elements)
        if element in node_elements
    ]
    if len(node_indices) >= 2:
        node_edges = _node_scaffold_edges(
            coords[node_indices],
            [normalized_elements[index] for index in node_indices],
            pair_cutoff_definitions=pair_cutoff_definitions,
        )
        for local_index_a, local_index_b in node_edges:
            global_index_a = node_indices[local_index_a]
            global_index_b = node_indices[local_index_b]
            pair_indices.add(
                (
                    min(global_index_a, global_index_b),
                    max(global_index_a, global_index_b),
                )
            )

    return pair_indices


def _first_shell_pair_distances_by_element(
    coordinates: np.ndarray,
    elements: list[str] | tuple[str, ...],
    *,
    node_elements: set[str],
    pair_cutoff_definitions: PairCutoffDefinitions,
) -> dict[tuple[str, str], np.ndarray]:
    pair_indices = _first_shell_pair_indices(
        coordinates,
        elements,
        node_elements=node_elements,
        pair_cutoff_definitions=pair_cutoff_definitions,
    )
    if not pair_indices:
        return {}
    coords = np.asarray(coordinates, dtype=float)
    normalized_elements = [
        _normalized_element_symbol(element) for element in elements
    ]
    pair_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for index_a, index_b in sorted(pair_indices):
        element_a = normalized_elements[index_a]
        element_b = normalized_elements[index_b]
        if not element_a or not element_b:
            continue
        pair_key = tuple(sorted((element_a, element_b)))
        pair_values[pair_key].append(
            float(np.linalg.norm(coords[index_a] - coords[index_b]))
        )
    return {
        pair_key: np.asarray(sorted(values), dtype=float)
        for pair_key, values in pair_values.items()
        if values
    }


def _element_pair_keys_from_counts(
    counts: dict[str, int],
) -> tuple[tuple[str, str], ...]:
    normalized = _normalized_counts(counts)
    active_elements = sorted(
        element for element, count in normalized.items() if int(count) > 0
    )
    pair_keys: list[tuple[str, str]] = []
    for index, element_a in enumerate(active_elements):
        if normalized.get(element_a, 0) >= 2:
            pair_keys.append((element_a, element_a))
        for element_b in active_elements[index + 1 :]:
            if normalized.get(element_b, 0) > 0:
                pair_keys.append((element_a, element_b))
    return tuple(pair_keys)


def _debye_waller_sigma_lookup_by_label(
    rows: list[DebyeWallerPairEstimate] | tuple[DebyeWallerPairEstimate, ...],
) -> dict[str, dict[tuple[str, str], float]]:
    lookup: dict[str, dict[tuple[str, str], float]] = defaultdict(dict)
    for row in rows:
        lookup[row.label][(row.element_a, row.element_b)] = float(row.sigma)
    return {label: dict(pair_map) for label, pair_map in lookup.items()}


def _candidate_feature_vector(
    counts: dict[str, int],
    *,
    node_elements: tuple[str, ...],
    non_node_elements: tuple[str, ...],
) -> np.ndarray:
    normalized = _normalized_counts(counts)
    node_count = max(
        float(sum(normalized.get(element, 0) for element in node_elements)),
        1.0,
    )
    total_atoms = float(sum(normalized.values()))
    ratios = [
        float(normalized.get(element, 0)) / node_count
        for element in non_node_elements
    ]
    return np.asarray([1.0, node_count, total_atoms, *ratios], dtype=float)


def _weighted_node_element_fractions(
    observations: list[ClusterDynamicsMLTrainingObservation],
    *,
    node_elements: tuple[str, ...],
) -> dict[str, float]:
    if not node_elements:
        return {}
    totals = {element: 0.0 for element in node_elements}
    total_weight = 0.0
    for row in observations:
        node_count = max(row.node_count, 0)
        if node_count <= 0:
            continue
        total_weight += row.stability_weight
        for element in node_elements:
            totals[element] += (
                row.stability_weight
                * row.element_counts.get(element, 0)
                / node_count
            )
    if total_weight <= 0.0:
        uniform = 1.0 / float(len(node_elements))
        return {element: uniform for element in node_elements}
    fractions = {
        element: totals[element] / total_weight for element in node_elements
    }
    fraction_sum = max(sum(fractions.values()), 1e-12)
    return {
        element: value / fraction_sum for element, value in fractions.items()
    }


def _allocate_node_counts(
    target_node_count: int,
    node_element_fractions: dict[str, float],
) -> dict[str, int]:
    if not node_element_fractions:
        return {}
    exact = {
        element: float(target_node_count) * float(fraction)
        for element, fraction in node_element_fractions.items()
    }
    allocated = {
        element: int(math.floor(value)) for element, value in exact.items()
    }
    remainder = int(target_node_count - sum(allocated.values()))
    ranked_remainders = sorted(
        exact.items(),
        key=lambda item: (-(item[1] - math.floor(item[1])), item[0]),
    )
    for index in range(remainder):
        element = ranked_remainders[index % len(ranked_remainders)][0]
        allocated[element] += 1
    return {
        element: count for element, count in allocated.items() if count > 0
    }


def _candidate_has_support(
    observations: list[ClusterDynamicsMLTrainingObservation],
    *,
    counts: dict[str, int],
    target_node_count: int,
    node_elements: tuple[str, ...],
    non_node_elements: tuple[str, ...],
) -> bool:
    del node_elements
    normalized = _normalized_counts(counts)
    if any(normalized.get(element, 0) > 0 for element in non_node_elements):
        return True

    supported_pure_node_sizes = sorted(
        {
            int(row.node_count)
            for row in observations
            if row.node_count > 0
            and not any(
                int(row.element_counts.get(element, 0)) > 0
                for element in non_node_elements
            )
        }
    )
    if not supported_pure_node_sizes:
        return False
    return int(target_node_count) <= max(supported_pure_node_sizes) + 1


def _required_non_node_count_floors(
    observations: list[ClusterDynamicsMLTrainingObservation],
    *,
    target_node_count: int,
    non_node_elements: tuple[str, ...],
    atom_type_by_element: dict[str, str],
) -> dict[str, int]:
    if int(target_node_count) <= 1:
        return {}
    multi_node_observations = [
        row
        for row in observations
        if int(row.node_count) >= 2 and int(row.node_count) > 0
    ]
    if not multi_node_observations:
        return {}

    required_counts: dict[str, int] = {}
    for element in non_node_elements:
        if atom_type_by_element.get(element) != "linker":
            continue
        ratios: list[float] = []
        for row in multi_node_observations:
            count = int(row.element_counts.get(element, 0))
            if count <= 0:
                ratios = []
                break
            ratios.append(float(count) / float(row.node_count))
        if not ratios:
            continue
        minimum_ratio = min(ratios)
        required_counts[element] = max(
            1,
            int(math.ceil(float(target_node_count) * minimum_ratio - 1e-9)),
        )
    return required_counts


def _composition_distance(
    counts_a: dict[str, int],
    counts_b: dict[str, int],
    *,
    node_count: int,
) -> float:
    elements = sorted(set(counts_a) | set(counts_b))
    denominator = max(float(node_count), 1.0)
    return float(
        sum(
            abs(counts_a.get(element, 0) - counts_b.get(element, 0))
            for element in elements
        )
        / denominator
    )


def _structure_descriptor_row(
    coordinates: np.ndarray,
) -> tuple[float, float, float, float, float, float]:
    coords = np.asarray(coordinates, dtype=float)
    if coords.size == 0:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    centered = coords - np.mean(coords, axis=0, keepdims=True)
    radial = np.linalg.norm(centered, axis=1)
    rg = float(np.sqrt(np.mean(np.sum(centered**2, axis=1))))
    max_radius = float(np.max(radial))
    covariance = np.cov(centered.T, bias=True)
    eigvals = np.linalg.eigvalsh(covariance)
    semiaxes = np.sqrt(np.clip(np.sort(eigvals)[::-1], 0.0, None))
    if semiaxes.size < 3:
        semiaxes = np.pad(semiaxes, (0, 3 - semiaxes.size))
    return (
        float(len(coords)),
        rg,
        max_radius,
        float(semiaxes[0]),
        float(semiaxes[1]),
        float(semiaxes[2]),
    )


def _empty_lifetime_summary(
    label: str,
    *,
    cluster_size: int,
) -> ClusterLifetimeSummary:
    return ClusterLifetimeSummary(
        label=label,
        cluster_size=int(cluster_size),
        total_observations=0,
        occupied_frames=0,
        mean_count_per_frame=0.0,
        occupancy_fraction=0.0,
        association_events=0,
        dissociation_events=0,
        association_rate_per_ps=0.0,
        dissociation_rate_per_ps=0.0,
        completed_lifetime_count=0,
        window_truncated_lifetime_count=0,
        mean_lifetime_fs=None,
        std_lifetime_fs=None,
    )


def _geometry_atom_type_label(atom_type: str) -> str:
    normalized = str(atom_type).strip().lower()
    if normalized == "node":
        return "node"
    if normalized == "linker":
        return "linker"
    return "shell"


def _structure_files_in_directory(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in {".xyz", ".pdb"}
    )


def _sorted_pair_key(value_a: str, value_b: str) -> tuple[str, str]:
    return tuple(sorted((str(value_a), str(value_b))))


def _median_or_default(
    values: list[float] | np.ndarray | tuple[float, ...],
    *,
    default: float,
) -> float:
    if len(values) == 0:
        return float(default)
    return float(np.median(np.asarray(values, dtype=float)))


def _pair_cutoff_distance(
    element_a: str,
    element_b: str,
    *,
    pair_cutoff_definitions: PairCutoffDefinitions,
) -> float | None:
    normalized_a = _normalized_element_symbol(element_a)
    normalized_b = _normalized_element_symbol(element_b)
    for (atom1, atom2), level_map in pair_cutoff_definitions.items():
        if {
            _normalized_element_symbol(atom1),
            _normalized_element_symbol(atom2),
        } != {normalized_a, normalized_b}:
            continue
        if 0 in level_map:
            return float(level_map[0])
        if level_map:
            return float(min(level_map.values()))
    return None


def _fallback_pair_distance(
    element_a: str,
    element_b: str,
    *,
    pair_cutoff_definitions: PairCutoffDefinitions,
    default: float,
) -> float:
    pair_cutoff = _pair_cutoff_distance(
        element_a,
        element_b,
        pair_cutoff_definitions=pair_cutoff_definitions,
    )
    if pair_cutoff is not None:
        return float(pair_cutoff)
    return float(default)


def _pair_contact_distance(
    element_a: str,
    element_b: str,
    *,
    geometry_statistics: _TrainingGeometryStatistics,
    pair_cutoff_definitions: PairCutoffDefinitions,
) -> float:
    pair_key = _sorted_pair_key(element_a, element_b)
    if pair_key in geometry_statistics.contact_distance_medians:
        return float(geometry_statistics.contact_distance_medians[pair_key])
    if pair_key in geometry_statistics.bond_length_medians:
        return float(geometry_statistics.bond_length_medians[pair_key])
    geometry_pair_key = _sorted_pair_key(
        geometry_statistics.atom_type_by_element.get(element_a, "shell"),
        geometry_statistics.atom_type_by_element.get(element_b, "shell"),
    )
    if (
        geometry_pair_key
        in geometry_statistics.geometry_contact_distance_medians
    ):
        return float(
            geometry_statistics.geometry_contact_distance_medians[
                geometry_pair_key
            ]
        )
    return float(
        _fallback_pair_distance(
            element_a,
            element_b,
            pair_cutoff_definitions=pair_cutoff_definitions,
            default=geometry_statistics.node_bond_length,
        )
    )


def _contact_distance_cutoff(
    element_a: str,
    element_b: str,
    *,
    preliminary_contact_medians: dict[tuple[str, str], float] | None = None,
    geometry_statistics: _TrainingGeometryStatistics | None = None,
    default_distance: float,
    pair_cutoff_definitions: PairCutoffDefinitions,
) -> float:
    pair_key = _sorted_pair_key(element_a, element_b)
    if geometry_statistics is not None:
        target_distance = _pair_contact_distance(
            element_a,
            element_b,
            geometry_statistics=geometry_statistics,
            pair_cutoff_definitions=pair_cutoff_definitions,
        )
    elif (
        preliminary_contact_medians is not None
        and pair_key in preliminary_contact_medians
    ):
        target_distance = float(preliminary_contact_medians[pair_key])
    else:
        target_distance = float(
            _fallback_pair_distance(
                element_a,
                element_b,
                pair_cutoff_definitions=pair_cutoff_definitions,
                default=default_distance,
            )
        )
    fallback_limit = float(
        _fallback_pair_distance(
            element_a,
            element_b,
            pair_cutoff_definitions=pair_cutoff_definitions,
            default=default_distance,
        )
    )
    return float(
        min(target_distance * 1.25 + 0.15, fallback_limit * 1.40 + 0.20)
    )


def _fallback_node_bond_length(
    *,
    node_elements: tuple[str, ...],
    pair_cutoff_definitions: PairCutoffDefinitions,
) -> float:
    explicit_node_lengths = [
        _pair_cutoff_distance(
            element_a,
            element_b,
            pair_cutoff_definitions=pair_cutoff_definitions,
        )
        for element_a, element_b in combinations(node_elements, 2)
    ]
    explicit_node_lengths.extend(
        _pair_cutoff_distance(
            element,
            element,
            pair_cutoff_definitions=pair_cutoff_definitions,
        )
        for element in node_elements
    )
    explicit_node_lengths = [
        float(value) for value in explicit_node_lengths if value is not None
    ]
    if explicit_node_lengths:
        return float(np.median(np.asarray(explicit_node_lengths, dtype=float)))
    bridging_distances = [
        float(level_value) * 2.0
        for (atom1, atom2), level_map in pair_cutoff_definitions.items()
        if (
            _normalized_element_symbol(atom1) in node_elements
            and _normalized_element_symbol(atom2) not in node_elements
        )
        or (
            _normalized_element_symbol(atom2) in node_elements
            and _normalized_element_symbol(atom1) not in node_elements
        )
        for level_value in level_map.values()
    ]
    if bridging_distances:
        return float(np.median(np.asarray(bridging_distances, dtype=float)))
    return 3.0


def _angle_between_vectors(
    vector_a: np.ndarray,
    vector_b: np.ndarray,
) -> float | None:
    norm_a = float(np.linalg.norm(vector_a))
    norm_b = float(np.linalg.norm(vector_b))
    if norm_a <= 1e-12 or norm_b <= 1e-12:
        return None
    cosine = float(np.dot(vector_a, vector_b) / (norm_a * norm_b))
    return float(math.degrees(math.acos(np.clip(cosine, -1.0, 1.0))))


def _safe_unit_vector(
    vector: np.ndarray,
    *,
    fallback: np.ndarray | None = None,
) -> np.ndarray:
    candidate = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(candidate))
    if norm > 1e-12:
        return candidate / norm
    if fallback is None:
        return np.asarray([1.0, 0.0, 0.0], dtype=float)
    return _safe_unit_vector(np.asarray(fallback, dtype=float))


def _orthonormal_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unit_axis = _safe_unit_vector(axis)
    if abs(float(unit_axis[0])) < 0.9:
        reference = np.asarray([1.0, 0.0, 0.0], dtype=float)
    else:
        reference = np.asarray([0.0, 1.0, 0.0], dtype=float)
    basis_u = np.cross(unit_axis, reference)
    basis_u = _safe_unit_vector(basis_u, fallback=np.asarray([0.0, 1.0, 0.0]))
    basis_v = np.cross(unit_axis, basis_u)
    basis_v = _safe_unit_vector(basis_v, fallback=np.asarray([0.0, 0.0, 1.0]))
    return basis_u, basis_v


def _cone_directions(
    axis: np.ndarray,
    *,
    angle_degrees: float,
    samples: int = 8,
) -> list[np.ndarray]:
    unit_axis = _safe_unit_vector(axis)
    theta = math.radians(float(angle_degrees))
    if theta <= 1e-6:
        return [unit_axis]
    if abs(theta - math.pi) <= 1e-6:
        return [unit_axis * -1.0]
    basis_u, basis_v = _orthonormal_basis(unit_axis)
    directions: list[np.ndarray] = []
    for sample_index in range(max(int(samples), 1)):
        phi = (2.0 * math.pi * float(sample_index)) / float(max(samples, 1))
        radial = math.cos(phi) * basis_u + math.sin(phi) * basis_v
        directions.append(
            _safe_unit_vector(
                math.cos(theta) * unit_axis + math.sin(theta) * radial
            )
        )
    return directions


def _direction_basis(reference: np.ndarray) -> list[np.ndarray]:
    axis = _safe_unit_vector(reference)
    basis_u, basis_v = _orthonormal_basis(axis)
    candidates = [
        axis,
        axis * -1.0,
        basis_u,
        basis_u * -1.0,
        basis_v,
        basis_v * -1.0,
        axis + basis_u,
        axis - basis_u,
        axis + basis_v,
        axis - basis_v,
        basis_u + basis_v,
        basis_u - basis_v,
    ]
    unique: list[np.ndarray] = []
    for candidate in candidates:
        direction = _safe_unit_vector(candidate)
        if any(np.allclose(direction, other) for other in unique):
            continue
        unique.append(direction)
    return unique


def _minimum_spanning_tree_edges(
    coordinates: np.ndarray,
) -> list[tuple[int, int]]:
    coords = np.asarray(coordinates, dtype=float)
    point_count = len(coords)
    if point_count < 2:
        return []
    parent = list(range(point_count))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(index_a: int, index_b: int) -> bool:
        root_a = find(index_a)
        root_b = find(index_b)
        if root_a == root_b:
            return False
        parent[root_b] = root_a
        return True

    all_edges = sorted(
        (
            float(np.linalg.norm(coords[index_a] - coords[index_b])),
            index_a,
            index_b,
        )
        for index_a, index_b in combinations(range(point_count), 2)
    )
    selected: list[tuple[int, int]] = []
    for _distance, index_a, index_b in all_edges:
        if not union(index_a, index_b):
            continue
        selected.append((index_a, index_b))
        if len(selected) == point_count - 1:
            break
    return selected


def _node_scaffold_edges(
    coordinates: np.ndarray,
    node_elements: list[str] | tuple[str, ...],
    *,
    pair_cutoff_definitions: PairCutoffDefinitions,
) -> list[tuple[int, int]]:
    coords = np.asarray(coordinates, dtype=float)
    point_count = len(coords)
    if point_count < 2:
        return []
    explicit_edges: set[tuple[int, int]] = set()
    all_edges = sorted(
        (
            float(np.linalg.norm(coords[index_a] - coords[index_b])),
            index_a,
            index_b,
        )
        for index_a, index_b in combinations(range(point_count), 2)
    )
    for distance, index_a, index_b in all_edges:
        cutoff = _pair_cutoff_distance(
            node_elements[index_a],
            node_elements[index_b],
            pair_cutoff_definitions=pair_cutoff_definitions,
        )
        if cutoff is None:
            continue
        if distance <= float(cutoff) * 1.15:
            explicit_edges.add((min(index_a, index_b), max(index_a, index_b)))
    if not explicit_edges:
        return _minimum_spanning_tree_edges(coords)

    parent = list(range(point_count))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(index_a: int, index_b: int) -> bool:
        root_a = find(index_a)
        root_b = find(index_b)
        if root_a == root_b:
            return False
        parent[root_b] = root_a
        return True

    for index_a, index_b in explicit_edges:
        union(index_a, index_b)
    for _distance, index_a, index_b in all_edges:
        if len({find(index) for index in range(point_count)}) == 1:
            break
        if union(index_a, index_b):
            explicit_edges.add((min(index_a, index_b), max(index_a, index_b)))
    return sorted(explicit_edges)


def _associate_non_node_atoms_to_nodes(
    coordinates: np.ndarray,
    *,
    elements: list[str],
    node_indices: list[int],
    pair_cutoff_definitions: PairCutoffDefinitions,
) -> dict[int, tuple[int, ...]]:
    coords = np.asarray(coordinates, dtype=float)
    if not node_indices:
        return {}
    node_index_set = set(node_indices)
    associations: dict[int, tuple[int, ...]] = {}
    for atom_index, element in enumerate(elements):
        if atom_index in node_index_set:
            continue
        distances = [
            (
                float(np.linalg.norm(coords[atom_index] - coords[node_index])),
                node_index,
            )
            for node_index in node_indices
        ]
        explicit_matches = [
            (distance, node_index)
            for distance, node_index in distances
            if (
                cutoff := _pair_cutoff_distance(
                    element,
                    elements[node_index],
                    pair_cutoff_definitions=pair_cutoff_definitions,
                )
            )
            is not None
            and distance <= float(cutoff) * 1.15
        ]
        if explicit_matches:
            associations[atom_index] = tuple(
                node_index
                for _distance, node_index in sorted(
                    explicit_matches,
                    key=lambda item: (item[0], item[1]),
                )
            )
            continue
        nearest_node = min(distances, key=lambda item: (item[0], item[1]))[1]
        associations[atom_index] = (nearest_node,)
    return associations


def _adjacency_from_edges(
    point_count: int,
    edges: list[tuple[int, int]],
) -> dict[int, set[int]]:
    adjacency = {index: set() for index in range(int(point_count))}
    for index_a, index_b in edges:
        adjacency.setdefault(index_a, set()).add(index_b)
        adjacency.setdefault(index_b, set()).add(index_a)
    return adjacency


def _unique_edges_from_adjacency(
    adjacency: dict[int, set[int]],
) -> list[tuple[int, int]]:
    return sorted(
        {
            (min(index_a, index_b), max(index_a, index_b))
            for index_a, neighbors in adjacency.items()
            for index_b in neighbors
        }
    )


def _next_pending_node_element(
    remaining_node_counts: Counter[str],
) -> str | None:
    for element in sorted(remaining_node_counts):
        if int(remaining_node_counts[element]) > 0:
            return element
    return None


def _node_bond_length_for_element(
    element: str,
    *,
    node_elements: tuple[str, ...],
    geometry_statistics: _TrainingGeometryStatistics,
    pair_cutoff_definitions: PairCutoffDefinitions,
) -> float:
    if element in node_elements:
        return float(geometry_statistics.node_bond_length)
    pair_distances = [
        geometry_statistics.bond_length_medians[
            _sorted_pair_key(element, node_element)
        ]
        for node_element in node_elements
        if _sorted_pair_key(element, node_element)
        in geometry_statistics.bond_length_medians
    ]
    if pair_distances:
        return float(np.median(np.asarray(pair_distances, dtype=float)))
    return float(
        _default_element_distance(
            element,
            pair_cutoff_definitions=pair_cutoff_definitions,
            node_elements=set(node_elements),
        )
    )


def _node_growth_position(
    node_positions: list[np.ndarray],
    adjacency: dict[int, set[int]],
    *,
    geometry_statistics: _TrainingGeometryStatistics,
) -> tuple[int, np.ndarray]:
    if not node_positions:
        return 0, np.zeros(3, dtype=float)
    coords = np.asarray(node_positions, dtype=float)
    if len(coords) == 1:
        bond_length = max(float(geometry_statistics.node_bond_length), 0.1)
        return 0, np.asarray(coords[0] + np.asarray([bond_length, 0.0, 0.0]))
    centroid = np.mean(coords, axis=0)
    desired_angle = float(
        geometry_statistics.node_angle_medians.get(("node", "node"), 180.0)
    )
    bond_length = max(float(geometry_statistics.node_bond_length), 0.1)
    anchor_indices = sorted(
        range(len(node_positions)),
        key=lambda index: (
            len(adjacency.get(index, set())),
            -float(np.linalg.norm(coords[index] - centroid)),
            index,
        ),
    )
    best_anchor = 0
    best_position = np.asarray(coords[0] + np.asarray([bond_length, 0.0, 0.0]))
    best_score: float | None = None
    for anchor_index in anchor_indices:
        anchor = coords[anchor_index]
        neighbor_indices = sorted(adjacency.get(anchor_index, set()))
        outward = anchor - centroid
        if neighbor_indices:
            outward = outward + np.sum(
                anchor - coords[neighbor_indices],
                axis=0,
            )
        candidate_directions = _direction_basis(outward)
        if len(neighbor_indices) == 1:
            inward = coords[neighbor_indices[0]] - anchor
            candidate_directions.extend(
                _cone_directions(
                    inward,
                    angle_degrees=desired_angle,
                    samples=12,
                )
            )
        unique_directions: list[np.ndarray] = []
        for direction in candidate_directions:
            unit_direction = _safe_unit_vector(direction, fallback=outward)
            if any(
                np.allclose(unit_direction, other)
                for other in unique_directions
            ):
                continue
            unique_directions.append(unit_direction)
        for direction in unique_directions:
            candidate = anchor + direction * bond_length
            anchor_distance = float(np.linalg.norm(candidate - anchor))
            collision_penalty = 0.0
            for existing_index, point in enumerate(coords):
                distance = float(np.linalg.norm(candidate - point))
                if existing_index == anchor_index:
                    collision_penalty += abs(distance - bond_length) * 5.0
                    continue
                if distance < bond_length * 0.75:
                    collision_penalty += (
                        (bond_length * 0.75) - distance
                    ) ** 2 * 500.0
                elif distance < bond_length * 0.95:
                    collision_penalty += (
                        (bond_length * 0.95) - distance
                    ) ** 2 * 60.0
            angle_penalty = 0.0
            for neighbor_index in neighbor_indices:
                angle = _angle_between_vectors(
                    candidate - anchor,
                    coords[neighbor_index] - anchor,
                )
                if angle is not None:
                    angle_penalty += abs(angle - desired_angle)
            outward_alignment = float(
                np.dot(
                    _safe_unit_vector(candidate - anchor),
                    _safe_unit_vector(outward, fallback=candidate - anchor),
                )
            )
            score = (
                collision_penalty
                + angle_penalty
                + abs(anchor_distance - bond_length) * 5.0
                - outward_alignment * 8.0
            )
            if best_score is None or score < best_score:
                best_score = float(score)
                best_anchor = int(anchor_index)
                best_position = np.asarray(candidate, dtype=float)
    return best_anchor, best_position


def _count_neighbor_type(
    neighbor_entries: list[tuple[str, np.ndarray]],
    geometry_type: str,
) -> int:
    return sum(
        1
        for entry_type, _vector in neighbor_entries
        if entry_type == geometry_type
    )


def _placement_sequence(
    target_counts: dict[str, int],
    *,
    node_elements: tuple[str, ...],
    geometry_statistics: _TrainingGeometryStatistics,
) -> list[str]:
    queued: list[tuple[float, int, str]] = []
    for element, count in sorted(target_counts.items()):
        if element in node_elements or int(count) <= 0:
            continue
        geometry_type = geometry_statistics.atom_type_by_element.get(
            element, "shell"
        )
        bridge_degree = float(
            geometry_statistics.non_node_node_coordination_medians.get(
                element, 1.0
            )
        )
        geometry_priority = 0 if geometry_type == "linker" else 1
        for _ in range(int(count)):
            queued.append((-bridge_degree, geometry_priority, element))
    queued.sort()
    return [element for _bridge_degree, _geometry_priority, element in queued]


def _select_bridge_edge(
    edges: list[tuple[int, int]],
    *,
    node_positions: list[np.ndarray],
    node_neighbor_entries: dict[int, list[tuple[str, np.ndarray]]],
    geometry_type: str,
    edge_assignments: Counter[tuple[int, int]],
    geometry_statistics: _TrainingGeometryStatistics,
) -> tuple[int, int]:
    coords = np.asarray(node_positions, dtype=float)
    centroid = (
        np.mean(coords, axis=0)
        if len(coords) > 0
        else np.zeros(3, dtype=float)
    )
    target_coordination = float(
        geometry_statistics.node_coordination_medians.get(geometry_type, 0.0)
    )
    return max(
        edges,
        key=lambda edge: (
            max(
                target_coordination
                - _count_neighbor_type(
                    node_neighbor_entries.get(edge[0], []), geometry_type
                ),
                0.0,
            )
            + max(
                target_coordination
                - _count_neighbor_type(
                    node_neighbor_entries.get(edge[1], []), geometry_type
                ),
                0.0,
            ),
            -float(edge_assignments[edge]),
            float(
                np.linalg.norm(
                    ((coords[edge[0]] + coords[edge[1]]) * 0.5) - centroid
                )
            ),
        ),
    )


def _bridge_atom_position(
    edge: tuple[int, int],
    *,
    node_positions: list[np.ndarray],
    existing_positions: list[np.ndarray],
    existing_elements: list[str],
    element: str,
    bond_length: float,
    geometry_statistics: _TrainingGeometryStatistics,
    pair_cutoff_definitions: PairCutoffDefinitions,
) -> np.ndarray:
    coords = np.asarray(node_positions, dtype=float)
    point_a = coords[edge[0]]
    point_b = coords[edge[1]]
    midpoint = (point_a + point_b) * 0.5
    edge_vector = point_b - point_a
    edge_length = float(np.linalg.norm(edge_vector))
    if edge_length <= 1e-12:
        return np.asarray(midpoint, dtype=float)
    desired_distance = max(float(bond_length), edge_length * 0.5)
    height = math.sqrt(
        max((desired_distance**2) - ((edge_length * 0.5) ** 2), 0.0)
    )
    if height <= 1e-8:
        return np.asarray(midpoint, dtype=float)
    centroid = (
        np.mean(np.asarray(node_positions, dtype=float), axis=0)
        if node_positions
        else np.zeros(3, dtype=float)
    )
    radial = midpoint - centroid
    edge_unit = _safe_unit_vector(edge_vector)
    radial_perpendicular = radial - np.dot(radial, edge_unit) * edge_unit
    basis_u, basis_v = _orthonormal_basis(edge_unit)
    preferred_normal = _safe_unit_vector(
        radial_perpendicular, fallback=basis_u
    )
    candidates = [
        midpoint + preferred_normal * height,
        midpoint - preferred_normal * height,
        midpoint + basis_u * height,
        midpoint - basis_u * height,
        midpoint + basis_v * height,
        midpoint - basis_v * height,
    ]
    existing = np.asarray(existing_positions, dtype=float)
    best_position = np.asarray(candidates[0], dtype=float)
    best_score: float | None = None
    for candidate in candidates:
        distances = [
            float(np.linalg.norm(candidate - position))
            for position in existing
            if float(np.linalg.norm(candidate - position)) > 1e-8
        ]
        minimum_distance = min(distances) if distances else math.inf
        interaction_penalty = _non_node_interaction_penalty(
            element,
            candidate,
            existing_elements=existing_elements,
            existing_positions=existing_positions,
            geometry_statistics=geometry_statistics,
            pair_cutoff_definitions=pair_cutoff_definitions,
        )
        node_distance_penalty = (
            abs(float(np.linalg.norm(candidate - point_a)) - bond_length) * 4.0
            + abs(float(np.linalg.norm(candidate - point_b)) - bond_length)
            * 4.0
        )
        score = (
            node_distance_penalty
            + interaction_penalty
            - minimum_distance * 3.0
        )
        if best_score is None or score < best_score:
            best_score = float(score)
            best_position = np.asarray(candidate, dtype=float)
    return best_position


def _select_attachment_node(
    *,
    node_positions: list[np.ndarray],
    node_neighbor_entries: dict[int, list[tuple[str, np.ndarray]]],
    geometry_type: str,
    geometry_statistics: _TrainingGeometryStatistics,
) -> int:
    coords = np.asarray(node_positions, dtype=float)
    centroid = (
        np.mean(coords, axis=0)
        if len(coords) > 0
        else np.zeros(3, dtype=float)
    )
    target_coordination = float(
        geometry_statistics.node_coordination_medians.get(geometry_type, 1.0)
    )
    return max(
        range(len(node_positions)),
        key=lambda index: (
            max(
                target_coordination
                - _count_neighbor_type(
                    node_neighbor_entries.get(index, []), geometry_type
                ),
                0.0,
            ),
            float(np.linalg.norm(coords[index] - centroid)),
            -len(
                [
                    1
                    for entry_type, _vector in node_neighbor_entries.get(
                        index, []
                    )
                    if entry_type != "node"
                ]
            ),
            -len(
                [
                    1
                    for entry_type, _vector in node_neighbor_entries.get(
                        index, []
                    )
                    if entry_type == "node"
                ]
            ),
        ),
    )


def _current_contact_counts(
    *,
    existing_elements: list[str],
    existing_positions: list[np.ndarray],
    geometry_statistics: _TrainingGeometryStatistics,
    pair_cutoff_definitions: PairCutoffDefinitions,
) -> list[Counter[str]]:
    counts = [Counter() for _ in existing_elements]
    if len(existing_elements) < 2:
        return counts
    positions = np.asarray(existing_positions, dtype=float)
    for atom_index_a, atom_index_b in combinations(
        range(len(existing_elements)), 2
    ):
        distance = float(
            np.linalg.norm(positions[atom_index_a] - positions[atom_index_b])
        )
        if distance > _contact_distance_cutoff(
            existing_elements[atom_index_a],
            existing_elements[atom_index_b],
            geometry_statistics=geometry_statistics,
            default_distance=geometry_statistics.node_bond_length,
            pair_cutoff_definitions=pair_cutoff_definitions,
        ):
            continue
        geometry_type_a = geometry_statistics.atom_type_by_element.get(
            existing_elements[atom_index_a],
            "shell",
        )
        geometry_type_b = geometry_statistics.atom_type_by_element.get(
            existing_elements[atom_index_b],
            "shell",
        )
        counts[atom_index_a][geometry_type_b] += 1
        counts[atom_index_b][geometry_type_a] += 1
    return counts


def _non_node_interaction_penalty(
    element: str,
    candidate_position: np.ndarray,
    *,
    existing_elements: list[str],
    existing_positions: list[np.ndarray],
    geometry_statistics: _TrainingGeometryStatistics,
    pair_cutoff_definitions: PairCutoffDefinitions,
) -> float:
    geometry_type = geometry_statistics.atom_type_by_element.get(
        element, "shell"
    )
    tracked_non_node_types = [
        atom_type
        for atom_type in geometry_statistics.tracked_atom_types
        if atom_type != "node"
    ]
    if not tracked_non_node_types or not existing_elements:
        return 0.0
    current_contact_counts = _current_contact_counts(
        existing_elements=existing_elements,
        existing_positions=existing_positions,
        geometry_statistics=geometry_statistics,
        pair_cutoff_definitions=pair_cutoff_definitions,
    )
    candidate_counts: Counter[str] = Counter()
    candidate_distance_errors: defaultdict[str, list[float]] = defaultdict(
        list
    )
    reciprocity_bonus = 0.0
    penalty = 0.0
    positions = np.asarray(existing_positions, dtype=float)
    for atom_index, existing_element in enumerate(existing_elements):
        existing_type = geometry_statistics.atom_type_by_element.get(
            existing_element, "shell"
        )
        if existing_type == "node":
            continue
        target_distance = _pair_contact_distance(
            element,
            existing_element,
            geometry_statistics=geometry_statistics,
            pair_cutoff_definitions=pair_cutoff_definitions,
        )
        distance = float(
            np.linalg.norm(candidate_position - positions[atom_index])
        )
        minimum_distance = max(target_distance * 0.65, 0.45)
        if distance < minimum_distance:
            penalty += ((minimum_distance - distance) ** 2) * 220.0
        cutoff = _contact_distance_cutoff(
            element,
            existing_element,
            geometry_statistics=geometry_statistics,
            default_distance=geometry_statistics.node_bond_length,
            pair_cutoff_definitions=pair_cutoff_definitions,
        )
        if distance > cutoff:
            continue
        candidate_counts[existing_type] += 1
        candidate_distance_errors[existing_type].append(
            abs(distance - target_distance)
        )
        desired_existing = int(
            round(
                geometry_statistics.atom_coordination_medians.get(
                    (existing_type, geometry_type),
                    0.0,
                )
            )
        )
        if desired_existing <= 0:
            continue
        current_count = int(
            current_contact_counts[atom_index].get(geometry_type, 0)
        )
        reciprocity_bonus += (
            float(max(desired_existing - current_count, 0)) * 6.0
        )
        if current_count >= desired_existing + 1:
            penalty += float(current_count - desired_existing) * 6.0
    for neighbor_type in tracked_non_node_types:
        available = sum(
            1
            for existing_element in existing_elements
            if geometry_statistics.atom_type_by_element.get(
                existing_element, "shell"
            )
            == neighbor_type
        )
        desired_count = int(
            round(
                geometry_statistics.atom_coordination_medians.get(
                    (geometry_type, neighbor_type),
                    0.0,
                )
            )
        )
        if available <= 0 and desired_count <= 0:
            continue
        target_count = min(max(desired_count, 0), available)
        observed_count = int(candidate_counts.get(neighbor_type, 0))
        if target_count > observed_count:
            penalty += float(target_count - observed_count) * 12.0
        elif observed_count > max(target_count, 0):
            penalty += float(observed_count - max(target_count, 0)) * 10.0
        if target_count > 0 and candidate_distance_errors.get(neighbor_type):
            penalty += float(
                sum(
                    sorted(candidate_distance_errors[neighbor_type])[
                        :target_count
                    ]
                )
                * 8.0
            )
        elif observed_count > 0 and candidate_distance_errors.get(
            neighbor_type
        ):
            penalty += (
                float(min(candidate_distance_errors[neighbor_type])) * 4.0
            )
    return float(penalty - reciprocity_bonus)


def _terminal_atom_position(
    anchor_index: int,
    *,
    node_positions: list[np.ndarray],
    node_neighbor_entries: dict[int, list[tuple[str, np.ndarray]]],
    existing_positions: list[np.ndarray],
    existing_elements: list[str],
    element: str,
    geometry_type: str,
    bond_length: float,
    geometry_statistics: _TrainingGeometryStatistics,
    pair_cutoff_definitions: PairCutoffDefinitions,
) -> np.ndarray:
    coords = np.asarray(node_positions, dtype=float)
    anchor = coords[anchor_index]
    centroid = (
        np.mean(coords, axis=0)
        if len(coords) > 0
        else np.zeros(3, dtype=float)
    )
    outward = anchor - centroid
    for neighbor_type, vector in node_neighbor_entries.get(anchor_index, []):
        if neighbor_type == "node":
            outward = outward + (anchor - (anchor + vector))
    candidate_directions = _direction_basis(outward)
    desired_node_angle = float(
        geometry_statistics.node_angle_medians.get(
            _sorted_pair_key("node", geometry_type),
            120.0,
        )
    )
    for neighbor_type, vector in node_neighbor_entries.get(anchor_index, []):
        neighbor_angle = float(
            geometry_statistics.node_angle_medians.get(
                _sorted_pair_key(geometry_type, neighbor_type),
                desired_node_angle if neighbor_type == "node" else 109.5,
            )
        )
        candidate_directions.extend(
            _cone_directions(
                vector,
                angle_degrees=neighbor_angle,
                samples=10,
            )
        )
    unique_directions: list[np.ndarray] = []
    for direction in candidate_directions:
        unit_direction = _safe_unit_vector(direction, fallback=outward)
        if any(
            np.allclose(unit_direction, other) for other in unique_directions
        ):
            continue
        unique_directions.append(unit_direction)
    existing = np.asarray(existing_positions, dtype=float)
    best_position = np.asarray(
        anchor + unique_directions[0] * bond_length, dtype=float
    )
    best_score: float | None = None
    for direction in unique_directions:
        candidate = anchor + direction * float(bond_length)
        collision_penalty = 0.0
        for position in existing:
            distance = float(np.linalg.norm(candidate - position))
            if distance <= 1e-8:
                continue
            if distance < bond_length * 0.7:
                collision_penalty += (
                    (bond_length * 0.7) - distance
                ) ** 2 * 500.0
            elif distance < bond_length * 0.95:
                collision_penalty += (
                    (bond_length * 0.95) - distance
                ) ** 2 * 40.0
        angle_penalty = 0.0
        for neighbor_type, vector in node_neighbor_entries.get(
            anchor_index, []
        ):
            desired_angle = float(
                geometry_statistics.node_angle_medians.get(
                    _sorted_pair_key(geometry_type, neighbor_type),
                    109.5 if geometry_type == neighbor_type else 120.0,
                )
            )
            angle = _angle_between_vectors(direction, vector)
            if angle is not None:
                angle_penalty += abs(angle - desired_angle)
        outward_alignment = float(
            np.dot(
                _safe_unit_vector(direction),
                _safe_unit_vector(outward, fallback=direction),
            )
        )
        interaction_penalty = _non_node_interaction_penalty(
            element,
            candidate,
            existing_elements=existing_elements,
            existing_positions=existing_positions,
            geometry_statistics=geometry_statistics,
            pair_cutoff_definitions=pair_cutoff_definitions,
        )
        score = (
            collision_penalty
            + angle_penalty
            + interaction_penalty
            - outward_alignment * 8.0
        )
        if best_score is None or score < best_score:
            best_score = float(score)
            best_position = np.asarray(candidate, dtype=float)
    return best_position


def _build_geometry_guided_structure(
    target_counts: dict[str, int],
    *,
    node_elements: tuple[str, ...],
    pair_cutoff_definitions: PairCutoffDefinitions,
    geometry_statistics: _TrainingGeometryStatistics,
    predicted_max_radius: float,
    seed_node_elements: list[str],
    seed_node_coordinates: np.ndarray,
) -> tuple[list[str], np.ndarray]:
    normalized_target_counts = _normalized_counts(target_counts)
    target_node_total = int(
        sum(
            normalized_target_counts.get(element, 0)
            for element in node_elements
        )
    )
    if target_node_total <= 0:
        return _build_fallback_structure(normalized_target_counts)

    remaining_node_counts: Counter[str] = Counter(
        {
            element: int(normalized_target_counts.get(element, 0))
            for element in node_elements
        }
    )
    seed_coords = np.asarray(seed_node_coordinates, dtype=float)
    kept_node_elements: list[str] = []
    kept_node_positions: list[np.ndarray] = []
    if seed_coords.ndim == 2 and len(seed_coords) == len(seed_node_elements):
        for element, coordinate in zip(
            seed_node_elements, seed_coords, strict=False
        ):
            if len(kept_node_elements) >= target_node_total:
                break
            if int(remaining_node_counts.get(element, 0)) <= 0:
                continue
            kept_node_elements.append(str(element))
            kept_node_positions.append(np.asarray(coordinate, dtype=float))
            remaining_node_counts[element] -= 1
    if not kept_node_positions:
        first_element = _next_pending_node_element(remaining_node_counts)
        if first_element is None:
            return _build_fallback_structure(normalized_target_counts)
        kept_node_elements.append(first_element)
        kept_node_positions.append(np.zeros(3, dtype=float))
        remaining_node_counts[first_element] -= 1

    adjacency = _adjacency_from_edges(
        len(kept_node_positions),
        _node_scaffold_edges(
            np.asarray(kept_node_positions, dtype=float),
            kept_node_elements,
            pair_cutoff_definitions=pair_cutoff_definitions,
        ),
    )

    while sum(int(value) for value in remaining_node_counts.values()) > 0:
        next_element = _next_pending_node_element(remaining_node_counts)
        if next_element is None:
            break
        anchor_index, new_position = _node_growth_position(
            kept_node_positions,
            adjacency,
            geometry_statistics=geometry_statistics,
        )
        new_index = len(kept_node_positions)
        kept_node_elements.append(next_element)
        kept_node_positions.append(np.asarray(new_position, dtype=float))
        adjacency.setdefault(anchor_index, set()).add(new_index)
        adjacency.setdefault(new_index, set()).add(anchor_index)
        remaining_node_counts[next_element] -= 1

    all_elements = list(kept_node_elements)
    all_positions = [
        np.asarray(position, dtype=float) for position in kept_node_positions
    ]
    node_neighbor_entries = {
        index: [
            (
                "node",
                kept_node_positions[neighbor] - kept_node_positions[index],
            )
            for neighbor in sorted(adjacency.get(index, set()))
        ]
        for index in range(len(kept_node_positions))
    }
    edge_assignments: Counter[tuple[int, int]] = Counter()
    scaffold_edges = _unique_edges_from_adjacency(adjacency)
    for element in _placement_sequence(
        normalized_target_counts,
        node_elements=node_elements,
        geometry_statistics=geometry_statistics,
    ):
        geometry_type = geometry_statistics.atom_type_by_element.get(
            element, "shell"
        )
        bridge_degree = int(
            max(
                1,
                min(
                    round(
                        geometry_statistics.non_node_node_coordination_medians.get(
                            element,
                            1.0,
                        )
                    ),
                    2,
                ),
            )
        )
        bond_length = max(
            _node_bond_length_for_element(
                element,
                node_elements=node_elements,
                geometry_statistics=geometry_statistics,
                pair_cutoff_definitions=pair_cutoff_definitions,
            ),
            0.1,
        )
        if bridge_degree >= 2 and scaffold_edges:
            selected_edge = _select_bridge_edge(
                scaffold_edges,
                node_positions=kept_node_positions,
                node_neighbor_entries=node_neighbor_entries,
                geometry_type=geometry_type,
                edge_assignments=edge_assignments,
                geometry_statistics=geometry_statistics,
            )
            placed_position = _bridge_atom_position(
                selected_edge,
                node_positions=kept_node_positions,
                existing_positions=all_positions,
                existing_elements=all_elements,
                element=element,
                bond_length=bond_length,
                geometry_statistics=geometry_statistics,
                pair_cutoff_definitions=pair_cutoff_definitions,
            )
            attached_nodes = selected_edge
            edge_assignments[selected_edge] += 1
        else:
            anchor_index = _select_attachment_node(
                node_positions=kept_node_positions,
                node_neighbor_entries=node_neighbor_entries,
                geometry_type=geometry_type,
                geometry_statistics=geometry_statistics,
            )
            placed_position = _terminal_atom_position(
                anchor_index,
                node_positions=kept_node_positions,
                node_neighbor_entries=node_neighbor_entries,
                existing_positions=all_positions,
                existing_elements=all_elements,
                element=element,
                geometry_type=geometry_type,
                bond_length=bond_length,
                geometry_statistics=geometry_statistics,
                pair_cutoff_definitions=pair_cutoff_definitions,
            )
            attached_nodes = (anchor_index,)
        all_elements.append(element)
        all_positions.append(np.asarray(placed_position, dtype=float))
        for node_index in attached_nodes:
            node_neighbor_entries.setdefault(node_index, []).append(
                (
                    geometry_type,
                    placed_position - kept_node_positions[node_index],
                )
            )

    coordinates = np.asarray(all_positions, dtype=float)
    coordinates = _scale_structure_to_radius(
        coordinates,
        target_max_radius=max(predicted_max_radius, 0.1),
    )
    return all_elements, coordinates


def _principal_axis(coordinates: np.ndarray) -> np.ndarray:
    coords = np.asarray(coordinates, dtype=float)
    if coords.size == 0 or len(coords) == 1:
        return np.asarray([1.0, 0.0, 0.0], dtype=float)
    centered = coords - np.mean(coords, axis=0, keepdims=True)
    covariance = np.cov(centered.T, bias=True)
    eigvals, eigvecs = np.linalg.eigh(covariance)
    axis = np.asarray(eigvecs[:, np.argmax(eigvals)], dtype=float)
    norm = np.linalg.norm(axis)
    if norm <= 0.0:
        return np.asarray([1.0, 0.0, 0.0], dtype=float)
    return axis / norm


def _estimate_node_spacing(
    coordinates: np.ndarray,
    *,
    elements: list[str],
    node_mask: np.ndarray,
    pair_cutoff_definitions: PairCutoffDefinitions,
    node_elements: set[str],
) -> float:
    del elements
    node_coords = np.asarray(coordinates, dtype=float)[
        np.asarray(node_mask, dtype=bool)
    ]
    if len(node_coords) >= 2:
        projections = node_coords @ _principal_axis(node_coords)
        ordered = np.sort(projections)
        diffs = np.diff(ordered)
        positive = diffs[diffs > 1e-6]
        if positive.size > 0:
            return float(np.median(positive))
    for element_a, element_b in pair_cutoff_definitions:
        if element_a in node_elements and element_b in node_elements:
            cutoffs = pair_cutoff_definitions[(element_a, element_b)]
            if cutoffs:
                return float(np.median(list(cutoffs.values())))
    return 3.0


def _terminal_anchor(
    coordinates: np.ndarray,
    *,
    node_mask: np.ndarray,
    axis: np.ndarray,
) -> np.ndarray:
    coords = np.asarray(coordinates, dtype=float)
    mask = np.asarray(node_mask, dtype=bool)
    if np.any(mask):
        anchors = coords[mask]
    else:
        anchors = coords
    if anchors.size == 0:
        return np.zeros(3, dtype=float)
    projections = anchors @ np.asarray(axis, dtype=float)
    return np.asarray(anchors[int(np.argmax(projections))], dtype=float)


def _remove_excess_atoms(
    coordinates: np.ndarray,
    elements: list[str],
    *,
    source_counts: Counter[str],
    target_counts: dict[str, int],
) -> tuple[np.ndarray, list[str]]:
    coords = np.asarray(coordinates, dtype=float)
    elem_list = list(elements)
    if coords.size == 0:
        return coords, elem_list
    centered = coords - np.mean(coords, axis=0, keepdims=True)
    projections = centered @ _principal_axis(centered)
    kept = np.ones(len(elem_list), dtype=bool)
    for element, current_count in source_counts.items():
        target_count = int(target_counts.get(element, 0))
        excess = int(current_count - target_count)
        if excess <= 0:
            continue
        indices = [
            index for index, value in enumerate(elem_list) if value == element
        ]
        ranked = sorted(
            indices, key=lambda index: float(projections[index]), reverse=True
        )
        for index in ranked[:excess]:
            kept[index] = False
    return coords[kept], [
        element for element, keep in zip(elem_list, kept) if keep
    ]


def _element_template_vectors(
    coordinates: np.ndarray,
    *,
    elements: list[str],
    element: str,
    node_mask: np.ndarray,
    axis: np.ndarray,
    pair_cutoff_definitions: PairCutoffDefinitions,
    node_elements: set[str],
) -> list[np.ndarray]:
    coords = np.asarray(coordinates, dtype=float)
    nodes = coords[np.asarray(node_mask, dtype=bool)]
    if nodes.size == 0:
        return []
    source_vectors: list[np.ndarray] = []
    default_distance = _default_element_distance(
        element,
        pair_cutoff_definitions=pair_cutoff_definitions,
        node_elements=node_elements,
    )
    for index, source_element in enumerate(elements):
        if source_element != element:
            continue
        atom = coords[index]
        deltas = nodes - atom
        distances = np.linalg.norm(deltas, axis=1)
        if distances.size == 0:
            continue
        nearest_node = nodes[int(np.argmin(distances))]
        vector = atom - nearest_node
        norm = np.linalg.norm(vector)
        if norm <= 1e-9:
            continue
        normalized = vector / norm
        if float(normalized @ axis) < 0.0:
            normalized = normalized * -1.0
        source_vectors.append(normalized * default_distance)
    unique_vectors: list[np.ndarray] = []
    for vector in source_vectors:
        if not any(np.allclose(vector, other) for other in unique_vectors):
            unique_vectors.append(vector)
    return unique_vectors


def _generic_element_vectors(
    *,
    count: int,
    distance: float,
) -> list[np.ndarray]:
    base = [
        np.asarray([0.0, 1.0, 0.0], dtype=float),
        np.asarray([0.0, -1.0, 0.0], dtype=float),
        np.asarray([0.0, 0.0, 1.0], dtype=float),
        np.asarray([0.0, 0.0, -1.0], dtype=float),
        np.asarray([0.0, 1.0, 1.0], dtype=float),
        np.asarray([0.0, -1.0, 1.0], dtype=float),
        np.asarray([0.0, 1.0, -1.0], dtype=float),
        np.asarray([0.0, -1.0, -1.0], dtype=float),
    ]
    vectors: list[np.ndarray] = []
    for index in range(max(count, 1)):
        direction = np.asarray(base[index % len(base)], dtype=float)
        direction_norm = np.linalg.norm(direction)
        if direction_norm <= 0.0:
            direction = np.asarray([0.0, 1.0, 0.0], dtype=float)
        else:
            direction = direction / direction_norm
        vectors.append(direction * float(distance))
    return vectors


def _default_element_distance(
    element: str,
    *,
    pair_cutoff_definitions: PairCutoffDefinitions,
    node_elements: set[str],
) -> float:
    distances: list[float] = []
    normalized_element = _normalized_element_symbol(element)
    for (atom1, atom2), level_map in pair_cutoff_definitions.items():
        if atom1 in node_elements and atom2 == normalized_element:
            distances.extend(float(value) for value in level_map.values())
        if atom2 in node_elements and atom1 == normalized_element:
            distances.extend(float(value) for value in level_map.values())
    if distances:
        return float(np.median(distances))
    return 2.5


def _scale_structure_to_radius(
    coordinates: np.ndarray,
    *,
    target_max_radius: float,
) -> np.ndarray:
    del target_max_radius
    return np.asarray(coordinates, dtype=float)


def _build_fallback_structure(
    target_counts: dict[str, int],
) -> tuple[list[str], np.ndarray]:
    elements: list[str] = []
    coords: list[np.ndarray] = []
    index = 0
    for element, count in sorted(target_counts.items()):
        for repeat in range(int(count)):
            shell = max(index, 1)
            angle = float(repeat) * (math.pi / 3.0)
            radius = 1.5 + 0.4 * shell
            coords.append(
                np.asarray(
                    [
                        float(shell),
                        radius * math.cos(angle),
                        radius * math.sin(angle),
                    ],
                    dtype=float,
                )
            )
            elements.append(element)
            index += 1
    return elements, np.asarray(coords, dtype=float)
