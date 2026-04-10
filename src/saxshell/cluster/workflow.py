from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from re import sub

from saxshell.structure import AtomTypeDefinitions

from .clusternetwork import (
    DEFAULT_SAVE_STATE_FREQUENCY,
    SEARCH_MODE_KDTREE,
    ExtractedFrameFolderClusterAnalyzer,
    PairCutoffDefinitions,
    format_search_mode_label,
    normalize_save_state_frequency,
    normalize_search_mode,
)

EXAMPLE_CLUSTER_ATOM_TYPE_DEFINITIONS: AtomTypeDefinitions = {
    "node": [("Pb", None)],
    "linker": [("I", None)],
    "shell": [("O", None)],
}
EXAMPLE_CLUSTER_PAIR_CUTOFFS: PairCutoffDefinitions = {
    ("Pb", "I"): {0: 3.36},
    ("Pb", "O"): {0: 3.36},
}


def example_atom_type_definitions() -> AtomTypeDefinitions:
    """Return a copy of the example atom-type definitions."""
    return {
        atom_type: list(criteria)
        for (
            atom_type,
            criteria,
        ) in EXAMPLE_CLUSTER_ATOM_TYPE_DEFINITIONS.items()
    }


def example_pair_cutoff_definitions() -> PairCutoffDefinitions:
    """Return a copy of the example pair-cutoff definitions."""
    return {
        pair: dict(levels)
        for pair, levels in EXAMPLE_CLUSTER_PAIR_CUTOFFS.items()
    }


def suggest_output_dir(frames_dir: str | Path) -> Path:
    """Suggest a new cluster output directory beside the frames
    folder."""
    source_path = Path(frames_dir)
    folder_name = _base_output_dir_name(source_path)
    return next_available_output_dir(source_path.parent, folder_name)


def next_available_output_dir(parent_dir: Path, folder_name: str) -> Path:
    """Return the next available directory path for a base folder
    name."""
    candidate = parent_dir / folder_name
    if not candidate.exists():
        return candidate

    index = 1
    while True:
        candidate = parent_dir / f"{folder_name}{index:04d}"
        if not candidate.exists():
            return candidate
        index += 1


def format_box_dimensions(
    box_dimensions: tuple[float, float, float] | None,
) -> str:
    """Format box lengths for CLI and preview output."""
    if box_dimensions is None:
        return "None"
    return " x ".join(f"{value:.3f}" for value in box_dimensions) + " A"


def _base_output_dir_name(frames_dir: Path) -> str:
    folder_label = sub(r"[^0-9A-Za-z]+", "_", frames_dir.name).strip("_")
    if not folder_label:
        folder_label = "frames"
    return f"clusters_{folder_label}"


def _coerce_box_dimensions(
    value: object | None,
) -> tuple[float, float, float] | None:
    if value is None:
        return None
    box = tuple(float(component) for component in value)
    if len(box) != 3:
        raise ValueError("Box dimensions must contain exactly three values.")
    return box


def _summary_box_dimensions(
    summary: dict[str, object],
) -> tuple[float, float, float] | None:
    value = summary.get("box_dimensions")
    if value is None:
        value = summary.get("estimated_box_dimensions")
    return _coerce_box_dimensions(value)


@dataclass(slots=True)
class ClusterSelectionResult:
    """Preview of the current cluster-analysis selection."""

    summary: dict[str, object]
    output_dir: Path
    resolved_box_dimensions: tuple[float, float, float] | None
    using_auto_box: bool
    use_pbc: bool
    smart_solvation_shells: bool
    include_shell_atoms_in_stoichiometry: bool
    search_mode: str
    save_state_frequency: int

    @property
    def frame_format(self) -> str:
        return str(self.summary["frame_format"])

    @property
    def mode_label(self) -> str:
        return str(self.summary["mode_label"])

    @property
    def output_file_extension(self) -> str:
        return str(self.summary["output_file_extension"])

    @property
    def estimated_box_dimensions(self) -> tuple[float, float, float] | None:
        return _summary_box_dimensions(self.summary)

    @property
    def box_dimensions_source(self) -> str | None:
        value = self.summary.get("box_dimensions_source")
        return None if value is None else str(value)

    @property
    def box_dimensions_source_kind(self) -> str | None:
        value = self.summary.get("box_dimensions_source_kind")
        return None if value is None else str(value)

    @property
    def total_frames(self) -> int:
        return int(self.summary["n_frames"])

    @property
    def search_mode_label(self) -> str:
        return format_search_mode_label(self.search_mode)

    def to_dict(self) -> dict[str, object]:
        return {
            "input_dir": str(self.summary["input_dir"]),
            "frame_format": self.frame_format,
            "mode_label": self.mode_label,
            "output_file_extension": self.output_file_extension,
            "output_dir": str(self.output_dir),
            "total_frames": self.total_frames,
            "first_frame": self.summary.get("first_frame"),
            "last_frame": self.summary.get("last_frame"),
            "estimated_box_dimensions": self.estimated_box_dimensions,
            "box_dimensions_source": self.box_dimensions_source,
            "box_dimensions_source_kind": self.box_dimensions_source_kind,
            "resolved_box_dimensions": self.resolved_box_dimensions,
            "using_auto_box": self.using_auto_box,
            "use_pbc": self.use_pbc,
            "smart_solvation_shells": self.smart_solvation_shells,
            "include_shell_atoms_in_stoichiometry": (
                self.include_shell_atoms_in_stoichiometry
            ),
            "search_mode": self.search_mode,
            "search_mode_label": self.search_mode_label,
            "save_state_frequency": self.save_state_frequency,
            "supports_full_molecule_shells": bool(
                self.summary.get("supports_full_molecule_shells", False)
            ),
        }


@dataclass(slots=True)
class ClusterExportResult:
    """Result of writing cluster files to disk."""

    output_dir: Path
    written_files: list[Path]
    analyzed_frames: int
    total_clusters: int
    selection: ClusterSelectionResult
    metadata_path: Path | None = None
    resumed: bool = False
    already_complete: bool = False
    previously_completed_frames: int = 0
    newly_processed_frames: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "output_dir": str(self.output_dir),
            "written_files": [str(path) for path in self.written_files],
            "written_count": len(self.written_files),
            "analyzed_frames": self.analyzed_frames,
            "total_clusters": self.total_clusters,
            "metadata_path": (
                None if self.metadata_path is None else str(self.metadata_path)
            ),
            "resumed": self.resumed,
            "already_complete": self.already_complete,
            "previously_completed_frames": self.previously_completed_frames,
            "newly_processed_frames": self.newly_processed_frames,
            "selection": self.selection.to_dict(),
        }


class ClusterWorkflow:
    """Headless workflow for cluster analysis from extracted frame
    folders."""

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
        include_shell_levels: tuple[int, ...] = (0,),
        shared_shells: bool = False,
        smart_solvation_shells: bool = True,
        include_shell_atoms_in_stoichiometry: bool = False,
        search_mode: str = SEARCH_MODE_KDTREE,
        save_state_frequency: int = DEFAULT_SAVE_STATE_FREQUENCY,
    ) -> None:
        self.frames_dir = Path(frames_dir)
        self.atom_type_definitions = atom_type_definitions
        self.pair_cutoff_definitions = pair_cutoff_definitions
        self.box_dimensions = box_dimensions
        self.use_pbc = bool(use_pbc)
        self.default_cutoff = default_cutoff
        self.shell_levels = shell_levels
        self.include_shell_levels = include_shell_levels
        self.shared_shells = shared_shells
        self.smart_solvation_shells = bool(smart_solvation_shells)
        self.include_shell_atoms_in_stoichiometry = bool(
            include_shell_atoms_in_stoichiometry
        )
        self.search_mode = normalize_search_mode(search_mode)
        self.save_state_frequency = normalize_save_state_frequency(
            save_state_frequency
        )
        self.summary: dict[str, object] | None = None

    def inspect(self) -> dict[str, object]:
        """Inspect the extracted frames folder metadata."""
        if self.summary is None:
            self.summary = self._build_analyzer().inspect()
        return dict(self.summary)

    def resolve_box_dimensions(
        self,
        *,
        box_dimensions: tuple[float, float, float] | None = None,
        use_pbc: bool | None = None,
    ) -> tuple[float, float, float] | None:
        """Resolve the box dimensions that should be used for
        analysis."""
        resolved_use_pbc = self.use_pbc if use_pbc is None else bool(use_pbc)
        if not resolved_use_pbc:
            return None

        resolved_box = (
            box_dimensions
            if box_dimensions is not None
            else self.box_dimensions
        )
        if resolved_box is not None:
            return resolved_box

        summary = self.inspect()
        estimated_box = _summary_box_dimensions(summary)
        if estimated_box is None:
            raise ValueError(
                "Periodic boundary conditions are enabled, but no box "
                "dimensions are available. Provide --box or use a frames "
                "folder with a usable coordinate extent."
            )
        return estimated_box

    def preview_selection(
        self,
        *,
        output_dir: str | Path | None = None,
        box_dimensions: tuple[float, float, float] | None = None,
        use_pbc: bool | None = None,
    ) -> ClusterSelectionResult:
        """Preview the selected frames, output directory, and box
        settings."""
        summary = self.inspect()
        smart_solvation_shells = (
            str(summary.get("frame_format", "")).lower() == "pdb"
            and self.smart_solvation_shells
        )
        resolved_use_pbc = self.use_pbc if use_pbc is None else bool(use_pbc)
        resolved_box = self.resolve_box_dimensions(
            box_dimensions=box_dimensions,
            use_pbc=resolved_use_pbc,
        )
        resolved_output_dir = (
            Path(output_dir)
            if output_dir is not None
            else suggest_output_dir(self.frames_dir)
        )
        return ClusterSelectionResult(
            summary=summary,
            output_dir=resolved_output_dir,
            resolved_box_dimensions=resolved_box,
            using_auto_box=(
                resolved_use_pbc
                and box_dimensions is None
                and self.box_dimensions is None
            ),
            use_pbc=resolved_use_pbc,
            smart_solvation_shells=smart_solvation_shells,
            include_shell_atoms_in_stoichiometry=(
                self.include_shell_atoms_in_stoichiometry
            ),
            search_mode=self.search_mode,
            save_state_frequency=self.save_state_frequency,
        )

    def export_clusters(
        self,
        *,
        output_dir: str | Path | None = None,
        box_dimensions: tuple[float, float, float] | None = None,
        use_pbc: bool | None = None,
        shell_levels: tuple[int, ...] | None = None,
        include_shell_levels: tuple[int, ...] | None = None,
        shared_shells: bool | None = None,
        smart_solvation_shells: bool | None = None,
    ) -> ClusterExportResult:
        """Analyze the extracted frames and write cluster files."""
        selection = self.preview_selection(
            output_dir=output_dir,
            box_dimensions=box_dimensions,
            use_pbc=use_pbc,
        )
        if selection.total_frames == 0:
            raise ValueError("No extracted frames were found to analyze.")

        analyzer = self._build_analyzer(
            box_dimensions=selection.resolved_box_dimensions,
            use_pbc=(self.use_pbc if use_pbc is None else bool(use_pbc)),
            smart_solvation_shells=(
                self.smart_solvation_shells
                if smart_solvation_shells is None
                else bool(smart_solvation_shells)
            ),
        )
        export = analyzer.export_cluster_files(
            selection.output_dir,
            shell_levels=(
                self.shell_levels if shell_levels is None else shell_levels
            ),
            include_shell_levels=(
                self.include_shell_levels
                if include_shell_levels is None
                else include_shell_levels
            ),
            shared_shells=(
                self.shared_shells
                if shared_shells is None
                else bool(shared_shells)
            ),
        )
        total_clusters = sum(
            frame_result.n_clusters for frame_result in export.frame_results
        )
        return ClusterExportResult(
            output_dir=selection.output_dir,
            written_files=export.written_files,
            analyzed_frames=len(export.frame_results),
            total_clusters=total_clusters,
            selection=selection,
            metadata_path=export.metadata_path,
            resumed=export.resumed,
            already_complete=export.already_complete,
            previously_completed_frames=export.previously_completed_frames,
            newly_processed_frames=export.newly_processed_frames,
        )

    def _build_analyzer(
        self,
        *,
        box_dimensions: tuple[float, float, float] | None = None,
        use_pbc: bool | None = None,
        smart_solvation_shells: bool | None = None,
    ) -> ExtractedFrameFolderClusterAnalyzer:
        return ExtractedFrameFolderClusterAnalyzer(
            frames_dir=self.frames_dir,
            atom_type_definitions=self.atom_type_definitions,
            pair_cutoffs_def=self.pair_cutoff_definitions,
            box_dimensions=box_dimensions,
            default_cutoff=self.default_cutoff,
            use_pbc=self.use_pbc if use_pbc is None else bool(use_pbc),
            smart_solvation_shells=(
                self.smart_solvation_shells
                if smart_solvation_shells is None
                else bool(smart_solvation_shells)
            ),
            include_shell_atoms_in_stoichiometry=(
                self.include_shell_atoms_in_stoichiometry
            ),
            search_mode=self.search_mode,
            save_state_frequency=self.save_state_frequency,
        )
