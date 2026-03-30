from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from re import sub

from saxshell.mdtrajectory.frame.cp2k_ener import CP2KEnergyData
from saxshell.mdtrajectory.frame.cutoff_analysis import (
    CP2KEnergyAnalyzer,
    SteadyStateResult,
)
from saxshell.mdtrajectory.frame.manager import (
    FrameSelectionPreview,
    TrajectoryManager,
)

EXPORT_METADATA_FILENAME = "mdtrajectory_export.json"


def suggest_output_dir(
    trajectory_file: str | Path,
    *,
    cutoff_fs: float | None = None,
) -> Path:
    """Suggest a new export directory beside the source trajectory."""
    source_path = Path(trajectory_file)
    parent_dir = source_path.parent
    folder_name = _base_output_dir_name(source_path, cutoff_fs=cutoff_fs)
    return next_available_output_dir(parent_dir, folder_name)


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


def format_cutoff_for_dir(cutoff_fs: float) -> str:
    """Format a cutoff value so it is safe to embed in a directory
    name."""
    rounded = f"{cutoff_fs:.3f}".rstrip("0").rstrip(".")
    return sub(r"[^0-9A-Za-z]+", "_", rounded)


def _base_output_dir_name(
    trajectory_file: Path,
    *,
    cutoff_fs: float | None = None,
) -> str:
    suffix = trajectory_file.suffix.lower()
    if suffix == ".xyz":
        base_name = "splitxyz"
    elif suffix == ".pdb":
        base_name = "splitpdb"
    else:
        base_name = "splitframes"

    if cutoff_fs is None:
        return base_name

    cutoff_text = format_cutoff_for_dir(cutoff_fs)
    return f"{base_name}_f{cutoff_text}fs"


@dataclass(slots=True)
class MDTrajectorySelectionResult:
    """Preview of the current selection and target output directory."""

    preview: FrameSelectionPreview
    output_dir: Path
    applied_cutoff_fs: float | None

    def to_dict(self) -> dict[str, object]:
        preview = self.preview
        return {
            "output_dir": str(self.output_dir),
            "applied_cutoff_fs": self.applied_cutoff_fs,
            "total_frames": preview.total_frames,
            "selected_frames": preview.selected_frames,
            "start": preview.start,
            "stop": preview.stop,
            "stride": preview.stride,
            "post_cutoff_stride": preview.post_cutoff_stride,
            "time_metadata_frames": preview.time_metadata_frames,
            "first_frame_index": preview.first_frame_index,
            "last_frame_index": preview.last_frame_index,
            "first_time_fs": preview.first_time_fs,
            "last_time_fs": preview.last_time_fs,
        }


@dataclass(slots=True)
class MDTrajectoryExportResult:
    """Result of writing the selected frames to disk."""

    output_dir: Path
    written_files: list[Path]
    selection: MDTrajectorySelectionResult
    metadata_file: Path | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "output_dir": str(self.output_dir),
            "written_files": [str(path) for path in self.written_files],
            "written_count": len(self.written_files),
            "metadata_file": (
                None if self.metadata_file is None else str(self.metadata_file)
            ),
            "selection": self.selection.to_dict(),
        }


class MDTrajectoryWorkflow:
    """Headless workflow for notebook and command-line usage."""

    def __init__(
        self,
        trajectory_file: str | Path,
        *,
        topology_file: str | Path | None = None,
        energy_file: str | Path | None = None,
        backend: str = "auto",
    ) -> None:
        self.trajectory_file = Path(trajectory_file)
        self.topology_file = (
            Path(topology_file) if topology_file is not None else None
        )
        self.energy_file = (
            Path(energy_file) if energy_file is not None else None
        )
        self.backend = backend
        self.manager = TrajectoryManager(
            input_file=self.trajectory_file,
            topology_file=self.topology_file,
            backend=backend,
        )
        self.summary: dict[str, object] | None = None
        self.energy_data: CP2KEnergyData | None = None
        self.suggested_cutoff_fs: float | None = None
        self.selected_cutoff_fs: float | None = None

    def inspect(self) -> dict[str, object]:
        """Inspect the trajectory metadata."""
        if self.summary is None:
            self.summary = self.manager.inspect()
        return dict(self.summary)

    def load_energy(self) -> CP2KEnergyData:
        """Load the configured CP2K energy file."""
        if self.energy_file is None:
            raise ValueError("No CP2K energy file was provided.")
        if self.energy_data is None:
            self.energy_data = CP2KEnergyData.from_file(self.energy_file)
        return self.energy_data

    def set_selected_cutoff(self, cutoff_fs: float | None) -> None:
        """Store the current cutoff to use for preview/export."""
        self.selected_cutoff_fs = cutoff_fs

    def suggest_cutoff(
        self,
        *,
        temp_target_k: float,
        temp_tol_k: float = 1.0,
        window: int = 3,
    ) -> SteadyStateResult:
        """Suggest a steady-state cutoff from the loaded energy data."""
        analyzer = CP2KEnergyAnalyzer(self.load_energy())
        result = analyzer.suggest_steady_state_cutoff(
            temp_target_k=temp_target_k,
            temp_tol_k=temp_tol_k,
            window=window,
        )
        self.suggested_cutoff_fs = result.cutoff_time_fs
        return result

    def resolve_cutoff(
        self,
        *,
        use_cutoff: bool = False,
        cutoff_fs: float | None = None,
    ) -> float | None:
        """Resolve the cutoff that should be applied to
        preview/export."""
        if not use_cutoff:
            return None

        resolved_cutoff = (
            cutoff_fs if cutoff_fs is not None else self.selected_cutoff_fs
        )
        if resolved_cutoff is None:
            raise ValueError(
                "Cutoff export is enabled, but no cutoff time is selected."
            )
        return resolved_cutoff

    def preview_selection(
        self,
        *,
        start: int | None = None,
        stop: int | None = None,
        stride: int = 1,
        post_cutoff_stride: int = 1,
        use_cutoff: bool = False,
        cutoff_fs: float | None = None,
        output_dir: str | Path | None = None,
    ) -> MDTrajectorySelectionResult:
        """Preview the selected frames and output target directory."""
        self.inspect()
        applied_cutoff_fs = self.resolve_cutoff(
            use_cutoff=use_cutoff,
            cutoff_fs=cutoff_fs,
        )
        preview = self.manager.preview_selection(
            start=start,
            stop=stop,
            stride=stride,
            min_time_fs=applied_cutoff_fs,
            post_cutoff_stride=post_cutoff_stride,
        )
        resolved_output_dir = (
            Path(output_dir)
            if output_dir is not None
            else suggest_output_dir(
                self.trajectory_file,
                cutoff_fs=applied_cutoff_fs,
            )
        )
        return MDTrajectorySelectionResult(
            preview=preview,
            output_dir=resolved_output_dir,
            applied_cutoff_fs=applied_cutoff_fs,
        )

    def export_frames(
        self,
        *,
        output_dir: str | Path | None = None,
        start: int | None = None,
        stop: int | None = None,
        stride: int = 1,
        post_cutoff_stride: int = 1,
        use_cutoff: bool = False,
        cutoff_fs: float | None = None,
    ) -> MDTrajectoryExportResult:
        """Write the current frame selection to disk."""
        selection = self.preview_selection(
            start=start,
            stop=stop,
            stride=stride,
            post_cutoff_stride=post_cutoff_stride,
            use_cutoff=use_cutoff,
            cutoff_fs=cutoff_fs,
            output_dir=output_dir,
        )
        if selection.preview.selected_frames == 0:
            raise ValueError("No frames match the current selection settings.")

        selected_frames = self.manager.get_selected_frames(
            start=start,
            stop=stop,
            stride=stride,
            min_time_fs=selection.applied_cutoff_fs,
            post_cutoff_stride=post_cutoff_stride,
        )
        written_files = self.manager.export_frames(
            output_dir=selection.output_dir,
            start=start,
            stop=stop,
            stride=stride,
            min_time_fs=selection.applied_cutoff_fs,
            post_cutoff_stride=post_cutoff_stride,
        )
        metadata_file = self._write_export_metadata(
            selection=selection,
            written_files=written_files,
            selected_frames=selected_frames,
        )
        return MDTrajectoryExportResult(
            output_dir=selection.output_dir,
            written_files=written_files,
            selection=selection,
            metadata_file=metadata_file,
        )

    def _write_export_metadata(
        self,
        *,
        selection: MDTrajectorySelectionResult,
        written_files: list[Path],
        selected_frames,
    ) -> Path:
        metadata_path = selection.output_dir / EXPORT_METADATA_FILENAME
        payload = {
            "version": 1,
            "trajectory_file": str(self.trajectory_file),
            "topology_file": (
                None if self.topology_file is None else str(self.topology_file)
            ),
            "energy_file": (
                None if self.energy_file is None else str(self.energy_file)
            ),
            "selection": selection.to_dict(),
            "written_frames": [
                {
                    "filename": path.name,
                    "frame_index": int(frame.frame_index),
                    "time_fs": (
                        None if frame.time_fs is None else float(frame.time_fs)
                    ),
                }
                for frame, path in zip(
                    selected_frames,
                    written_files,
                    strict=False,
                )
            ],
        }
        metadata_path.write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )
        return metadata_path
