from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .base import FrameRecord
from .cp2k_backend import CP2KTrajectoryBackend
from .exporters import export_pdb_frames, export_xyz_frames
from .selectors import select_frames

if TYPE_CHECKING:
    from .cp2k_ener import CP2KEnergyData
    from .cutoff_analysis import SteadyStateResult


@dataclass(slots=True)
class FrameSelectionPreview:
    """Summary of the frames matched by the current export settings."""

    total_frames: int
    selected_frames: int
    start: int | None
    stop: int | None
    stride: int
    min_time_fs: float | None
    first_frame_index: int | None
    last_frame_index: int | None
    first_time_fs: float | None
    last_time_fs: float | None
    time_metadata_frames: int


class TrajectoryManager:
    """Unified front-end for trajectory parsing and frame export."""

    def __init__(
        self,
        input_file: str | Path,
        topology_file: str | Path | None = None,
        backend: str = "auto",
    ) -> None:
        self.input_file = Path(input_file)
        self.topology_file = (
            Path(topology_file) if topology_file is not None else None
        )
        self.backend_name = backend
        self.backend = self._build_backend()
        self.frames: list[FrameRecord] | None = None

    def _build_backend(self):
        suffix = self.input_file.suffix.lower()

        if self.backend_name == "cp2k":
            return CP2KTrajectoryBackend(
                input_file=self.input_file,
                topology_file=self.topology_file,
            )

        if self.backend_name == "auto":
            if suffix in {".xyz", ".pdb"}:
                return CP2KTrajectoryBackend(
                    input_file=self.input_file,
                    topology_file=self.topology_file,
                )

        raise ValueError(
            "Only CP2K .xyz/.pdb backend is implemented in this version."
        )

    def inspect(self) -> dict[str, object]:
        return self.backend.inspect()

    def load_frames(self) -> list[FrameRecord]:
        self.frames = self.backend.iter_frames()
        return self.frames

    def get_selected_frames(
        self,
        start: int | None = None,
        stop: int | None = None,
        stride: int = 1,
        min_time_fs: float | None = None,
    ) -> list[FrameRecord]:
        if self.frames is None:
            self.load_frames()

        assert self.frames is not None
        self._validate_time_cutoff_support(
            frames=self.frames,
            min_time_fs=min_time_fs,
        )
        return select_frames(
            self.frames,
            start=start,
            stop=stop,
            stride=stride,
            min_time_fs=min_time_fs,
        )

    def preview_selection(
        self,
        start: int | None = None,
        stop: int | None = None,
        stride: int = 1,
        min_time_fs: float | None = None,
    ) -> FrameSelectionPreview:
        if self.frames is None:
            self.load_frames()

        assert self.frames is not None
        selected_frames = self.get_selected_frames(
            start=start,
            stop=stop,
            stride=stride,
            min_time_fs=min_time_fs,
        )
        selected_times = [
            frame.time_fs
            for frame in selected_frames
            if frame.time_fs is not None
        ]

        return FrameSelectionPreview(
            total_frames=len(self.frames),
            selected_frames=len(selected_frames),
            start=start,
            stop=stop,
            stride=stride,
            min_time_fs=min_time_fs,
            first_frame_index=(
                selected_frames[0].frame_index if selected_frames else None
            ),
            last_frame_index=(
                selected_frames[-1].frame_index if selected_frames else None
            ),
            first_time_fs=selected_times[0] if selected_times else None,
            last_time_fs=selected_times[-1] if selected_times else None,
            time_metadata_frames=sum(
                frame.time_fs is not None for frame in self.frames
            ),
        )

    def export_frames(
        self,
        output_dir: str | Path,
        start: int | None = None,
        stop: int | None = None,
        stride: int = 1,
        min_time_fs: float | None = None,
    ) -> list[Path]:
        frames = self.get_selected_frames(
            start=start,
            stop=stop,
            stride=stride,
            min_time_fs=min_time_fs,
        )

        if not frames:
            return []

        if frames[0].file_type == "xyz":
            return export_xyz_frames(frames, output_dir=output_dir)

        if frames[0].file_type == "pdb":
            return export_pdb_frames(frames, output_dir=output_dir)

        raise ValueError("Unsupported frame type for export.")

    def _validate_time_cutoff_support(
        self,
        frames: list[FrameRecord],
        min_time_fs: float | None,
    ) -> None:
        if min_time_fs is None:
            return
        if any(frame.time_fs is not None for frame in frames):
            return
        raise ValueError(
            "This trajectory does not include frame times, so a time "
            "cutoff cannot be applied."
        )


class CP2KFrameExtractionWorkflow:
    """Helper workflow that combines CP2K trajectory handling with
    optional .ener-based cutoff selection."""

    def __init__(
        self,
        trajectory_file: str | Path,
        energy_file: str | Path | None = None,
    ) -> None:
        self.trajectory_file = Path(trajectory_file)
        self.energy_file = (
            Path(energy_file) if energy_file is not None else None
        )
        self.trajectory = TrajectoryManager(
            input_file=self.trajectory_file,
            backend="cp2k",
        )
        self.energy_data: CP2KEnergyData | None = None
        self.steady_state: SteadyStateResult | None = None
        self.selected_cutoff_fs: float | None = None

    def load_energy(self) -> CP2KEnergyData:
        from .cp2k_ener import CP2KEnergyData

        if self.energy_file is None:
            raise ValueError("No CP2K energy file was provided.")
        self.energy_data = CP2KEnergyData.from_file(self.energy_file)
        return self.energy_data

    def suggest_cutoff(
        self,
        temp_target_k: float,
        temp_tol_k: float = 1.0,
        window: int = 10,
    ) -> SteadyStateResult:
        if self.energy_data is None:
            self.load_energy()

        assert self.energy_data is not None
        from .cutoff_analysis import CP2KEnergyAnalyzer

        analyzer = CP2KEnergyAnalyzer(self.energy_data)
        self.steady_state = analyzer.suggest_steady_state_cutoff(
            temp_target_k=temp_target_k,
            temp_tol_k=temp_tol_k,
            window=window,
        )
        return self.steady_state

    def choose_cutoff_interactively(self) -> float | None:
        if self.energy_data is None:
            self.load_energy()

        assert self.energy_data is not None
        from .cutoff_plot import CP2KEnergyCutoffSelector

        selector = CP2KEnergyCutoffSelector(
            energy_data=self.energy_data,
            steady_state=self.steady_state,
        )

        def _store_cutoff(cutoff_fs: float) -> None:
            self.selected_cutoff_fs = cutoff_fs

        selector.show(on_cutoff=_store_cutoff)
        return self.selected_cutoff_fs

    def export_frames(
        self,
        output_dir: str | Path,
        start: int | None = None,
        stop: int | None = None,
        stride: int = 1,
        use_selected_cutoff: bool = False,
    ) -> list[Path]:
        min_time_fs = self.selected_cutoff_fs if use_selected_cutoff else None
        return self.trajectory.export_frames(
            output_dir=output_dir,
            start=start,
            stop=stop,
            stride=stride,
            min_time_fs=min_time_fs,
        )
