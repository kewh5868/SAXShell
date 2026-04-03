from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class FrameMetadata:
    """Lightweight frame metadata used for selection previews."""

    frame_index: int
    time_fs: float | None = None


@dataclass(slots=True)
class FrameRecord:
    """In-memory representation of one trajectory frame."""

    frame_index: int
    file_type: str
    atom_count: int | None
    lines: list[str]
    time_fs: float | None = None


class TrajectoryBackend(ABC):
    """Abstract interface for trajectory parsing backends."""

    def __init__(
        self,
        input_file: str | Path,
        topology_file: str | Path | None = None,
    ) -> None:
        self.input_file = Path(input_file)
        self.topology_file = (
            Path(topology_file) if topology_file is not None else None
        )
        self._frame_metadata: list[FrameMetadata] | None = None

    def load_frame_metadata(self) -> list[FrameMetadata]:
        """Return cached frame metadata for lightweight inspection."""
        if self._frame_metadata is None:
            self._frame_metadata = self.iter_frame_metadata()
        return self._frame_metadata

    @abstractmethod
    def inspect(self) -> dict[str, object]:
        """Return basic metadata about the trajectory."""

    @abstractmethod
    def iter_frame_metadata(self) -> list[FrameMetadata]:
        """Return lightweight frame metadata for previews."""

    @abstractmethod
    def iter_frames(self) -> list[FrameRecord]:
        """Return parsed frame records."""
