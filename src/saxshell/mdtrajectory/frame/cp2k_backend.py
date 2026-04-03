from __future__ import annotations

import re
from pathlib import Path

from .base import FrameMetadata, FrameRecord, TrajectoryBackend

TIME_PATTERNS = (
    re.compile(
        r"time(?:\s*\[\s*fs\s*\])?\s*=\s*"
        r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)",
        re.IGNORECASE,
    ),
    re.compile(
        r"time_fs\s*=\s*" r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)",
        re.IGNORECASE,
    ),
)


class CP2KTrajectoryBackend(TrajectoryBackend):
    """Backend for CP2K-style XYZ and PDB trajectory files.

    Supports:
    - standard XYZ frames
    - metadata-style XYZ frames
    - prefix-style XYZ frames using 'frame' or 'NSTEP='
    - PDB trajectories split by MODEL/ENDMDL
    """

    def __init__(
        self,
        input_file: str | Path,
        topology_file: str | Path | None = None,
    ) -> None:
        super().__init__(input_file=input_file, topology_file=topology_file)
        suffix = self.input_file.suffix.lower()
        if suffix not in {".xyz", ".pdb"}:
            raise ValueError(
                "CP2KTrajectoryBackend supports only .xyz and .pdb files."
            )
        self.file_type = suffix.lstrip(".")

    def inspect(self) -> dict[str, object]:
        frame_metadata = self.load_frame_metadata()
        return {
            "input_file": str(self.input_file),
            "file_type": self.file_type,
            "n_frames": len(frame_metadata),
        }

    def iter_frame_metadata(self) -> list[FrameMetadata]:
        if self.file_type == "xyz":
            return self._parse_xyz_frame_metadata()
        if self.file_type == "pdb":
            return self._parse_pdb_frame_metadata()
        raise ValueError("Unsupported CP2K trajectory file type.")

    def iter_frames(self) -> list[FrameRecord]:
        if self.file_type == "xyz":
            return self._parse_xyz_frames()
        if self.file_type == "pdb":
            return self._parse_pdb_frames()
        raise ValueError("Unsupported CP2K trajectory file type.")

    def _estimate_frame_count(self) -> int:
        if self.file_type == "xyz":
            return self._estimate_frame_count_xyz()
        if self.file_type == "pdb":
            return self._estimate_frame_count_pdb()
        return 0

    def _estimate_frame_count_xyz(self) -> int:
        count = 0
        with self.input_file.open("r") as handle:
            while True:
                line = handle.readline()
                if not line:
                    break

                stripped = line.strip()
                if not stripped:
                    continue

                if stripped.startswith("frame") or stripped.startswith(
                    "NSTEP="
                ):
                    count += 1
                    atom_count_line = handle.readline()
                    if not atom_count_line:
                        break
                    atom_count = atom_count_line.strip()
                    if atom_count.isdigit():
                        for _ in range(int(atom_count)):
                            if not handle.readline():
                                break
                    continue

                if stripped.isdigit():
                    count += 1
                    atom_count = int(stripped)
                    if not handle.readline():
                        break
                    for _ in range(atom_count):
                            if not handle.readline():
                                break
        return count

    def _parse_xyz_frame_metadata(self) -> list[FrameMetadata]:
        frames: list[FrameMetadata] = []
        frame_idx = 0

        with self.input_file.open("r") as handle:
            while True:
                line = handle.readline()
                if not line:
                    break

                stripped = line.strip()
                if not stripped:
                    continue

                if stripped.startswith("frame") or stripped.startswith(
                    "NSTEP="
                ):
                    atom_count_line = handle.readline()
                    if not atom_count_line:
                        break
                    atom_count_text = atom_count_line.strip()
                    if not atom_count_text.isdigit():
                        continue
                    frames.append(
                        FrameMetadata(
                            frame_index=frame_idx,
                            time_fs=self._parse_time_from_header(line),
                        )
                    )
                    frame_idx += 1
                    for _ in range(int(atom_count_text)):
                        if not handle.readline():
                            break
                    continue

                if not stripped.isdigit():
                    continue

                atom_count = int(stripped)
                comment = handle.readline()
                time_val = (
                    None
                    if not comment
                    else self._parse_time_from_metadata(comment)
                )
                frames.append(
                    FrameMetadata(
                        frame_index=frame_idx,
                        time_fs=time_val,
                    )
                )
                frame_idx += 1
                if not comment:
                    break
                for _ in range(atom_count):
                    if not handle.readline():
                        break

        return frames

    def _parse_xyz_frames(self) -> list[FrameRecord]:
        lines = self.input_file.read_text().splitlines(keepends=True)
        frames: list[FrameRecord] = []
        frame_idx = 0
        atom_count: int | None = None
        buffer: list[str] = []
        time_val: float | None = None

        i = 0
        while i < len(lines):
            s = lines[i].strip()

            is_metadata_style = (
                s.isdigit()
                and i + 1 < len(lines)
                and lines[i + 1].strip().startswith("i =")
            )

            if is_metadata_style:
                if buffer and atom_count is not None:
                    frames.append(
                        FrameRecord(
                            frame_index=frame_idx,
                            file_type="xyz",
                            atom_count=atom_count,
                            lines=buffer.copy(),
                            time_fs=time_val,
                        )
                    )
                    frame_idx += 1

                atom_count = int(s)
                comment = lines[i + 1]
                time_val = self._parse_time_from_metadata(comment)
                buffer = [comment]
                i += 2
                continue

            if s.startswith("frame") or s.startswith("NSTEP="):
                if buffer and atom_count is not None:
                    frames.append(
                        FrameRecord(
                            frame_index=frame_idx,
                            file_type="xyz",
                            atom_count=atom_count,
                            lines=buffer.copy(),
                            time_fs=time_val,
                        )
                    )
                    frame_idx += 1
                atom_count = None
                time_val = self._parse_time_from_header(lines[i])
                buffer = [lines[i]]
                i += 1
                continue

            if s.isdigit():
                atom_count = int(s)
                if not buffer:
                    time_val = None
                i += 1
                continue

            if atom_count is not None:
                buffer.append(lines[i])

            i += 1

        if buffer and atom_count is not None:
            frames.append(
                FrameRecord(
                    frame_index=frame_idx,
                    file_type="xyz",
                    atom_count=atom_count,
                    lines=buffer.copy(),
                    time_fs=time_val,
                )
            )

        return frames

    def _parse_time_from_metadata(self, line: str) -> float | None:
        return self._parse_time_from_header(line)

    def _parse_time_from_header(self, line: str) -> float | None:
        text = line.strip()
        if "time" not in text.lower():
            return None
        try:
            for pattern in TIME_PATTERNS:
                match = pattern.search(text)
                if match is not None:
                    return float(match.group(1))
        except ValueError:
            pass
        return None

    def _estimate_frame_count_pdb(self) -> int:
        count = 0
        with self.input_file.open("r") as handle:
            for line in handle:
                if line.startswith("MODEL"):
                    count += 1
        return count

    def _parse_pdb_frame_metadata(self) -> list[FrameMetadata]:
        frames: list[FrameMetadata] = []
        frame_idx = 0
        with self.input_file.open("r") as handle:
            for line in handle:
                if not line.startswith("MODEL"):
                    continue
                frames.append(
                    FrameMetadata(frame_index=frame_idx, time_fs=None)
                )
                frame_idx += 1
        return frames

    def _parse_pdb_frames(self) -> list[FrameRecord]:
        frames: list[FrameRecord] = []
        frame_idx = 0
        buffer: list[str] = []

        with self.input_file.open("r") as handle:
            for line in handle:
                if line.startswith("MODEL"):
                    buffer = [line]
                    continue

                if line.startswith("ENDMDL") or line.startswith("END"):
                    if buffer:
                        buffer.append(line)
                        frames.append(
                            FrameRecord(
                                frame_index=frame_idx,
                                file_type="pdb",
                                atom_count=None,
                                lines=buffer.copy(),
                                time_fs=None,
                            )
                        )
                        frame_idx += 1
                        buffer = []
                    continue

                if buffer:
                    buffer.append(line)

        return frames
