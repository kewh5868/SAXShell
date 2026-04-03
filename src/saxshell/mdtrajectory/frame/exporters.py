from __future__ import annotations

from pathlib import Path
from typing import Callable

from .base import FrameRecord

ExportProgressCallback = Callable[[int, int, str], None]


def export_xyz_frames(
    frames: list[FrameRecord],
    output_dir: str | Path,
    *,
    progress_callback: ExportProgressCallback | None = None,
) -> list[Path]:
    """Write frame records as XYZ files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    written_files: list[Path] = []
    xyz_frames = [frame for frame in frames if frame.file_type == "xyz"]
    total_frames = len(xyz_frames)

    for index, frame in enumerate(xyz_frames, start=1):
        file_path = output_path / f"frame_{frame.frame_index:04d}.xyz"
        with file_path.open("w") as handle:
            handle.write(f"{frame.atom_count}\n")
            handle.write(frame.lines[0])
            for line in frame.lines[1:]:
                parts = line.split()
                if len(parts) != 4:
                    continue
                label = "".join(c for c in parts[0] if not c.isdigit())
                label = label.capitalize()
                x, y, z = parts[1], parts[2], parts[3]
                handle.write(f"{label:>4} {x:>10} {y:>10} {z:>10}\n")
        written_files.append(file_path)
        if progress_callback is not None:
            progress_callback(
                index,
                total_frames,
                f"Exporting frame {index} of {total_frames}: {file_path.name}",
            )

    return written_files


def export_pdb_frames(
    frames: list[FrameRecord],
    output_dir: str | Path,
    *,
    progress_callback: ExportProgressCallback | None = None,
) -> list[Path]:
    """Write frame records as PDB files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    written_files: list[Path] = []
    pdb_frames = [frame for frame in frames if frame.file_type == "pdb"]
    total_frames = len(pdb_frames)

    for index, frame in enumerate(pdb_frames, start=1):
        file_path = output_path / f"frame_{frame.frame_index:04d}.pdb"
        with file_path.open("w") as handle:
            handle.writelines(frame.lines)
        written_files.append(file_path)
        if progress_callback is not None:
            progress_callback(
                index,
                total_frames,
                f"Exporting frame {index} of {total_frames}: {file_path.name}",
            )

    return written_files
