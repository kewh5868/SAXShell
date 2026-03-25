from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from saxshell.mdtrajectory.frame.cp2k_ener import CP2KEnergyData
from saxshell.mdtrajectory.frame.manager import TrajectoryManager
from saxshell.mdtrajectory.ui.cutoff_panel import CutoffPanel
from saxshell.mdtrajectory.ui.export_panel import ExportPanel
from saxshell.mdtrajectory.ui.state import MDTrajectoryAppState
from saxshell.mdtrajectory.ui.trajectory_panel import TrajectoryPanel
from saxshell.mdtrajectory.workflow import suggest_output_dir


@dataclass(slots=True)
class InspectionResult:
    """Result payload for background trajectory inspection."""

    manager: TrajectoryManager
    summary: dict[str, object]
    energy_data: CP2KEnergyData | None


class InspectionWorker(QObject):
    """Background worker for trajectory metadata and energy loading."""

    metadata_ready = Signal(object)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        trajectory_file: Path,
        topology_file: Path | None,
        energy_file: Path | None,
        *,
        manager: TrajectoryManager | None = None,
        summary: dict[str, object] | None = None,
        reload_trajectory: bool = True,
    ) -> None:
        super().__init__()
        self.trajectory_file = trajectory_file
        self.topology_file = topology_file
        self.energy_file = energy_file
        self.manager = manager
        self.summary = summary
        self.reload_trajectory = reload_trajectory

    @Slot()
    def run(self) -> None:
        try:
            manager = self.manager
            summary = self.summary
            if self.reload_trajectory:
                manager = TrajectoryManager(
                    input_file=self.trajectory_file,
                    topology_file=self.topology_file,
                    backend="auto",
                )
                summary = manager.inspect()

            if manager is None or summary is None:
                raise ValueError(
                    "Inspection worker needs trajectory metadata to load "
                    "the optional energy file."
                )
            self.metadata_ready.emit(
                InspectionResult(
                    manager=manager,
                    summary=summary,
                    energy_data=None,
                )
            )
            if self.energy_file is not None:
                energy_data = CP2KEnergyData.from_file(self.energy_file)
            else:
                energy_data = None
            self.finished.emit(energy_data)
        except Exception as exc:
            self.failed.emit(str(exc))


class MDTrajectoryMainWindow(QMainWindow):
    """Main Qt window for the mdtrajectory application."""

    def __init__(self) -> None:
        super().__init__()
        self.state = MDTrajectoryAppState()
        self.manager: TrajectoryManager | None = None
        self._last_summary: dict[str, object] | None = None
        self._active_reload_trajectory = True
        self._inspect_thread: QThread | None = None
        self._inspect_worker: InspectionWorker | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        self.setWindowTitle("SAXSShell (mdtrajectory)")
        self.resize(1280, 780)

        central = QWidget()
        root = QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(12)

        self.trajectory_panel = TrajectoryPanel()
        self.cutoff_panel = CutoffPanel()
        self.export_panel = ExportPanel()

        left_layout.addWidget(self.trajectory_panel)
        left_layout.addWidget(self.export_panel)
        left_layout.addStretch(1)

        splitter.addWidget(self._wrap_scroll_area(left))
        splitter.addWidget(self._wrap_scroll_area(self.cutoff_panel))
        splitter.setSizes([420, 860])

        root.addWidget(splitter)
        self.setCentralWidget(central)
        self.statusBar().showMessage("Ready")

        self.trajectory_panel.inspect_requested.connect(
            self.inspect_trajectory
        )
        self.trajectory_panel.selection_changed.connect(
            self._refresh_selection_preview
        )
        self.trajectory_panel.trajectory_path_changed.connect(
            self._suggest_output_dir_from_trajectory
        )
        self.export_panel.export_requested.connect(self.export_frames)
        self.export_panel.settings_changed.connect(
            self._refresh_selection_preview
        )
        self.cutoff_panel.selection_changed.connect(
            self._refresh_selection_preview
        )
        self.cutoff_panel.suggestion_updated.connect(
            self._store_suggested_cutoff
        )
        self.cutoff_panel.cutoff_selected.connect(self._store_selected_cutoff)
        self.export_panel.set_selection_summary(
            "Inspect a trajectory to preview the export selection."
        )

    def inspect_trajectory(self) -> None:
        try:
            if self._inspect_thread is not None:
                return

            trajectory_file = self.trajectory_panel.get_trajectory_path()
            topology_file = self.trajectory_panel.get_topology_path()
            energy_file = self.trajectory_panel.get_energy_path()
            previous_trajectory = self.state.trajectory_file
            previous_topology = self.state.topology_file
            previous_energy = self.state.energy_file

            if trajectory_file is None:
                raise ValueError("No trajectory file selected.")

            trajectory_changed = (
                self.manager is None
                or trajectory_file != previous_trajectory
                or topology_file != previous_topology
            )
            energy_changed = energy_file != previous_energy

            self.state.trajectory_file = trajectory_file
            self.state.topology_file = topology_file
            self.state.energy_file = energy_file
            self.state.start = self.trajectory_panel.get_start()
            self.state.stop = self.trajectory_panel.get_stop()
            self.state.stride = self.trajectory_panel.get_stride()
            self._update_suggested_output_dir()

            if trajectory_changed:
                self.state.selected_cutoff_fs = None
                self.state.suggested_cutoff_fs = None
                self.manager = None
                self._last_summary = None
                self.cutoff_panel.reset()
                self.export_panel.set_log(
                    "Inspect request received.\n"
                    f"Loading trajectory metadata from: {trajectory_file}"
                )
                if energy_file is not None:
                    self.export_panel.append_log(
                        f"Will also load CP2K energy data from: {energy_file}"
                    )
                else:
                    self.export_panel.append_log(
                        "No CP2K energy file selected."
                    )
                self.trajectory_panel.set_summary_text(
                    "Inspecting trajectory metadata..."
                )
                self.export_panel.set_selection_summary(
                    "Inspecting trajectory and loading optional energy data..."
                )
                self._start_inspection_worker(
                    trajectory_file=trajectory_file,
                    topology_file=topology_file,
                    energy_file=energy_file,
                    reload_trajectory=True,
                )
                return

            if energy_changed and energy_file is not None:
                self.state.selected_cutoff_fs = None
                self.state.suggested_cutoff_fs = None
                self.cutoff_panel.reset()
                self.export_panel.set_log(
                    "Inspect request received.\n"
                    "Trajectory path unchanged; reusing cached trajectory "
                    "metadata.\n"
                    f"Loading CP2K energy data from: {energy_file}"
                )
                self.export_panel.set_selection_summary(
                    "Reusing cached trajectory metadata while the energy "
                    "profile loads..."
                )
                self._start_inspection_worker(
                    trajectory_file=trajectory_file,
                    topology_file=topology_file,
                    energy_file=energy_file,
                    manager=self.manager,
                    summary=self._last_summary,
                    reload_trajectory=False,
                )
                return

            if energy_changed and energy_file is None:
                self.state.selected_cutoff_fs = None
                self.state.suggested_cutoff_fs = None
                self.cutoff_panel.reset()
                self.export_panel.set_log(
                    "Inspect request received.\n"
                    "Trajectory path unchanged; reusing cached trajectory "
                    "metadata.\n"
                    "Cleared the previously loaded CP2K energy file."
                )
                self._refresh_selection_preview()
                self.statusBar().showMessage(
                    "Reused loaded trajectory metadata.", 5000
                )
                return

            self.export_panel.set_log(
                "Inspect request received.\n"
                "Trajectory and energy inputs are unchanged; using the "
                "already loaded data."
            )
            self._refresh_selection_preview()
            self.statusBar().showMessage(
                "Trajectory already loaded; nothing new to inspect.",
                5000,
            )

        except Exception as exc:
            self._show_error(str(exc))

    def export_frames(self) -> None:
        try:
            if self.manager is None:
                raise ValueError(
                    "Inspect a trajectory before exporting frames."
                )

            output_dir = self.export_panel.get_output_dir()
            if output_dir is None:
                raise ValueError("No output directory selected.")

            self._sync_state_from_controls()
            self.state.output_dir = output_dir
            min_time_fs = self._resolved_export_cutoff()
            preview = self.manager.preview_selection(
                start=self.state.start,
                stop=self.state.stop,
                stride=self.state.stride,
                min_time_fs=min_time_fs,
                post_cutoff_stride=self._resolved_post_cutoff_stride(
                    min_time_fs=min_time_fs
                ),
            )
            if preview.selected_frames == 0:
                raise ValueError(
                    "No frames match the current selection settings."
                )

            written_files = self.manager.export_frames(
                output_dir=output_dir,
                start=self.state.start,
                stop=self.state.stop,
                stride=self.state.stride,
                min_time_fs=min_time_fs,
                post_cutoff_stride=self._resolved_post_cutoff_stride(
                    min_time_fs=min_time_fs
                ),
            )

            lines = [
                "Frame export complete.",
                f"Output directory: {output_dir}",
                f"Frames written: {len(written_files)}",
                f"Start: {self.state.start}",
                f"Stop: {self.state.stop}",
                f"Stride: {self.state.stride}",
            ]
            if min_time_fs is not None:
                lines.append(f"Applied cutoff: {min_time_fs:.3f} fs")
                lines.append(
                    "Post-cutoff frame interval: "
                    f"{self._resolved_post_cutoff_stride(min_time_fs=min_time_fs)}"
                )
            else:
                lines.append("Applied cutoff: None")
            if preview.first_frame_index is not None:
                lines.append(
                    "Frame index range: "
                    f"{preview.first_frame_index} to "
                    f"{preview.last_frame_index}"
                )
            if preview.first_time_fs is not None:
                lines.append(
                    "Time range: "
                    f"{preview.first_time_fs:.3f} fs to "
                    f"{preview.last_time_fs:.3f} fs"
                )

            self.export_panel.set_log("\n".join(lines))
            self._update_suggested_output_dir()
            self._refresh_selection_preview()

        except Exception as exc:
            self._show_error(str(exc))

    def _store_suggested_cutoff(self, cutoff_fs: float) -> None:
        self.state.suggested_cutoff_fs = cutoff_fs

    def _store_selected_cutoff(self, cutoff_fs: float) -> None:
        self.state.selected_cutoff_fs = cutoff_fs

    def _sync_state_from_controls(self) -> None:
        self.state.start = self.trajectory_panel.get_start()
        self.state.stop = self.trajectory_panel.get_stop()
        self.state.stride = self.trajectory_panel.get_stride()
        self.state.use_cutoff_for_export = self.export_panel.use_cutoff()
        self.state.use_post_cutoff_stride = (
            self.export_panel.use_post_cutoff_stride()
        )
        self.state.post_cutoff_stride = (
            self.export_panel.get_post_cutoff_stride()
        )
        self.state.selected_cutoff_fs = self.cutoff_panel.get_selected_cutoff()
        self.state.suggested_cutoff_fs = (
            self.cutoff_panel.get_suggested_cutoff()
        )

    def _resolved_export_cutoff(self) -> float | None:
        if not self.state.use_cutoff_for_export:
            return None
        if self.state.selected_cutoff_fs is None:
            raise ValueError(
                "Cutoff export is enabled, but no cutoff time is selected."
            )
        return self.state.selected_cutoff_fs

    def _resolved_post_cutoff_stride(
        self,
        *,
        min_time_fs: float | None,
    ) -> int:
        if min_time_fs is None or not self.state.use_post_cutoff_stride:
            return 1
        return max(1, int(self.state.post_cutoff_stride))

    def _refresh_selection_preview(self) -> None:
        self._sync_state_from_controls()
        self._update_suggested_output_dir()

        if self.manager is None:
            self.export_panel.set_selection_summary(
                "Inspect a trajectory to preview the export selection."
            )
            return

        try:
            min_time_fs = None
            if self.state.use_cutoff_for_export:
                min_time_fs = self._resolved_export_cutoff()

            preview = self.manager.preview_selection(
                start=self.state.start,
                stop=self.state.stop,
                stride=self.state.stride,
                min_time_fs=min_time_fs,
                post_cutoff_stride=self._resolved_post_cutoff_stride(
                    min_time_fs=min_time_fs
                ),
            )
            self.export_panel.set_selection_summary(
                self._format_selection_summary(preview)
            )
        except Exception as exc:
            self.export_panel.set_selection_summary(
                f"Selection preview unavailable:\n{exc}"
            )

    def _wrap_scroll_area(self, widget: QWidget) -> QScrollArea:
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(widget)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        return scroll_area

    def _start_inspection_worker(
        self,
        trajectory_file: Path,
        topology_file: Path | None,
        energy_file: Path | None,
        *,
        manager: TrajectoryManager | None = None,
        summary: dict[str, object] | None = None,
        reload_trajectory: bool = True,
    ) -> None:
        self._set_inspection_busy(True)
        self._active_reload_trajectory = reload_trajectory
        self._inspect_thread = QThread(self)
        self._inspect_worker = InspectionWorker(
            trajectory_file=trajectory_file,
            topology_file=topology_file,
            energy_file=energy_file,
            manager=manager,
            summary=summary,
            reload_trajectory=reload_trajectory,
        )
        self._inspect_worker.moveToThread(self._inspect_thread)
        self._inspect_thread.started.connect(self._inspect_worker.run)
        self._inspect_worker.metadata_ready.connect(
            self._handle_inspection_metadata
        )
        self._inspect_worker.finished.connect(self._handle_energy_data_loaded)
        self._inspect_worker.failed.connect(self._handle_inspection_error)
        self._inspect_worker.finished.connect(self._inspect_thread.quit)
        self._inspect_worker.failed.connect(self._inspect_thread.quit)
        self._inspect_thread.finished.connect(self._cleanup_inspection_worker)
        self._inspect_thread.start()

    def _set_inspection_busy(self, is_busy: bool) -> None:
        if is_busy:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            self.statusBar().showMessage("Inspecting trajectory...")
        else:
            while QApplication.overrideCursor() is not None:
                QApplication.restoreOverrideCursor()
        self.trajectory_panel.setEnabled(not is_busy)
        self.cutoff_panel.setEnabled(not is_busy)
        self.export_panel.setEnabled(not is_busy)

    def _suggest_output_dir_from_trajectory(
        self,
        trajectory_path: Path | None,
    ) -> None:
        if trajectory_path is None:
            return
        self._update_suggested_output_dir(trajectory_path=trajectory_path)

    @Slot(object)
    def _handle_inspection_metadata(self, result: InspectionResult) -> None:
        self.manager = result.manager
        self._last_summary = result.summary
        self.trajectory_panel.set_summary(result.summary)
        n_frames = result.summary.get("n_frames", "unknown")
        file_type = result.summary.get("file_type", "unknown")
        self.export_panel.append_log(
            "Trajectory metadata loaded: "
            f"{n_frames} estimated {file_type} frames."
        )
        self._refresh_selection_preview()
        if self.state.energy_file is None:
            self.statusBar().showMessage(
                f"Loaded trajectory metadata for {self.state.trajectory_file}",
                5000,
            )
            return
        self.export_panel.append_log(
            "Trajectory is ready. Continuing with CP2K energy loading..."
        )
        self.statusBar().showMessage(
            "Trajectory metadata loaded. Loading CP2K energy profile...",
            5000,
        )

    @Slot(object)
    def _handle_energy_data_loaded(
        self,
        energy_data: CP2KEnergyData | None,
    ) -> None:
        if energy_data is not None:
            self.cutoff_panel.load_energy_data(energy_data)
            self.export_panel.append_log(
                "CP2K energy data loaded: "
                f"{energy_data.n_points} samples from {energy_data.filepath}"
            )
        else:
            self.export_panel.append_log(
                "Inspection complete. No CP2K energy file was loaded."
            )
        self._refresh_selection_preview()
        self.statusBar().showMessage(
            f"Loaded trajectory metadata for {self.state.trajectory_file}",
            5000,
        )

    @Slot(str)
    def _handle_inspection_error(self, message: str) -> None:
        if self._active_reload_trajectory:
            self.manager = None
            self._last_summary = None
            self.trajectory_panel.set_summary_text("Inspection failed.")
            self.export_panel.set_selection_summary(
                "Inspection failed. Fix the inputs and try again."
            )
        else:
            self._refresh_selection_preview()
        self.export_panel.append_log(f"Inspection failed: {message}")
        self.statusBar().showMessage("Inspection failed.", 5000)
        self._show_error(message)

    def _cleanup_inspection_worker(self) -> None:
        self._set_inspection_busy(False)
        self._active_reload_trajectory = True
        if self._inspect_worker is not None:
            self._inspect_worker.deleteLater()
            self._inspect_worker = None
        if self._inspect_thread is not None:
            self._inspect_thread.deleteLater()
            self._inspect_thread = None

    def _format_selection_summary(self, preview) -> str:
        output_dir = self.export_panel.get_output_dir()
        lines = [
            "Current export selection",
        ]
        if output_dir is not None:
            lines.extend(
                [
                    f"Output folder: {output_dir.name}",
                    f"Output path: {output_dir}",
                ]
            )
        else:
            lines.append("Output target: None")
        lines.extend(
            [
                f"Trajectory frames: {preview.total_frames}",
                f"Frames selected: {preview.selected_frames}",
                f"Start: {preview.start}",
                f"Stop: {preview.stop}",
                f"Stride: {preview.stride}",
                f"Time-tagged frames: {preview.time_metadata_frames}",
            ]
        )
        if preview.min_time_fs is not None:
            lines.append(f"Applied cutoff: {preview.min_time_fs:.3f} fs")
            lines.append(
                "Post-cutoff frame interval: " f"{preview.post_cutoff_stride}"
            )
        else:
            lines.append("Applied cutoff: None")
        if preview.first_frame_index is not None:
            lines.append(
                "Frame index range: "
                f"{preview.first_frame_index} to "
                f"{preview.last_frame_index}"
            )
        if preview.first_time_fs is not None:
            lines.append(
                "Time range: "
                f"{preview.first_time_fs:.3f} fs to "
                f"{preview.last_time_fs:.3f} fs"
            )
        return "\n".join(lines)

    def _update_suggested_output_dir(
        self,
        *,
        trajectory_path: Path | None = None,
    ) -> None:
        target_dir = self._build_suggested_output_dir(
            trajectory_path=trajectory_path,
        )
        if target_dir is None:
            return
        self.export_panel.suggest_output_dir(target_dir)

    def _build_suggested_output_dir(
        self,
        *,
        trajectory_path: Path | None = None,
    ) -> Path | None:
        self._sync_state_from_controls()
        source_path = (
            trajectory_path
            if trajectory_path is not None
            else self.state.trajectory_file
        )
        if source_path is None:
            return None

        min_time_fs = None
        if (
            self.export_panel.use_cutoff()
            and self.cutoff_panel.get_selected_cutoff() is not None
        ):
            min_time_fs = self.cutoff_panel.get_selected_cutoff()

        return suggest_output_dir(source_path, cutoff_fs=min_time_fs)

    def _show_error(self, message: str) -> None:
        QMessageBox.critical(self, "Error", message)


def launch_mdtrajectory_app() -> MDTrajectoryMainWindow:
    window = MDTrajectoryMainWindow()
    window.show()
    return window
