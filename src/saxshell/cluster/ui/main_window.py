from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from re import sub
from time import monotonic

from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from saxshell.cluster import (
    ExtractedFrameFolderClusterAnalyzer,
    PairCutoffDefinitions,
    detect_frame_folder_mode,
    format_box_dimensions,
    format_search_mode_label,
    frame_folder_label,
    frame_output_suffix,
)
from saxshell.cluster.ui.definitions_panel import ClusterDefinitionsPanel
from saxshell.cluster.ui.export_panel import ClusterExportPanel
from saxshell.cluster.ui.trajectory_panel import ClusterTrajectoryPanel
from saxshell.saxs.project_manager import (
    ProjectSettings,
    SAXSProjectManager,
    build_project_paths,
)
from saxshell.saxs.ui.branding import (
    configure_saxshell_application,
    load_saxshell_icon,
    prepare_saxshell_application_identity,
)
from saxshell.saxs.ui.project_status_label import CompactProjectStatusLabel
from saxshell.structure import AtomTypeDefinitions


@dataclass(slots=True)
class ClusterSelectionPreview:
    """Summary of the extracted frames currently selected for
    analysis."""

    total_frames: int
    selected_frames: int
    first_frame_name: str | None
    last_frame_name: str | None


def _save_state_frequency_label(frequency: int) -> str:
    if frequency >= 10**8:
        return "end of run only"
    return f"every {frequency} frame(s)"


@dataclass(slots=True)
class ClusterJobConfig:
    """Cluster-analysis settings assembled from the UI."""

    frames_dir: Path
    atom_type_definitions: AtomTypeDefinitions
    pair_cutoff_definitions: PairCutoffDefinitions
    box_dimensions: tuple[float, float, float] | None
    use_pbc: bool
    search_mode: str
    save_state_frequency: int
    default_cutoff: float | None
    shell_levels: tuple[int, ...]
    include_shell_levels: tuple[int, ...]
    shared_shells: bool
    smart_solvation_shells: bool
    include_shell_atoms_in_stoichiometry: bool
    output_dir: Path


@dataclass(slots=True)
class ClusterExportResult:
    """Background export payload returned to the UI thread."""

    summary: dict[str, object]
    preview: ClusterSelectionPreview
    written_files: list[Path]
    analyzed_frames: int
    total_clusters: int
    output_dir: Path
    metadata_path: Path | None = None
    resumed: bool = False
    already_complete: bool = False
    previously_completed_frames: int = 0
    newly_processed_frames: int = 0


class ClusterProgressDialog(QDialog):
    """Popup progress window for cluster extraction."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._start_time = monotonic()
        self._build_ui()

    def _build_ui(self) -> None:
        self.setWindowTitle("Cluster Extraction Progress")
        self.setModal(False)
        self.resize(540, 130)

        layout = QVBoxLayout(self)

        self.phase_label = QLabel("Processing extracted frames...")
        layout.addWidget(self.phase_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%v / %m frames")
        layout.addWidget(self.progress_bar)

        self.meter_label = QLabel()
        self.meter_label.setStyleSheet("font-family: monospace;")
        self.meter_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout.addWidget(self.meter_label)

    def begin(self, total_frames: int, phase: str = "extracting") -> None:
        self._start_time = monotonic()
        total = max(int(total_frames), 1)
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%v / %m frames")
        self.set_phase(phase)
        self._update_meter(0, total)
        self.show()
        self.raise_()

    def update_progress(self, processed: int, total: int) -> None:
        total = max(int(total), 1)
        processed = max(0, min(int(processed), total))
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(processed)
        self._update_meter(processed, total)

    def set_phase(self, phase: str) -> None:
        if phase == "sorting":
            self.phase_label.setText(
                "Sorting cluster files into stoichiometry folders..."
            )
        else:
            self.phase_label.setText("Processing extracted frames...")

    def _update_meter(self, processed: int, total: int) -> None:
        elapsed = max(monotonic() - self._start_time, 0.0)
        self.meter_label.setText(_format_tqdm_meter(processed, total, elapsed))


class ClusterInspectionWorker(QObject):
    """Background worker for extracted frame-folder inspection."""

    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, frames_dir: Path) -> None:
        super().__init__()
        self.frames_dir = frames_dir

    @Slot()
    def run(self) -> None:
        try:
            analyzer = ExtractedFrameFolderClusterAnalyzer(
                frames_dir=self.frames_dir,
                atom_type_definitions={},
                pair_cutoffs_def={},
            )
            self.finished.emit(analyzer.inspect())
        except Exception as exc:
            self.failed.emit(str(exc))


class ClusterExportWorker(QObject):
    """Background worker for cluster analysis and export."""

    progress = Signal(str)
    phase_changed = Signal(str)
    progress_count = Signal(int, int)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, config: ClusterJobConfig) -> None:
        super().__init__()
        self.config = config

    @Slot()
    def run(self) -> None:
        try:
            current_phase = "extracting"
            stoichiometry_bins_text = (
                "solute + shell atoms"
                if self.config.include_shell_atoms_in_stoichiometry
                else "solute only"
            )
            analyzer = ExtractedFrameFolderClusterAnalyzer(
                frames_dir=self.config.frames_dir,
                atom_type_definitions=self.config.atom_type_definitions,
                pair_cutoffs_def=self.config.pair_cutoff_definitions,
                box_dimensions=self.config.box_dimensions,
                default_cutoff=self.config.default_cutoff,
                use_pbc=self.config.use_pbc,
                smart_solvation_shells=self.config.smart_solvation_shells,
                include_shell_atoms_in_stoichiometry=(
                    self.config.include_shell_atoms_in_stoichiometry
                ),
                search_mode=self.config.search_mode,
                save_state_frequency=self.config.save_state_frequency,
            )
            summary = analyzer.inspect()
            preview = estimate_selection(
                total_frames=int(summary["n_frames"]),
                first_frame_name=summary.get("first_frame"),
                last_frame_name=summary.get("last_frame"),
            )
            frame_format = str(summary.get("frame_format", "pdb"))
            mode_label = str(
                summary.get("mode_label", frame_folder_label(frame_format))
            )
            output_suffix = str(
                summary.get(
                    "output_file_extension",
                    frame_output_suffix(frame_format),
                )
            )
            self.progress.emit(
                "Inspection complete. "
                f"Found {summary['n_frames']} extracted frame(s) in the "
                f"selected folder in {mode_label} mode."
            )
            self.progress.emit(
                "Preparing cluster analysis for "
                f"{preview.selected_frames} frame(s)."
            )
            self.progress_count.emit(0, preview.selected_frames)
            if self.config.shell_levels:
                shell_text = ", ".join(
                    str(level) for level in self.config.shell_levels
                )
                self.progress.emit(f"Growing shell levels: {shell_text}.")
            else:
                self.progress.emit("Running core-only cluster analysis.")
            self.progress.emit(
                f"Stoichiometry bins: {stoichiometry_bins_text}."
            )
            self.progress.emit(
                "Smart solvation shells: "
                f"{'on' if self.config.smart_solvation_shells and frame_format == 'pdb' else 'off'}."
            )
            self.progress.emit(
                "Periodic boundary conditions: "
                f"{'on' if self.config.use_pbc else 'off'}."
            )
            self.progress.emit(
                "Search mode: "
                f"{format_search_mode_label(self.config.search_mode)}."
            )
            self.progress.emit(
                "Save-state frequency: "
                f"{_save_state_frequency_label(self.config.save_state_frequency)}."
            )
            self.progress.emit(
                f"Writing cluster {output_suffix} files under: "
                f"{self.config.output_dir}"
            )
            progress_log_interval = (
                1
                if preview.selected_frames <= 10
                else max(preview.selected_frames // 8, 25)
            )

            def on_phase_changed(
                phase: str,
                processed: int,
                total: int,
            ) -> None:
                nonlocal current_phase
                current_phase = str(phase)
                self.phase_changed.emit(current_phase)
                self.progress_count.emit(processed, total)
                if current_phase == "sorting":
                    self.progress.emit(
                        "Frame extraction complete. Sorting cluster files "
                        "into stoichiometry folders..."
                    )
                else:
                    self.progress.emit("Extracting per-frame cluster files...")

            def on_progress(
                processed: int,
                total: int,
                frame_label: str,
            ) -> None:
                remaining = max(total - processed, 0)
                self.progress_count.emit(processed, total)
                if frame_label == "resume":
                    return
                should_log = (
                    processed == 1
                    or processed >= total
                    or processed % progress_log_interval == 0
                )
                if should_log:
                    if current_phase == "sorting":
                        self.progress.emit(
                            f"Sorted {processed} of {total} frame(s); "
                            f"{remaining} remaining. "
                            f"Last frame: {frame_label}."
                        )
                    else:
                        self.progress.emit(
                            f"Processed {processed} of {total} frame(s); "
                            f"{remaining} remaining. "
                            f"Last frame: {frame_label}."
                        )

            export = analyzer.export_cluster_files(
                self.config.output_dir,
                shell_levels=self.config.shell_levels,
                include_shell_levels=self.config.include_shell_levels,
                shared_shells=self.config.shared_shells,
                progress_callback=on_progress,
                phase_callback=on_phase_changed,
            )
            if export.already_complete:
                self.progress.emit(
                    "Matching metadata found. This extraction was already "
                    "completed, so existing cluster files were reused."
                )
            elif export.resumed:
                self.progress.emit(
                    "Matching metadata found. Resuming extraction from "
                    f"{export.previously_completed_frames} previously "
                    "completed frame(s)."
                )
            else:
                self.progress.emit("No prior extraction metadata found.")
            total_clusters = sum(
                frame_result.n_clusters
                for frame_result in export.frame_results
            )
            self.finished.emit(
                ClusterExportResult(
                    summary=summary,
                    preview=preview,
                    written_files=export.written_files,
                    analyzed_frames=len(export.frame_results),
                    total_clusters=total_clusters,
                    output_dir=self.config.output_dir,
                    metadata_path=export.metadata_path,
                    resumed=export.resumed,
                    already_complete=export.already_complete,
                    previously_completed_frames=(
                        export.previously_completed_frames
                    ),
                    newly_processed_frames=export.newly_processed_frames,
                )
            )
        except Exception as exc:
            self.failed.emit(str(exc))


_OPEN_WINDOWS: list["ClusterMainWindow"] = []


class ClusterMainWindow(QMainWindow):
    """Main Qt6 window for the cluster extraction application."""

    project_paths_registered = Signal(object)

    def __init__(
        self,
        initial_frames_dir: Path | None = None,
        initial_project_dir: Path | None = None,
    ) -> None:
        super().__init__()
        self._project_manager = SAXSProjectManager()
        self._project_dir = (
            None
            if initial_project_dir is None
            else Path(initial_project_dir).expanduser().resolve()
        )
        self._last_summary: dict[str, object] | None = None
        self._frame_format: str | None = None
        self._inspect_thread: QThread | None = None
        self._inspect_worker: ClusterInspectionWorker | None = None
        self._export_thread: QThread | None = None
        self._export_worker: ClusterExportWorker | None = None
        self._progress_dialog: ClusterProgressDialog | None = None
        self._export_phase = "idle"
        self._project_xyz_frames_dir: Path | None = None
        self._project_pdb_frames_dir: Path | None = None
        self._active_project_frames_kind = "xyz"
        self._build_ui()
        self._refresh_project_frame_sources()

        default_frames_dir = (
            self._project_xyz_frames_dir
            or initial_frames_dir
            or self._project_pdb_frames_dir
        )
        if default_frames_dir is not None:
            self.trajectory_panel.frames_dir_edit.setText(
                str(default_frames_dir)
            )
            self._update_suggested_output_dir(default_frames_dir)

    def closeEvent(self, event) -> None:
        if (
            self._inspect_thread is not None
            and self._inspect_thread.isRunning()
        ) or (
            self._export_thread is not None and self._export_thread.isRunning()
        ):
            QMessageBox.warning(
                self,
                "Cluster Extraction",
                "Please wait for the current frames inspection or cluster "
                "export to finish before closing this window.",
            )
            event.ignore()
            return
        super().closeEvent(event)

    def _build_ui(self) -> None:
        self.setWindowTitle("SAXSShell (cluster)")
        self.setWindowIcon(load_saxshell_icon())
        self.resize(1360, 860)

        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        self.project_banner = None

        splitter = QSplitter(Qt.Orientation.Horizontal)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(12)

        self.trajectory_panel = ClusterTrajectoryPanel()
        self.export_panel = ClusterExportPanel()
        self.definitions_panel = ClusterDefinitionsPanel()

        left_layout.addWidget(self.trajectory_panel)
        left_layout.addWidget(self.export_panel)
        left_layout.addStretch(1)

        splitter.addWidget(self._wrap_scroll_area(left))
        splitter.addWidget(self._wrap_scroll_area(self.definitions_panel))
        splitter.setSizes([520, 820])

        root.addWidget(splitter)
        self.setCentralWidget(central)
        self.project_status_label = self._build_project_status_label()
        if self.project_status_label is not None:
            self.statusBar().addPermanentWidget(self.project_status_label)
        self.statusBar().showMessage("Ready")

        self.trajectory_panel.inspect_requested.connect(
            self.inspect_frames_folder
        )
        self.trajectory_panel.settings_changed.connect(
            self._refresh_selection_preview
        )
        self.trajectory_panel.frames_dir_changed.connect(
            self._on_frames_dir_changed
        )
        self.trajectory_panel.project_source_changed.connect(
            self._on_project_source_changed
        )
        self.trajectory_panel.frames_dir_edit.editingFinished.connect(
            lambda: self._register_project_paths(
                **self._current_project_input_registration()
            )
        )
        self.export_panel.settings_changed.connect(
            self._refresh_selection_preview
        )
        self.export_panel.export_requested.connect(self.export_clusters)
        self.definitions_panel.settings_changed.connect(
            self._refresh_selection_preview
        )

        self.export_panel.set_selection_summary(
            "Select an extracted PDB or XYZ frames folder and define cluster "
            "rules to preview the analysis run."
        )
        self.export_panel.set_log(
            "Ready. Use mdtrajectory to extract frames first, then inspect "
            "the extracted PDB or XYZ frames folder here before running "
            "cluster analysis."
        )
        self._set_frame_format(None)

    def _load_project_settings(self) -> ProjectSettings | None:
        if self._project_dir is None:
            return None
        project_file = build_project_paths(self._project_dir).project_file
        if not project_file.is_file():
            return None
        try:
            return self._project_manager.load_project(self._project_dir)
        except Exception:
            return None

    def _project_status_text(self) -> str | None:
        if self._project_dir is None:
            return None
        return f"Active project: {self._project_dir}"

    def _project_status_tooltip(self) -> str | None:
        if self._project_dir is None:
            return None
        settings = self._load_project_settings()
        project_name = (
            self._project_dir.name
            if settings is None
            else settings.project_name.strip() or self._project_dir.name
        )
        return (
            f"Active project: {project_name}\n"
            f"{self._project_dir}\n\n"
            "This window is linked to the active SAXS project, so saved "
            "XYZ frames, optional PDB structure, and clusters folders are "
            "written back to that project when you inspect or export data "
            "here."
        )

    def _build_project_status_label(
        self,
    ) -> CompactProjectStatusLabel | None:
        status_text = self._project_status_text()
        if status_text is None:
            return None
        label = CompactProjectStatusLabel(self.statusBar())
        label.setToolTip(self._project_status_tooltip() or "")
        label.set_full_text(status_text)
        return label

    def _refresh_project_frame_sources(self) -> None:
        settings = self._load_project_settings()
        self._project_xyz_frames_dir = (
            None if settings is None else settings.resolved_frames_dir
        )
        self._project_pdb_frames_dir = (
            None if settings is None else settings.resolved_pdb_frames_dir
        )
        if (
            self._project_xyz_frames_dir is None
            and self._project_pdb_frames_dir
        ):
            self._active_project_frames_kind = "pdb"
        elif (
            self._active_project_frames_kind == "pdb"
            and self._project_pdb_frames_dir is None
        ):
            self._active_project_frames_kind = "xyz"
        self.trajectory_panel.set_project_frame_sources(
            self._project_xyz_frames_dir,
            self._project_pdb_frames_dir,
            active_kind=self._active_project_frames_kind,
        )

    def _on_project_source_changed(self, kind: object) -> None:
        if kind is None:
            return
        self._active_project_frames_kind = str(kind)

    def _current_project_input_registration(self) -> dict[str, Path | None]:
        frames_dir = self.trajectory_panel.get_frames_dir()
        if self._active_project_frames_kind == "pdb":
            return {"pdb_frames_dir": frames_dir}
        return {"frames_dir": frames_dir}

    def _emit_project_paths_registered(
        self,
        *,
        frames_dir: Path | None = None,
        pdb_frames_dir: Path | None = None,
        clusters_dir: Path | None = None,
    ) -> None:
        if self._project_dir is None:
            return
        payload: dict[str, object] = {
            "project_dir": Path(self._project_dir).expanduser().resolve(),
        }
        if frames_dir is not None:
            payload["frames_dir"] = Path(frames_dir).expanduser().resolve()
        if pdb_frames_dir is not None:
            payload["pdb_frames_dir"] = (
                Path(pdb_frames_dir).expanduser().resolve()
            )
        if clusters_dir is not None:
            payload["clusters_dir"] = Path(clusters_dir).expanduser().resolve()
        self.project_paths_registered.emit(payload)

    def _register_project_paths(
        self,
        *,
        frames_dir: Path | None = None,
        pdb_frames_dir: Path | None = None,
        clusters_dir: Path | None = None,
    ) -> str | None:
        settings = self._load_project_settings()
        if settings is None:
            return None
        try:
            settings.frames_dir = (
                None
                if frames_dir is None
                else str(Path(frames_dir).expanduser().resolve())
            )
            if pdb_frames_dir is not None:
                settings.pdb_frames_dir = str(
                    Path(pdb_frames_dir).expanduser().resolve()
                )
            if clusters_dir is not None:
                settings.clusters_dir = str(
                    Path(clusters_dir).expanduser().resolve()
                )
            self._project_manager.save_project(settings)
            self._emit_project_paths_registered(
                frames_dir=frames_dir,
                pdb_frames_dir=pdb_frames_dir,
                clusters_dir=clusters_dir,
            )
            self._refresh_project_frame_sources()
        except Exception as exc:
            return (
                "The project frames/PDB/clusters references could not be "
                f"updated: {exc}"
            )
        updates: list[str] = []
        if frames_dir is not None:
            updates.append(f"frames={Path(frames_dir).expanduser().resolve()}")
        if pdb_frames_dir is not None:
            updates.append(
                "pdb_frames=" f"{Path(pdb_frames_dir).expanduser().resolve()}"
            )
        if clusters_dir is not None:
            updates.append(
                f"clusters={Path(clusters_dir).expanduser().resolve()}"
            )
        if not updates:
            return None
        return "Updated project folder references: " + ", ".join(updates)

    def inspect_frames_folder(self) -> None:
        try:
            if self._inspect_thread is not None:
                return

            frames_dir = self.trajectory_panel.get_frames_dir()
            if frames_dir is None:
                raise ValueError("No extracted frames folder selected.")
            registration_message = self._register_project_paths(
                **self._current_project_input_registration()
            )

            self.export_panel.set_log(
                "Inspection request received.\n"
                f"Loading extracted frames from: {frames_dir}"
            )
            if registration_message is not None:
                self.export_panel.append_log(registration_message)
            self.trajectory_panel.set_summary_text(
                "Inspecting extracted frames folder..."
            )
            self.export_panel.set_selection_summary(
                "Inspecting extracted frames folder..."
            )
            self.statusBar().showMessage("Inspecting extracted frames...")
            self._start_inspection_worker(frames_dir)
        except Exception as exc:
            self._handle_error(
                "Frames-folder inspection failed",
                str(exc),
            )

    def export_clusters(self) -> None:
        try:
            if self._export_thread is not None:
                return
            config = self._build_job_config()
            stoichiometry_bins_text = (
                "solute + shell atoms"
                if config.include_shell_atoms_in_stoichiometry
                else "solute only"
            )
            mode_text = (
                frame_folder_label(self._frame_format)
                if self._frame_format is not None
                else "Auto-detect"
            )
            self.export_panel.set_log(
                "Cluster export request received.\n"
                f"Frames folder: {config.frames_dir}\n"
                f"Mode: {mode_text}\n"
                f"PBC: {'on' if config.use_pbc else 'off'}\n"
                "Smart solvation shells: "
                f"{'on' if config.smart_solvation_shells else 'off'}\n"
                "Search mode: "
                f"{format_search_mode_label(config.search_mode)}\n"
                "Save-state frequency: "
                f"{_save_state_frequency_label(config.save_state_frequency)}\n"
                f"Stoichiometry bins: {stoichiometry_bins_text}\n"
                f"Output directory: {config.output_dir}"
            )
            self.export_panel.reset_progress()
            self._export_phase = "extracting"
            self.export_panel.set_progress_phase(self._export_phase)
            self._show_progress_dialog(
                total_frames=(
                    int(self._last_summary.get("n_frames", 0))
                    if self._last_summary is not None
                    else 0
                )
            )
            self.export_panel.append_log(
                "Preparing cluster analysis configuration..."
            )
            self.statusBar().showMessage("Extracting clusters...")
            self._start_export_worker(config)
        except Exception as exc:
            self._handle_error(
                "Cluster export failed",
                str(exc),
            )

    def _build_job_config(self) -> ClusterJobConfig:
        frames_dir = self.trajectory_panel.get_frames_dir()
        if frames_dir is None:
            raise ValueError("No extracted frames folder selected.")

        atom_type_definitions = self.definitions_panel.atom_type_definitions()
        if not atom_type_definitions:
            raise ValueError(
                "Add at least one atom-type definition before exporting."
            )
        if not (
            atom_type_definitions.get("node")
            or atom_type_definitions.get("linker")
        ):
            raise ValueError("Define at least one node or linker atom type.")

        pair_cutoff_definitions = (
            self.definitions_panel.pair_cutoff_definitions()
        )
        default_cutoff = self.definitions_panel.default_cutoff()
        if not pair_cutoff_definitions and default_cutoff is None:
            raise ValueError(
                "Add at least one pair-cutoff definition or specify a "
                "default cutoff."
            )

        use_pbc = self.definitions_panel.use_pbc()
        manual_box_dimensions = self.definitions_panel.box_dimensions()
        resolved_box_dimensions = manual_box_dimensions
        if use_pbc and resolved_box_dimensions is None:
            resolved_box_dimensions = self._detected_box_dimensions()
            if resolved_box_dimensions is None:
                raise ValueError(
                    "Periodic boundary conditions are enabled, but no box "
                    "dimensions are available. Enter a manual box or inspect "
                    "a frames folder with a usable coordinate extent."
                )

        output_dir = self.export_panel.get_output_dir()
        if output_dir is None:
            raise ValueError("No output directory selected.")

        return ClusterJobConfig(
            frames_dir=frames_dir,
            atom_type_definitions=atom_type_definitions,
            pair_cutoff_definitions=pair_cutoff_definitions,
            box_dimensions=resolved_box_dimensions,
            use_pbc=use_pbc,
            search_mode=self.definitions_panel.search_mode(),
            save_state_frequency=self.definitions_panel.save_state_frequency(),
            default_cutoff=default_cutoff,
            shell_levels=self.definitions_panel.shell_growth_levels(),
            include_shell_levels=self.definitions_panel.include_shell_levels(),
            shared_shells=self.definitions_panel.shared_shells(),
            smart_solvation_shells=(
                self._frame_format == "pdb"
                and self.definitions_panel.smart_solvation_shells()
            ),
            include_shell_atoms_in_stoichiometry=(
                self.definitions_panel.include_shell_atoms_in_stoichiometry()
            ),
            output_dir=output_dir,
        )

    def _start_inspection_worker(self, frames_dir: Path) -> None:
        self._inspect_thread = QThread(self)
        self._inspect_worker = ClusterInspectionWorker(frames_dir)
        self._inspect_worker.moveToThread(self._inspect_thread)

        self._inspect_thread.started.connect(self._inspect_worker.run)
        self._inspect_worker.finished.connect(self._on_inspection_finished)
        self._inspect_worker.failed.connect(self._on_inspection_failed)
        self._inspect_worker.finished.connect(self._inspect_thread.quit)
        self._inspect_worker.failed.connect(self._inspect_thread.quit)
        self._inspect_thread.finished.connect(self._cleanup_inspection_thread)
        self._inspect_thread.finished.connect(self._inspect_thread.deleteLater)
        self._inspect_thread.finished.connect(self._inspect_worker.deleteLater)
        self._inspect_thread.start()

    def _start_export_worker(self, config: ClusterJobConfig) -> None:
        self._export_thread = QThread(self)
        self._export_worker = ClusterExportWorker(config)
        self._export_worker.moveToThread(self._export_thread)

        self._export_thread.started.connect(self._export_worker.run)
        self._export_worker.progress.connect(self.export_panel.append_log)
        self._export_worker.phase_changed.connect(
            self._on_export_phase_changed
        )
        self._export_worker.progress_count.connect(self._on_export_progress)
        self._export_worker.finished.connect(self._on_export_finished)
        self._export_worker.failed.connect(self._on_export_failed)
        self._export_worker.finished.connect(self._export_thread.quit)
        self._export_worker.failed.connect(self._export_thread.quit)
        self._export_thread.finished.connect(self._cleanup_export_thread)
        self._export_thread.finished.connect(self._export_thread.deleteLater)
        self._export_thread.finished.connect(self._export_worker.deleteLater)
        self._export_thread.start()

    def _on_inspection_finished(self, summary: dict[str, object]) -> None:
        self._last_summary = summary
        self._set_frame_format(summary.get("frame_format"))
        self.trajectory_panel.set_summary(summary)
        self.export_panel.append_log(
            "Inspection complete. "
            f"Detected {summary['n_frames']} extracted frame(s) in the "
            f"selected folder in {summary.get('mode_label', 'detected')}."
        )
        self.statusBar().showMessage("Inspection complete")
        self._refresh_selection_preview()

    def _on_inspection_failed(self, message: str) -> None:
        self.statusBar().showMessage("Frames-folder inspection failed")
        self._handle_error("Frames-folder inspection failed", message)

    def _on_export_finished(self, result: ClusterExportResult) -> None:
        self._last_summary = result.summary
        self._set_frame_format(result.summary.get("frame_format"))
        self.trajectory_panel.set_summary(result.summary)
        output_suffix = str(
            result.summary.get(
                "output_file_extension",
                frame_output_suffix(self._frame_format or "pdb"),
            )
        )
        self.export_panel.append_log(
            "Cluster analysis complete.\n"
            f"Frames analyzed: {result.analyzed_frames}\n"
            f"Clusters found: {result.total_clusters}\n"
            f"Cluster {output_suffix} files written: "
            f"{len(result.written_files)}\n"
            f"Output path: {result.output_dir}"
        )
        if result.already_complete:
            self.export_panel.append_log(
                "Resume status: extraction already complete; reused "
                "existing cluster files."
            )
        elif result.resumed:
            self.export_panel.append_log(
                "Resume status: resumed from "
                f"{result.previously_completed_frames} previously "
                "completed frame(s)."
            )
        else:
            self.export_panel.append_log("Resume status: new extraction.")
        self.export_panel.append_log(
            f"Newly processed frames: {result.newly_processed_frames}"
        )
        if result.metadata_path is not None:
            self.export_panel.append_log(
                f"Metadata file: {result.metadata_path}"
            )
        if result.written_files:
            self.export_panel.append_log(
                f"First written file: {result.written_files[0]}"
            )
        self._on_export_progress(
            result.analyzed_frames,
            result.analyzed_frames,
        )
        self._close_progress_dialog()
        self.statusBar().showMessage("Extraction Complete!")
        self._refresh_selection_preview()
        registration_message = self._register_project_paths(
            **self._current_project_input_registration(),
            clusters_dir=result.output_dir,
        )
        if registration_message is not None:
            self.export_panel.append_log(registration_message)
        next_dir = next_available_output_dir(
            result.output_dir.parent,
            result.output_dir.name,
        )
        self.export_panel.suggest_output_dir(next_dir)

    def _on_export_failed(self, message: str) -> None:
        self._close_progress_dialog()
        self.statusBar().showMessage("Cluster extraction failed")
        self._handle_error("Cluster export failed", message)

    def _cleanup_inspection_thread(self) -> None:
        self._inspect_thread = None
        self._inspect_worker = None

    def _cleanup_export_thread(self) -> None:
        self._export_thread = None
        self._export_worker = None
        self._export_phase = "idle"

    def _ensure_progress_dialog(self) -> ClusterProgressDialog:
        if self._progress_dialog is None:
            self._progress_dialog = ClusterProgressDialog(self)
        return self._progress_dialog

    def _show_progress_dialog(self, total_frames: int) -> None:
        dialog = self._ensure_progress_dialog()
        dialog.begin(total_frames, phase=self._export_phase)

    def _close_progress_dialog(self) -> None:
        if self._progress_dialog is not None:
            self._progress_dialog.close()

    def _on_export_progress(self, processed: int, total: int) -> None:
        self.export_panel.update_progress(processed, total)
        if self._progress_dialog is not None:
            self._progress_dialog.update_progress(processed, total)
        if self._export_phase == "sorting":
            self.statusBar().showMessage(
                f"Sorting cluster files... {processed}/{max(total, 1)} frames"
            )
        else:
            self.statusBar().showMessage(
                f"Extracting clusters... {processed}/{max(total, 1)} frames"
            )

    def _on_export_phase_changed(self, phase: str) -> None:
        self._export_phase = str(phase)
        self.export_panel.set_progress_phase(self._export_phase)
        if self._progress_dialog is not None:
            self._progress_dialog.set_phase(self._export_phase)
        if self._export_phase == "sorting":
            self.statusBar().showMessage("Sorting cluster files...")
        else:
            self.statusBar().showMessage("Extracting clusters...")

    def _estimated_box_dimensions(
        self,
    ) -> tuple[float, float, float] | None:
        if self._last_summary is None:
            return None
        value = self._last_summary.get("box_dimensions")
        if value is None:
            value = self._last_summary.get("estimated_box_dimensions")
        if value is None:
            return None
        return tuple(float(component) for component in value)

    def _detected_box_dimensions(
        self,
    ) -> tuple[float, float, float] | None:
        return self._estimated_box_dimensions()

    def _box_dimensions_source_kind(self) -> str | None:
        if self._last_summary is None:
            return None
        value = self._last_summary.get("box_dimensions_source_kind")
        return None if value is None else str(value)

    def _box_dimensions_source(self) -> str | None:
        if self._last_summary is None:
            return None
        value = self._last_summary.get("box_dimensions_source")
        return None if value is None else str(value)

    def _box_dimensions_label(self) -> str:
        if self._box_dimensions_source_kind() == "source_filename":
            return "Source box dimensions"
        return "Estimated box dimensions"

    def _sync_box_dimensions_from_summary(
        self,
        summary: dict[str, object] | None,
    ) -> None:
        if summary is None:
            self.definitions_panel.set_box_dimensions(None, emit_signal=False)
            return
        if summary.get("box_dimensions_source_kind") == "source_filename":
            value = summary.get("box_dimensions")
            if value is not None:
                self.definitions_panel.set_box_dimensions(
                    tuple(float(component) for component in value),
                    emit_signal=False,
                )
                return
        self.definitions_panel.set_box_dimensions(None, emit_signal=False)

    def _set_frame_format(self, frame_format: object | None) -> None:
        normalized = None if frame_format is None else str(frame_format)
        self._frame_format = normalized
        self.trajectory_panel.set_frame_mode(normalized)
        self.definitions_panel.set_frame_mode(normalized)
        self.export_panel.set_frame_mode(normalized)

    def _detect_frame_format(
        self,
        frames_dir: Path | None,
    ) -> tuple[str | None, str | None]:
        if frames_dir is None:
            return None, None
        try:
            frame_format, _frame_paths = detect_frame_folder_mode(frames_dir)
        except ValueError as exc:
            return None, str(exc)
        return frame_format, None

    def _on_frames_dir_changed(self, frames_dir: Path | None) -> None:
        self._last_summary = None
        if frames_dir is None:
            self._sync_box_dimensions_from_summary(None)
            self._set_frame_format(None)
            self.trajectory_panel.set_summary_text("")
            self._refresh_selection_preview()
            return

        try:
            analyzer = ExtractedFrameFolderClusterAnalyzer(
                frames_dir=frames_dir,
                atom_type_definitions={},
                pair_cutoffs_def={},
            )
            self._last_summary = analyzer.inspect()
            self._sync_box_dimensions_from_summary(self._last_summary)
            frame_format = str(self._last_summary.get("frame_format"))
            self._set_frame_format(frame_format)
            self.trajectory_panel.set_summary(self._last_summary)
        except ValueError as exc:
            self._sync_box_dimensions_from_summary(None)
            frame_format, detail = self._detect_frame_format(frames_dir)
            self._set_frame_format(frame_format)
            self.trajectory_panel.set_summary_text(str(exc))
            if detail is not None:
                self.trajectory_panel.set_frame_mode(None, detail=detail)
        self._update_suggested_output_dir(frames_dir)
        self._refresh_selection_preview()

    def _refresh_selection_preview(self) -> None:
        frames_dir = self.trajectory_panel.get_frames_dir()
        output_dir = self.export_panel.get_output_dir()

        if frames_dir is None:
            self.export_panel.set_selection_summary(
                "Select an extracted PDB or XYZ frames folder to preview "
                "the cluster analysis run."
            )
            return

        if output_dir is None:
            self._update_suggested_output_dir(frames_dir)
            output_dir = self.export_panel.get_output_dir()

        rule_warning: str | None = None
        try:
            atom_rule_count, pair_rule_count = (
                self.definitions_panel.rule_counts()
            )
        except ValueError as exc:
            atom_rule_count = 0
            pair_rule_count = 0
            rule_warning = str(exc)
        try:
            manual_box_dimensions = self.definitions_panel.box_dimensions()
        except ValueError as exc:
            manual_box_dimensions = None
            rule_warning = (
                str(exc) if rule_warning is None else f"{rule_warning}; {exc}"
            )

        if self._frame_format is None:
            detected_format, detect_warning = self._detect_frame_format(
                frames_dir
            )
            if detected_format is not None:
                self._set_frame_format(detected_format)
            elif detect_warning is not None and rule_warning is None:
                rule_warning = detect_warning

        if (
            self.definitions_panel.use_pbc()
            and manual_box_dimensions is None
            and self._detected_box_dimensions() is None
        ):
            pbc_warning = (
                "Periodic boundary conditions are enabled, but no manual or "
                "estimated box dimensions are available."
            )
            rule_warning = (
                pbc_warning
                if rule_warning is None
                else f"{rule_warning}; {pbc_warning}"
            )

        if self._last_summary is None:
            include_shell_stoichiometry = (
                self.definitions_panel.include_shell_atoms_in_stoichiometry()
            )
            stoichiometry_bins_text = (
                "solute + shell atoms"
                if include_shell_stoichiometry
                else "solute only"
            )
            mode_text = (
                frame_folder_label(self._frame_format)
                if self._frame_format is not None
                else "Auto-detect"
            )
            estimated_box_text = format_box_dimensions(
                self._detected_box_dimensions()
            )
            search_mode_label = format_search_mode_label(
                self.definitions_panel.search_mode()
            )
            preview_text = (
                "Inspect the selected frames folder to confirm the number "
                "of extracted frame files.\n"
                f"Mode: {mode_text}\n"
                f"PBC: {'on' if self.definitions_panel.use_pbc() else 'off'}\n"
                "Smart solvation shells: "
                f"{'on' if self._frame_format == 'pdb' and self.definitions_panel.smart_solvation_shells() else 'off'}\n"
                f"Search mode: {search_mode_label}\n"
                "Save-state frequency: "
                f"{_save_state_frequency_label(self.definitions_panel.save_state_frequency())}\n"
                f"Stoichiometry bins: {stoichiometry_bins_text}\n"
                f"{self._box_dimensions_label()}: {estimated_box_text}\n"
                f"Atom-type rules: {atom_rule_count}\n"
                f"Pair-cutoff rules: {pair_rule_count}"
            )
            box_source = self._box_dimensions_source()
            if box_source is not None:
                preview_text = f"{preview_text}\nBox source: {box_source}"
            if output_dir is not None:
                output_suffix = (
                    frame_output_suffix(self._frame_format)
                    if self._frame_format is not None
                    else "auto"
                )
                preview_text = (
                    f"{preview_text}\nOutput format: {output_suffix}\n"
                    f"Output folder: {output_dir.name}\n"
                    f"Output path: {output_dir}"
                )
            if rule_warning is not None:
                preview_text = (
                    f"{preview_text}\nValidation warning: {rule_warning}"
                )
            self.export_panel.set_selection_summary(preview_text)
            return

        preview = estimate_selection(
            total_frames=int(self._last_summary.get("n_frames", 0)),
            first_frame_name=self._last_summary.get("first_frame"),
            last_frame_name=self._last_summary.get("last_frame"),
        )
        summary_text = self._format_selection_summary(preview, output_dir)
        shared_shells_text = (
            "yes" if self.definitions_panel.shared_shells() else "no"
        )
        summary_text = (
            f"{summary_text}\nAtom-type rules: {atom_rule_count}\n"
            f"Pair-cutoff rules: {pair_rule_count}\n"
            f"Shell levels to grow: {self._shell_growth_text()}\n"
            "Smart solvation shells: "
            f"{'on' if self._frame_format == 'pdb' and self.definitions_panel.smart_solvation_shells() else 'off'}\n"
            f"Shared shells: {shared_shells_text}"
        )
        if rule_warning is not None:
            summary_text = (
                f"{summary_text}\nValidation warning: {rule_warning}"
            )
        if not bool(
            self._last_summary.get("supports_full_molecule_shells", False)
        ):
            summary_text = (
                f"{summary_text}\nMode note: XYZ exports preserve cluster "
                "atoms and shell atoms, but they cannot reconstruct full "
                "molecules the way PDB mode can."
            )
        self.export_panel.set_selection_summary(summary_text)

    def _format_selection_summary(
        self,
        preview: ClusterSelectionPreview,
        output_dir: Path | None = None,
    ) -> str:
        stoichiometry_bins_text = (
            "solute + shell atoms"
            if self.definitions_panel.include_shell_atoms_in_stoichiometry()
            else "solute only"
        )
        mode_text = (
            frame_folder_label(self._frame_format)
            if self._frame_format is not None
            else "Auto-detect"
        )
        search_mode_label = format_search_mode_label(
            self.definitions_panel.search_mode()
        )
        lines = [
            (
                "Project source: PDB structure folder"
                if self._active_project_frames_kind == "pdb"
                else "Project source: XYZ frames folder"
            ),
            f"Mode: {mode_text}",
            f"PBC: {'on' if self.definitions_panel.use_pbc() else 'off'}",
            "Smart solvation shells: "
            f"{'on' if self._frame_format == 'pdb' and self.definitions_panel.smart_solvation_shells() else 'off'}",
            f"Search mode: {search_mode_label}",
            "Save-state frequency: "
            f"{_save_state_frequency_label(self.definitions_panel.save_state_frequency())}",
            f"Stoichiometry bins: {stoichiometry_bins_text}",
            f"Frames in folder: {preview.total_frames}",
            f"Frames selected: {preview.selected_frames}",
        ]
        estimated_box = self._detected_box_dimensions()
        estimated_box_text = format_box_dimensions(estimated_box)
        lines.append(f"{self._box_dimensions_label()}: {estimated_box_text}")
        box_source = self._box_dimensions_source()
        if box_source is not None:
            lines.append(f"Box source: {box_source}")
        if self.definitions_panel.use_pbc():
            try:
                manual_box = self.definitions_panel.box_dimensions()
            except ValueError:
                manual_box = None
            resolved_box = manual_box or estimated_box
            resolved_label = (
                f"{format_box_dimensions(resolved_box)}"
                if manual_box is not None
                else f"{format_box_dimensions(resolved_box)} (auto)"
            )
            lines.append(f"Resolved box dimensions: {resolved_label}")
        else:
            lines.append("Resolved box dimensions: not used")
        if preview.selected_frames:
            lines.append(
                "Frame file range: "
                f"{preview.first_frame_name} to {preview.last_frame_name}"
            )
        else:
            lines.append("Frame file range: none")
        if output_dir is not None:
            if self._frame_format is not None:
                lines.append(
                    f"Output format: {frame_output_suffix(self._frame_format)}"
                )
            lines.append(f"Output folder: {output_dir.name}")
            lines.append(f"Output path: {output_dir}")
        return "\n".join(lines)

    def _shell_growth_text(self) -> str:
        levels = self.definitions_panel.shell_growth_levels()
        if not levels:
            return "core only"
        return ", ".join(str(level) for level in levels)

    def _update_suggested_output_dir(
        self,
        frames_dir: Path | None = None,
    ) -> None:
        frames_dir = frames_dir or self.trajectory_panel.get_frames_dir()
        if frames_dir is None:
            return
        self.export_panel.suggest_output_dir(
            suggest_cluster_output_dir(frames_dir)
        )

    def _handle_error(self, title: str, message: str) -> None:
        self.export_panel.append_log(f"{title}: {message}")
        QMessageBox.critical(self, title, message)

    @staticmethod
    def _wrap_scroll_area(widget: QWidget) -> QScrollArea:
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(widget)
        return scroll_area


def estimate_selection(
    *,
    total_frames: int,
    first_frame_name: object | None = None,
    last_frame_name: object | None = None,
) -> ClusterSelectionPreview:
    """Summarize the extracted frames selected for cluster analysis."""
    return ClusterSelectionPreview(
        total_frames=total_frames,
        selected_frames=total_frames,
        first_frame_name=(
            str(first_frame_name) if first_frame_name is not None else None
        ),
        last_frame_name=(
            str(last_frame_name) if last_frame_name is not None else None
        ),
    )


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "?"
    total_seconds = max(int(seconds), 0)
    minutes, secs = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _format_tqdm_meter(
    processed: int,
    total: int,
    elapsed_seconds: float,
) -> str:
    total = max(int(total), 1)
    processed = max(0, min(int(processed), total))
    fraction = processed / total
    filled = min(int(fraction * 20), 20)
    bar = "#" * filled + "-" * (20 - filled)
    elapsed = max(float(elapsed_seconds), 0.0)

    if processed > 0 and elapsed > 0.0:
        rate = processed / elapsed
        remaining_seconds = (total - processed) / rate if rate > 0.0 else None
        rate_text = f"{rate:0.2f} frame/s"
    else:
        remaining_seconds = None
        rate_text = "? frame/s"

    percentage = int(fraction * 100)
    return (
        f"{percentage:3d}%|{bar}| {processed}/{total} "
        f"[{_format_duration(elapsed)}<{_format_duration(remaining_seconds)}, "
        f"{rate_text}]"
    )


def suggest_cluster_output_dir(frames_dir: str | Path) -> Path:
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


def _base_output_dir_name(frames_dir: Path) -> str:
    folder_label = sub(r"[^0-9A-Za-z]+", "_", frames_dir.name).strip("_")
    if not folder_label:
        folder_label = "frames"
    return f"clusters_{folder_label}"


def launch_cluster_ui(
    frames_dir: str | Path | None = None,
    *,
    project_dir: str | Path | None = None,
) -> int:
    """Launch the Qt6 cluster extraction UI."""
    app = QApplication.instance()
    owns_app = app is None
    if app is None:
        prepare_saxshell_application_identity()
        app = QApplication(sys.argv)
    configure_saxshell_application(app)

    window = ClusterMainWindow(
        initial_frames_dir=(
            Path(frames_dir).expanduser().resolve()
            if frames_dir is not None
            else None
        ),
        initial_project_dir=(
            Path(project_dir).expanduser().resolve()
            if project_dir is not None
            else None
        ),
    )
    _OPEN_WINDOWS.append(window)
    window.show()
    if owns_app:
        return app.exec()
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point for launching the Qt6 cluster extraction UI."""
    parser = argparse.ArgumentParser(
        prog="cluster-ui",
        description=(
            "Launch the SAXSShell cluster extraction UI for extracted PDB "
            "or XYZ frame folders."
        ),
    )
    parser.add_argument(
        "frames_dir",
        nargs="?",
        help=(
            "Optional extracted PDB or XYZ frames folder to prefill in the "
            "UI."
        ),
    )
    args = parser.parse_args(argv)
    return launch_cluster_ui(args.frames_dir)
