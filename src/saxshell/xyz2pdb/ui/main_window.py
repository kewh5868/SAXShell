from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QMessageBox,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

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
from saxshell.xyz2pdb import (
    XYZToPDBExportResult,
    XYZToPDBReferenceUpdateCandidate,
    create_reference_molecule,
    default_reference_library_dir,
    list_reference_library,
    suggest_output_dir,
)
from saxshell.xyz2pdb.mapping_workflow import (
    FreeAtomMappingInput,
    MoleculeMappingInput,
    XYZToPDBEstimateResult,
    XYZToPDBMappingTestResult,
    XYZToPDBMappingWorkflow,
    XYZToPDBSampleAnalysis,
    reference_bond_tolerances,
)
from saxshell.xyz2pdb.ui.export_panel import XYZToPDBExportPanel
from saxshell.xyz2pdb.ui.input_panel import XYZToPDBInputPanel
from saxshell.xyz2pdb.ui.mapping_panel import XYZToPDBMappingPanel
from saxshell.xyz2pdb.ui.reference_panel import ReferenceLibraryPanel
from saxshell.xyz2pdb.ui.reference_update_dialog import (
    AssertionReferenceUpdateDialog,
)
from saxshell.xyz2pdb.workflow import XYZToPDBOperationCancelled


@dataclass(slots=True)
class XYZToPDBJobConfig:
    """Export settings assembled from the UI."""

    input_path: Path
    molecule_inputs: tuple[MoleculeMappingInput, ...]
    free_atom_inputs: tuple[FreeAtomMappingInput, ...]
    hydrogen_mode: str
    selected_solution_index: int
    library_dir: Path | None
    output_dir: Path | None
    assertion_mode: bool = False
    estimate_result: XYZToPDBEstimateResult | None = None


class XYZToPDBExportWorker(QObject):
    """Background worker that runs xyz2pdb export."""

    progress = Signal(str)
    progress_count = Signal(int, int, str)
    finished = Signal(object)
    failed = Signal(str)
    canceled = Signal()

    def __init__(self, config: XYZToPDBJobConfig) -> None:
        super().__init__()
        self.config = config
        self._cancel_requested = False

    @Slot()
    def cancel(self) -> None:
        self._cancel_requested = True

    def is_cancel_requested(self) -> bool:
        return (
            self._cancel_requested
            or QThread.currentThread().isInterruptionRequested()
        )

    @Slot()
    def run(self) -> None:
        try:
            workflow = XYZToPDBMappingWorkflow(
                self.config.input_path,
                reference_library_dir=self.config.library_dir,
                output_dir=self.config.output_dir,
            )
            self.progress.emit("Starting background xyz2pdb conversion.")

            def on_progress(processed: int, total: int, message: str) -> None:
                self.progress_count.emit(processed, total, message)

            result = workflow.export_with_mapping(
                molecule_inputs=self.config.molecule_inputs,
                free_atom_inputs=self.config.free_atom_inputs,
                hydrogen_mode=self.config.hydrogen_mode,
                selected_solution_index=self.config.selected_solution_index,
                output_dir=self.config.output_dir,
                assert_molecule_shapes=self.config.assertion_mode,
                estimate_result=self.config.estimate_result,
                progress_callback=on_progress,
                log_callback=self.progress.emit,
                cancel_callback=self.is_cancel_requested,
            )
            self.finished.emit(result)
        except XYZToPDBOperationCancelled:
            self.canceled.emit()
        except Exception as exc:
            self.failed.emit(str(exc))


class XYZToPDBMainWindow(QMainWindow):
    """Main Qt window for the xyz2pdb application."""

    project_paths_registered = Signal(object)

    def __init__(
        self,
        *,
        initial_project_dir: Path | None = None,
        input_path: Path | None = None,
        config_file: Path | None = None,
        reference_library_dir: Path | None = None,
    ) -> None:
        super().__init__()
        self._project_manager = SAXSProjectManager()
        self._project_dir = (
            None
            if initial_project_dir is None
            else Path(initial_project_dir).expanduser().resolve()
        )
        self._last_analysis: XYZToPDBSampleAnalysis | None = None
        self._last_estimate: XYZToPDBEstimateResult | None = None
        self._last_test_result: XYZToPDBMappingTestResult | None = None
        self._export_thread: QThread | None = None
        self._export_worker: XYZToPDBExportWorker | None = None
        self._closing_after_export_cancel = False
        self._build_ui()
        self._apply_initial_values(
            input_path=input_path,
            config_file=config_file,
            reference_library_dir=reference_library_dir,
        )

    def closeEvent(self, event) -> None:
        self._cancel_export_for_close()
        super().closeEvent(event)

    def _build_ui(self) -> None:
        self.setWindowTitle("SAXSShell (xyz2pdb)")
        self.setWindowIcon(load_saxshell_icon())
        self.resize(1500, 900)

        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        self.input_panel = XYZToPDBInputPanel()
        self.reference_panel = ReferenceLibraryPanel()
        self.mapping_panel = XYZToPDBMappingPanel()
        self.export_panel = XYZToPDBExportPanel()

        self._main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self._main_splitter.setChildrenCollapsible(False)
        self._main_splitter.setHandleWidth(10)
        self._left_splitter = QSplitter(Qt.Orientation.Vertical)
        self._left_splitter.setChildrenCollapsible(False)
        self._left_splitter.setHandleWidth(10)

        self._input_scroll_area = self._wrap_scroll_area(self.input_panel)
        self._reference_scroll_area = self._wrap_scroll_area(
            self.reference_panel
        )
        self._mapping_scroll_area = self._wrap_scroll_area(self.mapping_panel)
        self._export_scroll_area = self._wrap_scroll_area(self.export_panel)

        self._left_splitter.addWidget(self._input_scroll_area)
        self._left_splitter.addWidget(self._reference_scroll_area)
        self._left_splitter.addWidget(self._mapping_scroll_area)
        self._left_splitter.setSizes([180, 280, 440])

        self._main_splitter.addWidget(self._left_splitter)
        self._main_splitter.addWidget(self._export_scroll_area)
        self._main_splitter.setSizes([980, 480])

        root.addWidget(self._main_splitter)
        self.setCentralWidget(central)
        self.project_status_label = self._build_project_status_label()
        if self.project_status_label is not None:
            self.statusBar().addPermanentWidget(self.project_status_label)
        self.statusBar().showMessage("Ready")

        self.input_panel.inspect_requested.connect(self.inspect_input)
        self.input_panel.settings_changed.connect(self._invalidate_results)
        self.input_panel.input_path_changed.connect(
            self._suggest_output_dir_from_input
        )
        self.reference_panel.library_dir_changed.connect(
            self.refresh_reference_library
        )
        self.reference_panel.library_dir_changed.connect(
            lambda _path: self._invalidate_results()
        )
        self.reference_panel.refresh_requested.connect(
            self.refresh_reference_library
        )
        self.reference_panel.selection_changed.connect(
            self.reference_panel.update_details
        )
        self.reference_panel.create_requested.connect(
            self.create_reference_molecule
        )
        self.mapping_panel.settings_changed.connect(self._invalidate_results)
        self.export_panel.estimate_requested.connect(self.estimate_mapping)
        self.export_panel.export_requested.connect(self.export_conversion)
        self.export_panel.cancel_requested.connect(self.cancel_export)
        self.export_panel.set_preview_text(
            "Analyze an input, define free atoms and reference molecules, estimate the composition, then convert the first frame and reuse that mapping template for the remaining frames."
        )

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

    def _build_project_status_label(
        self,
    ) -> CompactProjectStatusLabel | None:
        if self._project_dir is None:
            return None
        settings = self._load_project_settings()
        project_name = (
            self._project_dir.name
            if settings is None
            else settings.project_name.strip() or self._project_dir.name
        )
        label = CompactProjectStatusLabel(self.statusBar())
        label.setToolTip(
            "Active project: "
            f"{project_name}\n{self._project_dir}\n\n"
            "This window is linked to the active SAXS project, so converted "
            "PDB structure folders are written back to that project."
        )
        label.set_full_text(f"Active project: {self._project_dir}")
        return label

    def _apply_initial_values(
        self,
        *,
        input_path: Path | None,
        config_file: Path | None,
        reference_library_dir: Path | None,
    ) -> None:
        resolved_input_path = input_path
        if resolved_input_path is None:
            settings = self._load_project_settings()
            if settings is not None:
                resolved_input_path = settings.resolved_frames_dir
        if resolved_input_path is not None:
            self.input_panel.input_edit.setText(str(resolved_input_path))
        library_dir = (
            default_reference_library_dir()
            if reference_library_dir is None
            else reference_library_dir
        )
        self.reference_panel.library_dir_edit.setText(str(library_dir))
        self.refresh_reference_library()
        self._suggest_output_dir_from_input(self.input_panel.get_input_path())

    def inspect_input(self) -> None:
        try:
            workflow = self._build_mapping_workflow()
            analysis = workflow.analyze_input()
            self._last_analysis = analysis
            self._last_estimate = None
            self._last_test_result = None
            self.input_panel.set_input_mode(analysis.inspection.input_mode)
            self.input_panel.set_summary_text(
                self._format_analysis_summary(analysis)
            )
            self.mapping_panel.set_available_elements(
                tuple(sorted(analysis.element_counts))
            )
            self.export_panel.set_solution_options([])
            self.export_panel.set_preview_text(
                "Input analysis complete. Define free atoms and reference molecules, estimate the composition, then convert the frames."
            )
            self.export_panel.set_log_text(
                "Input analysis complete.\n"
                f"Sample frame: {analysis.sample_file.name}\n"
                f"Total atoms: {analysis.total_atoms}"
            )
            self.statusBar().showMessage("Input analysis complete")
        except Exception as exc:
            self._show_error("Analyze input failed", str(exc))

    def estimate_mapping(self) -> None:
        try:
            workflow = self._build_mapping_workflow()
            estimate = workflow.estimate_mapping(
                molecule_inputs=self._collect_molecule_inputs(),
                free_atom_inputs=self._collect_free_atom_inputs(),
                hydrogen_mode=self.mapping_panel.hydrogen_mode(),
            )
            self._last_analysis = estimate.analysis
            self._last_estimate = estimate
            self._last_test_result = None
            self.export_panel.set_preview_text(
                self._format_estimate_summary(estimate)
            )
            self.export_panel.set_solution_options(
                self._solution_labels(estimate),
            )
            self.export_panel.set_log_text(
                "Estimate complete.\n"
                + "\n".join(estimate.warnings or ("No estimate warnings.",))
            )
            self.statusBar().showMessage("PDB mapping estimate complete")
        except Exception as exc:
            self._show_error("Estimate mapping failed", str(exc))

    def test_mapping(self) -> None:
        try:
            workflow = self._build_mapping_workflow()
            test_result = workflow.test_mapping(
                molecule_inputs=self._collect_molecule_inputs(),
                free_atom_inputs=self._collect_free_atom_inputs(),
                hydrogen_mode=self.mapping_panel.hydrogen_mode(),
                selected_solution_index=self.export_panel.selected_solution_index(),
                output_dir=self.export_panel.get_output_dir(),
            )
            self._last_analysis = test_result.analysis
            self._last_test_result = test_result
            self.export_panel.set_preview_text(
                self._format_test_summary(test_result)
            )
            log_lines = ["Test mapping complete."]
            log_lines.extend(test_result.console_messages)
            log_lines.extend(
                f"Warning: {warning}" for warning in test_result.warnings
            )
            self.export_panel.set_log_text("\n".join(log_lines))
            self.statusBar().showMessage("Test mapping complete")
        except Exception as exc:
            self._show_error("Test mapping failed", str(exc))

    def export_conversion(self) -> None:
        try:
            if self._export_thread is not None:
                return
            self._closing_after_export_cancel = False
            input_path = self.input_panel.get_input_path()
            if input_path is None:
                raise ValueError("Select an XYZ file or XYZ folder first.")
            job_config = XYZToPDBJobConfig(
                input_path=input_path,
                molecule_inputs=tuple(self._collect_molecule_inputs()),
                free_atom_inputs=tuple(self._collect_free_atom_inputs()),
                hydrogen_mode=self.mapping_panel.hydrogen_mode(),
                selected_solution_index=self.export_panel.selected_solution_index(),
                library_dir=self.reference_panel.get_library_dir(),
                output_dir=self.export_panel.get_output_dir(),
                assertion_mode=self.export_panel.assertion_mode_enabled(),
                estimate_result=self._last_estimate,
            )
            self.export_panel.reset_progress()
            self.export_panel.set_busy_progress(
                "Preparing first-frame mapping..."
            )
            self.export_panel.set_log_text(
                "Starting xyz2pdb export.\n"
                f"Input: {job_config.input_path}\n"
                f"Definitions: {len(job_config.molecule_inputs)} molecule type(s), "
                f"{len(job_config.free_atom_inputs)} free-atom type(s)\n"
                "Assertion mode: "
                + ("on" if job_config.assertion_mode else "off")
            )
            if self._last_estimate is not None:
                self.export_panel.append_log(
                    "Reusing the current estimate before writing PDB frames."
                )
            self.statusBar().showMessage("Converting XYZ to PDB...")
            self._set_export_running(True)
            self._export_thread = QThread(self)
            self._export_worker = XYZToPDBExportWorker(job_config)
            self._export_worker.moveToThread(self._export_thread)
            self._export_thread.started.connect(self._export_worker.run)
            self._export_worker.progress.connect(self.export_panel.append_log)
            self._export_worker.progress_count.connect(
                self._handle_export_progress
            )
            self._export_worker.finished.connect(self._on_export_finished)
            self._export_worker.failed.connect(self._on_export_failed)
            self._export_worker.canceled.connect(self._on_export_canceled)
            self._export_worker.finished.connect(self._cleanup_export_thread)
            self._export_worker.failed.connect(self._cleanup_export_thread)
            self._export_worker.canceled.connect(self._cleanup_export_thread)
            self._export_thread.start(QThread.Priority.LowPriority)
        except Exception as exc:
            self._set_export_running(False)
            self._show_error("Export failed", str(exc))

    def cancel_export(self) -> None:
        thread = self._export_thread
        if thread is None:
            return
        worker = self._export_worker
        if worker is not None:
            worker.cancel()
        if thread.isRunning():
            thread.requestInterruption()
        self.export_panel.cancel_button.setEnabled(False)
        self.export_panel.set_busy_progress("Canceling current conversion...")
        self.export_panel.append_log(
            "Cancellation requested. Stopping the current xyz2pdb run so you can adjust parameters and retry."
        )
        self.statusBar().showMessage("Canceling XYZ to PDB conversion...")

    def refresh_reference_library(
        self,
        _path: Path | None = None,
        *,
        preferred_name: str | None = None,
    ) -> None:
        try:
            library_dir = self.reference_panel.get_library_dir()
            entries = list_reference_library(library_dir)
            self.reference_panel.set_reference_entries(
                entries,
                preferred_name=preferred_name,
            )
            bond_defaults_by_name = {
                entry.name: reference_bond_tolerances(
                    entry.name,
                    library_dir=library_dir,
                )
                for entry in entries
            }
            self.mapping_panel.set_reference_entries(
                entries,
                bond_defaults_by_name=bond_defaults_by_name,
            )
            self.statusBar().showMessage("Reference library refreshed")
        except Exception as exc:
            self._show_error("Reference refresh failed", str(exc))

    def create_reference_molecule(self) -> None:
        try:
            source_path = self.reference_panel.get_source_path()
            reference_name = self.reference_panel.get_reference_name()
            if source_path is None:
                raise ValueError("Choose a PDB or XYZ source file first.")
            if not reference_name:
                raise ValueError("Enter a reference name before creating it.")
            result = create_reference_molecule(
                source_path,
                reference_name=reference_name,
                residue_name=self.reference_panel.get_residue_name(),
                library_dir=self.reference_panel.get_library_dir(),
                backbone_pairs=self.reference_panel.get_backbone_pairs(),
            )
            self.refresh_reference_library(preferred_name=result.name)
            self.export_panel.append_log(
                f"Created reference {result.name} at {result.path}"
            )
            self.statusBar().showMessage(
                f"Created reference molecule {result.name}"
            )
            self._invalidate_results()
        except Exception as exc:
            self._show_error("Create reference failed", str(exc))

    def _build_mapping_workflow(self) -> XYZToPDBMappingWorkflow:
        input_path = self.input_panel.get_input_path()
        if input_path is None:
            raise ValueError("Select an XYZ file or XYZ folder first.")
        return XYZToPDBMappingWorkflow(
            input_path,
            reference_library_dir=self.reference_panel.get_library_dir(),
            output_dir=self.export_panel.get_output_dir(),
        )

    def _collect_free_atom_inputs(self) -> list[FreeAtomMappingInput]:
        return self.mapping_panel.get_free_atom_inputs()

    def _collect_molecule_inputs(self) -> list[MoleculeMappingInput]:
        return self.mapping_panel.get_molecule_inputs()

    def _suggest_output_dir_from_input(self, input_path: Path | None) -> None:
        if input_path is None:
            return
        try:
            self.export_panel.suggest_output_dir(
                suggest_output_dir(input_path)
            )
        except Exception:
            return

    def _format_analysis_summary(
        self,
        analysis: XYZToPDBSampleAnalysis,
    ) -> str:
        inspection = analysis.inspection
        lines = [
            f"Input path: {inspection.input_path}",
            (
                "Mode: Single XYZ file"
                if inspection.input_mode == "single_xyz"
                else "Mode: XYZ folder"
            ),
            f"XYZ files found: {inspection.total_files}",
            f"Sample frame: {analysis.sample_file.name}",
            f"Sample comment: {analysis.sample_comment}",
            f"Sample atoms: {analysis.total_atoms}",
            f"Reference library: {inspection.reference_library_dir}",
            "Available references: "
            + (
                ", ".join(
                    entry.name for entry in inspection.available_references
                )
                if inspection.available_references
                else "none"
            ),
            "Element counts: "
            + ", ".join(
                f"{element} x{count}"
                for element, count in sorted(analysis.element_counts.items())
            ),
        ]
        return "\n".join(lines)

    def _format_estimate_summary(
        self,
        estimate: XYZToPDBEstimateResult,
    ) -> str:
        solution = estimate.solutions[0] if estimate.solutions else None
        lines = [
            f"Sample frame: {estimate.analysis.sample_file.name}",
            f"Solutions found: {len(estimate.solutions)}",
            f"Output directory: {self.export_panel.get_output_dir() or suggest_output_dir(estimate.analysis.inspection.input_path)}",
        ]
        if solution is not None:
            lines.append(
                "Estimated molecules: "
                + (
                    ", ".join(
                        f"{residue} x{count}"
                        for residue, count in sorted(
                            solution.molecule_count_by_residue(
                                estimate.plan
                            ).items()
                        )
                    )
                    or "none"
                )
            )
            lines.append(
                "Estimated free atoms: "
                + (
                    ", ".join(
                        f"{element} x{count}"
                        for element, count in sorted(
                            solution.free_atom_counts.items()
                        )
                    )
                    or "none"
                )
            )
            lines.append(
                "Unassigned atoms: "
                + (
                    ", ".join(
                        f"{element} x{count}"
                        for element, count in sorted(
                            solution.unassigned_counts.items()
                        )
                    )
                    or "none"
                )
            )
        if estimate.warnings:
            lines.append("Warnings: " + " | ".join(estimate.warnings))
        return "\n".join(lines)

    def _format_test_summary(
        self,
        result: XYZToPDBMappingTestResult,
    ) -> str:
        lines = [
            f"Output directory: {result.output_dir}",
            f"First output file: {result.first_output_file}",
            f"Matched molecule residues: {sum(result.molecule_counts.values())}",
            f"Total atoms to write: {result.total_atoms}",
            "Matched molecules: "
            + (
                ", ".join(
                    f"{name} x{count}"
                    for name, count in sorted(result.molecule_counts.items())
                )
                or "none"
            ),
            "Residues to write: "
            + (
                ", ".join(
                    f"{name} x{count}"
                    for name, count in sorted(result.residue_counts.items())
                )
                or "none"
            ),
            "Unassigned atoms: "
            + (
                ", ".join(
                    f"{element} x{count}"
                    for element, count in sorted(
                        result.unassigned_counts.items()
                    )
                )
                or "none"
            ),
        ]
        if result.warnings:
            lines.append("Warnings: " + " | ".join(result.warnings))
        return "\n".join(lines)

    def _solution_labels(
        self,
        estimate: XYZToPDBEstimateResult,
    ) -> list[str]:
        labels: list[str] = []
        for index, solution in enumerate(estimate.solutions, start=1):
            molecule_text = (
                ", ".join(
                    f"{residue} x{count}"
                    for residue, count in sorted(
                        solution.molecule_count_by_residue(
                            estimate.plan
                        ).items()
                    )
                )
                or "no molecules"
            )
            unassigned_text = (
                ", ".join(
                    f"{element} x{count}"
                    for element, count in sorted(
                        solution.unassigned_counts.items()
                    )
                )
                or "none"
            )
            labels.append(
                f"Solution {index}: {molecule_text} | unassigned: {unassigned_text}"
            )
        return labels

    def _invalidate_results(self) -> None:
        self._last_estimate = None
        self._last_test_result = None
        self.export_panel.set_solution_options([])

    def _set_export_running(self, is_running: bool) -> None:
        self.input_panel.setEnabled(not is_running)
        self.reference_panel.setEnabled(not is_running)
        self.mapping_panel.setEnabled(not is_running)
        self.export_panel.set_controls_enabled(not is_running)

    @Slot(int, int, str)
    def _handle_export_progress(
        self,
        processed: int,
        total: int,
        message: str,
    ) -> None:
        self.export_panel.update_progress(processed, total, message)
        self.export_panel.append_log(message)
        self.statusBar().showMessage(message)

    @Slot(object)
    def _on_export_finished(self, result: object) -> None:
        if self._closing_after_export_cancel:
            return
        self._closing_after_export_cancel = False
        assert isinstance(result, XYZToPDBExportResult)
        self._set_export_running(False)
        total_steps = max(
            int(
                result.progress_total_steps or (len(result.written_files) + 1)
            ),
            1,
        )
        self.export_panel.set_progress_complete(
            "Conversion complete.",
            total=total_steps,
        )
        if isinstance(result.preview, XYZToPDBMappingTestResult):
            self._last_test_result = result.preview
            self.export_panel.set_preview_text(
                self._format_test_summary(result.preview)
            )
        self.export_panel.append_log(
            f"Conversion complete. Wrote {len(result.written_files)} PDB file(s) to {result.output_dir}."
        )
        if result.assertion_result is not None:
            status_label = (
                "passed"
                if result.assertion_result.passed
                else "reported warnings"
            )
            self.export_panel.append_log(
                "Assertion mode "
                f"{status_label}. Molecule folder: {result.assertion_result.molecule_dir}"
            )
            self.export_panel.append_log(
                f"Assertion report: {result.assertion_result.report_file}"
            )
            for warning in result.assertion_result.warnings:
                self.export_panel.append_log(f"Assertion warning: {warning}")
        registration_message = self._register_exported_pdb_frames_folder(
            result.output_dir
        )
        if registration_message is not None:
            self.export_panel.append_log(registration_message)
        if result.assertion_result is not None:
            self._offer_assertion_reference_updates(
                result.assertion_result.reference_update_candidates
            )
        self.statusBar().showMessage("XYZ to PDB conversion complete")

    @Slot(str)
    def _on_export_failed(self, message: str) -> None:
        if self._closing_after_export_cancel:
            return
        self._closing_after_export_cancel = False
        self._set_export_running(False)
        self.export_panel.set_progress_failed("Progress: failed")
        self._show_error("Export failed", message)

    @Slot()
    def _on_export_canceled(self) -> None:
        if self._closing_after_export_cancel:
            return
        self._closing_after_export_cancel = False
        self._set_export_running(False)
        self.export_panel.set_progress_failed(
            "Conversion canceled. Adjust parameters and retry when ready."
        )
        self.export_panel.append_log("XYZ-to-PDB conversion canceled.")
        self.statusBar().showMessage("XYZ to PDB conversion canceled")

    @Slot()
    def _cleanup_export_thread(self) -> None:
        thread = self._export_thread
        worker = self._export_worker
        self._export_thread = None
        self._export_worker = None
        if thread is None:
            return
        if hasattr(thread, "quit"):
            thread.quit()
        if (
            hasattr(thread, "wait")
            and hasattr(thread, "isRunning")
            and thread.isRunning()
        ):
            thread.wait()
        if worker is not None:
            worker.deleteLater()
        if hasattr(thread, "deleteLater"):
            thread.deleteLater()

    def _cancel_export_for_close(self) -> None:
        thread = self._export_thread
        if thread is None:
            return
        self._closing_after_export_cancel = True
        worker = self._export_worker
        if worker is not None:
            worker.cancel()
        if thread.isRunning():
            thread.requestInterruption()
            thread.quit()
            if not thread.wait(25):
                thread.terminate()
                thread.wait(25)
        self._cleanup_export_thread()

    def _wrap_scroll_area(self, widget: QWidget) -> QScrollArea:
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(widget)
        return scroll_area

    def _show_error(self, title: str, message: str) -> None:
        QMessageBox.critical(self, title, message)
        self.statusBar().showMessage(message)

    def _offer_assertion_reference_updates(
        self,
        candidates: tuple[XYZToPDBReferenceUpdateCandidate, ...],
    ) -> None:
        if not candidates:
            return
        self.export_panel.append_log(
            "Assertion mode prepared "
            f"{len(candidates)} reference-update candidate(s)."
        )
        preferred_name: str | None = None
        for candidate in candidates:
            versioned_reference_name = self._versioned_reference_name(
                candidate.reference_name
            )
            decision = self._prompt_reference_update_candidate(
                candidate,
                versioned_reference_name=versioned_reference_name,
            )
            if decision == "skip":
                self.export_panel.append_log(
                    f"Skipped assertion-derived reference update for {candidate.reference_name}."
                )
                continue
            try:
                result = self._apply_reference_update_candidate(
                    candidate,
                    decision=decision,
                    versioned_reference_name=versioned_reference_name,
                )
            except Exception as exc:
                self._show_error(
                    "Reference update failed",
                    str(exc),
                )
                self.export_panel.append_log(
                    f"Reference update failed for {candidate.reference_name}: {exc}"
                )
                continue
            preferred_name = result.name
            if decision == "replace_existing":
                self.export_panel.append_log(
                    f"Updated reference {result.name} with assertion-averaged coordinates."
                )
            else:
                self.export_panel.append_log(
                    f"Saved assertion-derived reference version {result.name}."
                )
        if preferred_name is not None:
            self.refresh_reference_library(preferred_name=preferred_name)

    def _prompt_reference_update_candidate(
        self,
        candidate: XYZToPDBReferenceUpdateCandidate,
        *,
        versioned_reference_name: str,
    ) -> str:
        dialog = AssertionReferenceUpdateDialog(
            candidate,
            versioned_reference_name=versioned_reference_name,
            parent=self,
        )
        if dialog.exec():
            return dialog.decision
        return "skip"

    def _apply_reference_update_candidate(
        self,
        candidate: XYZToPDBReferenceUpdateCandidate,
        *,
        decision: str,
        versioned_reference_name: str,
    ):
        library_dir = self.reference_panel.get_library_dir()
        if library_dir is None:
            library_dir = candidate.reference_path.parent
        if decision == "replace_existing":
            reference_name = candidate.reference_name
        elif decision == "save_new_version":
            reference_name = versioned_reference_name
        else:
            raise ValueError(
                f"Unsupported reference update action: {decision}"
            )
        return create_reference_molecule(
            candidate.average_structure_file,
            reference_name=reference_name,
            residue_name=candidate.reference_residue_name,
            library_dir=library_dir,
            backbone_pairs=candidate.backbone_pairs,
        )

    def _versioned_reference_name(self, base_name: str) -> str:
        timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        return f"{base_name}_{timestamp}"

    def _register_exported_pdb_frames_folder(
        self,
        output_dir: Path,
    ) -> str | None:
        if self._project_dir is None:
            return None
        project_file = build_project_paths(self._project_dir).project_file
        if not project_file.is_file():
            return (
                "PDB conversion succeeded, but the associated project could "
                f"not be found: {self._project_dir}"
            )
        try:
            settings = self._project_manager.load_project(self._project_dir)
            settings.pdb_frames_dir = str(
                Path(output_dir).expanduser().resolve()
            )
            self._project_manager.save_project(settings)
            self.project_paths_registered.emit(
                {
                    "project_dir": Path(self._project_dir).resolve(),
                    "pdb_frames_dir": Path(output_dir).expanduser().resolve(),
                }
            )
        except Exception as exc:
            return (
                "PDB conversion succeeded, but the project PDB structure "
                f"folder reference could not be updated: {exc}"
            )
        return (
            "Registered the converted PDB structure folder with project "
            f"{self._project_dir.name}: {Path(output_dir).expanduser().resolve()}"
        )


def launch_xyz2pdb_ui(
    *,
    project_dir: str | Path | None = None,
    input_path: str | Path | None = None,
    config_file: str | Path | None = None,
    reference_library_dir: str | Path | None = None,
) -> XYZToPDBMainWindow:
    """Create, show, and return the xyz2pdb main window."""
    app = QApplication.instance()
    if app is None:
        prepare_saxshell_application_identity()
        app = QApplication(sys.argv)
    configure_saxshell_application(app)
    window = XYZToPDBMainWindow(
        initial_project_dir=(
            None
            if project_dir is None
            else Path(project_dir).expanduser().resolve()
        ),
        input_path=(
            None
            if input_path is None
            else Path(input_path).expanduser().resolve()
        ),
        config_file=(
            None
            if config_file is None
            else Path(config_file).expanduser().resolve()
        ),
        reference_library_dir=(
            None
            if reference_library_dir is None
            else Path(reference_library_dir).expanduser().resolve()
        ),
    )
    window.show()
    window.raise_()
    return window
