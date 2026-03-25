from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot
from PySide6.QtWidgets import (
    QMainWindow,
    QMessageBox,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from saxshell.xyz2pdb import (
    XYZToPDBExportResult,
    XYZToPDBInspectionResult,
    XYZToPDBPreviewResult,
    XYZToPDBWorkflow,
    create_reference_molecule,
    default_reference_library_dir,
    list_reference_library,
    suggest_output_dir,
)
from saxshell.xyz2pdb.ui.export_panel import XYZToPDBExportPanel
from saxshell.xyz2pdb.ui.input_panel import XYZToPDBInputPanel
from saxshell.xyz2pdb.ui.reference_panel import ReferenceLibraryPanel


@dataclass(slots=True)
class XYZToPDBJobConfig:
    """Export settings assembled from the UI."""

    input_path: Path
    config_file: Path
    library_dir: Path | None
    output_dir: Path | None


class XYZToPDBExportWorker(QObject):
    """Background worker that runs xyz2pdb export."""

    progress = Signal(str)
    progress_count = Signal(int, int)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, config: XYZToPDBJobConfig) -> None:
        super().__init__()
        self.config = config

    @Slot()
    def run(self) -> None:
        try:
            workflow = XYZToPDBWorkflow(
                self.config.input_path,
                config_file=self.config.config_file,
                reference_library_dir=self.config.library_dir,
                output_dir=self.config.output_dir,
            )
            preview = workflow.preview_conversion(
                output_dir=self.config.output_dir
            )
            self.progress.emit(
                f"Preview complete. First output will be written under {preview.output_dir}."
            )
            self.progress.emit(
                f"Converting {preview.inspection.total_files} XYZ file(s) to PDB."
            )
            self.progress_count.emit(0, preview.inspection.total_files)

            def on_progress(processed: int, total: int, filename: str) -> None:
                self.progress_count.emit(processed, total)
                self.progress.emit(f"[{processed}/{total}] Wrote {filename}")

            result = workflow.export_pdbs(
                output_dir=self.config.output_dir,
                progress_callback=on_progress,
            )
            self.finished.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


class XYZToPDBMainWindow(QMainWindow):
    """Main Qt window for the xyz2pdb application."""

    def __init__(
        self,
        *,
        input_path: Path | None = None,
        config_file: Path | None = None,
        reference_library_dir: Path | None = None,
    ) -> None:
        super().__init__()
        self._last_inspection: XYZToPDBInspectionResult | None = None
        self._last_preview: XYZToPDBPreviewResult | None = None
        self._export_thread: QThread | None = None
        self._export_worker: XYZToPDBExportWorker | None = None
        self._build_ui()
        self._apply_initial_values(
            input_path=input_path,
            config_file=config_file,
            reference_library_dir=reference_library_dir,
        )

    def _build_ui(self) -> None:
        self.setWindowTitle("SAXSShell (xyz2pdb)")
        self.resize(1360, 820)

        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(12)

        self.input_panel = XYZToPDBInputPanel()
        self.reference_panel = ReferenceLibraryPanel()
        self.export_panel = XYZToPDBExportPanel()

        left_layout.addWidget(self.input_panel)
        left_layout.addWidget(self.reference_panel)
        left_layout.addStretch(1)

        splitter.addWidget(self._wrap_scroll_area(left))
        splitter.addWidget(self._wrap_scroll_area(self.export_panel))
        splitter.setSizes([720, 640])

        root.addWidget(splitter)
        self.setCentralWidget(central)
        self.statusBar().showMessage("Ready")

        self.input_panel.inspect_requested.connect(self.inspect_input)
        self.input_panel.preview_requested.connect(self.preview_conversion)
        self.input_panel.input_path_changed.connect(
            self._suggest_output_dir_from_input
        )
        self.input_panel.library_dir_changed.connect(
            self.refresh_reference_library
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
        self.export_panel.export_requested.connect(self.export_conversion)
        self.export_panel.set_preview_text(
            "Inspect or preview a conversion to see the first-frame residue assignments."
        )

    def _apply_initial_values(
        self,
        *,
        input_path: Path | None,
        config_file: Path | None,
        reference_library_dir: Path | None,
    ) -> None:
        if input_path is not None:
            self.input_panel.input_edit.setText(str(input_path))
        if config_file is not None:
            self.input_panel.config_edit.setText(str(config_file))
        library_dir = (
            default_reference_library_dir()
            if reference_library_dir is None
            else reference_library_dir
        )
        self.input_panel.library_dir_edit.setText(str(library_dir))
        self.refresh_reference_library()
        self._suggest_output_dir_from_input(self.input_panel.get_input_path())

    def inspect_input(self) -> None:
        try:
            workflow = self._build_workflow()
            inspection = workflow.inspect()
            self._last_inspection = inspection
            self._last_preview = None
            self.input_panel.set_input_mode(inspection.input_mode)
            self.input_panel.set_summary_text(
                self._format_inspection_summary(inspection)
            )
            self.export_panel.set_preview_text(
                "Inspection complete. Use Preview to evaluate the first XYZ frame."
            )
            self.export_panel.set_log_text(
                "Inspection complete.\n"
                f"Found {inspection.total_files} XYZ file(s).\n"
                f"Reference library: {inspection.reference_library_dir}"
            )
            self.statusBar().showMessage("Inspection complete")
        except Exception as exc:
            self._show_error("Inspect failed", str(exc))

    def preview_conversion(self) -> None:
        try:
            workflow = self._build_workflow()
            preview = workflow.preview_conversion(
                output_dir=self.export_panel.get_output_dir()
            )
            self._last_inspection = preview.inspection
            self._last_preview = preview
            self.input_panel.set_input_mode(preview.inspection.input_mode)
            self.input_panel.set_summary_text(
                self._format_inspection_summary(preview.inspection)
            )
            self.export_panel.set_preview_text(
                self._format_preview_summary(preview)
            )
            self.export_panel.set_log_text(
                "Preview complete.\n"
                f"First output file: {preview.first_output_file}\n"
                f"Detected molecules: {', '.join(preview.molecule_counts)}"
            )
            self.statusBar().showMessage("Preview complete")
        except Exception as exc:
            self._show_error("Preview failed", str(exc))

    def export_conversion(self) -> None:
        try:
            if self._export_thread is not None:
                return
            input_path = self.input_panel.get_input_path()
            config_file = self.input_panel.get_config_path()
            if input_path is None:
                raise ValueError("Select an XYZ file or XYZ folder first.")
            if config_file is None:
                raise ValueError(
                    "Select a residue-assignment JSON file first."
                )
            job_config = XYZToPDBJobConfig(
                input_path=input_path,
                config_file=config_file,
                library_dir=self.input_panel.get_library_dir(),
                output_dir=self.export_panel.get_output_dir(),
            )
            self.export_panel.reset_progress()
            self.export_panel.set_log_text(
                "Starting xyz2pdb export.\n"
                f"Input: {job_config.input_path}\n"
                f"Config: {job_config.config_file}"
            )
            self.statusBar().showMessage("Converting XYZ to PDB...")
            self._export_thread = QThread(self)
            self._export_worker = XYZToPDBExportWorker(job_config)
            self._export_worker.moveToThread(self._export_thread)
            self._export_thread.started.connect(self._export_worker.run)
            self._export_worker.progress.connect(self.export_panel.append_log)
            self._export_worker.progress_count.connect(
                self.export_panel.update_progress
            )
            self._export_worker.finished.connect(self._on_export_finished)
            self._export_worker.failed.connect(self._on_export_failed)
            self._export_worker.finished.connect(self._cleanup_export_thread)
            self._export_worker.failed.connect(self._cleanup_export_thread)
            self._export_thread.start()
        except Exception as exc:
            self._show_error("Export failed", str(exc))

    def refresh_reference_library(
        self,
        _path: Path | None = None,
        *,
        preferred_name: str | None = None,
    ) -> None:
        try:
            library_dir = self.input_panel.get_library_dir()
            entries = list_reference_library(library_dir)
            self.reference_panel.set_reference_entries(
                entries,
                preferred_name=preferred_name,
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
                library_dir=self.input_panel.get_library_dir(),
            )
            self.refresh_reference_library(preferred_name=result.name)
            self.export_panel.append_log(
                f"Created reference {result.name} at {result.path}"
            )
            self.statusBar().showMessage(
                f"Created reference molecule {result.name}"
            )
        except Exception as exc:
            self._show_error("Create reference failed", str(exc))

    def _build_workflow(self) -> XYZToPDBWorkflow:
        input_path = self.input_panel.get_input_path()
        if input_path is None:
            raise ValueError("Select an XYZ file or XYZ folder first.")
        return XYZToPDBWorkflow(
            input_path,
            config_file=self.input_panel.get_config_path(),
            reference_library_dir=self.input_panel.get_library_dir(),
            output_dir=self.export_panel.get_output_dir(),
        )

    def _suggest_output_dir_from_input(self, input_path: Path | None) -> None:
        if input_path is None:
            return
        try:
            self.export_panel.suggest_output_dir(
                suggest_output_dir(input_path)
            )
        except Exception:
            return

    def _format_inspection_summary(
        self,
        inspection: XYZToPDBInspectionResult,
    ) -> str:
        lines = [
            f"Input path: {inspection.input_path}",
            (
                "Mode: Single XYZ file"
                if inspection.input_mode == "single_xyz"
                else "Mode: XYZ folder"
            ),
            f"XYZ files found: {inspection.total_files}",
            f"Reference library: {inspection.reference_library_dir}",
            "Available references: "
            + (
                ", ".join(
                    entry.name for entry in inspection.available_references
                )
                if inspection.available_references
                else "none"
            ),
        ]
        if inspection.config_file is not None:
            lines.append(f"Config JSON: {inspection.config_file}")
            lines.append(
                "Configured molecules: "
                + (
                    ", ".join(inspection.configured_molecules)
                    if inspection.configured_molecules
                    else "none"
                )
            )
            lines.append(
                "Configured references: "
                + (
                    ", ".join(inspection.configured_reference_names)
                    if inspection.configured_reference_names
                    else "none"
                )
            )
        return "\n".join(lines)

    def _format_preview_summary(
        self,
        preview: XYZToPDBPreviewResult,
    ) -> str:
        lines = [
            f"Output directory: {preview.output_dir}",
            f"XYZ files to convert: {preview.inspection.total_files}",
            f"First output file: {preview.first_output_file}",
            f"First-frame residues: {len(preview.residues)}",
            f"First-frame atoms: {preview.total_atoms}",
            "Detected molecules: "
            + ", ".join(
                f"{name} x{count}"
                for name, count in sorted(preview.molecule_counts.items())
            ),
            "Residues to write: "
            + ", ".join(
                f"{name} x{count}"
                for name, count in sorted(preview.residue_counts.items())
            ),
        ]
        return "\n".join(lines)

    @Slot(object)
    def _on_export_finished(self, result: object) -> None:
        assert isinstance(result, XYZToPDBExportResult)
        self.export_panel.update_progress(
            len(result.written_files),
            len(result.written_files),
        )
        self.export_panel.append_log(
            f"Conversion complete. Wrote {len(result.written_files)} PDB file(s) to {result.output_dir}."
        )
        self.statusBar().showMessage("XYZ to PDB conversion complete")

    @Slot(str)
    def _on_export_failed(self, message: str) -> None:
        self._show_error("Export failed", message)

    @Slot()
    def _cleanup_export_thread(self) -> None:
        if self._export_thread is None:
            return
        self._export_thread.quit()
        self._export_thread.wait()
        if self._export_worker is not None:
            self._export_worker.deleteLater()
        self._export_thread.deleteLater()
        self._export_worker = None
        self._export_thread = None

    def _wrap_scroll_area(self, widget: QWidget) -> QScrollArea:
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(widget)
        return scroll_area

    def _show_error(self, title: str, message: str) -> None:
        QMessageBox.critical(self, title, message)
        self.statusBar().showMessage(message)


def launch_xyz2pdb_ui(
    *,
    input_path: str | Path | None = None,
    config_file: str | Path | None = None,
    reference_library_dir: str | Path | None = None,
) -> XYZToPDBMainWindow:
    """Create, show, and return the xyz2pdb main window."""
    window = XYZToPDBMainWindow(
        input_path=None if input_path is None else Path(input_path),
        config_file=None if config_file is None else Path(config_file),
        reference_library_dir=(
            None
            if reference_library_dir is None
            else Path(reference_library_dir)
        ),
    )
    window.show()
    window.raise_()
    return window
