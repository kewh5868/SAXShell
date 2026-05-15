from __future__ import annotations

import json
import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from saxshell.cluster import DEFAULT_CLUSTER_EXTRACTION_PRESET_NAME
from saxshell.cluster.ui.definitions_panel import ClusterDefinitionsPanel
from saxshell.cluster.workflow import ClusterWorkflow, format_box_dimensions
from saxshell.clusterdynamics.ui.main_window import ClusterDynamicsTimePanel
from saxshell.clusterdynamicsml.run_config import (
    build_clusterdynamicsml_run_config,
    default_clusterdynamicsml_run_file_path,
    preview_clusterdynamicsml_run_config,
    save_clusterdynamicsml_run_config,
    suggest_clusterdynamicsml_output_file,
)
from saxshell.clusterdynamicsml.ui.main_window import (
    ClusterDynamicsMLSettingsPanel,
)
from saxshell.saxs.project_manager import SAXSProjectManager
from saxshell.saxs.ui.branding import (
    configure_saxshell_application,
    load_saxshell_icon,
    prepare_saxshell_application_identity,
)
from saxshell.xyz2pdb import list_reference_library


class ClusterDynamicsMLRunFileWindow(QMainWindow):
    def __init__(
        self,
        *,
        initial_project_dir: str | Path | None = None,
        initial_frames_dir: str | Path | None = None,
        initial_energy_file: str | Path | None = None,
        initial_clusters_dir: str | Path | None = None,
        initial_experimental_data_file: str | Path | None = None,
    ) -> None:
        super().__init__()
        self._browse_start_dir = Path.home()
        self._last_summary: dict[str, object] | None = None
        self._last_suggested_output_file: str | None = None

        project_dir = _optional_resolved_path(initial_project_dir)
        frames_dir = _optional_resolved_path(initial_frames_dir)
        energy_file = _optional_resolved_path(initial_energy_file)
        clusters_dir = _optional_resolved_path(initial_clusters_dir)
        experimental_data_file = _optional_resolved_path(
            initial_experimental_data_file
        )
        if project_dir is not None:
            self._browse_start_dir = project_dir
            defaults = self._project_defaults(project_dir)
            if frames_dir is None:
                frames_dir = defaults.get("frames_dir")
            if energy_file is None:
                energy_file = defaults.get("energy_file")
            if clusters_dir is None:
                clusters_dir = defaults.get("clusters_dir")
            if experimental_data_file is None:
                experimental_data_file = defaults.get("experimental_data_file")

        self.setWindowTitle("Cluster Dynamics ML CLI Setup (Beta)")
        self.setWindowIcon(load_saxshell_icon())
        self.resize(1120, 840)
        self._build_ui()
        self.definitions_panel.load_preset(
            DEFAULT_CLUSTER_EXTRACTION_PRESET_NAME
        )
        self.definitions_panel.set_shell_reference_editor_enabled(True)
        self._load_shell_reference_library_entries()

        if project_dir is not None:
            self.project_dir_edit.setText(str(project_dir))
            self._refresh_run_file_path()
        if frames_dir is not None and frames_dir.is_dir():
            self.frames_dir_edit.setText(str(frames_dir))
            self._browse_start_dir = frames_dir
        if energy_file is not None and energy_file.is_file():
            self.energy_file_edit.setText(str(energy_file))
        if clusters_dir is not None and clusters_dir.is_dir():
            self.prediction_panel.set_clusters_dir(
                clusters_dir,
                emit_signal=False,
            )
        if (
            experimental_data_file is not None
            and experimental_data_file.is_file()
        ):
            self.prediction_panel.set_experimental_data_file(
                experimental_data_file,
                emit_signal=False,
            )
        self._inspect_frames()
        self._update_preview()

    def _build_ui(self) -> None:
        central = QWidget(self)
        root = QVBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)
        self.setCentralWidget(central)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.setChildrenCollapsible(False)
        root.addWidget(splitter, stretch=1)

        left_scroll = QScrollArea(self)
        left_scroll.setWidgetResizable(True)
        left_panel = QWidget()
        self.left_layout = QVBoxLayout(left_panel)
        self.left_layout.setContentsMargins(10, 10, 10, 10)
        self.left_layout.setSpacing(10)
        left_scroll.setWidget(left_panel)

        right_scroll = QScrollArea(self)
        right_scroll.setWidgetResizable(True)
        right_panel = QWidget()
        self.right_layout = QVBoxLayout(right_panel)
        self.right_layout.setContentsMargins(10, 10, 10, 10)
        self.right_layout.setSpacing(10)
        right_scroll.setWidget(right_panel)

        splitter.addWidget(left_scroll)
        splitter.addWidget(right_scroll)
        splitter.setSizes([600, 520])

        self.left_layout.addWidget(self._build_project_group())
        self.left_layout.addWidget(self._build_input_group())
        self.prediction_panel = ClusterDynamicsMLSettingsPanel()
        self.prediction_panel.settings_changed.connect(self._update_preview)
        self.left_layout.addWidget(self.prediction_panel)
        self.definitions_panel = ClusterDefinitionsPanel()
        self.definitions_panel.settings_changed.connect(self._update_preview)
        self.left_layout.addWidget(self.definitions_panel)
        self.time_panel = ClusterDynamicsTimePanel()
        self.time_panel.settings_changed.connect(self._update_preview)
        self.left_layout.addWidget(self.time_panel)
        self.left_layout.addWidget(self._build_save_group())
        self.left_layout.addStretch(1)

        self.right_layout.addWidget(self._build_inspection_group())
        self.right_layout.addWidget(self._build_command_group())
        self.right_layout.addStretch(1)
        self.statusBar().showMessage("Ready")

    def _build_project_group(self) -> QGroupBox:
        group = QGroupBox("Project")
        form = QFormLayout(group)
        project_row = QHBoxLayout()
        self.project_dir_edit = QLineEdit()
        self.project_dir_edit.editingFinished.connect(
            self._on_project_dir_changed
        )
        project_row.addWidget(self.project_dir_edit, stretch=1)
        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self._browse_project_dir)
        project_row.addWidget(browse_button)
        project_widget = QWidget()
        project_widget.setLayout(project_row)
        form.addRow("Project folder", project_widget)

        self.run_file_edit = QLineEdit()
        self.run_file_edit.setReadOnly(True)
        form.addRow("Run file", self.run_file_edit)
        return group

    def _build_input_group(self) -> QGroupBox:
        group = QGroupBox("Input / Output")
        form = QFormLayout(group)
        self.frames_dir_edit = QLineEdit()
        self.frames_dir_edit.editingFinished.connect(self._inspect_frames)
        form.addRow(
            "Frames folder",
            self._make_path_row(
                self.frames_dir_edit,
                self._browse_frames_dir,
            ),
        )

        self.energy_file_edit = QLineEdit()
        self.energy_file_edit.editingFinished.connect(self._update_preview)
        form.addRow(
            "CP2K .ener file",
            self._make_path_row(
                self.energy_file_edit,
                self._browse_energy_file,
            ),
        )

        self.output_file_edit = QLineEdit()
        self.output_file_edit.editingFinished.connect(self._update_preview)
        form.addRow(
            "Output dataset",
            self._make_path_row(
                self.output_file_edit,
                self._browse_output_file,
            ),
        )
        return group

    def _build_save_group(self) -> QGroupBox:
        group = QGroupBox("Save")
        layout = QHBoxLayout(group)
        inspect_button = QPushButton("Inspect Frames")
        inspect_button.clicked.connect(self._inspect_frames)
        layout.addWidget(inspect_button)
        save_button = QPushButton("Save Run File")
        save_button.clicked.connect(self._save_run_file)
        layout.addWidget(save_button)
        layout.addStretch(1)
        return group

    def _build_inspection_group(self) -> QGroupBox:
        group = QGroupBox("Inspection")
        layout = QVBoxLayout(group)
        self.inspection_box = QPlainTextEdit()
        self.inspection_box.setReadOnly(True)
        self.inspection_box.setMinimumHeight(210)
        layout.addWidget(self.inspection_box)
        return group

    def _build_command_group(self) -> QGroupBox:
        group = QGroupBox("CLI Command / JSON")
        layout = QVBoxLayout(group)
        layout.addWidget(QLabel("Commands"))
        self.command_box = QPlainTextEdit()
        self.command_box.setReadOnly(True)
        self.command_box.setMinimumHeight(150)
        layout.addWidget(self.command_box)
        layout.addWidget(QLabel("Run file preview"))
        self.json_preview_box = QPlainTextEdit()
        self.json_preview_box.setReadOnly(True)
        self.json_preview_box.setMinimumHeight(300)
        layout.addWidget(self.json_preview_box)
        return group

    def _make_path_row(self, line_edit: QLineEdit, callback) -> QWidget:
        widget = QWidget()
        row = QHBoxLayout(widget)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(line_edit, stretch=1)
        button = QPushButton("Browse...")
        button.clicked.connect(callback)
        row.addWidget(button)
        return widget

    def _browse_project_dir(self, *_args: object) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "Select project folder",
            str(self._browse_start_dir),
        )
        if not selected:
            return
        self.project_dir_edit.setText(selected)
        self._on_project_dir_changed()

    def _browse_frames_dir(self, *_args: object) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "Select extracted frames folder",
            str(self._browse_start_dir),
        )
        if not selected:
            return
        self.frames_dir_edit.setText(selected)
        self._browse_start_dir = Path(selected).expanduser().resolve()
        self._inspect_frames()

    def _browse_energy_file(self, *_args: object) -> None:
        path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "Select CP2K .ener file",
            self.energy_file_edit.text().strip()
            or str(self._browse_start_dir),
            "Energy Files (*.ener);;All Files (*)",
        )
        if path:
            self.energy_file_edit.setText(path)
            self._update_preview()

    def _browse_output_file(self, *_args: object) -> None:
        path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "Select output dataset",
            self.output_file_edit.text().strip()
            or str(self._browse_start_dir),
            "JSON Files (*.json);;All Files (*)",
        )
        if path:
            self.output_file_edit.setText(path)
            self._update_preview()

    def _on_project_dir_changed(self, *_args: object) -> None:
        project_dir = self._project_dir()
        if project_dir is None:
            return
        self._browse_start_dir = project_dir
        self._refresh_run_file_path()
        defaults = self._project_defaults(project_dir)
        if not self.frames_dir_edit.text().strip():
            frames_dir = defaults.get("frames_dir")
            if frames_dir is not None and frames_dir.is_dir():
                self.frames_dir_edit.setText(str(frames_dir))
        if not self.energy_file_edit.text().strip():
            energy_file = defaults.get("energy_file")
            if energy_file is not None and energy_file.is_file():
                self.energy_file_edit.setText(str(energy_file))
        if self.prediction_panel.clusters_dir() is None:
            self.prediction_panel.set_clusters_dir(
                defaults.get("clusters_dir"),
                emit_signal=False,
            )
        if self.prediction_panel.experimental_data_file() is None:
            self.prediction_panel.set_experimental_data_file(
                defaults.get("experimental_data_file"),
                emit_signal=False,
            )
        self._inspect_frames()

    def _inspect_frames(self, *_args: object) -> None:
        frames_text = self.frames_dir_edit.text().strip()
        if not frames_text:
            self._last_summary = None
            self.inspection_box.setPlainText("No frames folder selected.")
            self._update_preview()
            return
        try:
            workflow = ClusterWorkflow(
                frames_dir=frames_text,
                atom_type_definitions={},
                pair_cutoff_definitions={},
            )
            summary = workflow.inspect()
        except Exception as exc:
            self._last_summary = None
            self.inspection_box.setPlainText(str(exc))
            self.statusBar().showMessage("Frames inspection failed")
            self._update_preview()
            return
        self._last_summary = summary
        frame_format = str(summary.get("frame_format", "") or "")
        self.definitions_panel.set_frame_mode(frame_format)
        self.inspection_box.setPlainText(_summary_text(summary))
        self._refresh_suggested_output_file()
        self.statusBar().showMessage(
            f"Discovered {int(summary.get('n_frames', 0))} frame(s)"
        )
        self._update_preview()

    def _refresh_run_file_path(self) -> None:
        project_dir = self._project_dir()
        if project_dir is None:
            self.run_file_edit.clear()
            return
        self.run_file_edit.setText(
            str(default_clusterdynamicsml_run_file_path(project_dir))
        )

    def _refresh_suggested_output_file(self) -> None:
        project_dir = self._project_dir()
        frames_text = self.frames_dir_edit.text().strip()
        if project_dir is None or not frames_text:
            return
        try:
            suggested = suggest_clusterdynamicsml_output_file(
                project_dir=project_dir,
                frames_dir=frames_text,
            )
        except Exception:
            return
        current = self.output_file_edit.text().strip()
        if not current or current == self._last_suggested_output_file:
            self.output_file_edit.setText(str(suggested))
        self._last_suggested_output_file = str(suggested)

    def _save_run_file(self, *_args: object) -> None:
        try:
            project_dir = self._require_project_dir()
            config = self._current_config(project_dir)
        except Exception as exc:
            QMessageBox.warning(
                self, "Cluster Dynamics ML CLI Setup", str(exc)
            )
            return
        run_file_path = default_clusterdynamicsml_run_file_path(project_dir)
        save_clusterdynamicsml_run_config(run_file_path, config)
        self.run_file_edit.setText(str(run_file_path))
        self.json_preview_box.setPlainText(save_preview_text(config.to_dict()))
        self._update_preview()
        self.statusBar().showMessage(f"Saved run file: {run_file_path}")
        QMessageBox.information(
            self,
            "Cluster Dynamics ML CLI Setup",
            f"Saved cluster dynamics ML CLI run file:\n{run_file_path}",
        )

    def _update_preview(self, *_args: object) -> None:
        project_dir = self._project_dir()
        if project_dir is None:
            self.command_box.setPlainText(
                "Select a project folder before saving the CLI run file."
            )
            self.json_preview_box.clear()
            return
        self._refresh_run_file_path()
        self.command_box.setPlainText(
            f'clusterdynamicsml run "{project_dir}"\n'
            f'saxshell clusterdynamicsml run "{project_dir}"'
        )
        try:
            config = self._current_config(project_dir)
            payload = config.to_dict()
            try:
                payload["selection_preview"] = (
                    preview_clusterdynamicsml_run_config(
                        project_dir=project_dir,
                        config=config,
                    )
                )
            except Exception as exc:
                payload["selection_preview_error"] = str(exc)
        except Exception as exc:
            self.json_preview_box.setPlainText(str(exc))
            return
        self.json_preview_box.setPlainText(save_preview_text(payload))

    def _current_config(self, project_dir: Path):
        frames_text = self.frames_dir_edit.text().strip()
        if not frames_text:
            raise ValueError("Choose a frames folder before saving.")
        output_text = self.output_file_edit.text().strip()
        energy_text = self.energy_file_edit.text().strip()
        frame_format = ""
        if self._last_summary is not None:
            frame_format = str(
                self._last_summary.get("frame_format", "") or ""
            )
        shell_references = (
            self.definitions_panel.shell_reference_definitions()
            if frame_format == "pdb"
            else ()
        )
        return build_clusterdynamicsml_run_config(
            project_dir=project_dir,
            frames_dir=frames_text,
            output_file=output_text or None,
            clusters_dir=self.prediction_panel.clusters_dir(),
            experimental_data_file=(
                self.prediction_panel.experimental_data_file()
            ),
            energy_file=energy_text or None,
            atom_type_definitions=self.definitions_panel.atom_type_definitions(),
            pair_cutoff_definitions=(
                self.definitions_panel.pair_cutoff_definitions()
            ),
            box_dimensions=self.definitions_panel.box_dimensions(),
            use_pbc=self.definitions_panel.use_pbc(),
            default_cutoff=self.definitions_panel.default_cutoff(),
            shell_levels=self.definitions_panel.shell_growth_levels(),
            shared_shells=self.definitions_panel.shared_shells(),
            include_shell_atoms_in_stoichiometry=(
                self.definitions_panel.include_shell_atoms_in_stoichiometry()
            ),
            search_mode=self.definitions_panel.search_mode(),
            shell_reference_definitions=shell_references,
            folder_start_time_fs=self.time_panel.folder_start_time_fs(),
            first_frame_time_fs=self.time_panel.first_frame_time_fs(),
            frame_timestep_fs=self.time_panel.frame_timestep_fs(),
            frames_per_colormap_timestep=(
                self.time_panel.frames_per_colormap_timestep()
            ),
            analysis_start_fs=self.time_panel.analysis_start_fs(),
            analysis_stop_fs=self.time_panel.analysis_stop_fs(),
            target_node_counts=self.prediction_panel.target_node_counts(),
            candidates_per_size=self.prediction_panel.candidates_per_size(),
            prediction_population_share_threshold=(
                self.prediction_panel.prediction_population_share_threshold()
            ),
            q_min=self.prediction_panel.q_min(),
            q_max=self.prediction_panel.q_max(),
            q_points=self.prediction_panel.q_points(),
        )

    def _project_dir(self) -> Path | None:
        text = self.project_dir_edit.text().strip()
        if not text:
            return None
        return Path(text).expanduser().resolve()

    def _require_project_dir(self) -> Path:
        project_dir = self._project_dir()
        if project_dir is None:
            raise ValueError("Choose a project folder before saving.")
        if not project_dir.is_dir():
            raise ValueError(f"Project folder does not exist: {project_dir}")
        return project_dir

    def _load_shell_reference_library_entries(self) -> None:
        try:
            entries = list(list_reference_library())
        except Exception:
            entries = []
        self.definitions_panel.set_shell_reference_library_entries(
            entries,
            emit_signal=False,
        )

    @staticmethod
    def _project_defaults(project_dir: Path) -> dict[str, Path | None]:
        defaults: dict[str, Path | None] = {
            "frames_dir": None,
            "energy_file": None,
            "clusters_dir": None,
            "experimental_data_file": None,
        }
        try:
            settings = SAXSProjectManager().load_project(project_dir)
        except Exception:
            return defaults
        defaults["frames_dir"] = settings.resolved_frames_dir
        defaults["energy_file"] = settings.resolved_energy_file
        defaults["clusters_dir"] = settings.resolved_clusters_dir
        defaults["experimental_data_file"] = (
            settings.resolved_experimental_data_path
        )
        return defaults


def _summary_text(summary: dict[str, object]) -> str:
    box_dimensions = summary.get("box_dimensions")
    if box_dimensions is None:
        box_dimensions = summary.get("estimated_box_dimensions")
    source_kind = summary.get("box_dimensions_source_kind")
    label = (
        "Source box dimensions"
        if source_kind == "source_filename"
        else "Estimated box dimensions"
    )
    lines = [
        f"Frames folder: {summary.get('input_dir')}",
        f"Mode: {summary.get('mode_label')}",
        f"Frames: {summary.get('n_frames')}",
        f"{label}: {format_box_dimensions(box_dimensions)}",
    ]
    if summary.get("box_dimensions_source") is not None:
        lines.append(f"Box source: {summary.get('box_dimensions_source')}")
    return "\n".join(lines)


def save_preview_text(payload: dict[str, object]) -> str:
    return json.dumps(payload, indent=2)


def _optional_resolved_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    return Path(value).expanduser().resolve()


def launch_clusterdynamicsml_run_file_ui(
    *,
    initial_project_dir: str | Path | None = None,
    initial_frames_dir: str | Path | None = None,
    initial_energy_file: str | Path | None = None,
    initial_clusters_dir: str | Path | None = None,
    initial_experimental_data_file: str | Path | None = None,
) -> ClusterDynamicsMLRunFileWindow:
    app = QApplication.instance()
    if app is None:
        prepare_saxshell_application_identity()
        app = QApplication(sys.argv)
    configure_saxshell_application(app)
    window = ClusterDynamicsMLRunFileWindow(
        initial_project_dir=initial_project_dir,
        initial_frames_dir=initial_frames_dir,
        initial_energy_file=initial_energy_file,
        initial_clusters_dir=initial_clusters_dir,
        initial_experimental_data_file=initial_experimental_data_file,
    )
    window.show()
    window.raise_()
    return window


__all__ = [
    "ClusterDynamicsMLRunFileWindow",
    "launch_clusterdynamicsml_run_file_ui",
]
