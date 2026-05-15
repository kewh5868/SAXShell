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

from saxshell.cluster import (
    DEFAULT_CLUSTER_EXTRACTION_PRESET_NAME,
    ClusterWorkflow,
    format_box_dimensions,
)
from saxshell.cluster.run_config import (
    build_cluster_run_config,
    default_cluster_run_file_path,
    preview_cluster_run_config,
    save_cluster_run_config,
    suggest_run_config_output_dir,
)
from saxshell.cluster.ui.definitions_panel import ClusterDefinitionsPanel


class ClusterRunFileWindow(QMainWindow):
    def __init__(
        self,
        *,
        initial_project_dir: str | Path | None = None,
        initial_frames_dir: str | Path | None = None,
    ) -> None:
        super().__init__()
        self._browse_start_dir = Path.home()
        self._last_suggested_output_dir: str | None = None
        self._last_summary: dict[str, object] | None = None

        project_dir = (
            None
            if initial_project_dir is None
            else Path(initial_project_dir).expanduser().resolve()
        )
        frames_dir = (
            None
            if initial_frames_dir is None
            else Path(initial_frames_dir).expanduser().resolve()
        )
        if project_dir is not None:
            self._browse_start_dir = project_dir
            if frames_dir is None:
                frames_dir = self._project_frames_dir(project_dir)

        self.setWindowTitle("Cluster Extraction CLI Setup (Beta)")
        self.setWindowIcon(_load_saxshell_icon())
        self.resize(1100, 780)
        self._build_ui()
        self.definitions_panel.load_preset(
            DEFAULT_CLUSTER_EXTRACTION_PRESET_NAME
        )

        if project_dir is not None:
            self.project_dir_edit.setText(str(project_dir))
            self._refresh_run_file_path()
        if frames_dir is not None and frames_dir.is_dir():
            self.frames_dir_edit.setText(str(frames_dir))
            self._browse_start_dir = frames_dir
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
        splitter.setSizes([560, 540])

        self.left_layout.addWidget(self._build_project_group())
        self.left_layout.addWidget(self._build_frames_group())
        self.definitions_panel = ClusterDefinitionsPanel()
        self.definitions_panel.settings_changed.connect(self._update_preview)
        self.left_layout.addWidget(self.definitions_panel)
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

    def _build_frames_group(self) -> QGroupBox:
        group = QGroupBox("Input / Output")
        form = QFormLayout(group)
        frames_row = QHBoxLayout()
        self.frames_dir_edit = QLineEdit()
        self.frames_dir_edit.editingFinished.connect(self._inspect_frames)
        frames_row.addWidget(self.frames_dir_edit, stretch=1)
        frames_button = QPushButton("Browse...")
        frames_button.clicked.connect(self._browse_frames_dir)
        frames_row.addWidget(frames_button)
        frames_widget = QWidget()
        frames_widget.setLayout(frames_row)
        form.addRow("Frames folder", frames_widget)

        output_row = QHBoxLayout()
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.editingFinished.connect(self._update_preview)
        output_row.addWidget(self.output_dir_edit, stretch=1)
        output_button = QPushButton("Browse...")
        output_button.clicked.connect(self._browse_output_dir)
        output_row.addWidget(output_button)
        output_widget = QWidget()
        output_widget.setLayout(output_row)
        form.addRow("Output clusters folder", output_widget)
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
        self.command_box.setMinimumHeight(130)
        layout.addWidget(self.command_box)
        layout.addWidget(QLabel("Run file preview"))
        self.json_preview_box = QPlainTextEdit()
        self.json_preview_box.setReadOnly(True)
        self.json_preview_box.setMinimumHeight(300)
        layout.addWidget(self.json_preview_box)
        return group

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

    def _browse_output_dir(self, *_args: object) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "Select output clusters folder",
            self.output_dir_edit.text().strip() or str(self._browse_start_dir),
        )
        if selected:
            self.output_dir_edit.setText(selected)
            self._update_preview()

    def _on_project_dir_changed(self, *_args: object) -> None:
        project_dir = self._project_dir()
        if project_dir is None:
            return
        self._browse_start_dir = project_dir
        self._refresh_run_file_path()
        if not self.frames_dir_edit.text().strip():
            frames_dir = self._project_frames_dir(project_dir)
            if frames_dir is not None and frames_dir.is_dir():
                self.frames_dir_edit.setText(str(frames_dir))
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
        self.definitions_panel.set_frame_mode(
            str(summary.get("frame_format", "") or "")
        )
        self.inspection_box.setPlainText(_summary_text(summary))
        self._refresh_suggested_output_dir()
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
            str(default_cluster_run_file_path(project_dir))
        )

    def _refresh_suggested_output_dir(self) -> None:
        frames_text = self.frames_dir_edit.text().strip()
        if not frames_text:
            return
        try:
            suggested = suggest_run_config_output_dir(frames_dir=frames_text)
        except Exception:
            return
        current = self.output_dir_edit.text().strip()
        if not current or current == self._last_suggested_output_dir:
            self.output_dir_edit.setText(str(suggested))
        self._last_suggested_output_dir = str(suggested)

    def _save_run_file(self, *_args: object) -> None:
        try:
            project_dir = self._require_project_dir()
            config = self._current_config(project_dir)
        except Exception as exc:
            QMessageBox.warning(self, "Cluster CLI Setup", str(exc))
            return
        run_file_path = default_cluster_run_file_path(project_dir)
        save_cluster_run_config(run_file_path, config)
        self.run_file_edit.setText(str(run_file_path))
        self.json_preview_box.setPlainText(save_preview_text(config.to_dict()))
        self._update_preview()
        self.statusBar().showMessage(f"Saved run file: {run_file_path}")
        QMessageBox.information(
            self,
            "Cluster CLI Setup",
            f"Saved cluster extraction CLI run file:\n{run_file_path}",
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
            f'clusters run "{project_dir}"\n'
            f'saxshell cluster run "{project_dir}"'
        )
        try:
            config = self._current_config(project_dir)
            payload = config.to_dict()
            try:
                payload["selection_preview"] = preview_cluster_run_config(
                    project_dir=project_dir,
                    config=config,
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
        output_text = self.output_dir_edit.text().strip()
        return build_cluster_run_config(
            project_dir=project_dir,
            frames_dir=frames_text,
            output_dir=output_text or None,
            atom_type_definitions=self.definitions_panel.atom_type_definitions(),
            pair_cutoff_definitions=(
                self.definitions_panel.pair_cutoff_definitions()
            ),
            box_dimensions=self.definitions_panel.box_dimensions(),
            use_pbc=self.definitions_panel.use_pbc(),
            default_cutoff=self.definitions_panel.default_cutoff(),
            shell_levels=self.definitions_panel.shell_growth_levels(),
            include_shell_levels=self.definitions_panel.include_shell_levels(),
            shared_shells=self.definitions_panel.shared_shells(),
            smart_solvation_shells=(
                self.definitions_panel.smart_solvation_shells()
            ),
            include_shell_atoms_in_stoichiometry=(
                self.definitions_panel.include_shell_atoms_in_stoichiometry()
            ),
            search_mode=self.definitions_panel.search_mode(),
            save_state_frequency=self.definitions_panel.save_state_frequency(),
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

    @staticmethod
    def _project_frames_dir(project_dir: Path) -> Path | None:
        try:
            payload = json.loads(
                (project_dir / "saxs_project.json").read_text(encoding="utf-8")
            )
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        frames_dir = _optional_project_path(payload.get("frames_dir"))
        return frames_dir or _optional_project_path(
            payload.get("pdb_frames_dir")
        )


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
        f"Output format: {summary.get('output_file_extension')}",
        f"{label}: {format_box_dimensions(box_dimensions)}",
    ]
    if summary.get("box_dimensions_source") is not None:
        lines.append(f"Box source: {summary.get('box_dimensions_source')}")
    return "\n".join(lines)


def save_preview_text(payload: dict[str, object]) -> str:
    return json.dumps(payload, indent=2)


def _optional_project_path(value: object) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    return Path(text).expanduser().resolve()


def _load_saxshell_icon():
    from saxshell.saxs.ui.branding import load_saxshell_icon

    return load_saxshell_icon()


def launch_cluster_run_file_ui(
    *,
    initial_project_dir: str | Path | None = None,
    initial_frames_dir: str | Path | None = None,
) -> ClusterRunFileWindow:
    from saxshell.saxs.ui.branding import (
        configure_saxshell_application,
        prepare_saxshell_application_identity,
    )

    app = QApplication.instance()
    if app is None:
        prepare_saxshell_application_identity()
        app = QApplication(sys.argv)
    configure_saxshell_application(app)
    window = ClusterRunFileWindow(
        initial_project_dir=initial_project_dir,
        initial_frames_dir=initial_frames_dir,
    )
    window.show()
    window.raise_()
    return window


__all__ = [
    "ClusterRunFileWindow",
    "launch_cluster_run_file_ui",
]
