from __future__ import annotations

import json
import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
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
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from saxshell.saxs.ui.branding import (
    configure_saxshell_application,
    load_saxshell_icon,
    prepare_saxshell_application_identity,
)
from saxshell.xyz2pdb.mapping_workflow import (
    XYZToPDBMappingWorkflow,
    reference_bond_tolerances,
)
from saxshell.xyz2pdb.run_config import (
    build_xyz2pdb_run_config,
    default_xyz2pdb_run_file_path,
    save_xyz2pdb_run_config,
)
from saxshell.xyz2pdb.ui.input_panel import XYZToPDBInputPanel
from saxshell.xyz2pdb.ui.mapping_panel import XYZToPDBMappingPanel
from saxshell.xyz2pdb.ui.reference_panel import ReferenceLibraryPanel
from saxshell.xyz2pdb.workflow import (
    list_reference_library,
    suggest_output_dir,
)


class XYZToPDBRunFileWindow(QMainWindow):
    def __init__(
        self,
        *,
        initial_project_dir: str | Path | None = None,
        initial_input_path: str | Path | None = None,
    ) -> None:
        super().__init__()
        self._browse_start_dir = Path.home()
        self._last_suggested_output_dir: str | None = None

        project_dir = (
            None
            if initial_project_dir is None
            else Path(initial_project_dir).expanduser().resolve()
        )
        input_path = (
            None
            if initial_input_path is None
            else Path(initial_input_path).expanduser().resolve()
        )
        if project_dir is not None:
            self._browse_start_dir = project_dir
            if input_path is None:
                input_path = self._project_frames_dir(project_dir)

        self.setWindowTitle("XYZ -> PDB CLI Setup")
        self.setWindowIcon(load_saxshell_icon())
        self.resize(1180, 820)
        self._build_ui()

        if project_dir is not None:
            self.project_dir_edit.setText(str(project_dir))
            self._refresh_run_file_path()
        if input_path is not None:
            self.input_panel.input_edit.setText(str(input_path))
            self._browse_start_dir = (
                input_path.parent if input_path.is_file() else input_path
            )
        self.refresh_reference_library()
        self.inspect_input()
        self._update_command_preview()

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
        splitter.setSizes([610, 570])

        self.input_panel = XYZToPDBInputPanel()
        self.reference_panel = ReferenceLibraryPanel()
        self.mapping_panel = XYZToPDBMappingPanel()
        self.input_panel.inspect_requested.connect(self.inspect_input)
        self.input_panel.input_path_changed.connect(
            lambda _path: self._refresh_suggested_output_dir()
        )
        self.input_panel.settings_changed.connect(self._update_command_preview)
        self.reference_panel.refresh_requested.connect(
            self.refresh_reference_library
        )
        self.reference_panel.library_dir_changed.connect(
            lambda _path: self.refresh_reference_library()
        )
        self.mapping_panel.settings_changed.connect(
            self._update_command_preview
        )

        self.left_layout.addWidget(self._build_project_group())
        self.left_layout.addWidget(self.input_panel)
        self.left_layout.addWidget(self.reference_panel)
        self.left_layout.addWidget(self.mapping_panel)
        self.left_layout.addWidget(self._build_options_group())
        self.left_layout.addWidget(self._build_save_group())
        self.left_layout.addStretch(1)

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

    def _build_options_group(self) -> QGroupBox:
        group = QGroupBox("Run Options")
        form = QFormLayout(group)
        form.addRow("Output folder", self._output_row())

        self.selected_solution_spin = QSpinBox()
        self.selected_solution_spin.setRange(0, 63)
        self.selected_solution_spin.valueChanged.connect(
            self._update_command_preview
        )
        form.addRow("Solution index", self.selected_solution_spin)

        self.assertion_mode_checkbox = QCheckBox("Assertion mode")
        self.assertion_mode_checkbox.toggled.connect(
            self._update_command_preview
        )
        form.addRow("", self.assertion_mode_checkbox)

        self.pbc_params_edit = QPlainTextEdit()
        self.pbc_params_edit.setPlaceholderText(
            '{"a": 20.0, "b": 20.0, "c": 20.0}'
        )
        self.pbc_params_edit.setMinimumHeight(80)
        self.pbc_params_edit.textChanged.connect(self._update_command_preview)
        form.addRow("PBC JSON", self.pbc_params_edit)
        return group

    def _output_row(self) -> QWidget:
        widget = QWidget()
        row = QHBoxLayout(widget)
        row.setContentsMargins(0, 0, 0, 0)
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.textChanged.connect(self._update_command_preview)
        row.addWidget(self.output_dir_edit, stretch=1)
        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self._browse_output_dir)
        row.addWidget(browse_button)
        return widget

    def _build_save_group(self) -> QGroupBox:
        group = QGroupBox("Save")
        layout = QHBoxLayout(group)
        inspect_button = QPushButton("Analyze Input")
        inspect_button.clicked.connect(self.inspect_input)
        layout.addWidget(inspect_button)
        save_button = QPushButton("Save Run File")
        save_button.clicked.connect(self._save_run_file)
        layout.addWidget(save_button)
        layout.addStretch(1)
        return group

    def _build_command_group(self) -> QGroupBox:
        group = QGroupBox("CLI Command")
        layout = QVBoxLayout(group)
        self.command_box = QPlainTextEdit()
        self.command_box.setReadOnly(True)
        self.command_box.setMinimumHeight(140)
        layout.addWidget(self.command_box)

        layout.addWidget(QLabel("Run File JSON Preview"))
        self.json_preview_box = QPlainTextEdit()
        self.json_preview_box.setReadOnly(True)
        self.json_preview_box.setMinimumHeight(420)
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

    def _browse_output_dir(self, *_args: object) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "Select xyz2pdb output folder",
            self.output_dir_edit.text().strip() or str(self._browse_start_dir),
        )
        if selected:
            self.output_dir_edit.setText(selected)
            self._update_command_preview()

    def _on_project_dir_changed(self, *_args: object) -> None:
        project_dir = self._project_dir()
        if project_dir is None:
            return
        self._browse_start_dir = project_dir
        self._refresh_run_file_path()
        if not self.input_panel.input_edit.text().strip():
            input_path = self._project_frames_dir(project_dir)
            if input_path is not None and input_path.exists():
                self.input_panel.input_edit.setText(str(input_path))
        self._update_command_preview()

    def refresh_reference_library(self, *_args: object) -> None:
        try:
            library_dir = self.reference_panel.get_library_dir()
            entries = list_reference_library(library_dir)
            self.reference_panel.set_reference_entries(entries)
            self.mapping_panel.set_reference_entries(
                entries,
                bond_defaults_by_name={
                    entry.name: reference_bond_tolerances(
                        entry.name,
                        library_dir=library_dir,
                    )
                    for entry in entries
                },
            )
            self.statusBar().showMessage("Reference library refreshed")
        except Exception as exc:
            self.statusBar().showMessage("Reference refresh failed")
            self.json_preview_box.setPlainText(str(exc))
        self._update_command_preview()

    def inspect_input(self, *_args: object) -> None:
        input_path = self.input_panel.get_input_path()
        if input_path is None:
            self.input_panel.set_summary_text("No XYZ input selected.")
            self.input_panel.set_input_mode(None)
            self._update_command_preview()
            return
        try:
            workflow = XYZToPDBMappingWorkflow(
                input_path,
                reference_library_dir=self.reference_panel.get_library_dir(),
                output_dir=self._output_dir(),
            )
            analysis = workflow.analyze_input()
            self.input_panel.set_input_mode(analysis.inspection.input_mode)
            self.input_panel.set_summary_text(
                "\n".join(
                    [
                        f"Input path: {analysis.inspection.input_path}",
                        f"XYZ files found: {analysis.inspection.total_files}",
                        f"Sample frame: {analysis.sample_file.name}",
                        f"Sample atoms: {analysis.total_atoms}",
                        "Element counts: "
                        + ", ".join(
                            f"{element} x{count}"
                            for element, count in sorted(
                                analysis.element_counts.items()
                            )
                        ),
                    ]
                )
            )
            self.mapping_panel.set_available_elements(
                tuple(sorted(analysis.element_counts))
            )
            self._refresh_suggested_output_dir()
        except Exception as exc:
            self.input_panel.set_summary_text(str(exc))
            self.input_panel.set_input_mode(None)
            self.statusBar().showMessage("Input analysis failed")
        self._update_command_preview()

    def _save_run_file(self, *_args: object) -> None:
        try:
            project_dir = self._require_project_dir()
            config = self._current_config(project_dir)
        except Exception as exc:
            QMessageBox.warning(self, "XYZ -> PDB CLI Setup", str(exc))
            return
        run_file_path = default_xyz2pdb_run_file_path(project_dir)
        save_xyz2pdb_run_config(run_file_path, config)
        self.run_file_edit.setText(str(run_file_path))
        self.json_preview_box.setPlainText(save_preview_text(config.to_dict()))
        self._update_command_preview()
        self.statusBar().showMessage(f"Saved run file: {run_file_path}")
        QMessageBox.information(
            self,
            "XYZ -> PDB CLI Setup",
            f"Saved XYZ -> PDB CLI run file:\n{run_file_path}",
        )

    def _update_command_preview(self, *_args: object) -> None:
        project_dir = self._project_dir()
        if project_dir is None:
            self.command_box.setPlainText(
                "Select a project folder before saving the CLI run file."
            )
            self.json_preview_box.clear()
            return
        run_file_path = default_xyz2pdb_run_file_path(project_dir)
        self.run_file_edit.setText(str(run_file_path))
        self.command_box.setPlainText(
            f'xyz2pdb run "{project_dir}"\n'
            f'saxshell xyz2pdb run "{project_dir}"'
        )
        try:
            config = self._current_config(project_dir)
        except Exception as exc:
            self.json_preview_box.setPlainText(str(exc))
            return
        self.json_preview_box.setPlainText(save_preview_text(config.to_dict()))

    def _current_config(self, project_dir: Path):
        input_path = self.input_panel.get_input_path()
        if input_path is None:
            raise ValueError("Choose an XYZ input before saving.")
        return build_xyz2pdb_run_config(
            project_dir=project_dir,
            input_path=input_path,
            output_dir=self._output_dir(),
            reference_library_dir=self.reference_panel.get_library_dir(),
            molecule_inputs=tuple(self.mapping_panel.get_molecule_inputs()),
            free_atom_inputs=tuple(self.mapping_panel.get_free_atom_inputs()),
            hydrogen_mode=self.mapping_panel.hydrogen_mode(),
            selected_solution_index=int(self.selected_solution_spin.value()),
            assertion_mode=bool(self.assertion_mode_checkbox.isChecked()),
            pbc_params=self._pbc_params(),
        )

    def _refresh_run_file_path(self) -> None:
        project_dir = self._project_dir()
        if project_dir is None:
            self.run_file_edit.clear()
            return
        self.run_file_edit.setText(
            str(default_xyz2pdb_run_file_path(project_dir))
        )

    def _refresh_suggested_output_dir(self) -> None:
        input_path = self.input_panel.get_input_path()
        if input_path is None:
            return
        try:
            suggested = suggest_output_dir(input_path)
        except Exception:
            return
        current = self.output_dir_edit.text().strip()
        if not current or current == self._last_suggested_output_dir:
            self.output_dir_edit.setText(str(suggested))
        self._last_suggested_output_dir = str(suggested)

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
        from saxshell.saxs.project_manager import SAXSProjectManager

        try:
            settings = SAXSProjectManager().load_project(project_dir)
        except Exception:
            return None
        return settings.resolved_frames_dir

    def _output_dir(self) -> Path | None:
        text = self.output_dir_edit.text().strip()
        return Path(text) if text else None

    def _pbc_params(self) -> dict[str, float | str]:
        text = self.pbc_params_edit.toPlainText().strip()
        if not text:
            return {}
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError("PBC JSON must be an object.")
        return dict(payload)


def save_preview_text(payload: dict[str, object]) -> str:
    return json.dumps(payload, indent=2)


def launch_xyz2pdb_run_file_ui(
    *,
    initial_project_dir: str | Path | None = None,
    initial_input_path: str | Path | None = None,
) -> XYZToPDBRunFileWindow:
    app = QApplication.instance()
    if app is None:
        prepare_saxshell_application_identity()
        app = QApplication(sys.argv)
    configure_saxshell_application(app)
    window = XYZToPDBRunFileWindow(
        initial_project_dir=initial_project_dir,
        initial_input_path=initial_input_path,
    )
    window.show()
    window.raise_()
    return window


__all__ = [
    "XYZToPDBRunFileWindow",
    "launch_xyz2pdb_run_file_ui",
]
