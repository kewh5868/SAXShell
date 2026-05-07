from __future__ import annotations

import os
import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
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

from saxshell.bondanalysis import (
    AngleTripletDefinition,
    BondAnalysisPreset,
    BondPairDefinition,
    load_presets,
    ordered_preset_names,
)
from saxshell.representativefinder.run_config import (
    build_representativefinder_run_config,
    default_representativefinder_run_file_path,
    save_representativefinder_run_config,
    suggest_run_config_output_dir,
)
from saxshell.representativefinder.workflow import (
    RepresentativeFinderInputInspection,
    RepresentativeFinderSettings,
    inspect_representative_structure_input,
)
from saxshell.saxs.project_manager import SAXSProjectManager
from saxshell.saxs.ui.branding import (
    configure_saxshell_application,
    load_saxshell_icon,
    prepare_saxshell_application_identity,
)

_ALGORITHM_ITEMS = (
    ("Quantile Distance", "target_distribution_quantile_distance"),
    ("Mean/Std Distance", "target_distribution_moment_distance"),
)
_ANALYSIS_MODE_ITEMS = (
    ("All Discovered Stoichiometries", "all"),
    ("Selected Stoichiometry Only", "single"),
)


class RepresentativeFinderRunFileWindow(QMainWindow):
    def __init__(
        self,
        *,
        initial_project_dir: str | Path | None = None,
        initial_input_path: str | Path | None = None,
    ) -> None:
        super().__init__()
        self._inspection: RepresentativeFinderInputInspection | None = None
        self._last_suggested_output_dir: str | None = None
        self._presets: dict[str, BondAnalysisPreset] = {}
        self._browse_start_dir = Path.home()

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
                input_path = self._project_clusters_dir(project_dir)

        self.setWindowTitle("Representative Structure CLI Setup (Beta)")
        self.setWindowIcon(load_saxshell_icon())
        self.resize(1040, 760)
        self._build_ui()
        self._reload_presets()

        if project_dir is not None:
            self.project_dir_edit.setText(str(project_dir))
            self._refresh_run_file_path()
        if input_path is not None and input_path.is_dir():
            self.input_dir_edit.setText(str(input_path))
            self._browse_start_dir = input_path
        self._inspect_input()
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
        splitter.setSizes([500, 540])

        self.left_layout.addWidget(self._build_project_group())
        self.left_layout.addWidget(self._build_input_group())
        self.left_layout.addWidget(self._build_preset_group())
        self.left_layout.addWidget(self._build_measurement_group())
        self.left_layout.addWidget(self._build_scoring_group())
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
        group = QGroupBox("Input")
        form = QFormLayout(group)
        input_row = QHBoxLayout()
        self.input_dir_edit = QLineEdit()
        self.input_dir_edit.editingFinished.connect(self._inspect_input)
        input_row.addWidget(self.input_dir_edit, stretch=1)
        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self._browse_input_dir)
        input_row.addWidget(browse_button)
        input_widget = QWidget()
        input_widget.setLayout(input_row)
        form.addRow("Input folder", input_widget)

        self.analysis_mode_combo = QComboBox()
        for label, value in _ANALYSIS_MODE_ITEMS:
            self.analysis_mode_combo.addItem(label, value)
        self.analysis_mode_combo.currentIndexChanged.connect(
            self._on_analysis_mode_changed
        )
        form.addRow("Analysis mode", self.analysis_mode_combo)

        self.stoichiometry_combo = QComboBox()
        self.stoichiometry_combo.currentIndexChanged.connect(
            self._update_command_preview
        )
        form.addRow("Stoichiometry", self.stoichiometry_combo)

        output_row = QHBoxLayout()
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.editingFinished.connect(
            self._update_command_preview
        )
        output_row.addWidget(self.output_dir_edit, stretch=1)
        output_browse_button = QPushButton("Browse...")
        output_browse_button.clicked.connect(self._browse_output_dir)
        output_row.addWidget(output_browse_button)
        output_widget = QWidget()
        output_widget.setLayout(output_row)
        form.addRow("Output folder", output_widget)
        return group

    def _build_preset_group(self) -> QGroupBox:
        group = QGroupBox("Bondanalysis Preset")
        layout = QHBoxLayout(group)
        self.preset_combo = QComboBox()
        layout.addWidget(self.preset_combo, stretch=1)
        load_button = QPushButton("Load")
        load_button.clicked.connect(self._load_selected_preset)
        layout.addWidget(load_button)
        return group

    def _build_measurement_group(self) -> QGroupBox:
        group = QGroupBox("Measurements")
        layout = QVBoxLayout(group)
        layout.addWidget(QLabel("Bond pairs"))
        self.bond_pairs_edit = QPlainTextEdit()
        self.bond_pairs_edit.setMinimumHeight(90)
        self.bond_pairs_edit.textChanged.connect(self._update_command_preview)
        layout.addWidget(self.bond_pairs_edit)

        layout.addWidget(QLabel("Angle triplets"))
        self.angle_triplets_edit = QPlainTextEdit()
        self.angle_triplets_edit.setMinimumHeight(90)
        self.angle_triplets_edit.textChanged.connect(
            self._update_command_preview
        )
        layout.addWidget(self.angle_triplets_edit)
        return group

    def _build_scoring_group(self) -> QGroupBox:
        group = QGroupBox("Scoring")
        form = QFormLayout(group)
        self.algorithm_combo = QComboBox()
        for label, value in _ALGORITHM_ITEMS:
            self.algorithm_combo.addItem(label, value)
        self.algorithm_combo.currentIndexChanged.connect(
            self._update_command_preview
        )
        form.addRow("Algorithm", self.algorithm_combo)

        self.bond_weight_spin = self._new_float_spin(value=1.0)
        self.bond_weight_spin.valueChanged.connect(
            self._update_command_preview
        )
        form.addRow("Bond weight", self.bond_weight_spin)
        self.angle_weight_spin = self._new_float_spin(value=1.0)
        self.angle_weight_spin.valueChanged.connect(
            self._update_command_preview
        )
        form.addRow("Angle weight", self.angle_weight_spin)
        self.solvent_weight_spin = self._new_float_spin(value=1.0)
        self.solvent_weight_spin.valueChanged.connect(
            self._update_command_preview
        )
        form.addRow("Solvent weight", self.solvent_weight_spin)

        worker_default = min(max(os.cpu_count() or 1, 1), 32)
        self.worker_spin = QSpinBox()
        self.worker_spin.setRange(0, 32)
        self.worker_spin.setValue(worker_default)
        self.worker_spin.valueChanged.connect(self._update_command_preview)
        form.addRow("Worker threads", self.worker_spin)

        self.generate_predicted_checkbox = QCheckBox(
            "Generate predicted optimized representative"
        )
        self.generate_predicted_checkbox.toggled.connect(
            self._update_command_preview
        )
        form.addRow("", self.generate_predicted_checkbox)

        self.overwrite_existing_checkbox = QCheckBox(
            "Overwrite existing project representatives"
        )
        self.overwrite_existing_checkbox.toggled.connect(
            self._update_command_preview
        )
        form.addRow("", self.overwrite_existing_checkbox)
        return group

    def _build_save_group(self) -> QGroupBox:
        group = QGroupBox("Save")
        layout = QHBoxLayout(group)
        inspect_button = QPushButton("Inspect Input")
        inspect_button.clicked.connect(self._inspect_input)
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
        self.inspection_box.setMinimumHeight(240)
        layout.addWidget(self.inspection_box)
        return group

    def _build_command_group(self) -> QGroupBox:
        group = QGroupBox("CLI Command")
        layout = QVBoxLayout(group)
        self.command_box = QPlainTextEdit()
        self.command_box.setReadOnly(True)
        self.command_box.setMinimumHeight(170)
        layout.addWidget(self.command_box)
        self.json_preview_box = QPlainTextEdit()
        self.json_preview_box.setReadOnly(True)
        self.json_preview_box.setMinimumHeight(280)
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

    def _browse_input_dir(self, *_args: object) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "Select representative input folder",
            str(self._browse_start_dir),
        )
        if not selected:
            return
        self.input_dir_edit.setText(selected)
        self._browse_start_dir = Path(selected).expanduser().resolve()
        self._inspect_input()

    def _browse_output_dir(self, *_args: object) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "Select representative output folder",
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
        if not self.input_dir_edit.text().strip():
            clusters_dir = self._project_clusters_dir(project_dir)
            if clusters_dir is not None and clusters_dir.is_dir():
                self.input_dir_edit.setText(str(clusters_dir))
        self._inspect_input()

    def _on_analysis_mode_changed(self, *_args: object) -> None:
        self._refresh_stoichiometry_enabled()
        self._refresh_suggested_output_dir()
        self._update_command_preview()

    def _inspect_input(self, *_args: object) -> None:
        input_text = self.input_dir_edit.text().strip()
        if not input_text:
            self._inspection = None
            self.stoichiometry_combo.clear()
            self.inspection_box.setPlainText("No input folder selected.")
            self._update_command_preview()
            return
        try:
            inspection = inspect_representative_structure_input(input_text)
        except Exception as exc:
            self._inspection = None
            self.stoichiometry_combo.clear()
            self.inspection_box.setPlainText(str(exc))
            self.statusBar().showMessage("Input inspection failed")
            self._update_command_preview()
            return
        self._inspection = inspection
        self.stoichiometry_combo.blockSignals(True)
        self.stoichiometry_combo.clear()
        for stoich in inspection.stoichiometry_folders:
            self.stoichiometry_combo.addItem(
                stoich.structure_label,
                stoich.structure_label,
            )
        self.stoichiometry_combo.blockSignals(False)
        self.inspection_box.setPlainText(inspection.summary_text())
        self._refresh_stoichiometry_enabled()
        self._refresh_suggested_output_dir()
        self.statusBar().showMessage(
            f"Discovered {inspection.stoichiometry_count} stoichiometry folder(s)"
        )
        self._update_command_preview()

    def _refresh_stoichiometry_enabled(self) -> None:
        self.stoichiometry_combo.setEnabled(
            self._analysis_mode() == "single"
            and self.stoichiometry_combo.count() > 0
        )

    def _refresh_run_file_path(self) -> None:
        project_dir = self._project_dir()
        if project_dir is None:
            self.run_file_edit.clear()
            return
        self.run_file_edit.setText(
            str(default_representativefinder_run_file_path(project_dir))
        )

    def _refresh_suggested_output_dir(self) -> None:
        project_dir = self._project_dir()
        input_text = self.input_dir_edit.text().strip()
        if project_dir is None or not input_text:
            return
        try:
            suggested = suggest_run_config_output_dir(
                project_dir=project_dir,
                input_dir=input_text,
                analysis_mode=self._analysis_mode(),
            )
        except Exception:
            return
        current = self.output_dir_edit.text().strip()
        if not current or current == self._last_suggested_output_dir:
            self.output_dir_edit.setText(str(suggested))
        self._last_suggested_output_dir = str(suggested)

    def _reload_presets(self, *, selected_name: str | None = None) -> None:
        self._presets = load_presets()
        previous_name = selected_name or self._selected_preset_name()
        self.preset_combo.blockSignals(True)
        self.preset_combo.clear()
        self.preset_combo.addItem("Select preset...", None)
        selected_index = 0
        for name in ordered_preset_names(self._presets):
            preset = self._presets[name]
            label = f"{name} (Built-in)" if preset.builtin else name
            self.preset_combo.addItem(label, name)
            if name == previous_name:
                selected_index = self.preset_combo.count() - 1
        self.preset_combo.setCurrentIndex(selected_index)
        self.preset_combo.blockSignals(False)

    def _selected_preset_name(self) -> str | None:
        payload = self.preset_combo.currentData()
        if payload is None:
            return None
        return str(payload)

    def _load_selected_preset(self, *_args: object) -> None:
        preset_name = self._selected_preset_name()
        if not preset_name:
            QMessageBox.information(
                self,
                "Representative CLI Setup",
                "Choose a preset first.",
            )
            return
        preset = self._presets.get(preset_name)
        if preset is None:
            QMessageBox.warning(
                self,
                "Representative CLI Setup",
                f"The selected preset is no longer available: {preset_name}",
            )
            return
        self.bond_pairs_edit.setPlainText(
            "\n".join(
                f"{pair.atom1}:{pair.atom2}:{pair.cutoff_angstrom:g}"
                for pair in preset.bond_pairs
            )
        )
        self.angle_triplets_edit.setPlainText(
            "\n".join(
                (
                    f"{triplet.vertex}:{triplet.arm1}:{triplet.arm2}:"
                    f"{triplet.cutoff1_angstrom:g}:"
                    f"{triplet.cutoff2_angstrom:g}"
                )
                for triplet in preset.angle_triplets
            )
        )
        self.statusBar().showMessage(f"Loaded preset: {preset_name}")
        self._update_command_preview()

    def _save_run_file(self, *_args: object) -> None:
        try:
            project_dir = self._require_project_dir()
            config = self._current_config(project_dir)
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Representative CLI Setup",
                str(exc),
            )
            return
        run_file_path = default_representativefinder_run_file_path(project_dir)
        save_representativefinder_run_config(run_file_path, config)
        self.run_file_edit.setText(str(run_file_path))
        self.json_preview_box.setPlainText(save_preview_text(config.to_dict()))
        self._update_command_preview()
        self.statusBar().showMessage(f"Saved run file: {run_file_path}")
        QMessageBox.information(
            self,
            "Representative CLI Setup",
            f"Saved representative CLI run file:\n{run_file_path}",
        )

    def _update_command_preview(self, *_args: object) -> None:
        project_dir = self._project_dir()
        if project_dir is None:
            self.command_box.setPlainText(
                "Select a project folder before saving the CLI run file."
            )
            self.json_preview_box.clear()
            return
        run_file_path = default_representativefinder_run_file_path(project_dir)
        command = f'representativefinder run "{project_dir}"'
        self.command_box.setPlainText(
            command
            + "\n"
            + f'saxshell representativefinder run "{project_dir}"'
        )
        try:
            config = self._current_config(project_dir)
        except Exception as exc:
            self.json_preview_box.setPlainText(str(exc))
            return
        self.run_file_edit.setText(str(run_file_path))
        self.json_preview_box.setPlainText(save_preview_text(config.to_dict()))

    def _current_config(
        self,
        project_dir: Path,
    ):
        input_text = self.input_dir_edit.text().strip()
        if not input_text:
            raise ValueError("Choose an input folder before saving.")
        output_text = self.output_dir_edit.text().strip()
        settings = RepresentativeFinderSettings(
            selection_algorithm=str(
                self.algorithm_combo.currentData()
                or "target_distribution_quantile_distance"
            ),
            bond_weight=float(self.bond_weight_spin.value()),
            angle_weight=float(self.angle_weight_spin.value()),
            solvent_weight=float(self.solvent_weight_spin.value()),
            generate_predicted_optimized_representative=bool(
                self.generate_predicted_checkbox.isChecked()
            ),
            parallel_workers=int(self.worker_spin.value()),
            bond_pairs=self._read_bond_pairs(),
            angle_triplets=self._read_angle_triplets(),
        )
        return build_representativefinder_run_config(
            project_dir=project_dir,
            input_dir=input_text,
            output_dir=output_text or None,
            analysis_mode=self._analysis_mode(),
            settings=settings,
            selected_stoichiometry=self._selected_stoichiometry(),
            overwrite_existing=bool(
                self.overwrite_existing_checkbox.isChecked()
            ),
        )

    def _read_bond_pairs(self) -> tuple[BondPairDefinition, ...]:
        definitions: list[BondPairDefinition] = []
        for raw in self.bond_pairs_edit.toPlainText().splitlines():
            text = raw.strip()
            if not text:
                continue
            parts = [part.strip() for part in text.split(":")]
            if len(parts) != 3:
                raise ValueError("Bond-pair rows must use ATOM1:ATOM2:CUTOFF.")
            definitions.append(
                BondPairDefinition(parts[0], parts[1], float(parts[2]))
            )
        return tuple(definitions)

    def _read_angle_triplets(self) -> tuple[AngleTripletDefinition, ...]:
        definitions: list[AngleTripletDefinition] = []
        for raw in self.angle_triplets_edit.toPlainText().splitlines():
            text = raw.strip()
            if not text:
                continue
            parts = [part.strip() for part in text.split(":")]
            if len(parts) != 5:
                raise ValueError(
                    "Angle-triplet rows must use "
                    "VERTEX:ARM1:ARM2:CUTOFF1:CUTOFF2."
                )
            definitions.append(
                AngleTripletDefinition(
                    parts[0],
                    parts[1],
                    parts[2],
                    float(parts[3]),
                    float(parts[4]),
                )
            )
        return tuple(definitions)

    def _analysis_mode(self) -> str:
        return str(self.analysis_mode_combo.currentData() or "all")

    def _selected_stoichiometry(self) -> str | None:
        if self._analysis_mode() != "single":
            return None
        payload = self.stoichiometry_combo.currentData()
        if payload is None:
            return None
        return str(payload)

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
    def _project_clusters_dir(project_dir: Path) -> Path | None:
        try:
            settings = SAXSProjectManager().load_project(project_dir)
        except Exception:
            return None
        return settings.resolved_clusters_dir

    @staticmethod
    def _new_float_spin(*, value: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(0.0, 100.0)
        spin.setSingleStep(0.1)
        spin.setDecimals(3)
        spin.setValue(value)
        return spin


def save_preview_text(payload: dict[str, object]) -> str:
    import json

    return json.dumps(payload, indent=2)


def launch_representativefinder_run_file_ui(
    *,
    initial_project_dir: str | Path | None = None,
    initial_input_path: str | Path | None = None,
) -> RepresentativeFinderRunFileWindow:
    app = QApplication.instance()
    if app is None:
        prepare_saxshell_application_identity()
        app = QApplication(sys.argv)
    configure_saxshell_application(app)
    window = RepresentativeFinderRunFileWindow(
        initial_project_dir=initial_project_dir,
        initial_input_path=initial_input_path,
    )
    window.show()
    window.raise_()
    return window


__all__ = [
    "RepresentativeFinderRunFileWindow",
    "launch_representativefinder_run_file_ui",
]
