from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class ClusterTrajectoryPanel(QGroupBox):
    """Panel for selecting and inspecting extracted frame folders."""

    inspect_requested = Signal()
    settings_changed = Signal()
    frames_dir_changed = Signal(object)
    project_source_changed = Signal(object)

    def __init__(self) -> None:
        super().__init__("Extracted Frames")
        self._project_xyz_frames_dir: Path | None = None
        self._project_pdb_frames_dir: Path | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout()
        form = QFormLayout()

        self.project_source_widget = QWidget()
        source_row = QHBoxLayout(self.project_source_widget)
        source_row.setContentsMargins(0, 0, 0, 0)
        self.project_source_combo = QComboBox()
        self.project_source_combo.currentIndexChanged.connect(
            self._on_project_source_changed
        )
        source_row.addWidget(self.project_source_combo)
        self.project_source_hint = QLabel(
            "Switch between the project XYZ folder and optional PDB folder."
        )
        self.project_source_hint.setWordWrap(True)
        source_row.addWidget(self.project_source_hint, stretch=1)
        form.addRow("Project source", self.project_source_widget)
        self.project_source_widget.setVisible(False)

        self.mode_label = QLabel("Mode: Auto-detect")
        self.mode_label.setToolTip(
            "The detected extracted-frame mode for the selected folder. "
            "Cluster analysis switches between PDB and XYZ behavior "
            "automatically."
        )
        layout.addWidget(self.mode_label)

        self.frames_dir_edit = QLineEdit()
        self.frames_dir_edit.setToolTip(
            "Path to a folder of extracted single-frame PDB or XYZ files, "
            "such as the output from mdtrajectory."
        )
        self.frames_dir_edit.textChanged.connect(
            lambda _text: self.frames_dir_changed.emit(self.get_frames_dir())
        )
        self.frames_dir_edit.textChanged.connect(
            lambda _text: self.settings_changed.emit()
        )
        form.addRow(
            "Frames folder",
            self._make_path_row(self.frames_dir_edit),
        )

        layout.addLayout(form)

        inspect_button = QPushButton("Inspect Frames Folder")
        inspect_button.setToolTip(
            "Inspect the selected extracted-frames folder, detect whether it "
            "contains PDB or XYZ frames, and count the files available for "
            "cluster analysis."
        )
        inspect_button.clicked.connect(
            lambda _checked=False: self.inspect_requested.emit()
        )
        layout.addWidget(inspect_button)

        layout.addWidget(QLabel("Folder Summary"))
        self.summary_box = QTextEdit()
        self.summary_box.setReadOnly(True)
        self.summary_box.setMinimumHeight(160)
        self.summary_box.setToolTip(
            "Summary of the extracted frames folder, including the detected "
            "mode and the number of frame files found."
        )
        layout.addWidget(self.summary_box)

        self.setLayout(layout)

    def _on_project_source_changed(self, index: int) -> None:
        if index < 0:
            return
        kind = self.project_source_combo.currentData()
        if kind == "xyz":
            path = self._project_xyz_frames_dir
        elif kind == "pdb":
            path = self._project_pdb_frames_dir
        else:
            path = None
        if path is not None and self.frames_dir_edit.text() != str(path):
            self.frames_dir_edit.setText(str(path))
        self.project_source_changed.emit(kind)

    def _make_path_row(self, line_edit: QLineEdit) -> QWidget:
        widget = QWidget()
        row = QHBoxLayout(widget)
        row.setContentsMargins(0, 0, 0, 0)

        button = QPushButton("Browse")
        button.setToolTip("Browse for an extracted frames folder.")
        button.clicked.connect(
            lambda _checked=False: self._choose_dir(line_edit)
        )

        row.addWidget(line_edit)
        row.addWidget(button)
        return widget

    def _choose_dir(self, line_edit: QLineEdit) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            "Select extracted frames folder",
        )
        if path:
            line_edit.setText(path)

    def set_frame_mode(
        self,
        frame_format: str | None,
        *,
        detail: str | None = None,
    ) -> None:
        if frame_format == "pdb":
            text = "Mode: PDB frames"
        elif frame_format == "xyz":
            text = "Mode: XYZ frames"
        else:
            text = "Mode: Auto-detect"
        if detail:
            text = f"{text} ({detail})"
        self.mode_label.setText(text)

    def set_project_frame_sources(
        self,
        xyz_frames_dir: Path | None,
        pdb_frames_dir: Path | None,
        *,
        active_kind: str | None = None,
    ) -> None:
        self._project_xyz_frames_dir = xyz_frames_dir
        self._project_pdb_frames_dir = pdb_frames_dir
        self.project_source_combo.blockSignals(True)
        self.project_source_combo.clear()
        if xyz_frames_dir is not None:
            self.project_source_combo.addItem("XYZ frames folder", "xyz")
        if pdb_frames_dir is not None:
            self.project_source_combo.addItem("PDB structure folder", "pdb")
        if self.project_source_combo.count() == 0:
            self.project_source_widget.setVisible(False)
            self.project_source_combo.blockSignals(False)
            return
        self.project_source_widget.setVisible(
            xyz_frames_dir is not None and pdb_frames_dir is not None
        )
        if active_kind is not None:
            index = self.project_source_combo.findData(active_kind)
            if index >= 0:
                self.project_source_combo.setCurrentIndex(index)
        self.project_source_combo.blockSignals(False)

    def selected_project_source_kind(self) -> str | None:
        if self.project_source_combo.count() == 0:
            return None
        current = self.project_source_combo.currentData()
        return None if current is None else str(current)

    def get_frames_dir(self) -> Path | None:
        text = self.frames_dir_edit.text().strip()
        return Path(text) if text else None

    def set_summary(self, summary: dict[str, object]) -> None:
        lines = [f"{key}: {value}" for key, value in summary.items()]
        self.summary_box.setPlainText("\n".join(lines))

    def set_summary_text(self, text: str) -> None:
        self.summary_box.setPlainText(text)
