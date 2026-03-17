from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QFileDialog
from PySide6.QtWidgets import QFormLayout
from PySide6.QtWidgets import QGroupBox
from PySide6.QtWidgets import QHBoxLayout
from PySide6.QtWidgets import QLabel
from PySide6.QtWidgets import QLineEdit
from PySide6.QtWidgets import QPushButton
from PySide6.QtWidgets import QSpinBox
from PySide6.QtWidgets import QTextEdit
from PySide6.QtWidgets import QVBoxLayout
from PySide6.QtWidgets import QWidget

from saxshell.mdtrajectory.frame.manager import TrajectoryManager


class TrajectoryPanel(QGroupBox):
    """Panel for trajectory input and inspection."""

    inspect_requested = Signal()
    selection_changed = Signal()
    trajectory_path_changed = Signal(object)

    def __init__(self) -> None:
        super().__init__("Trajectory")
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout()

        form = QFormLayout()

        self.trajectory_edit = QLineEdit()
        self.topology_edit = QLineEdit()
        self.energy_edit = QLineEdit()
        self.trajectory_edit.textChanged.connect(
            lambda _text: self._emit_trajectory_path_changed()
        )

        form.addRow(
            "Trajectory file",
            self._make_path_row(
                self.trajectory_edit,
                "trajectory",
                (
                    "Path to the CP2K trajectory file to inspect and export. "
                    "Currently .xyz and .pdb are supported."
                ),
            ),
        )
        form.addRow(
            "Topology file",
            self._make_path_row(
                self.topology_edit,
                "topology",
                (
                    "Optional topology/reference file for trajectory formats "
                    "that need extra structure metadata."
                ),
            ),
        )
        form.addRow(
            "CP2K .ener file",
            self._make_path_row(
                self.energy_edit,
                "energy",
                (
                    "Optional CP2K .ener file used to plot temperature and "
                    "suggest a steady-state cutoff."
                ),
            ),
        )

        self.start_spin = QSpinBox()
        self.start_spin.setRange(-1, 10**9)
        self.start_spin.setValue(-1)
        self.start_spin.setToolTip(
            "First frame index to include. Use -1 to start from the first "
            "available frame."
        )
        self.start_spin.valueChanged.connect(
            lambda _value: self.selection_changed.emit()
        )

        self.stop_spin = QSpinBox()
        self.stop_spin.setRange(-1, 10**9)
        self.stop_spin.setValue(-1)
        self.stop_spin.setToolTip(
            "Frame index where export stops. Use -1 to continue through the "
            "last available frame."
        )
        self.stop_spin.valueChanged.connect(
            lambda _value: self.selection_changed.emit()
        )

        self.stride_spin = QSpinBox()
        self.stride_spin.setRange(1, 10**9)
        self.stride_spin.setValue(1)
        self.stride_spin.setToolTip(
            "Keep every Nth frame after the start/stop slice is applied."
        )
        self.stride_spin.valueChanged.connect(
            lambda _value: self.selection_changed.emit()
        )

        form.addRow("Start frame", self.start_spin)
        form.addRow("Stop frame", self.stop_spin)
        form.addRow("Stride", self.stride_spin)

        layout.addLayout(form)

        button_row = QHBoxLayout()
        self.inspect_button = QPushButton("Inspect Trajectory")
        self.inspect_button.setToolTip(
            "Read trajectory metadata and, if provided, load the CP2K energy "
            "profile for cutoff selection."
        )
        self.inspect_button.clicked.connect(
            lambda _checked=False: self.inspect_requested.emit()
        )
        button_row.addWidget(self.inspect_button)
        button_row.addStretch(1)
        layout.addLayout(button_row)

        self.summary_box = QTextEdit()
        self.summary_box.setReadOnly(True)
        self.summary_box.setMinimumHeight(160)
        self.summary_box.setToolTip(
            "Inspection results such as file type and estimated frame count."
        )
        layout.addWidget(QLabel("Inspection Summary"))
        layout.addWidget(self.summary_box)

        self.setLayout(layout)

    def _make_path_row(
        self,
        line_edit: QLineEdit,
        file_kind: str,
        tooltip: str,
    ) -> QWidget:
        widget = QWidget()
        row = QHBoxLayout(widget)
        row.setContentsMargins(0, 0, 0, 0)

        button = QPushButton("Browse")
        button.setToolTip(f"Browse for the {file_kind} file.")
        button.clicked.connect(
            lambda _checked=False: self._choose_file(
                line_edit,
                file_kind,
            )
        )

        line_edit.setToolTip(tooltip)
        widget.setToolTip(tooltip)
        row.addWidget(line_edit)
        row.addWidget(button)
        return widget

    def _choose_file(
        self,
        line_edit: QLineEdit,
        file_kind: str,
    ) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            f"Select {file_kind} file",
            filter=self._file_filter(file_kind),
        )
        if path:
            line_edit.setText(path)

    def _file_filter(self, file_kind: str) -> str:
        if file_kind == "trajectory":
            return "Trajectory files (*.xyz *.pdb);;All files (*)"
        if file_kind == "energy":
            return "CP2K energy files (*.ener *.txt *.dat);;All files (*)"
        if file_kind == "topology":
            return "Topology files (*.pdb *.gro *.psf *.top);;All files (*)"
        return "All files (*)"

    def get_trajectory_path(self) -> Path | None:
        text = self.trajectory_edit.text().strip()
        return Path(text) if text else None

    def get_topology_path(self) -> Path | None:
        text = self.topology_edit.text().strip()
        return Path(text) if text else None

    def get_energy_path(self) -> Path | None:
        text = self.energy_edit.text().strip()
        return Path(text) if text else None

    def get_start(self) -> int | None:
        value = self.start_spin.value()
        return None if value < 0 else value

    def get_stop(self) -> int | None:
        value = self.stop_spin.value()
        return None if value < 0 else value

    def get_stride(self) -> int:
        return self.stride_spin.value()

    def set_summary(self, summary: dict[str, object]) -> None:
        lines = [f"{key}: {value}" for key, value in summary.items()]
        self.summary_box.setPlainText("\n".join(lines))

    def set_summary_text(self, text: str) -> None:
        self.summary_box.setPlainText(text)

    def _emit_trajectory_path_changed(self) -> None:
        self.trajectory_path_changed.emit(self.get_trajectory_path())

    def inspect_current(self) -> dict[str, object]:
        trajectory = self.get_trajectory_path()
        topology = self.get_topology_path()

        if trajectory is None:
            raise ValueError("No trajectory file selected.")

        manager = TrajectoryManager(
            input_file=trajectory,
            topology_file=topology,
            backend="auto",
        )
        return manager.inspect()
