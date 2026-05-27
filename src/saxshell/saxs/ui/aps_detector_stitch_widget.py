from __future__ import annotations

from pathlib import Path

from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg,
    NavigationToolbar2QT,
)
from matplotlib.figure import Figure
from PySide6.QtCore import QSize
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from saxshell.saxs.aps_detector_stitch import (
    APS_DETECTOR_ORDER,
    APSDetectorStitchResult,
    find_aps_detector_files,
    save_aps_stitched_data,
    stitch_aps_detector_files,
)


class APSDetectorStitchToolWindow(QWidget):
    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        initial_input_path: str | Path | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("APS Detector Stitch")
        self._selected_file_paths: list[Path] | None = None
        self._last_result: APSDetectorStitchResult | None = None
        self._build_ui()
        if initial_input_path is not None:
            self.input_path_edit.setText(str(initial_input_path))
        self.resize(self._default_window_size())

    def _build_ui(self) -> None:
        root_layout = QVBoxLayout(self)
        root_layout.setSpacing(8)

        input_group = QGroupBox("Input")
        input_layout = QVBoxLayout(input_group)
        folder_row = QHBoxLayout()
        self.input_path_edit = QLineEdit()
        folder_row.addWidget(self.input_path_edit, stretch=1)
        self.browse_folder_button = QPushButton("Folder...")
        self.browse_folder_button.clicked.connect(self._choose_folder)
        folder_row.addWidget(self.browse_folder_button)
        self.browse_files_button = QPushButton("Files...")
        self.browse_files_button.clicked.connect(self._choose_files)
        folder_row.addWidget(self.browse_files_button)
        input_layout.addLayout(folder_row)
        self.selected_files_label = QLabel("No individual files selected.")
        self.selected_files_label.setWordWrap(True)
        input_layout.addWidget(self.selected_files_label)
        root_layout.addWidget(input_group)

        output_group = QGroupBox("Output")
        output_layout = QFormLayout(output_group)
        output_row = QHBoxLayout()
        self.output_path_edit = QLineEdit()
        output_row.addWidget(self.output_path_edit, stretch=1)
        self.browse_output_button = QPushButton("Save As...")
        self.browse_output_button.clicked.connect(self._choose_output_path)
        output_row.addWidget(self.browse_output_button)
        output_layout.addRow("Stitched data", output_row)
        root_layout.addWidget(output_group)

        button_row = QHBoxLayout()
        self.run_button = QPushButton("Run Stitch")
        self.run_button.clicked.connect(self._run_stitch)
        button_row.addWidget(self.run_button)
        self.save_button = QPushButton("Save Stitched Data")
        self.save_button.clicked.connect(self._save_result)
        self.save_button.setEnabled(False)
        button_row.addWidget(self.save_button)
        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self._clear)
        button_row.addWidget(self.clear_button)
        button_row.addStretch(1)
        root_layout.addLayout(button_row)

        self.figure = Figure(figsize=(8.2, 5.2), tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        root_layout.addWidget(self.toolbar)
        root_layout.addWidget(self.canvas, stretch=1)

        self.output_box = QTextEdit()
        self.output_box.setReadOnly(True)
        self.output_box.setMinimumHeight(120)
        root_layout.addWidget(self.output_box)
        self._draw_empty_plot()

    @staticmethod
    def _default_window_size() -> QSize:
        app = QApplication.instance()
        screen = app.primaryScreen() if app is not None else None
        if screen is None:
            return QSize(1080, 760)
        available = screen.availableGeometry()
        width = min(1160, max(900, available.width() - 180))
        height = min(820, max(640, available.height() - 160))
        width = min(width, max(760, available.width() - 48))
        height = min(height, max(560, available.height() - 72))
        return QSize(width, height)

    def _choose_folder(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "Select APS detector folder",
            self.input_path_edit.text().strip(),
        )
        if not selected:
            return
        self._selected_file_paths = None
        self.input_path_edit.setText(selected)
        self.selected_files_label.setText("No individual files selected.")
        self._preview_folder_matches(Path(selected))

    def _choose_files(self) -> None:
        paths, _selected_filter = QFileDialog.getOpenFileNames(
            self,
            "Select APS detector files",
            self.input_path_edit.text().strip(),
            "Text data (*.txt *.dat *.csv);;All files (*)",
        )
        if not paths:
            return
        self._selected_file_paths = [Path(path) for path in paths]
        parent = self._common_parent(self._selected_file_paths)
        self.input_path_edit.setText(str(parent))
        self.selected_files_label.setText(
            "\n".join(str(path) for path in self._selected_file_paths)
        )
        self._set_default_output_path(parent)

    def _choose_output_path(self) -> None:
        start = self.output_path_edit.text().strip()
        if not start:
            start = str(self._default_output_path())
        selected, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save stitched APS detector data",
            start,
            "Text data (*.txt);;All files (*)",
        )
        if selected:
            self.output_path_edit.setText(selected)

    def _preview_folder_matches(self, folder: Path) -> None:
        try:
            matches = find_aps_detector_files(folder)
        except Exception as exc:
            self.output_box.setPlainText(str(exc))
            return
        self.selected_files_label.setText(
            "\n".join(
                f"{detector}: {matches[detector]}"
                for detector in APS_DETECTOR_ORDER
            )
        )
        self._set_default_output_path(folder)

    def _run_stitch(self) -> None:
        try:
            source = self._input_source()
            result = stitch_aps_detector_files(source)
        except Exception as exc:
            self._last_result = None
            self.save_button.setEnabled(False)
            self.output_box.setPlainText(
                f"Unable to stitch APS detector data:\n{exc}"
            )
            self._draw_empty_plot()
            return
        self._last_result = result
        if not self.output_path_edit.text().strip():
            self.output_path_edit.setText(str(self._default_output_path()))
        self.output_box.setPlainText(result.summary_text())
        self.save_button.setEnabled(True)
        self._plot_result(result)

    def _save_result(self) -> None:
        if self._last_result is None:
            self.output_box.setPlainText("Run the stitch before saving.")
            return
        destination_text = self.output_path_edit.text().strip()
        if not destination_text:
            destination_text = str(self._default_output_path())
            self.output_path_edit.setText(destination_text)
        try:
            destination = save_aps_stitched_data(
                self._last_result,
                destination_text,
            )
        except Exception as exc:
            QMessageBox.warning(
                self,
                "APS Detector Stitch",
                f"Could not save stitched data:\n{exc}",
            )
            return
        self.output_box.setPlainText(
            self._last_result.summary_text() + f"\nSaved: {destination}"
        )

    def _clear(self) -> None:
        self._selected_file_paths = None
        self._last_result = None
        self.input_path_edit.clear()
        self.output_path_edit.clear()
        self.selected_files_label.setText("No individual files selected.")
        self.output_box.clear()
        self.save_button.setEnabled(False)
        self._draw_empty_plot()

    def _input_source(self) -> list[Path] | Path:
        if self._selected_file_paths is not None:
            return self._selected_file_paths
        text = self.input_path_edit.text().strip()
        if not text:
            raise ValueError(
                "Select a folder or exactly three detector files."
            )
        return Path(text).expanduser()

    def _default_output_path(self) -> Path:
        if self._selected_file_paths:
            parent = self._common_parent(self._selected_file_paths)
        else:
            text = self.input_path_edit.text().strip()
            parent = Path(text).expanduser() if text else Path.cwd()
            if parent.is_file():
                parent = parent.parent
        if not parent.exists() or not parent.is_dir():
            parent = Path.cwd()
        return parent / "aps_detector_stitched.txt"

    def _set_default_output_path(self, parent: Path) -> None:
        if self.output_path_edit.text().strip():
            return
        if parent.exists() and parent.is_dir():
            self.output_path_edit.setText(
                str(parent / "aps_detector_stitched.txt")
            )

    @staticmethod
    def _common_parent(paths: list[Path]) -> Path:
        if not paths:
            return Path.cwd()
        parents = {path.expanduser().parent for path in paths}
        if len(parents) == 1:
            return next(iter(parents))
        return paths[0].expanduser().parent

    def _draw_empty_plot(self) -> None:
        self.figure.clear()
        axis = self.figure.add_subplot(111)
        axis.set_xlabel("q (1/A)")
        axis.set_ylabel("I(q)")
        axis.grid(True, which="both", alpha=0.25)
        self.canvas.draw_idle()

    def _plot_result(self, result: APSDetectorStitchResult) -> None:
        self.figure.clear()
        axis = self.figure.add_subplot(111)
        colors = {
            "hs104": "tab:blue",
            "hs103": "tab:orange",
            "hs102": "tab:green",
        }
        for detector in APS_DETECTOR_ORDER:
            data = result.scaled_traces[detector]
            positive = (data[:, 0] > 0.0) & (data[:, 1] > 0.0)
            axis.plot(
                data[positive, 0],
                data[positive, 1],
                marker=".",
                linestyle="None",
                markersize=3,
                alpha=0.42,
                color=colors.get(detector),
                label=f"{detector} scaled",
            )
        stitched = result.stitched_data
        positive = (stitched[:, 0] > 0.0) & (stitched[:, 1] > 0.0)
        axis.plot(
            stitched[positive, 0],
            stitched[positive, 1],
            color="black",
            linewidth=1.35,
            label="stitched",
        )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("q (1/A)")
        axis.set_ylabel("I(q)")
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(loc="best")
        self.canvas.draw_idle()


def launch_aps_detector_stitch_ui(
    *,
    initial_input_path: str | Path | None = None,
) -> APSDetectorStitchToolWindow:
    window = APSDetectorStitchToolWindow(
        initial_input_path=initial_input_path,
    )
    window.show()
    window.raise_()
    window.activateWindow()
    return window


__all__ = [
    "APSDetectorStitchToolWindow",
    "launch_aps_detector_stitch_ui",
]
