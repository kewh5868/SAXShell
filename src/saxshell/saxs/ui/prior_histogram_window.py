from __future__ import annotations

from pathlib import Path

from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg,
    NavigationToolbar2QT,
)
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from saxshell.saxs.project_manager import (
    export_prior_histogram_npy,
    export_prior_histogram_table,
    plot_md_prior_histogram,
)
from saxshell.saxs.ui.prior_export_dialog import PriorExportDialog


class PriorHistogramWindow(QMainWindow):
    def __init__(
        self,
        json_path: str | Path,
        *,
        mode: str,
        secondary_element: str | None = None,
        cmap: str = "summer",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.json_path = Path(json_path).expanduser().resolve()
        self.mode = mode
        self.secondary_element = secondary_element
        self.cmap = cmap
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.setWindowFlag(Qt.WindowType.Window, True)
        self._build_ui()
        self.refresh_plot()

    def _build_ui(self) -> None:
        mode_label = self._mode_label()
        self.setWindowTitle(f"Prior Histogram - {mode_label}")
        self.resize(980, 700)

        central = QWidget()
        root = QVBoxLayout(central)

        header = QHBoxLayout()
        header.addWidget(QLabel(f"Mode: {mode_label}"))
        if self.secondary_element:
            header.addWidget(
                QLabel(f"Secondary atom filter: {self.secondary_element}")
            )
        header.addStretch(1)
        save_button = QPushButton("Save Plot Data As...")
        save_button.clicked.connect(self.save_plot_data_as)
        header.addWidget(save_button)
        root.addLayout(header)

        plot_group = QGroupBox("Prior Histogram")
        plot_layout = QVBoxLayout(plot_group)
        self.figure = Figure(figsize=(9.5, 5.8))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas)
        root.addWidget(plot_group)
        self.setCentralWidget(central)

    def refresh_plot(self) -> None:
        self.figure.clear()
        axis = self.figure.add_subplot(111)
        plot_md_prior_histogram(
            self.json_path,
            mode=self.mode,
            secondary_element=self.secondary_element,
            cmap=self.cmap,
            ax=axis,
        )
        self.canvas.draw()

    def save_plot_data_as(self) -> None:
        suffix = f"_{self.secondary_element}" if self.secondary_element else ""
        dialog = PriorExportDialog(
            default_output_dir=self.json_path.parent,
            default_base_name=f"prior_histogram_{self.mode}{suffix}",
            parent=self,
        )
        if not dialog.exec():
            return
        options = dialog.selected_options
        if options is None:
            return

        saved_paths: list[Path] = []
        for value_mode in options.selected_value_modes():
            base_stem = f"{options.base_name}_{self.mode}_{value_mode}"
            if options.save_csv:
                saved_paths.append(
                    export_prior_histogram_table(
                        self.json_path,
                        options.output_dir / f"{base_stem}.csv",
                        mode=self.mode,
                        value_mode=value_mode,
                        secondary_element=self.secondary_element,
                    )
                )
            if options.save_npy:
                saved_paths.append(
                    export_prior_histogram_npy(
                        self.json_path,
                        options.output_dir / f"{base_stem}.npy",
                        mode=self.mode,
                        value_mode=value_mode,
                        secondary_element=self.secondary_element,
                    )
                )

        QMessageBox.information(
            self,
            "Prior histogram data saved",
            "Saved prior histogram data files:\n"
            + "\n".join(str(path) for path in saved_paths),
        )

    def _mode_label(self) -> str:
        labels = {
            "atom_fraction": "Atom Fraction",
            "structure_fraction": "Structure Fraction",
            "solvent_sort_atom_fraction": "Solvent Sort - Atom Fraction",
            "solvent_sort_structure_fraction": (
                "Solvent Sort - Structure Fraction"
            ),
        }
        return labels.get(self.mode, self.mode.replace("_", " ").title())


__all__ = ["PriorHistogramWindow"]
