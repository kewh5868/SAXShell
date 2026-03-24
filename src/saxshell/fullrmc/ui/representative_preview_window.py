from __future__ import annotations

from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg,
    NavigationToolbar2QT,
)
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QLabel,
    QMainWindow,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from saxshell.fullrmc.representatives import RepresentativePreviewCluster


class RepresentativePreviewTab(QWidget):
    def __init__(
        self,
        preview_cluster: RepresentativePreviewCluster,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.preview_cluster = preview_cluster
        self._build_ui()
        self.refresh_plot()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        details = [
            f"Representative: {self.preview_cluster.source_file_name}",
            f"Weight: {self.preview_cluster.selected_weight:.6g}",
            f"Analysis source: {self.preview_cluster.analysis_source}",
        ]
        if self.preview_cluster.score_total is not None:
            details.append(
                f"Selection score: {self.preview_cluster.score_total:.6g}"
            )

        self.header_label = QLabel(" | ".join(details))
        self.header_label.setWordWrap(True)
        root.addWidget(self.header_label)

        self.figure = Figure(figsize=(9.5, 7.0))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        root.addWidget(self.toolbar)
        root.addWidget(self.canvas, stretch=1)

    def refresh_plot(self) -> None:
        self.figure.clear()
        series = self.preview_cluster.all_series()
        if not series:
            axis = self.figure.add_subplot(111)
            axis.text(
                0.5,
                0.5,
                "No bond-pair or angle distributions are available for this representative selection.",
                ha="center",
                va="center",
                wrap=True,
                transform=axis.transAxes,
            )
            axis.set_axis_off()
            self.canvas.draw_idle()
            return

        axes = self.figure.subplots(len(series), 1, squeeze=False).flatten()
        for axis, preview_series in zip(axes, series, strict=True):
            if preview_series.distribution_values.size > 0:
                axis.hist(
                    preview_series.distribution_values,
                    bins=60,
                    color=(
                        "#355070"
                        if preview_series.category == "bond"
                        else "#bc6c25"
                    ),
                    edgecolor="white",
                    alpha=0.88,
                )
            else:
                axis.text(
                    0.5,
                    0.5,
                    "No analyzed values were available for this distribution.",
                    ha="center",
                    va="center",
                    wrap=True,
                    transform=axis.transAxes,
                )
            for index, value in enumerate(
                preview_series.representative_values
            ):
                axis.axvline(
                    value,
                    color="black",
                    linestyle="--",
                    linewidth=1.3,
                    label=("Representative value" if index == 0 else None),
                )
            axis.set_title(preview_series.display_label)
            axis.set_xlabel(preview_series.xlabel)
            axis.set_ylabel("Count")
            if preview_series.representative_values:
                axis.legend(frameon=False, loc="upper right")
        self.figure.tight_layout()
        self.canvas.draw_idle()


class RepresentativePreviewWindow(QMainWindow):
    def __init__(
        self,
        preview_clusters: list[RepresentativePreviewCluster],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.preview_clusters = list(preview_clusters)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.setWindowFlag(Qt.WindowType.Window, True)
        self.setWindowTitle("Representative Cluster Preview")
        self.resize(1100, 820)
        self._build_ui()

    def _build_ui(self) -> None:
        central = QWidget(self)
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        intro_label = QLabel(
            "Tab through the selected cluster bins to compare the analyzed bond-pair "
            "and angle distributions with the vertical dashed lines from the chosen "
            "representative structure."
        )
        intro_label.setWordWrap(True)
        root.addWidget(intro_label)

        self.tab_widget = QTabWidget()
        for preview_cluster in self.preview_clusters:
            tab = RepresentativePreviewTab(preview_cluster, parent=self)
            self.tab_widget.addTab(tab, preview_cluster.tab_label)
            self.tab_widget.setTabToolTip(
                self.tab_widget.count() - 1,
                preview_cluster.title,
            )
        root.addWidget(self.tab_widget, stretch=1)
        self.setCentralWidget(central)


__all__ = [
    "RepresentativePreviewTab",
    "RepresentativePreviewWindow",
]
