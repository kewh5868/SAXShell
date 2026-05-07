from __future__ import annotations

import pickle
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import (
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFontComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from saxshell.plotting.stacked_histogram import (
    STACKED_HISTOGRAM_LEGEND_LOCATIONS,
    StackedHistogramPlotDefaults,
    StackedHistogramPlotSettings,
)
from saxshell.saxs.stoichiometry import (
    format_stoich_for_axis,
    sort_stoich_labels,
)

PICKLED_PLOT_FILE_FILTER = "Pickled Plot Files (*.pkl);;All Files (*)"


def _default_pickled_plot_name(window_title: str) -> str:
    stem = re.sub(r"[^0-9a-zA-Z]+", "_", window_title.strip().lower()).strip(
        "_"
    )
    if not stem:
        stem = "plot"
    return f"{stem}.pkl"


def save_pickled_plot_figure(
    figure: Figure,
    destination: str | Path,
    *,
    window_title: str = "",
    extra_payload: Mapping[str, object] | None = None,
) -> Path:
    path = Path(destination)
    payload = {
        "kind": "saxshell_plot_figure",
        "version": 1,
        "window_title": window_title,
        "figure": figure,
    }
    if extra_payload is not None:
        payload.update(dict(extra_payload))
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_pickled_plot_payload(source: str | Path) -> dict[str, object]:
    path = Path(source)
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if isinstance(payload, Figure):
        return {
            "kind": "legacy_matplotlib_figure",
            "figure": payload,
        }
    if isinstance(payload, dict):
        figure = payload.get("figure")
        if isinstance(figure, Figure):
            return dict(payload)
        legacy_figure = payload.get("fig")
        if isinstance(legacy_figure, Figure):
            updated_payload = dict(payload)
            updated_payload["figure"] = legacy_figure
            return updated_payload
    raise ValueError(f"{path} does not contain a pickled Matplotlib figure")


def load_pickled_plot_figure(source: str | Path) -> Figure:
    payload = load_pickled_plot_payload(source)
    figure = payload.get("figure")
    if isinstance(figure, Figure):
        return figure
    raise ValueError(f"{source} does not contain a pickled Matplotlib figure")


@dataclass(slots=True)
class HeatmapPlotDefaults:
    title: str
    x_label: str
    y_label: str
    colorbar_label: str
    title_position_x: float = 0.5
    title_position_y: float = 1.0
    default_x_axis_unit_name: str = ""
    available_x_axis_unit_names: tuple[str, ...] = ()
    default_colormap_name: str = ""
    available_colormap_names: tuple[str, ...] = ()
    auto_color_limit_min: float = 0.0
    auto_color_limit_max: float = 1.0
    raw_cluster_labels: tuple[str, ...] = ()
    default_label_entries: tuple[tuple[str, str], ...] = ()


@dataclass(slots=True)
class HeatmapPlotSettings:
    title: str | None = None
    x_label: str | None = None
    y_label: str | None = None
    colorbar_label: str | None = None
    title_position_x: float | None = None
    title_position_y: float | None = None
    color_limit_min: float | None = None
    color_limit_max: float | None = None
    font_family: str = ""
    title_font_size: float = 12.0
    axis_label_font_size: float = 11.0
    tick_label_font_size: float = 9.0
    cluster_label_font_size: float = 9.0
    aspect_mode: str = "auto"
    custom_aspect: float = 1.0
    max_x_ticks: int = 8
    max_y_ticks: int = 24
    x_tick_rotation: int = 0
    y_tick_rotation: int = 0
    show_minor_x_ticks: bool = False
    show_minor_y_ticks: bool = False
    label_order: list[str] = field(default_factory=list)
    label_map: dict[str, str] = field(default_factory=dict)

    def resolve_title(self, defaults: HeatmapPlotDefaults) -> str:
        return defaults.title if self.title is None else self.title

    def resolve_x_label(self, defaults: HeatmapPlotDefaults) -> str:
        return defaults.x_label if self.x_label is None else self.x_label

    def resolve_y_label(self, defaults: HeatmapPlotDefaults) -> str:
        return defaults.y_label if self.y_label is None else self.y_label

    def resolve_colorbar_label(self, defaults: HeatmapPlotDefaults) -> str:
        return (
            defaults.colorbar_label
            if self.colorbar_label is None
            else self.colorbar_label
        )

    def resolve_title_position_x(self, defaults: HeatmapPlotDefaults) -> float:
        return (
            defaults.title_position_x
            if self.title_position_x is None
            else self.title_position_x
        )

    def resolve_title_position_y(self, defaults: HeatmapPlotDefaults) -> float:
        return (
            defaults.title_position_y
            if self.title_position_y is None
            else self.title_position_y
        )

    def has_manual_color_limits(self) -> bool:
        return (
            self.color_limit_min is not None
            or self.color_limit_max is not None
        )

    def resolve_color_limit_min(self, defaults: HeatmapPlotDefaults) -> float:
        return (
            defaults.auto_color_limit_min
            if self.color_limit_min is None
            else self.color_limit_min
        )

    def resolve_color_limit_max(self, defaults: HeatmapPlotDefaults) -> float:
        return (
            defaults.auto_color_limit_max
            if self.color_limit_max is None
            else self.color_limit_max
        )

    def sync_labels(
        self,
        raw_labels: Sequence[str],
        *,
        default_label_entries: Sequence[tuple[str, str]] | None = None,
    ) -> None:
        default_entries = (
            [
                (str(raw_label), format_stoich_for_axis(str(raw_label)))
                for raw_label in raw_labels
            ]
            if default_label_entries is None
            else [
                (str(raw_label), str(display_label))
                for raw_label, display_label in default_label_entries
            ]
        )
        default_map = {
            raw_label: display_label
            for raw_label, display_label in default_entries
        }
        existing = dict(self.label_map)
        preserved_order = [
            raw_label
            for raw_label in self.label_order
            if raw_label in default_map
        ]
        remaining = [
            raw_label
            for raw_label, _display_label in default_entries
            if raw_label not in preserved_order
        ]
        self.label_order = preserved_order + remaining
        self.label_map = {
            raw_label: existing.get(raw_label, default_map[raw_label])
            for raw_label in self.label_order
        }

    def display_label(self, raw_label: str) -> str:
        return self.label_map.get(raw_label, raw_label)

    def ordered_raw_labels(
        self,
        defaults: HeatmapPlotDefaults,
    ) -> list[str]:
        if self.label_order:
            available = set(defaults.raw_cluster_labels)
            ordered = [raw for raw in self.label_order if raw in available]
            remaining = [
                raw
                for raw in defaults.raw_cluster_labels
                if raw not in ordered
            ]
            return ordered + remaining
        return list(defaults.raw_cluster_labels)

    def to_dict(self) -> dict[str, object]:
        return {
            "title": self.title,
            "x_label": self.x_label,
            "y_label": self.y_label,
            "colorbar_label": self.colorbar_label,
            "title_position_x": self.title_position_x,
            "title_position_y": self.title_position_y,
            "color_limit_min": self.color_limit_min,
            "color_limit_max": self.color_limit_max,
            "font_family": self.font_family,
            "title_font_size": self.title_font_size,
            "axis_label_font_size": self.axis_label_font_size,
            "tick_label_font_size": self.tick_label_font_size,
            "cluster_label_font_size": self.cluster_label_font_size,
            "aspect_mode": self.aspect_mode,
            "custom_aspect": self.custom_aspect,
            "max_x_ticks": self.max_x_ticks,
            "max_y_ticks": self.max_y_ticks,
            "x_tick_rotation": self.x_tick_rotation,
            "y_tick_rotation": self.y_tick_rotation,
            "show_minor_x_ticks": self.show_minor_x_ticks,
            "show_minor_y_ticks": self.show_minor_y_ticks,
            "label_order": list(self.label_order),
            "label_map": dict(self.label_map),
        }

    def update_from_dict(self, payload: Mapping[str, object]) -> None:
        for field_name in (
            "title",
            "x_label",
            "y_label",
            "colorbar_label",
            "title_position_x",
            "title_position_y",
            "color_limit_min",
            "color_limit_max",
        ):
            if field_name in payload:
                setattr(self, field_name, payload[field_name])
        if "font_family" in payload:
            self.font_family = str(payload["font_family"] or "")
        for field_name in (
            "title_font_size",
            "axis_label_font_size",
            "tick_label_font_size",
            "cluster_label_font_size",
            "custom_aspect",
        ):
            if field_name in payload:
                setattr(self, field_name, float(payload[field_name]))
        if "aspect_mode" in payload:
            self.aspect_mode = str(payload["aspect_mode"])
        for field_name in (
            "max_x_ticks",
            "max_y_ticks",
            "x_tick_rotation",
            "y_tick_rotation",
        ):
            if field_name in payload:
                setattr(self, field_name, int(payload[field_name]))
        for field_name in ("show_minor_x_ticks", "show_minor_y_ticks"):
            if field_name in payload:
                setattr(self, field_name, bool(payload[field_name]))
        if "label_order" in payload:
            self.label_order = [str(value) for value in payload["label_order"]]
        if "label_map" in payload:
            label_map = payload["label_map"]
            if isinstance(label_map, Mapping):
                self.label_map = {
                    str(key): str(value) for key, value in label_map.items()
                }


class PlotEditorWindow(QWidget):
    """Reusable popup shell for plot editors with a live Matplotlib
    preview."""

    closed = Signal()

    def __init__(
        self,
        *,
        window_title: str,
        controls_widget: QWidget | None,
        render_preview: Callable[[Figure], None] | None,
        pickle_default_name: str | None = None,
        pickle_state_provider: (
            Callable[[], Mapping[str, object] | None] | None
        ) = None,
        apply_loaded_pickle_state: (
            Callable[[Mapping[str, object]], bool] | None
        ) = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent, Qt.WindowType.Window)
        self._render_preview = render_preview
        self._pickle_state_provider = pickle_state_provider
        self._apply_loaded_pickle_state = apply_loaded_pickle_state
        self._showing_pickled_plot = False
        self._last_pickle_path: Path | None = None
        self._pickle_default_name = (
            _default_pickled_plot_name(window_title)
            if pickle_default_name is None
            else pickle_default_name
        )
        self._preview_toolbar: NavigationToolbar | None = None
        self.canvas: FigureCanvas | None = None
        self.figure = Figure(figsize=(7.8, 6.2))
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.setWindowTitle(window_title)
        self.resize(1260, 760)

        root = QHBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        if controls_widget is not None:
            controls_scroll = QScrollArea()
            controls_scroll.setWidgetResizable(True)
            controls_scroll.setMinimumWidth(410)
            controls_scroll.setWidget(controls_widget)
            root.addWidget(controls_scroll, stretch=0)

        preview_panel = QWidget()
        preview_layout = QVBoxLayout(preview_panel)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.setSpacing(6)
        preview_layout.addWidget(QLabel("Preview"))
        preview_button_row = QHBoxLayout()
        preview_button_row.setContentsMargins(0, 0, 0, 0)
        preview_button_row.setSpacing(6)
        self.save_pickle_button = QPushButton("Save Pickled Plot")
        self.save_pickle_button.clicked.connect(self.save_pickled_plot_as)
        preview_button_row.addWidget(self.save_pickle_button)
        self.load_pickle_button = QPushButton("Load Pickled Plot")
        self.load_pickle_button.clicked.connect(self.load_pickled_plot_as)
        preview_button_row.addWidget(self.load_pickle_button)
        self.show_live_preview_button = QPushButton("Show Live Plot")
        self.show_live_preview_button.clicked.connect(self.show_live_preview)
        self.show_live_preview_button.setEnabled(False)
        preview_button_row.addWidget(self.show_live_preview_button)
        preview_button_row.addStretch(1)
        preview_layout.addLayout(preview_button_row)
        self._preview_canvas_layout = QVBoxLayout()
        self._preview_canvas_layout.setContentsMargins(0, 0, 0, 0)
        self._preview_canvas_layout.setSpacing(0)
        preview_layout.addLayout(self._preview_canvas_layout, stretch=1)
        self._set_preview_figure(self.figure)
        root.addWidget(preview_panel, stretch=1)

    def _set_preview_figure(self, figure: Figure) -> None:
        if self._preview_toolbar is not None:
            self._preview_canvas_layout.removeWidget(self._preview_toolbar)
            self._preview_toolbar.setParent(None)
            self._preview_toolbar.deleteLater()
            self._preview_toolbar = None
        if self.canvas is not None:
            self._preview_canvas_layout.removeWidget(self.canvas)
            self.canvas.setParent(None)
            self.canvas.deleteLater()
            self.canvas = None
        self.figure = figure
        self.canvas = FigureCanvas(self.figure)
        self._preview_toolbar = NavigationToolbar(self.canvas, self)
        self._preview_canvas_layout.addWidget(self._preview_toolbar)
        self._preview_canvas_layout.addWidget(self.canvas, stretch=1)

    def is_showing_pickled_plot(self) -> bool:
        return self._showing_pickled_plot

    def refresh_preview(self, *, force: bool = False) -> None:
        if self._showing_pickled_plot and not force:
            return
        if force:
            self._showing_pickled_plot = False
            self.show_live_preview_button.setEnabled(False)
        if self._render_preview is None:
            if self.canvas is not None:
                self.canvas.draw_idle()
            return
        self._render_preview(self.figure)
        if self.canvas is not None:
            self.canvas.draw_idle()

    def _default_pickle_path(self) -> Path:
        if self._last_pickle_path is not None:
            return self._last_pickle_path
        return Path.cwd() / self._pickle_default_name

    def save_pickled_plot(self, destination: str | Path) -> Path:
        extra_payload = (
            None
            if self._pickle_state_provider is None
            else self._pickle_state_provider()
        )
        saved_path = save_pickled_plot_figure(
            self.figure,
            destination,
            window_title=self.windowTitle(),
            extra_payload=extra_payload,
        )
        self._last_pickle_path = saved_path
        return saved_path

    def save_pickled_plot_as(self) -> Path | None:
        selected_path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save Pickled Plot",
            str(self._default_pickle_path()),
            PICKLED_PLOT_FILE_FILTER,
        )
        if not selected_path:
            return None
        destination = Path(selected_path)
        if destination.suffix == "":
            destination = destination.with_suffix(".pkl")
        try:
            return self.save_pickled_plot(destination)
        except Exception as exc:  # pragma: no cover - defensive UI guard
            QMessageBox.warning(
                self,
                "Save Pickled Plot",
                f"Could not save the pickled plot:\n{exc}",
            )
            return None

    def load_pickled_plot(self, source: str | Path) -> Figure:
        payload = load_pickled_plot_payload(source)
        self._last_pickle_path = Path(source)
        if (
            self._apply_loaded_pickle_state is not None
            and self._apply_loaded_pickle_state(payload)
        ):
            self.refresh_preview(force=True)
            return self.figure

        figure = payload.get("figure")
        if not isinstance(figure, Figure):
            raise ValueError(
                f"{source} does not contain a pickled Matplotlib figure"
            )
        self._set_preview_figure(figure)
        self._showing_pickled_plot = True
        self.show_live_preview_button.setEnabled(
            self._render_preview is not None
        )
        if self.canvas is not None:
            self.canvas.draw_idle()
        return figure

    def load_pickled_plot_as(self) -> Path | None:
        selected_path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "Load Pickled Plot",
            str(self._default_pickle_path()),
            PICKLED_PLOT_FILE_FILTER,
        )
        if not selected_path:
            return None
        source = Path(selected_path)
        try:
            self.load_pickled_plot(source)
        except Exception as exc:  # pragma: no cover - defensive UI guard
            QMessageBox.warning(
                self,
                "Load Pickled Plot",
                f"Could not load the pickled plot:\n{exc}",
            )
            return None
        return source

    def show_live_preview(self) -> None:
        self.refresh_preview(force=True)

    def closeEvent(self, event) -> None:  # noqa: N802
        self.closed.emit()
        super().closeEvent(event)


class HeatmapPlotEditorControls(QWidget):
    """Editable controls for reusable heatmap/colormap plot settings."""

    settings_changed = Signal()
    x_axis_unit_changed = Signal(str)
    colormap_changed = Signal(str)

    def __init__(
        self,
        *,
        settings: HeatmapPlotSettings,
        defaults: HeatmapPlotDefaults,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._settings = settings
        self._defaults = defaults
        self._last_synced_defaults: HeatmapPlotDefaults | None = None
        self._syncing = False
        self._build_ui()
        self.sync_defaults(defaults)

    def needs_default_sync(self, defaults: HeatmapPlotDefaults) -> bool:
        return self._last_synced_defaults != defaults

    def sync_defaults(self, defaults: HeatmapPlotDefaults) -> None:
        self._defaults = defaults
        self._settings.sync_labels(
            defaults.raw_cluster_labels,
            default_label_entries=defaults.default_label_entries,
        )
        self._syncing = True
        try:
            self.title_edit.setText(self._settings.resolve_title(defaults))
            self.x_label_edit.setText(self._settings.resolve_x_label(defaults))
            self.y_label_edit.setText(self._settings.resolve_y_label(defaults))
            self.colorbar_label_edit.setText(
                self._settings.resolve_colorbar_label(defaults)
            )
            self.title_position_x_spin.setValue(
                self._settings.resolve_title_position_x(defaults)
            )
            self.title_position_y_spin.setValue(
                self._settings.resolve_title_position_y(defaults)
            )
            self.x_axis_unit_combo.clear()
            for unit_name in defaults.available_x_axis_unit_names:
                self.x_axis_unit_combo.addItem(unit_name, unit_name)
            if defaults.available_x_axis_unit_names:
                self.x_axis_unit_combo.setCurrentIndex(
                    max(
                        0,
                        self.x_axis_unit_combo.findData(
                            defaults.default_x_axis_unit_name
                        ),
                    )
                )
            self.x_axis_unit_combo.setEnabled(
                bool(defaults.available_x_axis_unit_names)
            )
            self.colormap_combo.clear()
            for colormap_name in defaults.available_colormap_names:
                self.colormap_combo.addItem(colormap_name, colormap_name)
            if defaults.available_colormap_names:
                self.colormap_combo.setCurrentIndex(
                    max(
                        0,
                        self.colormap_combo.findData(
                            defaults.default_colormap_name
                        ),
                    )
                )
            self.colormap_combo.setEnabled(
                bool(defaults.available_colormap_names)
            )
            self.color_limit_min_spin.setValue(
                self._settings.resolve_color_limit_min(defaults)
            )
            self.color_limit_max_spin.setValue(
                self._settings.resolve_color_limit_max(defaults)
            )
            self._update_color_limit_reset_state()
            if self._settings.font_family:
                self.font_combo.setCurrentFont(
                    QFont(self._settings.font_family)
                )
            self.title_font_spin.setValue(self._settings.title_font_size)
            self.axis_label_font_spin.setValue(
                self._settings.axis_label_font_size
            )
            self.tick_label_font_spin.setValue(
                self._settings.tick_label_font_size
            )
            self.cluster_label_font_spin.setValue(
                self._settings.cluster_label_font_size
            )
            self.aspect_combo.setCurrentIndex(
                max(
                    0,
                    self.aspect_combo.findData(self._settings.aspect_mode),
                )
            )
            self.aspect_value_spin.setValue(self._settings.custom_aspect)
            self.max_x_ticks_spin.setValue(self._settings.max_x_ticks)
            self.max_y_ticks_spin.setValue(self._settings.max_y_ticks)
            self.x_tick_rotation_spin.setValue(self._settings.x_tick_rotation)
            self.y_tick_rotation_spin.setValue(self._settings.y_tick_rotation)
            self.minor_x_ticks_checkbox.setChecked(
                self._settings.show_minor_x_ticks
            )
            self.minor_y_ticks_checkbox.setChecked(
                self._settings.show_minor_y_ticks
            )
            self._sync_label_table()
            self._update_aspect_state()
        finally:
            self._last_synced_defaults = defaults
            self._syncing = False

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(10)

        note = QLabel(
            "Edit the heatmap title, labels, font, aspect ratio, and tick "
            "density. Use $_{n}$ for subscript and $^{n}$ for superscript "
            "(matplotlib mathtext). Igor-style inline text is also "
            "supported: \\f01 bold, \\f02 italics, \\f00 reset, and "
            "\\Z<NN> inline font size."
        )
        note.setWordWrap(True)
        root.addWidget(note)

        text_group = QGroupBox("Text")
        text_form = QFormLayout(text_group)
        self.title_edit = QLineEdit()
        self.title_edit.textChanged.connect(self._on_title_changed)
        text_form.addRow("Title", self.title_edit)
        self.x_axis_unit_combo = QComboBox()
        self.x_axis_unit_combo.currentIndexChanged.connect(
            self._on_x_axis_unit_changed
        )
        text_form.addRow("X Unit", self.x_axis_unit_combo)
        self.x_label_edit = QLineEdit()
        self.x_label_edit.textChanged.connect(self._on_x_label_changed)
        text_form.addRow("X Label", self.x_label_edit)
        self.y_label_edit = QLineEdit()
        self.y_label_edit.textChanged.connect(self._on_y_label_changed)
        text_form.addRow("Y Label", self.y_label_edit)
        self.colorbar_label_edit = QLineEdit()
        self.colorbar_label_edit.textChanged.connect(
            self._on_colorbar_label_changed
        )
        text_form.addRow("Colorbar", self.colorbar_label_edit)
        self.colormap_combo = QComboBox()
        self.colormap_combo.currentIndexChanged.connect(
            self._on_colormap_changed
        )
        text_form.addRow("Colormap", self.colormap_combo)
        self.reset_text_button = QPushButton("Reset Text Defaults")
        self.reset_text_button.clicked.connect(self._reset_text_defaults)
        text_form.addRow(self.reset_text_button)
        root.addWidget(text_group)

        color_group = QGroupBox("Color Scale")
        color_form = QFormLayout(color_group)
        self.color_limit_min_spin = self._build_color_limit_spin()
        self.color_limit_min_spin.valueChanged.connect(
            self._on_color_limit_min_changed
        )
        color_form.addRow("Min", self.color_limit_min_spin)
        self.color_limit_max_spin = self._build_color_limit_spin()
        self.color_limit_max_spin.valueChanged.connect(
            self._on_color_limit_max_changed
        )
        color_form.addRow("Max", self.color_limit_max_spin)
        self.reset_color_limits_button = QPushButton("Reset to Auto Limits")
        self.reset_color_limits_button.clicked.connect(
            self._reset_color_limits
        )
        color_form.addRow(self.reset_color_limits_button)
        root.addWidget(color_group)

        style_group = QGroupBox("Style")
        style_form = QFormLayout(style_group)
        self.font_combo = QFontComboBox()
        self.font_combo.currentFontChanged.connect(self._on_font_changed)
        style_form.addRow("Font", self.font_combo)
        self.title_font_spin = self._build_font_spin()
        self.title_font_spin.valueChanged.connect(
            self._on_title_font_size_changed
        )
        style_form.addRow("Title Size", self.title_font_spin)
        self.title_position_x_spin = self._build_position_spin()
        self.title_position_x_spin.valueChanged.connect(
            self._on_title_position_x_changed
        )
        style_form.addRow("Title X", self.title_position_x_spin)
        self.title_position_y_spin = self._build_position_spin()
        self.title_position_y_spin.valueChanged.connect(
            self._on_title_position_y_changed
        )
        style_form.addRow("Title Y", self.title_position_y_spin)
        self.axis_label_font_spin = self._build_font_spin()
        self.axis_label_font_spin.valueChanged.connect(
            self._on_axis_label_font_size_changed
        )
        style_form.addRow("Axis Label Size", self.axis_label_font_spin)
        self.tick_label_font_spin = self._build_font_spin()
        self.tick_label_font_spin.valueChanged.connect(
            self._on_tick_label_font_size_changed
        )
        style_form.addRow("Tick Label Size", self.tick_label_font_spin)
        self.cluster_label_font_spin = self._build_font_spin()
        self.cluster_label_font_spin.valueChanged.connect(
            self._on_cluster_label_font_size_changed
        )
        style_form.addRow("Cluster Label Size", self.cluster_label_font_spin)

        self.aspect_combo = QComboBox()
        self.aspect_combo.addItem("Auto", "auto")
        self.aspect_combo.addItem("Equal", "equal")
        self.aspect_combo.addItem("Custom Ratio", "custom")
        self.aspect_combo.currentIndexChanged.connect(
            self._on_aspect_mode_changed
        )
        style_form.addRow("Aspect", self.aspect_combo)
        self.aspect_value_spin = QDoubleSpinBox()
        self.aspect_value_spin.setDecimals(2)
        self.aspect_value_spin.setRange(0.1, 10.0)
        self.aspect_value_spin.setSingleStep(0.1)
        self.aspect_value_spin.valueChanged.connect(
            self._on_aspect_value_changed
        )
        style_form.addRow("Aspect Ratio", self.aspect_value_spin)
        root.addWidget(style_group)

        ticks_group = QGroupBox("Ticks")
        ticks_form = QFormLayout(ticks_group)
        self.max_x_ticks_spin = QSpinBox()
        self.max_x_ticks_spin.setRange(2, 20)
        self.max_x_ticks_spin.valueChanged.connect(
            self._on_max_x_ticks_changed
        )
        ticks_form.addRow("Max X Ticks", self.max_x_ticks_spin)
        self.max_y_ticks_spin = QSpinBox()
        self.max_y_ticks_spin.setRange(1, 60)
        self.max_y_ticks_spin.valueChanged.connect(
            self._on_max_y_ticks_changed
        )
        ticks_form.addRow("Max Y Ticks", self.max_y_ticks_spin)
        self.x_tick_rotation_spin = QSpinBox()
        self.x_tick_rotation_spin.setRange(-180, 180)
        self.x_tick_rotation_spin.valueChanged.connect(
            self._on_x_tick_rotation_changed
        )
        ticks_form.addRow("X Tick Rotation", self.x_tick_rotation_spin)
        self.y_tick_rotation_spin = QSpinBox()
        self.y_tick_rotation_spin.setRange(-180, 180)
        self.y_tick_rotation_spin.valueChanged.connect(
            self._on_y_tick_rotation_changed
        )
        ticks_form.addRow("Y Tick Rotation", self.y_tick_rotation_spin)
        self.minor_x_ticks_checkbox = QCheckBox("Show X Minor Ticks")
        self.minor_x_ticks_checkbox.toggled.connect(
            self._on_minor_x_ticks_changed
        )
        ticks_form.addRow(self.minor_x_ticks_checkbox)
        self.minor_y_ticks_checkbox = QCheckBox("Show Y Minor Ticks")
        self.minor_y_ticks_checkbox.toggled.connect(
            self._on_minor_y_ticks_changed
        )
        ticks_form.addRow(self.minor_y_ticks_checkbox)
        root.addWidget(ticks_group)

        labels_group = QGroupBox("Axis Labels")
        labels_layout = QVBoxLayout(labels_group)
        labels_layout.setContentsMargins(8, 8, 8, 8)
        labels_layout.setSpacing(8)
        labels_note = QLabel(
            "Rearrange rows to control axis-bin order and edit Display Label "
            "to customise tick text. The raw label column stays fixed so you "
            "can round-trip custom formatting."
        )
        labels_note.setWordWrap(True)
        labels_layout.addWidget(labels_note)
        self.label_table = QTableWidget(0, 2)
        self.label_table.setHorizontalHeaderLabels(
            ["Raw Label", "Display Label"]
        )
        self.label_table.horizontalHeader().setStretchLastSection(True)
        self.label_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.label_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.label_table.itemChanged.connect(self._on_label_item_changed)
        content_row = QHBoxLayout()
        content_row.addWidget(self.label_table, stretch=1)
        move_column = QVBoxLayout()
        self.move_label_up_button = QPushButton("Up")
        self.move_label_up_button.clicked.connect(self._move_label_up)
        move_column.addWidget(self.move_label_up_button)
        self.move_label_down_button = QPushButton("Down")
        self.move_label_down_button.clicked.connect(self._move_label_down)
        move_column.addWidget(self.move_label_down_button)
        move_column.addStretch(1)
        content_row.addLayout(move_column)
        labels_layout.addLayout(content_row, stretch=1)
        button_row = QHBoxLayout()
        self.histogram_order_button = QPushButton("Sort Like Histogram")
        self.histogram_order_button.clicked.connect(
            self._apply_histogram_order
        )
        button_row.addWidget(self.histogram_order_button)
        self.auto_subscript_button = QPushButton("Auto Stoich Subscripts")
        self.auto_subscript_button.clicked.connect(
            self._apply_stoich_subscripts
        )
        button_row.addWidget(self.auto_subscript_button)
        self.reset_labels_button = QPushButton("Reset Labels")
        self.reset_labels_button.clicked.connect(self._reset_labels)
        button_row.addWidget(self.reset_labels_button)
        button_row.addStretch(1)
        labels_layout.addLayout(button_row)
        root.addWidget(labels_group, stretch=1)
        root.addStretch(1)

    @staticmethod
    def _build_font_spin() -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setDecimals(1)
        spin.setRange(6.0, 40.0)
        spin.setSingleStep(0.5)
        return spin

    @staticmethod
    def _build_position_spin() -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setDecimals(2)
        spin.setRange(-1.0, 2.0)
        spin.setSingleStep(0.05)
        return spin

    @staticmethod
    def _build_color_limit_spin() -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setDecimals(6)
        spin.setRange(-1_000_000_000.0, 1_000_000_000.0)
        spin.setSingleStep(0.1)
        spin.setAccelerated(True)
        return spin

    def _sync_label_table(self) -> None:
        self.label_table.blockSignals(True)
        try:
            raw_labels = self._settings.ordered_raw_labels(self._defaults)
            self.label_table.setRowCount(len(raw_labels))
            for row, raw_label in enumerate(raw_labels):
                raw_item = QTableWidgetItem(raw_label)
                raw_item.setFlags(
                    raw_item.flags() & ~Qt.ItemFlag.ItemIsEditable
                )
                self.label_table.setItem(row, 0, raw_item)
                self.label_table.setItem(
                    row,
                    1,
                    QTableWidgetItem(self._settings.display_label(raw_label)),
                )
            if raw_labels:
                self.label_table.resizeColumnToContents(0)
                if self.label_table.currentRow() < 0:
                    self.label_table.selectRow(0)
        finally:
            self.label_table.blockSignals(False)

    def _update_aspect_state(self) -> None:
        aspect_mode = self.aspect_combo.currentData()
        self.aspect_value_spin.setEnabled(aspect_mode == "custom")

    def _update_color_limit_reset_state(self) -> None:
        self.reset_color_limits_button.setEnabled(
            self._settings.has_manual_color_limits()
        )

    def _emit_settings_changed(self) -> None:
        if not self._syncing:
            self.settings_changed.emit()

    def _on_title_changed(self, text: str) -> None:
        if self._syncing:
            return
        self._settings.title = text
        self._emit_settings_changed()

    def _on_x_label_changed(self, text: str) -> None:
        if self._syncing:
            return
        self._settings.x_label = text
        self._emit_settings_changed()

    def _on_x_axis_unit_changed(self) -> None:
        if self._syncing:
            return
        current = self.x_axis_unit_combo.currentData()
        if current is None:
            return
        self.x_axis_unit_changed.emit(str(current))

    def _on_y_label_changed(self, text: str) -> None:
        if self._syncing:
            return
        self._settings.y_label = text
        self._emit_settings_changed()

    def _on_colorbar_label_changed(self, text: str) -> None:
        if self._syncing:
            return
        self._settings.colorbar_label = text
        self._emit_settings_changed()

    def _on_colormap_changed(self) -> None:
        if self._syncing:
            return
        current = self.colormap_combo.currentData()
        if current is None:
            return
        self.colormap_changed.emit(str(current))

    @staticmethod
    def _color_limit_delta(anchor: float) -> float:
        return max(abs(float(anchor)) * 0.01, 1.0e-6)

    def _set_manual_color_limits(
        self,
        *,
        minimum: float | None = None,
        maximum: float | None = None,
        changed: str,
    ) -> None:
        current_min = self._settings.resolve_color_limit_min(self._defaults)
        current_max = self._settings.resolve_color_limit_max(self._defaults)
        new_min = current_min if minimum is None else float(minimum)
        new_max = current_max if maximum is None else float(maximum)
        if new_max <= new_min:
            if changed == "min":
                new_max = new_min + self._color_limit_delta(new_min)
            else:
                new_min = new_max - self._color_limit_delta(new_max)

        self._settings.color_limit_min = float(new_min)
        self._settings.color_limit_max = float(new_max)

        self._syncing = True
        try:
            self.color_limit_min_spin.setValue(float(new_min))
            self.color_limit_max_spin.setValue(float(new_max))
            self._update_color_limit_reset_state()
        finally:
            self._syncing = False

        self._emit_settings_changed()

    def _on_color_limit_min_changed(self, value: float) -> None:
        if self._syncing:
            return
        self._set_manual_color_limits(minimum=value, changed="min")

    def _on_color_limit_max_changed(self, value: float) -> None:
        if self._syncing:
            return
        self._set_manual_color_limits(maximum=value, changed="max")

    def _on_font_changed(self, font: QFont) -> None:
        if self._syncing:
            return
        self._settings.font_family = font.family()
        self._emit_settings_changed()

    def _on_title_font_size_changed(self, value: float) -> None:
        if self._syncing:
            return
        self._settings.title_font_size = float(value)
        self._emit_settings_changed()

    def _on_title_position_x_changed(self, value: float) -> None:
        if self._syncing:
            return
        self._settings.title_position_x = float(value)
        self._emit_settings_changed()

    def _on_title_position_y_changed(self, value: float) -> None:
        if self._syncing:
            return
        self._settings.title_position_y = float(value)
        self._emit_settings_changed()

    def _on_axis_label_font_size_changed(self, value: float) -> None:
        if self._syncing:
            return
        self._settings.axis_label_font_size = float(value)
        self._emit_settings_changed()

    def _on_tick_label_font_size_changed(self, value: float) -> None:
        if self._syncing:
            return
        self._settings.tick_label_font_size = float(value)
        self._emit_settings_changed()

    def _on_cluster_label_font_size_changed(self, value: float) -> None:
        if self._syncing:
            return
        self._settings.cluster_label_font_size = float(value)
        self._emit_settings_changed()

    def _on_aspect_mode_changed(self) -> None:
        if self._syncing:
            return
        aspect_mode = self.aspect_combo.currentData()
        self._settings.aspect_mode = (
            "auto" if aspect_mode is None else str(aspect_mode)
        )
        self._update_aspect_state()
        self._emit_settings_changed()

    def _on_aspect_value_changed(self, value: float) -> None:
        if self._syncing:
            return
        self._settings.custom_aspect = float(value)
        self._emit_settings_changed()

    def _on_max_x_ticks_changed(self, value: int) -> None:
        if self._syncing:
            return
        self._settings.max_x_ticks = int(value)
        self._emit_settings_changed()

    def _on_max_y_ticks_changed(self, value: int) -> None:
        if self._syncing:
            return
        self._settings.max_y_ticks = int(value)
        self._emit_settings_changed()

    def _on_x_tick_rotation_changed(self, value: int) -> None:
        if self._syncing:
            return
        self._settings.x_tick_rotation = int(value)
        self._emit_settings_changed()

    def _on_y_tick_rotation_changed(self, value: int) -> None:
        if self._syncing:
            return
        self._settings.y_tick_rotation = int(value)
        self._emit_settings_changed()

    def _on_minor_x_ticks_changed(self, checked: bool) -> None:
        if self._syncing:
            return
        self._settings.show_minor_x_ticks = bool(checked)
        self._emit_settings_changed()

    def _on_minor_y_ticks_changed(self, checked: bool) -> None:
        if self._syncing:
            return
        self._settings.show_minor_y_ticks = bool(checked)
        self._emit_settings_changed()

    def _on_label_item_changed(self, item: QTableWidgetItem) -> None:
        if self._syncing or item.column() != 1:
            return
        raw_item = self.label_table.item(item.row(), 0)
        if raw_item is None:
            return
        self._settings.label_map[raw_item.text()] = item.text()
        self._emit_settings_changed()

    def _move_label_up(self) -> None:
        row = self.label_table.currentRow()
        if row <= 0:
            return
        self._swap_label_rows(row, row - 1)
        self.label_table.selectRow(row - 1)
        self._store_label_order_from_table()
        self._emit_settings_changed()

    def _move_label_down(self) -> None:
        row = self.label_table.currentRow()
        if row < 0 or row >= self.label_table.rowCount() - 1:
            return
        self._swap_label_rows(row, row + 1)
        self.label_table.selectRow(row + 1)
        self._store_label_order_from_table()
        self._emit_settings_changed()

    def _swap_label_rows(self, row_a: int, row_b: int) -> None:
        self.label_table.blockSignals(True)
        try:
            for column in range(self.label_table.columnCount()):
                item_a = self.label_table.takeItem(row_a, column)
                item_b = self.label_table.takeItem(row_b, column)
                if item_a is not None:
                    self.label_table.setItem(row_b, column, item_a)
                if item_b is not None:
                    self.label_table.setItem(row_a, column, item_b)
        finally:
            self.label_table.blockSignals(False)

    def _store_label_order_from_table(self) -> None:
        self._settings.label_order = []
        for row in range(self.label_table.rowCount()):
            raw_item = self.label_table.item(row, 0)
            display_item = self.label_table.item(row, 1)
            if raw_item is None:
                continue
            raw_label = raw_item.text()
            self._settings.label_order.append(raw_label)
            self._settings.label_map[raw_label] = (
                display_item.text() if display_item is not None else raw_label
            )

    def _reset_text_defaults(self) -> None:
        self._settings.title = None
        self._settings.x_label = None
        self._settings.y_label = None
        self._settings.colorbar_label = None
        self._settings.title_position_x = None
        self._settings.title_position_y = None
        self.sync_defaults(self._defaults)
        self._emit_settings_changed()

    def _reset_color_limits(self) -> None:
        self._settings.color_limit_min = None
        self._settings.color_limit_max = None
        self.sync_defaults(self._defaults)
        self._emit_settings_changed()

    def _reset_labels(self) -> None:
        self._settings.label_order = list(self._defaults.raw_cluster_labels)
        default_map = dict(self._defaults.default_label_entries)
        self._settings.label_map = {
            raw_label: default_map.get(raw_label, raw_label)
            for raw_label in self._defaults.raw_cluster_labels
        }
        self.sync_defaults(self._defaults)
        self._emit_settings_changed()

    def _apply_stoich_subscripts(self) -> None:
        self._settings.label_map = {
            raw_label: format_stoich_for_axis(raw_label)
            for raw_label in self._settings.ordered_raw_labels(self._defaults)
        }
        self.sync_defaults(self._defaults)
        self._emit_settings_changed()

    def _apply_histogram_order(self) -> None:
        ordered = sort_stoich_labels(self._defaults.raw_cluster_labels)
        self._settings.label_order = list(ordered)
        self.sync_defaults(self._defaults)
        self._emit_settings_changed()


class StackedHistogramPlotEditorControls(QWidget):
    """Editable controls for reusable stacked-histogram plot
    settings."""

    settings_changed = Signal()
    colormap_changed = Signal(str)
    label_settings_changed = Signal()

    def __init__(
        self,
        *,
        settings: StackedHistogramPlotSettings,
        defaults: StackedHistogramPlotDefaults,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._settings = settings
        self._defaults = defaults
        self._last_synced_defaults: StackedHistogramPlotDefaults | None = None
        self._syncing = False
        self._build_ui()
        self.sync_defaults(defaults)

    def needs_default_sync(
        self, defaults: StackedHistogramPlotDefaults
    ) -> bool:
        return self._last_synced_defaults != defaults

    def sync_defaults(self, defaults: StackedHistogramPlotDefaults) -> None:
        self._defaults = defaults
        self._settings.sync_labels(
            defaults.raw_category_labels,
            default_label_entries=defaults.default_label_entries,
        )
        self._syncing = True
        try:
            self.title_edit.setText(self._settings.resolve_title(defaults))
            self.x_label_edit.setText(self._settings.resolve_x_label(defaults))
            self.y_label_edit.setText(self._settings.resolve_y_label(defaults))
            self.legend_title_edit.setText(
                self._settings.resolve_legend_title(defaults)
            )
            self.title_position_x_spin.setValue(
                self._settings.resolve_title_position_x(defaults)
            )
            self.title_position_y_spin.setValue(
                self._settings.resolve_title_position_y(defaults)
            )
            self.colormap_combo.clear()
            for colormap_name in defaults.available_colormap_names:
                self.colormap_combo.addItem(colormap_name, colormap_name)
            if defaults.available_colormap_names:
                self.colormap_combo.setCurrentIndex(
                    max(
                        0,
                        self.colormap_combo.findData(
                            defaults.default_colormap_name
                        ),
                    )
                )
            self.colormap_combo.setEnabled(
                bool(defaults.available_colormap_names)
            )
            if self._settings.font_family:
                self.font_combo.setCurrentFont(
                    QFont(self._settings.font_family)
                )
            self.title_font_spin.setValue(self._settings.title_font_size)
            self.axis_label_font_spin.setValue(
                self._settings.axis_label_font_size
            )
            self.tick_label_font_spin.setValue(
                self._settings.tick_label_font_size
            )
            self.legend_font_spin.setValue(self._settings.legend_font_size)
            self.annotation_font_spin.setValue(
                self._settings.annotation_font_size
            )
            self.show_total_annotations_checkbox.setChecked(
                self._settings.show_total_annotations
            )
            self.show_legend_checkbox.setChecked(self._settings.show_legend)
            self.legend_location_combo.setCurrentIndex(
                max(
                    0,
                    self.legend_location_combo.findData(
                        self._settings.legend_location
                    ),
                )
            )
            self.max_y_ticks_spin.setValue(self._settings.max_y_ticks)
            self.x_tick_rotation_spin.setValue(self._settings.x_tick_rotation)
            self.y_tick_rotation_spin.setValue(self._settings.y_tick_rotation)
            self.minor_y_ticks_checkbox.setChecked(
                self._settings.show_minor_y_ticks
            )
            self._sync_label_table()
            self._update_legend_state()
        finally:
            self._last_synced_defaults = defaults
            self._syncing = False

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(10)

        note = QLabel(
            "Edit stacked-histogram text, legend layout, tick styling, and "
            "category order. Use $_{n}$ for subscript and $^{n}$ for "
            "superscript (matplotlib mathtext). Igor-style inline text is "
            "also supported: \\f01 bold, \\f02 italics, \\f00 reset, and "
            "\\Z<NN> inline font size."
        )
        note.setWordWrap(True)
        root.addWidget(note)

        text_group = QGroupBox("Text")
        text_form = QFormLayout(text_group)
        self.title_edit = QLineEdit()
        self.title_edit.textChanged.connect(self._on_title_changed)
        text_form.addRow("Title", self.title_edit)
        self.x_label_edit = QLineEdit()
        self.x_label_edit.textChanged.connect(self._on_x_label_changed)
        text_form.addRow("X Label", self.x_label_edit)
        self.y_label_edit = QLineEdit()
        self.y_label_edit.textChanged.connect(self._on_y_label_changed)
        text_form.addRow("Y Label", self.y_label_edit)
        self.legend_title_edit = QLineEdit()
        self.legend_title_edit.textChanged.connect(
            self._on_legend_title_changed
        )
        text_form.addRow("Legend Title", self.legend_title_edit)
        self.colormap_combo = QComboBox()
        self.colormap_combo.currentIndexChanged.connect(
            self._on_colormap_changed
        )
        text_form.addRow("Colormap", self.colormap_combo)
        self.reset_text_button = QPushButton("Reset Text Defaults")
        self.reset_text_button.clicked.connect(self._reset_text_defaults)
        text_form.addRow(self.reset_text_button)
        root.addWidget(text_group)

        style_group = QGroupBox("Style")
        style_form = QFormLayout(style_group)
        self.font_combo = QFontComboBox()
        self.font_combo.currentFontChanged.connect(self._on_font_changed)
        style_form.addRow("Font", self.font_combo)
        self.title_font_spin = HeatmapPlotEditorControls._build_font_spin()
        self.title_font_spin.valueChanged.connect(
            self._on_title_font_size_changed
        )
        style_form.addRow("Title Size", self.title_font_spin)
        self.title_position_x_spin = (
            HeatmapPlotEditorControls._build_position_spin()
        )
        self.title_position_x_spin.valueChanged.connect(
            self._on_title_position_x_changed
        )
        style_form.addRow("Title X", self.title_position_x_spin)
        self.title_position_y_spin = (
            HeatmapPlotEditorControls._build_position_spin()
        )
        self.title_position_y_spin.valueChanged.connect(
            self._on_title_position_y_changed
        )
        style_form.addRow("Title Y", self.title_position_y_spin)
        self.axis_label_font_spin = (
            HeatmapPlotEditorControls._build_font_spin()
        )
        self.axis_label_font_spin.valueChanged.connect(
            self._on_axis_label_font_size_changed
        )
        style_form.addRow("Axis Label Size", self.axis_label_font_spin)
        self.tick_label_font_spin = (
            HeatmapPlotEditorControls._build_font_spin()
        )
        self.tick_label_font_spin.valueChanged.connect(
            self._on_tick_label_font_size_changed
        )
        style_form.addRow("Tick Label Size", self.tick_label_font_spin)
        self.legend_font_spin = HeatmapPlotEditorControls._build_font_spin()
        self.legend_font_spin.valueChanged.connect(
            self._on_legend_font_size_changed
        )
        style_form.addRow("Legend Size", self.legend_font_spin)
        self.annotation_font_spin = (
            HeatmapPlotEditorControls._build_font_spin()
        )
        self.annotation_font_spin.valueChanged.connect(
            self._on_annotation_font_size_changed
        )
        style_form.addRow("Totals Size", self.annotation_font_spin)
        root.addWidget(style_group)

        display_group = QGroupBox("Display")
        display_form = QFormLayout(display_group)
        self.show_total_annotations_checkbox = QCheckBox(
            "Show Total Percent Labels"
        )
        self.show_total_annotations_checkbox.toggled.connect(
            self._on_show_total_annotations_changed
        )
        display_form.addRow(self.show_total_annotations_checkbox)
        self.show_legend_checkbox = QCheckBox("Show Legend")
        self.show_legend_checkbox.toggled.connect(self._on_show_legend_changed)
        display_form.addRow(self.show_legend_checkbox)
        self.legend_location_combo = QComboBox()
        for label, value in STACKED_HISTOGRAM_LEGEND_LOCATIONS:
            self.legend_location_combo.addItem(label, value)
        self.legend_location_combo.currentIndexChanged.connect(
            self._on_legend_location_changed
        )
        display_form.addRow("Legend Position", self.legend_location_combo)
        root.addWidget(display_group)

        ticks_group = QGroupBox("Ticks")
        ticks_form = QFormLayout(ticks_group)
        self.max_y_ticks_spin = QSpinBox()
        self.max_y_ticks_spin.setRange(2, 20)
        self.max_y_ticks_spin.valueChanged.connect(
            self._on_max_y_ticks_changed
        )
        ticks_form.addRow("Max Y Ticks", self.max_y_ticks_spin)
        self.x_tick_rotation_spin = QSpinBox()
        self.x_tick_rotation_spin.setRange(-180, 180)
        self.x_tick_rotation_spin.valueChanged.connect(
            self._on_x_tick_rotation_changed
        )
        ticks_form.addRow("X Tick Rotation", self.x_tick_rotation_spin)
        self.y_tick_rotation_spin = QSpinBox()
        self.y_tick_rotation_spin.setRange(-180, 180)
        self.y_tick_rotation_spin.valueChanged.connect(
            self._on_y_tick_rotation_changed
        )
        ticks_form.addRow("Y Tick Rotation", self.y_tick_rotation_spin)
        self.minor_y_ticks_checkbox = QCheckBox("Show Y Minor Ticks")
        self.minor_y_ticks_checkbox.toggled.connect(
            self._on_minor_y_ticks_changed
        )
        ticks_form.addRow(self.minor_y_ticks_checkbox)
        root.addWidget(ticks_group)

        labels_group = QGroupBox("Axis Labels")
        labels_layout = QVBoxLayout(labels_group)
        labels_layout.setContentsMargins(8, 8, 8, 8)
        labels_layout.setSpacing(8)
        labels_note = QLabel(
            "Rearrange rows to control x-axis order and edit Display Label "
            "to customise tick text. The raw label column stays fixed so you "
            "can round-trip custom formatting."
        )
        labels_note.setWordWrap(True)
        labels_layout.addWidget(labels_note)
        self.label_table = QTableWidget(0, 2)
        self.label_table.setHorizontalHeaderLabels(
            ["Raw Label", "Display Label"]
        )
        self.label_table.horizontalHeader().setStretchLastSection(True)
        self.label_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.label_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.label_table.itemChanged.connect(self._on_label_item_changed)
        content_row = QHBoxLayout()
        content_row.addWidget(self.label_table, stretch=1)
        move_column = QVBoxLayout()
        self.move_label_up_button = QPushButton("Up")
        self.move_label_up_button.clicked.connect(self._move_label_up)
        move_column.addWidget(self.move_label_up_button)
        self.move_label_down_button = QPushButton("Down")
        self.move_label_down_button.clicked.connect(self._move_label_down)
        move_column.addWidget(self.move_label_down_button)
        move_column.addStretch(1)
        content_row.addLayout(move_column)
        labels_layout.addLayout(content_row, stretch=1)
        button_row = QHBoxLayout()
        self.histogram_order_button = QPushButton("Sort Like Histogram")
        self.histogram_order_button.clicked.connect(
            self._apply_histogram_order
        )
        button_row.addWidget(self.histogram_order_button)
        self.auto_subscript_button = QPushButton("Auto Stoich Subscripts")
        self.auto_subscript_button.clicked.connect(
            self._apply_stoich_subscripts
        )
        button_row.addWidget(self.auto_subscript_button)
        self.reset_labels_button = QPushButton("Reset Labels")
        self.reset_labels_button.clicked.connect(self._reset_labels)
        button_row.addWidget(self.reset_labels_button)
        button_row.addStretch(1)
        labels_layout.addLayout(button_row)
        root.addWidget(labels_group, stretch=1)
        root.addStretch(1)

    def _sync_label_table(self) -> None:
        self.label_table.blockSignals(True)
        try:
            raw_labels = self._settings.ordered_raw_labels(self._defaults)
            self.label_table.setRowCount(len(raw_labels))
            for row, raw_label in enumerate(raw_labels):
                raw_item = QTableWidgetItem(raw_label)
                raw_item.setFlags(
                    raw_item.flags() & ~Qt.ItemFlag.ItemIsEditable
                )
                self.label_table.setItem(row, 0, raw_item)
                self.label_table.setItem(
                    row,
                    1,
                    QTableWidgetItem(self._settings.display_label(raw_label)),
                )
            if raw_labels:
                self.label_table.resizeColumnToContents(0)
                if self.label_table.currentRow() < 0:
                    self.label_table.selectRow(0)
        finally:
            self.label_table.blockSignals(False)

    def _update_legend_state(self) -> None:
        self.legend_location_combo.setEnabled(
            self.show_legend_checkbox.isChecked()
        )

    def _emit_settings_changed(self) -> None:
        if not self._syncing:
            self.settings_changed.emit()

    def _emit_label_settings_changed(self) -> None:
        if not self._syncing:
            self.label_settings_changed.emit()

    def _on_title_changed(self, text: str) -> None:
        if self._syncing:
            return
        self._settings.title = text
        self._emit_settings_changed()

    def _on_x_label_changed(self, text: str) -> None:
        if self._syncing:
            return
        self._settings.x_label = text
        self._emit_settings_changed()

    def _on_y_label_changed(self, text: str) -> None:
        if self._syncing:
            return
        self._settings.y_label = text
        self._emit_settings_changed()

    def _on_legend_title_changed(self, text: str) -> None:
        if self._syncing:
            return
        self._settings.legend_title = text
        self._emit_settings_changed()

    def _on_colormap_changed(self) -> None:
        if self._syncing:
            return
        current = self.colormap_combo.currentData()
        if current is None:
            return
        self.colormap_changed.emit(str(current))

    def _on_font_changed(self, font: QFont) -> None:
        if self._syncing:
            return
        self._settings.font_family = font.family()
        self._emit_settings_changed()

    def _on_title_font_size_changed(self, value: float) -> None:
        if self._syncing:
            return
        self._settings.title_font_size = float(value)
        self._emit_settings_changed()

    def _on_title_position_x_changed(self, value: float) -> None:
        if self._syncing:
            return
        self._settings.title_position_x = float(value)
        self._emit_settings_changed()

    def _on_title_position_y_changed(self, value: float) -> None:
        if self._syncing:
            return
        self._settings.title_position_y = float(value)
        self._emit_settings_changed()

    def _on_axis_label_font_size_changed(self, value: float) -> None:
        if self._syncing:
            return
        self._settings.axis_label_font_size = float(value)
        self._emit_settings_changed()

    def _on_tick_label_font_size_changed(self, value: float) -> None:
        if self._syncing:
            return
        self._settings.tick_label_font_size = float(value)
        self._emit_settings_changed()

    def _on_legend_font_size_changed(self, value: float) -> None:
        if self._syncing:
            return
        self._settings.legend_font_size = float(value)
        self._emit_settings_changed()

    def _on_annotation_font_size_changed(self, value: float) -> None:
        if self._syncing:
            return
        self._settings.annotation_font_size = float(value)
        self._emit_settings_changed()

    def _on_show_total_annotations_changed(self, checked: bool) -> None:
        if self._syncing:
            return
        self._settings.show_total_annotations = bool(checked)
        self._emit_settings_changed()

    def _on_show_legend_changed(self, checked: bool) -> None:
        if self._syncing:
            return
        self._settings.show_legend = bool(checked)
        self._update_legend_state()
        self._emit_settings_changed()

    def _on_legend_location_changed(self) -> None:
        if self._syncing:
            return
        current = self.legend_location_combo.currentData()
        self._settings.legend_location = (
            "outside_upper_right" if current is None else str(current)
        )
        self._emit_settings_changed()

    def _on_max_y_ticks_changed(self, value: int) -> None:
        if self._syncing:
            return
        self._settings.max_y_ticks = int(value)
        self._emit_settings_changed()

    def _on_x_tick_rotation_changed(self, value: int) -> None:
        if self._syncing:
            return
        self._settings.x_tick_rotation = int(value)
        self._emit_settings_changed()

    def _on_y_tick_rotation_changed(self, value: int) -> None:
        if self._syncing:
            return
        self._settings.y_tick_rotation = int(value)
        self._emit_settings_changed()

    def _on_minor_y_ticks_changed(self, checked: bool) -> None:
        if self._syncing:
            return
        self._settings.show_minor_y_ticks = bool(checked)
        self._emit_settings_changed()

    def _on_label_item_changed(self, item: QTableWidgetItem) -> None:
        if self._syncing or item.column() != 1:
            return
        raw_item = self.label_table.item(item.row(), 0)
        if raw_item is None:
            return
        self._settings.label_map[raw_item.text()] = item.text()
        self._emit_label_settings_changed()
        self._emit_settings_changed()

    def _move_label_up(self) -> None:
        row = self.label_table.currentRow()
        if row <= 0:
            return
        self._swap_label_rows(row, row - 1)
        self.label_table.selectRow(row - 1)
        self._store_label_order_from_table()
        self._emit_label_settings_changed()
        self._emit_settings_changed()

    def _move_label_down(self) -> None:
        row = self.label_table.currentRow()
        if row < 0 or row >= self.label_table.rowCount() - 1:
            return
        self._swap_label_rows(row, row + 1)
        self.label_table.selectRow(row + 1)
        self._store_label_order_from_table()
        self._emit_label_settings_changed()
        self._emit_settings_changed()

    def _swap_label_rows(self, row_a: int, row_b: int) -> None:
        self.label_table.blockSignals(True)
        try:
            for column in range(self.label_table.columnCount()):
                item_a = self.label_table.takeItem(row_a, column)
                item_b = self.label_table.takeItem(row_b, column)
                if item_a is not None:
                    self.label_table.setItem(row_b, column, item_a)
                if item_b is not None:
                    self.label_table.setItem(row_a, column, item_b)
        finally:
            self.label_table.blockSignals(False)

    def _store_label_order_from_table(self) -> None:
        self._settings.label_order = []
        for row in range(self.label_table.rowCount()):
            raw_item = self.label_table.item(row, 0)
            display_item = self.label_table.item(row, 1)
            if raw_item is None:
                continue
            raw_label = raw_item.text()
            self._settings.label_order.append(raw_label)
            self._settings.label_map[raw_label] = (
                display_item.text() if display_item is not None else raw_label
            )

    def _reset_text_defaults(self) -> None:
        self._settings.title = None
        self._settings.x_label = None
        self._settings.y_label = None
        self._settings.legend_title = None
        self._settings.title_position_x = None
        self._settings.title_position_y = None
        self.sync_defaults(self._defaults)
        self._emit_settings_changed()

    def _reset_labels(self) -> None:
        self._settings.label_order = list(self._defaults.raw_category_labels)
        default_map = dict(self._defaults.default_label_entries)
        self._settings.label_map = {
            raw_label: default_map.get(raw_label, raw_label)
            for raw_label in self._defaults.raw_category_labels
        }
        self.sync_defaults(self._defaults)
        self._emit_label_settings_changed()
        self._emit_settings_changed()

    def _apply_stoich_subscripts(self) -> None:
        self._settings.label_map = {
            raw_label: format_stoich_for_axis(raw_label)
            for raw_label in self._settings.ordered_raw_labels(self._defaults)
        }
        self.sync_defaults(self._defaults)
        self._emit_label_settings_changed()
        self._emit_settings_changed()

    def _apply_histogram_order(self) -> None:
        ordered = sort_stoich_labels(self._defaults.raw_category_labels)
        self._settings.label_order = list(ordered)
        self.sync_defaults(self._defaults)
        self._emit_label_settings_changed()
        self._emit_settings_changed()


__all__ = [
    "HeatmapPlotDefaults",
    "HeatmapPlotEditorControls",
    "HeatmapPlotSettings",
    "PlotEditorWindow",
    "StackedHistogramPlotDefaults",
    "StackedHistogramPlotEditorControls",
    "StackedHistogramPlotSettings",
    "load_pickled_plot_payload",
    "load_pickled_plot_figure",
    "save_pickled_plot_figure",
]
