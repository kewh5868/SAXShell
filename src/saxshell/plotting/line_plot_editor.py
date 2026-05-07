from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFontComboBox,
    QFormLayout,
    QGroupBox,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

LINE_PLOT_LEGEND_LOCATIONS = (
    ("Best", "best"),
    ("Upper Right", "upper right"),
    ("Upper Left", "upper left"),
    ("Lower Right", "lower right"),
    ("Lower Left", "lower left"),
)


@dataclass(slots=True)
class LinePlotSeriesDefaults:
    key: str
    label: str
    axis_label: str = "Main"


@dataclass(slots=True)
class LinePlotDefaults:
    title: str
    x_label: str
    primary_y_label: str
    secondary_y_label: str = ""
    residual_y_label: str = ""
    title_position_x: float = 0.5
    title_position_y: float = 1.0
    has_secondary_y_axis: bool = False
    has_residual_y_axis: bool = False
    has_annotation: bool = False
    default_show_legend: bool = True
    default_legend_location: str = "best"
    default_show_annotation: bool = True
    series_defaults: tuple[LinePlotSeriesDefaults, ...] = ()


@dataclass(slots=True)
class LinePlotSettings:
    title: str | None = None
    x_label: str | None = None
    primary_y_label: str | None = None
    secondary_y_label: str | None = None
    residual_y_label: str | None = None
    title_position_x: float | None = None
    title_position_y: float | None = None
    font_family: str = ""
    title_font_size: float = 12.0
    axis_label_font_size: float = 11.0
    tick_label_font_size: float = 9.0
    primary_axis_label_font_size: float | None = None
    primary_tick_label_font_size: float | None = None
    secondary_axis_label_font_size: float | None = None
    secondary_tick_label_font_size: float | None = None
    legend_font_size: float = 9.0
    annotation_font_size: float = 9.0
    show_legend: bool | None = None
    legend_location: str | None = None
    show_annotation: bool | None = None
    series_label_map: dict[str, str] = field(default_factory=dict)

    def resolve_title(self, defaults: LinePlotDefaults) -> str:
        return defaults.title if self.title is None else self.title

    def resolve_x_label(self, defaults: LinePlotDefaults) -> str:
        return defaults.x_label if self.x_label is None else self.x_label

    def resolve_primary_y_label(self, defaults: LinePlotDefaults) -> str:
        return (
            defaults.primary_y_label
            if self.primary_y_label is None
            else self.primary_y_label
        )

    def resolve_secondary_y_label(self, defaults: LinePlotDefaults) -> str:
        return (
            defaults.secondary_y_label
            if self.secondary_y_label is None
            else self.secondary_y_label
        )

    def resolve_residual_y_label(self, defaults: LinePlotDefaults) -> str:
        return (
            defaults.residual_y_label
            if self.residual_y_label is None
            else self.residual_y_label
        )

    def resolve_title_position_x(self, defaults: LinePlotDefaults) -> float:
        return (
            defaults.title_position_x
            if self.title_position_x is None
            else self.title_position_x
        )

    def resolve_title_position_y(self, defaults: LinePlotDefaults) -> float:
        return (
            defaults.title_position_y
            if self.title_position_y is None
            else self.title_position_y
        )

    def resolve_show_legend(self, defaults: LinePlotDefaults) -> bool:
        return (
            defaults.default_show_legend
            if self.show_legend is None
            else bool(self.show_legend)
        )

    def resolve_primary_axis_label_font_size(
        self,
        defaults: LinePlotDefaults,
    ) -> float:
        del defaults
        return (
            self.axis_label_font_size
            if self.primary_axis_label_font_size is None
            else float(self.primary_axis_label_font_size)
        )

    def resolve_primary_tick_label_font_size(
        self,
        defaults: LinePlotDefaults,
    ) -> float:
        del defaults
        return (
            self.tick_label_font_size
            if self.primary_tick_label_font_size is None
            else float(self.primary_tick_label_font_size)
        )

    def resolve_secondary_axis_label_font_size(
        self,
        defaults: LinePlotDefaults,
    ) -> float:
        del defaults
        return (
            self.axis_label_font_size
            if self.secondary_axis_label_font_size is None
            else float(self.secondary_axis_label_font_size)
        )

    def resolve_secondary_tick_label_font_size(
        self,
        defaults: LinePlotDefaults,
    ) -> float:
        del defaults
        return (
            self.tick_label_font_size
            if self.secondary_tick_label_font_size is None
            else float(self.secondary_tick_label_font_size)
        )

    def resolve_legend_location(self, defaults: LinePlotDefaults) -> str:
        return (
            defaults.default_legend_location
            if self.legend_location is None
            else str(self.legend_location)
        )

    def resolve_show_annotation(self, defaults: LinePlotDefaults) -> bool:
        return (
            defaults.default_show_annotation
            if self.show_annotation is None
            else bool(self.show_annotation)
        )

    def sync_series(
        self,
        series_defaults: Sequence[LinePlotSeriesDefaults],
    ) -> None:
        default_map = {
            str(series.key): str(series.label) for series in series_defaults
        }
        existing = dict(self.series_label_map)
        self.series_label_map = {
            key: existing.get(key, label) for key, label in default_map.items()
        }

    def display_series_label(self, series_key: str, fallback: str) -> str:
        return self.series_label_map.get(series_key, fallback)

    def to_dict(self) -> dict[str, object]:
        return {
            "title": self.title,
            "x_label": self.x_label,
            "primary_y_label": self.primary_y_label,
            "secondary_y_label": self.secondary_y_label,
            "residual_y_label": self.residual_y_label,
            "title_position_x": self.title_position_x,
            "title_position_y": self.title_position_y,
            "font_family": self.font_family,
            "title_font_size": self.title_font_size,
            "axis_label_font_size": self.axis_label_font_size,
            "tick_label_font_size": self.tick_label_font_size,
            "primary_axis_label_font_size": self.primary_axis_label_font_size,
            "primary_tick_label_font_size": self.primary_tick_label_font_size,
            "secondary_axis_label_font_size": self.secondary_axis_label_font_size,
            "secondary_tick_label_font_size": self.secondary_tick_label_font_size,
            "legend_font_size": self.legend_font_size,
            "annotation_font_size": self.annotation_font_size,
            "show_legend": self.show_legend,
            "legend_location": self.legend_location,
            "show_annotation": self.show_annotation,
            "series_label_map": dict(self.series_label_map),
        }

    def update_from_dict(self, payload: Mapping[str, object]) -> None:
        optional_float_fields = {
            "primary_axis_label_font_size",
            "primary_tick_label_font_size",
            "secondary_axis_label_font_size",
            "secondary_tick_label_font_size",
        }
        for field_name in (
            "title",
            "x_label",
            "primary_y_label",
            "secondary_y_label",
            "residual_y_label",
            "title_position_x",
            "title_position_y",
            "show_legend",
            "legend_location",
            "show_annotation",
        ):
            if field_name in payload:
                setattr(self, field_name, payload[field_name])
        if "font_family" in payload:
            self.font_family = str(payload["font_family"] or "")
        for field_name in (
            "title_font_size",
            "axis_label_font_size",
            "tick_label_font_size",
            "primary_axis_label_font_size",
            "primary_tick_label_font_size",
            "secondary_axis_label_font_size",
            "secondary_tick_label_font_size",
            "legend_font_size",
            "annotation_font_size",
        ):
            if field_name in payload:
                value = payload[field_name]
                if value is None:
                    if field_name in optional_float_fields:
                        setattr(self, field_name, None)
                    continue
                setattr(self, field_name, float(value))
        if "series_label_map" in payload:
            series_label_map = payload["series_label_map"]
            if isinstance(series_label_map, Mapping):
                self.series_label_map = {
                    str(key): str(value)
                    for key, value in series_label_map.items()
                }


class LinePlotEditorControls(QWidget):
    """Editable controls for reusable line-plot settings."""

    settings_changed = Signal()
    label_settings_changed = Signal()

    def __init__(
        self,
        *,
        settings: LinePlotSettings,
        defaults: LinePlotDefaults,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._settings = settings
        self._defaults = defaults
        self._last_synced_defaults: LinePlotDefaults | None = None
        self._syncing = False
        self._build_ui()
        self.sync_defaults(defaults)

    def needs_default_sync(self, defaults: LinePlotDefaults) -> bool:
        return self._last_synced_defaults != defaults

    def sync_defaults(self, defaults: LinePlotDefaults) -> None:
        self._defaults = defaults
        self._settings.sync_series(defaults.series_defaults)
        self._syncing = True
        try:
            self.title_edit.setText(self._settings.resolve_title(defaults))
            self.x_label_edit.setText(self._settings.resolve_x_label(defaults))
            self.primary_y_label_edit.setText(
                self._settings.resolve_primary_y_label(defaults)
            )
            self.secondary_y_label_edit.setText(
                self._settings.resolve_secondary_y_label(defaults)
            )
            self.residual_y_label_edit.setText(
                self._settings.resolve_residual_y_label(defaults)
            )
            self.title_position_x_spin.setValue(
                self._settings.resolve_title_position_x(defaults)
            )
            self.title_position_y_spin.setValue(
                self._settings.resolve_title_position_y(defaults)
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
            self.primary_axis_label_font_spin.setValue(
                self._settings.resolve_primary_axis_label_font_size(defaults)
            )
            self.primary_tick_label_font_spin.setValue(
                self._settings.resolve_primary_tick_label_font_size(defaults)
            )
            self.secondary_axis_label_font_spin.setValue(
                self._settings.resolve_secondary_axis_label_font_size(defaults)
            )
            self.secondary_tick_label_font_spin.setValue(
                self._settings.resolve_secondary_tick_label_font_size(defaults)
            )
            self.legend_font_spin.setValue(self._settings.legend_font_size)
            self.annotation_font_spin.setValue(
                self._settings.annotation_font_size
            )
            self.show_legend_checkbox.setChecked(
                self._settings.resolve_show_legend(defaults)
            )
            self.legend_location_combo.setCurrentIndex(
                max(
                    0,
                    self.legend_location_combo.findData(
                        self._settings.resolve_legend_location(defaults)
                    ),
                )
            )
            self.show_annotation_checkbox.setChecked(
                self._settings.resolve_show_annotation(defaults)
            )
            self._sync_label_table()
            self._update_dynamic_field_visibility()
            self._update_display_state()
        finally:
            self._last_synced_defaults = defaults
            self._syncing = False

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(10)

        note = QLabel(
            "Edit line-plot titles, axis labels, legend layout, and series "
            "display labels. Axis-specific fields appear only when the "
            "current plot uses them."
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
        self.primary_y_label_edit = QLineEdit()
        self.primary_y_label_edit.textChanged.connect(
            self._on_primary_y_label_changed
        )
        text_form.addRow("Y Label", self.primary_y_label_edit)
        self.secondary_y_label_edit = QLineEdit()
        self.secondary_y_label_edit.textChanged.connect(
            self._on_secondary_y_label_changed
        )
        text_form.addRow("Secondary Y", self.secondary_y_label_edit)
        self.residual_y_label_edit = QLineEdit()
        self.residual_y_label_edit.textChanged.connect(
            self._on_residual_y_label_changed
        )
        text_form.addRow("Residual Y", self.residual_y_label_edit)
        self.reset_text_button = QPushButton("Reset Text Defaults")
        self.reset_text_button.clicked.connect(self._reset_text_defaults)
        text_form.addRow(self.reset_text_button)
        root.addWidget(text_group)

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
        style_form.addRow("X Label Size", self.axis_label_font_spin)
        self.tick_label_font_spin = self._build_font_spin()
        self.tick_label_font_spin.valueChanged.connect(
            self._on_tick_label_font_size_changed
        )
        style_form.addRow("X Tick Size", self.tick_label_font_spin)
        self.primary_axis_label_font_spin = self._build_font_spin()
        self.primary_axis_label_font_spin.valueChanged.connect(
            self._on_primary_axis_label_font_size_changed
        )
        style_form.addRow(
            "Primary Y Label Size",
            self.primary_axis_label_font_spin,
        )
        self.primary_tick_label_font_spin = self._build_font_spin()
        self.primary_tick_label_font_spin.valueChanged.connect(
            self._on_primary_tick_label_font_size_changed
        )
        style_form.addRow(
            "Primary Y Tick Size",
            self.primary_tick_label_font_spin,
        )
        self.secondary_axis_label_font_spin = self._build_font_spin()
        self.secondary_axis_label_font_spin.valueChanged.connect(
            self._on_secondary_axis_label_font_size_changed
        )
        style_form.addRow(
            "Secondary Y Label Size",
            self.secondary_axis_label_font_spin,
        )
        self.secondary_tick_label_font_spin = self._build_font_spin()
        self.secondary_tick_label_font_spin.valueChanged.connect(
            self._on_secondary_tick_label_font_size_changed
        )
        style_form.addRow(
            "Secondary Y Tick Size",
            self.secondary_tick_label_font_spin,
        )
        self.legend_font_spin = self._build_font_spin()
        self.legend_font_spin.valueChanged.connect(
            self._on_legend_font_size_changed
        )
        style_form.addRow("Legend Size", self.legend_font_spin)
        self.annotation_font_spin = self._build_font_spin()
        self.annotation_font_spin.valueChanged.connect(
            self._on_annotation_font_size_changed
        )
        style_form.addRow("Annotation Size", self.annotation_font_spin)
        root.addWidget(style_group)

        display_group = QGroupBox("Display")
        display_form = QFormLayout(display_group)
        self.show_legend_checkbox = QCheckBox("Show Legend")
        self.show_legend_checkbox.toggled.connect(self._on_show_legend_changed)
        display_form.addRow(self.show_legend_checkbox)
        self.legend_location_combo = QComboBox()
        for label, value in LINE_PLOT_LEGEND_LOCATIONS:
            self.legend_location_combo.addItem(label, value)
        self.legend_location_combo.currentIndexChanged.connect(
            self._on_legend_location_changed
        )
        display_form.addRow("Legend Position", self.legend_location_combo)
        self.show_annotation_checkbox = QCheckBox("Show Annotation")
        self.show_annotation_checkbox.toggled.connect(
            self._on_show_annotation_changed
        )
        display_form.addRow(self.show_annotation_checkbox)
        self.reset_display_button = QPushButton("Reset Display Defaults")
        self.reset_display_button.clicked.connect(self._reset_display_defaults)
        display_form.addRow(self.reset_display_button)
        root.addWidget(display_group)

        labels_group = QGroupBox("Series Labels")
        labels_layout = QVBoxLayout(labels_group)
        labels_layout.setContentsMargins(8, 8, 8, 8)
        labels_layout.setSpacing(8)
        labels_note = QLabel(
            "Edit display labels for the traces that are currently plotted."
        )
        labels_note.setWordWrap(True)
        labels_layout.addWidget(labels_note)
        self.label_table = QTableWidget(0, 3)
        self.label_table.setHorizontalHeaderLabels(
            ["Series", "Axis", "Display Label"]
        )
        self.label_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        self.label_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents
        )
        self.label_table.horizontalHeader().setStretchLastSection(True)
        self.label_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.label_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.label_table.itemChanged.connect(self._on_label_item_changed)
        labels_layout.addWidget(self.label_table)
        self.reset_labels_button = QPushButton("Reset Series Labels")
        self.reset_labels_button.clicked.connect(self._reset_labels)
        labels_layout.addWidget(self.reset_labels_button)
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

    def _set_form_row_visible(self, field: QWidget, visible: bool) -> None:
        label = self._form_label_for_field(field)
        if label is not None:
            label.setVisible(visible)
        field.setVisible(visible)

    def _form_label_for_field(self, field: QWidget) -> QWidget | None:
        for group in self.findChildren(QGroupBox):
            layout = group.layout()
            if isinstance(layout, QFormLayout):
                label = layout.labelForField(field)
                if label is not None:
                    return label
        return None

    def _update_dynamic_field_visibility(self) -> None:
        self._set_form_row_visible(
            self.secondary_y_label_edit,
            self._defaults.has_secondary_y_axis,
        )
        self._set_form_row_visible(
            self.secondary_axis_label_font_spin,
            self._defaults.has_secondary_y_axis,
        )
        self._set_form_row_visible(
            self.secondary_tick_label_font_spin,
            self._defaults.has_secondary_y_axis,
        )
        self._set_form_row_visible(
            self.residual_y_label_edit,
            self._defaults.has_residual_y_axis,
        )
        self._set_form_row_visible(
            self.annotation_font_spin,
            self._defaults.has_annotation,
        )
        self.show_annotation_checkbox.setVisible(self._defaults.has_annotation)

    def _update_display_state(self) -> None:
        self.legend_location_combo.setEnabled(
            self.show_legend_checkbox.isChecked()
        )
        self.annotation_font_spin.setEnabled(
            self._defaults.has_annotation
            and self.show_annotation_checkbox.isChecked()
        )

    def _sync_label_table(self) -> None:
        self.label_table.blockSignals(True)
        try:
            self.label_table.setRowCount(len(self._defaults.series_defaults))
            for row, series in enumerate(self._defaults.series_defaults):
                raw_item = QTableWidgetItem(series.label)
                raw_item.setFlags(
                    raw_item.flags() & ~Qt.ItemFlag.ItemIsEditable
                )
                raw_item.setData(Qt.ItemDataRole.UserRole, series.key)
                axis_item = QTableWidgetItem(series.axis_label)
                axis_item.setFlags(
                    axis_item.flags() & ~Qt.ItemFlag.ItemIsEditable
                )
                label_item = QTableWidgetItem(
                    self._settings.display_series_label(
                        series.key,
                        series.label,
                    )
                )
                self.label_table.setItem(row, 0, raw_item)
                self.label_table.setItem(row, 1, axis_item)
                self.label_table.setItem(row, 2, label_item)
            if (
                self._defaults.series_defaults
                and self.label_table.currentRow() < 0
            ):
                self.label_table.selectRow(0)
        finally:
            self.label_table.blockSignals(False)

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

    def _on_primary_y_label_changed(self, text: str) -> None:
        if self._syncing:
            return
        self._settings.primary_y_label = text
        self._emit_settings_changed()

    def _on_secondary_y_label_changed(self, text: str) -> None:
        if self._syncing:
            return
        self._settings.secondary_y_label = text
        self._emit_settings_changed()

    def _on_residual_y_label_changed(self, text: str) -> None:
        if self._syncing:
            return
        self._settings.residual_y_label = text
        self._emit_settings_changed()

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

    def _on_primary_axis_label_font_size_changed(self, value: float) -> None:
        if self._syncing:
            return
        self._settings.primary_axis_label_font_size = float(value)
        self._emit_settings_changed()

    def _on_primary_tick_label_font_size_changed(self, value: float) -> None:
        if self._syncing:
            return
        self._settings.primary_tick_label_font_size = float(value)
        self._emit_settings_changed()

    def _on_secondary_axis_label_font_size_changed(self, value: float) -> None:
        if self._syncing:
            return
        self._settings.secondary_axis_label_font_size = float(value)
        self._emit_settings_changed()

    def _on_secondary_tick_label_font_size_changed(self, value: float) -> None:
        if self._syncing:
            return
        self._settings.secondary_tick_label_font_size = float(value)
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

    def _on_show_legend_changed(self, checked: bool) -> None:
        if self._syncing:
            return
        self._settings.show_legend = bool(checked)
        self._update_display_state()
        self._emit_settings_changed()

    def _on_legend_location_changed(self) -> None:
        if self._syncing:
            return
        current = self.legend_location_combo.currentData()
        self._settings.legend_location = (
            self._defaults.default_legend_location
            if current is None
            else str(current)
        )
        self._emit_settings_changed()

    def _on_show_annotation_changed(self, checked: bool) -> None:
        if self._syncing:
            return
        self._settings.show_annotation = bool(checked)
        self._update_display_state()
        self._emit_settings_changed()

    def _on_label_item_changed(self, item: QTableWidgetItem) -> None:
        if self._syncing or item.column() != 2:
            return
        raw_item = self.label_table.item(item.row(), 0)
        if raw_item is None:
            return
        series_key = str(raw_item.data(Qt.ItemDataRole.UserRole) or "")
        if not series_key:
            return
        self._settings.series_label_map[series_key] = item.text()
        self._emit_label_settings_changed()
        self._emit_settings_changed()

    def _reset_text_defaults(self) -> None:
        self._settings.title = None
        self._settings.x_label = None
        self._settings.primary_y_label = None
        self._settings.secondary_y_label = None
        self._settings.residual_y_label = None
        self._settings.title_position_x = None
        self._settings.title_position_y = None
        self.sync_defaults(self._defaults)
        self._emit_settings_changed()

    def _reset_display_defaults(self) -> None:
        self._settings.show_legend = None
        self._settings.legend_location = None
        self._settings.show_annotation = None
        self.sync_defaults(self._defaults)
        self._emit_settings_changed()

    def _reset_labels(self) -> None:
        self._settings.series_label_map = {}
        self._settings.sync_series(self._defaults.series_defaults)
        self.sync_defaults(self._defaults)
        self._emit_label_settings_changed()
        self._emit_settings_changed()


__all__ = [
    "LINE_PLOT_LEGEND_LOCATIONS",
    "LinePlotDefaults",
    "LinePlotEditorControls",
    "LinePlotSeriesDefaults",
    "LinePlotSettings",
]
