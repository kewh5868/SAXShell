from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass, replace
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListView,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QProgressDialog,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QTextEdit,
    QToolButton,
    QTreeView,
    QVBoxLayout,
    QWidget,
)

from saxshell.pdf.debyer.workflow import (
    SUPPORTED_DEBYER_MODES,
    DebyerPDFCalculation,
    DebyerPDFSettings,
    DebyerPDFWorkflow,
    calculate_number_density,
    check_debyer_runtime,
    default_parallel_debyer_jobs,
    list_saved_debyer_calculations,
    load_debyer_calculation,
    rewrite_debyer_calculation_output,
    write_debyer_calculation_metadata,
)
from saxshell.saxs.project_manager import build_project_paths
from saxshell.saxs.ui.branding import (
    configure_saxshell_application,
    load_saxshell_icon,
    prepare_saxshell_application_identity,
)


def _new_item_id() -> str:
    return uuid.uuid4().hex


def _optional_path(text: str) -> Path | None:
    stripped = text.strip()
    if not stripped:
        return None
    return Path(stripped).expanduser().resolve()


def _normalize_solute_text(raw: str) -> tuple[str, ...]:
    values = [token.strip() for token in raw.replace(";", ",").split(",")]
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not value:
            continue
        element = value[:1].upper() + value[1:].lower()
        if element in seen:
            continue
        normalized.append(element)
        seen.add(element)
    return tuple(normalized)


def _solute_text(values: tuple[str, ...]) -> str:
    return ", ".join(values)


def _suggest_project_dir(frames_dir: Path | None) -> Path:
    if frames_dir is not None:
        return frames_dir.parent / f"{frames_dir.name}_pdfbatch"
    return Path.home() / "saxshell_pdf_batch_project"


def _project_reference_text(project_dir: Path | None) -> str:
    if project_dir is None:
        return "Project reference: choose a SAXSShell project folder."
    project_file = build_project_paths(project_dir).project_file
    if project_file.is_file():
        return f"Project reference: {project_file}"
    return (
        "Project reference: "
        f"{project_file} will be used when the calculation is saved."
    )


def _project_path(value: object, project_dir: Path) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = project_dir / path
    return path.resolve()


def _coerce_optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_optional_int(value: object) -> int | None:
    numeric = _coerce_optional_float(value)
    if numeric is None:
        return None
    return int(numeric)


def _coerce_optional_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return None


def _coerce_box_dimensions(
    value: object,
) -> tuple[float, float, float] | None:
    if isinstance(value, dict):
        values = [value.get(key) for key in ("a", "b", "c")]
    elif isinstance(value, str):
        normalized = (
            value.replace("x", ",")
            .replace("X", ",")
            .replace(";", ",")
            .replace(" ", ",")
        )
        values = [part for part in normalized.split(",") if part.strip()]
    elif isinstance(value, (list, tuple)):
        values = list(value)
    else:
        return None
    if len(values) != 3:
        return None
    coerced = tuple(_coerce_optional_float(entry) for entry in values)
    if any(entry is None for entry in coerced):
        return None
    return tuple(float(entry) for entry in coerced)  # type: ignore[arg-type]


def _payload_sources(payload: dict[str, object]) -> list[dict[str, object]]:
    sources: list[dict[str, object]] = []
    for key in (
        "debyer_pdf_settings",
        "pdf_debyer_settings",
        "debyer_settings",
        "pdf_settings",
    ):
        value = payload.get(key)
        if isinstance(value, dict):
            sources.append(value)
    sources.append(payload)
    return sources


def _payload_value(
    sources: list[dict[str, object]],
    keys: tuple[str, ...],
) -> object | None:
    for source in sources:
        for key in keys:
            value = source.get(key)
            if value is not None and str(value).strip():
                return value
    return None


def _load_project_payload(project_dir: Path) -> dict[str, object]:
    project_file = build_project_paths(project_dir).project_file
    if not project_file.is_file():
        return {}
    try:
        payload = json.loads(project_file.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_latest_project_debyer_defaults(
    project_dir: Path,
) -> DebyerPDFBatchItem | None:
    for summary in list_saved_debyer_calculations(project_dir):
        try:
            calculation = load_debyer_calculation(summary.calculation_dir)
        except Exception:
            continue
        return DebyerPDFBatchItem(
            item_id=_new_item_id(),
            project_dir=project_dir,
            frames_dir=calculation.frames_dir,
            filename_prefix=calculation.filename_prefix,
            mode=calculation.mode,
            from_value=calculation.from_value,
            to_value=calculation.to_value,
            step_value=calculation.step_value,
            box_dimensions=calculation.box_dimensions,
            atom_count=calculation.atom_count,
            store_frame_outputs=calculation.store_frame_outputs,
            solute_elements=calculation.solute_elements,
            max_parallel_jobs=calculation.parallel_jobs,
        )
    return None


def _queue_item_from_project_defaults(
    project_dir: Path,
    *,
    item_id: str | None = None,
    frames_dir_override: Path | None = None,
) -> DebyerPDFBatchItem:
    resolved_project_dir = Path(project_dir).expanduser().resolve()
    item = _load_latest_project_debyer_defaults(resolved_project_dir)
    if item is None:
        item = DebyerPDFBatchItem(
            item_id=item_id or _new_item_id(),
            project_dir=resolved_project_dir,
        )
    else:
        item = replace(item, item_id=item_id or _new_item_id())

    payload = _load_project_payload(resolved_project_dir)
    sources = _payload_sources(payload)

    frames_value = _payload_value(
        sources,
        (
            "frames_dir",
            "xyz_frames_dir",
            "xyz_file_path",
            "xyz_path",
            "debyer_frames_dir",
            "pdf_frames_dir",
        ),
    )
    frames_dir = _project_path(frames_value, resolved_project_dir)
    if frames_dir_override is not None:
        frames_dir = Path(frames_dir_override).expanduser().resolve()

    filename_prefix = str(
        _payload_value(
            sources,
            ("filename_prefix", "pdf_filename_prefix", "debyer_prefix"),
        )
        or item.filename_prefix
    )
    mode = str(
        _payload_value(sources, ("mode", "pdf_mode", "debyer_mode"))
        or item.mode
    )
    from_value = _coerce_optional_float(
        _payload_value(
            sources,
            (
                "from_value",
                "r_min",
                "r_range_min",
                "pdf_from_value",
                "debyer_from_value",
            ),
        )
    )
    to_value = _coerce_optional_float(
        _payload_value(
            sources,
            (
                "to_value",
                "r_max",
                "r_range_max",
                "pdf_to_value",
                "debyer_to_value",
            ),
        )
    )
    step_value = _coerce_optional_float(
        _payload_value(
            sources,
            (
                "step_value",
                "r_step",
                "r_range_step",
                "pdf_step_value",
                "debyer_step_value",
            ),
        )
    )
    box_dimensions = _coerce_box_dimensions(
        _payload_value(
            sources,
            (
                "box_dimensions",
                "bounding_box",
                "pdf_box_dimensions",
                "debyer_box_dimensions",
            ),
        )
    )
    atom_count = _coerce_optional_int(
        _payload_value(
            sources,
            ("atom_count", "pdf_atom_count", "debyer_atom_count"),
        )
    )
    solute_value = _payload_value(
        sources,
        (
            "solute_elements",
            "pdf_solute_elements",
            "debyer_solute_elements",
        ),
    )
    if isinstance(solute_value, str):
        solute_elements = _normalize_solute_text(solute_value)
    elif isinstance(solute_value, (list, tuple, set)):
        solute_elements = _normalize_solute_text(
            ",".join(str(value) for value in solute_value)
        )
    else:
        solute_elements = item.solute_elements
    store_frame_outputs = _coerce_optional_bool(
        _payload_value(
            sources,
            ("store_frame_outputs", "pdf_store_frame_outputs"),
        )
    )
    parallel_jobs = _coerce_optional_int(
        _payload_value(
            sources,
            ("max_parallel_jobs", "parallel_jobs", "pdf_parallel_jobs"),
        )
    )

    return replace(
        item,
        project_dir=resolved_project_dir,
        frames_dir=frames_dir or item.frames_dir,
        filename_prefix=filename_prefix.strip() or item.filename_prefix,
        mode=mode if mode in SUPPORTED_DEBYER_MODES else item.mode,
        from_value=item.from_value if from_value is None else from_value,
        to_value=item.to_value if to_value is None else to_value,
        step_value=item.step_value if step_value is None else step_value,
        box_dimensions=box_dimensions or item.box_dimensions,
        atom_count=item.atom_count if atom_count is None else atom_count,
        store_frame_outputs=(
            item.store_frame_outputs
            if store_frame_outputs is None
            else store_frame_outputs
        ),
        solute_elements=solute_elements,
        max_parallel_jobs=(
            item.max_parallel_jobs
            if parallel_jobs is None
            else max(int(parallel_jobs), 1)
        ),
    )


def _coerce_r_range_maximum_for_box(
    r_max: float,
    box_dimensions: tuple[float, float, float],
) -> tuple[float, bool]:
    if any(component <= 0.0 for component in box_dimensions):
        return r_max, False
    allowed_r_max = min(box_dimensions) * 0.5
    if r_max <= allowed_r_max:
        return r_max, False
    return allowed_r_max, True


def _choose_existing_directories(
    parent: QWidget,
    *,
    title: str,
    start_dir: str | Path,
) -> tuple[Path, ...]:
    dialog = QFileDialog(parent, title, str(start_dir))
    dialog.setFileMode(QFileDialog.FileMode.Directory)
    dialog.setOption(QFileDialog.Option.ShowDirsOnly, True)
    dialog.setOption(QFileDialog.Option.DontUseNativeDialog, True)
    for view in dialog.findChildren(QListView) + dialog.findChildren(
        QTreeView
    ):
        view.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
    if dialog.exec() != int(QFileDialog.DialogCode.Accepted):
        return ()
    return tuple(
        Path(path).expanduser().resolve() for path in dialog.selectedFiles()
    )


@dataclass(slots=True)
class DebyerPDFBatchItem:
    item_id: str
    project_dir: Path | None = None
    frames_dir: Path | None = None
    filename_prefix: str = "debyer_pdf"
    mode: str = "PDF"
    from_value: float = 0.5
    to_value: float = 15.0
    step_value: float = 0.01
    box_dimensions: tuple[float, float, float] = (0.0, 0.0, 0.0)
    atom_count: int = 0
    store_frame_outputs: bool = False
    solute_elements: tuple[str, ...] = ()
    max_parallel_jobs: int = default_parallel_debyer_jobs()

    def display_name(self) -> str:
        if self.project_dir is not None:
            return self.project_dir.name
        if self.frames_dir is not None:
            return self.frames_dir.name
        return "New PDF calculation"

    def to_settings(self) -> DebyerPDFSettings:
        frames_dir = self.frames_dir
        if frames_dir is None:
            raise ValueError("Select an XYZ frames folder.")
        project_dir = self.project_dir or _suggest_project_dir(frames_dir)
        if self.atom_count <= 0:
            raise ValueError("Atom count must be positive.")
        if any(component <= 0.0 for component in self.box_dimensions):
            raise ValueError("All bounding-box dimensions must be positive.")
        return DebyerPDFSettings(
            project_dir=project_dir,
            frames_dir=frames_dir,
            filename_prefix=self.filename_prefix.strip() or "debyer_pdf",
            mode=self.mode,
            from_value=float(self.from_value),
            to_value=float(self.to_value),
            step_value=float(self.step_value),
            box_dimensions=tuple(
                float(component) for component in self.box_dimensions
            ),
            atom_count=int(self.atom_count),
            store_frame_outputs=bool(self.store_frame_outputs),
            solute_elements=tuple(self.solute_elements),
            max_parallel_jobs=int(self.max_parallel_jobs),
        )


@dataclass(slots=True, frozen=True)
class DebyerPDFExistingPartialsJob:
    project_dir: Path
    solute_elements: tuple[str, ...] = ()


class DebyerPDFBatchItemWidget(QFrame):
    settings_changed = Signal(str)
    remove_requested = Signal(str)
    duplicate_requested = Signal(str)

    def __init__(
        self,
        item: DebyerPDFBatchItem,
        *,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._item = item
        self._loading = False
        self._locked = False
        self._selected = False
        self._append_grouped_mode = False
        self._build_ui()
        self._load_item(item)
        self._set_settings_visible(False)

    @property
    def item_id(self) -> str:
        return self._item.item_id

    def item(self) -> DebyerPDFBatchItem:
        return self._item

    def set_locked(self, locked: bool) -> None:
        self._locked = bool(locked)
        self.settings_group.setEnabled(not locked)
        self.remove_button.setEnabled(not locked)
        self.duplicate_button.setEnabled(not locked)
        self._refresh_setting_widget_states()

    def set_append_grouped_mode(self, enabled: bool) -> None:
        self._append_grouped_mode = bool(enabled)
        self._refresh_setting_widget_states()

    def set_status(self, message: str) -> None:
        self.status_label.setText(message)

    def set_progress(self, processed: int, total: int) -> None:
        self.progress_bar.setRange(0, max(int(total), 1))
        self.progress_bar.setValue(max(int(processed), 0))

    def set_selected(self, selected: bool) -> None:
        self._selected = bool(selected)
        self.header_frame.setProperty("selected", self._selected)
        self.header_frame.setStyleSheet(
            "QFrame#DebyerBatchItemHeader {"
            + (
                "background-color: #dce8f7; " "border: 1px solid #8fb0d7;"
                if self._selected
                else "background-color: #f6f8fb; " "border: 1px solid #cfd7e3;"
            )
            + "border-radius: 5px;}"
        )

    def collect_item(self) -> DebyerPDFBatchItem:
        frames_dir = _optional_path(self.frames_dir_edit.text())
        project_dir = _optional_path(self.project_dir_edit.text())
        if project_dir is None and frames_dir is not None:
            project_dir = _suggest_project_dir(frames_dir)
            self.project_dir_edit.setText(str(project_dir))
        box_dimensions = (
            float(self.box_a_edit.text().strip()),
            float(self.box_b_edit.text().strip()),
            float(self.box_c_edit.text().strip()),
        )
        to_value, changed = _coerce_r_range_maximum_for_box(
            float(self.to_edit.text().strip()),
            box_dimensions,
        )
        if changed:
            self.to_edit.setText(f"{to_value:g}")
            self.status_label.setText(
                "r max adjusted to half of the minimum box dimension."
            )
        self._item = DebyerPDFBatchItem(
            item_id=self._item.item_id,
            project_dir=project_dir,
            frames_dir=frames_dir,
            filename_prefix=self.filename_prefix_edit.text().strip()
            or "debyer_pdf",
            mode=self.mode_combo.currentText(),
            from_value=float(self.from_edit.text().strip()),
            to_value=to_value,
            step_value=float(self.step_edit.text().strip()),
            box_dimensions=box_dimensions,
            atom_count=int(float(self.atom_count_edit.text().strip())),
            store_frame_outputs=self.store_frame_outputs_checkbox.isChecked(),
            solute_elements=_normalize_solute_text(
                self.solute_elements_edit.text()
            ),
            max_parallel_jobs=int(self.parallel_jobs_spin.value()),
        )
        self._refresh_header()
        self._refresh_project_reference()
        self._refresh_rho0_label()
        return self._item

    def settings(self) -> DebyerPDFSettings:
        return self.collect_item().to_settings()

    def existing_partials_job(self) -> DebyerPDFExistingPartialsJob:
        frames_dir = _optional_path(self.frames_dir_edit.text())
        project_dir = _optional_path(self.project_dir_edit.text())
        if project_dir is None and frames_dir is not None:
            project_dir = _suggest_project_dir(frames_dir)
            self.project_dir_edit.setText(str(project_dir))
        if project_dir is None:
            raise ValueError(
                "Select a project folder before appending grouped partials."
            )
        solute_elements = _normalize_solute_text(
            self.solute_elements_edit.text()
        )
        self._item = replace(
            self._item,
            project_dir=project_dir,
            frames_dir=frames_dir,
            solute_elements=solute_elements,
        )
        self._refresh_header()
        self._refresh_project_reference()
        return DebyerPDFExistingPartialsJob(
            project_dir=project_dir,
            solute_elements=solute_elements,
        )

    def _build_ui(self) -> None:
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        self.header_frame = QFrame()
        self.header_frame.setObjectName("DebyerBatchItemHeader")
        header = QHBoxLayout(self.header_frame)
        header.setContentsMargins(8, 6, 8, 6)
        header.setSpacing(8)
        self.toggle_button = QToolButton()
        self.toggle_button.setCheckable(True)
        self.toggle_button.toggled.connect(self._set_settings_visible)
        header.addWidget(self.toggle_button)
        self.title_label = QLabel("New PDF calculation")
        self.title_label.setStyleSheet("font-weight: 600;")
        header.addWidget(self.title_label, stretch=1)
        self.status_label = QLabel("Ready")
        self.status_label.setMinimumWidth(180)
        header.addWidget(self.status_label)
        self.duplicate_button = QPushButton("Duplicate")
        self.duplicate_button.clicked.connect(
            lambda: self.duplicate_requested.emit(self.item_id)
        )
        header.addWidget(self.duplicate_button)
        self.remove_button = QPushButton("Remove")
        self.remove_button.clicked.connect(
            lambda: self.remove_requested.emit(self.item_id)
        )
        header.addWidget(self.remove_button)
        root.addWidget(self.header_frame)
        self.set_selected(False)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%v / %m frames")
        root.addWidget(self.progress_bar)

        self.settings_group = QGroupBox("Debyer Calculation Settings")
        root.addWidget(self.settings_group)
        form = QFormLayout(self.settings_group)

        project_row = QWidget()
        project_layout = QHBoxLayout(project_row)
        project_layout.setContentsMargins(0, 0, 0, 0)
        self.project_dir_edit = QLineEdit()
        self.project_dir_edit.editingFinished.connect(self._on_editor_changed)
        project_layout.addWidget(self.project_dir_edit, stretch=1)
        project_button = QPushButton("Browse...")
        project_button.clicked.connect(self._choose_project_dir)
        project_layout.addWidget(project_button)
        form.addRow("Project folder", project_row)

        self.project_reference_label = QLabel()
        self.project_reference_label.setWordWrap(True)
        self.project_reference_label.setFrameShape(QFrame.Shape.StyledPanel)
        form.addRow("", self.project_reference_label)

        frames_row = QWidget()
        frames_layout = QHBoxLayout(frames_row)
        frames_layout.setContentsMargins(0, 0, 0, 0)
        self.frames_dir_edit = QLineEdit()
        self.frames_dir_edit.editingFinished.connect(self._on_editor_changed)
        frames_layout.addWidget(self.frames_dir_edit, stretch=1)
        self.frames_button = QPushButton("Browse...")
        self.frames_button.clicked.connect(self._choose_frames_dir)
        frames_layout.addWidget(self.frames_button)
        form.addRow("XYZ frames folder", frames_row)

        self.filename_prefix_edit = QLineEdit("debyer_pdf")
        self.filename_prefix_edit.editingFinished.connect(
            self._on_editor_changed
        )
        form.addRow("Output prefix", self.filename_prefix_edit)

        self.mode_combo = QComboBox()
        for mode in SUPPORTED_DEBYER_MODES:
            self.mode_combo.addItem(mode)
        self.mode_combo.currentIndexChanged.connect(self._on_editor_changed)
        form.addRow("Mode", self.mode_combo)

        range_widget = QWidget()
        range_layout = QGridLayout(range_widget)
        range_layout.setContentsMargins(0, 0, 0, 0)
        self.from_edit = QLineEdit("0.5")
        self.to_edit = QLineEdit("15")
        self.step_edit = QLineEdit("0.01")
        for widget in (self.from_edit, self.to_edit, self.step_edit):
            widget.editingFinished.connect(self._on_editor_changed)
        range_layout.addWidget(QLabel("from"), 0, 0)
        range_layout.addWidget(self.from_edit, 0, 1)
        range_layout.addWidget(QLabel("to"), 0, 2)
        range_layout.addWidget(self.to_edit, 0, 3)
        range_layout.addWidget(QLabel("step"), 0, 4)
        range_layout.addWidget(self.step_edit, 0, 5)
        form.addRow("r-range (A)", range_widget)

        box_widget = QWidget()
        box_layout = QGridLayout(box_widget)
        box_layout.setContentsMargins(0, 0, 0, 0)
        self.box_a_edit = QLineEdit()
        self.box_b_edit = QLineEdit()
        self.box_c_edit = QLineEdit()
        for widget in (self.box_a_edit, self.box_b_edit, self.box_c_edit):
            widget.editingFinished.connect(self._on_editor_changed)
        box_layout.addWidget(QLabel("a"), 0, 0)
        box_layout.addWidget(self.box_a_edit, 0, 1)
        box_layout.addWidget(QLabel("b"), 0, 2)
        box_layout.addWidget(self.box_b_edit, 0, 3)
        box_layout.addWidget(QLabel("c"), 0, 4)
        box_layout.addWidget(self.box_c_edit, 0, 5)
        form.addRow("Bounding box (A)", box_widget)

        self.atom_count_edit = QLineEdit()
        self.atom_count_edit.editingFinished.connect(self._on_editor_changed)
        form.addRow("Atom count", self.atom_count_edit)
        self.rho0_label = QLabel(
            "rho0 will be computed from the atom count and box."
        )
        self.rho0_label.setWordWrap(True)
        form.addRow("", self.rho0_label)

        self.solute_elements_edit = QLineEdit()
        self.solute_elements_edit.setPlaceholderText("Optional, e.g. Pb, I")
        self.solute_elements_edit.setToolTip(
            "Defines solute atoms for grouped partial traces. In append "
            "mode, edit this value before running to rebuild the grouped "
            "columns with a different solute definition."
        )
        self.solute_elements_edit.editingFinished.connect(
            self._on_editor_changed
        )
        form.addRow("Solute elements", self.solute_elements_edit)

        self.store_frame_outputs_checkbox = QCheckBox(
            "Store per-frame Debyer output files"
        )
        self.store_frame_outputs_checkbox.toggled.connect(
            self._on_editor_changed
        )
        form.addRow("", self.store_frame_outputs_checkbox)

        self.parallel_jobs_spin = QSpinBox()
        self.parallel_jobs_spin.setRange(1, 64)
        self.parallel_jobs_spin.setValue(default_parallel_debyer_jobs())
        self.parallel_jobs_spin.valueChanged.connect(self._on_editor_changed)
        form.addRow("Parallel Debyer jobs", self.parallel_jobs_spin)
        self._full_calculation_widgets = (
            self.frames_dir_edit,
            self.frames_button,
            self.filename_prefix_edit,
            self.mode_combo,
            self.from_edit,
            self.to_edit,
            self.step_edit,
            self.box_a_edit,
            self.box_b_edit,
            self.box_c_edit,
            self.atom_count_edit,
            self.store_frame_outputs_checkbox,
            self.parallel_jobs_spin,
        )

    def _load_item(self, item: DebyerPDFBatchItem) -> None:
        self._loading = True
        self.project_dir_edit.setText(
            "" if item.project_dir is None else str(item.project_dir)
        )
        self.frames_dir_edit.setText(
            "" if item.frames_dir is None else str(item.frames_dir)
        )
        self.filename_prefix_edit.setText(item.filename_prefix)
        self.mode_combo.setCurrentText(item.mode)
        self.from_edit.setText(f"{item.from_value:g}")
        self.to_edit.setText(f"{item.to_value:g}")
        self.step_edit.setText(f"{item.step_value:g}")
        self.box_a_edit.setText(
            ""
            if item.box_dimensions[0] <= 0.0
            else f"{item.box_dimensions[0]:g}"
        )
        self.box_b_edit.setText(
            ""
            if item.box_dimensions[1] <= 0.0
            else f"{item.box_dimensions[1]:g}"
        )
        self.box_c_edit.setText(
            ""
            if item.box_dimensions[2] <= 0.0
            else f"{item.box_dimensions[2]:g}"
        )
        self.atom_count_edit.setText(
            "" if item.atom_count <= 0 else str(item.atom_count)
        )
        self.solute_elements_edit.setText(_solute_text(item.solute_elements))
        self.store_frame_outputs_checkbox.setChecked(item.store_frame_outputs)
        self.parallel_jobs_spin.setValue(item.max_parallel_jobs)
        self._loading = False
        self._refresh_header()
        self._refresh_project_reference()
        self._refresh_rho0_label()

    def _set_settings_visible(self, visible: bool) -> None:
        self.settings_group.setVisible(bool(visible))
        self.toggle_button.setChecked(bool(visible))
        self.toggle_button.setText("Hide Settings" if visible else "Settings")
        parent_item = self._list_item()
        if parent_item is not None:
            parent_item.setSizeHint(self.sizeHint())

    def _list_item(self) -> QListWidgetItem | None:
        parent = self.parent()
        while parent is not None and not isinstance(parent, QListWidget):
            parent = parent.parent()
        if not isinstance(parent, QListWidget):
            return None
        for row in range(parent.count()):
            list_item = parent.item(row)
            if parent.itemWidget(list_item) is self:
                return list_item
        return None

    def _choose_project_dir(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "Select SAXSShell project folder",
            self.project_dir_edit.text().strip() or str(Path.home()),
        )
        if not selected:
            return
        self.project_dir_edit.setText(selected)
        self._on_editor_changed()

    def _choose_frames_dir(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "Select XYZ frames folder",
            self.frames_dir_edit.text().strip()
            or self.project_dir_edit.text().strip()
            or str(Path.home()),
        )
        if not selected:
            return
        self.frames_dir_edit.setText(selected)
        self._on_editor_changed()

    def _on_editor_changed(self) -> None:
        if self._loading:
            return
        try:
            self.collect_item()
            self.status_label.setText("Ready")
        except Exception:
            self._refresh_project_reference()
            self._refresh_header()
            self._refresh_rho0_label()
        self.settings_changed.emit(self.item_id)

    def _refresh_header(self) -> None:
        self.title_label.setText(self._item.display_name())

    def _refresh_project_reference(self) -> None:
        project_dir = _optional_path(self.project_dir_edit.text())
        self.project_reference_label.setText(
            _project_reference_text(project_dir)
        )

    def _refresh_rho0_label(self) -> None:
        try:
            atom_count = int(float(self.atom_count_edit.text().strip()))
            box = (
                float(self.box_a_edit.text().strip()),
                float(self.box_b_edit.text().strip()),
                float(self.box_c_edit.text().strip()),
            )
            rho0 = calculate_number_density(atom_count, box)
        except Exception:
            self.rho0_label.setText(
                "rho0 will be computed from the atom count and box."
            )
            return
        self.rho0_label.setText(f"rho0 = {rho0:.6g} atoms/A^3")

    def _refresh_setting_widget_states(self) -> None:
        if not hasattr(self, "_full_calculation_widgets"):
            return
        enabled = not self._append_grouped_mode and not self._locked
        for widget in self._full_calculation_widgets:
            widget.setEnabled(enabled)


class DebyerPDFBatchWorker(QObject):
    item_started = Signal(str, int, int)
    item_progress = Signal(str, int, int, str)
    item_finished = Signal(str, object)
    item_failed = Signal(str, str)
    log = Signal(str)
    status = Signal(str)
    finished = Signal(object)
    failed = Signal(str, str)

    def __init__(
        self,
        queue_entries: list[tuple[str, DebyerPDFSettings]],
        *,
        debyer_executable: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.queue_entries = list(queue_entries)
        self.debyer_executable = debyer_executable
        self._cancel_requested = threading.Event()

    def request_cancel(self) -> None:
        self._cancel_requested.set()

    @Slot()
    def run(self) -> None:
        results: list[DebyerPDFCalculation] = []
        total_items = len(self.queue_entries)
        for index, (item_id, settings) in enumerate(
            self.queue_entries,
            start=1,
        ):
            if self._cancel_requested.is_set():
                self.log.emit("Batch queue stopped before the next project.")
                break
            label = settings.filename_prefix or settings.project_dir.name
            self.item_started.emit(item_id, index, total_items)
            self.status.emit(
                f"Running {index}/{total_items}: {settings.project_dir.name}"
            )
            self.log.emit(
                f"Starting {index}/{total_items}: {settings.project_dir}"
            )
            try:
                workflow = DebyerPDFWorkflow(
                    settings,
                    debyer_executable=self.debyer_executable,
                )
                result = workflow.run(
                    progress_callback=(
                        lambda processed, total, message, item_id=item_id: self.item_progress.emit(
                            item_id,
                            processed,
                            total,
                            message,
                        )
                    ),
                    log_callback=lambda message, label=label: self.log.emit(
                        f"[{label}] {message}"
                    ),
                    status_callback=lambda message, label=label: self.status.emit(
                        f"{label}: {message}"
                    ),
                    cancel_callback=self._cancel_requested.is_set,
                )
            except Exception as exc:
                message = str(exc)
                self.item_failed.emit(item_id, message)
                self.failed.emit(item_id, message)
                return
            results.append(result)
            self.item_finished.emit(item_id, result)
            if result.is_partial_average or self._cancel_requested.is_set():
                self.log.emit("Batch queue stopped after saving current work.")
                break
        self.status.emit("PDF batch queue finished")
        self.finished.emit(results)


class DebyerPDFExistingPartialsWorker(QObject):
    item_started = Signal(str, int, int)
    item_progress = Signal(str, int, int, str)
    item_finished = Signal(str, object)
    item_failed = Signal(str, str)
    log = Signal(str)
    status = Signal(str)
    finished = Signal(object)
    failed = Signal(str, str)

    def __init__(
        self,
        queue_entries: list[tuple[str, DebyerPDFExistingPartialsJob]],
    ) -> None:
        super().__init__()
        self.queue_entries = list(queue_entries)
        self._cancel_requested = threading.Event()

    def request_cancel(self) -> None:
        self._cancel_requested.set()

    @Slot()
    def run(self) -> None:
        results: list[DebyerPDFCalculation] = []
        total_items = len(self.queue_entries)
        for index, (item_id, job) in enumerate(
            self.queue_entries,
            start=1,
        ):
            if self._cancel_requested.is_set():
                self.log.emit("Batch queue stopped before the next project.")
                break
            self.item_started.emit(item_id, index, total_items)
            self.status.emit(
                f"Updating {index}/{total_items}: {job.project_dir.name}"
            )
            try:
                updated = self._update_project(item_id, job)
            except Exception as exc:
                message = str(exc)
                self.item_failed.emit(item_id, message)
                self.failed.emit(item_id, message)
                return
            results.extend(updated)
            self.item_finished.emit(item_id, updated)
        self.status.emit("Grouped partial column update finished")
        self.finished.emit(results)

    def _update_project(
        self,
        item_id: str,
        job: DebyerPDFExistingPartialsJob,
    ) -> list[DebyerPDFCalculation]:
        project_dir = Path(job.project_dir).expanduser().resolve()
        if not project_dir.is_dir():
            raise ValueError(
                f"The project folder does not exist: {project_dir}"
            )
        summaries = list_saved_debyer_calculations(project_dir)
        if not summaries:
            raise ValueError(
                "No saved Debyer calculations were found in " f"{project_dir}."
            )
        self.log.emit(
            f"Updating {len(summaries)} saved Debyer calculation(s) in "
            f"{project_dir}"
        )
        updated: list[DebyerPDFCalculation] = []
        total = len(summaries)
        for processed, summary in enumerate(summaries, start=1):
            if self._cancel_requested.is_set():
                self.log.emit(
                    f"Stopped before updating {summary.calculation_dir.name}."
                )
                break
            calculation = load_debyer_calculation(summary.calculation_dir)
            solute_elements = (
                job.solute_elements or calculation.solute_elements
            )
            if not solute_elements:
                raise ValueError(
                    "Solute elements are required to append grouped partial "
                    f"columns for {calculation.calculation_dir}."
                )
            calculation = replace(
                calculation,
                solute_elements=solute_elements,
                target_peak_markers={},
            )
            rewrite_debyer_calculation_output(calculation)
            write_debyer_calculation_metadata(calculation)
            updated.append(calculation)
            self.item_progress.emit(
                item_id,
                processed,
                total,
                f"Updated {processed}/{total}: {summary.filename_prefix}",
            )
            self.log.emit(
                "Appended grouped columns to "
                f"{calculation.averaged_output_file}"
            )
        return updated


class DebyerPDFBatchQueueWindow(QMainWindow):
    """Queue Debyer PDF calculations for multiple projects."""

    def __init__(
        self,
        initial_project_dir: str | Path | None = None,
        *,
        initial_frames_dir: str | Path | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._widgets_by_id: dict[str, DebyerPDFBatchItemWidget] = {}
        self._run_thread: QThread | None = None
        self._run_worker: (
            DebyerPDFBatchWorker | DebyerPDFExistingPartialsWorker | None
        ) = None
        self._initial_project_dir = (
            None
            if initial_project_dir is None
            else Path(initial_project_dir).expanduser().resolve()
        )
        self._initial_frames_dir = (
            None
            if initial_frames_dir is None
            else Path(initial_frames_dir).expanduser().resolve()
        )
        self._build_ui()
        self._refresh_runtime_status()
        if (
            self._initial_project_dir is not None
            or self._initial_frames_dir is not None
        ):
            initial_item = (
                _queue_item_from_project_defaults(
                    self._initial_project_dir,
                    frames_dir_override=self._initial_frames_dir,
                )
                if self._initial_project_dir is not None
                else DebyerPDFBatchItem(
                    item_id=_new_item_id(),
                    frames_dir=self._initial_frames_dir,
                    filename_prefix=(
                        self._initial_frames_dir.name
                        if self._initial_frames_dir is not None
                        else "debyer_pdf"
                    ),
                )
            )
            self.add_queue_item(initial_item)

    def closeEvent(self, event) -> None:
        if self._run_thread is not None and self._run_thread.isRunning():
            self._request_cancel()
            self.hide()
            while (
                self._run_thread is not None and self._run_thread.isRunning()
            ):
                QApplication.processEvents()
                if self._run_thread is not None:
                    self._run_thread.wait(50)
            event.accept()
            return
        super().closeEvent(event)

    def add_queue_item(
        self,
        item: DebyerPDFBatchItem | None = None,
    ) -> DebyerPDFBatchItemWidget:
        resolved_item = item or DebyerPDFBatchItem(item_id=_new_item_id())
        list_item = QListWidgetItem()
        list_item.setData(Qt.ItemDataRole.UserRole, resolved_item.item_id)
        self.queue_list.addItem(list_item)
        widget = DebyerPDFBatchItemWidget(
            resolved_item, parent=self.queue_list
        )
        widget.settings_changed.connect(self._on_item_settings_changed)
        widget.remove_requested.connect(self._remove_item)
        widget.duplicate_requested.connect(self._duplicate_item)
        self._widgets_by_id[resolved_item.item_id] = widget
        widget.set_append_grouped_mode(self._is_append_grouped_mode())
        list_item.setSizeHint(widget.sizeHint())
        self.queue_list.setItemWidget(list_item, widget)
        self.queue_list.setCurrentItem(list_item)
        self._refresh_order_labels()
        return widget

    def queue_settings_in_order(self) -> list[tuple[str, DebyerPDFSettings]]:
        entries: list[tuple[str, DebyerPDFSettings]] = []
        for row in range(self.queue_list.count()):
            list_item = self.queue_list.item(row)
            item_id = str(list_item.data(Qt.ItemDataRole.UserRole))
            widget = self._widgets_by_id[item_id]
            entries.append((item_id, widget.settings()))
        return entries

    def existing_partials_jobs_in_order(
        self,
    ) -> list[tuple[str, DebyerPDFExistingPartialsJob]]:
        entries: list[tuple[str, DebyerPDFExistingPartialsJob]] = []
        for row in range(self.queue_list.count()):
            list_item = self.queue_list.item(row)
            item_id = str(list_item.data(Qt.ItemDataRole.UserRole))
            widget = self._widgets_by_id[item_id]
            entries.append((item_id, widget.existing_partials_job()))
        return entries

    def _build_ui(self) -> None:
        self.setWindowTitle("SAXSShell PDF Batch Queue")
        self.setWindowIcon(load_saxshell_icon())
        self.resize(1120, 860)

        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        self.runtime_status_label = QLabel("Checking Debyer runtime...")
        self.runtime_status_label.setWordWrap(True)
        self.runtime_status_label.setFrameShape(QFrame.Shape.StyledPanel)
        root.addWidget(self.runtime_status_label)

        controls = QHBoxLayout()
        self.add_current_button = QPushButton("Add Current Project")
        self.add_current_button.clicked.connect(self._add_current_project)
        controls.addWidget(self.add_current_button)
        self.add_project_button = QPushButton("Add Projects...")
        self.add_project_button.clicked.connect(self._choose_project_to_add)
        controls.addWidget(self.add_project_button)
        self.add_frames_button = QPushButton("Add XYZ Frame Folders...")
        self.add_frames_button.clicked.connect(self._choose_frames_to_add)
        controls.addWidget(self.add_frames_button)
        controls.addStretch(1)
        root.addLayout(controls)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Queue mode"))
        self.queue_mode_combo = QComboBox()
        self.queue_mode_combo.addItem(
            "Run full Debyer calculations",
            "calculate",
        )
        self.queue_mode_combo.addItem(
            "Append grouped partial columns only",
            "append_grouped",
        )
        self.queue_mode_combo.currentIndexChanged.connect(
            self._on_queue_mode_changed
        )
        mode_row.addWidget(self.queue_mode_combo)
        self.queue_mode_status_label = QLabel(
            "Runs Debyer for each queue item in order."
        )
        self.queue_mode_status_label.setWordWrap(True)
        mode_row.addWidget(self.queue_mode_status_label, stretch=1)
        root.addLayout(mode_row)

        self.queue_list = QListWidget()
        self.queue_list.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.queue_list.setDragDropMode(
            QAbstractItemView.DragDropMode.InternalMove
        )
        self.queue_list.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.queue_list.setAlternatingRowColors(True)
        self.queue_list.setStyleSheet(
            "QListWidget::item:selected { background: transparent; }"
            "QListWidget::item:hover { background: transparent; }"
            "QListWidget::item { margin: 3px; }"
        )
        self.queue_list.model().rowsMoved.connect(self._refresh_order_labels)
        self.queue_list.itemSelectionChanged.connect(
            self._refresh_item_selection_styles
        )
        root.addWidget(self.queue_list, stretch=1)

        run_group = QGroupBox("Execute Queue")
        run_layout = QVBoxLayout(run_group)
        run_buttons = QHBoxLayout()
        self.run_button = QPushButton("Run Complete Queue")
        self.run_button.clicked.connect(self._start_queue)
        run_buttons.addWidget(self.run_button)
        self.cancel_button = QPushButton("Stop Queue")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._request_cancel)
        run_buttons.addWidget(self.cancel_button)
        run_buttons.addStretch(1)
        run_layout.addLayout(run_buttons)
        self.queue_status_label = QLabel("Queue idle")
        run_layout.addWidget(self.queue_status_label)
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setMinimumHeight(160)
        run_layout.addWidget(self.console)
        root.addWidget(run_group)

        self.setCentralWidget(central)
        self.statusBar().showMessage("Ready")
        self._on_queue_mode_changed()

    def _refresh_runtime_status(self) -> None:
        status = check_debyer_runtime()
        self.runtime_status_label.setText(status.message)

    def _is_append_grouped_mode(self) -> bool:
        return self.queue_mode_combo.currentData() == "append_grouped"

    def _on_queue_mode_changed(self, *_args) -> None:
        append_mode = self._is_append_grouped_mode()
        running = self._run_thread is not None and self._run_thread.isRunning()
        self.run_button.setText(
            "Append Grouped Partial Columns"
            if append_mode
            else "Run Complete Queue"
        )
        self.queue_mode_status_label.setText(
            "Updates existing saved Debyer calculations in each project. "
            "Debyer is not launched; project folder and solute elements are "
            "used."
            if append_mode
            else "Runs Debyer for each queue item in order."
        )
        self.add_frames_button.setEnabled(not append_mode and not running)
        for widget in self._widgets_by_id.values():
            widget.set_append_grouped_mode(append_mode)

    def _add_current_project(self) -> None:
        if (
            self._initial_project_dir is None
            and self._initial_frames_dir is None
        ):
            QMessageBox.information(
                self,
                "No active project",
                "The main UI did not provide an active project reference.",
            )
            return
        self.add_queue_item(
            (
                _queue_item_from_project_defaults(
                    self._initial_project_dir,
                    frames_dir_override=self._initial_frames_dir,
                )
                if self._initial_project_dir is not None
                else DebyerPDFBatchItem(
                    item_id=_new_item_id(),
                    frames_dir=self._initial_frames_dir,
                    filename_prefix=(
                        self._initial_frames_dir.name
                        if self._initial_frames_dir is not None
                        else "debyer_pdf"
                    ),
                )
            )
        )

    def _choose_project_to_add(self) -> None:
        selected_dirs = _choose_existing_directories(
            self,
            title="Select SAXSShell project folders",
            start_dir=self._initial_project_dir or Path.home(),
        )
        if not selected_dirs:
            return
        progress_dialog = self._project_load_progress_dialog(
            len(selected_dirs)
        )
        try:
            for index, project_dir in enumerate(selected_dirs, start=1):
                if progress_dialog is not None:
                    progress_dialog.setLabelText(
                        "Loading PDF project "
                        f"{index}/{len(selected_dirs)}:\n{project_dir}"
                    )
                    progress_dialog.setValue(index - 1)
                    QApplication.processEvents()
                item = _queue_item_from_project_defaults(project_dir)
                # Saved projects can reference very large frame folders; keep
                # project loading fast and rely on saved or manual settings.
                widget = self.add_queue_item(item)
                widget.set_status("Project loaded")
                if progress_dialog is not None:
                    progress_dialog.setValue(index)
                    QApplication.processEvents()
        finally:
            if progress_dialog is not None:
                progress_dialog.setValue(len(selected_dirs))
                progress_dialog.close()

    def _project_load_progress_dialog(
        self,
        project_count: int,
    ) -> QProgressDialog | None:
        if project_count <= 1:
            return None
        dialog = QProgressDialog(
            "Loading selected PDF projects...",
            None,
            0,
            project_count,
            self,
        )
        dialog.setWindowTitle("Loading PDF Projects")
        dialog.setWindowModality(Qt.WindowModality.WindowModal)
        dialog.setMinimumDuration(0)
        dialog.setAutoClose(False)
        dialog.setAutoReset(False)
        dialog.setValue(0)
        dialog.show()
        QApplication.processEvents()
        return dialog

    def _choose_frames_to_add(self) -> None:
        selected_dirs = _choose_existing_directories(
            self,
            title="Select XYZ frames folders",
            start_dir=(
                self._initial_frames_dir
                or self._initial_project_dir
                or Path.home()
            ),
        )
        if not selected_dirs:
            return
        for frames_dir in selected_dirs:
            self.add_queue_item(
                DebyerPDFBatchItem(
                    item_id=_new_item_id(),
                    project_dir=self._initial_project_dir,
                    frames_dir=frames_dir,
                    filename_prefix=frames_dir.name,
                )
            )

    def _on_item_settings_changed(self, _item_id: str) -> None:
        self._refresh_order_labels()

    def _refresh_order_labels(self, *_args) -> None:
        for row in range(self.queue_list.count()):
            list_item = self.queue_list.item(row)
            item_id = str(list_item.data(Qt.ItemDataRole.UserRole))
            widget = self._widgets_by_id.get(item_id)
            if widget is None:
                continue
            widget.title_label.setText(
                f"{row + 1}. {widget.item().display_name()}"
            )
            list_item.setSizeHint(widget.sizeHint())
        self._refresh_item_selection_styles()

    def _refresh_item_selection_styles(self) -> None:
        for row in range(self.queue_list.count()):
            list_item = self.queue_list.item(row)
            item_id = str(list_item.data(Qt.ItemDataRole.UserRole))
            widget = self._widgets_by_id.get(item_id)
            if widget is not None:
                widget.set_selected(list_item.isSelected())

    def _remove_item(self, item_id: str) -> None:
        if self._run_thread is not None and self._run_thread.isRunning():
            return
        for row in range(self.queue_list.count()):
            list_item = self.queue_list.item(row)
            if str(list_item.data(Qt.ItemDataRole.UserRole)) == item_id:
                self.queue_list.takeItem(row)
                break
        self._widgets_by_id.pop(item_id, None)
        self._refresh_order_labels()

    def _duplicate_item(self, item_id: str) -> None:
        widget = self._widgets_by_id.get(item_id)
        if widget is None:
            return
        try:
            item = widget.collect_item()
        except Exception:
            item = widget.item()
        self.add_queue_item(
            replace(
                item,
                item_id=_new_item_id(),
                filename_prefix=f"{item.filename_prefix}_copy",
            )
        )

    def _set_running(self, running: bool) -> None:
        self.add_current_button.setEnabled(not running)
        self.add_project_button.setEnabled(not running)
        self.add_frames_button.setEnabled(
            not running and not self._is_append_grouped_mode()
        )
        self.queue_mode_combo.setEnabled(not running)
        self.run_button.setEnabled(not running)
        self.cancel_button.setEnabled(running)
        self.queue_list.setDragEnabled(not running)
        self.queue_list.setAcceptDrops(not running)
        for widget in self._widgets_by_id.values():
            widget.set_locked(running)

    def _start_queue(self) -> None:
        if self.queue_list.count() == 0:
            QMessageBox.information(
                self,
                "PDF batch queue",
                "Add at least one project before running the queue.",
            )
            return
        append_mode = self._is_append_grouped_mode()
        try:
            entries = (
                self.existing_partials_jobs_in_order()
                if append_mode
                else self.queue_settings_in_order()
            )
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Invalid PDF batch settings",
                str(exc),
            )
            return

        if not append_mode:
            for _item_id, settings in entries:
                settings.project_dir.mkdir(parents=True, exist_ok=True)
        self.console.clear()
        self._set_running(True)
        self.queue_status_label.setText(
            (
                f"Updating 0/{len(entries)} queued project(s)"
                if append_mode
                else f"Running 0/{len(entries)} queued calculations"
            )
        )
        for widget in self._widgets_by_id.values():
            widget.set_progress(0, 1)
            widget.set_status("Queued")

        self._run_thread = QThread(self)
        self._run_worker = (
            DebyerPDFExistingPartialsWorker(entries)
            if append_mode
            else DebyerPDFBatchWorker(entries)
        )
        self._run_worker.moveToThread(self._run_thread)
        self._run_thread.started.connect(self._run_worker.run)
        self._run_worker.item_started.connect(self._on_item_started)
        self._run_worker.item_progress.connect(self._on_item_progress)
        self._run_worker.item_finished.connect(self._on_item_finished)
        self._run_worker.item_failed.connect(self._on_item_failed)
        self._run_worker.log.connect(self._append_log)
        self._run_worker.status.connect(self._on_status)
        self._run_worker.finished.connect(self._on_queue_finished)
        self._run_worker.failed.connect(self._on_queue_failed)
        self._run_worker.finished.connect(self._run_thread.quit)
        self._run_worker.failed.connect(self._run_thread.quit)
        self._run_thread.finished.connect(self._cleanup_run_thread)
        self._run_thread.finished.connect(self._run_thread.deleteLater)
        self._run_thread.start()

    def _request_cancel(self) -> None:
        self.cancel_button.setEnabled(False)
        self.queue_status_label.setText(
            "Stopping queue after the active project finishes"
        )
        self._append_log(
            "Stop requested; the current project will finish before the "
            "queue exits."
        )
        if self._run_worker is not None:
            self._run_worker.request_cancel()

    def _append_log(self, message: str) -> None:
        self.console.append(message)

    def _on_status(self, message: str) -> None:
        self.statusBar().showMessage(message)
        self.queue_status_label.setText(message)

    def _on_item_started(
        self,
        item_id: str,
        index: int,
        total: int,
    ) -> None:
        widget = self._widgets_by_id.get(item_id)
        if widget is not None:
            widget.set_status(
                f"Updating {index}/{total}"
                if self._is_append_grouped_mode()
                else f"Running {index}/{total}"
            )
            widget.set_progress(0, 1)
        self.queue_status_label.setText(
            (
                f"Updating {index}/{total} queued project(s)"
                if self._is_append_grouped_mode()
                else f"Running {index}/{total} queued calculations"
            )
        )

    def _on_item_progress(
        self,
        item_id: str,
        processed: int,
        total: int,
        message: str,
    ) -> None:
        widget = self._widgets_by_id.get(item_id)
        if widget is not None:
            widget.set_progress(processed, total)
            widget.set_status(message)

    def _on_item_finished(self, item_id: str, result: object) -> None:
        widget = self._widgets_by_id.get(item_id)
        if widget is None:
            return
        if isinstance(result, DebyerPDFCalculation):
            processed = (
                result.frame_count
                if result.processed_frame_count is None
                else result.processed_frame_count
            )
            widget.set_progress(processed, result.frame_count)
            widget.set_status(
                "Stopped early" if result.is_partial_average else "Complete"
            )
        elif isinstance(result, list):
            widget.set_progress(len(result), max(len(result), 1))
            widget.set_status(f"Updated {len(result)} calculation(s)")
        else:
            widget.set_status("Complete")

    def _on_item_failed(self, item_id: str, message: str) -> None:
        widget = self._widgets_by_id.get(item_id)
        if widget is not None:
            widget.set_status("Failed")
        self._append_log(message)

    def _on_queue_finished(self, results: object) -> None:
        self._set_running(False)
        result_count = len(results) if isinstance(results, list) else 0
        self.queue_status_label.setText(
            (
                f"Queue finished: {result_count} calculation(s) updated"
                if self._is_append_grouped_mode()
                else f"Queue finished: {result_count} calculation(s) saved"
            )
        )
        self.statusBar().showMessage("PDF batch queue finished")

    def _on_queue_failed(self, item_id: str, message: str) -> None:
        self._set_running(False)
        self.queue_status_label.setText("Queue stopped after a failure")
        self.statusBar().showMessage("PDF batch queue failed", 5000)
        QMessageBox.warning(
            self,
            "PDF batch queue failed",
            f"Queue item {item_id} failed:\n{message}",
        )

    def _cleanup_run_thread(self) -> None:
        self._run_thread = None
        self._run_worker = None


def launch_debyer_pdf_batch_queue_ui(
    initial_project_dir: str | Path | None = None,
    *,
    initial_frames_dir: str | Path | None = None,
) -> int:
    app = QApplication.instance()
    if app is None:
        prepare_saxshell_application_identity()
        app = QApplication([])
    configure_saxshell_application(app)
    window = DebyerPDFBatchQueueWindow(
        initial_project_dir=initial_project_dir,
        initial_frames_dir=initial_frames_dir,
    )
    window.show()
    return int(app.exec())


__all__ = [
    "DebyerPDFExistingPartialsJob",
    "DebyerPDFExistingPartialsWorker",
    "DebyerPDFBatchItem",
    "DebyerPDFBatchItemWidget",
    "DebyerPDFBatchQueueWindow",
    "DebyerPDFBatchWorker",
    "launch_debyer_pdf_batch_queue_ui",
]
