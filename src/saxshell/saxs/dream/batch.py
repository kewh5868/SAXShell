from __future__ import annotations

import json
import os
import re
import shutil
import signal
import textwrap
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from saxshell.saxs.dream.distributions import (
    DreamParameterEntry,
    distribution_guide_bounds,
    format_distribution_guide_value,
    format_prior_preset_summary,
    guide_clip_status,
    guide_clip_status_label,
    load_parameter_map,
    prior_preset_status_label,
    save_parameter_map,
)
from saxshell.saxs.dream.results import (
    SAXSDreamResultsLoader,
    dream_fit_q_bounds,
    dream_output_q_bounds,
    format_dream_q_bounds,
)
from saxshell.saxs.dream.runtime import DreamRunBundle, SAXSDreamWorkflow
from saxshell.saxs.dream.settings import (
    DreamRunSettings,
    PosteriorFilterSettings,
    format_dream_search_filter_preset,
    load_dream_settings,
    save_dream_settings,
)
from saxshell.saxs.prefit.workflow import PrefitParameterEntry
from saxshell.saxs.project_manager import (
    SAXSProjectManager,
    project_artifact_paths,
)

DREAM_BATCH_MANIFEST_NAME = "dream_backend_run_set.json"
DREAM_BATCH_SCRIPT_NAME = "run_dream_batch.sh"
DREAM_BATCH_COMMANDS_NAME = "dream_batch_commands.txt"
DREAM_BATCH_LOG_NAME = "dream_batch_session.log"
DREAM_BATCH_PID_NAME = "dream_batch.pid"
DREAM_BATCH_STATUS_NAME = "dream_batch_status.json"
DREAM_BATCH_COMPARISON_NAME = "dream_batch_filter_comparison.txt"
DREAM_BATCH_FIT_REPORT_PDF_NAME = "dream_batch_fit_report.pdf"
_DISTRIBUTION_PARAM_KEY_ORDER = ("loc", "scale", "s")
_PDF_TOC_INSERT_INDEX = 1
_PDF_TOC_ROWS_PER_PAGE = 24
_GUIDE_LOW_CLIP_COLOR = "#dbeafe"
_GUIDE_HIGH_CLIP_COLOR = "#ffedd5"
_FIT_Q_RANGE_SHADE_COLOR = "#fef3c7"
DEFAULT_DREAM_BATCH_CONDA_ENV = "saxshell-py312"


def _timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _print_line(message: object) -> None:
    print(message, flush=True)


@dataclass(slots=True)
class DreamBatchQueueItem:
    item_id: str
    label: str
    run_dir: str
    runtime_script_path: str
    metadata_path: str
    settings_path: str
    parameter_map_path: str
    created_at: str
    status: str = "queued"
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "DreamBatchQueueItem":
        return cls(
            item_id=str(payload.get("item_id", "")).strip(),
            label=str(payload.get("label", "")).strip(),
            run_dir=str(payload.get("run_dir", "")).strip(),
            runtime_script_path=str(
                payload.get("runtime_script_path", "")
            ).strip(),
            metadata_path=str(payload.get("metadata_path", "")).strip(),
            settings_path=str(payload.get("settings_path", "")).strip(),
            parameter_map_path=str(
                payload.get("parameter_map_path", "")
            ).strip(),
            created_at=str(payload.get("created_at", "")).strip(),
            status=str(payload.get("status", "queued")).strip() or "queued",
            started_at=_optional_text(payload.get("started_at")),
            finished_at=_optional_text(payload.get("finished_at")),
            error=_optional_text(payload.get("error")),
        )

    def bundle(self) -> DreamRunBundle:
        return DreamRunBundle(
            run_dir=Path(self.run_dir).expanduser().resolve(),
            runtime_script_path=Path(self.runtime_script_path)
            .expanduser()
            .resolve(),
            metadata_path=Path(self.metadata_path).expanduser().resolve(),
            settings_path=Path(self.settings_path).expanduser().resolve(),
            parameter_map_path=Path(self.parameter_map_path)
            .expanduser()
            .resolve(),
        )


@dataclass(slots=True)
class DreamBatchFilterSet:
    filter_id: str
    label: str
    settings: PosteriorFilterSettings
    created_at: str

    def to_dict(self) -> dict[str, object]:
        return {
            "filter_id": self.filter_id,
            "label": self.label,
            "posterior_filter_settings": self.settings.to_dict(),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "DreamBatchFilterSet":
        settings_payload = payload.get("posterior_filter_settings")
        if not isinstance(settings_payload, dict):
            settings_payload = payload.get("settings", {})
        if not isinstance(settings_payload, dict):
            settings_payload = {}
        return cls(
            filter_id=str(payload.get("filter_id", "")).strip(),
            label=str(payload.get("label", "")).strip(),
            settings=PosteriorFilterSettings.from_dict(dict(settings_payload)),
            created_at=str(payload.get("created_at", "")).strip(),
        )


@dataclass(slots=True)
class _PdfTocEntry:
    section: str
    label: str
    detail: str
    page_number: int


@dataclass(frozen=True, slots=True)
class _PdfTableCell:
    text: object
    fill_color: str | None = None


class _PdfReportBuffer:
    def __init__(self) -> None:
        self.figures: list[Figure] = []

    def savefig(self, figure: Figure) -> None:
        self.figures.append(figure)

    def get_pagecount(self) -> int:
        return len(self.figures)


def _fit_filter_report_index(
    fit_index: int | None,
    filter_index: int | None,
) -> str:
    if fit_index is None or filter_index is None:
        return ""
    return f"{int(fit_index)}.{int(filter_index)}"


def _pdf_indexed_title(title: str, index_label: object | None) -> str:
    text = str(index_label or "").strip()
    if not text:
        return title
    return f"[{text}] {title}"


@dataclass(slots=True)
class DreamBatchRunSet:
    project_dir: str
    run_set_dir: str
    label: str
    conda_env: str = DEFAULT_DREAM_BATCH_CONDA_ENV
    created_at: str = field(default_factory=_timestamp)
    updated_at: str = field(default_factory=_timestamp)
    queue_items: list[DreamBatchQueueItem] = field(default_factory=list)
    filter_sets: list[DreamBatchFilterSet] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "format_version": 1,
            "project_dir": self.project_dir,
            "run_set_dir": self.run_set_dir,
            "label": self.label,
            "conda_env": self.conda_env,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "queue_items": [item.to_dict() for item in self.queue_items],
            "filter_sets": [
                filter_set.to_dict() for filter_set in self.filter_sets
            ],
            "script_path": str(self.script_path),
            "commands_path": str(self.commands_path),
            "log_path": str(self.log_path),
            "pid_path": str(self.pid_path),
            "status_path": str(self.status_path),
            "comparison_report_path": str(self.comparison_report_path),
            "fit_report_pdf_path": str(self.fit_report_pdf_path),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "DreamBatchRunSet":
        return cls(
            project_dir=str(payload.get("project_dir", "")).strip(),
            run_set_dir=str(payload.get("run_set_dir", "")).strip(),
            label=str(payload.get("label", "")).strip() or "DREAM batch",
            conda_env=str(
                payload.get("conda_env", DEFAULT_DREAM_BATCH_CONDA_ENV)
            ).strip()
            or DEFAULT_DREAM_BATCH_CONDA_ENV,
            created_at=str(payload.get("created_at", "")).strip()
            or _timestamp(),
            updated_at=str(payload.get("updated_at", "")).strip()
            or _timestamp(),
            queue_items=[
                DreamBatchQueueItem.from_dict(dict(item))
                for item in payload.get("queue_items", [])
                if isinstance(item, dict)
            ],
            filter_sets=[
                DreamBatchFilterSet.from_dict(dict(item))
                for item in payload.get("filter_sets", [])
                if isinstance(item, dict)
            ],
        )

    @property
    def resolved_project_dir(self) -> Path:
        return Path(self.project_dir).expanduser().resolve()

    @property
    def resolved_run_set_dir(self) -> Path:
        return Path(self.run_set_dir).expanduser().resolve()

    @property
    def manifest_path(self) -> Path:
        return self.resolved_run_set_dir / DREAM_BATCH_MANIFEST_NAME

    @property
    def script_path(self) -> Path:
        return self.resolved_run_set_dir / DREAM_BATCH_SCRIPT_NAME

    @property
    def commands_path(self) -> Path:
        return self.resolved_run_set_dir / DREAM_BATCH_COMMANDS_NAME

    @property
    def log_path(self) -> Path:
        return self.resolved_run_set_dir / DREAM_BATCH_LOG_NAME

    @property
    def pid_path(self) -> Path:
        return self.resolved_run_set_dir / DREAM_BATCH_PID_NAME

    @property
    def status_path(self) -> Path:
        return self.resolved_run_set_dir / DREAM_BATCH_STATUS_NAME

    @property
    def comparison_report_path(self) -> Path:
        return self.resolved_run_set_dir / DREAM_BATCH_COMPARISON_NAME

    @property
    def fit_report_pdf_path(self) -> Path:
        return self.resolved_run_set_dir / DREAM_BATCH_FIT_REPORT_PDF_NAME


class DreamBatchRunSetManager:
    """Create project-local DREAM batch manifests and runtime
    bundles."""

    def __init__(
        self,
        project_dir: str | Path,
        *,
        run_set_dir: str | Path | None = None,
        label: str | None = None,
        conda_env: str = DEFAULT_DREAM_BATCH_CONDA_ENV,
        reuse_latest: bool = False,
    ) -> None:
        self.project_dir = Path(project_dir).expanduser().resolve()
        self.workflow = SAXSDreamWorkflow(self.project_dir)
        self._validate_project_ready()
        if run_set_dir is not None:
            resolved_run_set_dir = Path(run_set_dir).expanduser().resolve()
        elif reuse_latest and label is None:
            resolved_run_set_dir = self._latest_run_set_dir()
        else:
            resolved_run_set_dir = None
        if resolved_run_set_dir is None:
            resolved_run_set_dir = self._new_run_set_dir(label)
        resolved_run_set_dir.mkdir(parents=True, exist_ok=True)
        self.run_set = self._load_or_create_run_set(
            resolved_run_set_dir,
            label=label,
            conda_env=conda_env,
        )
        self.save_manifest()

    def _run_set_root(self) -> Path:
        settings = SAXSProjectManager().load_project(self.project_dir)
        artifacts = project_artifact_paths(settings)
        return artifacts.dream_dir / "backend_run_sets"

    def _validate_project_ready(self) -> None:
        if not self.workflow.prefit_workflow.can_run_prefit():
            raise ValueError(
                "DREAM batch setup requires an active SAXS Prefit with "
                "experimental data. Save or run a SAXS Prefit for the active "
                "project before creating a backend run set."
            )
        if not self.workflow.prefit_workflow.current_prefit_state_exists():
            raise ValueError(
                "DREAM batch setup requires a saved active SAXS Prefit state. "
                "Click Save Fit in SAXS Prefit, or open this setup from the "
                "main UI so the active Prefit settings can be pushed first."
            )

    def _new_run_set_dir(self, label: str | None) -> Path:
        root = self._run_set_root()
        root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = _sanitize_name(label or f"dream_batch_{stamp}")
        candidate = root / f"{stamp}_{name}"
        index = 2
        while candidate.exists():
            candidate = root / f"{stamp}_{name}_{index:02d}"
            index += 1
        return candidate

    def _latest_run_set_dir(self) -> Path | None:
        root = self._run_set_root()
        if not root.is_dir():
            return None
        candidates: list[tuple[float, str, Path]] = []
        for manifest_path in root.glob(f"*/{DREAM_BATCH_MANIFEST_NAME}"):
            try:
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
                if not isinstance(payload, dict):
                    continue
                manifest_project_dir = (
                    Path(str(payload.get("project_dir", "")))
                    .expanduser()
                    .resolve()
                )
                if manifest_project_dir != self.project_dir:
                    continue
                updated_key = manifest_path.stat().st_mtime
            except Exception:
                continue
            candidates.append(
                (
                    updated_key,
                    manifest_path.parent.name,
                    manifest_path.parent,
                )
            )
        if not candidates:
            return None
        return max(candidates)[2]

    def _load_or_create_run_set(
        self,
        run_set_dir: Path,
        *,
        label: str | None,
        conda_env: str,
    ) -> DreamBatchRunSet:
        manifest_path = run_set_dir / DREAM_BATCH_MANIFEST_NAME
        if manifest_path.is_file():
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError(
                    f"DREAM batch manifest is not a JSON object: {manifest_path}"
                )
            return DreamBatchRunSet.from_dict(payload)
        return DreamBatchRunSet(
            project_dir=str(self.project_dir),
            run_set_dir=str(run_set_dir),
            label=str(label or run_set_dir.name),
            conda_env=str(conda_env or DEFAULT_DREAM_BATCH_CONDA_ENV),
        )

    def save_manifest(self) -> Path:
        self.run_set.updated_at = _timestamp()
        self.run_set.resolved_run_set_dir.mkdir(parents=True, exist_ok=True)
        self.run_set.manifest_path.write_text(
            json.dumps(self.run_set.to_dict(), indent=2) + "\n",
            encoding="utf-8",
        )
        return self.run_set.manifest_path

    def add_queue_item(
        self,
        *,
        label: str,
        settings: DreamRunSettings,
        entries: list[DreamParameterEntry],
        prefit_parameter_entries: list[PrefitParameterEntry] | None = None,
    ) -> DreamBatchQueueItem:
        item_label = label.strip() or self.next_queue_item_label()
        active_settings = DreamRunSettings.from_dict(settings.to_dict())
        active_settings.run_label = item_label
        bundle = self.workflow.create_runtime_bundle(
            settings=active_settings,
            entries=entries,
            prefit_parameter_entries=prefit_parameter_entries,
            include_posterior_filter_settings=False,
        )
        item = DreamBatchQueueItem(
            item_id=_new_id("run"),
            label=item_label,
            run_dir=str(bundle.run_dir),
            runtime_script_path=str(bundle.runtime_script_path),
            metadata_path=str(bundle.metadata_path),
            settings_path=str(bundle.settings_path),
            parameter_map_path=str(bundle.parameter_map_path),
            created_at=_timestamp(),
        )
        if prefit_parameter_entries is not None:
            _save_prefit_entries_for_queue_item(
                item,
                [
                    PrefitParameterEntry.from_dict(entry.to_dict())
                    for entry in prefit_parameter_entries
                ],
            )
        self.run_set.queue_items.append(item)
        self._discard_generated_launch_files()
        self.save_manifest()
        return item

    def remove_queue_item(
        self,
        item_id: str,
        *,
        delete_runtime_bundle: bool = True,
    ) -> DreamBatchQueueItem:
        item_id = str(item_id or "").strip()
        for index, item in enumerate(self.run_set.queue_items):
            if item.item_id == item_id:
                break
        else:
            raise ValueError(f"DREAM queue item not found: {item_id}")
        if item.status != "queued":
            raise ValueError(
                "Only queued DREAM batch items can be removed from the "
                f"queue. Current status for {item.label!r}: {item.status}."
            )
        if delete_runtime_bundle:
            self._delete_queue_item_runtime_bundle(item)
        removed = self.run_set.queue_items.pop(index)
        self._discard_generated_launch_files()
        self.save_manifest()
        return removed

    def update_queue_item_fit_range(
        self,
        item_id: str,
        *,
        fit_q_min: float | None,
        fit_q_max: float | None,
    ) -> DreamBatchQueueItem:
        item_id = str(item_id or "").strip()
        for item in self.run_set.queue_items:
            if item.item_id == item_id:
                break
        else:
            raise ValueError(f"DREAM queue item not found: {item_id}")
        if item.status != "queued":
            raise ValueError(
                "Only queued DREAM batch items can be edited. Current status "
                f"for {item.label!r}: {item.status}."
            )
        q_min = self._optional_fit_q_bound(fit_q_min, "fit q min")
        q_max = self._optional_fit_q_bound(fit_q_max, "fit q max")
        if q_min is not None and q_max is not None and q_min >= q_max:
            raise ValueError("DREAM fit q min must be less than fit q max.")

        settings = load_dream_settings(item.settings_path)
        settings.fit_q_min = q_min
        settings.fit_q_max = q_max
        entries = load_parameter_map_for_queue_item(item)
        active_settings = self.workflow._prepare_runtime_settings(
            settings,
            entries,
        )
        prefit_entries = _load_prefit_entries_for_queue_item(item)
        save_dream_settings(
            item.settings_path,
            active_settings,
            include_posterior_filter_settings=False,
        )
        metadata = self.workflow._build_runtime_metadata(
            active_settings,
            entries,
            prefit_parameter_entries=prefit_entries,
            include_posterior_filter_settings=False,
        )
        Path(item.metadata_path).write_text(
            json.dumps(metadata, indent=2) + "\n",
            encoding="utf-8",
        )
        self._discard_generated_launch_files()
        self.save_manifest()
        return item

    def update_queue_item_active_weights(
        self,
        item_id: str,
        *,
        active_weight_names: set[str],
    ) -> DreamBatchQueueItem:
        item_id = str(item_id or "").strip()
        for item in self.run_set.queue_items:
            if item.item_id == item_id:
                break
        else:
            raise ValueError(f"DREAM queue item not found: {item_id}")
        if item.status != "queued":
            raise ValueError(
                "Only queued DREAM batch items can be edited. Current status "
                f"for {item.label!r}: {item.status}."
            )
        prefit_entries = _load_prefit_entries_for_queue_item(item)
        if not prefit_entries:
            raise ValueError(
                "This queued DREAM run does not include an editable Prefit "
                "snapshot."
            )
        known_weight_names = {
            str(entry.name).strip()
            for entry in prefit_entries
            if _is_prefit_weight_entry(entry)
        }
        requested_names = {str(name).strip() for name in active_weight_names}
        requested_names.discard("")
        unknown_names = requested_names - known_weight_names
        if unknown_names:
            raise ValueError(
                "Unknown queued DREAM weight name(s): "
                f"{', '.join(sorted(unknown_names))}"
            )
        if not requested_names:
            raise ValueError(
                "At least one queued DREAM weight must remain on."
            )
        for entry in prefit_entries:
            if _is_prefit_weight_entry(entry):
                entry.active = str(entry.name).strip() in requested_names

        settings = load_dream_settings(item.settings_path)
        current_entries = load_parameter_map_for_queue_item(item)
        parameter_entries = self.workflow._normalize_parameter_map_entries(
            current_entries,
            prefit_parameter_entries=prefit_entries,
        )
        active_settings = self.workflow._prepare_runtime_settings(
            settings,
            parameter_entries,
        )
        _save_prefit_entries_for_queue_item(item, prefit_entries)
        save_parameter_map(item.parameter_map_path, parameter_entries)
        save_dream_settings(
            item.settings_path,
            active_settings,
            include_posterior_filter_settings=False,
        )
        metadata = self.workflow._build_runtime_metadata(
            active_settings,
            parameter_entries,
            prefit_parameter_entries=prefit_entries,
            include_posterior_filter_settings=False,
        )
        Path(item.metadata_path).write_text(
            json.dumps(metadata, indent=2) + "\n",
            encoding="utf-8",
        )
        self._discard_generated_launch_files()
        self.save_manifest()
        return item

    @staticmethod
    def _optional_fit_q_bound(
        value: float | None,
        label: str,
    ) -> float | None:
        if value is None:
            return None
        numeric_value = float(value)
        if not np.isfinite(numeric_value):
            raise ValueError(f"DREAM {label} must be finite.")
        return numeric_value

    def _delete_queue_item_runtime_bundle(
        self,
        item: DreamBatchQueueItem,
    ) -> None:
        run_dir = Path(item.run_dir).expanduser().resolve()
        if not run_dir.exists():
            return
        runtime_root = self.workflow.dream_runtime_dir.resolve()
        if runtime_root not in run_dir.parents:
            raise ValueError(
                "Refusing to delete DREAM queue runtime bundle outside the "
                f"project DREAM runtime folder: {run_dir}"
            )
        shutil.rmtree(run_dir)

    def _discard_generated_launch_files(self) -> None:
        for path in (self.run_set.script_path, self.run_set.commands_path):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass

    def next_queue_item_label(self) -> str:
        existing_labels = {
            item.label.strip()
            for item in self.run_set.queue_items
            if item.label.strip()
        }
        index = len(self.run_set.queue_items) + 1
        while True:
            label = f"DREAM queue item {index}"
            if label not in existing_labels:
                return label
            index += 1

    def add_filter_set(
        self,
        *,
        label: str,
        settings: PosteriorFilterSettings | DreamRunSettings,
    ) -> DreamBatchFilterSet:
        filter_label = label.strip() or (
            f"Posterior filter {len(self.run_set.filter_sets) + 1}"
        )
        filter_set = DreamBatchFilterSet(
            filter_id=_new_id("filter"),
            label=filter_label,
            settings=_coerce_posterior_filter_settings(settings),
            created_at=_timestamp(),
        )
        self.run_set.filter_sets.append(filter_set)
        self._discard_generated_launch_files()
        self.save_manifest()
        return filter_set

    def remove_filter_set(self, filter_id: str) -> DreamBatchFilterSet:
        filter_id = str(filter_id or "").strip()
        for index, filter_set in enumerate(self.run_set.filter_sets):
            if filter_set.filter_id == filter_id:
                break
        else:
            raise ValueError(
                f"DREAM posterior filter set not found: {filter_id}"
            )
        removed = self.run_set.filter_sets.pop(index)
        self._discard_generated_launch_files()
        self.save_manifest()
        return removed

    def update_filter_set(
        self,
        filter_id: str,
        *,
        label: str | None = None,
        settings: PosteriorFilterSettings | DreamRunSettings | None = None,
    ) -> DreamBatchFilterSet:
        filter_id = str(filter_id or "").strip()
        for filter_set in self.run_set.filter_sets:
            if filter_set.filter_id == filter_id:
                break
        else:
            raise ValueError(
                f"DREAM posterior filter set not found: {filter_id}"
            )
        if label is not None:
            filter_label = str(label).strip()
            filter_set.label = filter_label or filter_set.label
        if settings is not None:
            filter_set.settings = _coerce_posterior_filter_settings(settings)
        self._discard_generated_launch_files()
        self.save_manifest()
        return filter_set

    def replace_filter_sets(
        self,
        filter_sets: list[
            tuple[str, PosteriorFilterSettings | DreamRunSettings]
        ],
    ) -> list[DreamBatchFilterSet]:
        self.run_set.filter_sets = [
            DreamBatchFilterSet(
                filter_id=_new_id("filter"),
                label=(str(label).strip() or f"Posterior filter {index}"),
                settings=_coerce_posterior_filter_settings(settings),
                created_at=_timestamp(),
            )
            for index, (label, settings) in enumerate(filter_sets, start=1)
        ]
        self._discard_generated_launch_files()
        self.save_manifest()
        return list(self.run_set.filter_sets)

    def _rotate_to_new_run_set_dir(self) -> None:
        new_dir = self._new_run_set_dir(None)
        cloned_run_set = DreamBatchRunSet.from_dict(self.run_set.to_dict())
        timestamp = _timestamp()
        cloned_run_set.run_set_dir = str(new_dir)
        cloned_run_set.label = new_dir.name
        cloned_run_set.created_at = timestamp
        cloned_run_set.updated_at = timestamp
        self.run_set = cloned_run_set

    def generate_shell_script(
        self,
        *,
        new_run_set_dir: bool = False,
    ) -> tuple[Path, Path]:
        if new_run_set_dir:
            self._rotate_to_new_run_set_dir()
        self.save_manifest()
        commands_text = command_text_for_run_set(self.run_set)
        self.run_set.commands_path.write_text(commands_text, encoding="utf-8")
        self.run_set.script_path.write_text(
            shell_script_text_for_run_set(self.run_set),
            encoding="utf-8",
        )
        try:
            mode = self.run_set.script_path.stat().st_mode
            self.run_set.script_path.chmod(mode | 0o111)
        except OSError:
            pass
        self.save_manifest()
        return self.run_set.script_path, self.run_set.commands_path


def load_dream_batch_manifest(path: str | Path) -> DreamBatchRunSet:
    manifest_path = Path(path).expanduser().resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"DREAM batch manifest is not a JSON object: {path}")
    return DreamBatchRunSet.from_dict(payload)


def run_dream_batch_manifest(
    manifest_path: str | Path,
    *,
    output_callback=_print_line,
) -> DreamBatchRunSet:
    run_set = load_dream_batch_manifest(manifest_path)
    run_set.resolved_run_set_dir.mkdir(parents=True, exist_ok=True)
    commands_text = command_text_for_run_set(run_set)
    run_set.commands_path.write_text(commands_text, encoding="utf-8")
    if output_callback is not None:
        output_callback("Starting SAXSShell DREAM backend run set.")
        output_callback(commands_text.rstrip())

    _write_status(run_set, "running_dream")
    project_dir = run_set.resolved_project_dir
    current_item: DreamBatchQueueItem | None = None

    def handle_termination(signum: int, _frame: object) -> None:
        signal_name = _signal_name(signum)
        message = f"DREAM backend run set terminated by {signal_name}."
        if current_item is not None:
            current_item.status = "terminated"
            current_item.finished_at = _timestamp()
            current_item.error = message
            _save_run_set(run_set)
        _write_status(run_set, "terminated", error=message)
        if output_callback is not None:
            output_callback(message)
        raise SystemExit(128 + int(signum))

    handled_signals = (signal.SIGINT, signal.SIGTERM)
    previous_handlers = {
        signum: signal.getsignal(signum) for signum in handled_signals
    }
    for signum in handled_signals:
        signal.signal(signum, handle_termination)
    try:
        completed_items: list[DreamBatchQueueItem] = []
        failed_items: list[DreamBatchQueueItem] = []
        for index, item in enumerate(run_set.queue_items, start=1):
            current_item = item
            item.status = "running"
            item.started_at = _timestamp()
            item.error = None
            _save_run_set(run_set)
            if output_callback is not None:
                output_callback(
                    f"[{index}/{len(run_set.queue_items)}] Running {item.label}"
                )
                output_callback(f"Run directory: {item.run_dir}")
            try:
                workflow = SAXSDreamWorkflow(project_dir)
                workflow.run_bundle(
                    item.bundle(),
                    output_callback=output_callback,
                )
                item.status = "completed"
                item.finished_at = _timestamp()
                completed_items.append(item)
                if output_callback is not None:
                    output_callback(
                        f"Completed DREAM fit {item.label}; posterior "
                        "filters will run after all DREAM fits complete."
                    )
            except Exception as exc:
                item.status = "failed"
                item.finished_at = _timestamp()
                item.error = str(exc)
                failed_items.append(item)
                _save_run_set(run_set)
                if output_callback is not None:
                    output_callback(
                        f"DREAM fit failed for {item.label}: {exc}"
                    )
            _save_run_set(run_set)
            current_item = None

        _write_status(run_set, "filtering")
        if output_callback is not None:
            output_callback(
                "All DREAM fits have finished. Applying posterior filters "
                "and building the comparison report."
            )
        total_reports = 0
        for item in completed_items:
            reports = _write_filter_reports(
                item,
                run_set.filter_sets,
                output_callback=output_callback,
            )
            total_reports += len(reports)
        if output_callback is not None:
            output_callback(
                f"Posterior filter reports written: {total_reports}"
            )

        _write_batch_comparison_report(
            run_set,
            output_callback=output_callback,
        )
        _write_batch_fit_pdf_report(
            run_set,
            output_callback=output_callback,
        )
        if failed_items:
            error = f"{len(failed_items)} DREAM queue item(s) failed."
            _write_status(run_set, "completed_with_errors", error=error)
        else:
            _write_status(run_set, "completed")
        if output_callback is not None:
            if failed_items:
                output_callback(
                    "DREAM backend run set complete with "
                    f"{len(failed_items)} failed queue item(s)."
                )
            else:
                output_callback("DREAM backend run set complete.")
        return run_set
    finally:
        for signum, previous_handler in previous_handlers.items():
            signal.signal(signum, previous_handler)


def dream_filter_kwargs(
    settings: PosteriorFilterSettings | DreamRunSettings,
) -> dict[str, object]:
    return {
        "bestfit_method": settings.bestfit_method,
        "posterior_filter_mode": settings.posterior_filter_mode,
        "posterior_top_percent": settings.posterior_top_percent,
        "posterior_top_n": settings.posterior_top_n,
        "credible_interval_low": settings.credible_interval_low,
        "credible_interval_high": settings.credible_interval_high,
        "violin_parameter_mode": settings.violin_parameter_mode,
        "violin_sample_source": settings.violin_sample_source,
        "stoichiometry_target_elements_text": (
            settings.stoichiometry_target_elements_text
        ),
        "stoichiometry_target_ratio_text": (
            settings.stoichiometry_target_ratio_text
        ),
        "stoichiometry_filter_enabled": settings.stoichiometry_filter_enabled,
        "stoichiometry_tolerance_percent": (
            settings.stoichiometry_tolerance_percent
        ),
    }


def _summary_filter_kwargs(
    settings: PosteriorFilterSettings | DreamRunSettings,
) -> dict[str, object]:
    return {
        "bestfit_method": settings.bestfit_method,
        "posterior_filter_mode": settings.posterior_filter_mode,
        "posterior_top_percent": settings.posterior_top_percent,
        "posterior_top_n": settings.posterior_top_n,
        "credible_interval_low": settings.credible_interval_low,
        "credible_interval_high": settings.credible_interval_high,
        "stoichiometry_target_elements_text": (
            settings.stoichiometry_target_elements_text
        ),
        "stoichiometry_target_ratio_text": (
            settings.stoichiometry_target_ratio_text
        ),
        "stoichiometry_filter_enabled": settings.stoichiometry_filter_enabled,
        "stoichiometry_tolerance_percent": (
            settings.stoichiometry_tolerance_percent
        ),
    }


def _violin_filter_kwargs(
    settings: PosteriorFilterSettings | DreamRunSettings,
) -> dict[str, object]:
    kwargs = dict(_summary_filter_kwargs(settings))
    kwargs.pop("bestfit_method", None)
    kwargs.update(
        {
            "mode": settings.violin_parameter_mode,
            "sample_source": settings.violin_sample_source,
            "weight_order": settings.violin_weight_order,
            "selected_parameter": settings.violin_selected_parameter or None,
        }
    )
    return kwargs


def command_text_for_run_set(run_set: DreamBatchRunSet) -> str:
    script = _shell_quote(run_set.script_path)
    log_path = _shell_quote(run_set.log_path)
    pid_path = _shell_quote(run_set.pid_path)
    status_path = _shell_quote(run_set.status_path)
    project_dir = _shell_quote(run_set.resolved_project_dir)
    run_set_dir = _shell_quote(run_set.resolved_run_set_dir)
    conda_env = _shell_quote(run_set.conda_env)
    return "\n".join(
        [
            "# DREAM backend run set commands",
            "# Copy one section at a time into Terminal.",
            "",
            "# Start batch process:",
            f"cd {project_dir}",
            f"conda activate {conda_env}",
            f"cd {run_set_dir}",
            f"bash {script}",
            "",
            "# Watch progress:",
            f"tail -f {log_path}",
            "",
            "# Check current status:",
            f"cat {status_path}",
            "",
            "# Stop process:",
            f'PID="$(cat {pid_path} 2>/dev/null || true)"',
            (
                'if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then '
                'kill "$PID"; else echo "No live DREAM batch process for '
                'PID ${PID:-<missing>}"; '
                f"cat {status_path}; fi"
            ),
            "",
            "# Suspend process:",
            f'PID="$(cat {pid_path} 2>/dev/null || true)"',
            (
                'if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then '
                'kill -STOP "$PID"; else echo "No live DREAM batch process '
                'for PID ${PID:-<missing>}"; '
                f"cat {status_path}; fi"
            ),
            "",
            "# Resume process:",
            f'PID="$(cat {pid_path} 2>/dev/null || true)"',
            (
                'if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then '
                'kill -CONT "$PID"; else echo "No live DREAM batch process '
                'for PID ${PID:-<missing>}"; '
                f"cat {status_path}; fi"
            ),
            "",
            "# Comparison report:",
            f"printf '%s\\n' {_shell_quote(run_set.comparison_report_path)}",
            "",
            "# Fit report PDF:",
            f"printf '%s\\n' {_shell_quote(run_set.fit_report_pdf_path)}",
            "",
        ]
    )


def shell_script_text_for_run_set(run_set: DreamBatchRunSet) -> str:
    manifest = _shell_quote(run_set.manifest_path)
    log_path = _shell_quote(run_set.log_path)
    pid_path = _shell_quote(run_set.pid_path)
    commands_path = _shell_quote(run_set.commands_path)
    run_set_dir = _shell_quote(run_set.resolved_run_set_dir)
    project_dir = _shell_quote(run_set.resolved_project_dir)
    conda_env = _shell_quote(run_set.conda_env)
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            f"cd {project_dir}",
            "if command -v conda >/dev/null 2>&1; then",
            '  CONDA_BASE="$(conda info --base)"',
            "  # shellcheck disable=SC1091",
            '  source "$CONDA_BASE/etc/profile.d/conda.sh"',
            f"  conda activate {conda_env}",
            "fi",
            f"cd {run_set_dir}",
            f"python -m saxshell.saxs dream-batch run {manifest} "
            f"> {log_path} 2>&1 &",
            "BATCH_PID=$!",
            f'echo "$BATCH_PID" > {pid_path}',
            f"cat {commands_path}",
            'echo "Started DREAM backend run set with PID $BATCH_PID"',
            f'echo "Progress log: {run_set.log_path}"',
            f'echo "Comparison report: {run_set.comparison_report_path}"',
            f'echo "Fit report PDF: {run_set.fit_report_pdf_path}"',
            "",
        ]
    )


def _write_filter_reports(
    item: DreamBatchQueueItem,
    filter_sets: list[DreamBatchFilterSet],
    *,
    output_callback=_print_line,
) -> list[Path]:
    run_dir = Path(item.run_dir).expanduser().resolve()
    filter_sets = _filter_sets_for_item(item, filter_sets)
    loader = SAXSDreamResultsLoader(run_dir)
    reports: list[Path] = []
    for filter_set in filter_sets:
        report_name = (
            f"dream_statistics_{_sanitize_name(filter_set.label)}.txt"
        )
        report_path = run_dir / report_name
        try:
            loader.save_statistics_report(
                report_path,
                **dream_filter_kwargs(filter_set.settings),
            )
        except Exception as exc:
            if output_callback is not None:
                output_callback(
                    "Skipped filter set "
                    f"{filter_set.label} for {item.label}: {exc}"
                )
            continue
        reports.append(report_path)
        if output_callback is not None:
            output_callback(
                f"Applied filter set {filter_set.label}: {report_path}"
            )
    return reports


def _write_batch_comparison_report(
    run_set: DreamBatchRunSet,
    *,
    output_callback=_print_line,
) -> Path:
    report_path = run_set.comparison_report_path
    lines = [
        "SAXSShell DREAM backend run set comparison",
        f"Run set: {run_set.label}",
        f"Run set folder: {run_set.run_set_dir}",
        f"Project folder: {run_set.project_dir}",
        f"Generated: {_timestamp()}",
        f"Queued DREAM runs: {len(run_set.queue_items)}",
        (
            "Posterior filter sets: "
            f"{len(run_set.filter_sets) if run_set.filter_sets else 1}"
        ),
        "Report index convention: fit_index.filter_subindex",
        "",
        "Filter settings:",
    ]
    explicit_filter_sets = list(run_set.filter_sets)
    filter_sets_for_settings = explicit_filter_sets
    if not filter_sets_for_settings and run_set.queue_items:
        filter_sets_for_settings = _filter_sets_for_item(
            run_set.queue_items[0],
            [],
        )
    for filter_index, filter_set in enumerate(
        filter_sets_for_settings,
        start=1,
    ):
        settings = filter_set.settings
        lines.extend(
            [
                f"- Filter {filter_index}: {filter_set.label}",
                f"  Best-fit method: {settings.bestfit_method}",
                f"  Posterior filter: {settings.posterior_filter_mode}",
                f"  Top percent: {settings.posterior_top_percent:g}",
                f"  Top N: {settings.posterior_top_n}",
                (
                    "  Credible interval: "
                    f"{settings.credible_interval_low:g} - "
                    f"{settings.credible_interval_high:g}"
                ),
                (
                    "  Stoichiometry filter: "
                    + (
                        "on"
                        if settings.stoichiometry_filter_enabled
                        else "off"
                    )
                ),
            ]
        )
        if settings.stoichiometry_filter_enabled:
            lines.extend(
                [
                    (
                        "  Stoichiometry target: "
                        f"{settings.stoichiometry_target_elements_text} = "
                        f"{settings.stoichiometry_target_ratio_text}"
                    ),
                    (
                        "  Stoichiometry tolerance (%): "
                        f"{settings.stoichiometry_tolerance_percent:g}"
                    ),
                ]
            )

    if run_set.queue_items:
        lines.extend(["", "Queue presets:"])
        for fit_index, item in enumerate(run_set.queue_items, start=1):
            lines.extend(
                [
                    f"- Fit {fit_index}: {item.label}",
                    (
                        "  DREAM search preset: "
                        f"{_queue_item_search_preset_text(item)}"
                    ),
                    (
                        "  Prior presets: "
                        f"{_queue_item_prior_preset_summary_text(item)}"
                    ),
                ]
            )

    matrix_rows: list[dict[str, str]] = []
    parameter_rows: list[tuple[str, str, str, str, str, dict[str, float]]] = []
    guide_rows: list[dict[str, str]] = []
    parameter_names: list[str] = []
    for fit_index, item in enumerate(run_set.queue_items, start=1):
        item_search_preset = _queue_item_search_preset_text(item)
        item_prior_preset_summary = _queue_item_prior_preset_summary_text(item)
        prior_lookup = _parameter_map_lookup_for_queue_item(item)
        item_filter_sets = _filter_sets_for_item(item, explicit_filter_sets)
        for filter_index, filter_set in enumerate(
            item_filter_sets,
            start=1,
        ):
            report_index = _fit_filter_report_index(
                fit_index,
                filter_index,
            )
            row = {
                "report_index": report_index,
                "fit_index": str(fit_index),
                "filter_subindex": str(filter_index),
                "queue_item": item.label,
                "run_dir": item.run_dir,
                "dream_search_preset": item_search_preset,
                "prior_presets": item_prior_preset_summary,
                "filter_set": filter_set.label,
                "status": item.status,
                "posterior_filter": filter_set.settings.posterior_filter_mode,
                "bestfit_method": filter_set.settings.bestfit_method,
                "fit_q_range": "",
                "output_q_range": "",
                "candidate_samples": "",
                "kept_samples": "",
                "map_chain": "",
                "map_step": "",
                "stoichiometry": "",
                "error": item.error or "",
            }
            if item.status != "completed":
                matrix_rows.append(row)
                continue
            try:
                loader = SAXSDreamResultsLoader(item.run_dir)
                summary = loader.get_summary(
                    **_summary_filter_kwargs(filter_set.settings)
                )
            except Exception as exc:
                row["status"] = "report_failed"
                row["error"] = str(exc)
                matrix_rows.append(row)
                continue
            row.update(
                {
                    "status": "completed",
                    "fit_q_range": format_dream_q_bounds(
                        dream_fit_q_bounds(
                            loader.q_values,
                            loader.output_fit_mask,
                        )
                    ),
                    "output_q_range": format_dream_q_bounds(
                        dream_output_q_bounds(loader.q_values)
                    ),
                    "candidate_samples": str(
                        summary.posterior_candidate_sample_count
                    ),
                    "kept_samples": str(summary.posterior_sample_count),
                    "map_chain": str(summary.map_chain),
                    "map_step": str(summary.map_step),
                    "stoichiometry": _stoichiometry_summary_text(summary),
                }
            )
            matrix_rows.append(row)
            params = {
                str(name): float(summary.bestfit_params[index])
                for index, name in enumerate(summary.full_parameter_names)
            }
            for name in params:
                if name not in parameter_names:
                    parameter_names.append(name)
            parameter_rows.append(
                (
                    report_index,
                    str(fit_index),
                    str(filter_index),
                    item.label,
                    filter_set.label,
                    params,
                )
            )
            for name, value in params.items():
                entry = prior_lookup.get(name)
                guide_low = None
                guide_high = None
                clip_status = ""
                if entry is not None:
                    guide_low, guide_high, _guide_kind = (
                        distribution_guide_bounds(entry)
                    )
                    clip_status = guide_clip_status(
                        value,
                        guide_low,
                        guide_high,
                    )
                guide_rows.append(
                    {
                        "report_index": report_index,
                        "fit_index": str(fit_index),
                        "filter_subindex": str(filter_index),
                        "queue_item": item.label,
                        "filter_set": filter_set.label,
                        "parameter": name,
                        "selected": _format_report_number(value),
                        "guide_low": format_distribution_guide_value(
                            guide_low
                        ),
                        "guide_high": format_distribution_guide_value(
                            guide_high
                        ),
                        "guide_clip": guide_clip_status_label(clip_status),
                    }
                )

    matrix_headers = [
        "report_index",
        "fit_index",
        "filter_subindex",
        "queue_item",
        "filter_set",
        "posterior_filter",
        "bestfit_method",
        "dream_search_preset",
        "prior_presets",
        "fit_q_range",
        "output_q_range",
        "candidate_samples",
        "kept_samples",
        "map_chain",
        "map_step",
        "stoichiometry",
        "status",
        "error",
        "run_dir",
    ]
    lines.extend(["", "Run/filter matrix:", "\t".join(matrix_headers)])
    for row in matrix_rows:
        lines.append(
            "\t".join(row.get(header, "") for header in matrix_headers)
        )

    if parameter_rows:
        parameter_headers = [
            "report_index",
            "fit_index",
            "filter_subindex",
            "queue_item",
            "filter_set",
            *parameter_names,
        ]
        lines.extend(
            [
                "",
                "Best-fit parameter values:",
                "\t".join(parameter_headers),
            ]
        )
        for (
            report_index,
            fit_index,
            filter_index,
            queue_label,
            filter_label,
            params,
        ) in parameter_rows:
            values = [
                report_index,
                fit_index,
                filter_index,
                queue_label,
                filter_label,
                *[
                    _format_report_number(params.get(name))
                    for name in parameter_names
                ],
            ]
            lines.append("\t".join(values))

    if guide_rows:
        guide_headers = [
            "report_index",
            "fit_index",
            "filter_subindex",
            "queue_item",
            "filter_set",
            "parameter",
            "selected",
            "guide_low",
            "guide_high",
            "guide_clip",
        ]
        lines.extend(
            ["", "Best-fit parameter guide bounds:", "\t".join(guide_headers)]
        )
        for row in guide_rows:
            lines.append(
                "\t".join(row.get(header, "") for header in guide_headers)
            )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if output_callback is not None:
        output_callback(f"DREAM batch comparison report: {report_path}")
    return report_path


def _write_batch_fit_pdf_report(
    run_set: DreamBatchRunSet,
    *,
    output_callback=_print_line,
) -> Path:
    report_path = run_set.fit_report_pdf_path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    explicit_filter_sets = list(run_set.filter_sets)
    filter_sets_for_overview = explicit_filter_sets
    if not filter_sets_for_overview and run_set.queue_items:
        filter_sets_for_overview = _filter_sets_for_item(
            run_set.queue_items[0],
            [],
        )

    report = _PdfReportBuffer()
    toc_entries: list[_PdfTocEntry] = []
    _add_pdf_title_page(report, run_set, filter_sets_for_overview)
    toc_entries.append(
        _PdfTocEntry(
            section="Overview",
            label="DREAM Queue",
            detail=f"{len(run_set.queue_items)} queued run(s)",
            page_number=report.get_pagecount() + 1,
        )
    )
    _add_pdf_queue_table(report, run_set)
    toc_entries.append(
        _PdfTocEntry(
            section="Overview",
            label="Posterior Filter Sets",
            detail=f"{len(filter_sets_for_overview) or 1} fit type(s)",
            page_number=report.get_pagecount() + 1,
        )
    )
    _add_pdf_filter_table(report, filter_sets_for_overview)
    for fit_index, item in enumerate(run_set.queue_items, start=1):
        toc_entries.append(
            _PdfTocEntry(
                section="Inputs",
                label=f"[Fit {fit_index}] {item.label}",
                detail="Runtime inputs, sampler settings, and prior map",
                page_number=report.get_pagecount() + 1,
            )
        )
        _add_pdf_queue_reproduction_pages(
            report,
            item,
            fit_index=fit_index,
        )
    for fit_index, item in enumerate(run_set.queue_items, start=1):
        item_filter_sets = _filter_sets_for_item(
            item,
            explicit_filter_sets,
        )
        for filter_index, filter_set in enumerate(
            item_filter_sets,
            start=1,
        ):
            report_index = _fit_filter_report_index(
                fit_index,
                filter_index,
            )
            toc_entries.append(
                _PdfTocEntry(
                    section="Fit",
                    label=(
                        f"[{report_index}] "
                        f"{item.label} / {filter_set.label}"
                    ),
                    detail=(
                        f"fit {fit_index}, filter {filter_index} | "
                        f"{_fit_toc_detail(filter_set)}"
                    ),
                    page_number=report.get_pagecount() + 1,
                )
            )
            _add_pdf_fit_result_pages(
                report,
                item,
                filter_set,
                report_index=report_index,
            )

    final_figures = _insert_pdf_table_of_contents(
        report.figures,
        report_path=report_path,
        toc_entries=toc_entries,
    )
    with PdfPages(report_path) as pdf:
        for figure in final_figures:
            pdf.savefig(figure)

    if output_callback is not None:
        output_callback(f"DREAM batch fit PDF report: {report_path}")
    return report_path


def _fit_toc_detail(filter_set: DreamBatchFilterSet) -> str:
    settings = filter_set.settings
    details = [
        f"best fit: {settings.bestfit_method}",
        f"posterior: {settings.posterior_filter_mode}",
    ]
    if settings.posterior_filter_mode == "top_percent_logp":
        details.append(f"top {settings.posterior_top_percent:g}%")
    elif settings.posterior_filter_mode == "top_n_logp":
        details.append(f"top {settings.posterior_top_n:g} samples")
    details.append(f"violin: {settings.violin_parameter_mode}")
    if settings.stoichiometry_filter_enabled:
        details.append(
            "stoich: "
            f"{settings.stoichiometry_target_elements_text}="
            f"{settings.stoichiometry_target_ratio_text}"
        )
    return " | ".join(details)


def _insert_pdf_table_of_contents(
    figures: list[Figure],
    *,
    report_path: Path,
    toc_entries: list[_PdfTocEntry],
) -> list[Figure]:
    if not figures:
        return figures
    toc_page_count = _pdf_toc_page_count(toc_entries)
    adjusted_entries = [
        _PdfTocEntry(
            section=entry.section,
            label=entry.label,
            detail=entry.detail,
            page_number=(
                entry.page_number + toc_page_count
                if entry.page_number > _PDF_TOC_INSERT_INDEX
                else entry.page_number
            ),
        )
        for entry in toc_entries
    ]
    toc_figures = _pdf_table_of_contents_figures(
        report_path,
        adjusted_entries,
    )
    insert_index = min(_PDF_TOC_INSERT_INDEX, len(figures))
    return [
        *figures[:insert_index],
        *toc_figures,
        *figures[insert_index:],
    ]


def _pdf_toc_page_count(toc_entries: list[_PdfTocEntry]) -> int:
    return max(
        1,
        int(np.ceil(max(1, len(toc_entries)) / _PDF_TOC_ROWS_PER_PAGE)),
    )


def _add_pdf_table_of_contents_pages(
    pdf: PdfPages,
    report_path: Path,
    toc_entries: list[_PdfTocEntry],
) -> None:
    for figure in _pdf_table_of_contents_figures(report_path, toc_entries):
        pdf.savefig(figure)


def _pdf_table_of_contents_figures(
    report_path: Path,
    toc_entries: list[_PdfTocEntry],
) -> list[Figure]:
    from matplotlib.lines import Line2D

    entries = toc_entries or [
        _PdfTocEntry(
            section="Report",
            label="No fit sections",
            detail="No DREAM queue items were available when the report was generated.",
            page_number=1,
        )
    ]
    figures: list[Figure] = []
    page_count = _pdf_toc_page_count(entries)
    for page_index, start in enumerate(
        range(0, len(entries), _PDF_TOC_ROWS_PER_PAGE),
        start=1,
    ):
        chunk = entries[start : start + _PDF_TOC_ROWS_PER_PAGE]
        fig = Figure(figsize=(8.5, 11))
        fig.text(
            0.08,
            0.95,
            "Table of Contents",
            fontsize=16,
            fontweight="bold",
        )
        if page_count > 1:
            fig.text(
                0.92,
                0.95,
                f"{page_index}/{page_count}",
                fontsize=8,
                ha="right",
            )
        fig.text(
            0.08,
            0.915,
            "Linked sections for the batch overview, inputs, and posterior-filter fit types.",
            fontsize=8.5,
        )
        fig.text(0.08, 0.875, "Section", fontsize=8, fontweight="bold")
        fig.text(
            0.23,
            0.875,
            "Fit type / report section",
            fontsize=8,
            fontweight="bold",
        )
        fig.text(
            0.80,
            0.875,
            "Page",
            fontsize=8,
            fontweight="bold",
            ha="right",
        )
        fig.lines.append(
            Line2D(
                [0.08, 0.80],
                [0.855, 0.855],
                transform=fig.transFigure,
                color="#d0d0d0",
                linewidth=0.7,
            )
        )
        y = 0.835
        for entry in chunk:
            link_url = _pdf_page_uri(report_path, entry.page_number)
            fig.text(0.08, y, entry.section, fontsize=8)
            fig.text(
                0.23,
                y,
                _shorten_pdf_text(entry.label, 54),
                fontsize=8.2,
                color="#1f5fbf",
                url=link_url,
            )
            fig.text(
                0.80,
                y,
                str(entry.page_number),
                fontsize=8.2,
                ha="right",
                color="#1f5fbf",
                url=link_url,
            )
            fig.text(
                0.23,
                y - 0.014,
                _shorten_pdf_text(entry.detail, 104),
                fontsize=7,
                color="#4d4d4d",
                url=link_url,
            )
            y -= 0.032
        figures.append(fig)
    return figures


def _pdf_page_uri(report_path: Path, page_number: int) -> str | None:
    file_uri = _path_file_uri(report_path)
    if file_uri is None:
        return None
    return f"{file_uri}#page={max(1, int(page_number))}"


def _add_pdf_title_page(
    pdf: PdfPages,
    run_set: DreamBatchRunSet,
    filter_sets: list[DreamBatchFilterSet],
) -> None:
    fields = [
        ("Run set", run_set.label),
        ("Generated", _timestamp()),
        ("Project folder", run_set.project_dir),
        ("Run set folder", run_set.run_set_dir),
        ("Manifest", run_set.manifest_path),
        ("Commands", run_set.commands_path),
        ("Comparison report", run_set.comparison_report_path),
        (
            "DREAM fit log",
            run_set.log_path,
            _path_file_uri(run_set.log_path),
        ),
        ("Conda environment", run_set.conda_env),
        ("Queued DREAM runs", len(run_set.queue_items)),
        ("Posterior filter sets", len(filter_sets) or 1),
        (
            "Report index convention",
            "Fit Index.Filter Subindex (for example 2.1)",
        ),
    ]
    _add_pdf_key_value_page(
        pdf,
        "SAXSShell DREAM Batch Fit Report",
        fields,
        subtitle=(
            "This report records the runtime inputs, prior maps, posterior "
            "filter settings, fit figures, and selected parameter summaries "
            "needed to reproduce the batch fit outputs."
        ),
        legend_items=_guide_clip_legend_items(),
    )


def _add_pdf_queue_table(pdf: PdfPages, run_set: DreamBatchRunSet) -> None:
    rows = [
        [
            str(index),
            item.label,
            item.status,
            _queue_item_search_preset_text(item),
            _queue_item_prior_preset_summary_text(item),
            item.started_at or "",
            item.finished_at or "",
            item.error or "",
            item.run_dir,
        ]
        for index, item in enumerate(run_set.queue_items, start=1)
    ]
    _add_pdf_table_pages(
        pdf,
        "Queued DREAM Runs",
        [
            "Fit Index",
            "Label",
            "Status",
            "Search preset",
            "Prior presets",
            "Started",
            "Finished",
            "Error",
            "Run folder",
        ],
        rows,
        max_rows=14,
        landscape=True,
    )


def _add_pdf_filter_table(
    pdf: PdfPages,
    filter_sets: list[DreamBatchFilterSet],
) -> None:
    rows = []
    for index, filter_set in enumerate(filter_sets, start=1):
        settings = filter_set.settings
        rows.append(
            [
                str(index),
                filter_set.label,
                settings.bestfit_method,
                settings.posterior_filter_mode,
                _format_report_number(settings.posterior_top_percent),
                str(settings.posterior_top_n),
                (
                    f"{settings.credible_interval_low:g} - "
                    f"{settings.credible_interval_high:g}"
                ),
                "on" if settings.stoichiometry_filter_enabled else "off",
                settings.stoichiometry_target_elements_text,
                settings.stoichiometry_target_ratio_text,
                _format_report_number(
                    settings.stoichiometry_tolerance_percent
                ),
            ]
        )
    _add_pdf_table_pages(
        pdf,
        "Posterior Filter Sets",
        [
            "Filter Subindex",
            "Label",
            "Best fit",
            "Posterior filter",
            "Top %",
            "Top N",
            "CI %",
            "Stoich",
            "Target elements",
            "Target ratio",
            "Tolerance %",
        ],
        rows,
        max_rows=20,
        landscape=True,
    )


def _add_pdf_queue_reproduction_pages(
    pdf: PdfPages,
    item: DreamBatchQueueItem,
    *,
    fit_index: int | None = None,
) -> None:
    settings_payload = _load_json_object(item.settings_path)
    prior_preset_summary = _queue_item_prior_preset_summary_text(item)
    fields = [
        ("Queue label", item.label),
        ("Status", item.status),
        ("Created", item.created_at),
        ("Started", item.started_at or ""),
        ("Finished", item.finished_at or ""),
        ("Error", item.error or ""),
        ("Run folder", item.run_dir),
        ("Runtime script", item.runtime_script_path),
        ("Runtime metadata", item.metadata_path),
        ("DREAM settings JSON", item.settings_path),
        ("Parameter map JSON", item.parameter_map_path),
        (
            "DREAM search preset",
            _settings_payload_search_preset_text(settings_payload),
        ),
        (
            "Requested fit q-range",
            _settings_payload_fit_q_range_text(settings_payload),
        ),
        ("Prior preset summary", prior_preset_summary),
    ]
    if fit_index is not None:
        fields.insert(0, ("Fit setting index", str(fit_index)))
    _add_pdf_key_value_page(
        pdf,
        _pdf_indexed_title(
            f"Reproduction Inputs: {item.label}",
            f"Fit {fit_index}" if fit_index is not None else None,
        ),
        fields,
    )

    setting_rows = [
        [str(key), _format_pdf_value(value)]
        for key, value in sorted(settings_payload.items())
    ]
    _add_pdf_table_pages(
        pdf,
        _pdf_indexed_title(
            f"DREAM Sampler Settings: {item.label}",
            f"Fit {fit_index}" if fit_index is not None else None,
        ),
        ["Setting", "Value"],
        setting_rows,
        max_rows=28,
    )

    try:
        entries = load_parameter_map_for_queue_item(item)
        prior_rows = []
        for entry in entries:
            guide_low_cell, guide_high_cell, clip_label = (
                _guide_cells_for_value(entry, entry.value)
            )
            clip_status = guide_clip_status(
                entry.value,
                *distribution_guide_bounds(entry)[:2],
            )
            prior_rows.append(
                [
                    entry.param,
                    entry.structure,
                    entry.motif,
                    entry.param_type,
                    _highlight_value_for_clip(
                        _format_report_number(entry.value),
                        clip_status,
                    ),
                    guide_low_cell,
                    guide_high_cell,
                    clip_label,
                    "yes" if entry.vary else "no",
                    entry.distribution,
                    _format_distribution_params(entry.dist_params),
                    prior_preset_status_label(entry.smart_preset_status),
                ]
            )
    except Exception as exc:
        prior_rows = [
            [
                "Parameter map failed to load",
                str(exc),
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
            ]
        ]
    _add_pdf_table_pages(
        pdf,
        _pdf_indexed_title(
            f"Prior Map: {item.label}",
            f"Fit {fit_index}" if fit_index is not None else None,
        ),
        [
            "Param",
            "Structure",
            "Motif",
            "Type",
            "Initial",
            "Guide Low",
            "Guide High",
            "Guide Clip",
            "Vary",
            "Distribution",
            "Distribution params",
            "Preset",
        ],
        prior_rows,
        max_rows=18,
        landscape=True,
    )


def _add_pdf_fit_result_pages(
    pdf: PdfPages,
    item: DreamBatchQueueItem,
    filter_set: DreamBatchFilterSet,
    *,
    report_index: str | None = None,
) -> None:
    if item.status != "completed":
        _add_pdf_key_value_page(
            pdf,
            _pdf_indexed_title(
                f"Fit Result Skipped: {item.label} / {filter_set.label}",
                report_index,
            ),
            [
                ("Report index", report_index or ""),
                ("Queue item status", item.status),
                ("Error", item.error or ""),
                ("Run folder", item.run_dir),
            ],
        )
        return
    try:
        loader = SAXSDreamResultsLoader(item.run_dir)
        summary_kwargs = _summary_filter_kwargs(filter_set.settings)
        summary = loader.get_summary(**summary_kwargs)
        model_data = loader.build_model_fit_data(**summary_kwargs)
    except Exception as exc:
        _add_pdf_key_value_page(
            pdf,
            _pdf_indexed_title(
                f"Fit Result Failed: {item.label} / {filter_set.label}",
                report_index,
            ),
            [
                ("Report index", report_index or ""),
                ("Queue item", item.label),
                ("Posterior filter set", filter_set.label),
                ("Error", str(exc)),
                ("Run folder", item.run_dir),
            ],
        )
        return

    _add_pdf_model_fit_page(
        pdf,
        item,
        filter_set,
        summary,
        model_data,
        report_index=report_index,
    )
    _add_pdf_parameter_summary_pages(
        pdf,
        item,
        filter_set,
        summary,
        model_data,
        report_index=report_index,
    )
    try:
        violin_data = loader.build_violin_data(
            **_violin_filter_kwargs(filter_set.settings)
        )
    except Exception as exc:
        _add_pdf_key_value_page(
            pdf,
            _pdf_indexed_title(
                f"Posterior Figure Failed: {item.label} / {filter_set.label}",
                report_index,
            ),
            [
                ("Report index", report_index or ""),
                ("Queue item", item.label),
                ("Posterior filter set", filter_set.label),
                ("Error", str(exc)),
            ],
        )
        return
    _add_pdf_violin_page(
        pdf,
        item,
        filter_set,
        summary,
        violin_data,
        report_index=report_index,
    )


def _add_pdf_model_fit_page(
    pdf: PdfPages,
    item: DreamBatchQueueItem,
    filter_set: DreamBatchFilterSet,
    summary: object,
    model_data: object,
    *,
    report_index: str | None = None,
) -> None:
    q_values = np.asarray(model_data.q_values, dtype=float)
    experimental = np.asarray(model_data.experimental_intensities, dtype=float)
    model = np.asarray(model_data.model_intensities, dtype=float)
    finite = (
        np.isfinite(q_values) & np.isfinite(experimental) & np.isfinite(model)
    )
    if not np.any(finite):
        _add_pdf_key_value_page(
            pdf,
            _pdf_indexed_title(
                "Model Fit Figure Failed: "
                f"{item.label} / {filter_set.label}",
                report_index,
            ),
            [
                ("Report index", report_index or ""),
                ("Error", "No finite model or experimental intensity points."),
            ],
        )
        return

    q_values = q_values[finite]
    experimental = experimental[finite]
    model = model[finite]
    residual = model - experimental
    active_fit_mask = getattr(model_data, "active_fit_mask", None)
    fit_q_bounds = dream_fit_q_bounds(
        getattr(model_data, "q_values", q_values),
        active_fit_mask,
    )
    output_q_bounds = dream_output_q_bounds(
        getattr(model_data, "q_values", q_values)
    )
    fit_point_text = ""
    if active_fit_mask is not None:
        mask = np.asarray(active_fit_mask, dtype=bool)
        if mask.shape == np.asarray(model_data.q_values).shape:
            finite_fit_mask = mask[np.asarray(finite, dtype=bool)]
            fit_point_text = (
                f" | Fit points: {int(np.count_nonzero(finite_fit_mask))} / "
                f"{int(finite_fit_mask.size)}"
            )
    fig = Figure(figsize=(8.5, 11))
    axes = fig.subplots(
        2,
        1,
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    fit_ax, residual_ax = axes
    _shade_fit_q_range(fit_ax, fit_q_bounds, label="Fit q-range")
    fit_ax.scatter(
        q_values,
        experimental,
        s=22,
        color="#1f77b4",
        label="Experimental",
        zorder=3,
    )
    fit_ax.plot(
        q_values, model, color="#d62728", linewidth=1.8, label="Best fit"
    )
    fit_ax.set_ylabel("Intensity")
    if np.all(q_values > 0.0):
        fit_ax.set_xscale("log")
    if np.all(experimental > 0.0) and np.all(model > 0.0):
        fit_ax.set_yscale("log")
    fit_ax.grid(True, alpha=0.25)
    fit_ax.legend(loc="best", fontsize=8)

    _shade_fit_q_range(residual_ax, fit_q_bounds)
    residual_ax.axhline(0.0, color="#444444", linewidth=0.8)
    residual_ax.plot(q_values, residual, color="#444444", linewidth=1.2)
    residual_ax.set_xlabel("q")
    residual_ax.set_ylabel("Residual")
    residual_ax.grid(True, alpha=0.25)

    title = _pdf_indexed_title(
        f"{item.label} / {filter_set.label}",
        report_index,
    )
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.985)
    info_lines = [
        f"Run folder: {item.run_dir}",
        (
            f"DREAM search preset: {_queue_item_search_preset_text(item)} | "
            f"Prior presets: {_queue_item_prior_preset_summary_text(item)}"
        ),
        (
            f"Best-fit method: {model_data.bestfit_method} | "
            f"Posterior filter: {summary.posterior_filter_mode} | "
            f"Kept samples: {summary.posterior_sample_count} / "
            f"{summary.posterior_candidate_sample_count}"
        ),
        (
            f"RMSE: {_format_report_number(model_data.rmse)} | "
            f"Mean abs residual: "
            f"{_format_report_number(model_data.mean_abs_residual)} | "
            f"R^2: {_format_report_number(model_data.r_squared)}"
            f"{fit_point_text}"
        ),
        (
            f"Fit q-range: {format_dream_q_bounds(fit_q_bounds)} | "
            f"Output q-range: {format_dream_q_bounds(output_q_bounds)}"
        ),
        (
            f"MAP chain/step: {summary.map_chain}/{summary.map_step} | "
            "Stoichiometry: "
            f"{_stoichiometry_summary_text(summary) or 'not evaluated'}"
        ),
    ]
    if report_index:
        info_lines.insert(0, f"Report index: {report_index}")
    y = 0.94
    for line in info_lines:
        y = _draw_wrapped_pdf_text(
            fig,
            0.08,
            y,
            line,
            width=108,
            fontsize=8.5,
            line_height=0.016,
        )
    fig.subplots_adjust(
        left=0.1, right=0.96, top=0.82, bottom=0.08, hspace=0.08
    )
    pdf.savefig(fig)


def _add_pdf_parameter_summary_pages(
    pdf: PdfPages,
    item: DreamBatchQueueItem,
    filter_set: DreamBatchFilterSet,
    summary: object,
    model_data: object,
    *,
    report_index: str | None = None,
) -> None:
    fit_parameter_lookup = {
        str(parameter.name): parameter
        for parameter in getattr(model_data, "fit_parameters", [])
    }
    prior_lookup = _parameter_map_lookup_for_queue_item(item)
    rows = []
    for index, name in enumerate(summary.full_parameter_names):
        parameter = fit_parameter_lookup.get(str(name))
        selected_value = summary.bestfit_params[index]
        guide_low_cell, guide_high_cell, clip_label = _guide_cells_for_value(
            prior_lookup.get(str(name)),
            selected_value,
        )
        clip_status = str(clip_label).lower().replace("guide ", "")
        rows.append(
            [
                str(name),
                ("yes" if getattr(parameter, "varied", False) else "no"),
                getattr(parameter, "structure", "") if parameter else "",
                getattr(parameter, "motif", "") if parameter else "",
                getattr(parameter, "param_type", "") if parameter else "",
                _highlight_value_for_clip(
                    _format_report_number(selected_value),
                    clip_status,
                ),
                guide_low_cell,
                guide_high_cell,
                clip_label,
                _format_report_number(summary.map_params[index]),
                _format_report_number(summary.chain_mean_params[index]),
                _format_report_number(summary.median_params[index]),
                _format_report_number(summary.interval_low_values[index]),
                _format_report_number(summary.interval_high_values[index]),
            ]
        )
    _add_pdf_table_pages(
        pdf,
        _pdf_indexed_title(
            "Selected Parameter Summary: "
            f"{item.label} / {filter_set.label}",
            report_index,
        ),
        [
            "Param",
            "Varied",
            "Structure",
            "Motif",
            "Type",
            "Selected",
            "Guide Low",
            "Guide High",
            "Guide Clip",
            "MAP",
            "Chain mean",
            "Median",
            f"p{summary.credible_interval_low:g}",
            f"p{summary.credible_interval_high:g}",
        ],
        rows,
        max_rows=15,
        landscape=True,
    )


def _add_pdf_violin_page(
    pdf: PdfPages,
    item: DreamBatchQueueItem,
    filter_set: DreamBatchFilterSet,
    summary: object,
    violin_data: object,
    *,
    report_index: str | None = None,
) -> None:
    samples = np.asarray(violin_data.samples, dtype=float)
    if samples.ndim == 1:
        samples = samples.reshape(-1, 1)
    if samples.size == 0 or samples.shape[0] == 0 or samples.shape[1] == 0:
        _add_pdf_key_value_page(
            pdf,
            _pdf_indexed_title(
                "Posterior Distribution: "
                f"{item.label} / {filter_set.label}",
                report_index,
            ),
            [
                ("Report index", report_index or ""),
                ("Status", "No posterior samples available for plotting."),
            ],
        )
        return

    limit = min(samples.shape[1], 24)
    plot_samples = []
    labels = []
    source_indices = []
    for index in range(limit):
        values = samples[:, index]
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        plot_samples.append(values)
        source_indices.append(index)
        display_names = getattr(violin_data, "display_names", [])
        label = (
            display_names[index]
            if index < len(display_names)
            else f"Param {index + 1}"
        )
        labels.append(_shorten_pdf_text(label, 24))
    if not plot_samples:
        _add_pdf_key_value_page(
            pdf,
            _pdf_indexed_title(
                "Posterior Distribution: "
                f"{item.label} / {filter_set.label}",
                report_index,
            ),
            [
                ("Report index", report_index or ""),
                (
                    "Status",
                    "No finite posterior samples available for plotting.",
                ),
            ],
        )
        return

    fig = Figure(figsize=(11, 8.5))
    ax = fig.add_subplot(111)
    parts = ax.violinplot(
        plot_samples,
        showmeans=False,
        showmedians=True,
        showextrema=True,
    )
    for body in parts.get("bodies", []):
        body.set_facecolor("#4c72b0")
        body.set_edgecolor("#1f1f1f")
        body.set_alpha(0.65)
    for key in ("cmedians", "cbars", "cmins", "cmaxes"):
        artist = parts.get(key)
        if artist is not None:
            artist.set_color("#333333")
            artist.set_linewidth(0.8)
    selected_positions, selected_values = _selected_violin_plot_points(
        summary,
        violin_data,
        source_indices,
    )
    if selected_positions:
        ax.scatter(
            selected_positions,
            selected_values,
            s=28,
            color="#d62728",
            edgecolors="white",
            linewidths=0.45,
            zorder=5,
            label="Selected model value",
        )
        ax.legend(loc="best", fontsize=8, framealpha=0.9)
    ax.set_title(
        _pdf_indexed_title(
            "Posterior Distribution: " f"{item.label} / {filter_set.label}",
            report_index,
        ),
        fontsize=13,
        fontweight="bold",
    )
    ax.set_ylabel("Parameter value")
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=7)
    ax.grid(True, axis="y", alpha=0.25)
    subtitle = (
        f"Mode: {violin_data.mode} | Source: {violin_data.sample_source} | "
        f"Samples: {violin_data.sample_count}"
    )
    if report_index:
        subtitle = f"Report index: {report_index} | {subtitle}"
    if samples.shape[1] > limit:
        subtitle += (
            f" | Showing first {limit} of {samples.shape[1]} parameters"
        )
    fig.text(0.08, 0.93, subtitle, fontsize=8.5)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.25)
    pdf.savefig(fig)


def _selected_violin_plot_points(
    summary: object,
    violin_data: object,
    source_indices: list[int],
) -> tuple[list[int], list[float]]:
    full_names = [
        str(name) for name in getattr(summary, "full_parameter_names", [])
    ]
    bestfit_params = np.asarray(
        getattr(summary, "bestfit_params", []),
        dtype=float,
    ).reshape(-1)
    if not full_names or bestfit_params.size == 0:
        return [], []
    selected_by_name = {
        name: float(value)
        for name, value in zip(full_names, bestfit_params, strict=False)
        if np.isfinite(value)
    }
    parameter_names = [
        str(name) for name in getattr(violin_data, "parameter_names", [])
    ]
    positions: list[int] = []
    values: list[float] = []
    for position, source_index in enumerate(source_indices, start=1):
        if source_index >= len(parameter_names):
            continue
        value = selected_by_name.get(parameter_names[source_index])
        if value is None:
            continue
        positions.append(position)
        values.append(value)
    return positions, values


def _add_pdf_key_value_page(
    pdf: PdfPages,
    title: str,
    fields: list[tuple[object, object] | tuple[object, object, str | None]],
    *,
    subtitle: str | None = None,
    legend_items: list[tuple[str, str]] | None = None,
) -> None:
    fig = Figure(figsize=(8.5, 11))
    fig.text(0.08, 0.95, title, fontsize=16, fontweight="bold")
    y = 0.9
    if subtitle:
        y = _draw_wrapped_pdf_text(
            fig,
            0.08,
            y,
            subtitle,
            width=86,
            fontsize=9,
            line_height=0.02,
        )
        y -= 0.02
    if legend_items:
        y = _draw_pdf_legend(fig, y, legend_items)
        y -= 0.02
    for pdf_field in fields:
        label, value, url = _normalize_pdf_field(pdf_field)
        if y < 0.08:
            pdf.savefig(fig)
            fig = Figure(figsize=(8.5, 11))
            fig.text(0.08, 0.95, title, fontsize=16, fontweight="bold")
            y = 0.9
        fig.text(0.08, y, f"{label}:", fontsize=9, fontweight="bold")
        y = _draw_wrapped_pdf_text(
            fig,
            0.32,
            y,
            _format_pdf_value(value),
            width=72,
            fontsize=8.5,
            line_height=0.017,
            url=url,
        )
        y -= 0.012
    pdf.savefig(fig)


def _draw_pdf_legend(
    fig: Figure,
    y: float,
    legend_items: list[tuple[str, str]],
) -> float:
    fig.text(0.08, y, "Guide clipping legend:", fontsize=9, fontweight="bold")
    y -= 0.024
    for label, color in legend_items:
        fig.patches.append(
            Rectangle(
                (0.08, y - 0.002),
                0.022,
                0.014,
                transform=fig.transFigure,
                facecolor=color,
                edgecolor="#808080",
                linewidth=0.5,
            )
        )
        fig.text(0.112, y, label, fontsize=8.5)
        y -= 0.022
    return y


def _guide_clip_legend_items() -> list[tuple[str, str]]:
    return [
        (
            "Guide Low clipping: selected value is at or below Guide Low",
            _GUIDE_LOW_CLIP_COLOR,
        ),
        (
            "Guide High clipping: selected value is at or above Guide High",
            _GUIDE_HIGH_CLIP_COLOR,
        ),
    ]


def _normalize_pdf_field(
    field: tuple[object, object] | tuple[object, object, str | None],
) -> tuple[object, object, str | None]:
    if len(field) >= 3:
        return field[0], field[1], field[2]
    return field[0], field[1], None


def _add_pdf_table_pages(
    pdf: PdfPages,
    title: str,
    headers: list[str],
    rows: list[list[object]],
    *,
    max_rows: int = 24,
    landscape: bool = False,
) -> None:
    display_rows = rows or [[""] * len(headers)]
    page_size = (11, 8.5) if landscape else (8.5, 11)
    max_rows = max(1, int(max_rows))
    for start in range(0, len(display_rows), max_rows):
        chunk = display_rows[start : start + max_rows]
        page_number = (start // max_rows) + 1
        page_count = int(np.ceil(len(display_rows) / max_rows))
        fig = Figure(figsize=page_size)
        fig.text(0.05, 0.94, title, fontsize=14, fontweight="bold")
        if page_count > 1:
            fig.text(
                0.92,
                0.94,
                f"{page_number}/{page_count}",
                fontsize=8,
                ha="right",
            )
        ax = fig.add_axes([0.04, 0.06, 0.92, 0.83])
        ax.axis("off")
        wrapped_rows = [
            [
                _wrap_table_cell(value, width=28 if landscape else 36)
                for value in row
            ]
            for row in chunk
        ]
        wrapped_headers = [
            _wrap_table_cell(header, width=18 if landscape else 24)
            for header in headers
        ]
        table = ax.table(
            cellText=wrapped_rows,
            colLabels=wrapped_headers,
            cellLoc="left",
            colLoc="left",
            loc="upper left",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(6.5 if landscape else 7)
        table.scale(1, 1.25)
        for (row_index, _column_index), cell in table.get_celld().items():
            cell.set_edgecolor("#d0d0d0")
            cell.set_linewidth(0.4)
            if row_index == 0:
                cell.set_facecolor("#f0f0f0")
                cell.set_text_props(weight="bold")
            else:
                row_values = chunk[row_index - 1]
                if _column_index < len(row_values):
                    fill_color = _pdf_table_cell_fill(
                        row_values[_column_index]
                    )
                    if fill_color:
                        cell.set_facecolor(fill_color)
        pdf.savefig(fig)


def _draw_wrapped_pdf_text(
    fig: Figure,
    x: float,
    y: float,
    text: object,
    *,
    width: int,
    fontsize: float,
    line_height: float,
    url: str | None = None,
) -> float:
    lines = textwrap.wrap(str(text), width=max(8, int(width))) or [""]
    for line in lines:
        fig.text(x, y, line, fontsize=fontsize, url=url)
        y -= line_height
    return y


def _wrap_table_cell(value: object, *, width: int) -> str:
    text = _format_pdf_value(value)
    if "\n" in text:
        pieces = text.splitlines()
    else:
        pieces = textwrap.wrap(text, width=max(8, int(width))) or [""]
    return "\n".join(pieces[:4])


def _pdf_table_cell_fill(value: object) -> str | None:
    if isinstance(value, _PdfTableCell):
        return value.fill_color
    return None


def _shorten_pdf_text(value: object, limit: int) -> str:
    text = str(value)
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def _format_pdf_value(value: object) -> str:
    if isinstance(value, _PdfTableCell):
        return _format_pdf_value(value.text)
    if value is None:
        return ""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (dict, list, tuple)):
        try:
            return json.dumps(value, sort_keys=True)
        except TypeError:
            return str(value)
    if isinstance(value, float):
        return _format_report_number(value)
    return str(value)


def _settings_payload_fit_q_range_text(payload: dict[str, object]) -> str:
    q_min = payload.get("fit_q_min")
    q_max = payload.get("fit_q_max")
    if q_min is None and q_max is None:
        return "Full computed q-range"
    min_text = "minimum" if q_min is None else _format_pdf_value(q_min)
    max_text = "maximum" if q_max is None else _format_pdf_value(q_max)
    return f"{min_text} to {max_text}"


def _settings_payload_search_preset_text(
    payload: dict[str, object],
) -> str:
    if "search_filter_preset" not in payload:
        return "Unavailable"
    return format_dream_search_filter_preset(
        payload.get("search_filter_preset")
    )


def _queue_item_search_preset_text(item: object) -> str:
    settings_path = str(getattr(item, "settings_path", "") or "").strip()
    if not settings_path:
        return "Unavailable"
    try:
        return _settings_payload_search_preset_text(
            _load_json_object(settings_path)
        )
    except Exception:
        return "Unavailable"


def _queue_item_prior_preset_summary_text(item: object) -> str:
    try:
        return format_prior_preset_summary(
            load_parameter_map_for_queue_item(item)
        )
    except Exception:
        return "Unavailable"


def _shade_fit_q_range(
    axis,
    bounds: tuple[float, float] | None,
    *,
    label: str | None = None,
) -> object | None:
    if bounds is None:
        return None
    q_min, q_max = bounds
    if not np.isfinite(q_min) or not np.isfinite(q_max):
        return None
    if q_max < q_min:
        q_min, q_max = q_max, q_min
    if q_max == q_min:
        return None
    return axis.axvspan(
        q_min,
        q_max,
        color=_FIT_Q_RANGE_SHADE_COLOR,
        alpha=0.35,
        zorder=0,
        label=label,
    )


def _parameter_map_lookup_for_queue_item(
    item: object,
) -> dict[str, DreamParameterEntry]:
    try:
        entries = load_parameter_map_for_queue_item(item)
    except Exception:
        return {}
    lookup: dict[str, DreamParameterEntry] = {}
    for entry in entries:
        name = str(entry.param).strip()
        if name and name not in lookup:
            lookup[name] = entry
    return lookup


def _guide_cells_for_value(
    entry: DreamParameterEntry | None,
    value: object,
) -> tuple[object, object, str]:
    if entry is None:
        return "n/a", "n/a", ""
    guide_low, guide_high, _guide_kind = distribution_guide_bounds(entry)
    clip_status = guide_clip_status(value, guide_low, guide_high)
    fill_color = _guide_clip_fill_color(clip_status)
    low_text = format_distribution_guide_value(guide_low)
    high_text = format_distribution_guide_value(guide_high)
    low_cell: object = low_text
    high_cell: object = high_text
    if clip_status == "low":
        low_cell = _PdfTableCell(low_text, fill_color)
    elif clip_status == "high":
        high_cell = _PdfTableCell(high_text, fill_color)
    return low_cell, high_cell, guide_clip_status_label(clip_status)


def _highlight_value_for_clip(value: object, clip_status: str) -> object:
    fill_color = _guide_clip_fill_color(clip_status)
    if not fill_color:
        return value
    return _PdfTableCell(value, fill_color)


def _guide_clip_fill_color(status: object) -> str | None:
    normalized = str(status or "").strip().lower()
    if normalized == "low":
        return _GUIDE_LOW_CLIP_COLOR
    if normalized == "high":
        return _GUIDE_HIGH_CLIP_COLOR
    return None


def _format_distribution_params(params: object) -> str:
    if not isinstance(params, dict):
        return _format_pdf_value(params)
    keys = sorted(
        params,
        key=lambda key: (
            (
                _DISTRIBUTION_PARAM_KEY_ORDER.index(str(key))
                if str(key) in _DISTRIBUTION_PARAM_KEY_ORDER
                else len(_DISTRIBUTION_PARAM_KEY_ORDER)
            ),
            str(key),
        ),
    )
    return ", ".join(
        f"{key}={_format_compact_pdf_number(params[key])}" for key in keys
    )


def _format_compact_pdf_number(value: object) -> str:
    if isinstance(value, bool):
        return str(value)
    try:
        return f"{float(value):.5g}"
    except (TypeError, ValueError):
        return str(value)


def _path_file_uri(path: str | Path) -> str | None:
    try:
        return Path(path).expanduser().resolve().as_uri()
    except ValueError:
        return None


def _load_json_object(path: str | Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    return payload


def _load_prefit_entries_for_queue_item(
    item: DreamBatchQueueItem,
) -> list[PrefitParameterEntry] | None:
    state_path = (
        Path(item.run_dir).expanduser().resolve() / "prefit_state.json"
    )
    if not state_path.is_file():
        return None
    payload = _load_json_object(state_path)
    entries = payload.get("parameter_entries")
    if not isinstance(entries, list):
        return None
    return [
        PrefitParameterEntry.from_dict(dict(entry))
        for entry in entries
        if isinstance(entry, dict)
    ]


def _save_prefit_entries_for_queue_item(
    item: DreamBatchQueueItem,
    entries: list[PrefitParameterEntry],
) -> None:
    state_path = (
        Path(item.run_dir).expanduser().resolve() / "prefit_state.json"
    )
    if not state_path.is_file():
        return
    payload = _load_json_object(state_path)
    payload["parameter_entries"] = [entry.to_dict() for entry in entries]
    state_path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


def queue_item_weight_states(
    item: DreamBatchQueueItem,
) -> list[dict[str, object]]:
    entries = _load_prefit_entries_for_queue_item(item) or []
    states = []
    for entry in entries:
        if not _is_prefit_weight_entry(entry):
            continue
        states.append(
            {
                "name": str(entry.name),
                "structure": str(entry.structure),
                "motif": str(entry.motif),
                "active": bool(getattr(entry, "active", True)),
            }
        )
    return states


def queue_item_weight_state_summary(item: DreamBatchQueueItem) -> str:
    states = queue_item_weight_states(item)
    if not states:
        return "n/a"
    return "; ".join(
        f"{state['name']} {'on' if state['active'] else 'off'}"
        for state in states
    )


def _is_prefit_weight_entry(entry: PrefitParameterEntry) -> bool:
    name = str(entry.name).strip()
    return (
        str(entry.category).strip() == "weight"
        and name.startswith("w")
        and name[1:].isdigit()
    )


def _filter_sets_for_item(
    item: DreamBatchQueueItem,
    filter_sets: list[DreamBatchFilterSet],
) -> list[DreamBatchFilterSet]:
    if filter_sets:
        return list(filter_sets)
    run_settings_path = Path(item.settings_path).expanduser().resolve()
    run_settings = DreamRunSettings()
    if run_settings_path.is_file():
        run_settings = load_dream_settings(run_settings_path)
    return [
        DreamBatchFilterSet(
            filter_id="default_filter",
            label="Saved run filter settings",
            settings=PosteriorFilterSettings.from_run_settings(run_settings),
            created_at=_timestamp(),
        )
    ]


def _coerce_posterior_filter_settings(
    settings: PosteriorFilterSettings | DreamRunSettings,
) -> PosteriorFilterSettings:
    if isinstance(settings, PosteriorFilterSettings):
        return PosteriorFilterSettings.from_dict(settings.to_dict())
    return PosteriorFilterSettings.from_run_settings(settings)


def _stoichiometry_summary_text(summary: object) -> str:
    evaluation = getattr(summary, "stoichiometry_evaluation", None)
    if evaluation is None:
        return ""
    observed_ratio = getattr(evaluation, "observed_ratio", None)
    if observed_ratio is None:
        return ""
    try:
        return ":".join(
            _format_report_number(value) for value in observed_ratio
        )
    except TypeError:
        return str(observed_ratio)


def _format_report_number(value: object) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.6g}"
    except (TypeError, ValueError):
        return str(value)


def _signal_name(signum: int) -> str:
    try:
        return signal.Signals(signum).name
    except ValueError:
        return f"signal {signum}"


def _save_run_set(run_set: DreamBatchRunSet) -> None:
    run_set.updated_at = _timestamp()
    run_set.manifest_path.write_text(
        json.dumps(run_set.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )


def _write_status(
    run_set: DreamBatchRunSet,
    status: str,
    *,
    error: str | None = None,
) -> None:
    payload = {
        "status": status,
        "updated_at": _timestamp(),
        "project_dir": run_set.project_dir,
        "run_set_dir": run_set.run_set_dir,
        "pid": os.getpid(),
        "error": error,
    }
    run_set.status_path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


def _new_id(prefix: str) -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _sanitize_name(value: object) -> str:
    text = str(value or "").strip()
    text = re.sub(r"[^\w.\-]+", "_", text)
    text = text.strip("._-")
    return text or "dream_batch"


def _shell_quote(value: str | Path) -> str:
    text = str(value)
    return "'" + text.replace("'", "'\"'\"'") + "'"


def load_parameter_map_for_queue_item(
    item: DreamBatchQueueItem,
) -> list[DreamParameterEntry]:
    return load_parameter_map(item.parameter_map_path)


__all__ = [
    "DEFAULT_DREAM_BATCH_CONDA_ENV",
    "DREAM_BATCH_COMPARISON_NAME",
    "DREAM_BATCH_COMMANDS_NAME",
    "DREAM_BATCH_FIT_REPORT_PDF_NAME",
    "DREAM_BATCH_LOG_NAME",
    "DREAM_BATCH_MANIFEST_NAME",
    "DREAM_BATCH_PID_NAME",
    "DREAM_BATCH_SCRIPT_NAME",
    "DREAM_BATCH_STATUS_NAME",
    "DreamBatchFilterSet",
    "DreamBatchQueueItem",
    "DreamBatchRunSet",
    "DreamBatchRunSetManager",
    "command_text_for_run_set",
    "dream_filter_kwargs",
    "load_dream_batch_manifest",
    "load_parameter_map_for_queue_item",
    "queue_item_weight_state_summary",
    "queue_item_weight_states",
    "run_dream_batch_manifest",
    "shell_script_text_for_run_set",
]
