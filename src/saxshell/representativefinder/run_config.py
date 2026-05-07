from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

from saxshell.bondanalysis import AngleTripletDefinition, BondPairDefinition
from saxshell.fullrmc.project_model import ensure_rmcsetup_structure
from saxshell.fullrmc.representatives import (
    load_representative_selection_metadata,
)

from .workflow import (
    RepresentativeFinderFolderInspection,
    RepresentativeFinderResult,
    RepresentativeFinderSettings,
    analyze_representative_structure_folder,
    inspect_representative_structure_input,
    persist_representativefinder_result_to_project,
    suggest_representativefinder_output_dir,
    suggest_representativefinder_target_output_dir,
)

DEFAULT_RUN_FILE_NAME = "representative_structure_cli_run.json"
RUN_CONFIG_VERSION = 1
RepresentativeFinderRunLogCallback = Callable[[str], None]
RepresentativeFinderRunProgressCallback = Callable[[int, int, str], None]


@dataclass(slots=True)
class RepresentativeFinderRunConfig:
    input_dir: str
    output_dir: str | None
    analysis_mode: str = "all"
    selected_stoichiometry: str | None = None
    overwrite_existing: bool = False
    settings: RepresentativeFinderSettings = field(
        default_factory=RepresentativeFinderSettings
    )
    created_at: str = field(
        default_factory=lambda: datetime.now().isoformat(timespec="seconds")
    )

    def to_dict(self) -> dict[str, object]:
        return {
            "version": RUN_CONFIG_VERSION,
            "created_at": self.created_at,
            "input_dir": self.input_dir,
            "output_dir": self.output_dir,
            "analysis_mode": _normalize_analysis_mode(self.analysis_mode),
            "selected_stoichiometry": self.selected_stoichiometry,
            "overwrite_existing": bool(self.overwrite_existing),
            "settings": self.settings.to_dict(),
        }

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, object],
    ) -> "RepresentativeFinderRunConfig":
        input_dir = str(payload.get("input_dir", "")).strip()
        if not input_dir:
            raise ValueError("Representative run file is missing input_dir.")
        output_dir = _optional_text(payload.get("output_dir"))
        return cls(
            input_dir=input_dir,
            output_dir=output_dir,
            analysis_mode=_normalize_analysis_mode(
                payload.get("analysis_mode", "all")
            ),
            selected_stoichiometry=_optional_text(
                payload.get("selected_stoichiometry")
            ),
            overwrite_existing=bool(payload.get("overwrite_existing", False)),
            settings=representativefinder_settings_from_dict(
                payload.get("settings")
            ),
            created_at=str(payload.get("created_at", "")).strip()
            or datetime.now().isoformat(timespec="seconds"),
        )


@dataclass(slots=True, frozen=True)
class RepresentativeFinderRunTarget:
    inspection: RepresentativeFinderFolderInspection
    output_dir: Path


@dataclass(slots=True, frozen=True)
class RepresentativeFinderRunFailure:
    input_dir: Path
    structure_label: str
    message: str


@dataclass(slots=True, frozen=True)
class RepresentativeFinderRunExecutionSummary:
    project_dir: Path
    run_file_path: Path | None
    targets: tuple[RepresentativeFinderRunTarget, ...]
    results: tuple[RepresentativeFinderResult, ...]
    project_representative_paths: tuple[Path, ...]
    failures: tuple[RepresentativeFinderRunFailure, ...]
    skipped_existing: tuple[str, ...]

    @property
    def completed_count(self) -> int:
        return len(self.results)

    @property
    def failed_count(self) -> int:
        return len(self.failures)


def default_representativefinder_run_file_path(
    project_dir: str | Path,
) -> Path:
    return Path(project_dir).expanduser().resolve() / DEFAULT_RUN_FILE_NAME


def representativefinder_settings_from_dict(
    payload: object,
) -> RepresentativeFinderSettings:
    source = dict(payload) if isinstance(payload, dict) else {}
    quantile_values = (
        source.get("quantiles") or RepresentativeFinderSettings().quantiles
    )
    quantiles = tuple(float(value) for value in quantile_values)
    return RepresentativeFinderSettings(
        selection_algorithm=str(
            source.get(
                "selection_algorithm",
                "target_distribution_quantile_distance",
            )
        ).strip()
        or "target_distribution_quantile_distance",
        bond_weight=_float_value(source.get("bond_weight"), 1.0),
        angle_weight=_float_value(source.get("angle_weight"), 1.0),
        solvent_weight=_float_value(source.get("solvent_weight"), 1.0),
        generate_predicted_optimized_representative=bool(
            source.get("generate_predicted_optimized_representative", False)
        ),
        parallel_workers=_int_value(source.get("parallel_workers"), 0),
        quantiles=quantiles or RepresentativeFinderSettings().quantiles,
        bond_pairs=tuple(
            _bond_pair_from_dict(entry)
            for entry in source.get("bond_pairs", [])
            if isinstance(entry, dict)
        ),
        angle_triplets=tuple(
            _angle_triplet_from_dict(entry)
            for entry in source.get("angle_triplets", [])
            if isinstance(entry, dict)
        ),
    )


def save_representativefinder_run_config(
    output_path: str | Path,
    config: RepresentativeFinderRunConfig,
) -> Path:
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(config.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def load_representativefinder_run_config(
    run_file_path: str | Path,
) -> RepresentativeFinderRunConfig:
    path = Path(run_file_path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(
            f"Representative run file must contain a JSON object: {path}"
        )
    return RepresentativeFinderRunConfig.from_dict(payload)


def path_text_for_run_config(
    path: str | Path | None,
    *,
    project_dir: str | Path,
) -> str | None:
    if path is None:
        return None
    resolved_project_dir = Path(project_dir).expanduser().resolve()
    resolved_path = Path(path).expanduser().resolve()
    try:
        return resolved_path.relative_to(resolved_project_dir).as_posix()
    except ValueError:
        return str(resolved_path)


def resolve_run_config_path(
    path_text: str | None,
    *,
    project_dir: str | Path,
) -> Path | None:
    text = str(path_text or "").strip()
    if not text:
        return None
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = Path(project_dir).expanduser().resolve() / path
    return path.resolve()


def build_representativefinder_run_config(
    *,
    project_dir: str | Path,
    input_dir: str | Path,
    output_dir: str | Path | None,
    analysis_mode: str,
    settings: RepresentativeFinderSettings,
    selected_stoichiometry: str | None = None,
    overwrite_existing: bool = False,
) -> RepresentativeFinderRunConfig:
    return RepresentativeFinderRunConfig(
        input_dir=path_text_for_run_config(
            input_dir,
            project_dir=project_dir,
        )
        or "",
        output_dir=path_text_for_run_config(
            output_dir,
            project_dir=project_dir,
        ),
        analysis_mode=_normalize_analysis_mode(analysis_mode),
        selected_stoichiometry=_optional_text(selected_stoichiometry),
        overwrite_existing=bool(overwrite_existing),
        settings=settings,
    )


def suggest_run_config_output_dir(
    *,
    project_dir: str | Path,
    input_dir: str | Path,
    analysis_mode: str,
) -> Path:
    inspection = inspect_representative_structure_input(input_dir)
    batch = (
        _normalize_analysis_mode(analysis_mode) == "all"
        or inspection.stoichiometry_count > 1
        or not inspection.input_is_stoichiometry_folder
    )
    suggestion_source = (
        inspection.input_dir
        if batch
        else inspection.stoichiometry_folders[0].input_dir
    )
    return suggest_representativefinder_output_dir(
        suggestion_source,
        project_dir=project_dir,
        batch=batch,
    )


def representativefinder_run_targets(
    *,
    project_dir: str | Path,
    config: RepresentativeFinderRunConfig,
) -> tuple[tuple[RepresentativeFinderRunTarget, ...], tuple[str, ...]]:
    resolved_project_dir = Path(project_dir).expanduser().resolve()
    input_dir = resolve_run_config_path(
        config.input_dir,
        project_dir=resolved_project_dir,
    )
    if input_dir is None:
        raise ValueError("Representative run file is missing input_dir.")
    inspection = inspect_representative_structure_input(input_dir)
    selected_stoichiometries = _selected_stoichiometries_for_config(
        inspection.stoichiometry_folders,
        analysis_mode=config.analysis_mode,
        selected_stoichiometry=config.selected_stoichiometry,
    )
    skipped_existing: tuple[str, ...] = ()
    if not config.overwrite_existing:
        saved_labels = _saved_project_representative_labels(
            resolved_project_dir
        )
        skipped_existing = tuple(
            stoich.structure_label
            for stoich in selected_stoichiometries
            if stoich.structure_label in saved_labels
        )
        selected_stoichiometries = tuple(
            stoich
            for stoich in selected_stoichiometries
            if stoich.structure_label not in saved_labels
        )

    output_root = resolve_run_config_path(
        config.output_dir,
        project_dir=resolved_project_dir,
    )
    if output_root is None:
        output_root = suggest_run_config_output_dir(
            project_dir=resolved_project_dir,
            input_dir=input_dir,
            analysis_mode=config.analysis_mode,
        )
    use_direct_output_dir = (
        inspection.input_is_stoichiometry_folder
        and inspection.stoichiometry_count == 1
    )
    targets = tuple(
        RepresentativeFinderRunTarget(
            inspection=stoich,
            output_dir=(
                output_root
                if use_direct_output_dir
                else suggest_representativefinder_target_output_dir(
                    output_root,
                    stoich.structure_label,
                )
            ),
        )
        for stoich in selected_stoichiometries
    )
    return targets, skipped_existing


def run_representativefinder_run_config(
    project_dir: str | Path,
    config: RepresentativeFinderRunConfig,
    *,
    run_file_path: str | Path | None = None,
    log_callback: RepresentativeFinderRunLogCallback | None = None,
    progress_callback: RepresentativeFinderRunProgressCallback | None = None,
) -> RepresentativeFinderRunExecutionSummary:
    resolved_project_dir = Path(project_dir).expanduser().resolve()
    targets, skipped_existing = representativefinder_run_targets(
        project_dir=resolved_project_dir,
        config=config,
    )
    results: list[RepresentativeFinderResult] = []
    failures: list[RepresentativeFinderRunFailure] = []
    project_paths: list[Path] = []

    if skipped_existing:
        _emit_run_log(
            log_callback,
            "Skipping saved representative structures: "
            + ", ".join(skipped_existing),
        )
    if not targets:
        _emit_run_log(
            log_callback,
            "No representative-structure targets need to be run.",
        )
    target_count = len(targets)
    for index, target in enumerate(targets, start=1):
        label = target.inspection.structure_label
        _emit_run_log(
            log_callback,
            f"[{index}/{target_count}] Starting {label}.",
        )

        def on_progress(
            processed: int,
            total: int,
            message: str,
        ) -> None:
            if progress_callback is not None:
                progress_callback(processed, total, f"[{label}] {message}")

        def on_log(message: str) -> None:
            _emit_run_log(log_callback, f"[{label}] {message}")

        try:
            result = analyze_representative_structure_folder(
                target.inspection.input_dir,
                settings=config.settings,
                output_dir=target.output_dir,
                project_dir=resolved_project_dir,
                progress_callback=on_progress,
                log_callback=on_log,
            )
            shared_path = persist_representativefinder_result_to_project(
                resolved_project_dir,
                result,
            )
        except Exception as exc:
            failures.append(
                RepresentativeFinderRunFailure(
                    input_dir=target.inspection.input_dir,
                    structure_label=label,
                    message=str(exc),
                )
            )
            _emit_run_log(
                log_callback,
                f"[{label}] Failed representative selection: {exc}",
            )
            continue
        results.append(result)
        project_paths.append(shared_path)
        _emit_run_log(
            log_callback,
            f"[{label}] Project representative: {shared_path}",
        )

    return RepresentativeFinderRunExecutionSummary(
        project_dir=resolved_project_dir,
        run_file_path=(
            None if run_file_path is None else Path(run_file_path).resolve()
        ),
        targets=targets,
        results=tuple(results),
        project_representative_paths=tuple(project_paths),
        failures=tuple(failures),
        skipped_existing=skipped_existing,
    )


def _selected_stoichiometries_for_config(
    stoichiometries: tuple[RepresentativeFinderFolderInspection, ...],
    *,
    analysis_mode: str,
    selected_stoichiometry: str | None,
) -> tuple[RepresentativeFinderFolderInspection, ...]:
    if not stoichiometries:
        raise ValueError("No stoichiometry folders were found.")
    if _normalize_analysis_mode(analysis_mode) == "all":
        return stoichiometries
    selected_label = str(selected_stoichiometry or "").strip()
    if selected_label:
        for stoich in stoichiometries:
            if stoich.structure_label == selected_label:
                return (stoich,)
        raise ValueError(
            "Selected stoichiometry was not found in the input folder: "
            f"{selected_label}"
        )
    return (stoichiometries[0],)


def _saved_project_representative_labels(project_dir: Path) -> set[str]:
    paths = ensure_rmcsetup_structure(project_dir)
    metadata = load_representative_selection_metadata(
        paths.representative_selection_path
    )
    if metadata is None:
        return set()
    labels: set[str] = set()
    for entry in metadata.representative_entries:
        source_file = str(entry.source_file or "").strip()
        if not source_file:
            continue
        if not Path(source_file).expanduser().resolve().is_file():
            continue
        structure = str(entry.structure or "").strip()
        if structure:
            labels.add(structure)
    return labels


def _bond_pair_from_dict(payload: dict[str, object]) -> BondPairDefinition:
    return BondPairDefinition(
        str(payload["atom1"]),
        str(payload["atom2"]),
        float(payload["cutoff_angstrom"]),
    )


def _angle_triplet_from_dict(
    payload: dict[str, object],
) -> AngleTripletDefinition:
    return AngleTripletDefinition(
        str(payload["vertex"]),
        str(payload["arm1"]),
        str(payload["arm2"]),
        float(payload["cutoff1_angstrom"]),
        float(payload["cutoff2_angstrom"]),
    )


def _normalize_analysis_mode(value: object) -> str:
    text = str(value or "").strip().lower()
    return "single" if text == "single" else "all"


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _float_value(value: object, default: float) -> float:
    if value is None:
        return float(default)
    text = str(value).strip()
    if not text:
        return float(default)
    return float(text)


def _int_value(value: object, default: int) -> int:
    if value is None:
        return int(default)
    text = str(value).strip()
    if not text:
        return int(default)
    return int(text)


def _emit_run_log(
    callback: RepresentativeFinderRunLogCallback | None,
    message: str,
) -> None:
    if callback is not None:
        callback(str(message).strip())


__all__ = [
    "DEFAULT_RUN_FILE_NAME",
    "RepresentativeFinderRunConfig",
    "RepresentativeFinderRunExecutionSummary",
    "RepresentativeFinderRunFailure",
    "RepresentativeFinderRunTarget",
    "build_representativefinder_run_config",
    "default_representativefinder_run_file_path",
    "load_representativefinder_run_config",
    "path_text_for_run_config",
    "representativefinder_run_targets",
    "representativefinder_settings_from_dict",
    "resolve_run_config_path",
    "run_representativefinder_run_config",
    "save_representativefinder_run_config",
    "suggest_run_config_output_dir",
]
