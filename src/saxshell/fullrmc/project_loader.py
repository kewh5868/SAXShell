from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from saxshell.saxs.dream.settings import DreamRunSettings, load_dream_settings
from saxshell.saxs.project_manager import (
    DreamBestFitSelection,
    ProjectPaths,
    ProjectSettings,
    SAXSProjectManager,
    build_project_paths,
)

from .constraint_generation import (
    ConstraintGenerationMetadata,
    load_constraint_generation_metadata,
)
from .packmol_planning import (
    PackmolPlanningMetadata,
    load_packmol_planning_metadata,
)
from .packmol_setup import PackmolSetupMetadata, load_packmol_setup_metadata
from .project_model import (
    ClusterSourceValidationResult,
    RMCSetupPaths,
    ensure_rmcsetup_structure,
    validate_cluster_source,
)
from .representatives import (
    RepresentativeSelectionMetadata,
    load_representative_selection_metadata,
)
from .solution_properties import (
    SolutionPropertiesMetadata,
    load_solution_properties_metadata,
)
from .solvent_handling import (
    SolventHandlingMetadata,
    load_solvent_handling_metadata,
)


@dataclass(slots=True)
class RMCDreamRunRecord:
    run_name: str
    relative_path: str
    run_dir: Path
    metadata_path: Path
    settings_path: Path | None
    template_name: str | None
    model_name: str | None
    settings: DreamRunSettings


@dataclass(slots=True)
class RMCDreamProjectSource:
    settings: ProjectSettings
    paths: ProjectPaths
    rmcsetup_paths: RMCSetupPaths
    valid_runs: list[RMCDreamRunRecord]
    cluster_validation: ClusterSourceValidationResult
    solution_properties: SolutionPropertiesMetadata
    representative_selection: RepresentativeSelectionMetadata | None
    solvent_handling: SolventHandlingMetadata | None
    packmol_planning: PackmolPlanningMetadata | None
    packmol_setup: PackmolSetupMetadata | None
    constraint_generation: ConstraintGenerationMetadata | None
    favorite_selection: DreamBestFitSelection | None
    favorite_history: list[DreamBestFitSelection]

    def find_run_for_selection(
        self,
        selection: DreamBestFitSelection | None,
    ) -> RMCDreamRunRecord | None:
        if selection is None:
            return None
        for run in self.valid_runs:
            if run.relative_path == selection.run_relative_path:
                return run
        for run in self.valid_runs:
            if run.run_name == selection.run_name:
                return run
        return None


def load_rmc_project_source(
    project_dir: str | Path,
) -> RMCDreamProjectSource:
    manager = SAXSProjectManager()
    settings = manager.load_project(project_dir)
    paths = build_project_paths(settings.project_dir)
    rmcsetup_paths = ensure_rmcsetup_structure(paths)
    return RMCDreamProjectSource(
        settings=settings,
        paths=paths,
        rmcsetup_paths=rmcsetup_paths,
        valid_runs=discover_valid_dream_runs(paths),
        cluster_validation=validate_cluster_source(
            settings,
            project_paths=paths,
        ),
        solution_properties=load_solution_properties_metadata(
            rmcsetup_paths.solution_properties_path
        ),
        representative_selection=load_representative_selection_metadata(
            rmcsetup_paths.representative_selection_path
        ),
        solvent_handling=load_solvent_handling_metadata(
            rmcsetup_paths.solvent_handling_path
        ),
        packmol_planning=load_packmol_planning_metadata(
            rmcsetup_paths.packmol_plan_path
        ),
        packmol_setup=load_packmol_setup_metadata(
            rmcsetup_paths.packmol_setup_path
        ),
        constraint_generation=load_constraint_generation_metadata(
            rmcsetup_paths.constraint_generation_path
        ),
        favorite_selection=settings.dream_favorite_selection,
        favorite_history=list(settings.dream_favorite_history),
    )


def discover_valid_dream_runs(
    paths_or_dir: ProjectPaths | str | Path,
) -> list[RMCDreamRunRecord]:
    if isinstance(paths_or_dir, ProjectPaths):
        paths = paths_or_dir
    else:
        paths = build_project_paths(paths_or_dir)
    records: list[RMCDreamRunRecord] = []
    for metadata_path in sorted(
        paths.dream_dir.rglob("dream_runtime_metadata.json")
    ):
        run_dir = metadata_path.parent
        if not _is_valid_run_dir(run_dir):
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        settings_path = run_dir / "pd_settings.json"
        if settings_path.is_file():
            settings = load_dream_settings(settings_path)
        else:
            settings = DreamRunSettings.from_dict(
                dict(metadata.get("settings", {}))
            )
        relative_path = str(run_dir.relative_to(paths.project_dir))
        template_name = _optional_text(metadata.get("template_name"))
        records.append(
            RMCDreamRunRecord(
                run_name=run_dir.name,
                relative_path=relative_path,
                run_dir=run_dir,
                metadata_path=metadata_path,
                settings_path=(
                    settings_path if settings_path.is_file() else None
                ),
                template_name=template_name,
                model_name=_optional_text(settings.model_name),
                settings=settings,
            )
        )
    records.sort(
        key=lambda record: (
            record.run_name.lower(),
            record.relative_path.lower(),
        ),
        reverse=True,
    )
    return records


def _is_valid_run_dir(run_dir: Path) -> bool:
    return (
        run_dir.is_dir()
        and (run_dir / "dream_runtime_metadata.json").is_file()
        and (run_dir / "dream_sampled_params.npy").is_file()
        and (run_dir / "dream_log_ps.npy").is_file()
    )


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


__all__ = [
    "RMCDreamProjectSource",
    "RMCDreamRunRecord",
    "discover_valid_dream_runs",
    "load_rmc_project_source",
]
