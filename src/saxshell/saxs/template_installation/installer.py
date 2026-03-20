from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from saxshell.saxs._model_templates import (
    clear_template_caches,
    default_template_dir,
    load_template_spec,
)

from .validator import TemplateValidationResult, validate_template_candidate


@dataclass(slots=True)
class InstalledTemplate:
    template_name: str
    installed_template_path: Path
    installed_metadata_path: Path | None
    validation_result: TemplateValidationResult


def install_template_candidate(
    template_path: str | Path,
    *,
    metadata_path: str | Path | None = None,
    destination_dir: str | Path | None = None,
    overwrite: bool = False,
) -> InstalledTemplate:
    validation_result = validate_template_candidate(
        template_path,
        metadata_path=metadata_path,
    )
    if not validation_result.passed:
        raise ValueError("Template validation failed. Install aborted.")

    destination = (
        Path(destination_dir).expanduser().resolve()
        if destination_dir is not None
        else default_template_dir()
    )
    destination.mkdir(parents=True, exist_ok=True)

    installed_template_path = destination / (
        f"{validation_result.template_name}.py"
    )
    installed_metadata_path = (
        destination / f"{validation_result.template_name}.json"
        if validation_result.metadata_path is not None
        else None
    )

    if not overwrite:
        if installed_template_path.exists():
            raise FileExistsError(
                f"Template already exists at {installed_template_path}"
            )
        if (
            installed_metadata_path is not None
            and installed_metadata_path.exists()
        ):
            raise FileExistsError(
                f"Template metadata already exists at {installed_metadata_path}"
            )

    shutil.copy2(validation_result.template_path, installed_template_path)
    if installed_metadata_path is not None:
        shutil.copy2(validation_result.metadata_path, installed_metadata_path)
    elif overwrite:
        stale_metadata = (
            destination / f"{validation_result.template_name}.json"
        )
        if stale_metadata.exists():
            stale_metadata.unlink()

    clear_template_caches()
    load_template_spec(validation_result.template_name, destination)

    return InstalledTemplate(
        template_name=validation_result.template_name,
        installed_template_path=installed_template_path,
        installed_metadata_path=installed_metadata_path,
        validation_result=validation_result,
    )
