from __future__ import annotations

import json
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from saxshell.saxs._model_templates import (
    clear_template_caches,
    default_template_dir,
    load_template_spec,
)

from .validator import (
    TemplateValidationResult,
    format_validation_report,
    validate_template_candidate,
)


@dataclass(slots=True)
class InstalledTemplate:
    template_name: str
    installed_template_path: Path
    installed_metadata_path: Path | None
    validation_result: TemplateValidationResult


def install_template_candidate(
    template_path: str | Path,
    *,
    model_name: str | None = None,
    model_description: str | None = None,
    metadata_path: str | Path | None = None,
    destination_dir: str | Path | None = None,
    overwrite: bool = False,
) -> InstalledTemplate:
    source_path = Path(template_path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Template file not found: {source_path}")

    if (model_name is None) != (model_description is None):
        raise ValueError(
            "model_name and model_description must be provided together."
        )
    if model_name is not None and metadata_path is not None:
        raise ValueError(
            "Provide either model_name/model_description or metadata_path, "
            "not both."
        )

    destination = (
        Path(destination_dir).expanduser().resolve()
        if destination_dir is not None
        else default_template_dir()
    )
    destination.mkdir(parents=True, exist_ok=True)

    validation_result: TemplateValidationResult
    installed_template_path: Path | None = None
    installed_metadata_path: Path | None = None
    with tempfile.TemporaryDirectory(
        prefix="saxs_template_install_candidate_"
    ) as tmp:
        staged_dir = Path(tmp)
        staged_template = source_path
        staged_metadata: Path | None = (
            Path(metadata_path).expanduser().resolve()
            if metadata_path is not None
            else None
        )

        if model_name is not None:
            template_name = _normalize_template_name(model_name)
            staged_template = staged_dir / f"{template_name}.py"
            shutil.copy2(source_path, staged_template)
            staged_metadata = staged_dir / f"{template_name}.json"
            staged_metadata.write_text(
                _render_generated_metadata(
                    display_name=model_name,
                    description=model_description or "",
                ),
                encoding="utf-8",
            )

        validation_result = validate_template_candidate(
            staged_template,
            metadata_path=staged_metadata,
        )
        if not validation_result.passed:
            raise ValueError(format_validation_report(validation_result))

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
                    "Template metadata already exists at "
                    f"{installed_metadata_path}"
                )

        shutil.copy2(validation_result.template_path, installed_template_path)
        if installed_metadata_path is not None:
            shutil.copy2(
                validation_result.metadata_path,
                installed_metadata_path,
            )
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
        installed_template_path=(
            installed_template_path
            if installed_template_path is not None
            else destination / f"{validation_result.template_name}.py"
        ),
        installed_metadata_path=installed_metadata_path,
        validation_result=validation_result,
    )


def _normalize_template_name(model_name: str) -> str:
    normalized = re.sub(r"[^0-9A-Za-z]+", "_", model_name.strip()).strip("_")
    if not normalized:
        raise ValueError("Enter a model name containing letters or numbers.")
    normalized = normalized.lower()
    if not normalized.startswith("template_"):
        normalized = f"template_{normalized}"
    if normalized[0].isdigit():
        normalized = f"template_{normalized}"
    return normalized


def _render_generated_metadata(
    *,
    display_name: str,
    description: str,
) -> str:
    cleaned_name = display_name.strip()
    cleaned_description = description.strip()
    if not cleaned_name:
        raise ValueError("Enter a model name before installing a template.")
    if not cleaned_description:
        raise ValueError(
            "Enter a model description before installing a template."
        )
    payload = {
        "display_name": cleaned_name,
        "description": cleaned_description,
    }
    return json.dumps(payload, indent=2) + "\n"
