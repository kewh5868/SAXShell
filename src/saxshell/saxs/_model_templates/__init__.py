from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import ModuleType


@dataclass(slots=True, frozen=True)
class TemplateParameter:
    name: str
    initial_value: float
    vary: bool
    minimum: float
    maximum: float


@dataclass(slots=True, frozen=True)
class TemplateSpec:
    name: str
    module_path: Path
    metadata_path: Path | None
    display_name: str
    description: str
    lmfit_model_name: str
    dream_model_name: str
    lmfit_inputs: tuple[str, ...]
    dream_inputs: tuple[str, ...]
    param_columns: tuple[str, ...]
    parameters: tuple[TemplateParameter, ...]

    @property
    def label(self) -> str:
        return self.display_name


def default_template_dir() -> Path:
    return Path(__file__).resolve().parent


def list_template_specs() -> list[TemplateSpec]:
    return [
        load_template_spec(path.stem)
        for path in sorted(default_template_dir().glob("*.py"))
        if path.name != "__init__.py"
    ]


@lru_cache(maxsize=None)
def load_template_spec(template_name: str) -> TemplateSpec:
    module_path = default_template_dir() / f"{template_name}.py"
    if not module_path.is_file():
        raise FileNotFoundError(
            f"Unknown SAXS model template: {template_name}"
        )
    directives = _parse_directives(module_path)
    metadata_path = default_template_dir() / f"{template_name}.json"
    metadata = _load_template_metadata(metadata_path, template_name)
    return TemplateSpec(
        name=template_name,
        module_path=module_path,
        metadata_path=metadata_path if metadata_path.is_file() else None,
        display_name=str(metadata["display_name"]),
        description=str(metadata["description"]),
        lmfit_model_name=directives["model_lmfit"],
        dream_model_name=directives["model_pydream"],
        lmfit_inputs=_split_csv(directives["inputs_lmfit"]),
        dream_inputs=_split_csv(directives["inputs_pydream"]),
        param_columns=_split_csv(directives["param_columns"]),
        parameters=tuple(_parse_param_lines(module_path)),
    )


@lru_cache(maxsize=None)
def load_template_module(template_name: str) -> ModuleType:
    spec = load_template_spec(template_name)
    import_spec = importlib.util.spec_from_file_location(
        f"saxshell.saxs._model_templates.{template_name}",
        spec.module_path,
    )
    if import_spec is None or import_spec.loader is None:
        raise ImportError(
            f"Unable to import template module from {spec.module_path}"
        )
    module = importlib.util.module_from_spec(import_spec)
    import_spec.loader.exec_module(module)
    return module


def _parse_directives(module_path: Path) -> dict[str, str]:
    directives: dict[str, str] = {}
    required = {
        "model_lmfit",
        "model_pydream",
        "inputs_lmfit",
        "inputs_pydream",
        "param_columns",
    }
    with module_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped.startswith("# "):
                continue
            if ":" not in stripped:
                continue
            key, raw_value = stripped[2:].split(":", 1)
            key = key.strip()
            if key in required:
                directives[key] = raw_value.strip()
    missing = required - set(directives)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(
            f"Template {module_path.name} is missing directives: "
            f"{missing_text}"
        )
    return directives


def _parse_param_lines(module_path: Path) -> list[TemplateParameter]:
    parameters: list[TemplateParameter] = []
    with module_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped.startswith("# param:"):
                continue
            parts = [
                part.strip() for part in stripped.split(":", 1)[1].split(",")
            ]
            if len(parts) != 5:
                raise ValueError(
                    "Template parameter lines must contain five values: "
                    "name, initial, vary, min, max."
                )
            parameters.append(
                TemplateParameter(
                    name=parts[0],
                    initial_value=float(parts[1]),
                    vary=parts[2].lower() in {"1", "true", "yes"},
                    minimum=_parse_bound(parts[3]),
                    maximum=_parse_bound(parts[4]),
                )
            )
    return parameters


def _parse_bound(value: str) -> float:
    lowered = value.lower()
    if lowered == "inf":
        return float("inf")
    if lowered == "-inf":
        return float("-inf")
    return float(value)


def _split_csv(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _load_template_metadata(
    metadata_path: Path,
    template_name: str,
) -> dict[str, str]:
    if not metadata_path.is_file():
        fallback_name = template_name.replace("_", " ")
        return {
            "display_name": fallback_name,
            "description": (
                f"{fallback_name}\n\n"
                "No explicit template description file was found for this "
                "SAXS model template. Add a paired JSON metadata file in "
                "the _model_templates folder to provide a friendly display "
                "name and a detailed description."
            ),
        }
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    display_name = str(payload.get("display_name", "")).strip()
    description = str(payload.get("description", "")).strip()
    if not display_name:
        raise ValueError(
            f"Template metadata file {metadata_path.name} is missing "
            "'display_name'."
        )
    if not description:
        raise ValueError(
            f"Template metadata file {metadata_path.name} is missing "
            "'description'."
        )
    return {
        "display_name": display_name,
        "description": description,
    }


__all__ = [
    "TemplateParameter",
    "TemplateSpec",
    "default_template_dir",
    "list_template_specs",
    "load_template_module",
    "load_template_spec",
]
