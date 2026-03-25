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
class TemplateRuntimeMetadataBinding:
    runtime_name: str
    metadata_field: str


@dataclass(slots=True, frozen=True)
class TemplateClusterGeometrySupport:
    supported: bool
    mapping_target: str = "component_weights"
    metadata_fields: tuple[str, ...] = ()
    runtime_bindings: tuple[TemplateRuntimeMetadataBinding, ...] = ()
    allowed_sf_approximations: tuple[str, ...] = (
        "sphere",
        "ellipsoid",
    )
    dynamic_parameters: bool = False
    sphere_parameter_prefix: str = "r_eff"
    ellipsoid_parameter_prefixes: tuple[str, str, str] = (
        "a_eff",
        "b_eff",
        "c_eff",
    )

    @property
    def runtime_input_names(self) -> tuple[str, ...]:
        return tuple(binding.runtime_name for binding in self.runtime_bindings)


@dataclass(slots=True, frozen=True)
class TemplateSpec:
    name: str
    module_path: Path
    metadata_path: Path | None
    deprecated: bool
    display_name: str
    description: str
    lmfit_model_name: str
    dream_model_name: str
    lmfit_inputs: tuple[str, ...]
    dream_inputs: tuple[str, ...]
    param_columns: tuple[str, ...]
    parameters: tuple[TemplateParameter, ...]
    cluster_geometry_support: TemplateClusterGeometrySupport

    @property
    def label(self) -> str:
        return self.display_name

    @property
    def extra_lmfit_inputs(self) -> tuple[str, ...]:
        if len(self.lmfit_inputs) < 4:
            return ()
        return self.lmfit_inputs[3:-1]


def default_template_dir() -> Path:
    return Path(__file__).resolve().parent


def _normalize_template_dir(template_dir: str | Path | None) -> Path:
    if template_dir is None:
        return default_template_dir()
    return Path(template_dir).expanduser().resolve()


def _deprecated_template_dir(template_dir: Path) -> Path:
    return template_dir / "_deprecated"


def _template_search_paths(
    template_dir: Path,
    *,
    include_deprecated: bool,
) -> list[Path]:
    search_paths = [template_dir]
    deprecated_dir = _deprecated_template_dir(template_dir)
    if include_deprecated and deprecated_dir.is_dir():
        search_paths.append(deprecated_dir)
    return search_paths


def _iter_template_module_paths(
    template_dir: Path,
    *,
    include_deprecated: bool,
) -> list[Path]:
    paths: list[Path] = []
    for search_dir in _template_search_paths(
        template_dir,
        include_deprecated=include_deprecated,
    ):
        paths.extend(
            sorted(
                path
                for path in search_dir.glob("*.py")
                if path.name != "__init__.py"
            )
        )
    return paths


def _resolve_template_module_path(
    template_name: str,
    template_dir: Path,
) -> Path | None:
    candidate_paths = [
        template_dir / f"{template_name}.py",
        _deprecated_template_dir(template_dir) / f"{template_name}.py",
    ]
    for candidate in candidate_paths:
        if candidate.is_file():
            return candidate
    return None


def list_template_specs(
    template_dir: str | Path | None = None,
    *,
    include_deprecated: bool = False,
) -> list[TemplateSpec]:
    resolved_dir = _normalize_template_dir(template_dir)
    seen_names: set[str] = set()
    specs: list[TemplateSpec] = []
    for path in _iter_template_module_paths(
        resolved_dir,
        include_deprecated=include_deprecated,
    ):
        if path.stem in seen_names:
            continue
        specs.append(load_template_spec(path.stem, resolved_dir))
        seen_names.add(path.stem)
    return specs


@lru_cache(maxsize=None)
def load_template_spec(
    template_name: str,
    template_dir: str | Path | None = None,
) -> TemplateSpec:
    resolved_dir = _normalize_template_dir(template_dir)
    module_path = _resolve_template_module_path(template_name, resolved_dir)
    if module_path is None:
        raise FileNotFoundError(
            f"Unknown SAXS model template: {template_name}"
        )
    directives = _parse_directives(module_path)
    metadata_path = module_path.with_suffix(".json")
    metadata = _load_template_metadata(
        metadata_path,
        template_name,
        directives=directives,
    )
    spec = TemplateSpec(
        name=template_name,
        module_path=module_path,
        metadata_path=metadata_path if metadata_path.is_file() else None,
        deprecated=module_path.parent.name == "_deprecated",
        display_name=str(metadata["display_name"]),
        description=str(metadata["description"]),
        lmfit_model_name=directives["model_lmfit"],
        dream_model_name=directives["model_pydream"],
        lmfit_inputs=_split_csv(directives["inputs_lmfit"]),
        dream_inputs=_split_csv(directives["inputs_pydream"]),
        param_columns=_split_csv(directives["param_columns"]),
        parameters=tuple(_parse_param_lines(module_path)),
        cluster_geometry_support=metadata["cluster_geometry_support"],
    )
    _validate_template_runtime_contract(spec)
    return spec


@lru_cache(maxsize=None)
def load_template_module(
    template_name: str,
    template_dir: str | Path | None = None,
) -> ModuleType:
    resolved_dir = _normalize_template_dir(template_dir)
    spec = load_template_spec(template_name, resolved_dir)
    relative_parts = (
        spec.module_path.relative_to(resolved_dir).with_suffix("").parts
    )
    import_spec = importlib.util.spec_from_file_location(
        ("saxshell.saxs._model_templates." + ".".join(relative_parts)),
        spec.module_path,
    )
    if import_spec is None or import_spec.loader is None:
        raise ImportError(
            f"Unable to import template module from {spec.module_path}"
        )
    module = importlib.util.module_from_spec(import_spec)
    import_spec.loader.exec_module(module)
    return module


def clear_template_caches() -> None:
    load_template_spec.cache_clear()
    load_template_module.cache_clear()


def _parse_directives(module_path: Path) -> dict[str, str]:
    directives: dict[str, str] = {}
    required = {
        "model_lmfit",
        "model_pydream",
        "inputs_lmfit",
        "inputs_pydream",
        "param_columns",
    }
    optional = {
        "cluster_geometry_metadata",
    }
    recognized = required | optional
    with module_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped.startswith("# "):
                continue
            if ":" not in stripped:
                continue
            key, raw_value = stripped[2:].split(":", 1)
            key = key.strip()
            if key in recognized:
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
    *,
    directives: dict[str, str],
) -> dict[str, object]:
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
            "cluster_geometry_support": TemplateClusterGeometrySupport(
                supported=False
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
    cluster_geometry_support = _parse_cluster_geometry_support(
        payload,
        metadata_path=metadata_path,
        directives=directives,
    )
    return {
        "display_name": display_name,
        "description": description,
        "cluster_geometry_support": cluster_geometry_support,
    }


def _validate_template_runtime_contract(spec: TemplateSpec) -> None:
    if not spec.cluster_geometry_support.supported:
        return

    runtime_names = set(spec.cluster_geometry_support.runtime_input_names)
    missing_from_lmfit = sorted(
        runtime_name
        for runtime_name in runtime_names
        if runtime_name not in spec.extra_lmfit_inputs
    )
    if missing_from_lmfit:
        raise ValueError(
            f"Template {spec.name} declares cluster geometry runtime "
            "bindings that are missing from inputs_lmfit: "
            + ", ".join(missing_from_lmfit)
        )

    missing_from_dream = sorted(
        runtime_name
        for runtime_name in runtime_names
        if runtime_name not in spec.dream_inputs
    )
    if missing_from_dream:
        raise ValueError(
            f"Template {spec.name} declares cluster geometry runtime "
            "bindings that are missing from inputs_pydream: "
            + ", ".join(missing_from_dream)
        )


def _parse_cluster_geometry_support(
    payload: dict[str, object],
    *,
    metadata_path: Path,
    directives: dict[str, str],
) -> TemplateClusterGeometrySupport:
    capabilities = payload.get("capabilities", {})
    if not isinstance(capabilities, dict):
        capabilities = {}
    raw_support = capabilities.get("cluster_geometry_metadata", {})
    if raw_support is None:
        raw_support = {}
    if not isinstance(raw_support, dict):
        raise ValueError(
            f"Template metadata file {metadata_path.name} defines "
            "'capabilities.cluster_geometry_metadata' with an invalid "
            "schema."
        )

    supported = bool(raw_support.get("supported", False))
    directive_text = directives.get("cluster_geometry_metadata")
    if supported and directive_text is None:
        raise ValueError(
            f"Template metadata file {metadata_path.name} enables "
            "cluster_geometry_metadata but the paired Python template "
            "header does not declare '# cluster_geometry_metadata: true'."
        )
    if directive_text is not None:
        directive_supported = _parse_bool_directive(
            directive_text,
            field_name="cluster_geometry_metadata",
            source_name=metadata_path.name,
        )
        if directive_supported != supported:
            raise ValueError(
                f"Template metadata file {metadata_path.name} and the paired "
                "Python template header disagree about "
                "cluster_geometry_metadata support."
            )
    if not supported:
        return TemplateClusterGeometrySupport(supported=False)

    mapping_target = (
        str(raw_support.get("mapping_target", "component_weights")).strip()
        or "component_weights"
    )
    if mapping_target != "component_weights":
        raise ValueError(
            f"Template metadata file {metadata_path.name} uses unsupported "
            f"cluster geometry mapping_target {mapping_target!r}."
        )

    metadata_fields = tuple(
        str(field_name).strip()
        for field_name in raw_support.get("metadata_fields", [])
        if str(field_name).strip()
    )
    if not metadata_fields:
        raise ValueError(
            f"Template metadata file {metadata_path.name} must declare at "
            "least one cluster geometry metadata field when support is "
            "enabled."
        )

    runtime_bindings_payload = raw_support.get("runtime_bindings", {})
    if not isinstance(runtime_bindings_payload, dict):
        raise ValueError(
            f"Template metadata file {metadata_path.name} defines "
            "'runtime_bindings' with an invalid schema."
        )
    runtime_bindings = tuple(
        TemplateRuntimeMetadataBinding(
            runtime_name=str(runtime_name).strip(),
            metadata_field=str(metadata_field).strip(),
        )
        for runtime_name, metadata_field in runtime_bindings_payload.items()
        if str(runtime_name).strip() and str(metadata_field).strip()
    )
    if not runtime_bindings:
        raise ValueError(
            f"Template metadata file {metadata_path.name} must declare "
            "runtime_bindings when cluster geometry metadata support is "
            "enabled."
        )
    invalid_fields = [
        binding.metadata_field
        for binding in runtime_bindings
        if binding.metadata_field not in metadata_fields
    ]
    if invalid_fields:
        raise ValueError(
            f"Template metadata file {metadata_path.name} references unknown "
            "cluster geometry metadata fields in runtime_bindings: "
            + ", ".join(sorted(set(invalid_fields)))
        )

    allowed_sf_payload = raw_support.get(
        "allowed_sf_approximations",
        ["sphere", "ellipsoid"],
    )
    if isinstance(allowed_sf_payload, str):
        allowed_sf_payload = [allowed_sf_payload]
    allowed_sf_approximations = tuple(
        str(option).strip().lower()
        for option in allowed_sf_payload
        if str(option).strip()
    )
    if not allowed_sf_approximations:
        raise ValueError(
            f"Template metadata file {metadata_path.name} must declare at "
            "least one allowed_sf_approximations entry when cluster "
            "geometry support is enabled."
        )
    invalid_approximations = [
        option
        for option in allowed_sf_approximations
        if option not in {"sphere", "ellipsoid"}
    ]
    if invalid_approximations:
        raise ValueError(
            f"Template metadata file {metadata_path.name} declares "
            "unsupported allowed_sf_approximations values: "
            + ", ".join(sorted(set(invalid_approximations)))
        )
    allowed_sf_approximations = tuple(dict.fromkeys(allowed_sf_approximations))

    dynamic_parameters = bool(raw_support.get("dynamic_parameters", False))
    sphere_parameter_prefix = (
        str(raw_support.get("sphere_parameter_prefix", "r_eff")).strip()
        or "r_eff"
    )
    ellipsoid_prefixes_payload = raw_support.get(
        "ellipsoid_parameter_prefixes",
        ["a_eff", "b_eff", "c_eff"],
    )
    if isinstance(ellipsoid_prefixes_payload, str):
        ellipsoid_prefixes_payload = [ellipsoid_prefixes_payload]
    ellipsoid_parameter_prefixes = tuple(
        str(prefix).strip()
        for prefix in ellipsoid_prefixes_payload
        if str(prefix).strip()
    )
    if dynamic_parameters and len(ellipsoid_parameter_prefixes) != 3:
        raise ValueError(
            f"Template metadata file {metadata_path.name} must declare "
            "exactly three ellipsoid_parameter_prefixes when "
            "dynamic_parameters is enabled."
        )

    return TemplateClusterGeometrySupport(
        supported=True,
        mapping_target=mapping_target,
        metadata_fields=metadata_fields,
        runtime_bindings=runtime_bindings,
        allowed_sf_approximations=allowed_sf_approximations,
        dynamic_parameters=dynamic_parameters,
        sphere_parameter_prefix=sphere_parameter_prefix,
        ellipsoid_parameter_prefixes=(
            ellipsoid_parameter_prefixes
            if ellipsoid_parameter_prefixes
            else ("a_eff", "b_eff", "c_eff")
        ),
    )


def _parse_bool_directive(
    value: str,
    *,
    field_name: str,
    source_name: str,
) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"{source_name} defines {field_name!r} with unsupported boolean "
        f"value {value!r}."
    )


__all__ = [
    "TemplateClusterGeometrySupport",
    "TemplateParameter",
    "TemplateRuntimeMetadataBinding",
    "TemplateSpec",
    "default_template_dir",
    "list_template_specs",
    "clear_template_caches",
    "load_template_module",
    "load_template_spec",
]
