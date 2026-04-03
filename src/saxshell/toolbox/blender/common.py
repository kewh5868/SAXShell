from __future__ import annotations

import json
import re
from dataclasses import dataclass
from itertools import combinations_with_replacement
from pathlib import Path

CPK_COLORS: dict[str, tuple[float, float, float, float]] = {
    "H": (1.0, 1.0, 1.0, 1.0),
    "C": (0.20, 0.20, 0.20, 1.0),
    "N": (0.19, 0.31, 0.97, 1.0),
    "O": (0.95, 0.05, 0.05, 1.0),
    "F": (0.56, 0.88, 0.31, 1.0),
    "P": (1.0, 0.50, 0.0, 1.0),
    "S": (1.0, 0.90, 0.19, 1.0),
    "Cl": (0.12, 0.94, 0.12, 1.0),
    "Br": (0.65, 0.16, 0.16, 1.0),
    "I": (0.58, 0.0, 0.58, 1.0),
    "B": (1.0, 0.71, 0.71, 1.0),
    "Li": (0.80, 0.50, 1.0, 1.0),
    "Na": (0.67, 0.36, 0.95, 1.0),
    "Mg": (0.54, 1.0, 0.0, 1.0),
    "Al": (0.75, 0.65, 0.65, 1.0),
    "Si": (0.94, 0.78, 0.63, 1.0),
    "K": (0.56, 0.25, 0.83, 1.0),
    "Ca": (0.24, 1.0, 0.0, 1.0),
    "Ti": (0.75, 0.76, 0.78, 1.0),
    "Cr": (0.54, 0.60, 0.78, 1.0),
    "Mn": (0.61, 0.48, 0.78, 1.0),
    "Fe": (0.88, 0.40, 0.20, 1.0),
    "Co": (0.94, 0.56, 0.63, 1.0),
    "Ni": (0.31, 0.82, 0.31, 1.0),
    "Cu": (0.78, 0.50, 0.20, 1.0),
    "Zn": (0.49, 0.50, 0.69, 1.0),
    "Se": (1.0, 0.63, 0.0, 1.0),
    "Ag": (0.75, 0.75, 0.75, 1.0),
    "Cd": (1.0, 0.85, 0.56, 1.0),
    "Sn": (0.40, 0.50, 0.50, 1.0),
    "Au": (1.0, 0.82, 0.14, 1.0),
    "Hg": (0.72, 0.72, 0.82, 1.0),
    "Pb": (0.34, 0.35, 0.38, 1.0),
}

COVALENT_RADII: dict[str, float] = {
    "H": 0.31,
    "B": 0.85,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "P": 1.07,
    "S": 1.05,
    "Cl": 1.02,
    "Br": 1.20,
    "I": 1.39,
    "Li": 1.28,
    "Na": 1.66,
    "Mg": 1.41,
    "Al": 1.21,
    "Si": 1.11,
    "K": 2.03,
    "Ca": 1.76,
    "Ti": 1.60,
    "Cr": 1.39,
    "Mn": 1.39,
    "Fe": 1.32,
    "Co": 1.26,
    "Ni": 1.24,
    "Cu": 1.32,
    "Zn": 1.22,
    "Se": 1.20,
    "Ag": 1.45,
    "Cd": 1.44,
    "Sn": 1.39,
    "Au": 1.36,
    "Hg": 1.32,
    "Pb": 1.46,
}

DEFAULT_ATOM_COLOR = (0.55, 0.55, 0.60, 1.0)
DEFAULT_BOND_COLOR = (0.44, 0.48, 0.54, 1.0)
DEFAULT_TEXT_COLOR = (0.12, 0.14, 0.18, 1.0)
DEFAULT_OUTPUT_EXTENSION = ".png"
DEFAULT_BLEND_EXTENSION = ".blend"
DEFAULT_BOND_THRESHOLD_SCALE = 1.18
DEFAULT_BOND_MIN_LENGTH = 0.0
DEFAULT_RENDER_WIDTH = 2600
DEFAULT_RENDER_HEIGHT = 2000
DEFAULT_RENDER_SAMPLES = 256
DEFAULT_ATOM_STYLE = "paper_gloss"
DEFAULT_LEGEND_FONT = "Arial"
ATOM_STYLE_LABELS: dict[str, str] = {
    "paper_gloss": "Paper Gloss",
    "vesta": "VESTA Ball and Stick",
    "soft_studio": "Soft Studio",
    "flat_diagram": "Flat Diagram",
    "toon_matte": "Toon Matte",
    "poster_pop": "Poster Pop",
    "pastel_cartoon": "Pastel Cartoon",
    "crystal_flat": "Crystal Flat",
    "crystal_cartoon": "Crystal Cartoon",
    "crystal_shadow_gloss": "Crystal Shadow Gloss",
    "cpk": "CPK Ball and Stick",
    "monochrome": "Monochrome Publication",
}
ATOM_STYLE_CHOICES = tuple(ATOM_STYLE_LABELS)
ATOM_STYLE_DESCRIPTIONS: dict[str, str] = {
    "paper_gloss": (
        "Glossy, paper-style spheres with tighter ball sizes, brighter studio "
        "color, and slender split-color bonds."
    ),
    "vesta": (
        "A familiar VESTA-like ball-and-stick look with larger atoms and clear bond "
        "separation."
    ),
    "soft_studio": (
        "A softer high-key publication preset with gentler highlights, warmer "
        "bonds, and a slightly fuller ball-and-stick profile."
    ),
    "flat_diagram": (
        "A high-key schematic preset with slimmer neutral bonds and flatter atom "
        "surfaces."
    ),
    "toon_matte": (
        "A true flat-cartoon preset tuned to match the visualizer more closely, "
        "with brighter color blocks, matte surfaces, and clean neutral bonds."
    ),
    "poster_pop": (
        "A bold illustrative preset with brighter punchy colors, lighter sheen, "
        "and presentation-friendly split bonds."
    ),
    "pastel_cartoon": (
        "A softer high-key cartoon preset with pastel atoms, matte shading, and "
        "gentle bond contrast."
    ),
    "crystal_flat": (
        "A flat schematic preset with brighter atom colors, "
        "clean neutral bonds, and minimal gloss."
    ),
    "crystal_cartoon": (
        "A punchy cartoon preset with brighter colors, simple highlights, and "
        "crisp split-color bonds."
    ),
    "crystal_shadow_gloss": (
        "A glossy preset with brighter directional shading, clean highlights, "
        "and stronger ball-and-stick depth."
    ),
    "cpk": (
        "Classic CPK-inspired colorization with cleaner reflections, brighter "
        "surfaces, and a balanced ball-and-stick ratio."
    ),
    "monochrome": (
        "A grayscale publication preset with subdued bond treatment and neutral "
        "surfaces."
    ),
}
ATOM_STYLE_DEFAULTS: dict[str, dict[str, float | str]] = {
    "paper_gloss": {
        "atom_scale": 0.82,
        "bond_radius": 0.11,
        "bond_color_mode": "split",
    },
    "vesta": {
        "atom_scale": 0.74,
        "bond_radius": 0.16,
        "bond_color_mode": "split",
    },
    "soft_studio": {
        "atom_scale": 0.78,
        "bond_radius": 0.14,
        "bond_color_mode": "split",
    },
    "flat_diagram": {
        "atom_scale": 0.72,
        "bond_radius": 0.10,
        "bond_color_mode": "neutral",
    },
    "toon_matte": {
        "atom_scale": 0.76,
        "bond_radius": 0.13,
        "bond_color_mode": "neutral",
    },
    "poster_pop": {
        "atom_scale": 0.74,
        "bond_radius": 0.12,
        "bond_color_mode": "split",
    },
    "pastel_cartoon": {
        "atom_scale": 0.80,
        "bond_radius": 0.12,
        "bond_color_mode": "neutral",
    },
    "crystal_flat": {
        "atom_scale": 0.78,
        "bond_radius": 0.11,
        "bond_color_mode": "neutral",
    },
    "crystal_cartoon": {
        "atom_scale": 0.78,
        "bond_radius": 0.13,
        "bond_color_mode": "split",
    },
    "crystal_shadow_gloss": {
        "atom_scale": 0.82,
        "bond_radius": 0.12,
        "bond_color_mode": "split",
    },
    "cpk": {
        "atom_scale": 0.68,
        "bond_radius": 0.14,
        "bond_color_mode": "split",
    },
    "monochrome": {
        "atom_scale": 0.70,
        "bond_radius": 0.15,
        "bond_color_mode": "neutral",
    },
}
DEFAULT_RENDER_QUALITY = "high"
RENDER_QUALITY_LABELS: dict[str, str] = {
    "high": "High Quality",
    "balanced": "Balanced",
    "draft": "Draft",
}
RENDER_QUALITY_CHOICES = tuple(RENDER_QUALITY_LABELS)
DEFAULT_LIGHTING_LEVEL = 2
LIGHTING_LEVEL_LABELS: dict[int, str] = {
    1: "1 - Lowest",
    2: "2 - Current",
    3: "3 - Bright",
    4: "4 - Brighter",
    5: "5 - Brightest",
}
LIGHTING_LEVEL_CHOICES = tuple(LIGHTING_LEVEL_LABELS)
RENDER_QUALITY_DEFAULTS: dict[str, dict[str, int | float]] = {
    "high": {
        "samples": 512,
        "camera_margin": 1.08,
    },
    "balanced": {
        "samples": 256,
        "camera_margin": 1.10,
    },
    "draft": {
        "samples": 128,
        "camera_margin": 1.12,
    },
}


@dataclass(slots=True, frozen=True)
class AtomAppearanceOverride:
    element: str
    color: tuple[float, float, float, float]
    size_scale: float

    def __post_init__(self) -> None:
        element = normalize_element_symbol(self.element)
        color = tuple(
            _clamp_color_channel(channel)
            for channel in (
                tuple(self.color)
                if isinstance(self.color, (list, tuple))
                else DEFAULT_ATOM_COLOR
            )[:4]
        )
        if len(color) < 4:
            color = tuple(color) + (1.0,) * (4 - len(color))
        object.__setattr__(self, "element", element)
        object.__setattr__(
            self,
            "color",
            (
                float(color[0]),
                float(color[1]),
                float(color[2]),
                float(color[3]),
            ),
        )
        object.__setattr__(self, "size_scale", max(float(self.size_scale), 0.05))


@dataclass(slots=True, frozen=True)
class CustomAestheticSpec:
    key: str
    name: str
    base_style: str
    overrides: tuple[AtomAppearanceOverride, ...] = ()

    def __post_init__(self) -> None:
        key = sanitize_custom_aesthetic_key(self.key or self.name)
        name = str(self.name).strip() or key.replace("_", " ").title()
        base_style = normalize_builtin_atom_style(self.base_style)
        override_map: dict[str, AtomAppearanceOverride] = {}
        for override in self.overrides:
            if isinstance(override, AtomAppearanceOverride):
                normalized = override
            elif isinstance(override, dict):
                normalized = AtomAppearanceOverride(**override)
            else:
                raise TypeError(
                    "Custom aesthetic overrides must be AtomAppearanceOverride "
                    "instances or dictionaries."
                )
            override_map[normalized.element] = normalized
        object.__setattr__(self, "key", key)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "base_style", base_style)
        object.__setattr__(
            self,
            "overrides",
            tuple(override_map[key] for key in sorted(override_map)),
        )

    def override_lookup(self) -> dict[str, AtomAppearanceOverride]:
        return {override.element: override for override in self.overrides}


_CUSTOM_AESTHETICS: dict[str, CustomAestheticSpec] = {}


@dataclass(slots=True, frozen=True)
class OrientationSpec:
    key: str
    label: str
    source: str
    x_degrees: float
    y_degrees: float
    z_degrees: float
    enabled: bool = True
    atom_style: str | None = None
    render_quality: str | None = None
    lighting_level: int | None = None
    save_legend: bool = False

    @property
    def euler_degrees(self) -> tuple[float, float, float]:
        return (
            float(self.x_degrees),
            float(self.y_degrees),
            float(self.z_degrees),
        )

    def effective_atom_style(self, default: str | None = None) -> str:
        return normalize_atom_style(self.atom_style or default)

    def effective_render_quality(self, default: str | None = None) -> str:
        return normalize_render_quality(self.render_quality or default)

    def effective_lighting_level(
        self,
        default: int | str | None = None,
    ) -> int:
        return normalize_lighting_level(
            self.lighting_level if self.lighting_level is not None else default
        )


@dataclass(slots=True, frozen=True)
class BondThresholdSpec:
    element_a: str
    element_b: str
    min_length: float = DEFAULT_BOND_MIN_LENGTH
    max_length: float = 0.0

    def __post_init__(self) -> None:
        element_a, element_b = normalize_element_pair(
            self.element_a,
            self.element_b,
        )
        min_length = max(float(self.min_length), 0.0)
        max_length = max(float(self.max_length), min_length)
        object.__setattr__(self, "element_a", element_a)
        object.__setattr__(self, "element_b", element_b)
        object.__setattr__(self, "min_length", min_length)
        object.__setattr__(self, "max_length", max_length)

    @property
    def pair_key(self) -> tuple[str, str]:
        return (self.element_a, self.element_b)


def sanitize_orientation_key(text: str) -> str:
    value = re.sub(r"[^0-9A-Za-z._-]+", "_", text.strip())
    value = value.strip("._-")
    value = re.sub(r"_+", "_", value)
    return value.lower() or "orientation"


def sanitize_custom_aesthetic_key(text: str) -> str:
    slug = sanitize_orientation_key(text)
    if slug.startswith("custom_"):
        return slug
    return f"custom_{slug}"


def normalize_element_symbol(symbol: str) -> str:
    text = (symbol or "").strip()
    if not text:
        return "X"
    return text[:1].upper() + text[1:].lower()


def normalize_element_pair(
    element_a: str,
    element_b: str,
) -> tuple[str, str]:
    left = normalize_element_symbol(element_a)
    right = normalize_element_symbol(element_b)
    return tuple(sorted((left, right)))


def default_bond_max_length(
    element_a: str,
    element_b: str,
    *,
    threshold_scale: float = DEFAULT_BOND_THRESHOLD_SCALE,
) -> float:
    left, right = normalize_element_pair(element_a, element_b)
    return (
        COVALENT_RADII.get(left, 0.85) + COVALENT_RADII.get(right, 0.85)
    ) * float(threshold_scale)


def default_bond_threshold_spec(
    element_a: str,
    element_b: str,
    *,
    threshold_scale: float = DEFAULT_BOND_THRESHOLD_SCALE,
    min_length: float = DEFAULT_BOND_MIN_LENGTH,
) -> BondThresholdSpec:
    left, right = normalize_element_pair(element_a, element_b)
    return BondThresholdSpec(
        element_a=left,
        element_b=right,
        min_length=min_length,
        max_length=default_bond_max_length(
            left,
            right,
            threshold_scale=threshold_scale,
        ),
    )


def encode_bond_threshold_arg(spec: BondThresholdSpec) -> str:
    return (
        f"{spec.element_a}:{spec.element_b}:"
        f"{float(spec.min_length):.6f}:{float(spec.max_length):.6f}"
    )


def parse_bond_threshold_arg(text: str) -> BondThresholdSpec:
    parts = [part.strip() for part in text.split(":")]
    if len(parts) != 4:
        raise ValueError(
            "Bond threshold must use element_a:element_b:min_length:max_length format."
        )
    return BondThresholdSpec(
        element_a=parts[0],
        element_b=parts[1],
        min_length=float(parts[2]),
        max_length=float(parts[3]),
    )


def complete_bond_threshold_specs(
    elements: tuple[str, ...] | list[str],
    overrides: (
        tuple[BondThresholdSpec, ...] | list[BondThresholdSpec] | None
    ) = None,
    *,
    threshold_scale: float = DEFAULT_BOND_THRESHOLD_SCALE,
    min_length: float = DEFAULT_BOND_MIN_LENGTH,
) -> tuple[BondThresholdSpec, ...]:
    unique_elements = sorted(
        {
            normalize_element_symbol(element)
            for element in elements
            if str(element).strip()
        }
    )
    defaults = {
        normalize_element_pair(left, right): default_bond_threshold_spec(
            left,
            right,
            threshold_scale=threshold_scale,
            min_length=min_length,
        )
        for left, right in combinations_with_replacement(unique_elements, 2)
    }
    if overrides:
        for spec in overrides:
            defaults[spec.pair_key] = BondThresholdSpec(
                element_a=spec.element_a,
                element_b=spec.element_b,
                min_length=spec.min_length,
                max_length=spec.max_length,
            )
    return tuple(defaults[key] for key in sorted(defaults))


def bond_threshold_lookup(
    specs: tuple[BondThresholdSpec, ...] | list[BondThresholdSpec],
) -> dict[tuple[str, str], BondThresholdSpec]:
    return {
        spec.pair_key: BondThresholdSpec(
            element_a=spec.element_a,
            element_b=spec.element_b,
            min_length=spec.min_length,
            max_length=spec.max_length,
        )
        for spec in specs
    }


def format_angle_for_filename(value: float) -> str:
    text = f"{float(value):.1f}"
    if text == "-0.0":
        text = "0.0"
    return text


def orientation_filename_metadata(orientation: OrientationSpec) -> str:
    return (
        f"{sanitize_orientation_key(orientation.key)}"
        f"_rx{format_angle_for_filename(orientation.x_degrees)}"
        f"_ry{format_angle_for_filename(orientation.y_degrees)}"
        f"_rz{format_angle_for_filename(orientation.z_degrees)}"
    )


def build_render_output_path(
    input_path: str | Path,
    output_dir: str | Path,
    orientation: OrientationSpec,
    *,
    default_atom_style: str | None = None,
    default_render_quality: str | None = None,
) -> Path:
    return build_output_path(
        input_path,
        output_dir,
        orientation,
        extension=DEFAULT_OUTPUT_EXTENSION,
        default_atom_style=default_atom_style,
        default_render_quality=default_render_quality,
    )


def build_blend_output_path(
    input_path: str | Path,
    output_dir: str | Path,
    orientation: OrientationSpec,
    *,
    default_atom_style: str | None = None,
    default_render_quality: str | None = None,
) -> Path:
    return build_output_path(
        input_path,
        output_dir,
        orientation,
        extension=DEFAULT_BLEND_EXTENSION,
        default_atom_style=default_atom_style,
        default_render_quality=default_render_quality,
    )


def build_legend_output_path(
    input_path: str | Path,
    output_dir: str | Path,
    orientation: OrientationSpec,
    *,
    default_atom_style: str | None = None,
    default_render_quality: str | None = None,
) -> Path:
    return build_output_path(
        input_path,
        output_dir,
        orientation,
        extension=DEFAULT_OUTPUT_EXTENSION,
        stem_suffix="_legend",
        default_atom_style=default_atom_style,
        default_render_quality=default_render_quality,
    )


def build_output_path(
    input_path: str | Path,
    output_dir: str | Path,
    orientation: OrientationSpec,
    *,
    extension: str,
    stem_suffix: str = "",
    default_atom_style: str | None = None,
    default_render_quality: str | None = None,
) -> Path:
    source = Path(input_path).expanduser()
    directory = Path(output_dir).expanduser()
    stem_parts = [source.stem]
    effective_atom_style = orientation.effective_atom_style(default_atom_style)
    effective_render_quality = orientation.effective_render_quality(
        default_render_quality
    )
    if effective_atom_style:
        stem_parts.append(f"style_{effective_atom_style}")
    if effective_render_quality:
        stem_parts.append(f"quality_{effective_render_quality}")
    stem_parts.append(orientation_filename_metadata(orientation))
    return directory / f"{'_'.join(stem_parts)}{stem_suffix}{extension}"


def encode_orientation_arg(orientation: OrientationSpec) -> str:
    base = (
        f"{sanitize_orientation_key(orientation.key)}:"
        f"{orientation.x_degrees:.6f}:"
        f"{orientation.y_degrees:.6f}:"
        f"{orientation.z_degrees:.6f}"
    )
    if orientation.lighting_level is not None:
        return (
            f"{base}:{orientation.atom_style or ''}:"
            f"{orientation.render_quality or ''}:"
            f"{orientation.effective_lighting_level()}"
        )
    if orientation.atom_style or orientation.render_quality:
        return (
            f"{base}:{orientation.atom_style or ''}:"
            f"{orientation.render_quality or ''}"
        )
    return base


def parse_orientation_arg(text: str) -> OrientationSpec:
    parts = [part.strip() for part in text.split(":")]
    if len(parts) not in {4, 5, 6, 7}:
        raise ValueError(
            "Orientation must use key:x_degrees:y_degrees:z_degrees"
            "[:atom_style:render_quality[:lighting_level]] format."
        )
    key, x_text, y_text, z_text = parts[:4]
    atom_style_text = parts[4] if len(parts) > 4 else ""
    render_quality_text = parts[5] if len(parts) > 5 else ""
    lighting_level_text = parts[6] if len(parts) > 6 else ""
    label = key.replace("_", " ").strip().title() or "Orientation"
    return OrientationSpec(
        key=sanitize_orientation_key(key),
        label=label,
        source="cli",
        x_degrees=float(x_text),
        y_degrees=float(y_text),
        z_degrees=float(z_text),
        enabled=True,
        atom_style=(
            normalize_atom_style(atom_style_text) if atom_style_text else None
        ),
        render_quality=(
            normalize_render_quality(render_quality_text)
            if render_quality_text
            else None
        ),
        lighting_level=(
            normalize_lighting_level(lighting_level_text)
            if lighting_level_text
            else None
        ),
    )


def infer_title(
    input_path: str | Path,
    *,
    structure_comment: str,
    explicit_title: str | None,
    uppercase: bool = False,
) -> str:
    if explicit_title:
        title = explicit_title.strip()
    elif structure_comment:
        title = structure_comment.strip()
    else:
        title = Path(input_path).stem.replace("_", " ").replace("-", " ")
    title = title.strip() or "STRUCTURE"
    return title.upper() if uppercase else title


def normalize_atom_style(value: str | None) -> str:
    text = (value or "").strip().lower()
    if text in ATOM_STYLE_CHOICES or text in _CUSTOM_AESTHETICS:
        return text
    return DEFAULT_ATOM_STYLE


def normalize_builtin_atom_style(value: str | None) -> str:
    text = (value or "").strip().lower()
    if text in ATOM_STYLE_CHOICES:
        return text
    return DEFAULT_ATOM_STYLE


def atom_style_is_custom(value: str | None) -> bool:
    text = (value or "").strip().lower()
    return text in _CUSTOM_AESTHETICS


def atom_style_base(value: str | None) -> str:
    style = normalize_atom_style(value)
    custom = _CUSTOM_AESTHETICS.get(style)
    if custom is not None:
        return custom.base_style
    return normalize_builtin_atom_style(style)


def atom_style_defaults(atom_style: str) -> dict[str, float | str]:
    return ATOM_STYLE_DEFAULTS[atom_style_base(atom_style)]


def atom_style_label(atom_style: str) -> str:
    style = normalize_atom_style(atom_style)
    custom = _CUSTOM_AESTHETICS.get(style)
    if custom is not None:
        return custom.name
    return ATOM_STYLE_LABELS[style]


def atom_style_description(atom_style: str) -> str:
    style = normalize_atom_style(atom_style)
    custom = _CUSTOM_AESTHETICS.get(style)
    if custom is not None:
        return (
            f"Custom aesthetic based on {ATOM_STYLE_LABELS[custom.base_style]} "
            "with saved per-element atom colors and sizes."
        )
    return ATOM_STYLE_DESCRIPTIONS[style]


def available_atom_style_labels() -> dict[str, str]:
    labels = dict(ATOM_STYLE_LABELS)
    for spec in sorted(
        _CUSTOM_AESTHETICS.values(),
        key=lambda item: (item.name.lower(), item.key),
    ):
        labels[spec.key] = spec.name
    return labels


def custom_aesthetics() -> tuple[CustomAestheticSpec, ...]:
    return tuple(
        spec
        for spec in sorted(
            _CUSTOM_AESTHETICS.values(),
            key=lambda item: (item.name.lower(), item.key),
        )
    )


def get_custom_aesthetic(atom_style: str) -> CustomAestheticSpec | None:
    return _CUSTOM_AESTHETICS.get(normalize_atom_style(atom_style))


def set_custom_aesthetics(
    specs: tuple[CustomAestheticSpec, ...] | list[CustomAestheticSpec],
) -> tuple[CustomAestheticSpec, ...]:
    normalized: dict[str, CustomAestheticSpec] = {}
    for spec in specs:
        if isinstance(spec, CustomAestheticSpec):
            normalized[spec.key] = spec
        elif isinstance(spec, dict):
            loaded = deserialize_custom_aesthetic(spec)
            normalized[loaded.key] = loaded
        else:
            raise TypeError(
                "Custom aesthetics must be CustomAestheticSpec instances or dictionaries."
            )
    _CUSTOM_AESTHETICS.clear()
    _CUSTOM_AESTHETICS.update(normalized)
    return custom_aesthetics()


def serialize_custom_aesthetic(spec: CustomAestheticSpec) -> dict[str, object]:
    return {
        "key": spec.key,
        "name": spec.name,
        "base_style": spec.base_style,
        "overrides": [
            {
                "element": override.element,
                "color": list(override.color),
                "size_scale": float(override.size_scale),
            }
            for override in spec.overrides
        ],
    }


def deserialize_custom_aesthetic(data: object) -> CustomAestheticSpec:
    if not isinstance(data, dict):
        raise ValueError("Custom aesthetic payload must be a dictionary.")
    raw_overrides = data.get("overrides", [])
    if not isinstance(raw_overrides, list):
        raise ValueError("Custom aesthetic overrides must be a list.")
    return CustomAestheticSpec(
        key=str(data.get("key", "")),
        name=str(data.get("name", "")),
        base_style=str(data.get("base_style", DEFAULT_ATOM_STYLE)),
        overrides=tuple(
            AtomAppearanceOverride(
                element=str(item.get("element", "")),
                color=tuple(item.get("color", DEFAULT_ATOM_COLOR)),
                size_scale=float(item.get("size_scale", 0.0)),
            )
            for item in raw_overrides
            if isinstance(item, dict)
        ),
    )


def encode_custom_aesthetic_arg(spec: CustomAestheticSpec) -> str:
    return json.dumps(
        serialize_custom_aesthetic(spec),
        sort_keys=True,
        separators=(",", ":"),
    )


def parse_custom_aesthetic_arg(text: str) -> CustomAestheticSpec:
    return deserialize_custom_aesthetic(json.loads(str(text)))


def style_atom_size_scale(
    element: str,
    *,
    atom_style: str,
) -> float:
    style = normalize_atom_style(atom_style)
    custom = _CUSTOM_AESTHETICS.get(style)
    if custom is not None:
        override = custom.override_lookup().get(normalize_element_symbol(element))
        if override is not None:
            return float(override.size_scale)
        return float(ATOM_STYLE_DEFAULTS[custom.base_style]["atom_scale"])
    return float(ATOM_STYLE_DEFAULTS[style]["atom_scale"])


def style_display_radius(
    element: str,
    *,
    atom_style: str,
    atom_scale_override: float | None = None,
) -> float:
    base = COVALENT_RADII.get(normalize_element_symbol(element), 0.85)
    scale = (
        float(atom_scale_override)
        if atom_scale_override is not None
        else style_atom_size_scale(element, atom_style=atom_style)
    )
    return max(base * scale, 0.18)


def normalize_render_quality(value: str | None) -> str:
    text = (value or "").strip().lower()
    if text in RENDER_QUALITY_CHOICES:
        return text
    return DEFAULT_RENDER_QUALITY


def normalize_lighting_level(value: int | str | None) -> int:
    try:
        level = int(value) if value is not None else DEFAULT_LIGHTING_LEVEL
    except (TypeError, ValueError):
        return DEFAULT_LIGHTING_LEVEL
    return min(max(level, 1), 5)


def _clamp_color_channel(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _lift_color(
    color: tuple[float, float, float, float],
    *,
    scale: float,
    bias: float,
) -> tuple[float, float, float, float]:
    return (
        _clamp_color_channel(color[0] * scale + bias),
        _clamp_color_channel(color[1] * scale + bias),
        _clamp_color_channel(color[2] * scale + bias),
        color[3],
    )


def _mix_color(
    color: tuple[float, float, float, float],
    *,
    mix: float,
    target: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> tuple[float, float, float, float]:
    return (
        _clamp_color_channel(color[0] * (1.0 - mix) + target[0] * mix),
        _clamp_color_channel(color[1] * (1.0 - mix) + target[1] * mix),
        _clamp_color_channel(color[2] * (1.0 - mix) + target[2] * mix),
        color[3],
    )


def _relative_luminance(
    color: tuple[float, float, float, float],
) -> float:
    return (
        0.2126 * float(color[0])
        + 0.7152 * float(color[1])
        + 0.0722 * float(color[2])
    )


def _adaptive_mix_color(
    color: tuple[float, float, float, float],
    *,
    mix: float,
    minimum_mix: float = 0.0,
    target: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> tuple[float, float, float, float]:
    luminance = _relative_luminance(color)
    effective_mix = max(
        float(minimum_mix),
        float(mix) * max(0.0, 1.0 - luminance),
    )
    return _mix_color(color, mix=effective_mix, target=target)


def style_atom_color(
    element: str,
    *,
    atom_style: str,
) -> tuple[float, float, float, float]:
    style = normalize_atom_style(atom_style)
    normalized_element = normalize_element_symbol(element)
    custom = _CUSTOM_AESTHETICS.get(style)
    if custom is not None:
        override = custom.override_lookup().get(normalized_element)
        if override is not None:
            return override.color
        style = custom.base_style
    color = CPK_COLORS.get(normalized_element, DEFAULT_ATOM_COLOR)
    if style == "paper_gloss":
        if normalized_element == "C":
            return (0.30, 0.38, 0.68, 1.0)
        return _adaptive_mix_color(color, mix=0.16)
    if style == "soft_studio":
        return _adaptive_mix_color(color, mix=0.22, minimum_mix=0.02)
    if style == "flat_diagram":
        return _adaptive_mix_color(color, mix=0.10)
    if style == "toon_matte":
        return _adaptive_mix_color(color, mix=0.12, minimum_mix=0.01)
    if style == "poster_pop":
        if normalized_element == "C":
            return (0.22, 0.29, 0.52, 1.0)
        return _adaptive_mix_color(color, mix=0.10, minimum_mix=0.01)
    if style == "pastel_cartoon":
        return _adaptive_mix_color(
            color,
            mix=0.28,
            minimum_mix=0.03,
            target=(1.0, 0.98, 0.96),
        )
    if style == "crystal_flat":
        return _adaptive_mix_color(color, mix=0.12)
    if style == "crystal_cartoon":
        return _adaptive_mix_color(color, mix=0.08, minimum_mix=0.01)
    if style == "crystal_shadow_gloss":
        return _adaptive_mix_color(
            color,
            mix=0.14,
            minimum_mix=0.02,
            target=(1.0, 0.99, 0.98),
        )
    if style == "monochrome":
        if normalized_element == "H":
            return (0.97, 0.97, 0.98, 1.0)
        return (0.78, 0.80, 0.83, 1.0)
    if style == "vesta":
        if normalized_element == "C":
            return (0.46, 0.48, 0.52, 1.0)
        return _adaptive_mix_color(color, mix=0.08)
    if style == "cpk":
        return _adaptive_mix_color(color, mix=0.06)
    return color


def style_neutral_bond_color(
    atom_style: str,
) -> tuple[float, float, float, float]:
    style = atom_style_base(atom_style)
    if style == "paper_gloss":
        return (0.54, 0.50, 0.57, 1.0)
    if style == "soft_studio":
        return (0.56, 0.51, 0.47, 1.0)
    if style == "flat_diagram":
        return (0.54, 0.58, 0.63, 1.0)
    if style == "toon_matte":
        return (0.50, 0.54, 0.60, 1.0)
    if style == "poster_pop":
        return (0.47, 0.51, 0.57, 1.0)
    if style == "pastel_cartoon":
        return (0.62, 0.58, 0.54, 1.0)
    if style == "crystal_flat":
        return (0.53, 0.57, 0.62, 1.0)
    if style == "crystal_cartoon":
        return (0.49, 0.55, 0.63, 1.0)
    if style == "crystal_shadow_gloss":
        return (0.57, 0.51, 0.57, 1.0)
    if style == "monochrome":
        return (0.60, 0.61, 0.64, 1.0)
    if style == "cpk":
        return (0.47, 0.56, 0.66, 1.0)
    if style == "vesta":
        return (0.52, 0.55, 0.60, 1.0)
    return DEFAULT_BOND_COLOR


def style_split_bond_color(
    element: str,
    *,
    atom_style: str,
) -> tuple[float, float, float, float]:
    style = atom_style_base(atom_style)
    atom_color = style_atom_color(element, atom_style=atom_style)
    if style == "monochrome":
        return (0.58, 0.60, 0.63, 1.0)
    if style in {"toon_matte", "pastel_cartoon"}:
        return _lift_color(atom_color, scale=0.84, bias=0.10)
    if style in {"poster_pop", "crystal_cartoon"}:
        return _lift_color(atom_color, scale=0.88, bias=0.08)
    if style in {"flat_diagram", "crystal_flat"}:
        return _lift_color(atom_color, scale=0.82, bias=0.10)
    return _lift_color(atom_color, scale=0.80, bias=0.08)


__all__ = [
    "ATOM_STYLE_CHOICES",
    "ATOM_STYLE_DEFAULTS",
    "ATOM_STYLE_DESCRIPTIONS",
    "ATOM_STYLE_LABELS",
    "AtomAppearanceOverride",
    "BondThresholdSpec",
    "CPK_COLORS",
    "COVALENT_RADII",
    "CustomAestheticSpec",
    "DEFAULT_ATOM_COLOR",
    "DEFAULT_ATOM_STYLE",
    "DEFAULT_BOND_COLOR",
    "DEFAULT_BOND_MIN_LENGTH",
    "DEFAULT_BOND_THRESHOLD_SCALE",
    "DEFAULT_BLEND_EXTENSION",
    "DEFAULT_LEGEND_FONT",
    "DEFAULT_LIGHTING_LEVEL",
    "DEFAULT_OUTPUT_EXTENSION",
    "DEFAULT_RENDER_HEIGHT",
    "DEFAULT_RENDER_QUALITY",
    "DEFAULT_RENDER_SAMPLES",
    "DEFAULT_RENDER_WIDTH",
    "DEFAULT_TEXT_COLOR",
    "LIGHTING_LEVEL_CHOICES",
    "LIGHTING_LEVEL_LABELS",
    "OrientationSpec",
    "RENDER_QUALITY_CHOICES",
    "RENDER_QUALITY_DEFAULTS",
    "RENDER_QUALITY_LABELS",
    "atom_style_base",
    "atom_style_defaults",
    "atom_style_description",
    "atom_style_is_custom",
    "atom_style_label",
    "available_atom_style_labels",
    "bond_threshold_lookup",
    "build_blend_output_path",
    "complete_bond_threshold_specs",
    "custom_aesthetics",
    "deserialize_custom_aesthetic",
    "default_bond_max_length",
    "default_bond_threshold_spec",
    "build_legend_output_path",
    "build_output_path",
    "build_render_output_path",
    "encode_bond_threshold_arg",
    "encode_custom_aesthetic_arg",
    "encode_orientation_arg",
    "format_angle_for_filename",
    "get_custom_aesthetic",
    "infer_title",
    "normalize_builtin_atom_style",
    "normalize_atom_style",
    "normalize_element_pair",
    "normalize_element_symbol",
    "normalize_lighting_level",
    "normalize_render_quality",
    "orientation_filename_metadata",
    "parse_bond_threshold_arg",
    "parse_custom_aesthetic_arg",
    "parse_orientation_arg",
    "sanitize_orientation_key",
    "sanitize_custom_aesthetic_key",
    "serialize_custom_aesthetic",
    "set_custom_aesthetics",
    "style_atom_color",
    "style_atom_size_scale",
    "style_display_radius",
    "style_neutral_bond_color",
    "style_split_bond_color",
]
