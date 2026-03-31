from __future__ import annotations

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
        "Glossy, paper-style spheres with tighter ball sizes, stronger reflections, "
        "and slender split-color bonds."
    ),
    "vesta": (
        "A familiar VESTA-like ball-and-stick look with larger atoms and clear bond "
        "separation."
    ),
    "soft_studio": (
        "A softer publication preset with gentler highlights, warmer bonds, and a "
        "slightly fuller ball-and-stick profile."
    ),
    "flat_diagram": (
        "A lower-sheen schematic preset with slimmer neutral bonds and flatter atom "
        "surfaces."
    ),
    "toon_matte": (
        "A flatter cartoon-style preset with stronger color blocks, soft matte "
        "surfaces, and clean neutral bonds."
    ),
    "poster_pop": (
        "A bold illustrative preset with punchier colors, lighter sheen, and "
        "presentation-friendly split bonds."
    ),
    "pastel_cartoon": (
        "A softer illustrated preset with pastel atoms, matte shading, and gentle "
        "bond contrast."
    ),
    "crystal_flat": (
        "A flat schematic preset with brighter atom colors, "
        "clean neutral bonds, and minimal gloss."
    ),
    "crystal_cartoon": (
        "A cartoon preset with punchy colors, simple "
        "highlights, and crisp split-color bonds."
    ),
    "crystal_shadow_gloss": (
        "A glossy preset with simple directional shading, "
        "clean highlights, and stronger ball-and-stick depth."
    ),
    "cpk": (
        "Classic CPK-inspired colorization with cleaner reflections and a balanced "
        "ball-and-stick ratio."
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
    if text in ATOM_STYLE_CHOICES:
        return text
    return DEFAULT_ATOM_STYLE


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


__all__ = [
    "ATOM_STYLE_CHOICES",
    "ATOM_STYLE_DEFAULTS",
    "ATOM_STYLE_DESCRIPTIONS",
    "ATOM_STYLE_LABELS",
    "BondThresholdSpec",
    "CPK_COLORS",
    "COVALENT_RADII",
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
    "bond_threshold_lookup",
    "build_blend_output_path",
    "complete_bond_threshold_specs",
    "default_bond_max_length",
    "default_bond_threshold_spec",
    "build_legend_output_path",
    "build_output_path",
    "build_render_output_path",
    "encode_bond_threshold_arg",
    "encode_orientation_arg",
    "format_angle_for_filename",
    "infer_title",
    "normalize_atom_style",
    "normalize_element_pair",
    "normalize_element_symbol",
    "normalize_lighting_level",
    "normalize_render_quality",
    "orientation_filename_metadata",
    "parse_bond_threshold_arg",
    "parse_orientation_arg",
    "sanitize_orientation_key",
]
