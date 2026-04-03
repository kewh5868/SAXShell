from __future__ import annotations

from pathlib import Path

from saxshell.toolbox.blender.common import (
    ATOM_STYLE_CHOICES,
    LIGHTING_LEVEL_CHOICES,
    normalize_atom_style,
    normalize_element_symbol,
    normalize_lighting_level,
)

REFERENCE_ATOM_ELEMENTS = ("C", "S")
REFERENCE_ATOM_LABELS = {
    "C": "Dark Atom (Carbon)",
    "S": "Light Atom (Sulfur)",
}
REFERENCE_ATOM_BACKGROUND = "#ffffff"
REFERENCE_ATOM_RENDER_QUALITY = "high"


def reference_atom_asset_dir() -> Path:
    return Path(__file__).resolve().parent / "reference_atoms"


def reference_atom_key(
    atom_style: str,
    lighting_level: int,
    element: str,
) -> str:
    style = normalize_atom_style(atom_style)
    level = normalize_lighting_level(lighting_level)
    symbol = normalize_element_symbol(element).lower()
    return f"{symbol}_reference_{style}_lighting_{level}"


def reference_atom_filename(
    atom_style: str,
    lighting_level: int,
    element: str,
) -> str:
    return f"{reference_atom_key(atom_style, lighting_level, element)}.png"


def reference_atom_path(
    atom_style: str,
    lighting_level: int,
    element: str,
) -> Path:
    return reference_atom_asset_dir() / reference_atom_filename(
        atom_style,
        lighting_level,
        element,
    )


def iter_reference_atom_matrix(
    elements: tuple[str, ...] | None = None,
) -> tuple[tuple[str, str, int], ...]:
    allowed_elements = elements or REFERENCE_ATOM_ELEMENTS
    return tuple(
        (normalize_element_symbol(element), atom_style, lighting_level)
        for element in allowed_elements
        for atom_style in ATOM_STYLE_CHOICES
        for lighting_level in LIGHTING_LEVEL_CHOICES
    )


__all__ = [
    "REFERENCE_ATOM_BACKGROUND",
    "REFERENCE_ATOM_ELEMENTS",
    "REFERENCE_ATOM_LABELS",
    "REFERENCE_ATOM_RENDER_QUALITY",
    "iter_reference_atom_matrix",
    "reference_atom_asset_dir",
    "reference_atom_filename",
    "reference_atom_key",
    "reference_atom_path",
]
