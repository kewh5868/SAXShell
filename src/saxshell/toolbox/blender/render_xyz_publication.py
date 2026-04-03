from __future__ import annotations

"""
Render an XYZ or PDB structure in Blender as one or more publication-ready images.

Run with:

    blender --background --python src/saxshell/toolbox/blender/render_xyz_publication.py -- \
        --input molecule.xyz \
        --output-dir renders \
        --orientation isometric:35.264:0:45
"""

import argparse
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parents[3]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from saxshell.toolbox.blender.common import (  # noqa: E402
    ATOM_STYLE_CHOICES,
    COVALENT_RADII,
    DEFAULT_ATOM_STYLE,
    DEFAULT_BOND_THRESHOLD_SCALE,
    DEFAULT_LIGHTING_LEVEL,
    DEFAULT_RENDER_HEIGHT,
    DEFAULT_RENDER_QUALITY,
    DEFAULT_RENDER_WIDTH,
    DEFAULT_TEXT_COLOR,
    LIGHTING_LEVEL_CHOICES,
    RENDER_QUALITY_CHOICES,
    RENDER_QUALITY_DEFAULTS,
    BondThresholdSpec,
    OrientationSpec,
    atom_style_base,
    atom_style_defaults,
    bond_threshold_lookup,
    build_blend_output_path,
    build_render_output_path,
    complete_bond_threshold_specs,
    infer_title,
    normalize_atom_style,
    normalize_element_pair,
    normalize_lighting_level,
    normalize_render_quality,
    parse_bond_threshold_arg,
    parse_custom_aesthetic_arg,
    parse_orientation_arg,
    set_custom_aesthetics,
    style_atom_color,
    style_display_radius,
    style_neutral_bond_color,
    style_split_bond_color,
)

try:
    import bpy  # type: ignore
    from mathutils import Vector  # type: ignore
except ImportError as exc:  # pragma: no cover - Blender runtime only
    raise SystemExit(
        "This script must be run from Blender, for example:\n"
        "blender --background --python render_xyz_publication.py -- --input "
        "molecule.xyz --output-dir renders --orientation isometric:35.264:0:45"
    ) from exc


@dataclass(slots=True)
class AtomRecord:
    element: str
    position: Vector


_XYZ_SUFFIXES = {".xyz"}
_PDB_SUFFIXES = {".pdb", ".ent"}


def _normalized_element(symbol: str) -> str:
    text = symbol.strip()
    if not text:
        return "X"
    return text[:1].upper() + text[1:].lower()


def _pdb_record_name(line: str) -> str:
    return line[:6].strip().upper()


def _read_text_lines(path: str | Path) -> list[str]:
    structure_path = Path(path).expanduser().resolve()
    return structure_path.read_text(
        encoding="utf-8", errors="ignore"
    ).splitlines()


def _peek_text_lines(path: str | Path, *, limit: int = 64) -> list[str]:
    structure_path = Path(path).expanduser().resolve()
    lines: list[str] = []
    with structure_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for index, line in enumerate(handle):
            lines.append(line.rstrip("\n"))
            if index + 1 >= limit:
                break
    return lines


def _collapse_text_fragments(fragments: list[str]) -> str:
    return " ".join(
        fragment.strip()
        for fragment in fragments
        if fragment and fragment.strip()
    ).strip()


def detect_structure_format(path: str | Path) -> str:
    structure_path = Path(path).expanduser().resolve()
    suffix = structure_path.suffix.lower()
    if suffix in _XYZ_SUFFIXES:
        return "xyz"
    if suffix in _PDB_SUFFIXES:
        return "pdb"

    preview_lines = _peek_text_lines(structure_path)
    for line in preview_lines:
        stripped = line.strip()
        if not stripped:
            continue
        try:
            int(stripped)
        except ValueError:
            break
        return "xyz"
    if any(
        _pdb_record_name(line)
        in {"HEADER", "TITLE", "COMPND", "ATOM", "HETATM", "MODEL"}
        for line in preview_lines
    ):
        return "pdb"
    raise ValueError(
        f"Unsupported structure file format for {structure_path}. "
        "Choose an XYZ or PDB file."
    )


def _blend_args(argv: list[str] | None = None) -> list[str]:
    args = sys.argv if argv is None else argv
    if "--" not in args:
        return []
    return args[args.index("--") + 1 :]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="render_xyz_publication.py",
        description=(
            "Render an XYZ or PDB structure to one or more publication-ready PNGs "
            "using ball-and-stick geometry, cleaner studio lighting, and "
            "orientation-aware output filenames."
        ),
    )
    parser.add_argument(
        "--input", required=True, help="Input XYZ or PDB file."
    )
    parser.add_argument(
        "--output",
        help="Legacy single-image output path. Use --output-dir for batches.",
    )
    parser.add_argument(
        "--output-dir",
        help="Destination folder for orientation-aware PNG outputs.",
    )
    parser.add_argument("--title", help="Optional title override.")
    parser.add_argument(
        "--hide-title",
        action="store_true",
        help="Do not draw a title label in the render.",
    )
    parser.add_argument(
        "--uppercase-title",
        action="store_true",
        help="Uppercase the rendered title text.",
    )
    parser.add_argument(
        "--orientation",
        action="append",
        default=[],
        help=(
            "Orientation as key:x_degrees:y_degrees:z_degrees"
            "[:atom_style:render_quality[:lighting_level]]."
        ),
    )
    parser.add_argument(
        "--width",
        type=int,
        default=DEFAULT_RENDER_WIDTH,
        help="Output width in pixels.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=DEFAULT_RENDER_HEIGHT,
        help="Output height in pixels.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Cycles sample count. Defaults to the selected render-quality preset.",
    )
    parser.add_argument(
        "--sample-floor",
        type=int,
        default=None,
        help=(
            "Optional minimum sample floor override. When omitted, the "
            "selected render-quality preset decides the floor."
        ),
    )
    parser.add_argument(
        "--atom-style",
        default=DEFAULT_ATOM_STYLE,
        help="Atom and bond styling preset or saved custom aesthetic key.",
    )
    parser.add_argument(
        "--atom-scale",
        type=float,
        default=None,
        help="Scale factor applied to covalent radii for the rendered atoms.",
    )
    parser.add_argument(
        "--bond-radius",
        type=float,
        default=None,
        help="Cylinder radius used for rendered bonds.",
    )
    parser.add_argument(
        "--bond-threshold",
        type=float,
        default=DEFAULT_BOND_THRESHOLD_SCALE,
        help="Fallback bond detection scale factor on covalent radii sums.",
    )
    parser.add_argument(
        "--bond-pair-threshold",
        action="append",
        default=[],
        help=(
            "VESTA-style pair cutoff as element_a:element_b:min_length:max_length. "
            "May be provided multiple times."
        ),
    )
    parser.add_argument(
        "--custom-aesthetic",
        action="append",
        default=[],
        help=(
            "JSON-encoded saved custom aesthetic definition. May be provided "
            "multiple times."
        ),
    )
    parser.add_argument(
        "--render-quality",
        choices=RENDER_QUALITY_CHOICES,
        default=DEFAULT_RENDER_QUALITY,
        help="Lighting and render tuning preset.",
    )
    parser.add_argument(
        "--lighting-level",
        type=int,
        choices=LIGHTING_LEVEL_CHOICES,
        default=DEFAULT_LIGHTING_LEVEL,
        help="Overall lighting brightness preset from 1 (lowest) to 5 (brightest).",
    )
    parser.add_argument(
        "--execution-device",
        choices=("auto", "cpu", "gpu"),
        default="auto",
        help=(
            "Cycles execution device. 'auto' prefers GPU when available, "
            "'cpu' skips GPU kernel startup for lightweight reference renders."
        ),
    )
    parser.add_argument(
        "--bond-color-mode",
        choices=("neutral", "split"),
        default=None,
        help=(
            "Use neutral gray bonds or split each bond into two halves colored "
            "by the connected atoms."
        ),
    )
    parser.add_argument(
        "--camera-margin",
        type=float,
        default=None,
        help="Extra framing margin for the orthographic camera.",
    )
    parser.add_argument(
        "--transparent",
        action="store_true",
        help="Deprecated: PNG renders are now always written with transparency.",
    )
    parser.add_argument(
        "--save-blend-files",
        action="store_true",
        help="Save a .blend scene beside each rendered PNG output.",
    )
    parser.add_argument(
        "--hide-hydrogen",
        action="store_true",
        help="Skip hydrogen atoms and hydrogen bonds in the render.",
    )
    parser.add_argument(
        "--title-scale",
        type=float,
        default=0.16,
        help="Title size relative to the structure width.",
    )
    return parser


def parse_xyz(
    path: str | Path,
    *,
    hide_hydrogen: bool = False,
) -> tuple[str, list[AtomRecord]]:
    xyz_path = Path(path).expanduser().resolve()
    lines = xyz_path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 3:
        raise ValueError(f"XYZ file is too short: {xyz_path}")
    try:
        atom_count = int(lines[0].strip())
    except ValueError as exc:
        raise ValueError(f"Invalid XYZ atom count in {xyz_path}") from exc

    comment = lines[1].strip() if len(lines) > 1 else ""
    atoms: list[AtomRecord] = []
    for line in lines[2 : 2 + atom_count]:
        tokens = line.split()
        if len(tokens) < 4:
            continue
        element = _normalized_element(tokens[0])
        if hide_hydrogen and element == "H":
            continue
        atoms.append(
            AtomRecord(
                element=element,
                position=Vector(
                    (float(tokens[1]), float(tokens[2]), float(tokens[3]))
                ),
            )
        )
    if not atoms:
        raise ValueError(f"No atoms were parsed from {xyz_path}")

    centroid = sum(
        (atom.position for atom in atoms),
        Vector((0.0, 0.0, 0.0)),
    ) / len(atoms)
    for atom in atoms:
        atom.position -= centroid
    return comment, atoms


def _pdb_text_field(line: str) -> str:
    return line[10:].strip() if len(line) > 10 else ""


def read_pdb_comment(path: str | Path) -> str:
    title_lines: list[str] = []
    compound_lines: list[str] = []
    header_line = ""
    for line in _read_text_lines(path):
        record_name = _pdb_record_name(line)
        if record_name == "TITLE":
            title_lines.append(_pdb_text_field(line))
        elif record_name == "COMPND":
            compound_lines.append(_pdb_text_field(line))
        elif record_name == "HEADER" and not header_line:
            header_line = _pdb_text_field(line)
    return (
        _collapse_text_fragments(title_lines)
        or _collapse_text_fragments(compound_lines)
        or header_line.strip()
    )


def _pdb_atom_site_key(line: str) -> tuple[str, str, str, str]:
    return (
        line[21:22].strip(),
        line[22:26].strip(),
        line[26:27].strip(),
        line[12:16].strip(),
    )


def _pdb_altloc_rank(value: str) -> int:
    altloc = value.strip().upper()
    if not altloc:
        return 0
    if altloc == "A":
        return 1
    if altloc == "1":
        return 2
    return 3


def _pdb_atom_element(line: str) -> str:
    if len(line) >= 78:
        element = _normalized_element(line[76:78])
        if element != "X":
            return element
    atom_name_field = line[12:16] if len(line) >= 16 else ""
    letters = "".join(
        character for character in atom_name_field if character.isalpha()
    )
    if not letters:
        return "X"
    if atom_name_field[:1].isalpha() and len(letters) >= 2:
        candidate = _normalized_element(letters[:2])
        if candidate in COVALENT_RADII:
            return candidate
    return _normalized_element(letters[:1])


def parse_pdb(
    path: str | Path,
    *,
    hide_hydrogen: bool = False,
) -> tuple[str, list[AtomRecord]]:
    pdb_path = Path(path).expanduser().resolve()
    chosen_sites: dict[tuple[str, str, str, str], tuple[int, str]] = {}
    site_order: list[tuple[str, str, str, str]] = []
    for line in _read_text_lines(pdb_path):
        record_name = _pdb_record_name(line)
        if record_name == "ENDMDL" and chosen_sites:
            break
        if record_name not in {"ATOM", "HETATM"}:
            continue
        site_key = _pdb_atom_site_key(line)
        altloc_rank = _pdb_altloc_rank(line[16:17] if len(line) >= 17 else "")
        current = chosen_sites.get(site_key)
        if current is None:
            chosen_sites[site_key] = (altloc_rank, line)
            site_order.append(site_key)
        elif altloc_rank < current[0]:
            chosen_sites[site_key] = (altloc_rank, line)

    atoms: list[AtomRecord] = []
    for site_key in site_order:
        _rank, line = chosen_sites[site_key]
        try:
            x_value = float(line[30:38].strip())
            y_value = float(line[38:46].strip())
            z_value = float(line[46:54].strip())
        except ValueError:
            continue
        element = _pdb_atom_element(line)
        if hide_hydrogen and element == "H":
            continue
        atoms.append(
            AtomRecord(
                element=element,
                position=Vector((x_value, y_value, z_value)),
            )
        )
    if not atoms:
        raise ValueError(f"No atoms were parsed from PDB file: {pdb_path}")

    centroid = sum(
        (atom.position for atom in atoms),
        Vector((0.0, 0.0, 0.0)),
    ) / len(atoms)
    for atom in atoms:
        atom.position -= centroid
    return read_pdb_comment(pdb_path), atoms


def parse_structure(
    path: str | Path,
    *,
    hide_hydrogen: bool = False,
) -> tuple[str, list[AtomRecord]]:
    structure_format = detect_structure_format(path)
    if structure_format == "pdb":
        return parse_pdb(path, hide_hydrogen=hide_hydrogen)
    return parse_xyz(path, hide_hydrogen=hide_hydrogen)


def display_radius(element: str, atom_scale: float) -> float:
    base = COVALENT_RADII.get(element, 0.85)
    return max(base * atom_scale, 0.18)


def detect_bonds(
    atoms: list[AtomRecord],
    *,
    threshold_scale: float,
    pair_thresholds: (
        list[BondThresholdSpec] | tuple[BondThresholdSpec, ...] | None
    ) = None,
) -> list[tuple[int, int]]:
    thresholds = complete_bond_threshold_specs(
        [atom.element for atom in atoms],
        pair_thresholds,
        threshold_scale=threshold_scale,
    )
    threshold_lookup = bond_threshold_lookup(thresholds)
    bonds: list[tuple[int, int]] = []
    for left_index, left_atom in enumerate(atoms):
        for right_index in range(left_index + 1, len(atoms)):
            right_atom = atoms[right_index]
            threshold_spec = threshold_lookup[
                normalize_element_pair(left_atom.element, right_atom.element)
            ]
            distance = (left_atom.position - right_atom.position).length
            if (
                float(threshold_spec.min_length)
                <= distance
                <= float(threshold_spec.max_length)
            ):
                bonds.append((left_index, right_index))
    return bonds


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for datablocks in (
        bpy.data.meshes,
        bpy.data.materials,
        bpy.data.curves,
        bpy.data.lights,
        bpy.data.cameras,
        bpy.data.fonts,
        bpy.data.objects,
    ):
        for datablock in list(datablocks):
            if datablock.users == 0:
                datablocks.remove(datablock)


def create_material(
    name: str,
    *,
    base_color: tuple[float, float, float, float],
    metallic: float,
    roughness: float,
    specular: float = 0.5,
    coat: float = 0.0,
    emission_strength: float = 0.0,
) -> bpy.types.Material:
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    principled = material.node_tree.nodes.get("Principled BSDF")
    if principled is None:
        return material
    principled.inputs["Base Color"].default_value = base_color
    principled.inputs["Metallic"].default_value = metallic
    principled.inputs["Roughness"].default_value = roughness
    principled.inputs["Specular IOR Level"].default_value = specular
    if "Coat Weight" in principled.inputs:
        principled.inputs["Coat Weight"].default_value = coat
    elif "Clearcoat" in principled.inputs:
        principled.inputs["Clearcoat"].default_value = coat
    emission_input = principled.inputs.get(
        "Emission Color"
    ) or principled.inputs.get("Emission")
    if emission_input is not None:
        emission_input.default_value = base_color
    emission_strength_input = principled.inputs.get("Emission Strength")
    if emission_strength_input is not None:
        emission_strength_input.default_value = max(
            float(emission_strength), 0.0
        )
    return material


def attach_material(
    obj: bpy.types.Object, material: bpy.types.Material
) -> None:
    if obj.data is None:
        return
    if hasattr(obj.data, "materials"):
        obj.data.materials.clear()
        obj.data.materials.append(material)


def shade_smooth(obj: bpy.types.Object) -> None:
    if not hasattr(obj.data, "polygons"):
        return
    for polygon in obj.data.polygons:
        polygon.use_smooth = True


def create_atom_object(
    atom: AtomRecord,
    *,
    radius: float,
    material: bpy.types.Material,
    parent: bpy.types.Object,
    index: int,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=96,
        ring_count=48,
        radius=radius,
        location=atom.position,
    )
    obj = bpy.context.object
    obj.name = f"Atom_{index:04d}_{atom.element}"
    obj.parent = parent
    shade_smooth(obj)
    attach_material(obj, material)
    return obj


def _cylinder_rotation(vector: Vector) -> tuple[float, float, float]:
    z_axis = Vector((0.0, 0.0, 1.0))
    quaternion = z_axis.rotation_difference(vector.normalized())
    return quaternion.to_euler()


def create_bond_segment(
    start: Vector,
    end: Vector,
    *,
    radius: float,
    material: bpy.types.Material,
    parent: bpy.types.Object,
    name: str,
) -> bpy.types.Object | None:
    delta = end - start
    length = delta.length
    if length <= 1.0e-6:
        return None
    midpoint = (start + end) * 0.5
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=48,
        radius=radius,
        depth=length,
        location=midpoint,
    )
    obj = bpy.context.object
    obj.name = name
    obj.rotation_euler = _cylinder_rotation(delta)
    obj.parent = parent
    shade_smooth(obj)
    attach_material(obj, material)
    return obj


def atom_surface_color(
    element: str,
    *,
    atom_style: str,
) -> tuple[float, float, float, float]:
    return style_atom_color(element, atom_style=atom_style)


def atom_material_profile(
    atom_style: str,
) -> dict[str, float]:
    style = atom_style_base(atom_style)
    if style == "paper_gloss":
        return {
            "metallic": 0.0,
            "roughness": 0.10,
            "specular": 0.70,
            "coat": 0.34,
            "emission": 0.02,
        }
    if style == "soft_studio":
        return {
            "metallic": 0.0,
            "roughness": 0.16,
            "specular": 0.60,
            "coat": 0.22,
            "emission": 0.03,
        }
    if style == "flat_diagram":
        return {
            "metallic": 0.0,
            "roughness": 0.46,
            "specular": 0.14,
            "coat": 0.0,
            "emission": 0.04,
        }
    if style == "toon_matte":
        return {
            "metallic": 0.0,
            "roughness": 0.70,
            "specular": 0.05,
            "coat": 0.0,
            "emission": 0.06,
        }
    if style == "poster_pop":
        return {
            "metallic": 0.0,
            "roughness": 0.48,
            "specular": 0.10,
            "coat": 0.0,
            "emission": 0.06,
        }
    if style == "pastel_cartoon":
        return {
            "metallic": 0.0,
            "roughness": 0.62,
            "specular": 0.08,
            "coat": 0.0,
            "emission": 0.06,
        }
    if style == "crystal_flat":
        return {
            "metallic": 0.0,
            "roughness": 0.50,
            "specular": 0.12,
            "coat": 0.0,
            "emission": 0.04,
        }
    if style == "crystal_cartoon":
        return {
            "metallic": 0.0,
            "roughness": 0.38,
            "specular": 0.18,
            "coat": 0.02,
            "emission": 0.05,
        }
    if style == "crystal_shadow_gloss":
        return {
            "metallic": 0.0,
            "roughness": 0.18,
            "specular": 0.60,
            "coat": 0.10,
            "emission": 0.03,
        }
    if style == "monochrome":
        return {
            "metallic": 0.0,
            "roughness": 0.22,
            "specular": 0.48,
            "coat": 0.08,
            "emission": 0.02,
        }
    if style == "cpk":
        return {
            "metallic": 0.01,
            "roughness": 0.16,
            "specular": 0.56,
            "coat": 0.16,
            "emission": 0.03,
        }
    return {
        "metallic": 0.0,
        "roughness": 0.20,
        "specular": 0.52,
        "coat": 0.18,
        "emission": 0.02,
    }


def bond_material_profile(
    atom_style: str,
) -> dict[str, float]:
    style = atom_style_base(atom_style)
    if style == "paper_gloss":
        return {
            "metallic": 0.0,
            "roughness": 0.18,
            "specular": 0.48,
            "coat": 0.06,
            "emission": 0.01,
        }
    if style == "soft_studio":
        return {
            "metallic": 0.0,
            "roughness": 0.28,
            "specular": 0.30,
            "coat": 0.01,
            "emission": 0.02,
        }
    if style == "flat_diagram":
        return {
            "metallic": 0.0,
            "roughness": 0.48,
            "specular": 0.12,
            "coat": 0.0,
            "emission": 0.03,
        }
    if style == "toon_matte":
        return {
            "metallic": 0.0,
            "roughness": 0.58,
            "specular": 0.05,
            "coat": 0.0,
            "emission": 0.03,
        }
    if style == "poster_pop":
        return {
            "metallic": 0.0,
            "roughness": 0.42,
            "specular": 0.10,
            "coat": 0.0,
            "emission": 0.03,
        }
    if style == "pastel_cartoon":
        return {
            "metallic": 0.0,
            "roughness": 0.52,
            "specular": 0.08,
            "coat": 0.0,
            "emission": 0.03,
        }
    if style == "crystal_flat":
        return {
            "metallic": 0.0,
            "roughness": 0.44,
            "specular": 0.10,
            "coat": 0.0,
            "emission": 0.03,
        }
    if style == "crystal_cartoon":
        return {
            "metallic": 0.0,
            "roughness": 0.38,
            "specular": 0.12,
            "coat": 0.0,
            "emission": 0.03,
        }
    if style == "crystal_shadow_gloss":
        return {
            "metallic": 0.0,
            "roughness": 0.24,
            "specular": 0.26,
            "coat": 0.02,
            "emission": 0.01,
        }
    if style == "monochrome":
        return {
            "metallic": 0.0,
            "roughness": 0.24,
            "specular": 0.34,
            "coat": 0.02,
            "emission": 0.01,
        }
    if style == "cpk":
        return {
            "metallic": 0.01,
            "roughness": 0.22,
            "specular": 0.42,
            "coat": 0.03,
            "emission": 0.02,
        }
    return {
        "metallic": 0.0,
        "roughness": 0.23,
        "specular": 0.38,
        "coat": 0.03,
        "emission": 0.01,
    }


def render_quality_sample_floor(render_quality: str) -> int:
    quality = normalize_render_quality(render_quality)
    if quality == "draft":
        return 128
    if quality == "balanced":
        return 256
    return 512


def render_quality_filter_size(render_quality: str) -> float:
    quality = normalize_render_quality(render_quality)
    if quality == "draft":
        return 0.75
    if quality == "balanced":
        return 0.55
    return 0.38


def render_quality_exposure(render_quality: str) -> float:
    quality = normalize_render_quality(render_quality)
    if quality == "draft":
        return 0.0
    if quality == "balanced":
        return 0.06
    return 0.12


def render_quality_light_scale(render_quality: str) -> float:
    quality = normalize_render_quality(render_quality)
    if quality == "draft":
        return 0.86
    if quality == "balanced":
        return 0.96
    return 1.06


def lighting_level_exposure_offset(lighting_level: int) -> float:
    level = normalize_lighting_level(lighting_level)
    return {
        1: -0.08,
        2: 0.0,
        3: 0.08,
        4: 0.16,
        5: 0.24,
    }[level]


def lighting_level_scale(lighting_level: int) -> float:
    level = normalize_lighting_level(lighting_level)
    return {
        1: 0.88,
        2: 1.00,
        3: 1.14,
        4: 1.28,
        5: 1.44,
    }[level]


def build_structure(
    atoms: list[AtomRecord],
    *,
    atom_style: str,
    atom_scale: float | None,
    bond_radius: float,
    bond_threshold: float,
    pair_thresholds: (
        list[BondThresholdSpec] | tuple[BondThresholdSpec, ...] | None
    ) = None,
    bond_color_mode: str,
) -> tuple[bpy.types.Object, list[bpy.types.Object]]:
    root = bpy.data.objects.new("StructureRoot", None)
    bpy.context.scene.collection.objects.link(root)
    root.rotation_euler = (0.0, 0.0, 0.0)

    atom_materials: dict[str, bpy.types.Material] = {}
    atom_objects: list[bpy.types.Object] = []
    atom_profile = atom_material_profile(atom_style)
    for index, atom in enumerate(atoms, start=1):
        color = atom_surface_color(atom.element, atom_style=atom_style)
        if atom.element not in atom_materials:
            atom_materials[atom.element] = create_material(
                f"Atom_{atom.element}",
                base_color=color,
                metallic=atom_profile["metallic"],
                roughness=atom_profile["roughness"],
                specular=atom_profile["specular"],
                coat=atom_profile["coat"],
                emission_strength=atom_profile["emission"],
            )
        atom_objects.append(
            create_atom_object(
                atom,
                radius=style_display_radius(
                    atom.element,
                    atom_style=atom_style,
                    atom_scale_override=atom_scale,
                ),
                material=atom_materials[atom.element],
                parent=root,
                index=index,
            )
        )

    bond_profile = bond_material_profile(atom_style)
    neutral_bond_material = create_material(
        "Bond_Neutral",
        base_color=style_neutral_bond_color(atom_style),
        metallic=bond_profile["metallic"],
        roughness=bond_profile["roughness"],
        specular=bond_profile["specular"],
        coat=bond_profile["coat"],
        emission_strength=bond_profile["emission"],
    )
    split_bond_materials: dict[str, bpy.types.Material] = {}
    created_bonds: list[bpy.types.Object] = []
    for left_index, right_index in detect_bonds(
        atoms,
        threshold_scale=bond_threshold,
        pair_thresholds=pair_thresholds,
    ):
        left_atom = atoms[left_index]
        right_atom = atoms[right_index]
        start = left_atom.position
        end = right_atom.position
        if bond_color_mode == "split":
            midpoint = (start + end) * 0.5
            for suffix, bond_start, bond_end, element in (
                ("A", start, midpoint, left_atom.element),
                ("B", midpoint, end, right_atom.element),
            ):
                if element not in split_bond_materials:
                    split_bond_materials[element] = create_material(
                        f"Bond_{element}",
                        base_color=style_split_bond_color(
                            element,
                            atom_style=atom_style,
                        ),
                        metallic=bond_profile["metallic"],
                        roughness=bond_profile["roughness"] + 0.02,
                        specular=bond_profile["specular"],
                        coat=bond_profile["coat"],
                        emission_strength=bond_profile["emission"],
                    )
                segment = create_bond_segment(
                    bond_start,
                    bond_end,
                    radius=bond_radius,
                    material=split_bond_materials[element],
                    parent=root,
                    name=(f"Bond_{left_index:04d}_{right_index:04d}_{suffix}"),
                )
                if segment is not None:
                    created_bonds.append(segment)
        else:
            segment = create_bond_segment(
                start,
                end,
                radius=bond_radius,
                material=neutral_bond_material,
                parent=root,
                name=f"Bond_{left_index:04d}_{right_index:04d}",
            )
            if segment is not None:
                created_bonds.append(segment)
    return root, atom_objects + created_bonds


def world_bounds(
    objects: list[bpy.types.Object],
) -> tuple[Vector, Vector]:
    min_corner = Vector((math.inf, math.inf, math.inf))
    max_corner = Vector((-math.inf, -math.inf, -math.inf))
    for obj in objects:
        if obj.type == "EMPTY":
            continue
        for corner in obj.bound_box:
            world_corner = obj.matrix_world @ Vector(corner)
            min_corner.x = min(min_corner.x, world_corner.x)
            min_corner.y = min(min_corner.y, world_corner.y)
            min_corner.z = min(min_corner.z, world_corner.z)
            max_corner.x = max(max_corner.x, world_corner.x)
            max_corner.y = max(max_corner.y, world_corner.y)
            max_corner.z = max(max_corner.z, world_corner.z)
    return min_corner, max_corner


def create_title_object(text: str) -> bpy.types.Object:
    bpy.ops.object.text_add(location=(0.0, 0.0, 0.0))
    title = bpy.context.object
    title.name = "StructureTitle"
    title.data.body = text
    title.data.align_x = "CENTER"
    title.data.align_y = "CENTER"
    title.data.extrude = 0.01
    title.data.bevel_depth = 0.001
    material = create_material(
        "TitleMaterial",
        base_color=DEFAULT_TEXT_COLOR,
        metallic=0.0,
        roughness=0.34,
        specular=0.24,
        coat=0.0,
    )
    attach_material(title, material)
    return title


def place_title_object(
    title: bpy.types.Object,
    *,
    text: str,
    structure_bounds: tuple[Vector, Vector],
    scale_factor: float,
    vertical_band: float,
) -> None:
    min_corner, max_corner = structure_bounds
    structure_width = max(max_corner.x - min_corner.x, 1.0)
    structure_height = max(max_corner.z - min_corner.z, 1.0)
    title.data.body = text
    title.data.size = max(
        structure_width * scale_factor,
        structure_height * 0.12,
    )
    title.location = (
        (min_corner.x + max_corner.x) * 0.5,
        0.0,
        min_corner.z - vertical_band * 0.58,
    )


def setup_world(*, transparent: bool, atom_style: str) -> None:
    scene = bpy.context.scene
    world = scene.world or bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    style = atom_style_base(atom_style)
    if background is not None:
        if style == "paper_gloss":
            background.inputs[0].default_value = (0.997, 0.994, 0.989, 1.0)
            background.inputs[1].default_value = 0.08
        elif style == "soft_studio":
            background.inputs[0].default_value = (0.990, 0.984, 0.976, 1.0)
            background.inputs[1].default_value = 0.09
        elif style == "flat_diagram":
            background.inputs[0].default_value = (1.0, 1.0, 1.0, 1.0)
            background.inputs[1].default_value = 0.08
        elif style == "toon_matte":
            background.inputs[0].default_value = (0.999, 0.978, 0.944, 1.0)
            background.inputs[1].default_value = 0.08
        elif style == "poster_pop":
            background.inputs[0].default_value = (0.999, 0.973, 0.937, 1.0)
            background.inputs[1].default_value = 0.08
        elif style == "pastel_cartoon":
            background.inputs[0].default_value = (0.999, 0.985, 0.975, 1.0)
            background.inputs[1].default_value = 0.09
        elif style == "crystal_flat":
            background.inputs[0].default_value = (1.0, 1.0, 1.0, 1.0)
            background.inputs[1].default_value = 0.08
        elif style == "crystal_cartoon":
            background.inputs[0].default_value = (0.999, 0.996, 0.985, 1.0)
            background.inputs[1].default_value = 0.08
        elif style == "crystal_shadow_gloss":
            background.inputs[0].default_value = (0.999, 0.990, 0.975, 1.0)
            background.inputs[1].default_value = 0.08
        elif style == "monochrome":
            background.inputs[0].default_value = (0.986, 0.986, 0.989, 1.0)
            background.inputs[1].default_value = 0.08
        elif style == "cpk":
            background.inputs[0].default_value = (0.972, 0.978, 0.990, 1.0)
            background.inputs[1].default_value = 0.08
        else:
            background.inputs[0].default_value = (0.994, 0.996, 0.999, 1.0)
            background.inputs[1].default_value = 0.08
    scene.render.film_transparent = True


def setup_render(
    *,
    width: int,
    height: int,
    samples: int,
    sample_floor_override: int | None,
    transparent: bool,
    render_quality: str,
    lighting_level: int,
    cycles_device: str = "auto",
) -> None:
    scene = bpy.context.scene
    quality = normalize_render_quality(render_quality)
    sample_floor = (
        max(int(sample_floor_override), 1)
        if sample_floor_override is not None
        else render_quality_sample_floor(quality)
    )
    tuned_samples = max(int(samples), sample_floor)
    scene.render.engine = (
        "CYCLES"
        if getattr(bpy.app.build_options, "cycles", False)
        else "BLENDER_EEVEE"
    )
    scene.render.resolution_x = int(width)
    scene.render.resolution_y = int(height)
    scene.render.resolution_percentage = 100
    scene.render.use_file_extension = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = True
    scene.render.filter_size = render_quality_filter_size(quality)
    scene.view_settings.look = "None"
    scene.view_settings.exposure = render_quality_exposure(
        quality
    ) + lighting_level_exposure_offset(lighting_level)
    if scene.render.engine == "CYCLES":
        scene.cycles.samples = tuned_samples
        scene.cycles.preview_samples = max(tuned_samples // 4, 32)
        scene.cycles.use_denoising = True
        if hasattr(scene.cycles, "use_adaptive_sampling"):
            scene.cycles.use_adaptive_sampling = True
        if hasattr(scene.cycles, "max_bounces"):
            scene.cycles.max_bounces = 10 if quality == "high" else 8
        if hasattr(scene.cycles, "diffuse_bounces"):
            scene.cycles.diffuse_bounces = 4 if quality == "high" else 3
        if hasattr(scene.cycles, "glossy_bounces"):
            scene.cycles.glossy_bounces = 4 if quality == "high" else 2
        if hasattr(scene.cycles, "transparent_max_bounces"):
            scene.cycles.transparent_max_bounces = 6
        if hasattr(scene.cycles, "caustics_reflective"):
            scene.cycles.caustics_reflective = False
        if hasattr(scene.cycles, "caustics_refractive"):
            scene.cycles.caustics_refractive = False
        requested_device = (
            str(
                cycles_device
                or os.environ.get("SAXSHELL_CYCLES_DEVICE", "auto")
            )
            .strip()
            .lower()
        )
        if requested_device == "cpu":
            scene.cycles.device = "CPU"
        else:
            try:
                scene.cycles.device = "GPU"
            except Exception:
                scene.cycles.device = "CPU"
    elif hasattr(scene, "eevee"):
        scene.eevee.taa_render_samples = max(tuned_samples, 64)
        scene.eevee.use_gtao = True
        if hasattr(scene.eevee, "use_bloom"):
            scene.eevee.use_bloom = quality == "high"


def add_area_light(
    name: str,
    *,
    location: tuple[float, float, float],
    rotation: tuple[float, float, float],
    energy: float,
    size: float,
) -> bpy.types.Object:
    light_data = bpy.data.lights.new(name=name, type="AREA")
    light_data.energy = energy
    light_data.shape = "RECTANGLE"
    light_data.size = size
    light_data.size_y = size * 0.72
    light_object = bpy.data.objects.new(name, light_data)
    light_object.location = location
    light_object.rotation_euler = rotation
    bpy.context.scene.collection.objects.link(light_object)
    return light_object


def add_sun_light(
    name: str,
    *,
    rotation: tuple[float, float, float],
    energy: float,
) -> bpy.types.Object:
    light_data = bpy.data.lights.new(name=name, type="SUN")
    light_data.energy = energy
    light_object = bpy.data.objects.new(name, light_data)
    light_object.rotation_euler = rotation
    bpy.context.scene.collection.objects.link(light_object)
    return light_object


def setup_lighting(
    bounds: tuple[Vector, Vector],
    *,
    atom_style: str,
    render_quality: str,
    lighting_level: int,
) -> None:
    min_corner, max_corner = bounds
    extent = max(
        max_corner.x - min_corner.x,
        max_corner.y - min_corner.y,
        max_corner.z - min_corner.z,
        1.0,
    )
    light_scale = render_quality_light_scale(
        render_quality
    ) * lighting_level_scale(lighting_level)
    style = atom_style_base(atom_style)
    key_energy = 3600.0 * light_scale
    fill_energy = 2100.0 * light_scale
    rim_energy = 1700.0 * light_scale
    top_energy = 1450.0 * light_scale
    sun_energy = 0.48 * light_scale
    if style == "monochrome":
        fill_energy *= 0.92
        rim_energy *= 1.06
        sun_energy *= 1.08
    elif style == "paper_gloss":
        key_energy *= 0.96
        fill_energy *= 0.96
        rim_energy *= 0.92
        top_energy *= 0.82
        sun_energy *= 0.88
    elif style == "soft_studio":
        key_energy *= 0.84
        fill_energy *= 1.00
        rim_energy *= 0.58
        top_energy *= 0.88
        sun_energy *= 0.84
    elif style == "flat_diagram":
        key_energy *= 0.70
        fill_energy *= 1.08
        rim_energy *= 0.22
        top_energy *= 0.84
        sun_energy *= 0.86
    elif style == "toon_matte":
        key_energy *= 0.68
        fill_energy *= 1.10
        rim_energy *= 0.18
        top_energy *= 0.82
        sun_energy *= 0.84
    elif style == "poster_pop":
        key_energy *= 0.72
        fill_energy *= 1.08
        rim_energy *= 0.22
        top_energy *= 0.86
        sun_energy *= 0.86
    elif style == "pastel_cartoon":
        key_energy *= 0.68
        fill_energy *= 1.08
        rim_energy *= 0.18
        top_energy *= 0.86
        sun_energy *= 0.84
    elif style == "crystal_flat":
        key_energy *= 0.72
        fill_energy *= 1.08
        rim_energy *= 0.20
        top_energy *= 0.84
        sun_energy *= 0.86
    elif style == "crystal_cartoon":
        key_energy *= 0.74
        fill_energy *= 1.10
        rim_energy *= 0.24
        top_energy *= 0.88
        sun_energy *= 0.88
    elif style == "crystal_shadow_gloss":
        key_energy *= 0.90
        fill_energy *= 0.98
        rim_energy *= 0.64
        top_energy *= 0.88
        sun_energy *= 0.88
    elif style == "vesta":
        key_energy *= 0.86
        fill_energy *= 0.98
        top_energy *= 0.88
        sun_energy *= 0.88
    add_area_light(
        "KeyLight",
        location=(-2.2 * extent, -1.9 * extent, 2.1 * extent),
        rotation=(math.radians(58.0), 0.0, math.radians(-30.0)),
        energy=key_energy,
        size=2.4 * extent,
    )
    add_area_light(
        "FillLight",
        location=(2.0 * extent, -1.5 * extent, 1.1 * extent),
        rotation=(math.radians(64.0), 0.0, math.radians(34.0)),
        energy=fill_energy,
        size=2.2 * extent,
    )
    add_area_light(
        "RimLight",
        location=(2.2 * extent, 1.9 * extent, 2.4 * extent),
        rotation=(math.radians(122.0), 0.0, math.radians(148.0)),
        energy=rim_energy,
        size=2.5 * extent,
    )
    add_area_light(
        "TopLight",
        location=(0.0, 0.0, 3.1 * extent),
        rotation=(math.radians(180.0), 0.0, 0.0),
        energy=top_energy,
        size=3.1 * extent,
    )
    add_sun_light(
        "AmbientSun",
        rotation=(math.radians(38.0), math.radians(14.0), math.radians(-20.0)),
        energy=sun_energy,
    )


def create_camera_rig() -> tuple[bpy.types.Object, bpy.types.Object]:
    camera_data = bpy.data.cameras.new("PublicationCamera")
    camera_data.type = "ORTHO"
    camera_data.clip_start = 0.01
    camera_data.clip_end = 10000.0
    camera_object = bpy.data.objects.new("PublicationCamera", camera_data)
    bpy.context.scene.collection.objects.link(camera_object)

    focus = bpy.data.objects.new("CameraFocus", None)
    bpy.context.scene.collection.objects.link(focus)

    track = camera_object.constraints.new(type="TRACK_TO")
    track.target = focus
    track.track_axis = "TRACK_NEGATIVE_Z"
    track.up_axis = "UP_Y"
    bpy.context.scene.camera = camera_object
    return camera_object, focus


def _render_aspect_ratio() -> float:
    render = bpy.context.scene.render
    pixel_width = max(
        float(render.resolution_x) * float(render.pixel_aspect_x), 1.0
    )
    pixel_height = max(
        float(render.resolution_y) * float(render.pixel_aspect_y),
        1.0,
    )
    return pixel_width / pixel_height


def frame_camera(
    camera_object: bpy.types.Object,
    focus: bpy.types.Object,
    *,
    structure_bounds: tuple[Vector, Vector],
    margin: float,
    title_band: float = 0.0,
) -> None:
    min_corner, max_corner = structure_bounds
    width = max(float(max_corner.x - min_corner.x), 1.0)
    depth = max(float(max_corner.y - min_corner.y), 1.0)
    frame_min_z = min_corner.z - title_band
    frame_max_z = max_corner.z + max(width, 1.0) * 0.04
    frame_height = max(float(frame_max_z - frame_min_z), 1.0)
    framing_extent = max(width, frame_height, 1.0)
    aspect_ratio = _render_aspect_ratio()

    # Blender's orthographic scale is width-based, so the required vertical
    # extent needs to be converted through the render aspect ratio.
    width_requirement = width + framing_extent * 0.04
    height_requirement = (frame_height + framing_extent * 0.04) * aspect_ratio

    camera_data = camera_object.data
    camera_data.ortho_scale = max(
        width_requirement,
        height_requirement,
    ) * max(float(margin), 1.0)
    camera_data.clip_end = max(depth, framing_extent) * 60.0
    center = Vector(
        (
            (min_corner.x + max_corner.x) * 0.5,
            0.0,
            (frame_min_z + frame_max_z) * 0.5,
        )
    )
    focus.location = center
    camera_object.location = center + Vector(
        (0.0, -max(depth, width, frame_height) * 10.0, 0.0)
    )


def render_scene(output_path: str | Path) -> None:
    bpy.context.scene.render.filepath = str(
        Path(output_path).expanduser().resolve()
    )
    bpy.ops.render.render(write_still=True)
    print(
        f"Saved render to {Path(output_path).expanduser().resolve()}",
        flush=True,
    )


def save_blend_scene(output_path: str | Path) -> None:
    blend_path = Path(output_path).expanduser().resolve()
    bpy.ops.wm.save_as_mainfile(
        filepath=str(blend_path),
        check_existing=False,
        copy=True,
    )
    print(f"Saved Blender scene to {blend_path}", flush=True)


def _parse_orientations(values: list[str]) -> list[OrientationSpec]:
    if not values:
        return [
            OrientationSpec(
                key="isometric",
                label="Isometric",
                source="default",
                x_degrees=35.264,
                y_degrees=0.0,
                z_degrees=45.0,
                enabled=True,
            )
        ]
    return [parse_orientation_arg(value) for value in values]


def _parse_bond_thresholds(values: list[str]) -> list[BondThresholdSpec]:
    return [parse_bond_threshold_arg(value) for value in values]


def _resolve_output_targets(
    args: argparse.Namespace,
    orientations: list[OrientationSpec],
) -> list[tuple[OrientationSpec, Path, Path | None]]:
    if args.output and args.output_dir:
        raise ValueError("Use either --output or --output-dir, not both.")
    if args.output:
        if len(orientations) != 1:
            raise ValueError(
                "--output only supports a single orientation. Use --output-dir "
                "for batch rendering."
            )
        png_path = Path(args.output).expanduser().resolve()
        blend_path = (
            png_path.with_suffix(".blend") if args.save_blend_files else None
        )
        return [(orientations[0], png_path, blend_path)]
    if not args.output_dir:
        raise ValueError("Provide --output-dir for batch rendering.")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return [
        (
            orientation,
            build_render_output_path(
                args.input,
                output_dir,
                orientation,
                default_atom_style=args.atom_style,
                default_render_quality=args.render_quality,
            ),
            (
                build_blend_output_path(
                    args.input,
                    output_dir,
                    orientation,
                    default_atom_style=args.atom_style,
                    default_render_quality=args.render_quality,
                )
                if args.save_blend_files
                else None
            ),
        )
        for orientation in orientations
    ]


def _effective_orientation_render_settings(
    args: argparse.Namespace,
    orientation: OrientationSpec,
) -> dict[str, str | float | int | None]:
    atom_style = orientation.effective_atom_style(args.atom_style)
    render_quality = orientation.effective_render_quality(args.render_quality)
    lighting_level = orientation.effective_lighting_level(args.lighting_level)
    atom_defaults = atom_style_defaults(atom_style)
    quality_defaults = RENDER_QUALITY_DEFAULTS[render_quality]
    return {
        "atom_style": atom_style,
        "render_quality": render_quality,
        "samples": (
            max(int(args.samples), 32)
            if args.samples is not None
            else int(quality_defaults["samples"])
        ),
        "atom_scale": (
            float(args.atom_scale) if args.atom_scale is not None else None
        ),
        "bond_radius": (
            float(args.bond_radius)
            if args.bond_radius is not None
            else float(atom_defaults["bond_radius"])
        ),
        "bond_color_mode": (
            str(args.bond_color_mode)
            if args.bond_color_mode
            else str(atom_defaults["bond_color_mode"])
        ),
        "camera_margin": (
            float(args.camera_margin)
            if args.camera_margin is not None
            else float(quality_defaults["camera_margin"])
        ),
        "lighting_level": lighting_level,
    }


def _apply_orientation(
    root: bpy.types.Object,
    orientation: OrientationSpec,
) -> None:
    root.rotation_euler = tuple(
        math.radians(value) for value in orientation.euler_degrees
    )


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(_blend_args(argv))
    set_custom_aesthetics(
        tuple(
            parse_custom_aesthetic_arg(text) for text in args.custom_aesthetic
        )
    )
    args.atom_style = normalize_atom_style(args.atom_style)
    args.render_quality = normalize_render_quality(args.render_quality)
    args.lighting_level = normalize_lighting_level(args.lighting_level)
    args.bond_pair_threshold = _parse_bond_thresholds(args.bond_pair_threshold)

    comment, atoms = parse_structure(
        args.input,
        hide_hydrogen=args.hide_hydrogen,
    )
    title = infer_title(
        args.input,
        structure_comment=comment,
        explicit_title=args.title,
        uppercase=args.uppercase_title,
    )
    orientations = _parse_orientations(args.orientation)
    output_targets = _resolve_output_targets(args, orientations)

    total_targets = len(output_targets)
    for index, (orientation, output_path, blend_path) in enumerate(
        output_targets,
        start=1,
    ):
        effective_settings = _effective_orientation_render_settings(
            args,
            orientation,
        )
        progress_label = output_path.stem
        print(
            f"BLENDERXYZ_PROGRESS_START {index} {total_targets} "
            f"{progress_label}",
            flush=True,
        )
        clear_scene()
        setup_render(
            width=args.width,
            height=args.height,
            samples=int(effective_settings["samples"]),
            sample_floor_override=args.sample_floor,
            transparent=args.transparent,
            render_quality=str(effective_settings["render_quality"]),
            lighting_level=int(effective_settings["lighting_level"]),
            cycles_device=str(args.execution_device),
        )
        setup_world(
            transparent=args.transparent,
            atom_style=str(effective_settings["atom_style"]),
        )

        root, structure_objects = build_structure(
            atoms,
            atom_style=str(effective_settings["atom_style"]),
            atom_scale=(
                float(effective_settings["atom_scale"])
                if effective_settings["atom_scale"] is not None
                else None
            ),
            bond_radius=float(effective_settings["bond_radius"]),
            bond_threshold=args.bond_threshold,
            pair_thresholds=args.bond_pair_threshold,
            bond_color_mode=str(effective_settings["bond_color_mode"]),
        )
        bpy.context.view_layer.update()
        setup_lighting(
            world_bounds(structure_objects),
            atom_style=str(effective_settings["atom_style"]),
            render_quality=str(effective_settings["render_quality"]),
            lighting_level=int(effective_settings["lighting_level"]),
        )
        camera_object, focus = create_camera_rig()
        title_object = None
        if not args.hide_title:
            title_object = create_title_object(title)

        _apply_orientation(root, orientation)
        bpy.context.view_layer.update()
        structure_bounds = world_bounds(structure_objects)

        title_band = 0.0
        if title_object is not None:
            structure_height = max(
                structure_bounds[1].z - structure_bounds[0].z,
                1.0,
            )
            title_band = structure_height * 0.28
            place_title_object(
                title_object,
                text=title,
                structure_bounds=structure_bounds,
                scale_factor=args.title_scale,
                vertical_band=title_band,
            )

        frame_camera(
            camera_object,
            focus,
            structure_bounds=structure_bounds,
            margin=float(effective_settings["camera_margin"]),
            title_band=title_band,
        )
        bpy.context.view_layer.update()
        render_scene(output_path)
        if blend_path is not None:
            save_blend_scene(blend_path)
        print(
            f"BLENDERXYZ_PROGRESS_DONE {index} {total_targets} "
            f"{progress_label}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
