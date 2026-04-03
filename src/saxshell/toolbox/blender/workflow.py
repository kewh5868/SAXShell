from __future__ import annotations

import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .common import (
    COVALENT_RADII,
    DEFAULT_ATOM_STYLE,
    DEFAULT_BOND_THRESHOLD_SCALE,
    DEFAULT_LEGEND_FONT,
    DEFAULT_LIGHTING_LEVEL,
    DEFAULT_RENDER_HEIGHT,
    DEFAULT_RENDER_QUALITY,
    DEFAULT_RENDER_WIDTH,
    BondThresholdSpec,
    CustomAestheticSpec,
    OrientationSpec,
    atom_style_base,
    bond_threshold_lookup,
    build_blend_output_path,
    build_legend_output_path,
    build_render_output_path,
    complete_bond_threshold_specs,
    custom_aesthetics,
    encode_bond_threshold_arg,
    encode_custom_aesthetic_arg,
    encode_orientation_arg,
    infer_title,
    normalize_atom_style,
    normalize_element_pair,
    normalize_lighting_level,
    normalize_render_quality,
    sanitize_orientation_key,
    set_custom_aesthetics,
    style_atom_color,
)

_PROGRESS_START_PREFIX = "BLENDERXYZ_PROGRESS_START "
_PROGRESS_DONE_PREFIX = "BLENDERXYZ_PROGRESS_DONE "

_COMMON_BLENDER_EXECUTABLES = (
    Path("/Applications/Blender.app/Contents/MacOS/Blender"),
)
_XYZ_SUFFIXES = {".xyz"}
_PDB_SUFFIXES = {".pdb", ".ent"}


@dataclass(slots=True, frozen=True)
class PreviewAtomRecord:
    element: str
    position: tuple[float, float, float]


@dataclass(slots=True, frozen=True)
class BlenderPreviewStructure:
    input_path: Path
    structure_comment: str
    atoms: tuple[PreviewAtomRecord, ...]
    bonds: tuple[tuple[int, int], ...]
    bond_thresholds: tuple[BondThresholdSpec, ...] = ()


@dataclass(slots=True, frozen=True)
class BlenderXYZRenderSettings:
    input_path: Path
    output_dir: Path
    orientations: tuple[OrientationSpec, ...] = ()
    title: str | None = None
    blender_executable: str | Path | None = None
    width: int = DEFAULT_RENDER_WIDTH
    height: int = DEFAULT_RENDER_HEIGHT
    samples: int | None = None
    atom_style: str = DEFAULT_ATOM_STYLE
    atom_scale: float | None = None
    bond_radius: float | None = None
    render_quality: str = DEFAULT_RENDER_QUALITY
    render_title: bool = False
    hide_hydrogen: bool = False
    bond_color_mode: str | None = None
    camera_margin: float | None = None
    bond_thresholds: tuple[BondThresholdSpec, ...] = ()
    transparent: bool = True
    save_blend_file: bool = False
    legend_font_family: str = DEFAULT_LEGEND_FONT
    lighting_level: int = DEFAULT_LIGHTING_LEVEL
    custom_aesthetics: tuple[CustomAestheticSpec, ...] | None = None
    execution_device: str = "auto"
    sample_floor_override: int | None = None


@dataclass(slots=True, frozen=True)
class BlenderXYZRenderResult:
    command: tuple[str, ...]
    output_dir: Path
    output_paths: tuple[Path, ...]
    blend_paths: tuple[Path, ...]
    legend_paths: tuple[Path, ...]
    stdout: str
    stderr: str


def render_script_path() -> Path:
    return Path(__file__).resolve().with_name("render_xyz_publication.py")


def resolve_desktop_dir() -> Path:
    return Path.home() / "Desktop"


def suggest_output_dir() -> Path:
    return resolve_desktop_dir()


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


def read_xyz_comment(path: str | Path) -> str:
    lines = _read_text_lines(path)
    if len(lines) < 2:
        return ""
    return lines[1].strip()


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


def read_structure_comment(path: str | Path) -> str:
    structure_format = detect_structure_format(path)
    if structure_format == "pdb":
        return read_pdb_comment(path)
    return read_xyz_comment(path)


def parse_xyz_records(
    path: str | Path,
    *,
    hide_hydrogen: bool = False,
) -> tuple[str, tuple[PreviewAtomRecord, ...]]:
    xyz_path = Path(path).expanduser().resolve()
    lines = xyz_path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 3:
        raise ValueError(f"XYZ file is too short: {xyz_path}")
    try:
        atom_count = int(lines[0].strip())
    except ValueError as exc:
        raise ValueError(f"Invalid XYZ atom count in {xyz_path}") from exc

    comment = lines[1].strip() if len(lines) > 1 else ""
    atoms: list[PreviewAtomRecord] = []
    for line in lines[2 : 2 + atom_count]:
        tokens = line.split()
        if len(tokens) < 4:
            continue
        element = _normalized_element(tokens[0])
        if hide_hydrogen and element == "H":
            continue
        atoms.append(
            PreviewAtomRecord(
                element=element,
                position=(
                    float(tokens[1]),
                    float(tokens[2]),
                    float(tokens[3]),
                ),
            )
        )
    if not atoms:
        raise ValueError(f"No atoms were parsed from {xyz_path}")

    positions = atom_positions_array(tuple(atoms))
    centroid = positions.mean(axis=0)
    centered_atoms = tuple(
        PreviewAtomRecord(
            element=atom.element,
            position=tuple((positions[index] - centroid).tolist()),
        )
        for index, atom in enumerate(atoms)
    )
    return comment, centered_atoms


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


def parse_pdb_records(
    path: str | Path,
    *,
    hide_hydrogen: bool = False,
) -> tuple[str, tuple[PreviewAtomRecord, ...]]:
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

    atoms: list[PreviewAtomRecord] = []
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
            PreviewAtomRecord(
                element=element,
                position=(x_value, y_value, z_value),
            )
        )
    if not atoms:
        raise ValueError(f"No atoms were parsed from PDB file: {pdb_path}")

    positions = atom_positions_array(tuple(atoms))
    centroid = positions.mean(axis=0)
    centered_atoms = tuple(
        PreviewAtomRecord(
            element=atom.element,
            position=tuple((positions[index] - centroid).tolist()),
        )
        for index, atom in enumerate(atoms)
    )
    return read_pdb_comment(pdb_path), centered_atoms


def parse_structure_records(
    path: str | Path,
    *,
    hide_hydrogen: bool = False,
) -> tuple[str, tuple[PreviewAtomRecord, ...]]:
    structure_format = detect_structure_format(path)
    if structure_format == "pdb":
        return parse_pdb_records(path, hide_hydrogen=hide_hydrogen)
    return parse_xyz_records(path, hide_hydrogen=hide_hydrogen)


def atom_positions_array(
    atoms: tuple[PreviewAtomRecord, ...] | list[PreviewAtomRecord],
) -> np.ndarray:
    return np.asarray([atom.position for atom in atoms], dtype=float)


def build_bond_thresholds_for_structure(
    atoms: tuple[PreviewAtomRecord, ...] | list[PreviewAtomRecord],
    *,
    overrides: (
        tuple[BondThresholdSpec, ...] | list[BondThresholdSpec] | None
    ) = None,
    threshold_scale: float = DEFAULT_BOND_THRESHOLD_SCALE,
) -> tuple[BondThresholdSpec, ...]:
    return complete_bond_threshold_specs(
        [atom.element for atom in atoms],
        overrides,
        threshold_scale=threshold_scale,
    )


def detect_bonds_records(
    atoms: tuple[PreviewAtomRecord, ...] | list[PreviewAtomRecord],
    *,
    threshold_scale: float = DEFAULT_BOND_THRESHOLD_SCALE,
    pair_thresholds: (
        tuple[BondThresholdSpec, ...] | list[BondThresholdSpec] | None
    ) = None,
) -> tuple[tuple[int, int], ...]:
    thresholds = build_bond_thresholds_for_structure(
        atoms,
        overrides=pair_thresholds,
        threshold_scale=threshold_scale,
    )
    threshold_lookup = bond_threshold_lookup(thresholds)
    bonds: list[tuple[int, int]] = []
    for left_index, left_atom in enumerate(atoms):
        left_position = np.asarray(left_atom.position, dtype=float)
        for right_index in range(left_index + 1, len(atoms)):
            right_atom = atoms[right_index]
            right_position = np.asarray(right_atom.position, dtype=float)
            threshold_spec = threshold_lookup[
                normalize_element_pair(left_atom.element, right_atom.element)
            ]
            distance = float(np.linalg.norm(left_position - right_position))
            if (
                float(threshold_spec.min_length)
                <= distance
                <= float(threshold_spec.max_length)
            ):
                bonds.append((left_index, right_index))
    return tuple(bonds)


def load_preview_structure(
    path: str | Path,
    *,
    hide_hydrogen: bool = False,
    pair_thresholds: (
        tuple[BondThresholdSpec, ...] | list[BondThresholdSpec] | None
    ) = None,
) -> BlenderPreviewStructure:
    structure_path = Path(path).expanduser().resolve()
    comment, atoms = parse_structure_records(
        structure_path,
        hide_hydrogen=hide_hydrogen,
    )
    thresholds = build_bond_thresholds_for_structure(
        atoms,
        overrides=pair_thresholds,
    )
    return BlenderPreviewStructure(
        input_path=structure_path,
        structure_comment=comment,
        atoms=atoms,
        bonds=detect_bonds_records(atoms, pair_thresholds=thresholds),
        bond_thresholds=thresholds,
    )


def display_radius(element: str, atom_scale: float = 0.62) -> float:
    base = COVALENT_RADII.get(element, 0.85)
    return max(base * atom_scale, 0.18)


def _legend_atom_color(
    element: str,
    *,
    atom_style: str,
) -> tuple[float, float, float, float]:
    return style_atom_color(element, atom_style=atom_style)


def _legend_text_color(atom_style: str) -> tuple[float, float, float, float]:
    style = atom_style_base(atom_style)
    if style in {"flat_diagram", "toon_matte", "crystal_flat"}:
        return (0.16, 0.17, 0.19, 1.0)
    if style in {"poster_pop", "crystal_cartoon"}:
        return (0.14, 0.15, 0.18, 1.0)
    if style in {"paper_gloss", "crystal_shadow_gloss"}:
        return (0.15, 0.14, 0.18, 1.0)
    return (0.18, 0.18, 0.20, 1.0)


def _legend_shadow_color(atom_style: str) -> tuple[float, float, float, float]:
    style = atom_style_base(atom_style)
    if style in {"paper_gloss", "soft_studio", "crystal_shadow_gloss"}:
        return (0.12, 0.10, 0.14, 0.24)
    if style in {"poster_pop", "crystal_cartoon"}:
        return (0.10, 0.10, 0.12, 0.20)
    return (0.10, 0.10, 0.10, 0.16)


def _structure_elements_for_legend(
    structure: BlenderPreviewStructure,
) -> tuple[str, ...]:
    seen: set[str] = set()
    elements: list[str] = []
    for atom in structure.atoms:
        if atom.element in seen:
            continue
        seen.add(atom.element)
        elements.append(atom.element)
    return tuple(elements)


def render_atom_legend_image(
    output_path: str | Path,
    *,
    structure: BlenderPreviewStructure,
    atom_style: str,
    font_family: str,
) -> Path:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.patches import Circle

    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    elements = _structure_elements_for_legend(structure)
    if not elements:
        raise ValueError("No atom types are available for legend rendering.")

    style = atom_style_base(atom_style)
    text_color = _legend_text_color(style)
    shadow_color = _legend_shadow_color(style)
    font_name = (
        font_family or DEFAULT_LEGEND_FONT
    ).strip() or DEFAULT_LEGEND_FONT

    item_width = 1.30
    left_margin = 0.22
    total_width = left_margin * 2.0 + item_width * len(elements)
    total_height = 0.90
    dpi = 220

    figure = Figure(figsize=(total_width, total_height), dpi=dpi)
    FigureCanvasAgg(figure)
    figure.patch.set_alpha(0.0)
    axis = figure.add_axes([0.0, 0.0, 1.0, 1.0])
    axis.set_xlim(0.0, total_width)
    axis.set_ylim(0.0, total_height)
    axis.set_axis_off()
    axis.set_aspect("equal")
    axis.patch.set_alpha(0.0)

    radius = 0.18
    center_y = 0.45
    for index, element in enumerate(elements):
        center_x = left_margin + (index * item_width) + 0.28
        atom_color = _legend_atom_color(element, atom_style=style)
        shadow = Circle(
            (center_x + 0.02, center_y - 0.02),
            radius=radius * 1.02,
            facecolor=shadow_color,
            edgecolor="none",
        )
        sphere = Circle(
            (center_x, center_y),
            radius=radius,
            facecolor=atom_color,
            edgecolor=(0.16, 0.16, 0.18, 0.65),
            linewidth=1.0,
        )
        highlight = Circle(
            (center_x - radius * 0.30, center_y + radius * 0.28),
            radius=radius * 0.24,
            facecolor=(1.0, 1.0, 1.0, 0.92),
            edgecolor="none",
        )
        axis.add_patch(shadow)
        axis.add_patch(sphere)
        axis.add_patch(highlight)
        axis.text(
            center_x + 0.30,
            center_y,
            element,
            va="center",
            ha="left",
            fontsize=18,
            fontfamily=font_name,
            color=text_color,
        )

    figure.savefig(
        output,
        dpi=dpi,
        transparent=True,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    figure.clear()
    return output


def resolve_blender_executable(
    candidate: str | Path | None = None,
) -> str:
    text = "" if candidate is None else str(candidate).strip()
    if text:
        path = Path(text).expanduser()
        if path.suffix.lower() == ".app":
            app_binary = path / "Contents" / "MacOS" / "Blender"
            if app_binary.is_file():
                return str(app_binary.resolve())
        if path.is_dir():
            for name in ("Blender", "blender"):
                executable = path / name
                if executable.is_file():
                    return str(executable.resolve())
        if path.exists():
            return str(path.resolve())
        discovered = shutil.which(text)
        return discovered or text

    discovered = shutil.which("blender")
    if discovered:
        return discovered
    for executable in _COMMON_BLENDER_EXECUTABLES:
        if executable.is_file():
            return str(executable.resolve())
    return "blender"


def rotation_matrix_from_euler_degrees(
    x_degrees: float,
    y_degrees: float,
    z_degrees: float,
) -> np.ndarray:
    x = math.radians(float(x_degrees))
    y = math.radians(float(y_degrees))
    z = math.radians(float(z_degrees))
    sin_x, cos_x = math.sin(x), math.cos(x)
    sin_y, cos_y = math.sin(y), math.cos(y)
    sin_z, cos_z = math.sin(z), math.cos(z)

    rotation_x = np.array(
        ((1.0, 0.0, 0.0), (0.0, cos_x, -sin_x), (0.0, sin_x, cos_x)),
        dtype=float,
    )
    rotation_y = np.array(
        ((cos_y, 0.0, sin_y), (0.0, 1.0, 0.0), (-sin_y, 0.0, cos_y)),
        dtype=float,
    )
    rotation_z = np.array(
        ((cos_z, -sin_z, 0.0), (sin_z, cos_z, 0.0), (0.0, 0.0, 1.0)),
        dtype=float,
    )
    return rotation_z @ rotation_y @ rotation_x


def matrix_to_euler_xyz_degrees(
    matrix: np.ndarray,
) -> tuple[float, float, float]:
    value = np.asarray(matrix, dtype=float)
    if abs(value[2, 0]) < 0.999999:
        y = math.asin(-value[2, 0])
        cos_y = math.cos(y)
        x = math.atan2(value[2, 1] / cos_y, value[2, 2] / cos_y)
        z = math.atan2(value[1, 0] / cos_y, value[0, 0] / cos_y)
    else:
        y = math.pi / 2.0 if value[2, 0] <= -0.999999 else -math.pi / 2.0
        x = math.atan2(-value[0, 1], value[1, 1])
        z = 0.0
    return (
        math.degrees(x),
        math.degrees(y),
        math.degrees(z),
    )


def compose_euler_degrees(
    base: tuple[float, float, float],
    extra: tuple[float, float, float],
) -> tuple[float, float, float]:
    composed = rotation_matrix_from_euler_degrees(
        *extra
    ) @ rotation_matrix_from_euler_degrees(*base)
    return matrix_to_euler_xyz_degrees(composed)


def apply_orientation_to_positions(
    positions: np.ndarray,
    orientation: OrientationSpec | tuple[float, float, float],
) -> np.ndarray:
    if isinstance(orientation, OrientationSpec):
        degrees = orientation.euler_degrees
    else:
        degrees = orientation
    matrix = rotation_matrix_from_euler_degrees(*degrees)
    return positions @ matrix.T


def _stabilize_principal_axes(axes: np.ndarray) -> np.ndarray:
    stabilized = np.asarray(axes, dtype=float).copy()
    for column in range(stabilized.shape[1]):
        axis = stabilized[:, column]
        dominant_index = int(np.argmax(np.abs(axis)))
        if axis[dominant_index] < 0.0:
            stabilized[:, column] *= -1.0
    if np.linalg.det(stabilized) < 0.0:
        stabilized[:, -1] *= -1.0
    return stabilized


def compute_principal_axes(positions: np.ndarray) -> np.ndarray:
    centered = np.asarray(positions, dtype=float)
    centered = centered - centered.mean(axis=0)
    if centered.shape[0] < 2 or np.allclose(centered, 0.0):
        return np.eye(3, dtype=float)
    covariance = centered.T @ centered
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    axes = eigenvectors[:, order]
    return _stabilize_principal_axes(axes)


def canonical_alignment_euler_degrees(
    atoms: tuple[PreviewAtomRecord, ...] | list[PreviewAtomRecord],
) -> tuple[float, float, float]:
    positions = atom_positions_array(tuple(atoms))
    axes = compute_principal_axes(positions)
    basis = np.column_stack((axes[:, 0], axes[:, 2], axes[:, 1]))
    return matrix_to_euler_xyz_degrees(basis.T)


def build_preset_orientations() -> tuple[OrientationSpec, ...]:
    return (
        OrientationSpec(
            key="isometric",
            label="Isometric",
            source="preset",
            x_degrees=35.264,
            y_degrees=0.0,
            z_degrees=45.0,
        ),
        OrientationSpec(
            key="down_a_axis",
            label="Down a-axis",
            source="preset",
            x_degrees=0.0,
            y_degrees=0.0,
            z_degrees=90.0,
        ),
        OrientationSpec(
            key="down_b_axis",
            label="Down b-axis",
            source="preset",
            x_degrees=0.0,
            y_degrees=0.0,
            z_degrees=0.0,
        ),
        OrientationSpec(
            key="down_c_axis",
            label="Down c-axis",
            source="preset",
            x_degrees=-90.0,
            y_degrees=0.0,
            z_degrees=0.0,
        ),
    )


def build_photoshoot_orientations(
    atoms: tuple[PreviewAtomRecord, ...] | list[PreviewAtomRecord],
) -> tuple[OrientationSpec, ...]:
    base = canonical_alignment_euler_degrees(tuple(atoms))
    photo_variants = (
        ("photoshoot_01", "Photoshoot 1", (0.0, 0.0, 0.0)),
        ("photoshoot_02", "Photoshoot 2", (24.0, 0.0, 32.0)),
        ("photoshoot_03", "Photoshoot 3", (-18.0, 26.0, -28.0)),
    )
    orientations: list[OrientationSpec] = []
    for key, label, extra in photo_variants:
        x_degrees, y_degrees, z_degrees = compose_euler_degrees(base, extra)
        orientations.append(
            OrientationSpec(
                key=key,
                label=label,
                source="photoshoot",
                x_degrees=x_degrees,
                y_degrees=y_degrees,
                z_degrees=z_degrees,
            )
        )
    return tuple(orientations)


def build_default_orientation_catalog(
    structure: BlenderPreviewStructure,
    *,
    include_presets: bool = True,
    include_photoshoot: bool = True,
) -> tuple[OrientationSpec, ...]:
    orientations: list[OrientationSpec] = []
    if include_presets:
        orientations.extend(build_preset_orientations())
    if include_photoshoot:
        orientations.extend(build_photoshoot_orientations(structure.atoms))
    return tuple(orientations)


def normalize_orientations(
    orientations: tuple[OrientationSpec, ...] | list[OrientationSpec],
) -> tuple[OrientationSpec, ...]:
    normalized: list[OrientationSpec] = []
    seen: dict[str, int] = {}
    for index, orientation in enumerate(orientations, start=1):
        if not orientation.enabled:
            continue
        label = orientation.label.strip() or f"Orientation {index}"
        base_key = sanitize_orientation_key(orientation.key or label)
        suffix = seen.get(base_key, 0)
        seen[base_key] = suffix + 1
        key = base_key if suffix == 0 else f"{base_key}_{suffix + 1:02d}"
        normalized.append(
            OrientationSpec(
                key=key,
                label=label,
                source=orientation.source or "custom",
                x_degrees=float(orientation.x_degrees),
                y_degrees=float(orientation.y_degrees),
                z_degrees=float(orientation.z_degrees),
                enabled=True,
                atom_style=(
                    orientation.effective_atom_style()
                    if orientation.atom_style
                    else None
                ),
                render_quality=(
                    orientation.effective_render_quality()
                    if orientation.render_quality
                    else None
                ),
                lighting_level=(
                    orientation.effective_lighting_level()
                    if orientation.lighting_level is not None
                    else None
                ),
                save_legend=bool(orientation.save_legend),
            )
        )
    return tuple(normalized)


def build_render_output_paths(
    input_path: str | Path,
    output_dir: str | Path,
    orientations: tuple[OrientationSpec, ...] | list[OrientationSpec],
    *,
    default_atom_style: str | None = None,
    default_render_quality: str | None = None,
) -> tuple[Path, ...]:
    return tuple(
        build_render_output_path(
            input_path,
            output_dir,
            orientation,
            default_atom_style=default_atom_style,
            default_render_quality=default_render_quality,
        )
        for orientation in normalize_orientations(tuple(orientations))
    )


def build_blend_output_paths(
    input_path: str | Path,
    output_dir: str | Path,
    orientations: tuple[OrientationSpec, ...] | list[OrientationSpec],
    *,
    default_atom_style: str | None = None,
    default_render_quality: str | None = None,
) -> tuple[Path, ...]:
    return tuple(
        build_blend_output_path(
            input_path,
            output_dir,
            orientation,
            default_atom_style=default_atom_style,
            default_render_quality=default_render_quality,
        )
        for orientation in normalize_orientations(tuple(orientations))
    )


def build_legend_output_paths(
    input_path: str | Path,
    output_dir: str | Path,
    orientations: tuple[OrientationSpec, ...] | list[OrientationSpec],
    *,
    default_atom_style: str | None = None,
    default_render_quality: str | None = None,
) -> tuple[Path, ...]:
    return tuple(
        build_legend_output_path(
            input_path,
            output_dir,
            orientation,
            default_atom_style=default_atom_style,
            default_render_quality=default_render_quality,
        )
        for orientation in normalize_orientations(tuple(orientations))
        if orientation.save_legend
    )


class BlenderXYZRenderWorkflow:
    """Launch the Blender XYZ publication renderer as a subprocess."""

    def __init__(self, settings: BlenderXYZRenderSettings) -> None:
        self.settings = settings

    def prepare_settings(self) -> BlenderXYZRenderSettings:
        if self.settings.custom_aesthetics is not None:
            set_custom_aesthetics(self.settings.custom_aesthetics)
        input_path = Path(self.settings.input_path).expanduser().resolve()
        output_dir = Path(self.settings.output_dir).expanduser().resolve()
        atom_style = normalize_atom_style(self.settings.atom_style)
        render_quality = normalize_render_quality(self.settings.render_quality)
        structure = load_preview_structure(
            input_path,
            hide_hydrogen=self.settings.hide_hydrogen,
            pair_thresholds=self.settings.bond_thresholds,
        )
        title = None
        if self.settings.render_title:
            title = infer_title(
                input_path,
                structure_comment=structure.structure_comment,
                explicit_title=self.settings.title,
            )

        orientations = normalize_orientations(
            tuple(self.settings.orientations)
        )
        if not orientations:
            orientations = (
                OrientationSpec(
                    key="isometric",
                    label="Isometric",
                    source="preset",
                    x_degrees=35.264,
                    y_degrees=0.0,
                    z_degrees=45.0,
                    enabled=True,
                ),
            )

        requested_samples = self.settings.samples
        samples = (
            max(int(requested_samples), 32)
            if requested_samples is not None
            else None
        )
        sample_floor_override = (
            max(int(self.settings.sample_floor_override), 1)
            if self.settings.sample_floor_override is not None
            else None
        )

        atom_scale = (
            float(self.settings.atom_scale)
            if self.settings.atom_scale is not None
            else None
        )
        bond_radius = (
            float(self.settings.bond_radius)
            if self.settings.bond_radius is not None
            else None
        )
        bond_color_mode = (
            str(self.settings.bond_color_mode)
            if self.settings.bond_color_mode
            else None
        )
        camera_margin = (
            float(self.settings.camera_margin)
            if self.settings.camera_margin is not None
            else None
        )
        lighting_level = normalize_lighting_level(self.settings.lighting_level)
        bond_thresholds = build_bond_thresholds_for_structure(
            structure.atoms,
            overrides=self.settings.bond_thresholds,
        )
        available_custom_aesthetics = (
            tuple(self.settings.custom_aesthetics)
            if self.settings.custom_aesthetics is not None
            else custom_aesthetics()
        )
        referenced_custom_keys = {
            atom_style,
            *(
                orientation.atom_style
                for orientation in orientations
                if orientation.atom_style
            ),
        }
        referenced_custom_aesthetics = tuple(
            spec
            for spec in available_custom_aesthetics
            if spec.key in referenced_custom_keys
        )

        return BlenderXYZRenderSettings(
            input_path=input_path,
            output_dir=output_dir,
            orientations=orientations,
            title=title,
            blender_executable=resolve_blender_executable(
                self.settings.blender_executable
            ),
            width=int(self.settings.width),
            height=int(self.settings.height),
            samples=samples,
            atom_style=atom_style,
            atom_scale=atom_scale,
            bond_radius=bond_radius,
            render_quality=render_quality,
            render_title=bool(self.settings.render_title),
            hide_hydrogen=bool(self.settings.hide_hydrogen),
            bond_color_mode=bond_color_mode,
            camera_margin=camera_margin,
            bond_thresholds=bond_thresholds,
            transparent=True,
            save_blend_file=bool(self.settings.save_blend_file),
            legend_font_family=(
                str(self.settings.legend_font_family).strip()
                or DEFAULT_LEGEND_FONT
            ),
            lighting_level=lighting_level,
            custom_aesthetics=referenced_custom_aesthetics,
            execution_device=(
                str(self.settings.execution_device).strip().lower()
                if str(self.settings.execution_device).strip().lower()
                in {"auto", "cpu", "gpu"}
                else "auto"
            ),
            sample_floor_override=sample_floor_override,
        )

    def build_command(
        self,
        settings: BlenderXYZRenderSettings | None = None,
    ) -> list[str]:
        prepared = self.prepare_settings() if settings is None else settings
        command = [
            str(prepared.blender_executable),
            "--background",
            "--factory-startup",
            "--python",
            str(render_script_path()),
            "--",
            "--input",
            str(prepared.input_path),
            "--output-dir",
            str(prepared.output_dir),
            "--width",
            str(prepared.width),
            "--height",
            str(prepared.height),
            "--atom-style",
            prepared.atom_style,
            "--render-quality",
            prepared.render_quality,
            "--lighting-level",
            str(prepared.lighting_level),
            "--execution-device",
            str(prepared.execution_device),
        ]
        if prepared.samples is not None:
            command.extend(["--samples", str(prepared.samples)])
        if prepared.sample_floor_override is not None:
            command.extend(
                [
                    "--sample-floor",
                    str(int(prepared.sample_floor_override)),
                ]
            )
        if prepared.atom_scale is not None:
            command.extend(
                ["--atom-scale", f"{float(prepared.atom_scale):.3f}"]
            )
        if prepared.bond_radius is not None:
            command.extend(
                ["--bond-radius", f"{float(prepared.bond_radius):.3f}"]
            )
        if prepared.bond_color_mode:
            command.extend(["--bond-color-mode", prepared.bond_color_mode])
        if prepared.camera_margin is not None:
            command.extend(
                [
                    "--camera-margin",
                    f"{float(prepared.camera_margin):.3f}",
                ]
            )
        if prepared.render_title and prepared.title:
            command.extend(["--title", prepared.title])
        else:
            command.append("--hide-title")
        if prepared.hide_hydrogen:
            command.append("--hide-hydrogen")
        command.append("--transparent")
        if prepared.save_blend_file:
            command.append("--save-blend-files")
        for spec in prepared.custom_aesthetics or ():
            command.extend(
                ["--custom-aesthetic", encode_custom_aesthetic_arg(spec)]
            )
        for bond_threshold in prepared.bond_thresholds:
            command.extend(
                [
                    "--bond-pair-threshold",
                    encode_bond_threshold_arg(bond_threshold),
                ]
            )
        for orientation in prepared.orientations:
            command.extend(
                ["--orientation", encode_orientation_arg(orientation)]
            )
        return command

    def run(self) -> BlenderXYZRenderResult:
        return self.run_streaming()

    def run_streaming(
        self,
        *,
        line_callback=None,
        progress_callback=None,
    ) -> BlenderXYZRenderResult:
        prepared = self.prepare_settings()
        if not prepared.input_path.is_file():
            raise FileNotFoundError(
                f"Structure input file was not found: {prepared.input_path}"
            )
        prepared.output_dir.mkdir(parents=True, exist_ok=True)

        command = self.build_command(prepared)
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                "Blender executable was not found. Select the Blender "
                "application/executable, or ensure 'blender' is on PATH."
            ) from exc

        collected_lines: list[str] = []
        stdout_pipe = process.stdout
        assert stdout_pipe is not None
        for raw_line in stdout_pipe:
            line = raw_line.rstrip()
            if not line:
                continue
            progress_event = self._parse_progress_event(line)
            if progress_event is not None:
                if progress_callback is not None:
                    progress_callback(*progress_event)
                continue
            collected_lines.append(line)
            if line_callback is not None:
                line_callback(line)
        process.wait()

        stdout = "\n".join(collected_lines).strip()
        stderr = ""
        if process.returncode != 0:
            parts = [
                "Blender render failed.",
                f"Command: {' '.join(command)}",
            ]
            if stdout:
                parts.append(f"output:\n{stdout}")
            raise RuntimeError("\n\n".join(parts))

        legend_paths = ()
        if any(
            orientation.save_legend for orientation in prepared.orientations
        ):
            structure = load_preview_structure(
                prepared.input_path,
                hide_hydrogen=prepared.hide_hydrogen,
            )
            legend_paths_list: list[Path] = []
            for orientation in prepared.orientations:
                if not orientation.save_legend:
                    continue
                legend_path = build_legend_output_path(
                    prepared.input_path,
                    prepared.output_dir,
                    orientation,
                    default_atom_style=prepared.atom_style,
                    default_render_quality=prepared.render_quality,
                )
                render_atom_legend_image(
                    legend_path,
                    structure=structure,
                    atom_style=orientation.effective_atom_style(
                        prepared.atom_style
                    ),
                    font_family=prepared.legend_font_family,
                )
                legend_paths_list.append(legend_path)
            legend_paths = tuple(legend_paths_list)

        return BlenderXYZRenderResult(
            command=tuple(command),
            output_dir=prepared.output_dir,
            output_paths=build_render_output_paths(
                prepared.input_path,
                prepared.output_dir,
                prepared.orientations,
                default_atom_style=prepared.atom_style,
                default_render_quality=prepared.render_quality,
            ),
            blend_paths=(
                build_blend_output_paths(
                    prepared.input_path,
                    prepared.output_dir,
                    prepared.orientations,
                    default_atom_style=prepared.atom_style,
                    default_render_quality=prepared.render_quality,
                )
                if prepared.save_blend_file
                else ()
            ),
            legend_paths=legend_paths,
            stdout=stdout,
            stderr=stderr,
        )

    @staticmethod
    def _parse_progress_event(
        line: str,
    ) -> tuple[str, int, int, str] | None:
        for prefix, event_name in (
            (_PROGRESS_START_PREFIX, "start"),
            (_PROGRESS_DONE_PREFIX, "done"),
        ):
            if not line.startswith(prefix):
                continue
            payload = line[len(prefix) :].strip()
            parts = payload.split(" ", 2)
            if len(parts) < 2:
                return None
            try:
                current = int(parts[0])
                total = int(parts[1])
            except ValueError:
                return None
            label = parts[2].strip() if len(parts) > 2 else ""
            return (event_name, current, total, label)
        return None


__all__ = [
    "BlenderPreviewStructure",
    "BlenderXYZRenderResult",
    "BlenderXYZRenderSettings",
    "BlenderXYZRenderWorkflow",
    "PreviewAtomRecord",
    "apply_orientation_to_positions",
    "atom_positions_array",
    "build_bond_thresholds_for_structure",
    "build_default_orientation_catalog",
    "build_blend_output_paths",
    "build_legend_output_paths",
    "build_photoshoot_orientations",
    "build_preset_orientations",
    "build_render_output_paths",
    "canonical_alignment_euler_degrees",
    "compose_euler_degrees",
    "detect_bonds_records",
    "detect_structure_format",
    "display_radius",
    "infer_title",
    "load_preview_structure",
    "parse_pdb_records",
    "parse_structure_records",
    "render_atom_legend_image",
    "parse_xyz_records",
    "read_pdb_comment",
    "read_structure_comment",
    "read_xyz_comment",
    "render_script_path",
    "resolve_blender_executable",
    "resolve_desktop_dir",
    "suggest_output_dir",
]
