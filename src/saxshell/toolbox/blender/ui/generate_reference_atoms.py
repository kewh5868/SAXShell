from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "saxshell_mplconfig"),
)

from saxshell.toolbox.blender.common import (
    ATOM_STYLE_CHOICES,
    LIGHTING_LEVEL_CHOICES,
    OrientationSpec,
    normalize_atom_style,
    normalize_element_symbol,
    normalize_lighting_level,
)
from saxshell.toolbox.blender.workflow import (
    BlenderXYZRenderSettings,
    BlenderXYZRenderWorkflow,
    build_render_output_paths,
)
from saxshell.toolbox.blender.ui.reference_atoms import (
    REFERENCE_ATOM_ELEMENTS,
    REFERENCE_ATOM_RENDER_QUALITY,
    iter_reference_atom_matrix,
    reference_atom_asset_dir,
    reference_atom_key,
    reference_atom_path,
)


def _write_reference_xyz(path: Path, *, element: str) -> Path:
    symbol = normalize_element_symbol(element)
    path.write_text(
        "1\n"
        f"{symbol} Reference\n"
        f"{symbol} 0.0 0.0 0.0\n",
        encoding="utf-8",
    )
    return path


def _build_reference_orientations(
    *,
    element: str,
    atom_styles: tuple[str, ...] | None = None,
    lighting_levels: tuple[int, ...] | None = None,
) -> tuple[OrientationSpec, ...]:
    symbol = normalize_element_symbol(element)
    allowed_styles = atom_styles or tuple(ATOM_STYLE_CHOICES)
    allowed_lighting = lighting_levels or tuple(LIGHTING_LEVEL_CHOICES)
    return tuple(
        OrientationSpec(
            key=reference_atom_key(atom_style, lighting_level, symbol),
            label=f"{symbol} {atom_style} Lighting {lighting_level}",
            source="reference",
            x_degrees=0.0,
            y_degrees=0.0,
            z_degrees=0.0,
            enabled=True,
            atom_style=atom_style,
            render_quality=REFERENCE_ATOM_RENDER_QUALITY,
            lighting_level=lighting_level,
        )
        for matrix_element, atom_style, lighting_level in iter_reference_atom_matrix(
            elements=(symbol,)
        )
        if atom_style in allowed_styles and lighting_level in allowed_lighting
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="generate_reference_atoms.py",
        description=(
            "Render carbon and sulfur reference atoms for every Blender "
            "aesthetic and "
            "lighting-level combination."
        ),
    )
    parser.add_argument(
        "--blender-executable",
        type=Path,
        help="Optional Blender executable or .app bundle override.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=512,
        help="Output width and height in pixels.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=96,
        help="Cycles sample count per reference render.",
    )
    parser.add_argument(
        "--atom-style",
        action="append",
        choices=ATOM_STYLE_CHOICES,
        help="Render only the selected aesthetic. Repeat as needed.",
    )
    parser.add_argument(
        "--lighting-level",
        action="append",
        type=int,
        choices=LIGHTING_LEVEL_CHOICES,
        help="Render only the selected lighting level. Repeat as needed.",
    )
    parser.add_argument(
        "--element",
        action="append",
        choices=REFERENCE_ATOM_ELEMENTS,
        help="Render only the selected reference atom element. Repeat as needed.",
    )
    parser.add_argument(
        "--execution-device",
        choices=("auto", "cpu", "gpu"),
        default="cpu",
        help="Cycles device override passed to the Blender render script.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    asset_dir = reference_atom_asset_dir()
    asset_dir.mkdir(parents=True, exist_ok=True)
    selected_elements = (
        tuple(normalize_element_symbol(value) for value in args.element)
        if args.element
        else REFERENCE_ATOM_ELEMENTS
    )
    selected_styles = (
        tuple(normalize_atom_style(value) for value in args.atom_style)
        if args.atom_style
        else None
    )
    selected_lighting = (
        tuple(
            normalize_lighting_level(value)
            for value in args.lighting_level
        )
        if args.lighting_level
        else None
    )
    clean_all = (
        not args.atom_style
        and not args.lighting_level
        and not args.element
    )
    if clean_all:
        for path in asset_dir.glob("*.png"):
            path.unlink()
    else:
        for element, atom_style, lighting_level in iter_reference_atom_matrix(
            elements=selected_elements
        ):
            if selected_styles and atom_style not in selected_styles:
                continue
            if selected_lighting and lighting_level not in selected_lighting:
                continue
            target_path = reference_atom_path(
                atom_style,
                lighting_level,
                element,
            )
            if target_path.is_file():
                target_path.unlink()

    with tempfile.TemporaryDirectory(prefix="saxshell_ref_atoms_") as tmp_root:
        temp_root = Path(tmp_root)
        for element in selected_elements:
            symbol = normalize_element_symbol(element)
            input_path = _write_reference_xyz(
                temp_root / f"{symbol.lower()}_reference.xyz",
                element=symbol,
            )
            orientations = _build_reference_orientations(
                element=symbol,
                atom_styles=selected_styles,
                lighting_levels=selected_lighting,
            )
            if not orientations:
                continue
            workflow = BlenderXYZRenderWorkflow(
                BlenderXYZRenderSettings(
                    input_path=input_path,
                    output_dir=temp_root,
                    orientations=orientations,
                    blender_executable=args.blender_executable,
                    width=int(args.size),
                    height=int(args.size),
                    samples=max(int(args.samples), 32),
                    sample_floor_override=max(int(args.samples), 32),
                    render_quality=REFERENCE_ATOM_RENDER_QUALITY,
                    render_title=False,
                    transparent=True,
                    execution_device=str(args.execution_device),
                )
            )
            prepared = workflow.prepare_settings()
            result = workflow.run_streaming()
            if not result.output_paths:
                raise SystemExit(
                    f"Reference atom render failed for {symbol}: no output paths were produced."
                )

            output_paths = build_render_output_paths(
                prepared.input_path,
                prepared.output_dir,
                prepared.orientations,
                default_atom_style=prepared.atom_style,
                default_render_quality=prepared.render_quality,
            )

            for orientation, output_path in zip(orientations, output_paths):
                target_path = reference_atom_path(
                    orientation.effective_atom_style(),
                    orientation.effective_lighting_level(),
                    symbol,
                )
                shutil.copy2(output_path, target_path)
                print(f"Saved {target_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
