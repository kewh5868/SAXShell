from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np

from saxshell.toolbox.blender.common import (
    ATOM_STYLE_DEFAULTS,
    CPK_COLORS,
    DEFAULT_ATOM_COLOR,
    DEFAULT_ATOM_STYLE,
    DEFAULT_LIGHTING_LEVEL,
    OrientationSpec,
)
from saxshell.toolbox.blender.workflow import (
    BlenderXYZRenderSettings,
    BlenderXYZRenderWorkflow,
    apply_orientation_to_positions,
    compose_euler_degrees,
    display_radius,
    load_preview_structure,
)

RENDER_AZIM = -90.0
RENDER_ELEV = 0.0

matplotlib.use("Agg")


@dataclass(frozen=True)
class OrientationCase:
    key: str
    label: str
    base_name: str
    correction_name: str
    orientation: OrientationSpec


def _preview_palette() -> tuple[str, tuple[float, float, float]]:
    return ("#fdfbf8", (0.57, 0.53, 0.62))


def _preview_color(element: str) -> tuple[float, float, float, float]:
    return CPK_COLORS.get(element, DEFAULT_ATOM_COLOR)


def _build_orientation_cases() -> list[OrientationCase]:
    base_orientations = {
        "down_b_axis": (0.0, 0.0, 0.0),
        "isometric": (35.264, 0.0, 45.0),
    }
    correction_orientations = {
        "identity": (0.0, 0.0, 0.0),
        "y180": (0.0, 180.0, 0.0),
        "x180": (180.0, 0.0, 0.0),
        "z180": (0.0, 0.0, 180.0),
    }
    cases: list[OrientationCase] = []
    for base_name, base_angles in base_orientations.items():
        for (
            correction_name,
            correction_angles,
        ) in correction_orientations.items():
            x_deg, y_deg, z_deg = compose_euler_degrees(
                base_angles,
                correction_angles,
            )
            key = f"{base_name}_{correction_name}"
            label = f"{base_name} + {correction_name}"
            cases.append(
                OrientationCase(
                    key=key,
                    label=label,
                    base_name=base_name,
                    correction_name=correction_name,
                    orientation=OrientationSpec(
                        key=key,
                        label=label,
                        source="probe",
                        x_degrees=x_deg,
                        y_degrees=y_deg,
                        z_degrees=z_deg,
                        enabled=True,
                        atom_style=DEFAULT_ATOM_STYLE,
                        render_quality="draft",
                        lighting_level=DEFAULT_LIGHTING_LEVEL,
                    ),
                )
            )
    return cases


def _draw_preview_image(
    structure,
    orientation: OrientationSpec,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    atom_style = orientation.effective_atom_style(DEFAULT_ATOM_STYLE)
    rotated = apply_orientation_to_positions(
        np.asarray([atom.position for atom in structure.atoms], dtype=float),
        orientation,
    )
    x_values = rotated[:, 0]
    y_values = rotated[:, 1]
    z_values = rotated[:, 2]
    background_color, bond_color = _preview_palette()
    atom_scale = float(ATOM_STYLE_DEFAULTS[atom_style]["atom_scale"])
    bond_width = max(
        2.4,
        10.0 * float(ATOM_STYLE_DEFAULTS[atom_style]["bond_radius"]),
    )

    figure = plt.figure(figsize=(5.2, 5.2), dpi=200)
    axis = figure.add_subplot(111, projection="3d")
    figure.set_facecolor(background_color)
    axis.set_facecolor(background_color)
    try:
        axis.set_proj_type("ortho")
    except Exception:
        pass
    axis.grid(False)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_zticks([])
    axis.set_xlabel("")
    axis.set_ylabel("")
    axis.set_zlabel("")
    try:
        axis.set_box_aspect((1.0, 1.0, 1.0))
    except Exception:
        pass
    for axis_obj in (axis.xaxis, axis.yaxis, axis.zaxis):
        try:
            axis_obj.line.set_alpha(0.0)
            axis_obj.pane.set_alpha(0.0)
        except Exception:
            pass

    for left_index, right_index in sorted(
        structure.bonds,
        key=lambda pair: float(y_values[pair[0]] + y_values[pair[1]]),
    ):
        start = rotated[left_index]
        end = rotated[right_index]
        axis.plot(
            (start[0], end[0]),
            (start[1], end[1]),
            (start[2], end[2]),
            color=bond_color,
            linewidth=bond_width,
            alpha=0.78,
            solid_capstyle="round",
        )

    for atom_index in np.argsort(y_values):
        atom = structure.atoms[int(atom_index)]
        color = _preview_color(atom.element)
        size = (
            display_radius(atom.element, atom_scale=atom_scale) * 30.0
        ) ** 2
        axis.scatter(
            [x_values[atom_index]],
            [y_values[atom_index]],
            [z_values[atom_index]],
            s=size,
            c=[color],
            edgecolors=[(0.18, 0.18, 0.22)],
            linewidths=0.9,
            alpha=0.96,
            depthshade=False,
        )

    minima = rotated.min(axis=0)
    maxima = rotated.max(axis=0)
    center = (minima + maxima) * 0.5
    radius = max(float(np.max(maxima - minima)) * 0.6, 1.0)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.view_init(elev=RENDER_ELEV, azim=RENDER_AZIM)
    axis.axis("off")
    figure.tight_layout(pad=0.0)
    figure.savefig(output_path, dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)


def _write_compare_image(
    preview_path: Path,
    render_path: Path,
    case: OrientationCase,
    output_path: Path,
) -> None:
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    preview_image = mpimg.imread(preview_path)
    render_image = mpimg.imread(render_path)
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(10.5, 5.3),
        dpi=180,
        facecolor="white",
    )
    for axis, image, title in (
        (axes[0], preview_image, "Visualizer"),
        (axes[1], render_image, "Blender Render"),
    ):
        axis.imshow(image)
        axis.set_title(title, fontsize=11)
        axis.axis("off")
    figure.suptitle(
        (
            f"{case.label}  |  "
            f"Euler = ({case.orientation.x_degrees:.3f}, "
            f"{case.orientation.y_degrees:.3f}, "
            f"{case.orientation.z_degrees:.3f})"
        ),
        fontsize=12,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    input_path = Path(
        "/Users/keithwhite/Desktop/blender_render/xyz_ref/pbi2_3dmso.xyz"
    ).expanduser()
    output_root = Path("/Users/keithwhite/Desktop/blender_render").expanduser()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        output_root / f"orientation_probe_{input_path.stem}_{timestamp}"
    )
    preview_dir = output_dir / "preview"
    render_dir = output_dir / "render"
    compare_dir = output_dir / "compare"
    for directory in (output_dir, preview_dir, render_dir, compare_dir):
        directory.mkdir(parents=True, exist_ok=True)

    structure = load_preview_structure(input_path)
    cases = _build_orientation_cases()

    manifest_path = output_dir / "orientation_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "index",
                "key",
                "label",
                "base_name",
                "correction_name",
                "x_degrees",
                "y_degrees",
                "z_degrees",
            ]
        )
        for index, case in enumerate(cases, start=1):
            writer.writerow(
                [
                    index,
                    case.key,
                    case.label,
                    case.base_name,
                    case.correction_name,
                    f"{case.orientation.x_degrees:.6f}",
                    f"{case.orientation.y_degrees:.6f}",
                    f"{case.orientation.z_degrees:.6f}",
                ]
            )

    settings = BlenderXYZRenderSettings(
        input_path=input_path,
        output_dir=render_dir,
        orientations=tuple(case.orientation for case in cases),
        width=1600,
        height=1600,
        atom_style=DEFAULT_ATOM_STYLE,
        render_quality="draft",
        lighting_level=DEFAULT_LIGHTING_LEVEL,
        transparent=True,
    )
    result = BlenderXYZRenderWorkflow(settings).run_streaming()

    render_map = {
        output_path.stem: output_path for output_path in result.output_paths
    }

    for index, case in enumerate(cases, start=1):
        prefix = f"{index:02d}_{case.key}"
        preview_path = preview_dir / f"{prefix}_visualizer.png"
        _draw_preview_image(structure, case.orientation, preview_path)

        render_match = None
        for stem, path in render_map.items():
            if case.orientation.key in stem:
                render_match = path
                break
        if render_match is None:
            raise FileNotFoundError(
                f"No render output was found for orientation key {case.orientation.key}"
            )
        compare_path = compare_dir / f"{prefix}_compare.png"
        _write_compare_image(
            preview_path,
            render_match,
            case,
            compare_path,
        )

    readme_path = output_dir / "README.txt"
    readme_path.write_text(
        "\n".join(
            [
                f"Input structure: {input_path}",
                f"Visualizer previews: {preview_dir}",
                f"Blender renders: {render_dir}",
                f"Side-by-side comparisons: {compare_dir}",
                f"Manifest: {manifest_path}",
                "",
                "Please tell me which compare image best matches the intended",
                "orientation so I can fix the visualizer/render transform.",
            ]
        ),
        encoding="utf-8",
    )

    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
