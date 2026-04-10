from __future__ import annotations

import argparse
import csv
import json
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from saxshell.saxs.electron_density_mapping import workflow as density_workflow

DEFAULT_SELECTION_PATH = Path(
    "/Users/keithwhite/repos/cluster_extraction/"
    "041_cp2k_pbi2_dmf_0p7M_RT/"
    "clusters_xyz2pdb_splitxyz_f1002_t497p5fs_smartshell/"
    "Pb2I4"
)
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "edm_contiguous_mode_report"
    / "pb2i4_backend_comparison"
)


def _center_mode_label(center_mode: str) -> str:
    if center_mode == "center_of_mass":
        return "Geometric Mass Center"
    if center_mode == "reference_element":
        return "Reference Element"
    return str(center_mode)


def _combo_slug(
    *,
    center_mode: str,
    use_contiguous_frame_mode: bool,
    pin_contiguous_geometric_tracking: bool,
) -> str:
    center_slug = (
        "geom_mass"
        if center_mode == "center_of_mass"
        else str(center_mode).replace("_", "-")
    )
    contiguous_slug = (
        "contiguous-on" if use_contiguous_frame_mode else "contiguous-off"
    )
    pin_slug = "pin-on" if pin_contiguous_geometric_tracking else "pin-off"
    return f"{center_slug}__{contiguous_slug}__{pin_slug}"


def _effective_combo_key(
    *,
    center_mode: str,
    use_contiguous_frame_mode: bool,
    pin_contiguous_geometric_tracking: bool,
) -> str:
    pin_is_effective = bool(
        center_mode == "center_of_mass"
        and use_contiguous_frame_mode
        and pin_contiguous_geometric_tracking
    )
    return _combo_slug(
        center_mode=center_mode,
        use_contiguous_frame_mode=use_contiguous_frame_mode,
        pin_contiguous_geometric_tracking=pin_is_effective,
    )


def _vector_components(
    values: np.ndarray | tuple[float, ...] | list[float],
) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    return (
        float(array[0]),
        float(array[1]),
        float(array[2]),
    )


def _vector_list(
    values: np.ndarray | tuple[float, ...] | list[float],
) -> list[float]:
    return [float(value) for value in np.asarray(values, dtype=float)]


def _profile_metrics(
    result_a: density_workflow.ElectronDensityProfileResult,
    result_b: density_workflow.ElectronDensityProfileResult,
) -> dict[str, float | bool]:
    profile_a = np.asarray(
        result_a.smeared_orientation_average_density,
        dtype=float,
    )
    profile_b = np.asarray(
        result_b.smeared_orientation_average_density,
        dtype=float,
    )
    radial_centers = np.asarray(result_a.radial_centers, dtype=float)
    difference = profile_a - profile_b
    return {
        "profiles_identical": bool(np.allclose(profile_a, profile_b)),
        "max_abs_diff_e_per_a3": float(np.max(np.abs(difference))),
        "mean_abs_diff_e_per_a3": float(np.mean(np.abs(difference))),
        "rms_diff_e_per_a3": float(np.sqrt(np.mean(np.square(difference)))),
        "integrated_abs_diff_e_per_a2": float(
            np.trapezoid(np.abs(difference), radial_centers)
        ),
    }


def _active_center_metrics(
    result_a: density_workflow.ElectronDensityProfileResult,
    result_b: density_workflow.ElectronDensityProfileResult,
) -> dict[str, float | bool]:
    centers_a = {
        Path(entry.file_path): np.asarray(entry.active_center, dtype=float)
        for entry in result_a.member_summaries
    }
    shifts: list[float] = []
    for entry in result_b.member_summaries:
        file_path = Path(entry.file_path)
        if file_path not in centers_a:
            continue
        shifts.append(
            float(
                np.linalg.norm(
                    centers_a[file_path]
                    - np.asarray(entry.active_center, dtype=float)
                )
            )
        )
    if not shifts:
        return {
            "active_centers_identical": True,
            "max_active_center_shift_a": 0.0,
            "mean_active_center_shift_a": 0.0,
        }
    shift_array = np.asarray(shifts, dtype=float)
    return {
        "active_centers_identical": bool(
            np.allclose(shift_array, np.zeros_like(shift_array))
        ),
        "max_active_center_shift_a": float(np.max(shift_array)),
        "mean_active_center_shift_a": float(np.mean(shift_array)),
    }


def _write_profile_csv(
    path: Path,
    result: density_workflow.ElectronDensityProfileResult,
) -> None:
    radial_edges = np.asarray(result.mesh_geometry.radial_edges, dtype=float)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "r_inner_a",
                "r_outer_a",
                "r_center_a",
                "orientation_average_density_e_per_a3",
                "smeared_orientation_average_density_e_per_a3",
                "orientation_density_stddev_e_per_a3",
                "smeared_orientation_density_stddev_e_per_a3",
                "electrons_in_shell",
                "shell_volume_a3",
            ]
        )
        for index, center in enumerate(result.radial_centers):
            writer.writerow(
                [
                    f"{float(radial_edges[index]):.8f}",
                    f"{float(radial_edges[index + 1]):.8f}",
                    f"{float(center):.8f}",
                    f"{float(result.orientation_average_density[index]):.8f}",
                    f"{float(result.smeared_orientation_average_density[index]):.8f}",
                    f"{float(result.orientation_density_stddev[index]):.8f}",
                    f"{float(result.smeared_orientation_density_stddev[index]):.8f}",
                    f"{float(result.shell_electron_counts[index]):.8f}",
                    f"{float(result.shell_volumes[index]):.8f}",
                ]
            )


def _write_detected_set_csv(
    path: Path,
    frame_sets: tuple[
        density_workflow.ElectronDensityContiguousFrameSetSummary,
        ...,
    ],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "set_index",
                "series_label",
                "frame_range_label",
                "frame_count",
                "first_frame_id",
                "last_frame_id",
                "first_file_name",
                "last_file_name",
            ]
        )
        for index, frame_set in enumerate(frame_sets, start=1):
            writer.writerow(
                [
                    index,
                    str(frame_set.series_label),
                    str(frame_set.frame_range_label),
                    int(frame_set.frame_count),
                    int(frame_set.frame_ids[0]),
                    int(frame_set.frame_ids[-1]),
                    frame_set.file_paths[0].name,
                    frame_set.file_paths[-1].name,
                ]
            )


def _write_run_center_csvs(
    output_dir: Path,
    run_slug: str,
    result: density_workflow.ElectronDensityProfileResult,
    frame_sets: tuple[
        density_workflow.ElectronDensityContiguousFrameSetSummary,
        ...,
    ],
) -> tuple[Path, Path]:
    summary_by_path = {
        Path(entry.file_path): entry for entry in result.member_summaries
    }
    frame_csv_path = output_dir / f"{run_slug}__frame_centers.csv"
    set_csv_path = output_dir / f"{run_slug}__set_summary.csv"

    with frame_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "set_index",
                "series_label",
                "frame_range_label",
                "frame_count",
                "frame_index_in_set",
                "frame_id",
                "frame_label",
                "file_name",
                "is_first_frame_in_set",
                "reference_element",
                "active_center_x_a",
                "active_center_y_a",
                "active_center_z_a",
                "center_of_mass_x_a",
                "center_of_mass_y_a",
                "center_of_mass_z_a",
                "geometric_center_x_a",
                "geometric_center_y_a",
                "geometric_center_z_a",
                "reference_element_geometric_center_x_a",
                "reference_element_geometric_center_y_a",
                "reference_element_geometric_center_z_a",
                "active_minus_reference_geom_x_a",
                "active_minus_reference_geom_y_a",
                "active_minus_reference_geom_z_a",
                "active_minus_center_of_mass_x_a",
                "active_minus_center_of_mass_y_a",
                "active_minus_center_of_mass_z_a",
            ]
        )
        for set_index, frame_set in enumerate(frame_sets, start=1):
            for frame_index, file_path in enumerate(
                frame_set.file_paths, start=1
            ):
                summary = summary_by_path[file_path]
                active_center = np.asarray(summary.active_center, dtype=float)
                center_of_mass = np.asarray(
                    summary.center_of_mass, dtype=float
                )
                geometric_center = np.asarray(
                    summary.geometric_center,
                    dtype=float,
                )
                reference_center = np.asarray(
                    summary.reference_element_geometric_center,
                    dtype=float,
                )
                writer.writerow(
                    [
                        set_index,
                        str(frame_set.series_label),
                        str(frame_set.frame_range_label),
                        int(frame_set.frame_count),
                        frame_index,
                        int(frame_set.frame_ids[frame_index - 1]),
                        str(frame_set.frame_labels[frame_index - 1]),
                        file_path.name,
                        int(frame_index == 1),
                        str(summary.reference_element),
                        f"{active_center[0]:.8f}",
                        f"{active_center[1]:.8f}",
                        f"{active_center[2]:.8f}",
                        f"{center_of_mass[0]:.8f}",
                        f"{center_of_mass[1]:.8f}",
                        f"{center_of_mass[2]:.8f}",
                        f"{geometric_center[0]:.8f}",
                        f"{geometric_center[1]:.8f}",
                        f"{geometric_center[2]:.8f}",
                        f"{reference_center[0]:.8f}",
                        f"{reference_center[1]:.8f}",
                        f"{reference_center[2]:.8f}",
                        f"{(active_center - reference_center)[0]:.8f}",
                        f"{(active_center - reference_center)[1]:.8f}",
                        f"{(active_center - reference_center)[2]:.8f}",
                        f"{(active_center - center_of_mass)[0]:.8f}",
                        f"{(active_center - center_of_mass)[1]:.8f}",
                        f"{(active_center - center_of_mass)[2]:.8f}",
                    ]
                )

    with set_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "set_index",
                "series_label",
                "frame_range_label",
                "frame_count",
                "first_file_name",
                "unique_active_center_count_rounded_1e-6",
                "first_frame_active_center_x_a",
                "first_frame_active_center_y_a",
                "first_frame_active_center_z_a",
                "first_frame_center_of_mass_x_a",
                "first_frame_center_of_mass_y_a",
                "first_frame_center_of_mass_z_a",
                "mean_active_center_x_a",
                "mean_active_center_y_a",
                "mean_active_center_z_a",
                "max_active_center_shift_from_first_a",
                "mean_active_center_shift_from_first_a",
                "mean_active_minus_reference_geom_x_a",
                "mean_active_minus_reference_geom_y_a",
                "mean_active_minus_reference_geom_z_a",
            ]
        )
        for set_index, frame_set in enumerate(frame_sets, start=1):
            summaries = [
                summary_by_path[file_path]
                for file_path in frame_set.file_paths
            ]
            active_centers = np.vstack(
                [
                    np.asarray(summary.active_center, dtype=float)
                    for summary in summaries
                ]
            )
            center_of_mass_centers = np.vstack(
                [
                    np.asarray(summary.center_of_mass, dtype=float)
                    for summary in summaries
                ]
            )
            reference_centers = np.vstack(
                [
                    np.asarray(
                        summary.reference_element_geometric_center,
                        dtype=float,
                    )
                    for summary in summaries
                ]
            )
            first_active_center = np.asarray(active_centers[0], dtype=float)
            shifts_from_first = np.linalg.norm(
                active_centers - first_active_center,
                axis=1,
            )
            rounded_active_centers = {
                tuple(np.round(center, 6)) for center in active_centers
            }
            mean_active_center = np.mean(active_centers, axis=0)
            mean_offset = np.mean(
                active_centers - reference_centers,
                axis=0,
            )
            writer.writerow(
                [
                    set_index,
                    str(frame_set.series_label),
                    str(frame_set.frame_range_label),
                    int(frame_set.frame_count),
                    frame_set.file_paths[0].name,
                    len(rounded_active_centers),
                    f"{first_active_center[0]:.8f}",
                    f"{first_active_center[1]:.8f}",
                    f"{first_active_center[2]:.8f}",
                    f"{center_of_mass_centers[0][0]:.8f}",
                    f"{center_of_mass_centers[0][1]:.8f}",
                    f"{center_of_mass_centers[0][2]:.8f}",
                    f"{mean_active_center[0]:.8f}",
                    f"{mean_active_center[1]:.8f}",
                    f"{mean_active_center[2]:.8f}",
                    f"{float(np.max(shifts_from_first)):.8f}",
                    f"{float(np.mean(shifts_from_first)):.8f}",
                    f"{mean_offset[0]:.8f}",
                    f"{mean_offset[1]:.8f}",
                    f"{mean_offset[2]:.8f}",
                ]
            )
    return frame_csv_path, set_csv_path


def _actual_run_display_label(run_key: str) -> str:
    labels = {
        "geom_mass__contiguous-off__pin-off": ("Geom Mass, contiguous off"),
        "geom_mass__contiguous-on__pin-off": ("Geom Mass, contiguous on"),
        "geom_mass__contiguous-on__pin-on": (
            "Geom Mass, contiguous on, pinned"
        ),
        "reference-element__contiguous-off__pin-off": (
            "Reference Element, contiguous off"
        ),
        "reference-element__contiguous-on__pin-off": (
            "Reference Element, contiguous on"
        ),
    }
    return labels.get(run_key, run_key)


def _actual_run_plot_style(run_key: str) -> dict[str, object]:
    styles: dict[str, dict[str, object]] = {
        "geom_mass__contiguous-off__pin-off": {
            "color": "#1d4ed8",
            "linestyle": "-",
            "linewidth": 2.0,
        },
        "geom_mass__contiguous-on__pin-off": {
            "color": "#0f766e",
            "linestyle": "--",
            "linewidth": 2.0,
        },
        "geom_mass__contiguous-on__pin-on": {
            "color": "#b45309",
            "linestyle": "-.",
            "linewidth": 2.2,
        },
        "reference-element__contiguous-off__pin-off": {
            "color": "#7c3aed",
            "linestyle": "-",
            "linewidth": 2.0,
        },
        "reference-element__contiguous-on__pin-off": {
            "color": "#a855f7",
            "linestyle": ":",
            "linewidth": 2.4,
        },
    }
    return dict(styles.get(run_key, {}))


def _ordered_actual_run_keys() -> list[str]:
    return [
        "geom_mass__contiguous-off__pin-off",
        "geom_mass__contiguous-on__pin-off",
        "geom_mass__contiguous-on__pin-on",
        "reference-element__contiguous-off__pin-off",
        "reference-element__contiguous-on__pin-off",
    ]


def _load_profile_csv(
    path: Path,
) -> dict[str, np.ndarray]:
    radial_centers: list[float] = []
    orientation_density: list[float] = []
    smeared_density: list[float] = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            radial_centers.append(float(row["r_center_a"]))
            orientation_density.append(
                float(row["orientation_average_density_e_per_a3"])
            )
            smeared_density.append(
                float(row["smeared_orientation_average_density_e_per_a3"])
            )
    return {
        "r_center_a": np.asarray(radial_centers, dtype=float),
        "orientation_average_density_e_per_a3": np.asarray(
            orientation_density,
            dtype=float,
        ),
        "smeared_orientation_average_density_e_per_a3": np.asarray(
            smeared_density,
            dtype=float,
        ),
    }


def _load_set_summary_csv(
    path: Path,
) -> dict[str, np.ndarray]:
    set_index: list[int] = []
    frame_count: list[int] = []
    unique_active_center_count: list[int] = []
    max_shift_from_first: list[float] = []
    mean_shift_from_first: list[float] = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            set_index.append(int(row["set_index"]))
            frame_count.append(int(row["frame_count"]))
            unique_active_center_count.append(
                int(row["unique_active_center_count_rounded_1e-6"])
            )
            max_shift_from_first.append(
                float(row["max_active_center_shift_from_first_a"])
            )
            mean_shift_from_first.append(
                float(row["mean_active_center_shift_from_first_a"])
            )
    return {
        "set_index": np.asarray(set_index, dtype=int),
        "frame_count": np.asarray(frame_count, dtype=int),
        "unique_active_center_count": np.asarray(
            unique_active_center_count,
            dtype=int,
        ),
        "max_active_center_shift_from_first_a": np.asarray(
            max_shift_from_first,
            dtype=float,
        ),
        "mean_active_center_shift_from_first_a": np.asarray(
            mean_shift_from_first,
            dtype=float,
        ),
    }


def _load_detected_set_csv(
    path: Path,
) -> dict[str, np.ndarray]:
    set_index: list[int] = []
    frame_count: list[int] = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            set_index.append(int(row["set_index"]))
            frame_count.append(int(row["frame_count"]))
    return {
        "set_index": np.asarray(set_index, dtype=int),
        "frame_count": np.asarray(frame_count, dtype=int),
    }


def _pair_row_from_summary(
    summary_payload: dict[str, object],
    left_key: str,
    right_key: str,
) -> dict[str, object]:
    rows = list(summary_payload["pairwise_profile_differences"])
    for row in rows:
        if (row["run_a"] == left_key and row["run_b"] == right_key) or (
            row["run_a"] == right_key and row["run_b"] == left_key
        ):
            return row
    raise KeyError(
        f"Could not find pair row for {left_key!r} vs {right_key!r}."
    )


def _generate_plot_artifacts(
    output_dir: Path,
    summary_payload: dict[str, object],
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    requested_runs = list(summary_payload["requested_runs"])
    actual_run_entries: dict[str, dict[str, object]] = {}
    for entry in requested_runs:
        actual_run_entries.setdefault(str(entry["actual_run_key"]), entry)

    profile_data = {
        run_key: _load_profile_csv(Path(entry["profile_csv_path"]))
        for run_key, entry in actual_run_entries.items()
    }
    set_summary_data = {
        run_key: _load_set_summary_csv(Path(entry["set_summary_csv_path"]))
        for run_key, entry in actual_run_entries.items()
    }
    detected_set_data = _load_detected_set_csv(
        Path(summary_payload["detected_contiguous_frame_sets"]["csv_path"])
    )
    plot_paths: list[str] = []

    overlay_path = output_dir / "profile_overlay_comparison.png"
    fig, ax = plt.subplots(figsize=(11, 7))
    for run_key in _ordered_actual_run_keys():
        if run_key not in profile_data:
            continue
        series = profile_data[run_key]
        ax.plot(
            series["r_center_a"],
            series["smeared_orientation_average_density_e_per_a3"],
            label=_actual_run_display_label(run_key),
            **_actual_run_plot_style(run_key),
        )
    ax.set_title("Pb2I4 EDM comparison: smeared profile overlays")
    ax.set_xlabel("r (A)")
    ax.set_ylabel("Smeared density (e-/A^3)")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(overlay_path, dpi=180)
    plt.close(fig)
    plot_paths.append(str(overlay_path))

    difference_path = output_dir / "key_profile_differences.png"
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
    difference_pairs = [
        (
            "Geom off -> geom contiguous",
            "geom_mass__contiguous-off__pin-off",
            "geom_mass__contiguous-on__pin-off",
        ),
        (
            "Geom contiguous -> geom pinned",
            "geom_mass__contiguous-on__pin-off",
            "geom_mass__contiguous-on__pin-on",
        ),
        (
            "Ref off -> ref contiguous",
            "reference-element__contiguous-off__pin-off",
            "reference-element__contiguous-on__pin-off",
        ),
        (
            "Geom off -> ref off",
            "geom_mass__contiguous-off__pin-off",
            "reference-element__contiguous-off__pin-off",
        ),
    ]
    for axis, (title, left_key, right_key) in zip(
        axes.flat,
        difference_pairs,
        strict=False,
    ):
        left_series = profile_data[left_key]
        right_series = profile_data[right_key]
        difference = (
            right_series["smeared_orientation_average_density_e_per_a3"]
            - left_series["smeared_orientation_average_density_e_per_a3"]
        )
        pair_row = _pair_row_from_summary(
            summary_payload,
            left_key,
            right_key,
        )
        axis.plot(
            left_series["r_center_a"],
            difference,
            color="#dc2626",
            linewidth=2.0,
        )
        axis.axhline(0.0, color="#475569", linewidth=1.0, alpha=0.8)
        axis.set_title(
            f"{title}\nmax abs diff={float(pair_row['max_abs_diff_e_per_a3']):.4f}"
        )
        axis.set_xlabel("r (A)")
        axis.set_ylabel("Delta smeared density (e-/A^3)")
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(difference_path, dpi=180)
    plt.close(fig)
    plot_paths.append(str(difference_path))

    center_behavior_path = output_dir / "per_set_center_behavior.png"
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    for run_key in [
        "geom_mass__contiguous-off__pin-off",
        "geom_mass__contiguous-on__pin-off",
        "geom_mass__contiguous-on__pin-on",
        "reference-element__contiguous-off__pin-off",
    ]:
        if run_key not in set_summary_data:
            continue
        series = set_summary_data[run_key]
        style = _actual_run_plot_style(run_key)
        axes[0].plot(
            series["set_index"],
            series["max_active_center_shift_from_first_a"],
            label=_actual_run_display_label(run_key),
            **style,
        )
        axes[1].plot(
            series["set_index"],
            series["unique_active_center_count"],
            label=_actual_run_display_label(run_key),
            **style,
        )
    axes[0].set_title("Per-set active-center drift relative to first frame")
    axes[0].set_ylabel("Max shift from first frame (A)")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="best", fontsize=9)
    axes[1].set_title(
        "Unique active centers observed within each contiguous set"
    )
    axes[1].set_xlabel("Contiguous set index")
    axes[1].set_ylabel("Unique active center count")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(center_behavior_path, dpi=180)
    plt.close(fig)
    plot_paths.append(str(center_behavior_path))

    set_size_path = output_dir / "contiguous_set_size_distribution.png"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(
        detected_set_data["set_index"],
        detected_set_data["frame_count"],
        color="#2563eb",
        alpha=0.85,
    )
    axes[0].set_title("Contiguous set sizes by set index")
    axes[0].set_xlabel("Contiguous set index")
    axes[0].set_ylabel("Frames in set")
    axes[0].grid(alpha=0.2, axis="y")
    axes[1].hist(
        detected_set_data["frame_count"],
        bins=min(16, len(detected_set_data["frame_count"])),
        color="#0f766e",
        alpha=0.85,
    )
    axes[1].set_title("Distribution of contiguous set sizes")
    axes[1].set_xlabel("Frames in set")
    axes[1].set_ylabel("Number of sets")
    axes[1].grid(alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(set_size_path, dpi=180)
    plt.close(fig)
    plot_paths.append(str(set_size_path))
    return plot_paths


def _render_report_lines(summary_payload: dict[str, object]) -> list[str]:
    frame_set_summary = dict(summary_payload["detected_contiguous_frame_sets"])
    key_comparisons = [
        (
            "Geometric Mass Center: contiguous off vs contiguous on (standard)",
            _pair_row_from_summary(
                summary_payload,
                "geom_mass__contiguous-off__pin-off",
                "geom_mass__contiguous-on__pin-off",
            ),
        ),
        (
            "Geometric Mass Center: contiguous on standard vs pinned",
            _pair_row_from_summary(
                summary_payload,
                "geom_mass__contiguous-on__pin-off",
                "geom_mass__contiguous-on__pin-on",
            ),
        ),
        (
            "Reference Element: contiguous off vs contiguous on",
            _pair_row_from_summary(
                summary_payload,
                "reference-element__contiguous-off__pin-off",
                "reference-element__contiguous-on__pin-off",
            ),
        ),
        (
            "Geometric Mass Center vs Reference Element, both contiguous off",
            _pair_row_from_summary(
                summary_payload,
                "geom_mass__contiguous-off__pin-off",
                "reference-element__contiguous-off__pin-off",
            ),
        ),
    ]
    mesh_settings = dict(summary_payload["mesh_settings"])
    smearing_settings = dict(summary_payload["smearing_settings"])
    report_lines = [
        "# EDM Contiguous-Mode Comparison",
        "",
        f"- Selection: `{summary_payload['selection_path']}`",
        f"- Output directory: `{summary_payload['output_dir']}`",
        f"- Structures: {int(summary_payload['structure_count'])}",
        (
            "- Mesh: "
            f"rstep={float(mesh_settings['rstep_a']):.3f} A, "
            f"theta={int(mesh_settings['theta_divisions'])}, "
            f"phi={int(mesh_settings['phi_divisions'])}, "
            f"rmax={float(mesh_settings['rmax_a']):.3f} A"
        ),
        (
            "- Smearing sigma: "
            f"{float(smearing_settings['gaussian_sigma_a']):.6f} A"
        ),
        (
            "- Detected contiguous sets: "
            f"{int(frame_set_summary['count'])} sets across "
            f"{int(frame_set_summary['total_frames'])} frames"
        ),
        (
            "- Frames per set: "
            f"min={int(frame_set_summary['min_frame_count'])}, "
            f"max={int(frame_set_summary['max_frame_count'])}, "
            f"mean={float(frame_set_summary['mean_frame_count']):.2f}"
        ),
        "",
        "## Requested Runs",
        "",
        "| Run | Actual run | Contiguous applied | Pin applied | Runtime (s) | Notes |",
        "| --- | --- | --- | --- | ---: | --- |",
    ]
    for entry in list(summary_payload["requested_runs"]):
        report_lines.append(
            "| "
            f"{entry['requested_slug']} | "
            f"{entry['actual_run_key']} | "
            f"{entry['contiguous_frame_mode_applied']} | "
            f"{entry['pinned_geometric_tracking_applied']} | "
            f"{float(entry['elapsed_seconds']):.2f} | "
            f"{'; '.join(entry['averaging_notes']) or 'None'} |"
        )
    report_lines.extend(
        [
            "",
            "## Key Comparisons",
            "",
        ]
    )
    for title, row in key_comparisons:
        report_lines.append(f"- {title}:")
        report_lines.append(
            "  "
            f"max abs profile diff={float(row['max_abs_diff_e_per_a3']):.8f} e-/A^3, "
            f"integrated abs diff={float(row['integrated_abs_diff_e_per_a2']):.8f} e-/A^2, "
            f"max active-center shift={float(row['max_active_center_shift_a']):.8f} A."
        )
    report_lines.append(
        "- Geometric Mass Center with contiguous off ignores the pin request, "
        "so the `pin-on` and `pin-off` requested cases share the same actual run."
    )
    report_lines.extend(
        [
            "",
            "## Output Files",
            "",
            f"- Summary JSON: `{Path(summary_payload['output_dir']) / 'summary.json'}`",
            (
                "- Pairwise CSV: "
                f"`{summary_payload['pairwise_profile_differences_csv_path']}`"
            ),
            f"- Detected contiguous sets CSV: `{frame_set_summary['csv_path']}`",
            "- Each computed run also writes:",
            "  - one profile CSV",
            "  - one frame-center CSV",
            "  - one set-summary CSV",
        ]
    )
    plot_files = list(summary_payload.get("plot_files") or [])
    if plot_files:
        report_lines.extend(
            [
                "",
                "## Plot Files",
                "",
            ]
        )
        for plot_path in plot_files:
            report_lines.append(f"- `{plot_path}`")
    return report_lines


@contextmanager
def _single_worker_mode(enabled: bool):
    if not enabled:
        yield
        return
    original = density_workflow._resolve_electron_density_worker_count
    density_workflow._resolve_electron_density_worker_count = lambda _count: 1
    try:
        yield
    finally:
        density_workflow._resolve_electron_density_worker_count = original


def _default_mesh_settings(
    reference_structure: density_workflow.ElectronDensityStructure,
) -> density_workflow.ElectronDensityMeshSettings:
    return density_workflow.ElectronDensityMeshSettings(
        rstep=0.5,
        theta_divisions=12,
        phi_divisions=6,
        rmax=min(max(float(reference_structure.rmax), 1.0), 10.0),
    )


def _prepare_element_caches(
    reference_structure: density_workflow.ElectronDensityStructure,
) -> list[str]:
    unique_elements = sorted(set(reference_structure.elements))
    for element in unique_elements:
        density_workflow._atomic_mass(element)
        density_workflow._atomic_number(element)
        density_workflow._waasmaier_parameters(element)
        density_workflow._effective_atomic_overlap_radius(element)
    return unique_elements


def _run_comparison(
    *,
    selection_path: Path,
    output_dir: Path,
    force_single_worker: bool,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    inspection = density_workflow.inspect_structure_input(selection_path)
    if inspection.input_mode != "folder":
        raise ValueError("The comparison report expects a folder selection.")
    reference_structure = density_workflow.load_electron_density_structure(
        inspection.reference_file
    )
    mesh_settings = _default_mesh_settings(reference_structure)
    unique_elements = _prepare_element_caches(reference_structure)
    detected_frame_sets, fallback_reason = (
        density_workflow._detect_contiguous_frame_sets(
            tuple(inspection.structure_files)
        )
    )
    if fallback_reason is not None:
        raise ValueError(fallback_reason)

    run_requests = [
        {
            "center_mode": "center_of_mass",
            "use_contiguous_frame_mode": False,
            "pin_contiguous_geometric_tracking": False,
        },
        {
            "center_mode": "center_of_mass",
            "use_contiguous_frame_mode": False,
            "pin_contiguous_geometric_tracking": True,
        },
        {
            "center_mode": "center_of_mass",
            "use_contiguous_frame_mode": True,
            "pin_contiguous_geometric_tracking": False,
        },
        {
            "center_mode": "center_of_mass",
            "use_contiguous_frame_mode": True,
            "pin_contiguous_geometric_tracking": True,
        },
        {
            "center_mode": "reference_element",
            "use_contiguous_frame_mode": False,
            "pin_contiguous_geometric_tracking": False,
        },
        {
            "center_mode": "reference_element",
            "use_contiguous_frame_mode": False,
            "pin_contiguous_geometric_tracking": True,
        },
        {
            "center_mode": "reference_element",
            "use_contiguous_frame_mode": True,
            "pin_contiguous_geometric_tracking": False,
        },
        {
            "center_mode": "reference_element",
            "use_contiguous_frame_mode": True,
            "pin_contiguous_geometric_tracking": True,
        },
    ]

    computed_results: dict[
        str,
        density_workflow.ElectronDensityProfileResult,
    ] = {}
    run_metadata: dict[str, dict[str, object]] = {}

    detected_set_csv_path = output_dir / "detected_contiguous_sets.csv"
    _write_detected_set_csv(detected_set_csv_path, detected_frame_sets)

    for request in run_requests:
        requested_slug = _combo_slug(**request)
        actual_key = _effective_combo_key(**request)
        run_metadata[requested_slug] = {
            "requested_slug": requested_slug,
            "actual_run_key": actual_key,
            "center_mode": str(request["center_mode"]),
            "center_mode_label": _center_mode_label(
                str(request["center_mode"])
            ),
            "use_contiguous_frame_mode": bool(
                request["use_contiguous_frame_mode"]
            ),
            "pin_contiguous_geometric_tracking_requested_in_mesh": bool(
                request["pin_contiguous_geometric_tracking"]
            ),
        }
        if actual_key in computed_results:
            continue
        run_mesh_settings = density_workflow.ElectronDensityMeshSettings(
            rstep=float(mesh_settings.rstep),
            theta_divisions=int(mesh_settings.theta_divisions),
            phi_divisions=int(mesh_settings.phi_divisions),
            rmax=float(mesh_settings.rmax),
            pin_contiguous_geometric_tracking=bool(
                request["pin_contiguous_geometric_tracking"]
            ),
        )
        print(
            "Running",
            actual_key,
            f"(center={request['center_mode']},",
            f"contiguous={request['use_contiguous_frame_mode']},",
            f"pin={request['pin_contiguous_geometric_tracking']})",
            flush=True,
        )
        start_time = time.perf_counter()
        with _single_worker_mode(force_single_worker):
            result = (
                density_workflow.compute_electron_density_profile_for_input(
                    inspection,
                    run_mesh_settings,
                    reference_structure=reference_structure,
                    center_mode=str(request["center_mode"]),
                    use_contiguous_frame_mode=bool(
                        request["use_contiguous_frame_mode"]
                    ),
                )
            )
        elapsed_seconds = float(time.perf_counter() - start_time)
        computed_results[actual_key] = result
        profile_csv_path = output_dir / f"{actual_key}__profile.csv"
        _write_profile_csv(profile_csv_path, result)
        frame_csv_path, set_csv_path = _write_run_center_csvs(
            output_dir,
            actual_key,
            result,
            detected_frame_sets,
        )
        run_metadata[requested_slug].update(
            {
                "computed": True,
                "elapsed_seconds": elapsed_seconds,
                "profile_csv_path": str(profile_csv_path),
                "frame_center_csv_path": str(frame_csv_path),
                "set_summary_csv_path": str(set_csv_path),
            }
        )
        print(
            "Completed",
            actual_key,
            f"in {elapsed_seconds:.2f}s",
            flush=True,
        )

    for request in run_requests:
        requested_slug = _combo_slug(**request)
        actual_key = _effective_combo_key(**request)
        result = computed_results[actual_key]
        metadata = run_metadata[requested_slug]
        if "elapsed_seconds" not in metadata:
            actual_metadata = next(
                value
                for value in run_metadata.values()
                if value["actual_run_key"] == actual_key
                and "elapsed_seconds" in value
            )
            metadata.update(
                {
                    "computed": False,
                    "aliased_to": actual_key,
                    "elapsed_seconds": float(
                        actual_metadata["elapsed_seconds"]
                    ),
                    "profile_csv_path": str(
                        actual_metadata["profile_csv_path"]
                    ),
                    "frame_center_csv_path": str(
                        actual_metadata["frame_center_csv_path"]
                    ),
                    "set_summary_csv_path": str(
                        actual_metadata["set_summary_csv_path"]
                    ),
                }
            )
        metadata.update(
            {
                "source_structure_count": int(result.source_structure_count),
                "averaging_mode": str(result.averaging_mode),
                "contiguous_frame_mode_requested": bool(
                    result.contiguous_frame_mode_requested
                ),
                "contiguous_frame_mode_applied": bool(
                    result.contiguous_frame_mode_applied
                ),
                "pinned_geometric_tracking_requested": bool(
                    result.pinned_geometric_tracking_requested
                ),
                "pinned_geometric_tracking_applied": bool(
                    result.pinned_geometric_tracking_applied
                ),
                "averaging_notes": [
                    str(note) for note in result.averaging_notes
                ],
                "smearing_sigma_a": float(
                    result.smearing_settings.gaussian_sigma_a
                ),
                "radial_shell_count": int(result.mesh_geometry.shell_count),
            }
        )

    pairwise_rows: list[dict[str, object]] = []
    unique_keys = sorted(computed_results)
    for left_index, left_key in enumerate(unique_keys):
        for right_key in unique_keys[left_index + 1 :]:
            left_result = computed_results[left_key]
            right_result = computed_results[right_key]
            row: dict[str, object] = {
                "run_a": left_key,
                "run_b": right_key,
            }
            row.update(_profile_metrics(left_result, right_result))
            row.update(_active_center_metrics(left_result, right_result))
            pairwise_rows.append(row)

    pairwise_csv_path = output_dir / "pairwise_profile_differences.csv"
    with pairwise_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "run_a",
                "run_b",
                "profiles_identical",
                "max_abs_diff_e_per_a3",
                "mean_abs_diff_e_per_a3",
                "rms_diff_e_per_a3",
                "integrated_abs_diff_e_per_a2",
                "active_centers_identical",
                "max_active_center_shift_a",
                "mean_active_center_shift_a",
            ],
        )
        writer.writeheader()
        for row in pairwise_rows:
            writer.writerow(row)

    frame_counts = [
        int(frame_set.frame_count) for frame_set in detected_frame_sets
    ]
    series_counts: dict[str, int] = {}
    for frame_set in detected_frame_sets:
        series_counts[str(frame_set.series_label)] = (
            int(series_counts.get(str(frame_set.series_label), 0)) + 1
        )

    summary_payload: dict[str, object] = {
        "selection_path": str(selection_path),
        "inspection_reference_file": str(inspection.reference_file),
        "output_dir": str(output_dir),
        "force_single_worker": bool(force_single_worker),
        "unique_elements": unique_elements,
        "mesh_settings": mesh_settings.to_dict(),
        "smearing_settings": density_workflow.ElectronDensitySmearingSettings()
        .normalized()
        .to_dict(),
        "structure_count": int(len(inspection.structure_files)),
        "detected_contiguous_frame_sets": {
            "count": int(len(detected_frame_sets)),
            "total_frames": int(sum(frame_counts)),
            "min_frame_count": int(min(frame_counts)),
            "max_frame_count": int(max(frame_counts)),
            "mean_frame_count": float(np.mean(frame_counts)),
            "series_counts": series_counts,
            "csv_path": str(detected_set_csv_path),
        },
        "requested_runs": [
            run_metadata[_combo_slug(**request)] for request in run_requests
        ],
        "pairwise_profile_differences_csv_path": str(pairwise_csv_path),
        "pairwise_profile_differences": pairwise_rows,
    }

    summary_payload["plot_files"] = _generate_plot_artifacts(
        output_dir,
        summary_payload,
    )
    summary_json_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"
    summary_payload["report_path"] = str(report_path)
    summary_json_path.write_text(
        json.dumps(summary_payload, indent=2),
        encoding="utf-8",
    )
    report_path.write_text(
        "\n".join(_render_report_lines(summary_payload)) + "\n",
        encoding="utf-8",
    )
    return summary_payload


def _generate_plots_only(
    output_dir: Path,
) -> dict[str, object]:
    summary_json_path = output_dir / "summary.json"
    if not summary_json_path.is_file():
        raise FileNotFoundError(
            f"Could not find existing summary JSON at {summary_json_path}."
        )
    summary_payload = json.loads(summary_json_path.read_text(encoding="utf-8"))
    summary_payload["plot_files"] = _generate_plot_artifacts(
        output_dir,
        summary_payload,
    )
    report_path = output_dir / "report.md"
    summary_payload["report_path"] = str(report_path)
    summary_json_path.write_text(
        json.dumps(summary_payload, indent=2),
        encoding="utf-8",
    )
    report_path.write_text(
        "\n".join(_render_report_lines(summary_payload)) + "\n",
        encoding="utf-8",
    )
    return summary_payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a backend EDM comparison across contiguous/pinned center-mode "
            "settings and write report artifacts under tests/."
        )
    )
    parser.add_argument(
        "--selection",
        type=Path,
        default=DEFAULT_SELECTION_PATH,
        help="Folder of contiguous PDB frames to analyze.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where report artifacts will be written.",
    )
    parser.add_argument(
        "--use-default-workers",
        action="store_true",
        help="Use the workflow's default worker-count resolution.",
    )
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Generate PNG plots from an existing summary/report bundle.",
    )
    args = parser.parse_args()
    resolved_output_dir = args.output_dir.expanduser().resolve()
    if args.plots_only:
        summary = _generate_plots_only(resolved_output_dir)
    else:
        summary = _run_comparison(
            selection_path=args.selection.expanduser().resolve(),
            output_dir=resolved_output_dir,
            force_single_worker=not bool(args.use_default_workers),
        )
    print(
        json.dumps(
            {
                "summary_json": str(
                    Path(summary["output_dir"]) / "summary.json"
                ),
                "report_path": str(summary["report_path"]),
                "pairwise_csv": str(
                    summary["pairwise_profile_differences_csv_path"]
                ),
                "plot_files": list(summary.get("plot_files") or []),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
