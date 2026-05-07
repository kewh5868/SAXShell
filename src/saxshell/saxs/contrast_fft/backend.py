from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from saxshell.xyz2pdb.workflow import _covalent_radius


@dataclass(slots=True, frozen=True)
class ContrastFFTSettings:
    spacing_a: float
    gaussian_sigma_a: float
    minimum_box_length_a: float
    padding_a: float = 0.0
    support_sigma: float = 4.0
    solvent_density_e_per_a3: float = 0.0
    exclusion_radius_scale: float = 1.0
    exclusion_radius_padding_a: float = 0.0
    use_cubic_box: bool = True

    def normalized(self) -> "ContrastFFTSettings":
        spacing = float(self.spacing_a)
        sigma = float(self.gaussian_sigma_a)
        minimum_box_length = float(self.minimum_box_length_a)
        padding = float(self.padding_a)
        support_sigma = float(self.support_sigma)
        solvent_density = float(self.solvent_density_e_per_a3)
        exclusion_radius_scale = float(self.exclusion_radius_scale)
        exclusion_radius_padding = float(self.exclusion_radius_padding_a)
        if spacing <= 0.0:
            raise ValueError("spacing_a must be greater than zero.")
        if sigma < 0.0:
            raise ValueError("gaussian_sigma_a must be non-negative.")
        if minimum_box_length < spacing:
            raise ValueError(
                "minimum_box_length_a must be at least one voxel wide."
            )
        if padding < 0.0:
            raise ValueError("padding_a must be non-negative.")
        if support_sigma <= 0.0:
            raise ValueError("support_sigma must be greater than zero.")
        if exclusion_radius_scale <= 0.0:
            raise ValueError(
                "exclusion_radius_scale must be greater than zero."
            )
        if exclusion_radius_padding < 0.0:
            raise ValueError(
                "exclusion_radius_padding_a must be non-negative."
            )
        return ContrastFFTSettings(
            spacing_a=spacing,
            gaussian_sigma_a=sigma,
            minimum_box_length_a=minimum_box_length,
            padding_a=padding,
            support_sigma=support_sigma,
            solvent_density_e_per_a3=solvent_density,
            exclusion_radius_scale=exclusion_radius_scale,
            exclusion_radius_padding_a=exclusion_radius_padding,
            use_cubic_box=bool(self.use_cubic_box),
        )


def default_contrast_fft_settings(
    *,
    solvent_density_e_per_a3: float = 0.0,
    exclusion_radius_scale: float = 1.0,
    exclusion_radius_padding_a: float = 0.0,
) -> ContrastFFTSettings:
    return ContrastFFTSettings(
        spacing_a=2.5,
        gaussian_sigma_a=0.75,
        minimum_box_length_a=640.0,
        padding_a=24.0,
        solvent_density_e_per_a3=float(solvent_density_e_per_a3),
        exclusion_radius_scale=float(exclusion_radius_scale),
        exclusion_radius_padding_a=float(exclusion_radius_padding_a),
    ).normalized()


@dataclass(slots=True, frozen=True)
class ContrastFFTGrid:
    density: np.ndarray
    grid_shape: tuple[int, int, int]
    origin_a: tuple[float, float, float]
    box_lengths_a: tuple[float, float, float]
    voxel_spacing_a: tuple[float, float, float]
    density_integral: float
    expected_weight: float


@dataclass(slots=True, frozen=True)
class ContrastFFTTiming:
    atomic_density_seconds: float
    contrast_density_seconds: float
    fft_seconds: float
    shell_average_seconds: float
    total_seconds: float


@dataclass(slots=True, frozen=True)
class ContrastFFTResult:
    settings: ContrastFFTSettings
    q_values: np.ndarray
    raw_intensity: np.ndarray
    kernel_corrected_intensity: np.ndarray
    q_shell_counts: np.ndarray
    density_integral: float
    expected_weight: float
    contrast_density_integral: float
    expected_contrast_weight: float
    solvent_exclusion_volume_a3: float
    grid_shape: tuple[int, int, int]
    box_lengths_a: tuple[float, float, float]
    voxel_spacing_a: tuple[float, float, float]
    q_nyquist_a_inverse: float
    q_frequency_step_a_inverse: tuple[float, float, float]
    q_convention: str
    uses_two_pi_frequency_conversion: bool
    density_subtraction_active: bool
    first_nonempty_q_a_inverse: float | None
    solvent_density_e_per_a3: float
    contrast_mode: str
    kernel_correction_supported: bool
    kernel_correction_applied: bool
    kernel_correction_model: str | None
    timing: ContrastFFTTiming


def _raise_if_cancelled(
    cancelled: Callable[[], bool] | None,
) -> None:
    if cancelled is not None and bool(cancelled()):
        raise RuntimeError("3D FFT Born calculation cancelled.")


def _box_axis_counts(
    coordinates: np.ndarray,
    settings: ContrastFFTSettings,
) -> tuple[int, int, int]:
    spacing = float(settings.spacing_a)
    span = np.ptp(np.asarray(coordinates, dtype=float), axis=0)
    requested_lengths = np.asarray(span + 2.0 * float(settings.padding_a))
    minimum_length = max(float(settings.minimum_box_length_a), spacing)
    if bool(settings.use_cubic_box):
        axis_counts = []
        longest = max(float(np.max(requested_lengths)), minimum_length)
        count = int(np.ceil(longest / spacing))
        if count % 2 == 0:
            count += 1
        axis_counts = [count, count, count]
    else:
        axis_counts = []
        for requested_length in requested_lengths:
            count = int(
                np.ceil(max(float(requested_length), minimum_length) / spacing)
            )
            if count % 2 == 0:
                count += 1
            axis_counts.append(count)
    return tuple(int(value) for value in axis_counts)


def _grid_origin_from_shape(
    grid_shape: tuple[int, int, int],
    spacing_a: float,
) -> np.ndarray:
    counts = np.asarray(grid_shape, dtype=float)
    return -0.5 * (counts - 1.0) * float(spacing_a)


def build_atomic_density_grid(
    coordinates: np.ndarray,
    weights: np.ndarray,
    settings: ContrastFFTSettings,
    *,
    cancelled: Callable[[], bool] | None = None,
) -> ContrastFFTGrid:
    normalized = settings.normalized()
    coords = np.asarray(coordinates, dtype=float)
    atom_weights = np.asarray(weights, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coordinates must be an Nx3 array.")
    if atom_weights.ndim != 1 or atom_weights.shape[0] != coords.shape[0]:
        raise ValueError(
            "weights must be a one-dimensional array matching coordinates."
        )
    if coords.shape[0] == 0:
        raise ValueError("At least one atom is required.")

    spacing = float(normalized.spacing_a)
    sigma = float(normalized.gaussian_sigma_a)
    voxel_volume = spacing**3
    grid_shape = _box_axis_counts(coords, normalized)
    origin = _grid_origin_from_shape(grid_shape, spacing)
    density = np.zeros(grid_shape, dtype=float)

    if sigma <= 0.0:
        grid_indices = np.rint(
            (coords - origin[np.newaxis, :]) / spacing
        ).astype(int)
        for grid_index, atom_weight in zip(
            grid_indices,
            atom_weights,
            strict=False,
        ):
            _raise_if_cancelled(cancelled)
            ix, iy, iz = (
                int(grid_index[0]),
                int(grid_index[1]),
                int(grid_index[2]),
            )
            if (
                ix < 0
                or iy < 0
                or iz < 0
                or ix >= grid_shape[0]
                or iy >= grid_shape[1]
                or iz >= grid_shape[2]
            ):
                raise ValueError("An atom fell outside the FFT grid bounds.")
            density[ix, iy, iz] += float(atom_weight) / voxel_volume
        return ContrastFFTGrid(
            density=density,
            grid_shape=grid_shape,
            origin_a=tuple(float(value) for value in origin),
            box_lengths_a=tuple(
                float(count * spacing) for count in grid_shape
            ),
            voxel_spacing_a=(spacing, spacing, spacing),
            density_integral=float(
                np.sum(density, dtype=float) * voxel_volume
            ),
            expected_weight=float(np.sum(atom_weights, dtype=float)),
        )

    support_radius = int(
        np.ceil(float(normalized.support_sigma) * sigma / spacing)
    )
    for point, atom_weight in zip(coords, atom_weights, strict=False):
        _raise_if_cancelled(cancelled)
        center_index = np.rint((point - origin) / spacing).astype(int)
        axis_ranges: list[np.ndarray] = []
        axis_kernels: list[np.ndarray] = []
        axis_sums: list[float] = []
        for axis in range(3):
            start = max(int(center_index[axis]) - support_radius, 0)
            stop = min(
                int(center_index[axis]) + support_radius + 1,
                grid_shape[axis],
            )
            index_range = np.arange(start, stop, dtype=int)
            axis_positions = origin[axis] + index_range.astype(float) * spacing
            kernel = np.exp(
                -0.5 * np.square((axis_positions - float(point[axis])) / sigma)
            )
            axis_ranges.append(index_range)
            axis_kernels.append(np.asarray(kernel, dtype=float))
            axis_sums.append(float(np.sum(kernel, dtype=float)))
        normalization = (
            axis_sums[0] * axis_sums[1] * axis_sums[2] * voxel_volume
        )
        if normalization <= 0.0:
            raise ValueError(
                "Encountered a non-positive Gaussian normalization."
            )
        contribution = (
            (float(atom_weight) / normalization)
            * axis_kernels[0][:, np.newaxis, np.newaxis]
            * axis_kernels[1][np.newaxis, :, np.newaxis]
            * axis_kernels[2][np.newaxis, np.newaxis, :]
        )
        density[
            np.ix_(axis_ranges[0], axis_ranges[1], axis_ranges[2])
        ] += contribution

    return ContrastFFTGrid(
        density=density,
        grid_shape=grid_shape,
        origin_a=tuple(float(value) for value in origin),
        box_lengths_a=tuple(float(count * spacing) for count in grid_shape),
        voxel_spacing_a=(spacing, spacing, spacing),
        density_integral=float(np.sum(density, dtype=float) * voxel_volume),
        expected_weight=float(np.sum(atom_weights, dtype=float)),
    )


def build_exclusion_mask_grid(
    coordinates: np.ndarray,
    elements: list[str] | tuple[str, ...],
    settings: ContrastFFTSettings,
    *,
    origin_a: tuple[float, float, float],
    grid_shape: tuple[int, int, int],
    cancelled: Callable[[], bool] | None = None,
) -> np.ndarray:
    normalized = settings.normalized()
    coords = np.asarray(coordinates, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coordinates must be an Nx3 array.")
    if len(elements) != coords.shape[0]:
        raise ValueError("elements must match the number of coordinates.")
    spacing = float(normalized.spacing_a)
    origin = np.asarray(origin_a, dtype=float)
    mask = np.zeros(grid_shape, dtype=bool)
    for point, element in zip(coords, elements, strict=False):
        _raise_if_cancelled(cancelled)
        radius = float(normalized.exclusion_radius_scale) * float(
            _covalent_radius(str(element))
        ) + float(normalized.exclusion_radius_padding_a)
        radius = max(radius, 0.0)
        if radius <= 0.0:
            continue
        support_radius = int(np.ceil(radius / spacing))
        center_index = np.rint((point - origin) / spacing).astype(int)
        axis_ranges: list[np.ndarray] = []
        axis_positions: list[np.ndarray] = []
        for axis in range(3):
            start = max(int(center_index[axis]) - support_radius, 0)
            stop = min(
                int(center_index[axis]) + support_radius + 1,
                grid_shape[axis],
            )
            index_range = np.arange(start, stop, dtype=int)
            positions = origin[axis] + index_range.astype(float) * spacing
            axis_ranges.append(index_range)
            axis_positions.append(np.asarray(positions, dtype=float))
        dx = axis_positions[0][:, np.newaxis, np.newaxis] - float(point[0])
        dy = axis_positions[1][np.newaxis, :, np.newaxis] - float(point[1])
        dz = axis_positions[2][np.newaxis, np.newaxis, :] - float(point[2])
        local_mask = np.square(dx) + np.square(dy) + np.square(dz) <= radius**2
        mask[
            np.ix_(axis_ranges[0], axis_ranges[1], axis_ranges[2])
        ] |= local_mask
    return np.asarray(mask, dtype=float)


def build_contrast_density_grid(
    atomic_grid: ContrastFFTGrid,
    settings: ContrastFFTSettings,
    *,
    coordinates: np.ndarray,
    elements: list[str] | tuple[str, ...] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> tuple[np.ndarray, float, float]:
    normalized = settings.normalized()
    atomic_density = np.asarray(atomic_grid.density, dtype=float)
    solvent_density = float(normalized.solvent_density_e_per_a3)
    if abs(solvent_density) <= 1.0e-15:
        return (
            np.asarray(atomic_density, dtype=float),
            0.0,
            float(atomic_grid.expected_weight),
        )
    if elements is None:
        raise ValueError(
            "elements are required when solvent_density_e_per_a3 is non-zero."
        )
    exclusion_mask = build_exclusion_mask_grid(
        coordinates,
        elements,
        normalized,
        origin_a=atomic_grid.origin_a,
        grid_shape=atomic_grid.grid_shape,
        cancelled=cancelled,
    )
    voxel_volume = float(normalized.spacing_a) ** 3
    solvent_exclusion_volume = float(
        np.sum(exclusion_mask, dtype=float) * voxel_volume
    )
    contrast_density = atomic_density - solvent_density * exclusion_mask
    expected_contrast_weight = float(
        atomic_grid.expected_weight
        - solvent_density * solvent_exclusion_volume
    )
    return (
        np.asarray(contrast_density, dtype=float),
        solvent_exclusion_volume,
        expected_contrast_weight,
    )


def _q_bin_edges(q_values: np.ndarray) -> np.ndarray:
    q_grid = np.asarray(q_values, dtype=float)
    if q_grid.ndim != 1:
        raise ValueError("q_values must be one-dimensional.")
    if q_grid.size == 0:
        raise ValueError("q_values must not be empty.")
    if q_grid.size == 1:
        q_step = max(float(q_grid[0]) * 0.5, 1.0e-6)
    else:
        q_step = float(np.median(np.diff(q_grid)))
    return np.concatenate(
        (
            [max(0.0, float(q_grid[0]) - 0.5 * q_step)],
            0.5 * (q_grid[:-1] + q_grid[1:]),
            [float(q_grid[-1]) + 0.5 * q_step],
        )
    )


def compute_contrast_fft_intensity(
    coordinates: np.ndarray,
    weights: np.ndarray,
    q_values: np.ndarray,
    settings: ContrastFFTSettings,
    *,
    elements: list[str] | tuple[str, ...] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> ContrastFFTResult:
    total_start = perf_counter()
    q_grid = np.asarray(q_values, dtype=float)
    if q_grid.ndim != 1:
        raise ValueError("q_values must be one-dimensional.")
    if q_grid.size == 0:
        raise ValueError("q_values must not be empty.")
    normalized = settings.normalized()
    atomic_start = perf_counter()
    atomic_grid = build_atomic_density_grid(
        coordinates,
        weights,
        normalized,
        cancelled=cancelled,
    )
    atomic_seconds = perf_counter() - atomic_start
    _raise_if_cancelled(cancelled)
    if (
        np.asarray(coordinates, dtype=float).shape[0] == 1
        and abs(float(normalized.solvent_density_e_per_a3)) <= 1.0e-15
    ):
        spacing = float(normalized.spacing_a)
        intensity = np.full(
            q_grid.shape,
            float(atomic_grid.expected_weight) ** 2,
            dtype=float,
        )
        total_seconds = perf_counter() - total_start
        return ContrastFFTResult(
            settings=normalized,
            q_values=np.asarray(q_grid, dtype=float),
            raw_intensity=intensity,
            kernel_corrected_intensity=np.asarray(intensity, dtype=float),
            q_shell_counts=np.ones_like(q_grid, dtype=int),
            density_integral=float(atomic_grid.density_integral),
            expected_weight=float(atomic_grid.expected_weight),
            contrast_density_integral=float(atomic_grid.density_integral),
            expected_contrast_weight=float(atomic_grid.expected_weight),
            solvent_exclusion_volume_a3=0.0,
            grid_shape=tuple(int(value) for value in atomic_grid.grid_shape),
            box_lengths_a=tuple(
                float(value) for value in atomic_grid.box_lengths_a
            ),
            voxel_spacing_a=tuple(
                float(value) for value in atomic_grid.voxel_spacing_a
            ),
            q_nyquist_a_inverse=float(np.pi / spacing),
            q_frequency_step_a_inverse=tuple(
                float(2.0 * np.pi / max(length, spacing))
                for length in atomic_grid.box_lengths_a
            ),
            q_convention=(
                "Direct single-atom Born evaluation for a bare-density "
                "single atom; the Cartesian FFT grid is still built for "
                "geometry diagnostics."
            ),
            uses_two_pi_frequency_conversion=True,
            density_subtraction_active=False,
            first_nonempty_q_a_inverse=float(q_grid[0]),
            solvent_density_e_per_a3=0.0,
            contrast_mode="single_atom_bare_density_direct_born",
            kernel_correction_supported=False,
            kernel_correction_applied=False,
            kernel_correction_model=None,
            timing=ContrastFFTTiming(
                atomic_density_seconds=float(atomic_seconds),
                contrast_density_seconds=0.0,
                fft_seconds=0.0,
                shell_average_seconds=0.0,
                total_seconds=float(total_seconds),
            ),
        )
    contrast_start = perf_counter()
    contrast_density, solvent_exclusion_volume, expected_contrast_weight = (
        build_contrast_density_grid(
            atomic_grid,
            normalized,
            coordinates=np.asarray(coordinates, dtype=float),
            elements=elements,
            cancelled=cancelled,
        )
    )
    contrast_seconds = perf_counter() - contrast_start
    _raise_if_cancelled(cancelled)
    spacing = float(normalized.spacing_a)
    voxel_volume = spacing**3
    fft_start = perf_counter()
    amplitude = np.fft.rfftn(contrast_density) * voxel_volume
    fft_seconds = perf_counter() - fft_start
    _raise_if_cancelled(cancelled)
    qx_axis = (
        2.0 * np.pi * np.fft.fftfreq(atomic_grid.grid_shape[0], d=spacing)
    )
    qy_axis = (
        2.0 * np.pi * np.fft.fftfreq(atomic_grid.grid_shape[1], d=spacing)
    )
    qz_axis = (
        2.0 * np.pi * np.fft.rfftfreq(atomic_grid.grid_shape[2], d=spacing)
    )
    q_edges = _q_bin_edges(q_grid)

    shell_sums = np.zeros_like(q_grid, dtype=float)
    corrected_shell_sums = np.zeros_like(q_grid, dtype=float)
    shell_counts = np.zeros_like(q_grid, dtype=int)
    qy_squared = np.square(qy_axis)[:, np.newaxis]
    qz_squared = np.square(qz_axis)[np.newaxis, :]
    z_plane_weights = np.ones_like(qz_axis, dtype=int)
    if qz_axis.size > 1:
        if atomic_grid.grid_shape[2] % 2 == 0:
            z_plane_weights[1:-1] = 2
        else:
            z_plane_weights[1:] = 2
    z_plane_weights_2d = z_plane_weights[np.newaxis, :]
    sigma = float(normalized.gaussian_sigma_a)
    raw_kernel_correction_valid = bool(
        sigma > 0.0
        and abs(float(normalized.solvent_density_e_per_a3)) <= 1.0e-15
    )
    shell_average_start = perf_counter()

    for x_index, qx_value in enumerate(qx_axis):
        _raise_if_cancelled(cancelled)
        q_magnitude = np.sqrt(float(qx_value) ** 2 + qy_squared + qz_squared)
        intensity_slice = np.square(np.abs(amplitude[x_index]))
        repeated_counts = np.broadcast_to(
            z_plane_weights_2d, intensity_slice.shape
        )
        bin_indices = np.digitize(q_magnitude.reshape(-1), q_edges) - 1
        valid_mask = (bin_indices >= 0) & (bin_indices < q_grid.size)
        flat_counts = repeated_counts.reshape(-1)
        flat_intensity = intensity_slice.reshape(-1)
        shell_counts += np.bincount(
            bin_indices[valid_mask],
            weights=flat_counts[valid_mask],
            minlength=q_grid.size,
        ).astype(int)
        shell_sums += np.bincount(
            bin_indices[valid_mask],
            weights=flat_intensity[valid_mask] * flat_counts[valid_mask],
            minlength=q_grid.size,
        )
        if raw_kernel_correction_valid:
            intensity_response = np.exp(
                -np.square(sigma) * np.square(q_magnitude)
            )
            corrected_slice = intensity_slice / np.maximum(
                intensity_response, 1.0e-12
            )
            corrected_shell_sums += np.bincount(
                bin_indices[valid_mask],
                weights=corrected_slice.reshape(-1)[valid_mask]
                * flat_counts[valid_mask],
                minlength=q_grid.size,
            )

    raw_shell_average = np.divide(
        shell_sums,
        shell_counts,
        out=np.full_like(q_grid, np.nan, dtype=float),
        where=shell_counts > 0,
    )
    if raw_kernel_correction_valid:
        corrected_shell_average = np.divide(
            corrected_shell_sums,
            shell_counts,
            out=np.full_like(q_grid, np.nan, dtype=float),
            where=shell_counts > 0,
        )
    else:
        corrected_shell_average = np.asarray(raw_shell_average, dtype=float)

    nonempty = np.flatnonzero(shell_counts > 0)
    first_nonempty_q = (
        None
        if nonempty.size == 0
        else float(np.asarray(q_grid)[int(nonempty[0])])
    )
    shell_average_seconds = perf_counter() - shell_average_start
    contrast_integral = float(
        np.sum(contrast_density, dtype=float) * voxel_volume
    )
    total_seconds = perf_counter() - total_start
    density_subtraction_active = bool(
        abs(float(normalized.solvent_density_e_per_a3)) > 1.0e-15
    )
    return ContrastFFTResult(
        settings=normalized,
        q_values=np.asarray(q_grid, dtype=float),
        raw_intensity=np.asarray(raw_shell_average, dtype=float),
        kernel_corrected_intensity=np.asarray(
            corrected_shell_average,
            dtype=float,
        ),
        q_shell_counts=np.asarray(shell_counts, dtype=int),
        density_integral=float(atomic_grid.density_integral),
        expected_weight=float(atomic_grid.expected_weight),
        contrast_density_integral=contrast_integral,
        expected_contrast_weight=float(expected_contrast_weight),
        solvent_exclusion_volume_a3=float(solvent_exclusion_volume),
        grid_shape=tuple(int(value) for value in atomic_grid.grid_shape),
        box_lengths_a=tuple(
            float(value) for value in atomic_grid.box_lengths_a
        ),
        voxel_spacing_a=tuple(
            float(value) for value in atomic_grid.voxel_spacing_a
        ),
        q_nyquist_a_inverse=float(np.pi / spacing),
        q_frequency_step_a_inverse=tuple(
            float(2.0 * np.pi / max(length, spacing))
            for length in atomic_grid.box_lengths_a
        ),
        q_convention=(
            "3D FFT of a Cartesian contrast-density grid with q = 2πf, "
            "followed by radial q-shell averaging of |A(qx, qy, qz)|^2."
        ),
        uses_two_pi_frequency_conversion=True,
        density_subtraction_active=density_subtraction_active,
        first_nonempty_q_a_inverse=first_nonempty_q,
        solvent_density_e_per_a3=float(normalized.solvent_density_e_per_a3),
        contrast_mode=(
            "constant_solvent_density_inside_union_of_atomic_spheres"
            if density_subtraction_active
            else "bare_atomic_density_only"
        ),
        kernel_correction_supported=raw_kernel_correction_valid,
        kernel_correction_applied=raw_kernel_correction_valid,
        kernel_correction_model=(
            "Gaussian deposition intensity factor exp(-sigma^2 q^2)"
            if raw_kernel_correction_valid
            else None
        ),
        timing=ContrastFFTTiming(
            atomic_density_seconds=float(atomic_seconds),
            contrast_density_seconds=float(contrast_seconds),
            fft_seconds=float(fft_seconds),
            shell_average_seconds=float(shell_average_seconds),
            total_seconds=float(total_seconds),
        ),
    )
