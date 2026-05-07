from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import cdist, pdist


def build_shared_q_grid(
    q_min: float,
    q_max: float,
    *,
    q_step: float = 0.01,
) -> np.ndarray:
    if float(q_step) <= 0.0:
        raise ValueError("q_step must be greater than zero.")
    if float(q_max) < float(q_min):
        raise ValueError("q_max must be greater than or equal to q_min.")
    step = float(q_step)
    count = int(np.floor((float(q_max) - float(q_min)) / step + 0.5)) + 1
    q_values = float(q_min) + step * np.arange(count, dtype=float)
    upper = float(q_max)
    q_values = q_values[q_values <= upper + 1.0e-12]
    if q_values.size == 0:
        raise ValueError("The requested q-grid did not contain any samples.")
    endpoint_tolerance = max(
        1.0e-12,
        1.0e-9 * max(abs(float(q_min)), abs(upper), 1.0),
    )
    if abs(float(q_values[-1]) - upper) <= endpoint_tolerance:
        q_values[-1] = upper
    elif float(q_values[-1]) < upper:
        q_values = np.append(q_values, upper)
    return np.asarray(q_values, dtype=float)


def fibonacci_sphere_directions(direction_count: int) -> np.ndarray:
    count = int(direction_count)
    if count < 1:
        raise ValueError("direction_count must be at least 1.")
    indices = np.arange(count, dtype=float)
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    y_values = 1.0 - (2.0 * indices + 1.0) / float(count)
    radial_values = np.sqrt(np.clip(1.0 - y_values * y_values, 0.0, None))
    theta_values = golden_angle * indices
    x_values = np.cos(theta_values) * radial_values
    z_values = np.sin(theta_values) * radial_values
    return np.asarray(
        np.column_stack((x_values, y_values, z_values)),
        dtype=float,
    )


def compute_constant_weight_debye_intensity(
    coordinates: np.ndarray,
    weights: np.ndarray,
    q_values: np.ndarray,
    *,
    atom_block_size: int = 128,
    q_chunk_size: int = 24,
) -> np.ndarray:
    q_grid = np.asarray(q_values, dtype=float)
    coords = np.asarray(coordinates, dtype=float)
    atom_weights = np.asarray(weights, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coordinates must be an Nx3 array.")
    if atom_weights.ndim != 1 or atom_weights.shape[0] != coords.shape[0]:
        raise ValueError(
            "weights must be a one-dimensional array matching coordinates."
        )
    if q_grid.ndim != 1:
        raise ValueError("q_values must be one-dimensional.")
    if coords.size == 0:
        return np.zeros_like(q_grid, dtype=float)

    intensity = np.sum(np.square(atom_weights), dtype=float) * np.ones_like(
        q_grid,
        dtype=float,
    )
    block = max(int(atom_block_size), 1)
    q_block = max(int(q_chunk_size), 1)
    atom_count = int(coords.shape[0])

    def accumulate_pair_terms(
        pair_distances: np.ndarray,
        pair_weights: np.ndarray,
    ) -> None:
        if pair_distances.size == 0 or pair_weights.size == 0:
            return
        for q_start in range(0, q_grid.size, q_block):
            q_stop = min(q_start + q_block, q_grid.size)
            q_chunk = q_grid[q_start:q_stop]
            kernel = np.sinc(
                pair_distances[:, np.newaxis] * q_chunk[np.newaxis, :] / np.pi
            )
            intensity[q_start:q_stop] += 2.0 * np.sum(
                pair_weights[:, np.newaxis] * kernel,
                axis=0,
                dtype=float,
            )

    for i_start in range(0, atom_count, block):
        i_stop = min(i_start + block, atom_count)
        coords_i = coords[i_start:i_stop]
        weights_i = atom_weights[i_start:i_stop]
        local_count = int(i_stop - i_start)
        if local_count > 1:
            intra_distances = pdist(coords_i, metric="euclidean")
            intra_weights = np.concatenate(
                [
                    weights_i[index] * weights_i[index + 1 :]
                    for index in range(local_count - 1)
                ]
            )
            accumulate_pair_terms(intra_distances, intra_weights)
        for j_start in range(i_stop, atom_count, block):
            j_stop = min(j_start + block, atom_count)
            coords_j = coords[j_start:j_stop]
            weights_j = atom_weights[j_start:j_stop]
            inter_distances = cdist(coords_i, coords_j).reshape(-1)
            inter_weights = (
                weights_i[:, np.newaxis] * weights_j[np.newaxis, :]
            ).reshape(-1)
            accumulate_pair_terms(inter_distances, inter_weights)
    return np.asarray(intensity, dtype=float)


def compute_spherical_average_point_born_intensity(
    coordinates: np.ndarray,
    weights: np.ndarray,
    q_values: np.ndarray,
) -> np.ndarray:
    coords = np.asarray(coordinates, dtype=float)
    atom_weights = np.asarray(weights, dtype=float)
    q_grid = np.asarray(q_values, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coordinates must be an Nx3 array.")
    if atom_weights.ndim != 1 or atom_weights.shape[0] != coords.shape[0]:
        raise ValueError(
            "weights must be a one-dimensional array matching coordinates."
        )
    radii = np.linalg.norm(coords, axis=1)
    amplitude = np.sum(
        atom_weights[np.newaxis, :]
        * np.sinc(q_grid[:, np.newaxis] * radii[np.newaxis, :] / np.pi),
        axis=1,
        dtype=float,
    )
    return np.square(np.abs(amplitude))


def compute_directional_born_intensity(
    coordinates: np.ndarray,
    weights: np.ndarray,
    q_values: np.ndarray,
    *,
    direction_count: int = 256,
    q_chunk_size: int = 12,
) -> np.ndarray:
    coords = np.asarray(coordinates, dtype=float)
    atom_weights = np.asarray(weights, dtype=float)
    q_grid = np.asarray(q_values, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coordinates must be an Nx3 array.")
    if atom_weights.ndim != 1 or atom_weights.shape[0] != coords.shape[0]:
        raise ValueError(
            "weights must be a one-dimensional array matching coordinates."
        )
    directions = fibonacci_sphere_directions(direction_count)
    projections = np.asarray(coords @ directions.T, dtype=float)
    intensity = np.zeros_like(q_grid, dtype=float)
    q_block = max(int(q_chunk_size), 1)
    for q_start in range(0, q_grid.size, q_block):
        q_stop = min(q_start + q_block, q_grid.size)
        q_chunk = q_grid[q_start:q_stop]
        phase = (
            q_chunk[:, np.newaxis, np.newaxis] * projections[np.newaxis, :, :]
        )
        amplitude = np.sum(
            atom_weights[np.newaxis, :, np.newaxis] * np.exp(1j * phase),
            axis=1,
        )
        intensity[q_start:q_stop] = np.mean(
            np.square(np.abs(amplitude)),
            axis=1,
            dtype=float,
        )
    return np.asarray(intensity, dtype=float)


@dataclass(slots=True, frozen=True)
class GridBornSettings:
    spacing_a: float
    padding_a: float
    sigma_a: float
    support_sigma: float = 4.0

    def normalized(self) -> "GridBornSettings":
        spacing = float(self.spacing_a)
        padding = float(self.padding_a)
        sigma = float(self.sigma_a)
        support_sigma = float(self.support_sigma)
        if spacing <= 0.0:
            raise ValueError("spacing_a must be greater than zero.")
        if padding < 0.0:
            raise ValueError("padding_a must be non-negative.")
        if sigma < 0.0:
            raise ValueError("sigma_a must be non-negative.")
        if support_sigma <= 0.0:
            raise ValueError("support_sigma must be greater than zero.")
        return GridBornSettings(
            spacing_a=spacing,
            padding_a=padding,
            sigma_a=sigma,
            support_sigma=support_sigma,
        )


@dataclass(slots=True, frozen=True)
class GridBornResult:
    settings: GridBornSettings
    q_values: np.ndarray
    intensity: np.ndarray
    q_shell_counts: np.ndarray
    density_integral: float
    expected_weight: float
    grid_shape: tuple[int, int, int]
    box_lengths_a: tuple[float, float, float]
    voxel_spacing_a: tuple[float, float, float]
    q_nyquist_a_inverse: float
    q_frequency_step_a_inverse: tuple[float, float, float]
    q_convention: str
    uses_two_pi_frequency_conversion: bool
    density_subtraction_active: bool


def _deposit_density_to_grid(
    coordinates: np.ndarray,
    weights: np.ndarray,
    settings: GridBornSettings,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normalized = settings.normalized()
    coords = np.asarray(coordinates, dtype=float)
    atom_weights = np.asarray(weights, dtype=float)
    spacing = float(normalized.spacing_a)
    padding = float(normalized.padding_a)
    sigma = float(normalized.sigma_a)
    d_volume = spacing**3

    coord_min = np.min(coords, axis=0) - padding
    coord_max = np.max(coords, axis=0) + padding
    span = np.asarray(coord_max - coord_min, dtype=float)
    grid_shape = tuple(
        int(np.ceil(float(axis_span) / spacing)) + 1 for axis_span in span
    )
    density = np.zeros(grid_shape, dtype=float)

    if sigma <= 0.0:
        grid_indices = np.rint((coords - coord_min) / spacing).astype(int)
        for grid_index, atom_weight in zip(
            grid_indices,
            atom_weights,
            strict=False,
        ):
            ix, iy, iz = (
                int(grid_index[0]),
                int(grid_index[1]),
                int(grid_index[2]),
            )
            density[ix, iy, iz] += float(atom_weight) / d_volume
        return density, coord_min, span

    support_radius = int(
        np.ceil(float(normalized.support_sigma) * sigma / spacing)
    )
    for point, atom_weight in zip(coords, atom_weights, strict=False):
        center_index = np.rint((point - coord_min) / spacing).astype(int)
        axis_ranges: list[np.ndarray] = []
        axis_kernels: list[np.ndarray] = []
        axis_sums: list[float] = []
        for axis in range(3):
            start = max(int(center_index[axis]) - support_radius, 0)
            stop = min(
                int(center_index[axis]) + support_radius + 1, grid_shape[axis]
            )
            index_range = np.arange(start, stop, dtype=int)
            axis_positions = (
                coord_min[axis] + index_range.astype(float) * spacing
            )
            kernel = np.exp(
                -0.5 * np.square((axis_positions - float(point[axis])) / sigma)
            )
            axis_ranges.append(index_range)
            axis_kernels.append(np.asarray(kernel, dtype=float))
            axis_sums.append(float(np.sum(kernel)))
        normalization = axis_sums[0] * axis_sums[1] * axis_sums[2] * d_volume
        contribution = (
            (float(atom_weight) / normalization)
            * axis_kernels[0][:, np.newaxis, np.newaxis]
            * axis_kernels[1][np.newaxis, :, np.newaxis]
            * axis_kernels[2][np.newaxis, np.newaxis, :]
        )
        density[
            np.ix_(axis_ranges[0], axis_ranges[1], axis_ranges[2])
        ] += contribution
    return density, coord_min, span


def compute_fft_grid_born_intensity(
    coordinates: np.ndarray,
    weights: np.ndarray,
    q_values: np.ndarray,
    settings: GridBornSettings,
) -> GridBornResult:
    q_grid = np.asarray(q_values, dtype=float)
    coords = np.asarray(coordinates, dtype=float)
    atom_weights = np.asarray(weights, dtype=float)
    normalized = settings.normalized()
    density, _origin, requested_span = _deposit_density_to_grid(
        coords,
        atom_weights,
        normalized,
    )
    spacing = float(normalized.spacing_a)
    d_volume = spacing**3
    amplitude = np.fft.fftn(density) * d_volume
    intensity_grid = np.square(np.abs(amplitude))
    q_axes = [
        2.0 * np.pi * np.fft.fftfreq(axis_count, d=spacing)
        for axis_count in density.shape
    ]
    qx, qy, qz = np.meshgrid(*q_axes, indexing="ij")
    q_magnitude = np.sqrt(qx * qx + qy * qy + qz * qz).reshape(-1)
    flat_intensity = intensity_grid.reshape(-1)

    if q_grid.size > 1:
        q_step = float(np.median(np.diff(q_grid)))
    else:
        q_step = max(float(q_grid[0]) * 0.5, 1.0e-6)
    q_edges = np.concatenate(
        (
            [max(0.0, float(q_grid[0]) - 0.5 * q_step)],
            0.5 * (q_grid[:-1] + q_grid[1:]),
            [float(q_grid[-1]) + 0.5 * q_step],
        )
    )
    bin_indices = np.digitize(q_magnitude, q_edges) - 1
    valid_mask = (bin_indices >= 0) & (bin_indices < q_grid.size)
    q_shell_counts = np.bincount(
        bin_indices[valid_mask],
        minlength=q_grid.size,
    )
    shell_sums = np.bincount(
        bin_indices[valid_mask],
        weights=flat_intensity[valid_mask],
        minlength=q_grid.size,
    )
    shell_average = np.divide(
        shell_sums,
        q_shell_counts,
        out=np.full(q_grid.shape, np.nan, dtype=float),
        where=q_shell_counts > 0,
    )
    box_lengths = tuple(
        float(axis_count * spacing) for axis_count in density.shape
    )
    q_frequency_steps = tuple(
        0.0 if len(axis_values) < 2 else float(axis_values[1] - axis_values[0])
        for axis_values in q_axes
    )
    return GridBornResult(
        settings=normalized,
        q_values=np.asarray(q_grid, dtype=float),
        intensity=np.asarray(shell_average, dtype=float),
        q_shell_counts=np.asarray(q_shell_counts, dtype=int),
        density_integral=float(np.sum(density, dtype=float) * d_volume),
        expected_weight=float(np.sum(atom_weights, dtype=float)),
        grid_shape=tuple(int(axis_count) for axis_count in density.shape),
        box_lengths_a=box_lengths,
        voxel_spacing_a=(spacing, spacing, spacing),
        q_nyquist_a_inverse=float(np.pi / spacing),
        q_frequency_step_a_inverse=q_frequency_steps,
        q_convention=(
            "3D FFT shell average with q = 2πf, where f is the Cartesian "
            "FFT frequency in cycles per Å."
        ),
        uses_two_pi_frequency_conversion=True,
        density_subtraction_active=False,
    )
