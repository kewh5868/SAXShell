import numpy as np

from saxshell.saxs.born_refinement.backend import build_shared_q_grid
from saxshell.saxs.contrast_fft import (
    ContrastFFTSettings,
    compute_contrast_fft_intensity,
)


def test_shared_q_grid_preserves_requested_upper_bound_for_partial_step():
    q_values = build_shared_q_grid(0.0101, 1.1976, q_step=0.01)

    np.testing.assert_allclose(q_values[0], 0.0101)
    np.testing.assert_allclose(q_values[-2], 1.1901)
    np.testing.assert_allclose(q_values[-1], 1.1976)


def test_single_atom_bare_density_uses_direct_born_trace_when_fft_bins_are_empty():
    coordinates = np.asarray([[0.0, 0.0, 0.0]], dtype=float)
    weights = np.asarray([6.0], dtype=float)
    q_values = np.asarray([0.01, 0.02, 0.03], dtype=float)
    settings = ContrastFFTSettings(
        spacing_a=2.5,
        gaussian_sigma_a=0.75,
        minimum_box_length_a=80.0,
        padding_a=4.0,
    ).normalized()

    result = compute_contrast_fft_intensity(
        coordinates,
        weights,
        q_values,
        settings,
        elements=("C",),
    )

    np.testing.assert_allclose(result.raw_intensity, [36.0, 36.0, 36.0])
    np.testing.assert_allclose(
        result.kernel_corrected_intensity,
        result.raw_intensity,
    )
    assert np.all(result.q_shell_counts == 1)
    assert result.contrast_mode == "single_atom_bare_density_direct_born"
    assert result.first_nonempty_q_a_inverse == q_values[0]
