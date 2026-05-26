from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from saxshell.saxs.aps_detector_stitch import (
    APS_DETECTOR_ORDER,
    detector_name_from_path,
    find_aps_detector_files,
    save_aps_stitched_data,
    stitch_aps_detector_files,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "aps_detector_stitch"


def _true_intensity(q: np.ndarray) -> np.ndarray:
    return 2.5 / (q + 0.08) ** 1.35 + 0.03


def _write_trace(path: Path, q: np.ndarray, raw_scale: float) -> None:
    intensity = _true_intensity(q) * raw_scale
    error = intensity * 0.025
    np.savetxt(path, np.column_stack((q, intensity, error)))


def test_detector_name_from_path_accepts_aps_detector_suffixes():
    assert detector_name_from_path("sample_hs102_389.txt") == "hs102"
    assert detector_name_from_path("sample-HS103-389.dat") == "hs103"
    assert detector_name_from_path("sample.hs104.txt") == "hs104"
    assert detector_name_from_path("sample_hs105_389.txt") is None


def test_stitch_aps_detector_files_corrects_known_offsets(tmp_path):
    q_104 = np.geomspace(0.01, 0.145, 90)
    q_103 = np.geomspace(0.132, 0.74, 120)
    q_102 = np.geomspace(0.69, 4.45, 150)
    _write_trace(tmp_path / "synthetic_hs104_001.txt", q_104, 1.0)
    _write_trace(tmp_path / "synthetic_hs103_001.txt", q_103, 0.5)
    _write_trace(tmp_path / "synthetic_hs102_001.txt", q_102, 2.0)

    result = stitch_aps_detector_files(tmp_path)

    assert list(result.detector_paths) == list(APS_DETECTOR_ORDER)
    assert np.all(np.diff(result.stitched_data[:, 0]) > 0.0)
    assert result.warnings == ()
    assert [join.method for join in result.joins] == [
        "overlap-median",
        "overlap-median",
    ]
    assert result.joins[0].scale_factor == pytest.approx(2.0, rel=1e-3)
    assert result.joins[1].scale_factor == pytest.approx(0.5, rel=1e-3)
    sample_q = np.geomspace(0.012, 4.2, 40)
    stitched_i = np.interp(
        sample_q,
        result.stitched_data[:, 0],
        result.stitched_data[:, 1],
    )
    assert stitched_i == pytest.approx(_true_intensity(sample_q), rel=0.015)


def test_stitch_aps_detector_files_loads_real_aps_fixture_and_saves(tmp_path):
    matches = find_aps_detector_files(FIXTURE_DIR)
    assert set(matches) == set(APS_DETECTOR_ORDER)

    result = stitch_aps_detector_files(FIXTURE_DIR)

    assert result.stitched_data.shape[1] == 3
    assert len(result.stitched_data) > 1000
    assert np.all(np.diff(result.stitched_data[:, 0]) > 0.0)
    assert result.stitched_data[0, 0] == pytest.approx(0.0025405)
    assert result.stitched_data[-1, 0] == pytest.approx(4.4578)
    assert result.warnings == ()
    assert all(join.scale_factor > 0.0 for join in result.joins)

    destination = save_aps_stitched_data(result, tmp_path / "stitched.txt")
    reloaded = np.loadtxt(destination)
    assert reloaded.shape == result.stitched_data.shape
    assert reloaded[:, 0] == pytest.approx(result.stitched_data[:, 0])
