from __future__ import annotations

import json
import os

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from saxshell.uvvis_fitting.model import (
    DEFAULT_MONTE_CARLO_SEED,
    FitResult,
    MonteCarloFitRecord,
    MonteCarloResult,
    MonteCarloSettings,
    PeakComponent,
    UVVisDataset,
    evaluate_components,
    fit_components,
    load_session_payload,
    load_uvvis_file,
    run_monte_carlo_fit,
    run_parameter_sweep,
    save_fit_bundle,
    session_payload,
)
from saxshell.uvvis_fitting.ui.main_window import (
    PARAMETER_SECTION_BOUNDARY_COLUMNS,
    TABLE_COLUMNS,
    MonteCarloResultsDialog,
    UVVisFitMainWindow,
)


@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _synthetic_dataset() -> UVVisDataset:
    x = np.linspace(200.0, 400.0, 151)
    truth = [
        PeakComponent("A", amplitude=8.0, center=260.0, fwhm=24.0, eta=0.2),
        PeakComponent("B", amplitude=5.0, center=310.0, fwhm=24.0, eta=0.2),
    ]
    _curves, y = evaluate_components(x, truth)
    return UVVisDataset(source_path="", x=x, y=y)


def test_load_uvvis_file_with_headers(tmp_path):
    path = tmp_path / "spectrum.txt"
    path.write_text(
        "wavelength_nm\tapparent_pb_molar_absorptivity\n"
        "300\t1.0\n"
        "301\t1.5\n",
        encoding="utf-8",
    )

    dataset = load_uvvis_file(path)

    assert dataset.x_label == "Wavelength (nm)"
    assert dataset.y_label == "Apparent Pb molar absorptivity"
    np.testing.assert_allclose(dataset.x, [300.0, 301.0])
    np.testing.assert_allclose(dataset.y, [1.0, 1.5])


def test_monte_carlo_settings_default_seed_is_fixed():
    settings = MonteCarloSettings()

    assert DEFAULT_MONTE_CARLO_SEED == 42
    assert settings.seed == 42


def test_peak_table_headers_group_each_constraint_parameter():
    constraint_headers = [
        title
        for title, _parameter, kind in TABLE_COLUMNS
        if kind == "constraint"
    ]

    assert constraint_headers == [
        "Amp Constraint",
        "Area Constraint",
        "Center Constraint",
        "FWHM Constraint",
        "η Constraint",
    ]
    assert PARAMETER_SECTION_BOUNDARY_COLUMNS == {0, 3, 6, 9, 12, 15}


def test_uvvis_plot_refresh_preserves_zoom_for_same_dataset(qapp):
    window = UVVisFitMainWindow()
    try:
        dataset = _synthetic_dataset()
        window.dataset = dataset
        window.components = [
            PeakComponent("A", amplitude=6.0, center=260.0, fwhm=22.0, eta=0.1)
        ]
        window._set_fit_window(dataset.x_min, dataset.x_max)
        window._refresh_table()
        window._refresh_plot()
        window.ax.set_xlim(240.0, 285.0)
        window.ax.set_ylim(-0.25, 8.5)

        window.components[0].center = 262.0
        window._refresh_plot()

        assert window.ax.get_xlim() == pytest.approx((240.0, 285.0))
        assert window.ax.get_ylim() == pytest.approx((-0.25, 8.5))
    finally:
        window.close()


def test_uvvis_toolbar_home_resets_preserved_zoom(qapp):
    window = UVVisFitMainWindow()
    try:
        dataset = _synthetic_dataset()
        window.dataset = dataset
        window.components = [
            PeakComponent("A", amplitude=6.0, center=260.0, fwhm=22.0, eta=0.1)
        ]
        window._set_fit_window(dataset.x_min, dataset.x_max)
        window._refresh_table()
        window._refresh_plot()
        home_xlim = window.ax.get_xlim()
        home_ylim = window.ax.get_ylim()
        window.ax.set_xlim(240.0, 285.0)
        window.ax.set_ylim(-0.25, 8.5)

        window.components[0].center = 262.0
        window._refresh_plot()
        window.toolbar.home()

        assert window.ax.get_xlim() == pytest.approx(home_xlim)
        assert window.ax.get_ylim() == pytest.approx(home_ylim)
    finally:
        window.close()


def test_monte_carlo_results_dialog_initial_selection_has_axes(qapp):
    dataset = _synthetic_dataset()
    components = [
        PeakComponent("A", amplitude=6.0, center=260.0, fwhm=22.0, eta=0.1)
    ]
    component_curves, total = evaluate_components(dataset.x, components)
    residual = total - dataset.y
    fit_result = FitResult(
        components=components,
        x_fit=dataset.x,
        y_data=dataset.y,
        total=total,
        component_curves=component_curves,
        residual=residual,
        chisq=float(np.sum(residual**2)),
        redchi=float(np.mean(residual**2)),
        success=True,
        message="test",
        nfev=0,
    )
    record = MonteCarloFitRecord(
        index=0,
        initial_components=[component.copy() for component in components],
        result=fit_result,
        retained=True,
    )
    result = MonteCarloResult(
        settings=MonteCarloSettings(iterations=1),
        records=[record],
        retained_records=[record],
        summaries=[],
    )

    dialog = MonteCarloResultsDialog(result=result)
    try:
        assert hasattr(dialog, "model_ax")
        assert dialog.selected_record is record
    finally:
        dialog.close()


def test_fit_components_honors_relational_constraints():
    dataset = _synthetic_dataset()
    components = [
        PeakComponent("A", amplitude=6.0, center=255.0, fwhm=20.0, eta=0.25),
        PeakComponent(
            "B",
            amplitude=4.0,
            center=305.0,
            fwhm=21.0,
            eta=0.25,
            constraints={"center": "@A + 50", "fwhm": "@A"},
        ),
    ]

    result = fit_components(dataset, components)

    center_offset = result.components[1].center - result.components[0].center
    assert abs(center_offset - 50.0) < 1e-8
    assert abs(result.components[1].fwhm - result.components[0].fwhm) < 1e-8
    assert result.chisq < 1e-5


def test_fit_components_forward_reference_to_later_peak():
    """Earlier peak referencing a later one must resolve (lmfit-name-
    order bug)."""
    dataset = _synthetic_dataset()
    components = [
        PeakComponent(
            "A",
            amplitude=7.0,
            center=255.0,
            fwhm=21.0,
            eta=0.25,
            constraints={
                "fwhm": "@B",
                "eta": "@B",
                "center": "@B - 50",
            },
        ),
        PeakComponent("B", amplitude=4.0, center=310.0, fwhm=24.0, eta=0.2),
    ]

    result = fit_components(dataset, components)

    assert abs(result.components[0].fwhm - result.components[1].fwhm) < 1e-8
    assert abs(result.components[0].eta - result.components[1].eta) < 1e-8
    assert (
        abs((result.components[1].center - result.components[0].center) - 50.0)
        < 1e-8
    )


def test_fit_components_handles_area_interval_penalty_shape_change():
    x = np.linspace(200.0, 400.0, 254)
    truth = PeakComponent(
        "A",
        amplitude=8.0,
        center=280.0,
        fwhm=18.0,
        eta=0.2,
    )
    _curves, y = evaluate_components(x, [truth])
    dataset = UVVisDataset(source_path="", x=x, y=y)
    target_area = truth.area
    components = [
        PeakComponent(
            "A",
            amplitude=2.0,
            center=272.0,
            fwhm=35.0,
            eta=0.35,
            constraints={
                "area": f"[{target_area * 0.85}, {target_area * 1.15}]"
            },
        )
    ]

    result = fit_components(dataset, components, max_nfev=1000)

    assert result.success is True
    assert result.residual.shape == (254,)
    assert result.chisq < 1.0


def test_session_and_sweep_outputs_roundtrip(tmp_path):
    dataset = _synthetic_dataset()
    components = [
        PeakComponent("A", amplitude=6.0, center=260.0, fwhm=22.0, eta=0.1),
        PeakComponent("B", amplitude=4.0, center=310.0, fwhm=22.0, eta=0.1),
    ]
    result = fit_components(dataset, components)

    payload = session_payload(dataset, result.components, result=result)
    loaded_dataset, loaded_components, _fit_min, _fit_max = (
        load_session_payload(json.loads(json.dumps(payload)))
    )

    assert loaded_dataset.x.size == dataset.x.size
    assert [component.label for component in loaded_components] == ["A", "B"]

    export_dir = tmp_path / "fit_export"
    save_fit_bundle(export_dir, dataset, result.components, result=result)
    assert (export_dir / "fit.json").is_file()
    assert (export_dir / "peaks.csv").is_file()
    assert (export_dir / "curve.csv").is_file()

    records = run_parameter_sweep(
        dataset,
        result.components,
        peak_label="A",
        parameter="eta",
        lower=0.0,
        upper=0.2,
        steps=2,
        output_root=tmp_path / "sweep",
    )
    assert len(records) == 2
    assert all(record.output_dir.is_dir() for record in records)


def test_monte_carlo_fit_reports_constrained_error_bars(tmp_path):
    dataset = _synthetic_dataset()
    components = [
        PeakComponent(
            "A",
            amplitude=6.0,
            center=258.0,
            fwhm=22.0,
            eta=0.15,
            constraints={"center": "[255, 265]", "fwhm": "[18, 30]"},
        ),
        PeakComponent(
            "B",
            amplitude=4.0,
            center=308.0,
            fwhm=22.0,
            eta=0.15,
            constraints={"center": "@A + 50", "fwhm": "@A", "eta": "[0, 0.5]"},
        ),
    ]
    settings = MonteCarloSettings(
        iterations=6,
        keep_fraction=0.5,
        seed=123,
        noise_sigma=0.01,
        amplitude_jitter=0.1,
        center_jitter=2.0,
        fwhm_jitter=0.1,
        eta_jitter=0.05,
        max_nfev=1000,
    )
    progress_calls: list[tuple[int, int, str]] = []

    def on_progress(processed: int, total: int, message: str) -> None:
        progress_calls.append((processed, total, message))

    result = run_monte_carlo_fit(
        dataset,
        components,
        settings=settings,
        output_root=tmp_path / "monte_carlo",
        progress_callback=on_progress,
    )

    assert result.completed == 6
    assert progress_calls[0] == (
        0,
        6,
        "Preparing Monte Carlo reference fit...",
    )
    assert progress_calls[-1] == (
        6,
        6,
        "Processed Monte Carlo fit 6 of 6.",
    )
    assert len(result.retained_records) == 3
    assert result.best_record.retained is True
    assert (tmp_path / "monte_carlo" / "monte_carlo_summary.json").is_file()
    assert (tmp_path / "monte_carlo" / "monte_carlo_population.csv").is_file()
    assert (tmp_path / "monte_carlo" / "monte_carlo_error_bars.csv").is_file()
    report_path = tmp_path / "monte_carlo" / "monte_carlo_fit_report.md"
    trace_path = (
        tmp_path / "monte_carlo" / "monte_carlo_traces_with_errors.csv"
    )
    assert report_path.is_file()
    assert trace_path.is_file()
    assert (tmp_path / "monte_carlo" / "best_fit" / "fit.json").is_file()
    assert "Monte Carlo UV-Vis Fit Report" in report_path.read_text(
        encoding="utf-8"
    )
    trace_text = trace_path.read_text(encoding="utf-8")
    assert (
        "x,experimental,in_fit_window,best_total,best_residual" in trace_text
    )
    assert "peak_A_best" in trace_text
    assert "peak_A_lower" in trace_text
    assert "total_lower" in trace_text

    for record in result.records:
        peak_a, peak_b = record.result.components
        assert peak_a.center == pytest.approx(260.0, abs=5.0)
        assert peak_b.center - peak_a.center == pytest.approx(50.0, abs=1e-8)
        assert peak_b.fwhm == pytest.approx(peak_a.fwhm, abs=1e-8)
        assert 0.0 <= peak_b.eta <= 0.5

    center_summary = next(
        summary
        for summary in result.summaries
        if summary.peak_label == "A" and summary.parameter == "center"
    )
    assert center_summary.count == 3
    assert (
        center_summary.lower <= center_summary.median <= center_summary.upper
    )
