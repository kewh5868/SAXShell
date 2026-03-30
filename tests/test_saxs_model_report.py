from __future__ import annotations

import numpy as np
import pytest
from matplotlib.colors import to_hex
from matplotlib.figure import Figure

from saxshell.saxs.model_report import (
    ReportComponentPlotData,
    ReportComponentSeries,
    _aligned_y_limits,
    _autoscale_component_plot_to_model_range,
    _draw_model_fit_axis,
    _draw_prefit_plot_axes,
)
from saxshell.saxs.prefit import PrefitEvaluation
from saxshell.saxs.project_manager import PowerPointExportSettings


def test_component_plot_autoscale_to_model_range_matches_project_setup_behavior():
    figure = Figure()
    experimental_axis = figure.add_subplot(111)
    component_axis = experimental_axis.twinx()

    experimental_q = np.asarray([0.05, 0.10, 0.15, 0.20, 0.25], dtype=float)
    experimental_i = np.asarray([200.0, 100.0, 80.0, 60.0, 40.0], dtype=float)
    model_q = np.asarray([0.10, 0.15, 0.20], dtype=float)
    model_i = np.asarray([2.0, 4.0, 8.0], dtype=float)

    experimental_axis.plot(
        experimental_q,
        experimental_i,
        color="black",
        alpha=0.35,
        linewidth=1.3,
        label="Experimental data",
    )
    experimental_axis.plot(
        model_q,
        experimental_i[1:4],
        color="black",
        linewidth=1.8,
        label="Selected q-range",
    )
    component_axis.plot(
        model_q,
        model_i,
        color="tab:blue",
        linewidth=1.4,
        label="component",
    )

    plot_data = ReportComponentPlotData(
        title="Initial SAXS traces without solvent",
        selected_q_min=0.10,
        selected_q_max=0.20,
        use_experimental_grid=False,
        log_x=False,
        log_y=False,
        experimental_q_values=experimental_q,
        experimental_intensities=experimental_i,
        solvent_q_values=None,
        solvent_intensities=None,
        component_series=(
            ReportComponentSeries(
                label="component",
                q_values=model_q,
                intensities=model_i,
                color="#1f77b4",
            ),
        ),
    )

    _autoscale_component_plot_to_model_range(
        experimental_axis,
        component_axis,
        plot_data,
    )

    assert experimental_axis.get_xlim() == pytest.approx((0.10, 0.20))
    assert component_axis.get_xlim() == pytest.approx((0.10, 0.20))
    assert experimental_axis.get_ylim() == pytest.approx((58.0, 102.0))

    expected_component_limits = _aligned_y_limits(
        experimental_axis.get_ylim(),
        60.0,
        100.0,
        2.0,
        8.0,
        log_scale=False,
    )
    assert component_axis.get_ylim() == pytest.approx(
        expected_component_limits
    )


def test_prefit_plot_axes_match_prefit_window_colors_and_rescale_without_solvent():
    figure_without = Figure()
    grid_without = figure_without.add_gridspec(2, 1, height_ratios=[3, 1])
    top_without = figure_without.add_subplot(grid_without[0, 0])
    bottom_without = figure_without.add_subplot(
        grid_without[1, 0],
        sharex=top_without,
    )

    figure_with = Figure()
    grid_with = figure_with.add_gridspec(2, 1, height_ratios=[3, 1])
    top_with = figure_with.add_subplot(grid_with[0, 0])
    bottom_with = figure_with.add_subplot(
        grid_with[1, 0],
        sharex=top_with,
    )

    q_values = np.asarray([0.05, 0.08, 0.12, 0.18, 0.24], dtype=float)
    experimental = np.asarray([120.0, 100.0, 90.0, 70.0, 55.0], dtype=float)
    model = np.asarray([110.0, 98.0, 88.0, 68.0, 52.0], dtype=float)
    solvent = np.asarray([8.0, 12.0, 18.0, 380.0, 520.0], dtype=float)
    evaluation = PrefitEvaluation(
        q_values=q_values,
        experimental_intensities=experimental,
        model_intensities=model,
        residuals=model - experimental,
        solvent_contribution=solvent,
    )
    settings = PowerPointExportSettings()

    _draw_prefit_plot_axes(
        top_without,
        bottom_without,
        evaluation,
        include_solvent=False,
        settings=settings,
    )
    _draw_prefit_plot_axes(
        top_with,
        bottom_with,
        evaluation,
        include_solvent=True,
        settings=settings,
    )

    without_lines = {
        str(line.get_label()): to_hex(line.get_color(), keep_alpha=False)
        for line in top_without.get_lines()
    }
    with_lines = {
        str(line.get_label()): to_hex(line.get_color(), keep_alpha=False)
        for line in top_with.get_lines()
    }

    assert without_lines == {
        "Experimental": "#000000",
        "Model": "#d62728",
    }
    assert with_lines == {
        "Experimental": "#000000",
        "Solvent contribution": "#008000",
        "Model": "#d62728",
    }
    assert (
        to_hex(bottom_without.get_lines()[-1].get_color(), keep_alpha=False)
        == "#1f77b4"
    )
    assert (
        to_hex(bottom_with.get_lines()[-1].get_color(), keep_alpha=False)
        == "#1f77b4"
    )
    assert top_without.get_ylim()[1] < top_with.get_ylim()[1]


def test_filter_fit_axis_matches_dream_output_colors_and_hides_solvent():
    figure = Figure()
    axis = figure.add_subplot(111)
    settings = PowerPointExportSettings()
    q_values = np.asarray([0.05, 0.08, 0.12, 0.18], dtype=float)
    experimental = np.asarray([120.0, 100.0, 88.0, 70.0], dtype=float)
    model = np.asarray([118.0, 97.0, 84.0, 68.0], dtype=float)
    structure_factor = np.asarray([1.0, 0.96, 0.91, 0.88], dtype=float)

    _draw_model_fit_axis(
        axis,
        q_values=q_values,
        experimental=experimental,
        model=model,
        solvent=None,
        structure_factor=structure_factor,
        title="All Post-burnin",
        metrics_lines=["RMSE: 0.1"],
        show_legend=False,
        compact=True,
        dream_output_style=True,
        settings=settings,
    )

    assert len(figure.axes) == 2
    left_axis, right_axis = figure.axes
    left_lines = {
        str(line.get_label()): to_hex(line.get_color(), keep_alpha=False)
        for line in left_axis.get_lines()
    }
    left_collections = [
        to_hex(collection.get_facecolor()[0], keep_alpha=False)
        for collection in left_axis.collections
        if collection.get_offsets().size > 0
    ]

    assert left_collections == ["#000000"]
    assert left_lines == {"Model": "#d62728"}
    assert "Solvent contribution" not in left_lines
    assert right_axis.get_ylabel() == "S(q)"
    assert (
        to_hex(right_axis.get_lines()[0].get_color(), keep_alpha=False)
        == "#9467bd"
    )
