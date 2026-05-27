#!/usr/bin/env python
##############################################################################
#
# (c) 2026 The University of Colorado Boulder in the City of Boulder.
# All rights reserved.
#
# File coded by: Keith White.
#
# See GitHub contributions for a more detailed list of contributors.
# https://github.com/kewh5868/saxshell/graphs/contributors
#
# See LICENSE.rst for license information.
#
##############################################################################
"""Python package for analysis of small-angle scattering data from
molecular dynamics derived liquid structures."""

from saxshell.plotting.axis_scale import (  # noqa
    apply_axis_scales,
    axis_has_positive_values,
    resolve_axis_scale,
    safe_set_axis_scale,
)
from saxshell.plotting.igor_inline import (  # noqa
    IgorInlineSegment,
    apply_igor_inline_text_artist,
    has_igor_inline_markup,
    igor_inline_to_mathtext,
    prepare_igor_inline_segments,
)
from saxshell.plotting.labels import Q_A_INVERSE_LABEL  # noqa
from saxshell.plotting.line_plot_editor import (  # noqa
    LINE_PLOT_LEGEND_LOCATIONS,
    LinePlotDefaults,
    LinePlotEditorControls,
    LinePlotSeriesDefaults,
    LinePlotSettings,
)
from saxshell.plotting.plot_editor import (  # noqa
    HeatmapPlotDefaults,
    HeatmapPlotEditorControls,
    HeatmapPlotSettings,
    PlotEditorWindow,
    StackedHistogramPlotDefaults,
    StackedHistogramPlotEditorControls,
    StackedHistogramPlotSettings,
    load_pickled_plot_figure,
    load_pickled_plot_payload,
    save_pickled_plot_figure,
)
from saxshell.plotting.stacked_histogram import (  # noqa
    STACKED_HISTOGRAM_LEGEND_LOCATIONS,
    render_stacked_histogram_export_payload,
)

# package version
from saxshell.version import __version__  # noqa

# silence the pyflakes syntax checker
assert __version__ or True

# End of file
