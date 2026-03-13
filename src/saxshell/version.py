#!/usr/bin/env python
##############################################################################
#
# (c) 2026 The University of Colorado Boulder in the City of Boulder.
# All rights reserved.
#
# File coded by: Keith White.
#
# See GitHub contributions for a more detailed list of contributors.
# https://github.com/kewh5868/saxshell/graphs/contributors  # noqa: E501
#
# See LICENSE.rst for license information.
#
##############################################################################
"""Definition of __version__."""

#  We do not use the other three variables, but can be added back if needed.
#  __all__ = ["__date__", "__git_commit__", "__timestamp__", "__version__"]

# obtain version information
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("saxshell")
except PackageNotFoundError:
    __version__ = "unknown"
