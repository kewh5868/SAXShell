"""Unit tests for __version__.py."""

import saxshell  # noqa


def test_package_version():
    """Ensure the package version is defined and not set to the initial
    placeholder."""
    assert hasattr(saxshell, "__version__")
    assert saxshell.__version__ != "0.0.0"
