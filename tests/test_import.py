import saxshell


def test_import() -> None:
    assert hasattr(saxshell, "__version__")


def test_version_is_string() -> None:
    assert isinstance(saxshell.__version__, str)
