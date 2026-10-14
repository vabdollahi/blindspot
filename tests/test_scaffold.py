from importlib.metadata import metadata


def test_project_declares_supported_python_version() -> None:
    """Check the installed package metadata matches the repository's Python floor."""
    assert set(metadata("blindspot")["Requires-Python"].split(",")) == {
        ">=3.12",
        "<3.13",
    }
