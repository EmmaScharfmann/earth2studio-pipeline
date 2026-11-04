from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_python_is_pinned_to_312():
    """The AIFS requirements pin cp312 flash-attn/torch wheels; on another
    Python uv builds the environment without flash-attn (see pyproject.toml)."""
    assert (ROOT / ".python-version").read_text().strip() == "3.12"
    assert 'requires-python = ">=3.12,<3.13"' in (ROOT / "pyproject.toml").read_text()
