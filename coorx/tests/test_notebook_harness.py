"""Tests for the notebook-execution plugin defined in the repository's conftest.py."""

import nbformat

from conftest import execute_notebook

STARTUP_SCRIPT = """
ip = get_ipython()
ip.display_formatter.formatters['text/html'].for_type(int, lambda value: '<noise/>')
"""


def test_execute_notebook_ignores_user_ipython_startup(tmp_path, monkeypatch):
    """A developer's IPython startup scripts must not add outputs to executed cells."""
    startup_dir = tmp_path / "profile_default" / "startup"
    startup_dir.mkdir(parents=True)
    (startup_dir / "50-noise.py").write_text(STARTUP_SCRIPT)
    monkeypatch.setenv("IPYTHONDIR", str(tmp_path))

    notebook = nbformat.v4.new_notebook()
    notebook.cells = [nbformat.v4.new_code_cell("1 + 1")]
    notebook.metadata["kernelspec"] = {"name": "python3", "language": "python", "display_name": "python3"}

    execute_notebook(notebook)

    outputs = notebook.cells[0].outputs
    assert len(outputs) == 1
    assert set(outputs[0]["data"]) == {"text/plain"}
