import difflib
import json
import base64
import io
import os

import numpy as np
from PIL import Image

import nbclient
import nbformat
import pytest

try:
    import nbdime
    from nbdime.diffing import diff_notebooks

    HAS_NBDIME = True
except ImportError:
    HAS_NBDIME = False


def pytest_collect_file(parent, file_path):
    """Collect test_*.ipynb files."""
    if file_path.suffix == ".ipynb" and file_path.name.startswith("test_"):
        return NotebookFile.from_parent(parent, path=file_path)


class NotebookFile(pytest.File):
    """Custom pytest File for Jupyter notebooks."""

    def collect(self):
        """Collect cells from the notebook as individual tests."""
        nb_path = self.fspath
        notebook = nbformat.read(nb_path, as_version=4)

        # Check if notebook is in pristine state
        yield PristineStateTest.from_parent(self, name="pristine_state", notebook=notebook)

        # Create a notebook executor for all cells
        yield NotebookExecutionTest.from_parent(
            self,
            name="notebook_execution",
            notebook=notebook,
            notebook_path=nb_path
        )


def image_to_ascii(img, width=80, height=40):
    """Convert an image to ASCII art."""
    # Resize image
    img = img.resize((width, height), Image.NEAREST)
    # Convert to grayscale
    img = img.convert('L')
    # Get pixel data
    pixels = np.array(img)

    # ASCII characters from dark to light
    chars = '.:-=+*#%@'

    def char_for_val(p):
        if p == 0:
            return ' '
        return chars[min(int(p * len(chars) / 256), len(chars) - 1)]

    # Map pixel values to ASCII characters
    ascii_img = []
    for row in pixels:
        ascii_row = ''.join(char_for_val(p) for p in row)
        ascii_img.append(ascii_row)

    return '\n'.join(ascii_img)


def do_images_differ(img1_data, img2_data, width=80, height=40):
    """Create a visual diff between two images in ASCII art."""
    # Decode base64 images
    try:
        img1_bytes = base64.b64decode(img1_data)
        img2_bytes = base64.b64decode(img2_data)

        img1 = Image.open(io.BytesIO(img1_bytes))
        img2 = Image.open(io.BytesIO(img2_bytes))

        if img1.width != img2.width or img1.height != img2.height:
            return True, f"Image dimensions differ: {img1.size} vs {img2.size}"

        # Convert to numpy arrays
        arr1 = np.array(img1.convert('RGB')).astype(float)
        arr2 = np.array(img2.convert('RGB')).astype(float)

        # Calculate difference
        diff = np.abs(arr1 - arr2)
        if diff.max() < 2:
            return False, None

        diff = diff * 255.0 / diff.max()  # Normalize to enhance subtle differences

        # Convert difference back to image
        diff_img = Image.fromarray(diff.astype(np.uint8))

        ascii_diff = image_to_ascii(diff_img, width, height)
        percent_diff = 100 * diff.sum() / (255 * diff.size)

        return True, f"Normalized difference ({percent_diff:.4f}%):\n{ascii_diff}"
    except Exception as e:
        return True, f"Failed to create image diff: {str(e)}"


def format_diff(expected, actual):
    """Format diff between expected and actual outputs."""
    if HAS_NBDIME and isinstance(expected, dict) and isinstance(actual, dict):
        # Use nbdime for structured diffs
        diff = diff_notebooks({"cells": [{"outputs": [expected]}]},
                              {"cells": [{"outputs": [actual]}]})
        return nbdime.prettyprint.pretty_print_notebook_diff(diff, '')
    else:
        # Use regular diff for text
        expected_str = json.dumps(expected, indent=2) if isinstance(expected, dict) else str(expected)
        actual_str = json.dumps(actual, indent=2) if isinstance(actual, dict) else str(actual)
        diff = difflib.unified_diff(
            expected_str.splitlines(),
            actual_str.splitlines(),
            n=3
        )
        return '\n'.join(diff)


def compare_outputs(expected_outputs, actual_outputs):
    """Compare expected and actual outputs, returning differences."""
    if len(expected_outputs) != len(actual_outputs):
        return [f"Number of outputs differ: expected {len(expected_outputs)}, got {len(actual_outputs)}"]

    differences = []

    for i, (expected, actual) in enumerate(zip(expected_outputs, actual_outputs)):
        # Compare output types
        if expected['output_type'] != actual['output_type']:
            differences.append(
                f"Output {i} types differ: expected {expected['output_type']}, got {actual['output_type']}")
            continue

        # Handle different output types
        if expected['output_type'] == 'stream':
            if expected.get('name') != actual.get('name'):
                differences.append(f"Stream name differs: expected {expected.get('name')}, got {actual.get('name')}")

            if expected.get('text') != actual.get('text'):
                differences.extend(
                    (
                        f"Stream text differs for output {i}:",
                        format_diff(expected.get('text'), actual.get('text')),
                    )
                )
        elif expected['output_type'] in ['display_data', 'execute_result']:
            # Compare data dictionaries
            expected_data = expected.get('data', {})
            actual_data = actual.get('data', {})

            # Check for missing keys
            for key in set(expected_data.keys()) | set(actual_data.keys()):
                if key not in expected_data:
                    differences.append(f"Missing key {key} in expected output {i}")
                elif key not in actual_data:
                    differences.append(f"Missing key {key} in actual output {i}")
                elif key == 'image/png':
                    images_differ, message = do_images_differ(expected_data[key], actual_data[key])
                    if images_differ:
                        # For images, create an ASCII representation of the diff
                        differences.extend([
                            f"Image data differs in output {i}:",
                            message
                        ])
                elif expected_data[key] != actual_data[key]:
                    differences.extend(
                        (
                            f"Data for {key} differs in output {i}:",
                            format_diff(
                                expected_data[key], actual_data[key]
                            ),
                        )
                    )
            # Compare metadata if present
            if expected.get('metadata', {}) != actual.get('metadata', {}):
                differences.extend(
                    (
                        f"Metadata differs in output {i}:",
                        format_diff(
                            expected.get('metadata', {}),
                            actual.get('metadata', {}),
                        ),
                    )
                )
        elif expected['output_type'] == 'error':
            # Compare error name, value and traceback
            for attr in ['ename', 'evalue']:
                if expected.get(attr) != actual.get(attr):
                    differences.extend(
                        (
                            f"Error {attr} differs in output {i}:",
                            format_diff(expected.get(attr), actual.get(attr)),
                        )
                    )
            # Compare tracebacks
            if expected.get('traceback') != actual.get('traceback'):
                differences.append(f"Error traceback differs in output {i}")

    return differences


class PristineStateTest(pytest.Item):
    """Test if the notebook is in a pristine state."""

    def __init__(self, parent, name, notebook):
        super().__init__(name, parent)
        self.notebook = notebook

    def runtest(self):
        """Check if the notebook is in a pristine state."""
        failures = []

        # Check for execution order
        expected_exec_count = 1
        for i, cell in enumerate(self.notebook.cells):
            if cell.cell_type == 'code':
                # Check if cell has been executed
                if cell.execution_count is None:
                    failures.append(f"Cell {i} has not been executed")
                elif cell.execution_count != expected_exec_count:
                    failures.append(
                        f"Cell {i} has execution count {cell.execution_count}, expected {expected_exec_count}")
                expected_exec_count += 1

        if failures:
            raise PristineStateError("\n".join(failures))

    def repr_failure(self, excinfo, **kwargs):
        """Custom failure representation."""
        if isinstance(excinfo.value, PristineStateError):
            return f"Notebook is not in a pristine state:\n{excinfo.value}"
        return super().repr_failure(excinfo)

    def reportinfo(self):
        return self.fspath, 0, "Checking notebook pristine state"


class NotebookExecutionTest(pytest.Item):
    """Test all cells in a notebook, maintaining kernel state between cells."""

    def __init__(self, parent, name, notebook, notebook_path):
        super().__init__(name, parent)
        self.notebook = notebook
        self.notebook_path = notebook_path

    def runtest(self):
        """Execute the notebook and compare cell outputs."""
        # Create a fresh copy of the notebook for execution
        test_notebook = nbformat.read(self.notebook_path, as_version=4)

        # Clear all outputs before execution
        for cell in test_notebook.cells:
            if cell.cell_type == 'code':
                cell.outputs = []
                cell.execution_count = None

        # Execute the entire notebook with a single kernel
        executor = nbclient.NotebookClient(
            test_notebook,
            timeout=600,  # Increase timeout for longer-running notebooks
            allow_errors=True  # Don't stop on errors, check them later
        )

        try:
            executor.execute()
        except Exception as e:
            raise ExecutionError("Notebook execution failed") from e

        # Load original notebook to compare outputs
        original_notebook = nbformat.read(self.notebook_path, as_version=4)

        # Compare outputs cell by cell
        cell_failures = []
        for i, (original_cell, executed_cell) in enumerate(zip(original_notebook.cells, test_notebook.cells)):
            if original_cell.cell_type != 'code':
                continue

            # Compare execution counts
            if original_cell.execution_count != executed_cell.execution_count:
                cell_failures.append(
                    f"Cell {i} execution count differs: expected {original_cell.execution_count}, "
                    f"got {executed_cell.execution_count}"
                )

            # Check for errors in executed cell that weren't in original
            executed_errors = [o for o in executed_cell.outputs if o.get('output_type') == 'error']
            original_errors = [o for o in original_cell.outputs if o.get('output_type') == 'error']

            if executed_errors and not original_errors:
                error = executed_errors[0]
                cell_failures.append(
                    f"Cell {i} execution produced an error that wasn't in the original:\n"
                    f"{error.get('ename', '')}: {error.get('evalue', '')}"
                )
                continue

            # Compare outputs
            differences = compare_outputs(original_cell.outputs, executed_cell.outputs)
            if differences:
                cell_failures.append(f"Cell {i} output differs:\n" + "\n".join(differences))

        if cell_failures:
            # save the new ipynb for comparison
            base, name = os.path.split(self.notebook_path)
            # if you need to get this file from a github-actions-like env, run `act --bind -j test`
            failure_path = os.path.join(base, f"test-failure-{name}")
            with open(failure_path, "w") as f:
                nbformat.write(test_notebook, f)
            raise OutputDiffError("\n\n".join(cell_failures))

    def repr_failure(self, excinfo, **kwargs):
        """Custom failure representation."""
        if isinstance(excinfo.value, OutputDiffError):
            return f"Notebook outputs differ:\n{excinfo.value}"
        elif isinstance(excinfo.value, ExecutionError):
            return f"Notebook execution failed:\n{excinfo.value}"
        return super().repr_failure(excinfo)

    def reportinfo(self):
        return self.fspath, 0, "Testing notebook execution"


class PristineStateError(Exception):
    """Error raised when notebook is not in pristine state."""
    pass


class OutputDiffError(Exception):
    """Error raised when outputs differ."""
    pass


class ExecutionError(Exception):
    """Error raised when cell execution fails."""
    pass
