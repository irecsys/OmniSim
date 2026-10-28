"""
Every .py file in the repository must compile under the Python running the tests (CI: 3.10). Catches files
that no test imports, such as scripts/sample_baseline.py, which did not compile on 3.10.
The byte code goes to tmp_path, so nothing is written into the repository.
"""
import os
import py_compile

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP_DIRS = {'.git', '__pycache__', '.venv', 'venv', 'env', 'node_modules', '.claude'}


def python_files():
    for root, dirs, files in os.walk(ROOT):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
        for f in sorted(files):
            if f.endswith('.py'):
                yield os.path.relpath(os.path.join(root, f), ROOT)


def test_there_are_python_files():
    assert len(list(python_files())) > 50


@pytest.mark.parametrize('path', list(python_files()))
def test_compiles(path, tmp_path):
    py_compile.compile(os.path.join(ROOT, path), cfile=str(tmp_path / 'x.pyc'), doraise=True)
