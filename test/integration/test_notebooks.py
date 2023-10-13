import os
import subprocess

import pytest
import tdir

import superduperdb as s

ENABLE = 'NB_TEST' in os.environ


def run_python_file(file_path, tmp_dir):
    subprocess.run(
        ['jupyter', 'nbconvert', file_path, '--output-dir', tmp_dir, '--to', 'python'],
        capture_output=True,
        text=True,
    )

    py_file_path = os.path.basename(os.path.splitext(file_path)[0]) + '.py'

    completed_process = subprocess.run(
        ['python3', os.path.join(tmp_dir, py_file_path)],
        capture_output=True,
        text=True,
        timeout=60 * 2,
    )
    print(completed_process.stdout)
    assert completed_process.returncode == 0


NOTEBOOKS = [
    'notebooks/mnist_clean.ipynb',
]


@pytest.mark.skipif(not ENABLE, reason='Notebook tests are disabled')
@pytest.mark.parametrize('nb_file', NOTEBOOKS)
@tdir(superduperdb=s.ROOT / 'superduperdb')
def test_notebooks(nb_file):
    run_python_file(nb_file, tmp_dir='.')
