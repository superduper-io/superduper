import glob
import subprocess
import os
import tempfile

import pytest

def run_python_file(file_path, tmp_dir):
    with pytest.raises(subprocess.TimeoutExpired):
        completed_process = subprocess.run(['jupyter', 'nbconvert', file_path, '--output-dir',tmp_dir, '--to', 'python'], capture_output=True, text=True)

        py_file_path = os.path.basename(os.path.splitext(file_path)[0]) + '.py'

        completed_process = subprocess.run(['python3', os.path.join(tmp_dir, py_file_path)], capture_output=True, text=True, timeout=10)
        assert completed_process.returncode == 0

def test_notebooks():
    with tempfile.TemporaryDirectory() as tmp_dir:
        for nb_file in glob.glob('notebooks/*'):
            run_python_file(nb_file, tmp_dir=tmp_dir)
