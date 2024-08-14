import os

import pytest

skip = not os.environ.get('SUPERDUPER_CONFIG', "").endswith('default.yaml')

if skip:
    pytest.skip("Skipping this file for now", allow_module_level=True)
