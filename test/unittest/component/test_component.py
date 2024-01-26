import os
import shutil

import pytest

from superduperdb import Model
from superduperdb.components.component import Component
from superduperdb.components.datatype import dill_serializer


@pytest.fixture
def cleanup():
    yield
    os.remove('test_export.tar.gz')
    shutil.rmtree('test_export')


def test_compile_decompile(cleanup):
    m = Model('test_export', object=lambda x: x, datatype=dill_serializer)
    m.version = 0
    m.datatype.version = 0
    m.export()
    assert os.path.exists('test_export.tar.gz')
    m_reload = Component.import_('test_export.tar.gz')
    assert isinstance(m_reload, Model)
