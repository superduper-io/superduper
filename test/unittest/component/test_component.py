import shutil
import pytest
import os

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
    m.compile()
    assert os.path.exists('test_export.tar.gz')
    m_reload = Component.decompile('test_export.tar.gz')
    assert isinstance(m_reload, Model)