import dataclasses as dc
import typing as t
from test.db_config import DBConfig

import pytest

from superduperdb.components.component import Component
from superduperdb.components.datatype import File, file_serializer


@dc.dataclass(kw_only=True)
class SpecialComponent(Component):
    type_id: t.ClassVar[str] = 'special'
    _artifacts: t.ClassVar = (('my_file', file_serializer),)
    my_file: str


@pytest.fixture
def a_file():
    import os

    os.system('touch .a_file')
    yield
    os.remove('.a_file')


@pytest.mark.parametrize("db", [DBConfig.mongodb_empty], indirect=True)
def test_reference(db, a_file):
    c = SpecialComponent('test', my_file='.a_file')

    db.add(c)

    reloaded = db.load('special', c.identifier)
    assert isinstance(reloaded.my_file, File)
    reloaded.init()
    assert isinstance(reloaded.my_file, str)
