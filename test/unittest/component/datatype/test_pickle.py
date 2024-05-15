import dataclasses as dc
import typing as t
from test.db_config import DBConfig

import pandas as pd
import pytest

from superduperdb.components.component import Component
from superduperdb.components.datatype import (
    pickle_lazy,
    pickle_serializer,
)


@dc.dataclass(kw_only=True)
class SpecialComponent(Component):
    type_id: t.ClassVar[str] = "special"
    my_data: str | None = None
    my_data_lazy: str | None = None
    _artifacts: t.ClassVar = (("my_data", pickle_serializer), ("my_file_lazy", pickle_lazy))


@pytest.fixture
def random_data():

    df = pd.DataFrame(
        [{"a": 1, "b": 2}, {"a": 3, "b": 4}, {"a": 5, "b": 6}, {"a": 7, "b": 8}]
    )
    return df



@pytest.mark.parametrize("db", [DBConfig.mongodb_empty], indirect=True)
def test_reference(db, random_data):
    c = SpecialComponent("test", my_data=random_data)

    db.add(c)

    reloaded = db.load("special", c.identifier)
    reloaded.init()
    assert isinstance(reloaded.my_data, pd.DataFrame)
    assert c.my_data.equals(reloaded.my_data)
