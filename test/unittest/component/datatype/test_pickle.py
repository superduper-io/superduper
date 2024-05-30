import dataclasses as dc
import typing as t
from test.db_config import DBConfig

import pandas as pd
import pytest

from superduperdb.components.component import Component
from superduperdb.components.datatype import (
    DataType,
    Empty,
    pickle_encoder,
    pickle_lazy,
    pickle_serializer,
)
from superduperdb.components.schema import Schema
from superduperdb.components.table import Table


@dc.dataclass(kw_only=True)
class SpecialComponent(Component):
    type_id: t.ClassVar[str] = "special"
    my_data: str | None = None
    my_data_lazy: str | None = None
    _artifacts: t.ClassVar = (
        ("my_data", pickle_serializer),
        ("my_file_lazy", pickle_lazy),
    )


@pytest.fixture
def random_data():
    df = pd.DataFrame(
        [{"a": 1, "b": 2}, {"a": 3, "b": 4}, {"a": 5, "b": 6}, {"a": 7, "b": 8}]
    )
    return df


@pytest.mark.parametrize("db", DBConfig.EMPTY_CASES, indirect=True)
def test_reference(db, random_data):
    c = SpecialComponent("test", my_data=random_data)

    db.add(c)

    reloaded = db.load("special", c.identifier)
    reloaded.init()
    assert isinstance(reloaded.my_data, pd.DataFrame)
    assert c.my_data.equals(reloaded.my_data)


@pytest.mark.parametrize(
    "datatype",
    [
        pickle_encoder,
        pickle_serializer,
        pickle_lazy,
    ],
)
@pytest.mark.parametrize("db", DBConfig.EMPTY_CASES, indirect=True)
def test_data(db, datatype: DataType, random_data: pd.DataFrame):
    schema = Schema(identifier="schema", fields={"x": datatype})
    table_or_collection = Table("documents", schema=schema)
    db.apply(table_or_collection)

    collection = db["documents"]
    collection.insert([{"x": random_data}]).execute()

    loaded_data = list(db.execute(collection.select()))[0]
    if datatype.encodable_cls.lazy:
        assert isinstance(loaded_data["x"].x, Empty)
        assert random_data.equals(loaded_data.unpack()["x"])
    else:
        assert random_data.equals(loaded_data.unpack()['x'])
