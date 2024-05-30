import dataclasses as dc
import os
import typing as t
from test.db_config import DBConfig

import pytest

from superduperdb import DataType, Schema
from superduperdb.components.component import Component
from superduperdb.components.datatype import (
    Empty,
    LazyFile,
    file_lazy,
    file_serializer,
)
from superduperdb.components.table import Table


@dc.dataclass(kw_only=True)
class SpecialComponent(Component):
    type_id: t.ClassVar[str] = "special"
    my_file: str | None = None
    my_file_lazy: str | None = None
    _artifacts: t.ClassVar = (("my_file", file_serializer), ("my_file_lazy", file_lazy))


@pytest.fixture
def random_file(tmpdir):
    file_name = os.path.join(tmpdir, "HelloWorld.txt")

    with open(file_name, "w") as f:
        f.write("Hello, World!")

    return file_name


@pytest.mark.parametrize("db", DBConfig.EMPTY_CASES, indirect=True)
def test_reference(db, random_file):
    c = SpecialComponent("test", my_file=random_file)

    db.add(c)

    reloaded = db.load("special", c.identifier)
    reloaded.init()
    assert isinstance(reloaded.my_file, str)
    assert reloaded.my_file != random_file
    with open(reloaded.my_file, "r") as f, open(random_file, "r") as f2:
        assert f.read() == f2.read()


@pytest.mark.parametrize("db", DBConfig.EMPTY_CASES, indirect=True)
def test_reference_lazy(db, random_file):
    c = SpecialComponent("test", my_file_lazy=random_file)

    db.add(c)

    reloaded = db.load("special", c.identifier)
    assert isinstance(reloaded.my_file_lazy, LazyFile)
    assert isinstance(reloaded.my_file_lazy.x, Empty)
    reloaded.init()
    assert isinstance(reloaded.my_file_lazy, str)
    with open(reloaded.my_file_lazy, "r") as f, open(random_file, "r") as f2:
        assert f.read() == f2.read()


@pytest.mark.parametrize("db", DBConfig.EMPTY_CASES, indirect=True)
def test_file(db, random_file):
    dt = DataType("my-file", encodable="file")
    schema = Schema(identifier="schema", fields={"x": dt})
    table_or_collection = Table("documents", schema=schema)
    db.apply(table_or_collection)

    collection = db['documents']
    collection.insert([{"x": random_file}]).execute()

    data = list(db.execute(collection.select()))[0].unpack()

    path = data["x"]
    assert os.path.exists(path)
    # Check that the file is the same
    assert path != random_file
    with open(path, "r") as f, open(random_file, "r") as f2:
        assert f.read() == f2.read()


@pytest.mark.parametrize("db", DBConfig.EMPTY_CASES, indirect=True)
def test_file_lazy(db, random_file):
    from superduperdb import DataType

    dt = DataType("my-file", encodable="lazy_file")
    schema = Schema(identifier="schema", fields={"x": dt})
    table_or_collection = Table("documents", schema=schema)
    db.apply(table_or_collection)

    collection = db['documents']
    collection.insert([{"x": random_file}]).execute()

    data = list(db.execute(collection.select()))[0]

    assert isinstance(data["x"].x, Empty)
    data = data.unpack()
    path = data["x"]
    assert isinstance(path, str)
    assert os.path.exists(path)
    # Check that the file is the same
    assert path != random_file
    with open(path, "r") as f, open(random_file, "r") as f2:
        assert f.read() == f2.read()
