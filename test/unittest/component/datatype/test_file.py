import dataclasses as dc
import os
import typing as t
from test.db_config import DBConfig

import pytest

from superduperdb import Document
from superduperdb.backends.mongodb import Collection
from superduperdb.components.component import Component
from superduperdb.components.datatype import (
    Empty,
    LazyFile,
    file_lazy,
    file_serializer,
)


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


@pytest.mark.parametrize("db", [DBConfig.mongodb_empty], indirect=True)
def test_reference(db, random_file):
    c = SpecialComponent("test", my_file=random_file)

    db.add(c)

    reloaded = db.load("special", c.identifier)
    assert isinstance(reloaded.my_file, str)
    with open(reloaded.my_file, "r") as f, open(random_file, "r") as f2:
        assert f.read() == f2.read()


@pytest.mark.parametrize("db", [DBConfig.mongodb_empty], indirect=True)
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


@pytest.mark.parametrize("db", [DBConfig.mongodb_empty], indirect=True)
def test_file(db, random_file):
    from superduperdb import DataType

    dt = DataType("my-file", encodable="file")
    db.apply(dt)
    collection = Collection("my-file")
    db.execute(collection.insert_one(Document({"x": dt(random_file)})))

    data = db.execute(collection.find_one())

    path = data["x"].x
    assert os.path.exists(path)
    # Check that the file is the same
    with open(path, "r") as f, open(random_file, "r") as f2:
        assert f.read() == f2.read()


@pytest.mark.parametrize("db", [DBConfig.mongodb_empty], indirect=True)
def test_file_lazy(db, random_file):
    from superduperdb import DataType

    dt = DataType("my-file", encodable="lazy_file")
    db.apply(dt)
    collection = Collection("my-file")
    db.execute(collection.insert_one(Document({"x": dt(random_file)})))

    data = db.execute(collection.find_one())

    assert isinstance(data["x"].x, Empty)
    data = data.unpack(db)
    path = data["x"]
    assert isinstance(path, str)
    assert os.path.exists(path)
    # Check that the file is the same
    with open(path, "r") as f, open(random_file, "r") as f2:
        assert f.read() == f2.read()
