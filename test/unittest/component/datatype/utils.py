import dataclasses as dc
import os
import typing as t
from pprint import pprint

import numpy as np

from superduperdb.base.datalayer import Datalayer
from superduperdb.base.document import Document
from superduperdb.base.enums import DBType
from superduperdb.components.component import Component
from superduperdb.components.datatype import (
    DataType,
    Empty,
    _BaseEncodable,
)
from superduperdb.components.schema import Schema
from superduperdb.components.table import Table


def assert_equal(expect, actual):
    assert isinstance(expect, type(actual))
    if isinstance(expect, np.ndarray):
        assert np.array_equal(expect, actual)
    elif isinstance(expect, _BaseEncodable):
        if actual.lazy:
            isinstance(actual.x, Empty)
            actual.init()
        expect = expect.x
        actual = actual.x

    if isinstance(expect, str) and os.path.exists(expect):
        assert isinstance(actual, str) and os.path.exists(actual)
        with open(expect, "rb") as f:
            expect = f.read()
        with open(actual, "rb") as f:
            actual = f.read()

    assert expect, actual


def print_sep():
    print("\n", "-" * 80, "\n")


def check_data_with_schema(data, datatype: DataType):
    print("datatype", datatype)
    print_sep()
    schema = Schema(identifier="schema", fields={"x": datatype, "y": int})

    document = Document({"x": data, "y": 1})
    print(document)
    print_sep()

    encoded = document.encode(schema=schema)
    pprint(encoded)
    print_sep()

    decoded = Document.decode(encoded, schema=schema)
    if datatype.encodable_cls.lazy:
        assert isinstance(decoded["x"], datatype.encodable_cls)
        assert isinstance(decoded["x"].x, type(data))
        decoded = Document(decoded.unpack())
    pprint(decoded)
    print_sep()

    assert_equal(document["x"], decoded["x"])
    assert_equal(document["y"], decoded["y"])

    return document, encoded, decoded


def check_data_with_schema_and_db(data, datatype: DataType, db: Datalayer):
    print("datatype", datatype)
    print_sep()
    schema = Schema(identifier="schema", fields={"x": datatype, "y": int})

    table = Table("documents", schema=schema)
    db.apply(table)

    document = Document({"x": data, "y": 1})
    print(document)
    print_sep()
    db["documents"].insert([document]).execute()

    if db.databackend.db_type == DBType.MONGODB:
        encoded = db.databackend.conn["test_db"]["documents"].find_one()
    else:
        t = db.databackend.conn.table("documents")
        encoded = dict(t.select(t).execute().iloc[0])

    pprint(encoded)
    print_sep()

    decoded = list(db["documents"].select().execute())[0]

    if datatype.encodable_cls.lazy:
        assert isinstance(decoded["x"], datatype.encodable_cls)
        assert isinstance(decoded["x"].x, Empty)
        decoded = Document(decoded.unpack())

    pprint(decoded)
    print_sep()

    assert_equal(document["x"], decoded["x"])
    assert_equal(document["y"], decoded["y"])

    return document, encoded, decoded


def check_data_without_schema(data, datatype: DataType):
    print("datatype", datatype)
    print_sep()

    document = Document({"x": datatype(data), "y": 1})
    pprint(document)
    print_sep()

    encoded = document.encode()
    pprint(encoded)
    print_sep()

    decoded = Document.decode(encoded)
    pprint(decoded)
    assert_equal(document["x"], decoded["x"])
    assert_equal(document["y"], decoded["y"])
    return document, encoded, decoded


def check_data_without_schema_and_db(data, datatype: DataType, db: Datalayer):
    print("datatype", datatype)
    print("\n", "-" * 80, "\n")

    document = Document({"x": datatype(data), "y": 1})
    print(document)
    print("\n", "-" * 80, "\n")
    db["documents"].insert([document]).execute()

    if db.databackend.db_type == DBType.MONGODB:
        encoded = db.databackend.conn["test_db"]["documents"].find_one()

    pprint(encoded)
    print("\n", "-" * 80, "\n")

    decoded = list(db["documents"].select().execute())[0]
    pprint(decoded)
    print("\n", "-" * 80, "\n")

    assert_equal(document["x"], decoded["x"])
    assert_equal(document["y"], decoded["y"])

    return document, encoded, decoded


@dc.dataclass(kw_only=True)
class ChildComponent(Component):
    type_id: t.ClassVar[str] = "ChildComponent"
    y: int | None = None


@dc.dataclass(kw_only=True)
class TestComponent(Component):
    type_id: t.ClassVar[str] = "TestComponent"
    y: int = 1
    x: np.ndarray | None = None
    child: ChildComponent | None = None
    _artifacts: t.ClassVar = ()


def check_component(data, datatype: DataType):
    print("datatype", datatype)
    print_sep()

    c = TestComponent(
        "test",
        x=data,
        child=ChildComponent("child", y=2, artifacts={"x": datatype}),
        artifacts={"x": datatype},
    )
    pprint(c)
    print_sep()

    encoded = c.encode()
    pprint(encoded)
    print_sep()

    decoded = Document.decode(encoded).unpack()
    decoded.init()
    pprint(decoded)

    assert_equal(c.x, decoded.x)

    return c, encoded, decoded


def check_component_with_db(data, datatype, db):
    print("datatype", datatype)
    print_sep()

    c = TestComponent(
        "test",
        x=data,
        child=ChildComponent("child", y=2, artifacts={"x": datatype}),
        artifacts={"x": datatype},
    )
    db.add(c)
    pprint(c)
    print_sep()

    encoded = db.metadata._get_component_by_uuid(c.uuid)
    pprint(encoded)
    print_sep()

    decoded = Document.decode(encoded, db=db).unpack()
    decoded.init()
    pprint(decoded)
    assert_equal(c.x, decoded.x)

    return c, encoded, decoded
