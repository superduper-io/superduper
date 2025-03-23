import dataclasses as dc
import os
import typing as t
from pprint import pprint

import numpy as np

from superduper.base.datalayer import Datalayer
from superduper.base.datatype import BaseDataType, pickle_serializer
from superduper.base.document import Document
from superduper.base.schema import Schema
from superduper.components.component import Component
from superduper.components.table import Table


def assert_equal(expect, actual):
    assert isinstance(expect, type(actual))
    if isinstance(expect, np.ndarray):
        assert np.array_equal(expect, actual)

    if isinstance(expect, str) and os.path.exists(expect):
        assert isinstance(actual, str) and os.path.exists(actual)
        with open(expect, "rb") as f:
            expect = f.read()
        with open(actual, "rb") as f:
            actual = f.read()

    assert expect, actual


def print_sep():
    print("\n", "-" * 80, "\n")


def check_data_with_schema(data, datatype, db):
    print("datatype", datatype)
    print_sep()

    schema = Schema(fields={"x": datatype, "y": int})

    document = Document({"x": data, "y": 1})
    print(document)
    print_sep()

    encoded = document.encode(schema=schema, db=db)

    pprint(encoded)
    print_sep()

    decoded = Document.decode(encoded, schema=schema).unpack()

    pprint(decoded)
    print_sep()

    assert_equal(document["x"], decoded["x"])
    assert_equal(document["y"], decoded["y"])

    return document, encoded, decoded


def check_data_with_schema_and_db(data, datatype: BaseDataType, db: Datalayer):
    print("datatype", datatype)
    print_sep()

    table = Table("documents", fields={"x": str(datatype).lower(), "y": 'int'})
    db.apply(table)

    document = {"x": data, "y": 1}
    print(document)
    print_sep()

    db["documents"].insert([document])

    decoded = db["documents"].select().execute()[0]

    decoded = decoded.unpack()

    pprint(decoded)
    print_sep()
    assert_equal(document["x"], decoded["x"])
    assert_equal(document["y"], decoded["y"])

    return document, decoded


class ChildComponent(Component):
    type_id: t.ClassVar[str] = "ChildComponent"
    y: int | None = None


class TestComponent(Component):
    type_id: t.ClassVar[str] = "TestComponent"
    y: int = 1
    x: np.ndarray | None = None
    child: ChildComponent | None = None
    _fields = {'x': pickle_serializer}


def check_component(data, datatype: BaseDataType):
    print("datatype", datatype)
    print_sep()

    c = TestComponent(
        "test",
        x=data,
        child=ChildComponent("child", y=2),
    )
    pprint(c)
    print_sep()

    encoded = c.encode()
    pprint(encoded)
    print_sep()

    decoded = Component.decode(encoded)
    decoded.setup()
    pprint(decoded)

    assert_equal(c.x, decoded.x)

    return c, encoded, decoded


def check_component_with_db(data, datatype, db):
    print("datatype", datatype)
    print_sep()

    c = TestComponent(
        "test",
        x=data,
        child=ChildComponent("child", y=2),
    )
    db.apply(c)
    pprint(c)
    print_sep()

    encoded = db.metadata.get_component_by_uuid(c.__class__.__name__, c.uuid)
    pprint(encoded)
    print_sep()

    decoded = Component.decode(encoded, db=db)
    decoded.setup()
    pprint(decoded)
    assert_equal(c.x, decoded.x)

    return c, encoded, decoded
