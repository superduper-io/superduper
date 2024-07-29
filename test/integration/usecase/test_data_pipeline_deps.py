import os
import typing as t

import pytest

from superduper import ObjectModel
from superduper.base.document import Document

skip = not os.environ.get('SUPERDUPER_CONFIG', "").endswith('mongodb.yaml')

if skip:
    # TODO: Enable this when select support filter
    pytest.skip("Skipping this file for now", allow_module_level=True)


class Tuple:
    def __init__(self, *values):
        self.values = values

    def __eq__(self, other):
        return self.values == other.values


if t.TYPE_CHECKING:
    from superduper.base.datalayer import Datalayer


def test_graph_deps(db: "Datalayer"):
    db.cfg.auto_schema = True
    data = [
        {"x": 1, "y": "2", "z": Tuple(1, 2, 3)},
        {"x": 2, "y": "3", "z": Tuple(4, 5, 6)},
        {"x": 3, "y": "4", "z": Tuple(7, 8, 9)},
    ]

    db["documents"].insert(data).execute()

    def func_a(x):
        return Tuple(x, "a")

    model_a = ObjectModel(identifier="model_a", object=func_a)

    listener_a = model_a.to_listener(
        key="x",
        select=db["documents"].select(),
        identifier="listener_a",
        uuid="a",
        predict_kwargs={"max_chunk_size": 1},
    )

    def func_b(x, y, o_a):
        return Tuple(x, y, o_a, "b")

    model_b = ObjectModel(identifier="model_b", object=func_b)
    listener_b = model_b.to_listener(
        key=("x", "y", "_outputs__a"),
        select=db["documents"].find({}, {'x': 1, 'y': 1}).outputs('a'),
        identifier="listener_b",
        uuid="b",
        predict_kwargs={"max_chunk_size": 1},
    )

    def func_c(x, y, z, o_a, o_b):
        return Tuple(x, y, z, o_a, o_b, "c")

    model_c = ObjectModel(identifier="model_c", object=func_c)
    listener_c = model_c.to_listener(
        key=("x", "y", "z", "_outputs__a", "_outputs__b"),
        select=db["documents"].find({}, {'x': 1, 'y': 1, 'z': 1}).outputs('a', 'b'),
        identifier="listener_c",
        uuid="c",
        predict_kwargs={"max_chunk_size": 1},
    )

    db.apply(listener_a)
    db.apply(listener_b)
    db.apply(listener_c)

    data = Document(
        list(db["documents"].select().outputs("a", "b", "c").execute())[0].unpack()
    )

    output_a = data["_outputs__a"]
    output_b = data["_outputs__b"]
    output_c = data["_outputs__c"]

    assert output_a == Tuple(1, "a")
    assert output_b == Tuple(1, "2", output_a, "b")
    assert output_c == Tuple(1, "2", Tuple(1, 2, 3), output_a, output_b, "c")

    new_data = [
        {"x": 4, "y": "5", "z": Tuple(10, 11, 12)},
        {"x": 5, "y": "6", "z": Tuple(13, 14, 15)},
        {"x": 6, "y": "7", "z": Tuple(16, 17, 18)},
    ]

    db["documents"].insert(new_data).execute()

    new_data = Document(
        list(db["documents"].select().outputs("a", "b", "c").execute())[-1].unpack()
    )

    output_a = new_data["_outputs__a"]
    output_b = new_data["_outputs__b"]
    output_c = new_data["_outputs__c"]

    assert output_a == Tuple(6, "a")
    assert output_b == Tuple(6, "7", output_a, "b")
    assert output_c == Tuple(6, "7", Tuple(16, 17, 18), output_a, output_b, "c")
