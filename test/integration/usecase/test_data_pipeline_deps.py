import time
import typing as t
from test.db_config import DBConfig

import pytest

from superduper import ObjectModel
from superduper.base.document import Document


class Tuple:
    def __init__(self, *values):
        self.values = values

    def __eq__(self, other):
        return self.values == other.values


if t.TYPE_CHECKING:
    from superduper.base.datalayer import Datalayer


# TODO: Need to support MongoDB query.outputs()
@pytest.mark.skip
@pytest.mark.parametrize("db", [DBConfig.mongodb_empty], indirect=True)
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
        key=("x", "y", "_outputs.a"),
        select=db["documents"].select(),
        identifier="listener_b",
        uuid="b",
        predict_kwargs={"max_chunk_size": 1},
    )

    def func_c(x, y, z, o_a, o_b):
        time.sleep(1)
        return Tuple(x, y, z, o_a, o_b, "c")

    model_c = ObjectModel(identifier="model_c", object=func_c)
    listener_c = model_c.to_listener(
        # key={"x": "x", "y": "y", "z": "z", "_outputs.a": "o_a", "_outputs.b": "o_b"},
        key=("x", "y", "z", "_outputs.a", "_outputs.b"),
        select=db["documents"].select(),
        identifier="listener_c",
        uuid="c",
        predict_kwargs={"max_chunk_size": 1},
    )

    db.apply(listener_a)
    db.apply(listener_b)
    db.apply(listener_c)

    data = Document(list(db["documents"].select().execute())[0].unpack())

    output_a = data["_outputs.a"]
    output_b = data["_outputs.b"]
    output_c = data["_outputs.c"]

    assert output_a == Tuple(1, "a")
    assert output_b == Tuple(1, "2", output_a, "b")
    assert output_c == Tuple(1, "2", Tuple(1, 2, 3), output_a, output_b, "c")

    new_data = [
        {"x": 4, "y": "5", "z": Tuple(10, 11, 12)},
        {"x": 5, "y": "6", "z": Tuple(13, 14, 15)},
        {"x": 6, "y": "7", "z": Tuple(16, 17, 18)},
    ]

    db["documents"].insert(new_data).execute()

    new_data = Document(list(db["documents"].select().execute())[-1].unpack())

    output_a = new_data["_outputs.a"]
    output_b = new_data["_outputs.b"]
    output_c = new_data["_outputs.c"]

    assert output_a == Tuple(6, "a")
    assert output_b == Tuple(6, "7", output_a, "b")
    assert output_c == Tuple(6, "7", Tuple(16, 17, 18), output_a, output_b, "c")
