import typing
from test.db_config import DBConfig

import pytest

from superduper import Document, ObjectModel


class Tuple:
    def __init__(self, *values):
        self.values = values

    def __eq__(self, other):
        return self.values == other.values


if typing.TYPE_CHECKING:
    from superduper.base.datalayer import Datalayer


@pytest.mark.parametrize("db", DBConfig.EMPTY_CASES, indirect=True)
def test_graph_deps(db: "Datalayer"):
    db.cfg.auto_schema = True
    data = [
        {"x": 1, "y": "2", "z": Tuple(1, 2, 3)},
        {"x": 2, "y": "3", "z": Tuple(4, 5, 6)},
        {"x": 3, "y": "4", "z": Tuple(7, 8, 9)},
    ]

    db["documents"].insert(data).execute()

    data = db["documents"].datas()[0]

    def func_a(x):
        return Tuple(x, "a")

    model_a = ObjectModel(identifier="a", object=func_a)

    output_a = model_a(data["x"])

    def func_b(x, y, o_a):
        return Tuple(x, y, o_a, "b")

    model_b = ObjectModel(identifier="b", object=func_b)

    output_b = model_b(data["x"], data["y"], output_a)

    def func_c(x, y, z, o_a, o_b):
        return Tuple(x, y, z, o_a, o_b, "c")

    model_c = ObjectModel(identifier="c", object=func_c)

    output_c = model_c(data["x"], data["y"], data["z"], output_a, output_b)

    output_c.apply()

    data = Document(
        list(db["documents"].select().outputs("a", "b", "c").execute())[0].unpack()
    )

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

    new_data = Document(
        list(db["documents"].select().outputs("a", "b", "c").execute())[-1].unpack()
    )

    output_a = new_data["_outputs.a"]
    output_b = new_data["_outputs.b"]
    output_c = new_data["_outputs.c"]

    assert output_a == Tuple(6, "a")
    assert output_b == Tuple(6, "7", output_a, "b")
    assert output_c == Tuple(6, "7", Tuple(16, 17, 18), output_a, output_b, "c")
