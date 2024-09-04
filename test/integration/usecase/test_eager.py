import typing
from pprint import pprint

from superduper import Document, ObjectModel


class Tuple:
    def __init__(self, *values):
        self.values = values

    def __eq__(self, other):
        return self.values == other.values


if typing.TYPE_CHECKING:
    from superduper.base.datalayer import Datalayer


def test_graph_deps(db: "Datalayer"):
    db.cfg.auto_schema = True
    data = [
        {"x": 1, "y": "2", "z": Tuple(1, 2, 3)},
        {"x": 2, "y": "3", "z": Tuple(4, 5, 6)},
        {"x": 3, "y": "4", "z": Tuple(7, 8, 9)},
    ]

    db["documents"].insert(data).execute()

    data = list(db["documents"].select().execute(eager_mode=True))[0]

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
        list(db["documents"].select().outputs(output_a.predict_id, output_b.predict_id, output_c.predict_id).execute())[0].unpack()
    )

    result_a = data[f"_outputs__{output_a.predict_id}"]
    result_b = data[f"_outputs__{output_b.predict_id}"]
    result_c = data[f"_outputs__{output_c.predict_id}"]

    assert result_a == Tuple(1, "a")
    assert result_b == Tuple(1, "2", result_a, "b")
    assert result_c == Tuple(1, "2", Tuple(1, 2, 3), result_a, result_b, "c")

    new_data = [
        {"x": 4, "y": "5", "z": Tuple(10, 11, 12)},
        {"x": 5, "y": "6", "z": Tuple(13, 14, 15)},
        {"x": 6, "y": "7", "z": Tuple(16, 17, 18)},
    ]

    db["documents"].insert(new_data).execute()

    new_data = Document(
        list(db["documents"].select().outputs(output_a.predict_id, output_b.predict_id, output_c.predict_id).execute())[-1].unpack()
    )

    result_a = new_data[f"_outputs__{output_a.predict_id}"]
    result_b = new_data[f"_outputs__{output_b.predict_id}"]
    result_c = new_data[f"_outputs__{output_c.predict_id}"]

    assert result_a == Tuple(6, "a")
    assert result_b == Tuple(6, "7", result_a, "b")
    assert result_c == Tuple(6, "7", Tuple(16, 17, 18), result_a, result_b, "c")


def test_flatten(db: "Datalayer"):
    db.cfg.auto_schema = True
    data = [
        {"n": 1, "x": "a"},
        {"n": 2, "x": "b"},
        {"n": 3, "x": "c"},
    ]

    db["documents"].insert(data).execute()

    data = list(db["documents"].select().execute(eager_mode=True))[0]

    def func_a(n, x):
        return [x] * n

    model_a = ObjectModel(identifier="flatten", object=func_a, flatten=True)

    output_a = model_a(data['n'], data['x'])[0]
    output_a.apply()
    pprint(output_a)

    outputs = list(db[f'_outputs__{output_a.predict_id}'].select().execute(eager_mode=True))
    assert len(outputs) == 6

    results = [o[f'_outputs__{output_a.predict_id}'].data for o in outputs]

    assert results.count("a") == 1
    assert results.count("b") == 2
    assert results.count("c") == 3


def test_predict_id(db: "Datalayer"):
    db.cfg.auto_schema = True
    data = [
        {"x": 1},
    ]

    db["documents"].insert(data).execute()

    data = list(db["documents"].select().execute(eager_mode=True))[0]

    model_a = ObjectModel(identifier="a", object=lambda x: f"{x}->a")

    output_a = model_a(data['x'])

    model_b = ObjectModel(identifier="b", object=lambda x: f"{x}->b")

    output_b = model_b(output_a)

    model_c = ObjectModel(identifier="c", object=lambda x: f"{x}->c")

    output_c = model_c(output_b)

    output_c.apply()

    output = list(db['documents'].select().outputs(output_a.predict_id, output_b.predict_id, output_c.predict_id).execute())[0]
    assert output[f"_outputs__{output_a.predict_id}"] == "1->a"
    assert output[f"_outputs__{output_b.predict_id}"] == "1->a->b"
    assert output[f"_outputs__{output_c.predict_id}"] == "1->a->b->c"


def test_condition(db: "Datalayer"):
    db.cfg.auto_schema = True
    data = [
        {"x": 1},
        {"x": 2},
        {"x": 3},
    ]

    db["documents"].insert(data).execute()

    data = list(db["documents"].select().execute(eager_mode=True))[0]

    model = ObjectModel(identifier="a", object=lambda x: f"{x}->a")

    output = model(data["x"])

    output.set_condition(data["x"] == 1)

    output.apply()

    outputs = list(db[f"_outputs__{output.predict_id}"].select().execute())
    assert len(outputs) == 1

    assert outputs[0][f"_outputs__{output.predict_id}"] == "1->a"
