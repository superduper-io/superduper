import typing as t

import numpy as np

from superduper import Document, ObjectModel

if t.TYPE_CHECKING:
    from superduper.base.datalayer import Datalayer


def _check_equal(a, b):
    if not isinstance(a, type(b)):
        return False

    if isinstance(a, np.ndarray):
        return np.array_equal(a, b)

    if isinstance(a, t.Mapping):
        if set(a.keys()) != set(b.keys()):
            return False

        for k, v in a.items():
            if not _check_equal(v, b[k]):
                return False
        else:
            return True

    return a == b


def build_graph_listener(db: "Datalayer"):
    db.cfg.auto_schema = True
    data = [
        {"x": 1, "y": "2", "z": np.array([1, 2, 3])},
        {"x": 2, "y": "3", "z": np.array([4, 5, 6])},
        {"x": 3, "y": "4", "z": np.array([7, 8, 9])},
    ]

    db["documents"].insert(data).execute()

    def func_a(x):
        return {"x": x, "model": "a"}

    primary_id = db["documents"].primary_id

    model_a = ObjectModel(identifier="model_a", object=func_a)

    listener_a = model_a.to_listener(
        key="x",
        select=db["documents"].select(),
        identifier="listener_a",
        predict_id="a",
        predict_kwargs={"max_chunk_size": 1},
    )

    def func_b(x, y, o_a):
        return {"x": x, "y": y, "o_a": o_a, "model": "b"}

    model_b = ObjectModel(identifier="model_b", object=func_b)
    listener_b = model_b.to_listener(
        key=("x", "y", "_outputs__a"),
        select=db["documents"].select(primary_id, "x", "y").outputs("a"),
        identifier="listener_b",
        predict_id="b",
        predict_kwargs={"max_chunk_size": 1},
    )

    def func_c(x, y, z, o_a, o_b):
        return {"x": x, "y": y, "z": z, "o_a": o_a, "o_b": o_b, "model": "c"}

    model_c = ObjectModel(identifier="model_c", object=func_c)
    listener_c = model_c.to_listener(
        key=("x", "y", "z", "_outputs__a", "_outputs__b"),
        select=db["documents"].select(primary_id, "x", "y", "z").outputs("a", "b"),
        identifier="listener_c",
        predict_id="c",
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

    assert _check_equal(output_a, {"x": 1, "model": "a"})
    assert _check_equal(output_b, {"x": 1, "y": "2", "o_a": output_a, "model": "b"})
    assert _check_equal(
        output_c,
        {
            "x": 1,
            "y": "2",
            "z": np.array([1, 2, 3]),
            "o_a": output_a,
            "o_b": output_b,
            "model": "c",
        },
    )

    new_data = [
        {"x": 4, "y": "5", "z": np.array([10, 11, 12])},
        {"x": 5, "y": "6", "z": np.array([13, 14, 15])},
        {"x": 6, "y": "7", "z": np.array([16, 17, 18])},
    ]

    db["documents"].insert(new_data).execute()

    new_data = Document(
        list(db["documents"].select().outputs("a", "b", "c").execute())[-1].unpack()
    )

    output_a = new_data["_outputs__a"]
    output_b = new_data["_outputs__b"]
    output_c = new_data["_outputs__c"]

    assert _check_equal(output_a, {"x": 6, "model": "a"})
    assert _check_equal(output_b, {"x": 6, "y": "7", "o_a": output_a, "model": "b"})
    assert _check_equal(
        output_c,
        {
            "x": 6,
            "y": "7",
            "z": np.array([16, 17, 18]),
            "o_a": output_a,
            "o_b": output_b,
            "model": "c",
        },
    )
