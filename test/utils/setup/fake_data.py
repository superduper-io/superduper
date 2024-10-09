import random

import numpy as np

# ruff: noqa: E402
from superduper.base.datalayer import Datalayer
from superduper.components.dataset import Dataset
from superduper.components.listener import Listener
from superduper.components.model import ObjectModel
from superduper.components.schema import Schema
from superduper.components.table import Table
from superduper.components.vector_index import VectorIndex
from superduper.ext.numpy.encoder import array

GLOBAL_TEST_N_DATA_POINTS = 100


def get_valid_dataset(db):
    table = db["documents"]
    select = db["documents"].select().filter(table["_fold"] == "valid")
    d = Dataset(
        identifier="my_valid",
        select=select,
        sample_size=100,
    )
    return d


def add_random_data(
    db: Datalayer,
    table_name: str = "documents",
    n: int = GLOBAL_TEST_N_DATA_POINTS,
):
    float_array = array(dtype="float", shape=(32,))

    schema = Schema(
        identifier=table_name,
        fields={
            "id": "str",
            "x": float_array,
            "y": "int32",
            "z": float_array,
        },
    )
    t = Table(identifier=table_name, schema=schema)
    db.apply(t)
    data = []
    for i in range(n):
        x = np.random.rand(32)
        y = int(random.random() > 0.5)
        z = np.random.rand(32)
        fold = int(random.random() > 0.5)
        fold = "valid" if fold else "train"
        data.append({"id": str(i), "x": x, "y": y, "z": z, "_fold": fold})
    db[table_name].insert(data).execute()


def add_datatypes(db: Datalayer):
    for n in [8, 16, 32]:
        db.apply(array(dtype="float", shape=(n,)))


def add_models(db: Datalayer):
    # identifier, weight_shape, encoder
    params = [
        ["linear_a", (32, 16), array(dtype="float", shape=(16,)), False],
        ["linear_a_multi", (32, 16), array(dtype="float", shape=(16,)), True],
        ["linear_b", (16, 8), array(dtype="float", shape=(8,)), False],
        ["linear_b_multi", (16, 8), array(dtype="float", shape=(8,)), True],
    ]
    for identifier, weight_shape, datatype, flatten in params:
        if flatten:
            weight = np.random.randn(weight_shape[1])
            m = ObjectModel(
                object=lambda x: list(np.outer(x, weight)),
                identifier=identifier,
                datatype=datatype,
                example=np.random.randn(weight_shape[0]),
                flatten=True,
            )
        else:
            weight = np.random.randn(*weight_shape)
            m = ObjectModel(
                object=lambda x: np.dot(x, weight),
                identifier=identifier,
                datatype=datatype,
                example=np.random.randn(weight_shape[0]),
            )
        db.add(m)


def add_listeners(db: Datalayer, collection_name="documents"):
    add_models(db)
    model = db.load("model", "linear_a")
    model.example = np.random.randn(32)
    select = db[collection_name].select()

    i_list = db.apply(
        Listener(
            identifier='vector-x',
            select=select,
            key="x",
            model=model,
        )
    )

    c_list = db.apply(
        Listener(
            identifier='vector-y',
            select=select,
            key="z",
            model=model,
        )
    )

    model = db.load("model", "linear_a_multi")

    i_list_flat = db.apply(
        Listener(
            identifier='vector-x-flat',
            select=select,
            key="x",
            model=model,
        )
    )

    return i_list, c_list, i_list_flat


def add_vector_index(
    db: Datalayer,
    identifier="test_vector_search",
):
    try:
        i_list = db.load("listener", "vector-x")
        c_list = db.load("listener", "vector-y")
    except FileNotFoundError:
        i_list, c_list, _ = add_listeners(db)

        db.apply(i_list)
        db.apply(c_list)

    vi = VectorIndex(
        identifier=identifier,
        indexing_listener=i_list,
        compatible_listener=c_list,
    )

    db.apply(vi)
