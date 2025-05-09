import random

import numpy as np

from superduper.base import exceptions
from superduper.base.datalayer import Datalayer
from superduper.base.datatype import Array

# ruff: noqa: E402
from superduper.components.dataset import Dataset
from superduper.components.listener import Listener
from superduper.components.model import ObjectModel
from superduper.components.table import Table
from superduper.components.vector_index import VectorIndex

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
    fields = {
        "id": "str",
        "x": "array[float:32]",
        "y": "int",
        "z": "array[float:32]",
        "_fold": "str",
    }
    t = Table(identifier=table_name, fields=fields)
    db.apply(t)
    data = []
    for i in range(n):
        x = np.random.rand(32)
        y = int(random.random() > 0.5)
        z = np.random.rand(32)
        fold = int(random.random() > 0.5)
        fold = "valid" if fold else "train"
        data.append({"id": str(i), "x": x, "y": y, "z": z, "_fold": fold})
    db[table_name].insert(data)


def add_datatypes(db: Datalayer):
    for n in [8, 16, 32]:
        db.apply(Array(dtype="float", shape=(n,)))


def add_models(db: Datalayer):

    m1 = ObjectModel(
        object=lambda x: np.dot(x, np.random.randn(32, 16)),
        identifier="linear_a",
        datatype="vector[float:16]",
    )

    m2 = ObjectModel(
        object=lambda x: np.outer(x, np.random.randn(16)),
        identifier="linear_a_multi",
        datatype="vector[float:16]",
    )

    m3 = ObjectModel(
        object=lambda x: np.dot(x, np.random.randn(16, 8)),
        identifier="linear_b",
        datatype="vector[float:8]",
    )

    m4 = ObjectModel(
        object=lambda x: np.outer(x, np.random.randn(8)),
        identifier="linear_b_multi",
        datatype="vector[float:8]",
    )

    db.apply(m1)
    db.apply(m2)
    db.apply(m3)
    db.apply(m4)

    db.show()


def add_listeners(db: Datalayer, collection_name="documents"):
    add_models(db)
    model = db.load("ObjectModel", "linear_a")
    select = db[collection_name].select()

    i_list = Listener(
        identifier='vector-x',
        select=select,
        key="x",
        model=model,
    )

    i_list = db.apply(i_list)

    c_list = db.apply(
        Listener(
            identifier='vector-y',
            select=select,
            key="z",
            model=model,
        )
    )

    model = db.load("ObjectModel", "linear_a_multi")

    i_list_flat = db.apply(
        Listener(
            identifier='vector-x-flat',
            select=select,
            key="x",
            model=model,
            flatten=True,
        )
    )

    return i_list, c_list, i_list_flat


def add_vector_index(
    db: Datalayer,
    identifier="test_vector_search",
):
    try:
        i_list = db.load("Listener", "vector-x")
        c_list = db.load("Listener", "vector-y")
    except exceptions.NotFound:
        i_list, c_list, _ = add_listeners(db)

        db.apply(i_list)
        db.apply(c_list)

    vi = VectorIndex(
        identifier=identifier,
        indexing_listener=i_list,
        compatible_listener=c_list,
    )

    db.apply(vi)
