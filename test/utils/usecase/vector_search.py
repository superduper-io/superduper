import typing as t

import numpy as np

from superduper.base import Base
from superduper.components.listener import Listener

if t.TYPE_CHECKING:
    from superduper.base.datalayer import Datalayer

VECTOR_SIZE = 300


class documents(Base):
    x: int
    label: int


def add_data(db: "Datalayer", start: int, end: int):
    """
    :param db: Datalayer
    :param start: int to start assigning to `x` column
    :param end: int to stop assigning to `x` column
    """
    db.insert([documents(x=i, label=int(i % 2 == 0)) for i in range(start, end)])


def build_vector_index(
    db: "Datalayer",
    n: int = 100,
    list_embeddings=False,
    measure=None,
):
    from superduper import ObjectModel, VectorIndex

    add_data(db, 0, n)

    def predict(x):
        vector = [0] * VECTOR_SIZE
        for offset in range(5):
            if offset + x < VECTOR_SIZE:
                vector[offset + x] = 1

            if x - offset >= 0:
                vector[x - offset] = 1

        if not list_embeddings:
            return np.array(vector)
        return vector

    indexing_model = ObjectModel(
        identifier="model", object=predict, datatype=f'vector[int:{VECTOR_SIZE}]'
    )

    indexing_listener = Listener(
        model=indexing_model,
        key="x",
        select=db["documents"].select(),
        identifier="vector",
    )

    compatible_model = ObjectModel(
        identifier="model-y",
        object=lambda y: predict(-y),
    )

    compatible_listener = Listener(
        model=compatible_model,
        key="y",
        select=None,
        identifier="compatible",
    )

    vector_index = VectorIndex(
        identifier="vector_index",
        indexing_listener=indexing_listener,
        compatible_listener=compatible_listener,
    )
    if measure:
        vector_index.measure = measure

    db.apply(vector_index)
