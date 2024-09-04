import typing as t

import numpy as np

if t.TYPE_CHECKING:
    from superduper.base.datalayer import Datalayer

VECTOR_SIZE = 300


def add_data(db: "Datalayer", start: int, end: int):
    data = []
    for i in range(start, end):
        data.append(
            {
                "x": i,
                "label": int(i % 2 == 0),
            }
        )
    db["documents"].insert(data).execute()


def build_vector_index(db: "Datalayer"):
    from superduper import ObjectModel, VectorIndex

    db.cfg.auto_schema = True

    add_data(db, 0, 100)

    def predict(x):
        vector = [0] * VECTOR_SIZE
        for offset in range(5):
            if offset + x < VECTOR_SIZE:
                vector[offset + x] = 1

            if x - offset >= 0:
                vector[x - offset] = 1

        return np.array(vector)

    indexing_model = ObjectModel(
        identifier="model",
        object=predict,
    )

    indexing_model = indexing_model.to_listener(
        key="x",
        select=db["documents"].select(),
        identifier="vector",
    )

    compatible_model = ObjectModel(
        identifier="model-y",
        object=lambda y: predict(-y),
    )

    compatible_listener = compatible_model.to_listener(
        key="y",
        select=None,
        identifier="compatible",
    )

    vector_index = VectorIndex(
        identifier="vector_index",
        indexing_listener=indexing_model,
        compatible_listener=compatible_listener,
    )

    db.apply(vector_index)
