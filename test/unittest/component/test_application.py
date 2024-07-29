import typing as t
from pprint import pprint

import numpy as np

from superduper import ObjectModel, Schema, superduper
from superduper.base.document import Document
from superduper.components.datatype import pickle_encoder
from superduper.components.table import Table

if t.TYPE_CHECKING:
    from superduper.base.datalayer import Datalayer


def test_wrap_as_application_from_db(db: "Datalayer"):
    data = {"x": 1, "y": 2, "z": np.array([1, 2, 3])}
    db.cfg.auto_schema = True

    schema = Schema(
        identifier="schema",
        fields={
            "x": int,
            "y": int,
            "z": pickle_encoder,
        },
    )

    table = Table("documents", schema=schema)

    db.apply(table)

    db["documents"].insert([data]).execute()

    model1 = ObjectModel(identifier="model1", object=lambda x: x + 1)

    listener1 = model1.to_listener(
        key="x",
        select=db["documents"].select(),
        identifier="listener1",
    )

    model2 = ObjectModel(identifier="model2", object=lambda y: y + 2)

    listener2 = model2.to_listener(
        key="y", select=db["documents"].select(), identifier="listener2"
    )

    model3 = ObjectModel(
        identifier="model3", object=lambda z: z * 3, datatype=pickle_encoder
    )

    listener3 = model3.to_listener(
        key="z", select=db["documents"].select(), identifier="listener3"
    )

    db.apply(listener1)
    db.apply(listener2)
    db.apply(listener3)

    pprint(db.show())

    from superduper.components.application import Application

    app = Application.build_from_db(identifier="app", db=db)

    assert app.identifier == "app"

    assert {c.identifier for c in app.components} == {
        "listener1",
        "listener2",
        "listener3",
        table.identifier,
    }

    db = superduper()
    db.cfg.auto_schema = True

    db["documents"].insert([data]).execute()

    assert "_outputs" not in list(db["documents"].select().execute())[0]
    db.apply(app)

    data = list(db["documents"].select().execute())[0]
    listener1.outputs_select

    def get_listener_output(listener):
        return Document(list(listener.outputs_select.execute())[0].unpack())[
            listener.outputs_key
        ]

    assert get_listener_output(listener1) == 2
    assert get_listener_output(listener2) == 4
    assert np.allclose(get_listener_output(listener3), data["z"] * 3)
