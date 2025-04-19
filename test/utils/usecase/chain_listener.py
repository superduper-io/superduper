import typing as t

from superduper import ObjectModel
from superduper.base.base import Base
from superduper.components.listener import Listener

if t.TYPE_CHECKING:
    from superduper.base.datalayer import Datalayer


class documents(Base):
    x: int


def build_chain_listener(db: "Datalayer"):

    db.create(documents)

    data = [
        {"x": 1},
        {"x": 2},
        {"x": 3},
    ]

    db["documents"].insert(data)

    model_a = ObjectModel(identifier="a", object=lambda x: f"{x}->a")

    model_b = ObjectModel(identifier="b", object=lambda x: f"{x}->b")

    model_c = ObjectModel(identifier="c", object=lambda x: f"{x}->c")

    listener_a = Listener(
        model=model_a,
        select=db["documents"].select(),
        key="x",
        identifier="a",
    )
    db.apply(listener_a)

    listener_b = Listener(
        model=model_b,
        select=db[listener_a.outputs].select(),
        key=listener_a.outputs,
        identifier="b",
        upstream=[listener_a],
    )
    db.apply(listener_b)

    listener_c = Listener(
        model=model_c,
        select=db[listener_b.outputs].select(),
        key=listener_b.outputs,
        identifier="c",
        upstream=[listener_a, listener_b],
    )
    db.apply(listener_c)

    data = [
        {"x": 4},
        {"x": 5},
        {"x": 6},
    ]

    db["documents"].insert(data)

    assert db.databackend.check_output_dest(listener_a.predict_id)
    assert db.databackend.check_output_dest(listener_b.predict_id)
    assert db.databackend.check_output_dest(listener_c.predict_id)

    assert len(list(db[listener_a.outputs].select().execute())) == 6
    assert len(list(db[listener_b.outputs].select().execute())) == 6
    assert len(list(db[listener_c.outputs].select().execute())) == 6
