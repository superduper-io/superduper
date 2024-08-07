import typing as t

from superduper import ObjectModel

if t.TYPE_CHECKING:
    from superduper.base.datalayer import Datalayer


def build_chain_listener(db: "Datalayer"):
    db.cfg.auto_schema = True
    data = [
        {"x": 1},
        {"x": 2},
        {"x": 3},
    ]

    db["documents"].insert(data).execute()

    data = list(db["documents"].select().execute(eager_mode=True))[0]

    model_a = ObjectModel(identifier="a", object=lambda x: f"{x}->a")

    model_b = ObjectModel(identifier="b", object=lambda x: f"{x}->b")

    model_c = ObjectModel(identifier="c", object=lambda x: f"{x}->c")

    listener_a = model_a.to_listener(
        select=db["documents"].select(),
        key="x",
        uuid="a",
        identifier="a",
    )
    db.apply(listener_a)

    listener_b = model_b.to_listener(
        select=listener_a.outputs_select,
        key=listener_a.outputs,
        uuid="b",
        identifier="b",
    )
    db.apply(listener_b)

    listener_c = model_c.to_listener(
        select=listener_b.outputs_select,
        key=listener_b.outputs,
        uuid="c",
        identifier="c",
    )
    db.apply(listener_c)

    data = [
        {"x": 4},
        {"x": 5},
        {"x": 6},
    ]

    db["documents"].insert(data).execute()

    assert db.databackend.check_output_dest('a')
    assert db.databackend.check_output_dest('b')
    assert db.databackend.check_output_dest('c')
