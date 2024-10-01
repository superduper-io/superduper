import typing as t

from superduper.components.listener import Listener
from superduper.components.model import Model

if t.TYPE_CHECKING:
    from superduper.base.datalayer import Datalayer


import uuid


def random_id():
    return str(uuid.uuid4())


def test_predict(model: Model, sample_data: t.Any):
    result = model.predict(sample_data)
    results = model.predict_batches([sample_data] * 10)
    assert len(results) == 10

    return result, results


def test_predict_in_db(model: Model, sample_data: t.Any, db: "Datalayer"):
    model.identifier = random_id()

    db.apply(model)

    db.cfg.auto_schema = True

    db["datas"].insert([{"data": sample_data, "i": i} for i in range(10)]).execute()

    listener = Listener(
        key="data",
        model=model,
        select=db["datas"].select(),
        identifier='test',
    )

    db.apply(listener)

    results = list(db[listener.outputs].select().execute())
    assert len(results) == 10

    return results


def test_model_as_a_listener(model: Model, sample_data: t.Any, db: "Datalayer"):
    db.cfg.auto_schema = True

    db["datas"].insert([{"data": sample_data, "i": i} for i in range(10)]).execute()

    model.identifier = f'test-{random_id()}'

    listener = model.to_listener(
        key="data",
        select=db["datas"].select(),
        identifier=f'test-{random_id()}',
    )

    db.apply(listener)

    results = list(db["_outputs__" + listener.predict_id].select().execute())
    assert len(results) == 10
    return results
