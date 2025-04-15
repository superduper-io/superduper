import typing as t

from superduper.components.listener import Listener
from superduper.components.model import Model
from superduper.components.table import Table

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


def test_predict_in_db(model: Model, sample_data: t.Any, db: "Datalayer", type: str):
    model.identifier = random_id()

    db.apply(Table('datas', fields={'data': type, 'i': 'int'}))

    db["datas"].insert([{"data": sample_data, "i": i} for i in range(10)])

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


# TODO remove these types of general fixtures/ utils (hard to read)
def test_model_as_a_listener(
    model: Model, sample_data: t.Any, db: "Datalayer", type: str
):

    db.apply(Table('datas', fields={'data': type, 'i': 'int'}))

    db["datas"].insert([{"data": sample_data, "i": i} for i in range(10)])

    model.identifier = f'test-{random_id()}'

    listener = Listener(
        model=model,
        key="data",
        select=db["datas"].select(),
        # TODO: Fix the issue caused by excessively long table names in PostgreSQL
        identifier=f'test-{random_id()[:4]}',
    )

    db.apply(listener)

    results = list(db["_outputs__" + listener.predict_id].select().execute())
    assert len(results) == 10
    return results
