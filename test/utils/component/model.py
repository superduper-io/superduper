import typing as t

from superduper.components.model import Model

if t.TYPE_CHECKING:
    from superduper.base.datalayer import Datalayer


def test_predict(model: Model, sample_data: t.Any):
    result = model.predict(sample_data)
    results = model.predict_batches([sample_data] * 10)
    assert len(results) == 10

    return result, results


def test_predict_in_db(model: Model, sample_data: t.Any, db: "Datalayer"):
    db.cfg.auto_schema = True

    db["datas"].insert([{"data": sample_data, "i": i} for i in range(10)]).execute()

    model.predict_in_db(
        X="data",
        select=db["datas"].select(),
        db=db,
        predict_id="test",
    )

    results = list(db["_outputs__test"].select().execute())
    assert len(results) == 10

    return results


def test_model_as_a_listener(model: Model, sample_data: t.Any, db: "Datalayer"):
    db.cfg.auto_schema = True

    db["datas"].insert([{"data": sample_data, "i": i} for i in range(10)]).execute()

    listener = model.to_listener(
        key="data",
        select=db["datas"].select(),
        identifier="test",
    )

    db.apply(listener)

    results = list(db["_outputs__" + listener.predict_id].select().execute())
    assert len(results) == 10
    return results
