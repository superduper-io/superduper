import pytest

from superduper.backends.query_dataset import QueryDataset
from superduper.components.model import Mapping

try:
    import torch
except ImportError:
    torch = None


def test_query_dataset(db):
    from test.utils.setup.fake_data import add_models, add_random_data, add_vector_index

    db.cfg.auto_schema = True
    add_random_data(db)
    add_models(db)
    add_vector_index(db)
    primary_id = db["documents"].primary_id

    listener_uuid = db.show('listener', 'vector-x', -1)['uuid']

    train_data = QueryDataset(
        db=db,
        mapping=Mapping("_base", signature="singleton"),
        select=db["documents"].select(primary_id, 'x', '_fold').outputs("vector-x__" + listener_uuid),
        fold="train",
    )
    r = train_data[0]
    assert r["_fold"] == "train"
    assert "y" not in r
    assert "x" in r

    db["documents"].select(primary_id, 'x', '_fold').outputs("vector-x__" + listener_uuid)
    assert r['_outputs__vector-x__' + listener_uuid].shape[0] == 16

    train_data = QueryDataset(
        db=db,
        select=db["documents"].select(),
        mapping=Mapping({"x": "x", "y": "y"}, signature="**kwargs"),
        fold="train",
    )

    r = train_data[0]
    assert "_id" not in r
    assert set(r.keys()) == {"x", "y"}


@pytest.mark.skipif(not torch, reason="Torch not installed")
def test_query_dataset_base(db):
    from test.utils.setup.fake_data import add_random_data

    add_random_data(db)
    train_data = QueryDataset(
        db=db,
        select=db["documents"].select(),
        mapping=Mapping(["_base", "y"], signature="*args"),
        fold="train",
    )
    r = train_data[0]
    assert isinstance(r, tuple)
    assert len(r) == 2
