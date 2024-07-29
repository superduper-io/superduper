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
    train_data = QueryDataset(
        db=db,
        mapping=Mapping("_base", signature="singleton"),
        select=db["documents"].select().outputs("vector-x"),
        fold="train",
    )
    r = train_data[0]
    assert r["_fold"] == "train"
    # TODO: Enable this test when select with fields is supported
    # assert "_id" not in r
    # assert "y" not in r

    assert r['_outputs__vector-x'].shape[0] == 16

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
