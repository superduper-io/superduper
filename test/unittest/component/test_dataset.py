import pytest

from superduper.components.dataset import DataInit, Dataset


@pytest.mark.parametrize("pin", [True, False])
def test_dataset_pin(db, pin):
    db.cfg.auto_schema = True

    datas = [{"x": i, "y": [1, 2, 3]} for i in range(10)]

    db["documents"].insert(datas).execute()

    select = db["documents"].select()

    d = Dataset(
        identifier="test_dataset",
        select=select,
        pin=pin,
    )
    db.apply(d)
    assert db.show("dataset") == ["test_dataset"]

    new_datas = [{"x": i, "y": [1, 2, 3]} for i in range(10, 20)]
    db["documents"].insert(new_datas).execute()
    dataset: Dataset = db.load("dataset", "test_dataset")
    dataset.init(db)
    if pin:
        len(dataset.data) == 10
    else:
        len(dataset.data) == 20


def test_init_data(db):
    db.cfg.auto_schema = True
    data = [{"x": i, "y": [1, 2, 3]} for i in range(10)]
    data_init = DataInit(data=data, table="documents", identifier="test_data_init")

    db.apply(data_init)

    data = list(db["documents"].select().execute())
    assert len(data) == 10
    for i, d in enumerate(data):
        assert d["x"] == i
        assert d["y"] == [1, 2, 3]
