from test.db_config import DBConfig

import pytest

from superduperdb.components.dataset import Dataset


@pytest.mark.parametrize("db", DBConfig.EMPTY_CASES, indirect=True)
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
