import pytest

from superduper.base.base import Base
from superduper.components.dataset import Dataset


class documents(Base):
    x: int
    y: list


@pytest.mark.parametrize("pin", [True, False])
def test_dataset_pin(db, pin):

    db.create(documents)

    datas = [{"x": i, "y": [1, 2, 3]} for i in range(10)]

    db["documents"].insert(datas)

    select = db["documents"].select()

    d = Dataset(
        identifier="test_dataset",
        select=select,
        pin=pin,
        db=db,
    )
    db.apply(d)
    assert db.show("Dataset") == ["test_dataset"]

    new_datas = [{"x": i, "y": [1, 2, 3]} for i in range(10, 20)]
    db["documents"].insert(new_datas)
    dataset: Dataset = db.load("Dataset", "test_dataset")
    dataset.setup()
    if pin:
        len(dataset.data) == 10
    else:
        len(dataset.data) == 20
