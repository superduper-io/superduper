import numpy as np
import pytest

from superduper import Document
from superduper.base.base import Base
from superduper.components.listener import Listener
from superduper.components.model import ObjectModel


def assert_output_is_correct(data, output):
    if isinstance(data, np.ndarray):
        assert np.allclose(output, data)
    else:
        assert output == data


class test(Base):
    x: int


@pytest.mark.parametrize(
    "data",
    [
        (2, 'int', 'json'),
        (np.array([[1, 1, 1], [1, 1, 1]]), 'pickleencoder', 'pickleencoder'),
    ],
)
@pytest.mark.parametrize("flatten", [True, False])
def test_downstream_task_workflows_are_triggered(db, data, flatten):

    db.create(test)

    data, datatype, datatype_flat = data

    db["test"].insert([{"x": 10}])

    upstream_model = ObjectModel(
        "m1",
        object=lambda x: data * x if not flatten else [data * x] * 10,
        datatype=datatype if not flatten else datatype_flat,
    )

    upstream_listener = Listener(
        model=upstream_model,
        key="x",
        select=db["test"].select(),
        identifier="upstream",
        flatten=flatten,
    )

    db.apply(upstream_listener)

    downstream_model = ObjectModel(
        "m2",
        object=lambda x: x / 2,
        datatype=datatype if not flatten else datatype_flat,
    )

    downstream_listener = Listener(
        model=downstream_model,
        key=upstream_listener.outputs,
        select=db[upstream_listener.outputs].select(),
        identifier="downstream",
        upstream=[upstream_listener],
    )

    db.apply(downstream_listener)

    outputs1 = db[upstream_listener.outputs].select().execute()
    outputs1 = [r[upstream_listener.outputs] for r in outputs1]

    outputs2 = db[downstream_listener.outputs].select().execute()
    outputs2 = [r[downstream_listener.outputs] for r in outputs2]

    assert len(outputs1) == 1 if not flatten else 10
    assert len(outputs2) == 1 if not flatten else 10

    assert_output_is_correct(data * 10, outputs1[0])
    assert_output_is_correct(data * 10 / 2, outputs2[0])

    db["test"].insert([{"x": 20}])

    # Check that the listeners are triggered when data is inserted later
    outputs1 = [
        Document(d.unpack())[upstream_listener.outputs]
        for d in db[upstream_listener.outputs].select().execute()
    ]

    outputs2 = [
        Document(d.unpack())[downstream_listener.outputs]
        for d in db[downstream_listener.outputs].select().execute()
    ]

    assert len(outputs1) == 2 if not flatten else 20
    assert len(outputs2) == 2 if not flatten else 20

    assert_output_is_correct(data * 20, outputs1[-1])
    assert_output_is_correct(data * 20 / 2, outputs2[-1])
