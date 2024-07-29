import ibis

ibis.options.interactive = True

import numpy as np
import pytest

from superduper import Document
from superduper.components.model import ObjectModel


def assert_output_is_correct(data, output):
    if isinstance(data, np.ndarray):
        assert np.allclose(output, data)
    else:
        assert output == data


@pytest.mark.parametrize(
    "data",
    [
        2,
        np.array([[1, 1, 1], [1, 1, 1]]),
    ],
)
@pytest.mark.parametrize("flatten", [True, False])
def test_downstream_task_workflows_are_triggered(db, data, flatten):
    db.cfg.auto_schema = True

    db.execute(db["test"].insert([{"x": 10}]))

    upstream_model = ObjectModel(
        "m1",
        object=lambda x: data * x if not flatten else [data * x] * 10,
        flatten=flatten,
    )

    upstream_listener = upstream_model.to_listener(
        key="x",
        select=db["test"].select(),
        uuid="upstream",
    )

    db.apply(upstream_listener)

    downstream_model = ObjectModel(
        "m2",
        object=lambda x: x / 2,
    )

    downstream_listener = downstream_model.to_listener(
        key=upstream_listener.outputs_key,
        select=upstream_listener.outputs_select,
        uuid="downstream",
    )

    db.apply(downstream_listener)

    outputs1 = [
        Document(d.unpack())[upstream_listener.outputs_key]
        for d in db.execute(upstream_listener.outputs_select)
    ]

    outputs2 = [
        Document(d.unpack())[downstream_listener.outputs_key]
        for d in db.execute(downstream_listener.outputs_select)
    ]

    assert len(outputs1) == 1 if not flatten else 10
    assert len(outputs2) == 1 if not flatten else 10

    assert_output_is_correct(data * 10, outputs1[0])
    assert_output_is_correct(data * 10 / 2, outputs2[0])

    db.execute(db["test"].insert([{"x": 20}]))

    # Check that the listeners are triggered when data is inserted later
    outputs1 = [
        Document(d.unpack())[upstream_listener.outputs_key]
        for d in db.execute(upstream_listener.outputs_select)
    ]

    outputs2 = [
        Document(d.unpack())[downstream_listener.outputs_key]
        for d in db.execute(downstream_listener.outputs_select)
    ]

    assert len(outputs1) == 2 if not flatten else 20
    assert len(outputs2) == 2 if not flatten else 20

    assert_output_is_correct(data * 20, outputs1[-1])
    assert_output_is_correct(data * 20 / 2, outputs2[-1])
