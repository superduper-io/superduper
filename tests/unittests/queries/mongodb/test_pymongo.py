# ruff: noqa: F401, F811
from superduperdb.queries.mongodb.queries import Collection

from tests.fixtures.collection import (
    empty,
    float_tensors_16,
    float_tensors_32,
    random_data_factory,
    random_data,
    a_watcher,
    a_model,
    a_watcher_base,
    a_model_base,
)


def test_find(random_data):
    r = random_data.execute(Collection(name='documents').find_one())
    print(r['x'].x.shape)

    cursor = random_data.execute(Collection(name='documents').find())
    print(next(cursor))
