from superduperdb.backends.mongodb.query import Collection
from superduperdb.components.listener import Listener
from superduperdb.components.model import Model


def test_listener_serializes_properly():
    q = Collection('test').find({}, {})
    l = Listener(
        model=Model('test', object=lambda x: x),
        select=q,
        key='test',
    )
    r = l.dict().encode()

    # check that the result is JSON-able
    import json

    print(json.dumps(r, indent=2))