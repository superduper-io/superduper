from superduperdb.backends.mongodb.query import Collection
from superduperdb.components.listener import Listener
from superduperdb.components.model import ObjectModel


def test_listener_serializes_properly():
    q = Collection('test').find({}, {})
    listener = Listener(
        model=ObjectModel('test', object=lambda x: x),
        select=q,
        key='test',
    )
    r = listener.dict().encode()

    # check that the result is JSON-able
    import json

    print(json.dumps(r, indent=2))
