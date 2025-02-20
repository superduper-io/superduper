import typing as t

from superduper import Model
from superduper.base.base import Base
from superduper.components.listener import Listener
from superduper.misc import typing as st  # noqa: F401


class MyModel(Model):
    breaks = ('b',)
    signature: str = 'singleton'
    a: str
    b: str

    def predict(self, x):
        return x + 1


class docs(Base):
    x: int


def test_reapply(db):
    db.create(docs)

    db['docs'].insert([{'x': i} for i in range(10)])

    def build(name, data):
        model = MyModel('test', a=name, b=data, datatype='int')
        listener_1 = Listener(
            model=model,
            key='x',
            select=db['docs'].select(),
            identifier='listener_1',
        )
        return Listener(
            model=model,
            key=listener_1.outputs,
            select=db[listener_1.outputs].select(),
            identifier='listener_2',
            upstream=[listener_1],
        )

    listener_2 = build('first', '1')

    db.apply(listener_2)

    outputs = db[listener_2.outputs].select().execute()

    import pprint

    pprint.pprint(outputs)

    assert outputs[0][listener_2.outputs] == 2

    listener_2 = build('second', '1')
    db.apply(listener_2)

    listener_2_update = build('second', '2')

    db.apply(listener_2_update)
