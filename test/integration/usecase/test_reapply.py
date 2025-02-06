import numpy

from superduper import Model
from superduper.components.dataset import Data
from superduper.components.model import model
from superduper.components.schema import Schema
from superduper.components.table import Table
from superduper.components.template import Template
from superduper.components.datatype import pickle_encoder


class MyModel(Model):
    breaks = ('b',)
    signature: str = 'singleton'
    a: str
    b: str

    def predict(self, x):
        return x + 1


def test_reapply(db):
    db.cfg.auto_schema = True

    db['docs'].insert([{'x': i} for i in range(10)]).execute()

    def build(name, data):
        model = MyModel('test', a=name, b=data)
        listener_1 = model.to_listener(
            key='x',
            select=db['docs'].select(),
            identifier='listener_1',
        )
        return model.to_listener(
            key=listener_1.outputs,
            select=db[listener_1.outputs].select(),
            identifier='listener_2',
            upstream=[listener_1],
        )

    listener_2 = build('first', '1')

    db.apply(listener_2)

    outputs = db[listener_2.outputs].select().tolist()

    import pprint

    pprint.pprint(outputs)

    assert outputs[0][listener_2.outputs] == 2

    listener_2 = build('second', '1')
    db.apply(listener_2)

    listener_2_update = build('second', '2')

    db.apply(listener_2_update)


def test_template_component_deps(db):

    @model
    def test(x):
        return x + 1

    test.datatype = pickle_encoder

    t = Template(
        template=test,
        identifier='test_template',
        default_tables=[
            Table(
                'test_table',
                schema=Schema('test_schema', fields={'x': 'str', 'y': pickle_encoder}),
                data=Data('test_data', raw_data=[{'x': '1', 'y': numpy.random.randn(3)}])
            )
        ]
    )

    db.apply(t, force=True)

    m = t()

    db.apply(m, force=True)

    db.remove('model', 'test', recursive=True, force=True)

    m = t()

    db.apply(m, force=True)

    db.remove('model', 'test', recursive=True, force=True)