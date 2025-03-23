import random
import typing as t

import numpy

from superduper import Listener, Model, VectorIndex, logging
from superduper.base import Base
from superduper.base.annotations import trigger
from superduper.base.datalayer import Datalayer
from superduper.components.model import Validation


class MyModel(Model):
    breaks: t.ClassVar = ('a', 'b')

    a: int = 1
    b: int = 2

    def predict(self, x):
        return numpy.array([x + self.a * self.b + 2 for _ in range(20)])

    @trigger('apply', requires='validation')
    def validate_in_db(self):
        out = self.validation.validate(self)
        self.metric_values = [out]
        self.db.apply(self, jobs=False, force=True)


class MyValidation(Validation):
    def validate(self, model):
        logging.info('Running my-validation')
        return 0.1


class docs(Base):
    x: int


def test(db: Datalayer):

    db.create(docs)

    db['docs'].insert([{'x': random.randrange(10)} for _ in range(10)])

    def build_vi(**kwargs):
        model = MyModel('my-model', datatype='vector[int:20]', **kwargs)

        listener = Listener(
            'my-listener',
            model=model,
            select=db['docs'].select(),
            key='x',
        )

        return VectorIndex(
            'my-vector-index',
            indexing_listener=listener,
        )

    db.apply(build_vi())
    db.apply(build_vi())

    # TODO for some reason the model is "updating" even after no changes.

    assert db.show('VectorIndex', 'my-vector-index') == [0]
    assert db.show('Listener', 'my-listener') == [0]
    assert db.show('MyModel', 'my-model') == [0]

    component = build_vi()
    t = db['docs']
    q = t.filter(t['x'] > 3)

    component.indexing_listener.select = q

    db.apply(component)

    assert db.show('Listener', 'my-listener') == [0, 1]
    assert db.show('VectorIndex', 'my-vector-index') == [0, 1]

    component.indexing_listener.model.validation = MyValidation(
        'test-validate', key='x', datasets=[], metrics=[]
    )

    db.apply(component)

    assert db.show('MyModel', 'my-model') == [0]

    m = db.load('MyModel', 'my-model')

    assert m.metric_values == [0.1]
