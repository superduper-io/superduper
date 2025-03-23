import dataclasses as dc
import typing as t  # noqa: F401
from test.utils.component import model as model_utils
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from sklearn.metrics import accuracy_score, f1_score

from superduper.backends.base.data_backend import BaseDataBackend
from superduper.base.datalayer import Datalayer
from superduper.base.datatype import pickle_serializer
from superduper.base.document import Document
from superduper.base.query import Query
from superduper.components.dataset import Dataset
from superduper.components.metric import Metric
from superduper.components.model import (
    Model,
    ObjectModel,
    QueryModel,
    SequentialModel,
    Trainer,
    Validation,
)
from superduper.misc import typing as st  # noqa: F401


# ------------------------------------------
# Test the _TrainingConfiguration class (tc)
# ------------------------------------------
@dc.dataclass
class Validator(ObjectModel): ...


# --------------------------------
# Test the _Predictor class (pm)
# --------------------------------


def return_self(x):
    return x


def return_self_multikey(x, y, z):
    return [x, y, z]


def to_call(x):
    return x * 5


def to_call_multi(x, y):
    return x


@pytest.fixture
def predict_mixin() -> Model:
    predict_mixin = ObjectModel('test', object=to_call)
    predict_mixin.version = 0
    return predict_mixin


@pytest.fixture
def predict_mixin_multikey() -> Model:
    predict_mixin = ObjectModel('test', object=to_call_multi)
    predict_mixin.version = 0
    return predict_mixin


def test_pm_predict(predict_mixin):
    X = np.random.randn(5)
    expect = to_call(X)
    assert np.allclose(predict_mixin.predict(X), expect)


def test_predict_core_multikey(predict_mixin_multikey):
    X = 1
    Y = 2
    expect = to_call_multi(X, Y)
    output = predict_mixin_multikey.predict(X, Y)
    assert output == expect

    output = predict_mixin_multikey.predict(x=X, y=Y)
    assert output == expect

    with pytest.raises(TypeError):
        predict_mixin_multikey.predict_batches(X, Y)

    output = predict_mixin_multikey.predict_batches([(X, Y), (X, Y)])
    assert isinstance(output, list)

    predict_mixin_multikey.num_workers = 2
    output = predict_mixin_multikey.predict_batches([(X, Y), (X, Y)])
    assert isinstance(output, list)


def test_pm_core_predict(predict_mixin):
    # make sure predict is called
    with patch.object(predict_mixin, 'predict', return_self):
        assert predict_mixin.predict(5) == return_self(5)


@pytest.mark.skip
def test_pm_predict_with_select_ids(monkeypatch, predict_mixin):
    xs = [np.random.randn(4) for _ in range(10)]

    docs = [Document({'x': x}) for x in xs]
    X = 'x'

    ids = [i for i in range(10)]

    select = MagicMock(spec=Query)
    db = MagicMock(spec=Datalayer)
    db.databackend = MagicMock(spec=BaseDataBackend)
    db.execute.return_value = docs
    predict_mixin.db = db

    with patch.object(predict_mixin, 'object') as my_object:
        my_object.return_value = 2
        # Check the base predict function
        predict_mixin.db = db
        with patch.object(select, 'select_using_ids') as select_using_ids:
            predict_mixin._predict_with_select_and_ids(
                X=X, select=select, ids=ids, predict_id='test'
            )
            select_using_ids.assert_called_once_with(ids)

    with (
        patch.object(predict_mixin, 'object') as my_object,
        patch.object(select, 'select_using_ids') as select_using_id,
    ):
        my_object.return_value = 2

        monkeypatch.setattr(
            predict_mixin,
            'datatype',
            pickle_serializer,
        )
        predict_mixin._predict_with_select_and_ids(
            X=X, select=select, ids=ids, predict_id='test'
        )
        select_using_id.assert_called_once_with(ids)

    with patch.object(predict_mixin, 'object') as my_object:
        my_object.return_value = {'out': 2}
        # Check the base predict function with output_schema
        from superduper.base.schema import Schema

        predict_mixin.datatype = None
        predict_mixin.output_schema = schema = MagicMock(spec=Schema)
        predict_mixin.db = db
        schema.side_effect = str
        with patch.object(select, 'select_using_ids') as select_using_ids:
            predict_mixin._predict_with_select_and_ids(
                X=X, select=select, ids=ids, predict_id='test'
            )
            select_using_ids.assert_called_once_with(ids)


def test_model_append_metrics():
    @dc.dataclass
    class _Tmp(ObjectModel): ...

    class MyTrainer(Trainer):
        def fit(self, *args, **kwargs): ...

    model = _Tmp(
        'test',
        object=object(),
        validation=Validation('test', key=('x', 'y')),
        trainer=MyTrainer('test', key='x', select='1'),
    )

    metric_values = {'acc': 0.5, 'loss': 0.5}

    model.append_metrics(metric_values)

    assert model.trainer.metric_values.get('acc') == [0.5]
    assert model.trainer.metric_values.get('loss') == [0.5]

    metric_values = {'acc': 0.6, 'loss': 0.4}
    model.append_metrics(metric_values)
    assert model.trainer.metric_values.get('acc') == [0.5, 0.6]
    assert model.trainer.metric_values.get('loss') == [0.5, 0.4]


def test_model_validate():
    # Check the metadadata recieves the correct values
    model = Validator('test', object=lambda x: x)
    model._signature = 'singleton'
    my_metric = MagicMock(spec=Metric)
    my_metric.identifier = 'acc'
    my_metric.return_value = 0.5
    dataset = MagicMock(spec=Dataset)
    dataset.data = [{'X': 1, 'y': 1} for _ in range(4)]

    def acc(x, y):
        return sum([xx == yy for xx, yy in zip(x, y)]) / len(x)

    with patch.object(model, 'predict_batches') as mock_predict:
        mock_predict.return_value = [1, 2, 1, 1]
        returned = model.validate(
            ('X', 'y'), dataset=dataset, metrics=[Metric('acc', object=acc)]
        )

    assert returned == {'acc': 0.75}


def test_model_validate_in_db(db):
    # Check the validation is done correctly
    from test.utils.setup.fake_data import add_random_data, get_valid_dataset

    add_random_data(db)
    valid_dataset = get_valid_dataset(db)

    model_predict = ObjectModel(
        identifier='test',
        object=lambda x: sum(x) > 0.5,
        datatype='str',
        validation=Validation(
            identifier='my-valid',
            metrics=[
                Metric('f1', object=f1_score),
                Metric('acc', object=accuracy_score),
            ],
            datasets=[valid_dataset],
            key=('x', 'y'),
        ),
    )

    db.apply(model_predict)
    reloaded = db.load('ObjectModel', model_predict.identifier)
    assert next(iter(reloaded.metric_values.keys())) == 'my_valid/0'


class MyTrainer(Trainer):
    def fit(self, *args, **kwargs):
        return


def test_model_create_fit_job(db):
    db.cfg.auto_schema = True
    from test.utils.setup.fake_data import add_random_data

    add_random_data(db)
    # Check the fit job is created correctly
    model = Validator('test', object=lambda x: x)
    # TODO move these parameters into the `Trainer` (same thing for validation)
    model.trainer = MyTrainer('test', select=db['documents'].select(), key='x')
    db.apply(model)
    model.db = db
    job = model.fit_in_db(job=True)
    assert job.identifier == model.identifier
    assert job.method == 'fit_in_db'


def test_model_fit(db):
    # Check the logic of the fit method, the mock method was tested above
    from test.utils.setup.fake_data import add_random_data

    add_random_data(db)

    class MyTrainer(Trainer):
        def fit(self, *args, **kwargs):
            return

    from superduper.components.dataset import Dataset

    valid_dataset = Dataset(identifier='test', select=db['documents'].select(), db=db)

    model = Validator(
        'test',
        object=object(),
        trainer=MyTrainer(
            'my-trainer', key='x', select=db['documents'].select(), db=db
        ),
        validation=Validation(
            'my-valid',
            datasets=[valid_dataset],
            metrics=[Metric('metric', object=lambda x, y: True)],
            key=('x', 'y'),
            db=db,
        ),
        db=db,
    )

    with patch.object(model, 'fit'):
        model.fit_in_db()
        model.fit.assert_called_once()

    with patch.object(model, 'validate'):
        with patch.object(model.db, 'apply'):
            model.validate_in_db()
            model.validate.assert_called_once()
            model.db.apply.assert_called_once()


@pytest.mark.skip
def test_query_model(db):
    from test.utils.setup.fake_data import add_models, add_random_data, add_vector_index

    db.cfg.auto_schema = True

    add_random_data(db)
    add_models(db)
    add_vector_index(db)
    q = (
        db['documents']
        .like(
            {'x': '<var:X>'},
            vector_index='test_vector_search',
            n=3,
        )
        .select()
    )

    check = q.set_variables(db=db, X='test')
    assert not check.variables

    primary_id = db['documents'].primary_id

    def postprocess(r):
        return list(r)[0][primary_id]

    m = QueryModel(
        identifier='test-query-model',
        select=q,
        postprocess=postprocess,
    )
    m.db = db

    out = m.predict(X=np.random.randn(32))

    out = m.predict_batches([{'X': np.random.randn(32)} for _ in range(4)])

    assert len(out) == 4

    db.apply(m)

    n = db.load('model', m.identifier)
    assert set(n.select.variables) == set(q.variables)


def test_sequential_model():
    m = SequentialModel(
        identifier='test-sequential-model',
        models=[
            ObjectModel(
                identifier='test-predictor-1',
                object=lambda x: x + 2,
            ),
            ObjectModel(
                identifier='test-predictor-2',
                object=lambda x: x + 1,
            ),
        ],
    )

    assert m.predict(1) == 4
    assert m.predict_batches([1 for _ in range(4)]) == [4, 4, 4, 4]


import numpy


def make_object_model(shape):
    return ObjectModel(
        'test', object=lambda x: numpy.array(x) + 1, datatype=f'array[float:{shape}]'
    )


def test_object_model_predict():

    object_model = make_object_model('10x10')

    sample_data = np.zeros((10, 10))
    result, results = model_utils.test_predict(object_model, sample_data)

    assert np.allclose(result, sample_data + 1)
    assert all(np.allclose(r, sample_data + 1) for r in results)


def test_object_model_as_a_listener(db):

    object_model = make_object_model('10x10')
    sample_data = np.zeros((10, 10))
    results = model_utils.test_model_as_a_listener(
        object_model, sample_data, db, type='array[float:10x10]'
    )
    r = results[0].unpack()
    key = next(k for k in r.keys() if k.startswith('_outputs__test'))
    assert all(np.allclose(r.unpack()[key], sample_data + 1) for r in results)


class MyModelSingleton(Model):
    def predict(self, X):
        return X


class MyModelArgs(Model):
    def predict(self, X, Y):
        return X


class MyModelArgsKwargs(Model):
    def predict(self, X, Y=None):
        return X


class MyModelKwargs(Model):
    def predict(self, Y=None):
        return Y


def test_signature_inference():
    assert MyModelSingleton._signature == 'singleton'
    assert MyModelArgs._signature == '*args'
    assert MyModelArgsKwargs._signature == '*args,**kwargs'
    assert MyModelKwargs._signature == '**kwargs'

    m = ObjectModel('test', object=lambda x, y: x + y)

    assert m.signature == '*args'
