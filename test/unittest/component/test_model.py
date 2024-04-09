import dataclasses as dc
from test.db_config import DBConfig
from unittest.mock import MagicMock, patch

import bson
import numpy as np
import pytest
from sklearn.metrics import accuracy_score, f1_score

from superduperdb.backends.base.query import Select
from superduperdb.backends.ibis.field_types import FieldType
from superduperdb.backends.local.compute import LocalComputeBackend
from superduperdb.backends.mongodb.query import Collection
from superduperdb.base.datalayer import Datalayer
from superduperdb.base.document import Document
from superduperdb.base.serializable import Variable
from superduperdb.components.dataset import Dataset
from superduperdb.components.datatype import DataType
from superduperdb.components.metric import Metric
from superduperdb.components.model import (
    Mapping,
    Model,
    ObjectModel,
    QueryModel,
    SequentialModel,
    Trainer,
    _Fittable,
    _Validator,
)
from superduperdb.jobs.job import ComponentJob


# ------------------------------------------
# Test the _TrainingConfiguration class (tc)
# ------------------------------------------
@dc.dataclass
class Validator(_Fittable, ObjectModel, _Validator):
    ...


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


def test_pm_predict_one(predict_mixin):
    X = np.random.randn(5)
    expect = to_call(X)
    assert np.allclose(predict_mixin.predict_one(X), expect)


def test_predict_core_multikey(predict_mixin_multikey):
    X = 1
    Y = 2
    expect = to_call_multi(X, Y)
    output = predict_mixin_multikey.predict_one(X, Y)
    assert output == expect

    output = predict_mixin_multikey.predict_one(x=X, y=Y)
    assert output == expect

    with pytest.raises(TypeError):
        predict_mixin_multikey.predict(X, Y)

    output = predict_mixin_multikey.predict([((X, Y), {}), ((X, Y), {})])
    assert isinstance(output, list)

    predict_mixin_multikey.num_workers = 2
    output = predict_mixin_multikey.predict([((X, Y), {}), ((X, Y), {})])
    assert isinstance(output, list)


def test_pm_core_predict(predict_mixin):
    # make sure _predict_one is called
    with patch.object(predict_mixin, 'predict_one', return_self):
        assert predict_mixin.predict_one(5) == return_self(5)


@patch('superduperdb.components.model.ComponentJob')
def test_pm_create_predict_job(mock_job, predict_mixin):
    mock_db = MagicMock()
    mock_select = MagicMock()
    mock_select.dict().encode.return_value = b'encoded_select'
    X = 'model_input'
    ids = ['id1', 'id2']
    max_chunk_size = 100
    in_memory = True
    overwrite = False
    predict_mixin.predict_in_db_job(
        X=X,
        db=mock_db,
        select=mock_select,
        ids=ids,
        max_chunk_size=max_chunk_size,
        predict_id='test',
    )
    mock_job.assert_called_once_with(
        component_identifier=predict_mixin.identifier,  # Adjust according to your setup
        method_name='predict_in_db',
        type_id='model',
        args=[X],
        kwargs={
            'select': b'encoded_select',
            'ids': ids,
            'predict_id': 'test',
            'max_chunk_size': max_chunk_size,
            'in_memory': in_memory,
            'overwrite': overwrite,
        },
        compute_kwargs={},
    )


def test_pm_predict(predict_mixin):
    # Check the logic of predict method, the mock method will be tested below
    db = MagicMock(spec=Datalayer)
    db.compute = MagicMock(spec=LocalComputeBackend)
    db.metadata = MagicMock()
    select = MagicMock(spec=Select)
    select.table_or_collection = MagicMock()

    with patch.object(predict_mixin, 'predict') as predict_func:
        predict_mixin.predict_in_db('x', db=db, select=select, predict_id='test')
        predict_func.assert_called_once()


def test_pm_predict_with_select_ids(monkeypatch, predict_mixin):
    xs = [np.random.randn(4) for _ in range(10)]

    docs = [Document({'x': x}) for x in xs]
    X = 'x'

    ids = [i for i in range(10)]

    select = MagicMock(spec=Select)
    db = MagicMock(spec=Datalayer)
    db.execute.return_value = docs

    with patch.object(predict_mixin, 'object') as my_object:
        my_object.return_value = 2
        # Check the base predict function
        predict_mixin.db = db
        with patch.object(select, 'select_using_ids') as select_using_ids, patch.object(
            select, 'model_update'
        ) as model_update:
            predict_mixin._predict_with_select_and_ids(
                X=X, db=db, select=select, ids=ids, predict_id='test'
            )
            select_using_ids.assert_called_once_with(ids)
            _, kwargs = model_update.call_args
            #  make sure the outputs are set
            assert kwargs.get('outputs') == [2] * 10

    with (
        patch.object(predict_mixin, 'object') as my_object,
        patch.object(select, 'select_using_ids') as select_using_id,
        patch.object(select, 'model_update') as model_update,
    ):
        my_object.return_value = 2

        monkeypatch.setattr(predict_mixin, 'datatype', DataType(identifier='test'))
        predict_mixin._predict_with_select_and_ids(
            X=X, db=db, select=select, ids=ids, predict_id='test'
        )
        select_using_id.assert_called_once_with(ids)
        _, kwargs = model_update.call_args
        datatype = predict_mixin.datatype
        assert kwargs.get('outputs') == [datatype(2).encode() for _ in range(10)]

    with patch.object(predict_mixin, 'object') as my_object:
        my_object.return_value = {'out': 2}
        # Check the base predict function with output_schema
        from superduperdb.components.schema import Schema

        predict_mixin.datatype = None
        predict_mixin.output_schema = schema = MagicMock(spec=Schema)
        schema.side_effect = str
        with patch.object(select, 'select_using_ids') as select_using_ids, patch.object(
            select, 'model_update'
        ) as model_update:
            predict_mixin._predict_with_select_and_ids(
                X=X, db=db, select=select, ids=ids, predict_id='test'
            )
            select_using_ids.assert_called_once_with(ids)
            _, kwargs = model_update.call_args
            assert kwargs.get('outputs') == [str({'out': 2}) for _ in range(10)]


def test_model_append_metrics():
    @dc.dataclass
    class _Tmp(ObjectModel, _Fittable):
        ...

    model = _Tmp('test', object=object())

    metric_values = {'acc': 0.5, 'loss': 0.5}

    model.append_metrics(metric_values)

    assert model.metric_values.get('acc') == [0.5]
    assert model.metric_values.get('loss') == [0.5]

    metric_values = {'acc': 0.6, 'loss': 0.4}
    model.append_metrics(metric_values)
    assert model.metric_values.get('acc') == [0.5, 0.6]
    assert model.metric_values.get('loss') == [0.5, 0.4]


@patch.object(Mapping, '__call__')
def test_model_validate(mock_call):
    # Check the metadadata recieves the correct values
    model = Validator('test', object=object())
    my_metric = MagicMock(spec=Metric)
    my_metric.identifier = 'acc'
    my_metric.return_value = 0.5
    mock_call.return_value = 1
    dataset = MagicMock(spec=Dataset)
    dataset.data = [None for _ in range(4)]

    def acc(x, y):
        return sum([xx == yy for xx, yy in zip(x, y)]) / len(x)

    with patch.object(model, 'predict') as mock_predict:
        mock_predict.return_value = [1, 2, 1, 1]
        returned = model.validate(
            'y', dataset=dataset, metrics=[Metric('acc', object=acc)]
        )

    assert returned == {'acc': 0.75}


@pytest.mark.parametrize(
    "db",
    [
        (DBConfig.mongodb_data, {'n_data': 500}),
        (DBConfig.sqldb_data, {'n_data': 500}),
    ],
    indirect=True,
)
def test_model_validate_in_db(valid_dataset, db):
    # Check the validation is done correctly

    model_predict = ObjectModel('test', object=lambda x: x, datatype=FieldType('str'))

    with patch.object(model_predict, 'validate') as mock_validate:
        mock_validate.return_value = {'acc': 0.7, 'f1': 0.2}

        model_predict.metrics = [
            Metric('f1', object=f1_score),
            Metric('acc', object=accuracy_score),
        ]
        model_predict.validation_sets = [valid_dataset]
        model_predict.validate_in_db(db)

        assert model_predict.metric_values == {'my_valid/0': {'acc': 0.7, 'f1': 0.2}}


class MyTrainer(Trainer):
    def fit(self, *args, **kwargs):
        return


@pytest.mark.parametrize("db", [DBConfig.mongodb_empty], indirect=True)
def test_model_create_fit_job(db):
    # Check the fit job is created correctly
    model = Validator('test', object=object())
    # TODO move these parameters into the `Trainer` (same thing for validation)
    model.train_X = 'x'
    model.train_select = Collection('test').find()
    model.trainer = MyTrainer('test')
    db.add(model)
    with patch.object(ComponentJob, '__call__') as mock_call:
        mock_call.return_value = None
    job = model.fit_in_db_job(db=db)
    assert job.component_identifier == model.identifier
    assert job.method_name == 'fit_in_db'


@pytest.mark.parametrize("db", [DBConfig.mongodb_empty], indirect=True)
def test_model_fit(db, valid_dataset):
    # Check the logic of the fit method, the mock method was tested above

    model = Validator(
        'test',
        object=object(),
        train_X='x',
        train_select=Collection('test').find(),
        validation_sets=[valid_dataset],
        metrics=[MagicMock(spec=Metric)],
    )

    with patch.object(model, 'fit'):
        model.fit_in_db(db)
        model.fit.assert_called_once()


@pytest.mark.parametrize(
    "db",
    [
        (DBConfig.mongodb, {'n_data': 500}),
    ],
    indirect=True,
)
def test_query_model(db):
    q = (
        Collection(identifier='documents')
        .like({'x': Variable('X')}, vector_index='test_vector_search', n=3)
        .find_one({}, {'_id': 1})
    )

    check = q.set_variables(db, X='test')
    assert not check.variables

    m = QueryModel(
        identifier='test-query-model',
        select=q,
        postprocess=lambda r: r['_id'],
    )
    m.db = db

    import torch

    out = m.predict_one(X=torch.randn(32))

    assert isinstance(out, bson.ObjectId)

    out = m.predict([{'X': torch.randn(32)} for _ in range(4)])

    assert len(out) == 4

    db.add(m)

    n = db.load('model', m.identifier)
    assert set(x.value for x in n.select.variables) == set(x.value for x in q.variables)


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
                signature='singleton',
            ),
        ],
    )

    assert m.predict_one(x=1) == 4
    assert m.predict([((1,), {}) for _ in range(4)]) == [4, 4, 4, 4]


def test_pm_predict_with_select_ids_multikey(monkeypatch, predict_mixin_multikey):
    xs = [np.random.randn(4) for _ in range(10)]

    def func(x, y):
        return 2

    monkeypatch.setattr(predict_mixin_multikey, 'object', func)

    def _test(X, docs):
        ids = [i for i in range(10)]

        select = MagicMock(spec=Select)
        db = MagicMock(spec=Datalayer)
        db.execute.return_value = docs

        # Check the base predict function
        predict_mixin_multikey.db = db
        with patch.object(select, 'select_using_ids') as select_using_ids, patch.object(
            select, 'model_update'
        ) as model_update:
            predict_mixin_multikey._predict_with_select_and_ids(
                X=X, predict_id='test', db=db, select=select, ids=ids
            )
            select_using_ids.assert_called_once_with(ids)
            _, kwargs = model_update.call_args
            #  make sure the outputs are set
            assert kwargs.get('outputs') == [2] * 10

    # TODO - I don't know how this works given that the `_outputs` field
    # should break...
    docs = [Document({'x': x, 'y': x}) for x in xs]
    X = ('x', 'y')

    _test(X, docs)

    # TODO this should also work
    # docs = [Document({'a': x, 'b': x}) for x in xs]
    # X = {'a': 'x', 'b': 'y'}
    # _test(X, docs)
