import inspect
import random
from test.db_config import DBConfig
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from sklearn.metrics import accuracy_score, f1_score

from superduperdb.backends.base.query import CompoundSelect, Select
from superduperdb.backends.local.compute import LocalComputeBackend
from superduperdb.base.artifact import Artifact
from superduperdb.base.datalayer import Datalayer
from superduperdb.base.document import Document
from superduperdb.components.component import Component
from superduperdb.components.dataset import Dataset
from superduperdb.components.encoder import Encoder
from superduperdb.components.listener import Listener
from superduperdb.components.metric import Metric
from superduperdb.components.model import (
    Model,
    TrainingConfiguration,
    _Predictor,
    _TrainingConfiguration,
)

# ------------------------------------------
# Test the _TrainingConfiguration class (tc)
# ------------------------------------------


def test_tc_type_id():
    config = _TrainingConfiguration('config')
    assert config.type_id == 'training_configuration'


def test_tc_get_method():
    config = _TrainingConfiguration('config', kwargs={'param1': 'value1'})
    config.version = 1

    assert config.get('identifier') == 'config'
    assert config.get('param1') == 'value1'
    assert config.get('version') == 1

    assert config.get('non_existent') is None
    assert config.get('non_existent', 'default_value') == 'default_value'

    # First get the properties of the instance
    config = _TrainingConfiguration('config', kwargs={'version': 2})
    config.version = 1
    assert config.get('version') == 1


# --------------------------------
# Test the PredictMixin class (pm)
# --------------------------------


def return_self(x):
    return x


def to_call(x):
    if isinstance(x, list):
        return [to_call(i) for i in x]
    return x * 5


def preprocess(x):
    return x + 1


def postprocess(x):
    return x + 0.1


def mock_forward(self, x, **kwargs):
    return to_call(x)


class TestModel(Component, _Predictor):
    batch_predict: bool = False


@pytest.fixture
def predict_mixin(request) -> _Predictor:
    cls_ = getattr(request, 'param', _Predictor)

    if 'identifier' in inspect.signature(cls_).parameters:
        predict_mixin = cls_(identifier='test')
    else:
        predict_mixin = cls_()
    predict_mixin.identifier = 'test'
    predict_mixin.to_call = to_call
    predict_mixin.preprocess = Artifact(preprocess)
    predict_mixin.postprocess = Artifact(postprocess)
    predict_mixin.takes_context = False
    predict_mixin.output_schema = None
    predict_mixin.encoder = None
    predict_mixin.model_update_kwargs = {}
    predict_mixin.version = 0
    return predict_mixin


def test_pm_predict_one(predict_mixin):
    X = np.random.randn(5)

    # preprocess -> to_call -> postprocess
    expect = postprocess(to_call(preprocess(X)))
    assert np.allclose(predict_mixin._predict_one(X), expect)

    # Bad preprocess
    with patch.object(predict_mixin, 'preprocess', lambda x: x), pytest.raises(
        ValueError
    ) as excinfo:
        predict_mixin._predict_one(X)
        assert 'preprocess' in str(excinfo.value)

    # to_call -> postprocess
    with patch.object(predict_mixin, 'preprocess', None):
        expect = postprocess(to_call(X))
        assert np.allclose(predict_mixin._predict_one(X), expect)

    # Bad postprocess
    with patch.object(predict_mixin, 'postprocess', lambda x: x), pytest.raises(
        ValueError
    ) as excinfo:
        predict_mixin._predict_one(X)
        assert 'postprocess' in str(excinfo.value)

    # preprocess -> to_call
    with patch.object(predict_mixin, 'postprocess', None):
        expect = to_call(preprocess(X))
        assert np.allclose(predict_mixin._predict_one(X), expect)


@pytest.mark.parametrize(
    'batch_predict, num_workers, expect_type',
    [
        [True, 0, np.ndarray],
        [False, 0, list],
        [False, 1, list],
        [False, 5, list],
    ],
)
def test_pm_forward(batch_predict, num_workers, expect_type):
    predict_mixin = _Predictor()
    X = np.random.randn(4, 5)

    predict_mixin.to_call = to_call
    predict_mixin.batch_predict = batch_predict

    output = predict_mixin._forward(X, num_workers=num_workers)
    assert isinstance(output, expect_type)
    assert np.allclose(output, to_call(X))


@patch.object(_Predictor, '_forward', mock_forward)
def test_pm_core_predict(predict_mixin):
    X = np.random.randn(4, 5)

    # make sure _predict_one is called
    with patch.object(predict_mixin, '_predict_one', return_self):
        assert predict_mixin._predict(5, one=True) == return_self(5)

    expect = postprocess(to_call(preprocess(X)))
    output = predict_mixin._predict(X)
    assert isinstance(output, list)
    assert np.allclose(output, expect)

    # Bad preprocess
    with patch.object(predict_mixin, 'preprocess', lambda x: x), pytest.raises(
        ValueError
    ) as excinfo:
        predict_mixin._predict(X)
        assert 'preprocess' in str(excinfo.value)

    # to_call -> postprocess
    with patch.object(predict_mixin, 'preprocess', None):
        expect = postprocess(to_call(X))
        output = predict_mixin._predict(X)
        assert isinstance(output, list)
        assert np.allclose(output, expect)

    # Bad postprocess
    with patch.object(predict_mixin, 'postprocess', lambda x: x), pytest.raises(
        ValueError
    ) as excinfo:
        predict_mixin._predict(X)
        assert 'postprocess' in str(excinfo.value)

    # preprocess -> to_call
    with patch.object(predict_mixin, 'postprocess', None):
        output = predict_mixin._predict(X)
        expect = to_call(preprocess(X))
        assert isinstance(output, list)
        assert np.allclose(output, expect)


@patch('superduperdb.backends.base.query.Select.serialize', MagicMock())
def test_pm_create_predict_job(predict_mixin):
    select = MagicMock(spec=Select)
    X = 'x'
    ids = [1, 2, 3]
    max_chunk_size = 2
    job = predict_mixin.create_predict_job(X, select, ids, max_chunk_size)
    assert job.component_identifier == predict_mixin.identifier
    assert job.method_name == 'predict'
    assert job.args == [X]
    assert job.kwargs['max_chunk_size'] == max_chunk_size
    assert job.kwargs['ids'] == ids


@patch.object(Datalayer, 'add')
@pytest.mark.parametrize("db", [DBConfig.mongodb_empty], indirect=True)
def test_pm_predict_and_listen(mock_add, predict_mixin, db):
    X = 'x'
    select = MagicMock(CompoundSelect)

    in_memory = False
    max_chunk_size = 2
    predict_mixin._predict_and_listen(
        X,
        select,
        db=db,
        max_chunk_size=max_chunk_size,
        in_memory=in_memory,
    )
    listener = mock_add.call_args[0][0]

    # Check whether create a correct listener
    assert isinstance(listener, Listener)
    assert listener.model == predict_mixin
    assert listener.predict_kwargs['in_memory'] == in_memory
    assert listener.predict_kwargs['max_chunk_size'] == max_chunk_size


@pytest.mark.parametrize('predict_mixin', [TestModel], indirect=True)
def test_pm_predict(predict_mixin):
    # Check the logic of predict method, the mock method will be tested below
    db = MagicMock(spec=Datalayer)
    db.compute = MagicMock(spec=LocalComputeBackend)
    db.metadata = MagicMock()
    select = MagicMock(spec=Select)
    select.table_or_collection = MagicMock()

    with patch.object(predict_mixin, '_predict_and_listen') as predict_func:
        predict_mixin.predict('x', db, select, listen=True)
        predict_func.assert_called_once()

    with patch.object(predict_mixin, '_predict') as predict_func:
        predict_mixin.predict('x')
        predict_func.assert_called_once()


def test_pm_predict_with_select(predict_mixin):
    # Check the logic about overwrite in _predict_with_select
    X = 'x'
    all_ids = ['1', '2', '3']
    ids_of_missing_outputs = ['1', '2']

    select = MagicMock(spec=Select)
    select.select_ids_of_missing_outputs.return_value = 'missing'

    def return_value(select_type):
        ids = ids_of_missing_outputs if select_type == 'missing' else all_ids
        query_result = [
            (
                {
                    'id_field': id,
                }
            )
            for id in ids
        ]
        return query_result

    db = MagicMock(spec=Datalayer)
    db.execute.side_effect = return_value
    db.databackend = MagicMock()
    db.databackend.id_field = 'id_field'

    # overwrite = True
    with patch.object(predict_mixin, '_predict_with_select_and_ids') as mock_predict:
        predict_mixin._predict_with_select(X, select, db, overwrite=True)
        _, kwargs = mock_predict.call_args
        assert kwargs.get('ids') == all_ids

    # overwrite = False
    with patch.object(predict_mixin, '_predict_with_select_and_ids') as mock_predict:
        predict_mixin._predict_with_select(
            X, select, db, overwrite=False, max_chunk_size=None, in_memory=True
        )
        _, kwargs = mock_predict.call_args
        assert kwargs.get('ids') == ids_of_missing_outputs


@patch.object(_Predictor, '_predict')
def test_pm_predict_with_select_ids(predict_mock, predict_mixin):
    xs = [np.random.randn(4) for _ in range(10)]
    ys = [int(random.random() > 0.5) for i in range(10)]
    docs = [Document({'x': x}) for x in xs]
    ids = [i for i in range(10)]

    select = MagicMock(spec=Select)
    db = MagicMock(spec=Datalayer)
    db.execute.return_value = docs

    # Check the base predict function
    predict_mock.return_value = ys
    with patch.object(select, 'select_using_ids') as select_using_ids, patch.object(
        select, 'model_update'
    ) as model_update:
        predict_mixin._predict_with_select_and_ids('x', db, select, ids)
        select_using_ids.assert_called_once_with(ids)
        _, kwargs = model_update.call_args
        #  make sure the outputs are set
        assert kwargs.get('outputs') == ys

    # Check the base predict function with encoder
    from superduperdb.components.encoder import Encoder

    predict_mixin.encoder = encoder = Encoder(identifier='test')
    with patch.object(select, 'model_update') as model_update:
        predict_mixin._predict_with_select_and_ids('x', db, select, ids)
        select_using_ids.assert_called_once_with(ids)
        _, kwargs = model_update.call_args
        #  make sure encoder is used
        encoder = predict_mixin.encoder
        assert kwargs.get('outputs') == [encoder(y).encode() for y in ys]

    # Check the base predict function with output_schema
    from superduperdb.components.schema import Schema

    predict_mixin.encoder = None
    predict_mixin.output_schema = schema = MagicMock(spec=Schema)
    schema.encode.side_effect = str
    predict_mock.return_value = [{'y': y} for y in ys]
    with patch.object(select, 'model_update') as model_update:
        predict_mixin._predict_with_select_and_ids('x', db, select, ids)
        select_using_ids.assert_called_once_with(ids)
        _, kwargs = model_update.call_args
        assert kwargs.get('outputs') == [str({'y': y}) for y in ys]


# --------------------
# Test the Model class
# --------------------


def test_model_init():
    # Check all the object are converted to Artifact
    obj = object()
    model = Model('test', object=obj)
    assert isinstance(model.object, Artifact)
    assert model.object.artifact is obj

    preprocess = object()
    model = Model('test', object=obj, preprocess=preprocess)
    assert isinstance(model.preprocess, Artifact)
    assert model.preprocess.artifact is preprocess

    postprocess = object()
    model = Model('test', object=obj, postprocess=postprocess)
    assert isinstance(model.postprocess, Artifact)
    assert model.postprocess.artifact is postprocess

    # Check the model_to_device_method is set correctly
    class SubModel(Model):
        def to(self, device):
            return self

    model = SubModel('test', object=obj, model_to_device_method='to')
    assert model._artifact_method == model.to


def test_model_child_components():
    # Check the child components are empty
    model = Model('test', object=object())
    assert model.child_components == []

    # if encoder or training_configuration is set
    model = Model('test', object=object(), encoder=Encoder(identifier='test'))
    assert model.child_components == [('encoder', 'encoder')]

    model = Model(
        'test',
        object=object(),
        training_configuration=TrainingConfiguration(identifier='test'),
    )
    assert model.child_components == [
        ('training_configuration', 'training_configuration')
    ]

    # if encoder and training_configuration are set
    model = Model(
        'test',
        object=object(),
        encoder=Encoder(identifier='test'),
        training_configuration=TrainingConfiguration(identifier='test'),
    )

    assert model.child_components == [
        ('encoder', 'encoder'),
        ('training_configuration', 'training_configuration'),
    ]


def test_model_on_create():
    db = MagicMock(spec=Datalayer)
    db.databackend = MagicMock()

    # Check the encoder is loaded if encoder is string
    model = Model('test', object=object(), encoder='test_encoder')
    with patch.object(db, 'load') as db_load:
        model.pre_create(db)
        db_load.assert_called_with('encoder', 'test_encoder')

    # Check the output_component table is added by datalayer
    model = Model('test', object=object(), encoder=Encoder(identifier='test'))
    output_component = MagicMock()
    db.databackend.create_model_table_or_collection.return_value = output_component
    with patch.object(db, 'add') as db_load:
        model.post_create(db)
        db_load.assert_called_with(output_component)


def test_model_append_metrics():
    model = Model('test', object=object())

    metric_values = {'acc': 0.5, 'loss': 0.5}

    model.append_metrics(metric_values)

    assert model.metric_values.get('acc') == [0.5]
    assert model.metric_values.get('loss') == [0.5]

    metric_values = {'acc': 0.6, 'loss': 0.4}
    model.append_metrics(metric_values)
    assert model.metric_values.get('acc') == [0.5, 0.6]
    assert model.metric_values.get('loss') == [0.5, 0.4]


@patch.object(Model, '_validate')
def test_model_validate(mock_validate):
    # Check the metadadata recieves the correct values
    mock_validate.return_value = {'acc': 0.5, 'loss': 0.5}
    model = Model('test', object=object())
    db = MagicMock(spec=Datalayer)
    db.metadata = MagicMock()
    with patch.object(db, 'add') as db_add, patch.object(
        db.metadata, 'update_object'
    ) as update_object:
        model.validate(db, MagicMock(spec=Dataset), [MagicMock(spec=Metric)])
        db_add.assert_called_once_with(model)
        _, kwargs = update_object.call_args
        assert kwargs.get('key') == 'dict.metric_values'
        assert kwargs.get('value') == {'acc': 0.5, 'loss': 0.5}


@patch.object(Model, '_predict')
@pytest.mark.parametrize(
    "db",
    [
        (DBConfig.mongodb_data, {'n_data': 500}),
        (DBConfig.sqldb_data, {'n_data': 500}),
    ],
    indirect=True,
)
def test_model_core_validate(model_predict, valid_dataset, db):
    # Check the validation is done correctly
    db.add(valid_dataset)
    model = Model('test', object=object(), train_X='x', train_y='y')
    model_predict.side_effect = lambda x: [random.randint(0, 1) for _ in range(len(x))]
    metrics = [
        Metric('f1', object=f1_score),
        Metric('acc', object=accuracy_score),
    ]
    results = model._validate(db, valid_dataset.identifier, metrics)
    assert len(results) == 2
    assert isinstance(results.get(f'{valid_dataset.identifier}/f1'), float)
    assert isinstance(results.get(f'{valid_dataset.identifier}/acc'), float)

    results = model._validate(db, valid_dataset, metrics)
    assert len(results) == 2
    assert isinstance(results.get(f'{valid_dataset.identifier}/f1'), float)
    assert isinstance(results.get(f'{valid_dataset.identifier}/acc'), float)


def test_model_create_fit_job():
    # Check the fit job is created correctly
    model = Model('test', object=object())
    job = model.create_fit_job('x')
    assert job.component_identifier == model.identifier
    assert job.method_name == 'fit'
    assert job.args == ['x']


def test_model_fit(valid_dataset):
    # Check the logic of the fit method, the mock method was tested above
    model = Model('test', object=object())
    with patch.object(model, '_fit') as model_fit:
        model.fit('x')
        model_fit.assert_called_once()

    with patch.object(model, '_fit') as model_fit:
        db = MagicMock(spec=Datalayer)
        db.compute = MagicMock(spec=LocalComputeBackend)
        model.fit(
            valid_dataset,
            db=db,
            validation_sets=[valid_dataset],
        )
        _, kwargs = model_fit.call_args
        assert kwargs.get('validation_sets') == [valid_dataset.identifier]
