import tempfile

import pytest

try:
    import torch
except ImportError:
    torch = None

from test.db_config import DBConfig

from superduperdb.backends.mongodb.query import MongoQuery
from superduperdb.base.document import Document as D
from superduperdb.components.dataset import Dataset
from superduperdb.ext.transformers.model import (
    TextClassificationPipeline,
    TransformersTrainer,
)


@pytest.fixture
@pytest.mark.parametrize(
    "db", [DBConfig.mongodb_empty, DBConfig.sqldb_empty], indirect=True
)
def transformers_model(db):
    data = [
        {'text': 'dummy text 1', 'label': 1},
        {'text': 'dummy text 2', 'label': 0},
        {'text': 'dummy text 1', 'label': 1},
    ]
    data = [D(d) for d in data]
    db.execute(MongoQuery('train_documents').insert_many(data))

    model = TextClassificationPipeline(
        identifier='my-sentiment-analysis',
        model_name='distilbert-base-uncased',
        model_kwargs={'num_labels': 2},
        device='cpu',
    )
    yield model


@pytest.mark.skipif(not torch, reason='Torch not installed')
def test_transformer_predict(transformers_model):
    one_prediction = transformers_model.predict('this is a test')
    assert isinstance(one_prediction, dict)
    predictions = transformers_model.predict_batches(
        ['this is a test', 'this is another']
    )
    assert isinstance(predictions, list)


@pytest.fixture
def td():
    with tempfile.TemporaryDirectory() as tmp:
        yield tmp


@pytest.mark.skipif(not torch, reason='Torch not installed')
# TODO: Test the sqldb
@pytest.mark.parametrize("db", [DBConfig.mongodb_empty], indirect=True)
def test_transformer_fit(transformers_model, db, td):
    repo_name = td
    trainer = TransformersTrainer(
        key={'text': 'text', 'label': 'label'},
        select=MongoQuery('train_documents').find(),
        identifier=repo_name,
        learning_rate=2e-5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=1,
        weight_decay=0.01,
        save_strategy="epoch",
        use_mps_device=False,
    )
    transformers_model.trainer = trainer
    transformers_model.validation_sets = [
        Dataset(
            identifier='my-eval',
            select=MongoQuery('train_documents').find({'_fold': 'valid'}),
        )
    ]
    transformers_model.fit_in_db(db)
