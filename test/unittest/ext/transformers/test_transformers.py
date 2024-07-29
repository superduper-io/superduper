import tempfile

import pytest

try:
    import torch
except ImportError:
    torch = None

from superduper.backends.mongodb.query import MongoQuery
from superduper.components.dataset import Dataset
from superduper.ext.transformers.model import (
    TextClassificationPipeline,
    TransformersTrainer,
)


@pytest.fixture
def transformers_model(db):
    db.cfg.auto_schema = True
    data = [
        {'text': 'dummy text 1', 'label': 1},
        {'text': 'dummy text 2', 'label': 0},
        {'text': 'dummy text 1', 'label': 1},
    ]
    db['train_documents'].insert(data).execute()
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
def test_transformer_fit(transformers_model, db, td):
    repo_name = td
    trainer = TransformersTrainer(
        key={'text': 'text', 'label': 'label'},
        select=db['train_documents'].select(),
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
            select=MongoQuery(table='train_documents').find({'_fold': 'valid'}),
        )
    ]
    transformers_model.fit_in_db(db)
