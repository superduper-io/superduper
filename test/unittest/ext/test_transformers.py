import tempfile

import pytest

try:
    import torch
except ImportError:
    torch = None

from test.db_config import DBConfig

from superduperdb.backends.mongodb.query import Collection
from superduperdb.base.document import Document as D
from superduperdb.components.dataset import Dataset
from superduperdb.ext.transformers.model import (
    Pipeline,
    TransformersTrainerConfiguration,
)


@pytest.fixture
@pytest.mark.parametrize(
    "db", [DBConfig.mongodb_empty, DBConfig.sqldb_empty], indirect=True
)
def transformers_model(db):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    data = [
        {'text': 'dummy text 1', 'label': 1},
        {'text': 'dummy text 2', 'label': 0},
        {'text': 'dummy text 1', 'label': 1},
    ]
    data = [D(d) for d in data]
    db.execute(Collection('train_documents').insert_many(data))
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2
    )

    model = Pipeline(
        identifier='my-sentiment-analysis',
        preprocess=tokenizer,
        object=model,
        device='cpu',
    )
    yield model


@pytest.mark.skipif(not torch, reason='Torch not installed')
def test_transformer_predict(transformers_model):
    one_prediction = transformers_model.predict_one('this is a test')
    assert isinstance(one_prediction, int)
    predictions = transformers_model.predict(['this is a test', 'this is another'])
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
    training_args = TransformersTrainerConfiguration(
        identifier=repo_name,
        output_dir=repo_name,
        learning_rate=2e-5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=1,
        weight_decay=0.01,
        save_strategy="epoch",
        use_mps_device=False,
    )
    transformers_model.fit(
        X='text',
        y='label',
        db=db,
        select=Collection('train_documents').find(),
        configuration=training_args,
        validation_sets=[
            Dataset(
                identifier='my-eval',
                select=Collection('train_documents').find({'_fold': 'valid'}),
            )
        ],
    )
