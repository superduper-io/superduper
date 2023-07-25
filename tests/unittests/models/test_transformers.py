import pytest
import tdir

from superduperdb.core.document import Document as D
from superduperdb.core.dataset import Dataset
from superduperdb.datalayer.mongodb.query import Collection

from superduperdb.models.transformers.wrapper import (
    TransformersTrainerConfiguration,
    Pipeline,
)


@pytest.fixture(scope="function")
def transformers_model(random_data):
    from transformers import AutoModelForSequenceClassification
    from transformers import AutoTokenizer

    data = [
        {'text': 'dummy text 1', 'label': 1},
        {'text': 'dummy text 2', 'label': 0},
        {'text': 'dummy text 1', 'label': 1},
    ]
    data = [D(d) for d in data]
    random_data.execute(Collection('train_documents').insert_many(data))
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


def test_transformer_predict(transformers_model):
    one_prediction = transformers_model.predict('this is a test', one=True)
    assert isinstance(one_prediction, int)
    predictions = transformers_model.predict(['this is a test', 'this is another'])
    assert isinstance(predictions, list)


@tdir
def test_tranformers_fit(transformers_model, random_data):
    repo_name = "test-superduperdb-sentiment-analysis"
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
        db=random_data,
        select=Collection('train_documents').find(),
        configuration=training_args,
        validation_sets=[
            Dataset(
                identifier='my-eval',
                select=Collection(name='train_documents').find({'_fold': 'valid'}),
            )
        ],
        data_prefetch=False,
    )
