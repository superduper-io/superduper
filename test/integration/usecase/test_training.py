import typing as t
from test.utils.setup.fake_data import add_random_data

import pytest

from superduper.base.datalayer import Datalayer
from superduper.base.datatype import pickle_serializer
from superduper.components.model import Model, Trainer
from superduper.misc import typing as st  # noqa: F401

if t.TYPE_CHECKING:
    from superduper.base.datalayer import Datalayer


class MyUseCaseTrainer(Trainer):
    def fit(self, model, db, train_dataset, valid_dataset):
        X, y = list(zip(*list(train_dataset)))
        model.estimator.fit(X, y)
        db.apply(model, force=True, jobs=False)


class MyUseCaseModel(Model):
    _fields = {'estimator': pickle_serializer}
    estimator: t.Any
    signature: str = 'singleton'

    def predict(self, x):
        return self.estimator.predict(x[None, :]).tolist()[0]

    def predict_batches(self, dataset):
        return self.estimator.predict(dataset).tolist()


# TODO: There is an issue of hash the sklearn model
@pytest.mark.skip
def test_training(db: "Datalayer"):
    add_random_data(db, 'documents', 100)

    from sklearn.svm import SVC

    model = MyUseCaseModel(
        identifier="my-model",
        estimator=SVC(),
        trainer=MyUseCaseTrainer(
            "my-trainer",
            key=('x', 'y'),
            select=db['documents'].select(),
        ),
    )

    db.apply(model)

    # Need to reload to get the fitted model
    reloaded = db.load('MyModel', 'my-model')

    r = db['documents'].get()

    # This only works if the model was trained
    prediction = reloaded.predict(r['x'])

    assert isinstance(prediction, int)
