import torch

from sddb import cf
from sddb.training.losses import NegativeLoss
from sddb.training.training import RepresentationTrainer, ImputationTrainer

from tests.material.models import Dummy, DummyClassifier, DummyLabel
from tests.fixtures.collection import collection_no_hashes


def test_representation_trainer(collection_no_hashes):

    def simple_split(x):
        output0 = x.copy()
        output1 = x.copy()
        output0['test'] = x['test'].split(' ')[0]
        output1['test'] = x['test'].split(' ')[1]
        return output0, output1

    model = Dummy()

    t = RepresentationTrainer(
        client=cf['mongodb'],
        database='test_db',
        collection='test_collection',
        encoders=model,
        fields='test',
        splitter=simple_split,
        batch_size=2,
        n_epochs=2,
        loss=NegativeLoss(),
    )

    t.train()


def test_imputation_trainer(collection_no_hashes):

    model = DummyClassifier()
    label = DummyLabel()

    t = ImputationTrainer(
        client=cf['mongodb'],
        database='test_db',
        collection='test_collection',
        encoders=(model, label),
        fields=['test', 'fruit'],
        batch_size=2,
        n_epochs=2,
        loss=torch.nn.CrossEntropyLoss(),
    )

    t.train()