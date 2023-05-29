# ruff: noqa: F401, F811
from superduperdb.core.learning_task import LearningTask
from superduperdb.datalayer.mongodb.query import Select
from superduperdb.models.torch.wrapper import SuperDuperModule
from superduperdb.training.pytorch_lightning.trainer import LightningConfiguration

from tests.fixtures.collection import random_data, float_tensors, empty

import pytorch_lightning as pl
import os
import pytest
import torch

DISABLE_TEST = os.environ.get('DISABLE_PYTORCH_LIGHTNING', '').lower().startswith('t')


class LightningModule(SuperDuperModule, pl.LightningModule):
    def __init__(self):
        pl.LightningModule.__init__(self)
        SuperDuperModule.__init__(self, torch.nn.Linear(32, 1), 'my-pl-module')

    def preprocess(self, r):
        return r['x']

    def preprocess_for_training(self, r):
        return r['x'], float(r['y'])

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)[:, 0]
        loss = torch.nn.functional.cross_entropy(pred, y)
        self.log("val_loss", loss, on_epoch=True)

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)[:, 0]
        return torch.nn.functional.binary_cross_entropy_with_logits(pred, y)

    def postprocess(self, out):
        return int((out.exp() > 0.5).item())


@pytest.mark.xfail(DISABLE_TEST, reason='See issue #94')
def test_classification(random_data):

    cf = LightningConfiguration(
        'my-pl-cf', loader_kwargs={'batch_size': 5, 'num_workers': 0}, max_epochs=10
    )
    random_data.database.create_component(LightningModule())
    random_data.database.create_component(cf)

    random_data.database.create_component(
        LearningTask(
            'my-pl-lt',
            model_ids=['my-pl-module'],
            keys=['_base'],
            training_configuration_id='my-pl-cf',
            select=Select('documents'),
        )
    )

    print(random_data.list_models())
