from tests.fixtures.collection import random_data, float_tensors, empty
from superduperdb.training.pytorch_lightning.trainer import (
    LightningConfiguration,
)

import pytorch_lightning as pl
import torch


class LightningModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(32, 1)

    def preprocess(self, r):
        return r['x']

    def preprocess_for_training(self, r):
        return r['x'], float(r['y'])

    def forward(self, x):
        return self.linear(x)

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


def test_classification(random_data):
    cf = LightningConfiguration(
        loader_kwargs={'batch_size': 5, 'num_workers': 0}, max_epochs=10
    )

    random_data.create_model('lightning_classifier', LightningModule())

    random_data.create_learning_task(
        models=['lightning_classifier'],
        keys=['_base'],
        configuration=cf,
    )

    print(random_data.list_models())
