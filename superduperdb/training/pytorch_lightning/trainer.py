import inspect
import typing as t
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from superduperdb.core.training_configuration import TrainingConfiguration
from superduperdb.datalayer.base.build import build_datalayer
from superduperdb.datalayer.base.query import Select
from superduperdb.models.torch.utils import eval


class CustomCheckpoint(pl.callbacks.ModelCheckpoint):
    def __init__(self, database, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.database = database

    def _save_checkpoint(self, trainer, filepath):
        filepath = filepath.split('.')[0]
        self.database._replace_model(filepath, trainer.lightning_module)


class LightningConfiguration(TrainingConfiguration):
    monitor: str
    loader_kwargs: t.Dict[str, t.Any]

    def __init__(self, identifier: str, loader_kwargs, monitor='val_loss', **kwargs):
        super().__init__(
            identifier, monitor=monitor, loader_kwargs=loader_kwargs, **kwargs
        )

    @classmethod
    def split_and_preprocess(cls, sample, model, splitter=None):
        if splitter is not None:
            sample = splitter(sample)
        return model.preprocess(sample)

    def __call__(
        self,
        identifier,
        models,
        keys,
        model_names,
        select: Select,
        splitter=None,
        validation_sets=(),
        metrics=None,
        features=None,
        download=False,
    ):
        assert len(models) == 1
        model = models[0]
        database = build_datalayer()

        with eval(model):
            train_data, valid_data = self._get_data(
                select,
                keys=None,
                features=features,
                transform=model.preprocess_for_training,
            )

        train_dataloader = DataLoader(train_data, **self.loader_kwargs)
        valid_dataloader = DataLoader(valid_data, **self.loader_kwargs)

        checkpoint_callback = CustomCheckpoint(
            database,
            monitor=self.monitor,
            mode='min',
            dirpath='',
            filename=model_names[0],
            save_top_k=1,
            verbose=True,
        )

        parameters = inspect.signature(pl.Trainer.__init__).parameters
        kwargs = {
            k: getattr(self, k) for k in parameters if k != 'self' and hasattr(self, k)
        }

        callbacks = self.get('callbacks', [])
        callbacks.append(checkpoint_callback)
        trainer = pl.Trainer(callbacks=callbacks, **kwargs)
        return lambda: trainer.fit(
            model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader
        )
