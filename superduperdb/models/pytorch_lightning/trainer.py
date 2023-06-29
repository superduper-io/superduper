import typing as t
import pytorch_lightning as pl

from superduperdb.datalayer.base.database import BaseDatabase


class CustomCheckpoint(pl.callbacks.ModelCheckpoint):
    def __init__(
        self, database: BaseDatabase, *args: t.Tuple, **kwargs: t.Dict[str, t.Any]
    ) -> None:
        super().__init__(*args, **kwargs)
        self.database = database

    def _save_checkpoint(self, trainer: t.Any, filepath: str) -> None:
        filepath = filepath.split('.')[0]
        self.database._replace_model(filepath, trainer.lightning_module)
