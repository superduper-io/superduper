import pytorch_lightning as pl


class CustomCheckpoint(pl.callbacks.ModelCheckpoint):
    def __init__(self, database, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.database = database

    def _save_checkpoint(self, trainer, filepath):
        filepath = filepath.split('.')[0]
        self.database.replace_object(filepath, trainer.lightning_module)
