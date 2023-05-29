from dataclasses import asdict, dataclass
from typing import Union, Optional

from superduperdb.core.base import Component, Placeholder
from superduperdb.core.model import Model


class Watcher(Component):
    variety = 'watcher'

    def __init__(
        self,
        select: dataclass,
        model: Optional[Model] = None,
        model_id: Optional[str] = None,
        key: str = '_base',
        features: Optional[dict] = None,
    ):
        self.model = model if model else Placeholder(model_id, 'model')
        self.select = select
        self.key = key
        self.select = select
        self.features = features
        identifier = f'{self.model.identifier}/{self.key}'
        super().__init__(identifier)

    def asdict(self):
        return {
            'model': self.model.identifier,
            'select': asdict(self.select),
            'key': self.key,
            'identifier': self.identifier,
            'features': self.features or {},
        }

    def schedule_jobs(self, database, verbose=False, dependencies=()):
        ids = database._get_ids_from_select(self.select)
        if not ids:
            return []
        return [database.apply_watcher(
            self.identifier,
            ids=ids,
            verbose=verbose,
            dependencies=dependencies,
        )]
