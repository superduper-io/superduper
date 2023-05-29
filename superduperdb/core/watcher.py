from dataclasses import asdict, dataclass
from typing import Union

from superduperdb.core.base import Component
from superduperdb.core.model import Model


class Watcher(Component):
    variety = 'watcher'

    def __init__(self, model: Union[Model, str], select: dataclass, key: str = '_base'):
        self.model = model
        self.select = select
        self.key = key
        self.select = select
        identifier = f'{model.identifier}/{key}' if isinstance(model, Model) else f'{model}/{key}'
        super().__init__(identifier)

    def asdict(self):
        return {
            'model': self.model.identifier if isinstance(self.model, Model) else self.model,
            'select': asdict(self.select),
            'key': self.key,
            'identifier': self.identifier,
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
