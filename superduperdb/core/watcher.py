from dataclasses import asdict, dataclass
from typing import Union, Optional

from superduperdb.core.base import Component, Placeholder
from superduperdb.core.model import Model


class Watcher(Component):
    """
    Watcher object which is used to process a column/ key of a collection or table,
    and store the outputs.

    :param select: Object for selecting which data is processed
    :param model: Model for processing data
    :param key: Key to be bound to model
    :param features: Dictionary of mappings from keys to models
    :param active: Toggle to ``False`` to deactivate change data triggering
    """

    variety = 'watcher'

    def __init__(
        self,
        select: dataclass,
        model: Union[Model, str],
        key: str = '_base',
        features: Optional[dict] = None,
        active: bool = True,
    ):
        self.model = model if isinstance(model, Model) else Placeholder(model, 'model')
        self.select = select
        self.key = key
        self.select = select
        self.features = features or {}
        self.active = active
        identifier = f'{self.model.identifier}/{self.key}'
        super().__init__(identifier)

    def asdict(self):
        return {
            'model': self.model.identifier,
            '_select': asdict(self.select),
            'key': self.key,
            'identifier': self.identifier,
            'features': self.features or {},
            'active': self.active,
        }

    def schedule_jobs(self, database, verbose=False, dependencies=()):
        if not self.active:
            return
        ids = database._get_ids_from_select(self.select)
        if not ids:
            return []
        return [
            database._apply_watcher(
                self.identifier,
                ids=ids,
                verbose=verbose,
                dependencies=dependencies,
            )
        ]
