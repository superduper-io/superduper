import typing as t

from superduperdb.core.base import Component, Placeholder
from superduperdb.core.model import Model
from superduperdb.datalayer.base.apply_watcher import apply_watcher
from superduperdb.datalayer.base.query import Select


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

    features: t.Dict
    key: str
    model: t.Union[Placeholder, Model]
    select: Select
    variety = 'watcher'

    def __init__(
        self,
        select: Select,
        model: t.Union[Model, str],
        key: str = '_base',
        features: t.Optional[t.Dict] = None,
        active: bool = True,
    ):
        self.model = model if isinstance(model, Model) else Placeholder(model, 'model')
        self.select = select
        self.key = key
        self.features = features or {}
        self.active = active
        identifier = f'{self.model.identifier}/{self.key}'
        super().__init__(identifier)

    def asdict(self) -> t.Dict[str, t.Any]:
        return {
            'model': self.model.identifier,
            'select': self.select.dict(),
            'key': self.key,
            'identifier': self.identifier,
            'features': self.features or {},
            'active': self.active,
        }

    @staticmethod
    def cleanup(info, database) -> None:
        database.db.unset_outputs(info)

    def schedule_jobs(self, database, verbose: bool = False, dependencies=()):
        if not self.active:
            return
        ids = database.db.get_ids_from_select(self.select)
        if not ids:
            return []
        return [
            apply_watcher(
                database,
                self.identifier,
                ids=ids,
                verbose=verbose,
                dependencies=dependencies,
            )
        ]
