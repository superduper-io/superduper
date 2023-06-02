from typing import List, Mapping, Optional

from superduperdb.core.base import ComponentList, PlaceholderList, Component, Placeholder
from superduperdb.core.model import Model
from superduperdb.core.watcher import Watcher
from superduperdb.vector_search import VanillaHashSet


class VectorIndex(Component):
    """
    Vector-index

    :param identifier: Unique ID of index
    :param keys: Keys which may be used to search index
    :param watcher: Watcher which is applied to create vectors
    :param watcher_id: ID of Watcher which is applied to create vectors
    :param models:  models which may be used to search index
    :param model_ids: ID of models which may be used to search index
    :param measure: Measure which is used to compare vectors in index
    :param hash_set_cls: Class which is used to execute similarity lookup
    """

    variety = 'vector_index'

    def __init__(
        self,
        identifier: str,
        keys: List[str],
        watcher: Optional[Watcher] = None,
        watcher_id: Optional[str] = None,
        model_ids: Optional[List[str]] = None,
        models: Optional[List[Model]] = None,
        measure: str= 'css',
        hash_set_cls: type = VanillaHashSet,
    ):
        super().__init__(identifier)
        self.keys = keys
        self.models = ComponentList('model', models) if models else PlaceholderList('model', model_ids)
        self.watcher = Placeholder(watcher_id, 'watcher') if watcher_id is not None else watcher
        assert len(self.keys) == len(self.models)
        self.measure = measure
        self.hash_set_cls = hash_set_cls

    def asdict(self):
        return {
            'identifier': self.identifier,
            'watcher': self.watcher.identifier,
            'keys': self.keys,
            'models': self.models.aslist(),
            'measure': self.measure,
            'hash_set_cls': self.hash_set_cls.name,
        }
