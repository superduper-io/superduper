from typing import List, Optional, Union

from superduperdb.core.base import ComponentList, PlaceholderList, Component, Placeholder, \
    is_placeholders_or_components
from superduperdb.core.metric import Metric
from superduperdb.core.model import Model
from superduperdb.core.watcher import Watcher
from superduperdb.datalayer.base.query import Select
from superduperdb.training.query_dataset import QueryDataset
from superduperdb.training.validation import validate_vector_search
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
        watcher: Union[Watcher, str],
        models: Union[List[Model], List[str]] = None,
        measure: str= 'css',
        hash_set_cls: type = VanillaHashSet,
    ):
        super().__init__(identifier)
        self.keys = keys
        self.watcher = Placeholder(watcher, 'watcher') if isinstance(watcher, str) else watcher

        is_placeholders, is_components = is_placeholders_or_components(models)
        assert is_placeholders or is_components
        if is_placeholders:
            self.models = PlaceholderList('model', models)
        else:
            self.models = ComponentList('model', models)
        assert len(self.keys) == len(self.models)
        self.measure = measure
        self.hash_set_cls = hash_set_cls

    def validate(
        self,
        database: 'superduperdb.datalayer.base.database.Database',
        validation_selects: List[Select],
        metrics: List[Metric],
    ):
        out = []
        for vs in validation_selects:
            validation_data = QueryDataset(
                vs,
                database_type=database._database_type,
                database=database.name,
                keys=self.keys,
                fold='valid',
            )
            res = validate_vector_search(
                validation_data=validation_data,
                models=self.models,
                keys=self.keys,
                metrics=metrics,
                hash_set_cls=self.hash_set_cls,
                measure=self.measure,
                predict_kwargs={},
            )
            out.append(res)
        return out

    def asdict(self):
        return {
            'identifier': self.identifier,
            'watcher': self.watcher.identifier,
            'keys': self.keys,
            'models': self.models.aslist(),
            'measure': self.measure,
            'hash_set_cls': self.hash_set_cls.name,
        }
