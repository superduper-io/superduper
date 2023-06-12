from typing import List, Union, Any, Dict, TYPE_CHECKING, Optional

from superduperdb.core.base import (
    ComponentList,
    PlaceholderList,
    Component,
    Placeholder,
    is_placeholders_or_components,
    DBPlaceholder,
)
from superduperdb.core.metric import Metric
from superduperdb.core.model import Model
from superduperdb.core.watcher import Watcher
from superduperdb.datalayer.base.query import Select
from superduperdb.misc import progress
from superduperdb.misc.logger import logging
from superduperdb.misc.special_dicts import MongoStyleDict
from superduperdb.training.query_dataset import QueryDataset
from superduperdb.training.validation import validate_vector_search
from superduperdb.vector_search import VanillaHashSet
from superduperdb.vector_search.base import BaseHashSet

if TYPE_CHECKING:
    from superduperdb.datalayer.mongodb.database import Database


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
    models: Union[PlaceholderList, ComponentList]
    watcher: Union[Watcher, Placeholder]
    _hash_set: Optional[BaseHashSet]

    def __init__(
        self,
        identifier: str,
        keys: List[str],
        watcher: Union[Watcher, str],
        models: Union[List[Model], List[str], None] = None,
        measure: str = 'css',
        hash_set_cls: type = VanillaHashSet,
    ):
        super().__init__(identifier)
        self.keys = keys
        self.watcher = (
            Placeholder(watcher, 'watcher') if isinstance(watcher, str) else watcher
        )

        is_placeholders, is_components = is_placeholders_or_components(models)
        assert is_placeholders or is_components
        if is_placeholders:
            self.models = PlaceholderList('model', models)
        else:
            self.models = ComponentList('model', models)
        assert len(self.keys) == len(self.models)
        self.measure = measure
        self.hash_set_cls = hash_set_cls
        self._hash_set = None
        self.database = DBPlaceholder()

    def repopulate(self, database: Optional[Any] = None):
        if database is None:
            database = self.database
            assert not isinstance(database, DBPlaceholder)
        super().repopulate(database)
        assert isinstance(self.watcher, Watcher)
        c = database.select(self.watcher.select)
        loaded = []
        ids = []
        docs = progress.progressbar(c)
        assert hasattr(logging, 'info')
        logging.info(f'loading hashes: "{self.identifier}')
        assert isinstance(self.watcher, Watcher)
        for r in docs:
            h = database._get_output_from_document(
                r, self.watcher.key, self.watcher.model.identifier
            )
            loaded.append(h)
            ids.append(r['_id'])

        self._hash_set = self.hash_set_cls(
            loaded,
            ids,
            measure=self.measure,
        )

    def get_nearest(
        self,
        like: Dict,
        database: Optional[Any] = None,
        outputs: Optional[dict] = None,
        featurize: bool = True,
        ids=None,
        n=100,
    ):
        if database is None:
            database = self.database
            assert not isinstance(database, DBPlaceholder)

        models = [m.identifier for m in self.models]
        keys = self.keys
        assert len(models) == len(keys)

        assert isinstance(self._hash_set, BaseHashSet)
        hash_set = self._hash_set
        if ids:
            hash_set = hash_set[ids]

        if database.id_field in like:
            return hash_set.find_nearest_from_id(like['_id'], n=n)

        document = MongoStyleDict(like)

        if featurize:
            outputs = outputs or {}
            if '_outputs' not in document:
                document['_outputs'] = {}
            document['_outputs'].update(outputs)
            assert isinstance(self.watcher, Watcher)
            for subkey in self.watcher.features or ():
                subout = document['_outputs'].setdefault(subkey, {})
                if self.watcher.features[subkey] not in subout:
                    subout[self.watcher.features[subkey]] = database.models[
                        self.watcher.features[subkey]
                    ].predict_one(document[subkey])
                document[subkey] = subout[self.watcher.features[subkey]]
        available_keys = list(document.keys()) + ['_base']
        try:
            model, key = next(
                (m, k) for m, k in zip(models, keys) if k in available_keys
            )
        except StopIteration:
            raise Exception(
                f'Keys in provided {like} don\'t match'
                f' VectorIndex keys: {self.keys}, {self.models}'
            )
        model_input = document[key] if key != '_base' else document

        model = database.models[model]
        h = model.predict_one(model_input)
        return hash_set.find_nearest_from_hash(h, n=n)

    def validate(
        self,
        database: 'Database',  # noqa: F821  why?
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
