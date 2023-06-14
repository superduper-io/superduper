from typing import List, Union, Optional, Any

from superduperdb.core.base import (
    ComponentList,
    PlaceholderList,
    Component,
    Placeholder,
    is_placeholders_or_components,
    DBPlaceholder,
)
from superduperdb.core.documents import Document
from superduperdb.core.metric import Metric
from superduperdb.core.type import DataVar
from superduperdb.core.watcher import Watcher
from superduperdb.datalayer.base.query import Select
from superduperdb.misc import progress
from superduperdb.misc.special_dicts import MongoStyleDict
from superduperdb.training.query_dataset import QueryDataset
from superduperdb.training.validation import validate_vector_search
from superduperdb.vector_search import VanillaHashSet
from superduperdb.misc.logger import logging


class VectorIndex(Component):
    """
    Vector-index

    :param identifier: Unique ID of index
    :param indexing_watcher: watcher which is applied to create vectors
    :param compatible_watchers: list of additional watchers which can
                                "talk" to the index (e.g. multi-modal)
    :param measure: Measure which is used to compare vectors in index
    :param hash_set_cls: Class which is used to execute similarity lookup
    """

    variety = 'vector_index'

    def __init__(
        self,
        identifier: str,
        indexing_watcher: Union[Watcher, str],
        compatible_watchers: Optional[Union[List[Watcher], List[str]]] = None,
        measure: str = 'css',
        hash_set_cls: type = VanillaHashSet,
    ):
        super().__init__(identifier)
        self.indexing_watcher = (
            Placeholder(indexing_watcher, 'watcher')
            if isinstance(indexing_watcher, str)
            else indexing_watcher
        )

        self.compatible_watchers = ()
        if compatible_watchers:
            is_placeholders, is_components = is_placeholders_or_components(
                compatible_watchers
            )
            assert is_placeholders or is_components
            if is_placeholders:
                self.compatible_watchers = PlaceholderList(
                    'watcher', compatible_watchers
                )
            else:
                self.compatible_watchers = ComponentList('watcher', compatible_watchers)
        self.measure = measure
        self.hash_set_cls = hash_set_cls
        self._hash_set = None
        self.database = DBPlaceholder()

    def repopulate(self, database: Optional[Any] = None):
        if database is None:
            database = self.database
            assert not isinstance(database, DBPlaceholder)
        super().repopulate(database)
        c = database.execute(self.indexing_watcher.select)
        loaded = []
        ids = []
        docs = progress.progressbar(c)
        logging.info(f'loading hashes: "{self.identifier}')
        for r in docs:
            h, id = database._get_output_from_document(
                r, self.indexing_watcher.key, self.indexing_watcher.model.identifier
            )
            if isinstance(h, DataVar):
                h = h.x
            loaded.append(h)
            ids.append(id)

        self._hash_set = self.hash_set_cls(
            loaded,
            ids,
            measure=self.measure,
        )

    def get_nearest(
        self,
        like: Document,
        database: Optional[Any] = None,
        outputs: Optional[dict] = None,
        featurize: bool = True,
        ids: Optional[List[str]] = None,
        n: int = 100,
    ):
        if database is None:
            database = self.database
            assert not isinstance(database, DBPlaceholder)

        models, keys = self.models_keys
        assert len(models) == len(keys)

        hash_set = self._hash_set
        if ids:
            hash_set = hash_set[ids]

        if database.id_field in like.content:
            return hash_set.find_nearest_from_id(like['_id'], n=n)

        document = MongoStyleDict(like.unpack())

        if featurize:
            outputs = outputs or {}
            if '_outputs' not in document:
                document['_outputs'] = {}
            document['_outputs'].update(outputs)

            for subkey in self.indexing_watcher.features or ():
                subout = document['_outputs'].setdefault(subkey, {})
                if self.indexing_watcher.features[subkey] not in subout:
                    subout[self.indexing_watcher.features[subkey]] = database.models[
                        self.indexing_watcher.features[subkey]
                    ].predict_one(document[subkey])
                document[subkey] = subout[self.indexing_watcher.features[subkey]]
        available_keys = list(document.keys()) + ['_base']
        try:
            model, key = next(
                (m, k) for m, k in zip(models, keys) if k in available_keys
            )
        except StopIteration:
            raise Exception(
                f'Keys in provided {like} don\'t match'
                f' VectorIndex keys: {keys}, with models: {models}'
            )
        model_input = document[key] if key != '_base' else document

        model = database.models[model]
        h = model.predict_one(model_input)
        return hash_set.find_nearest_from_hash(h, n=n)

    @property
    def models_keys(self):
        watchers = [self.indexing_watcher, *self.compatible_watchers]
        models = [w.model.identifier for w in watchers]
        keys = [w.key for w in watchers]
        return models, keys

    def validate(
        self,
        database: 'superduperdb.datalayer.base.database.Database',  # noqa: F821  why?
        validation_selects: List[Select],
        metrics: List[Metric],
    ):
        models, keys = self.models_keys
        models = [database.models[m] for m in models]
        out = []
        for vs in validation_selects:
            validation_data = QueryDataset(
                vs,
                database_type=database._database_type,
                database=database.name,
                keys=keys,
                fold='valid',
            )
            res = validate_vector_search(
                validation_data=validation_data,
                models=models,
                keys=keys,
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
            'indexing_watcher': self.indexing_watcher.identifier,
            'compatible_watchers': [w.identifier for w in self.compatible_watchers],
            'measure': self.measure,
            'hash_set_cls': self.hash_set_cls.name,
        }
