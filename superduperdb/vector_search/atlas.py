import copy
import json
import re
import typing as t
from functools import cached_property

import pymongo

from superduperdb import CFG, logging
from superduperdb.components.model import APIModel
from superduperdb.vector_search.base import BaseVectorSearcher

if t.TYPE_CHECKING:
    from superduperdb.components.vector_index import VectorIndex


class MongoAtlasVectorSearcher(BaseVectorSearcher):
    """
    Implementation of atlas vector search

    :param identifier: Unique string identifier of index
    """

    def __init__(
        self,
        identifier: str,
        collection: str,
        dimensions: t.Optional[int] = None,
        measure: t.Optional[str] = None,
        output_path: t.Optional[str] = None,
        **kwargs,
    ):
        self.identifier = identifier
        db_name = CFG.cluster.vector_search.split('/')[-1]
        self.database = getattr(pymongo.MongoClient(CFG.cluster.vector_search), db_name)
        assert output_path
        self.output_path = output_path
        self.collection = collection
        self.measure = measure
        self.dimensions = dimensions

        if not self._check_if_exists(identifier):
            self._create_index(collection, output_path)

    def __len__(self):
        pass

    @cached_property
    def index(self):
        return self.database[self.collection]

    @classmethod
    def from_component(cls, vi: 'VectorIndex'):
        from superduperdb.components.listener import Listener
        from superduperdb.components.model import ObjectModel

        assert isinstance(vi.indexing_listener, Listener)
        collection = vi.indexing_listener.select.table_or_collection.identifier

        indexing_key = vi.indexing_listener.key
        assert isinstance(
            indexing_key, str
        ), 'Only single key is support for atlas search'
        if indexing_key.startswith('_outputs'):
            indexing_key = indexing_key.split('.')[1]
        assert isinstance(vi.indexing_listener.model, ObjectModel) or isinstance(
            vi.indexing_listener.model, APIModel
        )
        assert isinstance(collection, str), 'Collection is required to be a string'
        indexing_model = vi.indexing_listener.model.identifier
        indexing_version = vi.indexing_listener.model.version
        output_path = f'_outputs.{indexing_key}.{indexing_model}.{indexing_version}'

        return MongoAtlasVectorSearcher(
            identifier=vi.identifier,
            dimensions=vi.dimensions,
            measure=vi.measure,
            output_path=output_path,
            collection=collection,
        )

    def _replace_document_with_vector(self, step):
        step = copy.deepcopy(step)
        assert "like" in step['$vectorSearch']
        vector = step['$vectorSearch']['like']
        step['$vectorSearch']['queryVector'] = vector

        step['$vectorSearch']['path'] = self.output_path
        step['$vectorSearch']['index'] = self.identifier
        del step['$vectorSearch']['like']
        return step

    def _prepare_pipeline(self, pipeline):
        pipeline = copy.deepcopy(pipeline)
        try:
            search_step = next(
                (i, step) for i, step in enumerate(pipeline) if '$vectorSearch' in step
            )
        except StopIteration:
            return pipeline
        pipeline[search_step[0]] = self._replace_document_with_vector(
            search_step[1],
        )
        return pipeline

    def _find(self, h, n=100):
        h = self.to_list(h)
        pl = [
            {
                "$vectorSearch": {
                    'like': h,
                    "limit": n,
                    'numCandidates': n,
                }
            },
            {'$addFields': {'score': {'$meta': 'vectorSearchScore'}}},
        ]
        pl = self._prepare_pipeline(
            pl,
        )
        cursor = self.index.aggregate(pl)
        scores = []
        ids = []
        for vector in cursor:
            scores.append(vector['score'])
            ids.append(str(vector['_id']))
        return ids, scores

    def find_nearest_from_id(self, id: str, n=100, within_ids=None):
        h = self.index.find_one({'id': id})
        return self.find_nearest_from_array(h, n=n, within_ids=within_ids)

    def find_nearest_from_array(self, h, n=100, within_ids=None):
        return self._find(h, n=n)

    def add(self, items):
        items = list(map(lambda x: x.to_dict(), items))
        if not CFG.cluster.vector_search == CFG.data_backend:
            self.index.insert_many(items)

    def delete(self, items):
        ids = list(map(lambda x: x.id, items))
        if not CFG.cluster.vector_search == CFG.data_backend:
            self.index.delete_many({'id': {'$in': ids}})

    def _create_index(self, collection: str, output_path: str):
        """
        Create a vector index in the data backend if an Atlas deployment.

        :param vector_index: vector index to create
        """
        _, key, model, version = output_path.split('.')
        if re.match('^_outputs\.[A-Za-z0-9_]+\.[A-Za-z0-9_]+', key):
            key = key.split('.')[1]

        fields4 = {
            str(version): [
                {
                    "dimensions": self.dimensions,
                    "similarity": self.measure,
                    "type": "knnVector",
                }
            ]
        }
        fields3 = {
            model: {
                "fields": fields4,
                "type": "document",
            }
        }
        fields2 = {
            key: {
                "fields": fields3,
                "type": "document",
            }
        }
        fields1 = {
            "_outputs": {
                "fields": fields2,
                "type": "document",
            }
        }
        index_definition = {
            "createSearchIndexes": collection,
            "indexes": [
                {
                    "name": self.identifier,
                    "definition": {
                        "mappings": {
                            "dynamic": True,
                            "fields": fields1,
                        }
                    },
                }
            ],
        }
        logging.info(json.dumps(index_definition, indent=2))
        self.database.command(index_definition)

    def _check_if_exists(self, index: str):
        indexes = self.index.list_search_indexes()
        return len(
            [i for i in indexes if i['name'] == index and i['status'] == 'READY']
        )
