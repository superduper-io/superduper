import copy
import dataclasses as dc
import typing as t

import mongomock
import pymongo
from bson import ObjectId
from pymongo import InsertOne as _InsertOne, UpdateOne as _UpdateOne

from superduperdb import CFG
from superduperdb.container.document import Document
from superduperdb.db.base.cursor import SuperDuperCursor
from superduperdb.db.base.query import (
    CompoundSelect,
    Delete,
    Featurize,
    Insert,
    Like,
    QueryComponent,
    QueryLinker,
    QueryType,
    Select,
    TableOrCollection,
    Update,
)


class FindOne(QueryComponent):
    """
    Wrapper around ``pymongo.Collection.find_one``

    :param args: Positional arguments to ``pymongo.Collection.find_one``
    :param kwargs: Named arguments to ``pymongo.Collection.find_one``
    """

    def select_using_ids(self, ids):
        ids = [ObjectId(id) for id in ids]
        args = list(self.args)[:]
        if not args:
            args = [{}]
        args[0].update({'_id': {'$in': ids}})
        return FindOne(
            name=self.name,
            type=self.type,
            args=args,
            kwargs=self.kwargs,
        )

    def add_fold(self, fold: str):
        """
        Modify the query to add a fold to filter {'_fold': fold}

        :param fold: The fold to add
        """
        if not self.args:
            args: t.List[t.Any] = [{}]
        args[0]['_fold'] = fold
        return FindOne(
            name=self.name,
            type=self.type,
            args=args,
            kwargs=self.kwargs,
        )


class Find(QueryComponent):
    """
    Wrapper around ``pymongo.Collection.find``

    :param args: Positional arguments to ``pymongo.Collection.find``
    :param kwargs: Named arguments to ``pymongo.Collection.find``
    """

    @property
    def select_ids(self):
        args = list(self.args)[:]
        if not args:
            args = [{}]
        else:
            args = list(self.args)
        if not args[1:]:
            args.append({})

        args[1].update({'_id': 1})
        return Find(
            name=self.name,
            type=self.type,
            args=args,
            kwargs=self.kwargs,
        )

    def select_using_ids(self, ids):
        ids = [ObjectId(id) for id in ids]
        args = list(self.args)[:]
        if not args:
            args = [{}]
        args[0].update({'_id': {'$in': ids}})
        return Find(
            name=self.name,
            type=self.type,
            args=args,
            kwargs=self.kwargs,
        )

    def select_ids_of_missing_outputs(self, key: str, model: str):
        assert self.type == QueryType.QUERY
        if self.args:
            args = [
                {'$and': [self.args[0], {f'_outputs.{key}.{model}': {'$exists': 0}}]},
                *self.args[1],
            ]
        else:
            args = [{f'_outputs.{key}.{model}': {'$exists': 0}}]

        return Find(
            name='find',
            type=QueryType.QUERY,
            args=args,
            kwargs=self.kwargs,
        )

    def select_single_id(self, id):
        assert self.type == QueryType.QUERY
        args = list(self.args)[:]
        if not args:
            args = [{}]
        args[0].update({'_id': id})
        return QueryComponent(
            name='find_one',
            type=QueryType.QUERY,
            args=args,
            kwargs=self.kwargs,
        )

    def add_fold(self, fold: str):
        args = self.args
        if not self.args:
            args = [{}]
        args[0]['_fold'] = fold
        return FindOne(
            name=self.name,
            type=self.type,
            args=args,
            kwargs=self.kwargs,
        )


@dc.dataclass
class Aggregate(Select):
    """
    Wrapper around ``pymongo.Collection.aggregate``

    :param table_or_collection: The table or collection to perform the query on
    :param vector_index: The vector index to use
    :param args: Positional arguments to ``pymongo.Collection.aggregate``
    :param kwargs: Named arguments to ``pymongo.Collection.aggregate``
    """

    table_or_collection: 'TableOrCollection'
    vector_index: t.Optional[str] = None
    args: t.Tuple[t.Any, ...] = dc.field(default_factory=tuple)
    kwargs: t.Dict[str, t.Any] = dc.field(default_factory=dict)

    @property
    def id_field(self):
        return self.table_or_collection.primary_id

    @property
    def select_table(self):
        raise NotImplementedError

    def add_fold(self):
        raise NotImplementedError

    def select_single_id(self, id: str):
        raise NotImplementedError

    @property
    def select_ids(self):
        raise NotImplementedError

    def select_using_ids(self):
        raise NotImplementedError

    def select_ids_of_missing_outputs(self, key: str, model: str):
        raise NotImplementedError

    @staticmethod
    def _replace_document_with_vector(step, vector_index, db):
        step = copy.deepcopy(step)
        assert "like" in step['$search']['knnBeta']
        vector_index = db.vector_indices[vector_index]
        models, keys = vector_index.models_keys
        step['$search']['knnBeta']['vector'], _, _ = vector_index.get_vector(
            like=step['$search']['knnBeta']['like'],
            models=models,
            keys=keys,
            db=db,
        )
        indexing_key = vector_index.indexing_listener.key
        if indexing_key.startswith('_outputs'):
            indexing_key = indexing_key.split('.')[1]
        indexing_model = vector_index.indexing_listener.model.identifier
        step['$search']['knnBeta']['path'] = f'_outputs.{indexing_key}.{indexing_model}'
        step['$search']['index'] = vector_index.identifier
        del step['$search']['knnBeta']['like']
        return step

    @staticmethod
    def _prepare_pipeline(pipeline, db, vector_index):
        pipeline = copy.deepcopy(pipeline)
        try:
            search_step = next(
                (i, step) for i, step in enumerate(pipeline) if '$search' in step
            )
        except StopIteration:
            return pipeline
        pipeline[search_step[0]] = Aggregate._replace_document_with_vector(
            search_step[1], vector_index, db
        )
        return pipeline

    def execute(self, db):
        collection = db.databackend.get_table_or_collection(
            self.table_or_collection.identifier
        )
        cursor = collection.aggregate(
            self._prepare_pipeline(self.args[0], db, self.vector_index)
        )
        return SuperDuperCursor(
            raw_cursor=cursor,
            id_field='_id',
            encoders=db.encoders,
        )


@dc.dataclass(repr=False)
class MongoCompoundSelect(CompoundSelect):
    def _get_query_linker(cls, table_or_collection, members) -> 'QueryLinker':
        return MongoQueryLinker(
            table_or_collection=table_or_collection, members=members
        )

    def change_stream(self, *args, **kwargs):
        return self.table_or_collection.change_stream(*args, **kwargs)

    def execute(self, db):
        output, scores = super().execute(db)
        if isinstance(output, (pymongo.cursor.Cursor, mongomock.collection.Cursor)):
            return SuperDuperCursor(
                raw_cursor=output,
                id_field='_id',
                scores=scores,
                encoders=db.encoders,
            )
        elif isinstance(output, dict):
            return Document(Document.decode(output, encoders=db.encoders))
        return output

    def featurize(self, features: t.Dict):
        return Featurize(features=features, parent=self)

    def download_update(self, db, id: str, key: str, bytes: bytearray) -> None:
        """
        Update to set the content of ``key`` in the document ``id``.

        :param db: The db to query
        :param id: The id to filter on
        :param key:
        :param bytes: The bytes to update
        """
        if self.collection is None:
            raise ValueError('collection cannot be None')
        update = {'$set': {f'{key}._content.bytes': bytes}}
        collection = db.databackend.get_table_or_collection(
            self.table_or_collection.identifier
        )
        return collection.update_one({'_id': id}, update)

    def select_table(self):
        return self.table_or_collection.find()


@dc.dataclass(repr=False)
class MongoQueryLinker(QueryLinker):
    @property
    def query_components(self):
        return self.table_or_collection.query_components

    def add_fold(self, fold):
        new_members = []
        for member in self.members:
            if hasattr(member, 'add_fold'):
                new_members.append(member.add_fold(fold))
            else:
                new_members.append(member)
        return MongoQueryLinker(
            table_or_collection=self.table_or_collection,
            members=new_members,
        )

    @property
    def select_ids(self):
        new_members = []
        for member in self.members:
            if hasattr(member, 'select_ids'):
                new_members.append(member.select_ids)
            else:
                new_members.append(member)

        return MongoQueryLinker(
            table_or_collection=self.table_or_collection,
            members=new_members,
        )

    def select_using_ids(self, ids):
        new_members = []
        for member in self.members:
            if hasattr(member, 'select_using_ids'):
                new_members.append(member.select_using_ids(ids))
            else:
                new_members.append(member)

        return MongoQueryLinker(
            table_or_collection=self.table_or_collection,
            members=new_members,
        )

    def select_ids_of_missing_outputs(self, key: str, model: str):
        new_members = []
        for member in self.members:
            if hasattr(member, 'select_ids_of_missing_outputs'):
                new_members.append(member.select_ids_of_missing_outputs(key, model))
        return MongoQueryLinker(
            table_or_collection=self.table_or_collection,
            members=new_members,
        )

    def select_single_id(self, id):
        assert (
            len(self.members) == 1
            and self.members[0].type == QueryType.QUERY
            and hasattr(self.members[0].name, 'select_single_id')
        )
        return MongoQueryLinker(
            table_or_collection=self.table_or_collection,
            members=[self.members[0].select_single_id(id)],
        )

    def execute(self, db):
        parent = db.databackend.get_table_or_collection(
            self.table_or_collection.identifier
        )
        for member in self.members:
            parent = member.execute(parent)
        return parent


@dc.dataclass(repr=False)
class MongoInsert(Insert):
    one: bool = False

    def execute(self, db):
        collection = db.databackend.get_table_or_collection(
            self.table_or_collection.identifier
        )
        documents = [r.encode() for r in self.documents]
        insert_result = collection.insert_many(documents, **self.kwargs)
        return insert_result.inserted_ids

    @property
    def select_table(self):
        return self.table_or_collection.find()


@dc.dataclass(repr=False)
class MongoDelete(Delete):
    one: bool = False

    @property
    def collection(self):
        return self.table_or_collection

    def execute(self, db):
        collection = db.databackend.get_table_or_collection(
            self.table_or_collection.identifier
        )
        if self.one:
            return collection.delete_one(*self.args, **self.kwargs)
        delete_result = collection.delete_many(*self.args, **self.kwargs)
        return delete_result.deleted_ids


@dc.dataclass(repr=False)
class MongoUpdate(Update):
    update: Document
    filter: t.Dict
    one: bool = False
    args: t.Sequence = dc.field(default_factory=list)
    kwargs: t.Dict = dc.field(default_factory=dict)

    @property
    def select_table(self):
        return self.table_or_collection.find()

    def execute(self, db):
        collection = db.databackend.get_table_or_collection(
            self.table_or_collection.identifier
        )

        update = self.update
        if isinstance(self.update, Document):
            update = update.encode()

        if self.one:
            id = collection.find_one(self.filter, {'_id': 1})['_id']
            collection.update_one({'_id': id}, update, *self.args, **self.kwargs)
            return [id]

        ids = [r['_id'] for r in collection.find(self.filter, {'_id': 1})]
        collection.update_many({'_id': {'$in': ids}}, update, *self.args, **self.kwargs)
        return ids


@dc.dataclass(repr=False)
class MongoReplaceOne(Update):
    replacement: Document
    filter: t.Dict
    args: t.Sequence = dc.field(default_factory=list)
    kwargs: t.Dict = dc.field(default_factory=dict)

    @property
    def collection(self):
        return self.table_or_collection

    @property
    def select_table(self):
        return self.table_or_collection.find()

    def execute(self, db):
        collection = db.databackend.get_table_or_collection(
            self.table_or_collection.identifier
        )

        replacement = self.replacement
        if isinstance(replacement, Document):
            replacement = replacement.encode()

        id = collection.find_one(self.filter, {'_id': 1})['_id']
        collection.replace_one({'_id': id}, replacement, *self.args, **self.kwargs)
        return [id]


@dc.dataclass
class ChangeStream:
    """Request a stream of changes from a db

    :param collection: The collection to perform the query on
    :param args: Positional query arguments to ``pymongo.Collection.watch``
    :param kwargs: Named query arguments to ``pymongo.Collection.watch``
    """

    collection: 'Collection'
    args: t.Sequence = dc.field(default_factory=list)
    kwargs: t.Dict = dc.field(default_factory=dict)

    def __call__(self, db):
        collection = db.databackend.get_table_or_collection(self.collection)
        return collection.watch(**self.kwargs)


@dc.dataclass(repr=False)
class Collection(TableOrCollection):
    query_components: t.ClassVar[t.Dict] = {'find': Find, 'find_one': FindOne}
    primary_id: t.ClassVar[str] = '_id'

    def get_table(self, db):
        collection = db.databackend.get_table_or_collection(self.collection.identifier)
        return collection

    def change_stream(self, *args, **kwargs):
        return ChangeStream(
            collection=self.identifier,
            args=args,
            kwargs=kwargs,
        )

    def _get_query_linker(self, members) -> MongoQueryLinker:
        return MongoQueryLinker(members=members, table_or_collection=self)

    def _get_query(
        self,
        pre_like: t.Optional[Like] = None,
        query_linker: t.Optional[QueryLinker] = None,
        post_like: t.Optional[Like] = None,
        i: int = 0,
    ) -> MongoCompoundSelect:
        return MongoCompoundSelect(
            pre_like=pre_like,
            query_linker=query_linker,
            post_like=post_like,
            table_or_collection=self,
        )

    def _delete(self, *args, one: bool = False, **kwargs):
        return MongoDelete(args=args, kwargs=kwargs, table_or_collection=self, one=one)

    def _insert(self, documents, **kwargs):
        return MongoInsert(documents=documents, kwargs=kwargs, table_or_collection=self)

    def _update(self, filter, update, *args, one: bool = False, **kwargs):
        return MongoUpdate(
            filter=filter,
            update=update,
            args=args,
            kwargs=kwargs,
            table_or_collection=self,
            one=one,
        )

    def aggregate(
        self, *args, vector_index: t.Optional[str] = None, **kwargs
    ) -> Aggregate:
        return Aggregate(
            args=args,
            kwargs=kwargs,
            vector_index=vector_index,
            table_or_collection=self,
        )

    def delete_one(self, *args, **kwargs):
        return self._delete(*args, one=True, **kwargs)

    def replace_one(self, filter, replacement, *args, **kwargs):
        return MongoReplaceOne(
            filter=filter,
            replacement=replacement,
            args=args,
            kwargs=kwargs,
            table_or_collection=self,
        )

    def update_one(self, filter, update, *args, **kwargs):
        return self._update(filter, update, *args, one=True, **kwargs)

    def update_many(self, filter, update, *args, **kwargs):
        return self._update(filter, update, *args, one=False, **kwargs)

    def insert_many(self, *args, **kwargs):
        return self._insert(*args, **kwargs)

    def insert_one(self, document, *args, **kwargs):
        return self._insert([document], *args, **kwargs)

    def like(self, r: Document, vector_index: str, n: int = 10):
        if CFG.data_backend != CFG.vector_search:
            return super().like(r=r, n=n, vector_index=vector_index)
        else:

            class LocalAggregate:
                def find(this, *args, **kwargs):
                    second_part = []
                    if args:
                        second_part.append({"$match": args[0] if args else {}})
                    if args[1:]:
                        second_part.append({'$project': args[1]})
                    pl = [
                        {
                            "$search": {
                                "knnBeta": {
                                    'like': r,
                                    "k": n,
                                }
                            }
                        },
                        *second_part,
                    ]
                    return Aggregate(
                        table_or_collection=self,
                        args=[pl],
                        vector_index=vector_index,
                    )

            return LocalAggregate()

    def model_update(  # type: ignore[override]
        self,
        db,
        ids: t.Sequence[t.Any],
        key: str,
        model: str,
        outputs: t.Sequence[t.Any],
        document_embedded: bool = True,
        flatten: bool = False,
    ):
        if key.startswith('_outputs'):
            key = key.split('.')[1]
        if not outputs:
            return
        if document_embedded:
            if flatten:
                raise AttributeError(
                    'Flattened outputs cannot be stored along with input documents.'
                    'Please use `document_embedded = False` option with flatten = True'
                )
            assert self.collection is not None
            collection = db.databackend.get_table_or_collection(self.identifier)
            collection.bulk_write(
                [
                    _UpdateOne(
                        {'_id': ObjectId(id)},
                        {'$set': {f'_outputs.{key}.{model}': outputs[i]}},
                    )
                    for i, id in enumerate(ids)
                ]
            )
        else:
            if flatten:
                bulk_docs = []
                for i, id in enumerate(ids):
                    _outputs = outputs[i]
                    if isinstance(_outputs, (list, tuple)):
                        for offset, output in enumerate(_outputs):
                            bulk_docs.append(
                                _InsertOne(
                                    {
                                        '_outputs': {key: {model: output}},
                                        '_source': ObjectId(id),
                                        '_offset': offset,
                                    }
                                )
                            )
                    else:
                        bulk_docs.append(
                            _InsertOne(
                                {
                                    '_outputs': {key: {model: _outputs}},
                                    '_source': ObjectId(id),
                                    '_offset': 0,
                                }
                            )
                        )

            else:
                bulk_docs = [
                    _InsertOne(
                        {'_id': ObjectId(id), '_outputs': {key: {model: outputs[i]}}}
                    )
                    for i, id in enumerate(ids)
                ]

            collection_name = f'_outputs.{key}.{model}'
            collection = db.databackend.get_table_or_collection(collection_name)
            collection.bulk_write(bulk_docs)
