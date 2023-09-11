from __future__ import annotations

import copy
import dataclasses as dc
import random
import typing as t

from bson import ObjectId
from overrides import override
from pymongo import UpdateOne as _UpdateOne

import superduperdb as s
from superduperdb.container.document import Document
from superduperdb.container.serializable import Serializable
from superduperdb.db.mongodb import CDC_COLLECTION_LOCKS

from ..base.cursor import SuperDuperCursor
from ..base.query import Delete, Insert, Like, Select, SelectOne, Update

if t.TYPE_CHECKING:
    from superduperdb.container.model import Model
    from superduperdb.db.base.db import DB


@dc.dataclass
class Collection(Serializable):
    """Collection wrapper"""

    #: The name of this Collection
    name: str

    @property
    def table(self) -> str:
        return self.name

    def count_documents(self, *args, **kwargs) -> 'CountDocuments':
        """
        Return a query counting the number of documents

        :param *args: Positional query arguments to
                      ``pymongo.Collection.count_documents``
        :param **kwargs: Named query arguments to
                         ``pymongo.Collection.count_documents``
        """
        return CountDocuments(collection=self, args=args, kwargs=kwargs)

    def like(self, r: Document, vector_index: str, n: int = 100) -> 'PreLike':
        """Return a query for Documents like a given one

        :param r: The document to match
        :param vector_index: The name of the vector index to use
        :param n: The number of documents to return
        """
        return PreLike(collection=self, r=r, vector_index=vector_index, n=n)

    def insert_one(
        self,
        *args,
        refresh: bool = True,
        encoders: t.Sequence = (),
        **kwargs,
    ) -> 'InsertMany':
        """Insert a single document

        :param *args: Positional query arguments to ``pymongo.Collection.insert_one``
        :param refresh: If true, refresh the underlying collection
        :param encoders: A tuple of encoders to use for encoding
        """
        return InsertMany(
            args=args[1:],
            collection=self,
            documents=[args[0]],
            encoders=encoders,
            kwargs=kwargs,
            refresh=refresh,
        )

    def insert_many(
        self,
        *args,
        refresh: bool = True,
        encoders: t.Sequence = (),
        **kwargs,
    ) -> 'InsertMany':
        """Insert many documents

        :param *args: Positional query arguments to ``pymongo.Collection.insert_many``
        :param refresh: If true, refresh the underlying collection
        :param encoders: A dictionary/ lookup of encoders to use for encoding
        :param **kwargs: Named query arguments to ``pymongo.Collection.insert_many``
        """
        return InsertMany(
            args=args[1:],
            collection=self,
            documents=args[0],
            encoders=encoders,
            kwargs=kwargs,
            refresh=refresh,
        )

    def delete_one(self, *args, **kwargs) -> 'DeleteOne':
        """
        Delete a single document

        :param *args: Positional query arguments to ``pymongo.Collection.delete_one``
        :param **kwargs: Named query arguments to ``pymongo.Collection.delete_one``
        """
        return DeleteOne(collection=self, args=args, kwargs=kwargs)

    def delete_many(self, *args, **kwargs) -> 'DeleteMany':
        """
        Delete many documents

        :param *args: Positional query arguments to ``pymongo.Collection.delete_many``
        :param **kwargs: Named query arguments to ``pymongo.Collection.delete_many``
        """
        return DeleteMany(collection=self, args=args, kwargs=kwargs)

    def update_one(self, *args, **kwargs) -> 'UpdateOne':
        """
        Update a single document

        :param *args: Positional query arguments to ``pymongo.Collection.update_one``
        :param **kwargs: Named query arguments to ``pymongo.Collection.update_one``
        """
        return UpdateOne(
            args=args,
            collection=self,
            filter=args[0],
            kwargs=kwargs,
            update=args[1],
        )

    def update_many(self, *args, **kwargs) -> 'UpdateMany':
        """
        Update many documents

        :param *args: Positional query arguments to ``pymongo.Collection.update_many``
        :param **kwargs: Named query arguments to ``pymongo.Collection.update_many``
        """
        return UpdateMany(
            args=args[2:],
            collection=self,
            filter=args[0],
            kwargs=kwargs,
            update=args[1],
        )

    def find(self, *args, **kwargs) -> 'Find':
        """
        Find many documents

        :param *args: Positional query arguments to ``pymongo.Collection.find``
        :param **kwargs: Named query arguments to ``pymongo.Collection.find``
        """
        return Find(collection=self, args=args, kwargs=kwargs)

    def find_one(self, *args, **kwargs) -> 'FindOne':
        """
        Find a single document

        :param *args: Positional query arguments to ``pymongo.Collection.find_one``
        :param **kwargs: Named query arguments to ``pymongo.Collection.find_one``
        """
        return FindOne(collection=self, args=args, kwargs=kwargs)

    def aggregate(
        self, *args, vector_index: t.Optional[str] = None, **kwargs
    ) -> 'Aggregate':
        """
        Prepare an aggregate query

        :param *args: Positional query arguments to ``pymongo.Collection.aggregate``
        :param vector_index: The name of the vector index to use
        :param **kwargs: Named query arguments to ``pymongo.Collection.aggregate``
        """
        return Aggregate(
            collection=self, args=args, vector_index=vector_index, kwargs=kwargs
        )

    def replace_one(self, *args, **kwargs) -> 'ReplaceOne':
        """
        Replace a single document

        :param *args: Positional query arguments to ``pymongo.Collection.replace_one``
        :param **kwargs: Named query arguments to ``pymongo.Collection.replace_one``
        """
        return ReplaceOne(
            args=args,
            collection=self,
            filter=args[0],
            kwargs=kwargs,
            update=args[1],
        )

    def change_stream(self, *args, **kwargs) -> 'ChangeStream':
        """
        Listen to a stream of changes from a collection

        :param *args: Positional query arguments to ``pymongo.Collection.watch``
        :param **kwargs: Named query arguments to ``pymongo.Collection.watch``
        """
        return ChangeStream(collection=self, args=args, kwargs=kwargs)


@dc.dataclass
class ReplaceOne(Update):
    """A query that replaces one record in a collection"""

    #: The collection to perform the query on
    collection: Collection

    #: Filter results by this dictionary
    filter: t.Dict

    #: The update to apply
    update: Document

    #: Positional query arguments to ``pymongo.Collection.replace_one``
    args: t.Sequence = dc.field(default_factory=list)

    #: Named query arguments to ``pymongo.Collection.replace_one``
    kwargs: t.Dict = dc.field(default_factory=dict)

    #: If true, refresh the model outputs
    refresh: bool = True

    #: If true, print more logging
    verbose: bool = True

    type_id: t.Literal['mongodb.ReplaceOne'] = 'mongodb.ReplaceOne'

    @override
    def __call__(self, db: DB):
        update = self.update.encode()
        return db.db[self.collection.name].replace_one(
            self.filter, update, *self.args[2:], **self.kwargs
        )

    @property
    @override
    def select(self) -> 'Find':
        return Find(like_parent=t.cast(PreLike, self.collection), args=[self.args[0]])

    @property
    @override
    def select_ids(self) -> 'Find':
        return self.select.select_ids


@dc.dataclass
class PreLike(Like):
    """A query that returns documents like another document"""

    #: The collection to perform the query on
    collection: Collection

    #: The document to match
    r: Document

    #: The name of the vector index to use
    vector_index: str

    #: The number of documents to return
    n: int = 100

    type_id: t.Literal['mongodb.PreLike'] = 'mongodb.PreLike'

    @property
    def table(self) -> str:
        """Extracts the table collection from the object"""
        return self.collection.name

    def find(self, *args, **kwargs) -> t.Union[Find, Aggregate]:
        """
        Find documents like this one filtered using *args and **kwargs on
        the basis of ``self.vector_index``

        :param args: Positional arguments to pass to the find query
        :param kwargs: Named arguments to pass to the find query
        """
        if s.CFG.vector_search == s.CFG.data_backend:
            second_part = []
            if args:
                second_part.append({"$match": args[0] if args else {}})
            if args[1:]:
                second_part.append({'$project': args[1]})
            pl = [
                {
                    "$search": {
                        "knnBeta": {
                            'like': self.r,
                            "k": self.n,
                        }
                    }
                },
                *second_part,
            ]
            return Aggregate(
                collection=self.collection,
                args=[pl],
                **kwargs,
                vector_index=self.vector_index,
            )
        else:
            return Find(like_parent=self, args=args, kwargs=kwargs)

    def find_one(self, *args, **kwargs) -> FindOne:
        """
        Find a document like this one on the basis of ``self.vector_index``

        :param **args: Positional arguments to ``pymongo.Collection.find_one``
        :param **kwargs: Named arguments to ``pymongo.Collection.find_one``
        """
        return FindOne(like_parent=self, args=args, kwargs=kwargs)

    @override
    def __call__(self, db: DB):
        ids, scores = db._select_nearest(
            like=self.r, vector_index=self.vector_index, n=self.n
        )
        cursor = db.db[self.collection.name].find(
            {'_id': {'$in': [ObjectId(_id) for _id in ids]}}
        )
        return SuperDuperCursor(
            raw_cursor=cursor,
            scores=dict(zip(ids, scores)),
            id_field='_id',
            encoders=db.encoders,
        )


@dc.dataclass
class Find(Select):
    """A query to find documents"""

    id_field: t.ClassVar[str] = '_id'

    #: The collection to perform the query on
    collection: t.Optional[Collection] = None

    #: The parent query to use for the like query (if applicable)
    like_parent: t.Optional[PreLike] = None

    #: Positional query arguments to ``pymongo.Collection.find``
    args: t.Sequence = dc.field(default_factory=list)

    #: Named query arguments to ``pymongo.Collection.find``
    kwargs: t.Dict = dc.field(default_factory=dict)

    type_id: t.Literal['mongodb.Find'] = 'mongodb.Find'

    def select_single_id(self, id, db, encoders=()):
        id = ObjectId(id)
        return Document(
            Document.decode(
                db.db[self.collection.name].find_one({'_id': id}), encoders=encoders
            )
        )

    @property
    def parent(self) -> t.Union[Collection, PreLike]:
        """Return whichever of self.like_parent or self.collection isn't None"""
        if self.like_parent is not None:
            if self.collection is None:
                return self.like_parent
        elif self.collection is not None:
            return self.collection
        raise ValueError('Exactly one of "like_parent" or "collection" must be set')

    @property
    def table(self) -> str:
        """Extracts the table collection from the object"""
        return self.parent.table

    def limit(self, n: int) -> 'Limit':
        """
        Return a new query with the number of documents limited to n

        :param n: The number of documents to return
        """
        return Limit(parent=self, n=n)  # type

    def like(
        self, r: Document, vector_index: str = '', n: int = 100, max_ids: int = 1000
    ) -> PostLike:
        """Retrurn a query for documents like this one

        :param vector_index: The name of the vector index to use
        :param n: The number of documents to return
        :param max_ids: The maximum number of ids to use for the like query
        """
        return PostLike(
            find_parent=self, r=r, n=n, max_ids=max_ids, vector_index=vector_index
        )

    def add_fold(self, fold: str) -> Find:
        """Create a select which selects the same data, but additionally restricts to
        the fold specified

        :param fold: possible values {'train', 'valid'}
        """
        args = []
        try:
            args.append(self.args[0])
        except IndexError:
            args.append({})
        args.extend(self.args[1:])
        args[0]['_fold'] = fold
        return Find(
            like_parent=self.like_parent,
            collection=self.collection,
            args=args,
            kwargs=self.kwargs,
        )

    @property
    @override
    def select_ids(self) -> Find:
        try:
            filter = self.args[0]
        except IndexError:
            filter = {}
        return Find(
            collection=self.collection,
            like_parent=self.like_parent,
            args=[filter, {'_id': 1}],
        )

    @override
    def select_using_ids(self, ids: t.Sequence[str]) -> Find:
        args = [*self.args, {}, {}][:2]
        args[0] = {'_id': {'$in': [ObjectId(_id) for _id in ids]}, **args[0]}

        return Find(
            collection=self.collection,
            like_parent=self.like_parent,
            args=args,
            kwargs=self.kwargs,
        )

    def select_ids_of_missing_outputs(self, key: str, model: str):
        if self.args:
            args = [
                {'$and': [self.args[0], {f'_outputs.{key}.{model}': {'$exists': 0}}]},
                *self.args[1],
            ]
        else:
            args = [{f'_outputs.{key}.{model}': {'$exists': 0}}]
        return Find(
            collection=self.collection,
            like_parent=self.like_parent,
            args=args,
            kwargs=self.kwargs,
        )

    def featurize(self, features: t.Dict[str, str]) -> 'Featurize':
        """Extract a feature vector

        :param features: The features to extract
        """
        return Featurize(parent=self, features=features)

    def get_ids(self, db: DB) -> t.Sequence[str]:
        """Get a list of matching IDs

        :param db: The db to query
        """
        args: t.List[t.Dict[str, t.Any]] = [{}, {}]
        args[: len(self.args)] = self.args
        args[1] = {'_id': 1}  # What happens to the data overwritten?
        cursor = Find(
            collection=self.collection,
            like_parent=self.like_parent,
            args=args,
            kwargs=self.kwargs,
        )(db)
        return [r['_id'] for r in cursor]

    @override
    def model_update(
        self,
        db: DB,
        ids: t.Sequence[t.Any],
        key: str,
        model: str,
        outputs: t.Sequence[t.Any],
    ) -> None:
        """
        Add the outputs of a model to the database

        :param db: The DB instance to use
        :param ids: The ids of the documents to update
        :param key: The key to update
        :param model: The model to update
        :param outputs: The outputs to add
        """

        if key.startswith('_outputs'):
            key = key.split('.')[1]
        if not outputs:
            return
        assert self.collection is not None
        db.db[self.collection.name].bulk_write(
            [
                _UpdateOne(
                    {'_id': ObjectId(id)},
                    {'$set': {f'_outputs.{key}.{model}': outputs[i]}},
                )
                for i, id in enumerate(ids)
            ]
        )

    def model_cleanup(self, db: DB, model: Model, key: str) -> None:
        """Clean up a model after the computation is done

        :param db: The db to use
        :param model: The model to clean
        :param key: The key to clean
        """
        assert self.collection is not None
        db.db[self.collection.name].update_many(
            {}, {'$unset': {f'_outputs.{key}.{model}': 1}}
        )

    @property
    @override
    def select_table(self) -> 'Find':
        if isinstance(self.parent, PreLike):
            return Find(like_parent=self.parent)
        raise TypeError('Expected PreLike')

    def download_update(self, db: DB, id: str, key: str, bytes: bytearray) -> None:
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
        return UpdateOne(
            collection=self.collection,
            filter={'_id': id},
            update=update,
        )(db)

    @override
    def __call__(self, db: DB) -> SuperDuperCursor:
        if isinstance(self.parent, Collection):
            assert self.collection, "Please set a valid collection name"
            cursor = db.db[self.collection.name].find(*self.args, **self.kwargs)
        elif isinstance(self.parent, Like):
            intermediate = self.parent(db)
            ids = [ObjectId(r['_id']) for r in intermediate]
            try:
                filter = self.args[0]
            except IndexError:
                filter = {}
            filter = {'$and': [filter, {'_id': {'$in': ids}}]}
            assert self.like_parent is not None
            cursor = db.db[self.like_parent.collection.name].find(
                filter, *self.args[1:], **self.kwargs
            )
        else:
            raise NotImplementedError
        return SuperDuperCursor(raw_cursor=cursor, id_field='_id', encoders=db.encoders)


@dc.dataclass
class CountDocuments(Select):
    """
    A query to count documents
    """

    #: The collection to perform the query on
    collection: t.Optional[Collection] = None

    #: The parent query to use for the like query (if applicable)
    like_parent: t.Optional[PreLike] = None

    #: Positional query arguments to ``pymongo.Collection.count_documents``
    args: t.Sequence = dc.field(default_factory=list)

    #: Named query arguments to ``pymongo.Collection.count_documents``
    kwargs: t.Dict = dc.field(default_factory=dict)

    #: Uniquely identifies serialized elements of this class
    type_id: t.Literal['mongodb.CountDocuments'] = 'mongodb.CountDocuments'

    @override
    def __call__(self, db: DB):
        assert self.collection is not None
        return db.db[self.collection.name].count_documents(*self.args, **self.kwargs)


@dc.dataclass
class FeaturizeOne(SelectOne):
    """A query to feature just one document"""

    #: The features to extract
    features: t.Dict[str, str]

    #: The parent query to use for the like query (if applicable)
    parent_find_one: t.Any = None

    type_id: t.Literal['mongodb.FeaturizeOne'] = 'mongodb.FeaturizeOne'

    @override
    def __call__(self, db: DB):
        r = self.parent_find_one(db)
        r = SuperDuperCursor.add_features(r.content, self.features)
        return Document(r)


@dc.dataclass
class FindOne(SelectOne):
    """A query to find one document."""

    # TODO add this to a base class (should be created)
    id_field: t.ClassVar[str] = '_id'

    #: The collection to perform the query on
    collection: t.Optional[Collection] = None

    #: The parent query to use for the like query (if applicable)
    like_parent: t.Optional[PreLike] = None

    #: Positional query arguments to ``pymongo.Collection.find_one``
    args: t.Sequence = dc.field(default_factory=list)

    #: Named query arguments to ``pymongo.Collection.find_one``
    kwargs: t.Dict = dc.field(default_factory=dict)

    type_id: t.Literal['mongodb.FindOne'] = 'mongodb.FindOne'

    @override
    def __call__(self, db: DB):
        find = Find(
            collection=self.collection,
            like_parent=self.like_parent,
            args=self.args,
            kwargs=self.kwargs,
        )
        return next(find(db=db))

    def featurize(self, features: t.Dict[str, str]) -> 'FeaturizeOne':
        """Extract a feature vector

        :param features: The features to extract
        """
        return FeaturizeOne(parent_find_one=self, features=features)


@dc.dataclass
class Aggregate(Select):
    """An aggregate query"""

    #: The collection to perform the query on
    collection: Collection

    #: Positional query arguments to ``pymongo.Collection.aggregate``
    args: t.Sequence = dc.field(default_factory=list)

    #: Named query arguments to ``pymongo.Collection.aggregate``
    kwargs: t.Dict = dc.field(default_factory=dict)

    #: The name of the vector index to use
    vector_index: t.Optional[str] = None

    type_id: t.Literal['mongodb.Aggregate'] = 'mongodb.Aggregate'

    @staticmethod
    def _replace_document_with_vector(step, vector_index, db):
        step = copy.deepcopy(step)
        assert "like" in step['$search']['knnBeta']
        vector_index = db.vector_indices[vector_index]
        models, keys = vector_index.models_keys
        step['$search']['knnBeta']['vector'], model, key = vector_index.get_vector(
            like=step['$search']['knnBeta']['like'],
            models=models,
            keys=keys,
            db=db,
        )
        step['$search']['knnBeta']['path'] = f'_outputs.{key}.{model}'
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

    @override
    def __call__(self, db: DB) -> SuperDuperCursor:
        args = self.args
        if self.vector_index is not None:
            args = [
                self._prepare_pipeline(args[0], db, self.vector_index),
                *self.args[1:],
            ]
        return SuperDuperCursor(
            id_field='_id',
            raw_cursor=db.db[self.collection.name].aggregate(*args, **self.kwargs),
            encoders=db.encoders,
        )


@dc.dataclass
class DeleteOne(Delete):
    """A query to delete a single document"""

    #: The collection to perform the query on
    collection: Collection

    #: Positional query arguments to ``pymongo.Collection.delete_one``
    args: t.Sequence = dc.field(default_factory=list)

    #: Named query arguments to ``pymongo.Collection.delete_one``
    kwargs: t.Dict = dc.field(default_factory=dict)

    type_id: t.Literal['mongodb.DeleteOne'] = 'mongodb.DeleteOne'

    @override
    def __call__(self, db: DB):
        return db.db[self.collection.name].delete_one(*self.args, **self.kwargs)


@dc.dataclass
class DeleteMany(Delete):
    """A query to delete many documents"""

    #: The collection to perform the query on
    collection: Collection

    #: Positional query arguments to ``pymongo.Collection.delete_many``
    args: t.Sequence = dc.field(default_factory=list)

    #: Named query arguments to ``pymongo.Collection.delete_many``
    kwargs: t.Dict = dc.field(default_factory=dict)

    type_id: t.Literal['mongodb.DeleteMany'] = 'mongodb.DeleteMany'

    @override
    def __call__(self, db: DB):
        return db.db[self.collection.name].delete_many(*self.args, **self.kwargs)


@dc.dataclass
class UpdateOne(Update):
    """A query that updates one document"""

    #: The collection to perform the query on
    collection: Collection

    #: The update to apply
    update: t.Any

    #: Filter results by this dictionary
    filter: t.Dict

    #: If true, refresh the underlying collection
    refresh: bool = True

    #: If true, print more logging
    verbose: bool = True

    #: Positional query arguments to ``pymongo.Collection.update_one``
    args: t.Sequence = dc.field(default_factory=list)

    #: Named query arguments to ``pymongo.Collection.update_one``
    kwargs: t.Dict = dc.field(default_factory=dict)

    #: Uniquely identifies serialized elements of this class
    type_id: t.Literal['mongodb.UpdateOne'] = 'mongodb.UpdateOne'

    @override
    def __call__(self, db: DB):
        return db.db[self.collection.name].update_one(
            self.filter,
            self.update,
            *self.args,
            **self.kwargs,
        )

    @property
    @override
    def select(self):
        return Find(collection=self.collection, args=[self.args[0]])


@dc.dataclass
class UpdateMany(Update):
    """A query that updates one document"""

    #: The collection to perform the query on
    collection: Collection

    #: Filter results by this dictionary
    filter: t.Dict
    update: Document

    #: If true, refresh the underlying collection
    refresh: bool = True

    #: If true, print more logging
    verbose: bool = True

    #: Positional query arguments
    args: t.Sequence = dc.field(default_factory=list)

    #: Named query arguments
    kwargs: t.Dict = dc.field(default_factory=dict)

    #: Uniquely identifies serialized elements of this class
    type_id: t.Literal['mongodb.UpdateMany'] = 'mongodb.UpdateMany'

    @override
    def __call__(self, db: DB):
        to_update = self.update.encode()
        ids = [
            r['_id'] for r in db.db[self.collection.name].find(self.filter, {'_id': 1})
        ]
        out = db.db[self.collection.name].update_many(
            {'_id': {'$in': ids}},
            to_update,
            *self.args[2:],
            **self.kwargs,
        )
        graph = None
        if self.refresh and not CDC_COLLECTION_LOCKS.get(self.collection.name, False):
            graph = db.refresh_after_update_or_insert(
                query=self, ids=ids, verbose=self.verbose
            )
        return out, graph

    @property
    @override
    def select_table(self) -> Find:
        return Find(collection=self.collection, args=[{}])

    @property
    @override
    def select(self):
        return Find(like_parent=self.args[0])


@dc.dataclass
class InsertMany(Insert):
    """A query that inserts many documents"""

    #: The collection to perform the query on
    collection: Collection

    #: The documents to insert
    documents: t.List[Document] = dc.field(default_factory=list)

    #: If true, refresh the underlying collection
    refresh: bool = True

    #: If true, print more logging
    verbose: bool = True

    #: Positional query arguments to ``pymongo.Collection.insert_many``
    args: t.Sequence = dc.field(default_factory=list)

    #: Named query arguments to ``pymongo.Collection.insert_many``
    kwargs: t.Dict = dc.field(default_factory=dict)

    #: Encoders to use
    encoders: t.Sequence = dc.field(default_factory=list)

    type_id: t.Literal['mongodb.InsertMany'] = 'mongodb.InsertMany'

    @property
    @override
    def table(self) -> str:
        return self.collection.name

    @property
    @override
    def select_table(self):
        return Find(collection=self.collection)

    def select_using_ids(self, ids: t.Sequence[str]) -> t.Any:
        """Create a select using the same Serializable, subset to the specified ids

        :param ids: string ids to which subsetting should occur
        """
        return Find(collection=self.collection, args=[{'_id': {'$in': ids}}])

    @override
    def __call__(self, db: DB):
        valid_prob = self.kwargs.get('valid_prob', 0.05)
        for e in self.encoders:
            db.add(e)
        documents = [r.encode() for r in self.documents]
        for r in documents:
            if '_fold' in r:
                continue
            if random.random() < valid_prob:
                r['_fold'] = 'valid'
            else:
                r['_fold'] = 'train'
        output = db.db[self.collection.name].insert_many(
            documents,
            *self.args,
            **self.kwargs,
        )
        graph = None
        if self.refresh and not CDC_COLLECTION_LOCKS.get(self.collection.name, False):
            graph = db.refresh_after_update_or_insert(
                query=self,
                ids=output.inserted_ids,
                verbose=self.verbose,
            )
        return output, graph


@dc.dataclass
class PostLike(Select):
    """Find documents like this one"""

    #: The parent query to use for the like query (if applicable)
    find_parent: t.Optional[Find]

    #: The document to match
    r: Document

    #: The name of the vector index to use
    vector_index: str

    #: The number of documents to return
    n: int = 100

    #: The maximum number of ids to use for the like query
    max_ids: int = 1000

    type_id: t.Literal['mongodb.PostLike'] = 'mongodb.PostLike'

    class Config:
        arbitrary_types_allowed = True

    @override
    def __call__(self, db: DB) -> SuperDuperCursor:
        assert self.find_parent is not None
        cursor = self.find_parent.select_ids.limit(self.max_ids)(db)
        ids = [r['_id'] for r in cursor]
        ids, scores = db._select_nearest(
            like=self.r,
            vector_index=self.vector_index,
            n=self.n,
            ids=[str(_id) for _id in ids],
        )
        ids = [ObjectId(_id) for _id in ids]
        return Find(
            collection=self.find_parent.collection,
            args=[{'_id': {'$in': ids}}],
        )(db)


@dc.dataclass
class Featurize(Select):
    """A feature-extraction query"""

    #: The features to extract
    features: t.Dict[str, str]

    #: The parent query to use for the like query (if applicable)
    parent: Find

    type_id: t.Literal['mongodb.Featurize'] = 'mongodb.Featurize'

    @property
    @override
    def select_table(self):
        return self.parent.select_table

    def get_ids(self, db: DB) -> t.Sequence[str]:
        """Get a list of matching IDs

        :param db: The db to query
        """
        assert isinstance(self.parent, Find)
        return self.parent.get_ids(db)

    @override
    def is_trivial(self) -> bool:
        return self.parent.is_trivial()

    @property
    def select_ids(self):
        return t.cast(Select, self.parent.select_ids)

    def add_fold(self, fold: str) -> Featurize:
        """Create a select which selects the same data, but additionally restricts to
        the fold specified

        :param fold: possible values {'train', 'valid'}
        """
        folded = self.parent.add_fold(fold)
        assert isinstance(folded, (Find, FindOne))
        folded.featurize(self.features)
        return self.parent.add_fold(fold).featurize(self.features)

    @override
    def select_using_ids(self, ids: t.Sequence[str]) -> t.Any:
        return self.parent.select_using_ids(ids=ids).featurize(features=self.features)

    def select_ids_of_missing_outputs(self, key: str, model: str) -> Select:
        parent = self.parent.select_ids_of_missing_outputs(key, model)
        assert isinstance(parent, Find)
        return Featurize(features=self.features, parent=parent)

    @override
    def model_update(
        self,
        db: DB,
        ids: t.Sequence[t.Any],
        key: str,
        model: str,
        outputs: t.Sequence[t.Any],
    ) -> None:
        self.parent.model_update(db=db, ids=ids, key=key, model=model, outputs=outputs)

    @override
    def __call__(self, db: DB) -> t.Any:
        if isinstance(self.parent, (Find, Like, Limit)):
            out = self.parent(db)
            out.features = self.features
            return out
        else:
            cursor = self.parent(db)
            # TODO: r is a SuperDuperCursor, but that has no .content attribute
            d = SuperDuperCursor.add_features(cursor.content, self.features)
            return Document(d)


@dc.dataclass
class Limit(Select):
    """A query that limits the number of returned documents"""

    #: The number of documents to return
    n: int

    #: The parent query to use for the like query (if applicable)
    parent: t.Union[Featurize, Find, PreLike, PostLike]

    type_id: t.Literal['mongodb.Limit'] = 'mongodb.Limit'

    @override
    def __call__(self, db: DB) -> SuperDuperCursor:
        return self.parent(db).limit(self.n)


@dc.dataclass
class ChangeStream:
    """Request a stream of changes from a db"""

    #: The collection to perform the query on
    collection: Collection

    #: Positional query arguments to ``pymongo.Collection.watch``
    args: t.Sequence = dc.field(default_factory=list)

    #: Named query arguments to ``pymongo.Collection.watch``
    kwargs: t.Dict = dc.field(default_factory=dict)

    @override
    def __call__(self, db: DB):
        collection = db.db[self.collection.name]
        return collection.watch(**self.kwargs)


all_items = {
    'Aggregate': Aggregate,
    'ChangeStream': ChangeStream,
    'CountDocuments': CountDocuments,
    'DeleteMany': DeleteMany,
    'DeleteOne': DeleteOne,
    'Featurize': Featurize,
    'FeaturizeOne': FeaturizeOne,
    'Find': Find,
    'FindOne': FindOne,
    'InsertMany': InsertMany,
    'Limit': Limit,
    'PostLike': PostLike,
    'PreLike': PreLike,
    'ReplaceOne': ReplaceOne,
    'UpdateOne': UpdateOne,
    'UpdateMany': UpdateMany,
}
