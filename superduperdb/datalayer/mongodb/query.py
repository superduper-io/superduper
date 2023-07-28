from __future__ import annotations

import dataclasses as dc
import random
import typing as t

from bson import ObjectId
from overrides import override
from pymongo import UpdateOne as _UpdateOne

import superduperdb as s
from superduperdb.core.document import Document
from superduperdb.core.serializable import Serializable

from ..base.cursor import SuperDuperCursor
from ..base.query import Delete, Insert, Like, Select, SelectOne, Update

if t.TYPE_CHECKING:
    from superduperdb.core.model import Model
    from superduperdb.datalayer.base.datalayer import Datalayer


@dc.dataclass
class Collection(Serializable):
    """A collection of query results"""

    #: The name of this Collection
    name: str

    @property
    def table(self) -> str:
        return self.name

    def count_documents(self, *args, **kwargs) -> 'CountDocuments':
        """Return a query counting the number of documents"""
        return CountDocuments(collection=self, args=args, kwargs=kwargs)

    def like(self, r: Document, vector_index: str, n: int = 100) -> 'PreLike':
        """Return a query for Documents like a given one

        :param r: The document to match
        :param vector_index: The name of the vector index to use
        :param n: The number of documents to return
        """
        return PreLike(collection=self, r=r, vector_index=vector_index, n=n)

    def insert_one(
        self, *args, refresh: bool = True, encoders=(), **kwargs
    ) -> 'InsertMany':
        """Insert a single document

        :param refresh: If true, refresh the underlying collection
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
        self, *args, refresh: bool = True, encoders=(), **kwargs
    ) -> 'InsertMany':
        """Insert many documents

        :param refresh: If true, refresh the underlying collection
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
        """Delete a single document"""
        return DeleteOne(collection=self, args=args, kwargs=kwargs)

    def delete_many(self, *args, **kwargs) -> 'DeleteMany':
        """Delete many documents"""
        return DeleteMany(collection=self, args=args, kwargs=kwargs)

    def update_one(self, *args, **kwargs) -> 'UpdateOne':
        """Update a single document"""
        return UpdateOne(
            args=args,
            collection=self,
            filter=args[0],
            kwargs=kwargs,
            update=args[1],
        )

    def update_many(self, *args, **kwargs) -> 'UpdateMany':
        """Update many documents"""
        return UpdateMany(
            args=args[2:],
            collection=self,
            filter=args[0],
            kwargs=kwargs,
            update=args[1],
        )

    def find(self, *args, **kwargs) -> 'Find':
        """Find many documents"""
        return Find(collection=self, args=args, kwargs=kwargs)

    def find_one(self, *args, **kwargs) -> 'FindOne':
        """Find a single document"""
        return FindOne(collection=self, args=args, kwargs=kwargs)

    def aggregate(self, *args, **kwargs) -> 'Aggregate':
        """Prepare an aggregate query"""
        return Aggregate(collection=self, args=args, kwargs=kwargs)

    def replace_one(self, *args, **kwargs) -> 'ReplaceOne':
        """Replace a single document"""
        return ReplaceOne(
            args=args,
            collection=self,
            filter=args[0],
            kwargs=kwargs,
            update=args[1],
        )

    def change_stream(self, *args, **kwargs) -> 'ChangeStream':
        """Listen to a stream of change from a datalayer"""
        return ChangeStream(collection=self, args=args, kwargs=kwargs)


@dc.dataclass
class ReplaceOne(Update):
    """A query that replaces one record in a collection"""

    #: The collection to perform the query on
    collection: Collection

    #: Filter results by this dictionary
    filter: t.Dict
    update: Document

    #: Positional query arguments
    args: t.Sequence = dc.field(default_factory=list)

    #: Named query arguments
    kwargs: t.Dict = dc.field(default_factory=dict)

    #: If true, refresh the underlying collection
    refresh: bool = True

    #: If true, print more logging
    verbose: bool = True

    #: Uniquely identifies serialized elements of this class
    type_id: t.Literal['mongodb.ReplaceOne'] = 'mongodb.ReplaceOne'

    @override
    def __call__(self, db: Datalayer):
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

    #: The document to match
    r: Document

    #: The name of the vector index to use
    vector_index: str

    #: The collection to perform the query on
    collection: Collection

    #: The number of documents to return
    n: int = 100

    #: Uniquely identifies serialized elements of this class
    type_id: t.Literal['mongodb.PreLike'] = 'mongodb.PreLike'

    @property
    def table(self) -> str:
        """Extracts the table collection from the object"""
        return self.collection.name

    def find(self, *args, **kwargs) -> Find:
        """Find documents like this one"""
        return Find(like_parent=self, args=args, kwargs=kwargs)

    def find_one(self, *args, **kwargs) -> FindOne:
        """Find documents like this one"""
        return FindOne(like_parent=self, args=args, kwargs=kwargs)

    @override
    def __call__(self, db: Datalayer):
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
    """A query to find couments"""

    #:
    id_field: t.ClassVar[str] = '_id'

    #: The collection to perform the query on
    collection: t.Optional[Collection] = None
    like_parent: t.Optional[PreLike] = None

    #: Positional query arguments
    args: t.Sequence = dc.field(default_factory=list)

    #: Named query arguments
    kwargs: t.Dict = dc.field(default_factory=dict)

    #: Uniquely identifies serialized elements of this class
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

    def count(self, *args, **kwargs) -> 'Count':
        """Return a new Count query from this Find"""
        return Count(self, *args, **kwargs)  # type

    def like(
        self, r: Document, vector_index: str = '', n: int = 100, max_ids: int = 1000
    ) -> PostLike:
        """Retrurn a query for documents like this one

        :param vector_index: The name of the vector index to use
        :param n: The number of documents to return
        """
        return PostLike(
            find_parent=self, r=r, n=n, max_ids=max_ids, vector_index=vector_index
        )

    # @override
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

    def featurize(self, features: t.Dict[str, str]) -> 'Featurize':
        """Extract a feature vector

        :param features: The features to extract
        """
        return Featurize(parent=self, features=features)

    def get_ids(self, db: Datalayer) -> t.Sequence[str]:
        """Get a list of matching IDs

        :param db: The datalayer to query
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
        db: Datalayer,
        ids: t.Sequence[t.Any],
        key: str,
        model: Model,
        outputs: t.Sequence[t.Any],
    ) -> None:
        if key.startswith('_outputs'):
            key = key.split('.')[1]
        db.db[self.collection.name].bulk_write(
            [
                _UpdateOne(
                    {'_id': ObjectId(id)},
                    {'$set': {f'_outputs.{key}.{model}': outputs[i]}},
                )
                for i, id in enumerate(ids)
            ]
        )

    def model_cleanup(self, db: Datalayer, model: Model, key: str) -> None:
        """Clean up a model after the computation is done

        :param db: The datalayer to use
        :param model: The model to clean
        :param key: The key to clean
        """
        db.db[self.collection.name].update_many(
            {}, {'$unset': {f'_outputs.{key}.{model}': 1}}
        )

    @property
    @override
    def select_table(self) -> 'Find':
        if isinstance(self.parent, PreLike):
            return Find(like_parent=self.parent)
        raise TypeError('Expected PreLike')

    def download_update(self, db: Datalayer, id: str, key: str, bytes) -> None:
        """Download an updated version of this query

        TODO: improve this comment.

        :param db: The datalayer to query
        :param id: The id to filter on
        :param key:
        :param b: The bytes to update
        """
        if self.collection is None:
            raise ValueError('collection cannot be None')

        update = {'$set': {f'{key}._content.bytes': bytes}}
        return UpdateOne(
            collection=self.collection,
            filter={'_id': id},
            update=update,  # type: ignore[arg-type]
        )(db)

    @override
    def __call__(self, db: Datalayer) -> SuperDuperCursor:
        if isinstance(self.parent, Collection):
            cursor = db.db[self.collection.name].find(*self.args, **self.kwargs)
        elif isinstance(self.parent, Like):
            intermediate = self.parent(db)
            ids = [ObjectId(r['_id']) for r in intermediate]
            try:
                filter = self.args[0]
            except IndexError:
                filter = {}
            filter = {'$and': [filter, {'_id': {'$in': ids}}]}
            cursor = db.db[self.like_parent.collection.name].find(
                filter, *self.args[1:], **self.kwargs
            )
        else:
            raise NotImplementedError
        return SuperDuperCursor(raw_cursor=cursor, id_field='_id', encoders=db.encoders)


@dc.dataclass
class CountDocuments(Find):
    """A query to count documents"""

    #: The collection to perform the query on
    collection: t.Optional[Collection] = None
    like_parent: t.Optional[PreLike] = None

    #: Positional query arguments
    args: t.Sequence = dc.field(default_factory=list)

    #: Named query arguments
    kwargs: t.Dict = dc.field(default_factory=dict)

    #: Uniquely identifies serialized elements of this class
    type_id: t.Literal[
        'mongodb.CountDocuments'
    ] = 'mongodb.CountDocuments'  # type: ignore[assignment]

    @override
    def __call__(self, db: Datalayer):
        return db.db[self.collection.name].count_documents(*self.args, **self.kwargs)


@dc.dataclass
class FeaturizeOne(SelectOne):
    """A query to feature just one document"""

    features: t.Dict[str, str]
    parent_find_one: t.Any = None

    #: Uniquely identifies serialized elements of this class
    type_id: t.Literal['mongodb.FeaturizeOne'] = 'mongodb.FeaturizeOne'

    @override
    def __call__(self, db: Datalayer):
        r = self.parent_find_one(db)
        r = SuperDuperCursor.add_features(r.content, self.features)
        return Document(r)


@dc.dataclass
class FindOne(SelectOne):
    #:
    id_field: t.ClassVar[str] = '_id'

    #: The collection to perform the query on
    collection: t.Optional[Collection] = None

    #:
    like_parent: t.Optional[PreLike] = None

    #: Positional query arguments
    args: t.Sequence = dc.field(default_factory=list)

    #: Named query arguments
    kwargs: t.Dict = dc.field(default_factory=dict)

    #: Uniquely identifies serialized elements of this class
    type_id: t.Literal['mongodb.FindOne'] = 'mongodb.FindOne'

    @override
    def __call__(self, db: Datalayer):
        if self.collection is not None:
            return SuperDuperCursor.wrap_document(
                db.db[self.collection.name].find_one(*self.args, **self.kwargs),
                encoders=db.encoders,
            )
        else:
            parent_cursor = self.like_parent(db)  # type: ignore[misc]
            ids = [r['_id'] for r in parent_cursor]
            filter = self.args[0] if self.args else {}
            filter['_id'] = {'$in': ids}
            r = db.db[self.like_parent.collection.name].find_one(
                filter,
                *self.args[1:],
                **self.kwargs,
            )
            return Document(Document.decode(r, encoders=db.encoders))

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

    #: Positional query arguments
    args: t.Sequence = dc.field(default_factory=list)

    #: Named query arguments
    kwargs: t.Dict = dc.field(default_factory=dict)

    #: Uniquely identifies serialized elements of this class
    type_id: t.Literal['mongodb.Aggregate'] = 'mongodb.Aggregate'

    @override
    def __call__(self, db: Datalayer):
        return SuperDuperCursor(
            id_field='_id',
            raw_cursor=db.db[self.collection.name].aggregate(*self.args, **self.kwargs),
            encoders=db.encoders,
        )


@dc.dataclass
class DeleteOne(Delete):
    """A query to delete a single document"""

    #: The collection to perform the query on
    collection: Collection

    #: Positional query arguments
    args: t.Sequence = dc.field(default_factory=list)

    #: Named query arguments
    kwargs: t.Dict = dc.field(default_factory=dict)

    #: Uniquely identifies serialized elements of this class
    type_id: t.Literal['mongodb.DeleteOne'] = 'mongodb.DeleteOne'

    @override
    def __call__(self, db: Datalayer):
        return db.db[self.collection.name].delete_one(*self.args, **self.kwargs)


@dc.dataclass
class DeleteMany(Delete):
    """A query to delete many documents"""

    #: The collection to perform the query on
    collection: Collection

    #: Positional query arguments
    args: t.Sequence = dc.field(default_factory=list)

    #: Named query arguments
    kwargs: t.Dict = dc.field(default_factory=dict)

    #: Uniquely identifies serialized elements of this class
    type_id: t.Literal['mongodb.DeleteMany'] = 'mongodb.DeleteMany'

    @override
    def __call__(self, db: Datalayer):
        return db.db[self.collection.name].delete_many(*self.args, **self.kwargs)


@dc.dataclass
class UpdateOne(Update):
    """A query that updates one document"""

    #: The collection to perform the query on
    collection: Collection
    update: Document

    #: Filter results by this dictionary
    filter: t.Dict

    #: If true, refresh the underlying collection
    refresh: bool = True

    #: If true, print more logging
    verbose: bool = True

    #: Positional query arguments
    args: t.Sequence = dc.field(default_factory=list)

    #: Named query arguments
    kwargs: t.Dict = dc.field(default_factory=dict)

    #: Uniquely identifies serialized elements of this class
    type_id: t.Literal['mongodb.UpdateOne'] = 'mongodb.UpdateOne'

    @override
    def __call__(self, db: Datalayer):
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
    def __call__(self, db: Datalayer):
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
        if self.refresh and not s.CFG.cdc:
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

    documents: t.List[Document] = dc.field(default_factory=list)

    #: If true, refresh the underlying collection
    refresh: bool = True

    #: If true, print more logging
    verbose: bool = True

    #: Positional query arguments
    args: t.Sequence = dc.field(default_factory=list)

    #: Named query arguments
    kwargs: t.Dict = dc.field(default_factory=dict)

    encoders: t.Sequence = dc.field(default_factory=list)

    #: Uniquely identifies serialized elements of this class
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
    def __call__(self, db: Datalayer):
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
        if self.refresh and not s.CFG.cdc:
            graph = db.refresh_after_update_or_insert(
                query=self,  # type: ignore[arg-type]
                ids=output.inserted_ids,
                verbose=self.verbose,
            )
        return output, graph


@dc.dataclass
class PostLike(Select):
    """Find documents like this one"""

    find_parent: t.Optional[Find]

    #: The document to match
    r: Document

    #: The name of the vector index to use
    vector_index: str

    #: The number of documents to return
    n: int = 100
    max_ids: int = 1000

    #: Uniquely identifies serialized elements of this class
    type_id: t.Literal['mongodb.PostLike'] = 'mongodb.PostLike'

    class Config:
        arbitrary_types_allowed = True

    @override
    def __call__(self, db: Datalayer):
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
            collection=self.find_parent.collection, args=[{'_id': {'$in': ids}}]
        )(db)


@dc.dataclass
class Featurize(Select):
    """A feature-extraction query"""

    features: t.Dict[str, str]
    parent: t.Union[PreLike, Find, PostLike]

    #: Uniquely identifies serialized elements of this class
    type_id: t.Literal['mongodb.Featurize'] = 'mongodb.Featurize'

    @property
    @override
    def select_table(self):
        return self.parent.select_table

    def get_ids(self, db: Datalayer) -> t.Sequence[str]:
        """Get a list of matching IDs

        :param db: The datalayer to query
        """
        return self.parent.get_ids(db)

    @override
    def is_trivial(self) -> bool:
        return self.parent.is_trivial()

    @property
    # @override
    def select_ids(self):
        """Converts the Serializable into a Serializable which only returns the id
        of each column/ document.
        """
        return t.cast(Select, self.parent.select_ids)

    # @override
    def add_fold(self, fold: str):
        """Create a select which selects the same data, but additionally restricts to
        the fold specified

        :param fold: possible values {'train', 'valid'}
        """
        return self.parent.add_fold(fold).featurize(self.features)

    @override
    def select_using_ids(self, ids: t.Sequence[str]) -> t.Any:
        return self.parent.select_using_ids(ids=ids).featurize(features=self.features)

    @override
    def model_update(
        self,
        db: Datalayer,
        ids: t.Sequence[t.Any],
        key: str,
        model: Model,
        outputs: t.Sequence[t.Any],
    ) -> None:
        self.parent.model_update(db=db, ids=ids, key=key, model=model, outputs=outputs)

    @override
    def __call__(self, db: Datalayer):
        if (
            isinstance(self.parent, Find)
            or isinstance(self.parent, Limit)
            or isinstance(self.parent, Like)
        ):
            out = self.parent(db)
            out.features = self.features
            return out
        else:
            r = self.parent(db)
            r = SuperDuperCursor.add_features(r.content, self.features)
            return Document(r)


@dc.dataclass
class Count(SelectOne):
    """A query to count matches"""

    parent: t.Union[Featurize, Find, PostLike, PreLike]

    #: Uniquely identifies serialized elements of this class
    type_id: t.Literal['mongodb.Count'] = 'mongodb.Count'

    #: Positional query arguments
    args: t.Sequence = dc.field(default_factory=list)

    #: Named query arguments
    kwargs: t.Dict = dc.field(default_factory=dict)

    @override
    def __call__(self, db: Datalayer):
        return db[self.parent.name].count_documents()  # type: ignore[index]


@dc.dataclass
class Limit(Select):
    """A query that limits the number of returned documents"""

    #: The number of documents to return
    n: int

    parent: t.Union[Featurize, Find, PreLike, PostLike]

    #: Uniquely identifies serialized elements of this class
    type_id: t.Literal['mongodb.Limit'] = 'mongodb.Limit'

    @override
    def __call__(self, db: Datalayer):
        return self.parent(db).limit(self.n)


@dc.dataclass
class ChangeStream:
    """Request a stream of changes from a datalayer"""

    #: The collection to perform the query on
    collection: Collection

    #: Positional query arguments
    args: t.Sequence = dc.field(default_factory=list)

    #: Named query arguments
    kwargs: t.Dict = dc.field(default_factory=dict)

    @override
    def __call__(self, db: Datalayer):
        collection = db.db[self.collection.name]
        return collection.watch(**self.kwargs)


all_items = {
    'Aggregate': Aggregate,
    'ChangeStream': ChangeStream,
    'Count': Count,
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
