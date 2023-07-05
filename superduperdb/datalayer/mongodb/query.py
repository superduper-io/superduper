from bson import ObjectId
from pydantic import BaseModel, Field
from pymongo import UpdateOne as _UpdateOne
import random
import typing as t

import superduperdb as s
from superduperdb.core.documents import Document
from superduperdb.datalayer.base.cursor import SuperDuperCursor
from superduperdb.datalayer.base.database import BaseDatabase
from superduperdb.datalayer.base.query import Select, SelectOne, Insert, Delete, Update
from superduperdb.datalayer.base.query import Like


class Collection(BaseModel):
    name: str

    @property
    def table(self):
        return self.name

    def to_dict(self):
        return {'cls': 'Collection', 'dict': {'name': self.name}}

    def like(
        self,
        r: t.Dict,
        vector_index: str,
        n: int = 100,
    ):
        return PreLike(collection=self, r=r, vector_index=vector_index, n=n)

    def insert_one(self, *args, **kwargs):
        return InsertMany(
            collection=self, documents=[args[0]], args=args[1:], kwargs=kwargs
        )

    def insert_many(self, *args, refresh=True, encoders=(), **kwargs):
        return InsertMany(
            collection=self,
            encoders=encoders,
            documents=args[0],
            args=args[1:],
            refresh=refresh,
            kwargs=kwargs,
        )

    def delete_one(self, *args, **kwargs):
        return DeleteOne(collection=self, args=args, kwargs=kwargs)

    def delete_many(self, *args, **kwargs):
        return DeleteMany(collection=self, args=args, kwargs=kwargs)

    def update_one(self, *args, **kwargs):
        return UpdateOne(collection=self, args=args, kwargs=kwargs)

    def update_many(self, *args, **kwargs):
        return UpdateMany(collection=self, args=args, kwargs=kwargs)

    def find(self, *args, **kwargs):
        return Find(collection=self, args=args, kwargs=kwargs)

    def find_one(self, *args, **kwargs):
        return FindOne(collection=self, args=args, kwargs=kwargs)

    def aggregate(self, *args, **kwargs):
        return Aggregate(collection=self, args=args, kwargs=kwargs)

    def replace_one(self, *args, **kwargs):
        return ReplaceOne(collection=self, args=args, kwargs=kwargs)

    def change_stream(self, *args, **kwargs):
        return ChangeStream(collection=self, args=args, kwargs=kwargs)


class ReplaceOne(Update):
    collection: Collection
    args: t.List = Field(default_factory=list)
    kwargs: t.Dict = Field(default_factory=dict)

    type_id: t.Literal['mongdb.ReplaceOne'] = 'mongdb.ReplaceOne'

    @property
    def select_table(self):
        raise NotImplementedError

    def __call__(self, db: BaseDatabase):
        filter, repl = self.args[:2]
        if isinstance(repl, Document):
            repl = repl.encode()
        elif isinstance(repl, dict):
            repl = Document(repl).encode()
        return db.db[self.collection.name].replace_one(
            filter, repl, *self.args[2:], **self.kwargs
        )

    def select(self):
        return Find(parent=self.collection, args=[self.args[0]])

    @property
    def select_ids(self):
        return Find(parent=self.collection, args=[self.args[0]]).select_ids


class PreLike(Like):
    r: t.Dict
    vector_index: str
    collection: Collection
    n: int = 100

    type_id: t.Literal['mongdb.PreLike'] = 'mongdb.PreLike'

    @property
    def table(self):
        return self.collection.name

    class Config:
        arbitrary_types_allowed = True

    def find(self, *args, **kwargs):
        return Find(like_parent=self, args=args, kwargs=kwargs)

    def find_one(self, *args, **kwargs):
        return Find(like_parent=self, args=args, kwargs=kwargs)

    def __call__(self, db):
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
            types=db.types,
        )


class Find(Select):
    collection: t.Optional[Collection] = None
    like_parent: t.Optional[PreLike] = None
    args: t.Optional[t.List] = Field(default_factory=lambda: [])
    kwargs: t.Optional[t.Dict] = Field(default_factory=lambda: {})

    type_id: t.Literal['mongdb.Find'] = 'mongdb.Find'

    @property
    def parent(self):
        msg = 'Must specify exactly one of "like_parent" and "collection"'
        assert not (self.like_parent and self.collection), msg
        assert self.like_parent or self.collection, msg
        if self.like_parent is not None:
            return self.like_parent
        return self.collection

    @property
    def table(self):
        return self.parent.table

    def limit(self, n: int):
        return Limit(parent=self, n=n)  # type

    def like(
        self, r: t.Dict, vector_index: str = '', n: int = 100, max_ids: int = 1000
    ):
        return PostLike(
            find_parent=self, r=r, n=n, max_ids=max_ids, vector_index=vector_index
        )

    def add_fold(self, fold: str) -> 'Find':
        args = []
        try:
            args.append(self.args[0])  # type: ignore[index, var-annotated]
        except IndexError:
            args.append({})
        args = args + self.args[1:]  # type: ignore[index]
        args[0]['_fold'] = fold  # type: ignore[index]
        return Find(
            like_parent=self.like_parent,
            collection=self.collection,
            args=args,
            kwargs=self.kwargs,
        )

    def is_trivial(self) -> bool:
        raise NotImplementedError

    @property
    def select_ids(self):
        try:
            filter = self.args[0]
        except IndexError:
            filter = {}
        return Find(
            collection=self.collection,
            like_parent=self.like_parent,
            args=[filter, {'_id': 1}],
        )

    def select_using_ids(self, ids: t.List[str]) -> t.Any:
        args = [{}, {}]  # type: ignore[var-annotated]
        if self.args and len(self.args) >= 1:  # type: ignore[var-annotated]
            args[0] = self.args[0]
        if self.args and len(self.args) >= 2:
            args[1] = self.args[1]
        args[0] = {'_id': {'$in': [ObjectId(_id) for _id in ids]}, **args[0]}

        return Find(
            collection=self.collection,
            like_parent=self.like_parent,
            args=args,
            kwargs=self.kwargs,
        )

    def featurize(self, features):
        return Featurize(parent=self, features=features)

    def get_ids(self, db: BaseDatabase):
        args = [{}, {}]  # type: ignore[var-annotated]
        args[: len(self.args)] = self.args  # type: ignore[var-annotated,arg-type,assignment]
        args[1] = {'_id': 1}  # type: ignore[arg-type,assigment]
        cursor = Find(
            collection=self.collection,
            like_parent=self.like_parent,
            args=args,
            kwargs=self.kwargs,
        )(db)
        return [r['_id'] for r in cursor]

    def model_update(self, db, model, key, outputs, ids):
        db.db[self.collection.name].bulk_write(
            [
                _UpdateOne(
                    {'_id': ObjectId(id)},
                    {'$set': {f'_outputs.{key}.{model}': outputs[i]}},
                )
                for i, id in enumerate(ids)
            ]
        )

    def model_cleanup(self, db, model, key):
        db.db[self.collection.name].update_many(
            {}, {'$unset': {f'_outputs.{key}.{model}': 1}}
        )

    @property
    def select_table(self):
        return Find(parent=self.parent)

    def download_update(self, db, id, key, bytes):
        return UpdateOne(
            collection=self.collection,
            args=[{'_id': id}, {'$set': {f'{key}._content.bytes': bytes}}],
        )(db)

    # ruff: noqq: E501
    def __call__(self, db: BaseDatabase):
        if isinstance(self.parent, Collection):
            cursor = db.db[self.collection.name].find(  # type: ignore[union-attr]
                *self.args, **self.kwargs
            )  #  type: ignore[union-attr]
        elif isinstance(self.parent, Like):  # type: ignore[union-attr]
            intermediate = self.parent(db)
            ids = [ObjectId(r['_id']) for r in intermediate]
            try:
                filter = self.args[0]  # type: ignore[index]
            except IndexError:  # type: ignore[index]
                filter = {}
            filter = {'$and': [filter, {'_id': {'$in': ids}}]}  # type: ignore[union-attr]
            cursor = db.db[self.like_parent.collection.name].find(  # type: ignore[index,union-attr]
                filter, *self.args[1:], **self.kwargs  # type: ignore[index]
            )
        else:
            raise NotImplementedError

        return SuperDuperCursor(raw_cursor=cursor, id_field='_id', types=db.types)


class FeaturizeOne(SelectOne):
    features: t.Dict[str, str]
    parent_find_one: t.Optional[Find] = None

    type_id: t.Literal['mongdb.FeaturizeOne'] = 'mongdb.FeaturizeOne'

    # ruff: noqa: E501
    def __call__(self, db: BaseDatabase):
        r = self.parent_find_one(db)  # type: ignore[misc]
        r = SuperDuperCursor.add_features(r.content, self.features)  # type: ignore[misc]
        return Document(r)


class FindOne(SelectOne):
    args: t.Optional[t.List] = Field(default_factory=list)
    kwargs: t.Optional[t.Dict] = Field(default_factory=dict)
    like_parent: t.Optional[PreLike] = None
    collection: t.Optional[Collection] = None

    type_id: t.Literal['mongdb.FindOne'] = 'mongdb.FindOne'

    def __call__(self, db: BaseDatabase):
        if self.collection is not None:
            return SuperDuperCursor.wrap_document(
                db.db[self.collection.name].find_one(*self.args, **self.kwargs),
                types=db.types,
            )
        else:
            parent_cursor = self.like_parent(db)  # type: ignore[misc]
            ids = [r['_id'] for r in parent_cursor]  # type: ignore[misc]
            filter = self.args[0] if self.args else {}
            filter['_id'] = {'$in': ids}
            r = db.db[self.like_parent.collection.name].find_one(  # type: ignore[union-attr]
                filter,  # type: ignore[union-attr]
                *self.args[1:],  # type: ignore[index]
                **self.kwargs,  # type: ignore[index]
            )
            return Document(Document.decode(r, types=db.types))

    def featurize(self, features):
        return FeaturizeOne(parent=self, features=features)


class Aggregate(Select):
    args: t.List = Field(default_factory=list)
    kwargs: t.Dict = Field(default_factory=dict)
    collection: Collection

    type_id: t.Literal['mongdb.Aggregate'] = 'mongdb.Aggregate'

    def __call__(self, db):
        return db.db[self.collection.name].aggregate(*self.args, **self.kwargs)


class DeleteOne(Delete):
    collection: Collection
    args: t.List = Field(default_factory=list)
    kwargs: t.Dict = Field(default_factory=dict)

    type_id: t.Literal['mongdb.DeleteOne'] = 'mongdb.DeleteOne'

    def __call__(self, db: BaseDatabase):
        return db.db[self.collection.name].delete_one(*self.args, **self.kwargs)


class DeleteMany(Delete):
    collection: Collection
    args: t.List = Field(default_factory=list)
    kwargs: t.Dict = Field(default_factory=dict)

    type_id: t.Literal['mongdb.DeleteMany'] = 'mongdb.DeleteMany'

    def __call__(self, db: BaseDatabase):
        return db.db[self.collection.name].delete_many(*self.args, **self.kwargs)


class UpdateOne(Update):
    collection: Collection
    args: t.List = Field(default_factory=list)
    kwargs: t.Dict = Field(default_factory=dict)

    type_id: t.Literal['mongdb.UpdateOne'] = 'mongdb.UpdateOne'

    def __call__(self, db: BaseDatabase):
        return db.db[self.collection.name].update_one(*self.args, **self.kwargs)

    @property
    def select_table(self):
        raise NotImplementedError

    @property
    def select(self):
        return Find(collection=self.collection, args=[self.args[0]])

    @property
    def select_ids(self):
        raise NotImplementedError


class UpdateMany(Update):
    collection: Collection
    args: t.List = Field(default_factory=list)
    kwargs: t.Dict = Field(default_factory=dict)

    my_refresh: bool = Field(default=True, alias='refresh')
    my_verbose: bool = Field(default=True, alias='verbose')

    type_id: t.Literal['mongdb.UpdateMany'] = 'mongdb.UpdateMany'

    def __call__(self, db: BaseDatabase):
        to_update = Document(self.args[1]).encode()
        ids = [
            r['_id'] for r in db.db[self.collection.name].find(self.args[0], {'_id': 1})
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
    def select_table(self):
        return Find(collection=self.collection, args=[{}])

    @property
    def select_ids(self):
        raise NotImplementedError

    @property
    def select(self):
        return Find(parent=self.args[0])


class InsertOne(Insert):
    collection: Collection
    args: t.List = Field(default_factory=list)
    kwargs: t.Dict = Field(default_factory=dict)
    my_refresh: bool = Field(default=True, alias='refresh')
    my_verbose: bool = Field(default=True, alias='verbose')

    def __call__(self, db: BaseDatabase):
        insert = db.db[self.collection.name].insert_one(*self.args, **self.kwargs)
        graph = None
        if self.refresh and not s.CFG.cdc:
            graph = db.refresh_after_update_or_insert(
                query=self,
                ids=[insert.id],
                verbose=self.verbose,
            )
        return insert, graph

    @property
    def table(self):
        return self.collection.name

    @property
    def select_table(self):
        return Find(collection=self.collection)

    def select_using_ids(self, ids):
        return Find(collection=self.collection, args=[{'_id': {'$in': ids}}])


class InsertMany(Insert):
    collection: Collection
    args: t.List = Field(default_factory=list)
    kwargs: t.Dict = Field(default_factory=dict)
    valid_prob: float = 0.05
    encoders: t.List = Field(default_factory=list)

    type_id: t.Literal['mongdb.InsertMany'] = 'mongdb.InsertMany'

    @property
    def table(self):
        return self.collection.name

    @property
    def select_table(self):
        return Find(collection=self.collection)

    def select_using_ids(self, ids):
        return Find(collection=self.collection, args=[{'_id': {'$in': ids}}])

    def __call__(self, db: BaseDatabase):
        for e in self.encoders:
            db.add(e)
        documents = [r.encode() for r in self.documents]
        for r in documents:
            if random.random() < self.valid_prob:
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


class PostLike(Select):
    find_parent: t.Optional[Find]
    r: t.Dict
    vector_index: str
    n: int = 100
    max_ids: int = 1000

    type_id: t.Literal['mongdb.PostLike'] = 'mongdb.PostLike'

    class Config:
        arbitrary_types_allowed = True
        # TODO: the server will crash when it tries to JSONize whatever it is that
        # this allows.

    def __call__(self, db: BaseDatabase):
        cursor = self.find_parent.select_ids.limit(self.max_ids)(db)  # type: ignore[union-attr]
        ids = [r['_id'] for r in cursor]  # type: ignore[union-attr]
        ids, scores = db._select_nearest(
            like=self.r,
            vector_index=self.vector_index,
            n=self.n,
            ids=[str(_id) for _id in ids],
        )
        ids = [ObjectId(_id) for _id in ids]
        # ruff: noqa: E501
        return Find(
            collection=self.find_parent.collection, args=[{'_id': {'$in': ids}}]  # type: ignore[union-attr]
        )(db)

    def add_fold(self, fold: str) -> 'Select':
        raise NotImplementedError

    def is_trivial(self) -> bool:
        raise NotImplementedError

    @property
    def select_ids(self) -> 'Select':
        raise NotImplementedError

    def model_update(self, db, model, key, outputs, ids):
        raise NotImplementedError

    @property
    def select_table(self):
        raise NotImplementedError

    def select_using_ids(
        self,
        ids: t.List[str],
    ) -> t.Any:
        raise NotImplementedError


class Featurize(Select):
    features: t.Dict[str, str]
    parent: t.Union[PreLike, Find, PostLike]

    type_id: t.Literal['mongdb.Featurize'] = 'mongdb.Featurize'

    @property
    def select_table(self):
        return self.parent.select_table

    def get_ids(self, *args, **kwargs):
        return self.parent.get_ids(*args, **kwargs)

    def is_trivial(self):
        return self.parent.is_trivial()

    @property
    def select_ids(self):
        return self.parent.select_ids

    # ruff: noqa: E501
    def add_fold(self, fold: str):
        return self.parent.add_fold(fold).featurize(self.features)  # type: ignore[union-attr]

    # ruff: noqa: E501
    def select_using_ids(self, ids: t.List[str]) -> t.Any:
        return self.parent.select_using_ids(ids=ids).featurize(features=self.features)  # type: ignore[union-attr]

    def model_update(self, *args, **kwargs):
        return self.parent.model_update(*args, **kwargs)

    def __call__(self, db: BaseDatabase):
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


class Limit(Select):
    n: int
    parent: t.Union[Find, PostLike, PreLike, Featurize]

    type_id: t.Literal['mongdb.Limit'] = 'mongdb.Limit'

    def __call__(self, db: BaseDatabase):
        return self.parent(db).limit(self.n)

    def add_fold(self, fold: str) -> 'Select':
        raise NotImplementedError

    def is_trivial(self) -> bool:
        raise NotImplementedError

    def select_using_ids(
        self,
        ids: t.List[str],
    ) -> t.Any:
        raise NotImplementedError

    def model_update(self, db, model, key, outputs, ids):
        raise NotImplementedError

    @property
    def select_ids(self) -> 'Select':
        raise NotImplementedError

    @property
    def select_table(self):
        raise NotImplementedError


class ChangeStream(BaseModel):
    collection: Collection
    args: t.List = Field(default_factory=list)
    kwargs: t.Dict = Field(default_factory=dict)

    def __call__(self, db: BaseDatabase):
        resume_token = self.kwargs.get("resume_token")
        # TODO (high): need to pass change pipeline into watch
        self.kwargs.get("change")

        collection = db.db[self.collection.name]
        options = {}
        if resume_token:
            options['resumeAfter'] = resume_token
        if options:
            change_stream_iterator = collection.watch()
        else:
            change_stream_iterator = collection.watch()
        return change_stream_iterator


all_items = {
    'Aggregate': Aggregate,
    'Collection': Collection,
    'DeleteMany': DeleteMany,
    'DeleteOne': DeleteOne,
    'Featurize': Featurize,
    'Find': Find,
    'FindOne': FindOne,
    'InsertMany': InsertMany,
    'PreLike': PreLike,
    'PostLike': PostLike,
    'Limit': Limit,
    'UpdateOne': UpdateOne,
    'UpdateMany': UpdateMany,
}
