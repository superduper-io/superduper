import copy
import dataclasses as dc
import typing as t

import mongomock
import pymongo
from bson import ObjectId

from superduperdb import CFG, logging
from superduperdb.backends.base.query import (
    CompoundSelect,
    Delete,
    Insert,
    Like,
    QueryComponent,
    QueryLinker,
    QueryType,
    Select,
    TableOrCollection,
    Update,
    Write,
)
from superduperdb.base.cursor import SuperDuperCursor
from superduperdb.base.document import Document
from superduperdb.components.schema import Schema
from superduperdb.misc.files import load_uris

SCHEMA_KEY = '_schema'


class FindOne(QueryComponent):
    """Wrapper around ``pymongo.Collection.find_one``.

    :param args: Positional arguments to ``pymongo.Collection.find_one``
    :param kwargs: Named arguments to ``pymongo.Collection.find_one``
    """

    def select_using_ids(self, ids):
        """Select documents using ids.

        :param ids: The ids to select
        """
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
        """Modify the query to add a fold to filter {'_fold': fold}.

        :param fold: The fold to add
        """
        args = self.args or [{}]
        args[0]['_fold'] = fold
        return FindOne(
            name=self.name,
            type=self.type,
            args=args,
            kwargs=self.kwargs,
        )


@dc.dataclass
class Find(QueryComponent):
    """Wrapper around ``pymongo.Collection.find``.

    :param output_fields: The output fields to project to
    """

    output_fields: t.Optional[t.Dict[str, str]] = None

    def __post_init__(self):
        if not self.args:
            self.args = [{}]
        else:
            self.args = list(self.args)
        if not self.args[1:]:
            self.args.append({})

        if self.output_fields is not None:
            # if outputs are specified, then project to those outputs
            self.args[1].update({f'_outputs.{x}': 1 for x in self.output_fields})
            if '_id' not in self.args[1]:
                self.args[1]['_id'] = 1

    @property
    def select_ids(self):
        """Select ids."""
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

    def outputs(self, *predict_ids):
        """
        Join the query with the outputs for a table.

        :param *predict_ids: The ids to predict
        """
        args = copy.deepcopy(list(self.args[:]))
        if not args:
            args = [{}]
        if not args[1:]:
            args.append({})

        for identifier in predict_ids:
            args[1][f'_outputs.{identifier}'] = 1
        return Find(
            name=self.name,
            type=self.type,
            args=args,
            kwargs=self.kwargs,
            output_fields=predict_ids,
        )

    def select_using_ids(self, ids):
        """Select documents using ids.

        :param ids: The ids to select
        """
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

    def select_ids_of_missing_outputs(self, predict_id: str):
        """Select ids of missing outputs.

        :param predict_id: The predict id to select
        """
        assert self.type == QueryType.QUERY
        if self.args:
            args = [
                {
                    '$and': [
                        self.args[0],
                        {f'_outputs.{predict_id}': {'$exists': 0}},
                    ]
                },
                *self.args[1:],
            ]
        else:
            args = [{f'_outputs.{predict_id}': {'$exists': 0}}]

        if len(args) == 1:
            args.append({})

        args[1] = {'_id': 1}

        return Find(
            name='find',
            type=QueryType.QUERY,
            args=args,
            kwargs=self.kwargs,
        )

    def select_single_id(self, id):
        """Select a single document by id.

        :param id: The id of the document to select
        """
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
        """Add a fold to the query.

        :param fold: The fold to add
        """
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
    """Wrapper around ``pymongo.Collection.aggregate``.

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
        """Return the id field."""
        return self.table_or_collection.primary_id

    @property
    def select_table(self):
        """Select the table to perform the query on."""
        raise NotImplementedError

    def add_fold(self):
        """Add a fold to the query."""
        raise NotImplementedError

    def select_single_id(self, id: str):
        """Select a single document by id.

        :param id: The id of the document to select
        """
        raise NotImplementedError

    @property
    def select_ids(self):
        """Select ids."""
        raise NotImplementedError

    def select_using_ids(self):
        """Select documents using ids."""
        raise NotImplementedError

    def select_ids_of_missing_outputs(self, key: str, model: str, version: int):
        """Select ids of missing outputs.

        :param key: The key to select
        :param model: The model to select
        :param version: The version to select
        """
        raise NotImplementedError

    @staticmethod
    def _replace_document_with_vector(step, vector_index, db):
        step = copy.deepcopy(step)
        assert "like" in step['$vectorSearch']
        vector = step['$vectorSearch']['like']

        if not isinstance(vector, Document):
            vector_index = db.vector_indices[vector_index]
            models, keys = vector_index.models_keys
            vector, _, _ = vector_index.get_vector(
                like=vector,
                models=models,
                keys=keys,
                db=db,
            )

        step['$vectorSearch']['queryVector'] = vector
        indexing_key = vector_index.indexing_listener.key
        if indexing_key.startswith('_outputs'):
            indexing_key = indexing_key.split('.')[1]
        indexing_model = vector_index.indexing_listener.model.identifier
        indexing_version = vector_index.indexing_listener.model.version
        step['$vectorSearch'][
            'path'
        ] = f'_outputs.{indexing_key}.{indexing_model}.{indexing_version}'
        step['$vectorSearch']['index'] = vector_index.identifier
        del step['$vectorSearch']['like']
        return step

    @staticmethod
    def _prepare_pipeline(pipeline, db, vector_index):
        pipeline = copy.deepcopy(pipeline)
        try:
            search_step = next(
                (i, step) for i, step in enumerate(pipeline) if '$vectorSearch' in step
            )
        except StopIteration:
            return pipeline
        pipeline[search_step[0]] = Aggregate._replace_document_with_vector(
            search_step[1],
            vector_index,
            db,
        )
        return pipeline

    def execute(self, db, reference=False):
        """Execute the query.

        :param db: The datalayer instance
        :param reference: Not used
        """
        collection = db.databackend.get_table_or_collection(
            self.table_or_collection.identifier
        )
        cursor = collection.aggregate(
            self._prepare_pipeline(
                self.args[0],
                db,
                self.vector_index,
            )
        )
        decode_function = _get_decode_function(db)
        return SuperDuperCursor(
            raw_cursor=cursor,
            id_field='_id',
            db=db,
            decode_function=decode_function,
        )


@dc.dataclass(repr=False)
class MongoCompoundSelect(CompoundSelect):
    """CompoundSelect class to perform compound queries on a collection."""

    DB_TYPE: t.ClassVar[str] = 'MONGODB'

    def _deep_flat_encode(self, cache):
        documents = {}
        query_chain = []

        if self.pre_like is not None:
            query, sub_docs = self.pre_like._deep_flat_encode(cache)
            query_chain.append(query)
            documents.update(sub_docs)

        for member in self.query_linker.members:
            query, sub_docs = member._deep_flat_encode(cache)
            query_chain.append(query)
            documents.update(sub_docs)

        if self.post_like is not None:
            query, sub_docs = self.pre_like._deep_flat_encode(cache)
            query_chain.append(query)
            documents.update(sub_docs)

        query = f'{self.table_or_collection.identifier}.' + '.'.join(query_chain)
        for i, k in enumerate(documents):
            query = query.replace(k, f'{i}')

        documents = list(documents.values())
        return {'query': query, 'documents': [r.encode() for r in documents]}

    def _get_query_linker(self, table_or_collection, members) -> 'QueryLinker':
        return MongoQueryLinker(
            table_or_collection=table_or_collection, members=members
        )

    @property
    def output_fields(self):
        """Return the output fields."""
        return self.query_linker.output_fields

    def outputs(self, *predict_ids):
        """Returns a query which joins a query with the outputs for a table.

        :param *predict_ids: The ids to predict

        >>> q = Collection(...).find(...).outputs('key', 'model_name')

        """
        assert self.query_linker is not None
        return MongoCompoundSelect(
            table_or_collection=self.table_or_collection,
            pre_like=self.pre_like,
            query_linker=self.query_linker.outputs(*predict_ids),
            post_like=self.post_like,
        )

    def change_stream(self, *args, **kwargs):
        """Change stream for the query."""
        return self.table_or_collection.change_stream(*args, **kwargs)

    def _execute(self, db):
        similar_scores = None
        query_linker = self.query_linker
        if self.pre_like:
            similar_ids, similar_scores = self.pre_like.execute(db)
            similar_scores = dict(zip(similar_ids, similar_scores))
            if not self.query_linker:
                return similar_ids, similar_scores
            # TODO -- pre-select ids (this logic is wrong)
            query_linker = query_linker.select_using_ids(similar_ids)

        if not self.post_like:
            return query_linker.execute(db), similar_scores

        assert self.pre_like is None
        cursor = query_linker.select_ids.execute(db)
        query_ids = [str(document[self.primary_id]) for document in cursor]
        similar_ids, similar_scores = self.post_like.execute(db, ids=query_ids)
        similar_scores = dict(zip(similar_ids, similar_scores))

        post_query_linker = self.query_linker.select_using_ids(similar_ids)
        return post_query_linker.execute(db), similar_scores

    def execute(self, db, reference=False):
        """Execute the query.

        :param db: The datalayer instance
        :param reference: If True, load the references
        """
        output, scores = self._execute(db)
        decode_function = _get_decode_function(db)
        if isinstance(output, (pymongo.cursor.Cursor, mongomock.collection.Cursor)):
            return SuperDuperCursor(
                raw_cursor=output,
                id_field='_id',
                scores=scores,
                db=db,
                decode_function=decode_function,
            )
        elif isinstance(output, dict):
            output = decode_function(output)
            if reference and CFG.hybrid_storage:
                load_uris(output, datatypes=db.datatypes)
            return Document.decode(output, db)
        return output

    def download_update(self, db, id: str, key: str, bytes: bytearray) -> None:
        """
        Update to set the content of ``key`` in the document ``id``.

        :param db: The db to query
        :param id: The id to filter on
        :param key: The key to update
        :param bytes: The bytes to update
        """
        if self.collection is None:
            raise ValueError('collection cannot be None')
        update = {'$set': {f'{key}._content.bytes': bytes}}
        collection = db.databackend.get_table_or_collection(
            self.table_or_collection.identifier
        )
        return collection.update_one({'_id': id}, update)

    def check_exists(self, db):
        """Check if the query exists in the database.

        :param db: The datalayer instance
        """
        ...

    @property
    def select_table(self):
        """Select the table to perform the query on."""
        return self.table_or_collection.find()


@dc.dataclass(repr=False)
class MongoQueryLinker(QueryLinker):
    """QueryLinker class to link queries together."""

    @property
    def query_components(self):
        """Return the query components."""
        return self.table_or_collection.query_components

    @property
    def output_fields(self):
        """Return the output fields."""
        out = {}
        for member in self.members:
            if hasattr(member, 'output_fields'):
                out.update(member.output_fields)
        return out

    def add_fold(self, fold):
        """Add a fold to the query.

        :param fold: The fold to add
        """
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

    def outputs(self, *predict_ids):
        """Join the query with the outputs for a table."""
        new_members = []
        for member in self.members:
            if hasattr(member, 'outputs'):
                new_members.append(member.outputs(*predict_ids))
            else:
                new_members.append(member)

        return MongoQueryLinker(
            table_or_collection=self.table_or_collection,
            members=new_members,
        )

    @property
    def select_ids(self):
        """Select ids."""
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
        """Select documents using ids.

        :param ids: The ids to select
        """
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

    def _select_ids_of_missing_outputs(self, predict_id: str):
        new_members = []
        for member in self.members:
            if hasattr(member, 'select_ids_of_missing_outputs'):
                new_members.append(
                    member.select_ids_of_missing_outputs(predict_id=predict_id)
                )
        return MongoQueryLinker(
            table_or_collection=self.table_or_collection,
            members=new_members,
        )

    def select_single_id(self, id):
        """Select a single document by id.

        :param id: The id of the document to select
        """
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
        """Execute the query.

        :param db: The datalayer instance
        """
        parent = db.databackend.get_table_or_collection(
            self.table_or_collection.identifier
        )
        for member in self.members:
            parent = member.execute(parent)
        return parent


@dc.dataclass(repr=False)
class MongoInsert(Insert):
    """Insert class to insert a single document in the database.

    :param one: If True, only one document will be inserted
    """

    one: bool = False

    def raw_query(self, db):
        """Returns a raw mongodb query for mongodb operation.

        :param db: The datalayer instance
        """
        schema = self.kwargs.pop('schema', None)
        schema = get_schema(db, schema) if schema else None
        documents = [r.encode(schema) for r in self.documents]
        if schema:
            for document in documents:
                document[SCHEMA_KEY] = schema.identifier
        return documents

    def execute(self, db):
        """Execute the query.

        :param db: The datalayer instance
        """
        collection = db.databackend.get_table_or_collection(
            self.table_or_collection.identifier
        )
        documents = self.raw_query(db)
        insert_result = collection.insert_many(documents, **self.kwargs)
        return insert_result.inserted_ids

    @property
    def select_table(self):
        """Select collection to be inserted."""
        return self.table_or_collection.find()


@dc.dataclass(repr=False)
class MongoDelete(Delete):
    """Delete class to delete a single document in the database.

    :param one: If True, only one document will be deleted
    """

    one: bool = False

    @property
    def collection(self):
        """Return the collection from the database."""
        return self.table_or_collection

    def to_operation(self, collection):
        """Returns a mongodb operation i.e `pymongo.InsertOne`.

        :param collection: The collection to perform the operation on
        """
        if self.one:
            return pymongo.DeleteOne(*self.args, **self.kwargs)
        return pymongo.DeleteMany(*self.args, **self.kwargs)

    def arg_ids(self, collection):
        """Returns the ids of the documents to be deleted.

        :param collection: The collection to be deleted from
        """
        ids = []

        if '_id' in self.kwargs:
            if self.one:
                ids = [str(self.kwargs['_id'])]
        for arg in self.args:
            if isinstance(arg, dict) and '_id' in arg:
                if self.one:
                    ids = [str(arg['_id'])]
                else:
                    ids = list(map(str, arg['_id']['$in']))
        return ids

    def execute(self, db):
        """Execute the query.

        :param db: The datalayer instance
        """
        collection = db.databackend.get_table_or_collection(
            self.table_or_collection.identifier
        )
        if self.one:
            ids = self.arg_ids(collection)

            result = collection.delete_one(*self.args, **self.kwargs)
            if result.deleted_count == 1:
                if not ids:
                    deleted_document = collection.find_one(*self.args, **self.kwargs)
                    return [str(deleted_document['_id'])]
                return ids
            else:
                return []

        collection.delete_many(*self.args, **self.kwargs)
        # TODO: check delete_result for counts
        deleted_ids = self.arg_ids(collection)
        if not deleted_ids:
            # This implies to delete all collections
            docs = list(collection.find(*self.args, **self.kwargs))
            return [str(d['_id']) for d in docs]
        return deleted_ids


@dc.dataclass(repr=False)
class MongoUpdate(Update):
    """Update class to update a single document in the database.

    :param update: The update document
    :param filter: The filter to apply
    :param one: If True, only one document will be updated
    :param args: Positional arguments to ``pymongo.Collection.update_one``
    :param kwargs: Named arguments to ``pymongo.Collection.update_one``
    """

    update: Document
    filter: t.Dict
    one: bool = False
    args: t.Sequence = dc.field(default_factory=list)
    kwargs: t.Dict = dc.field(default_factory=dict)

    @property
    def select_table(self):
        """Select collection to be updated."""
        return self.table_or_collection.find()

    def to_operation(self, collection):
        """Returns a mongodb operation i.e `pymongo.InsertOne`.

        :param collection: The collection to perform the operation on
        """
        filter, update = self.raw_query(collection)
        if self.one:
            return pymongo.UpdateOne(filter, update)
        return pymongo.UpdateMany(filter, update)

    def arg_ids(self, collection):
        """Returns the ids of the documents to be updated.

        :param collection: The collection to be updated from
        """
        filter, _ = self.raw_query(collection)
        if self.one is True:
            return [filter['_id']]
        else:
            return filter['_id']['$in']

    def raw_query(self, collection):
        """Returns a raw mongodb query for mongodb operation.

        :param collection: The collection to perform the operation on
        """
        update = self.update
        if isinstance(self.update, Document):
            update = update.encode()

        if self.one:
            id = collection.find_one(self.filter, {'_id': 1})['_id']
            return {'_id': id}, update

        ids = [r['_id'] for r in collection.find(self.filter, {'_id': 1})]
        return {'_id': {'$in': ids}}, update

    def execute(self, db):
        """Execute the query.

        :param db: The datalayer instance
        """
        collection = db.databackend.get_table_or_collection(
            self.table_or_collection.identifier
        )

        mongo_filter, update = self.raw_query(collection)
        if self.one:
            collection.update_one(mongo_filter, update, *self.args, **self.kwargs)
            return [mongo_filter['_id']]

        collection.update_many(mongo_filter, update, *self.args, **self.kwargs)
        return mongo_filter['_id']['$in']


@dc.dataclass(repr=False)
class MongoBulkWrite(Write):
    """MongoBulkWrite will help write multiple mongodb operations to database at once.

    Example:
    -------
        MongoBulkWrite(operations= [MongoUpdate(...), MongoDelete(...)])

    :param operations: List of operations to be performed
    :param args: Positional arguments to ``pymongo.Collection.bulk_write``
    :param kwargs: Named arguments to ``pymongo.Collection.bulk_write``

    """

    operations: t.List[t.Union[MongoUpdate, MongoDelete]]
    args: t.Sequence = dc.field(default_factory=list)
    kwargs: t.Dict = dc.field(default_factory=dict)

    def __post_init__(self):
        for query in self.operations:
            assert isinstance(query, (MongoDelete, MongoUpdate))
            if not query.arg_ids:
                raise ValueError(
                    'Please provided update/delete id in args',
                    'all ids selection e.g `\{\}` is not supported',
                )

    @property
    def select_table(self):
        """Select collection to be bulk written."""
        return self.table_or_collection.find()

    def execute(self, db):
        """Execute the query.

        :param db: The datalayer instance
        """
        collection = db.databackend.get_table_or_collection(
            self.table_or_collection.identifier
        )
        bulk_operations = []
        bulk_update_ids = []
        bulk_delete_ids = []
        bulk_result = {'delete': [], 'update': []}
        for query in self.operations:
            operation = query.to_operation(collection)

            bulk_operations.append(operation)
            ids = query.arg_ids(collection)
            if isinstance(query, MongoDelete):
                bulk_result['delete'].append({'query': query, 'ids': ids})
                bulk_delete_ids += ids
            else:
                bulk_update_ids += ids
                bulk_result['update'].append({'query': query, 'ids': ids})

        result = collection.bulk_write(bulk_operations, *self.args, **self.kwargs)
        if result.deleted_count != bulk_delete_ids:
            logging.warn(
                'Some delete ids are not executed',
                ', hence halting execution',
                'Please note the partially executed operations',
                'wont trigger any `model/listeners` unless CDC is active.',
            )
        elif (result.modified_count + result.upserted_count) != bulk_update_ids:
            logging.warn(
                'Some update ids are not executed',
                ', hence halting execution',
                'Please note the partially executed operations',
                'wont trigger any `model/listeners` unless CDC is active.',
            )
        return bulk_result, bulk_update_ids, bulk_delete_ids


@dc.dataclass(repr=False)
class MongoReplaceOne(Update):
    """Replace class to replace a single document in the database.

    :param replacement: The replacement document
    :param filter: The filter to apply
    :param args: Positional arguments to ``pymongo.Collection.replace_one``
    :param kwargs: Named arguments to ``pymongo.Collection.replace_one``
    """

    replacement: Document
    filter: t.Dict
    args: t.Sequence = dc.field(default_factory=list)
    kwargs: t.Dict = dc.field(default_factory=dict)

    @property
    def collection(self):
        """Return the collection from the database."""
        return self.table_or_collection

    @property
    def select_table(self):
        """Return the table from the database."""
        return self.table_or_collection.find()

    def execute(self, db):
        """Execute the query.

        :param db: The datalayer instance
        """
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
    """Change stream class to watch for changes in specified collection.

    :param collection: The collection to perform the query on
    :param args: Positional query arguments to ``pymongo.Collection.watch``
    :param kwargs: Named query arguments to ``pymongo.Collection.watch``
    """

    collection: 'Collection'
    args: t.Sequence = dc.field(default_factory=list)
    kwargs: t.Dict = dc.field(default_factory=dict)

    def __call__(self, db):
        """Watch for changes in the database in specified collection.

        :param db: The datalayer instance
        """
        collection = db.databackend.get_table_or_collection(self.collection)
        return collection.watch(**self.kwargs)


@dc.dataclass(repr=False)
class Collection(TableOrCollection):
    """Collection class to perform queries on a collection."""

    query_components: t.ClassVar[t.Dict] = {'find': Find, 'find_one': FindOne}
    type_id: t.ClassVar[str] = 'collection'

    primary_id: t.ClassVar[str] = '_id'

    def get_table(self, db):
        """Return the table from the database.

        :param db: The datalayer instance
        """
        collection = db.databackend.get_table_or_collection(self.collection.identifier)
        return collection

    def change_stream(self, *args, **kwargs):
        """Request a stream of changes from the collection."""
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

    def _bulk_write(self, operations, *args, **kwargs):
        return MongoBulkWrite(
            operations=operations,
            args=args,
            kwargs=kwargs,
            table_or_collection=self,
        )

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
        """Perform an aggregation on the collection.

        :param vector_index: The vector index to use
        """
        return Aggregate(
            args=args,
            kwargs=kwargs,
            vector_index=vector_index,
            table_or_collection=self,
        )

    def delete_one(self, *args, **kwargs):
        """Delete a single document in the database."""
        return self._delete(*args, one=True, **kwargs)

    def delete_many(self, *args, **kwargs):
        """Delete multiple documents in the database."""
        return self._delete(*args, one=False, **kwargs)

    def replace_one(self, filter, replacement, *args, **kwargs):
        """Replace a single document in the database.

        :param filter: The filter to apply
        :param replacement: The replacement to apply
        """
        return MongoReplaceOne(
            filter=filter,
            replacement=replacement,
            args=args,
            kwargs=kwargs,
            table_or_collection=self,
        )

    def update_one(self, filter, update, *args, **kwargs):
        """Update a single document in the database.

        :param filter: The filter to apply
        :param update: The update to apply
        """
        return self._update(filter, update, *args, one=True, **kwargs)

    def update_many(self, filter, update, *args, **kwargs):
        """Update multiple documents in the database.

        :param filter: The filter to apply
        :param update: The update to apply
        """
        return self._update(filter, update, *args, one=False, **kwargs)

    def bulk_write(self, operations, *args, **kwargs):
        """Bulk write multiple operations into the database.

        :param operations: The operations to perform
        """
        return self._bulk_write(operations, *args, **kwargs)

    def insert(self, *args, **kwargs):
        """Insert multiple documents into the database.

        :param args: The documents to insert
        """
        return self.insert_many(*args, **kwargs)

    def insert_many(self, *args, **kwargs):
        """Insert multiple documents into the database.

        :param args: The documents to insert
        """
        return self._insert(*args, **kwargs)

    def insert_one(self, document, *args, **kwargs):
        """Insert a single document into the database.

        :param document: The document to insert
        """
        return self._insert([document], *args, **kwargs)

    def model_update(
        self,
        db,
        ids: t.List[t.Any],
        predict_id: str,
        outputs: t.Sequence[t.Any],
        flatten: bool = False,
        **kwargs,
    ):
        """Update the outputs of a model in the database.

        :param db: The datalaer instance
        :param ids: The ids of the documents to update
        :param predict_id: The predict_id of outputs to store
        :param outputs: The outputs to store
        :param flatten: Whether to flatten the outputs
        :param kwargs: Additional arguments
        """
        document_embedded = kwargs.get('document_embedded', True)

        if not len(outputs):
            return
        if document_embedded:
            if flatten:
                raise AttributeError(
                    'Flattened outputs cannot be stored along with input documents.'
                    'Please use `document_embedded = False` option with flatten = True'
                )
            assert self.collection is not None
            bulk_operations = []
            for i, id in enumerate(ids):
                mongo_filter = {'_id': ObjectId(id)}
                update = Document({'$set': {f'_outputs.{predict_id}': outputs[i]}})
                bulk_operations.append(
                    MongoUpdate(
                        filter=mongo_filter,
                        update=update,
                        table_or_collection=self,
                        one=True,
                    )
                )

            db.execute(self.bulk_write(bulk_operations))

        else:
            collection = Collection(f'_outputs.{predict_id}')
            bulk_writes = []

            if flatten:
                for i, id in enumerate(ids):
                    _outputs = outputs[i]
                    if isinstance(_outputs, (list, tuple)):
                        for offset, output in enumerate(_outputs):
                            bulk_writes.append(
                                {
                                    '_outputs': {predict_id: output},
                                    '_source': ObjectId(id),
                                    '_offset': offset,
                                }
                            )
                    else:
                        bulk_writes.append(
                            {
                                '_outputs': {predict_id: _outputs},
                                '_source': ObjectId(id),
                                '_offset': 0,
                            }
                        )

            else:
                for i, id in enumerate(ids):
                    bulk_writes.append(
                        {'_id': ObjectId(id), '_outputs': {predict_id: outputs[i]}}
                    )

            if bulk_writes:
                db.execute(
                    collection.insert_many([Document(**doc) for doc in bulk_writes])
                )


def _get_decode_function(db) -> t.Callable[[t.Any], t.Any]:
    def decode(output):
        schema_identifier = output.get(SCHEMA_KEY)
        if schema_identifier is None:
            return output
        schema = get_schema(db, schema_identifier)
        for k in output.keys():
            if field := schema.fields.get(k):
                output[k] = field.decode_data(output[k])
        return output

    return decode


def get_schema(db, schema: t.Union[Schema, str]) -> Schema:
    """Handle schema caching and loading.

    :param db: the Datalayer instance
    :param schema: the schema to be loaded
    """
    if isinstance(schema, Schema):
        # If the schema is not in the db, it is added to the db.
        if schema.identifier not in db.show(Schema.type_id):
            db.add(schema)
        schema_identifier = schema.identifier

    else:
        schema_identifier = schema

    assert isinstance(schema_identifier, str)

    return db.schemas[schema_identifier]
