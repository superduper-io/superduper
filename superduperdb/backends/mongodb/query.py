import dataclasses as dc
import typing as t
from collections import defaultdict

from bson import ObjectId

from superduperdb.backends.base.query import (
    Query,
    applies_to,
    parse_query as _parse_query,
)
from superduperdb.base.cursor import SuperDuperCursor
from superduperdb.base.document import Document, QueryUpdateDocument
from superduperdb.base.leaf import Leaf

if t.TYPE_CHECKING:
    from superduperdb.base.datalayer import Datalayer


def parse_query(query, documents, db: t.Optional['Datalayer'] = None):
    return _parse_query(
        query=query,
        documents=documents,
        builder_cls=MongoQuery,
        db=db,
    )


@dc.dataclass(kw_only=True, repr=False)
class MongoQuery(Query):
    flavours: t.ClassVar[t.Dict[str, str]] = {
        'pre_like': '^.*\.like\(.*\)\.find',
        'post_like': '^.*\.find\(.*\)\.like(.*)$',
        'bulk_write': '^.*\.bulk_write\(.*\)$',
        'find_one': '^.*\.find_one\(.*\)',
        'find': '^.*\.find\(.*\)',
        'insert_many': '^.*\.insert_many\(.*\)$',
        'insert_one': '^.*\.insert_one\(.*\)$',
        'replace_one': '^.*\.replace_one\(.*\)$',
        'update_many': '^.*\.update_many\(.*\)$',
        'update_one': '^.*\.update_one\(.*\)$',
        'delete_many': '^.*\.delete_many\(.*\)$',
        'delete_one': '^.*\.delete_one\(.*\)$',
    }

    def _create_table_if_not_exists(self):
        return

    def _deep_flat_encode(self, cache, blobs, files, leaves_to_keep=(), schema=None):
        r = super()._deep_flat_encode(
            cache, blobs, files, leaves_to_keep=leaves_to_keep, schema=schema
        )
        cache[r[1:]]['_path'] = 'superduperdb/backends/mongodb/query/parse_query'
        return r

    @property
    def type(self):
        return defaultdict(
            lambda: 'select',
            {
                'update_many': 'update',
                'update_one': 'update',
                'delete_many': 'delete',
                'delete_one': 'delete',
                'bulk_write': 'write',
                'insert_many': 'insert',
                'insert_one': 'insert',
                'outputs': 'outputs',
            },
        )[self.flavour]

    def _prepare_inputs(self, inputs):
        if isinstance(inputs, BulkOp):
            return inputs.op
        if isinstance(inputs, (list, tuple)):
            return [self._prepare_inputs(i) for i in inputs]
        if isinstance(inputs, dict):
            return {k: self._prepare_inputs(v) for k, v in inputs.items()}
        return inputs

    def _execute_delete_one(self, parent):
        r = next(self.select_ids.limit(1)._execute(parent))
        self.table_or_collection.delete_one({'_id': r['_id']})._execute(parent)
        return [str(r['_id'])]

    def _execute_delete_many(self, parent):
        id_cursor = self.select_ids._execute(parent)
        ids = [r['_id'] for r in id_cursor]
        if not ids:
            return {}
        self.table_or_collection.delete_many({'_id': {'$in': ids}})._execute(parent)
        return [str(id) for id in ids]

    def _execute(self, parent, method='encode'):
        c = super()._execute(parent, method=method)
        import mongomock
        import pymongo

        if isinstance(c, (pymongo.cursor.Cursor, mongomock.collection.Cursor)):
            return SuperDuperCursor(
                raw_cursor=c,
                db=self.db,
                id_field='_id',
            )
        return c

    def _execute_pre_like(self, parent):
        assert self.parts[0][0] == 'like'
        assert self.parts[1][0] in ['find', 'find_one']
        like_args, like_kwargs = self.parts[0][1:]
        like_args = list(like_args)
        if not like_args:
            like_args = [{}]
        r = like_args[0] or like_kwargs.pop('r', {})
        if isinstance(r, Document):
            r = r.unpack()

        vector_index = like_kwargs.pop('vector_index')
        ids = like_kwargs.pop('within_ids', [])

        n = like_kwargs.pop('n', 100)

        ids, scores = self.db.select_nearest(
            like=r,
            vector_index=vector_index,
            ids=ids,
            n=n,
        )

        scores = dict(zip(ids , scores))
        find_args = self.parts[1][1]
        find_kwargs = self.parts[1][2]
        if find_args == ():
            find_args = ({},)
        find_args[0]['_id'] = {'$in': [ObjectId(id) for id in ids]}

        q = type(self)(
            db=self.db,
            identifier=self.identifier,
            parts=[(self.parts[1][0], tuple(find_args), find_kwargs), *self.parts[2:]],
        )
        result = q.execute(db=self.db)
        result.scores = scores
        return result

    def _execute_post_like(self, parent):
        assert len(self.parts) == 2
        assert self.parts[0][0] == 'find'
        assert self.parts[1][0] == 'like'

        find_args = self.parts[0][1]
        find_kwargs = self.parts[0][2]

        like_args = self.parts[1][1]

        like_kwargs = self.parts[1][2]
        r = like_args[0] or like_kwargs.pop('r')
        if isinstance(r, Document):
            r = r.unpack()
        range = like_kwargs.pop('range', None)

        parent_query = self[:-1].select_ids
        if range:
            parent_query = parent_query.limit(range)

        relevant_ids = [str(r['_id']) for r in parent_query.execute()]

        similar_ids, scores = self.db.select_nearest(
            like=r,
            ids=relevant_ids,
            vector_index = like_kwargs.pop('vector_index'),
            n=like_kwargs.get('n', 100),
        )
        scores = dict(zip(similar_ids, scores))
        similar_ids = [ObjectId(id) for id in similar_ids]

        final_args = find_args[:]
        if not final_args:
            final_args = [{}]
        final_args[0]['_id'] = {'$in': similar_ids}

        final_query = self.table_or_collection.find(*final_args, **find_kwargs)
        result = final_query._execute(parent)

        result.scores = scores
        return result

    def _execute_bulk_write(self, parent):
        for part in self.parts:
            if isinstance(part, str):
                parent = getattr(parent, part)
                continue
            args = self._prepare_inputs(part[1])
            kwargs = self._prepare_inputs(part[2])
            parent = getattr(parent, part[0])(*args, **kwargs)
        return parent

    def _execute_find(self, parent):
        return self._execute(parent, method='unpack')

    def _execute_replace_one(self, parent):
        documents = self.parts[0][1][0]
        trailing_args = list(self.parts[0][1][1:])
        kwargs = self.parts[0][2]

        schema = kwargs.pop('schema', None)

        replacement = trailing_args[0]
        if isinstance(replacement, Document):
            replacement = replacement.encode(schema)
        trailing_args.insert(0, replacement)

        q = self.table_or_collection.replace_one(documents, *trailing_args, **kwargs)
        q._execute(parent)

    def _execute_find_one(self, parent):
        r = self._execute(parent, method='unpack')
        if r is None:
            return
        return Document.decode(r, db=self.db)

    def _execute_insert_one(self, parent):
        insert_part = self.parts[0]
        parts = self.parts[1:]
        insert_part = (insert_part[0], [insert_part[1]], insert_part[2])

        self.parts = [insert_part] + parts
        return self._execute_insert_many(parent)

    def _execute_insert_many(self, parent):
        documents = self.parts[0][1][0]
        trailing_args = self.parts[0][1][1:]
        kwargs = self.parts[0][2]
        schema = kwargs.pop('schema', None)

        documents = [
            r.encode(schema) if isinstance(r, Document) else r for r in documents
        ]
        for r in documents:
            if '_blobs' in r:
                for file_id, bytes in r['_blobs'].items():
                    self.db.artifact_store._save_bytes(
                        bytes,
                        file_id,
                    )
                r['_blobs'] = list(r['_blobs'].keys())
        q = self.table_or_collection.insert_many(documents, *trailing_args, **kwargs)
        result = q._execute(parent)
        return result.inserted_ids

    def _execute_update_many(self, parent):
        ids = [r['_id'] for r in self.select_ids._execute(parent)]
        filter = self.parts[0][1][0]
        trailing_args = list(self.parts[0][1][1:])
        update = {}

        # Encode update document
        for ix, arg in enumerate(trailing_args):
            if '$set' in arg:
                if isinstance(arg, Document):
                    update = QueryUpdateDocument.from_document(arg)
                else:
                    update = arg
                del trailing_args[ix]
                break

        kwargs = self.parts[0][2]
        filter['_id'] = {'$in': ids}
        trailing_args.insert(0, update.encode())

        parent.update_many(filter, *trailing_args, **kwargs)
        return ids

    @applies_to('find')
    def outputs(self, *predict_ids):
        find_args, find_kwargs = self.parts[0][1:]
        find_args = list(find_args)

        if not find_args:
            find_args = [{}]

        if not find_args[1:]:
            find_args.append({})
        for identifier in predict_ids:
            find_args[1][f'_outputs.{identifier}'] = 1
            find_args[1]['_leaves'] = 1
            find_args[1]['_files'] = 1
            find_args[1]['_blobs'] = 1
        x = type(self)(
            db=self.db,
            identifier=self.identifier,
            parts=[
                ('find', tuple(find_args), find_kwargs),
                *self.parts[1:],
            ],
        )
        return x

    @applies_to('find')
    def add_fold(self, fold: str):
        find_args, find_kwargs = self.parts[0][1:]
        find_args = list(find_args)
        if not find_args:
            find_args = [{}]
        find_args[0]['_fold'] = fold
        return type(self)(
            db=self.db,
            identifier=self.identifier,
            parts=[
                ('find', tuple(find_args), find_kwargs),
                *self.parts[1:],
            ],
        )

    @property
    @applies_to('insert_many', 'insert_one')
    def documents(self):
        if self.parts[0][0] == 'insert_many':
            return self.parts[0][1][0]
        return [self.parts[0][1][0]]

    @property
    def primary_id(self):
        return '_id'

    @applies_to('find')
    def select_using_ids(self, ids: t.Sequence[str]):
        args, kwargs = self.parts[0][1:]
        args = list(args)[:]
        if not args:
            args = [{}]
        args[0]['_id'] = {'$in': [ObjectId(id) for id in ids]}
        return type(self)(
            db=self.db,
            identifier=self.identifier,
            parts=[
                ('find', args, kwargs),
                *self.parts[1:],
            ],
        )

    @property
    @applies_to('find', 'update_many', 'delete_many', 'delete_one')
    def select_ids(self):
        filter_ = {}
        if self.parts[0][1]:
            filter_ = self.parts[0][1][0]
        projection = {'_id': 1}
        coll = type(self)(identifier=self.identifier, db=self.db)
        return coll.find(filter_, projection)

    @property
    def table_or_collection(self):
        return type(self)(identifier=self.identifier, db=self.db)

    @applies_to('find')
    def select_ids_of_missing_outputs(self, predict_id: str):
        args, kwargs = self.parts[0][1:]
        if args:
            args = [
                {
                    '$and': [
                        args[0],
                        {f'_outputs.{predict_id}': {'$exists': 0}},
                    ]
                },
                *args[1:],
            ]
        else:
            args = [{f'_outputs.{predict_id}': {'$exists': 0}}]

        if len(args) == 1:
            args.append({})
        args[1] = {'_id': 1}

        return self.table_or_collection.find(*args, **kwargs)

    @property
    @applies_to('find')
    def select_single_id(self, id: str):
        args, kwargs = self.parts[0][1:]
        args = list(self.args)[:]
        if not args:
            args[0] = {}
        args[0]['_id'] = ObjectId(id)
        return type(self)(
            db=self.db, identifier=self.identifier, parts=[('find_one', args, kwargs)]
        )

    @property
    def select_table(self):
        return self.table_or_collection.find()

    def model_update(
        self,
        ids: t.List[t.Any],
        predict_id: str,
        outputs: t.Sequence[t.Any],
        flatten: bool = False,
        **kwargs,
    ):
        if not len(outputs):
            return

        document_embedded = kwargs.get('document_embedded', True)
        if document_embedded:
            if flatten:
                raise AttributeError(
                    'Flattened outputs cannot be stored along with input documents.'
                    'Please use `document_embedded = False` option with flatten = True'
                )

            bulk_operations = []
            for i, id in enumerate(ids):
                mongo_filter = {'_id': ObjectId(id)}

                output = outputs[i]

                if isinstance(output, dict) and '_base' in output:
                    output = output['_base']

                update = {f'_outputs.{predict_id}': output}

                update = QueryUpdateDocument._create_metadata_update(update, outputs[i])
                update = Document(update)
                bulk_operations.append(
                    UpdateOne(
                        filter=mongo_filter,
                        update=update,
                    )
                )
            return self.table_or_collection.bulk_write(bulk_operations)
        else:
            collection = MongoQuery(f'_outputs.{predict_id}')
            documents = []
            if flatten:
                for i, id in enumerate(ids):
                    _outputs = outputs[i]
                    if isinstance(_outputs, (list, tuple)):
                        for offset, output in enumerate(_outputs):
                            documents.append(
                                {
                                    f'_outputs': {predict_id: output},
                                    '_source': ObjectId(id),
                                    '_offset': offset,
                                }
                            )
                    else:
                        documents.append(
                            {
                                f'_outputs': {predict_id: outputs[i]},
                                '_source': ObjectId(id),
                                '_offset': 0,
                            }
                        )

            else:
                for i, id in enumerate(ids):
                    documents.append(
                        {
                            '_id': ObjectId(id),
                            f'_outputs': {predict_id: outputs[i]},
                        }
                    )
            return collection.insert_many(documents)

    @staticmethod
    def infer_schema(data: t.Mapping[str, t.Any], identifier: t.Optional[str] = None):
        """
        Infer a schema from a given data object

        :param data: The data object
        :param identifier: The identifier for the schema, if None, it will be generated
        :return: The inferred schema
        """
        from superduperdb.misc.auto_schema import infer_schema

        return infer_schema(data, identifier)


def InsertOne(**kwargs):
    return BulkOp(identifier='InsertOne', kwargs=kwargs)


def UpdateOne(**kwargs):
    return BulkOp(identifier='UpdateOne', kwargs=kwargs)


def DeleteOne(**kwargs):
    return BulkOp(identifier='DeleteOne', kwargs=kwargs)


def ReplaceOne(**kwargs):
    return BulkOp(identifier='ReplaceOne', kwargs=kwargs)


@dc.dataclass(kw_only=True)
class BulkOp(Leaf):
    ops: t.ClassVar[t.Sequence[str]] = [
        'InsertOne',
        'UpdateOne',
        'DeleteOne',
        'ReplaceOne',
    ]
    identifier: str
    kwargs: t.Dict = dc.field(default_factory=dict)

    def __post_init__(self, db):
        super().__post_init__(db)
        assert self.identifier in self.ops

    @property
    def op(self):
        import pymongo

        kwargs = {**self.kwargs}
        for k, v in self.kwargs.items():
            if isinstance(v, Document):
                kwargs[k] = v.unpack()
        return getattr(pymongo, self.identifier)(**kwargs)
