import copy
import dataclasses as dc
import functools
import re
import typing as t
from collections import defaultdict

import pymongo
from bson import ObjectId
from superduper import CFG, logging
from superduper.backends.base.query import (
    Query,
    applies_to,
    parse_query as _parse_query,
)
from superduper.base.cursor import SuperDuperCursor
from superduper.base.document import Document, QueryUpdateDocument
from superduper.base.leaf import Leaf

if t.TYPE_CHECKING:
    from superduper.base.datalayer import Datalayer

_SPECIAL_CHRS: list = ['$', '.']

OPS_MAP = {
    '__eq__': '$eq',
    '__ne__': '$ne',
    '__lt__': '$lt',
    '__le__': '$lte',
    '__gt__': '$gt',
    '__ge__': '$gte',
    'isin': '$in',
}


def _serialize_special_character(d, to='encode'):
    def extract_character(s):
        pattern = r'<(.)>'
        match = re.search(pattern, s)
        if match:
            return match.group(1)
        return None

    if not isinstance(d, dict):
        return d

    new_dict = {}
    for key, value in d.items():
        new_key = key
        if isinstance(key, str):
            if to == 'encode':
                if key[0] in _SPECIAL_CHRS:
                    new_key = f'<{key[0]}>' + key[1:]
            elif to == 'decode':
                k = extract_character(key[:3])
                if k in _SPECIAL_CHRS:
                    new_key = k + key[3:]

        if isinstance(value, dict):
            new_dict[new_key] = _serialize_special_character(value, to=to)
        elif isinstance(value, list):
            new_dict[new_key] = [
                (
                    _serialize_special_character(item, to=to)
                    if isinstance(item, dict)
                    else item
                )
                for item in value
            ]
        else:
            new_dict[new_key] = value

    return new_dict


def parse_query(
    query, documents: t.Sequence[t.Dict] = (), db: t.Optional['Datalayer'] = None
):
    """Parse a string query into a query object.

    :param query: The query to parse.
    :param documents: The documents to query.
    :param db: The datalayer to use to execute the query.
    """
    _decode = functools.partial(_serialize_special_character, to='decode')
    documents = list(map(_decode, documents))
    return _parse_query(
        query=query,
        builder_cls=MongoQuery,
        documents=list(documents),
        db=db,
    )


class MongoQuery(Query):
    """A query class for MongoDB.

    This class is used to build and execute queries on a MongoDB database.
    """

    flavours: t.ClassVar[t.Dict[str, str]] = {
        'pre_like': r'^.*\.like\(.*\)\.(find|select)',
        'post_like': r'^.*\.(find|select)\(.*\)\.like(.*)$',
        'bulk_write': r'^.*\.bulk_write\(.*\)$',
        'outputs': r'^.*\.outputs\(.*\)',
        'missing_outputs': r'^.*\.missing_outputs\(.*\)$',
        'find_one': r'^.*\.find_one\(.*\)',
        'find': r'^.*\.find\(.*\)',
        'select': r'^.*\.select\(.*\)$',
        'insert_one': r'^.*\.insert_one\(.*\)$',
        'insert_many': r'^.*\.insert_many\(.*\)$',
        'insert': r'^.*\.insert\(.*\)$',
        'replace_one': r'^.*\.replace_one\(.*\)$',
        'update_many': r'^.*\.update_many\(.*\)$',
        'update_one': r'^.*\.update_one\(.*\)$',
        'delete_many': r'^.*\.delete_many\(.*\)$',
        'delete_one': r'^.*\.delete_one\(.*\)$',
        'other': '.*',
    }

    # Use to control the behavior in the class construction method within LeafMeta
    _dataclass_params: t.ClassVar[t.Dict[str, t.Any]] = {
        'eq': False,
        'order': False,
    }

    def _create_table_if_not_exists(self):
        return

    def dict(self, metadata: bool = True, defaults: bool = True, uuid: bool = True):
        """Return the query as a dictionary."""
        r = super().dict()
        r['documents'] = list(map(_serialize_special_character, r['documents']))
        return r

    @property
    def type(self):
        """Return the type of the query."""
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
                'insert': 'insert',
                'outputs': 'select',
            },
        )[self.flavour]

    @property
    def _is_select_find(self):
        return self.parts and any([p[0] == 'select' for p in self.parts])

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

    def _execute_bulk_write(self, parent):
        """Execute the query.

        :param db: The datalayer instance
        """
        assert self.parts[0][0] == 'bulk_write'
        operations = self.parts[0][1][0]
        for query in operations:
            assert isinstance(query, (BulkOp))

            query.is_output_query = self.is_output_query
            if not query.kwargs.get('arg_ids', None):
                raise ValueError(
                    'Please provided update/delete id in args',
                    r'all ids selection e.g `\{\}` is not supported',
                )

        collection = self.db.databackend.get_table_or_collection(self.table)
        bulk_operations = []
        bulk_update_ids = []
        bulk_delete_ids = []
        bulk_result = {'delete': [], 'update': []}
        for query in operations:
            operation = query.op

            bulk_operations.append(operation)
            ids = query.kwargs['arg_ids']
            if query.identifier == 'DeleteOne':
                bulk_result['delete'].append({'query': query, 'ids': ids})
                bulk_delete_ids += ids
            else:
                bulk_update_ids += ids
                bulk_result['update'].append({'query': query, 'ids': ids})

        result = collection.bulk_write(bulk_operations)
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

    def filter(self, *args, **kwargs):
        """Return a query that filters the documents.

        :param args: The arguments to filter by.
        :param kwargs: Additional keyword arguments.
        """
        filters = {}
        for arg in args:
            if isinstance(arg, dict):
                filters.update(arg)
                continue
            if not isinstance(arg, Query):
                raise ValueError(f'Filter arguments must be queries, but got: {arg}')
            assert len(arg.parts) > 1, f'Invalid filter query: {arg}'
            key, (op, op_args, _) = arg.parts[-2:]

            if op not in OPS_MAP:
                raise ValueError(
                    f'Operation {op} not supported, '
                    f'supported operations are: {OPS_MAP.keys()}'
                )

            if not op_args:
                raise ValueError(f'No arguments found for operation {op}')

            value = op_args[0]

            filters[key] = {OPS_MAP[op]: value}

        query = self if self.parts else self.select()

        return type(self)(
            db=self.db,
            table=self.table,
            parts=[*query.parts, ('filter', (filters,), kwargs)],
        )

    def _execute_find(self, parent):
        return self._execute_select(parent)

    def _execute_replace_one(self, parent):
        documents = self.parts[0][1][0]
        trailing_args = list(self.parts[0][1][1:])
        kwargs = self.parts[0][2]

        schema = kwargs.pop('schema', None)

        replacement = trailing_args[0]
        if isinstance(replacement, Document):
            replacement = replacement.encode(schema)
        trailing_args[0] = replacement

        q = self.table_or_collection.replace_one(documents, *trailing_args, **kwargs)
        q._execute(parent)

    def _execute_find_one(self, parent):
        r = self._execute_select(parent)
        if r is None:
            return
        return Document.decode(r, db=self.db, schema=self._get_schema())

    def _execute_insert_one(self, parent):
        insert_part = self.parts[0]
        parts = self.parts[1:]
        insert_part = ('insert_many', [insert_part[1]], insert_part[2])

        self.parts = [insert_part] + parts
        return self._execute_insert_many(parent)

    def _execute_insert_many(self, parent):
        trailing_args = self.parts[0][1][1:]
        kwargs = self.parts[0][2]
        documents = self._prepare_documents()
        q = self.table_or_collection.insert_many(documents, *trailing_args, **kwargs)
        result = q._execute(parent)
        return result.inserted_ids

    def _execute_insert(self, parent):
        """Provide a unified insertion interface."""
        return self._execute_insert_many(parent)

    def _execute_update_many(self, parent):
        ids = [r['_id'] for r in self.select_ids._execute(parent)]
        filter = self.parts[0][1][0]
        trailing_args = list(self.parts[0][1][1:])
        update = {}
        kwargs = self.parts[0][2]

        # Encode update document
        for ix, arg in enumerate(trailing_args):
            if '$set' in arg:
                if isinstance(arg, Document):
                    update = QueryUpdateDocument.from_document(arg)
                else:
                    update = arg
                del trailing_args[ix]
                break

        filter['_id'] = {'$in': ids}

        try:
            table = self.db.load('table', self.table)
            schema = table.schema
        except FileNotFoundError:
            schema = None

        trailing_args.insert(
            0, update.encode(schema=schema) if isinstance(update, Document) else update
        )

        parent.update_many(filter, *trailing_args, **kwargs)
        return ids

    def drop_outputs(self, predict_id: str):
        """Return a query that removes output corresponding to the predict id.

        :param predict_ids: The ids of the predictions to select.
        """
        return self.db.databackend.drop_table_or_collection(
            f'{CFG.output_prefix}{predict_id}'
        )

    @applies_to('find', 'find_one', 'select', 'outputs')
    def add_fold(self, fold: str):
        """Return a query that adds a fold to the query.

        :param fold: The fold to add.
        """
        return self.filter(self['_fold'] == fold)

    @property
    @applies_to('insert_many', 'insert_one', 'insert')
    def documents(self):
        """Return the documents."""
        return super().documents

    @property
    def primary_id(self):
        """Return the primary id of the documents."""
        return '_id'

    def select_using_ids(self, ids: t.Sequence[str]):
        """Return a query that selects using the given ids.

        :param ids: The ids to select.
        """
        ids = [ObjectId(id) for id in ids]
        q = self.filter(self['_id'].isin(ids))
        return q

    @property
    def select_ids(self):
        """Select the ids of the documents."""
        if self._is_select_find:
            part = self.parts[0]
            new_part = ('select', ('_id',), part[2])
            return type(self)(
                db=self.db,
                table=self.table,
                parts=[new_part, *self.parts[1:]],
            )

        filter_ = {}
        if self.parts and self.parts[0][1]:
            filter_ = self.parts[0][1][0]
            if isinstance(filter_, str):
                filter_ = {}
        projection = {'_id': 1}
        coll = type(self)(table=self.table, db=self.db)
        return coll.find(filter_, projection)

    def select_ids_of_missing_outputs(self, predict_id: str):
        """Select the ids of documents that are missing the given output.

        :param predict_id: The id of the prediction.
        """
        return self.missing_outputs(predict_id=predict_id, ids_only=1)

    def _execute_missing_outputs(self, parent):
        """Select the documents that are missing the given output."""
        if len(self.parts[-1][2]) == 0:
            raise ValueError("Predict id is required")
        predict_id = self.parts[-1][2]["predict_id"]
        ids_only = self.parts[-1][2].get('ids_only', False)

        key = f'{CFG.output_prefix}{predict_id}'

        lookup = [
            {
                '$lookup': {
                    'from': key,
                    'localField': '_id',
                    'foreignField': '_source',
                    'as': key,
                }
            },
            {'$match': {key: {'$size': 0}}},
        ]

        raw_cursor = getattr(parent, 'aggregate')(lookup)

        def get_ids(result):
            return {"_id": result["_id"]}

        return SuperDuperCursor(
            raw_cursor=raw_cursor,
            db=self.db,
            id_field='_id',
            process_func=get_ids if ids_only else self._postprocess_result,
            schema=self._get_schema(),
        )

    @property
    @applies_to('find')
    def select_single_id(self, id: str):
        """Return a query that selects a single id.

        :param id: The id to select.
        """
        args, kwargs = self.parts[0][1:]
        args = list(self.args)[:]
        if not args:
            args[0] = {}
        args[0]['_id'] = ObjectId(id)
        return type(self)(
            db=self.db, table=self.table, parts=[('find_one', args, kwargs)]
        )

    @property
    def select_table(self):
        """Return the table or collection to select from."""
        return self.table_or_collection.find()

    def model_update(
        self,
        ids: t.List[t.Any],
        predict_id: str,
        outputs: t.Sequence[t.Any],
        flatten: bool = False,
        **kwargs,
    ):
        """Update the model outputs in the database.

        :param ids: The ids of the documents to update.
        :param predict_id: The id of the prediction.
        :param outputs: The outputs to store.
        :param flatten: Whether to flatten the outputs.
        :param kwargs: Additional keyword arguments.
        """
        if flatten:
            flattened_outputs = []
            flattened_ids = []
            for output, id in zip(outputs, ids):
                assert isinstance(output, (list, tuple)), 'Expected list or tuple'
                for o in output:
                    flattened_outputs.append(o)
                    flattened_ids.append(id)
            return self.model_update(
                ids=flattened_ids,
                predict_id=predict_id,
                outputs=flattened_outputs,
                flatten=False,
                **kwargs,
            )

        documents = []
        for output, id in zip(outputs, ids):
            documents.append(
                {
                    f'{CFG.output_prefix}{predict_id}': output,
                    '_source': ObjectId(id),
                }
            )

        from superduper.base.datalayer import Datalayer

        assert isinstance(self.db, Datalayer)
        output_query = self.db[f'{CFG.output_prefix}{predict_id}'].insert_many(
            documents
        )
        output_query.is_output_query = True
        output_query.updated_key = predict_id
        return output_query

    def _replace_part(self, part_name, replace_function):
        parts = copy.deepcopy(self.parts)

        for i, part in enumerate(parts):
            if part[0] == part_name:
                parts[i] = replace_function(part)
                break

        return type(self)(
            db=self.db,
            table=self.table,
            parts=parts,
        )

    def _execute_select(self, parent):
        parts = self._get_select_parts()
        output = self._get_chain_native_query(parent, parts, method='unpack')
        import mongomock
        import pymongo

        if isinstance(output, (pymongo.cursor.Cursor, mongomock.collection.Cursor)):
            return SuperDuperCursor(
                raw_cursor=output,
                db=self.db,
                id_field='_id',
                schema=self._get_schema(),
            )
        return output

    def _get_select_parts(self):
        def process_select_part(part):
            # Convert the select part to a find part
            _, args, _ = part
            projection = {key: 1 for key in args}
            if projection and '_id' not in projection:
                projection['_id'] = 0
            return ('find', ({}, projection), {})

        def process_find_part(part):
            method, args, kwargs = part
            # args: (filter, projection, *args)
            filter = copy.deepcopy(args[0]) if len(args) > 0 else {}
            filter = dict(filter)
            filter.update(self._get_filter_conditions())
            args = tuple((filter, *args[1:]))

            return (method, args, kwargs)

        parts = []
        for part in self.parts:
            if part[0] == 'select':
                part = process_select_part(part)

            if part[0] in {'find', 'find_one'}:
                part = process_find_part(part)

            if part[0] == 'filter':
                continue
            parts.append(part)

        return parts

    def _get_filter_conditions(self):
        filters = {}
        for part in self.parts:
            if part[0] == 'filter':
                sub_filters = part[1][0]
                assert isinstance(sub_filters, dict)
                filters.update(sub_filters)
        return filters

    def isin(self, other):
        """Create an isin query.

        :param other: The value to check against.
        """
        other = [ObjectId(o) for o in other]
        return self._ops('isin', other)

    def _get_method_parameters(self, method):
        args, kwargs = (), {}
        for part in self.parts:
            if part[0] == method:
                assert not args, 'Multiple find operations found'
                assert not kwargs, 'Multiple find operations found'
                args, kwargs = part[1:]

        return args, kwargs

    def _get_predict_ids(self):
        outputs_parts = [p for p in self.parts if p[0] == 'outputs']
        predict_ids = sum([p[1] for p in outputs_parts], ())
        return predict_ids

    def _execute_outputs(self, parent, method='encode'):
        project = self._get_project()

        limit_args, _ = self._get_method_parameters('limit')
        limit = {"$limit": limit_args[0]} if limit_args else None

        pipeline = []
        filter_mapping_base, filter_mapping_outputs = self._get_filter_mapping()
        if filter_mapping_base:
            pipeline.append({"$match": filter_mapping_base})
            project.update({k: 1 for k in filter_mapping_base.keys()})

        predict_ids_in_filter = list(filter_mapping_outputs.keys())

        predict_ids = self._get_predict_ids()
        predict_ids = list(set(predict_ids).union(predict_ids_in_filter))
        # After the join, the complete outputs data can be queried as
        # {CFG.output_prefix}{predict_id}._outputs.{predict_id} : result.
        for predict_id in predict_ids:
            key = f'{CFG.output_prefix}{predict_id}'
            lookup = {
                "$lookup": {
                    "from": key,
                    "localField": "_id",
                    "foreignField": "_source",
                    "as": key,
                }
            }

            project[key] = 1
            pipeline.append(lookup)

            if predict_id in filter_mapping_outputs:
                filter_key, filter_value = list(
                    filter_mapping_outputs[predict_id].items()
                )[0]
                pipeline.append({"$match": {f'{key}.{filter_key}': filter_value}})

            pipeline.append(
                {"$unwind": {"path": f"${key}", "preserveNullAndEmptyArrays": True}}
            )

        if project:
            pipeline.append({"$project": project})

        if limit:
            pipeline.append(limit)

        try:
            import json

            logging.debug(f'Executing pipeline: {json.dumps(pipeline, indent=2)}')
        except TypeError:
            pass

        raw_cursor = getattr(parent, 'aggregate')(pipeline)

        return SuperDuperCursor(
            raw_cursor=raw_cursor,
            db=self.db,
            id_field='_id',
            process_func=self._postprocess_result,
            schema=self._get_schema(),
        )

    def _get_schema(self):
        outputs_parts = [p for p in self.parts if p[0] == 'outputs']
        predict_ids = sum([p[1] for p in outputs_parts], ())

        try:
            table = self.db.load('table', self.table)
            if not predict_ids:
                return table.schema
            fields = table.schema.fields
        except FileNotFoundError:
            fields = {}

        for predict_id in predict_ids:
            key = f'{CFG.output_prefix}{predict_id}'
            try:
                output_table = self.db.load('table', key)
            except FileNotFoundError:
                logging.warn(
                    f'No schema found for table {key}. Using default projection'
                )
                continue
            fields[key] = output_table.schema.fields[key]

        from superduper.components.datatype import BaseDataType
        from superduper.components.schema import Schema

        fields = {k: v for k, v in fields.items() if isinstance(v, BaseDataType)}

        return Schema(f"_tmp:{self.table}", fields=fields, db=self.db)

    def _get_project(self):
        find_params, _ = self._get_method_parameters('find')
        select_params, _ = self._get_method_parameters('select')
        if self._is_select_find:
            project = {key: 1 for key in select_params}
        else:
            project = copy.deepcopy(find_params[1]) if len(find_params) > 1 else {}

        if not project:
            try:
                table = self.db.load('table', self.table)
                project = {key: 1 for key in table.schema.fields.keys()}
            except FileNotFoundError:
                logging.warn(
                    'No schema found for table',
                    f'{self.table}. Using default projection',
                )

        return project

    def _get_filter_mapping(self):
        find_params, _ = self._get_method_parameters('find')
        filter = find_params[0] if find_params else {}
        filter.update(self._get_filter_conditions())

        if not filter:
            return {}, {}

        filter_mapping_base = {}
        filter_mapping_outputs = defaultdict(dict)

        for key, value in filter.items():
            if '{CFG.output_prefix}' not in key:
                filter_mapping_base[key] = value
                continue

            if key.startswith('{CFG.output_prefix}'):
                predict_id = key.split('__')[1]
                filter_mapping_outputs[predict_id] = {key: value}

        return filter_mapping_base, filter_mapping_outputs

    def _postprocess_result(self, result):
        """Postprocess the result of the query.

        Merge the outputs from the output keys to the result

        :param result: The result to postprocess.
        """
        merge_outputs = {}
        predict_ids = self._get_predict_ids()
        output_keys = [f"{CFG.output_prefix}{predict_id}" for predict_id in predict_ids]
        for output_key in output_keys:
            output_data = result[output_key]
            output_result = output_data[output_key]
            merge_outputs[output_key] = output_result

        result.update(merge_outputs)
        return result


def InsertOne(**kwargs):
    """InsertOne operation for MongoDB.

    :param kwargs: The arguments to pass to the operation.
    """
    return BulkOp(identifier='InsertOne', kwargs=kwargs)


def UpdateOne(**kwargs):
    """UpdateOne operation for MongoDB.

    :param kwargs: The arguments to pass to the operation.
    """
    try:
        filter = kwargs['filter']
    except Exception as e:
        raise KeyError('Filter not found in `UpdateOne`') from e

    id = filter['_id']
    if isinstance(id, ObjectId):
        ids = [id]
    else:
        ids = id['$in']
    kwargs['arg_ids'] = ids
    return BulkOp(identifier='UpdateOne', kwargs=kwargs)


def DeleteOne(**kwargs):
    """DeleteOne operation for MongoDB.

    :param kwargs: The arguments to pass to the operation.
    """
    return BulkOp(identifier='DeleteOne', kwargs=kwargs)


def ReplaceOne(**kwargs):
    """ReplaceOne operation for MongoDB.

    :param kwargs: The arguments to pass to the operation.
    """
    return BulkOp(identifier='ReplaceOne', kwargs=kwargs)


class BulkOp(Leaf):
    """A bulk operation for MongoDB.

    :param kwargs: The arguments to pass to the operation.
    """

    ops: t.ClassVar[t.Sequence[str]] = [
        'InsertOne',
        'UpdateOne',
        'DeleteOne',
        'ReplaceOne',
    ]
    kwargs: t.Dict = dc.field(default_factory=dict)

    def __post_init__(self, db):
        super().__post_init__(db)
        assert self.identifier in self.ops

    @property
    def op(self):
        """Return the operation."""
        kwargs = copy.deepcopy(self.kwargs)
        kwargs.pop('arg_ids')
        for k, v in kwargs.items():
            if isinstance(v, Document):
                kwargs[k] = v.unpack()
        return getattr(pymongo, self.identifier)(**kwargs)
