import base64
import copy
import dataclasses as dc
import pickle
import typing as t
import uuid
from collections import defaultdict

import pandas

from superduperdb import Document
from superduperdb.backends.base.query import (
    Query,
    applies_to,
    parse_query as _parse_query,
)
from superduperdb.base.cursor import SuperDuperCursor
from superduperdb.base.exceptions import DatabackendException
from superduperdb.components.schema import Schema
from superduperdb.misc.annotations import merge_docstrings
from superduperdb.misc.special_dicts import SuperDuperFlatEncode

if t.TYPE_CHECKING:
    from superduperdb.base.datalayer import Datalayer


def parse_query(
    query, documents: t.Sequence[t.Dict] = (), db: t.Optional['Datalayer'] = None
):
    """Parse a string query into a query object.

    :param query: The query to parse.
    :param documents: The documents to query.
    :param db: The datalayer to use to execute the query.
    """
    return _parse_query(
        query=query,
        documents=list(documents),
        builder_cls=IbisQuery,
        db=db,
    )


def _load_keys_with_blob(output):
    if isinstance(output, SuperDuperFlatEncode):
        return output.load_keys_with_blob()
    elif isinstance(output, dict):
        return SuperDuperFlatEncode(output).load_keys_with_blob()
    return output


def _model_update_impl_flatten(
    db,
    ids: t.List[t.Any],
    predict_id: str,
    outputs: t.Sequence[t.Any],
):
    """Flatten the outputs and ids and update the model outputs in the database."""
    flattened_outputs = []
    flattened_ids = []
    for output, id in zip(outputs, ids):
        assert isinstance(output, (list, tuple)), 'Expected list or tuple'
        for o in output:
            flattened_outputs.append(o)
            flattened_ids.append(id)

    return _model_update_impl(
        db=db,
        ids=flattened_ids,
        predict_id=predict_id,
        outputs=flattened_outputs,
    )


def _model_update_impl(
    db,
    ids: t.List[t.Any],
    predict_id: str,
    outputs: t.Sequence[t.Any],
):
    if not outputs:
        return

    documents = []
    for output, source_id in zip(outputs, ids):
        d = {
            '_source': str(source_id),
            'output': output,
            'id': str(uuid.uuid4()),
        }
        documents.append(Document(d))
    return db[f'_outputs.{predict_id}'].insert(documents)


@merge_docstrings
@dc.dataclass(kw_only=True, repr=False)
class IbisQuery(Query):
    """A query that can be executed on an Ibis database."""

    flavours: t.ClassVar[t.Dict[str, str]] = {
        'pre_like': '^.*\.like\(.*\)\.select',
        'post_like': '^.*\.([a-z]+)\(.*\)\.like(.*)$',
        'insert': '^[^\(]+\.insert\(.*\)$',
        'filter': '^[^\(]+\.filter\(.*\)$',
        'delete': '^[^\(]+\.delete\(.*\)$',
        'select': '^[^\(]+\.select\(.*\)$',
        'join': '^.*\.join\(.*\)$',
        'anti_join': '^[^\(]+\.anti_join\(.*\)$',
    }

    def _deep_flat_encode(self, cache, blobs, files, leaves_to_keep=(), schema=None):
        r = super()._deep_flat_encode(
            cache, blobs, files, leaves_to_keep=leaves_to_keep, schema=schema
        )
        cache[r[1:]]['_path'] = 'superduperdb/backends/ibis/query/parse_query'
        return r

    @property
    @applies_to('insert')
    def documents(self):
        """Return the documents."""

        def _wrap_document(document):
            if not isinstance(document, Document):
                if isinstance(document, dict):
                    document = Document(document)
                else:
                    schema = self.db[self.identifier]._get_schema()
                    field = [
                        k
                        for k in schema.fields
                        if k not in [self.primary_id, '_fold', '_outputs']
                    ]
                    assert len(field) == 1
                    document = Document({field[0]: document})
            return document

        def _update_part(documents):
            nonlocal self
            doc_args = (documents, *self.parts[0][1][1:])
            insert_part = (self.parts[0][0], doc_args, self.parts[0][2])
            return [insert_part] + self.parts[1:]

        documents = self.parts[0][1][0]
        wrapped_documents = []
        for document in documents:
            document = _wrap_document(document)
            wrapped_documents.append(document)
        self.parts = _update_part(wrapped_documents)
        return wrapped_documents

    def _get_tables(self):
        out = {self.identifier: self.db.tables[self.identifier]}

        for part in self.parts:
            if isinstance(part, str):
                return out
            args = part[1]
            for a in args:
                if isinstance(a, IbisQuery):
                    out.update(a._get_tables())
            kwargs = part[2].values()
            for v in kwargs:
                if isinstance(v, IbisQuery):
                    out.update(v._get_tables())
        return out

    def _get_schema(self):
        fields = {}
        tables = self._get_tables()

        table_renamings = self.renamings({})
        if len(tables) == 1 and not table_renamings:
            return self.db.tables[self.identifier].schema
        for identifier, c in tables.items():
            renamings = table_renamings.get(identifier, {})

            tmp = c.schema.fields
            to_update = dict(
                (renamings[k], v) if k in renamings else (k, v) for k, v in tmp.items()
            )
            fields.update(to_update)

        return Schema(f'_tmp:{self.identifier}', fields=fields)

    def renamings(self, r={}):
        """Return the renamings.

        :param r: Renamings.
        """
        for part in self.parts:
            if isinstance(part, str):
                continue
            if part[0] == 'rename':
                r[self.identifier] = part[1][0]
            if part[0] == 'relabel':
                r[self.identifier] = part[1][0]
            else:
                queries = list(part[1]) + list(part[2].values())
                for query in queries:
                    if isinstance(query, IbisQuery):
                        query.renamings(r)
        return r

    def _execute_pre_like(self, parent):
        assert self.parts[0][0] == 'like'
        assert self.parts[1][0] in ['select']
        similar_ids, similar_scores = self._prepare_pre_like(parent)

        t = self.db[self.identifier]
        filter_query = t.select_using_ids(similar_ids)
        query = type(self)(
            db=self.db,
            identifier=self.identifier,
            parts=[
                *filter_query.parts,
                *self.parts[1:],
            ],
        )
        result = query.do_execute(db=self.db)
        result.scores = similar_scores
        return result

    def __eq__(self, other):
        return super().__eq__(other)

    def __leq__(self, other):
        return super().__leq__(other)

    def __geq__(self, other):
        return super().__geq__(other)

    def _execute_post_like(self, parent):
        pre_like_parts = []
        like_part = []
        like_part_index = 0
        for i, part in enumerate(self.parts):
            if not isinstance(part, str):
                if part[0] == 'like':
                    like_part = part
                    like_part_index = i
                    break
            pre_like_parts.append(part)
        post_like_parts = self.parts[like_part_index + 1 :]

        like_args = like_part[1]
        like_kwargs = like_part[2]
        vector_index = like_kwargs['vector_index']
        like = like_args[0] if like_args else like_kwargs['r']
        if isinstance(like, Document):
            like = like.unpack()
        pre_like_query = IbisQuery(
            db=self.db, identifier=self.identifier, parts=pre_like_parts
        )
        within_ids = [
            r[self.primary_id] for r in pre_like_query.select_ids._execute(parent)
        ]
        similar_ids, similar_scores = self.db.select_nearest(
            like, vector_index=vector_index, n=like_kwargs.get('n', 10), ids=within_ids
        )
        similar_scores = dict(zip(similar_ids, similar_scores))

        t = self.db[pre_like_query._get_parent().get_name()]
        filter_query = pre_like_query.filter(
            getattr(t, self.primary_id).isin(similar_ids)
        )

        parts = filter_query.parts + post_like_parts

        q = IbisQuery(db=self.db, identifier=self.identifier, parts=parts)
        outputs = q._execute(parent)
        outputs.scores = similar_scores
        return outputs

    def _execute_select(self, parent):
        return self._execute(parent)

    def _execute_insert(self, parent):
        documents = self._prepare_documents()
        documents = [self._refactor_insert_data_info(d) for d in documents]
        for r in documents:
            if self.primary_id not in r:
                pid = str(uuid.uuid4())
                r[self.primary_id] = pid
        ids = [r[self.primary_id] for r in documents]
        self.db.databackend.insert(self.identifier, raw_documents=documents)
        return ids

    def _create_table_if_not_exists(self):
        tables = self.db.databackend.list_tables_or_collections()
        if self.identifier in tables:
            return
        self.db.databackend.create_table_and_schema(
            self.identifier,
            self._get_schema(),
        )

    def _execute(self, parent, method='encode'):
        q = super()._execute(parent, method=method)
        try:
            output = q.execute()
        except Exception as e:
            raise DatabackendException(
                f'Error while executing ibis query {self}'
            ) from e

        assert isinstance(output, pandas.DataFrame)
        output = output.to_dict(orient='records')
        output = [self._recover_insert_data_info(d) for d in output]
        component_table = self.db.tables[self.identifier]
        return SuperDuperCursor(
            raw_cursor=output,
            db=self.db,
            id_field=component_table.primary_id,
            schema=self._get_schema(),
        )

    @staticmethod
    def _refactor_insert_data_info(data):
        merge_data = {}
        for key in ['_leaves', '_blobs', '_files', '_schema']:
            if key in data:
                merge_data[key] = data.pop(key)

        merge_data = base64.b64encode(pickle.dumps(merge_data)).decode()
        data['_info'] = merge_data
        return data

    @staticmethod
    def _recover_insert_data_info(data):
        if '_info' in data:
            merge_data = pickle.loads(base64.b64decode(data.pop('_info')))
            data.update(merge_data)
        return data

    @property
    def type(self):
        """Return the type of the query."""
        return defaultdict(
            lambda: 'select',
            {
                'replace': 'update',
                'delete': 'delete',
                'filter': 'select',
                'insert': 'insert',
            },
        )[self.flavour]

    @property
    def primary_id(self):
        """Return the primary id."""
        return self.db.tables[self.identifier].primary_id

    def model_update(
        self,
        ids: t.List[t.Any],
        predict_id: str,
        outputs: t.Sequence[t.Any],
        flatten: bool = False,
        **kwargs,
    ):
        """Update the model outputs in the database.

        :param ids: The ids of the inputs.
        :param predict_id: The predict id.
        :param outputs: The outputs.
        :param flatten: Whether to flatten the outputs.
        :param kwargs: Additional keyword arguments.
        """
        if not flatten:
            return _model_update_impl(
                db=self.db,
                ids=ids,
                predict_id=predict_id,
                outputs=outputs,
            )
        else:
            return _model_update_impl_flatten(
                db=self.db,
                ids=ids,
                predict_id=predict_id,
                outputs=outputs,
            )

    def add_fold(self, fold: str):
        """Return a query that adds a fold.

        :param fold: The fold to add.
        """
        return self.filter(self._fold == fold)

    def select_using_ids(self, ids: t.Sequence[str]):
        """Return a query that selects using ids.

        :param ids: The ids to select.
        """
        filter_query = self.filter(getattr(self, self.primary_id).isin(ids))
        return filter_query

    @property
    def select_ids(self):
        """Return a query that selects ids."""
        return self.select(self.primary_id)

    @applies_to('select')
    def outputs(self, *predict_ids):
        """Return a query that selects outputs.

        :param predict_ids: The predict ids.
        """
        find_args = ()
        if self.parts:
            find_args, _ = self.parts[0][1:]
        find_args = copy.deepcopy(list(find_args))

        if not find_args:
            find_args = [{}]

        if not find_args[1:]:
            find_args.append({})

        for identifier in predict_ids:
            identifier = (
                identifier if '_outputs' in identifier else f'_outputs.{identifier}'
            )
            symbol_table = self.db[identifier]

            symbol_table = symbol_table.relabel(
                # TODO: Check for folds
                {'output': identifier, '_fold': f'fold.{identifier}'}
            )

            attr = getattr(self, self.primary_id)
            other_query = self.join(symbol_table, symbol_table._source == attr)
            other_query._get_schema()
            return other_query

    @applies_to('select', 'join')
    def select_ids_of_missing_outputs(self, predict_id: str):
        """Return a query that selects ids of missing outputs.

        :param predict_id: The predict id.
        """
        output_table = self.db[f'_outputs.{predict_id}']
        output_table = output_table.relabel({'output': '_outputs.' + predict_id})
        return self.anti_join(
            output_table,
            output_table._source == getattr(self, self.primary_id),
        )

    def select_single_id(self, id: str):
        """Return a query that selects a single id.

        :param id: The id to select.
        """
        filter_query = eval(f'table.{self.primary_id} == {id}')
        return self.filter(filter_query)

    @property
    def select_table(self):
        """Return a query that selects the table."""
        t = self.db[self.table_or_collection.identifier]
        return t.select(t)

    def __call__(self, *args, **kwargs):
        """Add a method call to the query.

        :param args: The arguments to pass to the method.
        :param kwargs: The keyword arguments to pass to the method.
        """
        assert isinstance(self.parts[-1], str)
        if self.parts[-1] == 'select' and not args:
            # support table.select() without column args
            table = self.db.databackend.get_table_or_collection(self.identifier)
            args = tuple(table.columns)
        return super().__call__(*args, **kwargs)

    def compile(self, db):
        """
        Compile `IbisQuery` to native ibis query format.

        :param db: Datalayer instance.
        """
        parent = self._get_parent()
        return super()._execute(parent).compile()


class _SQLDictIterable:
    def __init__(self, iterable):
        self.iterable = iter(iterable)

    def next(self):
        element = next(self.iterable)
        return dict(element)

    def __iter__(self):
        return self

    __next__ = next


@dc.dataclass
class RawSQL:
    """Raw SQL query.

    :param query: The raw SQL query
    :param id_field: The field to use as the primary id
    """

    query: str
    id_field: str = 'id'
    type: t.ClassVar[str] = 'select'

    def do_execute(self, db):
        """Run the query.

        :param db: The DataLayer instance
        """
        cursor = db.databackend.conn.raw_sql(self.query)
        try:
            cursor = cursor.mappings().all()
            cursor = _SQLDictIterable(cursor)
            return SuperDuperCursor(cursor, id_field=self.id_field)
        except Exception:
            return cursor
