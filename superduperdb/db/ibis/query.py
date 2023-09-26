"""
TODO:
- select_ids_of_missing_outputs
"""

import dataclasses as dc
import enum
import re
import typing as t

import ibis

from superduperdb.base.logger import logging
from superduperdb.container.component import Component
from superduperdb.container.encoder import Encoder
from superduperdb.container.schema import Schema
from superduperdb.db.base.cursor import SuperDuperCursor
from superduperdb.db.base.db import DB
from superduperdb.db.base.query import (
    CompoundSelect,
    Insert,
    Like,
    QueryComponent,
    QueryLinker,
    Select,
    TableOrCollection,
    _ReprMixin,
)

PRIMARY_ID: str = 'id'

IbisTableType = t.TypeVar('IbisTableType')
ParentType = t.TypeVar('ParentType')


@dc.dataclass(repr=False)
class IbisCompoundSelect(CompoundSelect):
    """
    A query incorporating vector-search and a standard ``ibis`` query
    """

    __doc__ = __doc__ + CompoundSelect.__doc__  # type: ignore[operator]

    def __eq__(self, __value: object) -> bool:
        assert self.query_linker is not None
        return self.query_linker.__eq__(__value)

    def __lt__(self, __value: object) -> bool:
        assert self.query_linker is not None
        return self.query_linker.__lt__(__value)

    def __gt__(self, __value: object) -> bool:
        assert self.query_linker is not None
        return self.query_linker.__gt__(__value)

    def __and__(self, __value: object) -> bool:
        assert self.query_linker is not None
        return self.query_linker.__and__(__value)

    def __or__(self, __value: object) -> bool:
        assert self.query_linker is not None
        return self.query_linker.__or__(__value)

    def __not__(self) -> bool:
        assert self.query_linker is not None
        return self.query_linker.__not__()

    def __getitem__(self, item) -> bool:
        assert self.query_linker is not None
        return self.query_linker.__getitem__(item)

    def _get_query_linker(cls, table_or_collection, members) -> 'IbisQueryLinker':
        return IbisQueryLinker(table_or_collection=table_or_collection, members=members)

    def _get_query_component(
        self,
        name: str,
        type: str,
        args: t.Optional[t.Sequence] = None,
        kwargs: t.Optional[t.Dict] = None,
    ):
        if args is None:
            args = []
        if kwargs is None:
            kwargs = {}
        return IbisQueryComponent(name, type=type, args=args, kwargs=kwargs)

    def outputs(self, model: str):
        """
        This method returns a query which joins a query with the outputs
        for a table.

        :param model: The model identifier for which to get the outputs

        >>> q = t.filter(t.age > 25).outputs('model_name', db)

        The above query will return the outputs of the `model_name` model
        with t.filter() ids.
        """
        assert self.query_linker is not None
        return IbisCompoundSelect(
            table_or_collection=self.table_or_collection,
            pre_like=self.pre_like,
            query_linker=self.query_linker.outputs(model),
            post_like=self.post_like,
        )

    def compile(self, db: DB, tables: t.Optional[t.Dict] = None):
        """
        Convert the current query to an ``ibis`` native query.

        :param db: The superduperdb connection
        :param tables: A dictionary of ``ibis`` tables to use for the query
        """
        assert self.pre_like is None, "pre_like must be None"
        assert self.post_like is None, "post_like must be None"
        assert self.query_linker is not None, "query_linker must be set"

        table_id = self.table_or_collection.identifier
        if tables is None:
            tables = {}
        if table_id not in tables:
            tables[table_id] = db.databackend.conn.table(table_id)
        return self.query_linker.compile(db, tables=tables)

    def get_all_tables(self):
        tables = [self.table_or_collection.identifier]
        tmp = self.query_linker.get_all_tables()
        tables.extend(tmp)
        tables = list(set(tables))
        return tables

    def _get_all_fields(self, db):
        tables = self.get_all_tables()
        component_tables = []
        for tab in tables:
            component_tables.append(db.load('table', tab))
        fields = {}
        for tab in component_tables:
            fields.update(tab.schema.fields)
        return fields

    @property
    def select_table(self):
        return self.table_or_collection

    def execute(self, db):
        output, scores = super().execute(db)
        fields = self._get_all_fields(db)
        for column in output.columns:
            try:
                type = fields[column]
            except KeyError:
                logging.warn(f'Disambiguation not yet supported of {column}: TODO!')
                continue
            if isinstance(type, Encoder):
                output[column] = output[column].map(type.decode)
        if scores is not None:
            output['scores'] = list(
                map(scores, output[self.table_or_collection.primary_id].tolist())
            )
        output = output.to_dict(orient='records')
        return SuperDuperCursor(output, id_field=self.table_or_collection.primary_id)


class _LogicalExprMixin:
    def _logical_expr(self, members, collection, k, other: t.Optional[t.Any] = None):
        if other:
            args = [other]
        else:
            args = []
        members.append(
            IbisQueryComponent(k, args=args, kwargs={}, type=QueryType.QUERY)
        )
        return IbisQueryLinker(collection, members=members)

    def eq(self, other, members, collection):
        k = '__eq__'
        return self._logical_expr(members, collection, k, other=other)

    def or_(self, other, members, collection):
        k = '__or__'
        return self._logical_expr(members, collection, k, other=other)

    def not_(self, members, collection):
        k = '__not__'
        return self._logical_expr(members, collection, k)

    def and_(self, other, members, collection):
        k = '__and__'
        return self._logical_expr(members, collection, k, other=other)

    def gt(self, other, members, collection):
        k = '__gt__'
        return self._logical_expr(members, collection, k, other=other)

    def lt(self, other, members, collection):
        k = '__lt__'
        return self._logical_expr(other, members, collection, k, other=other)

    def getitem(self, other, members, collection):
        k = '__getitem__'
        return self._logical_expr(members, collection, k, other=other)


@dc.dataclass(repr=False)
class IbisQueryLinker(QueryLinker, _LogicalExprMixin):
    def repr_(self) -> str:
        out = super().repr_()
        out = re.sub('\. ', ' ', out)
        return out

    def __eq__(self, other):
        return self.eq(other, members=self.members, collection=self.table_or_collection)

    def __lt__(self, other):
        return self.lt(other, members=self.members, collection=self.table_or_collection)

    def __gt__(self, other):
        return self.gt(other, members=self.members, collection=self.table_or_collection)

    def __or__(self, other):
        return self.or_(
            other, members=self.members, collection=self.table_or_collection
        )

    def __and__(self, other):
        return self.and_(
            other, members=self.members, collection=self.table_or_collection
        )

    def __not__(self, other):
        return self.not_(
            other, members=self.members, collection=self.table_or_collection
        )

    def __getitem__(self, other):
        return self.getitem(
            other, members=self.members, collection=self.table_or_collection
        )

    def _get_query_component(self, k):
        return IbisQueryComponent(name=k, type=QueryType.ATTR)

    @property
    def select_ids(self):
        return self.select(self.table_or_collection.primary_id)

    def select_single_id(self, id):
        return self.filter(
            self.table_or_collection.__getattr__(self.table_or_collection.primary_id)
            == id
        )

    def select_using_ids(self, ids):
        return self.filter(
            self.table_or_collection.__getattr__(
                self.table_or_collection.primary_id
            ).isin(ids)
        )

    def select_ids_of_missing_outputs(self, key: str, model: str):
        raise NotImplementedError

    def get_all_tables(self):
        out = []
        for member in self.members:
            out.extend(member.get_all_tables())
        return list(set(out))

    def outputs(self, model: str):
        symbol_table = IbisQueryTable(
            identifier=f'_outputs/{model}',
            primary_id='id',
        )
        attr = getattr(  # type: ignore[call-overload]
            self, self.table_or_collection.primary_id
        )
        other_query = self.join(
            symbol_table,
            [
                symbol_table.input_id == attr,
                symbol_table.query_id == self.table_or_collection.identifier,
            ],
        )
        return other_query

    def compile(self, db: DB, tables: t.Optional[t.Dict] = None):
        table_id = self.table_or_collection.identifier
        if tables is None:
            tables = {}
        if table_id not in tables:
            tables = {table_id: db.databackend.conn.table(table_id)}
        result = tables[table_id]
        for member in self.members:
            result, tables = member.compile(parent=result, db=db, tables=tables)
        return result, tables

    def execute(self, db):
        native_query, _ = self.compile(db)
        result = native_query.execute()
        return result


class QueryType(str, enum.Enum):
    QUERY = 'query'
    ATTR = 'attr'


@dc.dataclass(repr=False)
class IbisTable(Component):
    """
    This is a representation of an SQL table in ibis,
    saving the important meta-data associated with the table
    in the ``superduperdb`` meta-data store.

    :param identifier: The name of the table
    :param schema: The schema of the table
    :param primary_id: The primary id of the table
    :param version: The version of the table
    """

    identifier: str
    schema: Schema
    primary_id: str = 'id'
    version: t.Optional[int] = None
    type_id: t.ClassVar[str] = 'table'

    def on_create(self, db: DB):
        assert self.schema is not None, "Schema must be set"
        for e in self.schema.encoders:
            db.add(e)
        if db.databackend.in_memory:
            logging.info(f'Using in-memory tables "{self.identifier}" so doing nothing')
            return

        try:
            db.databackend.create_ibis_table(  # type: ignore[attr-defined]
                self.identifier, schema=ibis.schema(self.schema.raw)
            )
        except Exception as e:
            if 'already exists' in str(e):
                pass
            else:
                raise e

    def insert(self, documents, **kwargs):
        return IbisQueryTable(
            identifier=self.identifier, primary_id=self.primary_id
        ).insert(documents, **kwargs)

    def __getattr__(self, item):
        return IbisQueryTable(
            identifier=self.identifier, primary_id=self.primary_id
        ).__getattr__(item)

    def __getitem__(self, item):
        return IbisQueryTable(
            identifier=self.identifier, primary_id=self.primary_id
        ).__getitem__(item)


@dc.dataclass(repr=False)
class IbisQueryTable(_ReprMixin, TableOrCollection, Select):
    """
    This is a symbolic representation of a table
    for building ``IbisCompoundSelect`` queries.

    :param identifier: The name of the table
    :param primary_id: The primary id of the table
    :param version: The version of the table
    """

    primary_id: str = 'id'

    def compile(self, db: DB, tables: t.Optional[t.Dict] = None):
        if tables is None:
            tables = {}
        if self.identifier not in tables:
            tables[self.identifier] = db.databackend.conn.table(self.identifier)
        return tables[self.identifier], tables

    def repr_(self):
        return self.identifier

    def add_fold(self, fold: str) -> Select:
        return self.filter(self.fold == fold)

    @property
    def id_field(self):
        return self.primary_id

    @property
    def select_table(self) -> Select:
        return self

    @property
    def select_ids(self) -> Select:
        return self.select(self.primary_id)

    def select_using_ids(self, ids: t.Sequence[t.Any]) -> Select:
        return self.filter(self[self.primary_id].isin(ids))

    def select_ids_of_missing_outputs(self, key: str, model: str) -> Select:
        raise NotImplementedError

    def select_single_id(self, id):
        return self.filter(getattr(self, self.primary_id) == id)

    def __getitem__(self, item):
        return IbisCompoundSelect(
            table_or_collection=self,
            query_linker=self._get_query_linker(
                members=[IbisQueryComponent('__getitem__', type=QueryType.ATTR)]
            ),
        )(item)

    def _insert(self, documents, **kwargs):
        return IbisInsert(documents=documents, kwargs=kwargs, table_or_collection=self)

    def _get_query(
        self,
        pre_like: t.Optional[Like] = None,
        query_linker: t.Optional[QueryLinker] = None,
        post_like: t.Optional[Like] = None,
    ) -> IbisCompoundSelect:
        return IbisCompoundSelect(
            pre_like=pre_like,
            query_linker=query_linker,
            post_like=post_like,
            table_or_collection=self,
        )

    def _get_query_component(
        self,
        k,
    ):
        return IbisQueryComponent(name=k, type=QueryType.ATTR)

    def _get_query_linker(self, members) -> IbisQueryLinker:
        return IbisQueryLinker(table_or_collection=self, members=members)

    def model_update(  # type: ignore[override]
        self,
        db,
        ids: t.Sequence[t.Any],
        key: str,
        model: str,
        outputs: t.Sequence[t.Any],
        flatten: bool = False,
    ):
        if flatten:
            raise NotImplementedError('Flatten not yet supported for ibis')

        if key.startswith('_outputs'):
            key = key.split('.')[1]

        if not outputs:
            return

        # TODO generalize this to multiple queries per table/ model
        table_records = []
        for ix in range(len(outputs)):
            d = {
                'id': str(ids[ix]),
                'input_id': str(ids[ix]),
                'query_id': self.identifier,
                'output': outputs[ix],
                'key': key,
            }
            table_records.append(d)

        for r in table_records:
            if isinstance(r['output'], dict) and '_content' in r['output']:
                r['output'] = r['output']['_content']['bytes']

        db.databackend.insert(f'_outputs/{model}', table_records)

    def insert(
        self,
        *args,
        **kwargs,
    ):
        return self._insert(*args, **kwargs)

    def _delete(self, *args, **kwargs):
        return super()._delete(*args, **kwargs)

    def execute(self, db):
        return db.databackend.conn.table(self.identifier).execute()


def _compile_item(item, db, tables):
    if isinstance(item, (IbisQueryLinker, IbisCompoundSelect, IbisQueryTable)):
        return item.compile(db, tables=tables)
    if isinstance(item, list) or isinstance(item, tuple):
        compiled = []
        for x in item:
            c, tables = _compile_item(x, db, tables=tables)
            compiled.append(c)
        return compiled, tables
    elif isinstance(item, dict):
        c = {}
        for k, v in item.items():
            c[k], tables = _compile_item(v, db, tables=tables)
        return c, tables
    return item, tables


def _get_all_tables(item):
    if isinstance(item, (IbisQueryLinker, IbisCompoundSelect)):
        return item.get_all_tables()
    elif isinstance(item, IbisQueryTable):
        return [item.identifier]
    elif isinstance(item, list) or isinstance(item, tuple):
        return sum([_get_all_tables(x) for x in item], [])
    elif isinstance(item, dict):
        return sum([_get_all_tables(x) for x in item.values()], [])
    else:
        return []


@dc.dataclass
class IbisQueryComponent(QueryComponent):
    """
    This class represents a component of an ``ibis`` query.
    For example ``filter`` in ``t.filter(t.age > 25)``.
    """

    __doc__ = __doc__ + QueryComponent.__doc__  # type: ignore[operator]

    def repr_(self) -> str:
        out = super().repr_()
        match = re.match('.*__([a-z]{2,2})__\(([a-z0-9_\.\']+)\)', out)
        lookup = {'gt': '>', 'lt': '<', 'eq': '=='}
        if match is not None and match.groups()[0] in lookup:
            out = f' {lookup[match.groups()[0]]} {match.groups()[1]}'
        return out

    def compile(self, parent: t.Any, db: DB, tables: t.Optional[t.Dict] = None):
        if self.type == QueryType.ATTR:
            return getattr(parent, self.name), tables
        args, tables = _compile_item(self.args, db, tables=tables)
        kwargs, tables = _compile_item(self.kwargs, db, tables=tables)
        return getattr(parent, self.name)(*args, **kwargs), tables

    def get_all_tables(self):
        out = []
        out.extend(_get_all_tables(self.args))
        out.extend(_get_all_tables(self.kwargs))
        return list(set(out))


@dc.dataclass
class IbisInsert(Insert):
    def _encode_documents(self, table: IbisTable) -> t.List[t.Dict]:
        return [r.encode(table.schema) for r in self.documents]

    def execute(self, db):
        table = db.load(
            'table',
            self.table_or_collection.identifier,
        )
        encoded_documents = self._encode_documents(table=table)
        ids = [r[table.primary_id] for r in encoded_documents]
        db.databackend.insert(
            self.table_or_collection.identifier, raw_documents=encoded_documents
        )
        return ids

    @property
    def select_table(self):
        return self.table_or_collection
