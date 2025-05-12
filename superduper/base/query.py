"""
Permitted patterns.

type_1: table.like()[.filter(...)][.select(...)][.get() | .limit(...)]'
type_2: table[.filter(...)][.select(...)][.like()][.get() | .limit(...)]'

Select always comes last, unless with `.get`, `.limit`.

"""

import dataclasses as dc
import functools
import json
import re
import typing as t
import uuid
from types import MethodType

from superduper import CFG, logging
from superduper.base import exceptions
from superduper.base.base import Base
from superduper.base.constant import KEY_BLOBS, KEY_BUILDS, KEY_FILES, KEY_PATH
from superduper.base.datatype import BaseDataType
from superduper.base.document import Document, _unpack

if t.TYPE_CHECKING:
    from superduper.base.datalayer import Datalayer


@dc.dataclass
class QueryPart:
    """A method part of a query.

    :param name: The name of the method.
    :param args: The arguments of the method.
    :param kwargs: The keyword arguments of the method.
    """

    name: str
    args: t.Sequence
    kwargs: t.Dict


@dc.dataclass
class Op(QueryPart):
    """An operation part of a query.

    :param name: The name of the method.
    :param args: The arguments of the method.
    :param kwargs: The keyword arguments of the method.
    :param symbol: The symbol of the operation.
    """

    symbol: str


@dc.dataclass
class Decomposition:
    """
    Decompose a query into its parts.

    :param table: The table to use.
    :param db: The datalayer to use.
    :param col: The column to use.
    :param insert: The insert part of the query.
    :param pre_like: The pre-like part of the query.
    :param post_like: The post-like part of the query.
    :param filter: The filter part of the query.
    :param select: The select part of the query.
    :param get: The get part of the query.
    :param limit: The limit part of the query.
    :param outputs: The outputs part of the query.
    :param op: The operation part of the query.
    """

    table: str
    db: 'Datalayer'
    col: str | None = None
    insert: QueryPart | None = None
    pre_like: QueryPart | None = None
    post_like: QueryPart | None = None
    filter: QueryPart | None = None
    select: QueryPart | None = None
    get: QueryPart | None = None
    limit: QueryPart | None = None
    outputs: QueryPart | None = None
    op: Op | None = None

    @property
    def predict_ids(self):
        if self.outputs:
            return self.outputs.args
        return []

    def to_query(self):
        """Convert decomposition back to a ``Query``."""
        if self.db is None:
            self.db = db

        q = self.db[self.table]

        if self.pre_like:
            q = q + self.pre_like

        if self.filter:
            q = q + self.filter

        if self.outputs:
            q = q + self.outputs

        if self.select:
            q = q + self.select

        if self.post_like:
            q = q + self.post_like

        if self.get:
            assert not self.limit
            q = q + self.get

        if self.limit:
            q = q + self.limit

        return q

    def copy(self):
        """Copy the decomposition."""
        return self.to_query().copy().decomposition


def _stringify(item, documents, queries):
    if isinstance(item, dict):
        documents.append(item)
        out = f'documents[{len(documents) - 1}]'
    elif isinstance(item, list):
        old_len = len(documents)
        documents.extend(item)
        out = f'documents[{old_len}:{len(documents)}]'
    elif isinstance(item, Query):
        out = f'query[{len(queries)}]'
        queries.append(item.stringify(documents, queries))
    elif isinstance(item, Op):
        arg = _stringify(item.args[0], documents, queries)
        return f' {item.symbol} {arg}'
    elif isinstance(item, QueryPart):
        args = [_stringify(a, documents, queries) for a in item.args]
        kwargs = {k: _stringify(v, documents, queries) for k, v in item.kwargs.items()}
        parameters = ''
        if args and kwargs:
            parameters = (
                ', '.join(args)
                + ', '
                + ', '.join([f'{k}={v}' for k, v in kwargs.items()])
            )
        elif args:
            parameters = ', '.join(args)
        elif kwargs:
            parameters = ', '.join([f'{k}={v}' for k, v in kwargs.items()])
        return f'.{item.name}({parameters})'
    else:
        try:
            out = json.dumps(item)
        except Exception:
            documents.append(item)
            out = f'documents[{len(documents) - 1}]'
    return out


# TODO add to regular Query class
class _BaseQuery(Base):
    parts: t.Sequence[t.Union[QueryPart, str]] = dc.field(default_factory=list)

    def __post_init__(self, db: t.Optional['Datalayer'] = None):
        super().__post_init__(db)
        if not self.identifier:
            self.identifier = self._build_hr_identifier()
        self.identifier = re.sub('[^a-zA-Z0-9\-]', '-', self.identifier)
        self.identifier = re.sub('[\-]+', '-', self.identifier)

    def unpack(self):
        """Unpack the query."""
        parts = _unpack(self.parts)
        return _from_parts(
            impl=self.__class__, table=self.table, parts=parts, db=self.db
        )

    def _build_hr_identifier(self):
        identifier = str(self).split('\n')[-1]
        variables = re.findall(r'(<var:[a-zA-Z0-9]+>)', identifier)
        variables = sorted(list(set(variables)))
        for i, v in enumerate(variables):
            identifier = identifier.replace(v, f'#{i}')
        identifier = re.sub(r'[^a-zA-Z0-9#]', '-', identifier)
        identifier = re.sub('[-]+$', '', identifier)
        identifier = re.sub('[-]+', '-', identifier)
        identifier = re.sub('^[-]+', '', identifier)
        for i, v in enumerate(variables):
            identifier = identifier.replace(f'#{i}', v)
        return identifier

    def _to_str(self):
        documents = []
        queries = {}
        out = str(self.table)
        for part in self.parts:
            if isinstance(part, str):
                if isinstance(getattr(self.__class__, part, None), property):
                    out += f'.{part}'
                    continue
                else:
                    out += f'["{part}"]'
                    continue
            args = []
            for a in part.args:
                args.append(self._update_item(a, documents, queries))
            args = ', '.join(args)
            kwargs = {}
            for k, v in part.kwargs.items():
                kwargs[k] = self._update_item(v, documents, queries)
            kwargs = ', '.join([f'{k}={v}' for k, v in kwargs.items()])
            if part.args and part.kwargs:
                out += f'.{part.name}({args}, {kwargs})'
            if not part.args and part.kwargs:
                out += f'.{part.name}({kwargs})'
            if part.args and not part.kwargs:
                out += f'.{part.name}({args})'
            if not part.args and not part.kwargs:
                out += f'.{part.name}()'
        return out, documents, queries

    def _dump_query(self):
        output, documents, queries = self._to_str()
        if queries:
            output = '\n'.join(list(queries.values())) + '\n' + output
        for i, k in enumerate(queries):
            output = output.replace(k, str(i))
        return output, documents

    @staticmethod
    def _update_item(a, documents, queries):
        if isinstance(a, Query):
            a, sub_documents, sub_queries = a._to_str()
            if documents:
                for i in range(len(sub_documents)):
                    a = a.replace(f'documents[{i}]', f'documents[{i + len(documents)}]')
            documents.extend(sub_documents)
            queries.update(sub_queries)
            id_ = uuid.uuid4().hex[:5].upper()
            queries[id_] = a
            arg = f'query[{id_}]'
        else:
            if isinstance(a, dict):
                documents.append(a)
                arg = f'documents[{len(documents) - 1}]'
            elif isinstance(a, list):
                old_len = len(documents)
                documents.extend(a)
                arg = f'documents[{old_len}:{len(documents)}]'
            else:
                try:
                    arg = json.dumps(a)
                except Exception:
                    documents.append(a)
                    arg = f'documents[{len(documents) - 1}]'
        return arg


def bind(f):
    """Bind a method to a query object.

    :param f: The method to bind.
    """

    @functools.wraps(f)
    def decorated(self, *args, **kwargs):
        out = f(self, *args, **kwargs)
        children = self.mapping[f.__name__]
        for method in children:
            out._bind_base_method(method, eval(method))
        return out

    decorated.__name__ = f.__name__
    return decorated


@bind
def limit(self, n: int):
    """Limit the number of results returned by the query.

    # noqa

    :param n: The number of results to return.
    """
    # always the last one
    assert not self.decomposition.limit
    assert not self.decomposition.get
    return self + QueryPart('limit', (n,), {})


def insert(self: 'Query', documents, raw: bool = False):
    """Insert documents into the table.

    # noqa
    """
    # FIXME: Access to a protected member _do_insert of a class
    out = self.db.databackend._do_insert(self.table, documents, raw=raw)
    self.db._post_query(self.table, ids=out, type_='insert')
    return out


def update(self, condition: t.Dict, key: str, value: t.Any):
    """Update documents in the table.

    # noqa
    """
    s = self.db.metadata.get_schema(self.table)
    if isinstance(s[key], BaseDataType):
        value = s[key].encode_data(value, None)
        if s[key].dtype == 'json' and not CFG.json_native:
            value = json.dumps(value)
    out = self.db.databackend.update(self.table, condition, key=key, value=value)

    # FIXME: Access to a protected member _post_query of a class
    self.db._post_query(self.table, ids=out, type_='update')
    return out


def delete(self, condition: t.Dict):
    """Update documents in the table.

    # noqa
    """
    out = self.db.databackend.delete(self.table, condition)
    self.db._post_query(self.table, ids=out, type_='delete')
    return out


def replace(self, condition: t.Dict, r: t.Dict | Document):
    """Update documents in the table.

    # noqa
    """
    try:
        r.pop(KEY_BUILDS)
    except KeyError:
        pass
    try:
        r.pop(KEY_BLOBS)
    except KeyError:
        pass
    try:
        r.pop(KEY_FILES)
    except KeyError:
        pass
    try:
        r.pop(KEY_PATH)
    except KeyError:
        pass

    if isinstance(r, Document):
        s = self.db.metadata.get_schema(self.table)
        r = s.encode_data(r)

    out = self.db.databackend.replace(self.table, condition, r)
    self.db._post_query(self.table, ids=out, type_='update')
    return out


@bind
def outputs(self, *predict_ids):
    """Add outputs to the query.

    # noqa

    :param predict_ids: The predict_ids to add. # noqa
    """
    d: Decomposition = self.decomposition

    assert not d.outputs

    d.outputs = QueryPart('outputs', predict_ids, {})

    return d.to_query()


def get(self, raw: bool = False, **kwargs):
    """Get a single row of data.

    # noqa
    """
    query = self
    if kwargs:
        filters = []
        t = self.db[self.table]
        for k, v in kwargs.items():
            filters.append(t[k] == v)
        query = self.filter(*filters)

    result = query.db.databackend.get(query, raw=raw)
    return result


def ids(self):
    """Get the primary ids of the query.

    # noqa
    """
    msg = '.ids only applicable to select queries'
    assert self.type == 'select', msg
    q = self.select(self.primary_id)
    pid = self.primary_id.execute()
    results = q.execute()
    return [str(r[pid]) for r in results]


def distinct(self, key: str):
    """Get distinct values of a column.

    # noqa
    """
    q = self.select(key)
    msg = '.distinct only applicable to select queries'
    assert self.type == 'select', msg
    q = self.select(key)
    results = q.execute()
    return list(set([r[key] for r in results]))


# TODO use this/ test this
def missing_outputs(self, predict_id):
    """Get missing outputs for a given predict_id.

    # noqa

    :param predict_id: The predict_id to check.
    """
    return self.db.databackend.missing_outputs(self, predict_id)


# TODO use this in the code to split jobs in parts
def chunks(self, n: int):
    """Split a query into chunks of size n.

    # noqa

    :param n: The size of the chunks.
    """
    assert self.type == 'select'
    t = self.db[self.table]
    ids = self.select(t.primary_id).execute()
    for i in range(0, len(ids), n):
        yield self.subset(ids[i : i + n])


@bind
def select(self, *cols):
    """Create a select query selecting certain fields/ cols.

    # noqa

    :param cols: The columns to select.

    >>> from superduper import superduper
    >>> db = superduper()
    >>> db['table'].insert({'col': 1, 'other': 2})
    >>> results = db['table'].select('col').execute()
    >>> list(results[0].keys())
    ['col']
    """
    d = self.decomposition

    if d.select:
        d.select = QueryPart(
            'select',
            (*d.select.args, *cols),
            {},
        )
    else:
        d.select = QueryPart('select', cols, {})

    return d.to_query()


@bind
def filter(self, *filters):
    """Create a filter query.

    # noqa

    :param filters: The filters to apply.

    >>> from superduper import superduper
    >>> db = superduper()
    >>> t = db['table']
    >>> t.insert({'col': 1})
    >>> results = t.filter(t['col'] == 1, t['col'] > 0).execute()
    >>> len(results)
    1
    """
    d = self.decomposition

    if d.filter:
        d.filter = QueryPart('filter', args=(*d.filter.args, *filters), kwargs={})
    else:
        d.filter = QueryPart('filter', args=filters, kwargs={})

    return d.to_query()


@bind
def like(self, r: t.Dict, vector_index: str, n: int = 10):
    """Create a similarity query with a vector_index.

    # noqa

    :param r: The vector to compare against.
    :param vector_index: The index of the vector.
    :param n: The number of results to return.
    """
    return self + QueryPart('like', args=(r, vector_index), kwargs={'n': n})


SYMBOLS = {
    '__eq__': '==',
    '__ne__': '!=',
    '__le__': '<=',
    '__ge__': '>=',
    '__lt__': '<',
    '__gt__': '>',
    'isin': 'in',
}


class Query(_BaseQuery):
    """A query object.

    This base class is used to create a query object that can be executed
    in the datalayer.

    :param table: The table to use.
    :param parts: The parts of the query.
    :param db: The `Datalayer` instance to use.
    """

    # mapping between methods and allowed downstream methods
    # base methods are at the key level
    mapping: t.ClassVar[t.Dict] = {
        'insert': [],
        'update': [],
        'delete': [],
        'replace': [],
        'missing_outputs': [],
        'select': [
            'filter',
            'outputs',
            'like',
            'limit',
            'select',
            'ids',
            'distinct',
            'missing_outputs',
            'chunks',
            'get',
        ],
        'filter': [
            'filter',
            'outputs',
            'like',
            'limit',
            'select',
            'ids',
            'distinct',
            'missing_outputs',
            'chunks',
            'get',
        ],
        'like': [
            'select',
            'filter',
            'ids',
            'distinct',
            'missing_outputs',
            'get',
            'limit',
        ],
        'outputs': [
            'filter',
            'limit',
            'ids',
            'distinct',
            'missing_outputs',
            'chunks',
            'get',
            'select',
        ],
        'limit': [],
        'ids': [],
        'distinct': [],
        'get': [],
    }

    flavours: t.ClassVar[t.Dict[str, str]] = {}
    table: str

    db: dc.InitVar[t.Optional['Datalayer']] = None

    @classmethod
    def _alternative_init(cls, documents, query, db):
        return parse_query(query, documents, db)

    def __post_init__(self, db=None):

        self.db: t.Union['Datalayer', None] = db

        if not self.parts:
            for method in self.mapping:
                self._bind_base_method(method, eval(method))
        elif self.parts:
            if isinstance(self.parts[-1], str):
                name = self.parts[-1]
                self._bind_base_method('filter', filter)
            else:
                name = self.parts[-1].name

                try:
                    for method in self.mapping[name]:
                        self._bind_base_method(method, eval(method))
                except KeyError:
                    pass

        if self.type == 'insert':
            self._add_fold_to_insert()

    def _add_fold_to_insert(self):
        assert self.type == 'insert'
        documents = self[-1].args[0]
        import random

        for r in documents:
            r.setdefault(
                '_fold',
                'train' if random.random() >= CFG.fold_probability else 'valid',
            )

    @property
    def decomposition(self):
        out = Decomposition(table=self.table, db=self.db)

        for i, part in enumerate(self.parts):
            if isinstance(part, str):
                out.col = part
                continue

            if i == 0 and part.name == 'like':
                out.pre_like = part
                continue

            if part.name == 'like':
                out.post_like = part
                continue

            if isinstance(part, Op):
                out.op = Op
                continue

            msg = f'Found unexpected query part "{part.name}"'
            assert part.name in [f.name for f in dc.fields(out)], msg
            setattr(out, part.name, part)

        return out

    def _bind_base_method(self, name, method):
        method = MethodType(method, self)
        setattr(self, name, method)

    def stringify(self, documents, queries):
        """Stringify the query.

        :param documents: The documents to stringify.
        :param queries: The queries to stringify.
        """
        parts = []
        for part in self.parts:
            if isinstance(part, str):
                if part == 'primary_id':
                    parts.append('.primary_id')
                else:
                    parts.append(f'["{part}"]')
                continue
            parts.append(_stringify(part, documents, queries))
        parts = ''.join(parts)
        return f'{self.table}{parts}'

    @property
    def type(self):
        if self.parts and isinstance(self[-1], QueryPart) and self[-1].name == 'insert':
            return 'insert'
        if 'delete' in str(self):
            return 'delete'
        return 'select'

    @property
    def tables(self):
        """Tables contained in the ``Query`` object."""
        out = []
        for part in self.parts:
            if part.name == 'outputs':
                out.extend([f'{CFG.output_prefix}{x}' for x in part.args])
        out.append(self.table)
        return list(set(out))

    def __len__(self):
        return len(self.parts) + 1

    def __getitem__(self, item):
        # supports queries which use strings to index
        if isinstance(item, str):
            return self + item

        if isinstance(item, int):
            return self.parts[item]

        if not isinstance(item, slice):
            raise TypeError('Query index must be a string or a slice')

        assert isinstance(item, slice)

        parts = self.parts[item]

        return self.__class__(db=self.db, table=self.table, parts=parts)

    def copy(self):
        """Copy the query."""
        r = self.dict()
        del r['_path']
        return parse_query(**r, db=self.db)

    def dict(self, *args, **kwargs):
        """Return the query as a dictionary.

        :param args: Additional arguments.
        :param kwargs: Additional keyword arguments.
        """
        documents = []
        queries = []
        _stringify(self, documents=documents, queries=queries)
        query = '\n'.join(queries)
        return Document(
            {
                '_path': 'superduper.base.query.parse_query',
                'documents': documents,
                'query': query,
            }
        )

    def __repr__(self):
        r = self.dict()
        query = r['query'].split('\n')[-1]
        queries = r['query'].split('\n')[:-1]
        for i, q in enumerate(queries):
            query = query.replace(f'query[{i}]', q)

        doc_refs = re.findall('documents\[([0-9]+)\]', query)
        if doc_refs:
            for numeral in doc_refs:
                query = query.replace(
                    f'documents[{numeral}]', str(r['documents'][int(numeral)])
                )

        doc_segs = re.findall('documents\[([0-9]+):([0-9]+)\]', query)
        if doc_segs:
            for n1, n2 in doc_segs:
                query = query.replace(
                    f'documents[{n1}:{n2}]', str(r['documents'][int(n1) : int(n2)])
                )

        return query

    def __add__(self, other: QueryPart | str):
        return Query(
            table=self.table,
            parts=[*self.parts, other],
            db=self.db,
        )

    def _ops(self, op, other):
        msg = 'Can only compare based on a column'
        assert isinstance(self.parts[-1], str), msg
        return self + Op(op, args=(other,), kwargs={}, symbol=SYMBOLS[op])

    def __eq__(self, other):
        return self._ops('__eq__', other)

    def __ne__(self, other):
        return self._ops('__ne__', other)

    def __lt__(self, other):
        return self._ops('__lt__', other)

    def __gt__(self, other):
        return self._ops('__gt__', other)

    def __le__(self, other):
        return self._ops('__le__', other)

    def __ge__(self, other):
        return self._ops('__ge__', other)

    def isin(self, other):
        """Create an isin query.

        :param other: The value to check against.
        """
        return self._ops('isin', other)

    def _encode_or_unpack_args(self, r, db, method='encode', parent=None):
        if isinstance(r, Document):
            out = getattr(r, method)()
            try:
                out.pop('_builds')
                out.pop('_files')
                out.pop('_blobs')
            except KeyError:
                pass

            if '_base' in out:
                return out['_base']
            return out

        if isinstance(r, (tuple, list)):
            out = [
                self._encode_or_unpack_args(x, db, method=method, parent=parent)
                for x in r
            ]
            if isinstance(r, tuple):
                return tuple(out)
            return out
        if isinstance(r, dict):
            return {
                k: self._encode_or_unpack_args(v, db, method=method, parent=parent)
                for k, v in r.items()
            }
        if isinstance(r, Query):
            r.db = db
            parent = r._get_parent()
            return super(type(self), r)._execute(parent, method=method)

        return r

    def complete_uuids(
        self, db: 'Datalayer', listener_uuids: t.Optional[t.Dict] = None
    ) -> 'Query':
        """Complete the UUIDs which have been omitted from output-tables.

        :param db: ``db`` instance.
        :param listener_uuids: identifier to UUIDs of listeners lookup
        """
        listener_uuids = listener_uuids or {}
        import copy

        r = copy.deepcopy(self.dict())
        lines = r['query'].split('\n')
        parser = parse_query

        def _get_uuid(identifier):
            if '.' in identifier:
                identifier = identifier.split('.')[0]
            msg = (
                'Couldn\'t complete `Listener` key '
                'based on ellipsis {predict_id}__????????????????. '
                'Please specify using upstream_listener.outputs'
            )
            try:
                return listener_uuids[identifier]
            except KeyError as e:
                logging.warn(f'Error in completing UUIDs from cache: {e}')
                pass

            try:
                return db.show('Listener', identifier, -1)['uuid']
            except FileNotFoundError as e:
                logging.warn(
                    f'Error in completing UUIDs from saved components,'
                    f' based on `Listener={identifier}`: {e}'
                )
                pass

            raise Exception(msg.format(predict_id=identifier))

        for i, line in enumerate(lines):
            output_query_groups = re.findall('\.outputs\((.*?)\)', line)

            for group in output_query_groups:
                predict_ids = [eval(x.strip()) for x in group.split(',')]
                replace_ids = []
                for predict_id in predict_ids:
                    if re.match(r'^.*__([0-9a-z]{8,})$', predict_id):
                        replace_ids.append(f'"{predict_id}"')
                        continue
                    listener_uuid = _get_uuid(predict_id)
                    replace_ids.append(f'"{predict_id}__{listener_uuid}"')
                new_group = ', '.join(replace_ids)
                lines[i] = lines[i].replace(group, new_group)

            output_table_groups = re.findall(f'^{CFG.output_prefix}.*?\.', line)

            for group in output_table_groups:
                if re.match(f'^{CFG.output_prefix}[^\.]+__([0-9a-z]{{8,}})\.$', group):
                    continue
                identifier = group[len(CFG.output_prefix) : -1]
                listener_uuid = _get_uuid(identifier)
                new_group = f'{CFG.output_prefix}{identifier}__{listener_uuid}.'
                lines[i] = lines[i].replace(group, new_group)

        def swap_keys(r: str | list | dict):
            if isinstance(r, str):
                if (
                    r.startswith(CFG.output_prefix)
                    and '__' not in r[len(CFG.output_prefix) :]
                ):
                    parts = [r]
                    if '.' in r:
                        parts = list(r.split('.'))
                    parts[0] += '__' + _get_uuid(r[len(CFG.output_prefix) :])
                    return '.'.join(parts)
                return r
            if isinstance(r, list):
                return [swap_keys(x) for x in r]
            if isinstance(r, dict):
                return {swap_keys(k): swap_keys(v) for k, v in r.items()}
            return r

        r['query'] = '\n'.join(lines)
        r['documents'] = swap_keys(r['documents'])

        del r['_path']
        out = parser(**r, db=db)
        return out

    @property
    def primary_id(self):
        return Query(table=self.table, parts=(), db=self.db) + 'primary_id'

    @property
    def documents(self):
        return self.dict()['documents']

    def subset(self, ids: t.Sequence[str]):
        """Subset the query based on primary ids.

        :param ids: The primary ids to subset on.
        """
        assert self.type == 'select'

        # mypy nonsense
        from superduper.base.datalayer import Datalayer

        assert isinstance(self.db, Datalayer)

        t = self.db[self.table]
        modified_query = self.filter(t.primary_id.isin(ids))

        return modified_query.execute()

    def execute(self, raw: bool = False):
        """Execute the query.

        :param raw: Whether to return raw results.
        """
        db = self.db
        if self.table in db.metadata.db.databackend.list_tables():
            db = db.metadata.db
        if self.parts and self.parts[0] == 'primary_id':
            return db.databackend.primary_id(self.table)
        results = db.databackend.execute(self, raw=raw)
        return results


def _parse_op_part(table, col, symbol, operand, db, documents=()):
    operand = eval(operand, {'documents': documents})

    reverse = dict(zip(SYMBOLS.values(), SYMBOLS.keys()))

    if col != 'primary_id':
        out = getattr(db[table][col], reverse[symbol])(operand)
    else:
        out = getattr(db[table].primary_id, reverse[symbol])(operand)

    return out


def _parse_query_part(part, documents, query, db):
    pattern = (
        '^([a-zA-Z0-9_]+)\["([a-zA-Z0-9_]+)"\][ ]{0,}'
        '([!=><]=|==|!=|<=|>=|<|>|in)[ ]{0,}(.*)[ ]{0,}$'
    )

    if match := re.match(pattern, part):
        return _parse_op_part(*match.groups(), db, documents=documents)

    pattern = (
        '^([a-zA-Z0-9_]+)\.primary_id[ ]{0,}'
        '([!=><]=|==|!=|<=|>=|<|>|in)[ ]{0,}(.*)[ ]{0,}$'
    )

    if match := re.match(pattern, part):
        return _parse_op_part(
            match.groups()[0],
            'primary_id',
            *match.groups()[1:],
            db,
            documents=documents,
        )

    table = part.split('.', 1)[0]

    rest_part = part[len(table) + 1 :]

    col_match = re.match('^([a-zA-Z0-9]+)\["[a-zA-Z0-9]+"\]$', table)
    if col_match:
        table = col_match.groups()[0]

    parts = re.findall(r'\.([a-zA-Z0-9_]+)(\(.*?\))?', "." + rest_part)

    # TODO what's this clause?
    recheck_part = ".".join(p[0] + p[1] for p in parts)
    if recheck_part != rest_part:
        raise ValueError(f'Invalid query part: {part} != {recheck_part}')

    new_parts = []
    for part in parts:
        if (
            isinstance(part, str)
            and re.match('^[a-zA-Z0-9]+\["[a-zA-Z0-9]+"\]$', part) is not None
        ):
            new_parts.extend(part.split('[')[0], part.split(']').strip()[:-1])
            continue
        new_parts.append(part)

    current = Query(table=table, parts=(), db=db)

    for part in parts:
        comp = part[0] + part[1]
        match = re.match(r'^([a-zA-Z0-9_]+)\((.*)\)$', comp)

        if match is None:
            current = getattr(current, comp)
            continue

        if not match.groups()[1].strip():
            current = getattr(current, match.groups()[0])()
            continue

        comp = getattr(current, match.groups()[0])
        args_kwargs = [x.strip() for x in match.groups()[1].split(',')]
        args = []
        kwargs = {}
        for x in args_kwargs:
            if '=' in x:
                k, v = x.split('=')
                kwargs[k] = eval(v, {'documents': documents, 'query': query})
            else:
                args.append(eval(x, {'documents': documents, 'query': query}))
        current = comp(*args, **kwargs)

    return current


def _from_parts(impl, table, parts, db):
    current = impl(table=table, parts=(), db=db)
    for part in parts:
        if isinstance(part, str):
            try:
                current = getattr(current, part)
            except AttributeError:
                current = current[part]
            continue
        current = getattr(current, part.name)(*part.args, **part.kwargs)
    return current


def parse_query(
    query: t.Union[str, list],
    documents: t.Sequence[t.Any] = (),
    db: t.Optional['Datalayer'] = None,
):
    """Parse a string query into a query object.

    :param query: The query to parse.
    :param documents: The documents to query.
    :param db: The datalayer to use to execute the query.
    """
    if isinstance(query, str):
        query = [x.strip() for x in query.split('\n') if x.strip()]
    for i, q in enumerate(query):
        query[i] = _parse_query_part(q, documents, query[:i], db=db)

    return query[-1]


class _PlaceholderDB:
    def __getitem__(self, item):
        return Query(table=item, parts=(), db=self)


db = _PlaceholderDB()
