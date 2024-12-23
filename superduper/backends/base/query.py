"""
Permitted patterns

type_1: table.like()[.filter(...)][.select(...)][.get() | .limit(...)]'
type_2: table[.filter(...)][.select(...)][.like()][.get() | .limit(...)]'

Select always comes last, unless with `.get`, `.limit`.

"""
from abc import ABC, abstractmethod
import copy
import dataclasses as dc
import functools
import json
import re
import typing as t
from types import MethodType
import uuid

from superduper import CFG, logging
from superduper.base.document import Document, _unpack
from superduper.base.leaf import Leaf
from superduper.components.schema import Schema

if t.TYPE_CHECKING:
    from superduper.base.datalayer import Datalayer


@dc.dataclass
class QueryPart:
    name: str
    args: t.Sequence
    kwargs: t.Dict


@dc.dataclass
class Op(QueryPart):
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

    def to_query(self):

        q = self.db[self.table]

        if self.pre_like:
            q = q + self.pre_like

        if self.outputs:
            q = q + self.outputs

        if self.filter:
            q = q + self.filter

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
        return self.to_query().copy().decomposition


def stringify(item, documents, queries):
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
        arg = stringify(item.args[0], documents, queries)
        return f' {item.symbol} {arg}'
    elif isinstance(item, QueryPart):
        args = [stringify(a, documents, queries) for a in item.args]
        kwargs = {k: stringify(v, documents, queries) for k, v in item.kwargs.items()}
        parameters = ''
        if args and kwargs:
            parameters = ', '.join(args) + ', ' + ', '.join([f'{k}={v}' for k, v in kwargs.items()])
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


class _BaseQuery(Leaf):
    parts: t.Sequence[t.Union[QueryPart, str]] = dc.field(default_factory=list)

    def __post_init__(self, db: t.Optional['Datalayer'] = None):
        super().__post_init__(db)
        if not self.identifier:
            self.identifier = self._build_hr_identifier()
        self.identifier = re.sub('[^a-zA-Z0-9\-]', '-', self.identifier)
        self.identifier = re.sub('[\-]+', '-', self.identifier)

    def unpack(self):
        parts = _unpack(self.parts)
        return from_parts(impl=self.__class__, table=self.table, parts=parts, db=self.db)

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
    # always the last one
    assert not self.decomposition.limit
    assert not self.decomposition.get
    return self + QueryPart('limit', (n,), {})


@bind
def insert(self, documents):
    # no children
    return self + QueryPart('insert', (documents,), {})


@bind
def outputs(self, *predict_ids):

    d = self.decomposition

    assert not d.outputs

    d.outputs = QueryPart('outputs', predict_ids, {})

    return d.to_query()


def ids(self):
    msg = '.ids only applicable to select queries'
    assert self.type == 'select', msg
    q = self.select(self.primary_id)
    pid = self.primary_id.execute()
    results = q.execute()
    return [str(r[pid]) for r in results]


@bind
def select(self, *cols):

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

    d = self.decomposition

    if d.filter:
        d.filter = QueryPart(
            'filter',
            args=(*d.filter.args, *filters),
            kwargs={}
        )
    else:
        d.filter = QueryPart(
            'filter',
            args=filters,
            kwargs={}
        )

    return d.to_query()


@bind
def like(self, r: t.Dict, vector_index: str, n: int = 10):
    return self + QueryPart('like', args=(r, vector_index), kwargs={'n': n})


SYMBOLS = {
    '__eq__': '==',
    '__ne__': '!=',
    '__le__': '<=',
    '__ge__': '>=',
    '__lt__': '<',
    '__gt__': '>',
    'isin': 'in'
}



class Query(_BaseQuery):
    """A query object.

    This base class is used to create a query object that can be executed
    in the datalayer.

    :param table: The table to use.
    :param parts: The parts of the query.
    """

    # mapping between methods and allowed downstream methods
    # base methods are at the key level
    mapping: t.ClassVar[t.Dict] = {
        'insert': [],
        'select': ['filter', 'outputs', 'like', 'limit', 'select', 'ids'],
        'filter': ['filter', 'outputs', 'like', 'limit', 'select', 'ids'],
        'like': ['select', 'filter', 'ids'],
        'outputs': ['filter', 'limit', 'ids'],
        'limit': [],
        'ids': [],
    }

    flavours: t.ClassVar[t.Dict[str, str]] = {}
    table: str
    identifier: str = ''

    def __post_init__(self, db = None):
        out = super().__post_init__(db)
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

        return out

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
        parts = []
        for part in self.parts:
            if isinstance(part, str):
                if part == 'primary_id':
                    parts.append('.primary_id')
                else:
                    parts.append(f'["{part}"]')
                continue
            parts.append(stringify(part, documents, queries))
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
        r = self.dict()
        del r['_path']
        del r['identifier']
        return parse_query(**r, db=self.db)

    def dict(self, *args, **kwargs):
        """Return the query as a dictionary."""
        documents = []
        queries = []
        stringify(self, documents=documents, queries=queries)
        query = '\n'.join(queries)
        return Document(
            {
                '_path': 'superduper.backends.base.query.parse_query',
                'documents': documents,
                'identifier': self.identifier,
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
                query = query.replace(f'documents[{numeral}]', str(r['documents'][int(numeral)]))

        doc_segs = re.findall('documents\[([0-9]+):([0-9]+)\]', query)
        if doc_segs:
            for n1, n2 in doc_segs:
                query = query.replace(f'documents[{n1}:{n2}]', ' '.join([str(rr) for rr in  r['documents'][int(n1):int(n2)]]))

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
            from superduper.misc.special_dicts import SuperDuperFlatEncode

            if isinstance(out, SuperDuperFlatEncode):
                out.pop_builds()
                out.pop_files()
                out.pop_blobs()

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
                return db.show('listener', identifier, -1)['uuid']
            except FileNotFoundError as e:
                logging.warn(
                    f'Error in completing UUIDs from saved components,'
                    f' based on `listenerr={identifier}`: {e}'
                )
                pass

            raise Exception(msg.format(predict_id=identifier))

        for i, line in enumerate(lines):
            output_query_groups = re.findall('\.outputs\((.*?)\)', line)

            for group in output_query_groups:
                predict_ids = [eval(x.strip()) for x in group.split(',')]
                replace_ids = []
                for predict_id in predict_ids:
                    if re.match(r'^.*__([0-9a-z]{15,})$', predict_id):
                        replace_ids.append(f'"{predict_id}"')
                        continue
                    listener_uuid = _get_uuid(predict_id)
                    replace_ids.append(f'"{predict_id}__{listener_uuid}"')
                new_group = ', '.join(replace_ids)
                lines[i] = lines[i].replace(group, new_group)

            output_table_groups = re.findall(f'^{CFG.output_prefix}.*?\.', line)

            for group in output_table_groups:
                if re.match(f'^{CFG.output_prefix}[^\.]+__([0-9a-z]{{15,}})\.$', group):
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
        del r['identifier']
        out = parser(**r, db=db)
        return out

    @property
    def primary_id(self):
        return Query(table=self.table, parts=(), db=self.db) + 'primary_id'

    @property
    def documents(self):
        return self.dict()['documents']

    def subset(self, ids: t.Sequence[str]):
        assert self.type == 'select'
        t = self.db[self.table]
        modified_query = self.filter(t.primary_id.isin(ids))
        return modified_query.execute()

    def execute(self):
        if self.type == 'insert':
            self.db._prepare_insert(self)
        return self.db.databackend.plugin.Executor(self, db=self.db).execute()

    def get(self):
        return self.db.databackend.plugin.Executor(self, db=self.db).get()

    def _convert_eager_mode_results(self, results):
        from superduper.base.cursor import SuperDuperCursor
        from superduper.misc.eager import SuperDuperData, SuperDuperDataType

        new_results = []
        query = self
        if not len(query.parts):
            query = query.select()
        if isinstance(results, (SuperDuperCursor, list)):
            for r in results:
                r = Document(r.unpack())
                sdd = SuperDuperData(r, type=SuperDuperDataType.DATA, query=query)
                new_results.append(sdd)

            return new_results

        elif isinstance(results, dict):
            return SuperDuperData(results, type=SuperDuperDataType.DATA, query=query)

        raise ValueError(f'Cannot convert {results} to eager mode results')


def _parse_query_part(part, documents, query, db):
    if part.startswith(CFG.output_prefix):
        predict_id = part[len(CFG.output_prefix) :]
        table = f'{CFG.output_prefix}{predict_id}'
    else:
        table = part.split('.', 1)[0]
    rest_part = part[len(table) + 1 :]

    col = None
    col_match = re.match('^([a-zA-Z0-9]+)\["[a-zA-Z0-9]+"\]$', table)
    if col_match:
        table = col_match.groups()[0]
        col = col_match.groups()[1]

    parts = re.findall(r'\.([a-zA-Z0-9_]+)(\(.*?\))?', "." + rest_part)

    # TODO what's this clause?
    recheck_part = ".".join(p[0] + p[1] for p in parts)
    if recheck_part != rest_part:
        raise ValueError(f'Invalid query part: {part} != {recheck_part}')

    new_parts = []
    for part in parts:
        if isinstance(part, str) and re.match('^[a-zA-Z0-9]+\["[a-zA-Z0-9]+"\]$', part) is not None:
            new_parts.extend(part.split('[')[0], part.split(']').strip()[:-1])
            continue
        new_parts.append(part)

    if col is None:
        current = Query(table=table, parts=(), db=db)
    else:
        current = Query(table=table, parts=(col,), db=db)

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


def from_parts(impl, table, parts, db):
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
    :param builder_cls: The class to use to build the query.
    :param documents: The documents to query.
    :param db: The datalayer to use to execute the query.
    """

    if isinstance(query, str):
        query = [x.strip() for x in query.split('\n') if x.strip()]
    for i, q in enumerate(query):
        query[i] = _parse_query_part(q, documents, query[:i], db=db)

    return query[-1]


class Executor(ABC):

    def __init__(self, parent, db: 'Datalayer'):
        self.parent = parent
        self.db = db

    @property
    def decomposition(self) -> Decomposition:
        return self.parent.decomposition

    def get(self):
        assert self.parent.type == 'select'

        if self.decomposition.pre_like:
            return list(self._execute_pre_like(n=1))[0]

        elif self.decomposition.post_like:
            return list(self._execute_post_like(n=1))[0]

        return Document(next(iter(self._execute_select())))

    def to_id(self, id):
        return id

    @functools.cached_property
    def primary_id(self):
        return self.parent.primary_id.execute()

    def _wrap_results(self, result):
        for r in result:
            if self.primary_id in r:
                r[self.primary_id] = str(r[self.primary_id])
        result = [Document.decode(r, schema=self.schema) for r in result]
        return [Document(r) for r in result]

    def execute(self):

        if self.decomposition.insert:
            return self._do_execute_insert()

        if self.decomposition.col == 'primary_id':
            return self._execute_primary_id()

        elif self.decomposition.pre_like:
            return self._wrap_results(self._execute_pre_like())

        elif self.decomposition.post_like:
            return self._wrap_results(self._execute_post_like())

        return self._wrap_results(self._execute_select())

    @functools.cached_property
    def schema(self) -> Schema:
        base_schema = self.db.load('table', self.parent.table).schema

        # TODO handle outputs
        return base_schema

    def _do_execute_insert(self):
        documents = self.parent.documents
        if not self.schema.trivial:
            blobs = []
            files = []
            for i, r in enumerate(documents):
                documents[i] = self.schema.encode_data(r, builds={}, blobs=blobs, files=files)
        self._execute_insert(documents)

    @abstractmethod
    def _execute_insert(self, documents):
        pass

    @abstractmethod
    def _execute_primary_id(self):
        pass

    @abstractmethod
    def _execute_select(self):
        pass

    def _execute_pre_like(self):

        assert self.decomposition.pre_like is not None

        ids, scores = self.db.select_nearest(
            like=self.decomposition.pre_like.args[0],
            vector_index=self.decomposition.pre_like.args[1],
            n=self.decomposition.pre_like.kwargs.get('n', 10),
        )

        lookup = {id: score for id, score in zip(ids, scores)}

        t = self.db[self.decomposition.table]
        new_filter = t.primary_id.isin(ids)

        copy = self.decomposition.copy()
        copy.pre_like = None

        new = copy.to_query()
        new = new.filter(new_filter)

        results = new.execute()

        for r in results:
            r['score'] = lookup[r[self.primary_id]]

        return results

    def _execute_post_like(self):

        like_part = self.parent[-1]
        prepare_query = self.parent[:-1]
        relevant_ids = prepare_query.ids()

        ids, scores = self.db.select_nearest(
            like=like_part.args[0],
            vector_index=like_part.args[1],
            n=like_part.kwargs['n'],
            ids=relevant_ids,
        )

        lookup = {id: score for id, score in zip(ids, scores)}

        t = self.db[self.parent.table]

        results = prepare_query.filter(t.primary_id.isin(ids)).execute()

        for r in results:
            r['score'] = lookup[r[self.primary_id]]

        return results