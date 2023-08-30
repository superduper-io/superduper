import dataclasses as dc
import enum
import typing as t

import ibis
from ibis.expr.types.relations import Table as IbisTable

from superduperdb import CFG
from superduperdb.container.component import Component
from superduperdb.container.document import Document
from superduperdb.container.encoder import Encoder
from superduperdb.container.serializable import Serializable
from superduperdb.db.base.db import DB
from superduperdb.db.ibis.cursor import SuperDuperIbisCursor
from superduperdb.db.ibis.schema import IbisSchema

PRIMARY_ID: str = 'id'


class QueryType(enum.Enum):
    QUERY = 'query'
    ATTR = 'attr'


class Query:
    def __init__(
        self,
        name,
        type=QueryType.QUERY.value,
        args=[],
        kwargs={},
        sddb_kwargs={},
        connection_parent=False,
    ):
        self.name = name
        query = query_lookup.get(name, None)
        if query is None:
            self.query = PlaceHolderQuery(name)
        else:
            self.query = query(**sddb_kwargs)
        self.type = type
        self.args = args
        self.kwargs = kwargs
        self.connection_parent = connection_parent

    def execute(self, db, parent, table, ibis_table=None):
        if self.type == QueryType.ATTR.value:
            return getattr(parent, self.query.name)

        self.query.pre(db, table=table, kwargs=self.kwargs)
        if len(self.args) == 1 and isinstance(self.args[0], QueryLinker):
            self.args = [self.args[0].execute(db, parent, ibis_table)]

        if self.query.namespace == 'ibis':
            if self.connection_parent:
                parent = getattr(db.db, self.query.name)(*self.args, **self.kwargs)
            else:
                parent = getattr(parent, self.query.name)(*self.args, **self.kwargs)

        parent = self.query.post(
            db,
            parent,
            table=table,
            ibis_table=ibis_table,
            args=self.args,
            kwargs=self.kwargs,
        )
        return parent


class QueryChain:
    def __init__(self, seed=None, type=QueryType.QUERY.value):
        if isinstance(seed, str):
            query = Query(seed, type=type)
        elif isinstance(seed, Query):
            query = seed
        else:
            query = None

        self.chain = [query]

    def append(self, data, type=QueryType.QUERY.value):
        query = Query(data, type=type)
        self.chain.append(query)

    def append_query(self, query, create=False):
        if create:
            return self.chain + [query]

        self.chain.append(query)

    def get(self, ix):
        return self.chain[ix]

    def update_last_query(self, args, kwargs, type=None):
        self.chain[-1].args = args
        self.chain[-1].kwargs = kwargs
        if type:
            self.chain[-1].type = type

    def __iter__(self):
        for query in self.chain:
            if query.type in [QueryType.QUERY.value, QueryType.ATTR.value]:
                yield query

    def __repr__(self):
        return str(self.chain)


@dc.dataclass
class OutputTable:
    model: str
    primary_id: str = 'id'
    table: t.Any = None
    output_type: t.Any = None

    def create(self, conn):
        self.table = conn.create_table(self.model, schema=self.schema)
        return self.table

    @property
    def schema(self):
        assert self.output_type is not None, "Output type must be set"

        if isinstance(self.output_type, Encoder):
            output_type = 'binary'
        else:
            output_type = self.output_type

        schema = {
            'id': 'int32',
            'input_id': 'int32',
            'query_id': 'string',
            'output': output_type,
            'key': 'string',
        }
        return ibis.schema(schema)


@dc.dataclass
class InMemoryTable(Component):
    identifier: str
    table: t.Any = None
    primary_id: str = 'id'
    type_id: t.ClassVar[str] = 'inmemory_table'

    def mutate_args(self, args):
        return args

    def __getattr__(self, k):
        if k in self.__dict__:
            return self.__getattr__(k)

        if not hasattr(self.table, k):
            raise AttributeError(k)

        return QueryLinker(
            self, query_type=k, members=QueryChain(k, type=QueryType.ATTR.value)
        )


@dc.dataclass
class Table(Component):
    identifier: str
    schema: t.Optional[IbisSchema] = None
    table: t.Any = None
    primary_id: str = 'id'
    #: A unique name for the class
    type_id: t.ClassVar[str] = 'table'
    version: t.ClassVar[t.Optional[int]] = None

    def on_create(self, db: DB):
        self.create(db)

    def create(self, db: DB):
        assert self.schema is not None, "Schema must be set"
        for e in self.schema.encoders:
            db.add(e)
        try:
            db.db.create_table(self.identifier, schema=self.schema.map())
        except Exception as e:
            if 'already exists' in str(e):
                pass
            else:
                raise e

    def get_table(self, conn):
        if self.table is None:
            self.table = conn.table(self.identifier)
        return self.table

    @property
    def name(self):
        return self.identifier

    def mutate_args(self, args):
        mutated_args = []
        for attr in args:
            if isinstance(attr, str):
                if attr in self.schema.fields:
                    mutated_args.append(self.schema.mutate_column(attr))
                else:
                    mutated_args.append(attr)
            elif isinstance(attr, QueryLinker):
                attr_query = attr.get_latest_query()
                if attr_query.type == QueryType.ATTR.value:
                    mutated_args.append(self.schema.mutate_column(attr.query_type))
                else:
                    mutated_args.append(attr)
            else:
                mutated_args.append(attr)

        return mutated_args

    def __getattr__(self, k):
        if k in self.__dict__:
            return self.__getattr__(k)

        if not (hasattr(IbisTable, k) or k in self.schema.fields):
            breakpoint()
            raise AttributeError(k)

        return QueryLinker(
            self, query_type=k, members=QueryChain(k, type=QueryType.ATTR.value)
        )

    def like(self, r=None, n=10, vector_index=None):
        k = 'prelike'
        kwargs = {'r': r, 'n': n, 'vector_index': vector_index}
        query = Query(k, args=[], sddb_kwargs=kwargs)
        return QueryLinker(self, query_type=k, members=QueryChain(query))

    def insert(
        self,
        *args,
        refresh: bool = True,
        verbose: bool = True,
        encoders: t.Sequence = [],
        valid_prob: float = 0.05,
        **kwargs,
    ):
        sddb_kwargs = {
            'refresh': refresh,
            'verbose': verbose,
            'encoders': encoders,
            'kwargs': {'valid_prob': valid_prob},
            'documents': args[0],
        }
        kwargs.update({'table_name': self.identifier})
        args = args[1:]

        insert = Query(
            'insert',
            type=QueryType.QUERY.value,
            args=args,
            kwargs=kwargs,
            sddb_kwargs=sddb_kwargs,
            connection_parent=True,
        )

        qc = QueryChain(insert)
        return QueryLinker(self, query_type='insert', members=qc)


class LogicalExprMixin:
    def _logical_expr(self, other, members, collection, k):
        args = [other]
        members.append_query(Query(k, args=args, kwargs={}))
        return QueryLinker(collection, query_type=k, members=members)

    def eq(self, other, members, collection):
        k = '__eq__'
        return self._logical_expr(other, members, collection, k)

    def gt(self, other, members, collection):
        k = '__gt__'
        return self._logical_expr(other, members, collection, k)

    def lt(self, other, members, collection):
        k = '__lt__'
        return self._logical_expr(other, members, collection, k)


@dc.dataclass
class QueryLinker(Serializable, LogicalExprMixin):
    collection: Table
    query_type: str = 'find'
    args: t.Sequence = dc.field(default_factory=list)
    kwargs: t.Dict = dc.field(default_factory=dict)
    members: QueryChain = dc.field(default_factory=QueryChain)

    def get_latest_query(self):
        return self.members.get(-1)

    def __getattr__(self, k):
        if k in self.__dict__:
            return self.__getattr__(k)
        self.members.append(k)
        return QueryLinker(self.collection, query_type=k, members=self.members)

    def __eq__(self, other):
        return self.eq(other, members=self.members, collection=self.collection)

    def __lt__(self, other):
        return self.lt(other, members=self.members, collection=self.collection)

    def __gt__(self, other):
        return self.gt(other, members=self.members, collection=self.collection)

    def select_ids(self):
        k = 'select'
        args = [self.collection.primary_id]
        members = self.members.append_query(Query(k, args=args, kwargs={}), create=True)
        return QueryLinker(self.collection, query_type='select', members=members)

    def select_using_ids(self, ids):
        isin_query = self.collection.__getattr__(self.collection.primary_id).isin(ids)
        k = 'filter'
        args = [isin_query]
        members = self.members.append_query(Query(k, args=args, kwargs={}), create=True)

        return QueryLinker(self.collection, query_type='filter', members=members)

    def build(self, db):
        table = parent = self.collection.table
        for member in self.members:
            parent = member.execute(db, parent, self.collection, table)
        return parent

    def outputs(self, model, db):
        curr_query = self.build(db)
        model_table = db.db.table(model)
        query = curr_query.join(
            model_table,
            [
                model_table.id == curr_query.id,
                model_table.query_id == self.collection.identifier,
            ],
        )
        cursor = SuperDuperIbisCursor(
            query, self.collection.primary_id, encoders=db.encoders
        )
        return cursor

    def model_update(
        self,
        db,
        ids: t.Sequence[t.Any],
        key: str,
        model: str,
        outputs: t.Sequence[t.Any],
    ):
        if key.startswith('_outputs'):
            key = key.split('.')[1]
        if not outputs:
            return

        input_table = self.collection.identifier

        table_record = []
        for ix in range(len(outputs)):
            d = Document(
                {
                    'id': int(ids[ix]),
                    'input_id': int(ids[ix]),
                    'output': outputs[ix],
                    'query_id': input_table,
                    'key': key,
                }
            )
            table_record.append(d)
        db.execute(Table(model).insert(table_record))

    def __call__(self, *args, **kwargs):
        args = self.collection.mutate_args(args)

        # TODO: handle kwargs
        self.members.update_last_query(args, kwargs, type=QueryType.QUERY.value)

        return QueryLinker(
            collection=self.collection,
            query_type=self.query_type,
            members=self.members,
            args=args,
            kwargs=kwargs,
        )

    def execute(self, db, parent, ibis_table):
        for member in self.members:
            parent = member.execute(db, parent, self.collection, ibis_table=ibis_table)
        return parent


class PlaceHolderQuery:
    type_id: t.Literal['query'] = 'query'  # type: ignore[annotation-unchecked]

    def __init__(self, name, *args, **kwargs):
        self.name = name
        self.args = args
        self.kwargs = kwargs

        self.namespace: str = 'ibis'  # type: ignore[annotation-unchecked]

    def pre(self, db, **kwargs):
        ...

    def post(self, db, output, *args, **kwargs):
        return output


@dc.dataclass
class PostLike:
    type_id: t.Literal['Ibis.PostLike'] = 'Ibis.PostLike'
    name: str = 'postlike'
    namespace: str = 'sddb'

    def pre(self, db, **kwargs):
        pass

    def post(self, db, output, table=None, ibis_table=None, args=[], kwargs={}):
        r = kwargs.get('r', None)
        assert r is not None, 'r must be provided'
        n = kwargs.get('n', 10)

        vector_index = kwargs.get('vector_index', 'vector_index')
        ids = output.select(table.primary_id)
        ids, scores = db._select_nearest(
            like=r,
            vector_index=vector_index,
            n=n,
            ids=ids,
        )
        return output.filter(  # type: ignore[annotation-unchecked]
            ibis_table.__getattr__(table.primary_id).isin(ids)
        )


@dc.dataclass
class PreLike:
    r: t.Any
    vector_index: str = 'vector_index'
    n: int = 10
    collection: str = 'collection'
    primary_id: str = 'id'
    type_id: t.Literal['Ibis.PreLike'] = 'Ibis.PreLike'
    name: str = 'prelike'
    namespace: str = 'sddb'

    def pre(self, db, **kwargs):
        pass

    def post(self, db, output, table=None, ibis_table=None, args=[], kwargs={}):
        '''

        ids, _ = db._select_nearest(
            like=self.r, vector_index=self.vector_index, n=self.n
        )
        '''
        ids = [1, 2, 3]
        f = output.filter(ibis_table.__getattr__(self.primary_id).isin(ids))
        return f


@dc.dataclass
class Insert:
    documents: t.List['Document'] = dc.field(default_factory=list)
    refresh: bool = True
    verbose: bool = True
    kwargs: t.Dict = dc.field(default_factory=dict)
    encoders: t.Sequence = dc.field(default_factory=list)
    type_id: t.Literal['Ibis.insert'] = 'Ibis.insert'
    name: str = 'insert'
    namespace: str = 'ibis'

    def pre(self, db, table=None, **kwargs):
        # TODO: handle adding table later
        '''
        if self.base_table.identifier not in db.tables:
            db.add(self.base_table)
        '''
        for e in self.encoders:
            db.add(e)

        documents = [r.encode(table.schema) for r in self.documents]

        # TODO: handle _fold
        '''
        for r in documents:
            if '_fold' in r:
                continue
            if random.random() < 0.05:
                r['_fold'] = 'valid'
            else:
                r['_fold'] = 'train'
        '''

        # mutate kwargs with documents
        documents = [tuple(d.values()) for d in documents]
        kwargs['kwargs']['obj'] = documents
        return

    def post(self, db, output, *args, **kwargs):
        graph = None
        # inserted_ids = [d[PRIMARY_ID] for d in kwargs['kwargs']['obj']]
        if self.refresh and not CFG.cdc:
            '''
            graph = db.refresh_after_update_or_insert(
                query=self,  # type: ignore[arg-type]
                ids=inserted_ids,
                verbose=self.verbose,
            )
            '''
            pass
        return graph, output


query_lookup = {'insert': Insert, 'prelike': PreLike, 'like': PostLike}
