import dataclasses as dc
import random
import typing as t

from superduperdb import CFG


class Insert:
    collection: 'Collection'
    documents: t.List['Document'] = dc.field(default_factory=list)
    refresh: bool = True
    verbose: bool = True
    args: t.Sequence = dc.field(default_factory=list)
    kwargs: t.Dict = dc.field(default_factory=dict)
    encoders: t.Sequence = dc.field(default_factory=list)
    type_id: t.Literal['mongodb.InsertMany'] = 'mongodb.InsertMany'

    def post(self, db, output):
        graph = None
        if self.refresh and not CFG.cdc:
            graph = db.refresh_after_update_or_insert(
                query=self,  # type: ignore[arg-type]
                ids=output.inserted_ids,
                verbose=self.verbose,
            )
        return output, graph

    def pre(self, db):
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

        return documents


@dc.dataclass
class InsertMany(Insert):
    ...


class PlaceHolderQuery:
    def __init__(self, data, *args, **kwargs):
        self.name = data
        self.args = args
        self.kwargs = kwargs

    def pre(self, db):
        pass

    def post(self, db, output):
        pass


query_lookup = {'insert': InsertMany}


class Query:
    def __init__(self, name, type='query', args=[], kwargs={}):
        self.name = name
        self.query = query_lookup.get(name, PlaceHolderQuery(name))
        self.next = None
        self.db = None
        self.type = type
        self.args = args
        self.kwargs = kwargs

    def execute(self, db, parent, table):
        if self.type == "attr":
            return getattr(parent, self.query.name)

        self.query.pre(self.db)
        if len(self.args) == 1:
            if isinstance(self.args[0], Queries):
                self.args = [self.args[0].condition(db, table)]
        parent = getattr(parent, self.query.name)(*self.args, **self.kwargs)
        self.query.post(self.db, parent)
        return parent


class QueryChain:
    def __init__(self, seed=None, type='attr'):
        if seed:
            query = Query(seed, type=type)
        else:
            query = None

        self.chain = [query]

    def append(self, data, type='query'):
        query = Query(data, type=type)
        self.chain.append(query)

    def append_query(self, query):
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
            if query.type in ['query', 'attr']:
                yield query


@dc.dataclass
class Table:
    name: str
    primary_id: str = 'id'

    def __getattr__(self, k):
        return Queries(self, query_type=k, members=QueryChain(k, type='attr'))


class LogicalExprMixin:
    def _logical_expr(self, other, members, collection, k):
        args = [other]
        members.append_query(Query(k, args=args, kwargs={}))
        return Queries(collection, query_type=k, members=members)

    def eq(self, other, members, collection):
        k = '__eq__'
        self._logical_expr(other, members, collection, k)

    def gt(self, other, members, collection):
        k = '__gt__'
        self._logical_expr(other, members, collection, k)

    def lt(self, other, members, collection):
        k = '__lt__'
        self._logical_expr(other, members, collection, k)


@dc.dataclass
class Queries(LogicalExprMixin):
    collection: Table
    query_type: str = 'find'
    args: t.Sequence = dc.field(default_factory=list)
    kwargs: t.Dict = dc.field(default_factory=dict)
    is_callable: t.Optional[bool] = None
    members: QueryChain = dc.field(default_factory=QueryChain)

    def __getattr__(self, k):
        if k in self.__dict__:
            return self.__getattr__(k)
        self.members.append(k)
        return Queries(self.collection, query_type=k, members=self.members)

    def __eq__(self, other):
        self.eq(other, members=self.members, collection=self.collection)

    def __lt__(self, other):
        self.lt(other, members=self.members, collection=self.collection)

    def __gt__(self, other):
        self.gt(other, members=self.members, collection=self.collection)

    def select_ids(self):
        k = 'select'
        args = [self.collection.primary_id]
        self.members.append_query(Query(k, args=args, kwargs={}))
        return Queries(self.collection, query_type='select', members=self.members)

    def select_from_ids(self, ids):
        isin_query = self.collection.__getattr__(self.collection.primary_id).isin(ids)
        k = 'filter'
        args = [isin_query]
        self.members.append_query(Query(k, args=args, kwargs={}))

        return Queries(self.collection, query_type='filter', members=self.members)

    def __call__(self, *args, **kwargs):
        self.members.update_last_query(args, kwargs, type='query')

        return Queries(
            collection=self.collection,
            query_type=self.query_type,
            members=self.members,
            args=args,
            kwargs=kwargs,
        )

    def condition(self, db, parent):
        for member in self.members:
            parent = member.execute(db, parent, parent)
        return parent


class IbisConnection:
    def __init__(self, db):
        self.db = db

    def _execute(self, db, query, parent):
        table = parent
        for member in query.members:
            parent = member.execute(db, parent, table)
        return parent.execute()

    def execute(self, query):
        return self._execute(self.db, query, self.db.table(query.collection.name))
