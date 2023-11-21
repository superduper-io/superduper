import dataclasses as dc
import enum
import typing as t
from abc import ABC, abstractmethod, abstractproperty
from typing import Any

from superduperdb import logging
from superduperdb.base.document import Document
from superduperdb.base.serializable import Serializable, Variable

GREEN = '\033[92m'
BOLD = '\033[1m'
END = '\033[0m}'


class _ReprMixin(ABC):
    @abstractmethod
    def repr_(self) -> str:
        pass

    def __repr__(self) -> str:
        return (
            f'<{self.__class__.__module__}.{self.__class__.__name__}'
            f'[\n    {GREEN + BOLD}{self.repr_()}{END}\n] object at {hex(id(self))}>'
        )


@dc.dataclass(repr=False)
class Select(Serializable, ABC):
    """
    Base class for all select queries.
    """

    @abstractproperty
    def id_field(self):
        pass

    @property
    def query_components(self):
        return self.table_or_collection.query_components

    def model_update(
        self,
        db,
        ids: t.Sequence[str],
        key: str,
        model: str,
        version: int,
        outputs: t.Sequence[t.Any],
        **kwargs,
    ):
        """
        Update model outputs for a set of ids.

        :param db: The DB instance to use
        :param ids: The ids to update
        :param key: The key to update
        :param model: The model to update
        :param outputs: The outputs to update
        """
        return self.table_or_collection.model_update(
            db=db,
            ids=ids,
            key=key,
            model=model,
            outputs=outputs,
            version=version,
            **kwargs,
        )

    @abstractproperty
    def select_table(self):
        pass

    @abstractmethod
    def add_fold(self, fold: str) -> 'Select':
        pass

    @abstractmethod
    def select_using_ids(self, ids: t.Sequence[str]) -> 'Select':
        pass

    @abstractproperty
    def select_ids(self) -> 'Select':
        pass

    @abstractmethod
    def select_ids_of_missing_outputs(
        self, key: str, model: str, version: int
    ) -> 'Select':
        pass

    @abstractmethod
    def select_single_id(self, id: str) -> 'Select':
        pass

    @abstractmethod
    def execute(self, db):
        """
        Execute the query on the DB instance.
        """
        pass


@dc.dataclass(repr=False)
class Delete(Serializable, ABC):
    """
    Base class for all deletion queries

    :param table_or_collection: The table or collection that this query is linked to
    """

    table_or_collection: 'TableOrCollection'
    args: t.Sequence = dc.field(default_factory=list)
    kwargs: t.Dict = dc.field(default_factory=dict)

    @abstractmethod
    def execute(self, db):
        pass


@dc.dataclass(repr=False)
class Update(Serializable, ABC):
    """
    Base class for all update queries

    :param table_or_collection: The table or collection that this query is linked to
    """

    table_or_collection: 'TableOrCollection'

    @abstractmethod
    def select_table(self):
        pass

    @abstractmethod
    def execute(self, db):
        pass


@dc.dataclass(repr=False)
class CompoundSelect(_ReprMixin, Select, ABC):
    """
    A query with multiple parts.

    like ----> select ----> like

    :param table_or_collection: The table or collection that this query is linked to
    :param pre_like: The pre_like part of the query (e.g. ``table.like(...)...``)
    :param post_like: The post_like part of the query
                      (e.g. ``table.filter(...)....like(...)``)
    :param query_linker: The query linker that is responsible for linking the
                         query chain. E.g. ``table.filter(...).select(...)``.
    :param i: The index of the query in the query chain
    """

    table_or_collection: 'TableOrCollection'
    pre_like: t.Optional['Like'] = None
    post_like: t.Optional['Like'] = None
    query_linker: t.Optional['QueryLinker'] = None

    @abstractproperty
    def output_fields(self):
        pass

    @property
    def id_field(self):
        return self.primary_id

    @property
    def primary_id(self):
        return self.table_or_collection.primary_id

    def add_fold(self, fold: str):
        assert self.pre_like is None
        assert self.post_like is None
        assert self.query_linker is not None
        return self._query_from_parts(
            table_or_collection=self.table_or_collection,
            query_linker=self.query_linker.add_fold(fold),
        )

    @property
    def select_ids(self):
        """
        Query which selects the same documents/ rows but only ids.
        """

        assert self.pre_like is None
        assert self.post_like is None

        return self._query_from_parts(
            table_or_collection=self.table_or_collection,
            query_linker=self.query_linker.select_ids,
        )

    def select_ids_of_missing_outputs(self, key: str, model: str, version: int):
        """
        Query which selects ids where outputs are missing.
        """

        assert self.pre_like is None
        assert self.post_like is None
        assert self.query_linker is not None

        return self._query_from_parts(
            table_or_collection=self.table_or_collection,
            query_linker=self.query_linker._select_ids_of_missing_outputs(
                key=key, model=model, version=version
            ),
        )

    def select_single_id(self, id: str):
        """
        Query which selects a single id.

        :param id: The id to select.
        """
        assert self.pre_like is None
        assert self.post_like is None
        assert self.query_linker is not None

        return self._query_from_parts(
            table_or_collection=self.table_or_collection,
            query_linker=self.query_linker.select_single_id(id),
        )

    def select_using_ids(self, ids):
        """
        Subset a query to only these ids.

        :param ids: The ids to subset to.
        """

        assert self.pre_like is None
        assert self.post_like is None

        return self._query_from_parts(
            table_or_collection=self.table_or_collection,
            query_linker=self.query_linker.select_using_ids(ids),
        )

    def repr_(self):
        """
        String representation of the query.
        """

        components = []
        components.append(str(self.table_or_collection.identifier))
        if self.pre_like:
            components.append(str(self.pre_like))
        if self.query_linker:
            components.extend(self.query_linker.repr_().split('.')[1:])
        if self.post_like:
            components.append(str(self.post_like))
        return '.'.join(components)

    @classmethod
    def _query_from_parts(
        cls,
        table_or_collection,
        pre_like=None,
        post_like=None,
        query_linker=None,
    ):
        return cls(
            table_or_collection=table_or_collection,
            pre_like=pre_like,
            post_like=post_like,
            query_linker=query_linker,
        )

    def _get_query_component(
        self,
        name: str,
        type: str,
        args: t.Optional[t.Sequence] = None,
        kwargs: t.Optional[t.Dict] = None,
    ):
        query_component_cls = self.table_or_collection.query_components.get(
            name, QueryComponent
        )
        return query_component_cls(name, type=type, args=args, kwargs=kwargs)

    @abstractmethod
    def _get_query_linker(cls, table_or_collection, members) -> 'QueryLinker':
        pass

    def __getattr__(self, name):
        assert self.post_like is None
        if self.query_linker is not None:
            query_linker = getattr(self.query_linker, name)
        else:
            query_linker = self._get_query_linker(
                self.table_or_collection,
                members=[self._get_query_component(name, type=QueryType.ATTR)],
            )
        return self._query_from_parts(
            table_or_collection=self.table_or_collection,
            pre_like=self.pre_like,
            query_linker=query_linker,
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        assert self.post_like is None
        assert self.query_linker is not None
        return self._query_from_parts(
            table_or_collection=self.table_or_collection,
            pre_like=self.pre_like,
            query_linker=self.query_linker(*args, **kwargs),
        )

    @abstractmethod
    def execute(self, db):
        """
        Execute the compound query on the DB instance.

        :param db: The DB instance to use
        """

    def like(self, r: Document, vector_index: str, n: int = 10):
        assert self.query_linker is not None
        assert self.pre_like is None
        return self._query_from_parts(
            table_or_collection=self.table_or_collection,
            pre_like=None,
            query_linker=self.query_linker,
            post_like=Like(r=r, n=n, vector_index=vector_index),
        )


@dc.dataclass(repr=False)
class Insert(_ReprMixin, Serializable, ABC):
    """
    Base class for all insert queries.

    :param table_or_collection: The table or collection that this query is linked to
    :param documents: The documents to insert
    :param refresh: Whether to refresh the task-graph after inserting
    :param verbose: Whether to print the progress of the insert
    :param kwargs: Any additional keyword arguments to pass to the insert method
    :param encoders: The encoders to use to encode the documents
    """

    table_or_collection: 'TableOrCollection'
    documents: t.Sequence['Document'] = dc.field(default_factory=list)
    verbose: bool = True
    kwargs: t.Dict = dc.field(default_factory=dict)

    def repr_(self):
        documents_str = (
            str(self.documents)[:25] + '...'
            if len(self.documents) > 25
            else str(self.documents)
        )
        return f'{self.table_or_collection.identifier}.insert_many({documents_str})'

    @abstractmethod
    def select_table(self):
        pass

    @abstractmethod
    def execute(self, parent: t.Any):
        """
        Insert the data.

        :param parent: The parent instance to use for insertion
        """
        pass

    def to_select(self, ids=None):
        if ids is None:
            ids = [r['_id'] for r in self.documents]
        return self.table.find({'_id': ids})


class QueryType(str, enum.Enum):
    """
    The type of a query. Either `query` or `attr`.
    """

    QUERY = 'query'
    ATTR = 'attr'


@dc.dataclass(repr=False)
class QueryComponent(Serializable):
    """
    This is a representation of a single query object in ibis query chain.
    This is used to build a query chain that can be executed on a database.
    Query will be executed in the order they are added to the chain.

    If we have a query chain like this:
        query = t.select(['id', 'name']).limit(10)
    here we have 2 query objects, `select` and `limit`.

    `select` will be wrapped with this class and added to the chain.

    :param name: The name of the query
    :param type: The type of the query, either `query` or `attr`
    :param args: The arguments to pass to the query
    :param kwargs: The keyword arguments to pass to the query
    """

    name: str
    type: str = QueryType.ATTR
    args: t.Sequence = dc.field(default_factory=list)
    kwargs: t.Dict = dc.field(default_factory=dict)

    def repr_(self) -> str:
        if self.type == QueryType.ATTR:
            return self.name

        def to_str(x):
            if isinstance(x, str):
                return f"'{x}'"
            elif isinstance(x, list):
                return f'[{", ".join([to_str(a) for a in x])}]'
            elif isinstance(x, dict):
                return str({k: to_str(v) for k, v in x.items()})
            elif isinstance(x, _ReprMixin):
                return x.repr_()
            else:
                return str(x)

        args_as_strs = [to_str(a) for a in self.args]
        args_as_strs += [f'{k}={to_str(v)}' for k, v in self.kwargs.items()]
        joined = ', '.join(args_as_strs)
        return f'{self.name}({joined})'

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        try:
            assert (
                self.type == QueryType.ATTR
            ), '__call__ method must be called on an attribute query'
        except AssertionError as e:
            logging.warn('QUERY_COMPONENT: ' + self.name)
            raise e
        return type(self)(
            name=self.name,
            type=QueryType.QUERY,
            args=args,
            kwargs=kwargs,
        )

    def execute(self, parent: t.Any):
        if self.type == QueryType.ATTR:
            return getattr(parent, self.name)
        assert self.type == QueryType.QUERY
        parent = getattr(parent, self.name)(*self.args, **self.kwargs)
        return parent


@dc.dataclass(repr=False)
class QueryLinker(_ReprMixin, Serializable, ABC):
    """
    This class is responsible for linking together a query using
    `getattr` and `__call__`.

    This allows ``superduperdb`` to serialize queries from a range of APIs.
    Intuitively this allows us to do something like this:

    >>> collection.find({}).limit(10).sort('name')
    -->
    [
        ('<NAME>', <ARGS>, <KWARGS>),
        ('find', {}, None),
        ('limit', 10, None),
        ('sort', 'name', None),
    ]

    table.filter(t.select('id') == '1')

    :param table_or_collection: The table or collection that this query is linked to.
    :param members: The members of the query chain.
    """

    table_or_collection: 'TableOrCollection'
    members: t.List = dc.field(default_factory=list)

    @property
    def query_components(self):
        return self.table_or_collection.query_components

    def repr_(self) -> str:
        return (
            f'{self.table_or_collection.identifier}'
            + '.'
            + '.'.join([m.repr_() for m in self.members])
        )

    def _get_query_component(self, k):
        if k in self.query_components:
            return self.query_components[k](name=k, type=QueryType.ATTR)
        return QueryComponent(name=k, type=QueryType.ATTR)

    @classmethod
    def _get_query_linker(cls, table_or_collection, members):
        return cls(
            table_or_collection=table_or_collection,
            members=members,
        )

    def __getattr__(self, k):
        return self._get_query_linker(
            self.table_or_collection,
            members=[*self.members, self._get_query_component(k)],
        )

    @property
    @abstractmethod
    def select_ids(self):
        pass

    @abstractmethod
    def select_single_id(self, id):
        pass

    @abstractmethod
    def select_using_ids(self, ids):
        pass

    def __call__(self, *args, **kwargs):
        members = [*self.members[:-1], self.members[-1](*args, **kwargs)]
        return type(self)(table_or_collection=self.table_or_collection, members=members)

    @abstractmethod
    def execute(self, db):
        pass


@dc.dataclass
class Like(Serializable):
    """
    Base class for all like (vector-search) queries.

    :param r: The vector to search for
    :param vector_index: The vector index to use
    :param n: The number of results to return
    """

    r: Document
    vector_index: str
    n: int = 10

    def execute(self, db, ids: t.Optional[t.Sequence[str]] = None):
        return db.select_nearest(
            like=self.r,
            vector_index=self.vector_index,
            ids=ids,
            n=self.n,
        )


@dc.dataclass
class TableOrCollection(Serializable, ABC):
    """
    This is a representation of an SQL table in ibis.

    :param identifier: The name of the table
    """

    query_components: t.ClassVar[t.Dict] = {}
    type_id: t.ClassVar[str] = 'table_or_collection'
    identifier: t.Union[str, Variable]

    @abstractmethod
    def _get_query_linker(self, members) -> QueryLinker:
        pass

    def _get_query_component(self, name: str) -> QueryComponent:
        return self.query_components.get(name, QueryComponent)(
            name=name, type=QueryType.ATTR
        )

    @abstractmethod
    def _get_query(
        self,
        pre_like: t.Optional[Like] = None,
        query_linker: t.Optional[QueryLinker] = None,
        post_like: t.Optional[Like] = None,
    ) -> CompoundSelect:
        pass

    def __getattr__(self, k: str) -> 'CompoundSelect':
        # This method is responsible for dynamically creating a query chain,
        # which can be executed on a database. This is done by creating a
        # QueryLinker object, which is a representation of a query chain.
        # Under the hood, this is done by creating a QueryChain object, which
        # is a representation of a query chain.
        query_linker = self._get_query_linker([self._get_query_component(k)])
        return self._get_query(query_linker=query_linker)

    def like(
        self,
        r: Document,
        vector_index: str,
        n: int = 10,
    ):
        """
        This method appends a query to the query chain where the query is repsonsible
        for performing a vector search on the parent query chain inputs.

        :param r: The vector to search for
        :param vector_index: The vector index to use
        :param n: The number of results to return
        """
        return self._get_query(
            pre_like=Like(
                r=r,
                n=n,
                vector_index=vector_index,
            ),
        )

    @abstractmethod
    def _insert(
        self,
        documents: t.Sequence[Document],
        *,
        refresh: bool = False,
        encoders: t.Sequence = (),
        verbose: bool = True,
        **kwargs,
    ):
        pass


@dc.dataclass
class RawQuery:
    query: t.Any

    @abstractmethod
    def execute(self, db):
        '''
        A raw query method which executes the query and returns the result
        '''
