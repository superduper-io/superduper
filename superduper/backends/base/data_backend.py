import functools
import hashlib
import typing as t
import uuid
from abc import ABC, abstractmethod

from superduper import CFG, logging
from superduper.base import exceptions
from superduper.base.constant import KEY_BLOBS, KEY_BUILDS, KEY_FILES
from superduper.base.document import Document
from superduper.base.query import Query

if t.TYPE_CHECKING:
    from superduper.base.schema import Schema


class BaseDataBackend(ABC):
    """Base data backend for the database.

    :param uri: URI to the databackend database.
    :param plugin: Plugin implementing the databackend.
    :param flavour: Flavour of the databackend.
    """

    batched: bool = False
    id_field: str = 'id'

    # TODO plugin not required
    # TODO flavour required?
    def __init__(self, uri: str, plugin: t.Any, flavour: t.Optional[str] = None):
        self.conn = None
        self.flavour = flavour
        self.in_memory: bool = False
        self.in_memory_tables: t.Dict = {}
        self.plugin = plugin
        self._db = None
        self.uri = uri
        self.bytes_encoding = 'bytes'

    @property
    def database(self):
        raise NotImplementedError

    @property
    def backend_name(self):
        return self.__class__.__name__.split('DataBackend')[0].lower()

    @property
    def type(self):
        """Return databackend."""
        raise NotImplementedError

    def create_id(self, item: str):
        """Create a unique id for an item.

        The id is determistic and based on the item itself.

        :param item: The item to create an id for.
        """
        return hashlib.sha256(item.encode()).hexdigest()[:16]

    @abstractmethod
    def drop_table(self, table: str):
        """Drop data from table.

        :param table: The table to drop.
        """

    # TODO needed?
    @abstractmethod
    def random_id(self):
        """Generate random-id."""
        pass

    @property
    def db(self):
        """Return the datalayer."""
        return self._db

    @db.setter
    def db(self, value):
        """Set the datalayer.

        :param value: The datalayer.
        """
        self._db = value

    @abstractmethod
    def create_table_and_schema(
        self, identifier: str, schema: 'Schema', primary_id: str
    ):
        """Create a schema in the data-backend.

        :param identifier: The identifier of the schema.
        :param schema: The schema to create.
        :param primary_id: The primary id of the schema.
        """

    @abstractmethod
    def check_output_dest(self, predict_id) -> bool:
        """Check if the output destination exists.

        :param predict_id: The identifier of the output destination.
        """
        pass

    @abstractmethod
    def get_table(self, identifier):
        """Get a table or collection from the database.

        :param identifier: The identifier of the table or collection.
        """
        pass

    @abstractmethod
    def drop(self, force: bool = False):
        """Drop the databackend.

        :param force: If ``True``, don't ask for confirmation.
        """

    @abstractmethod
    def list_tables(self):
        """List all tables or collections in the database."""

    @abstractmethod
    def reconnect(self):
        """Reconnect to the databackend."""

    ########################################################
    # Abstract methods/ optional methods to be implemented #
    ########################################################

    @abstractmethod
    def insert(self, table: str, documents: t.Sequence[t.Dict]) -> t.List[str]:
        """Insert data into the database.

        :param table: The table to insert into.
        :param documents: The documents to insert.
        """

    @abstractmethod
    def replace(self, table: str, condition: t.Dict, r: t.Dict) -> t.List[str]:
        """Replace data.

        :param table: The table to insert into.
        :param condition: The condition to update.
        :param r: The document to replace.
        """

    @abstractmethod
    def update(self, table: str, condition: t.Dict, key: str, value: t.Any):
        """Update data in the database.

        :param table: The table to update.
        :param condition: The condition to update.
        :param key: The key to update.
        :param value: The value to update.
        """

    @abstractmethod
    def delete(self, table: str, condition: t.Dict):
        """Update data in the database.

        :param table: The table to update.
        :param condition: The condition to update.
        """

    @abstractmethod
    def missing_outputs(self, query: Query, predict_id: str) -> t.List[str]:
        """Get missing outputs from an outputs query.

        This method will be used to perform an anti-join between
        the input and the outputs table, and return the missing ids.

        :param query: The query to perform.
        :param predict_id: The predict id.
        """

    def primary_id(self, table: str) -> str:
        """Get the primary id of a table.

        :param table: The table to get the primary id of.
        """
        return self.db.metadata.get_primary_id(table)

    @abstractmethod
    def select(self, query: Query) -> t.List[t.Dict]:
        """Select data from the database.

        :param query: The query to perform.
        """

    def to_id(self, id: t.Any) -> str:
        """Convert an id to a string.

        :param id: The id to convert.
        """
        return id

    ##########################################
    # Methods which leverage implementations #
    ##########################################

    def get(self, query: Query, raw: bool = False):
        """Get a single result from a query.

        :param query: The query to perform.
        :param raw: If ``True``, return raw results.
        """
        assert query.type == 'select'

        if query.decomposition.pre_like:
            return list(self.pre_like(query, n=1, raw=raw))[0]

        elif query.decomposition.post_like:
            return list(self.post_like(query, n=1, raw=raw))[0]

        try:
            return query.limit(1).execute(raw=raw)[0]
        except IndexError:
            return None

    def _wrap_results(self, query: Query, result, schema, raw: bool = False):
        pid = self.primary_id(query.table)
        for r in result:
            if pid in r:
                r[pid] = str(r[pid])
            if '_source' in r:
                r['_source'] = str(r['_source'])

        if raw:
            return result
        return [Document.decode(r, schema=schema, db=self.db) for r in result]

    def execute(self, query: Query, raw: bool = False):
        """Execute a query.

        :param query: The query to execute.
        :param raw: If ``True``, return raw results.
        """
        query = query if '.outputs' not in str(query) else query.complete_uuids(self.db)

        schema = self.get_schema(query)

        if query.decomposition.pre_like:
            return self.pre_like(query, raw=raw)

        if query.decomposition.post_like:
            return self.post_like(query, raw=raw)

        return self._wrap_results(query, self.select(query), schema=schema, raw=raw)

    def get_schema(self, query) -> 'Schema':
        """Get the schema of a query.

        :param query: The query to get the schema of.
        """
        base_schema = self.db.metadata.get_schema(query.table)
        if query.decomposition.outputs:
            for predict_id in query.decomposition.outputs.args:
                base_schema += self.db.metadata.get_schema(
                    f'{CFG.output_prefix}{predict_id}'
                )

        return base_schema

    def _do_insert(self, table, documents, raw: bool = False):

        schema = self.get_schema(self.db[table])

        if not raw and not schema.trivial:
            schema = self.get_schema(self.db[table])
            for i, r in enumerate(documents):
                r = Document(r).encode(schema=schema, db=self.db)
                if r.get(KEY_BLOBS) or r.get(KEY_FILES):
                    self.db.artifact_store.save_artifact(r)
                r.pop(KEY_BUILDS)
                r.pop(KEY_BLOBS)
                r.pop(KEY_FILES)
                documents[i] = r
        else:
            for i, r in enumerate(documents):
                r = dict(r)
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
                documents[i] = r

        out = self.insert(table, documents)
        return [str(x) for x in out]

    def pre_like(self, query: Query, **kwargs):
        """Perform a pre-like query.

        :param query: The query to perform.
        :param kwargs: Additional keyword arguments.
        """
        assert query.decomposition.pre_like is not None

        ids, scores = self.db.select_nearest(
            like=query.decomposition.pre_like.args[0],
            vector_index=query.decomposition.pre_like.args[1],
            n=query.decomposition.pre_like.kwargs.get('n', 10),
        )

        lookup = {id: score for id, score in zip(ids, scores)}

        t = self.db[query.decomposition.table]
        new_filter = t.primary_id.isin(ids)

        copy = query.decomposition.copy()
        copy.pre_like = None

        new = copy.to_query()
        new = new.filter(new_filter)

        results = new.execute(**kwargs)

        pid = self.primary_id(query.table)
        for r in results:
            r['score'] = lookup[r[pid]]

        results = sorted(results, key=lambda x: x['score'], reverse=True)
        return results

    def post_like(self, query: Query, **kwargs):
        """Perform a post-like query.

        :param query: The query to perform.
        :param kwargs: Additional keyword arguments.
        """
        like_part = query[-1]
        prepare_query = query[:-1]
        relevant_ids = prepare_query.ids()

        ids, scores = self.db.select_nearest(
            like=like_part.args[0],
            vector_index=like_part.args[1],
            n=like_part.kwargs['n'],
            ids=relevant_ids,
        )

        lookup = {id: score for id, score in zip(ids, scores)}

        t = self.db[query.table]

        results = prepare_query.filter(t.primary_id.isin(ids)).execute(**kwargs)

        pid = self.primary_id(query.table)

        for r in results:
            r['score'] = lookup[r[pid]]

        results = sorted(results, key=lambda x: x['score'], reverse=True)
        return results

    @abstractmethod
    def execute_native(self, query: str):
        """Execute a native query provided as a str.

        :param query: The query to execute.
        """
        pass


class DataBackendProxy:
    """
    Proxy class to DataBackend which acts as middleware for performing fallbacks.

    :param backend: Instance of `BaseDataBackend`.
    """

    def __init__(self, backend):
        self._backend = backend

    @property
    def db(self):
        """Return the datalayer."""
        return self._backend._datalayer

    @db.setter
    def db(self, value):
        """Set the datalayer.

        :param value: The datalayer.
        """
        self._backend._db = value

    @property
    def type(self):
        """Instance of databackend."""
        return self._backend

    def _try_execute(self, attr):
        @functools.wraps(attr)
        def wrapper(*args, **kwargs):
            try:
                result = attr(*args, **kwargs)
                return result
            except Exception as e:
                error_message = str(e).lower()
                if "expire" in error_message and "token" in error_message:
                    logging.warn(
                        f"Authentication expiry detected: {e}. "
                        "Attempting to reconnect..."
                    )
                    self._backend.reconnect()
                    return attr(*args, **kwargs)
                else:
                    raise e

        return wrapper

    def __getattr__(self, name):
        attr = getattr(self._backend, name)

        if callable(attr):
            return self._try_execute(attr)
        return attr


class KeyedDatabackend(BaseDataBackend):
    """Keyed databackend for the database.

    :param uri: URI to the databackend database.
    :param plugin: Plugin implementing the databackend.
    :param flavour: Flavour of the databackend.
    """

    @abstractmethod
    def __getitem__(self, key: t.Tuple[str, str, str]) -> t.Dict:
        pass

    @abstractmethod
    def __setitem__(self, key: t.Tuple[str, str, str], value: t.Any):
        pass

    def get_many(self, *pattern: t.Sequence[str]):
        """Get many items from the database.

        :param pattern: The pattern to match.
        """
        keys = self.keys(*pattern)
        if not keys:
            return []
        else:
            return [self[key] for key in keys]

    def check_output_dest(self, predict_id):
        """Check if the output destination exists.

        :param predict_id: The identifier of the output destination.
        """
        raise NotImplementedError

    def create_table_and_schema(
        self, identifier: str, schema: 'Schema', primary_id: str
    ):
        """Create a schema in the data-backend.

        :param identifier: The identifier of the schema.
        :param schema: The schema to create.
        :param primary_id: The primary id of the schema.
        """
        pass

    def delete(self, table, condition):
        """
        Delete data from the database.

        :param table: The table to delete from.
        :param condition: The condition to delete.
        """
        r_table = self._get_with_component_identifier('Table', table)

        if not r_table['is_component']:
            pid = self.primary_id(table)
            if pid in condition:
                docs = self.get_many(table, condition[pid])
            else:
                docs = self.get_many(table, '*')
            docs = self._do_filter(docs, condition)
            for r in docs:
                del self[table, r[pid]]
        else:
            if 'uuid' in condition:
                docs = self.get_many(table, '*', condition['uuid'])
            elif 'identifier' in condition:
                docs = self.get_many(table, condition['identifier'], '*')
                docs = self._do_filter(docs, condition)
            else:
                docs = self.get_many(table, '*', '*')
                docs = self._do_filter(docs, condition)
            for r in docs:
                del self[table, r['identifier'], r['uuid']]

    def drop_table(self, table):
        """Drop data from table.

        :param table: The table to drop.
        """
        for k in self.keys(table, '*', '*'):
            del self[k]

    def execute_native(self, query):
        """Execute a native query provided as a str.

        (Not implemented in this class)

        :param query: The query to execute
        """
        raise NotImplementedError

    def get_table(self, identifier):
        """Get a table or collection from the database.

        (Not implemented in this class)

        :param identifier: The identifier of the table or collection.
        """
        raise NotImplementedError

    def list_tables(self):
        """List all tables in the database."""
        keys = self.keys('*', '*', '*') + self.keys('*', '*')
        return sorted(list(set(k[0] for k in keys)))

    def missing_outputs(self, query, predict_id):
        """Get missing outputs from an outputs query.

        (Not implemented in this class)

        :param query: The query to perform.
        :param predict_id: The predict id.
        """
        raise NotImplementedError

    def random_id(self):
        """Generate a random id."""
        return str(uuid.uuid4())[:16]

    def _do_filter(self, docs, condition):
        if not condition:
            return docs

        def do_test(r):
            for k, v in condition.items():
                if r.get(k) != v:
                    return False
            return True

        return [r for r in docs if do_test(r)]

    def replace(self, table, condition, r):
        """Replace data.

        :param table: The table to insert into.
        :param condition: The condition to update.
        :param r: The document to replace.
        """
        r_table = self._get_with_component_identifier('Table', table)

        if not r_table['is_component']:
            pid = self.primary_id(table)
            docs = self.get_many(table, condition[pid])
            docs = self._do_filter(docs, condition)
            for s in docs:
                r[pid] = s[pid]
                self[table, s[pid]] = r
        else:
            if 'uuid' in condition:
                s = self.get_many(table, '*', condition['uuid'])[0]
                self[table, s['identifier'], condition['uuid']] = r
            elif 'identifier' in condition:
                docs = self.get_many(table, condition['identifier'], '*')
                docs = self._do_filter(docs, condition)
                for s in docs:
                    self[table, s['identifier'], s['uuid']] = r
            else:
                docs = self.get_many(table, '*', '*')
                docs = self._do_filter(docs, condition)
                for s in docs:
                    self[table, s['identifier'], s['uuid']] = r

    def update(self, table, condition, key, value):
        """Update data in the database.

        :param table: The table to update.
        :param condition: The condition to update.
        :param key: The key to update.
        :param value: The value to update.
        """
        r_table = self._get_with_component_identifier('Table', table)

        if not r_table['is_component']:
            pid = self.primary_id(table)
            docs = self.get_many(table, condition[pid])
            docs = self._do_filter(docs, condition)
            for s in docs:
                s[key] = value
                self[table, s[pid]] = s
        else:
            if 'uuid' in condition:
                s = self.get_many(table, '*', condition['uuid'])[0]
                s[key] = value
                self[table, s['identifier'], condition['uuid']] = s
            elif 'identifier' in condition:
                docs = self.get_many(table, condition['identifier'], '*')
                docs = self._do_filter(docs, condition)
                for s in docs:
                    s[key] = value
                    self[table, s['identifier'], s['uuid']] = s
            else:
                docs = self.get_many(table, '*', '*')
                docs = self._do_filter(docs, condition)
                for s in docs:
                    s[key] = value
                    self[table, s['identifier'], s['uuid']] = s

    @abstractmethod
    def keys(self, *pattern) -> t.List[t.Tuple[str, str, str]]:
        """Get the keys from the cache.

        :param pattern: The pattern to match.

        >>> cache.keys('*', '*', '*')
        >>> cache.keys('*', '*')
        >>> cache.keys('Model', '*', '*')
        >>> cache.keys('my_table', '*')
        >>> cache.keys('Model', 'my_model', '*')
        >>> cache.keys('*', '*', '1234567890')
        """

    def _get_with_component(self, component: str):
        """Get all components from the cache of a certain type.

        :param component: The component to get.
        """
        keys = self.keys(component, '*', '*')
        return [self[k] for k in keys]

    def _get_all_with_component_identifier(self, component: str, identifier: str):
        """Get a component from the cache with a specific identifier.

        :param component: The component to get.
        :param identifier: The identifier of the component to
        """
        keys = self.keys(component, identifier, '*')
        out = [self[k] for k in keys]
        if not out:
            return []
        return out

    def _get_with_component_identifier(self, component: str, identifier: str):
        """Get a component from the cache with a specific identifier.

        :param component: The component to get.
        :param identifier: The identifier of the component to
        """
        keys = self.keys(component, identifier, '*')
        out = [self[k] for k in keys]
        if not out:
            return None

        out = max(out, key=lambda x: x['version'])  # type: ignore[arg-type,call-overload]
        return out

    def _get_with_component_identifier_version(
        self, component: str, identifier: str, version: int
    ):
        """Get a component from the cache with a specific version.

        :param component: The component to get.
        :param identifier: The identifier of the component to get.
        :param version: The version of the component to get.
        """
        keys = self.keys(component, identifier, '*')
        out = [self[k] for k in keys]
        try:
            return next(r for r in out if r['version'] == version)
        except StopIteration:
            return

    @abstractmethod
    def __delitem__(self, key: t.Tuple[str, str, str]):
        pass

    def insert(self, table, documents):
        """Insert data into the database.

        :param table: The table to insert into.
        :param documents: The documents to insert.
        """
        ids = []
        try:
            pid = self.primary_id(table)
        except exceptions.NotFound:
            pid = None

        if ('uuid' == pid or not pid) and "uuid" in documents[0]:
            for r in documents:
                self[table, r['identifier'], r['uuid']] = r
                ids.append(r['uuid'])
        elif pid:
            pid = self.primary_id(table)
            for r in documents:
                if pid not in r:
                    r[pid] = self.random_id()
                self[table, r[pid]] = r
                ids.append(r[pid])
        else:
            raise exceptions.NotFound("Table", table)
        return ids

    def select(self, query):
        """Select data from the database.

        :param query: The query to perform.
        """
        if query.decomposition.outputs:
            raise NotImplementedError(
                "KeyedDatabackend does not support outputs queries."
            )

        ops = {
            '==': lambda x, y: x == y,
            'in': lambda x, y: x in y,
            '!=': lambda x, y: x != y,
        }

        filter_kwargs = {}

        def do_test(r):
            return True

        if query.decomposition.filter:

            filters = query.decomposition.filter.args
            for f in filters:
                col, op = f.parts
                assert (
                    op.symbol in ops
                ), f"KeyedDatabackend only supports these filters {list(ops.keys())}."
                value = op.args[0]
                filter_kwargs[col] = {
                    'value': value,
                    'op': op.symbol,
                }

            def do_test(r):
                for k, v in filter_kwargs.items():
                    if k not in r:
                        return False

                    op = ops[v['op']]
                    v = v['value']

                    if not op(r[k], v):
                        return False
                return True

        tables = self.get_many('Table', query.table, '*')
        if not tables:
            raise exceptions.NotFound("Table", query.table)

        is_component = max(tables, key=lambda x: x['version'])['is_component']

        if not is_component:
            pid = self.primary_id(query.table)
            if pid in filter_kwargs:
                keys = self.keys(query.table, filter_kwargs[pid]['value'])
                del filter_kwargs[pid]
            else:
                keys = self.keys(query.table, '*')

            docs = [self[k] for k in keys]
            docs = [r for r in docs if do_test(r)]
        else:

            if not filter_kwargs:
                keys = self.keys(query.table, '*', '*')
                docs = [self[k] for k in keys]
            elif set(filter_kwargs.keys()) == {'uuid'}:
                keys = self.keys(query.table, '*', filter_kwargs['uuid']['value'])
                docs = [self[k] for k in keys]
            elif set(filter_kwargs.keys()) == {'identifier'}:
                assert filter_kwargs['identifier']['op'] == '=='

                keys = self.keys(query.table, filter_kwargs['identifier']['value'], '*')
                docs = [self[k] for k in keys]
            elif set(filter_kwargs.keys()) == {'identifier', 'uuid'}:
                assert filter_kwargs['identifier']['op'] == '=='
                assert filter_kwargs['uuid']['op'] == '=='

                r = self[
                    query.table,
                    filter_kwargs['identifier']['value'],
                    filter_kwargs['uuid']['value'],
                ]
                if r is None:
                    docs = []
                else:
                    docs = [r]
            elif set(filter_kwargs.keys()) == {'identifier', 'version'}:
                assert filter_kwargs['identifier']['op'] == '=='
                assert filter_kwargs['version']['op'] == '=='

                keys = self.keys(query.table, filter_kwargs['identifier']['value'], '*')
                docs = [self[k] for k in keys]
                docs = [
                    r for r in docs if r['version'] == filter_kwargs['version']['value']
                ]
            else:
                keys = self.keys(query.table, '*', '*')
                docs = [self[k] for k in keys]
                docs = [r for r in docs if do_test(r)]

        if filter_kwargs:
            docs = [r for r in docs if do_test(r)]

        if query.decomposition.select:
            cols = query.decomposition.select.args
            for i, r in enumerate(docs):
                docs[i] = {k: v for k, v in r.items() if k in cols}
        return docs
