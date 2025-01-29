import functools
import typing as t
from abc import ABC, abstractmethod

from superduper import CFG, logging
from superduper.backends.base.query import Query
from superduper.base.constant import KEY_BLOBS, KEY_BUILDS, KEY_FILES, KEY_SCHEMA
from superduper.base.document import Document

if t.TYPE_CHECKING:
    from superduper.components.schema import Schema


class BaseDataBackend(ABC):
    """Base data backend for the database.

    :param uri: URI to the databackend database.
    :param plugin: Plugin implementing the databackend.
    :param flavour: Flavour of the databackend.
    """

    id_field: str = 'id'

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

    @abstractmethod
    def drop_table(self, table: str):
        """Drop data from table.

        :param table: The table to drop.
        """

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
    def build_metadata(self):
        """Build a default metadata store based on current connection."""
        pass

    @abstractmethod
    def build_artifact_store(self):
        """Build a default artifact store based on current connection."""
        pass

    @abstractmethod
    def create_table_and_schema(self, identifier: str, schema: "Schema"):
        """Create a schema in the data-backend.

        :param identifier: The identifier of the table.
        :param schema: The schema to create.
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
    def missing_outputs(self, query: Query, predict_id: str) -> t.List[str]:
        """Get missing outputs from an outputs query.

        This method will be used to perform an anti-join between
        the input and the outputs table, and return the missing ids.

        :param query: The query to perform.
        :param predict_id: The predict id.
        """

    @abstractmethod
    def primary_id(self, query: Query) -> str:
        """Get the primary id of a query.

        :param query: The query to get the primary id of.
        """

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

    def get(self, query: Query):
        """Get a single result from a query.

        :param query: The query to perform.
        """
        assert query.type == 'select'

        if query.decomposition.pre_like:
            return list(self.pre_like(query, n=1))[0]

        elif query.decomposition.post_like:
            return list(self.post_like(query, n=1))[0]

        return query.limit(1).execute()[0]

    def _wrap_results(self, query: Query, result, schema):
        pid = self.primary_id(query)
        for r in result:
            if pid in r:
                r[pid] = str(r[pid])
            if '_source' in r:
                r['_source'] = str(r['_source'])
        return [Document.decode(r, schema=schema, db=self.db) for r in result]

    def execute(self, query: Query):
        """Execute a query.

        :param query: The query to execute.
        """
        query = query if '.outputs' not in str(query) else query.complete_uuids(self.db)

        schema = self.get_schema(query)

        if query.decomposition.pre_like:
            return self._wrap_results(query, self.pre_like(query), schema=schema)

        if query.decomposition.post_like:
            return self._wrap_results(query, self.post_like(query), schema=schema)

        return self._wrap_results(query, self.select(query), schema=schema)

    def get_schema(self, query) -> 'Schema':
        """Get the schema of a query.

        :param query: The query to get the schema of.
        """
        base_schema = self.db.load('table', query.table).schema

        if query.decomposition.outputs:
            for predict_id in query.decomposition.outputs.args:
                base_schema += self.db.load(
                    'table', f'{CFG.output_prefix}{predict_id}'
                ).schema

        return base_schema

    def _do_insert(self, table, documents):
        schema = self.get_schema(self.db[table])

        if not schema.trivial:
            for i, r in enumerate(documents):
                r = Document(r).encode(schema=self.get_schema(self.db[table]))
                self.db.artifact_store.save_artifact(r)
                r.pop(KEY_BUILDS)
                r.pop(KEY_BLOBS)
                r.pop(KEY_FILES)
                r.pop(KEY_SCHEMA, None)
                documents[i] = r

        out = self.insert(table, documents)
        return [str(x) for x in out]

    def pre_like(self, query: Query):
        """Perform a pre-like query.

        :param query: The query to perform.
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

        results = new.execute()

        pid = self.primary_id(query)
        for r in results:
            r['score'] = lookup[r[pid]]

        results = sorted(results, key=lambda x: x['score'], reverse=True)
        return results

    def post_like(self, query: Query):
        """Perform a post-like query.

        :param query: The query to perform.
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

        results = prepare_query.filter(t.primary_id.isin(ids)).execute()

        pid = self.primary_id(query)

        for r in results:
            r['score'] = lookup[r[pid]]

        results = sorted(results, key=lambda x: x['score'], reverse=True)
        return results


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

    @abstractmethod
    def execute_native(self, query: str):
        """Execute a native query provided as a str.

        :param query: The query to execute.
        """
        pass

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
