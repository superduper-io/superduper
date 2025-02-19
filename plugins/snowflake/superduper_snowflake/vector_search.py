from functools import wraps
import os
import re
import typing as t

from snowflake.snowpark import Session
from superduper import CFG
from superduper.backends.base.vector_search import BaseVectorSearcher, VectorItem

if t.TYPE_CHECKING:
    from superduper.components.vector_index import VectorIndex


def retry(f):
    @wraps(f)
    def wrapper(self, *args, **kwargs):
        try:
            return f(self, *args, **kwargs)
        except Exception as e:
            if 'token' in str(e):
                self.session = SnowflakeVectorSearcher.create_session(CFG.data_backend)
            return f(self, *args, **kwargs)

    return wrapper


class SnowflakeVectorSearcher(BaseVectorSearcher):
    """Vector searcher implementation of atlas vector search.

    :param identifier: Unique string identifier of index
    :param collection: Collection name
    :param dimensions: Dimension of the vector embeddings
    :param measure: measure to assess similarity
    :param output_path: Path to the output
    """

    native_service: t.ClassVar[bool] = False

    def __init__(
        self,
        identifier: str,
        collection: str,
        dimensions: t.Optional[int],
        measure: t.Optional[str] = None,
        output_path: t.Optional[str] = None,
    ):
        self.identifier = identifier
        vector_search_uri = CFG.data_backend
        assert vector_search_uri, "Vector search URI is required"

        self.session = SnowflakeVectorSearcher.create_session(vector_search_uri)

        assert output_path
        self.output_path = output_path
        self.collection = collection
        if measure != 'l2':
            raise TypeError(
                "NOTE: Only L2 similary function is supported by ",
                "snowflake vector search",
            )

        self.measure = measure
        self.dimensions = dimensions
        self._cache = {}
        self._db = None

    @classmethod
    def create_session(cls, vector_search_uri):
        """Creates a snowflake session.

        :param vector_search_uri: Connection URI.
        """
        if vector_search_uri == 'snowflake://':
            host = os.environ['SNOWFLAKE_HOST']
            port = int(os.environ['SNOWFLAKE_PORT'])
            account = os.environ['SNOWFLAKE_ACCOUNT']
            token = open('/snowflake/session/token').read()
            warehouse = os.environ['SNOWFLAKE_WAREHOUSE']
            database = os.environ['SNOWFLAKE_DATABASE']
            schema = os.environ['SUPERDUPER_DATA_SCHEMA']

            connection_parameters = {
                "token": token,
                "account": account,
                "database": database,
                "schema": schema,
                "warehouse": warehouse,
                "authenticator": "oauth",
                "port": port,
                "host": host,
            }
        else:
            if '?warehouse=' not in vector_search_uri:
                match = re.match(
                    '^snowflake:\/\/(.*):(.*)\@(.*)\/(.*)\/(.*)$', vector_search_uri
                )
                user, password, account, database, schema = match.groups()
                warehouse = None
            else:
                match = re.match(
                    '^snowflake://(.*):(.*)@(.*)/(.*)/(.*)?warehouse=(.*)$',
                    vector_search_uri,
                )
                user, password, account, database, schema, warehouse = match.groups()
            if match:
                connection_parameters = {
                    "user": user,
                    "password": password,
                    "account": account,
                    "database": database,
                    "schema": schema,
                    "warehouse": warehouse,
                }
            else:
                raise ValueError(f"URI `{vector_search_uri}` is invalid!")

        session = Session.builder.configs(connection_parameters).create()
        return session

    def __len__(self):
        pass

    def is_initialized(self, identifier):
        """Check if vector index initialized."""
        return True

    @classmethod
    def from_component(cls, vi: "VectorIndex"):
        """Create a vector searcher from a vector index.

        :param vi: VectorIndex instance
        """
        from superduper.components.listener import Listener

        assert isinstance(vi.indexing_listener, Listener)
        assert vi.indexing_listener.select is not None
        path = collection = vi.indexing_listener.outputs

        return SnowflakeVectorSearcher(
            identifier=vi.identifier,
            dimensions=vi.dimensions,
            measure=vi.measure,
            output_path=path,
            collection=collection,
        )

    def add(self, items: t.Sequence[VectorItem], cache: bool = False) -> None:
        """
        Add items to the index.

        :param items: t.Sequence of VectorItems
        """
        # NOTE: Since we will be doing vector search on tables directly
        # seperate vector search is not required.

    def delete(self, ids: t.Sequence[str]) -> None:
        """Remove items from the index.

        :param ids: t.Sequence of ids of vectors.
        """

    def find_nearest_from_id(self, id: str, n=100, within_ids=None):
        """Find the nearest vectors to the given ID.

        :param id: ID of the vector
        :param n: number of nearest vectors to return
        :param within_ids: list of IDs to search within
        """
        result = self.session.sql(
            f"""
        SELECT "{self.output_path}"
        FROM "{self.collection}"
        WHERE id = '{id}'
        LIMIT 1
        """
        ).collect()
        return self.find_nearest_from_array(result, n=n, within_ids=within_ids)

    @retry
    def find_nearest_from_array(self, h, n=100, within_ids=None):
        """Find the nearest vectors to the given vector.

        :param h: vector
        :param n: number of nearest vectors to return
        :param within_ids: list of IDs to search within
        """
        from snowflake.snowpark.functions import col, lit, vector_l2_distance
        from snowflake.snowpark.types import VectorType

        if hasattr(h, 'tolist'):
            h = h.tolist()

        vector_table = self.session.table(f'"{self.collection}"')
        result_df = (
            vector_table.select(
                '"_source"',
                f'"{self.output_path}"',
                vector_l2_distance(
                    col(f'"{self.output_path}"').cast(
                        VectorType(float, self.dimensions)
                    ),
                    lit(h).cast(VectorType(float, self.dimensions)),
                ).as_("distance"),
            )
            .sort("distance")
            .limit(n)
        )
        result_list = result_df.collect()
        ids = [row["_source"] for row in result_list]
        scores = [-row["distance".upper()] for row in result_list]
        return ids, scores

    def initialize(self):
        """Initialize vector search."""
