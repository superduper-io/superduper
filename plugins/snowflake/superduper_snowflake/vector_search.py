import typing as t

import numpy as np
from superduper import CFG
from superduper.backends.base.vector_search import BaseVectorSearcher, VectorItem

from superduper_snowflake.connect import connect, watch_token_file

if t.TYPE_CHECKING:
    from superduper.components.vector_index import VectorIndex


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
        self.uri = CFG.data_backend
        assert self.uri, "Vector search URI is required"

        self.session, self.schema = connect(self.uri)
        self.observer = None
        if self.uri == 'snowflake://':
            self.observer = watch_token_file(self)

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

    def reconnect(self):
        """Reconnect to the data backend."""
        self.session, self.schema = connect(self.uri)

    def __len__(self):
        pass

    def is_initialized(self, identifier):
        """Check if vector index initialized."""
        return True

    def drop(self):
        """Drop the vector index."""
        pass

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

    def find_nearest_from_array(self, h, n=100, within_ids=None):
        """Find the nearest vectors to the given vector.

        :param h: vector
        :param n: number of nearest vectors to return
        :param within_ids: list of IDs to search within
        """
        from snowflake.snowpark.functions import col, lit, vector_l2_distance
        from snowflake.snowpark.types import VectorType

        if isinstance(h, np.ndarray):
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
