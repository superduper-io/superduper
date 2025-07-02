import numpy
import typing as t

import psycopg2
from pgvector.psycopg2 import register_vector
from pgvector import Vector          # <- new import

from superduper.backends.base.vector_search import BaseVectorSearcher, VectorItem
from superduper import VectorIndex


class PGVectorSearcher(BaseVectorSearcher):

    def __init__(
        self,
        table: str,
        vector_column: str,
        primary_id: str,
        dimensions: int,
        measure: str,
        uri: str,
    ):
        self.conn = psycopg2.connect(uri)
        register_vector(self.conn)
        self.table = table
        self.vector_column = vector_column
        self.dimensions = dimensions
        self.measure = measure
        self.primary_id = primary_id

    def drop(self):
        """Drop the vector index."""
        self.conn.close()

    def initialize(self):
        """Initialize the vector-searcher.

        :param db: ``Datalayer`` instance.
        """
        cur = self.conn.cursor()
        # check that this is a vector table
        try:
            cur.execute(f"""
                SELECT *
                FROM information_schema.columns
                WHERE table_name = '{self.table}' AND column_name = '{self.vector_column}'
            """)
        except Exception as e:
            self.conn.rollback()
            raise e
        self.conn.commit()
        if not cur.fetchone():
            raise ValueError(f"Table {self.table} is not a vector table")

    def add(self, items: t.Sequence['VectorItem']) -> None:
        """
        Add items to the index.

        :param items: t.Sequence of VectorItems
        """
        return

    def delete(self, ids: t.Sequence[str]) -> None:
        """Remove items from the index.

        :param ids: t.Sequence of ids of vectors.
        """
        return

    def find_nearest_from_array(
        self,
        h: numpy.typing.ArrayLike,
        n: int = 100,
        within_ids: t.Sequence[str] = (),
    ) -> t.Tuple[t.List[str], t.List[float]]:
        """
        Find the nearest vectors to the given vector.

        :param h: vector
        :param n: number of nearest vectors to return
        :param within_ids: list of ids to search within
        """
        # use pg_vector to find nearest vectors
        cur = self.conn.cursor()
        operator = {
            'l2': '<->',
            'css': '<=>',
            'cosine': '<=>',
            'dot': '<#>',
        }[self.measure]

        if within_ids:
            query = f"""
                SELECT id, {self.vector_column} {operator} %s AS score
                FROM {self.table}
                WHERE {self.primary_id} = ANY(%s)
                ORDER BY score
                LIMIT %s
            """
        else:
            query = f"""
                SELECT id, {self.vector_column} {operator} %s AS score
                FROM {self.table}
                ORDER BY score
                LIMIT %s
            """
        with self.conn.cursor() as cur:
            if isinstance(h, numpy.ndarray):
                h = h.tolist()
            try:
                if within_ids:
                    cur.execute(query, (Vector(h), within_ids, n))
                else:
                    cur.execute(query, (Vector(h), n))
                results = cur.fetchall()
            except Exception as e:
                self.conn.rollback()
                raise e
        self.conn.commit()
        return [r[0] for r in results], [1 - r[1] for r in results]

    def find_nearest_from_id(
        self,
        id: str,
        n: int = 100,
        within_ids: t.Sequence[str] = (),
    ) -> t.Tuple[t.List[str], t.List[float]]:
        """
        Find the nearest vectors to the given vector.

        :param id: id of the vector to search with
        :param n: number of nearest vectors to return
        :param within_ids: list of ids to search within
        """

    def __len__(self):
        with self.conn.cursor() as cur:
            result = cur.execute(f"SELECT COUNT(*) FROM {self.table}")

        return result.fetchone()[0] if result else 0

    @classmethod
    def from_component(cls, vi: VectorIndex):
        """Create a PGVectorSearcher from component and vector index."""
        output_table = vi.db.load(vi.indexing_listener.outputs)
        pid = output_table.primary_id
        from superduper import CFG
        return cls(
            table=vi.indexing_listener.outputs,
            vector_column=vi.indexing_listener.key,
            primary_id=pid,
            dimensions=vi.dimensions,
            measure=vi.measure,
            uri=CFG.data_backend,
        )