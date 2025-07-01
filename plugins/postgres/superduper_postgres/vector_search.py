import numpy
import typing as t

import psycopg2
from pgvector.psycopg2 import register_vector

from superduper.backends.base.vector_search import BaseVectorSearcher, VectorItem


class PGVectorSearcher(BaseVectorSearcher):

    def __init__(
        self,
        identifier: str,
        dimensions: int,
        measure: str,
        component: str = 'VectorIndex',
    ):
        self.conn = psycopg2.connect(
            dbname="your_db",
            user="your_user",
            password="your_password",
            host="localhost",
            port="5432"
        )
        register_vector(self.conn)
        self.identifier = identifier
        self.dimensions = dimensions
        self.measure = measure
        self.component = component

    def drop(self):
        """Drop the vector index."""
        pass

    def initialize(self, db):
        """Initialize the vector-searcher.

        :param db: ``Datalayer`` instance.
        """
        cur = self.conn.cursor()
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.identifier} (
                id SERIAL PRIMARY KEY,
                embedding VECTOR({self.dimensions})
            )
        """)

        # Insert a vector
        cur.execute("INSERT INTO items (embedding) VALUES (%s)", ([0.1, 0.2, 0.3],))

        # Find the most similar vector (L2 distance)
        cur.execute("""
            SELECT id, embedding
            FROM items
            ORDER BY embedding <-> %s
            LIMIT 1
        """, ([0.1, 0.2, 0.3],))

        pass

    def add(self, items: t.Sequence['VectorItem']) -> None:
        """
        Add items to the index.

        :param items: t.Sequence of VectorItems
        """

    def delete(self, ids: t.Sequence[str]) -> None:
        """Remove items from the index.

        :param ids: t.Sequence of ids of vectors.
        """

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