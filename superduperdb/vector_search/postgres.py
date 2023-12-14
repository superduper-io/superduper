import os
import typing as t

from pgvector.psycopg2 import psycopg2, register_vector
import numpy
import pyarrow as pa

from superduperdb import CFG
from superduperdb.vector_search.base import BaseVectorSearcher, VectorItem


class PostgresVectorSearcher(BaseVectorSearcher):
    """
    Implementation of a vector index using the ``pgvector`` library.

    :param identifier: Unique string identifier of index
    :param dimensions: Dimension of the vector embeddings in the Lance dataset
    :param uri: connection string to postgres
    :param h: ``torch.Tensor``
    :param index: list of IDs
    :param measure: measure to assess similarity
    """

    def __init__(
        self,
        identifier: str,
        dimensions: int,
        uri: str,
        h: t.Optional[numpy.ndarray] = None,
        index: t.Optional[t.List[str]] = None,
        measure: t.Optional[str] = None,
    ):
        self.engine = psycopg2.connect(dsn=uri)
        self.dimensions = dimensions
        self.identifier = identifier
        self.measure = measure
        with self.engine.connect() as conn:
            register_vector(conn)
            cursor = conn.cursor()
            cursor.execute('CREATE EXTENSION IF NOT EXISTS vector')
            cursor.execute('CREATE TABLE %s (id varchar, embedding vector(%d)' % (self.identifier, self.dimensions))
        if h:
            self._create_or_append_to_dataset(h, index)


    def __len__(self):
        with self.engine.connect().cursor() as curr:
            length = curr.execute('SELECT COUNT(*) FROM %s' % self.identifier).fetchone()[0]
        return length


    def _create_or_append_to_dataset(self, vectors, ids):
        with self.engine.connect().cursor().copy('COPY %s (id, embedding) PRIMARY KEY id FROM STDIN WITH (FORMAT BINARY)' % self.identifier) as copy:
            for id_vector, vector in zip(ids, vectors):
                copy.write_row([id_vector, vector])
            copy.commit()


    def add(self, items: t.Sequence[VectorItem]) -> None:
        """
        Add items to the index.

        :param items: t.Sequence of VectorItems
        """
        ids = [item.id for item in items]
        vectors = [item.vector for item in items]
        self._create_or_append_to_dataset(vectors, ids)


    def delete(self, ids: t.Sequence[str]) -> None:
        """
        Remove items from the index

        :param ids: t.Sequence of ids of vectors.
        """
        with self.engine.connect().cursor() as curr:
            for id_vector in ids:
                curr.execute('DELETE FROM %s WHERE id = %d' % (self.identifier, id_vector))
            curr.commit()
    

    def find_nearest_from_id(
        self,
        _id,
        n: int = 100,
        within_ids: t.Sequence[str] = (),
    ) -> t.Tuple[t.List[str], t.List[float]]:
        """
        Find the nearest vectors to the vector with the given id.

        :param _id: id of the vector
        :param n: number of nearest vectors to return
        """
        with self.engine.connect().cursor() as curr:
            curr.execute("""
                SELECT embedding 
                FROM %s 
                WHERE id = %s""" % (self.identifier, _id)
            )
            h = curr.fetchone()[0]
        return self.find_nearest_from_array(h, n, within_ids)

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
        """
        if len(within_ids) == 0:
            condition = "1=1"
        else:
            within_ids_str = ', '.join([f"'{i}'" for i in within_ids])
            condition = f"id in ({within_ids_str})"
        with self.engine.connect().cursor() as curr:
            curr.execute("""
                SELECT id,  1 - (embedding <=> '%s') as cosine_similarity
                FROM %s
                WHERE %s
                ORDER BY cosine_similarity
                LIMIT %d
                """ % (h, self.identifier, condition, n)
            )
            nearest_items = curr.fetchall()
        ids = [row[0] for row in nearest_items]
        scores = [row[1] for row in nearest_items]
        return ids, scores