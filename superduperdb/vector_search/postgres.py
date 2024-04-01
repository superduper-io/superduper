import json
import typing as t
import numpy
from pgvector.psycopg import psycopg, register_vector


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
        conninfo: str,
        h: t.Optional[numpy.ndarray] = None,
        index: t.Optional[t.List[str]] = None,
        measure: t.Optional[str] = None,
    ):
        self.connection = psycopg.connect(conninfo=conninfo)
        self.dimensions = dimensions
        self.identifier = identifier
        if measure == "l2" or not measure:
            self.measure_query = "embedding <-> '%s'"
        elif measure == "dot":
            self.measure_query = "(embedding <#> '%s') * -1"
        elif measure == "cosine":
            self.measure_query = "1 - (embedding <=> '%s')"
        else:
            raise NotImplementedError("Unrecognized measure format")
        with self.connection.cursor() as cursor:
            cursor.execute('CREATE EXTENSION IF NOT EXISTS vector')
            cursor.execute(
                'CREATE TABLE IF NOT EXISTS %s (id varchar, embedding vector(%d))'
                % (self.identifier, self.dimensions)
            )
        register_vector(self.connection)
        if h:
            self._create_or_append_to_dataset(h, index)


    def __len__(self):
        with self.connection.cursor() as curr:
            length = curr.execute(
                'SELECT COUNT(*) FROM %s' % self.identifier
            ).fetchone()[0]
        return length


    def _create_or_append_to_dataset(self, vectors, ids):
        with self.connection.cursor().copy(
            'COPY %s (id, embedding) FROM STDIN WITH (FORMAT BINARY)' % self.identifier
        ) as copy:
            copy.set_types(['varchar', 'vector'])
            for id_vector, vector in zip(ids, vectors):
                copy.write_row([id_vector, vector])
        self.connection.commit()


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
        with self.connection.cursor() as curr:
            for id_vector in ids:
                curr.execute(
                    "DELETE FROM %s WHERE id = '%s'" % (self.identifier, id_vector)
                )
        self.connection.commit()


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
        with self.connection.cursor() as curr:
            curr.execute(
                """
                SELECT embedding 
                FROM %s 
                WHERE id = '%s'"""
                % (self.identifier, _id)
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
        h = self.to_numpy(h)[None, :]
        if len(within_ids) == 0:
            condition = "1=1"
        else:
            within_ids_str = ', '.join([f"'{i}'" for i in within_ids])
            condition = f"id in ({within_ids_str})"
        query_search_nearest = f"""
        SELECT id, {self.measure_query} as distance
                FROM %s
                WHERE %s
                ORDER BY distance
                LIMIT %d
            """
        with self.connection.cursor() as curr:
            curr.execute(
                query_search_nearest % (json.dumps(h), self.identifier, condition, n)
            )
            nearest_items = curr.fetchall()
        ids = [row[0] for row in nearest_items]
        scores = [row[1] for row in nearest_items]
        return ids, scores
