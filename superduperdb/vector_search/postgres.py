import json
import typing as t
import numpy
import psycopg2

from superduperdb import CFG, logging
if t.TYPE_CHECKING:
    from superduperdb.components.vector_index import VectorIndex
from superduperdb.components.model import APIModel, Model



from superduperdb.vector_search.base import BaseVectorSearcher, VectorItem, VectorIndexMeasureType

class PostgresIndexing:
    cosine = "vector_cosine_ops"
    l2 = "vector_l2_ops"
    inner_product = "vector_ip_ops"

    
class IVFFlat(PostgresIndexing):
    """
    An IVFFlat index divides vectors into lists, and then searches a subset of those lists that are closest to the query vector. 
    It has faster build times and uses less memory than HNSW, but has lower query performance (in terms of speed-recall tradeoff).

    :param lists 
    :param probes
    """
    def __init__(self, lists: t.Optional[int] = 100, probes: t.Optional[int] = 1):
        self.name = "ivfflat"
        self.lists = lists
        self.probes = probes

class HNSW(PostgresIndexing):
    """
    An HNSW index creates a multilayer graph. It has better query performance than IVFFlat (in terms of speed-recall tradeoff), 
    but has slower build times and uses more memory. Also, an index can be created without any data in the table 
    since there isn’t a training step like IVFFlat.

    :param m: the max number of connections per layer 
    :param ef_construction: the size of the dynamic candidate list for constructing the graph
    """
    def __init__(self, m: t.Optional[int] = 16, ef_construction: t.Optional[int] = 64, ef_search: t.Optional[int] = 40):
        self.name = "hnsw"
        self.m = m
        self.ef_construction = ef_construction
        self.ef_search: ef_search = ef_search


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
        measure: t.Optional[str] = VectorIndexMeasureType.cosine,
        indexing : t.Optional[HNSW | IVFFlat] = None,
        indexing_measure : t.Optional[PostgresIndexing] = PostgresIndexing.cosine
    ):
        self.connection = psycopg2.connect(uri)
        self.dimensions = dimensions
        self.identifier = identifier
        self.measure = measure 
        self.measure_query = self.get_measure_query()
        self.indexing = indexing
        self.indexing_measure = indexing_measure
        with self.connection.cursor() as cursor:
            cursor.execute('CREATE EXTENSION IF NOT EXISTS vector')
            cursor.execute(
                'CREATE TABLE IF NOT EXISTS "%s" (id varchar, txt VARCHAR, embedding vector(%d))'
                % (self.identifier, self.dimensions)
            )
        self.connection.commit()
        if h:
            self._create_or_append_to_dataset(h, index)


    def __len__(self):
        with self.connection.cursor() as curr:
            length = curr.execute(
                'SELECT COUNT(*) FROM %s' % self.identifier
            ).fetchone()[0]
        return length
    
    def get_measure_query(self):
        if self.measure.value == "l2":
            return "embedding <-> '%s'"
        elif self.measure.value == "dot":
            return "(embedding <#> '%s') * -1"
        elif self.measure.value == "cosine":
            return "1 - (embedding <=> '%s')"
        else:
            raise NotImplementedError("Unrecognized measure format")


    def _create_or_append_to_dataset(self, vectors, ids):
        with self.connection.cursor() as cursor:
            for id_, vector in zip(ids, vectors):
                try:
                    cursor.execute(
                        "INSERT INTO %s (id, embedding) VALUES (%s, '%s');" % (self.identifier, id_, vector)
                    )
                except Exception as e:
                    pass
        self.connection.commit()

    def _create_index(self):
        with self.connection.cursor() as cursor:
            if self.indexing.name == 'hnsw':
                cursor.execute("""CREATE INDEX ON %s
                                USING %s (embedding %s)
                                WITH (m = %s, ef_construction = %s);""" % (self.identifier, self.indexing.name, self.indexing_measure, self.indexing.m, self.indexing.ef_construction))
                
                cursor.execute("""SET %s.ef_search = %s;""" % (self.indexing.name, self.indexing.ef_search))
            elif self.indexing.name == 'ivfflat':
                cursor.execute("""CREATE INDEX ON %s
                                USING %s (embedding %s)
                                WITH (lists = %s);""" % (self.identifier, self.indexing.name, self.indexing_measure, self.indexing.lists))
                
                cursor.execute("""SET %s.probes = %s;""" % (self.indexing.name, self.indexing.probes))

        self.connection.commit()


    def add(self, items: t.Sequence[VectorItem]) -> None:
        """
        Add items to the index.
        :param items: t.Sequence of VectorItems
        """
        ids = [item.id for item in items]
        vectors = [item.vector for item in items]
        self._create_or_append_to_dataset(vectors, ids)

        if self.indexing:
            self._create_index()


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
        # h = self.to_numpy(h)[None, :]
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

    @classmethod
    def from_component(cls, vi: 'VectorIndex'):
        from superduperdb.components.listener import Listener
        from superduperdb.components.model import ObjectModel

        assert isinstance(vi.indexing_listener, Listener)
        collection = vi.indexing_listener.select.table_or_collection.identifier


        indexing_key = vi.indexing_listener.key

        assert isinstance(
            indexing_key, str
        ), 'Only single key is support for atlas search'
        if indexing_key.startswith('_outputs'):
            indexing_key = indexing_key.split('.')[1]
        assert isinstance(vi.indexing_listener.model, Model) or isinstance(
            vi.indexing_listener.model, APIModel
        )
        assert isinstance(collection, str), 'Collection is required to be a string'
        indexing_model = vi.indexing_listener.model.identifier

        indexing_version = vi.indexing_listener.model.version

        output_path = f'_outputs.{vi.indexing_listener.predict_id}'

        return PostgresVectorSearcher(
            uri=CFG.data_backend,
            identifier=output_path,
            dimensions=vi.dimensions,
            measure=VectorIndexMeasureType.cosine,
        )