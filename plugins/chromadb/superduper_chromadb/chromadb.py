import re
import typing as t

import chromadb
import numpy as np
from superduper import CFG, logging
from superduper.backends.base.vector_search import (
    BaseVectorSearcher,
    VectorItem,
)

ID_PAYLOAD_KEY = "_id"


class ChromaDBVectorSearcher(BaseVectorSearcher):
    """
    Implementation of a vector index using [Qdrant](https://qdrant.tech/).

    :param identifier: Unique string identifier of index
    :param dimensions: Dimension of the vector embeddings
    :param h: Seed vectors ``numpy.ndarray``
    :param index: list of IDs
    :param measure: measure to assess similarity
    :param batch_size: Number of vectors to upsert in a single batch (default: 512)
    """

    def __init__(
        self,
        identifier: str,
        dimensions: int,
        measure: t.Optional[str] = None,
        component: str = 'VectorIndex',
        batch_size: int = 512,
    ):
        try:
            plugin, uri = CFG.vector_search_engine.split("://")
            port = int(CFG.vector_search_engine.split(":")[-1])
            assert plugin == "chromadb"
            assert uri.startswith('localhost'), 'ChromaDB only supported on localhost'

        except ValueError as e:
            if 'not enough values to unpack' in str(e):
                plugin = CFG.vector_search_engine
            else:
                raise e

        self.client = chromadb.HttpClient(host="localhost", port=port)

        self.identifier = identifier
        self.measure = measure
        self.batch_size = batch_size
        self.identifier = re.sub("\W+", "", identifier)

        self.collection = self.client.get_or_create_collection(
            name=self.identifier,
            metadata={"hnsw:space": self._distance_mapping(self.measure)},
            embedding_function=None,  # we'll supply vectors manually
        )

        self.component = component

    def initialize(self):
        """Initialize the vector index.

        :param db: Datalayer instance
        """
        pass

    def __len__(self):
        return self.collection.count()

    def add(self, items: t.Sequence[VectorItem], cache: bool = False) -> None:
        """Add vectors to the index.

        :param items: List of vectors to add
        :param cache: Cache vectors (not used in Qdrant implementation).
        """
        if not items:
            return

        vectors = []
        ids = []
        for item in items:
            if hasattr(item.vector, "tolist"):
                vector = item.vector.tolist()
            else:
                vector = item.vector
            vectors.append(vector)
            ids.append(item.id)

        total_batches = (len(vectors) + self.batch_size - 1) // self.batch_size

        logging.info(
            f"Adding {len(vectors)} points to ChromaDB index '{self.identifier}' "
            f"in {total_batches} batches (batch_size={self.batch_size})"
        )

        for batch_idx, i in enumerate(range(0, len(vectors), self.batch_size), 1):
            sub_vectors = vectors[i : i + self.batch_size]
            sub_ids = ids[i : i + self.batch_size]

            logging.info(f"Processing batch {batch_idx}/{total_batches} ")

            self.collection.add(
                ids=sub_ids,
                embeddings=sub_vectors,
            )

        logging.info(
            f"✓ Successfully added all {len(vectors)} points to Qdrant index "
            f"'{self.identifier}' in {total_batches} batches"
        )

    def drop(self):
        """Drop the vector index."""
        try:
            self.client.delete_collection(self.identifier)
        except Exception:
            pass

    def delete(self, ids: t.Sequence[str]) -> None:
        """Delete vectors from the Chroma collection by ID."""
        self.collection.delete(ids=list(ids))

    def find_nearest_from_id(
        self,
        _id,
        n: int = 100,
        within_ids: t.Sequence[str] = (),
    ) -> t.Tuple[t.List[str], t.List[float]]:
        """Find the nearest vectors to a given ID.

        :param _id: ID to search
        :param n: Number of results to return
        :param within_ids: List of IDs to search within
        """
        got = self.collection.get(ids=[_id], include=["embeddings", "ids"])
        if not got or not got.get("ids"):
            raise ValueError(f"id not found: {_id}")
        h = got["embeddings"][0]  # the stored embedding for that id
        return self.find_nearest_from_array(h=h, n=n, within_ids=within_ids)

    def find_nearest_from_array(
        self,
        h: np.typing.ArrayLike,
        n: int = 100,
        within_ids: t.Sequence[str] = (),
    ) -> t.Tuple[t.List[str], t.List[float]]:
        """Find the nearest vectors to a given vector.

        :param h: Vector to search
        :param n: Number of results to return
        :param within_ids: List of IDs to search within
        """
        if isinstance(h, np.ndarray):
            h = h.tolist()
        else:
            assert isinstance(h, list), "Input vector must be a list or numpy array"

        res = self.collection.query(
            query_embeddings=[h], n_results=n, include=["distances"]
        )

        ids = res.get("ids", [[]])[0]
        dists = res.get("distances", [[]])[0]

        if within_ids:
            logging.warning(f"Searching within specific IDs: {within_ids}")

            ix = [i for i, id in enumerate(ids) if id in within_ids]
            ids = [ids[i] for i in ix]
            dists = [dists[i] for i in ix]
        return ids, dists

    def _distance_mapping(self, measure: t.Optional[str] = None):
        if measure == "cosine":
            return 'cosine'
        if measure == "l2":
            return 'l2'
        if measure == "dot":
            return 'ip'
        else:
            raise ValueError(f"Unsupported measure: {measure}")
