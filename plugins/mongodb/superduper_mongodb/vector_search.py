import typing as t
from functools import cached_property

from superduper import CFG, logging
from superduper.backends.base.vector_search import BaseVectorSearcher, VectorItem

if t.TYPE_CHECKING:
    from superduper.components.vector_index import VectorIndex


class MongoAtlasVectorSearcher(BaseVectorSearcher):
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
        dimensions: int,
        measure: t.Optional[str] = None,
        output_path: t.Optional[str] = None,
    ):
        import pymongo

        self.identifier = identifier
        vector_search_uri = CFG.cluster.vector_search.uri or CFG.data_backend
        assert vector_search_uri, "Vector search URI is required"
        db_name = vector_search_uri.split("/")[-1]
        self.database = getattr(pymongo.MongoClient(vector_search_uri), db_name)
        assert output_path
        self.output_path = output_path
        self.collection = collection
        self.measure = measure
        self.dimensions = dimensions
        self._is_exists = False

        self._check_if_exists(create=True)
        super().__init__(identifier=identifier, dimensions=dimensions, measure=measure)

    def __len__(self):
        pass

    @cached_property
    def index(self):
        """Return the index collection."""
        return self.database[self.collection]

    def is_initialized(self, identifier):
        """Check if vector index initialized."""
        return self._check_if_exists(create=False)

    @classmethod
    def from_component(cls, vi: "VectorIndex"):
        """Create a vector searcher from a vector index.

        :param vi: VectorIndex instance
        """
        from superduper.components.listener import Listener

        assert isinstance(vi.indexing_listener, Listener)
        assert vi.indexing_listener.select is not None
        path = collection = vi.indexing_listener.outputs

        return MongoAtlasVectorSearcher(
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
        self._check_if_exists(create=True)

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
        h = self.index.find_one({"_id": id})[self.output_path]
        return self.find_nearest_from_array(h, n=n, within_ids=within_ids)

    def find_nearest_from_array(self, h, n=100, within_ids=None):
        """Find the nearest vectors to the given vector.

        :param h: vector
        :param n: number of nearest vectors to return
        :param within_ids: list of IDs to search within
        """
        self._check_if_exists(create=True)
        self._check_queryable()
        vector_search = {
            "index": self.identifier,
            "path": self.output_path,
            "queryVector": h,
            "numCandidates": n,
            "limit": n,
        }
        if within_ids:
            vector_search["filter"] = {"_id": {"$in": within_ids}}

        project = {
            "_id": 1,
            "_source": 1,
            "score": {"$meta": "vectorSearchScore"},
        }

        pipeline = [
            {"$vectorSearch": vector_search},
            {"$project": project},
        ]

        cursor = self.index.aggregate(pipeline)
        scores = []
        ids = []
        for vector in cursor:
            scores.append(vector["score"])
            ids.append(str(vector["_source"]))
        return ids, scores

    def _create_index(self):
        """Create a vector index in the data backend if an Atlas deployment."""
        if self.collection not in self.database.list_collection_names():
            logging.warn(
                f"Collection {self.collection} does not exist. " "Cannot create index."
            )
            return
        from pymongo.operations import SearchIndexModel

        definition = {
            "fields": [
                {
                    "type": "vector",
                    "numDimensions": self.dimensions,
                    "path": self.output_path,
                    "similarity": self.measure,
                },
            ]
        }

        search_index_model = SearchIndexModel(
            definition=definition,
            name=self.identifier,
            type="vectorSearch",
        )
        logging.info(
            f"Creating search index [{self.identifier}] on {self.collection} "
            f"-- Definition: {definition}"
        )
        result = self.index.create_search_index(model=search_index_model)
        return result

    def _check_if_exists(self, create=True):
        if self._is_exists:
            return True
        index = self._get_index()
        if bool(index):
            self._is_exists = True
        elif create:
            self._create_index()
        return self._is_exists

    def _check_queryable(self):
        index = self._get_index()
        if not index:
            raise FileNotFoundError(
                f"Index {self.identifier} does not exist in the collection "
                f"{self.collection}. Cannot perform query."
            )

        if not index.get("queryable"):
            raise FileNotFoundError(
                f"Index {self.identifier} is pending and not yet queryable. "
                "Please wait until the index is fully ready for queries."
                f"Cannot perform query. "
                f"You need to wait for the index to be queryable. "
                f"Index: {index}"
            )
        return True

    def _get_index(self):
        try:
            indexes = self.index.list_search_indexes()
        except Exception:
            return False

        indexes = [i for i in indexes if i["name"] == self.identifier]

        if not indexes:
            return None
        else:
            return indexes[0]
