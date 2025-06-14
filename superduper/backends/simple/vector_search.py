import traceback
import typing as t

import numpy
import requests
from pydantic import BaseModel

from superduper import logging
from superduper.backends.base.vector_search import VectorSearchBackend


class VectorItem(BaseModel):
    """A vector item model for storing vectors with their IDs."""

    id: str
    vector: t.List

    def __post_init__(self):
        """Convert vector to list if it's a numpy array."""
        if hasattr(self.vector, "tolist"):
            self.vector = self.vector.tolist()


class SimpleVectorSearchClient(VectorSearchBackend):
    """Client for interacting with the vector search service.

    Inherits from both Client and VectorSearchBackend to provide vector search
    functionality with a REST API interface.
    """

    def __init__(self):
        self.uri = 'http://localhost:8001/'

    def add(self, uuid: str, vectors: t.List['VectorItem']):
        """Add vectors to a vector index.

        :param uuid: Identifier of index
        :param vectors: Vectors to add
        :return: Response from the add operation
        """
        vectors = [{'id': x.id, 'vector': x.vector.tolist()} for x in vectors]
        response = requests.post(
            f'{self.uri}/vector_search/add', json={'uuid': uuid, 'vectors': vectors}
        )

        if response.status_code != 200:
            raise Exception(f"Failed to add vectors: {response.text}")
        return response.json()

    def initialize(self):
        """Initialize the vector search service.

        This method is a placeholder for any initialization logic needed.
        """
        return requests.post(f'{self.uri}/vector_search/initialize')

    def delete(self, uuid, ids):
        """Delete ids from index.

        :param uuid: Identifier of index
        :param ids: Ids to delete
        :return: Response from the delete operation
        """
        response = requests.post(
            f'{self.uri}/vector_search/delete',
            json={'uuid': uuid, 'ids': ids},
        )
        if response.status_code != 200:
            raise Exception(f"Failed to delete vectors: {response.text}")
        return response.json()

    def describe(self, component: str, vector_index: str):
        """Describe the vector index in the backend.

        :param component: component class name.
        :param vector_index: vector index identifier.
        """
        return requests.get(
            f'{self.uri}/vector_search/describe?'
            f'component={component}&vector_index={vector_index}'
        )

    def put_component(self, component: str, uuid: str):
        """Add a component to the vector search service.

        :param component: Component to add
        :return: Response from the put operation
        """
        response = requests.post(
            f'{self.uri}/vector_search/put_component?component={component}&uuid={uuid}',
        )
        if response.status_code != 200:
            raise Exception(f"Failed to put component: {response.text}")
        return response.json()

    def find_nearest_from_id(
        self,
        id: str,
        component: str,
        vector_index: str,
        n: int = 100,
        within_ids: t.List | None = None,
    ):
        """Find nearest vectors to a vector with the given id.

        :param id: ID of the vector to find nearest neighbors for
        :param vector_index: Name of the vector index to search
        :param n: Number of results to return
        :param within_ids: Optional list of IDs to search within
        :return: Tuple of (ids, scores) of nearest neighbors
        """
        response = requests.post(
            f'{self.uri}/vector_search/find_nearest_from_id',
            json={
                'id': id,
                'component': component,
                'vector_index': vector_index,
                'n': n,
                'within_ids': within_ids,
            },
        )
        if response.status_code != 200:
            raise Exception(f"Failed to find nearest vectors: {response.text}")

        out = response.json()
        return out['ids'], out['scores']

    def find_nearest_from_array(
        self,
        h: numpy.typing.ArrayLike,
        component: str,
        vector_index: str,
        n: int = 100,
        within_ids: t.List | None = None,
    ):
        """Find nearest vectors to a given vector array.

        :param h: Vector array to find nearest neighbors for
        :param vector_index: Name of the vector index to search
        :param n: Number of results to return
        :param within_ids: Optional list of IDs to search within
        :return: Tuple of (ids, scores) of nearest neighbors
        """
        response = requests.post(
            f'{self.uri}/vector_search/find_nearest_from_array',
            json={
                'h': h.tolist(),  # type: ignore[union-attr]
                'component': component,
                'vector_index': vector_index,
                'n': n,
                'within_ids': within_ids,
            },
        )
        if response.status_code != 200:
            raise Exception(f"Failed to find nearest vectors: {response.text}")
        out = response.json()
        return out['ids'], out['scores']


class SimpleVectorSearch(VectorSearchBackend):
    """Service for vector similarity search.

    Provides APIs for managing vector indices and performing similarity searches.
    """

    def __init__(self, backend: VectorSearchBackend, *args, **kwargs):
        """Initialize the VectorSearch service.

        :param backend: Vector search backend implementation
        :param args: Additional arguments passed to parent constructor
        :param kwargs: Additional keyword arguments passed to parent constructor
        """
        super().__init__(*args, **kwargs)
        self.backend = backend

    @property
    def db(self):
        """Get the database instance.

        :return: Database instance
        """
        return self._db

    @db.setter
    def db(self, db):
        """Set the database instance for both this service and the backend.

        :param db: Database instance to set
        """
        self._db = db
        self.backend.db = db

    def initialize(self):
        """Initialize the vector search backend."""
        self.backend.initialize()

    def find_nearest_from_id(
        self,
        id: str,
        vector_index: str,
        n: int = 100,
        within_ids: t.List | None = None,
        component: str = 'VectorIndex',
    ):
        """Query the vector index with an ID.

        :param id: ID to query
        :param vector_index: Vector index to query
        :param n: Number of results to return
        :param within_ids: Optional list of IDs to search within
        :param component: Component type
        :return: Dictionary with ids and scores of nearest neighbors
        :raises HTTPException: If vectors are not yet loaded in vector database
        """
        ids, scores = self.backend.find_nearest_from_id(
            id,
            component=component,
            vector_index=vector_index,
            n=n,
            within_ids=within_ids,  # type: ignore[arg-type]
        )
        return {"ids": ids, "scores": scores}

    def find_nearest_from_array(
        self,
        h: t.List,
        vector_index: str,
        n: int = 100,
        within_ids: t.List | None = None,
        component: str = 'VectorIndex',
    ):
        """Query the vector index with a vector.

        :param h: Vector to query
        :param vector_index: Vector index to query
        :param n: Number of results to return
        :param within_ids: Optional list of IDs to search within
        :param component: Component type
        :return: Dictionary with ids and scores of nearest neighbors
        :raises HTTPException: If vectors are not yet loaded in vector database
        """
        ids, scores = self.backend.find_nearest_from_array(
            h,
            vector_index=vector_index,
            component=component,
            n=n,
            within_ids=within_ids,  # type: ignore[arg-type]
        )
        return {"ids": ids, "scores": scores}

    def add(self, vectors: t.List[VectorItem], uuid: str):
        """Add vectors to a vector index.

        :param vectors: Vectors to be added
        :param uuid: Vector index identifier where vectors need to be added
        """
        self.backend.add(uuid=uuid, vectors=vectors)

    def put_component(self, component: str, uuid: str):
        """Create a vector index.

        :param component: Vector index component to create
        :param uuid: UUID of the component
        """
        logging.info(
            f"Putting vector index {component}/{uuid} on vector-search "
            f"backend {self}..."
        )
        try:
            self.backend.put_component(component=component, uuid=uuid)
        except Exception as e:
            self.db.metadata.set_component_failed(
                component=component,
                uuid=uuid,
                reason=str(e),
                message=traceback.format_exc(),
            )
            raise e
        logging.info(
            f"Putting vector index {component}/{uuid} on vector-search backend "
            f"{self}... DONE"
        )

    def describe(self, component: str, vector_index: str):
        """Describe the vector index in the backend.

        :param component: component class name.
        :param vector_index: vector index identifier.
        """
        return self.backend.describe(component=component, vector_index=vector_index)

    def drop(self):
        """Remove all components from the vector search service.

        :return: Result from the drop operation
        """
        return self.backend.drop()

    def drop_component(
        self, component: str, identifier: str | None = None, uuid: str | None = None
    ):
        """Remove a specific component from the vector search service.

        :param component: Component type
        :param identifier: Component identifier
        :return: Result from the drop operation
        """
        return self.backend.drop_component(component, identifier=identifier, uuid=uuid)

    def build(self, app):
        """Set up FastAPI routes for this service.

        :param app: FastAPI application to set routes on
        """

        @app.post("/vector_search/initialize")
        def initialize():
            """Initialize the vector search service.

            :return: Status response
            """
            self.initialize()
            return {"status": "ok"}

        @app.post("/vector_search/add")
        def add(kwargs: t.Dict):
            """Add vectors to a vector index.

            :param kwargs: Dictionary containing 'vectors' and 'uuid' keys
            :return: Result of the add operation
            """
            kwargs['vectors'] = [VectorItem(**x) for x in kwargs['vectors']]
            return self.add(**kwargs)

        @app.post("/vector_search/delete")
        def delete(kwargs: t.Dict):
            """Delete vectors from a vector index.

            :param kwargs: Dictionary containing 'uuid' and 'ids' keys
            :return: Result of the delete operation
            """
            return self.delete(**kwargs)

        @app.post("/vector_search/drop")
        def drop():
            """Remove all components from the vector search service.

            :return: Status response
            """
            self.backend.drop()
            return {"status": "ok"}

        @app.post("/vector_search/drop_component")
        def drop_component(
            component: str, identifier: str | None = None, uuid: str | None = None
        ):
            """Remove a specific component from the vector search service.

            :param component: Component type
            :param identifier: Component identifier
            :return: Status response
            """
            self.backend.drop_component(component, identifier=identifier, uuid=uuid)
            return {"status": "ok"}

        @app.post("/vector_search/put_component")
        def put_component(component: str, uuid: str):
            """Add a component to the vector search service.

            :param component: Component type
            :param uuid: Component UUID
            :return: Status response
            :raises AssertionError: If component is not a VectorIndex
            """
            self.put_component(component, uuid)
            return {"status": "ok"}

        @app.post("/vector_search/find_nearest_from_id")
        def find_nearest_from_id(kwargs: t.Dict):
            """Find nearest vectors to a vector with the given ID.

            :param kwargs: Dictionary containing search parameters
            :return: Dictionary with ids and scores of nearest neighbors
            """
            return self.find_nearest_from_id(**kwargs)

        @app.post("/vector_search/find_nearest_from_array")
        def find_nearest_from_array(kwargs: t.Dict):
            """Find nearest vectors to a given vector array.

            :param kwargs: Dictionary containing search parameters
            :return: Dictionary with ids and scores of nearest neighbors
            """
            return self.find_nearest_from_array(**kwargs)
