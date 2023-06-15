from __future__ import annotations
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from contextlib import contextmanager
import numpy
import numpy.typing
import torch
from typing import List, Iterator, Sequence, Callable, Any, Mapping, Union, Literal

from ..misc.config import VectorSearchConfig


class BaseHashSet:
    name = None

    def __init__(self, h, index, measure):
        if isinstance(h, list) and isinstance(h[0], torch.Tensor):
            h = torch.stack(h).numpy()
        elif isinstance(h, list) and isinstance(h[0], numpy.ndarray):
            h = numpy.stack(h)
        elif isinstance(h, list) and isinstance(h[0], list):
            h = numpy.array(h)
        elif isinstance(h, torch.Tensor):
            h = h.numpy()
        self.h = h
        self.index = index
        if index is not None:
            self.lookup = dict(zip(index, range(len(index))))
        self.measure = measure

    @property
    def shape(self):  # pragma: no cover
        return self.h.shape

    def find_nearest_from_id(self, _id, n=100):
        _ids, scores = self.find_nearest_from_ids([_id], n=n)
        return _ids[0], scores[0]

    def find_nearest_from_ids(self, _ids, n=100):
        ix = list(map(self.lookup.__getitem__, _ids))
        return self.find_nearest_from_hashes(self.h[ix, :], n=n)

    def find_nearest_from_hash(self, h, n=100):
        h = to_numpy(h)
        _ids, scores = self.find_nearest_from_hashes(h[None, :], n=n)
        return _ids[0], scores[0]

    def find_nearest_from_hashes(self, h, n=100):
        raise NotImplementedError

    def __getitem__(self, item):
        raise NotImplementedError


ArrayLike = Union[numpy.typing.ArrayLike, torch.Tensor]


def to_numpy(x: ArrayLike) -> numpy.ndarray:
    if isinstance(x, numpy.ndarray):
        return x
    if isinstance(x, torch.Tensor):
        return x.numpy()
    return numpy.array(x)


VectorCollectionId = str
VectorCollectionItemId = str
VectorIndexMeasureType = Literal["l2", "dot", "css"]
VectorIndexMeasureFunction = Callable[[numpy.ndarray, numpy.ndarray], float]
VectorIndexMeasure = Union[VectorIndexMeasureType, VectorIndexMeasureFunction]


class VectorCollectionItemNotFound(Exception):
    pass


@dataclass(frozen=True)
class VectorCollectionConfig:
    id: VectorCollectionId
    dimensions: int
    measure: VectorIndexMeasure = "l2"
    parameters: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VectorCollectionItem:
    id: VectorCollectionItemId
    vector: numpy.ndarray

    @classmethod
    def create(
        self, *, id: VectorCollectionItemId, vector: ArrayLike
    ) -> VectorCollectionItem:
        return VectorCollectionItem(id=id, vector=to_numpy(vector))


@dataclass(frozen=True)
class VectorCollectionResult:
    id: VectorCollectionItemId
    score: float


class VectorCollection(ABC):
    """A vector collection within a vector database.

    Concrete implementations of this class are responsible for lifecycle management of
    a single vector collection within a specific vector database.

    It is assumed that a vector collection is associated with a single vector column or
    field within a vector database.
    """

    @contextmanager
    @abstractmethod
    def init(self) -> Iterator["VectorCollection"]:
        ...

    @abstractmethod
    def add(self, items: Sequence[VectorCollectionItem]) -> None:
        """Add items to the collection."""
        ...

    @abstractmethod
    def find_nearest_from_id(
        self,
        identifier: VectorCollectionItemId,
        *,
        within_ids: Sequence[VectorCollectionItemId] = (),
        limit: int = 100,
        offset: int = 0,
    ) -> List[VectorCollectionResult]:
        """Find items that are nearest to the item with the given identifier."""
        ...

    @abstractmethod
    def find_nearest_from_array(
        self,
        array: ArrayLike,
        *,
        within_ids: Sequence[VectorCollectionItemId] = (),
        limit: int = 100,
        offset: int = 0,
    ) -> List[VectorCollectionResult]:
        """Find items that are nearest to the given vector."""
        ...


class VectorDatabase(ABC):
    """A manager for vector indexes within a vector database.
    A vector database that manages indexed vector collections.

    Concrete implementations of this class are responsible for lifecycle management of
    vector collections within a specific vector database.

    Clients of this class retrieve vector collections by their identifier for subsequent
    operations.
    """

    def __init__(self, *, config: VectorSearchConfig) -> None:
        self._config = config

    @classmethod
    def create(cls, *, config: VectorSearchConfig) -> VectorDatabase:
        if config.milvus:
            return cls.create_milvus(config=config)
        return cls.create_in_memory(config=config)

    @classmethod
    def create_milvus(self, *, config: VectorSearchConfig) -> VectorDatabase:
        # avoiding circular import
        from .milvus import MilvusVectorDatabase

        return MilvusVectorDatabase(config=config)

    @classmethod
    def create_in_memory(self, *, config: VectorSearchConfig) -> VectorDatabase:
        # avoiding circular import
        from .inmemory import InMemoryVectorDatabase

        return InMemoryVectorDatabase(config=config)

    @contextmanager
    @abstractmethod
    def init(self) -> Iterator["VectorDatabase"]:
        """Initialize the database.

        This method makes sure all necessary connections to the underlying vector
        database are established.
        """
        ...

    @contextmanager
    @abstractmethod
    def get_collection(
        self,
        config: VectorCollectionConfig,
    ) -> Iterator[VectorCollection]:
        """Get a vector collection by its identifier.

        If the collection does not exist, it is created with the specified
        dimensionality.
        """
        ...
