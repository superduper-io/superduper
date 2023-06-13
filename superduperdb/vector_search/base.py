from __future__ import annotations
from dataclasses import dataclass
from abc import ABC, abstractmethod
from contextlib import contextmanager
import numpy
import torch
from typing import List, Iterator, Sequence, Callable

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
        if isinstance(h, list):
            h = numpy.array(h)
        elif isinstance(h, torch.Tensor):
            h = h.numpy()
        _ids, scores = self.find_nearest_from_hashes(h[None, :], n=n)
        return _ids[0], scores[0]

    def find_nearest_from_hashes(self, h, n=100):
        raise NotImplementedError

    def __getitem__(self, item):
        raise NotImplementedError


VectorIndexId = str
VectorIndexItemId = str
VectorIndexMeasureFunction = Callable[[numpy.ndarray, numpy.ndarray], float]


class VectorIndexItemNotFound(Exception):
    pass


@dataclass(frozen=True)
class VectorIndexItem:
    id: VectorIndexItemId
    vector: numpy.ndarray


@dataclass(frozen=True)
class VectorIndexResult:
    id: VectorIndexItemId
    score: float


class VectorIndex(ABC):
    """A vector index within a vector database.

    Concrete implementations of this class are responsible for lifecycle management of
    a single vector index within a specific vector database.

    It is assumed that a vector index is associated with a single vector column or
    field within a vector database.
    """

    @contextmanager
    @abstractmethod
    def init(self) -> Iterator["VectorIndex"]:
        ...

    @abstractmethod
    def add(self, items: Sequence[VectorIndexItem]) -> None:
        """Add items to the index."""
        ...

    @abstractmethod
    def find_nearest_from_id(
        self, identifier: VectorIndexItemId, *, limit: int = 100, offset: int = 0
    ) -> List[VectorIndexResult]:
        """Find items that are nearest to the item with the given identifier."""
        ...

    @abstractmethod
    def find_nearest_from_array(
        self, array: numpy.ndarray, *, limit: int = 100, offset: int = 0
    ) -> List[VectorIndexResult]:
        """Find items that are nearest to the given vector."""
        ...


class VectorIndexManager(ABC):
    """A manager for vector indexes within a vector database.

    Concrete implementations of this class are responsible for lifecycle management of
    vector indexes within a specific vector database.

    Clients of this class retrieve vector indexes by their identifier for subsequent
    operations.
    """

    def __init__(self, *, config: VectorSearchConfig) -> None:
        self._config = config

    @contextmanager
    @abstractmethod
    def init(self) -> Iterator["VectorIndexManager"]:
        """Initialize the manager.

        This method makes sure all necessary connections to the underlying vector
        database are established.
        """
        ...

    @contextmanager
    @abstractmethod
    def get_index(
        self, identifier: VectorIndexId, *, dimensions: int
    ) -> Iterator[VectorIndex]:
        """Get a vector index by its identifier.

        If the index does not exist, it is created with the specified dimensionality.
        """
        ...
