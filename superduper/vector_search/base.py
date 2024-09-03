from __future__ import annotations

import enum
import typing as t
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass

import numpy
import numpy.typing

if t.TYPE_CHECKING:
    from superduper.components.vector_index import VectorIndex


class BaseVectorSearcher(ABC):
    """Base class for vector searchers.

    :param identifier: Unique string identifier of index
    :param dimensions: Dimension of the vector embeddings
    :param h: Seed vectors ``numpy.ndarray``
    :param index: list of IDs
    :param measure: measure to assess similarity
    """

    native_service: t.ClassVar[bool] = True

    @abstractmethod
    def __init__(
        self,
        identifier: str,
        dimensions: int,
        h: t.Optional[numpy.ndarray] = None,
        index: t.Optional[t.List[str]] = None,
        measure: t.Optional[str] = None,
    ):
        self._init_vi: t.Dict = defaultdict(lambda: False)

    def initialize(self, identifier):
        """Initialize vector index."""
        self._init_vi[identifier] = True

    def is_initialized(self, identifier):
        """Check if vector index initialized."""
        return self._init_vi[identifier]

    @classmethod
    def from_component(cls, vi: 'VectorIndex'):
        """Create a vector searcher from a vector index.

        :param vi: VectorIndex instance
        """
        return cls(
            identifier=vi.identifier, dimensions=vi.dimensions, measure=vi.measure
        )

    @property
    def is_native(self):
        """Check if the vector searcher is native."""
        return self.native_service

    @abstractmethod
    def __len__(self):
        pass

    @staticmethod
    def to_numpy(h):
        """Converts a vector to a numpy array.

        :param h: vector, numpy.ndarray, or list
        """
        if isinstance(h, numpy.ndarray):
            return h
        if hasattr(h, 'numpy'):
            return h.numpy()
        if isinstance(h, list):
            return numpy.array(h)
        raise ValueError(str(h))

    @staticmethod
    def to_list(h):
        """Converts a vector to a list.

        :param h: vector
        """
        if hasattr(h, 'tolist'):
            return h.tolist()
        if isinstance(h, list):
            return h
        raise ValueError(str(h))

    @abstractmethod
    def add(self, items: t.Sequence[VectorItem]) -> None:
        """
        Add items to the index.

        :param items: t.Sequence of VectorItems
        """

    @abstractmethod
    def delete(self, ids: t.Sequence[str]) -> None:
        """Remove items from the index.

        :param ids: t.Sequence of ids of vectors.
        """

    @abstractmethod
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
        :param within_ids: list of ids to search within
        """

    @abstractmethod
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

    def post_create(self):
        """Post create method.

        This method is used for searchers which requires
        to perform a task after all vectors have been added
        """


class VectorIndexMeasureType(str, enum.Enum):
    """Enum for vector index measure types # noqa."""

    cosine = 'cosine'
    css = 'css'
    dot = 'dot'
    l2 = 'l2'


@dataclass(frozen=True)
class VectorItem:
    """Class for representing a vector in vector search with id and vector.

    :param id: ID of the vector
    :param vector: Vector of the item
    """

    id: str
    vector: numpy.ndarray

    @classmethod
    def create(
        cls,
        *,
        id: str,
        vector: numpy.typing.ArrayLike,
    ) -> VectorItem:
        """Creates a vector item from id and vector.

        :param id: ID of the vector
        :param vector: Vector of the item
        """
        return VectorItem(id=id, vector=BaseVectorSearcher.to_numpy(vector))

    def to_dict(self) -> t.Dict:
        """Converts the vector item to a dictionary."""
        return {'id': self.id, 'vector': self.vector}


def l2(x, y):
    """L2 function for vector similarity search.

    :param x: numpy.ndarray
    :param y: numpy.ndarray
    """
    return numpy.array([-numpy.linalg.norm(x - y, axis=1)])


def dot(x, y):
    """Dot function for vector similarity search.

    :param x: numpy.ndarray
    :param y: numpy.ndarray
    """
    return numpy.dot(x, y.T)


def cosine(x, y):
    """Cosine similarity function for vector search.

    :param x: numpy.ndarray
    :param y: numpy.ndarray, y should be normalized!
    """
    x = x / numpy.linalg.norm(x, axis=1)[:, None]
    # y which implies all vectors in vectordatabase
    # has normalized vectors.
    return dot(x, y)


measures = {'cosine': cosine, 'dot': dot, 'l2': l2}
