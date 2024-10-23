import enum
import typing as t
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy
import numpy.typing

from superduper.backends.base.backends import BaseBackend

if t.TYPE_CHECKING:
    from superduper.base.datalayer import Datalayer
    from superduper.components.vector_index import VectorIndex


class VectorSearchBackend(BaseBackend):
    """Base vector-search backend."""

    def __init__(self):
        self._cache = {}

    @abstractmethod
    def __getitem__(self, identifier):
        pass

    def add(self, identifier, vectors):
        """Add vectors to a vector-index.

        :param identifier: Identifier of index.
        :param vectors: Vectors.
        """
        self.get(identifier).add(vectors)

    def delete(self, identifier, ids):
        """Delete ids from index.

        :param identifier: Identifier of index.
        :param ids: Ids to delete.
        """
        self.get(identifier).delete(ids)

    @property
    def db(self) -> 'Datalayer':
        """Get the ``db``."""
        return self._db

    @db.setter
    def db(self, value: 'Datalayer'):
        """Set the ``db``.

        :param value: ``Datalayer`` instance.
        """
        self._db = value


class VectorSearcherInterface(ABC):
    """Interface for vector searchers.

    # noqa
    """

    @abstractmethod
    def add(self, items: t.Sequence['VectorItem']) -> None:
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


class BaseVectorSearcher(VectorSearcherInterface):
    """Base class for vector searchers.

    :param uuid: Unique string identifier of index
    :param dimensions: Dimension of the vector embeddings
    :param measure: measure to assess similarity
    """

    native_service: t.ClassVar[bool] = True

    @abstractmethod
    def __init__(
        self,
        uuid: str,
        dimensions: int,
        measure: str,
    ):
        pass

    @classmethod
    def from_component(cls, index: 'VectorIndex'):
        """Create a vector searcher from a vector index.

        :param vi: ``VectorIndex`` instance
        """
        return cls(uuid=index.uuid, dimensions=index.dimensions, measure=index.measure)

    @abstractmethod
    def initialize(self, db):
        """Initialize the vector-searcher.

        :param db: ``Datalayer`` instance.
        """
        pass

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
    ) -> 'VectorItem':
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
