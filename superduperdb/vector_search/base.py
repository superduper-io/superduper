from __future__ import annotations

import enum
import typing as t
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy
import numpy.typing


class BaseVectorSearcher(ABC):
    @abstractmethod
    def __init__(
        self,
        identifier: str,
        dimensions: int,
        h: t.Optional[numpy.ndarray] = None,
        index: t.Optional[t.List[str]] = None,
        measure: t.Optional[str] = None,
    ):
        pass

    @abstractmethod
    def __len__(self):
        pass

    @staticmethod
    def to_numpy(h):
        if isinstance(h, numpy.ndarray):
            return h
        if hasattr(h, 'numpy'):
            return h.numpy()
        if isinstance(h, list):
            return numpy.array(h)
        raise ValueError(str(h))

    @staticmethod
    def to_list(h):
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
        """
        Remove items from the index

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
        """


class VectorIndexMeasureType(str, enum.Enum):
    cosine = 'cosine'
    css = 'css'
    dot = 'dot'
    l2 = 'l2'


@dataclass(frozen=True)
class VectorSearchConfig:
    '''
    Represents search config which helps initiate a vector
    searcher class.
    '''

    id: str
    dimensions: int
    measure: VectorIndexMeasureType = VectorIndexMeasureType.l2
    parameters: t.Mapping[str, t.Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VectorItem:
    '''
    Class for representing a vector in vector search with
    id and vector.

    '''

    id: str
    vector: numpy.ndarray

    @classmethod
    def create(
        cls,
        *,
        id: str,
        vector: numpy.typing.ArrayLike,
    ) -> VectorItem:
        return VectorItem(id=id, vector=BaseVectorSearcher.to_numpy(vector))

    def to_dict(self) -> t.Dict:
        return {'id': self.id, 'vector': self.vector}


@dataclass(frozen=True)
class VectorSearchResult:
    '''
    Dataclass for representing vector search results with
    `id` and `score`.
    '''

    id: str
    score: float


def l2(x, y):
    '''
    L2 function for vector similarity search
    '''
    return numpy.array([-numpy.linalg.norm(x - y, axis=1)])


def dot(x, y):
    '''
    Dot function for vector similarity search
    '''
    return numpy.dot(x, y.T)


def cosine(x, y):
    '''
    Cosine similarity function for vector search
    '''
    x = x / numpy.linalg.norm(x, axis=1)[:, None]
    y = y / numpy.linalg.norm(y, axis=1)[:, None]
    return dot(x, y)


measures = {'cosine': cosine, 'dot': dot, 'l2': l2}
