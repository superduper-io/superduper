from __future__ import annotations

import enum
import typing as t
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy
import numpy.typing


class BaseVectorIndex:
    name: t.Optional[str] = None
    index: t.List[str]
    lookup: t.Dict[str, t.Union[t.Iterator[int], int]]
    measure: str

    def __init__(self, h, index, measure):
        if hasattr(h, 'tolist'):
            h = h.tolist()
        self.h_list = h
        self._h = numpy.array(h)
        self.index = index
        if index is not None:
            self.lookup = dict(zip(index, range(len(index))))
        self.measure = measure

    @property
    def h(self):
        if len(self.h_list) != self._h.shape[0]:
            self._h = numpy.array(self.h_list)
        return self._h

    @property
    def shape(self):
        return self.h.shape

    def find_nearest_from_id(self, _id, n=100):
        _ids, scores = self.find_nearest_from_ids([_id], n=n)
        return _ids[0], scores[0]

    def find_nearest_from_ids(self, _ids, n=100):
        ix = list(map(self.lookup.__getitem__, _ids))
        return self.find_nearest_from_arrays(self.h[ix, :], n=n)

    def find_nearest_from_array(self, h, n=100):
        h = to_numpy(h)
        _ids, scores = self.find_nearest_from_arrays(h[None, :], n=n)
        return _ids[0], scores[0]

    def find_nearest_from_arrays(self, h, n=100):
        raise NotImplementedError

    def __getitem__(self, item):
        raise NotImplementedError


def to_numpy(x: numpy.typing.ArrayLike) -> numpy.ndarray:
    return numpy.array(x)


class VectorIndexMeasureType(str, enum.Enum):
    cosine = 'cosine'
    css = 'css'
    dot = 'dot'
    l2 = 'l2'


VectorCollectionId = str
VectorCollectionItemId = str
VectorIndexMeasure = t.Union[VectorIndexMeasureType, str]


class VectorCollectionItemNotFound(Exception):
    pass


@dataclass(frozen=True)
class VectorCollectionConfig:
    id: VectorCollectionId
    dimensions: int
    measure: VectorIndexMeasure = VectorIndexMeasureType.l2
    parameters: t.Mapping[str, t.Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VectorCollectionItem:
    id: VectorCollectionItemId
    vector: numpy.ndarray

    @classmethod
    def create(
        cls,
        *,
        id: VectorCollectionItemId,
        vector: numpy.typing.ArrayLike,
    ) -> VectorCollectionItem:
        return VectorCollectionItem(id=id, vector=to_numpy(vector))

    def to_dict(self) -> t.Dict:
        return {'id': self.id, 'vector': self.vector}


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

    @abstractmethod
    def add(self, items: t.Sequence[VectorCollectionItem]) -> None:
        """
        Add items to the collection.

        :param items: items to add
        """
        pass

    @abstractmethod
    def find_nearest_from_id(
        self,
        identifier: VectorCollectionItemId,
        *,
        within_ids: t.Sequence[VectorCollectionItemId] = (),
        limit: int = 100,
        offset: int = 0,
    ) -> t.List[VectorCollectionResult]:
        """
        Find items that are nearest to the item with the given identifier.

        :param identifier: identifier of the item
        :param within_ids: identifiers to search within
        :param limit: maximum number of nearest items to return
        :param offset: offset of the first item to return
        """
        pass

    @abstractmethod
    def find_nearest_from_array(
        self,
        array: numpy.typing.ArrayLike,
        *,
        within_ids: t.Sequence[VectorCollectionItemId] = (),
        limit: int = 100,
        offset: int = 0,
    ) -> t.List[VectorCollectionResult]:
        """
        Find items that are nearest to the given vector.

        :param array: array representing the vector
        :param within_ids: identifiers to search within
        :param limit: maximum number of nearest items to return
        :param offset: offset of the first item to return
        """
        pass


class VectorDatabase(ABC):
    """
    A vector database that manages vector collections.

    Concrete implementations of this class are responsible for lifecycle management of
    vector collections within a specific vector database.

    Implementations of this abstract base class retrieve vector collections by their
    identifier for subsequent operations.
    """

    @abstractmethod
    def get_table(self, config: VectorCollectionConfig) -> VectorCollection:
        """Get a vector collection by its identifier.

        :param config: configuration for the vector database
        """
        pass
