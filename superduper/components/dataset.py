from __future__ import annotations

import typing as t
from functools import cached_property

import numpy

from superduper.backends.base.query import Query
from superduper.base.datalayer import Datalayer
from superduper.base.document import Document
from superduper.components.component import Component, ensure_initialized
from superduper.components.schema import Schema


class Dataset(Component):
    """A dataset is an immutable collection of documents.

    :param select: A query to select the documents for the dataset.
    :param sample_size: The number of documents to sample from the query.
    :param random_seed: The random seed to use for sampling.
    :param creation_date: The date the dataset was created.
    :param raw_data: The raw data for the dataset.
    :param schema: Optional schema for decoding the data.
    :param pin: Whether to pin the dataset.
                If True, the dataset will load the datas from the database every time.
                If False, the dataset will cache the datas after we apply to db.
    """

    type_id: t.ClassVar[str] = 'dataset'

    select: t.Optional[Query] = None
    sample_size: t.Optional[int] = None
    random_seed: t.Optional[int] = None
    creation_date: t.Optional[str] = None
    raw_data: t.Optional[t.Sequence[t.Any]] = None
    schema: t.Optional[Schema] = None
    pin: bool = False

    def postinit(self):
        """Post initialization method."""
        self._data = None
        super().postinit()

    @property
    @ensure_initialized
    def data(self):
        """Property representing the dataset's data."""
        return self._data

    def init(self):
        """Initialization method."""
        super().init()
        if self.pin:
            assert self.raw_data is not None
            if self.schema is not None:
                self._data = [
                    Document.decode(r, db=self.db, schema=self.schema).unpack()
                    for r in self.raw_data
                ]
            else:
                self._data = self.raw_data
        else:
            self._data = self._load_data(self.db)

    def _pre_create(self, db: 'Datalayer', startup_cache: t.Dict = {}) -> None:
        if self.raw_data is None and self.pin:
            data = self._load_data(db)
            self.raw_data = [r.encode() for r in data]

    def _load_data(self, db: 'Datalayer'):
        assert db is not None, 'Database must be set'
        assert self.select is not None, 'Select must be set'
        data = self.select.execute()
        if self.sample_size is not None and self.sample_size < len(data):
            perm = self.random.permutation(len(data)).tolist()
            data = [data[perm[i]] for i in range(self.sample_size)]
        return data

    @cached_property
    def random(self):
        """Cached property representing the random number generator."""
        return numpy.random.default_rng(seed=self.random_seed)

    def __str__(self):
        """String representation of the dataset."""
        return f'Dataset(identifier={self.identifier}, select={self.select})'

    __repr__ = __str__


class RemoteData(Component):
    """Class to fetch dataset from remote.

    :param getter: Function to fetch data.
    """

    type_id: t.ClassVar[str] = 'dataset'
    getter: t.Callable

    def __post_init__(self, db):
        self._data = None
        return super().__post_init__(db)

    @property
    def data(self):
        """Get the data."""
        if self._data is None:
            self._data = self.getter()
        return self._data


class Data(Component):
    """Class to store data.

    :param raw_data: The raw data.
    """

    type_id: t.ClassVar[str] = 'data'
    raw_data: t.Any

    @property
    def data(self):
        return self.raw_data
