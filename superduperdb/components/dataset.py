from __future__ import annotations

import dataclasses as dc
import typing as t
from functools import cached_property

import numpy
from overrides import override

from superduperdb.backends.mongodb.query import Select
from superduperdb.base.artifact import Artifact
from superduperdb.base.datalayer import Datalayer
from superduperdb.base.document import Document
from superduperdb.components.component import Component


@dc.dataclass
class Dataset(Component):
    """A dataset is an immutable collection of documents that used for training

    :param identifier: A unique identifier for the dataset
    :param select: A query to select the documents for the dataset
    :param sample_size: The number of documents to sample from the query
    :param random_seed: The random seed to use for sampling
    :param creation_date: The date the dataset was created
    :param raw_data: The raw data for the dataset
    :param version: The version of the dataset
    """

    identifier: str
    select: t.Optional[Select] = None
    sample_size: t.Optional[int] = None
    random_seed: t.Optional[int] = None
    creation_date: t.Optional[str] = None
    raw_data: t.Optional[t.Union[Artifact, t.Any]] = None

    # Don't set these manually
    version: t.Optional[int] = None
    type_id: t.ClassVar[str] = 'dataset'

    @override
    def pre_create(self, db: 'Datalayer') -> None:
        if self.raw_data is None:
            if self.select is None:
                raise ValueError('select cannot be None')
            data = list(db.execute(self.select))
            if self.sample_size is not None and self.sample_size < len(data):
                perm = self.random.permutation(len(data)).tolist()
                data = [data[perm[i]] for i in range(self.sample_size)]
            self.raw_data = Artifact(artifact=[r.encode() for r in data])

    @override
    def post_create(self, db: 'Datalayer') -> None:
        return self.on_load(db)

    @override
    def on_load(self, db: 'Datalayer') -> None:
        assert isinstance(self.raw_data, Artifact)
        self.data = [
            Document(Document.decode(r.copy(), encoders=db.encoders))
            for r in self.raw_data.artifact
        ]

    @cached_property
    def random(self):
        return numpy.random.default_rng(seed=self.random_seed)
