from __future__ import annotations

import dataclasses as dc
import typing as t
from functools import cached_property

import numpy
from overrides import override

from superduperdb.container.artifact import Artifact
from superduperdb.container.component import Component
from superduperdb.container.document import Document
from superduperdb.db.mongodb.query import Find

if t.TYPE_CHECKING:
    from superduperdb.db.base.db import DB


@dc.dataclass
class Dataset(Component):
    identifier: str
    select: Find | None = None
    sample_size: int | None = None
    random_seed: int | None = None
    creation_date: str | None = None
    raw_data: Artifact | t.Any | None = None
    version: int | None = None

    #: A unique name for the class
    type_id: t.ClassVar[str] = 'dataset'

    @override
    def on_create(self, db: DB) -> None:
        if self.raw_data is None:
            if self.select is None:
                raise ValueError('select cannot be None')
            data = list(db.execute(self.select))
            if self.sample_size is not None and self.sample_size < len(data):
                perm = self.random.permutation(len(data)).tolist()
                data = [data[perm[i]] for i in range(self.sample_size)]
            self.raw_data = Artifact(artifact=[r.encode() for r in data])

    @override
    def on_load(self, db: DB) -> None:
        self.data = [
            Document(Document.decode(r.copy(), encoders=db.encoders))
            for r in self.raw_data.artifact  # type: ignore[union-attr]
        ]

    @cached_property
    def random(self):
        return numpy.random.default_rng(seed=self.random_seed)
