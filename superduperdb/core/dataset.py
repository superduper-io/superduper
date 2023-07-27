from __future__ import annotations
from overrides import override
from functools import cached_property
import dataclasses as dc
import numpy
import typing as t

from superduperdb.core.artifact import Artifact
from superduperdb.core.component import Component
from superduperdb.core.document import Document
from superduperdb.datalayer.mongodb.query import Find

if t.TYPE_CHECKING:
    from superduperdb.datalayer.base.datalayer import Datalayer


@dc.dataclass
class Dataset(Component):
    variety: t.ClassVar[str] = 'dataset'

    identifier: str
    select: t.Optional[Find] = None
    sample_size: t.Optional[int] = None
    random_seed: t.Optional[int] = None
    creation_date: t.Optional[str] = None
    raw_data: t.Optional[t.Union[Artifact, t.Any]] = None
    version: t.Optional[int] = None

    @override
    def on_create(self, db: Datalayer) -> None:
        if self.raw_data is None:
            data = list(db.execute(self.select))  # type: ignore[arg-type]
            if self.sample_size is not None and self.sample_size < len(data):
                perm = self.random.permutation(len(data)).tolist()
                data = [data[perm[i]] for i in range(self.sample_size)]
            self.raw_data = Artifact(artifact=[r.encode() for r in data])

    @override
    def on_load(self, db: Datalayer) -> None:
        self.data = [
            Document(Document.decode(r.copy(), encoders=db.encoders))
            for r in self.raw_data.artifact
        ]

    @cached_property
    def random(self):
        return numpy.random.default_rng(seed=self.random_seed)
