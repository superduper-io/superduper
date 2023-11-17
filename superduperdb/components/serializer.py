import dataclasses as dc
import typing as t

from superduperdb.components.component import Component
from superduperdb.misc.serialization import serializers

if t.TYPE_CHECKING:
    from superduperdb.base.datalayer import Datalayer


@dc.dataclass
class Serializer(Component):
    identifier: str
    object: t.Type

    type_id: t.ClassVar[str] = 'serializer'

    version: t.Optional[int]

    def pre_create(self, db: 'Datalayer'):
        serializers.add(self.identifier, self.object)

        self.object = t.cast(t.Type, self.identifier)
