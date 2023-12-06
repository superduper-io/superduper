import dataclasses as dc
import typing as t

from superduperdb.components.component import Component
from superduperdb.misc.annotations import public_api
from superduperdb.misc.serialization import serializers

if t.TYPE_CHECKING:
    from superduperdb.base.datalayer import Datalayer


@public_api(stability='beta')
@dc.dataclass(kw_only=True)
class Serializer(Component):
    """
    A component carrying the information to apply a serializer to a
    model.
    {component_parameters}
    :param object: The serializer
    """

    __doc__ = __doc__.format(component_parameters=Component.__doc__)

    type_id: t.ClassVar[str] = 'serializer'

    object: t.Type

    def pre_create(self, db: 'Datalayer'):
        super().pre_create(db)
        serializers.add(self.identifier, self.object)
        self.object = t.cast(t.Type, self.identifier)
