import dataclasses as dc
import importlib
import json
import os
import re
import typing as t

from superduperdb.misc.annotations import public_api

from .component import Component


@public_api(stability='alpha')
@dc.dataclass(kw_only=True)
class Stack(Component):
    """
    A placeholder to hold list of components under a namespace and packages them as
    a tarball
    This tarball can be retrieved back to a `Stack` instance with ``load`` method.
    {component_parameters}
    :param components: List of components to stack together and add to database.
    """

    __doc__ = __doc__.format(component_parameters=Component.__doc__)

    type_id: t.ClassVar[str] = 'stack'
    components: t.Sequence[Component] = ()