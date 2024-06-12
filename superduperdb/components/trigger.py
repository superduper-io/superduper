from superduperdb import Component
import typing as t


class Trigger(Component):
    """Trigger a function when a condition is met.

    ***Note that this feature deploys on SuperDuperDB Enterprise.***
    
    :param on: When to trigger the function `{'insert', 'update', 'delete'}`.
    :param condition: Additional condition to trigger the function.
    """
    table: str
    on: str = 'insert'
    condition: t.Optional[t.Callable] = None

    def if_change(self, ids):
        raise NotImplementedError