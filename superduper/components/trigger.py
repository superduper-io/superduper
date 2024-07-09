import typing as t

from superduper import Component


class Trigger(Component):
    """Trigger a function when a condition is met.

    ***Note that this feature deploys on superduper.io Enterprise.***

    :param on: When to trigger the function `{'insert', 'update', 'delete'}`.
    :param condition: Additional condition to trigger the function.
    """

    table: str
    on: str = 'insert'
    condition: t.Optional[t.Callable] = None

    def pull(self, ids):
        """Pull the trigger."""
        raise NotImplementedError
