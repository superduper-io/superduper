from abc import abstractmethod

from superduper.backends.base.backends import BaseBackend
from superduper.components.component import Component


class Cache(BaseBackend):
    """Cache object for caching components.

    # noqa
    """

    @abstractmethod
    def __getitem__(self, *item) -> Component:
        """Get a component from the cache."""
        pass
