import typing as t
from abc import abstractmethod

from superduper.components.component import Component


class Cache:
    """Cache object for caching components.

    # noqa
    """

    @abstractmethod
    def __getitem__(self, *item) -> t.Dict | t.List:
        """Get a component from the cache."""
        pass

    @abstractmethod
    def keys(self, *pattern) -> t.List[str]:
        """Get the keys from the cache.

        :param pattern: The pattern to match.

        >>> cache.keys('*', '*', '*')
        >>> cache.keys('Model', '*', '*')
        >>> cache.keys('Model', 'my_model', '*')
        >>> cache.keys('*', '*', '1234567890')
        """

    def get_with_uuid(self, uuid: str):
        """Get a component from the cache with a specific uuid.

        :param uuid: The uuid of the component to get.
        """
        key = self.keys('*', '*', uuid)
        if not key:
            return None
        else:
            key = key[0]

        try:
            return self[key]
        except KeyError:
            return

    def get_with_component(self, component: str):
        """Get all components from the cache of a certain type.

        :param component: The component to get.
        """
        keys = self.keys(component, '*', '*')
        return [self[k] for k in keys]

    def get_with_component_identifier(self, component: str, identifier: str):
        """Get a component from the cache with a specific identifier.

        :param component: The component to get.
        :param identifier: The identifier of the component to
        """
        keys = self.keys(component, identifier, '*')
        out = [self[k] for k in keys]
        if not out:
            return None
        out = max(out, key=lambda x: x['version'])  # type: ignore[arg-type, call-overload]
        return out

    def get_with_component_identifier_version(
        self, component: str, identifier: str, version: int
    ):
        """Get a component from the cache with a specific version.

        :param component: The component to get.
        :param identifier: The identifier of the component to get.
        :param version: The version of the component to get.
        """
        keys = self.keys(component, identifier, '*')
        out = [self[k] for k in keys]
        try:
            return next(r for r in out if r['version'] == version)  # type: ignore[call-overload]
        except StopIteration:
            return

    def __contains__(self, key: str) -> bool:
        return key in self.keys()

    @abstractmethod
    def __setitem__(self, key: t.Tuple[str, str, str], value: t.Dict) -> None:
        pass

    def delete_uuid(self, uuid: str):
        """Delete a component from the cache.

        :param uuid: The uuid of the component to delete.
        """
        keys = self.keys('*', '*', uuid)
        for key in keys:
            del self[key]  # type: ignore[arg-type]

    @abstractmethod
    def __delitem__(self, key: t.Tuple[str, str, str]):
        pass
