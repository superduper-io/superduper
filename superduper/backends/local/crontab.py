from superduper.backends.base.crontab import CrontabBackend


class LocalCrontabBackend(CrontabBackend):
    """Local crontab backend."""

    def _put(self, item):
        raise NotImplementedError

    def list(self):
        """List crontab items."""
        raise NotImplementedError

    def __delitem__(self, item):
        raise NotImplementedError

    def list_components(self):
        """List components."""
        raise NotImplementedError

    def list_uuids(self):
        """List UUIDs of components."""
        raise NotImplementedError

    def drop(self):
        """Drop the crontab."""
        raise NotImplementedError

    def initialize(self):
        """Initialize the crontab."""
