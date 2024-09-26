from superduper.backends.base.crontab import CrontabBackend


class LocalCrontabBackend(CrontabBackend):
    """Local crontab backend."""

    def _put(self, item):
        """Put component."""
        raise NotImplementedError

    def list(self):
        """List all components."""
        raise NotImplementedError

    def __delitem__(self, item):
        raise NotImplementedError

    def list_components(self):
        """List components."""
        raise NotImplementedError

    def list_uuids(self):
        """List uuids."""
        raise NotImplementedError

    def drop(self):
        """Drop component."""
        raise NotImplementedError

    def initialize(self):
        """Initialize."""
        raise NotImplementedError
