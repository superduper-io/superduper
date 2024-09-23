from superduper.backends.base.crontab import CrontabBackend


class LocalCrontabBackend(CrontabBackend):
    """Local crontab backend."""

    def _put(self, item):
        raise NotImplementedError

    def list(self):
        raise NotImplementedError

    def __delitem__(self, item):
        raise NotImplementedError

    def list_components(self):
        raise NotImplementedError

    def list_uuids(self):
        raise NotImplementedError

    def drop(self):
        raise NotImplementedError

    def initialize(self):
        raise NotImplementedError
