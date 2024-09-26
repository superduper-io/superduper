from abc import abstractmethod

from superduper.backends.base.backends import BaseBackend


class CDCBackend(BaseBackend):
    """Base backend for CDC."""

    @abstractmethod
    def handle_event(self, event_type, table, ids):
        pass

    @property
    def db(self) -> 'Datalayer':
        return self._db

    @db.setter
    def db(self, value: 'Datalayer'):
        self._db = value
        self.initialize()
