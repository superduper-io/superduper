import typing as t
from abc import abstractmethod

from superduper.backends.base.backends import BaseBackend

if t.TYPE_CHECKING:
    from superduper.base.datalayer import Datalayer


class CDCBackend(BaseBackend):
    """Base backend for CDC."""

    @abstractmethod
    def handle_event(self, event_type, table, ids):
        """Abstract method to handle events."""

    @property
    def db(self) -> 'Datalayer':
        """Datalayer instance property."""
        return self._db

    @db.setter
    def db(self, value: 'Datalayer'):
        self._db = value
        self.initialize()
