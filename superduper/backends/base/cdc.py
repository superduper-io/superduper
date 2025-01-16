import typing as t
from abc import abstractmethod

from superduper.backends.base.backends import BaseBackend

if t.TYPE_CHECKING:
    from superduper.base.datalayer import Datalayer


class CDCBackend(BaseBackend):
    """Base backend for CDC."""

    @abstractmethod
    def handle_event(self, event_type, table, ids):
        """Handle an incoming event.

        :param event_type: The type of event.
        :param table: The table to handle.
        :param ids: The ids to handle.
        """
        pass

    @property
    def db(self) -> 'Datalayer':
        """Get the ``db``."""
        return self._db

    @db.setter
    def db(self, value: 'Datalayer'):
        """Set the ``db``.

        :param value: ``Datalayer`` instance.
        """
        self._db = value
