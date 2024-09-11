from abc import ABC, abstractmethod
from superduper.backends.base.backends import BaseBackend


class CDCBackend(BaseBackend):
    """Base backend for CDC."""

    @abstractmethod
    def handle_event(self, event_type, query, ids):
        pass

