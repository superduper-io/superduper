# TODO deprecate the implementations in favour of the plugin paradigm
from abc import ABC, abstractmethod
import typing as t
from superduper.vector_search.atlas import MongoAtlasVectorSearcher
from superduper.vector_search.lance import LanceVectorSearcher
from superduper.vector_search.qdrant import QdrantVectorSearcher

if t.TYPE_CHECKING:
    from superduper.components.component import Component
    from superduper.base.datalayer import Datalayer

vector_searcher_implementations = {
    "lance": LanceVectorSearcher,
    "mongodb+srv": MongoAtlasVectorSearcher,
    "qdrant": QdrantVectorSearcher,
}


class BaseBackend(ABC):
    """Base backend class for cluster client."""
    def __init__(self):
        self._db = None

    @abstractmethod
    def drop(self):
        pass

    @abstractmethod
    def list_uuids(self):
        pass

    @abstractmethod
    def list_components(self):
        pass

    @abstractmethod
    def _put(self, item):
        pass

    @abstractmethod
    def __delitem__(self, item):
        pass

    @abstractmethod
    def initialize(self):
        """To be called on program start."""
        pass

    def put(self, component: 'Component', **kwargs):
        # This is to make sure that we only have 1 version 
        # of each component implemented at any given time
        # TODO: get identifier in string component argument.
        identifier = ''
        if isinstance(component, str):
            uuid = component
        else:
            uuid = component.uuid
            identifier = component.identifier

        if uuid in self.list_uuids():
            return
        if identifier in self.list_components():
            del self[component.identifier]
        self._put(component, **kwargs)

    @property
    def db(self) -> 'Datalayer':
        return self._db

    @db.setter
    def db(self, value: 'Datalayer'):
        self._db = value

    
