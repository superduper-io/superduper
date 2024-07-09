from ibis.backends import BaseBackend
from pymongo import MongoClient

from superduper.backends.ibis.data_backend import IbisDataBackend
from superduper.backends.local.artifacts import FileSystemArtifactStore
from superduper.backends.mongodb.artifacts import MongoArtifactStore
from superduper.backends.mongodb.data_backend import MongoDataBackend
from superduper.backends.mongodb.metadata import MongoMetaDataStore
from superduper.backends.sqlalchemy.metadata import SQLAlchemyMetadata
from superduper.vector_search.atlas import MongoAtlasVectorSearcher
from superduper.vector_search.in_memory import InMemoryVectorSearcher
from superduper.vector_search.lance import LanceVectorSearcher

data_backends = {
    'mongodb': MongoDataBackend,
    'ibis': IbisDataBackend,
}

artifact_stores = {
    'mongodb': MongoArtifactStore,
    'filesystem': FileSystemArtifactStore,
}

metadata_stores = {
    'mongodb': MongoMetaDataStore,
    'sqlalchemy': SQLAlchemyMetadata,
}

vector_searcher_implementations = {
    'lance': LanceVectorSearcher,
    'in_memory': InMemoryVectorSearcher,
    'mongodb+srv': MongoAtlasVectorSearcher,
}

CONNECTIONS = {
    'pymongo': MongoClient,
    'ibis': BaseBackend,
}
