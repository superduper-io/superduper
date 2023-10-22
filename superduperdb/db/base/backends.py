from ibis.backends.base import BaseBackend
from pymongo import MongoClient

from superduperdb.db.filesystem.artifacts import FileSystemArtifactStore
from superduperdb.db.ibis.data_backend import IbisDataBackend
from superduperdb.db.mongodb.artifacts import MongoArtifactStore
from superduperdb.db.mongodb.data_backend import MongoDataBackend
from superduperdb.db.mongodb.metadata import MongoMetaDataStore
from superduperdb.db.sqlalchemy.metadata import SQLAlchemyMetadata
from superduperdb.vector_search.in_memory import InMemoryVectorSearcher
from superduperdb.vector_search.lance import LanceVectorSearcher

data_backends = {'mongodb': MongoDataBackend, 'ibis': IbisDataBackend}

artifact_stores = {'mongodb': MongoArtifactStore, 'filesystem': FileSystemArtifactStore}

metadata_stores = {'mongodb': MongoMetaDataStore, 'sqlalchemy': SQLAlchemyMetadata}

vector_searcher_implementations = {
    'lance': LanceVectorSearcher,
    'in_memory': InMemoryVectorSearcher,
}

CONNECTIONS = {
    'pymongo': MongoClient,
    'ibis': BaseBackend,
}
