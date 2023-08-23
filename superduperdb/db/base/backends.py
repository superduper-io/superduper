from pymongo import MongoClient
from ibis.backends.base import BaseBackend

from superduperdb.base import config
from superduperdb.db.mongodb.artifacts import MongoArtifactStore
from superduperdb.db.mongodb.data_backend import MongoDataBackend
from superduperdb.db.filesystem.artifacts import FileSystemArtifactStore
from superduperdb.db.sqlalchemy.metadata import  SQLAlchemyMetadata
from superduperdb.db.ibis.data_backend import IbisDataBackend
from superduperdb.db.mongodb.metadata import MongoMetaDataStore
from superduperdb.vector_search.inmemory import InMemoryVectorDatabase
from superduperdb.vector_search.lancedb_client import LanceVectorIndex

data_backends = {'mongodb': MongoDataBackend, 'ibis': IbisDataBackend}

artifact_stores = {'mongodb': MongoArtifactStore, 'filesystem': FileSystemArtifactStore}

metadata_stores = {'mongodb': MongoMetaDataStore, 'sqlalchemy': SQLAlchemyMetadata}

vector_database_stores = {
    config.LanceDB: LanceVectorIndex,
    config.InMemory: InMemoryVectorDatabase,
}

connections = {
    'pymongo': MongoClient,
    'ibis': BaseBackend,
}
