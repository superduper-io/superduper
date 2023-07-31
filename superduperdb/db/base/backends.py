from pymongo import MongoClient

from superduperdb.base import config
from superduperdb.db.mongodb.artifacts import MongoArtifactStore
from superduperdb.db.mongodb.data_backend import MongoDataBackend
from superduperdb.db.mongodb.metadata import MongoMetaDataStore
from superduperdb.vector_search.inmemory import InMemoryVectorDatabase
from superduperdb.vector_search.lancedb_client import LanceVectorIndex

data_backends = {'mongodb': MongoDataBackend}

artifact_stores = {'mongodb': MongoArtifactStore}

metadata_stores = {'mongodb': MongoMetaDataStore}

vector_database_stores = {
    config.LanceDB: LanceVectorIndex,
    config.InMemory: InMemoryVectorDatabase,
}

connections = {
    'pymongo': MongoClient,
}
