from pymongo import MongoClient

from superduperdb.base import config
from superduperdb.db.mongodb.artifacts import MongoArtifactStore
from superduperdb.db.mongodb.data_backend import MongoDataBackend
from superduperdb.db.mongodb.metadata import MongoMetaDataStore
from superduperdb.vector_search.inmemory import InMemoryVectorDatabase
from superduperdb.vector_search.lancedb_client import LanceVectorIndex

DATA_BACKENDS = {'mongodb': MongoDataBackend}

ARTIFACT_STORES = {'mongodb': MongoArtifactStore}

METADATA_STORES = {'mongodb': MongoMetaDataStore}

VECTOR_DATA_STORES = {
    config.LanceDB: LanceVectorIndex,
    config.InMemory: InMemoryVectorDatabase,
}

CONNECTIONS = {
    'pymongo': MongoClient,
}
