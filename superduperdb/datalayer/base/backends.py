from pymongo import MongoClient

from superduperdb.datalayer.mongodb.artifacts import MongoArtifactStore
from superduperdb.datalayer.mongodb.data_backend import MongoDataBackend
from superduperdb.datalayer.mongodb.metadata import MongoMetaDataStore
from superduperdb.misc import config
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
