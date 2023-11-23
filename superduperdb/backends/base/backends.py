from ibis.backends.base import BaseBackend
from pymongo import MongoClient

from superduperdb.backends.ibis.data import IbisDataStore
from superduperdb.backends.local.artifacts import FileSystemArtifactStore
from superduperdb.backends.mongodb.artifacts import MongoArtifactStore
from superduperdb.backends.mongodb.data import MongoDataStore
from superduperdb.backends.mongodb.metadata import MongoMetadataStore
from superduperdb.backends.sqlalchemy.metadata import SQLAlchemyMetadata
from superduperdb.vector_search.in_memory import InMemoryVectorSearcher
from superduperdb.vector_search.lance import LanceVectorSearcher

data_stores = {'mongodb': MongoDataStore, 'ibis': IbisDataStore}

artifact_stores = {'mongodb': MongoArtifactStore, 'filesystem': FileSystemArtifactStore}

metadata_stores = {'mongodb': MongoMetadataStore, 'sqlalchemy': SQLAlchemyMetadata}

vector_searcher_implementations = {
    'lance': LanceVectorSearcher,
    'in_memory': InMemoryVectorSearcher,
}

CONNECTIONS = {
    'pymongo': MongoClient,
    'ibis': BaseBackend,
}
