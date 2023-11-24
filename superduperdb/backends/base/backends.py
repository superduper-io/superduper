from ibis.backends.base import BaseBackend
from pymongo import MongoClient

from superduperdb.backends.dask.compute import DaskComputeEngine
from superduperdb.backends.ibis.data import IbisDataStore
from superduperdb.backends.local.artifacts import FileSystemArtifactStore
from superduperdb.backends.local.compute import LocalComputeEngine
from superduperdb.backends.mongodb.artifacts import MongoArtifactStore
from superduperdb.backends.mongodb.data import MongoDataStore
from superduperdb.backends.mongodb.metadata import MongoMetadataStore
from superduperdb.backends.sqlalchemy.metadata import SQLAlchemyMetadata
from superduperdb.vector_search.in_memory import InMemoryVectorSearchEngine
from superduperdb.vector_search.lance import LanceVectorSearchEngine

data_stores = {
    'mongodb': MongoDataStore,
    'ibis': IbisDataStore,
}

metadata_stores = {
    'mongodb': MongoMetadataStore,
    'sqlalchemy': SQLAlchemyMetadata,
}

artifact_stores = {
    'mongodb': MongoArtifactStore,
    'filesystem': FileSystemArtifactStore,
}

compute_engines = {
    'local',
    LocalComputeEngine,
    'dask',
    DaskComputeEngine,
}

vector_search_engines = {
    'lance': LanceVectorSearchEngine,
    'in_memory': InMemoryVectorSearchEngine,
}

CONNECTIONS = {
    'pymongo': MongoClient,
    'ibis': BaseBackend,
}
