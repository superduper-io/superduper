from pymongo import MongoClient

from superduperdb.datalayer.mongodb.artifacts import MongoArtifactStore
from superduperdb.datalayer.mongodb.data_backend import MongoDataBackend
from superduperdb.datalayer.mongodb.metadata import MongoMetaDataStore


data_backends = {'mongodb': MongoDataBackend}

artifact_stores = {'mongodb': MongoArtifactStore}

metadata_stores = {'mongodb': MongoMetaDataStore}

connections = {
    'pymongo': MongoClient,
}
