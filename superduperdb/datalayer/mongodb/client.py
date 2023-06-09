from typing import Optional

import click
from pymongo.mongo_client import MongoClient

import superduperdb.datalayer.mongodb.database
from superduperdb.datalayer.base.artifacts import ArtifactStore
from superduperdb.datalayer.base.metadata import MetaDataStore
from superduperdb.datalayer.mongodb.artifacts import MongoArtifactStore
from superduperdb.datalayer.mongodb.metadata import MongoMetaDataStore
from superduperdb.misc.logger import logging


class SuperDuperClient(MongoClient):
    """
    Client building on top of :code:`pymongo.MongoClient`.

    Databases and collections in the client are SuperDuperDB objects.
    """

    def __init__(
        self,
        *args,
        user=None,
        artifact_store: Optional[ArtifactStore] = None,
        metadata: Optional[MetaDataStore] = None,
        **kwargs,
    ):
        self.artifact_store = artifact_store
        self.metadata = metadata
        if user is not None:
            kwargs.setdefault('username', user)
        super().__init__(*args, **kwargs)
        self.args = args
        self.kwargs = kwargs

    def __getitem__(self, name: str):
        artifact_store = self.artifact_store
        if artifact_store is None:
            artifact_db = super().__getitem__(f'_{name}:files')
            artifact_store = MongoArtifactStore(artifact_db)
        metadata = self.metadata
        if metadata is None:
            metadata_db = super().__getitem__(name)
            metadata = MongoMetaDataStore(
                metadata_db, '_objects', '_meta', '_jobs', '_parent_child_mappings'
            )
        return superduperdb.datalayer.mongodb.database.Database(
            artifact_store,
            metadata,
            self,
            name,
        )

    def get_database_from_name(self, name):
        return self[name]

    def list_database_names(self, **kwargs):
        names = super().list_database_names(**kwargs)
        names = [
            x
            for x in names
            if (not x.endswith(':files') and x not in {'admin', 'local', 'config'})
        ]
        return names

    def drop_database(
        self,
        name,
        force=False,
        **kwargs,
    ):
        if force or click.confirm(
            'Are you sure you want to delete this database and all its models?',
            default=False,
        ):
            super().drop_database(f'_{name}:files')
            super().drop_database(name)
        else:
            logging.warning('aborting...')
