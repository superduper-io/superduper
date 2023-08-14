import typing as t

import click
from pymongo import MongoClient

from superduperdb import logging
from superduperdb.container.serializable import Serializable
from superduperdb.db.base.data_backend import BaseDataBackend
from superduperdb.misc.colors import Colors
from superduperdb.misc.special_dicts import MongoStyleDict

from .metadata import MongoMetaDataStore
from .artifacts import MongoArtifactStore


class MongoDataBackend(BaseDataBackend):
    id_field = '_id'

    def __init__(self, conn: MongoClient, name: str):
        super().__init__(conn=conn, name=name)
        self._db = self.conn[self.name]

    @property
    def db(self):
        return self._db

    @property
    def default_metadata_store(self):
        return MongoMetaDataStore(self.db.client, name=self.db.name)

    @property
    def default_artifact_store(self):
        return MongoArtifactStore(self.db.client, name=f'_filesystem:{self.db.name}')

    def drop(self, force: bool = False):
        if not force:
            if not click.confirm(
                f'{Colors.RED}[!!!WARNING USE WITH CAUTION AS YOU '
                f'WILL LOSE ALL DATA!!!]{Colors.RESET} '
                'Are you sure you want to drop the data-backend? ',
                default=False,
            ):
                print('Aborting...')
        return self.db.client.drop_database(self.db.name)

    def set_content_bytes(self, r, key, bytes_):
        if not isinstance(r, MongoStyleDict):
            r = MongoStyleDict(r)
        r[f'{key}._content.bytes'] = bytes_
        return r

    def unset_outputs(self, info: t.Dict):
        select = Serializable.deserialize(info['select'])
        logging.info(f'unsetting output field _outputs.{info["key"]}.{info["model"]}')
        doc = {'$unset': {f'_outputs.{info["key"]}.{info["model"]}': 1}}

        update = select.update(doc)
        return self.db[select.collection].update_many(update.filter, update.update)
