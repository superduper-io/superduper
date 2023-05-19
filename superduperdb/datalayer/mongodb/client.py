import click
from pymongo.mongo_client import MongoClient

import superduperdb.datalayer.mongodb.database
from superduperdb.misc.logger import logging


class SuperDuperClient(MongoClient):
    """
    Client building on top of :code:`pymongo.MongoClient`. Databases and collections in the
    client are SuperDuperDB objects.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.args = args
        self.kwargs = kwargs

    def __getitem__(self, name: str):
        return superduperdb.datalayer.mongodb.database.Database(self, name)

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
            'are you sure you want to delete this database and all of the models, etc. in it?',
            default=False,
        ):
            super().drop_database(f'_{name}:files')
            super().drop_database(name)
        else:
            logging.warning('aborting...')
