from pymongo.mongo_client import MongoClient
import gridfs

from . import database
from sddb import cf


class SddbClient(MongoClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.args = args
        self.kwargs = kwargs
        self.filesystem = gridfs.GridFS(self['_sddb_files'])

    def __repr__(self):
        return f'SddbClient({self.args, self.kwargs})'

    def __getitem__(self, name: str):
        return database.Database(self, name)


client = SddbClient(**cf['mongodb'])
