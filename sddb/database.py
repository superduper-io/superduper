from pymongo.database import Database as BaseDatabase
from pymongo.collection import Collection as BaseCollection
from sddb import collection


class Database(BaseDatabase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, name: str):
        if name.endswith('_outputs') or name.endswith('_models') or name.endswith('_meta'):
            return BaseCollection(self, name)
        else:
            return collection.Collection(self, name)
