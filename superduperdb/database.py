from pymongo.database import Database as BaseDatabase
import superduperdb.collection


class Database(BaseDatabase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, name: str):
        return superduperdb.collection.Collection(self, name)
