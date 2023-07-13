from pymongo.database import Database as BaseDatabase
import superduperdb.collection


class Database(BaseDatabase):
    """
    Database building on top of :code:`pymongo.database.Database`. Collections in the
    database are SuperDuperDB objects :code:`superduperdb.collection.Collection`.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, name: str):
        return superduperdb.collection.Collection(self, name)
