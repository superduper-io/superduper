from pymongo.mongo_client import MongoClient

import superduperdb.mongodb.database


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
        return superduperdb.mongodb.database.Database(self, name)

    def get_database_from_name(self, name):
        return self[name]
