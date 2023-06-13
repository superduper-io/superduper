from pymongo.collection import Collection as MongoCollection

from superduperdb.datalayer.mongodb.query import Insert, Update, Delete


class Collection(MongoCollection):
    """
    Collection building on top of ``pymongo.collection.Collection``.
    Implements additional methods required by ``superduperdb`` for AI/ machine learning.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _base_delete(self, delete: Delete):
        if delete.one:
            return super().delete_one(delete.filter)
        else:
            return super().delete_many(delete.filter)

    def _base_find(self, *args, **kwargs):
        return super().find(*args, **kwargs)

    def _base_update(self, update: Update):
        if update.replacement is not None:
            return super().replace_one(update.filter, update.replacement.encode())
        if update.one:
            return super().update_one(update.filter, update.update.encode())
        return super().update_many(update.filter, update.update.encode())

    @property
    def remote(self):
        return self.database.remote

    @remote.setter
    def remote(self, value):
        self.database.remote = value

    def __getitem__(self, item):
        if item == '_validation_sets':
            return self.database[f'{self.name}._validation_sets']
        return super().__getitem__(item)

    def _base_insert_many(self, insert: Insert):
        encoded = [r.encode() for r in insert.documents]
        return super().insert_many(
            encoded,
            ordered=insert.ordered,
            bypass_document_validation=insert.bypass_document_validation,
        )
