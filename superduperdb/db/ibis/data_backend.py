import typing as t

from superduperdb import logging
from superduperdb.container.document import Document
from superduperdb.container.serializable import Serializable
from superduperdb.db.base.data_backend import BaseDataBackend
from superduperdb.misc.special_dicts import MongoStyleDict


class IbisDataBackend(BaseDataBackend):
    id_field = 'id'

    def __init__(self, conn, name: str):
        super().__init__(conn=conn, name=name)
        # Get database.
        self._db = self.conn[self.name]

    @property
    def db(self):
        return self._db
