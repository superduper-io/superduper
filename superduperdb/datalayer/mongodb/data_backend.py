import typing as t

from pymongo import MongoClient

from superduperdb.core.documents import Document
from superduperdb.datalayer.base.data_backend import BaseDataBackend
from superduperdb.misc.special_dicts import MongoStyleDict
from superduperdb.misc.logger import logging
from superduperdb.queries.serialization import from_dict


class MongoDataBackend(BaseDataBackend):
    id_field = '_id'

    def __init__(self, conn: MongoClient, name: str):
        super().__init__(conn=conn, name=name)
        self._db = self.conn[self.name]

    @property
    def db(self):
        return self._db

    def get_output_from_document(
        self, r: Document, key: str, model: str
    ) -> MongoStyleDict:
        return (
            MongoStyleDict(r.content)[f'_outputs.{key}.{model}'],  # type: ignore
            r.content['_id'],  # type: ignore
        )

    def set_content_bytes(self, r, key, bytes_):
        if not isinstance(r, MongoStyleDict):
            r = MongoStyleDict(r)
        r[f'{key}._content.bytes'] = bytes_
        return r

    def unset_outputs(self, info: t.Dict):
        select = from_dict(info['select'])
        logging.info(f'unsetting output field _outputs.{info["key"]}.{info["model"]}')
        doc = {'$unset': {f'_outputs.{info["key"]}.{info["model"]}': 1}}

        # doc = Document(doc)
        # TODO: The above looks like it should be correct, and fixes the mypy error, but
        # not passing in the dict breaks two unit tests!

        update = select.update(doc)  # type: ignore
        return self.db[select.collection].update_many(update.filter, update.update)
