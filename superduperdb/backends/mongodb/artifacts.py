import click
import gridfs

from superduperdb import logging
from superduperdb.backends.base.artifact import ArtifactStore
from superduperdb.base import exceptions
from superduperdb.misc.colors import Colors


class MongoArtifactStore(ArtifactStore):
    """
    Artifact store for MongoDB.

    :param conn: MongoDB client connection
    :param name: Name of database to host filesystem
    """

    def __init__(self, conn, name: str):
        super().__init__(name=name, conn=conn)
        self.db = self.conn[self.name]
        self.filesystem = gridfs.GridFS(self.db)

    def url(self):
        return self.conn.HOST + ':' + str(self.conn.PORT) + '/' + self.name

    def drop(self, force: bool = False):
        try:
            if not force:
                if not click.confirm(
                    f'{Colors.RED}[!!!WARNING USE WITH CAUTION AS YOU '
                    f'WILL LOSE ALL DATA!!!]{Colors.RESET} '
                    'Are you sure you want to drop all artifacts? ',
                    default=False,
                ):
                    logging.warn('Aborting...')
            return self.db.client.drop_database(self.db.name)
        except Exception as e:
            raise exceptions.ArtifactStoreDeleteException(
                'Error while dropping in artifact store'
            ) from e

    def delete(self, file_id: str):
        try:
            return self.filesystem.delete(file_id)
        except Exception as e:
            raise exceptions.ArtifactStoreDeleteException(
                'Error while dropping in artifact store'
            ) from e

    def load_bytes(self, file_id: str):
        try:
            return self.filesystem.get(file_id).read()
        except Exception as e:
            raise exceptions.ArtifactStoreLoadException(
                'Error while saving artifacts'
            ) from e

    def save_artifact(self, serialized: bytes):
        try:
            return self.filesystem.put(serialized)
        except Exception as e:
            raise exceptions.ArtifactStoreSaveException(
                'Error while saving artifacts'
            ) from e

    def disconnect(self):
        """
        Disconnect the client
        """

        # TODO: implement me
