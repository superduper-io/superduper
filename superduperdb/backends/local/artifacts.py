import os
import shutil
import typing as t
import uuid

import click

from superduperdb import logging
from superduperdb.backends.base.artifact import ArtifactStore
from superduperdb.base import exceptions
from superduperdb.misc.colors import Colors


class FileSystemArtifactStore(ArtifactStore):
    """
    Abstraction for storing large artifacts separately from primary data.

    :param conn: root directory of the artifact store
    :param name: subdirectory to use for this artifact store
    """

    def __init__(
        self,
        conn: t.Any,
        name: t.Optional[str] = None,
    ):
        self.name = name
        self.conn = conn
        if not os.path.exists(self.conn):
            logging.info('Creating artifact store directory')
            os.makedirs(self.conn, exist_ok=True)

    def url(self):
        return self.conn

    def delete(self, file_id: str):
        """
        Delete artifact from artifact store
        :param file_id: File id uses to identify artifact in store
        """
        try:
            os.remove(f'{self.conn}/{file_id}')
        except Exception as e:
            raise exceptions.ArtifactStoreDeleteException(
                f'Error while deleting {file_id}'
            ) from e

    def drop(self, force: bool = False):
        """
        Drop the artifact store.
        """
        try:
            if not force:
                if not click.confirm(
                    f'{Colors.RED}[!!!WARNING USE WITH CAUTION AS YOU '
                    f'WILL LOSE ALL DATA!!!]{Colors.RESET} '
                    'Are you sure you want to drop all artifacts? ',
                    default=False,
                ):
                    logging.warn('Aborting...')
            shutil.rmtree(self.conn, ignore_errors=force)
        except Exception as e:
            raise exceptions.ArtifactStoreDeleteException(
                'Error while dropping in artifact store'
            ) from e

    def save_artifact(self, serialized: bytes) -> t.Any:
        try:
            h = uuid.uuid4().hex
            with open(os.path.join(self.conn, h), 'wb') as f:
                f.write(serialized)
            return h
        except Exception as e:
            raise exceptions.ArtifactStoreSaveException(
                'Error while saving artifacts'
            ) from e

    def load_bytes(self, file_id: str) -> bytes:
        try:
            with open(os.path.join(self.conn, file_id), 'rb') as f:
                return f.read()
        except Exception as e:
            raise exceptions.ArtifactStoreLoadException(
                'Error while loading artifacts'
            ) from e

    def disconnect(self):
        """
        Disconnect the client
        """

        # TODO: implement me
