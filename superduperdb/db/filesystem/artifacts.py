import os
import shutil
import typing as t
import uuid

import click

from superduperdb.db.base.artifact import ArtifactStore
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
            print('Creating artifact store directory')
            os.makedirs(self.conn, exist_ok=True)

    def delete_artifact(self, file_id: str):
        """
        Delete artifact from artifact store
        :param file_id: File id uses to identify artifact in store
        """
        os.remove(f'{self.conn}/{file_id}')

    def drop(self, force: bool = False):
        """
        Drop the artifact store.
        """
        if not force:
            if not click.confirm(
                f'{Colors.RED}[!!!WARNING USE WITH CAUTION AS YOU '
                f'WILL LOSE ALL DATA!!!]{Colors.RESET} '
                'Are you sure you want to drop all artifacts? ',
                default=False,
            ):
                print('Aborting...')
        shutil.rmtree(self.conn, ignore_errors=force)

    def _save_artifact(self, serialized: bytes) -> t.Any:
        h = uuid.uuid4().hex
        with open(os.path.join(self.conn, h), 'wb') as f:
            f.write(serialized)
        return h

    def _load_bytes(self, file_id: str) -> bytes:
        with open(os.path.join(self.conn, file_id), 'rb') as f:
            return f.read()
