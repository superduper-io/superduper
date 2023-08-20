import hashlib
import os
import shutil
import typing as t

import click
from superduperdb.misc.colors import Colors

from superduperdb.misc.serialization import Info, serializers
from superduperdb.db.base.artifact import ArtifactStore


class FilesystemArtifactStore(ArtifactStore):
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
        assert self.conn.startswith('./')

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
        h = hashlib.sha1(serialized).hexdigest()
        with open(f'{self.conn}/{h}', 'wb') as f:
            f.write(serialized)
        return h

    def _load_bytes(self, file_id: str) -> bytes:
        with open(f'{self.conn}/{file_id}', 'rb') as f:
            return f.read()