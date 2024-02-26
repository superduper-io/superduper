import os
import shutil
import typing as t
from pathlib import Path

import click

from superduperdb import logging
from superduperdb.backends.base.artifact import ArtifactStore
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

    def _exists(self, file_id: str):
        path = os.path.join(self.conn, file_id)
        return os.path.exists(path)

    def url(self):
        return self.conn

    def _delete_artifact(self, file_id: str):
        """
        Delete artifact from artifact store
        :param file_id: File id uses to identify artifact in store
        """
        path = os.path.join(self.conn, file_id)
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)

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
                logging.warn('Aborting...')
        shutil.rmtree(self.conn, ignore_errors=force)
        os.makedirs(self.conn)

    def _save_bytes(
        self,
        serialized: bytes,
        file_id: str,
    ) -> t.Any:
        with open(os.path.join(self.conn, file_id), 'wb') as f:
            f.write(serialized)

    def _load_bytes(self, file_id: str) -> bytes:
        with open(os.path.join(self.conn, file_id), 'rb') as f:
            return f.read()

    def _save_file(self, file_path: str, file_id: str):
        """
        Save file in artifact store and return the relative path
        return the relative path {file_id}/{name}
        """
        path = Path(file_path)
        name = path.name
        file_id_folder = os.path.join(self.conn, file_id)
        os.makedirs(file_id_folder, exist_ok=True)
        save_path = os.path.join(file_id_folder, name)
        logging.info(f"Copying file {file_path} to {save_path}")
        if path.is_dir():
            shutil.copytree(file_path, save_path)
        else:
            shutil.copy(file_path, save_path)
        # return the relative path {file_id}/{name}
        return os.path.join(file_id, name)

    def _load_file(self, file_id: str) -> str:
        """Return the path to the file in the artifact store"""
        logging.info(f"Loading file {file_id} from {self.conn}")
        return os.path.join(self.conn, file_id)

    def disconnect(self):
        """
        Disconnect the client
        """
        # Not necessary since just local filesystem
        pass
