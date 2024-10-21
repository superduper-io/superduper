import os
import shutil
import typing as t
from pathlib import Path

import click

from superduper import logging
from superduper.backends.base.artifacts import ArtifactStore
from superduper.misc.colors import Colors


class FileSystemArtifactStore(ArtifactStore):
    """
    Abstraction for storing large artifacts separately from primary data.

    :param conn: root directory of the artifact store
    :param name: subdirectory to use for this artifact store
    :param flavour: Flavour of the artifact store
    """

    def __init__(
        self,
        conn: t.Any,
        name: t.Optional[str] = None,
        flavour: t.Optional[str] = None,
    ):
        if conn.startswith('filesystem://'):
            conn = conn.split('filesystem://')[-1]
        super().__init__(conn, name, flavour)
        if not os.path.exists(self.conn):
            logging.info('Creating artifact store directory')
            os.makedirs(self.conn, exist_ok=True)

    def _exists(self, file_id: str):
        path = os.path.join(self.conn, file_id)
        return os.path.exists(path)

    def url(self):
        """Return the URL of the artifact store."""
        return self.conn

    def _delete_bytes(self, file_id: str):
        """Delete artifact from artifact store.

        :param file_id: File id uses to identify artifact in store
        """
        path = os.path.join(self.conn, file_id)
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)

    def drop(self, force: bool = False):
        """Drop the artifact store.

        Please use with caution as this will delete all data in the artifact store.

        :param force: Whether to force the drop.
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
        if os.path.exists(self.conn):
            logging.warn('Failed to drop artifact store')
        os.makedirs(self.conn, exist_ok=True)

    def put_bytes(
        self,
        serialized: bytes,
        file_id: str,
    ) -> t.Any:
        """
        Save bytes in artifact store.

        :param serialized: The bytes to be saved.
        :param file_id: The id of the file.
        """
        path = os.path.join(self.conn, file_id)
        if os.path.exists(path):
            logging.warn(f"File {path} already exists")

        with open(path, 'wb') as f:
            f.write(serialized)
        os.chmod(path, 0o777)

    def get_bytes(self, file_id: str) -> bytes:
        """
        Return the bytes from the artifact store.

        :param file_id: The id of the file.
        """
        with open(os.path.join(self.conn, file_id), 'rb') as f:
            return f.read()

    def put_file(self, file_path: str, file_id: str):
        """Save file in artifact store and return the relative path.

        return the relative path {file_id}/{name}

        :param file_path: The path to the file to be saved.
        :param file_id: The id of the file.
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
        os.chmod(save_path, 0o777)
        return file_id

    def get_file(self, file_id: str) -> str:
        """Return the path to the file in the artifact store.

        :param file_id: The id of the file.
        """
        logging.info(f"Loading file {file_id} from {self.conn}")
        path = os.path.join(self.conn, file_id)
        files = os.listdir(path)
        assert len(files) == 1, f"Expected 1 file, got {len(files)}"
        name = files[0]
        return os.path.join(path, name)

    def disconnect(self):
        """Disconnect the client."""
        # Not necessary since just local filesystem
        pass
