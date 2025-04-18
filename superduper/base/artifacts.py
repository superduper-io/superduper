import os
import shutil
import typing as t
from abc import ABC, abstractmethod
from pathlib import Path

import click

from superduper import logging
from superduper.base.constant import KEY_BLOBS, KEY_FILES


class ArtifactStore(ABC):
    """
    Abstraction for storing large artifacts separately from primary data.

    :param conn: connection to the meta-data store
    :param name: Name to identify DB using the connection
    :param flavour: Flavour of the artifact store
    """

    def __init__(
        self,
        conn: t.Any,
        name: t.Optional[str] = None,
        flavour: t.Optional[str] = None,
    ):
        self.name = name
        self.conn = conn
        self._db = None
        self.flavour = flavour

    @property
    def db(self):
        """Return the db."""
        assert self._db is not None, 'db not initialized!'
        return self._db

    @db.setter
    def db(self, value):
        """Set the db.

        :param value: The db.
        """
        self._db = value

    @abstractmethod
    def url(self):
        """Artifact store connection url."""
        pass

    @abstractmethod
    def _delete_bytes(self, file_id: str):
        """Delete artifact from artifact store.

        :param file_id: File id uses to identify artifact in store
        """

    @abstractmethod
    def drop(self, force: bool = False):
        """
        Drop the artifact store.

        :param force: If ``True``, don't ask for confirmation
        """
        pass

    @abstractmethod
    def put_bytes(self, serialized: bytes, file_id: str):
        """Save bytes in artifact store.

        :param serialized: Bytes to save
        :param file_id: Identifier of artifact in the store
        """
        pass

    @abstractmethod
    def put_file(self, file_path: str, file_id: str) -> str:
        """Save file in artifact store and return file_id.

        :param file_path: Path to file
        :param file_id: Identifier of artifact in the store
        """
        pass

    def save_artifact(self, r: t.Dict):
        """Save serialized object in the artifact store.

        :param r: dictionary with mandatory fields
        """
        blobs = r.get(KEY_BLOBS, {})
        files = r.get(KEY_FILES, {})

        for file_id, blob in blobs.items():
            if blob is None:
                continue
            try:
                self.put_bytes(blob, file_id=file_id)
            except FileExistsError:
                continue

        for file_id, file_path in files.items():
            try:
                self.put_file(file_path, file_id=file_id)
            except (FileExistsError, shutil.SameFileError):
                continue

        # After we save the artifacts, we can remove the blobs and files
        # TODO move this logic
        r[KEY_FILES] = {}
        r[KEY_BLOBS] = {}

        return r

    def delete_artifact(self, artifact_ids: t.List[str]):
        """Delete artifact from artifact store.

        :param artifact_ids: list of artifact ids to delete.
        """
        for artifact_id in artifact_ids:
            try:
                self._delete_bytes(artifact_id)
            except FileNotFoundError:
                logging.warn(f'Blob {artifact_id} not found in artifact store')

    @abstractmethod
    def get_bytes(self, file_id: str) -> bytes:
        """
        Load bytes from artifact store.

        :param file_id: Identifier of artifact in the store
        """
        pass

    @abstractmethod
    def get_file(self, file_id: str) -> str:
        """
        Load file from artifact store and return path.

        :param file_id: Identifier of artifact in the store
        """
        pass

    @abstractmethod
    def disconnect(self):
        """Disconnect the client."""
        pass

    @abstractmethod
    def list(self):
        """List all artifacts in the store."""


class FileSystemArtifactStore(ArtifactStore):
    """
    Abstraction for storing large artifacts separately from primary data.

    :param conn: Root directory of the artifact store.
    :param name: Name of the artifact store.
    :param flavour: Flavour of the artifact store.
    :param files: Subdirectory to use for files.
    :param blobs: Subdirectory to use for blobs.
    """

    def __init__(
        self,
        conn: t.Any,
        name: t.Optional[str] = None,
        flavour: t.Optional[str] = None,
        files: str = '',
        blobs: str = '',
    ):
        if conn.startswith('filesystem://'):
            conn = conn.split('filesystem://')[-1]
        super().__init__(conn, name, flavour)

        if not os.path.exists(self.conn):
            logging.info('Creating artifact store directory')
            os.makedirs(self.conn, exist_ok=True)

        self.files = os.path.join(self.conn, files) if files else self.conn
        if self.files != self.conn and not os.path.exists(self.files):
            logging.info('Creating file store directory')
            os.makedirs(self.files, exist_ok=True)

        self.blobs = os.path.join(self.conn, blobs) if blobs else self.conn
        if self.blobs != self.conn and not os.path.exists(self.blobs):
            logging.info('Creating file store directory')
            os.makedirs(self.blobs, exist_ok=True)

    # def _exists(self, file_id: str):
    #     path = os.path.join(self.conn, file_id)
    #     return os.path.exists(path)

    def url(self):
        """Return the URL of the artifact store."""
        return self.conn

    def _delete_bytes(self, file_id: str):
        """Delete artifact from artifact store.

        :param file_id: File id uses to identify artifact in store
        """
        path = os.path.join(self.blobs, file_id)
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
                '!!!WARNING USE WITH CAUTION AS YOU '
                'WILL LOSE ALL DATA!!!]\n'
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
        path = os.path.join(self.blobs, file_id)
        if os.path.exists(path):
            logging.debug(f"File {path} already exists")

        with open(path, 'wb') as f:
            f.write(serialized)
        os.chmod(path, 0o777)

    def get_bytes(self, file_id: str) -> bytes:
        """
        Return the bytes from the artifact store.

        :param file_id: The id of the file.
        """
        with open(os.path.join(self.blobs, file_id), 'rb') as f:
            return f.read()

    def put_file(self, file_path: str, file_id: str):
        """Save file in artifact store and return the relative path.

        return the relative path {file_id}/{name}

        :param file_path: The path to the file to be saved.
        :param file_id: The id of the file.
        """
        path = Path(file_path)
        name = path.name
        file_id_folder = os.path.join(self.files, file_id)

        os.makedirs(file_id_folder, exist_ok=True)
        os.chmod(file_id_folder, 0o777)
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
        logging.debug(f"Loading file {file_id} from {self.files}")
        path = os.path.join(self.files, file_id)
        files = os.listdir(path)
        assert len(files) == 1, f"Expected 1 file, got {len(files)}"
        name = files[0]
        return os.path.join(path, name)

    def disconnect(self):
        """Disconnect the client."""
        # Not necessary since just local filesystem
        pass

    def list(self):
        """List all files in the artifact store."""
        return sorted(list(set(os.listdir(self.blobs) + os.listdir(self.files))))
