import os
import tempfile
import typing as t
from pathlib import Path

import click
import gridfs
from superduper import CFG, logging
from superduper.backends.base.artifacts import ArtifactStore
from superduper.misc.colors import Colors
from tqdm import tqdm

from superduper_mongodb.utils import connection_callback


class MongoArtifactStore(ArtifactStore):
    """
    Artifact store for MongoDB.

    :param conn: MongoDB client connection
    :param name: Name of database to host filesystem
    :param flavour: Flavour of the artifact store
    """

    def __init__(
        self, conn, name: t.Optional[str] = None, flavour: t.Optional[str] = None
    ):
        super().__init__(name=name, conn=conn)
        if isinstance(conn, str):
            self.conn, url_name = connection_callback(conn, flavour)
            name = name or url_name
            self.name = f"_filesystem:{name}"
        else:
            self.conn = conn
            self.name = name
        self.filesystem = gridfs.GridFS(self.conn[self.name])

    def url(self):
        """Return the URL of the database."""
        return self.conn.HOST + ':' + str(self.conn.PORT) + '/' + self.name

    def drop(self, force: bool = False):
        """Drop the database.

        Please use with caution as this will delete all artifacts.

        :param force: If True, will not prompt for confirmation
        """
        if not force:
            if not click.confirm(
                f'{Colors.RED}[!!!WARNING USE WITH CAUTION AS YOU '
                f'WILL LOSE ALL DATA!!!]{Colors.RESET} '
                'Are you sure you want to drop all artifacts? ',
                default=False,
            ):
                logging.warn('Aborting...')
        return self.conn.drop_database(self.name)

    def _exists(self, file_id):
        return self.filesystem.find_one({'file_id': file_id}) is not None

    def _delete_bytes(self, file_id: str):
        r = self.filesystem.find({'file_id': file_id})
        ids = [x._id for x in r]
        if not ids:
            raise FileNotFoundError(f'File not found in {file_id}')
        for _id in ids:
            self.filesystem.delete(_id)

    def get_bytes(self, file_id: str):
        """
        Get the bytes of the file from GridFS.

        :param file_id: The file_id of the file to get
        """
        cur = self.filesystem.find_one({'file_id': file_id})
        if cur is None:
            raise FileNotFoundError(f'File not found in {file_id}')
        return cur.read()

    def put_file(self, file_path: str, file_id: str):
        """Save file to GridFS.

        :param file_path: The path to the file to save
        :param file_id: The file_id of the file
        """
        path = Path(file_path)
        if path.is_dir():
            upload_folder(file_path, file_id, self.filesystem)
        else:
            _upload_file(file_path, file_id, self.filesystem)
        return file_id

    def get_file(self, file_id: str) -> str:
        """Download file from GridFS and return the path.

        The path is a temporary directory, `{tmp_prefix}/{file_id}/{filename or folder}`
        :param file_id: The file_id of the file to download
        """
        return _download(file_id, self.filesystem)

    def put_bytes(self, serialized: bytes, file_id: str):
        """
        Save bytes in GridFS.

        :param serialized: The bytes to save
        :param file_id: The file_id of the file
        """
        cur = self.filesystem.find_one({'file_id': file_id})
        if cur is not None:
            logging.warn(f"File {file_id} already exists")
            self._delete_bytes(file_id)
        return self.filesystem.put(serialized, filename=file_id, file_id=file_id)

    def disconnect(self):
        """Disconnect the client."""

        # TODO: implement me


def _upload_file(path, file_id, fs):
    """Upload file to GridFS.

    :param path: The path to the file to upload
    :param file_id: The file_id of the file
    :param fs: The GridFS object
    """
    logging.info(f"Uploading file {path} to GridFS with file_id {file_id}")
    path = Path(path)
    with open(path, 'rb') as file_to_upload:
        fs.put(
            file_to_upload,
            filename=path.name,
            file_id=file_id,
            metadata={"type": "file"},
        )


def upload_folder(path, file_id, fs, parent_path=""):
    """Upload folder to GridFS.

    :param path: The path to the folder to upload
    :param file_id: The file_id of the folder
    :param fs: The GridFS object
    :param parent_path: The parent path of the folder
    """
    path = Path(path)
    if not parent_path:
        logging.info(f"Uploading folder {path} to GridFS with file_id {file_id}")
        parent_path = os.path.basename(path)

    # if the folder is empty, create an empty file
    if not os.listdir(path):
        fs.put(
            b'',
            filename=os.path.join(parent_path, os.path.basename(path)),
            file_id=file_id,
            metadata={"is_empty_dir": True, 'type': 'dir'},
        )
    else:
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            if os.path.isdir(item_path):
                upload_folder(item_path, file_id, fs, os.path.join(parent_path, item))
            else:
                with open(item_path, 'rb') as file_to_upload:
                    fs.put(
                        file_to_upload,
                        filename=os.path.join(parent_path, item),
                        file_id=file_id,
                        metadata={"type": "dir"},
                    )


def _download(file_id, fs):
    """Download file or folder from GridFS and return the path.

    The path is a temporary directory, `{tmp_prefix}/{file_id}/{filename or folder}`

    :param file_id: The file_id of the file or folder to download
    :param fs: The GridFS object
    """
    download_folder = CFG.downloads.folder

    if not download_folder:
        download_folder = os.path.join(
            tempfile.gettempdir(), "superduper", "ArtifactStore"
        )

    save_folder = os.path.join(download_folder, file_id)
    os.makedirs(save_folder, exist_ok=True)

    file = fs.find_one({"file_id": file_id})
    if file is None:
        raise FileNotFoundError(f"File not found in {file_id}")

    type_ = file.metadata.get("type")
    if type_ not in {"file", "dir"}:
        raise ValueError(
            f"Unknown type '{type_}' for file_id {file_id}, expected file or dir"
        )

    if type_ == 'file':
        save_path = os.path.join(save_folder, os.path.split(file.filename)[-1])
        logging.info(f"Downloading file_id {file_id} to {save_path}")
        with open(save_path, 'wb') as f:
            f.write(file.read())
        logging.info(f"Downloaded file_id {file_id} to {save_path}")
        return save_path

    logging.info(f"Downloading folder with file_id {file_id} to {save_folder}")
    for grid_out in tqdm(fs.find({"file_id": file_id, "metadata.type": "dir"})):
        file_path = os.path.join(save_folder, grid_out.filename)
        if grid_out.metadata.get("is_empty_dir", False):
            if not os.path.exists(file_path):
                os.makedirs(file_path)
        else:
            directory = os.path.dirname(file_path)
            if not os.path.exists(directory):
                os.makedirs(directory)
            with open(file_path, 'wb') as file_to_write:
                file_to_write.write(grid_out.read())

    folders = os.listdir(save_folder)
    assert len(folders) == 1, f"Expected only one folder, got {folders}"
    folder_path = os.path.join(save_folder, folders[0])
    logging.info(f"Downloaded folder with file_id {file_id} to {folder_path}")
    return folder_path
