import os
import tempfile
from pathlib import Path

import click
import gridfs
from tqdm import tqdm

from superduperdb import CFG, logging
from superduperdb.backends.base.artifact import ArtifactStore
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
        if not force:
            if not click.confirm(
                f'{Colors.RED}[!!!WARNING USE WITH CAUTION AS YOU '
                f'WILL LOSE ALL DATA!!!]{Colors.RESET} '
                'Are you sure you want to drop all artifacts? ',
                default=False,
            ):
                logging.warn('Aborting...')
        return self.db.client.drop_database(self.db.name)

    def _exists(self, file_id):
        return self.filesystem.find_one({'filename': file_id}) is not None

    def _delete_artifact(self, file_id: str):
        r = self.filesystem.find({'metadata.file_id': file_id})
        ids = [x._id for x in r]
        if not ids:
            raise FileNotFoundError(f'File not found in {file_id}')
        for _id in ids:
            self.filesystem.delete(_id)

    def _load_bytes(self, file_id: str):
        cur = self.filesystem.find_one({'filename': file_id})
        if cur is None:
            raise FileNotFoundError(f'File not found in {file_id}')
        return cur.read()

    def _save_file(self, file_path: str, file_id: str):
        """Save file to GridFS"""
        path = Path(file_path)
        if path.is_dir():
            upload_folder(file_path, file_id, self.filesystem)
        else:
            upload_file(file_path, file_id, self.filesystem)
        return file_id

    def _load_file(self, file_id: str) -> str:
        """
        Download file from GridFS and return the path
        The path is a temporary directory, {tmp_prefix}/{file_id}/{filename or folder}
        """
        return download(file_id, self.filesystem)

    def _save_bytes(self, serialized: bytes, file_id: str):
        return self.filesystem.put(
            serialized, filename=file_id, metadata={"file_id": file_id}
        )

    def disconnect(self):
        """
        Disconnect the client
        """

        # TODO: implement me


def upload_file(path, file_id, fs):
    """Upload file to GridFS"""
    logging.info(f"Uploading file {path} to GridFS with file_id {file_id}")
    path = Path(path)
    with open(path, 'rb') as file_to_upload:
        fs.put(
            file_to_upload,
            filename=path.name,
            metadata={"file_id": file_id, "type": "file"},
        )


def upload_folder(path, file_id, fs, parent_path=""):
    """Upload folder to GridFS"""
    path = Path(path)
    if not parent_path:
        logging.info(f"Uploading folder {path} to GridFS with file_id {file_id}")
        parent_path = os.path.basename(path)

    # if the folder is empty, create an empty file
    if not os.listdir(path):
        fs.put(
            b'',
            filename=os.path.join(parent_path, os.path.basename(path)),
            metadata={"file_id": file_id, "is_empty_dir": True, 'type': 'dir'},
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
                        metadata={"file_id": file_id, "type": "dir"},
                    )


def download(file_id, fs):
    """Download file or folder from GridFS and return the path"""

    download_folder = CFG.downloads.folder

    if not download_folder:
        download_folder = os.path.join(
            tempfile.gettempdir(), "superduperdb", "ArtifactStore"
        )

    save_folder = os.path.join(download_folder, file_id)
    os.makedirs(save_folder, exist_ok=True)

    file = fs.find_one({"metadata.file_id": file_id})
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
        return save_path

    logging.info(f"Downloading folder with file_id {file_id} to {save_folder}")
    for grid_out in tqdm(
        fs.find({"metadata.file_id": file_id, "metadata.type": "dir"})
    ):
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
    save_folder = os.path.join(save_folder, folders[0])
    logging.info(f"Downloaded folder with file_id {file_id} to {save_folder}")
    return save_folder
