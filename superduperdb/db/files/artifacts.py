import os
import shutil
import uuid
import click

from superduperdb.db.base.artifact import ArtifactStore
from superduperdb.misc.colors import Colors


class FilesystemArtifactStore(ArtifactStore):
    def __init__(self, conn: str, name: str):
        self.root = conn
        self.path = name

    def drop(self, force: bool = False):
        """
        Drop the filesystem on disk.
        :param force: Force drop without confirmation (TO BE USED WITH EXTREME CAUTION!)
        """
        if not force:
            if not click.confirm(
                f'{Colors.RED}[!!!WARNING USE WITH CAUTION AS YOU '
                f'WILL LOSE ALL DATA!!!]{Colors.RESET} '
                'Are you sure you want to drop all artifacts? ',
                default=False,
            ):
                print('Aborting...')
        return shutil.rmtree(f'{self.root}/{self.path}', ignore_errors=True)

    def delete_artifact(self, file_id: str):
        """
        Delete artifact from artifact store
        :param file_id: File id uses to identify artifact in store
        """
        os.remove(f'{self.root}/{self.path}/{file_id}')

    def _save_artifact(self, serialized: bytes) -> t.Any:
        file_id = str(uuid.uuid4())
        with open(f'{self.root}/{self.path}/{file_id}', 'wb') as f:
            f.write(serialized)
        return file_id

    def _load_bytes(self, file_id: str) -> bytes:
        with open(f'{self.root}/{self.path}/{file_id}', 'rb') as f:
            return f.read()
