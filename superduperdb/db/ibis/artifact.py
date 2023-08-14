import os

from superduperdb.db.base.artifact import ArtifactStore


class Filesystem:
    ROOT_PATH = '.artifacts/'
    def __init__(self):
        os.makedirs(self.ROOT_PATH, exist_ok=True)

    def delete(self, file_id: str):
        # use os to delete file with file_id name

        os.remove(os.path.join(self.ROOT_PATH, file_id))

    def get(self, file_id: str):
        return open(os.path.join(self.ROOT_PATH, file_id), 'rb')

    def put(self, bytes):
        file_id = os.urandom(8).hex()
        with open(os.path.join(self.ROOT_PATH, file_id), 'wb') as f:
            f.write(bytes)
        return file_id


class IbisArtifactStore(ArtifactStore):
    def __init__(self, conn, name: str):
        """
        :param conn: Ibis client connection
        :param name: Name of database to host filesystem
        """
        super().__init__(name=name, conn=conn)
        self.filesystem = Filesystem()

    def delete_artifact(self, file_id: str):
        return self.filesystem.delete(file_id)

    def _load_bytes(self, file_id: str):
        return self.filesystem.get(file_id).read()

    def _save_artifact(self, serialized: bytes):
        return self.filesystem.put(serialized)
