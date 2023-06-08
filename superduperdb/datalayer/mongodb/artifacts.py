import gridfs

from superduperdb.datalayer.base.artifacts import ArtifactStore


class MongoArtifactStore(ArtifactStore):
    def __init__(self, db):
        """
        :param db: MongoDB database connection
        """
        self.filesystem = gridfs.GridFS(db)

    def delete_artifact(self, file_id: str):
        return self.filesystem.delete(file_id)

    def _load_bytes(self, file_id: str):
        return self.filesystem.get(file_id).read()

    def _save_artifact(self, serialized: bytes):
        return self.filesystem.put(serialized)
