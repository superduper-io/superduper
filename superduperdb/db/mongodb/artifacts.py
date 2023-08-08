import gridfs

from superduperdb.db.base.artifact import ArtifactStore


class MongoArtifactStore(ArtifactStore):
    def __init__(self, conn, name: str):
        """
        :param conn: MongoDB client connection
        :param name: Name of database to host filesystem
        """
        super().__init__(name=name, conn=conn)
        self.db = self.conn[self.name]
        self.filesystem = gridfs.GridFS(self.db)

    def drop(self):
        return self.db.client.drop_database(self.db.name)

    def delete_artifact(self, file_id: str):
        return self.filesystem.delete(file_id)

    def _load_bytes(self, file_id: str):
        return self.filesystem.get(file_id).read()

    def _save_artifact(self, serialized: bytes):
        return self.filesystem.put(serialized)
