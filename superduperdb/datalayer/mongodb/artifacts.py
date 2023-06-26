import typing as t

import gridfs

from superduperdb.datalayer.base.artifacts import ArtifactStore


class MongoArtifactStore(ArtifactStore):
    def __init__(self, conn, name: t.Optional[str] = None):
        """
        :param conn: MongoDB client connection
        :param name: Name of database to host filesystem
        """
        if name is None:
            name = '_filesystem'
        super().__init__(name=name, conn=conn)
        db = self.conn[self.name]
        self.filesystem = gridfs.GridFS(db)

    def delete_artifact(self, file_id: str) -> None:
        return self.filesystem.delete(file_id)

    def _load_bytes(self, file_id: str) -> bytes:
        return self.filesystem.get(file_id).read()

    def _save_artifact(self, serialized: bytes):
        return self.filesystem.put(serialized)
