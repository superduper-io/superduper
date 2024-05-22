import hashlib
import os
import typing as t
from abc import ABC, abstractmethod


def _construct_file_id_from_uri(uri):
    return str(hashlib.sha1(uri.encode()).hexdigest())


class ArtifactStore(ABC):
    """
    Abstraction for storing large artifacts separately from primary data.

    :param conn: connection to the meta-data store
    :param name: Name to identify DB using the connection
    """

    def __init__(
        self,
        conn: t.Any,
        name: t.Optional[str] = None,
    ):
        self.name = name
        self.conn = conn
        self._serializers = None

    @property
    def serializers(self):
        """Return the serializers."""
        assert self._serializers is not None, 'Serializers not initialized!'
        return self._serializers

    @serializers.setter
    def serializers(self, value):
        """Set the serializers.

        :param value: The serializers.
        """
        self._serializers = value

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
    def _exists(self, file_id: str):
        pass

    def exists(
        self,
        file_id: t.Optional[str] = None,
        datatype: t.Optional[str] = None,
        uri: t.Optional[str] = None,
    ):
        """Check if artifact exists in artifact store.

        :param file_id: file id of artifact in the store
        :param datatype: Datatype of the artifact
        :param uri: URI of the artifact
        """
        if file_id is None:
            assert uri is not None, "if file_id is None, uri can\'t be None"
            file_id = _construct_file_id_from_uri(uri)
            if self.serializers[datatype].directory:
                assert datatype is not None
                file_id = os.path.join(datatype.directory, file_id)
        return self._exists(file_id)

    @abstractmethod
    def put_bytes(self, serialized: bytes, file_id: str):
        """Save bytes in artifact store""" ""
        pass

    @abstractmethod
    def put_file(self, file_path: str, file_id: str) -> str:
        """Save file in artifact store and return file_id."""
        pass

    def save_artifact(self, r: t.Dict):
        """Save serialized object in the artifact store.

        :param r: dictionary with mandatory fields
        """
        blobs = r.get('_blobs', {})
        files = r.get('_files', {})

        for file_id, blob in blobs.items():
            try:
                self.put_bytes(blob, file_id=file_id)
            except FileExistsError:
                continue

        for file_id, file_path in files.items():
            try:
                self.put_file(file_path, file_id=file_id)
            except FileExistsError:
                continue

        r['_blobs'] = list(blobs.keys())
        r['_files'] = list(files.keys())
        return r

    def delete_artifact(self, r: t.Dict):
        """Delete artifact from artifact store.

        :param r: dictionary with mandatory fields
        """
        for blob in r['_blobs']:
            self._delete_bytes(blob)

        for file_path in r['_files']:
            self._delete_bytes(file_path)

    def update_artifact(self, old_r: t.Dict, new_r: t.Dict):
        """Update artifact in artifact store.

        This method deletes the old artifact and saves the new artifact.

        :param old_r: dictionary with mandatory fields
        :param new_r: dictionary with mandatory fields
        """
        self.delete_artifact(old_r)
        return self.save_artifact(new_r)

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

    def load_artifact(self, r):
        """
        Load artifact from artifact store, and deserialize.

        :param r: Mandatory fields {'file_id', 'datatype'}
        """
        datatype = self.serializers[r['datatype']]
        file_id = r.get('file_id')
        if r.get('encodable') == 'file':
            x = self.get_file(file_id)
        else:
            # TODO We should always have file_id available at load time (because saved)
            uri = r.get('uri')
            if file_id is None:
                assert uri is not None, '"uri" and "file_id" can\'t both be None'
                file_id = _construct_file_id_from_uri(uri)
            x = self.get_bytes(file_id)
        return datatype.decode_data(x)

    @abstractmethod
    def disconnect(self):
        """Disconnect the client."""
        pass


class ArtifactSavingError(Exception):
    """
    Error when saving artifact in artifact store fails.

    :param args: *args for `Exception`
    :param kwargs: **kwargs for `Exception`
    """
