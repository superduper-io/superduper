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
        assert self._serializers is not None, 'Serializers not initialized!'
        return self._serializers

    @serializers.setter
    def serializers(self, value):
        self._serializers = value

    @abstractmethod
    def url(self):
        """
        Artifact store connection url
        """
        pass

    @abstractmethod
    def _delete_artifact(self, file_id: str):
        """
        Delete artifact from artifact store
        :param file_id: File id uses to identify artifact in store
        """

    def delete(self, r: t.Dict):
        if '_content' in r and 'file_id' in r['_content']:
            return self._delete_artifact(r['_content']['file_id'])
        for v in r.values():
            if isinstance(v, dict):
                self.delete(v)

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
        if file_id is None:
            assert uri is not None, "if file_id is None, uri can\'t be None"
            file_id = _construct_file_id_from_uri(uri)
            if self.serializers[datatype].directory:
                assert datatype is not None
                file_id = os.path.join(datatype.directory, file_id)
        return self._exists(file_id)

    @abstractmethod
    def _save_bytes(self, serialized: bytes, file_id: str):
        """Save bytes in artifact store""" ""
        pass

    @abstractmethod
    def _save_file(self, file_path: str, file_id: str) -> str:
        """Save file in artifact store and return file_id"""
        pass

    def save_artifact(self, r: t.Dict):
        """
        Save serialized object in the artifact store.

        :param r: dictionary with mandatory fields
                  {'bytes', 'datatype'}
                  and optional fields
                  {'file_id', 'uri'}
        """
        if r.get('leaf_type') == 'file':
            assert 'file_id' in r, 'file_id is missing!'
            file_id = self._save_file(r['uri'], r['file_id'])
        else:
            assert 'bytes' in r, 'serialized bytes are missing!'
            assert 'datatype' in r, 'no datatype specified!'
            datatype = self.serializers[r['datatype']]
            uri = r.get('uri')
            file_id = r.get('file_id')
            if uri is not None:
                file_id = _construct_file_id_from_uri(uri)
            else:
                file_id = r.get('sha1') or hashlib.sha1(r['bytes']).hexdigest()
            if r.get('directory'):
                file_id = os.path.join(datatype.directory, file_id)
            self._save_bytes(r['bytes'], file_id=file_id)
            del r['bytes']
        r['file_id'] = file_id
        return r

    @abstractmethod
    def _load_bytes(self, file_id: str) -> bytes:
        """
        Load bytes from artifact store.

        :param file_id: Identifier of artifact in the store
        """
        pass

    @abstractmethod
    def _load_file(self, file_id: str) -> str:
        """
        Load file from artifact store and return path

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
            x = self._load_file(file_id)
        else:
            # We should always have file_id available at load time (because saved)
            uri = r.get('uri')
            if file_id is None:
                assert uri is not None, '"uri" and "file_id" can\'t both be None'
                file_id = _construct_file_id_from_uri(uri)
            x = self._load_bytes(file_id)
        return datatype.decode_data(x)

    def save(self, r: t.Dict) -> t.Dict:
        """
        Save list of artifacts and replace the artifacts with file reference
        :param r: `dict` of artifacts
        """
        if isinstance(r, dict):
            if '_content' in r and r['_content']['leaf_type'] in {
                'artifact',
                'file',
                'lazy_artifact',
            }:
                self.save_artifact(r['_content'])
            else:
                for k in r:
                    self.save(r[k])
        if isinstance(r, list):
            for x in r:
                self.save(x)
        return r

    @abstractmethod
    def disconnect(self):
        """
        Disconnect the client
        """
        pass


class ArtifactSavingError(Exception):
    pass
