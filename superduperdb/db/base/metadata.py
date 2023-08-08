import time
import typing as t
from abc import ABC, abstractmethod


class MetaDataStore(ABC):
    """
    Abstraction for storing meta-data separately from primary data.

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

    @abstractmethod
    def create_component(self, info: t.Dict):
        pass

    @abstractmethod
    def create_job(self, info: t.Dict):
        pass

    @abstractmethod
    def create_parent_child(self, parent: str, child: str):
        pass

    @abstractmethod
    def drop(self):
        """
        Drop the metadata store.
        """
        pass

    @abstractmethod
    def get_job(self, job_id: str):
        pass

    @abstractmethod
    def update_job(self, job_id: str, key: str, value: t.Any):
        pass

    def listen_job(self, identifier: str):
        try:
            status = 'pending'
            n_lines = 0
            n_lines_stderr = 0
            while status in {'pending', 'running'}:
                r = self.get_job(identifier)
                status = r['status']
                if status == 'running':
                    if len(r['stdout']) > n_lines:
                        print(''.join(r['stdout'][n_lines:]), end='')
                        n_lines = len(r['stdout'])
                    if len(r['stderr']) > n_lines_stderr:
                        print(''.join(r['stderr'][n_lines_stderr:]), end='')
                        n_lines_stderr = len(r['stderr'])
                    time.sleep(0.2)
                else:
                    time.sleep(0.2)
            r = self.get_job(identifier)
            if status == 'success':
                if len(r['stdout']) > n_lines:
                    print(''.join(r['stdout'][n_lines:]), end='')
                if len(r['stderr']) > n_lines_stderr:
                    print(''.join(r['stderr'][n_lines_stderr:]), end='')
            elif status == 'failed':  # pragma: no cover
                print(r['msg'])
        except KeyboardInterrupt:  # pragma: no cover
            return

    @abstractmethod
    def show_jobs(self):
        pass

    @abstractmethod
    def show_components(self, type_id: str, **kwargs):
        pass

    @abstractmethod
    def show_component_versions(self, type_id: str, identifier: str):
        pass

    @abstractmethod
    def delete_component_version(self, type_id: str, identifier: str, version: int):
        pass

    @abstractmethod
    def component_version_has_parents(
        self, type_id: str, identifier: str, version: int
    ):
        pass

    @abstractmethod
    def get_metadata(self, key):
        pass

    @abstractmethod
    def get_latest_version(
        self, type_id: str, identifier: str, allow_hidden: bool = False
    ):
        pass

    @abstractmethod
    def _get_component(
        self,
        type_id: str,
        identifier: str,
        version: int,
        allow_hidden: bool = False,
    ):
        pass

    def get_component(
        self,
        type_id: str,
        identifier: str,
        version: t.Optional[int] = None,
        allow_hidden: bool = False,
    ) -> t.Dict[str, t.Any]:
        if version is None:
            version = self.get_latest_version(
                type_id=type_id, identifier=identifier, allow_hidden=allow_hidden
            )
        r = self._get_component(
            type_id=type_id,
            identifier=identifier,
            version=version,
            allow_hidden=allow_hidden,
        )
        if r is None:
            raise FileNotFoundError(f'Object {identifier} does not exist in metadata')
        return r

    @abstractmethod
    def _update_object(
        self,
        identifier: str,
        type_id: str,
        key: str,
        value: t.Any,
        version: int,
    ):
        pass

    def update_object(self, identifier, type_id, key, value, version=None):
        if version is not None:
            version = self.get_latest_version(type_id, identifier)
        return self._update_object(
            identifier=identifier,
            type_id=type_id,
            key=key,
            value=value,
            version=version,
        )

    @abstractmethod
    def _replace_object(self, info, identifier, type_id, version):
        pass

    def replace_object(
        self,
        info: t.Dict[str, t.Any],
        identifier: str,
        type_id: str,
        version: t.Optional[int] = None,
    ) -> None:
        if version is not None:
            version = self.get_latest_version(type_id, identifier)
        return self._replace_object(
            info=info,
            identifier=identifier,
            type_id=type_id,
            version=version,
        )

    @abstractmethod
    def write_output_to_job(self, identifier, msg, stream):
        pass

    @abstractmethod
    def get_component_version_parents(self, unique_id: str):
        pass

    @abstractmethod
    def hide_component_version(self, type_id: str, identifier: str, version: int):
        pass
