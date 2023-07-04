import time
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod


class MetaDataStore(ABC):
    """
    Abstraction for storing meta-data separately from primary data.

    :param conn: connection to the meta-data store
    :param name: Name to identify DB using the connection
    """

    def __init__(
        self,
        conn: Any,
        name: Optional[str] = None,
    ):
        self.name = name
        self.conn = conn

    @abstractmethod
    def create_component(self, info: Dict):
        pass

    @abstractmethod
    def create_job(self, info: Dict):
        pass

    @abstractmethod
    def create_parent_child(self, parent: str, child: str):
        pass

    @abstractmethod
    def get_job(self, job_id: str):
        pass

    @abstractmethod
    def update_job(self, job_id: str, key: str, value: Any):
        pass

    def watch_job(self, identifier: str):
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
    def show_components(self, variety: str, **kwargs):
        pass

    @abstractmethod
    def show_component_versions(self, variety: str, identifier: str):
        pass

    @abstractmethod
    def delete_component_version(self, variety: str, identifier: str, version: int):
        pass

    @abstractmethod
    def component_version_has_parents(
        self, variety: str, identifier: str, version: int
    ):
        pass

    @abstractmethod
    def get_metadata(self, key):
        pass

    @abstractmethod
    def get_latest_version(
        self, variety: str, identifier: str, allow_hidden: bool = False
    ):
        pass

    @abstractmethod
    def _get_component(
        self,
        variety: str,
        identifier: str,
        version: int,
        allow_hidden: bool = False,
    ):
        pass

    def get_component(
        self,
        variety: str,
        identifier: str,
        version: Optional[int] = None,
        allow_hidden: bool = False,
    ):
        if version is None:
            version = self.get_latest_version(
                variety=variety, identifier=identifier, allow_hidden=allow_hidden
            )
        r = self._get_component(
            variety=variety,
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
        variety: str,
        key: str,
        value: Any,
        version: int,
    ):
        pass

    def update_object(self, identifier, variety, key, value, version=None):
        if version is not None:
            version = self.get_latest_version(variety, identifier)
        return self._update_object(
            identifier=identifier,
            variety=variety,
            key=key,
            value=value,
            version=version,
        )

    @abstractmethod
    def write_output_to_job(self, identifier, msg, stream):
        pass

    @abstractmethod
    def get_component_version_parents(self, unique_id: str):
        pass

    @abstractmethod
    def hide_component_version(self, variety: str, identifier: str, version: int):
        pass
