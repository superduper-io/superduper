import time
import typing as t
from abc import ABC, abstractmethod

if t.TYPE_CHECKING:
    from superduperdb.datalayer.base.database import (
        UpdateResult,
        InsertResult,
        DeleteResult,
    )


class MetaDataStore(ABC):
    """
    Abstraction for storing meta-data separately from primary data.
    """

    def __init__(
        self,
        conn: t.Any,
        name: t.Optional[str] = None,
    ):
        self.name = name
        self.conn = conn

    @abstractmethod
    def create_component(self, info: t.Dict) -> 'InsertResult':
        pass

    @abstractmethod
    def create_parent_child(self, parent: str, child: str) -> 'InsertResult':
        pass

    @abstractmethod
    def get_job(self, job_id: str) -> t.Any:
        pass

    @abstractmethod
    def update_job(self, job_id: str, key: str, value: t.Any) -> 'UpdateResult':
        pass

    @abstractmethod
    def watch_job(self, identifier: str) -> None:
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
    def show_jobs(self) -> t.List:
        pass

    @abstractmethod
    def show_components(self, variety: str, **kwargs) -> t.List:
        pass

    @abstractmethod
    def show_component_versions(self, variety: str, identifier: str) -> t.List:
        pass

    @abstractmethod
    def delete_component_version(
        self, variety: str, identifier: str, version: int
    ) -> 'DeleteResult':
        pass

    @abstractmethod
    def component_version_has_parents(
        self, variety: str, identifier: str, version: int
    ) -> bool:
        pass

    @abstractmethod
    def get_metadata(self, key) -> t.Any:
        pass

    @abstractmethod
    def get_latest_version(
        self, variety: str, identifier: str, allow_hidden: bool = False
    ) -> t.List:
        pass

    @abstractmethod
    def _get_component(
        self,
        variety: str,
        identifier: str,
        version: int,
        allow_hidden: bool = False,
    ) -> t.Any:
        pass

    def get_component(
        self,
        variety: str,
        identifier: str,
        version: t.Optional[int] = None,
        allow_hidden: bool = False,
    ) -> t.Any:
        if version is None:
            version = self.get_latest_version(
                variety, identifier, allow_hidden=allow_hidden
            )
        r = self._get_component(
            variety, identifier, version=version, allow_hidden=allow_hidden
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
        value: t.Any,
        version: int,
    ) -> 'UpdateResult':
        pass

    def update_object(
        self,
        identifier: str,
        variety: str,
        key: str,
        value: t.Any,
        version: t.Optional[int] = None,
    ) -> 'UpdateResult':
        if version is not None:
            version = self.get_latest_version(variety, identifier)
        return self._update_object(identifier, variety, key, value, version=version)

    @abstractmethod
    def write_output_to_job(self, identifier: str, msg: str, stream: str) -> None:
        pass

    @abstractmethod
    def get_component_version_parents(self, unique_id: str) -> t.List[str]:
        pass

    @abstractmethod
    def hide_component_version(
        self, variety: str, identifier: str, version: int
    ) -> None:
        pass
