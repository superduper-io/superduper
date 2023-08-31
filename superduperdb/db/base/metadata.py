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
        """
        Create a component in the metadata store.

        :param info: dictionary containing information about the component.
        """
        pass

    @abstractmethod
    def create_job(self, info: t.Dict):
        """
        Create a job in the metadata store.
        :param info: dictionary containing information about the job.
        """
        pass

    @abstractmethod
    def create_parent_child(self, parent: str, child: str):
        """
        Create a parent-child relationship between two components.

        :param parent: parent component
        :param child: child component
        """
        pass

    @abstractmethod
    def drop(self, force: bool = False):
        """
        Drop the metadata store.

        :param force: whether to force the drop (without confirmation)
        """
        pass

    # ------------------ COMPONENTS ------------------

    @abstractmethod
    def get_job(self, job_id: str):
        """
        Get a job from the metadata store.

        :param job_id: job identifier
        """
        pass

    @abstractmethod
    def update_job(self, job_id: str, key: str, value: t.Any):
        """
        Update a job in the metadata store.

        :param job_id: job identifier
        :param key: key to be updated
        :param value: value to be updated
        """
        pass

    def watch_job(self, identifier: str):
        """
        Listen to a job.

        :param identifier: job identifier
        """
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
            time.sleep(0.01)
        r = self.get_job(identifier)
        if status == 'success':
            if len(r['stdout']) > n_lines:
                print(''.join(r['stdout'][n_lines:]), end='')
            if len(r['stderr']) > n_lines_stderr:
                print(''.join(r['stderr'][n_lines_stderr:]), end='')
        elif status == 'failed':
            print(r['msg'])

    @abstractmethod
    def show_jobs(self):
        """
        Show all jobs in the metadata store.
        """
        pass

    @abstractmethod
    def show_components(self, type_id: str, **kwargs):
        """
        Show all components in the metadata store.

        :param type_id: type of component
        :param **kwargs: additional arguments
        """
        pass

    @abstractmethod
    def show_component_versions(self, type_id: str, identifier: str):
        """
        Show all versions of a component in the metadata store.

        :param type_id: type of component
        :param identifier: identifier of component
        """
        pass

    @abstractmethod
    def delete_component_version(self, type_id: str, identifier: str, version: int):
        """
        Delete a component version from the metadata store.

        :param type_id: type of component
        :param identifier: identifier of component
        :param version: version of component
        """
        pass

    @abstractmethod
    def component_version_has_parents(
        self, type_id: str, identifier: str, version: int
    ):
        """
        Check if a component version has parents.

        :param type_id: type of component
        :param identifier: identifier of component
        :param version: version of component
        """
        pass

    @abstractmethod
    def get_metadata(self, key):
        """
        Get metadata from the metadata store.

        :param key: key of metadata
        """
        pass

    @abstractmethod
    def get_latest_version(
        self, type_id: str, identifier: str, allow_hidden: bool = False
    ):
        """
        Get the latest version of a component.

        :param type_id: type of component
        :param identifier: identifier of component
        :param allow_hidden: whether to allow hidden components
        """
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
        """
        Get a component from the metadata store.

        :param type_id: type of component
        :param identifier: identifier of component
        :param version: version of component
        :param allow_hidden: whether to allow hidden components
        """
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

    def update_object(
        self,
        identifier: str,
        type_id: str,
        key: str,
        value: str,
        version: t.Optional[int] = None,
    ):
        """
        Update an object in the metadata store.

        :param identifier: identifier of object
        :param type_id: type of object
        :param key: key to be updated
        :param value: value to be updated
        :param version: version of object
        """
        if version is not None:
            version = self.get_latest_version(type_id, identifier)
        assert isinstance(version, int)
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
        """
        Replace an object in the metadata store.

        :param info: dictionary containing information about the object
        :param identifier: identifier of object
        :param type_id: type of object
        :param version: version of object
        """
        if version is not None:
            version = self.get_latest_version(type_id, identifier)
        return self._replace_object(
            info=info,
            identifier=identifier,
            type_id=type_id,
            version=version,
        )

    @abstractmethod
    def write_output_to_job(self, identifier: str, msg: str, stream: str):
        """
        Write output to a job in the metadata store.

        :param identifier: identifier of job
        :param msg: message to be written
        :param stream: stream to be written to
        """
        pass

    @abstractmethod
    def get_component_version_parents(self, unique_id: str):
        """
        Get the parents of a component version.

        :param unique_id: unique identifier of component version
        """
        pass

    @abstractmethod
    def hide_component_version(self, type_id: str, identifier: str, version: int):
        """
        Hide a component version.

        :param type_id: type of component
        :param identifier: identifier of component
        :param version: version of component
        """
        pass

    @abstractmethod
    def update_metadata(self, key, value):
        """
        Update metadata in the metadata store.

        :param key: Key of metadata
        :param value: Value of metadata
        """
        pass
