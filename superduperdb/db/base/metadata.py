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
    def drop(self, force: bool = False):
        """
        Drop the metadata store.
        """
        pass

    # ------------------ COMPONENTS ------------------

    @abstractmethod
    def create_component(self, info: t.Dict):
        """
        Create a component in the metadata store.

        :param info: Information about the component found in ``component.serialize()``

        See also ``superduperdb.container.component.Component``
        """
        pass

    @abstractmethod
    def component_version_has_parents(
        self, type_id: str, identifier: str, version: int
    ):
        """
        Check if a component version has parents.

        :param type_id: Type of component
        :param identifier: Identifier of component
        :param version: Version of component
        """
        pass

    @abstractmethod
    def create_parent_child(self, parent: str, child: str):
        """
        Create a parent-child relationship between two components.

        :param parent: Parent component Unique ID ``{type_id}/{identifier}/{version}``
        :param child: Child component Unique ID ``{type_id}/{identifier}/{version}``
        """
        pass

    @abstractmethod
    def delete_component_version(self, type_id: str, identifier: str, version: int):
        """
        Delete a component version.

        :param type_id: Type of component
        :param identifier: Identifier of component
        :param version: Version of component
        """
        pass

    @abstractmethod
    def get_latest_version(
        self, type_id: str, identifier: str, allow_hidden: bool = False
    ):
        """
        Get the latest version of a component.

        :param type_id: Type of component
        :param identifier: Identifier of component
        :param allow_hidden: Allow hidden components to be returned
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
        Hidden components are components which have been marked for deletion,
        but are kept in the metadata store to preserve the integrity of
        other components which may depend on them.

        :param type_id: Type of component
        :param identifier: Identifier of component
        :param version: Version of component
        :param allow_hidden: Allow hidden components to be returned
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
            raise FileNotFoundError(
                f'Component {identifier} does not exist in metadata'
            )
        return r

    @abstractmethod
    def get_component_version_parents(self, unique_id: str):
        """
        Get the parents of a component version.

        :param unique_id: Unique ID of component
        version ``{type_id}/{identifier}/{version}``
        """
        pass

    @abstractmethod
    def hide_component_version(self, type_id: str, identifier: str, version: int):
        """
        Hide a component version.

        :param type_id: Type of component
        :param identifier: Identifier of component
        :param version: Version of component
        """
        pass

    @abstractmethod
    def _replace_component(self, info, identifier, type_id, version):
        pass

    def replace_component(
        self,
        info: t.Dict[str, t.Any],
        identifier: str,
        type_id: str,
        version: t.Optional[int] = None,
    ) -> None:
        """
        Replace a component in the metadata store.

        :param info: Information about the component found in ``component.serialize()``
        :param identifier: Identifier of component
        :param type_id: Type of component
        :param version: Version of component
        """
        if version is not None:
            version = self.get_latest_version(type_id, identifier)
        return self._replace_component(
            info=info,
            identifier=identifier,
            type_id=type_id,
            version=version,
        )

    @abstractmethod
    def show_components(self, type_id: str, **kwargs):
        """
        Show components in the metadata store.

        :param type_id: Type of component
        """
        pass

    @abstractmethod
    def show_component_versions(self, type_id: str, identifier: str):
        """
        Show component versions in the metadata store.

        :param type_id: Type of component
        :param identifier: Identifier of component
        """
        pass

    @abstractmethod
    def _update_component(
        self,
        identifier: str,
        type_id: str,
        key: str,
        value: t.Any,
        version: int,
    ):
        pass

    def update_component(self, identifier, type_id, key, value, version=None):
        """
        Update a component in the metadata store.

        :param identifier: Identifier of component
        :param type_id: Type of component
        :param version: Version of component (optional - defaults to latest)
        """
        if version is not None:
            version = self.get_latest_version(type_id, identifier)
        return self._update_component(
            identifier=identifier,
            type_id=type_id,
            key=key,
            value=value,
            version=version,
        )

    # --------------- JOBS ------------------

    @abstractmethod
    def create_job(self, info: t.Dict):
        """
        Create a job in the metadata store.

        :param info: Information about the job found in ``job.dict()``
        """
        raise NotImplementedError

    @abstractmethod
    def get_job(self, job_id: str):
        """
        Get a job from the metadata store.
        """
        raise NotImplementedError

    def listen_job(self, identifier: str):
        """
        Listen to a job from the metadata store.

        :param identifier: Identifier of job
        """
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
        """
        Show jobs in the metadata store.
        """
        pass

    # TODO move this to job logger component
    @abstractmethod
    def write_output_to_job(self, identifier, msg, stream):
        """
        Write output to a job in the metadata store.

        :param identifier: Identifier of job
        :param msg: Message to write
        """
        pass

    @abstractmethod
    def update_job(self, job_id: str, key: str, value: t.Any):
        """
        Update a job in the metadata store.

        :param job_id: ID of job
        :param key: Key to update
        :param value: Value to update
        """
        pass

    # -------------- METADATA ----------------

    @abstractmethod
    def create_metadata(self, key, value):
        """
        Create metadata in the metadata store.

        :param key: Key of metadata
        :param value: Value of metadata
        """
        pass

    @abstractmethod
    def get_metadata(self, key):
        """
        Get metadata from the metadata store.

        :param key: Key of metadata
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
