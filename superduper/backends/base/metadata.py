import time
import typing as t
from abc import ABC, abstractmethod

from .data_backend import DataBackendProxy

if t.TYPE_CHECKING:
    from superduper.backends.base.query import Select


class NonExistentMetadataError(Exception):
    """NonExistentMetadataError.

    :param args: *args for `Exception`
    :param kwargs: **kwargs for `Exception`
    """


class MetaDataStore(ABC):
    """
    Abstraction for storing meta-data separately from primary data.

    :param uri: URI to the databackend database.
    :param flavour: Flavour of the databackend.
    :param callback: Optional callback to create connection.
    """

    def __init__(
        self,
        uri: t.Optional[str] = None,
        flavour: t.Optional[str] = None,
        callback: t.Optional[t.Callable] = None,
    ):
        self.uri = uri
        self.flavour = flavour

    @property
    def batched(self):
        """Batched metadata updates."""
        return False

    def expire(self, uuid: str):
        """Expire metadata batch cache if any.

        :param uuid: uuid to expire.
        """

    @abstractmethod
    def delete_parent_child(self, parent: str, child: str):
        """
        Delete parent-child mappings.

        :param parent: parent component
        :param child: child component
        """
        pass

    @abstractmethod
    def url(self):
        """Metadata store connection url."""
        pass

    @abstractmethod
    def create_component(self, info: t.Dict):
        """
        Create a component in the metadata store.

        :param info: dictionary containing information about the component.
        """
        pass

    @abstractmethod
    def create_job(self, info: t.Dict):
        """Create a job in the metadata store.

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

    def create_artifact_relation(self, uuid, artifact_ids):
        """
        Create a relation between an artifact and a component version.

        :param uuid: UUID of component version
        :param artifact: artifact
        """
        artifact_ids = (
            [artifact_ids] if not isinstance(artifact_ids, list) else artifact_ids
        )
        data = []
        for artifact_id in artifact_ids:
            data.append({'uuid': uuid, 'artifact_id': artifact_id})

        if data:
            self._create_data('_artifact_relations', data)

    def delete_artifact_relation(self, uuid, artifact_ids):
        """
        Delete a relation between an artifact and a component version.

        :param uuid: UUID of component version
        :param artifact: artifact
        """
        artifact_ids = (
            [artifact_ids] if not isinstance(artifact_ids, list) else artifact_ids
        )
        for artifact_id in artifact_ids:
            self._delete_data(
                '_artifact_relations',
                {
                    'uuid': uuid,
                    'artifact_id': artifact_id,
                },
            )

    def get_artifact_relations(self, uuid=None, artifact_id=None):
        """
        Get all relations between an artifact and a component version.

        :param artifact_id: artifact
        """
        if uuid is None and artifact_id is None:
            raise ValueError('Either `uuid` or `artifact_id` must be provided')
        elif uuid:
            relations = self._get_data(
                '_artifact_relations',
                {'uuid': uuid},
            )
            ids = [relation['artifact_id'] for relation in relations]
        else:
            relations = self._get_data(
                '_artifact_relations',
                {'artifact_id': artifact_id},
            )
            ids = [relation['uuid'] for relation in relations]
        return ids

    def get_children_relations(self, parent: str):
        """
        Get all children of a component.

        :param parent: parent component
        """
        relations = self._get_data('_parent_child', {'parent_id': parent})
        return [relation['child_id'] for relation in relations]

    # TODO: Refactor to use _create_data, _delete_data, _get_data
    @abstractmethod
    def _create_data(self, table_name, datas):
        pass

    @abstractmethod
    def _delete_data(self, table_name, filter):
        pass

    @abstractmethod
    def _get_data(self, table_name, filter):
        pass

    @abstractmethod
    def drop(self, force: bool = False):
        """
        Drop the metadata store.

        :param force: whether to force the drop (without confirmation)
        """
        pass

    @abstractmethod
    def set_component_status(self, uuid: str, status: str):
        """
        Set the status of a component.

        :param uuid: ``Component.uuid``
        :param status: ``Status`` to set
        """
        pass

    def get_component_status(self, uuid: str):
        """
        Get the status of a component.

        :param uuid: ``Component.uuid``
        """
        status = self._get_component_status(uuid)
        return status

    # ------------------ JOBS ------------------

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
    def show_jobs(
        self,
        identifier: t.Optional[str],
        type_id: t.Optional[str],
    ):
        """Show all jobs in the metadata store.

        :param identifier: identifier of component
        :param type_id: type of component
        """
        pass

    @abstractmethod
    def show_job_ids(self, uuids: t.Optional[str] = None, status: str = 'running'):
        """Show all jobs-id in the metadata store, with optional filters.

        :param uuids: Optional list of component uuids.
        :param status: Optional status of the job.
        """
        pass

    def show_components(self, type_id: t.Optional[str] = None):
        """
        Show all components in the metadata store.

        :param type_id: type of component
        :param **kwargs: additional arguments
        """
        results = self._show_components(type_id)
        if type_id is not None:
            identifiers = [data['identifier'] for data in results]
            identifiers = sorted(set(identifiers), key=lambda x: identifiers.index(x))
            return identifiers
        else:
            components = [
                {'identifier': data['identifier'], 'type_id': data['type_id']}
                for data in results
            ]
            # Remove duplicates
            components = [dict(t) for t in {tuple(d.items()) for d in components}]
            components = sorted(
                components, key=lambda x: (x['type_id'], x['identifier'])
            )
            return components

    # TODO is this used?
    @abstractmethod
    def show_cdc_tables(self):
        """List the tables used for CDC."""
        pass

    def show_cdcs(self, table):
        """
        Show the ``CDC`` components running on a given table.

        :param table: ``Table`` to consider.
        """
        results = self._show_cdcs(table)
        lookup = {}
        results = list(results)
        # Forces the latest versions out
        results = sorted(results, key=lambda r: r['version'])
        for r in results:
            lookup[r['type_id'], r['identifier']] = r['uuid']
        return list(lookup.values())

    @abstractmethod
    def _show_cdcs(self, table):
        pass

    @abstractmethod
    def _show_components(self, type_id: t.Optional[str] = None):
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
    def _get_component_uuid(self, type_id: str, identifier: str, version: int):
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

    @abstractmethod
    def get_component_by_uuid(self, uuid: str, allow_hidden: bool = False):
        """Get a component by UUID.

        :param uuid: UUID of component
        :param allow_hidden: whether to load hidden components
        """
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

    @abstractmethod
    def _replace_object(
        self,
        info,
        identifier: str | None = None,
        type_id: str | None = None,
        version: int | None = None,
        uuid: str | None = None,
        batch: bool = False,
    ):
        pass

    def replace_object(
        self,
        info: t.Dict[str, t.Any],
        identifier: str | None = None,
        type_id: str | None = None,
        version: int | None = None,
        uuid: str | None = None,
    ) -> None:
        """
        Replace an object in the metadata store.

        :param info: dictionary containing information about the object
        :param identifier: identifier of object
        :param type_id: type of object
        :param version: version of object
        """
        if version is None and uuid is None:
            assert isinstance(type_id, str)
            assert isinstance(identifier, str)
            version = self.get_latest_version(type_id, identifier)
        return self._replace_object(
            info=info,
            identifier=identifier,
            type_id=type_id,
            version=version,
            uuid=uuid,
        )

    @abstractmethod
    def get_component_version_parents(self, uuid: str):
        """
        Get the parents of a component version.

        :param uuid: unique identifier of component version
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

    def add_query(self, query: 'Select', model: str):
        """Add query id to query table.

        :param query: query object
        :param model: model identifier
        """
        raise NotImplementedError

    def get_queries(self, model: str):
        """Get all queries from query table corresponding to the model.

        :param model: model identifier
        """

    @abstractmethod
    def disconnect(self):
        """Disconnect the client."""


class MetaDataStoreProxy(DataBackendProxy):
    """
    Proxy class to DataBackend which acts as middleware for performing fallbacks.

    :param backend: Instance of `MetaDataStore`.
    """

    def __init__(self, backend):
        super().__init__(backend=backend)
