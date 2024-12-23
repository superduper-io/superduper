import typing as t

import click
from pymongo.results import DeleteResult, InsertOneResult, UpdateResult
from superduper import logging
from superduper.backends.base.metadata import MetaDataStore
from superduper.components.component import Status
from superduper.misc.colors import Colors


class MongoMetaDataStore(MetaDataStore):
    """
    Metadata store for MongoDB.

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
        super().__init__(uri=uri, flavour=flavour)

        if callback:
            self.connection_callback = callback
        else:
            assert uri
            from .utils import connection_callback

            self.connection_callback = lambda: connection_callback(uri, flavour)

        self.conn, self.name = self.connection_callback()
        self._setup()

    def _setup(self):
        self.db = self.conn[self.name]
        self.meta_collection = self.db['_meta']
        self.component_collection = self.db['_objects']
        self.job_collection = self.db['_jobs']
        self.parent_child_mappings = self.db['_parent_child_mappings']
        self.artifact_relations = self.db['_artifact_relations']

    def reconnect(self):
        """Reconnect to metdata store."""
        self.conn, self.name = self.connection_callback()
        self._setup()

    def url(self):
        """Metadata store connection url."""
        return self.conn.HOST + ':' + str(self.conn.PORT) + '/' + self.name

    def drop(self, force: bool = False):
        """Drop all meta-data from the metadata store.

        Please always use with caution. This will drop all the meta-data collections.
        :param force: whether to force the drop, defaults to False
        """
        if not force:
            if not click.confirm(
                f'{Colors.RED}[!!!WARNING USE WITH CAUTION AS YOU '
                f'WILL LOSE ALL DATA!!!]{Colors.RESET} '
                'Are you sure you want to drop all meta-data? ',
                default=False,
            ):
                logging.warn('Aborting...')
        self.db.drop_collection(self.meta_collection.name)
        self.db.drop_collection(self.component_collection.name)
        self.db.drop_collection(self.job_collection.name)
        self.db.drop_collection(self.parent_child_mappings.name)
        self.db.drop_collection(self.artifact_relations.name)

    def delete_parent_child(self, parent: str, child: str) -> None:
        """
        Delete parent-child relationships between two components.

        :param parent: parent component uuid
        :param child: child component uuid
        """
        self.parent_child_mappings.delete_many(
            {
                'parent': parent,
                'child': child,
            }
        )

    def create_parent_child(self, parent: str, child: str) -> None:
        """Create a parent-child relationship between two components.

        :param parent: parent component
        :param child: child component
        """
        self.parent_child_mappings.insert_one(
            {
                'parent': parent,
                'child': child,
            }
        )

    def _create_data(self, table_name, datas):
        collection = self.db[table_name]
        collection.insert_many(datas)

    def _delete_data(self, table_name, filter):
        collection = self.db[table_name]
        collection.delete_many(filter)

    def _get_data(self, table_name, filter):
        collection = self.db[table_name]
        return list(collection.find(filter))

    def create_component(self, info: t.Dict) -> InsertOneResult:
        """Create a component in the metadata store.

        :param info: dictionary containing information about the component.
        """
        if 'hidden' not in info:
            info['hidden'] = False
        return self.component_collection.insert_one(info)

    def create_job(self, info: t.Dict) -> InsertOneResult:
        """Create a job in the metadata store.

        :param info: dictionary containing information about the job.
        """
        return self.job_collection.insert_one(info)

    def get_job(self, identifier: str):
        """Get a job from the metadata store.

        :param identifier: identifier of job
        """
        return self.job_collection.find_one({'job_id': identifier})

    def get_latest_version(
        self, type_id: str, identifier: str, allow_hidden: bool = False
    ) -> int:
        """Get the latest version of a component.

        :param type_id: type of component
        :param identifier: identifier of component
        :param allow_hidden: whether to allow hidden components
        """
        try:
            if allow_hidden:
                return sorted(
                    self.component_collection.distinct(
                        'version', {'identifier': identifier, 'type_id': type_id}
                    )
                )[-1]
            else:

                def f():
                    return self.component_collection.distinct(
                        'version',
                        {
                            '$or': [
                                {
                                    'identifier': identifier,
                                    'type_id': type_id,
                                    'hidden': False,
                                },
                                {
                                    'identifier': identifier,
                                    'type_id': type_id,
                                    'hidden': {'$exists': 0},
                                },
                            ]
                        },
                    )

                return sorted(f())[-1]
        except IndexError:
            raise FileNotFoundError(f'Can\'t find {type_id}: {identifier} in metadata')

    def update_job(self, identifier: str, key: str, value: t.Any) -> UpdateResult:
        """Update a job in the metadata store.

        :param identifier: identifier of job
        :param key: key to be updated
        :param value: value to be updated
        """
        return self.job_collection.update_one(
            {'job_id': identifier}, {'$set': {key: value}}
        )

    def show_cdc_tables(self):
        """Show tables to be consumed with cdc."""
        return self.component_collection.distinct('cdc_table')

    def _show_cdcs(self, table):
        return list(
            self.component_collection.find(
                {'cdc_table': table},
                {'identifier': 1, '_id': 0, 'type_id': 1, 'version': 1, 'uuid': 1},
            )
        )

    def _show_components(self, type_id: t.Optional[str] = None):
        """Show components in the metadata store.

        :param type_id: type of component
        """
        filter = {}
        if type_id is not None:
            filter['type_id'] = type_id
        return list(
            self.component_collection.find(
                filter, {'identifier': 1, '_id': 0, 'type_id': 1}
            )
        )

    def show_component_versions(
        self, type_id: str, identifier: str
    ) -> t.List[t.Union[t.Any, int]]:
        """Show component versions in the metadata store.

        :param type_id: type of component
        :param identifier: identifier of component
        """
        return sorted(
            self.component_collection.distinct(
                'version', {'type_id': type_id, 'identifier': identifier}
            )
        )

    def show_job_ids(self, uuids: t.Optional[str] = None, status: str = 'running'):
        """Show all jobs in the metadata store.

        :param uuids: list of uuids
        :param status: status of the job
        """
        if uuids:
            return self.job_collection.distinct(
                'job_id', {'status': status, 'uuid': {'$in': uuids}}
            )
        return self.job_collection.distinct('job_id', {'status': status})

    def show_jobs(
        self,
        identifier: t.Optional[str] = None,
        type_id: t.Optional[str] = None,
    ):
        """Show jobs in the metadata store.

        :param component_identifier: identifier of component
        :param type_id: type of component
        """
        filter_ = {}
        if identifier is not None:
            filter_['identifier'] = identifier
        if type_id is not None:
            filter_['type_id'] = type_id
        return list(self.job_collection.find(filter_))

    def _get_component_uuid(self, type_id: str, identifier: str, version: int) -> str:
        return self.component_collection.find_one(
            {'type_id': type_id, 'identifier': identifier, 'version': version},
            {'uuid': 1},
        )['uuid']

    def component_version_has_parents(
        self, type_id: str, identifier: str, version: int
    ) -> int:
        """Check if a component version has parents.

        :param type_id: type of component
        :param identifier: identifier of component
        :param version: version of component
        """
        uuid = self.component_collection.find_one(
            {'type_id': type_id, 'identifier': identifier, 'version': version},
            {'uuid': 1, 'id': 1},
        )['uuid']
        doc = {'child': uuid}
        return self.parent_child_mappings.count_documents(doc)

    def delete_component_version(
        self, type_id: str, identifier: str, version: int
    ) -> DeleteResult:
        """Delete a component version from the metadata store.

        :param type_id: type of component
        :param identifier: identifier of component
        :param version: version of component
        """
        uuid = self._get_component_uuid(type_id, identifier, version)

        self.parent_child_mappings.delete_many({'parent': uuid})

        return self.component_collection.delete_many(
            {
                'identifier': identifier,
                'type_id': type_id,
                'version': version,
            }
        )

    def get_component_by_uuid(self, uuid: str, allow_hidden: bool = False):
        """Get a component by UUID.

        :param uuid: UUID of component
        :param allow_hidden: whether to load hidden components
        """
        r = self.component_collection.find_one({'uuid': uuid})
        if r is None:
            raise FileNotFoundError(f'Can\'t find {uuid} in metadata')
        return self._get_component(
            type_id=r['type_id'],
            identifier=r['identifier'],
            version=r['version'],
            allow_hidden=allow_hidden,
        )

    def _get_component(
        self,
        type_id: str,
        identifier: str,
        version: int,
        allow_hidden: bool = False,
    ) -> t.Dict[str, t.Any]:
        if not allow_hidden:
            r = self.component_collection.find_one(
                {
                    '$or': [
                        {
                            'identifier': identifier,
                            'type_id': type_id,
                            'version': version,
                            'hidden': False,
                        },
                        {
                            'identifier': identifier,
                            'type_id': type_id,
                            'version': version,
                            'hidden': {'$exists': 0},
                        },
                    ]
                },
                {'_id': 0},
            )
        else:
            r = self.component_collection.find_one(
                {
                    'identifier': identifier,
                    'type_id': type_id,
                    'version': version,
                },
                {'_id': 0},
            )
        return r

    def get_component_version_parents(self, uuid: str) -> t.List[str]:
        """Get the parents of a component version.

        :param uuid: unique identifier of component
        """
        return [r['parent'] for r in self.parent_child_mappings.find({'child': uuid})]

    def _replace_object(
        self,
        info: t.Dict[str, t.Any],
        identifier: str | None = None,
        type_id: str | None = None,
        version: int | None = None,
        uuid: str | None = None,
    ) -> None:
        if uuid:
            return self.component_collection.replace_one({'uuid': uuid}, info)
        else:
            assert type_id, 'type_id cannot be False in replace'
            assert identifier, "identifier cannot be False in replace"
            assert version is not None, 'version cannot be None in replace'
            self.component_collection.replace_one(
                {'identifier': identifier, 'type_id': type_id, 'version': version},
                info,
            )

    def _update_object(
        self,
        identifier: str,
        type_id: str,
        key: str,
        value: t.Any,
        version: int,
    ):
        """Update a component in the metadata store.

        :param identifier: identifier of component
        :param type_id: type of component
        :param key: key to be updated
        :param value: value to be updated
        :param version: version of component
        """
        return self.component_collection.update_one(
            {'identifier': identifier, 'type_id': type_id, 'version': version},
            {'$set': {key: value}},
        )

    def hide_component_version(
        self, type_id: str, identifier: str, version: int
    ) -> None:
        """Hide a component version.

        :param type_id: type of component
        :param identifier: identifier of component
        :param version: version of component
        """
        self.component_collection.update_one(
            {'type_id': type_id, 'identifier': identifier, 'version': version},
            {'$set': {'hidden': True}},
        )

    def disconnect(self):
        """Disconnect the client."""

        # TODO: implement me

    def set_component_status(self, uuid, status: Status):
        """Set status of component with `status`."""
        return self.component_collection.update_one(
            {'uuid': uuid}, {'$set': {'status': status}}
        )

    def _get_component_status(self, uuid):
        """Get status of component."""
        data = self.component_collection.find_one({'uuid': uuid}, {'status': 1})
        return data['status'] if data else None
