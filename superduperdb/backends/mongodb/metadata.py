import typing as t

import click
from pymongo.results import DeleteResult, InsertOneResult, UpdateResult

from superduperdb import logging
from superduperdb.backends.base.metadata import MetaDataStore
from superduperdb.misc.colors import Colors


class MongoMetaDataStore(MetaDataStore):
    """
    Metadata store for MongoDB.

    :param conn: MongoDB client connection
    :param name: Name of database to host filesystem
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
            from .data_backend import _connection_callback

            self.connection_callback = lambda: _connection_callback(uri, flavour)

        self.conn, self.name = self.connection_callback()
        self._setup()

    def _setup(self):
        self.db = self.conn[self.name]
        self.meta_collection = self.db['_meta']
        self.cdc_collection = self.db['_cdc_tables']
        self.component_collection = self.db['_objects']
        self.job_collection = self.db['_jobs']
        self.parent_child_mappings = self.db['_parent_child_mappings']

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
        return self.job_collection.find_one({'identifier': identifier})

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
            {'identifier': identifier}, {'$set': {key: value}}
        )

    def show_components(self, type_id: t.Optional[str] = None):
        """Show components in the metadata store.

        :param type_id: type of component
        """
        # TODO: Should this be sorted?
        if type_id is not None:
            return self.component_collection.distinct(
                'identifier', {'type_id': type_id}
            )
        else:
            return list(
                self.component_collection.find(
                    {}, {'identifier': 1, '_id': 0, 'type_id': 1}
                )
            )

    def show_component_versions(
        self, type_id: str, identifier: str
    ) -> t.List[t.Union[t.Any, int]]:
        """Show component versions in the metadata store.

        :param type_id: type of component
        :param identifier: identifier of component
        """
        return self.component_collection.distinct(
            'version', {'type_id': type_id, 'identifier': identifier}
        )

    def show_jobs(
        self,
        component_identifier: t.Optional[str] = None,
        type_id: t.Optional[str] = None,
    ):
        """Show jobs in the metadata store.

        :param component_identifier: identifier of component
        :param type_id: type of component
        """
        filter_ = {}
        if component_identifier is not None:
            filter_['component_identifier'] = component_identifier
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
        identifier: str,
        type_id: str,
        version: int,
    ) -> None:
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
