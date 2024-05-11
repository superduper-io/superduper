import typing as t

import click
import tenacity
from pymongo.results import DeleteResult, InsertOneResult, UpdateResult

from superduperdb import logging
from superduperdb.backends.base.metadata import MetaDataStore
from superduperdb.components.component import Component
from superduperdb.misc.colors import Colors


class MongoMetaDataStore(MetaDataStore):
    """
    Metadata store for MongoDB.

    :param conn: MongoDB client connection
    :param name: Name of database to host filesystem
    """

    def __init__(
        self,
        conn: t.Any,
        name: t.Optional[str] = None,
    ) -> None:
        self.name = name
        self.db = conn[name]
        self.meta_collection = self.db['_meta']
        self.cdc_collection = self.db['_cdc_tables']
        self.component_collection = self.db['_objects']
        self.job_collection = self.db['_jobs']
        self.parent_child_mappings = self.db['_parent_child_mappings']
        self.conn = conn

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

    def get_parent_child_relations(self):
        """Get parent-child relations from the metadata store."""
        c = self.parent_child_mappings.find()
        return [(r['parent'], r['child']) for r in c]

    def get_component_version_children(self, unique_id: str):
        """Get the children of a component version.

        :param unique_id: unique identifier of component
        """
        return self.parent_child_mappings.distinct('child', {'parent': unique_id})

    def get_job(self, identifier: str):
        """Get a job from the metadata store.

        :param identifier: identifier of job
        """
        return self.job_collection.find_one({'identifier': identifier})

    def create_metadata(self, key: str, value: str):
        """Create metadata in the metadata store.

        :param key: key to be created
        :param value: value to be created
        """
        return self.meta_collection.insert_one({'key': key, 'value': value})

    def get_metadata(self, key: str):
        """Get metadata from the metadata store.

        :param key: key to be retrieved
        """
        return self.meta_collection.find_one({'key': key})['value']

    def update_metadata(self, key: str, value: str):
        """Update metadata in the metadata store.

        :param key: key to be updated
        :param value: value to be updated
        """
        return self.meta_collection.update_one({'key': key}, {'$set': {'value': value}})

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

    # TODO: Why is this is needed to prevent failures in CI?
    @tenacity.retry(stop=tenacity.stop_after_attempt(10))
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

    def list_components_in_scope(self, scope: str):
        """List components in a scope.

        :param scope: scope of components
        """
        out = []
        for r in self.component_collection.find({'parent': scope}):
            out.append((r['type_id'], r['identifier']))
        return out

    def show_job(self, job_id: str):
        """Show a job in the metadata store.

        :param job_id: identifier of job
        """
        return self.job_collection.find_one({'identifier': job_id})

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

    def _component_used(
        self, type_id: str, identifier: str, version: t.Optional[int] = None
    ) -> bool:
        """Check if a component is used in other components.

        :param type_id: type of component
        :param identifier: identifier of component
        :param version: version of component
        """
        if version is None:
            members: t.Union[t.Dict, str] = {'$regex': f'^{identifier}/{type_id}'}
        else:
            members = Component.make_unique_id(type_id, identifier, version)

        return bool(self.component_collection.count_documents({'members': members}))

    def component_has_parents(self, type_id: str, identifier: str) -> int:
        """Check if a component has parents.

        :param type_id: type of component
        :param identifier: identifier of component
        """
        doc = {'child': {'$regex': f'^{type_id}/{identifier}/'}}
        return self.parent_child_mappings.count_documents(doc)

    def component_version_has_parents(
        self, type_id: str, identifier: str, version: int
    ) -> int:
        """Check if a component version has parents.

        :param type_id: type of component
        :param identifier: identifier of component
        :param version: version of component
        """
        doc = {'child': Component.make_unique_id(type_id, identifier, version)}
        return self.parent_child_mappings.count_documents(doc)

    def delete_component_version(
        self, type_id: str, identifier: str, version: int
    ) -> DeleteResult:
        """Delete a component version from the metadata store.

        :param type_id: type of component
        :param identifier: identifier of component
        :param version: version of component
        """
        if self._component_used(type_id, identifier, version=version):
            raise Exception('Component version already in use in other components!')

        self.parent_child_mappings.delete_many(
            {'parent': Component.make_unique_id(type_id, identifier, version)}
        )

        return self.component_collection.delete_many(
            {
                'identifier': identifier,
                'type_id': type_id,
                'version': version,
            }
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
                }
            )
        else:
            r = self.component_collection.find_one(
                {
                    'identifier': identifier,
                    'type_id': type_id,
                    'version': version,
                },
            )
        return r

    def get_component_version_parents(self, unique_id: str) -> t.List[str]:
        """Get the parents of a component version.

        :param unique_id: unique identifier of component
        """
        return [
            r['parent'] for r in self.parent_child_mappings.find({'child': unique_id})
        ]

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

    def write_output_to_job(self, identifier, msg, stream):
        """Write output to a job in the metadata store.

        :param identifier: identifier of job
        :param msg: message to be written
        :param stream: stream to be written to
        """
        if stream not in ('stdout', 'stderr'):
            raise ValueError(f'stream is "{stream}", should be stdout or stderr')
        self.job_collection.update_one(
            {'identifier': identifier}, {'$push': {stream: msg}}
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
