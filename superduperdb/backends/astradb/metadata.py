import typing as t
import click
from superduperdb import logging
from superduperdb.backends.base.metadata import MetaDataStore
from superduperdb.components.component import Component
from superduperdb.misc.colors import Colors
from astrapy.db import AstraDB, AstraDBCollection


class AstraMetaDataStore(MetaDataStore):
    """
    Metadata store for AstraDB.

    :param conn: AstraDB client connection
    :param name: Name of database to host filesystem
    """

    def __init__(
            self,
            conn: AstraDB, name: str
    ) -> None:
        self.name = name
        self.db = conn
        self.parent_child_mappings = None
        self.job_collection = None
        self.component_collection = None
        self.cdc_collection = None
        self.meta_collection = None
        self.initialize_collections(self.db)

    @classmethod
    def get_table_names(cls) -> list:
        """Get table names."""
        return ["meta", "cdc_tables", "objects", "jobs", "parent_child_mappings"]

    def initialize_collections(self, conn: AstraDB) -> None:
        """Initialize collections."""
        table_names = self.get_table_names()
        for table_name in table_names:
            if table_name != "cdc_tables":
                self.db.create_collection(collection_name=table_name)
        self.meta_collection = AstraDBCollection(collection_name=table_names[0], astra_db=conn)
        # self.cdc_collection = AstraDBCollection(collection_name=table_names[1], astra_db=conn)
        self.component_collection = AstraDBCollection(collection_name=table_names[2], astra_db=conn)
        self.job_collection = AstraDBCollection(collection_name=table_names[3], astra_db=conn)
        self.parent_child_mappings = AstraDBCollection(collection_name=table_names[4], astra_db=conn)

    def url(self):
        """
        Databackend connection url
        """
        return self.db.base_url

    def drop(self, force: bool = False):
        if not force:
            if not click.confirm(
                    f'{Colors.RED}[!!!WARNING USE WITH CAUTION AS YOU '
                    f'WILL LOSE ALL DATA!!!]{Colors.RESET} '
                    'Are you sure you want to drop all meta-data? ',
                    default=False,
            ):
                logging.warn('Aborting...')
        table_names = self.get_table_names()
        for table_name in table_names:
            if table_name != "cdc_tables":
                self.db.delete_collection(collection_name=table_name)

    def create_parent_child(self, parent: str, child: str) -> None:
        self.parent_child_mappings.insert_one(
            document={
                'parent': parent,
                'child': child,
            }
        )

    def create_component(self, info: t.Dict):
        if 'hidden' not in info:
            info['hidden'] = False
        return self.component_collection.insert_one(document=info).get('status')

    def create_job(self, info: t.Dict):
        return self.job_collection.insert_one(document=info).get('status')

    def get_job(self, identifier: str):
        return self.job_collection.find_one(filter={'identifier': identifier}).get('data')['document']

    def create_metadata(self, key: str, value: str):
        return self.meta_collection.insert_one(document={'key': key, 'value': value}).get('status')

    def get_metadata(self, key: str):
        return self.meta_collection.find_one(filter={'key': key}).get('data')['document']

    def update_metadata(self, key: str, value: str):
        return self.meta_collection.update_one(filter={'key': key}, update={'$set': {'value': value}}).get('status')

    def get_latest_version(
            self, type_id: str, identifier: str, allow_hidden: bool = False
    ) -> int:
        try:
            distinct_values = []
            if allow_hidden:
                response_generator = self.component_collection.paginated_find(
                    filter={'identifier': identifier, 'type_id': type_id})
                for document in response_generator:
                    if document['version'] not in distinct_values:
                        distinct_values.append(document['version'])
                if distinct_values:  # Check if the list is not empty
                    return sorted(distinct_values)[-1]
                else:
                    return None 
            else:
                response_generator = self.component_collection.paginated_find(
                    filter={
                        '$or': [
                            {
                                'identifier': identifier,
                                'type_id': type_id,
                                'hidden': False,
                            },
                            {
                                'identifier': identifier,
                                'type_id': type_id,
                                'hidden': {'$exists': False},
                            },
                        ]
                    },
                )
                for document in response_generator:
                    if document['version'] not in distinct_values:
                        distinct_values.append(document['version'])
                if distinct_values:  # Check if the list is not empty
                    return sorted(distinct_values)[-1]
                else:
                    return None 
        except IndexError:
            raise FileNotFoundError(f'Can\'t find {type_id}: {identifier} in metadata')

    def update_job(self, identifier: str, key: str, value: t.Any):
        return self.job_collection.update_one(
            filter={'identifier': identifier}, update={'$set': {key: value}}
        ).get('status')

    def show_components(self, type_id: str, **kwargs) -> t.List[t.Union[t.Any, str]]:
        distinct_values = []
        results = []
        response_generator = self.component_collection.paginated_find(
            filter={'type_id': type_id, **kwargs}
        )
        for document in response_generator:
            if document['identifier'] not in distinct_values:
                distinct_values.append(document['identifier'])
                results.append(document)
        return results

    def show_component_versions(
            self, type_id: str, identifier: str
    ) -> t.List[t.Union[t.Any, int]]:
        result = self.component_collection.find(
            filter={'type_id': type_id, 'identifier': identifier}
        )
        distinct_values = []
        results = []
        for doc in result:
            if doc['version'] not in distinct_values:
                distinct_values.append(doc['version'])
                results.append(doc)
        return results

    def show_job(self, job_id: str):
        return self.job_collection.find_one(filter={'identifier': job_id}).get('data')['document']

    def show_jobs(self, status=None):
        document_list = []
        status = {} if status is None else {'status': status}
        response_generator = self.job_collection.paginated_find(
            filter=status, projection={'identifier': 1, '_id': 0, 'method': 1, 'status': 1, 'time': 1}
        )
        for document in response_generator:
            document_list.append(document)
        return document_list

    def _component_used(
            self, type_id: str, identifier: str, version: t.Optional[int] = None
    ) -> bool:
        if version is None:
            members: t.Union[t.Dict, str] = {'$regex': f'^{identifier}/{type_id}'}
        else:
            members = Component.make_unique_id(type_id, identifier, version)

        return bool(self.component_collection.count_documents(filter={'members': members}).get('status')['count'])

    def component_version_has_parents(
            self, type_id: str, identifier: str, version: int
    ) -> int:
        doc = {'child': Component.make_unique_id(type_id, identifier, version)}
        return self.parent_child_mappings.count_documents(filter=doc).get('status')['count']

    def delete_component_version(
            self, type_id: str, identifier: str, version: int
    ):
        if self._component_used(type_id, identifier, version=version):
            raise Exception('Component version already in use in other components!')

        self.parent_child_mappings.delete_many(
            {'parent': Component.make_unique_id(type_id, identifier, version)}
        )

        return self.component_collection.delete_many(
            filter={
                'identifier': identifier,
                'type_id': type_id,
                'version': version,
            }
        ).get('status')

    def _get_component(
            self,
            type_id: str,
            identifier: str,
            version: int,
            allow_hidden: bool = False,
    ) -> t.Dict[str, t.Any]:
        print("inside child _get_component")
        if not allow_hidden:
            r = self.component_collection.find_one(
                filter={
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
                            'hidden': {'$exists': False},
                        },
                    ]
                }
            )
        else:
            r = self.component_collection.find_one(
                filter={
                    'identifier': identifier,
                    'type_id': type_id,
                    'version': version,
                },
            )
        return r.get('data')['document']

    def get_component_version_parents(self, unique_id: str) -> t.List[str]:
        response_generator = self.parent_child_mappings.paginated_find(filter={'child': unique_id})
        document_list = []
        for document in response_generator:
            document_list.append(document)
        return [
            r['parent'] for r in document_list
        ]

    def _replace_object(
            self,
            info: t.Dict[str, t.Any],
            identifier: str,
            type_id: str,
            version: int,
    ) -> None:
        self.component_collection.find_one_and_replace(
            filter={'identifier': identifier, 'type_id': type_id, 'version': version},
            replacement=info,
        )

    def _update_object(
            self,
            identifier: str,
            type_id: str,
            key: str,
            value: t.Any,
            version: int,
    ):
        return self.component_collection.update_one(
            filter={'identifier': identifier, 'type_id': type_id, 'version': version},
            update={'$set': {key: value}},
        ).get('status')

    def write_output_to_job(self, identifier, msg, stream):
        if stream not in ('stdout', 'stderr'):
            raise ValueError(f'stream is "{stream}", should be stdout or stderr')
        self.job_collection.update_one(
            filter={'identifier': identifier}, update={'$push': {stream: msg}}
        )

    def hide_component_version(
            self, type_id: str, identifier: str, version: int
    ) -> None:
        self.component_collection.update_one(
            filter={'type_id': type_id, 'identifier': identifier, 'version': version},
            update={'$set': {'hidden': True}},
        )

    def disconnect(self):
        """
        Disconnect the client
        """
