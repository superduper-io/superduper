from typing import Dict, Any

from superduperdb.datalayer.base.metadata import MetaDataStore


class MongoMetaDataStore(MetaDataStore):
    def __init__(self, db, object_collection, meta_collection, job_collection):
        """
        :param db: pymongo database connection
        :param object_collection: name of collection to store component information
        :param meta_collection: name of collection to store meta-data
        :param job_collection: name of collection to store job information
        """
        self.meta_collection = db[meta_collection]
        self.object_collection = db[object_collection]
        self.job_collection = db[job_collection]

    def create_component(self, info: Dict):
        return self.object_collection.insert_one(info)

    def create_job(self, info: Dict):
        return self.job_collection.insert_one(info)

    def get_job(self, identifier: str):
        return self.job_collection.find_one({'identifier': identifier})

    def get_metadata(self, key):
        return self.meta_collection.find_one({'key': key})['value']

    def update_job(self, identifier: str, key: str, value: Any):
        return self.job_collection.update_one(
            {'identifier': identifier}, {'$set': {key: value}}
        )

    def watch_job(self, job_id: str):
        pass

    def list_components(self, variety: str):
        return self.object_collection.distinct('identifier', {'variety': variety})

    def list_jobs(self, status=None):
        status = {} if status is None else {'status': status}
        return list(
            self.job_collection.find(
                status, {'identifier': 1, '_id': 0, 'method': 1, 'status': 1, 'time': 1}
            )
        )

    def delete_component(self, identifier: str, variety: str):
        return self.object_collection.delete_one(
            {'identifier': identifier, 'variety': variety}
        )

    def _get_object(self, identifier: str, variety: str):
        return self.object_collection.find_one(
            {'identifier': identifier, 'variety': variety}
        )

    def update_object(self, identifier, variety, key, value):
        ...

    def write_output_to_job(self, identifier, msg, stream):
        ...
