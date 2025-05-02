import hashlib
import json
import typing as t

import click
from bson.objectid import ObjectId
from superduper import CFG, logging
from superduper.backends.base.data_backend import BaseDataBackend
from superduper.base.query import Query
from superduper.base.schema import Schema

from superduper_mongodb.utils import connection_callback

OPS_MAP = {
    '__eq__': '$eq',
    '__ne__': '$ne',
    '__lt__': '$lt',
    '__le__': '$lte',
    '__gt__': '$gt',
    '__ge__': '$gte',
    'isin': '$in',
}


class MongoDBDataBackend(BaseDataBackend):
    """
    Data backend for MongoDB.

    :param uri: URI to the databackend database.
    :param flavour: Flavour of the databackend.
    """

    id_field = "_id"

    def __init__(self, uri: str, plugin: t.Any, flavour: t.Optional[str] = None):
        self.connection_callback = lambda: connection_callback(uri, flavour)

        super().__init__(uri, flavour=flavour, plugin=plugin)

        self.conn, self.name = connection_callback(uri, flavour)
        self._database = self.conn[self.name]

        self.datatype_presets = {
            'vector': 'superduper.components.datatype.NativeVector'
        }

    def create_id(self, item: str):
        """Create a unique ID for the item.

        :param item: The item to create an ID for.
        """
        hash_object = hashlib.sha256(item.encode())
        return hash_object.hexdigest()[:24]

    def random_id(self):
        """Generate a random ID."""
        return ObjectId()

    def reconnect(self):
        """Reconnect to MongoDB databackend."""
        conn, _ = self.connection_callback()
        self.conn = conn
        self._database = self.conn[self.name]

    def drop_table(self, name: str):
        """Drop the table or collection.

        Please use with caution as you will lose all data.
        :param name: Collection to drop.
        """
        return self._database.drop_collection(name)

    def drop(self, force: bool = False):
        """Drop the data backend.

        Please use with caution as you will lose all data.
        :param force: Force the drop, default is False.
                      If False, a confirmation prompt will be displayed.
        """
        if not force:
            if not click.confirm(
                '!!!WARNING USE WITH CAUTION AS YOU '
                "WILL LOSE ALL DATA!!!]\n"
                "Are you sure you want to drop the data-backend? ",
                default=False,
            ):
                logging.warn("Aborting...")
        return self._database.client.drop_database(self._database.name)

    def get_table(self, identifier):
        """Get a table or collection from the data backend.

        :param identifier: table or collection identifier
        """
        return self._database[identifier]

    def list_tables(self):
        """List all tables or collections in the data backend."""
        return self._database.list_collection_names()

    def check_output_dest(self, predict_id) -> bool:
        """Check if the output destination exists.

        :param predict_id: identifier of the prediction
        """
        return self._database[f"{CFG.output_prefix}{predict_id}"].find_one() is not None

    def create_table_and_schema(self, identifier: str, schema: Schema, primary_id: str):
        """Create a table and schema in the data backend.

        :param identifier: The identifier for the table
        :param mapping: The mapping for the schema
        """
        pass

    ###################################
    # Query execution implementations #
    ###################################

    def primary_id(self, query):
        """Get the primary ID for the query."""
        return '_id'

    def replace(self, table: str, condition: t.Dict, r: t.Dict) -> t.List[str]:
        """Replace data.

        :param table: The table to insert into.
        :param condition: The condition to replace.
        :param r: The data to replace.
        """
        if '_id' in r:
            del r['_id']
        self._database[table].replace_one(condition, r, upsert=True)

    def insert(self, table, documents):
        """Insert documents into the table."""
        for doc in documents:
            if '_id' in doc:
                doc['_id'] = ObjectId(doc['_id'])
            if '_source' in doc:
                doc['_source'] = ObjectId(doc['_source'])
        return self._database[table].insert_many(documents).inserted_ids

    def update(self, table, condition, key, value):
        """Update the table with the key and value."""
        self._database[table].update_many(condition, {'$set': {key: value}})

    def delete(self, table, condition):
        """Delete data from the table."""
        return self._database[table].delete_many(condition)

    def missing_outputs(self, query, predict_id: str):
        """Get the missing outputs for the prediction."""
        key = f'{CFG.output_prefix}{predict_id}'
        lookup = [
            {
                '$lookup': {
                    'from': key,
                    'localField': '_id',
                    'foreignField': '_source',
                    'as': key,
                }
            },
            {'$match': {key: {'$size': 0}}},
        ]
        collection = self._database[query.table]
        results = list(collection.aggregate(lookup))
        return [r['_id'] for r in results]

    def select(self, query: Query):
        """Select data from the table."""
        if query.decomposition.outputs:
            return self._outputs(query)

        collection = self._database[query.table]

        logging.debug(str(query))

        limit = self._get_limit(query)
        if limit:
            return list(
                collection.find(
                    self._mongo_filter(query), self._get_project(query)
                ).limit(limit)
            )

        return list(
            collection.find(self._mongo_filter(query), self._get_project(query))
        )

    def to_id(self, id):
        """Convert the ID to the correct format."""
        return ObjectId(id)

    ########################
    # Helper methods below #
    ########################

    @staticmethod
    def _get_project(query):
        if query.decomposition.select is None:
            return {}

        if not query.decomposition.select.args:
            return {}

        project = {}
        for k in query.decomposition.select.args:
            if isinstance(k, Query):
                assert k.parts[0] == 'primary_id'
                project['_id'] = 1
            else:
                project[k] = 1

        if '_id' not in project:
            project['_id'] = 0

        return project

    @staticmethod
    def _mongo_filter(query):
        if query.decomposition.filter is None:
            return {}

        filters = query.decomposition.filter.args

        mongo_filter = {}
        for f in filters:
            assert len(f) > 2, f'Invalid filter query: {f}'
            key = f.parts[0]
            if key == 'primary_id':
                key = '_id'

            op = f.parts[1]

            if op.name not in OPS_MAP:
                raise ValueError(
                    f'Operation {op} not supported, '
                    f'supported operations are: {OPS_MAP.keys()}'
                )

            if not op.args:
                raise ValueError(f'No arguments found for operation {op}')

            value = op.args[0]

            if key == '_id':
                if isinstance(value, str):
                    value = ObjectId(value)
                elif isinstance(value, list):
                    value = [ObjectId(x) for x in value]

            mongo_filter[key] = {OPS_MAP[op.name]: value}
        return mongo_filter

    @staticmethod
    def _get_limit(query):
        try:
            out = query.decomposition.limit.args[0]
            assert out > 0
            return out
        except AttributeError:
            return

    def _outputs(self, query):
        pipeline = []

        project = self._get_project(query).copy()

        filter_mapping_base = {
            k: v
            for k, v in self._mongo_filter(query).items()
            if not k.startswith(CFG.output_prefix)
        }
        filter_mapping_outputs = {
            k: v
            for k, v in self._mongo_filter(query).items()
            if k.startswith(CFG.output_prefix)
        }

        if filter_mapping_base:
            pipeline.append({"$match": filter_mapping_base})
            if project:
                project.update({k: 1 for k in filter_mapping_base.keys()})

        predict_ids = query.decomposition.predict_ids

        if filter_mapping_outputs:
            predict_ids = [
                pid
                for pid in predict_ids
                if f'{CFG.output_prefix}{pid}' in filter_mapping_outputs
            ]

        for predict_id in predict_ids:
            key = f'{CFG.output_prefix}{predict_id}'
            lookup = {
                "$lookup": {
                    "from": key,
                    "localField": "_id",
                    "foreignField": "_source",
                    "as": key,
                }
            }

            if project:
                project[key] = 1

            pipeline.append(lookup)

            if key in filter_mapping_outputs:
                pipeline.append({"$match": {key: filter_mapping_outputs[key]}})

            pipeline.append(
                {"$unwind": {"path": f"${key}", "preserveNullAndEmptyArrays": True}}
            )

        if project:
            pipeline.append({"$project": project})

        if self._get_limit(query):
            pipeline.append({"$limit": self._get_limit(query)})

        try:
            import json

            logging.debug(f'Executing pipeline: {json.dumps(pipeline, indent=2)}')
        except TypeError:
            pass

        collection = self._database[query.table]
        result = list(collection.aggregate(pipeline))

        for pid in predict_ids:
            k = f'{CFG.output_prefix}{pid}'
            for r in result:
                r[k] = r[k][k]
        return result

    def execute_native(self, query: str):
        """Execute a native MongoDB pipeline with aggregate.

        :param query: JSON string with the collection and pipeline as a list.

        >>> query = '{"collection": "c", "pipeline": [{"$match": {"field": "value"}}]}'
        >>> backend.execute_native(query)
        """
        parsed = json.loads(query)
        collection = parsed['collection']
        pipeline = parsed['pipeline']
        results = list(self._database[collection].aggregate(pipeline))
        for r in results:
            if '_id' in r:
                r['_id'] = str(r['_id'])
        return results
