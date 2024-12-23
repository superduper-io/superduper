import typing as t
from functools import cached_property

from bson import ObjectId

from superduper import CFG, logging
from superduper.backends.base.query import Executor, Query


OPS_MAP = {
    '__eq__': '$eq',
    '__ne__': '$ne',
    '__lt__': '$lt',
    '__le__': '$lte',
    '__gt__': '$gt',
    '__ge__': '$gte',
    'isin': '$in',
}


class MongoDBExecutor(Executor):

    @cached_property
    def collection(self):
        return self.db.databackend.database[self.decomposition.table]

    @property
    def primary_id(self):
        return '_id'

    def _execute_insert(self, documents):
        return self.collection.insert_many(documents).inserted_ids

    def to_id(self, id):
        return ObjectId(id)

    @property
    def predict_ids(self):
        return self.decomposition.outputs.args

    @property
    def project(self):
        if self.decomposition.select is None:
            return {}

        if not self.decomposition.select.args:
            return {}

        project = {}
        for k in self.decomposition.select.args:
            if isinstance(k, Query):
                assert k.parts[0] == 'primary_id'
                project['_id'] = 1
            else:
                project[k] = 1

        if '_id' not in project:
            project['_id'] = 0

        return project

    @cached_property
    def filter(self):
        if self.decomposition.filter is None:
            return {}

        filters = self.decomposition.filter.args

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

            if f.decomposition.col == 'primary_id':
                if isinstance(value, str):
                    value = ObjectId(value)
                elif isinstance(value, list):
                    value = [ObjectId(x) for x in value]

            mongo_filter[key] = {OPS_MAP[op.name]: value}
        return mongo_filter

    @cached_property
    def limit(self):
        try:
            out = self.decomposition.limit.args[0]
            assert out > 0
            return out
        except AttributeError:
            return

    def _execute_outputs(self):

        pipeline = []

        project = self.project.copy()

        filter_mapping_base = {
            k: v for k, v in self.filter.items() 
            if not k.startswith(CFG.output_prefix)
        }
        filter_mapping_outputs = {
            k: v for k, v in self.filter.items() 
            if k.startswith(CFG.output_prefix)
        }

        if self.filter:
            pipeline.append({"$match": filter_mapping_base})
            project.update({k: 1 for k in filter_mapping_base.keys()})

        predict_ids = self.predict_ids

        if filter_mapping_outputs:
            predict_ids = [
                pid for pid in self.predict_ids 
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

        if self.limit:
            pipeline.append({"$limit": self.limit})

        try:
            import json
            logging.debug(f'Executing pipeline: {json.dumps(pipeline, indent=2)}')
        except TypeError:
            pass

        result = list(self.collection.aggregate(pipeline))

        for pid in predict_ids:
            k = f'{CFG.output_prefix}{pid}'
            for r in result:
                r[k] = r[k][k]
        return result

    def _execute_primary_id(self):
        return '_id'

    def _execute_select(self):

        if self.decomposition.outputs:
            return self._execute_outputs()

        if self.limit:
            return list(self.collection.find(self.filter, self.project).limit(self.limit))

        return list(self.collection.find(self.filter, self.project))