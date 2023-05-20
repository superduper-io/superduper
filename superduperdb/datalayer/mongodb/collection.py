import functools

from pymongo.collection import Collection as MongoCollection

from superduperdb.datalayer.base.database import BaseDatabase
from superduperdb.cluster.client_decorators import vector_search


class Collection(MongoCollection):
    """
    Collection building on top of ``pymongo.collection.Collection``.
    Implements additional methods required by ``superduperdb`` for AI/ machine learning.
    """

    _id = '_id'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._hash_set = None
        self._semantic_index = None

        self.metrics = self.database.metrics
        self.models = self.database.models
        self.types = self.database.types

        self._all_hash_sets = self.database._all_hash_sets

    def _base_delete(self, filter_, *args, **kwargs):
        return super().delete_many(filter_, *args, **kwargs)

    def _base_update(self, op, filter_, update, *args, **kwargs):
        if op == 'update_many':
            return super().update_many(filter_, update, *args, **kwargs)
        elif op == 'replace_one':
            return super().replace_one(filter_, update, *args, **kwargs)

    @property
    def remote(self):
        return self.database.remote

    @remote.setter
    def remote(self, value):
        self.database.remote = value

    def __getitem__(self, item):
        if item == '_validation_sets':
            return self.database[f'{self.name}._validation_sets']
        return super().__getitem__(item)

    def predict(self, *args, **kwargs):
        return self.database.predict(*args, **kwargs)

    def predict_one(self, *args, **kwargs):
        return self.database.predict_one(*args, **kwargs)

    def _base_insert_many(self, *args, **kwargs):
        return super().insert_many(*args, **kwargs)

    def cancel_job(self, job_id):
        return self.database.cancel_job(job_id)

    @vector_search
    def clear_remote_cache(self):
        """
        Drop the hash_set currently in-use.
        """
        for k in self._all_hash_sets:
            del self._all_hash_sets[k]

    @functools.wraps(BaseDatabase.create_learning_task)
    def create_learning_task(self, models, keys, *query_params, **kwargs):
        """
        Create learning task.

        :param args: positional arguments to ``self.database.create_learning_task``
        :param kwargs: passed to ``self.database.create_learning_task``
        """
        return self.database.create_learning_task(
            models, keys, *('find', self.name, *query_params), **kwargs
        )

    def create_metric(self, *args, **kwargs):
        """
        Create metric.

        :param args: positional arguments to ``self.database.create_metric``
        :param kwargs: passed to ``self.database.create_metric``
        """
        return self.database.create_metric(*args, **kwargs)

    def create_model(self, *args, **kwargs):
        """
        Create a model.

        :param args: positional arguments to ``self.database.create_model``
        :param kwargs: passed to ``self.database.create_model``
        """
        return self.database.create_model(*args, **kwargs)

    def create_neighbourhood(self, *args, **kwargs):
        """
        Create neighbourhood.

        :param args: positional arguments to ``self.database.create_neighbourhood``
        :param kwargs: passed to ``self.database.create_neighbourhood``
        """
        return self.database.create_neighbourhood(*args, **kwargs)

    def create_trainer(self, *args, **kwargs):
        """
        Create trainer.

        :param args: positional arguments to ``self.database.create_trainer``
        :param kwargs: passed to ``self.database.create_trainer``
        """
        return self.database.create_trainer(*args, **kwargs)

    def create_type(self, *args, **kwargs):
        """
        Create type.

        :param args: positional arguments to ``self.database.create_type``
        :param kwargs: passed to ``self.database.create_type``
        """
        return self.database.create_type(*args, **kwargs)

    def create_validation_set(self, identifier, filter_=None, *args, **kwargs):
        """
        Create validation set.

        :param identifier: identifier of validation-set
        :param filter_: filter_ defining where to get data from
        :param args: positional arguments to ``self.database.create_validation_set``
        :param kwargs: passed to ``self.database.create_validation_set``
        """
        if filter_ is None:
            filter_ = {'_fold': 'valid'}
        else:
            filter_['_fold'] = 'valid'
        return self.database.create_validation_set(
            identifier, self.name, filter_, *args, **kwargs
        )

    def create_watcher(self, *args, **kwargs):
        """
        Create watcher.

        :param args: positional arguments to ``self.database.create_watcher``
        :param kwargs: passed to ``self.database.create_watcher``
        """
        return self.database.create_watcher(self.name, *args, **kwargs)

    def delete_agent(self, *args, **kwargs):
        """
        Delete agent
        """
        return self.database.delete_agent(*args, **kwargs)

    def delete_function(self, *args, **kwargs):
        """
        Delete function
        """
        return self.database.delete_function(*args, **kwargs)

    def delete_imputation(self, *args, **kwargs):
        """
        Delete imputation
        """
        return self.database.delete_imputation(*args, **kwargs)

    def delete_learning_task(self, *args, **kwargs):
        """
        Delete learning task
        """
        return self.database.delete_learning_task(*args, **kwargs)

    def delete_metric(self, *args, **kwargs):
        """
        Delete metric
        """
        return self.database.delete_metric(*args, **kwargs)

    def delete_model(self, *args, **kwargs):
        """
        Delete model
        """
        return self.database.delete_model(*args, **kwargs)

    def delete_neighbourhood(self, *args, **kwargs):
        """
        Delete neighbourhood
        """
        return self.database.delete_neighbourhood(*args, **kwargs)

    def delete_type(self, *args, **kwargs):
        """
        Delete type
        """
        return self.database.delete_type(*args, **kwargs)

    def delete_validation_set(self, *args, **kwargs):
        """
        Delete validation-set
        """
        return self.database.delete_validation_set(*args, **kwargs)

    def delete_watcher(self, *args, **kwargs):
        """
        Delete watcher
        """
        return self.database.delete_watcher(*args, **kwargs)

    def list_learning_tasks(self):
        """
        List learning-tasks.
        """
        return self.database.list_learning_tasks()

    def list_jobs(self):
        """
        List jobs.
        """
        return self.database.list_jobs()

    def list_metrics(self):
        """
        List metrics.
        """
        return self.database.list_metrics()

    def list_models(self):
        """
        List models.
        """
        return self.database.list_models()

    def list_types(self):
        """
        List types.
        """
        return self.database.list_types()

    def list_watchers(self):
        """
        List watchers.
        """
        return self.database.list_watchers()

    def find(self, *args, **kwargs):
        """
        Behaves like MongoDB ``find`` with similarity search as additional option.

        :param args: args passed to super(), along with ``"find"`` and ``self.name``
        :param kwargs: kwargs passed to super()
        """
        return self.database._select('find', self.name, *args, **kwargs)

    def find_one(self, *args, **kwargs):
        """
        Behaves like MongoDB ``find_one`` with exception of ``$like`` operator.
        See *Collection.find* for more details.

        :param args: args passed to super()
        :param kwargs: kwargs to be passed to super()
        """
        cursor = self.find(*args, **kwargs)
        for result in cursor.limit(-1):
            return result

    def insert_one(self, document, *args, **kwargs):
        """
        Insert a document into database.

        :param document: list of documents
        :param args: args to be passed to super()
        :param kwargs: kwargs to be passed to super()
        """
        return self.insert_many([document], *args, **kwargs)

    def insert_many(self, *args, **kwargs):
        """
        Insert many items into database.

        :param args: args to be passed to super()
        :param verbose: toggle to ``True`` to display outputs during computation
        :param refresh: toggle to ``False`` to suppress model processing
        :param kwargs: kwargs to be passed to super()
        """
        return self.database._insert(
            args[0], 'insert_many', self.name, *args[1:], **kwargs
        )

    def refresh_watcher(self, *args, **kwargs):
        """
        Recompute model outputs.

        :param args: position args passed to ``self.database.refresh_watcher``
        :param kwargs: kwargs passed to ``self.database.refresh_watcher``
        """
        return self.database.refresh_watcher(self.name, *args, **kwargs)

    def replace_one(self, filter, replacement, *args, **kwargs):
        """
        Replace a document in the database.
        The outputs of models will be refreshed for this document.

        :param filter: MongoDB like filter
        :param replacement: Replacement document
        :param args: args to be passed to super()
        :param kwargs: kwargs to be passed to super()
        """
        return self.database._update('replace_one', self.name, filter, replacement, *args, **kwargs)

    def update_many(self, *args, refresh=True, **kwargs):
        """
        Update the collection at the documents specified by the filter.
        If there are active models these are applied to the updated documents.

        :param args: Arguments to be passed to ``super()``
        :param refresh: Toggle to ``False`` to stop models being applied to
                        updated documents
        :param kwargs: Keyword arguments to be passed to ``super()``
        :return: ``result`` or ``(result, job_ids)`` depending on ``self.remote``
        """
        return self.database._update('update_many', self.name, *args, **kwargs)

    def update_one(self, filter, *args, **kwargs):
        """
        Update a single document specified by the filter.

        :param filter: Filter dictionary selecting documents to be updated
        :param args: Arguments to be passed to ``super()``
        :param kwargs: Keyword arguments to be passed to ``super()``
        :return: ``result`` or ``(result, job_ids)`` depending on ``self.remote``
        """
        id_ = super().find_one(filter, {'_id': 1})['_id']
        return self.update_many({'_id': id_}, *args, **kwargs)

    def watch_job(self, *args, **kwargs):
        """
        Watch stdout/stderr of worker job.
        """
        return self.database.watch_job(*args, **kwargs)
