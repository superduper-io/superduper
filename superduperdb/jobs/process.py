import contextlib
import datetime
import traceback

from superduperdb.client import the_client


class Logger:
    def __init__(self, collection, id_, stream='stdout'):
        self.collection = collection
        self.id_ = id_
        self.stream = stream

    def write(self, message):
        if self.stream == 'stdout':
            self.collection['_jobs'].update_one(
                {'identifier': self.id_},
                {'$push': {'stdout': message}}
            )
        elif self.stream == 'stderr':
            self.collection['_jobs'].update_one(
                {'identifier': self.id_},
                {'$push': {'stderr': message}}
            )
        else:
            raise NotImplementedError

    def flush(self):
        pass


def handle_function_output(function, collection, identifier, *args, **kwargs):
    with contextlib.redirect_stdout(Logger(collection, identifier)):
        with contextlib.redirect_stderr(Logger(collection, identifier, stream='stderr')):
            return function(*args, **kwargs)


def _function_job(database_name, collection_name, function_name, identifier,
                  args_, kwargs_):
    collection = the_client[database_name][collection_name]
    function = getattr(collection, function_name)
    collection['_jobs'].update_one({'identifier': identifier},
                                   {'$set': {'status': 'running'}})
    try:
        handle_function_output(
            function,
            collection,
            identifier,
            *args_,
            **kwargs_,
        )
    except Exception as e:
        tb = traceback.format_exc()
        collection['_jobs'].update_one(
            {'identifier': identifier},
            {'$set': {'status': 'failed', 'msg': tb}}
        )
        raise e
    collection['_jobs'].update_one({'identifier': identifier},
                                   {'$set': {'status': 'success'}})

