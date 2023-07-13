import contextlib
import datetime
import traceback

from sddb.client import the_client


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


def _handle_function_output(func, database, collection, *args, identifier=None, **kwargs):
    try:
        collection = the_client[database][collection]
        collection['_jobs'].insert_one({
            'identifier': identifier,
            'time': datetime.datetime.now(),
            'status': 'running',
            'func': func.__name__,
            'args': args,
            'kwargs': kwargs,
            'stdout': [],
            'stderr': [],
        })
        with contextlib.redirect_stdout(Logger(collection, identifier)):
            with contextlib.redirect_stderr(Logger(collection, identifier, stream='stderr')):
                collection.single_thread = True
                out = func(collection, *args, **kwargs)
                collection['_jobs'].update_one(
                    {'identifier': identifier},
                    {'$set': {'status': 'success'}}
                )
                return out
    except Exception as e:
        tb = traceback.format_exc()
        collection['_jobs'].update_one(
            {'identifier': identifier},
            {'$set': {'status': 'failed', 'msg': tb}}
        )
        raise e


def handle_function_output(function, collection, identifier, *args, **kwargs):
    with contextlib.redirect_stdout(Logger(collection, identifier)):
        with contextlib.redirect_stderr(Logger(collection, identifier, stream='stderr')):
            return function(*args, **kwargs)


def _function_job(database_name, collection_name, function_name, identifier,
                  *args, **kwargs):
    collection = the_client[database_name][collection_name]
    function = getattr(collection, function_name)
    collection['_jobs'].update_one({'identifier': identifier},
                                   {'$set': {'status': 'running'}})
    try:
        handle_function_output(
            function,
            collection,
            identifier,
            *args,
            **kwargs,
        )
    except Exception as e:
        tb = traceback.format_exc()
        collection['_jobs'].update_one(
            {'identifier': identifier},
            {'$set': {'status': 'failed', 'msg': tb}}
        )
        raise e

