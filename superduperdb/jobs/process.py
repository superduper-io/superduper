import contextlib
import traceback


class Logger:
    def __init__(self, database, id_, stream='stdout'):
        self.database = database
        self.id_ = id_
        self.stream = stream

    def write(self, message):
        if self.stream == 'stdout':
            self.database['_jobs'].update_one(
                {'identifier': self.id_},
                {'$push': {'stdout': message}}
            )
        elif self.stream == 'stderr':
            self.database['_jobs'].update_one(
                {'identifier': self.id_},
                {'$push': {'stderr': message}}
            )
        else:
            raise NotImplementedError

    def flush(self):
        pass


def handle_function_output(function, database, identifier, *args, **kwargs):
    with contextlib.redirect_stdout(Logger(database, identifier)):
        with contextlib.redirect_stderr(Logger(database, identifier, stream='stderr')):
            return function(*args, **kwargs)


def _function_job(database_name, function_name, identifier,
                  args_, kwargs_):
    from superduperdb.client import the_client
    database = the_client[database_name]
    function = getattr(database, function_name)
    database['_jobs'].update_one({'identifier': identifier},
                                 {'$set': {'status': 'running'}})
    try:
        handle_function_output(
            function,
            database,
            identifier,
            *args_,
            **kwargs_,
        )
    except Exception as e:
        tb = traceback.format_exc()
        database['_jobs'].update_one(
            {'identifier': identifier},
            {'$set': {'status': 'failed', 'msg': tb}}
        )
        raise e
    database['_jobs'].update_one(
        {'identifier': identifier},
        {'$set': {'status': 'success'}}
    )

