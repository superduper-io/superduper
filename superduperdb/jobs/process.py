import contextlib
import traceback

from superduperdb.utils import get_database_from_database_type


class Logger:
    def __init__(self, database, id_, stream='stdout'):
        self.database = database
        self.id_ = id_
        self.stream = stream

    def write(self, message):
        self.database.write_output_to_job(self.id_, message, stream=self.stream)

    def flush(self):
        pass


def handle_function_output(function, database, identifier, *args, **kwargs):
    with contextlib.redirect_stdout(Logger(database, identifier)):
        with contextlib.redirect_stderr(Logger(database, identifier, stream='stderr')):
            return function(*args, **kwargs)


def _function_job(database_type, database_name, function_name, identifier,
                  args_, kwargs_):
    database = get_database_from_database_type(database_type, database_name)
    database.remote = False
    function = getattr(database, function_name)
    database.set_job_flag(identifier, ('status', 'running'))
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
        database.set_job_flag(identifier, ('status', 'failed'))
        database.set_job_flag(identifier, ('msg', tb))
        raise e
    database.set_job_flag(identifier, ('status', 'success'))
