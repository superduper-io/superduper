import contextlib


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


