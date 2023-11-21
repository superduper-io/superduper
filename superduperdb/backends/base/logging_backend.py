import typing as t
from abc import ABC


class BaseLoggingBackend(ABC):
    def __init__(self, database: t.Any, session_id: str, stream='stdout'):
        self.database = database
        self.session_id = session_id
        self.stream = stream

    def write(self, message):
        print(message)

    #        self.database.metadata.write_output_to_job(
    #            self.id_, message, stream=self.stream
    #        )

    def flush(self):
        pass
