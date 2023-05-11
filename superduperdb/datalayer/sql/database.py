from superduperdb.datalayer.base.database import BaseDatabase
from sqlparse import parse


class PartialParser:
    def __init__(self, query):
        self.query = query
        self._type = None
        self._parsed = parse(query)

    @property
    def n_clauses(self):
        return len(self._parsed)

    @property
    def type(self):
        ...

    @property
    def type(self):
        if self._type is None:
            ...
        return self._type


class BaseSQLDatabase(BaseDatabase):

    def execute_query(self, query_string, like=None):
        parsed = parse(query_string)
        if len(parsed) > 1 and like:
            ...