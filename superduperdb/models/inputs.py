

class QueryInput:
    def __init__(self, *query_args, **query_kwargs):
        self.args = query_args
        self.kwargs = query_kwargs

    def execute(self, database):
        return database.execute_query(*self.args)