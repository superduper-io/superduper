from torch.utils import data

from superduperdb.utils import get_database_from_database_type, MongoStyleDict


class QueryDataset(data.Dataset):
    """
    Dataset object wrapping a database query

    :param database_type: type of database
    :param database: database name
    :param query_params: parameters of database query
    :param fold: fold of data
    :param suppress: supply to suppress certain fields/ columns
    :param transform: function to apply to the output
    """

    def __init__(self, database_type, database, query_params, fold='train', suppress=(), transform=None):
        super().__init__()

        self._database = None
        self._database_type = database_type
        self._database_name = database

        self.transform = transform if transform else lambda x: x
        query_params = self.database._format_fold_to_query(query_params, fold)
        self._documents = list(self.database.execute_query(*query_params))
        self.suppress = suppress

    @property
    def database(self):
        if self._database is None:
            self._database = get_database_from_database_type(self._database_type, self._database_name)
        return self._database

    def __len__(self):
        return len(self._documents)

    def __getitem__(self, item):
        r = MongoStyleDict(self._documents[item])
        for k in self.suppress:
            del r[k]
        return self.transform(self._documents[item])
