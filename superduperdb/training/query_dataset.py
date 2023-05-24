from superduperdb.datalayer.base.imports import get_database_from_database_type
from superduperdb.datalayer.base.query import Select
from superduperdb.misc.special_dicts import MongoStyleDict


class QueryDataset:
    """
    Dataset object wrapping a database query

    :param database_type: type of database
    :param database: database name
    :param query_params: parameters of database query
    :param fold: fold of data
    :param suppress: supply to suppress certain fields/ columns
    :param transform: function to apply to the output
    """

    def __init__(
        self,
        select: Select,
        database,
        database_type=None,
        keys=None,
        fold='train',
        suppress=(),
        transform=None,
        features=None,
    ):
        super().__init__()

        if isinstance(database, str):
            self._database = None
            self._database_name = database
        else:
            self._database = database
        self._database_type = database_type
        self.keys = keys

        self.transform = transform if transform else lambda x: x
        select = select.add_fold(fold)
        self._documents = list(self.database.select(select))
        self.suppress = suppress
        self.features = features or {}

    @property
    def database(self):
        if self._database is None:
            self._database = get_database_from_database_type(
                self._database_type, self._database_name
            )
        return self._database

    def __len__(self):
        return len(self._documents)

    def __getitem__(self, item):
        r = MongoStyleDict(self._documents[item])
        s = MongoStyleDict({})
        for k in self.features:
            r[k] = r['_outputs'][k][self.features[k]]

        if self.keys is not None:
            for k in self.keys:
                if k == '_base' and k not in self.features:
                    s[k] = r
                else:
                    s[k] = r[k]
        else:
            s = r
        return self.transform(s)
