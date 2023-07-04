from superduperdb.datalayer.base.query import Select
from superduperdb.misc.special_dicts import MongoStyleDict


class QueryDataset:
    def __init__(
        self,
        select: Select,
        keys=None,
        fold='train',
        suppress=(),
        transform=None,
        features=None,
        database=None,
    ):
        super().__init__()

        self._database = database
        self.keys = keys

        self.transform = transform if transform else lambda x: x
        select = select.add_fold(fold)
        self._documents = list(self.database.execute(select))
        self.suppress = suppress
        self.features = features or {}

    @property
    def database(self):
        if self._database is None:
            from superduperdb.datalayer.base.build import build_datalayer

            self._database = build_datalayer()
        return self._database

    def __len__(self):
        return len(self._documents)

    def __getitem__(self, item):
        r = MongoStyleDict(self._documents[item].unpack())
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
