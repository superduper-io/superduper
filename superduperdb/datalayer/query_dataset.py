import random

from torch.utils.data import Dataset

from superduperdb.datalayer.base.query import Select
from superduperdb.misc.special_dicts import MongoStyleDict


class ExpiryCache(list):
    def __getitem__(self, index):
        item = super().__getitem__(index)
        del self[index]
        return item


class QueryDataset(Dataset):
    def __init__(
        self,
        select: Select,
        keys=None,
        fold='train',
        suppress=(),
        transform=None,
        features=None,
        database=None,
        **kwargs,
    ):
        super().__init__()

        self._database = database
        self.keys = keys

        self.transform = transform if transform else lambda x: x
        self._documents = list(self.database.execute(select))
        self.select = select.add_fold(fold)
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


class CachedQueryDataset(Dataset):
    """
    A cached dataset class which can be used to define a torch dataset class.
    This class which fetch the document corresponding to the given ``index``.
    This class prefetches documents from database and stores in the memory.

    This can drastically reduce database read operations and hence reduce the overall
    load on the database.
    """

    _BACKFILL_INDEX = 0.2

    def __init__(
        self,
        select: Select,
        keys=None,
        fold='train',
        suppress=(),
        transform=None,
        features=None,
        database=None,
        prefetch_size: int = 100,
    ):
        super().__init__()

        self._database = database
        self.keys = keys

        self.transform = transform if transform else lambda x: x
        self.select = select.add_fold(fold)
        self.ids = [doc.id for doc in self.database.execute(self.select.select_ids)]
        self.suppress = suppress
        self.features = features or {}
        self._max_cache_size = prefetch_size
        self._cache: ExpiryCache = self._fetch_cache()
        self._total_documents = self.count_documents()

    def count_documents(
        self,
    ):
        return self.database.execute(self.select).count()

    def _fetch_cache(self):
        cache_ids = random.sample(self.ids, self._max_cache_size)
        return ExpiryCache(
            list(self.database.execute(self.select.select_using_ids(cache_ids)))
        )

    @property
    def database(self):
        if self._database is None:
            from superduperdb.datalayer.base.build import build_datalayer

            self._database = build_datalayer()
        return self._database

    def _unpack(self, documents):
        batch = []
        for document in documents:
            r = MongoStyleDict(document.unpack())
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
            s = self.transform(s)
            batch.append(s)
        return batch

    def _get_random_index(self, index):
        return int((index * len(self._cache)) / self._total_documents)

    def _backfill_cache(self):
        if len(self._cache) <= int(self._max_cache_size * self._BACKFILL_INDEX):
            self._cache = self._fetch_cache()

    def __len__(self):
        return self._total_documents

    def __getitem__(self, index):
        index = self._get_random_index(index)
        document = self._cache[index]
        self._backfill_cache()
        r = MongoStyleDict(document.unpack())
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


def query_dataset_factory(data_prefetch: bool = False, **kwargs):
    if data_prefetch:
        return CachedQueryDataset(**kwargs)
    return QueryDataset(**kwargs)
