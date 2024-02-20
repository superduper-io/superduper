import inspect
import random
import typing as t

from superduperdb.backends.base.query import Select
from superduperdb.misc.special_dicts import MongoStyleDict

if t.TYPE_CHECKING:
    from superduperdb.components.model import Mapping


class ExpiryCache(list):
    def __getitem__(self, index):
        item = super().__getitem__(index)
        del self[index]
        return item


class QueryDataset:
    """
    A dataset class which can be used to define a torch dataset class.

    :param select: A select query object which defines the query to be executed.
    :param keys: A list of keys to be returned from the dataset.
    :param fold: The fold to be used for the dataset.
    :param suppress: A list of keys to be suppressed from the dataset.
    :param transform: A callable which can be used to transform the dataset.
    :param db: A ``DB`` object to be used for the dataset.
    :param ids: A list of ids to be used for the dataset.
    :param in_memory: A boolean flag to indicate if the dataset should be loaded
                      in memory.
    :param extract: A key to be extracted from the dataset.
    """

    def __init__(
        self,
        select: Select,
        mapping: t.Optional['Mapping'] = None,
        ids: t.Optional[t.List[str]] = None,
        fold: t.Union[str, None] = 'train',
        transform: t.Optional[t.Callable] = None,
        db=None,
        in_memory: bool = True,
    ):
        self._db = db

        self.transform = transform
        if fold is not None:
            self.select = select.add_fold(fold)
        else:
            self.select = select

        self.in_memory = in_memory
        if self.in_memory:
            if ids is None:
                self._documents = list(self.db.execute(self.select))
            else:
                self._documents = list(
                    self.db.execute(self.select.select_using_ids(ids))
                )
        else:
            if ids is None:
                self._ids = [
                    r[self.select.id_field]
                    for r in self.db.execute(self.select.select_ids)
                ]
            else:
                self._ids = ids
            self.select_one = self.select.select_single_id

        self.mapping = mapping

    @property
    def db(self):
        if self._db is None:
            from superduperdb.base.build import build_datalayer

            self._db = build_datalayer()
        return self._db

    def __len__(self):
        if self.in_memory:
            return len(self._documents)
        else:
            return len(self._ids)

    def __getitem__(self, item):
        if self.in_memory:
            input = self._documents[item]
        else:
            input = self.select_one(
                self._ids[item], self.db, encoders=self.db.datatypes
            )
        input = MongoStyleDict(input.unpack(db=self.db))
        from superduperdb.components.model import Signature

        out = input
        if self.mapping is not None:
            out = self.mapping(out)
        if self.transform is not None and self.mapping is not None:
            if self.mapping.signature == Signature.args_kwargs:
                out = self.transform(*out[0], **out[1])
            elif self.mapping.signature == Signature.args:
                out = self.transform(*out)
            elif self.mapping.signature == Signature.kwargs:
                out = self.transform(**out)
            elif self.mapping.signature == Signature.singleton:
                out = self.transform(out)
        elif self.transform is not None:
            out = self.transform(out)
        return out


class CachedQueryDataset:
    """
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
        database=None,
        prefetch_size: int = 100,
    ):
        self._database = database
        self.keys = keys

        self.transform = transform if transform else lambda x: x
        self.select = select.add_fold(fold)

        self.ids = [doc.id for doc in self.database.execute(self.select.select_ids)]
        self.suppress = suppress
        self._max_cache_size = prefetch_size
        self._cache: ExpiryCache = self._fetch_cache()
        self._total_documents = self.count_documents()

    def count_documents(self) -> int:
        """Return the number of matching documents"""
        return self.database.execute(self.select).count()

    def _fetch_cache(self):
        cache_ids = random.sample(self.ids, self._max_cache_size)
        return ExpiryCache(
            list(self.database.execute(self.select.select_using_ids(cache_ids)))
        )

    @property
    def database(self):
        if self._database is None:
            from superduperdb.base.build import build_datalayer

            self._database = build_datalayer()
        return self._database

    def _unpack(self, documents):
        batch = []
        for document in documents:
            r = MongoStyleDict(document.unpack())
            s = MongoStyleDict({})

            if self.keys is not None:
                for k in self.keys:
                    if k == '_base':
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

        if self.keys is not None:
            for k in self.keys:
                if k == '_base':
                    s[k] = r
                else:
                    s[k] = r[k]
        else:
            s = r
        return self.transform(s)


def query_dataset_factory(**kwargs):
    if kwargs.get('data_prefetch', False):
        return CachedQueryDataset(**kwargs)
    kwargs = {
        k: v
        for k, v in kwargs.items()
        if k in inspect.signature(QueryDataset.__init__).parameters
    }
    return QueryDataset(**kwargs)
