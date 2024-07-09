import inspect
import random
import typing as t

from superduper.backends.base.query import Query
from superduper.misc.special_dicts import MongoStyleDict

if t.TYPE_CHECKING:
    from superduper.base.datalayer import Datalayer
    from superduper.components.model import Mapping


class ExpiryCache(list):
    """Expiry Cache for storing documents.

    The document will be removed from the cache after fetching it from the cache.

    :param args: *args for `list`
    :param kwargs: **kwargs for `list`
    """

    def __getitem__(self, index):
        item = super().__getitem__(index)
        del self[index]
        return item


class QueryDataset:
    """Query Dataset for fetching documents from database.

    :param select: A select query object which defines the query to be executed.
    :param mapping: A mapping object to be used for the dataset.
    :param ids: A list of ids to be used for the dataset.
    :param fold: The fold to be used for the dataset.
    :param transform: A callable which can be used to transform the dataset.
    :param db: A datalayer instance to be used for the dataset.
    :param in_memory: A boolean flag to indicate if the dataset should be loaded
                      in memory.
    """

    def __init__(
        self,
        select: Query,
        mapping: t.Optional['Mapping'] = None,
        ids: t.Optional[t.List[str]] = None,
        fold: t.Union[str, None] = 'train',
        transform: t.Optional[t.Callable] = None,
        db: t.Optional['Datalayer'] = None,
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
        """Return the datalayer instance."""
        if self._db is None:
            from superduper.base.build import build_datalayer

            self._db = build_datalayer()
        return self._db

    def __len__(self):
        if self.in_memory:
            return len(self._documents)
        else:
            return len(self._ids)

    def _get_item(self, input):
        out = input
        if self.mapping is not None:
            out = self.mapping(out)
        if self.transform is not None and self.mapping is not None:
            if self.mapping.signature == '*args,**kwargs':
                out = self.transform(*out[0], **out[1])
            elif self.mapping.signature == '*args':
                out = self.transform(*out)
            elif self.mapping.signature == '**kwargs':
                out = self.transform(**out)
            elif self.mapping.signature == 'singleton':
                out = self.transform(out)
        elif self.transform is not None:
            out = self.transform(out)
        return out

    def __getitem__(self, item):
        if self.in_memory:
            input = self._documents[item]
        else:
            input = self.select_one(
                self._ids[item], self.db, encoders=self.db.datatypes
            )
        input = MongoStyleDict(input.unpack())
        return self._get_item(input)


class CachedQueryDataset(QueryDataset):
    """Cached Query Dataset for fetching documents from database.

    This class which fetch the document corresponding to the given ``index``.
    This class prefetches documents from database and stores in the memory.

    This can drastically reduce database read operations and hence reduce the overall
    load on the database.

    :param select: A select query object which defines the query to be executed.
    :param mapping: A mapping object to be used for the dataset.
    :param ids: A list of ids to be used for the dataset.
    :param fold: The fold to be used for the dataset.
    :param transform: A callable which can be used to transform the dataset.
    :param db: A datalayer instance to be used for the dataset.
    :param in_memory: A boolean flag to indicate if the dataset should be loaded
    :param prefetch_size: The number of documents to prefetch from the database.
    """

    _BACKFILL_INDEX = 0.2

    def __init__(
        self,
        select: Query,
        mapping: t.Optional['Mapping'] = None,
        ids: t.Optional[t.List[str]] = None,
        fold: t.Union[str, None] = 'train',
        transform: t.Optional[t.Callable] = None,
        db=None,
        in_memory: bool = True,
        prefetch_size: int = 100,
    ):
        super().__init__(
            select=select,
            mapping=mapping,
            ids=ids,
            fold=fold,
            transform=transform,
            db=db,
            in_memory=in_memory,
        )
        self.prefetch_size = prefetch_size

    def _fetch_cache(self):
        cache_ids = random.sample(self.ids, self._max_cache_size)
        return ExpiryCache(
            list(self.database.execute(self.select.select_using_ids(cache_ids)))
        )

    @property
    def database(self):
        """Return the database object."""
        if self._database is None:
            from superduper.base.build import build_datalayer

            self._database = build_datalayer()
        return self._database

    def _get_random_index(self, index):
        return int((index * len(self._cache)) / self._total_documents)

    def _backfill_cache(self):
        if len(self._cache) <= int(self._max_cache_size * self._BACKFILL_INDEX):
            self._cache = self._fetch_cache()

    def __len__(self):
        return self._total_documents

    def __getitem__(self, index):
        index = self._get_random_index(index)
        self._backfill_cache()
        input = self._cache[index]
        return self._get_item(input)


def query_dataset_factory(**kwargs):
    """Create a query dataset object.

    If ``data_prefetch`` is set to ``True``, then a ``CachedQueryDataset`` object is
    created, otherwise a ``QueryDataset`` object is created.

    :param kwargs: Keyword arguments to be passed to the query dataset object.
    """
    if kwargs.get('data_prefetch', False):
        return CachedQueryDataset(**kwargs)
    kwargs = {
        k: v
        for k, v in kwargs.items()
        if k in inspect.signature(QueryDataset.__init__).parameters
    }
    return QueryDataset(**kwargs)
