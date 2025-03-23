import typing as t

from superduper.base.query import Query
from superduper.misc.special_dicts import DeepKeyedDict

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
            assert db is not None
            fold_filter = db[select.table]['_fold'] == fold
            self.select = select.filter(fold_filter)
        else:
            self.select = select

        self.in_memory = in_memory
        if self.in_memory:
            if ids is None:
                self._documents = self.select.execute()
            else:
                self._documents = self.select.subset(ids)
        else:
            if ids is None:
                self._ids = self.select.ids()
            else:
                self._ids = ids

            # TODO replace by adding parameters to `.get`
            assert db is not None
            t = db[self.select.table]
            self.select_one = lambda x: next(
                self.select.filter(t[t.primary_id] == x).execute()
            )

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
        input = DeepKeyedDict(input.unpack())
        return self._get_item(input)
