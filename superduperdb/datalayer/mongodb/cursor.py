import typing as t

from pymongo.collection import Collection
from pymongo.cursor import Cursor

from superduperdb.misc.special_dicts import MongoStyleDict
from superduperdb.core.documents import Document


class SuperDuperCursor(Cursor):
    def __init__(
        self,
        collection: Collection,
        *args: t.Any,
        features: t.Optional[t.Dict] = None,
        scores: t.Optional[t.Dict] = None,
        similar_join: t.Any = None,
        **kwargs: t.Any,
    ) -> None:
        """
        Cursor subclassing *pymongo.cursor.Cursor*.
        If *features* are specified, these are substituted in the records
        for the raw data. This is useful, for instance if images are present, and they
        should be featurized by a certain model. If *scores* are added, these are added
        to the results records.

        :param collection: collection
        :param *args: args to pass to super()
        :param features: dictionary of features to set (replace record entries with
                         model outputs)
        :param scores: similarity scores to add to records
        :param similar_join: replace ids by documents in subfield of _like
        :param **kwargs: kwargs to pass to super() (see pymongo.cursor.Cursor)
        """
        super().__init__(collection, *args, **kwargs)
        self.attr_collection = collection
        self.features = features
        self.scores = scores
        self.similar_join = similar_join
        self._args = args
        self._kwargs = kwargs
        if self.scores is not None:
            self._results = []
            while True:
                try:
                    self._results.append(super().next())
                except StopIteration:
                    break
            self._results = sorted(self._results, key=lambda r: -self.scores[r['_id']])
            self.it = 0

    def limit(self, limit: int) -> Cursor:
        if self.scores is None:
            return super().limit(limit)
        self._results = self._results[:limit]
        return self

    def __getitem__(self, item: int) -> t.Union[Cursor, MongoStyleDict]:
        r = super().__getitem__(item)
        if self.features is not None and self.features:
            r = self._add_features(r)
        return r

    def _add_features(self, r: t.Dict) -> MongoStyleDict:
        r = MongoStyleDict(r)
        for k in self.features:
            r[k] = r['_outputs'][k][self.features[k]]
        if '_other' in r:
            for k in self.features:
                if k in r['_other']:
                    r['_other'][k] = r['_outputs'][k][self.features[k]]
        return r

    def next(self) -> Document:
        if self.scores is not None:
            try:
                r = self._results[self.it]
            except IndexError:
                raise StopIteration
            self.it += 1
        else:
            r = super().next()
        if self.scores is not None:
            r['_score'] = self.scores[r['_id']]

        if self.features is not None and self.features:
            r = self._add_features(r)

        if self.similar_join is not None:
            if self.similar_join in r.get('_like', {}):
                ids = r['_like'][self.similar_join]
                lookup = {
                    r['_id']: r
                    for r in self.collection.find(
                        {'_id': {'$in': ids}}, *self._args[1:], **self._kwargs
                    )
                }
                for i, id_ in enumerate(r['_like'][self.similar_join]):
                    r['_like'][self.similar_join][i] = lookup[id_]

        return Document(Document.decode(r, self.collection.database.types))

    __next__ = next
