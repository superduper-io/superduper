from pymongo.cursor import Cursor

from superduperdb.types.utils import convert_types
from superduperdb.utils import MongoStyleDict


class SuperDuperCursor(Cursor):
    def __init__(self, collection, *args, features=None, scores=None,
                 similar_join=None, **kwargs):
        """
        Cursor subclassing *pymongo.cursor.Cursor*.
        If *features* are specified, these are substituted in the records
        for the raw data. This is useful, for instance if images are present, and they should
        be featurized by a certain model. If *scores* are added, these are added to the results
        records.

        :param collection: collection
        :param *args: args to pass to super()
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

    def next(self):
        r = super().next()
        if self.scores is not None:
            r['_score'] = self.scores[r['_id']]
        if self.features is not None and self.features:
            r = MongoStyleDict(r)
            for k in self.features:
                r[k] =  r['_outputs'][k][self.features[k]]
            if '_other' in r:
                for k in self.features:
                    r['_other'][k] = r['_outputs'][k][self.features[k]]

        if self.similar_join is not None:
            if self.similar_join in r.get('_like', {}):
                ids = r['_like'][self.similar_join]
                lookup = {r['_id']: r for r in self.collection.find({'_id': {'$in': ids}},
                                                                    *self._args[1:],
                                                                    **self._kwargs)}
                for i, id_ in enumerate(r['_like'][self.similar_join]):
                    r['_like'][self.similar_join][i] = lookup[id_]

        return convert_types(r, converters=self.attr_collection.converters)

    __next__ = next