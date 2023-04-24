import numpy
from sklearn.pipeline import Pipeline as BasePipeline
from superduperdb import progress
from superduperdb.models.inputs import QueryInput
from superduperdb import cf


class Pipeline(BasePipeline):
    def __init__(self, database_or_table, identifier, steps, memory=None, verbose=False, postprocessor=None):
        super().__init__(steps=steps, memory=memory, verbose=verbose)
        self.database_or_table = database_or_table
        self.identifier = identifier
        self.postprocessor = postprocessor

    def _get_data(self, X, y, query_params):
        if not isinstance(query_params, tuple):
            query_params = (query_params,)
        documents = []
        for r in progress.progressbar(self.database_or_table.execute_query(*query_params)):
            documents.append(r)
        X = [r[X] for r in documents]
        _ids = [r[self.database_or_table._id] for r in documents]
        if isinstance(X[0], numpy.ndarray):
            X = numpy.stack(X)
        if y is not None:
            y = [r[y] for r in documents]
            if isinstance(y[0], numpy.ndarray):
                y = numpy.stack(y)
        return X, y, _ids

    def fit(self, X, y=None, query_params=None, remote=cf['remote'], **fit_params):
        if remote:
            self.database_or_table.create_function(self.identifier)
            return self.database_or_table.fit(self.identifier, X, y, **fit_params)
        return self._fit_locally(X, y, query_params=query_params, **fit_params)

    def _fit_locally(self, X, y=None, query_params=None, **fit_params):
        if query_params is not None:
            X, y, _ids = self._get_data(X, y, query_params)
        return super().fit(X, y=y, **fit_params)

    def fit_predict(self, X, y=None, remote=cf['remote'], **fit_params):
        if remote:
            fit_id = self.fit(X, y=y, remote=remote, **fit_params)
            return self.predict(X, dependencies=(fit_id,))

    def _fit_predict_locally(self, X, y=None, query_params=None, **fit_params):
        if isinstance(X, QueryInput):
            X, y, _ids = self._get_data(X, y, query_params)
        return super().fit_predict(X, y=y, **fit_params)

    def score(self, X, remote=cf['remote'], persist_key=None, **predict_params):
        if remote:
            out = self.database_or_table.score(self.identifier, X, persist_key=persist_key,
                                               **predict_params)
            return out

    def _score_locally(self, X, y=None, query_params=None, sample_weight=None):
        if query_params is not None:
            X, y, _ids = self._get_data(X, y, query_params)
        return super().score(X, y=y, sample_weight=sample_weight)

    def predict(self, X, query_params=None, remote=cf['remote'], dependencies=(),
                **predict_params):
        if remote:
            out = self.database_or_table.predict(self.identifier, X, query_params=query_params,
                                                 dependencies=dependencies,
                                                 **predict_params)
            return out
        return self._predict_locally(X, query_params=query_params, **predict_params)

    def predict_and_update(self, X, y, query_params, remote=cf['remote'], **predict_params):
        assert query_params is not None
        assert y is not None
        out = self.predict(X, query_params=query_params, remote=remote, **predict_params)
        self.database_or_table._write_watcher_outputs(
            {'model': self.identifier, 'key': y, 'query_params': query_params},
            outputs=[r['y_pred'] for r in out],
            _ids=[r['_id'] for r in out],
        )

    def _predict_locally(self, X, query_params=None, **predict_params):
        use_db = query_params is not None and isinstance(X, str)

        if use_db:
            X, _, _ids = self._get_data(X, None, query_params)
        y_pred = super().predict(X, **predict_params)
        if self.postprocessor is not None:
            y_pred = self.postprocessor(y_pred)

        if use_db:
            y_pred = list(y_pred)
            return [
                {'_id': _id, 'y_pred': y_pred[i]}
                for i, _id in enumerate(_ids)
            ]
        return y_pred

