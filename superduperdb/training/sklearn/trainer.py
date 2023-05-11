from superduperdb.datalayer.base.imports import get_database_from_database_type
from superduperdb.misc import progress
from superduperdb.training.base.config import TrainerConfiguration
import numpy


class SklearnTrainerConfiguration(TrainerConfiguration):
    def __init__(self, fit_params=None, predict_params=None):
        super().__init__(fit_params=fit_params or {}, predict_params=predict_params or {})

    @classmethod
    def _get_data(cls, database, X, query_params, y=None, y_preprocess=None):
        if not isinstance(query_params, tuple):
            query_params = (query_params,)
        documents = []
        for r in progress.progressbar(database.execute_query(*query_params)):
            documents.append(r)
        X = [r[X] for r in documents]
        if isinstance(X[0], numpy.ndarray):
            X = numpy.stack(X)
        if y is not None:
            y = [r[y] for r in documents]
            if y_preprocess is not None:
                y = [y_preprocess(item) for item in y]
            if isinstance(y[0], numpy.ndarray):
                y = numpy.stack(y)
        return X, y

    def __call__(self,
                 identifier,
                 models,
                 keys,
                 model_names,
                 database_type,
                 database_name,
                 query_params,
                 splitter=None,
                 validation_sets=(),
                 metrics=None,
                 features=None,
                 download=False):

        database = get_database_from_database_type(database_type, database_name)
        train_query_params = database._format_fold_to_query(query_params, 'train')
        valid_query_params = database._format_fold_to_query(query_params, 'valid')
        X_train, y_train = self._get_data(database,
                                          keys[0],
                                          train_query_params,
                                          keys[1] if keys[1:] else None,
                                          y_preprocess=models[1].predict_one)
        X_valid, y_valid = self._get_data(database,
                                          keys[0], valid_query_params,
                                          y=keys[1] if keys[1:] else None,
                                          y_preprocess=models[1].predict_one)
        return SklearnTrainer(models,
                              model_names,
                              X_train, X_valid,
                              save=database._replace_model,
                              y_train=y_train, y_valid=y_valid,
                              fit_params=self.fit_params,
                              predict_params=self.predict_params)


class SklearnTrainer:
    def __init__(self, models, model_names, X_train, X_valid, save, y_train=None, y_valid=None,
                 fit_params=None, predict_params=None):
        self.models = models
        self.model_names = model_names
        self.save = save

        self.X_train = X_train
        self.X_valid = X_valid
        self.y_train = y_train
        self.y_valid = y_valid
        self.fit_params = fit_params or {}
        self.predict_params = predict_params or {}

    def __call__(self):
        if self.y_train is not None:
            self.models[0].fit(self.X_train, self.y_train, **self.fit_params)
        else:
            self.models[0].fit(self.X_train, **self.fit_params)

        score = self.models[0].score(self.X_valid, self.y_valid, **self.predict_params)
        self.save(self.model_names[0], self.models[0])
        return score
