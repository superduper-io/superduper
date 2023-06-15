from superduperdb.core.training_configuration import TrainingConfiguration
from superduperdb.datalayer.base.build import build_datalayer
from superduperdb.datalayer.base.query import Select
from superduperdb.misc import progress
import numpy


class SklearnTrainingConfiguration(TrainingConfiguration):
    def __init__(self, identifier, fit_params=None, predict_params=None):
        super().__init__(
            identifier, fit_params=fit_params or {}, predict_params=predict_params or {}
        )

    @classmethod
    def _get_data(cls, database, X, select: Select, y=None, y_preprocess=None):
        documents = []
        for r in progress.progressbar(database._select(select)):
            documents.append(r.unpack())
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

    def __call__(
        self,
        identifier,
        models,
        keys,
        model_names,
        select: Select,
        splitter=None,
        validation_sets=(),
        metrics=None,
        features=None,
        download=False,
    ):
        database = build_datalayer()
        X_train, y_train = self._get_data(
            database=database,
            X=keys[0],
            select=select.add_fold('train'),
            y=keys[1] if keys[1:] else None,
            y_preprocess=models[1].predict_one,
        )
        X_valid, y_valid = self._get_data(
            database=database,
            X=keys[0],
            select=select.add_fold('valid'),
            y=keys[1] if keys[1:] else None,
            y_preprocess=models[1].predict_one,
        )
        return SklearnTrainer(
            models=models,
            model_names=model_names,
            X_train=X_train,
            X_valid=X_valid,
            save=database._replace_model,
            y_train=y_train,
            y_valid=y_valid,
            fit_params=self.fit_params,
            predict_params=self.predict_params,
        )


class SklearnTrainer:
    def __init__(
        self,
        models,
        model_names,
        X_train,
        X_valid,
        save,
        y_train=None,
        y_valid=None,
        fit_params=None,
        predict_params=None,
    ):
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
