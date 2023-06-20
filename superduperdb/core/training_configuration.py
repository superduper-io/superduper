import typing as t

from superduperdb.core.base import Component
from superduperdb.training.query_dataset import QueryDataset


class TrainingConfiguration(Component):
    """
    Training configuration object, containing all settings necessary for a particular
    learning-task use-case to be serialized and initiated. The object is ``callable``
    and returns a class which may be invoked to apply training.

    :param identifier: Unique identifier of configuration
    :param **parameters: Key-values pairs, the variables which configure training.
    """

    compute_metrics: t.Callable[..., t.Any]
    loader_kwargs: t.Dict[str, t.Any]
    max_iterations: float
    no_improve_then_stop: int
    objective: t.Any
    optimizer_classes: t.Dict
    optimizer_kwargs: t.Dict
    validation_interval: int
    watch: str

    variety = 'training_configuration'

    def __init__(self, identifier, **parameters):
        super().__init__(identifier)
        for k, v in parameters.items():
            setattr(self, k, v)

    @classmethod
    def split_and_preprocess(cls, r, models):
        raise NotImplementedError

    @classmethod
    def save_models(cls, database, models, model_names):
        for model, mn in zip(models, model_names):
            database._replace_model(model, mn)

    @classmethod
    def _get_data(cls, select, keys, features, transform):
        train_data = QueryDataset(
            select=select,
            keys=keys,
            fold='train',
            transform=transform,
            features=features,
        )

        valid_data = QueryDataset(
            select=select,
            keys=keys,
            fold='valid',
            transform=transform,
            features=features,
        )

        return train_data, valid_data

    def get(self, k, default=None):
        return getattr(self, k, default)

    def __call__(
        self,
        identifier,
        models,
        keys,
        model_names,
        select,
        validation_sets=(),
        metrics=None,
        features=None,
        download=False,
    ):
        raise NotImplementedError
