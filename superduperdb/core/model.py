import typing as t
from contextlib import contextmanager

from superduperdb.core.base import Component, Placeholder
from superduperdb.core.encoder import Encoder
from superduperdb.datalayer.base.query import Select
from superduperdb.training.query_dataset import QueryDataset

EncoderArg = t.Union[Encoder, Placeholder, None, str]


class TrainingConfiguration(Component):
    """
    Training configuration object, containing all settings necessary for a particular
    learning-task use-case to be serialized and initiated. The object is ``callable``
    and returns a class which may be invoked to apply training.

    :param identifier: Unique identifier of configuration
    :param **parameters: Key-values pairs, the variables which configure training.
    """

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
            with model.saving():
                database._replace_model(mn, model)

    @classmethod
    def _get_data(cls, select, keys, features, transform):
        train_data = QueryDataset(select=select, keys=keys, fold='train', transform=transform)

        valid_data = QueryDataset(select=select, keys=keys, fold='valid', transform=transform)

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


class Model(Component):
    """
    Model component which wraps a model to become serializable

    :param object: Model object, e.g. sklearn model, etc..
    :param identifier: Unique identifying ID
    :param encoder: Encoder instance (optional)
    """

    variety: str = 'model'
    object: t.Any
    identifier: str
    encoder: EncoderArg

    def __init__(
        self,
        object: t.Any,
        identifier: str,
        encoder: EncoderArg = None,
        training_configuration: t.Optional[TrainingConfiguration] = None,
        training_select: t.Optional[Select] = None,
        training_keys: t.Optional[t.Dict] = None
    ):
        super().__init__(identifier)
        self.object = object

        if isinstance(encoder, str):
            self.type = Placeholder(encoder, 'type')
        else:
            self.type = encoder

        try:
            self.predict_one = object.predict_one
        except AttributeError:
            pass

        if not hasattr(self, 'predict'):
            try:
                self.predict = object.predict
            except AttributeError:
                pass
                self.predict = self._predict

        self.training_configuration = training_configuration
        self.training_select = training_select
        self.training_keys = training_keys

    @contextmanager
    def saving(self):
        yield

    @property
    def encoder(self) -> EncoderArg:
        return self.type

    def _predict(self, inputs, **kwargs):
        return [self.predict_one(x, **kwargs) for x in inputs]

    def asdict(self):
        return {
            'identifier': self.identifier,
            'type': None if self.encoder is None else self.encoder.identifier,
        }
