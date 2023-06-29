import typing as t

from superduperdb.core.metric import Metric
from superduperdb.core.base import Component, Placeholder
from superduperdb.core.encoder import Encoder
from superduperdb.datalayer.base.query import Select

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

    def __init__(self, identifier: str, **parameters: t.Dict[str, t.Any]) -> None:
        super().__init__(identifier)
        for k, v in parameters.items():
            setattr(self, k, v)

    def get(self, k: str, default: t.Any = None) -> t.Any:
        return getattr(self, k, default)


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
        training_keys: t.Optional[t.List[str]] = None,
        metrics: t.Optional[t.List[Metric]] = None,
    ):
        super().__init__(identifier)
        self.object = object

        if isinstance(encoder, str):
            self.encoder: EncoderArg = Placeholder(encoder, 'type')
        else:
            self.encoder: EncoderArg = encoder  # type: ignore

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
        self.metrics = metrics
        self.metric_values: t.Dict = {}

    def _predict(self, inputs: t.Any, **kwargs: t.Dict[str, t.Any]) -> t.List:
        return [self.predict_one(x, **kwargs) for x in inputs]

    def asdict(self) -> t.Dict[str, t.Any]:
        return {
            'identifier': self.identifier,
            'type': None if self.encoder is None else self.encoder.identifier,
        }

    def append_metrics(self, d: t.Dict) -> None:
        for k in d:
            if k not in self.metric_values:
                self.metric_values[k] = []
            self.metric_values[k].append(d[k])


class ModelEnsemblePredictionError(Exception):
    pass


# TODO make Component less dogmatic about having just one ``self.object`` type thing
class ModelEnsemble:
    variety: str = 'model'

    def __init__(self, models: t.List[t.Union[Model, str]]):
        self._model_ids = []
        for m in models:
            if isinstance(m, Model):
                setattr(self, m.identifier, m)
                self._model_ids.append(m.identifier)
            elif isinstance(m, str):
                setattr(self, m, Placeholder('model', m))
                self._model_ids.append(m)

    def __getitem__(self, submodel: t.Union[int, str]) -> Model:
        if isinstance(submodel, int):
            submodel = next(m for i, m in enumerate(self._model_ids) if i == submodel)
        submodel = getattr(self, submodel)
        assert isinstance(submodel, Model)
        return submodel
