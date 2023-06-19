import typing as t

from superduperdb.core.base import Component, Placeholder
from superduperdb.core.encoder import Encoder

EncoderArg = t.Union[Encoder, Placeholder, None, str]


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
    type: EncoderArg

    def __init__(
        self,
        object: t.Any,
        identifier: str,
        encoder: EncoderArg = None,
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
