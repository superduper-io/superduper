from typing import Any, Optional, Union

from superduperdb.core.base import Component
from superduperdb.core.type import Type


class Model(Component):
    variety = 'model'

    def __init__(self, object: Any, identifier: str, type: Optional[Union[Type, str]] = None):
        super().__init__(identifier)
        self.object = object
        self.type = type

    def asdict(self):
        return {
            'identifier': self.identifier,
            'type': self.type.identifier if isinstance(self.type, Type) else self.type,
        }

    def predict_one(self, r, **kwargs):
        if hasattr(self.object, 'predict_one'):
            return self.object.predict_one(r, **kwargs)
        raise NotImplementedError

    def predict(self, docs, **kwargs):
        if hasattr(self.object, 'predict'):
            return self.object.predict(docs, **kwargs)
        raise NotImplementedError