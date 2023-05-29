from superduperdb.core.base import Component


class Type(Component):
    variety = 'type'

    def __init__(self, identifier: str):
        super().__init__(identifier)

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def decode(self, bytes):
        raise NotImplementedError

    def encode(self, item):
        raise NotImplementedError