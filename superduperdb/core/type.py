from superduperdb.core.base import Component


class Type(Component):
    """
    Type component, responsible for encoding and decoding model inputs and outputs
    to and from the datalayer as blobs of bytes.

    :param identifier: Unique identifier
    """

    variety = 'type'

    def __init__(self, identifier: str):
        super().__init__(identifier)

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def decode(self, bytes):
        raise NotImplementedError

    def encode(self, item):
        raise NotImplementedError
