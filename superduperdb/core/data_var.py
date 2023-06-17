import io
import pickle
import typing as t


class DataVar:
    """
    Data variable wrapping encode-able item. Encoding is controlled by the referred
    to ``Type`` instance.

    :param x: Wrapped content
    :param type: Identifier of type component used to encode
    :param encoder: Encoder used to dump to `bytes`
    """

    def __init__(
        self,
        x: t.Any,
        type: str,
        encoder: t.Optional[t.Callable] = None,
        shape: t.Optional[t.Tuple] = None,
    ):
        if shape is not None:
            assert hasattr(x, 'shape')
            assert tuple(x.shape) == shape
        self.x = x
        self._encoder = encoder
        self.type = type
        self.shape = shape

    def __repr__(self):
        if self.shape is not None:
            return f'DataVar[{self.type}: {tuple(self.shape)}]({self.x.__repr__()})'
        else:
            return f'DataVar[{self.type}]({self.x.__repr__()})'

    def encode(self):
        if self._encoder is None:
            f = io.BytesIO()
            pickle.dump(self.x, f)
            return f.getvalue()
        return {'_content': {'bytes': self._encoder(self.x), 'type': self.type}}
