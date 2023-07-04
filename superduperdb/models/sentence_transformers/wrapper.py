import typing as t

from sentence_transformers import SentenceTransformer as _SentenceTransformer

from superduperdb.core.model import Model
from superduperdb.encoders.numpy.array import array


class SentenceTransformer(Model):
    def __init__(
        self,
        identifier: str,
        shape: int,
    ):
        sentence_transformer = _SentenceTransformer(identifier)  # type: ignore
        super().__init__(
            sentence_transformer,
            identifier,
            encoder=array('float32', shape=(shape,)),
        )

    def _predict_one(self, X, **kwargs):
        return self.object.encode(X, **kwargs)

    def _predict(self, X: t.Union[str, t.List[str]], **kwargs):
        if isinstance(X, str):
            return self._predict_one(X, **kwargs)
        return self.object.encode(X, **kwargs)
