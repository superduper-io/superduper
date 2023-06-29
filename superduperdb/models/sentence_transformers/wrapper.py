import typing as t

from sentence_transformers import SentenceTransformer as _SentenceTransformer

from superduperdb.core.encoder import Encoder
from superduperdb.core.model import Model


class SentenceTransformer(Model):
    def __init__(
        self,
        model_name_or_path: t.Optional[str] = None,
        identifier: t.Optional[str] = None,
        encoder: t.Union[Encoder, None, str] = None,
    ):
        if identifier is None:
            identifier = model_name_or_path
        sentence_transformer = _SentenceTransformer(model_name_or_path)  # type: ignore
        super().__init__(
            sentence_transformer, identifier, encoder=encoder  # type: ignore
        )

    def predict_one(self, sentence: str, **kwargs: t.Any) -> t.Dict[str, t.Any]:
        return self.object.encode(sentence)

    def predict(
        self, sentences: t.Union[str, t.List[str]], **kwargs: t.Any
    ) -> t.Dict[str, t.Any]:
        return self.object.encode(sentences)
