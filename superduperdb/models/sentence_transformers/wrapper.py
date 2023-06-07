from typing import Optional, Union

from sentence_transformers import SentenceTransformer as _SentenceTransformer

from superduperdb.core.model import Model
from superduperdb.core.type import Type


class SentenceTransformer(Model):
    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        identifier: Optional[str] = None,
        type: Optional[Union[str, Type]] = None,
    ):
        if identifier is None:
            identifier = model_name_or_path
        sentence_transformer = _SentenceTransformer(model_name_or_path)
        super().__init__(sentence_transformer, identifier, type=type)

    def predict_one(self, sentence, **kwargs):
        return self.object.encode(sentence)

    def predict(self, sentences, **kwargs):
        return self.object.encode(sentences)
