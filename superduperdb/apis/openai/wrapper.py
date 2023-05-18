import os

from openai import ChatCompletion as _ChatCompletion
from openai import Embedding as _Embedding
from openai import Model
from openai.error import Timeout, RateLimitError, TryAgain, ServiceUnavailableError

from superduperdb.apis.utils import DoRetry
from superduperdb.models.base import SuperDuperModel


do_retry = DoRetry((Timeout, RateLimitError, TryAgain, ServiceUnavailableError))

AVAILABLE_MODELS = [r['id'] for r in Model.list()['data']]


class BaseOpenAI(SuperDuperModel):
    def __init__(self, model_id):
        self.model_id = model_id
        assert model_id in AVAILABLE_MODELS, "model not in list of OpenAI available models"
        assert 'OPENAI_API_KEY' in os.environ, "OPENAI_API_KEY not set"


class Embedding(BaseOpenAI):

    @do_retry
    def predict_one(self, text, **kwargs):
        return _Embedding.create(input=text, model=self.model_id, **kwargs)['data'][0]['embedding']

    @do_retry
    def _predict_a_batch(self, texts, **kwargs):
        out = _Embedding.create(input=texts, model=self.model_id, **kwargs)['data']
        return [r['embedding'] for r in out]

    def predict(self, texts, batch_size=100, **kwargs):  # asyncio?
        out = []
        for i in range(0, len(texts), batch_size):
            out.extend(self._predict_a_batch(texts[i: i + batch_size], **kwargs))
        return out


class ChatCompletion(BaseOpenAI):

    @do_retry
    def predict_one(self, message, **kwargs):
        return _ChatCompletion.create(
            messages=[{'role': 'user', 'content': message}],
            model=self.model_id,
            **kwargs
        )['choices'][0]['message']['content']

    def predict(self, messages, **kwargs):
        return [self.predict_one(msg) for msg in messages]  # use asyncio

