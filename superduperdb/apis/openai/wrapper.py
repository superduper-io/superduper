import os

import tqdm

import superduperdb as s

from openai import ChatCompletion as _ChatCompletion
from openai import Embedding as _Embedding
from openai import Model as OpenAIModel
from openai.error import Timeout, RateLimitError, TryAgain, ServiceUnavailableError

from superduperdb.apis.retry import Retry
from superduperdb.core.model import Model
from superduperdb.misc.compat import cache

import typing as t

from superduperdb.types.vectors.vector import vector

retry = Retry((RateLimitError, ServiceUnavailableError, Timeout, TryAgain))


def init_fn():
    s.log.info('Setting OpenAI api-key...')
    os.environ['OPENAI_API_KEY'] = s.CFG.apis.providers['openai'].api_key


@cache
def _available_models():
    return tuple([r['id'] for r in OpenAIModel.list()['data']])


class BaseOpenAI(Model):
    def __init__(self, identifier: str):
        super().__init__(None, identifier)
        msg = "model not in list of OpenAI available models"
        assert identifier in _available_models(), msg
        assert 'OPENAI_API_KEY' in os.environ, "OPENAI_API_KEY not set"


class Embedding(BaseOpenAI):
    shapes = {'text-embedding-ada-002': (1536,)}

    def __init__(self, identifier: str, shape: t.Optional[int] = None):
        super().__init__(identifier)
        if shape is None:
            shape = self.shapes[identifier]
        self.encoder = vector(shape)

    @retry
    def _predict_one(self, X, **kwargs):
        e = _Embedding.create(input=X, model=self.identifier, **kwargs)
        return e['data'][0]['embedding']

    @retry
    def _predict_a_batch(self, texts, **kwargs):
        out = _Embedding.create(input=texts, model=self.identifier, **kwargs)['data']
        return [r['embedding'] for r in out]

    def _predict(self, X, batch_size=100, **kwargs):  # asyncio?
        if isinstance(X, str):
            return self._predict_one(X)
        out = []
        for i in tqdm.tqdm(range(0, len(X), batch_size)):
            out.extend(self._predict_a_batch(X[i : i + batch_size], **kwargs))
        return out


class ChatCompletion(BaseOpenAI):
    @retry
    def predict_one(self, message, **kwargs):
        return _ChatCompletion.create(
            messages=[{'role': 'user', 'content': message}],
            model=self.identifier,
            **kwargs,
        )['choices'][0]['message']['content']

    def predict(self, messages, **kwargs):
        return [self.predict_one(msg) for msg in messages]  # use asyncio
