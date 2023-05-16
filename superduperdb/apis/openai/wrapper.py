import os

from superduperdb import cf
from openai import ChatCompletion as _ChatCompletion
from openai import Embedding as _Embedding
from superduperdb.models.base import SuperDuperModel


class BaseOpenAI(SuperDuperModel):
    def __init__(self, model_id):
        self.model_id = model_id
        assert 'OPENAI_API_KEY' in os.environ, "OPENAI_API_KEY not set"


class Embedding(BaseOpenAI):
    def __init__(self, model_id):
        super().__init__(model_id)

    async def _predict_one(self, text, **kwargs):
        # This doesn't work...
        out = await _Embedding.acreate(input=text, model=self.model_id, **kwargs)
        return out['data'][0]['embedding']

    def predict_one(self, text, **kwargs):
        return _Embedding.create(input=text, model=self.model_id, **kwargs)['data'][0]['embedding']

    def predict(self, texts, **kwargs):  # asyncio?
        out = _Embedding.create(input=texts, model=self.model_id, **kwargs)['data']
        return [r['embedding'] for r in out]


class ChatCompletion(BaseOpenAI):
    def __init__(self, model_id):
        super().__init__(model_id)

    def predict_one(self, message, **kwargs):
        return _ChatCompletion.create(
            messages=[{'role': 'user', 'content': message}],
            model=self.model_id,
            **kwargs
        )['choices'][0]['message']['content']

    def predict(self, messages, **kwargs):
        return [self.predict_one(msg) for msg in messages]  # use asyncio

