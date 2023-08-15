import dataclasses as dc
import os
import typing as t

import tqdm
from openai import ChatCompletion, Embedding
from openai import Model as OpenAIModel
from openai.error import RateLimitError, ServiceUnavailableError, Timeout, TryAgain

import superduperdb as s
from superduperdb.container.component import Component
from superduperdb.container.encoder import Encoder
from superduperdb.container.model import PredictMixin
from superduperdb.ext.vector.encoder import vector
from superduperdb.misc.compat import cache
from superduperdb.misc.retry import Retry

retry = Retry(
    exception_types=(RateLimitError, ServiceUnavailableError, Timeout, TryAgain)
)


def init_fn():
    s.log.info('Setting OpenAI api-key...')
    os.environ['OPENAI_API_KEY'] = s.CFG.apis.providers['openai'].api_key


@cache
def _available_models():
    return tuple([r['id'] for r in OpenAIModel.list()['data']])


@dc.dataclass
class OpenAI(Component, PredictMixin):
    model: str
    identifier: t.Optional[str] = None  # type: ignore[assignment]
    version: t.Optional[int] = None
    takes_context: bool = False
    encoder: t.Union[Encoder, str, None] = None

    #: A unique name for the class
    type_id: t.ClassVar[str] = 'model'

    @property
    def child_components(self):
        if self.encoder is not None:
            return [('encoder', 'encoder')]
        return []

    def __post_init__(self):
        if self.model not in (mo := _available_models()):
            msg = f'model {self.model} not in OpenAI available models, {mo}'
            raise ValueError(msg)

        if self.identifier is None:
            self.identifier = self.model

        if 'OPENAI_API_KEY' not in os.environ:
            raise ValueError('OPENAI_API_KEY not set')


@dc.dataclass
class OpenAIEmbedding(OpenAI):
    shapes = {'text-embedding-ada-002': (1536,)}
    shape: t.Optional[t.Sequence[int]] = None

    def __post_init__(self):
        super().__post_init__()
        if self.shape is None:
            self.shape = self.shapes[self.identifier]
        self.encoder = vector(self.shape)

    @retry
    def _predict_one(self, X: str, **kwargs):
        e = Embedding.create(input=X, model=self.identifier, **kwargs)
        return e['data'][0]['embedding']

    @retry
    async def _apredict_one(self, X: str, **kwargs):
        e = await Embedding.acreate(input=X, model=self.identifier, **kwargs)
        return e['data'][0]['embedding']

    @retry
    def _predict_a_batch(self, texts, **kwargs):
        out = Embedding.create(input=texts, model=self.identifier, **kwargs)['data']
        return [r['embedding'] for r in out]

    @retry
    async def _apredict_a_batch(self, texts, **kwargs):
        out = await Embedding.acreate(input=texts, model=self.identifier, **kwargs)[
            'data'
        ]
        return [r['embedding'] for r in out]

    def _predict(self, X, batch_size=100, **kwargs):
        if isinstance(X, str):
            return self._predict_one(X)
        out = []
        for i in tqdm.tqdm(range(0, len(X), batch_size)):
            out.extend(self._predict_a_batch(X[i : i + batch_size], **kwargs))
        return out

    async def _apredict(self, X, batch_size=100, **kwargs):
        if isinstance(X, str):
            return await self._apredict_one(X)
        out = []
        for i in range(0, len(X), batch_size):
            out.extend(await self._apredict_a_batch(X[i : i + batch_size], **kwargs))
        return out


@dc.dataclass
class OpenAIChatCompletion(OpenAI):
    takes_context: bool = True
    prompt: t.Optional[str] = None

    def _format_prompt(self, context, X):
        prompt = self.prompt.format(  # type: ignore[union-attr]
            context='\n'.join(context)
        )
        return prompt + X

    @retry
    def _predict_one(self, X, context: t.Optional[t.List[str]] = None, **kwargs):
        if context is not None:
            X = self._format_prompt(context, X)
        return ChatCompletion.create(
            messages=[{'role': 'user', 'content': X}],
            model=self.identifier,
            **kwargs,
        )['choices'][0]['message']['content']

    @retry
    async def _apredict_one(self, X, context: t.Optional[t.List[str]] = None, **kwargs):
        if context is not None:
            X = self._format_prompt(context, X)
        return (
            await ChatCompletion.acreate(
                messages=[{'role': 'user', 'content': X}],
                model=self.identifier,
                **kwargs,
            )
        )['choices'][0]['message']['content']

    def _predict(
        self, X, one: bool = True, context: t.Optional[t.List[str]] = None, **kwargs
    ):
        if context:
            assert one, 'context only works with ``one=True``'
        if one:
            return self._predict_one(X, context=context, **kwargs)
        return [self._predict_one(msg) for msg in X]  # type: ignore[attr-defined]

    # ruff: noqa: E501
    async def _apredict(
        self, X, one: bool = True, context: t.Optional[t.List[str]] = None, **kwargs
    ):
        if context:
            assert one, 'context only works with ``one=True``'
        if one:
            return await self._apredict_one(X, context=context, **kwargs)
        return [await self._apredict_one(msg) for msg in X]  # type: ignore[attr-defined]
