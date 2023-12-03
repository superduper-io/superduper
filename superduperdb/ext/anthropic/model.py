import dataclasses as dc
import typing as t

import anthropic
from anthropic import APIConnectionError, APIError, APIStatusError, APITimeoutError

from superduperdb.backends.ibis.data_backend import IbisDataBackend
from superduperdb.backends.ibis.field_types import dtype
from superduperdb.base.datalayer import Datalayer
from superduperdb.components.model import APIModel
from superduperdb.ext.utils import format_prompt, get_key
from superduperdb.misc.retry import Retry

retry = Retry(
    exception_types=(APIConnectionError, APIError, APIStatusError, APITimeoutError)
)

KEY_NAME = 'ANTHROPIC_API_KEY'


@dc.dataclass
class Anthropic(APIModel):
    """Anthropic predictor."""

    client_kwargs: t.Dict[str, t.Any] = dc.field(default_factory=dict)

    def __post_init__(self):
        self.identifier = self.identifier or self.model


@dc.dataclass
class AnthropicCompletions(Anthropic):
    """Cohere completions (chat) predictor.

    :param takes_context: Whether the model takes context into account.
    :param prompt: The prompt to use to seed the response.
    """

    takes_context: bool = True
    prompt: str = ''

    def pre_create(self, db: Datalayer) -> None:
        super().pre_create(db)
        if isinstance(db.databackend, IbisDataBackend) and self.encoder is None:
            self.encoder = dtype('str')

    @retry
    def _predict_one(self, X, context: t.Optional[t.List[str]] = None, **kwargs):
        if context is not None:
            X = format_prompt(X, self.prompt, context=context)
        client = anthropic.Anthropic(api_key=get_key(KEY_NAME), **self.client_kwargs)
        resp = client.completions.create(prompt=X, model=self.identifier, **kwargs)
        return resp.completion

    @retry
    async def _apredict_one(self, X, context: t.Optional[t.List[str]] = None, **kwargs):
        if context is not None:
            X = format_prompt(X, self.prompt, context=context)
        client = anthropic.AsyncAnthropic(
            api_key=get_key(KEY_NAME), **self.client_kwargs
        )
        resp = await client.completions.create(
            prompt=X, model=self.identifier, **kwargs
        )
        return resp.completion

    def _predict(
        self, X, one: bool = True, context: t.Optional[t.List[str]] = None, **kwargs
    ):
        if context:
            assert one, 'context only works with ``one=True``'
        if one:
            return self._predict_one(X, context=context, **kwargs)
        return [self._predict_one(msg) for msg in X]

    async def _apredict(
        self, X, one: bool = True, context: t.Optional[t.List[str]] = None, **kwargs
    ):
        if context:
            assert one, 'context only works with ``one=True``'
        if one:
            return await self._apredict_one(X, context=context, **kwargs)
        return [await self._apredict_one(msg) for msg in X]
