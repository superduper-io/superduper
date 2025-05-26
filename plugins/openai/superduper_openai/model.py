import dataclasses as dc
import os
import traceback
import typing as t

import numpy
import tqdm
from httpx import ResponseNotRead
from openai import (
    APIError,
    APITimeoutError,
    InternalServerError,
    OpenAI as SyncOpenAI,
    RateLimitError,
)
from superduper import CFG, logging
from superduper.base.datalayer import Datalayer
from superduper.components.model import APIBaseModel, method_wrapper
from superduper.misc.files import load_secrets
from superduper.misc.retry import Retry, safe_retry

retry = Retry(
    exception_types=(
        RateLimitError,
        InternalServerError,
        APITimeoutError,
        ResponseNotRead,
    )
)


class _OpenAI(APIBaseModel):
    """Base class for OpenAI models.

    :param openai_api_key: The OpenAI API key.
    :param openai_api_base: The server to use for requests.
    :param client_kwargs: The kwargs to be passed to OpenAI
    """

    breaks: t.ClassVar[t.Tuple[str]] = ('model',)

    openai_api_key: t.Optional[str] = None
    openai_api_base: t.Optional[str] = None
    client_kwargs: t.Optional[dict] = dc.field(default_factory=dict)

    def _wrapper(self, item: t.Any):
        """Wrap the item with the model.

        :param item: Item to wrap.
        """
        return method_wrapper(self.predict, item, self.signature)

    def postinit(self):
        assert isinstance(self.client_kwargs, dict)
        if self.openai_api_key is not None:
            self.client_kwargs['api_key'] = self.openai_api_key
        if self.openai_api_base is not None:
            self.client_kwargs['base_url'] = self.openai_api_base
            self.client_kwargs['default_headers'] = self.openai_api_base

    @property
    def sync_client(self):
        if 'OPENAI_API_KEY' not in self.client_kwargs and os.path.exists(
            CFG.secrets_volume
        ):
            try:
                load_secrets()
            except Exception as e:
                logging.error(traceback.format_exc())
                logging.error(f'Error loading secrets: {e}')
                raise e
        return SyncOpenAI(**self.client_kwargs)

    @safe_retry(APIError)
    def predict_batches(self, dataset: t.List) -> t.List:
        """Predict on a dataset.

        :param dataset: The dataset to predict on.
        """
        out = []
        for i in tqdm.tqdm(range(0, len(dataset), self.batch_size)):
            batch = [
                dataset[i] for i in range(i, min(len(dataset), i + self.batch_size))
            ]
            out.extend(self._predict_a_batch(batch))
        return out


class OpenAIEmbedding(_OpenAI):
    """OpenAI embedding predictor.

    :param shape: The shape as ``tuple`` of the embedding.
    :param batch_size: The batch size to use.

    Example:
    -------
    >>> from superduper_openai.model import OpenAIEmbedding
    >>> model = OpenAIEmbedding(identifier='text-embedding-ada-002')
    >>> model.predict('Hello, world!')

    """

    signature: str = 'singleton'
    batch_size: int = 100

    @retry
    @safe_retry(APIError)
    def predict(self, X: str):
        """Generates embeddings from text.

        :param X: The text to generate embeddings for.
        """
        e = self.sync_client.embeddings.create(
            input=X, model=self.model, **self.predict_kwargs
        )

        out = numpy.array(e.data[0].embedding).astype('float32')
        if self.postprocess is not None:
            out = self.postprocess(out)
        return out

    @retry
    @safe_retry(APIError)
    def _predict_a_batch(self, texts: t.List[t.Dict]):
        out = self.sync_client.embeddings.create(
            input=texts, model=self.model, **self.predict_kwargs
        )
        out = [numpy.array(r.embedding).astype('float32') for r in out.data]
        if self.postprocess is not None:
            out = list(map(self.postprocess, out))
        return out


class OpenAIChatCompletion(_OpenAI):
    """OpenAI chat completion predictor.

    :param batch_size: The batch size to use.
    :param prompt: The prompt to use to seed the response.

    Example:
    -------
    >>> from superduper_openai.model import OpenAIChatCompletion
    >>> model = OpenAIChatCompletion(model='gpt-3.5-turbo', prompt='Hello, {context}')
    >>> model.predict('Hello, world!')

    """

    signature: str = 'singleton'
    batch_size: int = 1
    prompt: str = ''

    def postinit(self):
        """Post initialization of the model."""
        self.takes_context = True

    def _format_prompt(self, context, X):
        prompt = self.prompt.format(context='\n'.join(context))
        return prompt + X

    def _pre_create(self, db: Datalayer) -> None:
        """Pre creates the model.

        :param db: The datalayer instance.
        """
        self.datatype = self.datatype or 'str'

    @retry
    @safe_retry(APIError)
    def predict(self, X: str, context: t.Optional[str] = None, **kwargs):
        """Generates text completions from prompts.

        :param X: The prompt.
        :param context: The context to use for the prompt.
        :param kwargs: Additional keyword arguments.
        """
        if context is not None:
            X = self._format_prompt(context, X)
        return (
            self.sync_client.chat.completions.create(
                messages=[{'role': 'user', 'content': X}],
                model=self.model,
                **{**self.predict_kwargs, **kwargs},
            )
            .choices[0]
            .message.content
        )

    def predict_batches(self, dataset: t.List) -> t.List:
        """Generates text completions from prompts.

        :param dataset: The dataset of prompts.
        """
        out = []
        for i in range(len(dataset)):
            out.append(method_wrapper(self.predict, dataset[i], self.signature))
        return out
