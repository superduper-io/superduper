import dataclasses as dc
import json
import os
import typing as t
from functools import lru_cache as cache

import numpy
import tqdm
from httpx import ResponseNotRead
from openai import (
    APITimeoutError,
    InternalServerError,
    OpenAI as SyncOpenAI,
    RateLimitError,
)
from superduper.base import exceptions
from superduper.components.model import APIBaseModel
from superduper.misc.retry import Retry, safe_retry

retry = Retry(
    exception_types=(
        RateLimitError,
        InternalServerError,
        APITimeoutError,
        ResponseNotRead,
    )
)


@cache
@retry
def _available_models(skwargs):
    kwargs = json.loads(skwargs)
    return tuple([r.id for r in SyncOpenAI(**kwargs).models.list().data])


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

    def postinit(self):
        assert isinstance(self.client_kwargs, dict)
        if self.openai_api_key is not None:
            self.client_kwargs['api_key'] = self.openai_api_key
        if self.openai_api_base is not None:
            self.client_kwargs['base_url'] = self.openai_api_base
            self.client_kwargs['default_headers'] = self.openai_api_base

        super().postinit()

    @safe_retry(exceptions.NotFound, verbose=0)
    def setup(self):
        """Initialize the model.

        :param db: Database instance.
        """
        super().setup()

        # dall-e is not currently included in list returned by OpenAI model endpoint
        if 'OPENAI_API_KEY' not in os.environ or (
            'api_key' not in self.client_kwargs.keys() and self.client_kwargs
        ):
            raise exceptions.NotFound("secret", "OPENAI_API_KEY")

        if self.model not in (
            mo := _available_models(json.dumps(self.client_kwargs))
        ) and self.model not in ('dall-e'):
            msg = f'model {self.model} not in OpenAI available models, {mo}'
            raise ValueError(msg)

        self.syncClient = SyncOpenAI(**self.client_kwargs)


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
    def predict(self, X: str):
        """Generates embeddings from text.

        :param X: The text to generate embeddings for.
        """
        e = self.syncClient.embeddings.create(
            input=X, model=self.model, **self.predict_kwargs
        )

        out = numpy.array(e.data[0].embedding).astype('float32')
        if self.postprocess is not None:
            out = self.postprocess(out)
        return out

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

    @retry
    def _predict_a_batch(self, texts: t.List[t.Dict]):
        out = self.syncClient.embeddings.create(
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
        """Post-initialization method."""
        self.takes_context = True
        return super().postinit()

    def _format_prompt(self, context, X):
        prompt = self.prompt.format(context='\n'.join(context))
        return prompt + X

    @retry
    def predict(self, X: str, context: t.Optional[str] = None, **kwargs):
        """Generates text completions from prompts.

        :param X: The prompt.
        :param context: The context to use for the prompt.
        :param kwargs: Additional keyword arguments.
        """
        if context is not None:
            X = self._format_prompt(context, X)
        return (
            self.syncClient.chat.completions.create(
                messages=[{'role': 'user', 'content': X}],
                model=self.model,
                **{**self.predict_kwargs, **kwargs},
            )
            .choices[0]
            .message.content
        )


class OpenAIAudioTranscription(_OpenAI):
    """OpenAI audio transcription predictor.

    :param takes_context: Whether the model takes context into account.
    :param prompt: The prompt to guide the model's style.

    The prompt should contain the `"context"` format variable.

    Example:
    -------
    >>> import io
    >>> from superduper_openai.model import OpenAIAudioTranscription
    >>> with open('test/material/data/test.wav', 'rb') as f:
    >>>     buffer = io.BytesIO(f.read())
    >>> buffer.name = 'test.wav'
    >>> prompt = (
    >>>     'i have some advice for you. write all text in lower-case.'
    >>>     'only make an exception for the following words: {context}'
    >>> )
    >>> model = OpenAIAudioTranscription(identifier='whisper-1', prompt=prompt)
    >>> model.predict(buffer, context=['United States'])

    """

    takes_context: bool = True
    prompt: str = ''

    @retry
    def predict(self, file: t.BinaryIO, context: t.Optional[t.List[str]] = None):
        """Converts a file-like Audio recording to text.

        :param file: The file-like Audio recording to transcribe.
        :param context: The context to use for the prompt.
        """
        if context is not None:
            self.prompt = self.prompt.format(context='\n'.join(context))
        return self.syncClient.audio.transcriptions.create(
            file=file,
            model=self.model,
            prompt=self.prompt,
            **self.predict_kwargs,
        ).text


class OpenAIAudioTranslation(_OpenAI):
    """OpenAI audio translation predictor.

    :param takes_context: Whether the model takes context into account.
    :param prompt: The prompt to guide the model's style.
    :param batch_size: The batch size to use.

    The prompt should contain the `"context"` format variable.

    Example:
    -------
    >>> import io
    >>> from superduper_openai.model import OpenAIAudioTranslation
    >>> with open('test/material/data/german.wav', 'rb') as f:
    >>>     buffer = io.BytesIO(f.read())
    >>> buffer.name = 'test.wav'
    >>> prompt = (
    >>>     'i have some advice for you. write all text in lower-case.'
    >>>     'only make an exception for the following words: {context}'
    >>> )
    >>> e = OpenAIAudioTranslation(identifier='whisper-1', prompt=prompt)
    >>> resp = e.predict(buffer, context=['Emmerich'])
    >>> buffer.close()

    """

    signature: str = 'singleton'

    takes_context: bool = True
    prompt: str = ''
    batch_size: int = 1

    @retry
    def predict(
        self,
        file: t.BinaryIO,
        context: t.Optional[t.List[str]] = None,
    ):
        """Translates a file-like Audio recording to English.

        :param file: The file-like Audio recording to translate.
        :param context: The context to use for the prompt.
        """
        if context is not None:
            self.prompt = self.prompt.format(context='\n'.join(context))
        return (
            self.syncClient.audio.translations.create(
                file=file,
                model=self.model,
                prompt=self.prompt,
                **self.predict_kwargs,
            )
        ).text
