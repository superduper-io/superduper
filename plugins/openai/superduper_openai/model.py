import base64
import dataclasses as dc
import json
import os
import typing as t

import numpy
import requests
import tqdm
from httpx import ResponseNotRead
from openai import (
    APITimeoutError,
    InternalServerError,
    OpenAI as SyncOpenAI,
    RateLimitError,
)
from openai._types import NOT_GIVEN
from superduper.backends.query_dataset import QueryDataset
from superduper.base import exceptions
from superduper.base.datalayer import Datalayer
from superduper.components.model import APIBaseModel, Inputs
from superduper.misc.compat import cache
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

    def __post_init__(self, db, example):
        super().__post_init__(db, example)

        assert isinstance(self.client_kwargs, dict)
        if self.openai_api_key is not None:
            self.client_kwargs['api_key'] = self.openai_api_key
        if self.openai_api_base is not None:
            self.client_kwargs['base_url'] = self.openai_api_base
            self.client_kwargs['default_headers'] = self.openai_api_base

    @safe_retry(exceptions.MissingSecretsException, verbose=0)
    def init(self, db=None):
        """Initialize the model.

        :param db: Database instance.
        """
        super().init(db=db)

        # dall-e is not currently included in list returned by OpenAI model endpoint
        if 'OPENAI_API_KEY' not in os.environ or (
            'api_key' not in self.client_kwargs.keys() and self.client_kwargs
        ):
            raise exceptions.MissingSecretsException(
                'OPENAI_API_KEY not available neither in environment vars '
                'nor in `client_kwargs`'
            )

        if self.model not in (
            mo := _available_models(json.dumps(self.client_kwargs))
        ) and self.model not in ('dall-e'):
            msg = f'model {self.model} not in OpenAI available models, {mo}'
            raise ValueError(msg)
        self.syncClient = SyncOpenAI(**self.client_kwargs)

    def predict_batches(self, dataset: t.Union[t.List, QueryDataset]) -> t.List:
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

    @property
    def inputs(self):
        """The inputs of the model."""
        return Inputs(['input'])

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

    def __post_init__(self, db, example):
        super().__post_init__(db, example)
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

    def predict_batches(self, dataset: t.Union[t.List, QueryDataset]) -> t.List:
        """Generates text completions from prompts.

        :param dataset: The dataset of prompts.
        """
        out = []
        for i in range(len(dataset)):
            args, kwargs = self.handle_input_type(
                data=dataset[i], signature=self.signature
            )
            out.append(self.predict(*args, **kwargs))
        return out


class OpenAIImageCreation(_OpenAI):
    """OpenAI image creation predictor.

    :param takes_context: Whether the model takes context into account.
    :param prompt: The prompt to use to seed the response.
    :param n: The number of images to generate.
    :param response_format: The response format to use.

    Example:
    -------
    >>> from superduper_openai.model import OpenAIImageCreation
    >>>
    >>> model = OpenAIImageCreation(
    >>>     model="dall-e",
    >>>     prompt="a close up, studio photographic portrait of a {context}",
    >>>     response_format="url",
    >>> )
    >>> model.predict("cat")

    """

    signature: str = 'singleton'
    takes_context: bool = True
    prompt: str = ''
    n: int = 1
    response_format: str = 'b64_json'

    def _pre_create(self, db: Datalayer):
        """Pre creates the model.

        :param db: The datalayer instance.
        """
        self.datatype = self.datatype or 'bytes'

    def _format_prompt(self, context, X):
        prompt = self.prompt.format(context='\n'.join(context))
        return prompt + X

    @retry
    def predict(self, X: str):
        """Generates images from text prompts.

        :param X: The text prompt.
        """
        if self.response_format == 'b64_json':
            resp = self.syncClient.images.generate(
                prompt=X,
                n=self.n,
                response_format='b64_json',
                **self.predict_kwargs,
            )
            b64_json = resp.data[0].b64_json
            assert b64_json is not None
            return base64.b64decode(b64_json)
        else:
            url = (
                self.syncClient.images.generate(
                    prompt=X, n=self.n, **self.predict_kwargs
                )
                .data[0]
                .url
            )
            return requests.get(url).content

    def predict_batches(self, dataset: t.Union[t.List, QueryDataset]) -> t.List:
        """Generates images from text prompts.

        :param dataset: The dataset of text prompts.
        """
        out = []
        for i in range(len(dataset)):
            args, kwargs = self.handle_input_type(
                data=dataset[i], signature=self.signature
            )
            out.append(self.predict(*args, **kwargs))
        return out


class OpenAIImageEdit(_OpenAI):
    """OpenAI image edit predictor.

    :param takes_context: Whether the model takes context into account.
    :param prompt: The prompt to use to seed the response.
    :param response_format: The response format to use.
    :param n: The number of images to generate.

    Example:
    -------
    >>> import io
    >>>
    >>> from superduper_openai.model import OpenAIImageEdit
    >>>
    >>> model = OpenAIImageEdit(
    >>>     model="dall-e",
    >>>     prompt="A celebration party at the launch of {context}",
    >>>     response_format="url",
    >>> )
    >>> with open("test/material/data/rickroll.png", "rb") as f:
    >>>     buffer = io.BytesIO(f.read())
    >>> model.predict(buffer, context=["superduper"])

    """

    takes_context: bool = True
    prompt: str = ''
    response_format: str = 'b64_json'
    n: int = 1

    def _format_prompt(self, context):
        prompt = self.prompt.format(context='\n'.join(context))
        return prompt

    def _pre_create(self, db: Datalayer):
        """Pre creates the model.

        :param db: The datalayer instance.
        """
        self.datatype = self.datatype or 'bytes'

    @retry
    def predict(
        self,
        image: t.BinaryIO,
        mask: t.Optional[t.BinaryIO] = None,
        context: t.Optional[t.List[str]] = None,
    ):
        """Edits an image.

        :param image: The image to edit.
        :param mask: The mask to apply to the image.
        :param context: The context to use for the prompt.
        """
        if context is not None:
            self.prompt = self._format_prompt(context)

        maybe_mask = mask or NOT_GIVEN

        if self.response_format == 'b64_json':
            b64_json = (
                self.syncClient.images.edit(
                    image=image,
                    mask=maybe_mask,
                    prompt=self.prompt,
                    n=self.n,
                    response_format='b64_json',
                    **self.predict_kwargs,
                )
                .data[0]
                .b64_json
            )
            out = base64.b64decode(b64_json)
        else:
            url = (
                self.syncClient.images.edit(
                    image=image,
                    mask=maybe_mask,
                    prompt=self.prompt,
                    n=self.n,
                    **self.predict_kwargs,
                )
                .data[0]
                .url
            )
            out = requests.get(url).content
        return out

    def predict_batches(self, dataset: t.Union[t.List, QueryDataset]) -> t.List:
        """Predicts the output for a dataset of images.

        :param dataset: The dataset of images.
        """
        out = []
        for i in range(len(dataset)):
            args, kwargs = self.handle_input_type(
                data=dataset[i], signature=self.signature
            )
            out.append(self.predict(*args, **kwargs))
        return out


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

    def _pre_create(self, db: Datalayer):
        """Pre creates the model.

        :param db: The datalayer instance.
        """
        self.datatype = self.datatype or 'str'

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

    @retry
    def _predict_a_batch(self, files: t.List[t.BinaryIO], **kwargs):
        """Converts multiple file-like Audio recordings to text."""
        resps = [
            self.syncClient.audio.transcriptions.create(
                file=file, model=self.model, **self.predict_kwargs
            )
            for file in files
        ]
        return [resp.text for resp in resps]


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

    def _pre_create(self, db: Datalayer):
        """Translates a file-like Audio recording to English.

        :param db: The datalayer to use for the model.
        """
        self.datatype = self.datatype or 'str'

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

    @retry
    def _predict_a_batch(self, files: t.List[t.BinaryIO]):
        """Translates multiple file-like Audio recordings to English."""
        # TODO use async or threads
        resps = [
            self.syncClient.audio.translations.create(
                file=file, model=self.model, **self.predict_kwargs
            )
            for file in files
        ]
        return [resp.text for resp in resps]
