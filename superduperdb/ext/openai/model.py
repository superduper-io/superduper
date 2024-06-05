import base64
import dataclasses as dc
import json
import os
import typing as t

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

from superduperdb.backends.ibis.data_backend import IbisDataBackend
from superduperdb.backends.ibis.field_types import dtype
from superduperdb.backends.query_dataset import QueryDataset
from superduperdb.base.datalayer import Datalayer
from superduperdb.components.model import APIBaseModel, Inputs
from superduperdb.components.vector_index import sqlvector, vector
from superduperdb.misc.compat import cache
from superduperdb.misc.retry import Retry

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

    openai_api_key: t.Optional[str] = None
    openai_api_base: t.Optional[str] = None
    client_kwargs: t.Optional[dict] = dc.field(default_factory=dict)

    def __post_init__(self, db, artifacts):
        super().__post_init__(db, artifacts)

        assert isinstance(self.client_kwargs, dict)

        if self.openai_api_key is not None:
            self.client_kwargs['api_key'] = self.openai_api_key
        if self.openai_api_base is not None:
            self.client_kwargs['base_url'] = self.openai_api_base

        # dall-e is not currently included in list returned by OpenAI model endpoint
        if self.model not in (
            mo := _available_models(json.dumps(self.client_kwargs))
        ) and self.model not in ('dall-e'):
            msg = f'model {self.model} not in OpenAI available models, {mo}'
            raise ValueError(msg)

        self.syncClient = SyncOpenAI(**self.client_kwargs)

        if 'OPENAI_API_KEY' not in os.environ and (
            'api_key' not in self.client_kwargs.keys() and self.client_kwargs
        ):
            raise ValueError(
                'OPENAI_API_KEY not available neither in environment vars '
                'nor in `client_kwargs`'
            )

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
    """

    shapes: t.ClassVar[t.Dict] = {'text-embedding-ada-002': (1536,)}

    shape: t.Optional[t.Sequence[int]] = None
    signature: str = 'singleton'
    batch_size: int = 100

    @property
    def inputs(self):
        """The inputs of the model."""
        return Inputs(['input'])

    def __post_init__(self, db, artifacts):
        super().__post_init__(db, artifacts)
        if self.shape is None:
            self.shape = self.shapes[self.model]

    def pre_create(self, db):
        """Pre creates the model.

        If the datatype is not set and the datalayer is an IbisDataBackend,
        the datatype is set to ``sqlvector`` or ``vector``.

        :param db: The datalayer instance.
        """
        super().pre_create(db)
        if isinstance(db.databackend, IbisDataBackend):
            if self.datatype is None:
                self.datatype = sqlvector(self.shape)
        elif self.datatype is None:
            self.datatype = vector(shape=self.shape)

    @retry
    def predict(self, X: str):
        """Generates embeddings from text.

        :param X: The text to generate embeddings for.
        """
        e = self.syncClient.embeddings.create(
            input=X, model=self.model, **self.predict_kwargs
        )
        return e.data[0].embedding

    @retry
    def _predict_a_batch(self, texts: t.List[t.Dict]):
        out = self.syncClient.embeddings.create(
            input=texts, model=self.model, **self.predict_kwargs
        )
        return [r.embedding for r in out.data]


class OpenAIChatCompletion(_OpenAI):
    """OpenAI chat completion predictor.

    :param batch_size: The batch size to use.
    :param prompt: The prompt to use to seed the response.
    """

    signature: str = 'singleton'
    batch_size: int = 1
    prompt: str = ''

    def __post_init__(self, db, artifacts):
        super().__post_init__(db, artifacts)
        self.takes_context = True

    def _format_prompt(self, context, X):
        prompt = self.prompt.format(context='\n'.join(context))
        return prompt + X

    def pre_create(self, db: Datalayer) -> None:
        """Pre creates the model.

        If the datatype is not set and the datalayer is an IbisDataBackend,
        the datatype is set to ``dtype('str')``.

        :param db: The datalayer instance.
        """
        super().pre_create(db)
        if isinstance(db.databackend, IbisDataBackend) and self.datatype is None:
            self.datatype = dtype('str')

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
    """

    signature: str = 'singleton'
    takes_context: bool = True
    prompt: str = ''
    n: int = 1
    response_format: str = 'b64_json'

    def pre_create(self, db: Datalayer) -> None:
        """Pre creates the model.

        If the datatype is not set and the datalayer is an IbisDataBackend,
        the datatype is set to ``dtype('bytes')``.

        :param db: The datalayer instance.
        """
        super().pre_create(db)
        if isinstance(db.databackend, IbisDataBackend) and self.datatype is None:
            self.datatype = dtype('bytes')

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
    """

    takes_context: bool = True
    prompt: str = ''
    response_format: str = 'b64_json'
    n: int = 1

    def _format_prompt(self, context):
        prompt = self.prompt.format(context='\n'.join(context))
        return prompt

    def pre_create(self, db: Datalayer) -> None:
        """Pre creates the model.

        If the datatype is not set and the datalayer is an IbisDataBackend,
        the datatype is set to ``dtype('bytes')``.

        :param db: The datalayer instance.
        """
        super().pre_create(db)
        if isinstance(db.databackend, IbisDataBackend) and self.datatype is None:
            self.datatype = dtype('bytes')

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
    """

    takes_context: bool = True
    prompt: str = ''

    def pre_create(self, db: Datalayer) -> None:
        """Pre creates the model.

        If the datatype is not set and the datalayer is an IbisDataBackend,
        the datatype is set to ``dtype('str')``.

        :param db: The datalayer instance.
        """
        super().pre_create(db)
        if isinstance(db.databackend, IbisDataBackend) and self.datatype is None:
            self.datatype = dtype('str')

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
    """

    signature: str = 'singleton'

    takes_context: bool = True
    prompt: str = ''
    batch_size: int = 1

    def pre_create(self, db: Datalayer) -> None:
        """Translates a file-like Audio recording to English.

        :param db: The datalayer to use for the model.
        """
        super().pre_create(db)
        if isinstance(db.databackend, IbisDataBackend) and self.datatype is None:
            self.datatype = dtype('str')

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
