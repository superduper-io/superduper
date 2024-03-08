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
from superduperdb.components.model import APIModel, Inputs
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


@dc.dataclass(kw_only=True)
class _OpenAI(APIModel):
    '''
    :param client_kwargs: The kwargs to be passed to OpenAI
    '''

    client_kwargs: t.Optional[dict] = dc.field(default_factory=dict)
    __doc__ = APIModel.__doc__  # type: ignore[assignment]

    def __post_init__(self, artifacts):
        super().__post_init__(artifacts)

        assert isinstance(self.client_kwargs, dict)

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

    def predict(self, dataset: t.Union[t.List, QueryDataset]) -> t.List:
        out = []
        for i in tqdm.tqdm(range(0, len(dataset), self.batch_size)):
            batch = [
                dataset[i] for i in range(i, min(len(dataset), i + self.batch_size))
            ]
            out.extend(self._predict_a_batch(batch))
        return out


@dc.dataclass(kw_only=True)
class OpenAIEmbedding(_OpenAI):
    """
    OpenAI embedding predictor
    {_openai_parameters}
    :param shape: The shape as ``tuple`` of the embedding.
    """

    __doc__ = __doc__.format(_openai_parameters=_OpenAI.__doc__)

    shape: t.Optional[t.Sequence[int]] = None
    shapes: t.ClassVar[t.Dict] = {'text-embedding-ada-002': (1536,)}
    signature: t.ClassVar[str] = 'singleton'
    batch_size: int = 100

    @property
    def inputs(self):
        return Inputs(['input'])

    def __post_init__(self, artifacts):
        super().__post_init__(artifacts)
        if self.shape is None:
            self.shape = self.shapes[self.model]

    def pre_create(self, db):
        super().pre_create(db)
        if isinstance(db.databackend, IbisDataBackend):
            if self.datatype is None:
                self.datatype = sqlvector(self.shape)
        elif self.datatype is None:
            self.datatype = vector(self.shape)

    @retry
    def predict_one(self, X: str):
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


@dc.dataclass(kw_only=True)
class OpenAIChatCompletion(_OpenAI):
    """OpenAI chat completion predictor.
    {_openai_parameters}
    :param prompt: The prompt to use to seed the response.
    """

    signature: t.ClassVar[str] = 'singleton'
    __doc__ = __doc__.format(_openai_parameters=_OpenAI.__doc__)

    batch_size: int = 1
    prompt: str = ''

    @property
    def inputs(self):
        return Inputs(['content', 'context'])

    def __post_init__(self, artifacts):
        super().__post_init__(artifacts)
        self.takes_context = True

    def _format_prompt(self, context, X):
        prompt = self.prompt.format(context='\n'.join(context))
        return prompt + X

    def pre_create(self, db: Datalayer) -> None:
        super().pre_create(db)
        if isinstance(db.databackend, IbisDataBackend) and self.datatype is None:
            self.datatype = dtype('str')

    @retry
    def predict_one(self, X: str, context: t.Optional[str] = None):
        if context is not None:
            X = self._format_prompt(context, X)
        return (
            self.syncClient.chat.completions.create(
                messages=[{'role': 'user', 'content': X}],
                model=self.model,
                **self.predict_kwargs,
            )
            .choices[0]
            .message.content
        )

    def predict(self, dataset: t.Union[t.List, QueryDataset]) -> t.List:
        out = []
        for i in range(len(dataset)):
            args, kwargs = self.handle_input_type(
                data=dataset[i], signature=self.signature
            )
            out.append(self.predict_one(*args, **kwargs))
        return out


@dc.dataclass(kw_only=True)
class OpenAIImageCreation(_OpenAI):
    """OpenAI image creation predictor.
    {_openai_parameters}
    :param takes_context: Whether the model takes context into account.
    :param prompt: The prompt to use to seed the response.
    """

    signature: t.ClassVar[str] = 'singleton'

    __doc__ = __doc__.format(_openai_parameters=_OpenAI.__doc__)

    takes_context: bool = True
    prompt: str = ''
    n: int = 1
    response_format: str = 'b64_json'

    def pre_create(self, db: Datalayer) -> None:
        super().pre_create(db)
        if isinstance(db.databackend, IbisDataBackend) and self.datatype is None:
            self.datatype = dtype('bytes')

    def _format_prompt(self, context, X):
        prompt = self.prompt.format(context='\n'.join(context))
        return prompt + X

    @retry
    def predict_one(self, X: str):
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

    def predict(self, dataset: t.Union[t.List, QueryDataset]) -> t.List:
        out = []
        for i in range(len(dataset)):
            args, kwargs = self.handle_input_type(
                data=dataset[i], signature=self.signature
            )
            out.append(self.predict_one(*args, **kwargs))
        return out


@dc.dataclass(kw_only=True)
class OpenAIImageEdit(_OpenAI):
    """OpenAI image edit predictor.
    {_openai_parameters}
    :param takes_context: Whether the model takes context into account.
    :param prompt: The prompt to use to seed the response.
    """

    __doc__ = __doc__.format(_openai_parameters=_OpenAI.__doc__)

    takes_context: bool = True
    prompt: str = ''
    response_format: str = 'b64_json'
    n: int = 1

    def _format_prompt(self, context):
        prompt = self.prompt.format(context='\n'.join(context))
        return prompt

    def pre_create(self, db: Datalayer) -> None:
        super().pre_create(db)
        if isinstance(db.databackend, IbisDataBackend) and self.datatype is None:
            self.datatype = dtype('bytes')

    @retry
    def predict_one(
        self,
        image: t.BinaryIO,
        mask: t.Optional[t.BinaryIO] = None,
        context: t.Optional[t.List[str]] = None,
    ):
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

    def predict(self, dataset: t.Union[t.List, QueryDataset]) -> t.List:
        out = []
        for i in range(len(dataset)):
            args, kwargs = self.handle_input_type(
                data=dataset[i], signature=self.signature
            )
            out.append(self.predict_one(*args, **kwargs))
        return out


@dc.dataclass(kw_only=True)
class OpenAIAudioTranscription(_OpenAI):
    """OpenAI audio transcription predictor.
    {_openai_parameters}
    :param takes_context: Whether the model takes context into account.
    :param prompt: The prompt to guide the model's style. Should contain ``{context}``.
    """

    __doc__ = __doc__.format(_openai_parameters=_OpenAI.__doc__, context='{context}')

    takes_context: bool = True
    prompt: str = ''

    def pre_create(self, db: Datalayer) -> None:
        super().pre_create(db)
        if isinstance(db.databackend, IbisDataBackend) and self.datatype is None:
            self.datatype = dtype('str')

    @retry
    def predict_one(self, file: t.BinaryIO, context: t.Optional[t.List[str]] = None):
        "Converts a file-like Audio recording to text."
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
        "Converts multiple file-like Audio recordings to text."
        resps = [
            self.syncClient.audio.transcriptions.create(
                file=file, model=self.model, **self.predict_kwargs
            )
            for file in files
        ]
        return [resp.text for resp in resps]


@dc.dataclass(kw_only=True)
class OpenAIAudioTranslation(_OpenAI):
    """OpenAI audio translation predictor.
    {_openai_parameters}
    :param takes_context: Whether the model takes context into account.
    :param prompt: The prompt to guide the model's style. Should contain ``{context}``.
    """

    signature: t.ClassVar[str] = 'singleton'

    __doc__ = __doc__.format(_openai_parameters=_OpenAI.__doc__, context='{context}')

    takes_context: bool = True
    prompt: str = ''
    batch_size: int = 1

    def pre_create(self, db: Datalayer) -> None:
        super().pre_create(db)
        if isinstance(db.databackend, IbisDataBackend) and self.datatype is None:
            self.datatype = dtype('str')

    @retry
    def predict_one(
        self,
        file: t.BinaryIO,
        context: t.Optional[t.List[str]] = None,
    ):
        "Translates a file-like Audio recording to English."
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
        "Translates multiple file-like Audio recordings to English."
        # TODO use async or threads
        resps = [
            self.syncClient.audio.translations.create(
                file=file, model=self.model, **self.predict_kwargs
            )
            for file in files
        ]
        return [resp.text for resp in resps]
