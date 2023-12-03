import asyncio
import base64
import dataclasses as dc
import itertools
import os
import typing as t

import aiohttp
import requests
import tqdm
from openai import (
    APITimeoutError,
    AsyncOpenAI,
    InternalServerError,
    OpenAI as SyncOpenAI,
    RateLimitError,
)

from superduperdb.backends.ibis.data_backend import IbisDataBackend
from superduperdb.backends.ibis.field_types import dtype
from superduperdb.base.datalayer import Datalayer
from superduperdb.components.model import APIModel
from superduperdb.components.vector_index import sqlvector, vector
from superduperdb.misc.compat import cache
from superduperdb.misc.retry import Retry

retry = Retry(exception_types=(RateLimitError, InternalServerError, APITimeoutError))


@cache
def _available_models():
    return tuple([r.id for r in SyncOpenAI().models.list().data])


@dc.dataclass
class OpenAI(APIModel):
    """OpenAI predictor."""

    def __post_init__(self):
        # dall-e is not currently included in list returned by OpenAI model endpoint
        if self.model not in (mo := _available_models()) and self.model not in (
            'dall-e'
        ):
            msg = f'model {self.model} not in OpenAI available models, {mo}'
            raise ValueError(msg)

        self.identifier = self.identifier or self.model

        self.syncClient = SyncOpenAI()
        self.asyncClient = AsyncOpenAI()

        if 'OPENAI_API_KEY' not in os.environ:
            raise ValueError('OPENAI_API_KEY not set')


@dc.dataclass
class OpenAIEmbedding(OpenAI):
    """OpenAI embedding predictor

    :param shape: The shape as ``tuple`` of the embedding.
    """

    shape: t.Optional[t.Sequence[int]] = None

    shapes: t.ClassVar[t.Dict] = {'text-embedding-ada-002': (1536,)}

    def __post_init__(self):
        super().__post_init__()
        if self.shape is None:
            self.shape = self.shapes[self.identifier]

    def pre_create(self, db):
        super().pre_create(db)
        if isinstance(db.databackend, IbisDataBackend):
            if self.encoder is None:
                self.encoder = sqlvector(self.shape)
        elif self.encoder is None:
            self.encoder = vector(self.shape)

    @retry
    def _predict_one(self, X: str, **kwargs):
        e = self.syncClient.embeddings.create(input=X, model=self.identifier, **kwargs)
        return e.data[0].embedding

    @retry
    async def _apredict_one(self, X: str, **kwargs):
        e = await self.asyncClient.embeddings.create(
            input=X, model=self.identifier, **kwargs
        )
        return e.data[0].embedding

    @retry
    def _predict_a_batch(self, texts: t.List[str], **kwargs):
        out = self.syncClient.embeddings.create(
            input=texts, model=self.identifier, **kwargs
        )
        return [r.embedding for r in out.data]

    @retry
    async def _apredict_a_batch(self, texts: t.List[str], **kwargs):
        out = await self.asyncClient.embeddings.create(
            input=texts, model=self.identifier, **kwargs
        )
        return [r.embedding for r in out.data]

    def _predict(self, X, one: bool = False, **kwargs):
        if isinstance(X, str):
            return self._predict_one(X)
        out = []
        batch_size = kwargs.pop('batch_size', 100)
        for i in tqdm.tqdm(range(0, len(X), batch_size)):
            out.extend(self._predict_a_batch(X[i : i + batch_size], **kwargs))
        return out

    async def _apredict(self, X, one: bool = False, **kwargs):
        if isinstance(X, str):
            return await self._apredict_one(X)
        out = []
        batch_size = kwargs.pop('batch_size', 100)
        # Note: we submit the async requests in serial to avoid rate-limiting
        for i in range(0, len(X), batch_size):
            out.extend(await self._apredict_a_batch(X[i : i + batch_size], **kwargs))
        return out


@dc.dataclass
class OpenAIChatCompletion(OpenAI):
    """OpenAI chat completion predictor.

    :param takes_context: Whether the model takes context into account.
    :param prompt: The prompt to use to seed the response.
    """

    takes_context: bool = True
    prompt: str = ''

    def _format_prompt(self, context, X):
        prompt = self.prompt.format(context='\n'.join(context))
        return prompt + X

    def pre_create(self, db: Datalayer) -> None:
        super().pre_create(db)
        if isinstance(db.databackend, IbisDataBackend) and self.encoder is None:
            self.encoder = dtype('str')

    @retry
    def _predict_one(self, X, context: t.Optional[t.List[str]] = None, **kwargs):
        if context is not None:
            X = self._format_prompt(context, X)
        return (
            self.syncClient.chat.completions.create(
                messages=[{'role': 'user', 'content': X}],
                model=self.identifier,
                **kwargs,
            )
            .choices[0]
            .message.content
        )

    @retry
    async def _apredict_one(self, X, context: t.Optional[t.List[str]] = None, **kwargs):
        if context is not None:
            X = self._format_prompt(context, X)
        return (
            (
                await self.asyncClient.chat.completions.create(
                    messages=[{'role': 'user', 'content': X}],
                    model=self.identifier,
                    **kwargs,
                )
            )
            .choices[0]
            .message.content
        )

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


@dc.dataclass
class OpenAIImageCreation(OpenAI):
    """OpenAI image creation predictor.

    :param takes_context: Whether the model takes context into account.
    :param prompt: The prompt to use to seed the response.
    """

    takes_context: bool = True
    prompt: str = ''

    def pre_create(self, db: Datalayer) -> None:
        super().pre_create(db)
        if isinstance(db.databackend, IbisDataBackend) and self.encoder is None:
            self.encoder = dtype('bytes')

    def _format_prompt(self, context, X):
        prompt = self.prompt.format(context='\n'.join(context))
        return prompt + X

    @retry
    def _predict_one(
        self,
        X,
        n: int,
        response_format: str,
        context: t.Optional[t.List[str]] = None,
        **kwargs,
    ):
        if context is not None:
            X = self._format_prompt(context, X)
        if response_format == 'b64_json':
            b64_json = (
                self.syncClient.images.generate(
                    prompt=X, n=n, response_format='b64_json'
                )
                .data[0]
                .b64_json
            )
            return base64.b64decode(b64_json)
        else:
            url = self.syncClient.images.generate(prompt=X, n=n, **kwargs).data[0].url
            return requests.get(url).content

    @retry
    async def _apredict_one(
        self,
        X,
        n: int,
        response_format: str,
        context: t.Optional[t.List[str]] = None,
        **kwargs,
    ):
        if context is not None:
            X = self._format_prompt(context, X)
        if response_format == 'b64_json':
            b64_json = (
                (
                    await self.asyncClient.images.generate(
                        prompt=X, n=n, response_format='b64_json'
                    )
                )
                .data[0]
                .b64_json
            )
            return base64.b64decode(b64_json)
        else:
            url = (
                (await self.asyncClient.images.generate(prompt=X, n=n, **kwargs))
                .data[0]
                .url
            )
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as resp:
                    return await resp.read()

    def _predict(
        self, X, one: bool = True, context: t.Optional[t.List[str]] = None, **kwargs
    ):
        response_format = kwargs.pop('response_format', 'b64_json')
        if context:
            assert one, 'context only works with ``one=True``'
        if one:
            return self._predict_one(
                X, n=1, response_format=response_format, context=context, **kwargs
            )
        return [
            self._predict_one(msg, n=1, response_format=response_format) for msg in X
        ]

    async def _apredict(
        self, X, one: bool = True, context: t.Optional[t.List[str]] = None, **kwargs
    ):
        response_format = kwargs.pop('response_format', 'b64_json')
        if context:
            assert one, 'context only works with ``one=True``'
        if one:
            return await self._apredict_one(
                X, context=context, n=1, response_format=response_format, **kwargs
            )
        return [
            await self._apredict_one(msg, n=1, response_format=response_format)
            for msg in X
        ]


@dc.dataclass
class OpenAIImageEdit(OpenAI):
    """OpenAI image edit predictor.

    :param takes_context: Whether the model takes context into account.
    :param prompt: The prompt to use to seed the response.
    """

    takes_context: bool = True
    prompt: str = ''

    def _format_prompt(self, context):
        prompt = self.prompt.format(context='\n'.join(context))
        return prompt

    def pre_create(self, db: Datalayer) -> None:
        super().pre_create(db)
        if isinstance(db.databackend, IbisDataBackend) and self.encoder is None:
            self.encoder = dtype('bytes')

    @retry
    def _predict_one(
        self,
        image: t.BinaryIO,
        n: int,
        response_format: str,
        context: t.Optional[t.List[str]] = None,
        mask_png_path: t.Optional[str] = None,
        **kwargs,
    ):
        if context is not None:
            self.prompt = self._format_prompt(context)

        if mask_png_path is not None:
            with open(mask_png_path, 'rb') as f:
                mask = f.read()
            kwargs['mask'] = mask

        if response_format == 'b64_json':
            b64_json = (
                self.syncClient.images.edit(
                    image=image,
                    prompt=self.prompt,
                    n=n,
                    response_format='b64_json',
                    **kwargs,
                )
                .data[0]
                .b64_json
            )
            return base64.b64decode(b64_json)
        else:
            url = (
                self.syncClient.images.edit(
                    image=image, prompt=self.prompt, n=n, **kwargs
                )
                .data[0]
                .url
            )
            return requests.get(url).content

    @retry
    async def _apredict_one(
        self,
        image: t.BinaryIO,
        n: int,
        response_format: str,
        context: t.Optional[t.List[str]] = None,
        mask_png_path: t.Optional[str] = None,
        **kwargs,
    ):
        if context is not None:
            self.prompt = self._format_prompt(context)

        if mask_png_path is not None:
            with open(mask_png_path, 'rb') as f:
                mask = f.read()
            kwargs['mask'] = mask

        if response_format == 'b64_json':
            b64_json = (
                (
                    await self.asyncClient.images.edit(
                        image=image,
                        prompt=self.prompt,
                        n=n,
                        response_format='b64_json',
                        **kwargs,
                    )
                )
                .data[0]
                .b64_json
            )
            return base64.b64decode(b64_json)
        else:
            url = (
                (
                    await self.asyncClient.images.edit(
                        image=image, prompt=self.prompt, n=n, **kwargs
                    )
                )
                .data[0]
                .url
            )
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as resp:
                    return await resp.read()

    def _predict(
        self, X, one: bool = True, context: t.Optional[t.List[str]] = None, **kwargs
    ):
        response_format = kwargs.pop('response_format', 'b64_json')
        if context:
            assert one, 'context only works with ``one=True``'
        if one:
            return self._predict_one(
                image=X, n=1, response_format=response_format, context=context, **kwargs
            )
        return [
            self._predict_one(
                image=image, n=1, response_format=response_format, **kwargs
            )
            for image in X
        ]

    async def _apredict(
        self, X, one: bool = True, context: t.Optional[t.List[str]] = None, **kwargs
    ):
        response_format = kwargs.pop('response_format', 'b64_json')
        if context:
            assert one, 'context only works with ``one=True``'
        if one:
            return await self._apredict_one(
                image=X, context=context, n=1, response_format=response_format, **kwargs
            )
        return [
            await self._apredict_one(
                image=image, n=1, response_format=response_format, **kwargs
            )
            for image in X
        ]


@dc.dataclass
class OpenAIAudioTranscription(OpenAI):
    """OpenAI audio transcription predictor.

    :param takes_context: Whether the model takes context into account.
    :param prompt: The prompt to guide the model's style. Should contain ``{context}``.
    """

    takes_context: bool = True
    prompt: str = ''

    def pre_create(self, db: Datalayer) -> None:
        super().pre_create(db)
        if isinstance(db.databackend, IbisDataBackend) and self.encoder is None:
            self.encoder = dtype('str')

    @retry
    def _predict_one(
        self, file: t.BinaryIO, context: t.Optional[t.List[str]] = None, **kwargs
    ):
        "Converts a file-like Audio recording to text."
        if context is not None:
            self.prompt = self.prompt.format(context='\n'.join(context))
        return self.syncClient.audio.transcriptions.create(
            file=file,
            model=self.identifier,
            prompt=self.prompt,
            **kwargs,
        ).text

    @retry
    async def _apredict_one(
        self, file: t.BinaryIO, context: t.Optional[t.List[str]] = None, **kwargs
    ):
        "Converts a file-like Audio recording to text."
        if context is not None:
            self.prompt = self.prompt.format(context='\n'.join(context))
        return (
            await self.asyncClient.audio.transcriptions.create(
                file=file,
                model=self.identifier,
                prompt=self.prompt,
                **kwargs,
            )
        ).text

    @retry
    def _predict_a_batch(self, files: t.List[t.BinaryIO], **kwargs):
        "Converts multiple file-like Audio recordings to text."
        resps = [
            self.syncClient.audio.transcriptions.create(
                file=file, model=self.identifier, **kwargs
            )
            for file in files
        ]
        return [resp.text for resp in resps]

    @retry
    async def _apredict_a_batch(self, files: t.List[t.BinaryIO], **kwargs):
        "Converts multiple file-like Audio recordings to text."
        resps = await asyncio.gather(
            *[
                self.asyncClient.audio.transcriptions.create(
                    file=file, model=self.identifier, **kwargs
                )
                for file in files
            ]
        )
        return [resp.text for resp in resps]

    def _predict(
        self, X, one: bool = True, context: t.Optional[t.List[str]] = None, **kwargs
    ):
        if context:
            assert one, 'context only works with ``one=True``'
        if one:
            return self._predict_one(X, context=context, **kwargs)
        out = []
        batch_size = kwargs.pop('batch_size', 10)
        for i in tqdm.tqdm(range(0, len(X), batch_size)):
            out.extend(self._predict_a_batch(X[i : i + batch_size], **kwargs))
        return out

    async def _apredict(
        self, X, one: bool = True, context: t.Optional[t.List[str]] = None, **kwargs
    ):
        if context:
            assert one, 'context only works with ``one=True``'
        if one:
            return await self._apredict_one(X, context=context, **kwargs)
        batch_size = kwargs.pop('batch_size', 10)
        list_of_lists = await asyncio.gather(
            *[
                self._apredict_a_batch(X[i : i + batch_size], **kwargs)
                for i in range(0, len(X), batch_size)
            ]
        )
        return list(itertools.chain(*list_of_lists))


@dc.dataclass
class OpenAIAudioTranslation(OpenAI):
    """OpenAI audio translation predictor.

    :param takes_context: Whether the model takes context into account.
    :param prompt: The prompt to guide the model's style. Should contain ``{context}``.
    """

    takes_context: bool = True
    prompt: str = ''

    def pre_create(self, db: Datalayer) -> None:
        super().pre_create(db)
        if isinstance(db.databackend, IbisDataBackend) and self.encoder is None:
            self.encoder = dtype('str')

    @retry
    def _predict_one(
        self, file: t.BinaryIO, context: t.Optional[t.List[str]] = None, **kwargs
    ):
        "Translates a file-like Audio recording to English."
        if context is not None:
            self.prompt = self.prompt.format(context='\n'.join(context))
        return (
            self.syncClient.audio.translations.create(
                file=file,
                model=self.identifier,
                prompt=self.prompt,
                **kwargs,
            )
        ).text

    @retry
    async def _apredict_one(
        self, file: t.BinaryIO, context: t.Optional[t.List[str]] = None, **kwargs
    ):
        "Translates a file-like Audio recording to English."
        if context is not None:
            self.prompt = self.prompt.format(context='\n'.join(context))
        return (
            await self.asyncClient.audio.translations.create(
                file=file,
                model=self.identifier,
                prompt=self.prompt,
                **kwargs,
            )
        ).text

    @retry
    def _predict_a_batch(self, files: t.List[t.BinaryIO], **kwargs):
        "Translates multiple file-like Audio recordings to English."
        resps = [
            self.syncClient.audio.translations.create(
                file=file, model=self.identifier, **kwargs
            )
            for file in files
        ]
        return [resp.text for resp in resps]

    @retry
    async def _apredict_a_batch(self, files: t.List[t.BinaryIO], **kwargs):
        "Translates multiple file-like Audio recordings to English."
        resps = await asyncio.gather(
            *[
                self.asyncClient.audio.translations.create(
                    file=file, model=self.identifier, **kwargs
                )
                for file in files
            ]
        )
        return [resp.text for resp in resps]

    def _predict(
        self, X, one: bool = True, context: t.Optional[t.List[str]] = None, **kwargs
    ):
        if context:
            assert one, 'context only works with ``one=True``'
        if one:
            return self._predict_one(X, context=context, **kwargs)
        out = []
        batch_size = kwargs.pop('batch_size', 10)
        for i in tqdm.tqdm(range(0, len(X), batch_size)):
            out.extend(self._predict_a_batch(X[i : i + batch_size], **kwargs))
        return out

    async def _apredict(
        self, X, one: bool = True, context: t.Optional[t.List[str]] = None, **kwargs
    ):
        if context:
            assert one, 'context only works with ``one=True``'
        if one:
            return await self._apredict_one(X, context=context, **kwargs)
        batch_size = kwargs.pop('batch_size', 10)
        list_of_lists = await asyncio.gather(
            *[
                self._apredict_a_batch(X[i : i + batch_size], **kwargs)
                for i in range(0, len(X), batch_size)
            ]
        )
        return list(itertools.chain(*list_of_lists))
