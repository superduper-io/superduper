from .registry import NotJSONableError, Registry
from enum import Enum
from fastapi import FastAPI, HTTPException, Response, UploadFile
from fastapi.responses import PlainTextResponse
from functools import cached_property
from http import HTTPStatus as status
from uvicorn import run
import dataclasses as dc
import fastapi
import superduperdb as s
import typing as t

MEDIA_TYPE = 'application/octet-stream'


@dc.dataclass
class Server:
    cfg: s.config.Server = dc.field(default_factory=s.CFG.server.deepcopy)
    stats: t.Dict = dc.field(default_factory=dict)
    artifact_store: t.Dict = dc.field(default_factory=dict)
    # TODO: this is a non-thread-safe proof of concept

    @cached_property
    def app(self) -> FastAPI:
        return FastAPI(**self.cfg.fastapi.dict())

    @cached_property
    def registry(self) -> Registry:
        return Registry()

    def register(self, *methods) -> t.Optional[t.Callable]:
        result = None
        for result in methods:
            self.registry.register(result)
        return result

    def run(self, obj: t.Any) -> t.NoReturn:
        self.add_endpoints(obj)
        run(self.app, **self.cfg.web_server.dict())
        assert False, 'We never get here'

    def auto_register(self, obj: t.Any) -> None:
        for k, v in vars(obj.__class__).items():
            if callable(v) and k.islower() and not k.startswith('_'):
                try:
                    self.registry.register(v)
                except NotJSONableError:
                    pass

    def auto_run(self, obj: t.Any) -> t.NoReturn:
        self.auto_register(obj)
        self.run(obj)

    def add_endpoints(self, obj: t.Any):
        count = 0

        @self.app.get('/', response_class=PlainTextResponse)
        def root() -> str:
            return self.cfg.fastapi.title

        count += 1

        @self.app.get('/health', response_class=PlainTextResponse)
        def health() -> str:
            return 'ok'

        count += 1

        @self.app.get('/stats')
        def stats() -> t.Dict[str, int]:
            return self.stats

        count += 1

        @self.app.get('/download/{artifact_key}', response_class=Response)
        def download(artifact_key: str) -> Response:
            if not (content := self.artifact_store.get(artifact_key)):
                print(self.artifact_store, artifact_key)
                raise HTTPException(
                    status.NOT_FOUND, f'Unknown artifact {artifact_key}'
                )

            elif not isinstance(content, bytes):
                if not callable(encode := getattr(content, 'encode', None)):
                    msg = f'Uncodable artifact {artifact_key}'
                    raise HTTPException(status.NOT_FOUND, msg)

                content = encode()

            return fastapi.Response(
                content=content, status_code=status.OK, media_type=MEDIA_TYPE
            )

        count += 1

        @self.app.post('/upload/{artifact_key}')
        async def upload(file: UploadFile):
            exists = file.filename in self.artifact_store
            self.artifact_store[file.filename] = await file.read()

            action = 'replaced' if exists else 'created'
            return {action: file.filename}

        count += 1

        for method_name in self.registry.entries:
            method = getattr(obj, method_name)
            self.app.post('/' + method_name)(method)

            count += 1

        # TODO: unfortunately, mypy won't accept these computed types
        # even though FastAPI has no issue with them.

        methods = sorted((e, e) for e in self.registry.entries)
        Method = Enum('Method', methods)  # type: ignore
        Parameter = self.registry.Parameter
        Result = self.registry.Result

        @self.app.post('/execute')
        def execute(
            method: Method,  # type: ignore
            args: t.List[Parameter] = (),  # type: ignore
        ) -> Result:  # type: ignore
            return self.registry.execute(obj, method.value, args)

        count += 1

        print(count, 'endpoints added')
