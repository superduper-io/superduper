from .registry import NotJSONableError, Registry
from enum import Enum
from fastapi import FastAPI, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse
from functools import cached_property
from http import HTTPStatus
from pathlib import Path
from superduperdb.misc.typed_cache import TypedCache
import dataclasses as dc
import superduperdb as s
import typing as t
import uvicorn

MEDIA_TYPE = 'application/octet-stream'
HERE = Path(__file__).parent
FAVICON = HERE / 'favicon.ico'
INDEX_HTML = Path(__file__).parent / 'index.html'
OK, NOT_FOUND = HTTPStatus.OK, HTTPStatus.NOT_FOUND


@dc.dataclass
class Server:
    cfg: s.config.Server = dc.field(default_factory=s.CFG.server.deepcopy)
    stats: t.Dict = dc.field(default_factory=dict)
    document_store: TypedCache = dc.field(default_factory=TypedCache)
    count: int = 0

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

    def run(self, obj: t.Any) -> None:
        d = self.cfg.web_server.dict()
        d.pop('protocol', None)
        uvicorn.run(self.app, **d)

    def auto_register(self, obj: t.Any) -> None:
        for k, v in vars(obj.__class__).items():
            if callable(v) and k.islower() and not k.startswith('_'):
                try:
                    self.registry.register(v)
                except NotJSONableError:
                    pass

    def add_endpoints(self, obj: t.Any):
        self.count = 0

        @self.app.get('/')
        def root():
            return FileResponse(INDEX_HTML)

        self.count += 1

        @self.app.get('/health')
        def health():
            return PlainTextResponse('ok')

        self.count += 1

        @self.app.get('/stats')
        def stats() -> t.Dict[str, t.Any]:
            return dict(self.stats, perhaps='redoc makes this pointless')

        self.count += 1

        @self.app.get('/favicon.ico', include_in_schema=False)
        def favicon():
            return FileResponse(FAVICON)

        @self.app.get('/download/{document_key}', response_class=Response)
        def download(document_key: str) -> Response:
            try:
                doc = self.document_store.get(document_key)
            except KeyError:
                raise HTTPException(NOT_FOUND, f'Unknown document {document_key}')

            try:
                content = doc.encode()
            except Exception:
                content = doc

            if not isinstance(content, bytes):
                raise HTTPException(NOT_FOUND, f'Uncodable document {document_key}')

            return Response(content=content, status_code=OK, media_type=MEDIA_TYPE)

        self.count += 1

        @self.app.post('/upload/{document_key}')
        async def upload(file: UploadFile):
            exists = file.filename in self.document_store
            data = await file.read()
            self.document_store.put(data, file.filename)

            action = 'replaced' if exists else 'created'
            return {action: file.filename}

        self.count += 1

        for method_name in self.registry.entries:
            method = getattr(obj, method_name)
            self.app.post('/' + method_name)(method)

            self.count += 1

    def add_execute(self, obj: t.Any):
        # TODO: unfortunately, mypy won't accept these computed types
        # even though FastAPI has no issue with them.

        methods = sorted((e, e) for e in self.registry.entries)
        Method = Enum('Method', methods)  # type: ignore
        Args = t.List[self.registry.Parameter]  # type: ignore
        Result = self.registry.Result

        @self.app.post('/execute')
        def execute(method: Method, args: Args = ()) -> Result:  # type: ignore
            return self.registry.execute(obj, method.value, args)

        self.count += 1
        print(self.count, 'endpoints added')
