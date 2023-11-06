
import uvicorn
from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from superduperdb.base.build import build_datalayer
from superduperdb.base.datalayer import Datalayer

app = FastAPI()


def create_datalayer() -> Datalayer:
    return build_datalayer()


@app.on_event('startup')
def startup_db_client():
    db = create_datalayer()
    app.state.pool = db


@app.on_event('shutdown')
def shutdown_db_client():
    try:
        app.state.pool.close()
    except AttributeError:
        raise Exception('Could not close the database properly')


# --------------- Create exception handler middleware-----------------


class ExceptionHandlerMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        try:
            return await call_next(request)
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={'error': e.__class__.__name__, 'messages': e.args},
            )


app.add_middleware(ExceptionHandlerMiddleware)


class SuperDuperApp:
    def __init__(self, service='vector_search', host='0.0.0.0', port=8000):
        self.service = service
        self.host = host
        self.port = port

        self.router = APIRouter(prefix='/' + self.service)

    def add(self, *args, method='post', **kwargs):
        def decorator(function):
            self.router.add_api_route(
                *args, **kwargs, endpoint=function, methods=[method]
            )
            return

        return decorator

    def start(self):
        app.include_router(self.router)

        uvicorn.run(
            app,
            host=self.host,
            port=self.port,
            reload=False,
        )


def database(request: Request) -> Datalayer:
    return request.app.state.pool


def DatalayerDependency():
    return Depends(database)
