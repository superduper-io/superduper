import json
import sys
import threading
import time
from functools import cached_property
from traceback import format_exc

import uvicorn
from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from prettytable import PrettyTable
from starlette.middleware.base import BaseHTTPMiddleware

from superduperdb import logging
from superduperdb.base.build import build_datalayer
from superduperdb.base.datalayer import Datalayer

# --------------- Create exception handler middleware-----------------


class ExceptionHandlerMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        try:
            return await call_next(request)
        except Exception as e:
            host = getattr(getattr(request, "client", None), "host", None)
            port = getattr(getattr(request, "client", None), "port", None)
            url = (
                f"{request.url.path}?{request.query_params}"
                if request.query_params
                else request.url.path
            )
            exception_type, exception_value, _ = sys.exc_info()
            exception_traceback = format_exc()
            exception_name = getattr(exception_type, "__name__", None)
            msg = f'{host}:{port} - "{request.method} {url}"\
                    500 Internal Server Error <{exception_name}:\
                    {exception_value}>'
            logging.exception(msg, e=e)
            return JSONResponse(
                status_code=500,
                content={
                    'error': exception_name,
                    'messages': msg,
                    'traceback': exception_traceback,
                },
            )


class SuperDuperApp:
    """
    This is a wrapper class that prepares helper functions used to create a
    fastapi application in the realm of superduperdb.
    """

    def __init__(self, service='vector_search', port=8000):
        self.service = service
        self.port = port

        self.app_host = '0.0.0.0'
        self._app = FastAPI()

        self.router = APIRouter()
        self._user_startup = False
        self._user_shutdown = False

        self._app.add_middleware(ExceptionHandlerMiddleware)

    @cached_property
    def app(self):
        self._app.include_router(self.router)
        return self._app

    @cached_property
    def db(self):
        return self._app.state.pool

    def add(self, *args, method='post', **kwargs):
        """
        Register an endpoint with this method.
        """

        def decorator(function):
            self.router.add_api_route(
                *args, **kwargs, endpoint=function, methods=[method]
            )

        return decorator

    def add_default_endpoints(self):
        """
        Add a list of default endpoints which comes out of the box with `SuperDuperApp`
        """

        @self.router.get('/health')
        def health():
            return {'status': 200}

        @self.router.post('/handshake/config')
        def handshake(cfg: str):
            from superduperdb import CFG

            cfg_dict = json.loads(cfg)
            if CFG.match(cfg_dict):
                return {'status': 200, 'msg': 'matched'}

            return JSONResponse(
                status_code=400,
                content={'error': 'Config is not match'},
            )

    def print_routes(self):
        table = PrettyTable()

        # Define the table headers
        table.field_names = ["Path", "Methods", "Function"]

        # Add rows to the table
        for route in self._app.routes:
            table.add_row([route.path, ", ".join(route.methods), route.name])

        logging.info(f"Routes for '{self.service}' app: \n{table}")

    def pre_start(self, cfg=None):
        self.add_default_endpoints()

        if not self._user_startup:
            self.startup(cfg=cfg)
        if not self._user_shutdown:
            self.shutdown()
        assert self.app

    def start(self):
        """
        This method is used to start the application server
        """
        self.pre_start()

        self.print_routes()

        uvicorn.run(
            self._app,
            host=self.app_host,
            port=self.port,
            reload=False,
        )

    def startup(self, function=None, cfg=None):
        """
        This method is used to register a startup function
        """
        self._user_startup = True

        @self._app.on_event('startup')
        def startup_db_client():
            sys.path.append('./')
            db = build_datalayer(cfg)
            db.server_mode = True
            if function:
                function(db=db)
            self._app.state.pool = db

        return

    def shutdown(self, function=None):
        """
        This method is used to register a shutdown function
        """
        self._user_shutdown = True

        @self._app.on_event('shutdown')
        def shutdown_db_client():
            try:
                if function:
                    function(db=self._app.state.pool)
                self._app.state.pool.close()
            except AttributeError:
                raise Exception('Could not close the database properly')


def database(request: Request) -> Datalayer:
    return request.app.state.pool


def DatalayerDependency():
    """
    A helper method to be used for injecting datalayer instance
    into endpoint implementation
    """
    return Depends(database)


class Server(uvicorn.Server):
    def install_signal_handlers(self):
        pass

    def run_in_thread(self):
        self._thread = threading.Thread(target=self.run)
        self._thread.start()

        while not self.started:
            time.sleep(1e-3)

    def stop(self):
        self.should_exit = True
        self._thread.join()
