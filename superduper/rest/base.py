import os
import sys
import threading
import time
import typing as t
from functools import cached_property
from traceback import format_exc

import uvicorn
from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.exceptions import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prettytable import PrettyTable
from starlette.middleware.base import BaseHTTPMiddleware

from superduper import logging
from superduper.base.build import build_datalayer
from superduper.base.config import Config
from superduper.base.datalayer import Datalayer

# --------------- Create exception handler middleware-----------------


class ExceptionHandlerMiddleware(BaseHTTPMiddleware):
    """Middleware to handle exceptions and log them."""

    async def dispatch(self, request: Request, call_next):
        """Dispatch the request and handle exceptions.

        :param request: request to dispatch
        :param call_next: next call to make
        """
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
    """A wrapper class for creating a fastapi application.

    The class provides a simple interface for creating a fastapi application
    with custom endpoints.

    :param service: name of the service
    :param port: port to run the service on
    :param db: datalayer instance
    """

    def __init__(
        self,
        service='rest',
        port=8000,
        db: Datalayer = None,
        prefix: str = '',
        data_backend: str | None = None,
        templates: t.List[str] | None = None,
    ):
        if prefix and not prefix.startswith('/'):
            prefix = f'/{prefix}'

        self.service = service
        self.prefix = prefix

        self.port = port

        self.app_host = '0.0.0.0'
        self._app = FastAPI(
            root_path=prefix,
        )

        self.router = APIRouter()

        self._app.add_middleware(ExceptionHandlerMiddleware)
        self._app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # You can specify a list of allowed origins here
            allow_credentials=True,
            allow_methods=[
                "GET",
                "POST",
                "PUT",
                "DELETE",
            ],  # You can adjust these as per your needs
            allow_headers=["*"],  # You can specify allowed headers here
        )
        self._db = db
        self.data_backend = data_backend
        self.templates = templates

    @cached_property
    def app(self):
        """Return the application instance."""
        self._app.include_router(self.router)
        return self._app

    def raise_error(self, msg: str, code: int):
        """Raise an error with the given message and code.

        :param msg: message to raise
        :param code: code to raise
        """
        raise HTTPException(code, detail=msg)

    @cached_property
    def db(self) -> Datalayer:
        """Return the database instance from the app state."""
        return self._app.state.pool

    def add(self, *args, method='post', **kwargs):
        """Register an endpoint with this method.

        :param method: method to use
        """

        def decorator(function):
            self.router.add_api_route(
                *args, **kwargs, endpoint=function, methods=[method]
            )

        return decorator

    def add_default_endpoints(self):
        """Add default endpoints to the application.

        - /health: Health check endpoint
        - /handshake/config: Handshake endpoint
        """
        logging.info(f"Adding default endpoints to '{self.service}' app")

        @self.add('/health', method='get')
        def health():
            return {'status': 200}

    def print_routes(self):
        """Print the routes of the application."""
        table = PrettyTable()

        # Define the table headers
        table.field_names = ["Path", "Methods", "Function"]

        # Add rows to the table
        for route in self._app.routes:
            try:
                table.add_row([route.path, ", ".join(route.methods), route.name])
            except AttributeError:
                logging.warn(f"Route {route} has no name")

        logging.info(f"Routes for '{self.service}' app: \n{table}")

    def pre_start(self, cfg: t.Union[Config, None] = None):
        """Pre-start the application.

        :param cfg: Configurations to use
        """
        self.add_default_endpoints()
        self.startup(cfg=cfg)
        assert self.app

    def run(self):
        """Run the application."""
        uvicorn.run(
            self._app,
            host=self.app_host,
            port=self.port,
        )

    def start(self):
        """Start the application."""
        self.pre_start()
        self.print_routes()
        self.run()

    def _add_templates(self, db):
        if self.templates:
            from superduper import templates

            existing = db.show('template')
            for t in self.templates:
                logging.info(f'Applying template: {t}')

                if t is None:
                    continue

                if os.path.exists(t):
                    from superduper import Template

                    t = Template.read(t)
                else:
                    t = templates.get(t)

                if t.identifier in existing:
                    logging.warn(f'Template {t.identifier} already applied')
                    continue
                db.apply(t, force=True)

    def startup(
        self,
        cfg: t.Union[Config, None] = None,
    ):
        """Startup the application.

        :param cfg: Configurations to use
        """

        @self._app.on_event('startup')
        def startup_db_client():
            sys.path.append('./')
            if self._db is None:
                if self.data_backend:
                    db = build_datalayer(
                        cfg,
                        data_backend=self.data_backend,
                    )
                else:
                    db = build_datalayer(cfg)
            else:
                db = self._db

            self._add_templates(db)

            self._app.state.pool = db
            self._app.state.pool.cluster.initialize()
            self._db = db

        return

    def shutdown(self, function: t.Union[t.Callable, None] = None):
        """Shutdown the application.

        :param function: function to run on shutdown
        """

        @self._app.on_event('shutdown')
        def shutdown_db_client():
            try:
                self._app.state.pool.close()
            except AttributeError:
                raise Exception('Could not close the database properly')


def database(request: Request) -> Datalayer:
    """Return the database instance from the app state.

    :param request: request object
    """
    return request.app.state.pool


def DatalayerDependency():
    """Dependency for injecting datalayer instance into endpoint implementation."""
    return Depends(database)


class Server(uvicorn.Server):
    """Custom server class."""

    def install_signal_handlers(self):
        """Install signal handlers."""
        pass

    def run_in_thread(self):
        """Run the server in a separate thread."""
        self._thread = threading.Thread(target=self.run)
        self._thread.start()

        while not self.started:
            time.sleep(1e-3)

    def stop(self):
        """Stop the server."""
        self.should_exit = True
        self._thread.join()
