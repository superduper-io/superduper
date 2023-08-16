from backend.ai.artifacts import load_ai_artifacts
from backend.ai.components import install_ai_components
from backend.config import settings
from backend.documents.routes import documents_router
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient

from superduperdb import superduper

# TODO: Fix before deployment
origins = [
    '*',
]


def init_routers(app: FastAPI) -> None:
    app.include_router(documents_router)


def create_app() -> FastAPI:
    _app = FastAPI(title='Question the Docs')

    _app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=['*'],
        allow_headers=['*'],
    )

    @_app.on_event('startup')
    def startup_db_client():
        _app.mongodb_client = MongoClient(settings.mongo_uri)
        _app.mongodb = _app.mongodb_client[settings.mongo_db_name]

        # We wrap our MongoDB to make it a SuperDuperDB!
        _app.superduperdb = superduper(_app.mongodb)

        # An Artifact is information that has been pre-processed
        # for use with AI models.
        load_ai_artifacts(_app.superduperdb)

        # A Component is an AI Model. Each Component can process
        # one or more types of Artifact.
        install_ai_components(_app.superduperdb)

    @_app.on_event('shutdown')
    def shutdown_db_client():
        _app.mongodb_client.close()

    init_routers(app=_app)
    return _app
