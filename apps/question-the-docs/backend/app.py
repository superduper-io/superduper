from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient

from backend.ai.artifacts import load_ai_artifacts
from backend.ai.components import install_ai_components
from backend.config import settings
from backend.documents.routes import documents_router
from superduperdb import superduper


def create_app() -> FastAPI:
    app = FastAPI(title='Question the Docs')

    app.add_middleware(
        CORSMiddleware,
        allow_origins=['*'],
        allow_credentials=True,
        allow_methods=['*'],
        allow_headers=['*'],
    )

    @app.on_event('startup')
    def startup_db_client():
        app.mongodb_client = MongoClient(settings.mongo_uri)
        app.mongodb = app.mongodb_client[settings.mongo_db_name]

        # We wrap our MongoDB to make it a SuperDuperDB!
        app.superduperdb = superduper(app.mongodb)

        # An Artifact is information that has been pre-processed
        # for use with AI models.
        load_ai_artifacts(app.superduperdb)

        # A Component is an AI Model. Each Component can process
        # one or more types of Artifact.
        install_ai_components(app.superduperdb)

    @app.on_event('shutdown')
    def shutdown_db_client():
        app.mongodb_client.close()

    app.include_router(documents_router)
    return app
