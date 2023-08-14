from backend.config import settings
from backend.document.routes import document_router
from fastapi import FastAPI
from pymongo import MongoClient

from .ai.bot import setup_qa_documentation


def init_routers(app: FastAPI) -> None:
    app.include_router(document_router)


def create_app() -> FastAPI:
    _app = FastAPI(title="KPI Dashboards")

    @_app.on_event("startup")
    def startup_db_client():
        _app.mongodb_client = MongoClient(settings.DB_URL)
        _client = _app.mongodb_client[settings.DB_NAME]
        _app.mongodb = _client

        setup_qa_documentation(_client)



    @_app.on_event("shutdown")
    def shutdown_db_client():
        _app.mongodb_client.close()

    init_routers(app=_app)
    return _app
