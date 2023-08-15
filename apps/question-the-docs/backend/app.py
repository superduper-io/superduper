from apps.question-the-docs.backend.ai.components import setup_ai
from apps.question-the-docs.backend.ai.data import setup_data
from backend.config import settings
from backend.document.routes import document_router
from backend.ai.data import setup_qa_documentation
from fastapi import FastAPI
from pymongo import MongoClient

from superduperdb.misc import superduper
from backend.ai.setup import setup as setup_ai_and_data


def init_routers(app: FastAPI) -> None:
    app.include_router(document_router)


def create_app() -> FastAPI:
    _app = FastAPI(title="KPI Dashboards")

    @_app.on_event("startup")
    def startup_db_client():
        _app.mongodb_client = MongoClient(settings.DB_URL)
        _app.superduperdb = superduper(_app.mongodb_client[settings.MONGO_DB_NAME])

        _app.superduperdb.drop(force=True)
        setup_data(_app.superduperdb)
        setup_ai(_app.superduperdb)

    @_app.on_event("shutdown")
    def shutdown_db_client():
        _app.mongodb_client.close()

    init_routers(app=_app)
    return _app
