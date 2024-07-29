import os
from pathlib import Path
from typing import Iterator

SUPERDUPER_CONFIG = os.environ.get("SUPERDUPER_CONFIG", "test/configs/default.yaml")

os.environ["SUPERDUPER_CONFIG"] = SUPERDUPER_CONFIG

import pytest

from superduper import superduper
from superduper.base.datalayer import Datalayer
from superduper.base.enums import DBType


@pytest.fixture
def db() -> Iterator[Datalayer]:
    db = superduper()

    yield db
    db_type = db.databackend.db_type
    if db_type == DBType.MONGODB:
        db.drop(force=True, data=True)
    elif db_type == DBType.SQL:
        db.artifact_store.drop(force=True)
        tables = db.databackend.conn.list_tables()
        for table in tables:
            db.databackend.conn.drop_table(table, force=True)


@pytest.fixture(scope='session')
def image_url():
    path = Path(__file__).parent / 'material' / 'data' / '1x1.png'
    return f'file://{path}'
