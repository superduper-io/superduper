import os
from pathlib import Path
from typing import Iterator

import pytest

SUPERDUPER_CONFIG = os.environ.get("SUPERDUPER_CONFIG", "test/configs/default.yaml")

os.environ["SUPERDUPER_CONFIG"] = SUPERDUPER_CONFIG


from superduper import superduper
from superduper.base.datalayer import Datalayer


@pytest.fixture
def db() -> Iterator[Datalayer]:
    db = superduper(force_apply=True)

    yield db
    db.drop(force=True, data=True)


@pytest.fixture(scope='session')
def image_url():
    path = Path(__file__).parent / 'material' / 'data' / '1x1.png'
    return f'file://{path}'
