from typing import Iterator

import pytest
from superduper import superduper
from superduper.base.datalayer import Datalayer


@pytest.fixture
def db() -> Iterator[Datalayer]:
    db = superduper()

    yield db
    db.drop(force=True, data=True)
