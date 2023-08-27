from __future__ import annotations

import pytest

try:
    import torch
except ImportError:
    torch = None

from superduperdb import superduper
from superduperdb.misc.superduper import MongoDbTyper, SklearnTyper, TorchTyper


@pytest.mark.skipif(not torch, reason='Torch not installed')
def test_mongodb_typer(test_db):
    assert MongoDbTyper.accept(test_db.db) is True


def test_sklearn_typer():
    from sklearn.linear_model import LinearRegression

    assert SklearnTyper.accept(LinearRegression()) is True


@pytest.mark.skipif(not torch, reason='Torch not installed')
def test_torch_typer():
    assert TorchTyper.accept(torch.nn.Linear(1, 1)) is True


def test_superduper_db(test_db):
    db = superduper(test_db.db)
    assert db.db == test_db.db


@pytest.mark.skipif(not torch, reason='Torch not installed')
def test_superduper_model():
    from sklearn.linear_model import LinearRegression

    model = superduper(torch.nn.Linear(1, 1))
    assert isinstance(model.object.artifact, torch.nn.modules.linear.Linear)
    model = superduper(LinearRegression())
    assert isinstance(model.object.artifact, LinearRegression)


def test_superduper_raise():
    with pytest.raises(NotImplementedError):
        superduper(1)

    with pytest.raises(NotImplementedError):
        superduper("string")
