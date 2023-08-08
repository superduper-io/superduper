import pytest

from superduperdb import superduper
from superduperdb.misc.superduper import MongoDbTyper, SklearnTyper, TorchTyper


def test_mongodb_typer(empty_database):
    assert MongoDbTyper.accept(empty_database.db) is True


def test_sklearn_typer():
    from sklearn.linear_model import LinearRegression

    assert SklearnTyper.accept(LinearRegression()) is True


def test_torch_typer():
    import torch

    assert TorchTyper.accept(torch.nn.Linear(1, 1)) is True


def test_superduper_db(empty_database):
    db = superduper(empty_database.db)
    assert db.db == empty_database.db


def test_superduper_model():
    import torch
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
