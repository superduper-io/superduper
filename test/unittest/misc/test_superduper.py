import pytest

try:
    import torch
except ImportError:
    torch = None

from superduperdb import superduper
from superduperdb.misc.superduper import SklearnTyper, TorchTyper


def test_sklearn_typer():
    from sklearn.linear_model import LinearRegression

    assert SklearnTyper.accept(LinearRegression()) is True


@pytest.mark.skipif(not torch, reason='Torch not installed')
def test_torch_typer():
    assert TorchTyper.accept(torch.nn.Linear(1, 1)) is True


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


# TODO: add MongoDbTyper test
