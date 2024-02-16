from superduperdb.components.model import ObjectModel


def test_auto():
    import torch as torch_native

    from superduperdb.ext.auto import torch

    m = torch.nn.Linear(2, 4, identifier='my-linear')

    assert isinstance(m, ObjectModel)
    assert isinstance(m.object, torch_native.nn.Module)
