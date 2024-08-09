import pytest

try:
    import torch
    from superduper.ext.torch.utils import device_of, eval, set_device, to_device
except ImportError:
    torch = None


@pytest.fixture
def model():
    return torch.nn.Linear(10, 2)


@pytest.mark.skipif(not torch, reason='Torch not installed')
def test_device_of_cpu(model):
    device = device_of(model)
    assert device.type == 'cpu'


@pytest.mark.skipif(not torch, reason='Torch not installed')
def test_device_of_cuda(model):
    if torch.cuda.is_available():
        model.to(torch.device('cuda'))
        device = device_of(model)
        assert device.type == 'cuda'


@pytest.mark.skipif(not torch, reason='Torch not installed')
def test_eval_context_manager(model):
    with eval(model):
        assert not model.training


@pytest.mark.skipif(not torch, reason='Torch not installed')
def test_set_device_context_manager(model):
    device_before = device_of(model)
    if torch.cuda.is_available():
        with set_device(model, torch.device('cuda', 0)):
            device_after = device_of(model)
            assert device_after.type == 'cuda'
        assert device_of(model) == device_before


@pytest.mark.skipif(not torch, reason='Torch not installed')
def test_to_device_tensor(model):
    tensor = torch.tensor([1, 2, 3])
    device = (
        torch.device('cuda', 0) if torch.cuda.is_available() else torch.device('cpu')
    )
    tensor_device = to_device(tensor, device)
    assert tensor_device.device == device


@pytest.mark.skipif(not torch, reason='Torch not installed')
def test_to_device_nested_list(model):
    nested_list = [
        torch.tensor([1, 2, 3]),
        [torch.tensor([4, 5]), torch.tensor([6, 7])],
    ]
    device = (
        torch.device('cuda', 0) if torch.cuda.is_available() else torch.device('cpu')
    )
    nested_list_device = to_device(nested_list, device)
    for item in nested_list_device:
        if isinstance(item, list):
            assert all(i.device == device for i in item)
        else:
            assert item.device == device


@pytest.mark.skipif(not torch, reason='Torch not installed')
def test_to_device_nested_dict(model):
    nested_dict = {'a': torch.tensor([1, 2, 3]), 'b': {'c': torch.tensor([4, 5])}}
    device = (
        torch.device('cuda', 0) if torch.cuda.is_available() else torch.device('cpu')
    )
    nested_dict_device = to_device(nested_dict, device)
    for item in nested_dict_device.values():
        if isinstance(item, dict):
            for i in item.values():
                assert i.device == device
        else:
            assert item.device == device
