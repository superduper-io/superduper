import torch

from superduperdb.models.torch.wrapper import TorchPipeline


class ToDict:
    def __init__(self):
        self.dict = dict(zip(list('abcdefghiklmnopqrstuvwyz'), range(26)))

    def __call__(self, input: str):
        return [self.dict[k] for k in input]


class TensorLookup:
    def __init__(self):
        self.d = torch.randn(26, 32)

    def __call__(self, x):
        return torch.stack([self.d[y] for y in x])


def pad_to_ten(x):
    to_stack = []
    for i, y in enumerate(x):
        out = torch.zeros(10, 32)
        y = y[:10]
        out[: y.shape[0], :] = y
        to_stack.append(out)
    return torch.stack(to_stack)


def test_pipeline():
    pl = TorchPipeline(
        'my-pipeline',
        [
            ('encode', ToDict()),
            ('lookup', TensorLookup()),
            ('forward', torch.nn.Linear(32, 256)),
            ('top1', lambda x: x.topk(1)[1]),
        ],
        collate_fn=pad_to_ten,
    )

    out = pl.predict('bla')

    print(out)

    assert isinstance(out, torch.Tensor)

    out = pl.predict(['bla', 'testing'], batch_size=2)

    assert isinstance(out, list)
