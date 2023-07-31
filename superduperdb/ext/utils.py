import typing as t


def str_shape(shape: t.Sequence[int]) -> str:
    return str(shape[0]) if len(shape) == 0 else 'x'.join([str(x) for x in shape])
