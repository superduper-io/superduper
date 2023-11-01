import itertools
import typing as t

T = t.TypeVar('T')


def ibatch(iterable: t.Iterable[T], batch_size: int) -> t.Iterator[t.List[T]]:
    """
    Batch an iterable into chunks of size `batch_size`

    :param iterable: the iterable to batch
    :param batch_size: the number of groups to write
    """

    iterator = iter(iterable)
    while True:
        batch = list(itertools.islice(iterator, batch_size))
        if not batch:
            break
        yield batch
