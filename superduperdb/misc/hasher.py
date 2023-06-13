import hashlib
import typing as t

if t.TYPE_CHECKING:
    from hashlib import _Hash as Hash
else:
    Hash = t.Any

HASHER = hashlib.sha256
BaseHashable = t.Union[bytearray, bytes, memoryview, str]
Hashable = t.Union[BaseHashable, t.Iterator[BaseHashable]]


def hash_all(x: Hashable, hasher: t.Optional[Hash] = None) -> Hash:
    """Hashes either a single Hashable, or an iterator of Hashable"""
    if hasher is None:
        hasher = HASHER()

    if isinstance(x, BaseHashable.__args__):  # type: ignore
        x = iter([x])

    for i in x:
        if isinstance(i, str):
            i = i.encode()
        elif not isinstance(i, t.ByteString):
            raise TypeError(f'Expecting str or ByteString, got {i}')
        hasher.update(i)

    return hasher


def hash_hashes(it: t.Iterator[Hashable], hasher: t.Optional[Hash] = None) -> Hash:
    """Hashes together the hashes of each item in an iterable"""
    return hash_all((hash_all(i).digest() for i in it), hasher)
