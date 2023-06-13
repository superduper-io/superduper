from superduperdb.misc.hasher import hash_all, hash_hashes
import pytest


def test_empty():
    expected = 'e3b0c44298fc1c149afbf4c8996fb924'
    assert hash_all([]).hexdigest()[:32] == expected
    assert hash_all(['']).hexdigest()[:32] == expected
    assert hash_hashes([]).hexdigest()[:32] == expected

    assert hash_hashes(['']).hexdigest()[:32] == '5df6e0e2761359d30a8275058e299fcc'


def test_space():
    assert hash_all([' ']).hexdigest()[:32] == '36a9e7f1c95b82ffb99743e0c5c4ce95'


def test_concat():
    expected = 'c2e2e7015634e96133f1ccf6a010105c'
    assert hash_all(['a sentence']).hexdigest()[:32] == expected
    assert hash_all(['a ', 'sentence']).hexdigest()[:32] == expected
    assert hash_all(['a', ' sentence']).hexdigest()[:32] == expected
    assert hash_all([b'a', ' sentence']).hexdigest()[:32] == expected
    assert hash_all('a sentence').hexdigest()[:32] == expected
    assert hash_all(b'a sentence').hexdigest()[:32] == expected


def test_hash_hashes():
    assert hash_hashes(['a sentence']).hexdigest()[:16] == '0f35d998685e4256'

    assert hash_hashes(['a ', 'sentence']).hexdigest()[:16] == '00f5b511d408c004'

    assert hash_hashes(['a', ' sentence']).hexdigest()[:16] == '10f4f34d4c6829a2'
    assert hash_hashes([b'a', ' sentence']).hexdigest()[:16] == '10f4f34d4c6829a2'

    assert hash_hashes('a sentence').hexdigest()[:16] == '0f7d36b24e30001e'

    with pytest.raises(TypeError) as e:
        hash_hashes(b'a sentence')
    assert e.value.args == ("'int' object is not iterable",)
