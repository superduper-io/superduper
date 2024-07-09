import pytest

from superduper.misc.anonymize import anonymize_url

params = [
    (
        "mongodb://localhost:27017/test_db",
        "mongodb://localhost:27017/test_db",
    ),
    (
        "mongodb://user:password@localhost:27017/test_db",
        "mongodb://us******er:pa******rd@localhost:27017/test_db",
    ),
    (
        "postgresql://localhost:5432/test_db",
        "postgresql://localhost:5432/test_db",
    ),
    (
        "postgresql://user:password@localhost:5432/test_db",
        "postgresql://us******er:pa******rd@localhost:5432/test_db",
    ),
    (
        "postgresql://user:@localhost:5432/test_db",
        "postgresql://us******er:@localhost:5432/test_db",
    ),
    (
        None,
        None,
    ),
    (
        "",
        "",
    ),
]


@pytest.mark.parametrize("input_url, expected", params)
def test_anonymize_url(input_url, expected):
    assert anonymize_url(input_url) == expected
