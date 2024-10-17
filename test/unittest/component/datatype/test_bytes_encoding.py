from superduper.components.vector_index import sqlvector


def test_bytes_encoding_str():
    dt = sqlvector(shape=(3,))

    dt.intermediate_type = 'bytes'
    dt.bytes_encoding = 'str'
    encoded = dt.encode_data([1.1, 2.2, 3.3])
    assert isinstance(encoded, str)
