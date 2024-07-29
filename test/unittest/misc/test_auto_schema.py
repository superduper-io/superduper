import threading

import numpy as np
import pandas as pd
import PIL
import pytest
import torch

from superduper.base.document import Document
from superduper.base.exceptions import UnsupportedDatatype
from superduper.components.table import Table
from superduper.misc.auto_schema import infer_datatype, infer_schema


@pytest.fixture
def data():
    data = {
        "int": 1,
        "float": 1.0,
        "str": "1",
        "bool": True,
        "bytes": b"1",
        "np_array": np.array([1, 2, 3]),
        "torch_tensor": torch.tensor([1, 2, 3]),
        "dict": {"a": 1, "b": "2", "c": {"d": 3}},
        "dict_np_array": {"a": np.array([1, 2, 3])},
        "df": pd.DataFrame({"col1": [1, 2], "col2": [3, 4]}),
    }
    return data


def test_infer_datatype():
    from superduper.ext import pillow as pillow_ext, torch as torch_ext

    print(torch_ext, pillow_ext)
    assert infer_datatype(1) is int
    assert infer_datatype(1.0) is float
    assert infer_datatype("1") is str
    assert infer_datatype(True) is bool
    assert infer_datatype(b"1") is bytes

    assert infer_datatype(np.array([1, 2, 3])).identifier == "numpy-int64[3]"
    assert infer_datatype(torch.tensor([1, 2, 3])).identifier == "torch-int64[3]"

    assert infer_datatype({"a": 1}).identifier == "json"

    assert infer_datatype({"a": np.array([1, 2, 3])}).identifier == "DEFAULT"

    assert (
        infer_datatype(pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})).identifier
        == "DEFAULT"
    )

    assert (
        infer_datatype(PIL.Image.open('test/material/data/test.png')).identifier
        == 'pil_image'
    )

    with pytest.raises(UnsupportedDatatype):
        thread = threading.Thread(target=lambda x: x)
        infer_datatype(thread)


def test_infer_schema_mongo(data):
    schema = infer_schema(data)
    encode_data = Document(data).encode(schema)

    expected_datatypes = [
        "np_array",
        "torch_tensor",
        "dict_np_array",
        "df",
    ]
    assert sorted(schema.encoded_types) == sorted(expected_datatypes)
    for key in expected_datatypes:
        if schema.fields.get(key) is not None:
            assert str(data[key]) != str(encode_data[key])

    decode_data = Document.decode(encode_data, schema).unpack()
    for key in data:
        assert isinstance(data[key], type(decode_data[key]))
        assert str(data[key]) == str(decode_data[key])


def test_infer_schema_ibis(data):
    schema = infer_schema(data, ibis=True)
    encode_data = Document(data).encode(schema)

    expected_datatypes = [
        "np_array",
        "torch_tensor",
        "dict",
        "dict_np_array",
        "df",
    ]
    assert sorted(schema.encoded_types) == sorted(expected_datatypes)
    for key in expected_datatypes:
        if schema.fields.get(key) is not None:
            assert str(data[key]) != str(encode_data[key])

    decode_data = Document.decode(encode_data, schema).unpack()
    for key in data:
        assert isinstance(data[key], type(decode_data[key]))
        assert str(data[key]) == str(decode_data[key])


def test_schema(db, data):
    schema = db.infer_schema(data)

    t = Table(identifier="my_table", schema=schema)

    db.apply(t)

    db.execute(db["my_table"].insert([Document(data)]))

    select = db["my_table"].select().limit(1)
    decode_data = db.execute(select).next().unpack()
    for key in data:
        assert isinstance(data[key], type(decode_data[key]))
        assert str(data[key]) == str(decode_data[key])
