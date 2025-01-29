import threading

import numpy as np
import pandas as pd
import pytest

from superduper.base.document import Document
from superduper.base.exceptions import UnsupportedDatatype
from superduper.components.table import Table
from superduper.misc.auto_schema import infer_datatype


@pytest.fixture
def data():
    data = {
        "int": 1,
        "float": 1.0,
        "str": "1",
        "bool": True,
        "bytes": b"1",
        "np_array": np.array([1, 2, 3]),
        "dict": {"a": 1, "b": "2", "c": {"d": 3}},
        "dict_np_array": {"a": np.array([1, 2, 3])},
        "df": pd.DataFrame({"col1": [1, 2], "col2": [3, 4]}),
    }
    return data


def test_infer_datatype():
    assert infer_datatype(1) is int
    assert infer_datatype(1.0) is float
    assert infer_datatype("1") is str
    assert infer_datatype(True) is bool
    assert infer_datatype(b"1") is bytes

    assert infer_datatype(np.array([1, 2, 3])).identifier == "vector[3]"

    assert infer_datatype({"a": 1}).identifier == "json"

    assert infer_datatype({"a": np.array([1, 2, 3])}).identifier == "PickleEncoder"

    assert (
        infer_datatype(pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})).identifier
        == "PickleEncoder"
    )

    with pytest.raises(UnsupportedDatatype):
        thread = threading.Thread(target=lambda x: x)
        infer_datatype(thread)


def test_schema(db, data):
    schema = db.infer_schema(data)

    t = Table(identifier="my_table", schema=schema)

    db.apply(t)

    db["my_table"].insert([Document(data)])

    select = db["my_table"].select()
    decode_data = select.get().unpack()
    for key in data:
        assert isinstance(data[key], type(decode_data[key]))
        assert str(data[key]) == str(decode_data[key])
