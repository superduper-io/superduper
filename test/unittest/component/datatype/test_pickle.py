from test.utils.component import datatype as datatype_utils

import numpy as np
import pandas as pd
import pytest

from superduper.base.datatype import (
    BaseDataType,
    pickle_encoder,
    pickle_serializer,
)


@pytest.fixture
def random_data():
    array = np.array([1])
    return array


datatypes = [
    pickle_encoder,
    pickle_serializer,
]


@pytest.mark.parametrize("datatype", datatypes)
def test_data_with_schema(db, datatype: BaseDataType, random_data: pd.DataFrame):
    datatype_utils.check_data_with_schema(random_data, datatype, db)


@pytest.mark.parametrize("datatype", datatypes)
def test_data_with_schema_and_db(datatype: BaseDataType, random_data: pd.DataFrame, db):
    datatype_utils.check_data_with_schema_and_db(random_data, datatype, db)


@pytest.mark.parametrize("datatype", datatypes)
def test_component(random_data, datatype):
    datatype_utils.check_component(random_data, datatype)


@pytest.mark.parametrize("datatype", datatypes)
def test_component_with_db(db, random_data, datatype):
    # TODO: Need to fix the encodable in component when db is SQL
    # Some bytes are not serializable, then can't be stored in SQL
    # if datatype.encodable == "encodable" and db.databackend.db_type == DBType.SQL:
    #     return
    datatype_utils.check_component_with_db(random_data, datatype, db)
