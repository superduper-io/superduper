from test.unittest.component.datatype import utils

import numpy as np
import pandas as pd
import pytest

from superduper.base.enums import DBType
from superduper.components.datatype import (
    DataType,
    pickle_encoder,
    pickle_lazy,
    pickle_serializer,
)


@pytest.fixture
def random_data():
    array = np.array([1])
    return array


datatypes = [
    pickle_encoder,
    pickle_serializer,
    pickle_lazy,
]


@pytest.mark.parametrize("datatype", datatypes)
def test_data_with_schema(datatype: DataType, random_data: pd.DataFrame):
    utils.check_data_with_schema(random_data, datatype)


@pytest.mark.parametrize("datatype", datatypes)
def test_data_with_schema_and_db(datatype: DataType, random_data: pd.DataFrame, db):
    utils.check_data_with_schema_and_db(random_data, datatype, db)


@pytest.mark.parametrize("datatype", datatypes)
def test_data_without_schema(datatype: DataType, random_data: pd.DataFrame):
    utils.check_data_without_schema(random_data, datatype)


@pytest.mark.parametrize("datatype", datatypes)
def test_data_without_schema_and_db(datatype: DataType, random_data: pd.DataFrame, db):
    utils.check_data_without_schema_and_db(random_data, datatype, db)


@pytest.mark.parametrize("datatype", datatypes)
def test_component(random_data, datatype):
    utils.check_component(random_data, datatype)


@pytest.mark.parametrize("datatype", datatypes)
def test_component_with_db(db, random_data, datatype):
    # TODO: Need to fix the encodable in component when db is SQL
    # Some bytes are not serializable, then can't be stored in SQL
    if datatype.encodable == "encodable" and db.databackend.db_type == DBType.SQL:
        return
    utils.check_component_with_db(random_data, datatype, db)
