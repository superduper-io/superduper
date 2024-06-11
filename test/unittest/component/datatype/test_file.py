import os
from test.db_config import DBConfig
from test.unittest.component.datatype import utils

import pytest

from superduperdb import DataType


@pytest.fixture
def random_data(tmpdir):
    file_name = os.path.join(tmpdir, "HelloWorld.txt")

    with open(file_name, "w") as f:
        f.write("Hello, World!")

    return file_name


def dt_file():
    return DataType("my-file", encodable="file")


def dt_file_lazy():
    return DataType("my-file", encodable="lazy_file")


datatypes = [
    dt_file(),
    dt_file_lazy(),
]


@pytest.mark.parametrize("datatype", datatypes)
def test_data_with_schema(datatype: DataType, random_data):
    utils.check_data_with_schema(random_data, datatype)


@pytest.mark.parametrize("datatype", datatypes)
@pytest.mark.parametrize("db", DBConfig.EMPTY_CASES, indirect=True)
def test_data_with_schema_and_db(datatype: DataType, random_data, db):
    utils.check_data_with_schema_and_db(random_data, datatype, db)


@pytest.mark.parametrize("datatype", datatypes)
def test_data_without_schema(datatype: DataType, random_data):
    utils.check_data_without_schema(random_data, datatype)


@pytest.mark.parametrize("datatype", datatypes)
@pytest.mark.parametrize("db", DBConfig.EMPTY_CASES[:1], indirect=True)
def test_data_without_schema_and_db(datatype: DataType, random_data, db):
    utils.check_data_without_schema_and_db(random_data, datatype, db)


@pytest.mark.parametrize("datatype", datatypes)
def test_component(random_data, datatype):
    utils.check_component(random_data, datatype)


@pytest.mark.parametrize("datatype", datatypes)
@pytest.mark.parametrize("db", DBConfig.EMPTY_CASES, indirect=True)
def test_component_with_db(db, random_data, datatype):
    utils.check_component_with_db(random_data, datatype, db)
