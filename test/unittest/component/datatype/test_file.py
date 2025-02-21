import os
from test.utils.component import datatype as datatype_utils

import pytest

from superduper.base.datatype import file


@pytest.fixture
def random_data(tmpdir):
    file_name = os.path.join(tmpdir, "HelloWorld.txt")

    with open(file_name, "w") as f:
        f.write("Hello, World!")

    return file_name


def test_data_with_schema(db, random_data):
    datatype_utils.check_data_with_schema(random_data, file, db=db)


def test_data_with_schema_and_db(random_data, db):
    datatype_utils.check_data_with_schema_and_db(random_data, file, db)


def test_component(random_data):
    datatype_utils.check_component(random_data, file)


def test_component_with_db(db, random_data):
    datatype_utils.check_component_with_db(random_data, file, db)
