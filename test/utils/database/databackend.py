from superduper.backends.base.data_backend import BaseDataBackend
from superduper.base.datatype import FieldType
from superduper.base.schema import Schema


def test_list_tables_or_collections(databackend: BaseDataBackend):
    fields = {
        'a': FieldType('int'),
    }

    for i in range(10):
        table_name = f"table_{i}"
        databackend.create_table_and_schema(
            table_name, schema=Schema(fields=fields), primary_id='id'
        )

    tables = databackend.list_tables()
    assert len(tables) == 10
    assert [f"table_{i}" for i in range(10)] == sorted(tables)
