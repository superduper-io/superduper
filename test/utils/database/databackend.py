from superduper.backends.base.data_backend import BaseDataBackend
from superduper.components.schema import Schema


def test_list_tables_or_collections(databackend: BaseDataBackend):
    fields = {
        'a': int,
    }

    for i in range(10):
        table_name = f"table_{i}"
        databackend.create_table_and_schema(
            table_name, schema=Schema(identifier="schema", fields=fields)
        )

    tables = databackend.list_tables()
    assert len(tables) == 10
    assert [f"table_{i}" for i in range(10)] == sorted(tables)
