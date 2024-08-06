from superduper import CFG
from superduper.backends.base.data_backend import BaseDataBackend
from superduper.backends.base.query import Query
from superduper.components.datatype import pickle_serializer
from superduper.components.schema import Schema


def test_output_dest(databackend: BaseDataBackend):
    assert isinstance(databackend, BaseDataBackend)
    # Create an output destination for the database
    predict_id = "predict_id"

    assert not databackend.check_output_dest(predict_id)

    table = databackend.create_output_dest(predict_id, pickle_serializer)

    assert table.identifier.startswith(CFG.output_prefix)

    databackend.create_table_and_schema(table.identifier, table.schema)

    assert databackend.check_output_dest(predict_id)

    # Drop the output destination
    #
    databackend.drop_outputs()

    assert not databackend.check_output_dest(predict_id)


def test_query_builder(databackend: BaseDataBackend):
    query = databackend.get_query_builder("datas")

    assert isinstance(query, Query)
    assert query.table == "datas"


def test_list_tables_or_collections(databackend: BaseDataBackend):
    fields = {
        'a': int,
    }

    for i in range(10):
        table_name = f"table_{i}"
        databackend.create_table_and_schema(
            table_name, schema=Schema(identifier="schema", fields=fields)
        )

    tables = databackend.list_tables_or_collections()
    assert len(tables) == 10
    assert [f"table_{i}" for i in range(10)] == sorted(tables)
