import pandas as pd


def _default_insert_processor(table_name, datas):
    """Default insert processor for SQL dialects."""
    return table_name, datas


def _clickhouse_insert_processor(table_name, datas):
    """Insert processor for ClickHouse."""
    return f'`{table_name}`', pd.DataFrame(datas)


def get_insert_processor(dialect):
    """Get the insert processor for the given dialect."""
    funcs = {
        'clickhouse': _clickhouse_insert_processor,
    }
    return funcs.get(dialect, _default_insert_processor)
