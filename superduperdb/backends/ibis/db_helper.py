import base64

import pandas as pd

BASE64_PREFIX = 'base64:'


class Base64Mixin:
    def convert_data_format(self, data):
        """Convert byte data to base64 format for storage in the database."""
        if isinstance(data, bytes):
            return BASE64_PREFIX + base64.b64encode(data).decode('utf-8')
        else:
            return data

    def recover_data_format(self, data):
        """Recover byte data from base64 format stored in the database."""
        if isinstance(data, str) and data.startswith(BASE64_PREFIX):
            return base64.b64decode(data[len(BASE64_PREFIX) :])
        else:
            return data

    def process_schema_types(self, schema_mapping):
        """Convert bytes to string in the schema."""
        for key, value in schema_mapping.items():
            if value == 'Bytes':
                schema_mapping[key] = 'String'
        return schema_mapping


class DBHelper:
    match_dialect = 'base'

    def __init__(self, dialect):
        self.dialect = dialect

    def process_before_insert(self, table_name, datas):
        return table_name, pd.DataFrame(datas)

    def process_schema_types(self, schema_mapping):
        return schema_mapping

    def convert_data_format(self, data):
        return data

    def recover_data_format(self, data):
        return data


class ClickHouseHelper(Base64Mixin, DBHelper):
    match_dialect = 'clickhouse'

    def process_before_insert(self, table_name, datas):
        return f'`{table_name}`', pd.DataFrame(datas)


def get_db_helper(dialect) -> DBHelper:
    """Get the insert processor for the given dialect."""
    for helper in DBHelper.__subclasses__():
        if helper.match_dialect == dialect:
            return helper(dialect)

    return DBHelper(dialect)
