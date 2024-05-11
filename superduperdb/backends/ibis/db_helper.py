import base64

import pandas as pd

BASE64_PREFIX = 'base64:'


class Base64Mixin:
    """Mixin class for converting byte data to base64 format.

    This class is used to convert byte data to base64 format for storage in the
    database.  # noqa
    """

    def convert_data_format(self, data):
        """Convert byte data to base64 format for storage in the database.

        :param data: The data to convert.
        """
        if isinstance(data, bytes):
            return BASE64_PREFIX + base64.b64encode(data).decode('utf-8')
        else:
            return data

    def recover_data_format(self, data):
        """Recover byte data from base64 format stored in the database.

        :param data: The data to recover.
        """
        if isinstance(data, str) and data.startswith(BASE64_PREFIX):
            return base64.b64decode(data[len(BASE64_PREFIX) :])
        else:
            return data

    def process_schema_types(self, schema_mapping):
        """Convert bytes to string in the schema.

        :param schema_mapping: The schema mapping to convert.
        """
        for key, value in schema_mapping.items():
            if value == 'Bytes':
                schema_mapping[key] = 'String'
        return schema_mapping


class DBHelper:
    """Generic helper class for database.

    :param dialect: The dialect of the database.
    """

    match_dialect = 'base'

    def __init__(self, dialect):
        self.dialect = dialect

    def process_before_insert(self, table_name, datas):
        """Convert byte data to base64 format for storage in the database.

        :param table_name: The name of the table.
        :param datas: The data to insert.
        """
        return table_name, pd.DataFrame(datas)

    def process_schema_types(self, schema_mapping):
        """Convert bytes to string in the schema.

        :param schema_mapping: The schema mapping to convert.
        """
        return schema_mapping

    def convert_data_format(self, data):
        """Convert data to the format for storage in the database.

        :param data: The data to convert.
        """
        return data

    def recover_data_format(self, data):
        """Recover data from the format stored in the database.

        :param data: The data to recover.
        """
        return data


class ClickHouseHelper(Base64Mixin, DBHelper):
    """Helper class for ClickHouse database.

    This class is used to convert byte data to base64 format for storage in the
    database.

    :param dialect: The dialect of the database.
    """

    match_dialect = 'clickhouse'

    def process_before_insert(self, table_name, datas):
        """Convert byte data to base64 format for storage in the database.

        :param table_name: The name of the table.
        :param datas: The data to insert.
        """
        return f'`{table_name}`', pd.DataFrame(datas)


def get_db_helper(dialect) -> DBHelper:
    """Get the insert processor for the given dialect.

    :param dialect: The dialect of the database.
    """
    for helper in DBHelper.__subclasses__():
        if helper.match_dialect == dialect:
            return helper(dialect)

    return DBHelper(dialect)
