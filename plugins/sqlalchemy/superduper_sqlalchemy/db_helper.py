import json
from typing import Tuple

from sqlalchemy import (
    Boolean,
    DateTime,
    Integer,
    String,
    Text,
    TypeDecorator,
)

DEFAULT_LENGTH = 255


class JsonMixin:
    """Mixin for JSON type columns # noqa.

    Converts dict to JSON strings before saving to database
    and converts JSON strings to dict when loading from database.

    # noqa
    """

    def process_bind_param(self, value, dialect):
        """Convert dict to JSON string.

        :param value: The dict to convert.
        :param dialect: The dialect of the database.
        """
        if value is not None:
            value = json.dumps(value)
        return value

    def process_result_value(self, value, dialect):
        """Convert JSON string to dict.

        :param value: The JSON string to convert.
        :param dialect: The dialect of the database.
        """
        if value is not None:
            value = json.loads(value)
        return value


class JsonAsString(JsonMixin, TypeDecorator):
    """JSON type column for short JSON strings # noqa."""

    impl = String(DEFAULT_LENGTH)


class JsonAsText(JsonMixin, TypeDecorator):
    """JSON type column for long JSON strings # noqa."""

    impl = Text


class DefaultConfig:
    """Default configuration for database types # noqa."""

    type_string = String(DEFAULT_LENGTH)
    type_string_long = String(1000)
    type_json_as_string = JsonAsString
    type_json_as_text = JsonAsText
    type_integer = Integer
    type_datetime = DateTime
    type_boolean = Boolean

    query_id_table_args: Tuple = tuple()
    job_table_args: Tuple = tuple()
    parent_child_association_table_args: Tuple = tuple()
    component_table_args: Tuple = tuple()
    meta_table_args: Tuple = tuple()


def create_clickhouse_config():
    """Create configuration for ClickHouse database."""
    # lazy import
    try:
        from clickhouse_sqlalchemy import engines, types
    except ImportError:
        raise ImportError(
            'The clickhouse_sqlalchemy package is required to use the '
            'clickhouse dialect. Please install it with pip install '
            'clickhouse-sqlalchemy'
        )

    class ClickHouseConfig:
        class JsonAsString(JsonMixin, TypeDecorator):
            impl = types.String

        class JsonAsText(JsonMixin, TypeDecorator):
            impl = types.String

        type_string = types.String
        type_json_as_string = JsonAsString
        type_json_as_text = JsonAsText
        type_integer = types.Int32
        type_datetime = types.DateTime
        type_boolean = types.Boolean

        # clickhouse need engine args to create table
        query_id_table_args = (engines.MergeTree(order_by='query_id'),)
        job_table_args = (engines.MergeTree(order_by='identifier'),)
        parent_child_association_table_args = (engines.MergeTree(order_by='parent_id'),)
        component_table_args = (engines.MergeTree(order_by='id'),)
        meta_table_args = (engines.MergeTree(order_by='key'),)

    return ClickHouseConfig


def get_db_config(dialect):
    """Get the configuration class for the specified dialect.

    :param dialect: The dialect of the database.
    """
    if dialect == 'clickhouse':
        return create_clickhouse_config()
    else:
        return DefaultConfig
