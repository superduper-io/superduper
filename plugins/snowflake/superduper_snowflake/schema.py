from collections import defaultdict

from snowflake.snowpark.types import (
    BooleanType,
    DoubleType,
    IntegerType,
    StringType,
    VariantType,
)
from superduper.base.datatype import BaseDataType, FieldType
from superduper.base.schema import Schema

LOOKUP = defaultdict(
    lambda: ('VARIANT', VariantType()),
    {
        'str': ('VARCHAR', StringType()),
        'int': ('NUMBER', IntegerType()),
        'float': ('FLOAT', DoubleType()),
        'bool': ('BOOLEAN', BooleanType()),
        'json': ('VARIANT', VariantType()),
    },
)


def superduper_to_snowflake_schema(schema: Schema, primary_id: str):
    """Convert a SuperDuper schema to a Snowflake schema.

    :param schema: The SuperDuper schema.
    :param primary_id: The primary ID column.
    """
    snowflake_schema = []
    snowflake_schema.append(f'"{primary_id}" VARCHAR PRIMARY KEY')
    for f, dt in schema.fields.items():
        if f == primary_id:
            continue
        if isinstance(dt, BaseDataType):
            value = LOOKUP[dt.dtype][0]
        elif isinstance(dt, FieldType):
            value = LOOKUP[dt.identifier][0]
        else:
            raise ValueError(
                f'Unknown datatype: {dt}; expected {BaseDataType} or {FieldType}'
            )
        snowflake_schema.append(f'"{f}" {value}')
    return snowflake_schema
