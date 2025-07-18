from superduper.base.datatype import Vector
from superduper.base.schema import Schema


def superduper_to_postgres_schema(schema: Schema, primary_id: str = 'id') -> dict:
    """Convert a SuperDuper schema to a Postgres schema.

    :param schema: The SuperDuper schema.
    """
    out = {}
    out[primary_id] = 'VARCHAR(64) PRIMARY KEY'
    for f in schema.fields:
        if f == primary_id:
            continue
        if str(schema[f]).lower() == 'str':
            out[f] = 'TEXT'
        elif str(schema[f]).lower() == 'int':
            out[f] = 'INT'
        elif str(schema[f]).lower() == 'float':
            out[f] = 'FLOAT'
        elif str(schema[f]).lower() == 'json':
            out[f] = 'JSONB'
        elif str(schema[f]).lower() == 'bool':
            out[f] = 'BOOLEAN'
        elif isinstance(schema[f], Vector):
            out[f] = f'VECTOR({schema[f].shape})'
        else:
            if schema[f].dtype == 'str':
                out[f] = 'TEXT'
            elif schema[f].dtype == 'json':
                out[f] = 'JSONB'
            else:
                raise ValueError(f"Unsupported field type: {schema[f].dtype} for field {f}")
    return out