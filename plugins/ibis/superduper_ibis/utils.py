from ibis.expr.datatypes import dtype
from superduper.components.datatype import (
    BaseDataType,
    File,
)
from superduper.components.schema import ID, FieldType, Schema

SPECIAL_ENCODABLES_FIELDS = {
    File: "str",
}


def _convert_field_type_to_ibis_type(field_type: FieldType):
    if field_type.identifier == ID.identifier:
        ibis_type = "String"
    else:
        ibis_type = field_type.identifier
    return dtype(ibis_type)


def convert_schema_to_fields(schema: Schema):
    """Return the raw fields.

    Get a dictionary of fields as keys and datatypes as values.
    This is used to create ibis tables.
    :param schema: The schema to convert
    """
    fields = {}

    for k, v in schema.fields.items():
        if isinstance(v, FieldType):
            fields[k] = _convert_field_type_to_ibis_type(v)
        elif not isinstance(v, BaseDataType):
            fields[k] = v.identifier
        else:
            if v.encodable == 'encodable':
                fields[k] = dtype(
                    'str'
                    if schema.db.databackend.bytes_encoding == 'base64'
                    else 'bytes'
                )
            else:
                fields[k] = dtype('str')

    return fields
