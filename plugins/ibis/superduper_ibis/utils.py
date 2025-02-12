from ibis.expr.datatypes import dtype
from superduper.components.datatype import (
    ID,
    BaseDataType,
    FieldType,
    FileItem,
)
from superduper.components.schema import Schema

SPECIAL_ENCODABLES_FIELDS = {
    FileItem: "str",
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
        else:
            assert isinstance(schema.fields[k], BaseDataType)

            fields[k] = dtype(schema.fields[k].dtype)

    return fields
