from ibis.expr.datatypes import dtype
from superduper.components.datatype import DataType
from superduper.components.schema import ID, FieldType, Schema

SPECIAL_ENCODABLES_FIELDS = {
    "file": "String",
    "lazy_file": "String",
    "artifact": "String",
    "lazy_artifact": "String",
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
        elif not isinstance(v, DataType):
            fields[k] = v.identifier
        else:
            if v.encodable_cls.leaf_type in SPECIAL_ENCODABLES_FIELDS:
                fields[k] = SPECIAL_ENCODABLES_FIELDS[v.encodable_cls.leaf_type]
            else:
                fields[k] = v.bytes_encoding

    return fields
