from ibis.expr.datatypes import dtype
from superduper import CFG
from superduper.base.datatype import (
    ID,
    Array,
    BaseDataType,
    FieldType,
    FileItem,
    Vector,
)
from superduper.base.schema import Schema

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

            if not CFG.json_native and schema.fields[k].dtype == "json":
                fields[k] = dtype("str")
            elif isinstance(v, Array):
                fields[k] = dtype("str")
            elif isinstance(v, Vector) and isinstance(v.datatype_impl, Array):
                fields[k] = dtype("str")
            else:
                fields[k] = dtype(schema.fields[k].dtype)

    return fields
