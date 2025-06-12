import typing as t

from ibis.expr.datatypes import dtype
from superduper.base.datatype import (
    ID,
    Array,
    BaseDataType,
    BaseVector,
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


def convert_schema_to_fields(
    schema: Schema, json_native: bool, vector_impl: t.Type[BaseVector]
) -> dict:
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
            if isinstance(v, Vector):
                v = vector_impl(shape=v.shape, dtype=v.dtype)

            assert isinstance(v, BaseDataType)

            if not json_native and v.dtype == "json":
                fields[k] = dtype("str")
            elif isinstance(v, Array):
                fields[k] = dtype("str")
            else:
                fields[k] = dtype(v.dtype)

    return fields
