from ibis.expr.datatypes import dtype
from superduper import CFG
from superduper.components.datatype import (
    Artifact,
    BaseDataType,
    File,
    LazyArtifact,
    LazyFile,
    Native,
)
from superduper.components.schema import ID, FieldType, Schema

SPECIAL_ENCODABLES_FIELDS = {
    File: "str",
    LazyFile: "str",
    Artifact: "str",
    LazyArtifact: "str",
    Native: "json",
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
            if v.encodable_cls in SPECIAL_ENCODABLES_FIELDS:
                fields[k] = dtype(SPECIAL_ENCODABLES_FIELDS[v.encodable_cls])
            else:
                fields[k] = CFG.bytes_encoding

    return fields
