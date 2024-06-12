import typing as t
from functools import cached_property

from overrides import override

from superduperdb.components.component import Component
from superduperdb.components.datatype import DataType
from superduperdb.misc.reference import parse_reference
from superduperdb.misc.special_dicts import SuperDuperFlatEncode

SCHEMA_KEY = '_schema'


class _Native:
    _TYPES = {str: 'str', int: 'int', float: 'float'}

    def __init__(self, x):
        if x in self._TYPES:
            x = self._TYPES[x]
        self.identifier = x


class Schema(Component):
    """A component carrying the `DataType` of columns.

    :param fields: A mapping of field names to types or `Encoders`
    """

    type_id: t.ClassVar[str] = 'schema'
    fields: t.Mapping[str, DataType]

    def __post_init__(self, db, artifacts):
        assert self.identifier is not None, 'Schema must have an identifier'
        assert self.fields is not None, 'Schema must have fields'
        super().__post_init__(db, artifacts)

        for k, v in self.fields.items():
            if isinstance(v, str):
                self.fields[k] = _Native(v)
            elif v in (str, bool, int, float):
                self.fields[k] = _Native(v)

    @override
    def pre_create(self, db) -> None:
        """Database pre-create hook to add datatype to the database.

        :param db: Datalayer instance.
        """
        return super().pre_create(db)

    @cached_property
    def encoded_types(self):
        """List of fields of type DataType."""
        return [k for k, v in self.fields.items() if isinstance(v, DataType)]

    @cached_property
    def trivial(self):
        """Determine if the schema contains only trivial fields."""
        return not any([isinstance(v, DataType) for v in self.fields.values()])

    @property
    def encoders(self):
        """An iterable to list DataType fields."""
        for v in self.fields.values():
            if isinstance(v, DataType):
                yield v

    @property
    def fields_set(self):
        """Get the fields set for the schema."""
        fields = set()
        for k, v in self.fields.items():
            if hasattr(v, 'identifier'):
                fields.add((k, v.identifier))
        return fields

    def encode_data(self, out, builds, blobs, files, leaves_to_keep=()):
        """Encode data using the schema's encoders.

        :param out: Data to encode.
        :param builds: Builds.
        :param blobs: Blobs.
        :param files: Files.
        """
        for k, field in self.fields.items():
            if not isinstance(field, DataType):
                continue

            if isinstance(out[k], leaves_to_keep):
                continue

            data, identifier = field.encode_data_with_identifier(out[k])
            if field.encodable_cls.artifact:
                reference = field.encodable_cls.build_reference(identifier, data)
                ref_obj = parse_reference(reference)

                if ref_obj.name == 'blob':
                    blobs[identifier] = data
                elif ref_obj.name == 'file':
                    files[identifier] = data
                else:
                    assert False, f'Unknown reference type {ref_obj.name}'
                out[k] = reference
            else:
                out[k] = data

        out['_schema'] = self.identifier

        return out

    def decode_data(self, data: dict[str, t.Any]) -> dict[str, t.Any]:
        """Decode data using the schema's encoders.

        :param data: Data to decode.
        """
        if self.trivial:
            return data
        decoded = {}
        for k, value in data.items():
            field = self.fields.get(k)
            if not isinstance(field, DataType):
                decoded[k] = value
                continue

            value = data[k]

            if reference := parse_reference(value):
                if not reference.is_in_document:
                    file_id = value.split(':')[-1]
                    value = data.get(f'_{reference.name}s', {}).get(file_id, value)
                encodable = field.encodable_cls(datatype=field, x=value)
                if not field.encodable_cls.lazy:
                    encodable = encodable.unpack()
                decoded[k] = encodable
            else:
                decoded[k] = field.decode_data(data[k])

        decoded.pop(SCHEMA_KEY, None)
        return decoded

    def __call__(self, data: dict[str, t.Any]) -> dict[str, t.Any]:
        """Encode data using the schema's encoders.

        :param data: Data to encode.
        """
        if self.trivial:
            return data

        encoded_data = {}
        for k, v in data.items():
            if k in self.fields and isinstance(self.fields[k], DataType):
                field_encoder = self.fields[k]
                assert callable(field_encoder)
                encoded_data.update({k: field_encoder(v)})
            else:
                encoded_data.update({k: v})
        return SuperDuperFlatEncode(encoded_data)


def get_schema(db, schema: t.Union[Schema, str]) -> t.Optional[Schema]:
    """Handle schema caching and loading.

    :param db: Datalayer instance.
    :param schema: Schema to get. If a string, it will be loaded from the database.
    """
    if schema is None:
        return None
    if isinstance(schema, Schema):
        return schema
    assert isinstance(schema, str)
    if db is None:
        raise ValueError(
            f'A Datalayer instance is required for encoding with schema {schema}'
        )

    return db.schemas[schema]
