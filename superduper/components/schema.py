import base64
import typing as t
from functools import cached_property

from superduper.base.constant import KEY_SCHEMA
from superduper.base.leaf import Leaf
from superduper.components.component import Component
from superduper.components.datatype import BaseDataType, Saveable
from superduper.misc.reference import parse_reference
from superduper.misc.special_dicts import SuperDuperFlatEncode

if t.TYPE_CHECKING:
    from superduper.base.document import Getters


class FieldType(Leaf):
    """Field type to represent the type of a field in a table.

    This is a wrapper around native datatype

    :param identifier: The name of the data type.
    """

    identifier: t.Union[str, BaseDataType]

    def __post_init__(self, db):
        super().__post_init__(db)

        # TODO why would this happen?
        if isinstance(self.identifier, BaseDataType):
            self.identifier = self.identifier.name

        elif isinstance(self.identifier, self.__class__):
            self.identifier = self.identifier.identifier

        elif isinstance(self.identifier, str):
            self.identifier = self.identifier

        elif self.identifier in (str, bool, int, float, bytes):
            self.identifier = self.identifier.__name__
        else:
            raise ValueError(f'Invalid field type {self.identifier}')


ID = FieldType(identifier='ID')


def _convert_base64_to_bytes(str_: str) -> bytes:
    return base64.b64decode(str_)


def _convert_bytes_to_base64(bytes_: bytes) -> str:
    return base64.b64encode(bytes_).decode('utf-8')


class Schema(Component):
    """A component carrying the `DataType` of columns.

    :param fields: A mapping of field names to types or `Encoders`
    """

    type_id: t.ClassVar[str] = 'schema'
    fields: t.Mapping[str, BaseDataType]

    def __post_init__(self, db):
        assert self.identifier is not None, 'Schema must have an identifier'
        assert self.fields is not None, 'Schema must have fields'
        super().__post_init__(db)

        for k, v in self.fields.items():
            if isinstance(v, (BaseDataType, FieldType)):
                continue

            try:
                v = FieldType(identifier=v)
            except ValueError:
                raise ValueError(f'Invalid field type {v} for field {k}')

            self.fields[k] = v

    def update(self, other: 'Schema'):
        new_fields = self.fields.copy()
        new_fields.update(other.fields)
        return Schema(self.identifier, fields=new_fields)

    # TODO why do we need this?
    @cached_property
    def encoded_types(self):
        """List of fields of type DataType."""
        return [k for k, v in self.fields.items() if isinstance(v, BaseDataType)]

    @cached_property
    def trivial(self):
        """Determine if the schema contains only trivial fields."""
        return not any([isinstance(v, BaseDataType) for v in self.fields.values()])

    @property
    def encoders(self):
        """An iterable to list DataType fields."""
        for v in self.fields.values():
            if isinstance(v, BaseDataType):
                yield v

    @property
    def fields_set(self):
        """Get the fields set for the schema."""
        fields = set()
        for k, v in self.fields.items():
            if hasattr(v, 'identifier'):
                fields.add((k, v.identifier))
        return fields

    def decode_data(
        self, data: dict[str, t.Any], getters: 'Getters'
    ) -> dict[str, t.Any]:
        """Decode data using the schema's encoders.

        :param data: Data to decode.
        :param getters: Getters to decode.
        """
        if self.trivial:
            return data
        decoded = {}
        for k, value in data.items():
            field = self.fields.get(k)
            if not isinstance(field, BaseDataType) or value is None:
                decoded[k] = value
                continue

            value = data[k]

            if reference := parse_reference(value):
                saveable: Saveable = getters.run(reference.name, reference.path)
                decoded[k] = saveable
            elif isinstance(value, str) and value.startswith('?'):
                decoded[k] = value
                continue
            else:
                b = data[k]
                if (
                    field.encodable == 'encodable'
                    and self.db.databackend.bytes_encoding == 'base64'
                ):
                    b = _convert_base64_to_bytes(b)
                decoded[k] = field.decode_data(b)

        decoded.pop(KEY_SCHEMA, None)
        return decoded

    def encode_data(self, out, builds, blobs, files, leaves_to_keep=()):
        """Encode data using the schema's encoders.

        :param out: Data to encode.
        :param builds: Builds.
        :param blobs: Blobs.
        :param files: Files.
        """
        for k, field in self.fields.items():
            if not isinstance(field, BaseDataType):
                continue

            if k not in out:
                continue

            if isinstance(out[k], Saveable):
                continue

            if isinstance(out[k], leaves_to_keep):
                continue

            if isinstance(out[k], Leaf):
                from superduper.base.document import _deep_flat_encode

                out[k] = _deep_flat_encode(
                    out[k],
                    builds,
                    blobs=blobs,
                    files=files,
                    leaves_to_keep=leaves_to_keep,
                )
                continue

            if out[k] is None:
                continue

            data = field.encode_data(out[k])

            if (
                field.encodable == 'encodable'
                and self.db.databackend.bytes_encoding == 'base64'
            ):
                assert isinstance(data, bytes)
                data = _convert_bytes_to_base64(data)
                out[k] = data

            elif isinstance(data, Saveable):
                ref_obj = parse_reference(data.reference)

                if ref_obj.name == 'blob':
                    blobs[data.identifier] = data.bytes

                elif ref_obj.name == 'file':
                    files[data.identifier] = data.path
                else:
                    assert False, f'Unknown reference type {ref_obj.name}'

                out[k] = data.reference
            else:
                out[k] = data

        out['_schema'] = self.identifier

        return out

    def __call__(self, data: dict[str, t.Any]) -> dict[str, t.Any]:
        """Encode data using the schema's encoders.

        :param data: Data to encode.
        """
        if self.trivial:
            return data

        encoded_data = {}
        for k, v in data.items():
            if k in self.fields and isinstance(self.fields[k], BaseDataType):
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
    return db.load('schema', schema)
