import dataclasses as dc
import typing as t
from functools import cached_property

from overrides import override

from superduperdb.components.component import Component
from superduperdb.components.datatype import DataType, _BaseEncodable
from superduperdb.misc.annotations import merge_docstrings
from superduperdb.misc.special_dicts import SuperDuperFlatEncode

SCHEMA_KEY = '_schema'


class _Native:
    _TYPES = {str: 'str', int: 'int', float: 'float'}

    def __init__(self, x):
        if x in self._TYPES:
            x = self._TYPES[x]
        self.identifier = x


@merge_docstrings
@dc.dataclass(kw_only=True)
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
        for v in self.fields.values():
            if isinstance(v, DataType):
                db.add(v)
        return super().pre_create(db)

    def deep_flat_encode_data(self, r, cache, blobs, files, leaves_to_keep=()):
        """Deep flat encode data.

        :param r: Data to encode.
        :param cache: Cache for encoding.
        :param blobs: Blobs for encoding.
        :param files: Files for encoding.
        :param leaves_to_keep: Leaves to keep.
        """
        for k, datatype in self.fields.items():
            if k not in r:
                continue
            value = r[k]
            if isinstance(datatype, DataType):
                if isinstance(value, _BaseEncodable):
                    assert value.datatype.identifier == datatype.identifier
                    encodable = value
                else:
                    encodable = datatype(value)
                if isinstance(encodable, leaves_to_keep):
                    continue
                r[k] = encodable._deep_flat_encode(
                    cache, blobs, files, leaves_to_keep=leaves_to_keep, schema=self
                )
        r[SCHEMA_KEY] = self.identifier
        return r

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

            if isinstance(value, str) and value.startswith('?'):
                decoded[k] = field.encodable_cls.build_from_reference(
                    value, datatype=field
                )
            else:
                decoded[k] = field.decode_data(data[k])
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
