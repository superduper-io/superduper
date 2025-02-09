import inspect
import typing as t
from functools import cached_property

from superduper.base.leaf import Leaf
from superduper.components.datatype import BaseDataType

if t.TYPE_CHECKING:
    pass


class FieldType(Leaf):
    """Field type to represent the type of a field in a table.

    This is a wrapper around native datatype

    :param identifier: The name of the data type.
    """

    identifier: t.Union[str, BaseDataType]

    def postinit(self):
        """Post initialization method."""
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

        super().postinit()


ID = FieldType(identifier='ID')


class Schema(BaseDataType):
    """A component carrying the `DataType` of columns.

    :param fields: A mapping of field names to types or `Encoders`
    """

    _fields = {'fields': 'sdict'}

    type_id: t.ClassVar[str] = 'schema'
    fields: t.Mapping[str, BaseDataType]

    def postinit(self):
        """Post initialization method."""
        assert self.identifier is not None, 'Schema must have an identifier'
        assert self.fields is not None, 'Schema must have fields'

        for k, v in self.fields.items():
            if isinstance(v, (BaseDataType, FieldType)):
                v.db = self.db
                continue

            if inspect.isclass(v) and issubclass(v, Leaf):
                continue

            try:
                v = FieldType(identifier=v)
            except ValueError:
                raise ValueError(f'Invalid field type {v} for field {k}')

            self.fields[k] = v
        return super().postinit()

    def __add__(self, other: 'Schema'):
        new_fields = self.fields.copy()
        new_fields.update(other.fields)
        id = self.identifier + '+' + other.identifier
        return Schema(id, fields=new_fields, db=self.db)

    @cached_property
    def trivial(self):
        """Determine if the schema contains only trivial fields."""
        return not any([isinstance(v, BaseDataType) for v in self.fields.values()])

    def decode_data(self, data: dict[str, t.Any], builds: t.Dict) -> dict[str, t.Any]:
        """Decode data using the schema's encoders.

        :param data: Data to decode.
        :param builds: build cache for decoding references.
        """
        if self.trivial:
            return data

        decoded = {}

        for k, value in data.items():
            field = self.fields.get(k)

            if not isinstance(field, BaseDataType) or value is None:
                decoded[k] = value
                continue

            decoded[k] = field.decode_data(value, builds=builds)

            if isinstance(value, str) and value.startswith('?'):
                decoded[k] = builds[value[1:]]
                continue
            else:
                decoded[k] = field.decode_data(value, builds=builds)

        return decoded

    def encode_data(self, out, builds, blobs, files, leaves_to_keep=()):
        """Encode data using the schema's encoders.

        :param out: Data to encode.
        :param builds: Builds.
        :param blobs: Blobs.
        :param files: Files.
        :param leaves_to_keep: `Leaf` instances to keep (don't encode)
        """
        result = {k: v for k, v in out.items()}

        for k in out:
            field = self.fields.get(k)

            if not isinstance(field, BaseDataType):
                continue

            if isinstance(out[k], str) and (
                out[k].startswith('?') or out[k].startswith('&')
            ):
                continue

            if out[k] is None:
                continue

            result[k] = field.encode_data(
                out[k],
                builds=builds,
                blobs=blobs,
                files=files,
                leaves_to_keep=leaves_to_keep,
            )

        return result


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
