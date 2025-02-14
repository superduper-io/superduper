import dataclasses as dc
import json
import typing as t
from functools import cached_property

from superduper import CFG
from superduper.components.datatype import BaseDataType
from superduper.misc.special_dicts import dict_to_ascii_table

if t.TYPE_CHECKING:
    pass


@dc.dataclass
class Schema(BaseDataType):
    """A component carrying the `DataType` of columns.

    :param fields: A mapping of field names to types or `Encoders`
    """

    fields: t.Dict[str, BaseDataType]

    @staticmethod
    def build(fields: t.Dict[str, str]) -> 'Schema':
        """Build a schema from a dictionary of fields.

        :param fields: The fields of the schema
        """
        from superduper.components.datatype import INBUILT_DATATYPES

        fields = {k: INBUILT_DATATYPES[fields[k]] for k in fields}
        return Schema(fields)  # type: ignore[arg-type]

    def __add__(self, other: 'Schema'):
        new_fields = self.fields.copy()
        new_fields.update(other.fields)
        return Schema(fields=new_fields)

    @cached_property
    def trivial(self):
        """Determine if the schema contains only trivial fields."""
        return not any([isinstance(v, BaseDataType) for v in self.fields.values()])

    def __repr__(self):
        return dict_to_ascii_table(self.fields)

    def decode_data(
        self, data: dict[str, t.Any], builds: t.Dict, db
    ) -> dict[str, t.Any]:
        """Decode data using the schema's encoders.

        :param data: Data to decode.
        :param builds: build cache for decoding references.
        :param db: Datalayer instance
        """
        if self.trivial:
            return data

        decoded = {}

        for k, value in data.items():
            field = self.fields.get(k)

            if not isinstance(field, BaseDataType) or value is None:
                decoded[k] = value
                continue

            if (
                isinstance(value, str)
                and value.startswith('?')
                and not isinstance(builds[value[1:]], dict)
            ):
                decoded[k] = builds[value[1:]]
                continue
            else:
                if field.dtype == 'json' and not CFG.json_native:
                    value = json.loads(value)
                decoded[k] = field.decode_data(value, builds=builds, db=db)

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

            encoded = field.encode_data(
                out[k],
                builds=builds,
                blobs=blobs,
                files=files,
                leaves_to_keep=leaves_to_keep,
            )

            if field.dtype == 'json' and not CFG.json_native:
                encoded = json.dumps(encoded)

            result[k] = encoded

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
