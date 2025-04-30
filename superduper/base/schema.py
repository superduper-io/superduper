# TODO move to base
import dataclasses as dc
import json
import re
import typing as t

from superduper import CFG, logging
from superduper.base.constant import KEY_PATH
from superduper.base.datatype import INBUILT_DATATYPES, BaseDataType
from superduper.base.encoding import EncodeContext
from superduper.misc.importing import isreallyinstance
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
    def parse(spec):
        """Parse a schema from a string.

        :param spec: The string to parse.
        """
        parts = dict([tuple(x.strip().split('=')) for x in spec.split('|')])
        fields = {k: INBUILT_DATATYPES[v] for k, v in parts.items()}
        return Schema(fields)

    @staticmethod
    def build(**fields: t.Dict[str, str]) -> 'Schema':
        """Build a schema from a dictionary of fields.

        :param fields: The fields of the schema

        # noqa
        """
        from superduper.base.datatype import INBUILT_DATATYPES

        fields = {k: INBUILT_DATATYPES[fields[k]] for k in fields}
        return Schema(fields)  # type: ignore[arg-type]

    def __add__(self, other: 'Schema'):
        new_fields = self.fields.copy()
        new_fields.update(other.fields)
        return Schema(fields=new_fields)

    def __getitem__(self, item):
        return self.fields[item]

    @property
    def trivial(self):
        """Determine if the schema contains only trivial fields."""
        return not any(
            [isreallyinstance(v, BaseDataType) for v in self.fields.values()]
        )

    def __repr__(self):
        return dict_to_ascii_table(self.fields)

    @staticmethod
    def handle_references(item, builds):
        """Handle references in a schema.

        :param item: Item to handle references in.
        :param builds: Build cache for decoding references.
        """
        if '?(' not in str(item):
            return item

        if isinstance(item, str):
            instances = re.findall(r'\?\((.*?)\)', item)

            for k in instances:
                name = k.split('.')[0]
                attr = k.split('.')[-1]

                if name not in builds:
                    logging.warn(
                        f'Could not find reference {name} '
                        f'from reference in {item} in builds'
                    )
                    return item

                to_replace = getattr(builds[name], attr)
                item = item.replace(f'?({k})', str(to_replace))

            return item
        elif isinstance(item, list):
            return [Schema.handle_references(i, builds) for i in item]
        elif isinstance(item, dict):
            return {
                Schema.handle_references(k, builds): Schema.handle_references(v, builds)
                for k, v in item.items()
            }
        else:
            return item

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

        # reorder the component so that references go first
        def is_ref(x):
            return isinstance(x, str) and x.startswith('?')

        data_is_ref = {k: v for k, v in data.items() if is_ref(v)}
        data_not_ref = {k: v for k, v in data.items() if not is_ref(v)}
        data = {**data_is_ref, **data_not_ref}

        for k, value in data.items():
            field = self.fields.get(k)

            value = self.handle_references(value, builds)

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
                msg = (
                    'Expected a string since databackend is not json native;'
                    ' CFG.json_native is False'
                )
                # If the item has been cached, it will not be a string
                if (
                    isinstance(value, str)
                    and field.dtype == 'json'
                    and not CFG.json_native
                ):
                    assert isinstance(value, str), msg
                    try:
                        value = json.loads(value)
                    except json.JSONDecodeError:
                        pass

                decoded[k] = field.decode_data(value, builds=builds, db=db)

        return decoded

    def encode_data(self, out, context: t.Optional[EncodeContext] = None, **kwargs):
        """Encode data using the schema's encoders.

        :param out: Data to encode.
        :param context: Encoding context.
        :param kwargs: Additional encoding arguments.
        """
        result = {k: v for k, v in out.items()}

        if context is None:
            context = EncodeContext()
        for k, v in kwargs.items():
            setattr(context, k, v)

        for k in out:
            field = self.fields.get(k)

            if not isreallyinstance(field, BaseDataType):
                continue

            if isinstance(out[k], str) and (
                out[k].startswith('?') or out[k].startswith('&')
            ):
                continue

            if out[k] is None:
                continue

            assert field is not None

            # TODO
            # field.validate(out[k])

            try:
                encoded = field.encode_data(
                    out[k],
                    context=context(name=context.name + f'.{k}'),
                )
            except Exception as e:
                logging.error(
                    f'Error encoding field {context.name + "." + k} with '
                    f'value {out[k]} and field {field}. '
                    'Have you chosen an appropriate encoder?'
                )
                raise e

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


class NotSupportedError(Exception):
    """Exception raised when a schema is not supported.

    # noqa
    """


def create_pydantic(
    name: str, schema: Schema, components: t.Dict[str, t.Type] | None = None
):
    """Create pydantic model from schema.

    :param name: Name of the model.
    :param schema: Schema to create the model from.
    :param components: Additional components to add to the model.
    """
    lookup = {
        'str': str | None,
        'int': int | None,
        'float': float | None,
        'bool': bool | None,
        'bytes': bytes | None,
        'dict': dict | None,
        'list': list | None,
        'json': dict | list | None,
    }

    from pydantic import BaseModel, create_model

    components = components or {}

    to_pass = {}
    for k, v in schema.fields.items():
        if str(v).lower() not in lookup and str(v).lower() not in components:
            continue
        to_pass[k] = (lookup[str(v).lower()], None)

    if 'status' in to_pass:
        del to_pass['status']
    if 'version' in to_pass:
        del to_pass['version']

    return create_model(name, **to_pass, __base__=BaseModel)  # type: ignore[call-overload]
