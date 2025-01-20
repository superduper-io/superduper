# TODO - remove in favour of annotations on classes
import importlib
import typing as t

import numpy as np

from superduper import CFG, logging
from superduper.base.exceptions import UnsupportedDatatype
from superduper.components.datatype import (
    INBUILT_DATATYPES,
    JSON,
    BaseDataType,
    DataTypeFactory,
    Vector,
)
from superduper.components.schema import FieldType, Schema

if t.TYPE_CHECKING:
    from superduper.base.datalayer import Datalayer


def register_module(module_name):
    """Register a module for datatype inference.

    Only modules with a check and create function will be registered
    :param module_name: The module name, e.g. "superduper_plugins.numpy.encoder"
    """
    try:
        importlib.import_module(module_name)
    except Exception:
        logging.debug(f"Could not register module: {module_name}")


BASE_TYPES = (
    int,
    str,
    float,
    bool,
    bytes,
    bytearray,
)


def infer_datatype(
    data: t.Any, db: 'Datalayer'
) -> t.Optional[t.Union[BaseDataType, type]]:
    """Infer the datatype of a given data object.

    If the data object is a base type, return None,
    Otherwise, return the inferred datatype

    :param data: The data object.
    :param db: The datalayer.
    """
    datatype = None

    try:
        from bson import ObjectId

        if isinstance(data, ObjectId):
            return datatype
    except ImportError:
        pass

    if isinstance(data, BASE_TYPES):
        datatype: BaseDataType = type(data)
        logging.debug(f"Inferred base type: {datatype} for data:", data)
        return datatype

    for factory in FACTORIES:
        if factory.check(data):
            datatype: BaseDataType = factory.create(data, db=db)
            assert isinstance(datatype, BaseDataType) or isinstance(datatype, FieldType)
            logging.debug(f"Inferred datatype: {datatype.identifier} for data: {data}")
            break

    if datatype is None:
        datatype: BaseDataType = INBUILT_DATATYPES['encodable']('encodable', db=db)
        try:
            encoded_data = datatype.encode_data(data)
            decoded_data = datatype.decode_data(encoded_data)
            assert isinstance(decoded_data, type(data))
        except Exception as e:
            import traceback

            logging.error(traceback.format_exc())
            raise UnsupportedDatatype(
                f"Could not infer datatype for data: {data}"
            ) from e
        logging.debug(f"Inferring default datatype for data: {data}")

    return datatype


def infer_schema(
    data: t.Mapping[str, t.Any],
    db: 'Datalayer',
    identifier: t.Optional[str] = None,
) -> Schema:
    """Infer a schema from a given data object.

    :param data: The data object.
    :param db: The datalayer.
    :param identifier: The identifier for the schema, if None, it will be generated.
    """
    assert isinstance(data, dict), "Data must be a dictionary"

    schema_data = {}
    for k, v in data.items():
        try:
            data_type = infer_datatype(v, db=db)
        except UnsupportedDatatype as e:
            raise UnsupportedDatatype(
                f"Could not infer datatype for key: {k}, value: {v}"
            ) from e
        if data_type is not None:
            schema_data[k] = data_type

    if identifier is None:
        if not schema_data:
            identifier = "empty"
        else:
            key_value_pairs = []
            for k, v in sorted(schema_data.items()):
                if hasattr(v, "identifier"):
                    key_value_pairs.append(f"{k}={v.identifier}")
                else:
                    key_value_pairs.append(f"{k}={str(v)}")
            identifier = "&".join(key_value_pairs)

    identifier = "AUTO-" + identifier

    return Schema(identifier=identifier, fields=schema_data, db=db)  # type: ignore


class VectorTypeFactory(DataTypeFactory):
    """A factory for Vector datatypes # noqa."""

    @staticmethod
    def check(data: t.Any) -> bool:
        """Check if the data is able to be encoded by the JSON serializer.

        :param data: The data object
        """
        return isinstance(data, np.ndarray) and len(data.shape) == 1

    @staticmethod
    def create(data: t.Any, db: 'Datalayer') -> BaseDataType | FieldType:
        """Create a vector datatype.

        :param data: The data object.
        :param db: The datalayer.
        """
        return Vector(shape=(len(data),), dtype=str(data.dtype), db=db)


class JsonDataTypeFactory(DataTypeFactory):
    """A factory for JSON datatypes # noqa."""

    @staticmethod
    def check(data: t.Any) -> bool:
        """Check if the data is able to be encoded by the JSON serializer.

        :param data: The data object.
        """
        try:
            JSON('json').encode_data(data)
            return True
        except Exception:
            return False

    @staticmethod
    def create(data: t.Any, db: 'Datalayer') -> BaseDataType | FieldType:
        """Create a JSON datatype.

        :param data: The data object.
        :param db: The datalayer.
        """
        if CFG.json_native:
            return FieldType(identifier='json')
        return JSON('json', db=db)


register_module("superduper_torch.encoder")
register_module("superduper_pillow.encoder")


FACTORIES = DataTypeFactory.__subclasses__()
FACTORIES = sorted(FACTORIES, key=lambda x: 0 if x.__module__ == __name__ else 1)
