import importlib
import typing as t

import numpy as np

from superduper import CFG, logging
from superduper.base.exceptions import UnsupportedDatatype
from superduper.components.datatype import (
    DEFAULT_ENCODER,
    BaseDataType,
    DataTypeFactory,
    Vector,
    json_encoder,
)
from superduper.components.schema import FieldType, Schema


def register_module(module_name):
    """Register a module for datatype inference.

    Only modules with a check and create function will be registered
    :param module_name: The module name, e.g. "superduper.ext.numpy.encoder"
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


def infer_datatype(data: t.Any) -> t.Optional[t.Union[BaseDataType, type]]:
    """Infer the datatype of a given data object.

    If the data object is a base type, return None,
    Otherwise, return the inferred datatype

    :param data: The data object
    """
    datatype = None

    # # TODO - why this?
    # if isinstance(data, _BaseEncodable):
    #     return datatype

    try:
        from bson import ObjectId

        if isinstance(data, ObjectId):
            return datatype
    except ImportError:
        pass

    if isinstance(data, BASE_TYPES):
        datatype = type(data)
        logging.debug(f"Inferred base type: {datatype} for data:", data)
        return datatype

    for factory in FACTORIES:
        if factory.check(data):
            datatype = factory.create(data)
            assert isinstance(datatype, BaseDataType) or isinstance(datatype, FieldType)
            logging.debug(f"Inferred datatype: {datatype.identifier} for data: {data}")
            break

    if datatype is None:
        try:
            encoded_data = DEFAULT_ENCODER.encode_data(data)
            decoded_data = DEFAULT_ENCODER.decode_data(encoded_data)
            assert isinstance(decoded_data, type(data))
        except Exception as e:
            raise UnsupportedDatatype(
                f"Could not infer datatype for data: {data}"
            ) from e

        logging.debug(f"Inferring default datatype for data: {data}")
        datatype = DEFAULT_ENCODER

    return datatype


def infer_schema(
    data: t.Mapping[str, t.Any],
    identifier: t.Optional[str] = None,
) -> Schema:
    """Infer a schema from a given data object.

    :param data: The data object
    :param identifier: The identifier for the schema, if None, it will be generated
    """
    assert isinstance(data, dict), "Data must be a dictionary"

    schema_data = {}
    for k, v in data.items():
        try:
            data_type = infer_datatype(v)
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

    return Schema(identifier=identifier, fields=schema_data)  # type: ignore


class VectorTypeFactory(DataTypeFactory):
    """A factory for Vector datatypes # noqa."""

    @staticmethod
    def check(data: t.Any) -> bool:
        """Check if the data is able to be encoded by the JSON serializer.

        :param data: The data object
        """
        return isinstance(data, np.ndarray) and len(data.shape) == 1

    @staticmethod
    def create(data: t.Any) -> BaseDataType | FieldType:
        """Create a JSON datatype.

        :param data: The data object
        """
        return Vector(shape=(len(data),), dtype=str(data.dtype))


class JsonDataTypeFactory(DataTypeFactory):
    """A factory for JSON datatypes # noqa."""

    @staticmethod
    def check(data: t.Any) -> bool:
        """Check if the data is able to be encoded by the JSON serializer.

        :param data: The data object
        """
        try:
            json_encoder.encode_data(data)
            return True
        except Exception:
            return False

    @staticmethod
    def create(data: t.Any) -> BaseDataType | FieldType:
        """Create a JSON datatype.

        :param data: The data object
        """
        if CFG.json_native:
            return FieldType(identifier='json')
        return json_encoder


register_module("superduper.ext.numpy.encoder")
register_module("superduper.ext.torch.encoder")
register_module("superduper.ext.pillow.encoder")


FACTORIES = DataTypeFactory.__subclasses__()
FACTORIES = sorted(FACTORIES, key=lambda x: 0 if x.__module__ == __name__ else 1)
