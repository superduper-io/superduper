# TODO - remove in favour of annotations on classes
import importlib
import typing as t

import numpy as np

from superduper import logging
from superduper.base.exceptions import UnsupportedDatatype
from superduper.components.datatype import (
    INBUILT_DATATYPES,
    JSON,
    DataTypeFactory,
)

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


def infer_datatype(data: t.Any, db: 'Datalayer') -> t.Optional[str]:
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
        datatype: str = type(data).__name__
        logging.debug(f"Inferred base type: {datatype} for data:", data)
        return datatype

    for factory in FACTORIES:
        if factory.check(data):
            datatype: str = factory.create(data)
            logging.debug(f"Inferred datatype: {datatype} for data: {data}")
            break

    if datatype is None:
        datatype = 'dillencoder'
        datatype_impl = INBUILT_DATATYPES[datatype]
        try:
            encoded_data = datatype_impl.encode_data(
                data, builds={}, blobs={}, files={}
            )
            decoded_data = datatype_impl.decode_data(encoded_data, builds={}, db=db)
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
) -> t.Dict:
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
    return schema_data


class VectorTypeFactory(DataTypeFactory):
    """A factory for Vector datatypes # noqa."""

    @staticmethod
    def check(data: t.Any) -> bool:
        """Check if the data is able to be encoded by the JSON serializer.

        :param data: The data object
        """
        return isinstance(data, np.ndarray) and len(data.shape) == 1

    @staticmethod
    def create(data: t.Any) -> str:
        """Create a vector datatype.

        :param data: The data object.
        """
        return f'vector[{str(data.dtype)}:{len(data)}]'


class JsonDataTypeFactory(DataTypeFactory):
    """A factory for JSON datatypes # noqa."""

    @staticmethod
    def check(data: t.Any) -> bool:
        """Check if the data is able to be encoded by the JSON serializer.

        :param data: The data object.
        """
        try:
            JSON().encode_data(data, builds={}, blobs={}, files={})
            return True
        except Exception:
            return False

    @staticmethod
    def create(data: t.Any) -> str:
        """Create a JSON datatype.

        :param data: The data object.
        """
        return 'json'


register_module("superduper_torch.encoder")
register_module("superduper_pillow.encoder")


FACTORIES = DataTypeFactory.__subclasses__()
FACTORIES = sorted(FACTORIES, key=lambda x: 0 if x.__module__ == __name__ else 1)
