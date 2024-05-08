import json
import typing as t

import numpy as np

from superduperdb import logging
from superduperdb.components.datatype import (
    DataType,
    dill_base,
    json_serializer,
    serializers,
)
from superduperdb.components.schema import Schema
from superduperdb.ext.numpy.encoder import array

try:
    import torch

    torch_cls = (torch.Tensor, torch.nn.Module)
except ImportError:
    torch_cls = ()

_BASE_TYPES = (
    int,
    str,
    float,
    bool,
    bytes,
    bytearray,
)


def infer_datatype(data: t.Any, ibis=False) -> t.Optional[DataType]:
    """
    Infer the datatype of a given data object
    If the data object is a base type, return None,
    Otherwise, return the inferred datatype

    :param data: The data object
    """

    datatype = dill_base

    if isinstance(data, _BASE_TYPES):
        if not ibis:
            datatype = None
        else:
            from superduperdb.backends.ibis.field_types import dtype

            datatype = dtype(type(data))

    elif isinstance(data, np.ndarray):
        dtype = data.dtype.name
        assert isinstance(dtype, str), f"Expected dtype to be a string, got {dtype}"
        shape = tuple(data.shape)
        datatype = array(dtype, shape)
        serializers[datatype.identifier] = datatype

    elif isinstance(data, torch_cls):
        import torch

        if isinstance(data, torch.Tensor):
            from superduperdb.ext.torch.encoder import tensor

            dtype = data.dtype
            shape = data.shape
            datatype = tensor(dtype, shape)
            serializers[datatype.identifier] = datatype
        else:
            datatype = serializers["torch"]

    elif isinstance(data, dict):
        try:
            json.dumps(data)
            # Only use json_serializer when using ibis.
            if ibis:
                datatype = json_serializer
                serializers[datatype.identifier] = datatype
            else:
                datatype = None
        except Exception:
            pass

    if datatype is not None:
        logging.info(f"Inferred datatype: {datatype} for data: {data}")

    return datatype


def infer_schema(
    data: t.Mapping[str, t.Any],
    identifier: t.Optional[str] = None,
    ibis=False,
) -> Schema:
    """
    Infer a schema from a given data object

    :param data: The data object
    :param identifier: The identifier for the schema, if None, it will be generated
    :return: The inferred schema
    """

    assert isinstance(data, dict), "Data must be a dictionary"

    schema_data = {}
    for k, v in data.items():
        data_type = infer_datatype(v, ibis)
        if data_type is not None:
            schema_data[k] = data_type

    if identifier is None:
        key_value_pairs = sorted(schema_data.items())
        id_ = "&".join([f"{k}={v.identifier}" for k, v in key_value_pairs])

        identifier = f"schema_{id_}"

    return Schema(identifier=identifier, fields=schema_data)
