from superduperdb import logging
from superduperdb.components.datatype import DataType


def convert_model_datatype(datatype, ibis=False):
    """Convert a model datatype to a SQL vector datatype.

    :param datatype: The datatype to convert
    :param ibis: Whether to use ibis
    """
    if not isinstance(datatype, DataType):
        return datatype

    # If the datatype is a vector, convert it to a SQL vector
    if ibis and datatype.identifier.startswith('vector['):
        from superduperdb.components.vector_index import sqlvector

        new_datatype = sqlvector(shape=datatype.shape)
        logging.warn(f"Detected useing ibis, converting {datatype} to {new_datatype}")
        return new_datatype

    return datatype
