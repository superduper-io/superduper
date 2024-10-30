# TODO not needed
from enum import Enum


class DBType(str, Enum):
    """
    DBType is an enumeration of the supported database types.

    # noqa
    """

    SQL = "SQL"
    MONGODB = "MONGODB"
