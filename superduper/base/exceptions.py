from superduper import logging


class ComponentInUseError(Exception):
    """
    Exception raised when a component is already in use.

    :param args: *args for Exception
    :param kwargs: **kwargs for Exception
    """

    pass


class ComponentInUseWarning(Warning):
    """
    Warning raised when a component is already in use.

    :param args: *args for Exception
    :param kwargs: **kwargs for Exception
    """

    pass


class BaseException(Exception):
    """BaseException which logs a message after exception.

    :param msg: msg for Exception
    """

    def __init__(self, msg):
        self.msg = msg
        logging.exception(self.msg, e=self)

    def __str__(self):
        return self.msg


class ServiceRequestException(BaseException):
    """ServiceRequestException.

    :param msg: msg for BaseException
    """


class QueryException(BaseException):
    """QueryException.

    :param msg: msg for BaseException
    """


class DatabackendException(BaseException):
    """
    DatabackendException.

    :param msg: msg for BaseException
    """


class TableNotFoundError(Exception):
    """Table not found in database.

    :param args: *args for Exception
    :param kwargs: **kwargs for Exception
    """


class MetadataException(BaseException):
    """
    MetadataException.

    :param msg: msg for BaseException
    """


class ComponentException(BaseException):
    """
    ComponentException.

    :param msg: msg for BaseException
    """


class UnsupportedDatatype(BaseException):
    """
    UnsupportedDatatype.

    :param msg: msg for BaseException
    """
