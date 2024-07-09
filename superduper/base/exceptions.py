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


class RequiredPackageVersionsNotFound(ImportError):
    """
    Exception raised when one or more required packages are not found.

    :param args: *args for ImportError
    :param kwargs: **kwargs for ImportError
    """


class RequiredPackageVersionsWarning(ImportWarning):
    """
    Exception raised when one or more required packages are not found.

    :param args: *args for ImportWarning
    :param kwargs: **kwargs for ImportWarning
    """


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
