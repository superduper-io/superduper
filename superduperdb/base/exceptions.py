from superduperdb import logging


class ComponentInUseError(Exception):
    """Exception raised when a component is already in use."""

    pass


class ComponentInUseWarning(Warning):
    """Warning raised when a component is already in use."""

    pass


class BaseException(Exception):
    """BaseException which logs a message after exception.

    :param msg: The message to log.
    """

    def __init__(self, msg):
        self.msg = msg
        logging.exception(self.msg, e=self)

    def __str__(self):
        return self.msg


class RequiredPackageVersionsNotFound(ImportError):
    """Exception raised when one or more required packages are not found."""


class RequiredPackageVersionsWarning(ImportWarning):
    """Exception raised when one or more required packages are not found."""


class ServiceRequestException(BaseException):
    """ServiceRequestException."""


class QueryException(BaseException):
    """QueryException."""


class DatabackendException(BaseException):
    """DatabackendException."""


class MetadataException(BaseException):
    """MetadataException."""


class ComponentException(BaseException):
    """ComponentException."""
