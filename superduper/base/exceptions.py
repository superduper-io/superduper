# TODO add more exceptions
# for example ArtifactNotFoundError
from superduper import logging


class ComponentInUseError(Exception):
    """
    Exception raised when a component is already in use.

    :param args: *args for Exception
    :param kwargs: **kwargs for Exception
    """


class DatabackendError(Exception):
    """Exception raised when a databackend error occurs.

    # noqa
    """


class BaseException(Exception):
    """BaseException which logs a message after exception.

    :param msg: msg for Exception
    """

    def __init__(self, msg):
        self.msg = msg
        logging.exception(self.msg, e=self)

    def __str__(self):
        return self.msg


class TableNotFoundError(Exception):
    """Table not found in database.

    :param args: *args for Exception
    :param kwargs: **kwargs for Exception
    """


class UnsupportedDatatype(BaseException):
    """
    UnsupportedDatatype.

    :param msg: msg for BaseException
    """


class MissingSecretsException(BaseException):
    """
    Missing secrets.

    :param msg: msg for BaseException
    """


class ServiceRequestException(BaseException):
    """
    Service request exception.

    :param msg: msg for BaseException
    """
