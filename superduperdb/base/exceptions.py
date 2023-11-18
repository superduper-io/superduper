from superduperdb import logging


class ComponentInUseError(Exception):
    pass


class ComponentInUseWarning(Warning):
    pass


class BaseException(Exception):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        logging.error(self.msg)
        return self.msg


class ComponentException(BaseException):
    pass


class DatabaseConnectionException(BaseException):
    pass


class DatabackendException(BaseException):
    pass


class MetadatastoreException(BaseException):
    pass


class ArtifactStoreException(BaseException):
    pass


class DatalayerException(BaseException):
    pass


class FileNotFoundException(BaseException):
    pass


class ServiceException(BaseException):
    pass


class ModelException(BaseException):
    pass


class VectorSearchException(ComponentException):
    pass


class EncoderException(ComponentException):
    pass


class QueryException(ComponentException):
    pass


class SelectQueryException(ComponentException):
    pass


class DeleteQueryException(ComponentException):
    pass


class InsertQueryException(ComponentException):
    pass


class UpdateQueryException(ComponentException):
    pass


class TableQueryException(ComponentException):
    pass


class RawQueryException(ComponentException):
    pass


class JobException(ComponentException):
    pass


class DistributedJobException(ComponentException):
    pass


class TaskWorklowException(ComponentException):
    pass


query_exceptions = {
    'Delete': DeleteQueryException,
    'Update': UpdateQueryException,
    'Table': TableQueryException,
    'Insert': InsertQueryException,
    'Select': SelectQueryException,
    'RawQuery': RawQueryException,
}
