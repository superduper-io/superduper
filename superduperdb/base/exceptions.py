from superduperdb import logging


class ComponentInUseError(Exception):
    pass


class ComponentInUseWarning(Warning):
    pass


class BaseException(Exception):
    '''
    BaseException which logs a message after
    exception
    '''

    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        logging.error(self.msg)
        return self.msg


class ComponentException(BaseException):
    '''
    ComponentException
    '''


class ComponentAddException(ComponentException):
    '''
    ComponentAddException
    '''

    pass


class ComponentReplaceException(ComponentException):
    '''
    ComponentReplaceException
    '''

    pass


class ComponentLoadException(ComponentException):
    '''
    ComponentLoadException
    '''

    pass


class DatabaseConnectionException(BaseException):
    '''
    DatabackendException
    '''

    pass


class DatabackendException(BaseException):
    '''
    DatabackendException
    '''

    pass


class MetadatastoreException(BaseException):
    '''
    MetadatastoreException
    '''

    pass


class ArtifactStoreException(BaseException):
    '''
    ArtifactStoreException
    '''

    pass


class DatalayerException(BaseException):
    '''
    DatalayerException
    '''

    pass


class FileNotFoundException(BaseException):
    '''
    FileNotFoundException
    '''

    pass


class ServiceException(BaseException):
    '''
    ServiceException
    '''

    pass


class ModelException(BaseException):
    '''
    ModelException
    '''

    pass


class VectorSearchException(ComponentException):
    '''
    VectorSearchException
    '''

    pass


class EncoderException(ComponentException):
    '''
    EncoderException
    '''

    pass


class QueryException(ComponentException):
    '''
    QueryException
    '''

    pass


class SelectQueryException(QueryException):
    '''
    SelectQueryException
    '''

    pass


class DeleteQueryException(QueryException):
    '''
    DeleteQueryException
    '''

    pass


class InsertQueryException(QueryException):
    '''
    InsertQueryException
    '''

    pass


class UpdateQueryException(QueryException):
    '''
    UpdateQueryException
    '''

    pass


class TableQueryException(QueryException):
    '''
    TableQueryException
    '''

    pass


class RawQueryException(QueryException):
    '''
    RawQueryException
    '''

    pass


class JobException(ComponentException):
    '''
    JobException
    '''

    pass


class DistributedJobException(ComponentException):
    '''
    DistributedJobException
    '''

    pass


class TaskWorkflowException(ComponentException):
    '''
    TaskWorkflowException
    '''

    pass


query_exceptions = {
    'Delete': DeleteQueryException,
    'Update': UpdateQueryException,
    'Table': TableQueryException,
    'Insert': InsertQueryException,
    'Select': SelectQueryException,
    'RawQuery': RawQueryException,
}
