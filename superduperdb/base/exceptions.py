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

    def __init__(self, msg, exception_on_raise=True):
        self.msg = msg
        self.exception_on_raise = exception_on_raise

    def __str__(self):
        if self.exception_on_raise:
            self.exception_on_raise = False
            logging.exception(self.msg, e=self)
        return self.msg


class ComponentException(BaseException):
    '''
    ComponentException
    '''


class ComponentAddException(ComponentException):
    '''
    ComponentAddException
    '''


class ComponentReplaceException(ComponentException):
    '''
    ComponentReplaceException
    '''


class ComponentLoadException(ComponentException):
    '''
    ComponentLoadException
    '''


class DatabaseConnectionException(BaseException):
    '''
    DatabackendException
    '''


class DatabackendException(BaseException):
    '''
    DatabackendException
    '''


class MetadatastoreException(BaseException):
    '''
    MetadatastoreException
    '''


class ArtifactStoreException(BaseException):
    '''
    ArtifactStoreException
    '''


class ArtifactStoreDeleteException(ArtifactStoreException):
    '''
    ArtifactStoreException
    '''


class ArtifactStoreLoadException(ArtifactStoreException):
    '''
    ArtifactStoreException
    '''


class ArtifactStoreSaveException(ArtifactStoreException):
    '''
    ArtifactStoreException
    '''


class DatalayerException(BaseException):
    '''
    DatalayerException
    '''


class FileNotFoundException(BaseException):
    '''
    FileNotFoundException
    '''


class ServiceRequestException(BaseException):
    '''
    ServiceException
    '''


class ModelException(BaseException):
    '''
    ModelException
    '''


class VectorSearchException(ComponentException):
    '''
    VectorSearchException
    '''


class EncoderException(ComponentException):
    '''
    EncoderException
    '''


class QueryException(ComponentException):
    '''
    QueryException
    '''


class SelectQueryException(QueryException):
    '''
    SelectQueryException
    '''


class DeleteQueryException(QueryException):
    '''
    DeleteQueryException
    '''


class InsertQueryException(QueryException):
    '''
    InsertQueryException
    '''


class UpdateQueryException(QueryException):
    '''
    UpdateQueryException
    '''


class TableQueryException(QueryException):
    '''
    TableQueryException
    '''


class RawQueryException(QueryException):
    '''
    RawQueryException
    '''


class JobException(ComponentException):
    '''
    JobException
    '''


class DistributedJobException(ComponentException):
    '''
    DistributedJobException
    '''


class TaskWorkflowException(ComponentException):
    '''
    TaskWorkflowException
    '''


class MetaDataStoreDeleteException(MetadatastoreException):
    '''
    MetaDataStoreDeleteException
    '''


class MetaDataStoreJobException(MetadatastoreException):
    '''
    MetaDataStoreJobException
    '''


class MetaDataStoreCreateException(MetadatastoreException):
    '''
    MetaDataStoreCreateException
    '''


class MetaDataStoreUpdateException(MetadatastoreException):
    '''
    MetaDataStoreUpdateException
    '''


class ModelPredictException(ModelException):
    '''
    ModelPredictException
    '''


class ModelFitException(ModelException):
    '''
    ModelFitException
    '''


_query_exceptions = {
    'Delete': DeleteQueryException,
    'Update': UpdateQueryException,
    'Table': TableQueryException,
    'Insert': InsertQueryException,
    'Select': SelectQueryException,
    'RawQuery': RawQueryException,
}


def query_exceptions(query):
    query = str(query)
    for k, v in _query_exceptions.items():
        if k in query:
            return v
    else:
        return QueryException
