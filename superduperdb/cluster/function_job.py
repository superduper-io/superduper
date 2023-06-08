from superduperdb.datalayer.base.imports import get_database_from_database_type
from superduperdb.cluster.annotations import decode_args, decode_kwargs
import traceback
from superduperdb.cluster.logging import handle_function_output


def function_job(database_type, database_name, function_name, args_, kwargs_, job_id):
    database = get_database_from_database_type(database_type, database_name)
    database.remote = False
    function = getattr(database, function_name)
    database.metadata.update_job(job_id, 'status', 'running')
    args_ = decode_args(database, function.signature, args_)
    kwargs_ = decode_kwargs(database, function.signature, kwargs_)
    try:
        handle_function_output(
            function,
            database,
            job_id,
            *args_,
            **kwargs_,
        )
    except Exception as e:
        tb = traceback.format_exc()
        database.metadata.update_job(job_id, 'status', 'failed')
        database.metadata.update_job(job_id, 'msg', tb)
        raise e
    database.set_job_flag(job_id, ('status', 'success'))
