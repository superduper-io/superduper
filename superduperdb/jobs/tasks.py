import traceback
import typing as t

if t.TYPE_CHECKING:
    from superduperdb.base.datalayer import Datalayer


def method_job(
    cfg,
    type_id,
    identifier,
    method_name,
    args,
    kwargs,
    job_id,
    dependencies=(),
    db: t.Optional['Datalayer'] = None,
):
    """
    Run a method on a component in the database.

    :param cfg: user config
    :param type_id: type of component
    :param identifier: identifier of component
    :param method_name: name of method to run
    :param args: positional arguments to pass to the method
    :param kwargs: keyword arguments to pass to the method
    :param job_id: unique identifier for this job
    :param dependencies: other jobs that this job depends on
    :param db: datalayer to use
    """
    import sys

    sys.path.append('./')

    from superduperdb import CFG
    from superduperdb.base.build import build_datalayer

    if isinstance(cfg, dict):
        cfg = CFG(**cfg)

    # Set the compute as local since otherwise a new
    # Ray cluster would be created inside the job
    if db is None:
        db = build_datalayer(cfg=cfg, cluster__compute__uri=None)

    component = db.load(type_id, identifier)
    method = getattr(component, method_name)
    db.metadata.update_job(job_id, 'status', 'running')

    try:
        method(*args, db=db, **kwargs)
    except Exception as e:
        tb = traceback.format_exc()
        db.metadata.update_job(job_id, 'status', 'failed')
        db.metadata.update_job(job_id, 'msg', tb)
        raise e
    db.metadata.update_job(job_id, 'status', 'success')


# TODO: Is this class used?
class Logger:
    """Logger class for writing to the database.

    :param database: database to write to
    :param id_: job id
    :param stream: stream to write to
    """

    def __init__(self, database, id_, stream='stdout'):
        self.database = database
        self.id_ = id_
        self.stream = stream

    def write(self, message):
        """Write a message to the database.

        :param message: message to write
        """
        self.database.metadata.write_output_to_job(
            self.id_, message, stream=self.stream
        )

    def flush(self):
        """Flush something."""
        pass


def callable_job(
    cfg,
    function_to_call,
    args,
    kwargs,
    job_id,
    dependencies=(),
    db: t.Optional['Datalayer'] = None,
):
    """Run a function in the database.

    :param cfg: configuration
    :param function_to_call: function to call
    :param args: positional arguments to pass to the function
    :param kwargs: keyword arguments to pass to the function
    :param job_id: unique identifier for this job
    :param dependencies: other jobs that this job depends on
    :param db: datalayer to use
    """
    import sys

    from superduperdb import CFG
    from superduperdb.base.build import build_datalayer

    sys.path.append('./')

    if isinstance(cfg, dict):
        cfg = CFG(**cfg)

    # Set the compute as local since otherwise a new
    # Ray cluster would be created inside the job
    if db is None:
        db = build_datalayer(cfg=cfg, cluster__compute__uri=None)

    db.metadata.update_job(job_id, 'status', 'running')
    output = None
    try:
        output = function_to_call(*args, db=db, **kwargs)
    except Exception as e:
        tb = traceback.format_exc()
        db.metadata.update_job(job_id, 'status', 'failed')
        db.metadata.update_job(job_id, 'msg', tb)
        raise e
    else:
        db.metadata.update_job(job_id, 'status', 'success')
    return output
