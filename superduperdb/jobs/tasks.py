import contextlib
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
    local=True,
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
    """
    from superduperdb.base.build import build_datalayer
    from superduperdb.base.configs import build_config

    if db is None:
        cfg = build_config(cfg)
        cfg.force_set('cluster.compute', None)
        db = build_datalayer(cfg)

    component = db.load(type_id, identifier)
    method = getattr(component, method_name)
    db.metadata.update_job(job_id, 'status', 'running')

    try:
        if local:
            method(*args, db=db, **kwargs)
        else:
            handle_function_output(
                method,
                db=db,
                job_id=job_id,
                args=args,
                kwargs=kwargs,
            )
    except Exception as e:
        tb = traceback.format_exc()
        db.metadata.update_job(job_id, 'status', 'failed')
        db.metadata.update_job(job_id, 'msg', tb)
        raise e
    db.metadata.update_job(job_id, 'status', 'success')


class Logger:
    def __init__(self, database, id_, stream='stdout'):
        self.database = database
        self.id_ = id_
        self.stream = stream

    def write(self, message):
        self.database.metadata.write_output_to_job(
            self.id_, message, stream=self.stream
        )

    def flush(self):
        pass


def handle_function_output(function, db, job_id, args, kwargs):
    with contextlib.redirect_stdout(Logger(db, job_id)):
        with contextlib.redirect_stderr(Logger(db, job_id, stream='stderr')):
            return function(db=db, *args, **kwargs)


def callable_job(
    cfg,
    function_to_call,
    args,
    kwargs,
    job_id,
    dependencies=(),
    local=True,
    db: t.Optional['Datalayer'] = None,
):
    from superduperdb.base.build import build_datalayer
    from superduperdb.base.configs import build_config

    if db is None:
        CFG = build_config(cfg)
        cfg.force_set('cluster.compute', None)
        db = build_datalayer(CFG)
    db.metadata.update_job(job_id, 'status', 'running')
    output = None
    try:
        if local:
            output = function_to_call(*args, db=db, **kwargs)
        else:
            output = handle_function_output(
                function_to_call,
                db=db,
                job_id=job_id,
                args=args,
                kwargs=kwargs,
            )
    except Exception as e:
        tb = traceback.format_exc()
        db.metadata.update_job(job_id, 'status', 'failed')
        db.metadata.update_job(job_id, 'msg', tb)
        raise e
    else:
        db.metadata.update_job(job_id, 'status', 'success')
    return output
