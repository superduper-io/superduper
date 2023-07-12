import contextlib
import traceback


def method_job(
    cfg,
    variety,
    identifier,
    method_name,
    args,
    kwargs,
    job_id,
    dependencies=(),
):
    from superduperdb.datalayer.base.build import build_datalayer

    db = build_datalayer(cfg)
    component = db.load(variety, identifier)
    method = getattr(component, method_name)
    db.metadata.update_job(job_id, 'status', 'running')
    try:
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
):
    from superduperdb.datalayer.base.build import build_datalayer

    db = build_datalayer(cfg)
    db.metadata.update_job(job_id, 'status', 'running')
    try:
        handle_function_output(
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
    db.metadata.update_job(job_id, 'status', 'success')
