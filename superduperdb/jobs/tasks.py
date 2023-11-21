import inspect
import traceback


def method_job(
    cfg,
    type_id,
    identifier,
    method_name,
    args,
    kwargs,
    job_id,
    dependencies=(),
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

    cfg = build_config(cfg)
    cfg.force_set('cluster.distributed', False)
    db = build_datalayer(cfg)

    component = db.load(type_id, identifier)
    method = getattr(component, method_name)
    db.metadata.update_job(job_id, 'status', 'running')

    if 'distributed' in inspect.signature(method).parameters:
        kwargs['distributed'] = False
    try:
        method(db=db, *args, **kwargs)
    except Exception as e:
        tb = traceback.format_exc()
        db.metadata.update_job(job_id, 'status', 'failed')
        db.metadata.update_job(job_id, 'msg', tb)
        raise e
    db.metadata.update_job(job_id, 'status', 'success')


def callable_job(
    cfg,
    function_to_call,
    args,
    kwargs,
    job_id,
    dependencies=(),
):
    from superduperdb.base.build import build_datalayer
    from superduperdb.base.configs import build_config

    cfg = build_config(cfg)
    cfg.force_set('cluster.distributed', False)
    db = build_datalayer(cfg)
    db.metadata.update_job(job_id, 'status', 'running')
    output = None
    if 'distributed' in inspect.signature(function_to_call).parameters:
        kwargs['distributed'] = False
    try:
        output = function_to_call(db=db, *args, **kwargs)
    except Exception as e:
        tb = traceback.format_exc()
        db.metadata.update_job(job_id, 'status', 'failed')
        db.metadata.update_job(job_id, 'msg', tb)
        raise e
    else:
        db.metadata.update_job(job_id, 'status', 'success')
    return output
