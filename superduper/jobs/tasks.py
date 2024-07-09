import typing as t

if t.TYPE_CHECKING:
    from superduper.base.datalayer import Datalayer
    from superduper.components.component import Component


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
    component: 'Component' = None,
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
    :param component: component to run method on
    """
    import sys

    sys.path.append('./')

    from superduper import CFG
    from superduper.base.build import build_datalayer

    if isinstance(cfg, dict):
        cfg = CFG(**cfg)

    # Set the compute as local since otherwise a new
    # Ray cluster would be created inside the job
    if db is None:
        db = build_datalayer(cfg=cfg, cluster__compute___path=None)

    if not component:
        component = db.load(type_id, identifier)
        from superduper.components.component import Component

        component = t.cast(Component, component)
        component.unpack()

    component.unpack()

    method = getattr(component, method_name)
    db.metadata.update_job(job_id, 'status', 'running')

    try:
        method(*args, db=db, **kwargs)
    except Exception as e:
        db.metadata.update_job(job_id, 'status', 'failed')
        raise e
    db.metadata.update_job(job_id, 'status', 'success')


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

    from superduper import CFG
    from superduper.base.build import build_datalayer

    sys.path.append('./')

    if isinstance(cfg, dict):
        cfg = CFG(**cfg)

    # Set the compute as local since otherwise a new
    # Ray cluster would be created inside the job
    if db is None:
        db = build_datalayer(cfg=cfg, cluster__compute___path=None)

    db.metadata.update_job(job_id, 'status', 'running')
    output = None
    try:
        output = function_to_call(*args, db=db, **kwargs)
    except Exception as e:
        db.metadata.update_job(job_id, 'status', 'failed')
        raise e
    else:
        db.metadata.update_job(job_id, 'status', 'success')
    return output
