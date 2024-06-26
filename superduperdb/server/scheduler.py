import typing as t

from pydantic import BaseModel

from superduperdb import logging
from superduperdb.base.datalayer import Datalayer
from superduperdb.server.app import DatalayerDependency, SuperDuperApp

app = SuperDuperApp('scheduler', port=8181)


class JobSubmit(BaseModel):
    """A vector item model."""
    identifier: str
    dependencies: t.List[str]
    compute_kwargs: t.Dict[str, t.Any]


@app.startup
def scheduler_startup(db: Datalayer) -> Datalayer:
    """Start the cdc server.

    :param db: Datalayer instance.
    """
    db.compute.connect()
    return db


@app.add("/job/create/component_hook", status_code=200, method='get')
def component_hook(component_id: str, db: Datalayer = DatalayerDependency()):
    db.compute.component_hook(component_id, to='create')

@app.add("/job/submit", status_code=200, method='post')
def submit(job: JobSubmit, db: Datalayer = DatalayerDependency()):
    from superduperdb.jobs.job import ComponentJob, FunctionJob
    
    logging.info(f"Running remote job {job.identifier}")
    logging.info(f"Dependencies: {job.dependencies}")
    logging.info(f"Compute kwargs: {job.compute_kwargs}")


    assert db.compute.remote is True, "Compute is not a distributed backend type."

    deps_future = []
    for dep in job.dependencies:
        if future:=db.compute._futures_collection.get(dep, None):
            deps_future.append(future)

    info = db.metadata.get_job(job.identifier)

    args = info['args']
    kwargs = info['kwargs']
    path = info['_path']
    logging.info(f"Running remote task: {info}")

    if 'ComponentJob' in path:
        component_identifier = info['component_identifier']
        method_name = info['method_name']
        type_id = info['type_id']
        job = ComponentJob(
            args=args,
            kwargs=kwargs,
            method_name=method_name,
            component_identifier=component_identifier,
            type_id=type_id,
            db=db,
        )
    elif 'FunctionJob' in path:
        import importlib

        function = path.split('/')[-1]
        import_path, function = function.split(';')
        callable = getattr(importlib.import_module(import_path), function)

        job = FunctionJob(args=args, kwargs=kwargs, callable=callable, db=db)
    else:
        raise TypeError
    job.submit(dependencies=deps_future, update_job=True)

    return {'message': 'Job created successfully'}


