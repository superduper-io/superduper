import typing as t

from pydantic import BaseModel

import json

from superduperdb import CFG
from superduperdb import CFG, logging
from superduperdb.base.datalayer import Datalayer
from superduperdb.server.app import DatalayerDependency, SuperDuperApp

app = SuperDuperApp('scheduler', port=8181)


class JobSubmit(BaseModel):
    """A vector item model."""
    identifier: str
    dependencies: t.List[str]
    compute_kwargs: t.Dict[str, t.Any]

@app.add("/job/submit", status_code=200, method='post')
def submit(job: JobSubmit, db: Datalayer = DatalayerDependency()):
    

    logging.info(f"Running remote job {job.identifier}")
    logging.info(f"Dependencies: {job.dependencies}")
    logging.info(f"Compute kwargs: {job.compute_kwargs}")


    assert db.compute.remote is True, "Compute is not a distributed backend type."
    deps_future = []
    for dep in job.dependencies:
        if future:=db.compute._futures_collection.get(dep, None):
            deps_future.append(future)

    db.compute.execute_task(
        job.identifier, dependencies=deps_future, compute_kwargs=job.compute_kwargs
    )

    return {'message': 'Job created successfully'}
