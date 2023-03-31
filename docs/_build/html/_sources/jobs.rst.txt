Jobs
====

Scheduling of training and model outputs
----------------------------------------
In order to most efficiently marshall computational resources,
SuperDuperDB may be configured to run in asynchronous mode ``{"remote": True}``.

See the section on :ref:`setting up a cluster <Setting up a SuperDuperDB cluster>` for an overview
of what this means from an infrastructural point of view.

There are several key functionalities in SuperDuperDB which cause asynchronous jobs to be
spawned in the SuperDuperDB cluster's worker pool:

* Inserting data
* Updating data
* Creating watchers
* Training semantic indexes and imputations

When a command is executed which creates jobs, its output will contain the job ids of the jobs
created. For example when inserting data, we get as many jobs as there are models in the database.
Each of these jobs will compute outputs on those data for a single model. The order of the jobs
is determined by which features are necessary for a given model. Those models with no necessary
input features which result from another model go first.

.. code-block:: python

    >>> job_ids = docs.insert_many(data)[1]
    >>> print(job_ids)
    {'resnet': ['5ebf5272-95ac-11ed-9436-1e00f226d551'],
     'visual_classifier': ['69d283c8-95ac-11ed-9436-1e00f226d551']}

The standard output of these asynchronous jobs is logged to MongoDB. One may watch this
output using:

.. code-block:: python

    >>> docs.watch_job(job_ids['resnet'])
    # ... lots of lines of stdout/ stderr
