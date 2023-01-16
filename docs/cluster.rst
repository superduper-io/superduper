*********************************
Setting up a SuperDuperDB cluster
*********************************

The exact setup of your SuperDuperDB cluster will depend on the requires of the use-cases you
plan to execute with the cluster. Factors to consider will be:

- Latency
- Where the MongoDB deployment is located
- Whether you want to scale the cluster according to demand
- What hardware you'd like to run
- And more...

++++++++++
Components
++++++++++

The basic topology of a SuperDuperDB cluster is given in the graphic below:

.. image:: img/architecture.png
    :width: 80%

Client
======

The client is analagous to the client used in MongoDB. This is the programmer's interface to
the SuperDuperDB cluster and provides a unified user-experience very similar to the MongoDB
user experience.

MongoDB
=======

This is a standard MongoDB deployment. The deployment can either sit in the same infrastructure
as the remained of the SuperDuperDB, or it can be situated remotely. Performance and latency
concerns here will play a role in which version works best and is most convenient.

Master/ Vector Search
=====================

This node returns real time semantic index search outputs to the client. The node loads
model outputs which are of vector or tensor type, and creates an in-memory search index over
them.

Job-master
==========

This node schedules jobs to run on the job-workers. These jobs compute model outputs and
also run training to create new models.

Job-worker
==========

These nodes perform the long computations necessary to update model outputs when new data
come in, and also perform model training for models which are set up to be trained on creation.

+++++++++++++++++++++++++
Basic local cluster setup
+++++++++++++++++++++++++

The following ``config.json`` and ``supervisord.conf`` configuration runs a test cluster
on the ``localhost``:

.. code-block:: json

    {
      "remote": true,
      "master": {
        "host": "localhost",
        "port": 5001
      },
      "jobs": {
        "host": "localhost",
        "port": 5002
      },
      "redis": {
        "host": "localhost",
        "port": 6379
      },
      "mongodb": {
        "host": "localhost",
        "port": 27017
      }
    }

.. code-block::

    [supervisord]

    logfile=/dev/null
    logfile_maxbytes=0

    [program:master]

    command=/bin/bash -c "python3 -m superduperdb.servers.master $(cat config.json | jq .master.port)"
    process_name=%(program_name)s_%(process_num)s
    numprocs=1
    stdout_logfile=logs/master.out
    stderr_logfile=logs/master.out
    autorestart=false
    startretries=1

    [program:jobs-master]

    command=/bin/bash -c "python3 -m superduperdb.servers.jobs $(cat config.json | jq .hash_set.port)"
    process_name=%(program_name)s_%(process_num)s
    numprocs=1
    stdout_logfile=logs/jobs-master.out
    stderr_logfile=logs/jobs-master.out
    autorestart=false
    startretries=1

    [program:jobs-worker]

    command=/bin/bash -c "rq worker -v --url redis://:@localhost:$(cat config.json | jq .redis.port)"
    process_name=%(program_name)s_%(process_num)s
    numprocs=2
    stdout_logfile=logs/jobs-worker.out
    stderr_logfile=logs/jobs-worker.out
    autorestart=false
    startretries=1

    [program:redis]

    command=/bin/bash -c "redis-server --port $(cat config.json | jq .redis.port)"
    process_name=%(program_name)s_%(process_num)s
    numprocs=1
    stdout_logfile=logs/redis.out
    stderr_logfile=logs/redis.out
    autorestart=false
    startretries=1

The cluster may be started with this command:

.. code-block:: bash

    OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES supervisord -n
