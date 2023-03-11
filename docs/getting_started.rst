Quick  start
============

Prerequisites
-------------

We assume you are running a recent version of Python3:

* Python3.9+

SuperDuperDB requires access to working deployments of:

* MongoDB
* Redis

The deployments can be existing deployments already running in your infrastructure, or
bespoke deployments which you'll use specifically for SuperDuperDB.

Here are instructions for 2 very popular systems:

Ubuntu
^^^^^^

* `MongoDB installation <https://www.mongodb.com/docs/manual/tutorial/install-mongodb-on-ubuntu/>`_
* `Redis installation <https://redis.io/docs/getting-started/installation/install-redis-on-linux/>`_

Mac OSX
^^^^^^^

* `MongoDB installation <https://www.mongodb.com/docs/manual/tutorial/install-mongodb-on-os-x/>`_
* `Redis installation <https://redis.io/docs/getting-started/installation/install-redis-on-mac-os/>`_

Installation
------------

.. code-block:: bash

    pip install superduperdb

Cluster Setup
-------------

To run SuperDuperDB in asynchronous mode, one is required to set up a SuperDuperDB cluster.
Read :ref:`here <Setting up a SuperDuperDB cluster>` for information regarding setting up a SuperDuperDB cluster.
