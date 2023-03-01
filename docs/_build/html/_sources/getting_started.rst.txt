Getting started
===============

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

Configuration
-------------

SuperDuperDB looks for a configuration file by default in your working directory. Here's
a minimal example with all components running on ``localhost``.

.. code-block:: json

    {
      "remote": true,
      "password": "<this-is-the-secure-password>"
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
        "port": 27017,
      }
    }