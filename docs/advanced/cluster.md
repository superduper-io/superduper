# Cluster

There are 3 features of a SuperDuperDB cluster:

1. When users access SuperDuperDB, they invoke the SuperDuperDB **client** which manages communication
   with the various components of the SuperDuperDB cluster.
2. Data storage: occurs in **MongoDB**, where the raw data,
   [models](Models - an extension of PyTorch models) and model outputs are stored.
3. Job management: whenever SuperDuperDB creates a [job](Jobs - scheduling of training and model outputs)
   (i.e. during data inserts, updates, downloads, and model training), a job is queued and
   then executed by a pool of **workers**.
4. Querying SuperDuperDB: when using queries which use tensor similarity, the SuperDuperDB client
   combines calls to MongoDB with calls to the **linear-algebra** component.
5. SuperDuperDB includes a **model-server**, which may be used to serve the models which have
   been uploaded to SuperDuperDB.

The exact setup of your SuperDuperDB cluster will depend on the use-cases you
plan to execute with the cluster. Factors to consider will be:

- Latency
- Where the MongoDB deployment is located
- Whether you want to scale the cluster according to demand
- What hardware you'd like to run
- And more...

## Components

The basic topology of a SuperDuperDB cluster is given in the graphic below:

![SuperDuperDB cluster topology](img/architecture_now.png)

### Client

The client is analagous to the client used in MongoDB. This is the programmer's interface to
the SuperDuperDB cluster and provides a unified user-experience very similar to the MongoDB
user experience.

### MongoDB

This is a standard MongoDB deployment. The deployment can either sit in the same infrastructure
as the remainder of the SuperDuperDB cluster, or it can be situated remotely. Performance and latency
concerns here will play a role in which version works best and is most convenient.

### Linear algebra 

This node returns real time semantic index search outputs to the client. The node loads
model outputs which are of vector or tensor type, and creates an in-memory search index over
them.

### Model-server

SuperDuperDB contains a component which serves models which has been created.

### Worker

These nodes perform the long computations necessary to update model outputs when new data
come in, and also perform model training for models which are set up to be trained on creation.

## Basic local cluster setup

The following `config.json` and `supervisord.conf` configuration runs a test cluster
on the `localhost`:

```json
{
  "remote": true,
  "linear_algebra": {
    "host": "localhost",
    "port": 5001
  },
  "model_server": {
    "host": "localhost",
    "port": 5003
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
