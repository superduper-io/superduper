---
sidebar_position: 1
---

# Overview of SuperDuperDB clusters

There are 3 features of a SuperDuperDB cluster:

1. When users access SuperDuperDB, they invoke the SuperDuperDB **client** which manages communication
   with the various components of the SuperDuperDB cluster.
2. Data storage: occurs in the underlying datastore, where the raw data,
   [models](Models - an extension of PyTorch models) and model outputs are stored.
3. Job management: whenever SuperDuperDB creates a [job](Jobs - scheduling of training and model outputs)
   (i.e. during data inserts, updates, downloads, and model training), a job is queued and
   then executed by a pool of **workers**.
4. Querying SuperDuperDB: when using queries which use tensor similarity, the SuperDuperDB client
   combines calls to the datastore with calls to the **vector-search** component.
5. SuperDuperDB includes a **model-server**, which may be used to serve the models which have
   been uploaded to SuperDuperDB.

The exact setup of your SuperDuperDB cluster will depend on the use-cases you
plan to execute with the cluster. Factors to consider will be:

- Latency
- Where the MongoDB deployment is located
- Whether you want to scale the cluster according to demand
- What hardware you'd like to run
- And more...