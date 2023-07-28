# SuperDuperDB Usage

## Important Concepts

The SuperDuperDB workflow looks like this:

1. The user adds data to the SuperDuperDB **datalayer**, which may include user defined **encoders**, and external web or other content.
2. The user uploads one or more **models** including weights or parameters to SuperDuperDB, configuring which models should be applied to which data, by linking models to a
   query and key, in a **watcher**. 
3. (Optionally) SuperDuperDB creates a **job** to train the uploaded models on the data contained in the database.
4. SuperDuperDB applies the **models** to the configured data and the outputs are stored in the documents to which the **models** were applied.
5. SuperDuperDB **watches** for when new data comes in, when the **models** which have already been uploaded are reactivated.
6. (Optionally) SuperDuperDB retrains **models** on the latest data.
7. SuperDuperDB creates a **job** to apply the **models** to data, which has yet to be processed, and the outputs
   are stored in the documents to which the **models** were applied.
8. At inference time, the outputs of the applied **models** may be queried using classifical DB queries,
   or, if the outputs are vectors, searched using a **vector-index**.

![](../img/cycle-linear.svg)

The key concepts are detailed below.

## Contents

```{toctree}
:maxdepth: 3

datalayer
queries
encoders
models
datasets
vector_search
jobs
```
