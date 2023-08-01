# SuperDuperDB Usage

## Important Concepts

The SuperDuperDB workflow looks like this:

1. The user adds data to the **[DB](DB)**, which may include user defined **[encoders](encoder)**, and external web or other content.
2. The user uploads one or more **[models](modelz)** including weights or parameters to SuperDuperDB, configuring which models should be applied to which data, by linking models to a
   query and key, in a **[listener](listener)**. 
3. (Optionally) SuperDuperDB creates a **[job](jobs)** to train the uploaded models on the data contained in the database.
4. SuperDuperDB applies the **[models](modelz)** to the configured data and the outputs are stored in the documents to which the **[models](modelz)** were applied.
5. SuperDuperDB **[listenes](listener)** for when new data comes in, when the **[models](modelz)** which have already been uploaded are reactivated.
6. (Optionally) SuperDuperDB retrains **[models](modelz)** on the latest data.
7. SuperDuperDB creates a **[job](jobs)** to apply the **[models](modelz)** to data, which has yet to be processed, and the outputs
   are stored in the documents to which the **[models](modelz)** were applied.
8. At inference time, the outputs of the applied **[models](modelz)** may be queried using classical DB queries,
   or, if the outputs are vectors, searched using a **[vector-index](vectorsearch)**.

![](../img/cycle-linear.svg)

The key concepts are detailed below.

## Contents

```{toctree}
:maxdepth: 3

db
queries
encoders
models
datasets
vector_index
jobs
```
