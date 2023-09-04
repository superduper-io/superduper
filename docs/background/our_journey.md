# How did we get here?

SuperDuperDB was born out of the need to work quickly and flexibly with AI models in 
conjunction with a database of individual "entities", as for instance commonly encountered
in content-based recommendation and semantic search.

For example, in e-Commerce, product data is often multi-faceted, involving:

- Multiple textual attributes, which are hierarchically organized and not of equivalent relevance
- Multiple images, which may have different significance for customers

In order to make sense of this data using AI, a canonical approach is to achieve a vectorial 
embedding of each of the sub-attributes/ images and also a single embedding of the entire document.
This implies:

- Maintaining a database of outputs for each of several models, applied to each of the sub-entities.
- Updating this database whenever new data comes in
- Monitoring the performance of models, as new data comes in

Once we have a database of vectors/ features for each of the entities (products) in the database, 
then these can be used in downstream tasks. For example:

- Vector-search
- Building models on top of these features, e.g. classifiers

In addition, once we have features for several collections or tables or data, these 
can be "semantically joined". This might mean, for example:

- Conditioning vector-search on additional inputs (e.g. product search, conditioned on user embeddings)
- Training cross-modality search, for instance, aligning textual queries resulting for typed queries, 
  with product features.

From this point of view, we believe that the ultimate production environment for AI, is a database, where 
all documents/ rows are game to be fed through AI models, and the outputs may be saved in the database, 
to be used downstream as inputs to further AI models, and for use in vector-search, and further approaches
to navigating data with AI models. This is an extension of the now-standard vector-database paradigm, 
which enables documents to be encoded as vectors and for these to be searched. We believe that the 
vector-database paradigm is unnecessarily limiting, as evidenced, for example, by the fact that 
vectors are simultaneously featurizations of their documents, and logically can and should be used
in additional machine learning/ AI tasks.

## Desiderata for AI-data environment

In building SuperDuperDB, considering this background, we wanted to build an AI-data environment whicih could 
do the following:

```{important}
Any row (row+column) or document (subdocument/ subfield) of data should be game to be processed by configured AI models
```

This is inline with the background of individual documents corresponding to entities in the application-domain.
For instance, in e-Commerce, one wants to be able to encode individual attributes, as well as entire products using AI-models.

```{important}
AI models should be built and defined in a way that they can read data directly from the database and produce outputs which may be written back to the database
```

Only if this is achieved, is it possible to automate computation whereby a model is reloaded and 
applied to incoming data.

```{important}
AI outputs should also be considered data, in order to enable outputs to be used as input features
```

By saving AI-outputs to the database, our system becomes essentially recursive. We can 
configure further models to consume the outputs of upstream models. This is a vital component 
of modern AI.

```{important}
AI models should be configurable to optionally depend on upstream models
```

This and the previous point are two sides of the same coin. If model outputs are treatable
as data, then we still want to order model computations, to respect the fact that certain 
models depend on the computations of their upstream models, for feasibility.

```{important}
The environment should be flexible enough to enable AI models with diverse inputs and outputs
```

This is an absolute must if we are to cater to the most promising recent areas in AI, 
including computer vision, video comprehension, speech recognition and generation, and more.

