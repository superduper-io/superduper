# Datasets 

```{note}
Datasets are immutable snapshots of the `Datalayer`
```

When building AI models, in many cases, especially for validation, it's important that 
immutable snapshots of data are maintained for posterity. The purpose of this practice is:

- To foster reproducibility and transparency in experimentation.
- To have a permanent benchmark via which models may be compared.
- To maintain a record for auditors and oversight committees to refer to in the worst case.

Correspondingly, SuperDuperDB distinguishes between datasets and data queries from 
the `Datalayer`. The `Dataset` is designed to be fetched only once, and preserved
as an `Artifact`, saved in the configured artifact store. 

Here is an example of creating a SuperDuperDB dataset:

```python
db.add(
    Dataset(
        identifier='my-validation-set',
        select=Collection('documents').find({'_fold': 'valid'})
    )
)
```

This dataset may be used for validation of diverse `Model` instances during training and otherwise, simply
by referring to the `identifier`:

```python
model.fit(
    X='x',
    y='y',
    db=db,
    select=collection.find(),
    validation_sets=['my-validation-set'],
    metrics=['acc', 'roc']
)
```

Datasets, as with other `Component` descendants, may be created inline: 

```python
data = Dataset(
    identifier='my-validation-set',
    select=Collection('documents').find({'_fold': 'valid'})
)

model.fit(
    X='x',
    y='y',
    db=db,
    select=collection.find(),
    validation_sets=[data],
    metrics=['acc', 'roc']
)
```