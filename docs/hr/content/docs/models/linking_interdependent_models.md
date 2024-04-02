---
sidebar_position: 24
---

# Configuring models to ingest features from other models

Sometimes the outputs of one model should be "chained together" to become inputs of another model.
Typical examples of this are:

- Chunking and vectorizing text data, to become input to a retrieval augmented LLM.
- Transfer learning in computer vision, where prepared features become input to a shallow classifier

## Procedural API

In procedural mode, pass the output of one `.predict` as a dependency of
the second, as well as specifying that the model should ingest the outputs 
of the first, using the `.outputs` query.

```python
j1 = m1.predict(X='my_key_1', select=collection.find())
m2.predict(
    X='my_key_1',
    select=collection.find().outputs('my_key_1', m1.identifier),
    dependencies=(j1,),
)
```

In `CFG.production = True` mode, the `j1` object is a `Job` object, which 
doesn't block the main thread of the program. The second `.predict` waits
for this `Job` to finish before starting.

## Declarative API

With the declarative API this 
behaviour may be achieved by linking the `.outputs` of one `Listener` as the `select=...` of another:

```python
l1 = Listener(
    model_1,
    key='my_key_1',
    select=collection.find(),
)

l2 = Listener(
    model_2,
    key='my_key_1',
    select=l1.outputs,
)
```

This implies that whenever data is inserted to `collection`, `model_1` will compute outputs on that data first, 
which will subsequently be consumed by `model_2` as inputs; its outputs will then also be saved to `db`.