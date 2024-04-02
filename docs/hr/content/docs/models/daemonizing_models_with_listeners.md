---
sidebar_position: 22
---

# Daemonizing `.predict` with listeners

In many AI applications, it's important that a catalogue of predictions is maintained for 
all data in the database, updated as soon after data-updates and streaming inserts as possible.

In order to allow developers to implement this functionality, `superduperdb` offers
the `Listener` abstraction.

## Creating listeners in-line with `.predict`

### Procedural API

To create a `Listener`, when applying a `Predictor` instance to `db`, the following 
procedural pattern applies:

```python
my_model.predict(
    X='<input-field>',
    db=db,
    select=query,
    listen=True,
)
```

### Declarative API

This is equivalent to:

```python
db.add(
    Listener(
        model=my_model,
        key='my-key',
        select=<query>,
        predict_kwargs={**<model_dot_predict_kwargs>},
    )
)
```

## Outcome

If a `Listener` has been created, whenever new data is added to `db`, 
the `Predictor` instance is loaded and predictions are evaluated on the inserted data.

:::info
In MongoDB, if [change-data-capture (CDC)](../production/change_data_capture.md) has been configured, 
data may even be inserted from third-party clients such as `pymongo`, and is nonetheless still processed
by configured `Listeners` via the CDC service.
:::