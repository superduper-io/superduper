---
sidebar_position: 22
---

# Computing model outputs with listeners

### Declarative API

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