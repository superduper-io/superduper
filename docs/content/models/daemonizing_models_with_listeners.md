# Computing model outputs with listeners

## Usage

```python
db.apply(
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
the `Model` instance is loaded and outputs as computed with `Model.predict` are evaluated on the inserted data.

:::info
If [change-data-capture (CDC)](../production/change_data_capture.md) has been configured, 
data may even be inserted from third-party clients such as `pymongo`, and is nonetheless still processed
by configured `Listeners` via the CDC service.
:::