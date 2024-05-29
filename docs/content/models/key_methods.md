# Key methods of `Model`

All usage in `superduperdb` proceeds by changing or setting the attributes of a `Component`
and then calling `db.apply`. 

However it may be useful to know that the following methods specific to `Model` play a key role.
See [here](../apply_api/overview#key-methods) for an overview of key-methods specific to `Component`.

| Method | Description | Optional |
| --- | --- | --- |
| `Model.predict_one` | Predict on a single data-point | `FALSE` | 
| `Model.predict` | Predict on batches of data-points | `FALSE` |
| `Model.predict_in_db` | Predict and save predictions in `db` | `FALSE` |
| `Model.predict_in_db_job` | `predict_in_db` as compute job | `FALSE` |
| `Model.validate` | Validate on datasets with metrics | `FALSE` |
| `Model.validate_in_db` | Validate on datasets with metrics and save in `db` | `FALSE` |
| `Model.validate_in_db_job` | `validate_in_db` as job | `FALSE` |
| `Model.fit` | Fit on datasets | `TRUE` |
| `Model.fit_in_db` | Fit on data in `db` | `TRUE` |
| `Model.fit_in_db_job` | `.fit_in_db` as job | `TRUE` |
