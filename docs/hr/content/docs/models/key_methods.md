# Key methods of `Model`

All usage in `superduperdb` proceeds by changing or setting the attributes of a `Component`
and then calling `db.apply`. 

However it may be useful to know that the following methods specific to `Model` play a key role.
See [here](../apply_api/overview#key-methods) for an overview of key-methods specific to `Component`.

| Method | Description | Optional |
| --- | --- | --- |
| `Model.predict_one` | | `FALSE` | 
| `Model.predict` | | `FALSE` |
| `Model.predict_in_db` | | `FALSE` |
| `Model.predict_in_db_job` | | `FALSE` |
| `Model.validate` | | `FALSE` |
| `Model.validate_in_db` | | `FALSE` |
| `Model.validate_in_db_job` | | `FALSE` |
| `Model.fit` | | `TRUE` |
| `Model.fit_in_db` | | `TRUE` |
| `Model.fit_in_db_job` | | `TRUE` |
