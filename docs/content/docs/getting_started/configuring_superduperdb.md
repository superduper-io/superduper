# Configuration

SuperDuperDB ships with configuration via the `pydantic`. Read more about `pydantic` [here](https://docs.pydantic.dev/latest/).
Using `pydantic`, SuperDuperDB comes with wall-to-wall defaults in `superduperdb.base.config.py`.

These configurations may be overridden by:

- import `superduperdb.CFG` and overwriting values in that object
- environment variable 'SUPERDUPERB_...`
- by a file `configs.json` in the
working directory.

Here are the key configuration classes: