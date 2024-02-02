---
sidebar_position: 4
---

# Testing

Make sure you have the following prerequistes
- `Python 3.11`
- `Docker`

To run the `superduperdb` tests, for a particular version of the code, checkout to that version with this command:

```bash
git clone https://github.com/SuperDuperDB/superduperdb.git
git checkout <MAJOR>.<MINOR>.<PATCH>      # e.g. 0.1.0
```


Install the requirements for that version:

```bash
pip install -e '.[dev]'
```

To run the unittests run:

```bash
pytest test/unittest
```

To run the integration tests, it's necessary to build and initializing the testing environment:

```bash
make testenv_image
make testenv_init
```
When you run `make testenv_init`,  you may be prompted to  add the following host names to your `/etc/hosts/` file

```
127.0.0.1 mongodb
127.0.0.1 vector-search
127.0.0.1 cdc
127.0.0.1 scheduler
```
After copy-pasting the above host names into the `/etc/host/` file , run `make testenv_init` again to be successful.

Next, Run the integration tests with:

```bash
pytest test/integration
```

To run the additional quality checks which we run on the CI/ CD on GitHub, run:

```bash
make lint-and-type-check
```