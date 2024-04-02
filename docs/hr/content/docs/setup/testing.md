---
sidebar_position: 4
---

# Testing

To run the `superduperdb` tests, for a particular version of the code, checkout that version with this command:

```bash
git clone https://github.com/SuperDuperDB/superduperdb.git
git checkout <MAJOR>.<MINOR>.<PATCH>      # e.g. 0.1.0
```

Install the requirements for that version:

```bash
pip install -e .[dev]
```

To run the unittests run:

```bash
pytest -n auto test/unittest
```

To run the integration tests, it's necessary to build and initializing the testing environment:

```bash
make testenv_image
make testenv_init
```

Run the integration tests with:

```bash
pytest test/integration
```

To run the additional quality checks which we run on the CI/ CD on GitHub, run:

```bash
make lint-and-type-check
```