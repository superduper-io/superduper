## How To Contribute :rocket:


Hello! :wave: 

Thank you for considering contributing to `superduper`. There are many ways to contribute, and they are not limited to writing code. We welcome all contributions such as:

- bug reports
- documentation improvements
- enhancement suggestions
- expanding the reusable-snippets and use-cases

This project is intended to be a community effort, and it won't be possible without your support and enthusiasm.

## Where to Start? 

If you're new to open-source development, we recommend going through the [GitHub “issues” tab](https://github.com/superduper-io/superduper/issues) to find items that interest you. Once you’ve found something interesting, the next step is to create your development environment.

## Installation and setup

Once you've 'forked' and 'cloned' the code to your local machine, please follow these steps:

Get the code on your local:

```shell
# Clone and change location directory of the superduper repository, change the `<FORKED_NAME>` to your GitHub id
git clone git@github.com:<FORKED_NAME>/superduper.git
cd superduper
```

Set up your python environment:


```shell
# Create your Python virtual environment
python3 -m venv .venv

# Activate the Python virtual environment
. .venv/bin/activate  
```

Install the dependencies:

```shell
# Install pip-tools and latest version of pip
pip install --upgrade pip-tools

# Install the SuperDuper project in editable mode, along with the necessary developer tools.
pip install -e '.[test]'
```

Install the required plugins for your development environment.
```shell
# The mongodb plugin is required for the tests (nosql)
pip install -e 'plugins/mongodb[test]'
# The ibis and sqlalchemy plugins are required for the tests (sql)
pip install -e 'plugins/ibis[test]'
pip install -e 'plugins/sqlalchemy[test]'
```

You can install any additional plugins needed for your development environment.

## Branch workflow

We follow something called a "fork and pull request" workflow for collaborating on our project. See [here](https://gist.github.com/Chaser324/ce0505fbed06b947d962) for a great overview on what some of these mysterious terms mean! 

## Running the tests

### Unittests

These tests check that there are no basic programming errors in how 
classes and functions work internally.

```shell
make unit_testing
```

### Extension integration tests

We use specific use cases to test the entire system.

```shell
make usecase_testing
```

### Plugin tests

We maintain a set of plugins that are tested independently.. If you change the plugin code, you can run the tests for that plugin.

```shell
export PYTHONPATH=./
# Install the plugin you want to test
pip install -e 'plugins/<PLUGIN_NAME>[test]'
# Run the tests
pytest plugins/<PLUGIN_NAME>/plugin_test
```


## Lint and type-check the code

We use `black` for code formatting, `run` for linting, and `mypy` for type-checking.

You can run the following commands to check the code:

```
make lint-and-type-check
```

If you want to format the code, you can run the following command:
```
make fix-and-check
```

## Contributing to new plugins and templates

The `superduper` project is designed to be extensible and customizable. You can create new plugins and templates to extend the functionality of the project.

### Create a new plugin

Plugins are Python packages that extend the functionality of the SuperDuper project.
More details about the plugins can be found in the documentation:

- [Data plugins](https://docs.superduper.io/docs/data_plugins/intro)
- [AI plugins](https://docs.superduper.io/docs/category/ai-plugins)

If you want to create a new plugin, you can follow the steps below:

1. Copy `plugins/template` to `plugins/<PLUGIN_NAME>`
2. Modify the name of the plugin in the following names:
   - Plugin name in `pyproject.toml`
   - Package name `superduper_<PLUGIN_NAME>`
3. Write the code for the plugin
4. Create the tests directory `plugin_test/` and write the tests
5. Modify the `__version__` in the `superduper_<PLUGIN_NAME>/__init__.py` file. We use this version for releasing the plugin.

We follow x.y.z versioning for the plugins, where the x.y version matches the superduper x.y version.

We increment the z version for new releases and increment the x.y version for major releases.

### Create a new template

Templates are reusable components that facilitate the quick and easy building of AI applications.
More details about the templates can be found in the documentation:

- [Templates](https://docs.superduper.io/docs/category/templates)

If you want to create a new template, you can follow the steps below:

1. Create a new directory in `templates/<TEMPLATE_NAME>`
2. Create a `build.ipynb` file with the code to build the template
   1. Present the build process in a Jupyter notebook
   2. Package all the components into the application
   3. Build a template from the application
   4. Export the template using `template.export('.')`, and then you can get `component.json` in the directory

## Contributing to the documentation
We maintain the documentation in the [superduper-docs](https://github.com/superduper-io/superduper-docs) repository.


Please go to the repository and create a pull request with the changes you want to make.

### Fork and clone the repository

```
git clone git@github.com:<FORKED_NAME>/superduper-docs.git
cd superduper-docs
```

### Updating the documentation

For most document updates and additions, you can directly modify the files under `superduper-docs/content`.

But there are some special cases:

- Plugin documentation
- Template documentation

**Creating of updating documentation for a plugin**

After you create or update a plugin, you need to update the documentation.

1. Update the `README.md` file in the plugin directory.
2. Copy the file to the `superduper-docs/content/plugins` directory and change the file name to `<PLUGIN_NAME>.md`.

**Creating of updating documentation for a template**

After you create or update a template, you need to update the documentation.

We can use the `to_docusaurus_markdown.py` script to convert the Jupyter notebook to the markdown file.

```
python3 to_docusaurus_markdown.py <Your superduper project path>/templates/<TEMPLATE_NAME>/build.ipynb
```

Then a new markdown file `<Your superduper project path>/templates/<TEMPLATE_NAME>/build.md`.

You can copy the file to the `superduper-docs/content/templates` directory and change the file name to `<TEMPLATE_NAME>.md`.




## Create an issue

If you have an unsolvable problem or find a bug with the code, we
would love it if you could create a useful [issue on GitHub](https://github.com/superduper-io/superduper-stealth/issues).

Creating a useful issue, is itself a useful skill. Think about following these pointers:

- Add the "bug label" to flag the issue as a bug
- Make sure the issue contains the ***minimal code*** needed to create the issue:
  - Remove padding code, unnecessary setup etc. 
  - Make it as easy as possible to recreate the problem.
- Always include the traceback in the issue
- To flag the issue to the team, escalate this in the Slack channels
- Tag relevant people who have worked on that issue, or know the feature

## CI workflow

We have two ci workflows that run on the pull requests:

- Code Testing: Unittests, Extension Integration Tests. The code testing is run on every pull request.
- Plugin Testing: Plugin tests. The plugin testing only runs on the pull requests that change the plugin code.

Additionally, we have a plugin release workflow that runs on the main branch. The plugin release workflow will release the plugins to the PyPI.

### Code Testing

1. Lint and type-check
2. Unit Testing, will run the unittests with mongodb and sqlite
3. Usecase Testing, will run the usecases with mongodb and sqlite

### Plugin Testing

The plugin use matrix testing to test the plugins which are changed in the pull request.

1. Lint and type-check
2. Run `plugins/<PLUGIN_NAME>/plugin_test`
3. Run the base testing(Optional): If the config file `plugins/<PLUGIN_NAME>/plugin_test/config.yaml` exists, the unittests and usecases will be run with the plugin.

### Release plugins on PyPI

The workflow detects commit messages like `[PLUGINS] Bump Version [plugin_1 | plugin_2]` and releases the corresponding plugins to PyPI.

## Getting Help 🙋

[![Slack](https://img.shields.io/badge/Slack-superduper-8A2BE2?logo=slack)](https://join.slack.com/t/superduper-public/shared_invite/zt-1yodhtx8y-KxzECued5QBtT6JFnsSNrQ)
[![Issues](https://img.shields.io/badge/Issues-superduper-8A2BE2?logo=github)](https://github.com/superduper-io/superduper/issues)
[![Wiki](https://img.shields.io/badge/Project%20Wiki-superduper-8A2BE2?logo=github)](https://github.com/superduper-io/superduper/wiki)

If you have any problems please contact a maintainer or community volunteer. The GitHub issues or the Slack channel are a great place to start. We look forward to seeing you there! :purple_heart:
