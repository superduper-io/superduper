# `Plugin`

- Supports a plugin system that dynamically loads Python modules and packages at runtime.
- Supports functioning as subcomponents, providing dependency support for custom models and other local code.
- Capable of applying to a database and storing in the artifact_store, exporting as a superduper format package for sharing with others.
- Supports automatic installation of dependencies listed in the requirements.txt file under the plugin.

***Usage pattern***

Create plugin

```python
from superduperdb.components.plugin import Plugin
plugin_path = 'path/to/my_module.py'
my_plugin = Plugin(path=plugin_path)
```

Pip install without python code

```python
from superduperdb.components.plugin import Plugin
# If there is only one requirements file, the path must be a file that ends with requirements.txt.
plugin = Plugin(path="deploy/installations/testenv_requirements.txt")
```

Python package with requirements
```python
from superduperdb.components.plugin import Plugin
plugin_path = 'path/to/my_package'
# |_my_package
#    |_requirements.txt
#    |_file1.py
#    |_file2.py
#    |_sub_module
#       |_file_a.py
my_plugin = Plugin(path=plugin_path)
```

Python module with requirements

> If you want to add requirements to a Python file, you can create a requirement_plugin as a submodule of this module. 
> Then, the requirement_plugin will be loaded prior to the Python code.
```python
from superduperdb.components.plugin import Plugin
requirements_path = 'path/to/my_requirements.txt'
requirement_plugin = Plugin(path=requirements_path)

plugin_path = 'path/to/my_module.py'
my_plugin = Plugin(path=plugin_path, plugins=[requirement_plugin])
```

Export plugin

```python
from superduperdb.components.plugin import Plugin
plugin_path = 'plugin_path'
my_plugin = Plugin(path=plugin_path)

my_plugin.export("exports/plugin")
```

Load plugin

```python
from superduperdb.components.plugin import Plugin
my_plugin = Plugin.read("exports/plugin")
```

As a sub-component

```python
from utils import function
class Model:
  def predict(self, X):
    return function(X)

from superduperdb.components.plugin import Plugin
plugin = Plugin(path="./utils.py")
model = Model(identifier="test", plugins=[plugin])
db.apply(model)

# Then we can execute db.load("model", "test") from any location.
```

***Explanation***

Initialization and installation

- During plugin initialization, superduperdb loads the component’s Python module or package into `sys.modules`, allowing subsequent use of native import statements for plugin utilization.
- If the plugin package includes a `requirements.txt`, dependencies are installed prior to loading the Python code.
- The plugin is installed only once per process; if it detects that the same plugin has already been installed in the current runtime, the installation is skipped.

Usage

- When exported locally, the plugin component saves all necessary dependency files for the plugins into the superduperdb package, allowing for sharing to different locations.
- When executing `db.apply(plugin)`, the necessary Python dependency environment files for the plugin are saved in the artifact_store. During `db.load("plugin", "plugin_identifier")`, these files are downloaded to the local `~/.superduperdb/plugin/` directory, followed by the initialization and installation of the plugin.
- As a sub-component, superduperdb’s encode and decode logic ensures that the plugin is loaded prior to the parent component to maintain dependency integrity.


***See also***

- [superduperdb.components.plugin](../api/components/plugin.md)