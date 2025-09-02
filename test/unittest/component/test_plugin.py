import json
import os
import sys

import pytest

from superduper.base.base import REGISTRY
from superduper.components.plugin import Plugin

PYTHON_CODE = """
from superduper import Model
import typing as t

class {class_name}(Model):
    def predict(self, X) -> int:
        return "{plugin_type}"
"""


def write_path(path, code):
    with open(path, "w") as f:
        f.write(code)


@pytest.fixture
def cleanup():
    yield
    # REGISTRY.drop()


def create_module_plugin(tempdirname, module_name, component):
    code = PYTHON_CODE.format(plugin_type=module_name, class_name=component)
    write_path(os.path.join(tempdirname, f"p_{module_name}.py"), code)
    plugin_path = os.path.join(tempdirname, f"p_{module_name}.py")
    return plugin_path


def create_package_plugin(
    tempdirname,
    package_name,
    component,
    include_init=True,
    requirements=None,
):
    code = PYTHON_CODE.format(plugin_type=package_name, class_name=component)
    plugin_path = os.path.join(tempdirname, f"p_{package_name}")
    os.makedirs(plugin_path)

    if include_init:
        write_path(os.path.join(plugin_path, "__init__.py"), "")

    if requirements:
        code = PYTHON_CODE.format(plugin_type=package_name, class_name=component)
        code = "import matplotlib\nversion=matplotlib.__version__" + code
        write_path(
            os.path.join(plugin_path, "requirements.txt"),
            "matplotlib",
        )
    else:
        code = PYTHON_CODE.format(plugin_type=package_name, class_name=component)

    write_path(os.path.join(plugin_path, f"p_{package_name}.py"), code)
    return plugin_path


def create_import_plugin(tempdirname, component):

    import_path = os.path.join(tempdirname, "import")
    component_dict = {
        "identifier": "plugin",
        "component": "Plugin",
        "path": "&:file:p_import:file_id",
    }
    os.makedirs(import_path)
    write_path(os.path.join(import_path, "component.json"), json.dumps(component_dict))
    create_package_plugin(
        os.path.join(import_path, "files", "file_id"), "import", component=component
    )
    return import_path


def test_module(tmpdir, cleanup):
    with pytest.raises(ImportError):
        import p_module

        print(p_module)

    path = create_module_plugin(tmpdir, "module", "PModelModule")
    Plugin(path=path)
    from p_module import PModelModule

    model = PModelModule("test")
    assert model.predict(2) == "module"


def test_package(tmpdir, cleanup):
    with pytest.raises(ImportError):
        import p_package

        print(p_package)

    path = create_package_plugin(tmpdir, "package", "PModelPackage")
    Plugin(path=path)
    from p_package.p_package import PModelPackage

    model = PModelPackage("test")
    assert model.predict(2) == "package"


def test_directory(tmpdir, cleanup):
    with pytest.raises(ImportError):
        import p_directory

        print(p_directory)

    path = create_package_plugin(tmpdir, "directory", "PModelDirectory")
    Plugin(path=path)
    from p_directory.p_directory import PModelDirectory

    model = PModelDirectory("test")

    assert model.predict(2) == "directory"


def test_repeated_loading(tmpdir, cleanup):
    path = create_module_plugin(tmpdir, "repeated", component="PModelRepeated")
    p = Plugin(path=path)
    assert "p_repeated" in sys.modules
    assert f"_PLUGIN_{p.uuid}" in os.environ

    sys.modules.pop("p_repeated")
    Plugin(path=path)
    assert "p_repeated" not in sys.modules


def test_requirements(tmpdir, cleanup):
    with pytest.raises(ImportError):
        import p_pip

        print(p_pip)

    path = create_package_plugin(
        tmpdir, "pip", requirements=True, component="PModelPip"
    )
    Plugin(path=path)

    from p_pip.p_pip import version

    assert version


def test_import(tmpdir, cleanup):
    with pytest.raises(ImportError):
        import p_import

        print(p_import)

    path = create_import_plugin(tmpdir, component="PModelImport")
    Plugin.read(path)

    from p_import.p_import import PModelImport

    model = PModelImport("test")
    assert model.predict(2) == "import"


def test_apply(db, tmpdir, cleanup):
    path = create_package_plugin(tmpdir, "apply", component="PModelApply")
    plugin = Plugin(identifier="test", path=path)
    db.apply(plugin)

    plugin_reload = db.load("Plugin", "test")

    assert plugin_reload.path.startswith(os.path.expanduser("~"))
