import importlib.util
import os
import shutil
import subprocess
import sys

from superduper import Component, logging
from superduper.base.datatype import FileItem
from superduper.misc import typing as st


class Plugin(Component):
    """Plugin component allows to install and use external python packages as plugins.

    :param path: Path to the plugin package or module.
    :param identifier: Unique identifier for the plugin.
    :param cache_path: Path to the cache directory where the plugin will be stored.
    """

    breaks = ('path',)

    path: st.File
    identifier: str = ""
    cache_path: str = "~/.superduper/plugins"

    def postinit(self):
        """Post initialization method."""
        self.setup()
        self._prepare_plugin()

        path_name = os.path.basename(self.path.rstrip("/"))
        self.identifier = self.identifier or f"plugin-{path_name}".replace(".", "_")
        self._install()
        super().postinit()

    def _install(self):
        logging.debug(f"Installing plugin {self.identifier}")
        package_path = self.path
        module_name = os.path.basename(self.path.rstrip("/"))

        # Check if plugin is already installed
        check_tag = f"_PLUGIN_{self.uuid}"

        if check_tag in os.environ:
            logging.debug(f"Plugin {self.identifier} already installed")
            return

        if os.path.isdir(package_path):
            import_package_path = os.path.join(package_path, "__init__.py")

            if not os.path.exists(import_package_path):
                logging.info(f"Creating __init__.py file in {package_path}")
                open(import_package_path, "w").close()

            logging.debug(f"Plugin {self.identifier} is a package")
            self._pip_install(os.path.join(package_path, "requirements.txt"))

        else:
            module_name = module_name.split(".")[0]
            import_package_path = package_path

            if package_path.endswith(".py"):
                logging.debug(f"Plugin {self.identifier} is a standalone Python file")

            elif package_path.endswith("requirements.txt"):
                os.environ[check_tag] = "1"
                self._pip_install(package_path)
                return

            else:
                raise ValueError(
                    (
                        f"Plugin {self.identifier} path "
                        "is not a valid Python file or requirements file."
                    )
                )

        spec = importlib.util.spec_from_file_location(module_name, import_package_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules[module_name] = module

        os.environ[check_tag] = "1"

    def _pip_install(self, requirement_path):
        if not os.path.exists(requirement_path):
            logging.debug(f"No requirements file found for plugin {self.identifier}")
            return
        logging.debug(f"Installing requirements for plugin {self.identifier}")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", requirement_path],
            check=True,
        )

    def _prepare_plugin(self):
        plugin_name_tag = f"{self.identifier}"
        if isinstance(self.path, FileItem):
            self.path = self.path.unpack()

        cache_path = os.path.expanduser(self.cache_path)
        uuid_path = os.path.join(cache_path, self.uuid)

        # Check if plugin is already in cache
        if os.path.exists(uuid_path):
            logging.info(f'Plugin {self.path} already exists in cache')
            names = os.listdir(uuid_path)
            names = [name for name in names if name != "__pycache__"]
            assert len(names) == 1, f"Multiple plugins found in {uuid_path}"
            self.path = os.path.join(uuid_path, names[0])
            sys.path.append(uuid_path)
            return

        logging.info(f"Preparing plugin {plugin_name_tag}")
        assert os.path.exists(
            self.path
        ), f"Plugin {plugin_name_tag} not found at {self.path}"

        # Pull the plugin to cache
        logging.info(f"Downloading plugin {self.identifier} to {self.path}")
        dist = os.path.join(cache_path, self.uuid, os.path.basename(self.path))
        if os.path.exists(dist):
            logging.info(f"Plugin {self.identifier} already exists in cache : {dist}")
        else:
            logging.info(f"Copying plugin [{self.identifier}] to {dist}")
            os.makedirs(os.path.dirname(dist), exist_ok=True)
            if os.path.isdir(self.path):
                shutil.copytree(self.path, dist)
            else:
                shutil.copy(self.path, dist)

        self.path = dist
