import importlib.util
import os
import shutil
import sys
import typing as t

from superduperdb import Component, logging
from superduperdb.components.datatype import LazyFile, file_lazy


class Plugin(Component):
    type_id: t.ClassVar[str] = "plugin"
    _artifacts: t.ClassVar = (("path", file_lazy),)
    path: str
    identifier: str = ""
    cache_path: str = ".superduperdb/plugins"

    def __post_init__(self, db, artifacts):
        if isinstance(self.path, LazyFile):
            self.prepare_plugin()
        else:
            path_name = os.path.basename(self.path.rstrip("/"))
            self.identifier = self.identifier or f"plugin-{path_name}"
        super().__post_init__(db, artifacts)
        self.install()

    def install(self):
        logging.info(f"Installing plugin {self.identifier}")
        package_path = self.path
        path_name = os.path.basename(self.path.rstrip("/"))
        if "__init__.py" in os.listdir(package_path):
            logging.info(f"Plugin {self.identifier} is a package")
            spec = importlib.util.spec_from_file_location(
                path_name, os.path.join(package_path, "__init__.py")
            )
            legal_tech = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = legal_tech
        else:
            logging.info(f"Plugin {self.identifier} is a module")
            sys.path.append(package_path)

    def prepare_plugin(self):
        plugin_name_tag = f"{self.identifier}"
        logging.info(f"Preparing plugin {plugin_name_tag}")
        assert isinstance(self.path, LazyFile)
        self.path = self.path.unpack()
        assert os.path.exists(
            self.path
        ), f"Plugin {plugin_name_tag} not found at {self.path}"
        logging.info(f"Downloading plugin {self.identifier} to {self.path}")
        dist = os.path.join(self.cache_path, self.uuid, os.path.basename(self.path))
        if os.path.exists(dist):
            logging.info(f"Plugin {self.identifier} already exists in cache : {dist}")
        else:
            logging.info(f"Copying plugin [{self.identifier}] to {dist}")
            shutil.copytree(self.path, dist)
        self.path = dist
