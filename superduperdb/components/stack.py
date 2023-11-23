import dataclasses as dc
import json
import os
import tarfile
import typing as t

from superduperdb.backends.local.artifacts import FileSystemArtifactStore
from superduperdb.base.serializable import Serializable

from .component import Component

if t.TYPE_CHECKING:
    from superduperdb.base.datalayer import Datalayer as DB


ARTIFACTS = 'artifacts'
_STACK_JSON_FILE = 'stack.json'


@dc.dataclass
class Stack(Component):
    """
    A placeholder to hold list of components under a namespace and packages them as
    a tarball
    This tarball can be retrieved back to a `Stack` instance with ``load`` method.

    :param identifier: A string used to identify the model.
    :param components: List of components to stack together and add to database.
    :param version: Version number of the model(?)
    """

    identifier: t.Optional[str] = None
    components: t.Optional[t.Sequence[Component]] = ()
    version: t.Optional[int] = None

    type_id: t.ClassVar[str] = 'stack'

    def __post_init__(self):
        self._load_components()

    def _load_components(self):
        self._component_type_store = {}
        self._component_store = {}
        if self.components:
            for component in self.components:
                type_id = component.type_id
                identifier = component.identifier

                if not self._component_type_store.get(type_id, None):
                    self._component_type_store.update({type_id: [component]})
                else:
                    self._component_type_store[type_id] = self._component_type_store[
                        type_id
                    ].append(component)

                self._component_store.update({identifier: component})

    @staticmethod
    def to_tarball(folder_path: str):
        """
        Create a tarball (compressed archive) from a folder.

        :param folder_path: Path to the folder to be archived.
        """
        path = os.path.basename(folder_path) + '.tar.gz'
        with tarfile.open(path, "w:gz") as tar:
            tar.add(folder_path, arcname=os.path.basename(folder_path))

    @staticmethod
    def from_tarball(tarball_path: str):
        """
        Extract the contents of stack tarball

        :param tarball_path: Path to the tarball file.
        """
        extract_path = tarball_path.split('.tar.gz')[0]
        with tarfile.open(tarball_path, "r:gz") as tar:
            tar.extractall(path=extract_path)
        return extract_path

    def save(self, db: 'DB', name: str):
        """
        Save method takes a stack name and exports ``self``
        to tarball by serializing ``Stack`` along with storing
        artifacts to filesystem

        The json serialized file and artifacts folders are packaged by tar
        ball.
        """
        os.makedirs(name, exist_ok=True)
        fs_store = FileSystemArtifactStore(os.path.join(name, ARTIFACTS))

        serialized, artifacts = self.serialized
        artifact_info = fs_store.save(artifacts)
        serialized = db.artifact_store.replace(serialized, artifact_info)

        with open(os.path.join(name, _STACK_JSON_FILE), 'w') as outfile:
            json.dump(serialized, outfile)

        self.to_tarball(name)

    def load(self, path: str):
        """
        Load method loads the stack tarball, loads the `stack.json` and `artifacts`.
        It binds the loaded stack object to ``self``

        Stack should be empty for it to load the given tarball else it will
        override the exisitng artibutes.

        :param: Path to tarball which contains artifacts and stack json.
        """
        # Extract tarball
        extracted_path = self.from_tarball(path + '.tar.gz')

        # Load stack.josn file
        with open(os.path.join(extracted_path, _STACK_JSON_FILE), 'r') as inputfile:
            serialized = json.load(inputfile)

        # Load the stack
        fs_store = FileSystemArtifactStore(os.path.join(extracted_path, ARTIFACTS))
        info = fs_store.load(serialized, lazy=True)
        loaded_stack = Serializable.deserialize(info)

        self.identifier = loaded_stack.identifier
        self.components = loaded_stack.components
        self._load_components()

    def get(self, identifier: str):
        return self._component_store.get(identifier, None)

    def get_component_type(self, type_id: str):
        return self._component_type_store.get(type_id, None)
