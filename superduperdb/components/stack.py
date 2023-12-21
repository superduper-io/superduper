import dataclasses as dc
import importlib
import json
import os
import re
import tarfile
import typing as t

from superduperdb.backends.local.artifacts import FileSystemArtifactStore
from superduperdb.base.serializable import Serializable
from superduperdb.components.serializer import serializers
from superduperdb.misc.annotations import public_api
from superduperdb.misc.download import Fetcher

from .component import Component

if t.TYPE_CHECKING:
    from superduperdb.base.datalayer import Datalayer


ARTIFACTS = 'artifacts'
_STACK_JSON_FILE = 'stack.json'


@public_api(stability='alpha')
@dc.dataclass(kw_only=True)
class Stack(Component):
    """
    A placeholder to hold list of components under a namespace and packages them as
    a tarball
    This tarball can be retrieved back to a `Stack` instance with ``load`` method.
    {component_parameters}
    :param components: List of components to stack together and add to database.
    :param version: Version number of the model(?)
    """

    __doct__ = __doc__.format(component_parameters=Component.__doc__)

    type_id: t.ClassVar[str] = 'stack'

    components: t.Sequence[Component] = ()

    @property
    def child_components(self):
        return (('components', 'component_list'),)

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

    def save(self, db: 'Datalayer', name: str):
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
        fs_store = os.path.join(extracted_path, ARTIFACTS)
        info = fs_store.load(serialized, lazy=True)
        loaded_stack = Serializable.deserialize(info)

        self.identifier = loaded_stack.identifier
        self.components = loaded_stack.components
        self._load_components()

    def get(self, identifier: str):
        return self._component_store.get(identifier, None)

    def get_component_type(self, type_id: str):
        return self._component_type_store.get(type_id, None)

    @staticmethod
    def from_dict(d: t.Dict, db: t.Optional['Datalayer'] = None):
        """
        Load a stack from a dictionary.

        :param d: Dictionary containing the stack definition.
        :param db: Datalayer instance to use for loading artifacts.

        The dictionary should look something like below.
        1. An ``identifier`` field
        2. An ``artifact`` field containing a list of artifacts including the
           serializer used
        3. A ``components`` field containing a list of component definitions.
           These definitions can recursively refer to other components or artifacts,
           resp. ``$components.<identifier>`` or ``$artifacts[<index>]``.

        >>> d = {
        ...    'identifier': 'my_stack',
        ...    'artifacts': [
        ...        {
        ...            'serializer': 'pickle',
        ...            'bytes': bytes_,
        ...        },
        ...    ],
        ...    'components': [
        ...        {'module': 'superduperdb.ext.pillow', 'variable': 'pil_image'},
        ...        {
        ...            'module': 'superduperdb.components.model',
        ...            'cls': 'Model',
        ...            'dict': {
        ...                'identifier': 'my_model',
        ...                'object': '$artifacts[0]',
        ...                'encoder': '$components.pil_image',
        ...            },
        ...        },
        ...    ],
        ... }
        >>> s = Stack.from_dict(d)
        >>> isinstance(s, Stack)
        True
        >>> isinstance(s.components[1], Model)
        True

        """
        return _load_stack(d, db)


def _find_variables(r):
    if 'dict' not in r:
        return []
    variables = []
    for k in r['dict']:
        if isinstance(r['dict'][k], str) and r['dict'][k].startswith('$components'):
            variables.append(r['dict'][k])
    return variables


def _find_leaves(components, existing):
    def contains_new_variable(r):
        if set(_find_variables(r)).issubset(set(existing)):
            return False
        return True

    out = [c for c in components if not contains_new_variable(c)]
    components = [c for c in components if contains_new_variable(c)]
    return out, components


def _load_artifact(variable, artifacts, serializers, dir=None, fetcher=None):
    serializer = serializers[artifacts[variable]['serializer']]
    if 'bytes' not in artifacts[variable]:
        fetcher = fetcher or Fetcher()
        try:
            uri = artifacts[variable]['uri']
        except KeyError:
            raise Exception(f'No uri or bytes found for artifact {variable}')

        artifacts[variable]['bytes'] = fetcher(uri)
    return serializer.decode(artifacts[variable]['bytes'])


def _build_component(d, artifacts, components, serializers):
    module = importlib.import_module(d['module'])
    if 'dict' in d:
        defn = d['dict'].copy()
        for k in defn:
            if isinstance(defn[k], str) and defn[k].startswith('$'):
                if '.' in defn[k]:
                    assert defn[k].startswith('$components.')
                    type, variable = defn[k][1:].split('.')
                else:
                    match = re.match('\$artifacts\[(\d+)\]', defn[k])
                    assert (
                        match
                    ), 'Unknown variable type {defn[k]}, expected $artifacts[<index>]'
                    type = 'artifacts'
                    variable = int(match.groups()[0].strip())

                if type == 'artifacts':
                    if 'artifact' not in artifacts[variable]:
                        artifacts[variable]['artifact'] = _load_artifact(
                            variable, artifacts, serializers=serializers
                        )
                    defn[k] = artifacts[variable]['artifact']
                elif type == 'components':
                    defn[k] = {c.identifier: c for c in components}[variable]
                else:
                    raise Exception(f'Unknown variable type {type}')

        component_cls = getattr(module, d['cls'])
        built = component_cls(**defn)
        if built.type_id == 'serializer':
            serializers[built.identifier] = built.object
    else:
        built = getattr(module, d['variable'])
    return built


def _load_stack(d, db=None):
    components = []
    to_build = d['components'].copy()
    while True:
        current, to_build = _find_leaves(
            to_build,
            existing=[f'$components.{c.identifier}' for c in components],
        )
        for c in current:
            built = _build_component(
                c,
                artifacts=d['artifacts'],
                components=components,
                serializers=db.serializers if db else serializers,
            )
            components.append(built)
        if not to_build:
            break
    return Stack(identifier=d['identifier'], components=components)
