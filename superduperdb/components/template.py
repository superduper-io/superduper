import dataclasses as dc
import json
import os
import typing as t

from superduperdb.base.constant import KEY_BLOBS, KEY_BUILDS, KEY_FILES
from superduperdb.base.datalayer import Datalayer
from superduperdb.base.document import Document
from superduperdb.base.variables import _replace_variables
from superduperdb.components.datatype import DataType, dill_lazy
from superduperdb.misc.special_dicts import SuperDuperFlatEncode

from .component import Component, ensure_initialized


class Template(Component):
    """
    Application template component.

    :param component: Template component with variables.
    :param info: Info.
    :param _component_blobs: Blobs in `Template.component`
                             NOTE: This is only for internal
                             use.
    """

    literals: t.ClassVar[t.Tuple[str]] = ('component',)
    _artifacts: t.ClassVar[t.Sequence[t.Tuple[str, 'DataType']]] = (
        ('_component_blobs', dill_lazy),
    )
    type_id: t.ClassVar[str] = "template"

    component: t.Union[Component, t.Dict]
    variables: t.Optional[t.List[str]] = None
    blobs: t.Optional[t.List[str]] = None
    info: t.Optional[t.Dict] = dc.field(default_factory=dict)

    def pre_create(self, db: Datalayer) -> None:
        super().pre_create(db)
        for identifier, blob in self.component[KEY_BLOBS]:
            db.artifact_store.put_bytes(blob, identifier)
        self.blobs = list(self.component[KEY_BLOBS].keys())
        self.component.pop(KEY_BLOBS)

    def __post_init__(self, db, artifacts):
        if isinstance(self.component, Component):
            self.component = self.component.encode()
        else:
            self.component = SuperDuperFlatEncode(self.component)
        if self.variables is None:
            self.variables = self.component.variables
        return super().__post_init__(db, artifacts)

    @ensure_initialized
    def __call__(self, **kwargs):
        """Method to create component from the given template and `kwargs`."""
        assert set(kwargs.keys()) == set(self.variables)
        _replace_variables(self.component, **kwargs)
        return Document.decode(self.component, db=self.db)

    def export(self, path: str, format: str = 'json', zip: bool = True):
        """
        Save `self` to a directory using super-duper protocol.

        :param path: Path to the directory to save the component.

        Created directory structure:
        ```
        |_component.json/yaml
        |_blobs/*
        |_files/*
        ```
        """
        super().export(path, format, zip=False)
        if self.blobs:
            os.makedirs(os.path.join(path, 'blobs'), exist_ok=True)
            for identifier in self.blobs:
                blob = self.db.artifact_store.get_bytes(identifier)
                with open(path + f'/blobs/{identifier}', 'wb') as f:
                    f.write(blob)
        if zip:
            self.zip_export(path)
