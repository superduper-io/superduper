import dataclasses as dc
import os
import typing as t

from superduperdb.base.constant import KEY_BLOBS
from superduperdb.base.datalayer import Datalayer
from superduperdb.base.document import Document
from superduperdb.base.variables import _replace_variables
from superduperdb.misc.special_dicts import SuperDuperFlatEncode

from .component import Component, ensure_initialized


class Template(Component):
    """
    Application template component.

    :param component: Template component with variables.
    :param variables: Variables to be set.
    :param blobs: Blob identifiers in `Template.component`
    :param info: Additional information.
    """

    literals: t.ClassVar[t.Tuple[str]] = ('template',)
    type_id: t.ClassVar[str] = "template"
    template: t.Dict
    variables: t.Optional[t.List[str]] = None
    blobs: t.Optional[t.List[str]] = None
    info: t.Optional[t.Dict] = dc.field(default_factory=dict)

    def pre_create(self, db: Datalayer) -> None:
        """Run before the object is created."""
        super().pre_create(db)
        if KEY_BLOBS in self.template:
            for identifier, blob in self.template[KEY_BLOBS].items():
                db.artifact_store.put_bytes(blob, identifier)
            self.blobs = list(self.template[KEY_BLOBS].keys())
            self.template.pop(KEY_BLOBS)

    def __post_init__(self, db, artifacts):
        self.template = SuperDuperFlatEncode(self.template)
        if self.variables is None:
            self.variables = self.template.variables
        return super().__post_init__(db, artifacts)

    @ensure_initialized
    def __call__(self, **kwargs):
        """Method to create component from the given template and `kwargs`."""
        assert set(kwargs.keys()) == set(self.variables)
        component = _replace_variables(self.template, **kwargs)
        return Document.decode(component, db=self.db).unpack()

    def export(
        self,
        path: t.Optional[str] = None,
        format: str = 'json',
        zip: bool = False,
        defaults: bool = False,
        metadata: bool = False,
    ):
        """
        Save `self` to a directory using super-duper protocol.

        :param path: Path to the directory to save the component.
        :param format: Format to save the component in.
        :param zip: Whether to zip the directory.
        :param defaults: Whether to save default values.
        :param metadata: Whether to save metadata.

        Created directory structure:
        ```
        |_component.(json|yaml)
        |_blobs/*
        |_files/*
        ```
        """
        if self.blobs is not None and self.blobs:
            assert self.db is not None
            assert self.identifier in self.db.show('template')
        if path is None:
            path = './' + self.identifier
        super().export(path, format, zip=False, defaults=defaults, metadata=metadata)
        if self.blobs is not None and self.blobs:
            os.makedirs(os.path.join(path, 'blobs'), exist_ok=True)
            for identifier in self.blobs:
                blob = self.db.artifact_store.get_bytes(identifier)
                with open(path + f'/blobs/{identifier}', 'wb') as f:
                    f.write(blob)
        if zip:
            self._zip_export(path)
