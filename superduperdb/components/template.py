import dataclasses as dc
import json
import os
import typing as t

from superduperdb.base.document import Document
from superduperdb.components.datatype import DataType, dill_lazy

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

    _artifacts: t.ClassVar[t.Sequence[t.Tuple[str, 'DataType']]] = (
        ('_component_blobs', dill_lazy),
    )
    type_id: t.ClassVar[str] = 'template'

    component: t.Union[Component, t.Dict]
    info: t.Optional[t.Dict] = dc.field(default_factory=dict)
    _component_blobs: t.Optional[t.Union[t.Dict, bytes]] = dc.field(
        default_factory=dict
    )

    def __post_init__(self, db, artifacts):
        self._variables = []
        if isinstance(self.component, Component):
            self._variables = self.component.variables
            self.component = self.component.dict().encode()
            if not self._component_blobs:
                self._component_blobs = self.component.pop_blobs()
        return super().__post_init__(db, artifacts)

    @ensure_initialized
    def __call__(self, **kwargs):
        """Method to create component from the given template and `kwargs`."""
        if self.info:
            assert set(kwargs.keys()) == (set(self.info.keys())), 'Invalid variables'
        t = Document.decode(
            {**self.component, '_blobs': self._component_blobs}, db=self.db
        )
        t.init(db=self.db)
        t = t.set_variables(db=self.db, **kwargs)
        t.init(db=self.db)
        return t

    @property
    def variables(self):
        """Variables in `Template` property."""
        return self._variables

    @staticmethod
    def _append_component_metadata(r, component={}):
        leaves = r.get('_leaves', {})
        files = r.get('_files', {})
        leaves.update(component.pop('_leaves', {}))
        files.update(component.pop('_files', {}))
        r['_leaves'] = leaves
        r['_files'] = files
        return r

    def dict(self):
        """Updates base dict document with `component` metadata."""
        r = super().dict()
        component = r['component']
        r = self._append_component_metadata(r, component=component)

        return r

    def export(self, path: str, format: str = 'json'):
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
        r = self.dict().encode()

        os.makedirs(path, exist_ok=True)
        if self._component_blobs:
            os.makedirs(os.path.join(path, 'blobs'), exist_ok=True)
            for file_id, bytestr_ in r.blobs.items():
                filepath = os.path.join(path, 'blobs', file_id)
                with open(filepath, 'wb') as f:
                    f.write(bytestr_)
            r.pop_blobs()
        else:
            del r['_blobs']

        if format == 'json':
            with open(os.path.join(path, 'component.json'), 'w') as f:
                json.dump(r, f, indent=2)
        elif format == 'yaml':
            import yaml

            with open(os.path.join(path, 'component.yaml'), 'w') as f:
                yaml.safe_dump(r, f)
        else:
            raise ValueError(f'Invalid format: {format}')

    def on_load(self, db):
        """
        Datalayer `on_load` hook.

        :param db: Datalayer instance.
        """
        super().on_load(db=db)
        self.init(db=db)
