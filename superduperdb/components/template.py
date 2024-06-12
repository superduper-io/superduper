import dataclasses as dc
import json
import os
import typing as t

from superduperdb.base.document import Document

from .component import Component, ensure_initialized


class Template(Component):
    """
    Application template component.

    :param component: Template component with variables.
    :param info: Info.
    :param _component_blobs: Blobs in `Template.component`
                             NOTE: This is only for internal
                             use.
    :param _component_leaves: Leaves in `Template.component`
                              NOTE: This is only for internal
                              use.
    """

    _literals: t.ClassVar[t.Tuple[str]] = ('component', '_component_leaves')
    type_id: t.ClassVar[str] = 'template'

    component: t.Union[Component, t.Dict]
    info: t.Optional[t.Dict] = dc.field(default_factory=dict)
    _component_blobs: t.Optional[t.Union[t.Dict, bytes]] = dc.field(
        default_factory=dict
    )
    _component_leaves: t.Optional[t.Dict] = dc.field(default_factory=dict)

    def __post_init__(self, db, artifacts):
        self._variables = []
        if isinstance(self.component, Component):
            self._variables = self.component.variables
            self.component = self.component.dict().encode()
            if not self._component_blobs:
                self._component_blobs = self.component.pop_blobs()
            if not self._component_leaves:
                self._component_leaves = dict(self.component.pop_leaves())
        return super().__post_init__(db, artifacts)

    @ensure_initialized
    def __call__(self, **kwargs):
        """Method to create component from the given template and `kwargs`."""
        if self.info:
            assert set(kwargs.keys()) == (set(self.info.keys())), 'Invalid variables'
        t = Document.decode(
            {
                **self.component,
                '_blobs': self._component_blobs or {},
                '_leaves': self._component_leaves or {},
            },
            db=self.db,
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
    def read(path: str):
        """
        Read a `Component` instance from a directory created with `.export`.

        :param path: Path to the directory containing the component.

        Expected directory structure:
        ```
        |_component.json/yaml
        |_blobs/*
        |_files/*
        ```
        """
        from superduperdb.components.component import _build_info_from_path

        config_object = _build_info_from_path(path=path)

        from superduperdb import Document

        def load_blob(blob):
            with open(path + '/blobs/' + blob, 'rb') as f:
                return f.read()

        # Move back the top level component leaves and blobs to the component
        total_leaves = config_object.get('_leaves', {})
        component_leaves = {}
        for v in config_object.pop('_component_leaves_keys', []):
            component_leaves[v] = total_leaves.pop(v)
        config_object['_component_leaves'] = component_leaves

        component_blobs = {}
        for v in config_object.pop('_component_blobs_keys', []):
            component_blobs[v] = load_blob(v)
        config_object['_component_blobs'] = component_blobs

        return Document.decode(config_object).unpack()

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
        # Move the component leaves and blobs to the top level
        r = self.dict().encode()
        component_leaves = r.pop('_component_leaves', {})
        component_blobs = r.pop('_component_blobs', {})
        component_leaves_keys = list(component_leaves.keys())
        component_blobs_keys = list(component_blobs.keys())
        r['_leaves'].update(component_leaves)
        r['_blobs'].update(component_blobs)
        r['_component_leaves_keys'] = component_leaves_keys
        r['_component_blobs_keys'] = component_blobs_keys

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
