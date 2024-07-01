import dataclasses as dc
import os
import typing as t

from superduperdb.base.constant import KEY_BLOBS
from superduperdb.base.datalayer import Datalayer
from superduperdb.base.document import Document
from superduperdb.base.leaf import Leaf
from superduperdb.base.variables import _replace_variables
from superduperdb.components.component import Component
from superduperdb.misc.special_dicts import SuperDuperFlatEncode

from .component import ensure_initialized


class _BaseTemplate(Component):
    """
    Base template component.

    :param template: Template component with variables.
    :param template_variables: Variables to be set.
    :param info: Additional information.
    :param blobs: Blob identifiers in `Template.component`.
    :param substitutions: Substitutions to be made to create variables.
    """

    literals: t.ClassVar[t.Tuple[str]] = ('template',)

    template: t.Union[t.Dict, Component]
    template_variables: t.Optional[t.List[str]] = None
    info: t.Optional[t.Dict] = dc.field(default_factory=dict)
    blobs: t.Optional[t.List[str]] = None
    substitutions: dc.InitVar[t.Optional[t.Dict]] = None

    def __post_init__(self, db, artifacts, substitutions):
        if isinstance(self.template, Leaf):
            self.template = self.template.encode(defaults=False, metadata=False)
        self.template = SuperDuperFlatEncode(self.template)
        if substitutions is not None:
            self.template = self.template.to_template(**substitutions)
        if self.template_variables is None:
            self.template_variables = self.template.variables
        super().__post_init__(db, artifacts)

    @ensure_initialized
    def __call__(self, **kwargs):
        """Method to create component from the given template and `kwargs`."""
        assert set(kwargs.keys()) == set(self.template_variables)
        component = _replace_variables(self.template, **kwargs)
        return Document.decode(component, db=self.db).unpack()

    @property
    def form_template(self):
        """Form to be diplayed to user."""
        return {
            'identifier': '<enter-a-unique-identifier>',
            '_variables': {
                k: f'<value-{i}>' for i, k in enumerate(self.template_variables)
            },
            **{k: v for k, v in self.template.items() if k != 'identifier'},
        }


class Template(_BaseTemplate):
    """Application template component."""

    type_id: t.ClassVar[str] = "template"

    def pre_create(self, db: Datalayer) -> None:
        """Run before the object is created."""
        super().pre_create(db)
        assert isinstance(self.template, dict)
        if KEY_BLOBS in self.template:
            for identifier, blob in self.template[KEY_BLOBS].items():
                db.artifact_store.put_bytes(blob, identifier)
            self.blobs = list(self.template[KEY_BLOBS].keys())
            self.template.pop(KEY_BLOBS)

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


class QueryTemplate(_BaseTemplate):
    """
    Query template component.

    Example:
    -------
    >>> q = db['docs'].select().limit('<var:limit>')
    >>> t = QueryTemplate('select_lim', template=q)
    >>> t.variables
    ['limit']

    """

    type_id: t.ClassVar[str] = 'query_template'

    def __post_init__(self, db, artifacts, substitutions):
        if isinstance(self.template, Leaf):
            self.template = self.template.dict(metadata=False, defaults=False).encode()
        return super().__post_init__(db, artifacts, substitutions)

    @property
    def form_template(self):
        """Form to be diplayed to user."""
        return {
            '_variables': {
                k: f'<value-{i}>' for i, k in enumerate(self.template_variables)
            },
            **{
                k: v for k, v in self.template.items()
                if k not in {'_builds', '_blobs', 'identifier', '_path'}
            },
        }

    def execute(self, **kwargs):
        """Execute the query with the given variables.

        :param kwargs: Variables to be set in the query.
        """
        query = self.query.set_variables(**kwargs)
        return self.db.execute(query)
