import dataclasses as dc
import os
import typing as t

from superduper import logging
from superduper.base.constant import KEY_BLOBS, KEY_FILES
from superduper.base.datalayer import Datalayer
from superduper.base.document import Document, QueryUpdateDocument
from superduper.base.leaf import Leaf
from superduper.base.variables import _replace_variables
from superduper.components.component import Component, _build_info_from_path
from superduper.components.datatype import pickle_serializer
from superduper.misc.special_dicts import SuperDuperFlatEncode

from .component import ensure_initialized


class _BaseTemplate(Component):
    """
    Base template component.

    :param template: Template component with variables.
    :param template_variables: Variables to be set.
    :param types: Additional information about types of variables.
    :param blobs: Blob identifiers in `Template.component`.
    :param files: File identifiers in `Template.component`.
    :param substitutions: Substitutions to be made to create variables.
    :param default_values: Default values for the variables.
    """

    literals: t.ClassVar[t.Tuple[str]] = ('template',)

    template: t.Union[t.Dict, Component]
    template_variables: t.Optional[t.List[str]] = None
    types: t.Optional[t.Dict] = dc.field(default_factory=dict)
    blobs: t.Optional[t.List[str]] = None
    files: t.Optional[t.List[str]] = None
    substitutions: dc.InitVar[t.Optional[t.Dict]] = None
    default_values: t.Dict = dc.field(default_factory=dict)

    def __post_init__(self, db, artifacts, substitutions):
        if isinstance(self.template, Leaf):
            self.template = self.template.encode(defaults=False, metadata=False)
        self.template = SuperDuperFlatEncode(self.template)
        if substitutions is not None:
            self.template = QueryUpdateDocument(self.template).to_template(
                **substitutions
            )
        if self.template_variables is None:
            self.template_variables = self.template.variables
        logging.info(f"Template variables: {self.template_variables}")
        super().__post_init__(db, artifacts)

    @ensure_initialized
    def __call__(self, **kwargs):
        """Method to create component from the given template and `kwargs`."""
        kwargs.update({k: v for k, v in self.default_values.items() if k not in kwargs})
        assert set(kwargs.keys()) == set(self.template_variables)
        component = _replace_variables(self.template, **kwargs)
        return Document.decode(component, db=self.db).unpack()

    @property
    def form_template(self):
        """Form to be diplayed to user."""
        return {
            'identifier': '<enter-a-unique-identifier>',
            '_variables': {
                k: (
                    f'<value-{i}>'
                    if k not in self.default_values
                    else self.default_values[k]
                )
                for i, k in enumerate(self.template_variables)
            },
            **{k: v for k, v in self.template.items() if k != 'identifier'},
            '_template_name': self.identifier,
        }


class Template(_BaseTemplate):
    """Application template component.

    :param data: Sample data to test the template.
    """

    _artifacts: t.ClassVar[t.Tuple[str]] = (('data', pickle_serializer),)

    type_id: t.ClassVar[str] = "template"

    data: t.List[t.Dict] | None = None

    def pre_create(self, db: Datalayer) -> None:
        """Run before the object is created."""
        super().pre_create(db)
        assert isinstance(self.template, dict)
        self.blobs = list(self.template.get(KEY_BLOBS, {}).keys())
        self.files = list(self.template.get(KEY_FILES, {}).keys())
        db.artifact_store.save_artifact(self.template)
        if self.data is not None:
            if not db.cfg.auto_schema:
                logging.warn('Auto schema is disabled. Skipping data insertion.')
                return
            db[self.default_table].insert(self.data).execute()

    @property
    def default_table(self):
        return f'_sample_{self.identifier}'

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
        assert isinstance(self.template, dict)
        blobs = self.template.pop(KEY_BLOBS, None)
        files = self.template.pop(KEY_FILES, None)

        if self.blobs or self.files:
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

        if self.files is not None and self.files:
            os.makedirs(os.path.join(path, 'files'), exist_ok=True)
            for identifier in self.files:
                file = self.db.artifact_store.get_bytes(identifier)
                with open(path + f'/files/{identifier}', 'wb') as f:
                    f.write(file)

        self._save_blobs_for_export(blobs, path)
        self._save_files_for_export(files, path)

        if zip:
            self._zip_export(path)

    @staticmethod
    def read(path: str, db: t.Optional[Datalayer] = None):
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
        object = super(Template, Template).read(path, db)
        if not hasattr(object, 'variables'):
            raise Exception(
                f"Expected a `Template` object, got {object.__class__.__name__}"
            )
        # Add blobs and files back to the template
        config_object = _build_info_from_path(path=path)
        object.template.update(
            {
                KEY_BLOBS: config_object.get(KEY_BLOBS, {}),
                KEY_FILES: config_object.get(KEY_FILES, {}),
            }
        )
        return object


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
                k: (
                    f'<value-{i}>'
                    if k not in self.default_values
                    else self.default_values[k]
                )
                for i, k in enumerate(self.template_variables)
            },
            **{
                k: v
                for k, v in self.template.items()
                if k not in {'_blobs', 'identifier', '_path'}
            },
            '_path': self.template['_path'],
        }

    def execute(self, **kwargs):
        """Execute the query with the given variables.

        :param kwargs: Variables to be set in the query.
        """
        query = self.query.set_variables(**kwargs)
        return self.db.execute(query)
