import dataclasses as dc
import os
import typing as t

from superduper import CFG
from superduper.base.constant import KEY_BLOBS, KEY_FILES
from superduper.base.datalayer import Datalayer
from superduper.base.document import Document, QueryUpdateDocument
from superduper.base.leaf import Leaf
from superduper.base.variables import _replace_variables
from superduper.components.component import Component, _build_info_from_path
from superduper.components.table import Table
from superduper.misc.special_dicts import SuperDuperFlatEncode

from .component import ensure_initialized


class _BaseTemplate(Component):
    """
    Base template component.

    :param template: Template component with variables.
    :param template_variables: Variables to be set.
    :param types: Additional information about types of variables.
    :param schema: How to structure frontend form.
    :param blobs: Blob identifiers in `Template.component`.
    :param files: File identifiers in `Template.component`.
    :param substitutions: Substitutions to be made to create variables.
    """

    literals: t.ClassVar[t.Tuple[str]] = ('template',)

    template: t.Union[t.Dict, Component]
    template_variables: t.Optional[t.List[str]] = None
    types: t.Optional[t.Dict] = None
    schema: t.Optional[t.Dict] = None
    blobs: t.Optional[t.List[str]] = None
    files: t.Optional[t.List[str]] = None
    substitutions: dc.InitVar[t.Optional[t.Dict]] = None

    def __post_init__(self, db, substitutions):
        if isinstance(self.template, Leaf):
            self.template = self.template.encode(defaults=True, metadata=False)
        self.template = SuperDuperFlatEncode(self.template)
        if substitutions is not None:
            substitutions = {
                db.databackend.backend_name: 'databackend',
                CFG.output_prefix: 'output_prefix',
                **substitutions,
            }
        if substitutions is not None:
            self.template = QueryUpdateDocument(self.template).to_template(
                **substitutions
            )
        if self.template_variables is None:
            self.template_variables = self.template.variables
        super().__post_init__(db)

    @ensure_initialized
    def __call__(self, **kwargs):
        """Method to create component from the given template and `kwargs`."""
        kwargs.update({k: v for k, v in self.default_values.items() if k not in kwargs})

        assert set(kwargs.keys()) == (
            set(self.template_variables) - {'output_prefix', 'databackend'}
        )
        kwargs['output_prefix'] = CFG.output_prefix
        kwargs['databackend'] = self.db.databackend.backend_name
        component = _replace_variables(
            self.template,
            **kwargs,
            build_variables=kwargs,
            build_template=self.identifier,
        )
        out = Document.decode(component, db=self.db)
        if isinstance(out, Document) and '_base' in out:
            out = out['_base']
        return out

    @property
    def default_values(self):
        default_values = {}
        if self.types:
            for k in self.template_variables:
                if k not in self.types:
                    continue
                if 'default' in self.types[k]:
                    default_values[k] = self.types[k]['default']
        return default_values

    @property
    def form_template(self):
        """Form to be diplayed to user."""
        return {
            'types': self.types,
            'schema': self.schema,
            **{k: v for k, v in self.template.items() if k != 'identifier'},
            'build_template': self.identifier,
        }


class Template(_BaseTemplate):
    """Application template component.

    :param requirements: pip requirements for the template.
    :param default_tables: Default table to be used with the template.
    :param staged_file: A file which should be staged after installing the template.
    :param queries: `QueryTemplate` instances to be used with the template.
    """

    _fields = {'staged_file': 'file'}
    type_id: t.ClassVar[str] = "template"

    requirements: t.List[str] | None = None
    default_tables: t.List[Table] | None = None
    staged_file: str | None = None
    queries: t.List['QueryTemplate'] | None = None

    def _pre_create(self, db: Datalayer) -> None:
        """Run before the object is created."""
        assert isinstance(self.template, dict)
        self.blobs = list(self.template.get(KEY_BLOBS, {}).keys())
        self.files = list(self.template.get(KEY_FILES, {}).keys())
        db.artifact_store.save_artifact(self.template)
        self.init(db)

    def export(
        self,
        path: t.Optional[str] = None,
        format: str = 'json',
        zip: bool = False,
        defaults: bool = True,
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

    @property
    def default_values(self):
        default_values = super().default_values
        return self._replace_stage_file(default_values)

    @property
    def form_template(self):
        """Form to be diplayed to user."""
        form_template = super().form_template
        return self._replace_stage_file(form_template)

    def _replace_stage_file(self, data):
        if self.staged_file:
            self.unpack()
            data = _replace_variables(data, template_staged_file=self.staged_file)
        return data

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

    def __post_init__(self, db, substitutions):
        if isinstance(self.template, Leaf):
            self.template = self.template.dict(metadata=False, defaults=False).encode()
        return super().__post_init__(db, substitutions)

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
