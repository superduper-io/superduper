import dataclasses as dc
import os
import subprocess
import typing as t

from superduper import CFG
from superduper.base.base import Base
from superduper.base.constant import KEY_BLOBS, KEY_FILES
from superduper.base.datalayer import Datalayer
from superduper.base.document import Document
from superduper.base.variables import _find_variables, _replace_variables
from superduper.components.component import Component, _build_info_from_path
from superduper.components.table import Table
from superduper.misc import typing as st

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

    template: st.JSON
    template_variables: t.Optional[t.List[str]] = None
    types: t.Optional[t.Dict] = None
    schema: t.Optional[t.Dict] = None
    blobs: t.Any = dc.field(default_factory=dict)
    files: t.Dict = dc.field(default_factory=dict)
    substitutions: dc.InitVar[t.Optional[t.Dict]] = None

    def __post_init__(self, db, substitutions):
        self.substitutions = substitutions
        super().__post_init__(db=db)

    def postinit(self):
        """Post initialization method."""
        if isinstance(self.template, Base):
            self.template = self.template.encode(defaults=True, metadata=False)

        if '_blobs' in self.template:
            self.blobs = self.template.pop('_blobs')

        if '_files' in self.template:
            self.files = self.template.pop('_files')

        if self.substitutions is not None:
            self.substitutions = {
                self.db.databackend.backend_name: 'databackend',
                CFG.output_prefix: 'output_prefix',
                **self.substitutions,
            }
        if self.substitutions is not None:
            self.template = self._document_to_template(
                self.template, self.substitutions
            )
        if self.template_variables is None:
            self.template_variables = sorted(list(set(_find_variables(self.template))))

        super().postinit()

    @staticmethod
    def _document_to_template(r, substitutions):
        substitutions.setdefault(CFG.output_prefix, 'output_prefix')

        def substitute(x):
            if isinstance(x, str):
                for k, v in substitutions.items():
                    x = x.replace(k, f'<var:{v}>')
                return x
            if isinstance(x, dict):
                return {substitute(k): substitute(v) for k, v in x.items()}
            if isinstance(x, (list, tuple)):
                return [substitute(v) for v in x]
            return x

        return substitute(dict(r))

    @ensure_initialized
    def __call__(self, **kwargs):
        """Method to create component from the given template and `kwargs`.

        :param kwargs: Variables to be set in the template.
        """
        kwargs.update({k: v for k, v in self.default_values.items() if k not in kwargs})

        wants = set(self.template_variables) - {'output_prefix', 'databackend'}
        got = set(kwargs.keys()) - {'output_prefix', 'databackend'}

        assert got == wants, f"Expected {wants}, got {got}"

        kwargs['output_prefix'] = CFG.output_prefix
        try:
            kwargs['databackend'] = self.db.databackend.backend_name
        except AttributeError:
            pass

        component = _replace_variables(
            self.template,
            **kwargs,
            build_variables=kwargs,
            build_template=self.identifier,
        )

        from superduper.components.component import build_uuid

        lookup = {}
        for k, v in component.get('_builds', {}).items():
            v['uuid'] = build_uuid()
            lookup[v['uuid']] = k

        component['uuid'] = build_uuid()

        def _replace_uuids(r):
            import json

            encoded = json.dumps(r)
            for k, v in lookup.items():
                encoded = encoded.replace(f'?({v}.uuid)', k)
            return json.loads(encoded)

        component = _replace_uuids(component)

        component['_blobs'] = self.blobs
        component['_files'] = self.files

        return Component.decode(component)

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
            'template': self.template,
            'build_template': self.identifier,
        }


class Template(_BaseTemplate):
    """Application template component.

    :param requirements: pip requirements for the template.
    :param default_tables: Default table to be used with the template.
    :param staged_file: A file which should be staged after installing the template.
    :param queries: `QueryTemplate` instances to be used with the template.
    """

    requirements: t.List[str] | None = None
    default_tables: t.List[Table] | None = None
    staged_file: str | None = None
    queries: t.List['QueryTemplate'] | None = None

    def download(self, name: str = '*', path='./templates'):
        """Download the templates to the given path.

        :param name: Name of the template to download.
        :param path: Path to download the templates.

        Here are the supported templates:

        - llm_finetuning
        - multimodal_image_search
        - multimodal_video_search
        - pdf_rag
        - rag
        - simple_rag
        - text_vector_search
        - transfer_learning
        """
        base_url = 'https://superduper-public-templates.s3.us-east-2.amazonaws.com'
        versions = {
            'llm_finetuning': '0.5.0',
            'multimodal_image_search': '0.5.0',
            'multimodal_video_search': '0.5.0',
            'pdf_rag': '0.5.0',
            'rag': '0.5.0',
            'simple_rag': '0.5.0',
            'text_vector_search': '0.5.0',
            'transfer_learning': '0.5.0',
        }
        templates = {k: base_url + f'/{k}-{versions[k]}.zip' for k in versions}

        if name == '*':
            for a_name in templates:
                self.download(a_name, path + '/' + a_name)
            return

        assert name in templates, '{} not in supported templates {}'.format(
            name, list(templates.keys())
        )

        file = name + '.zip'
        url = templates[name]

        if not os.path.exists(f'/tmp/{file}'):
            subprocess.run(['curl', '-O', '-k', url])
            subprocess.run(['mv', file, f'/tmp/{file}'])
            subprocess.run(['unzip', f'/tmp/{file}', '-d', path])

    def export(
        self,
        path: t.Optional[str] = None,
        defaults: bool = True,
        metadata: bool = False,
    ):
        """
        Save `self` to a directory using super-duper protocol.

        :param path: Path to the directory to save the component.
        :param defaults: Whether to save default values.
        :param metadata: Whether to save metadata.

        Created directory structure:
        ```
        |_component.json
        |_blobs/*
        |_files/*
        ```
        """
        self.init()
        assert isinstance(self.template, dict)

        if path is None:
            path = './' + self.identifier
        super().export(path, defaults=defaults, metadata=metadata)

        self._save_blobs_for_export(self.blobs, path)
        self._save_files_for_export(self.files, path)

    @property
    def default_values(self):
        default_values = super().default_values
        return self._replace_stage_file(default_values)

    @property
    @ensure_initialized
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
        :param db: Datalayer instance to be used to read the component.

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
        object.blobs = config_object.get(KEY_BLOBS, {})
        object.files = config_object.get(KEY_FILES, {})
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

    def postinit(self):
        """Post initialization method."""
        if isinstance(self.template, Base):
            self.template = self.template.dict(metadata=False, defaults=False).encode()
        super().postinit()

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
