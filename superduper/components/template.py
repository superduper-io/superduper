import dataclasses as dc
import os
import pathlib
import typing as t
from contextlib import contextmanager

from superduper import CFG
from superduper.base.base import Base
from superduper.base.constant import KEY_BLOBS, KEY_FILES
from superduper.base.datalayer import Datalayer
from superduper.base.variables import _find_variables, _replace_variables
from superduper.components.component import Component, _build_info_from_path
from superduper.components.table import Table
from superduper.misc import typing as st

from .component import ensure_setup


class Template(Component):
    """Application template component.

    :param template: Template to be used.
    :param template_variables: Variables in the template.
    :param types: Types of variables in the template.
    :param schema: Schema of the template.
    :param blobs: Blobs to be saved with the template.
    :param files: Files to be staged with the template.
    :param substitutions: dict of substitutions to be made in the template.
    :param requirements: pip requirements for the template.
    :param default_tables: Default table to be used with the template.
    """

    template: st.JSON
    template_variables: t.Optional[t.List[str]] = None
    types: t.Optional[t.Dict] = None
    schema: t.Optional[t.Dict] = None
    blobs: t.Any = dc.field(default_factory=dict)
    files: st.FileDict = dc.field(default_factory=dict)
    substitutions: dc.InitVar[t.Optional[t.Dict]] = None
    requirements: t.List[str] | None = None
    default_tables: t.List[Table] | None = None

    def __post_init__(self, db, substitutions):
        self.substitutions = substitutions
        super().__post_init__(db=db)

    @staticmethod
    def read(path):
        """Read the template from the given path.

        :param path: Path to the template.

        If the template has yet to be built, it will be built using
        the `build.ipynb` notebook.
        """
        path = pathlib.Path(path)

        if 'SUPERDUPER_CONFIG' in os.environ:
            os.environ['SUPERDUPER_CONFIG'] = os.path.abspath(
                os.environ['SUPERDUPER_CONFIG']
            )
            os.environ['SUPERDUPER_CONFIG'] = os.path.expanduser(
                os.environ['SUPERDUPER_CONFIG']
            )

        @contextmanager
        def change_dir(destination):
            prev_dir = os.getcwd()
            os.chdir(destination)
            try:
                yield
            finally:
                os.chdir(prev_dir)

        if not os.path.exists(str(path / 'component.json')) and os.path.exists(
            str(path / 'build.ipynb')
        ):
            with change_dir(path):
                import papermill

                papermill.execute_notebook(
                    './build.ipynb', '/tmp/build.ipynb', parameters={'APPLY': False}
                )
        return Component.read(path)

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
    def download(name: str = '*', path='./templates'):
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
            'llm_finetuning': '0.6.1',
            'multimodal_video_search': '0.6.1',
            'pdf_rag': '0.6.1',
            'simple_rag': '0.6.1',
            'transfer_learning': '0.6.0',
        }
        templates = {k: base_url + f'/{k}-{versions[k]}.zip' for k in versions}

        if name == '*':
            for a_name in templates:
                Template.download(a_name, path + '/' + a_name)
            return

        assert name in templates, '{} not in supported templates {}'.format(
            name, list(templates.keys())
        )

        file = name + '-' + versions[name] + '.zip'
        url = templates[name]

        import subprocess

        if not os.path.exists(f'/tmp/{file}'):
            subprocess.run(['curl', '-O', '-k', url])

        subprocess.run(['rm', '-rf', f'{path}/{name}'])
        subprocess.run(['mv', file, f'/tmp/{file}'])
        subprocess.run(['mkdir', '-p', path])
        subprocess.run(['unzip', f'/tmp/{file}', '-d', path])

    @property
    @ensure_setup
    def form_template(self):
        """Form to be diplayed to user."""
        return {
            'types': self.types,
            'schema': self.schema,
            'template': self.template,
            'build_template': self.identifier,
        }

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

    @ensure_setup
    def __call__(self, identifier=None, **kwargs):
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
            identifier=identifier,
            build_variables=kwargs,
            build_template=self.identifier,
        )

        component['_blobs'] = self.blobs
        component['_files'] = self.files

        obj = Component.decode(component)
        if identifier is not None:
            obj.identifier = identifier

        return obj
