import os
import sys


sys.path.insert(0, os.path.abspath('..'))

extensions =[
    'sphinx.ext.autodoc',
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx.ext.napoleon',
    'nbsphinx',
    'nbsphinx_link',
    'myst_parser',
    'sphinx_copybutton',
    'sphinxcontrib.mermaid',
    'sphinx_markdown_builder'
]

autoclass_content = "both"

copyright = 'SuperDuperDB Inc., opensource@superduperdb.com'

exclude_patterns = ['_build']

html_theme = 'furo'
html_static_path = ['img']
html_logo = 'img/SuperDuperDB_logo.png'
master_doc = 'index'

napoleon_google_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = True
napoleon_include_special_with_doc = True
napoleon_numpy_docstring = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

nbsphinx_allow_errors = True
nbsphinx_execute = 'never'

# project = ''
pygments_style = 'sphinx'

source_suffix = '.rst'
