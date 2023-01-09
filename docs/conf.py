import os
import sys

import sphinx_rtd_theme

sys.path.insert(0, os.path.abspath('..'))

extensions = [
    'sphinx.ext.autodoc',
    'sphinx_rtd_theme',
]
source_suffix = '.rst'
master_doc = 'index'
project = u'SuperDuperDB'
copyright = u'Duncan Blythe duncan@superduperdb.com'
exclude_patterns = ['_build']
pygments_style = 'sphinx'
autoclass_content = "both"
html_theme = 'sphinx_rtd_theme'
