import enum

import safer
from semver import Version

import superduperdb as s
from superduperdb.misc import run

from . import command

INIT = 'superduperdb/__init__.py'


class Release(str, enum.Enum):
    major = 'major'
    minor = 'minor'
    patch = 'patch'


@command(help='Create a release commit')
def release(update: Release):
    old = Version.parse(s.__version__)
    new = getattr(old, 'bump_' + update)()
    print('Updating', old, 'to', new)
    with open(INIT) as fp_in, safer.open(INIT, 'w') as fp_out:
        for line in fp_in:
            if line.startswith('__version__'):
                line = f"__version__ = '{new}'\n"
            fp_out.write(line)
    run.run(('git', 'commit', INIT, '-m', f'Bump Version v{new}'))
