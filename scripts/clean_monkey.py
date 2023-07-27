#!/usr/bin/env python3

import re
import safer
import subprocess as sp

IMPORT_LINE = 'from typing import '
TYPES = '|'.join((
    'Any',
    'Dict',
    'Iterator',
    'List',
    'Optional',
    'Sequence',
    'Set',
    'Tuple',
    'Union',
))

REPLACE = re.compile(fr'(?<!\bt\.)\b({TYPES})\b')
INIT = '/__init__'


def clean_one(file):
    with safer.open(file, 'w') as fp_out:
        with open(file) as fp_in:
            for line in fp_in:
                if not (ls := line.strip()).startswith(IMPORT_LINE):
                    if not (ls and ls[0] in '#:'):
                        line = REPLACE.sub(r't.\1', line)
                    fp_out.write(line)


def clean(files):
    for file in files:
        name, suffix = file.split('.')
        assert suffix == 'py'

        if name.endswith(INIT):
            name = name[:-len(INIT)]

        module = name.replace('/', '.')
        cmd = 'monkeytype', 'apply', module

        # print('$', *cmd)
        res = sp.run(cmd, text=True, stdout=sp.PIPE, stderr=sp.PIPE)
        if res.returncode:
            print('ERROR', file)
        else:
            clean_one(file)
            print('ok   ', file)


if __name__ == '__main__':
    import sys

    clean(sys.argv[1:])
