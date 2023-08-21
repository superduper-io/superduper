import superduperdb as s
import typing as t
from pathlib import Path
import safer
import sys


def remove_type(line):
    before, _, after = line.partition(' # type:')
    if after:
        return before.rstrip() + '\n'
    return before


def remove_types(file):
    with open(file) as fp_in:
        with safer.open(file, 'w') as fp_out:
            for line in fp_in:
                fp_out.write(remove_type(line))


def remove_all(root):
    root = Path(root)
    if root.suffix == '.py':
        remove_types(root)
    else:
        for file in root.glob('**/*.py'):
            remove_types(file)



if  __name__ == '__main__':
    for a in sys.argv[1:] or ['superduperdb']:
        remove_all(a)
