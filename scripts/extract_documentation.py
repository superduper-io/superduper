import csv
import typing as t
from pathlib import Path
from bs4 import BeautifulSoup
import re

NEWLINE = r'\n'
assert len(NEWLINE) == 2

FIELD = 'text'
FIELDNAMES = (FIELD,)
CSV_CFG = dict(delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
TAGS = 'p', 'h1', 'h2', 'h3', 'h4', 'h5',


def categorize(f: Path):
    parents = [p.name for p in f.parents]
    if not any(p.startswith('.') or p.startswith('_') for p in parents):
        if parents[0] == 'source':
            return 'source'
        else:
            return 'docs'


def extract(source: Path = Path(), target: Path = Path()):
    result = {}
    total_words = total_lines = 0

    for f in sorted(source.glob('**/*.html')):
        # print(f)
        if category := categorize(f.relative_to(source)):
            items = result.setdefault(category, [])
            for p in BeautifulSoup(f.read_text(), 'html.parser').find_all(TAGS):
                items.extend(ps := p.text.split())
                if ps:
                    items.append(NEWLINE)

    for category, words in sorted(result.items()):
        filename = target / f'{category}.csv'
        with filename.open('w', newline='') as csvfile:
            w = csv.DictWriter(csvfile, fieldnames=[FIELD], quoting=csv.QUOTE_ALL)
            w.writeheader()
            for word in words:
                w.writerow({FIELD: word})
        lines = sum(w == NEWLINE for w in words)
        print(f'Wrote {len(words)} words and {lines} lines to {filename}')
        total_words += len(words)
        total_lines += lines

    print(f'Wrote {total_words} words and {total_lines} lines total')


if __name__ == '__main__':
    import sys

    extract(*(Path(a) for a in sys.argv[1:]))
