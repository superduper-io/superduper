import re

import superduperdb as s

CODE_ROOTS = s.ROOT / 'superduperdb', s.ROOT / 'test'

# DEFECTS maps defect names to functions that match a defect in a line of code.
# Each pattern matches its own definition :-D so 1 is the lowest possible defect count.
DEFECTS = {
    'noqa': re.compile(r'# .*noqa: ').search,
    'type_ignore': re.compile(r'# .*type: ignore').search,
}

# ALLOWABLE_DEFECTS has the allowable defect counts, which should be non-increasing
# over time.
ALLOWABLE_DEFECTS = {
    'noqa': 13,
    'type_ignore': 133,
}


def test_quality():
    files = (f for root in CODE_ROOTS for f in sorted(root.glob('**/*.py')))
    lines = [line for f in files for line in f.read_text().splitlines()]
    defects = {k: sum(bool(v(line)) for line in lines) for k, v in DEFECTS.items()}

    assert defects == ALLOWABLE_DEFECTS
