from __future__ import annotations

import pandas as pd

from superduperdb.misc.text import contextualize


def test_simple_contextualize():
    df = pd.DataFrame({'text': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']})
    context = contextualize(df, 3, 1)
    assert len(context) == 7
