"AI helper functions for text processing."

import enum
import re

import pandas as pd
from backend.ai.utils.github import URL_CACHE


class TextProcessing(enum.Enum):
    stride = 5
    window_size = 10


def chunk_text_with_sliding_window(
    repo: str,
    df: pd.DataFrame,
    window_size: int,
    stride: int,
    combine: str = ' ',
    text_col: str = 'text',
) -> pd.DataFrame:
    context = []
    n = len(df)

    curr_title = ""
    titles = []
    for i in range(0, n, stride):
        if i + window_size <= n or n - i >= 2:
            window_text = combine.join(df[text_col].iloc[i : min(i + window_size, n)])
            title = re.findall(r'^\s*(#+)\s*(.*)', window_text, re.MULTILINE)
            if title:
                curr_title = title[0][-1]
            context.append(window_text)
            url = URL_CACHE.get((repo, curr_title), 'nan')
            titles.append(url)

    return pd.DataFrame({text_col: context, 'src_url': titles})


def chunk_file_contents(repo, files):
    context_dfs = []
    for file in files:
        with open(file, 'r') as f:
            content = f.readlines()
        content_df = pd.DataFrame({"text": content})
        df = chunk_text_with_sliding_window(
            repo,
            content_df,
            window_size=TextProcessing.window_size.value,
            stride=TextProcessing.stride.value,
        )
        context_dfs.append(df)
    df = pd.concat(context_dfs)
    return df
