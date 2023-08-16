"AI helper functions for text processing."

import enum

import pandas as pd


class TextProcessing(enum.Enum):
    stride = 5
    window_size = 10


def chunk_text_with_sliding_window(
    df: pd.DataFrame,
    window_size: int,
    stride: int,
    combine: str = ' ',
    text_col: str = 'text',
) -> pd.DataFrame:
    context = []
    n = len(df)

    for i in range(0, n, stride):
        if i + window_size <= n or n - i >= 2:
            window_text = combine.join(df[text_col].iloc[i : min(i + window_size, n)])
            context.append(window_text)

    return pd.DataFrame({text_col: context})


def chunk_file_contents(files):
    context_dfs = []
    for file in files:
        with open(file, 'r') as f:
            content = f.readlines()
        content_df = pd.DataFrame({"text": content})
        df = chunk_text_with_sliding_window(
            content_df,
            window_size=TextProcessing.window_size.value,
            stride=TextProcessing.stride.value,
        )
        context_dfs.append(df)
    df = pd.concat(context_dfs)
    return df["text"].values
