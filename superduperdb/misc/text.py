import pandas as pd


def contextualize(
    df: pd.DataFrame,
    window_size: int,
    stride: int,
    combine: str = ' ',
    text_col: str = 'text',
) -> pd.DataFrame:
    """
    A method to contextualize a dataframe of text into a dataframe of windows of text.

    :param df: The dataframe to contextualize.
    :param window_size: The size of the window.
    :param stride: The stride of the window.
    :param combine: The string to use to combine the text.
    :param text_col: The column to use for the text.
    """
    context = []
    n = len(df)

    for i in range(0, n, stride):
        if i + window_size <= n or n - i >= 2:
            window_text = combine.join(df[text_col].iloc[i : min(i + window_size, n)])
            context.append(window_text)

    context = pd.DataFrame({text_col: context})
    return context
