import glob
import os
import subprocess
from warnings import warn

from backend.config import settings

import pandas as pd

from superduperdb.container.document import Document
from superduperdb.db.mongodb.query import Collection


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


def clone_repo(git_url, target_dir):
    subprocess.run(['git', 'clone', git_url, target_dir])


def setup_data(db):

    if not os.path.exists(settings.PATH_TO_REPO):
        warn(f"Path to repo: {settings.PATH_TO_REPO} does not exist, fetching...")
        clone_repo(settings.GIT_URL, settings.PATH_TO_REPO)

    context_dfs = []
    for level in range(0, settings.DOC_FILE_LEVELS):
        md_path = os.path.join(
            settings.PATH_TO_REPO,
            *["*"] * level if level else '',
            f"*.{settings.DOC_FILE_EXT}",
        )

        for file in glob.glob(md_path):
            print(f"Contextualizing file: {file}")
            content = open(file).readlines()
            content_df = pd.DataFrame({"text": content})
            df = chunk_text_with_sliding_window(content_df, window_size=settings.WINDOW_SIZE, stride=settings.STRIDE)
            context_dfs.append(df)

    df = pd.concat(context_dfs)
    data = [Document({settings.VECTOR_EMBEDDING_KEY: v}) for v in df["text"].values]

    db.execute(
        Collection(settings.MONGO_COLLECTION_NAME).insert_many(data)
    )
