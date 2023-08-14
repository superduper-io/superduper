import glob
import os

import pandas as pd
from backend.ai.utils import contextualize
from backend.config import settings

from superduperdb import superduper
from superduperdb.container.document import Document as D
from superduperdb.container.listener import Listener
from superduperdb.container.vector_index import VectorIndex
from superduperdb.db.mongodb.query import Collection
from superduperdb.ext.openai.model import OpenAIChatCompletion, OpenAIEmbedding


def concept_assist_prompt_build():
    return (
        f'Use the following description and code-snippets aboout SuperDuperDB to answer this question about SuperDuperDB\n'
        'Do not use any other information you might have learned about other python packages\n'
        'Only base your answer on the code-snippets retrieved\n'
        '{context}\n\n'
        'Here\'s the question:\n'
    )


def setup_qa_documentation(mongodb_client):
    db = superduper(mongodb_client.my_database_name)
    if db.show('vector_index'):
        return

    if not os.path.exists(settings.PATH_TO_REPO):
        print(f"Path to repo: {settings.PATH_TO_REPO} does not exist")
        return

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
            df = contextualize(content_df, window_size=10, stride=5)
            context_dfs.append(df)

    # merge all the dataframes
    df = pd.concat(context_dfs)
    documents = [D({"text": v}) for v in df["text"].values]

    db.execute(Collection("markdown").insert_many(documents))

    db.add(
        VectorIndex(
            identifier="documentation_index",
            indexing_listener=Listener(
                model=OpenAIEmbedding(model="text-embedding-ada-002"),
                key="text",
                select=Collection(name="markdown").find(),
            ),
        )
    )

    # Setup the chatbot into the database
    prompt = concept_assist_prompt_build()
    model = OpenAIChatCompletion(
        takes_context=True, prompt=prompt, model="gpt-3.5-turbo"
    )
    db.add(model)
