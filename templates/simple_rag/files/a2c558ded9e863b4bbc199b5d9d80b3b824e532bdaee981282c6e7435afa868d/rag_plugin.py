from superduper import Model
from superduper.base.query import Query

import typing as t


class Chunker(Model):
    chunk_size: int = 200

    def predict(self, text):
        text = text.split()
        chunks = [' '.join(text[i:i + self.chunk_size]) for i in range(0, len(text), self.chunk_size)]
        return chunks


class RAGModel(Model):
    """Model to use for RAG.

    :param prompt_template: Prompt template.
    :param select: Query to retrieve data.
    :param key: Key to use for get text out of documents.
    :param llm: Language model to use.
    """

    breaks: t.ClassVar[t.Sequence] = ('llm', 'prompt_template')

    prompt_template: str
    select: Query
    key: str
    llm: Model

    def _build_prompt(self, query, docs):
        chunks = [doc[self.key] for doc in docs]
        context = "\n\n".join(chunks)
        return self.prompt_template.format(context=context, query=query)

    def predict(self, query: str):
        """Predict on a single query string.

        :param query: Query string.
        """
        from superduper.base.datalayer import Datalayer

        assert isinstance(self.db, Datalayer)
        select = self.select.set_variables(db=self.db, query=query)
        results = [r.unpack() for r in select.execute()]
        prompt = self._build_prompt(query, results)
        return self.llm.predict(prompt)


def demo_func(db):
    import streamlit as st
    import openai

    openai.api_key = "your_openai_api_key"

    st.title("Chat with the Superduper docs!")

    user_input = st.text_input("Your Question:", key="question_input")

    rag = db.load('RAGModel', 'simple_rag')

    if st.button("Get Answer") and user_input:
        with st.spinner("Thinking..."):
            answer = rag.predict(user_input)
            st.write(f"**Answer:** {answer}")
