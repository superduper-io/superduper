import dataclasses as dc
import typing as t
from functools import cached_property

from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.base import Chain
from langchain.schema import BaseRetriever, Document
from langchain.schema.language_model import BaseLanguageModel

from superduperdb.core import document
from superduperdb.core.model import Model
from superduperdb.core.vector_index import VectorIndex


class LangchainRetriever(BaseRetriever):
    """
    Retriever wrapping SuperDuperDB vector search, for compatibility with LangChain
    RetrievalQAWithSourcesChain

    :param key: Key/field/ column to search
    :param vector_index: Vector-index to use for search
    :param n: Number of documents to use for context
    """

    def __init__(
        self,
        key: str,
        db: t.Any,
        vector_index: VectorIndex,
        n: int = 100,
    ):
        self.vector_index = vector_index
        self.key = key
        self.n = n
        self.db = db

    def get_relevant_documents(self, query: str) -> t.List[Document]:  # type: ignore
        document_to_search = document.Document(content={self.key: query})
        ids, scores = self.vector_index.get_nearest(
            document_to_search,
            n=self.n,
            featurize=False,
        )
        select = self.vector_index.indexing_watcher.select.select_using_ids(ids)
        out = list(self.db.execute(select))
        out = [
            Document(
                page_content=x[self.key],
                metadata={'source': x[self.db.databackend.id_field]},  # type: ignore
            )
            for x in out
        ]
        return out

    async def aget_relevant_documents(  # type:ignore
        self, query: str
    ) -> t.List[Document]:
        raise NotImplementedError


class ChainWrapper(Model):
    """
    Wrapper to serialize and apply langchain Chain

    :params chain: Langchain Chain object
    :param identifier: unique ID
    """

    def predict_one(self, input):
        return self.object.run(input)


@dc.dataclass
class DBQAWithSourcesChain(Model):
    """
    Model which applies ``langchain.chains.RetrievalQAWithSourcesChain`` to the
    database, levering vector search.

    :param identifier: Unique ID
    :param key: Field/ column to apply chain
    :param llm: LangChain language model
    :param vector_index: VectorIndex component to apply
    :param chain_type: important parameter to RetrievalQAWithSourcesChain
    :param n: Number of documents to retrieve as seed for chain
    """

    vector_index: t.Union[VectorIndex, str] = None
    chain_type: str = ('stuff',)
    n: int = (5,)

    def _on_load(self, db):
        if isinstance(self.vector_index, str):
            self.vector_index = db.load('vector_index', self.vector_index)
        self.db = db

    @cached_property
    def retriever(self) -> LangchainRetriever:
        return LangchainRetriever(  # type: ignore
            key=self.vector_index.indexing_watcher.key,
            db=self.db,
            vector_index=self.vector_index,  # type: ignore[arg-type]
            n=self.n,
        )

    @cached_property
    def chain(self) -> Chain:
        return RetrievalQAWithSourcesChain.from_chain_type(
            llm=t.cast(BaseLanguageModel, self.object.artifact),
            retriever=self.retriever,
            chain_type=self.chain_type,
        )

    def _predict_one(self, question, outputs=None, **kwargs):
        return self.chain(question)

    def _predict(self, question):
        if isinstance(question, list):
            return [self._predict_one(q) for q in question]
        return self._predict_one(question)
