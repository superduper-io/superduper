import typing as t
from functools import cached_property

from langchain.base_language import BaseLanguageModel
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.base import Chain
from langchain.schema import BaseRetriever, Document

from superduperdb.core import documents
from superduperdb.core.base import Placeholder
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
        vector_index: VectorIndex,
        n: int = 100,
    ):
        self.vector_index = vector_index
        self.key = key
        self.n = n

    def get_relevant_documents(self, query: str) -> t.List[Document]:
        document_to_search = documents.Document({self.key: query})
        ids, scores = self.vector_index.get_nearest(
            document_to_search,
            n=self.n,
            featurize=False,
        )
        select = self.vector_index.select.select_using_ids(ids)
        out = list(
            self.vector_index.database.select(  # type: ignore
                select, features=self.vector_index.watcher.features  # type: ignore
            )
        )
        out = [
            Document(
                page_content=x[self.key],
                metadata={
                    'source': x[self.vector_index.database.id_field]  # type: ignore
                },
            )
            for x in out
        ]
        return out

    async def aget_relevant_documents(self, query: str) -> t.List[Document]:
        raise NotImplementedError


class ChainWrapper(Model):
    """
    Wrapper to serialize and apply langchain Chain

    :params chain: Langchain Chain object
    :param identifier: unique ID
    """

    def __init__(self, chain: Chain, identifier: str):
        msg = (
            "Chains which include a retrieval pass are handled by "
            f"{DBQAWithSourcesChain}"
        )
        assert not isinstance(chain, RetrievalQAWithSourcesChain), msg
        super().__init__(chain, identifier=identifier)

    def predict_one(self, input):
        return self.object.run(input)


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

    vector_index: t.Union[Placeholder, VectorIndex]

    def __init__(
        self,
        identifier: str,
        key: str,
        llm: BaseLanguageModel,
        vector_index: t.Union[VectorIndex, str],
        chain_type: str = 'stuff',
        n: int = 5,
    ):
        super().__init__(llm, identifier=identifier)
        self.chain_type = chain_type
        self._retrieval_chain = None
        self.vector_index = (
            vector_index
            if isinstance(vector_index, VectorIndex)
            else Placeholder(variety='vector_index', identifier=vector_index)
        )
        self.key = key
        self.n = n

    @cached_property
    def retriever(self) -> LangchainRetriever:
        assert not isinstance(self.vector_index, Placeholder)
        return LangchainRetriever(
            self.key,
            vector_index=self.vector_index,
            n=self.n,
        )

    @cached_property
    def chain(self) -> Chain:
        assert hasattr(self, 'retriever')
        return RetrievalQAWithSourcesChain.from_chain_type(
            llm=self.object,
            retriever=self.retriever,
            chain_type=self.chain_type,
        )

    def predict_one(self, question, outputs=None, **kwargs):
        return self.chain(question)

    def predict(self):
        raise NotImplementedError
