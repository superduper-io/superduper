from langchain import OpenAI
import os
import pytest

from superduperdb.models.sentence_transformers.wrapper import SentenceTransformer
from superduperdb.core.watcher import Watcher
from superduperdb.core.vector_index import VectorIndex
from superduperdb.datalayer.mongodb.query import Select
from superduperdb.models.langchain.retriever import DBQAWithSourcesChain
from superduperdb.types.numpy.array import Array


SKIP_PAID = os.environ.get('OPENAI_API_KEY') is None
if not SKIP_PAID:
    SKIP_PAID = os.environ.get('SKIP_PAID', 'true') == 'true'


@pytest.mark.skipif(SKIP_PAID, reason='don\'t test paid API')
def test_db_qa_with_sources_chain(nursery_rhymes):
    nursery_rhymes.database.create_component(Array('array', dtype='float32'))
    pl = SentenceTransformer(model_name_or_path='all-MiniLM-L6-v2', type='array')
    nursery_rhymes.database.create_component(pl)
    nursery_rhymes.database.create_component(
        Watcher(model='all-MiniLM-L6-v2', key='text', select=Select('documents'))
    )
    nursery_rhymes.database.create_component(
        VectorIndex(
            'my-index',
            models=['all-MiniLM-L6-v2'],
            watcher='all-MiniLM-L6-v2/text',
            keys=['text'],
        )
    )
    llm = OpenAI(model_name="text-davinci-003")
    m = DBQAWithSourcesChain(
        'retrieval_chain',
        llm=llm,
        vector_index='my-index',
        key='text',
        n=2,
    )
    m.repopulate(nursery_rhymes.database)
    output = m.predict_one('Complete this: Do you know the ...')
    assert 'muffin' in output['answer'].lower()
