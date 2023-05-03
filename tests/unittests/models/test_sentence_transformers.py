from tests.fixtures.collection import sentences, empty, arrays
from sentence_transformers import SentenceTransformer


def test_transformers(sentences, arrays):
    model = SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens')
    sentences.create_model('bert_embeddings', model, type='array')
    sentences.create_watcher('bert_embeddings', key='text')
    r = sentences.find_one()
    assert 'bert_embeddings' in r['_outputs']['text']
