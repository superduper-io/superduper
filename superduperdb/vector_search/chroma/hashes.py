import chromadb
from superduperdb import cf
from superduperdb.vector_search.base import BaseHashSet

client = chromadb.Client(**cf.get('chroma'))


class ChromaHashSet(BaseHashSet):
    def __init__(self, collection_name, h=None, index=None):
        super().__init__(h, index)
        if h is not None:
            assert index is not None
            self.collection = client.create_collection(collection_name)

    def find_nearest_from_hash(self, h, n=100):
        ...

    def find_nearest_from_hashes(self, h, n=100):
        raise NotImplementedError
