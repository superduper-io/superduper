from superduper.vector_search.atlas import MongoAtlasVectorSearcher
from superduper.vector_search.in_memory import InMemoryVectorSearcher
from superduper.vector_search.lance import LanceVectorSearcher
from superduper.vector_search.qdrant import QdrantVectorSearcher

vector_searcher_implementations = {
    "lance": LanceVectorSearcher,
    "in_memory": InMemoryVectorSearcher,
    "mongodb+srv": MongoAtlasVectorSearcher,
    "qdrant": QdrantVectorSearcher,
}
