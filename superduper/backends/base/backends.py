from superduper.vector_search.atlas import MongoAtlasVectorSearcher
from superduper.vector_search.in_memory import InMemoryVectorSearcher
from superduper.vector_search.lance import LanceVectorSearcher

vector_searcher_implementations = {
    "lance": LanceVectorSearcher,
    "in_memory": InMemoryVectorSearcher,
    "mongodb+srv": MongoAtlasVectorSearcher,
}
