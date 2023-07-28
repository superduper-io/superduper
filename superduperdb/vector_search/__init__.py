from .faiss_index import FaissVectorIndex
from .table_scan import VanillaVectorIndex

hash_set_classes = {'faiss': FaissVectorIndex, 'vanilla': VanillaVectorIndex}
