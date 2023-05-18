from .faiss.hashes import FaissHashSet
from .vanilla.hashes import VanillaHashSet


hash_set_classes = {'faiss': FaissHashSet, 'vanilla': VanillaHashSet}