from superduperdb.core.metric import Metric
from superduperdb.core.model import Model
from superduperdb.core.model import _TrainingConfiguration
from superduperdb.core.encoder import Encoder
from superduperdb.core.vector_index import VectorIndex
from superduperdb.core.watcher import Watcher

components = {
    'watcher': Watcher,
    'model': Model,
    'metric': Metric,
    'training_configuration': _TrainingConfiguration,
    'type': Encoder,
    'vector_index': VectorIndex,
}
