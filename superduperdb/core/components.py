from superduperdb.core.fit import Fit
from superduperdb.core.metric import Metric
from superduperdb.core.model import Model
from superduperdb.core.model import TrainingConfiguration
from superduperdb.core.encoder import Encoder
from superduperdb.core.vector_index import VectorIndex
from superduperdb.core.watcher import Watcher

components = {
    'watcher': Watcher,
    'model': Model,
    'fit': Fit,
    'metric': Metric,
    'training_configuration': TrainingConfiguration,
    'type': Encoder,
    'vector_index': VectorIndex,
}
