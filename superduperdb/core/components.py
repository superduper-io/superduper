from superduperdb.core.learning_task import LearningTask
from superduperdb.core.metric import Metric
from superduperdb.core.model import Model
from superduperdb.core.training_configuration import TrainingConfiguration
from superduperdb.core.encoder import Encoder
from superduperdb.core.vector_index import VectorIndex
from superduperdb.core.watcher import Watcher

components = {
    'watcher': Watcher,
    'model': Model,
    'learning_task': LearningTask,
    'metric': Metric,
    'training_configuration': TrainingConfiguration,
    'type': Encoder,
    'vector_index': VectorIndex,
}
