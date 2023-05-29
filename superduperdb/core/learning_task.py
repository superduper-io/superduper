from typing import List, Mapping, Optional

from superduperdb.core.base import Component, ComponentList, PlaceholderList, Placeholder
from superduperdb.core.metric import Metric
from superduperdb.core.model import Model
from superduperdb.core.training_configuration import TrainingConfiguration
from superduperdb.datalayer.base.query import Select


class LearningTask(Component):
    variety = 'learning_task'

    def __init__(
        self,
        identifier: str,
        keys: List[str],
        select: Select,
        validation_sets: List[str] = (),
        training_configuration_id: Optional[str] = None,
        training_configuration: Optional[TrainingConfiguration] = None,
        metrics: Optional[List[Metric]] = None,
        metric_ids: Optional[List[str]] = None,
        models: Optional[List[Model]] = None,
        model_ids: Optional[List[str]] = None,
        features: Optional[Mapping[str, str]] = None,
    ):
        super().__init__(identifier)

        assert training_configuration_id or training_configuration
        assert model_ids or models

        self.models = ComponentList('model', models) if models else PlaceholderList('model', model_ids)
        self.keys = keys
        self.training_configuration = (
            training_configuration if training_configuration
            else Placeholder(training_configuration_id, 'training_configuration')
        )
        self.identifier = identifier
        self.validation_sets = validation_sets
        self.metrics = ComponentList('metric', metrics) \
            if metrics else (PlaceholderList('metric', metric_ids) if metric_ids else ())
        self.features = features or {}
        self.select = select

    def asdict(self):
        return {
            'identifier': self.identifier,
            'keys': list(self.keys),
            'validation_sets': list(self.validation_sets),
            'training_configuration': self.training_configuration.identifier,
            'metrics': [m.identifier for m in self.metrics],
            'models': [m.identifier for m in self.models],
            'features': self.features,
        }

    def schedule_jobs(self, database, verbose=True, dependencies=()):
        return [database.fit(self.identifier)]
