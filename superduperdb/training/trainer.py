from superduperdb.training.base import Trainer
from superduperdb.training.validation import validate_representations, validate_imputation


class ImputationTrainer(Trainer):

    variety = 'imputation'

    def __init__(self, *args, inference_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.inference_model = inference_model if inference_model is not None else self.models[0]

    def validate_model(self, data_loader, *args, **kwargs):
        for e in self.models:
            if hasattr(e, 'eval'):
                e.eval()
        results = {}
        if self.metrics:
            for vs in self.validation_sets:
                results[vs] = validate_imputation(self.database, vs, self.train_name,
                                                  self.metrics, model=self.models[0],
                                                  features=self.features)
        objective_values = []
        for batch in data_loader:
            outputs = self.apply_models_to_batch(batch, self.learn_encoders, self.device)
            objective_values.append(self.objective(*outputs).item())
        results['objective'] = sum(objective_values) / len(objective_values)

        for e in self.models:
            if hasattr(e, 'train'):
                e.train()
        return results


class SemanticIndexTrainer(Trainer):
    variety = 'semantic_index'

    def __init__(self, *args, n_retrieve=100, **kwargs):
        self.n_retrieve = n_retrieve
        super().__init__(*args, **kwargs)

    def validate_model(self, data_loader, epoch):
        for m in self.models:
            if hasattr(m, 'eval'):
                m.eval()
        results = {}
        if self.metrics:
            for vs in self.validation_sets:
                results[vs] = validate_representations(self.database,
                                                       vs,
                                                       self.train_name,
                                                       self.metrics,
                                                       self.models)
        objective_values = []
        for batch in data_loader:
            outputs = self.apply_models_to_batch(batch, self.learn_encoders, self.device)
            objective_values.append(self.objective(*outputs).item())
        results['objective'] = sum(objective_values) / len(objective_values)
        for m in self.models:
            if hasattr(m, 'train'):
                m.train()
        return results

