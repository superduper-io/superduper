import random
import time
import uuid
from collections import defaultdict

import pymongo
import torch.utils.data

from superduperdb import client
from superduperdb.training.loading import QueryDataset
from superduperdb.utils import MongoStyleDict
from superduperdb.training.validation import validate_representations, validate_imputation


class Trainer:
    def __init__(self,
                 train_name,
                 client,
                 database,
                 collection,
                 models,
                 keys,
                 model_names,
                 use_grads=None,
                 splitter=None,
                 validation_sets=(),
                 metrics=None,
                 objective=None,
                 batch_size=100,
                 optimizers=(),
                 lr=0.0001,
                 num_workers=0,
                 projection=None,
                 filter=None,
                 features=None,
                 n_epochs=None,
                 n_iterations=None,
                 save=None,
                 watch='objective',
                 log_weights=False,
                 validation_interval=1000,
                 no_improve_then_stop=10,
                 download=False):

        self.id = uuid.uuid4()
        self.train_name = train_name
        self._client = client
        self._database = database
        self._collection = collection
        self._collection_object = None

        if use_grads is None:
            self.use_grads = {mn: True for mn in model_names}
        else:
            self.use_grads = use_grads
            for mn in model_names:
                if mn not in use_grads:
                    self.use_grads[mn] = True

        self.splitter = splitter
        self.models = models
        self.model_names = model_names
        self.keys = keys
        self.objective = objective
        self.features = features
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.filter = filter if filter is not None else {}
        self.valid_ids = [
            r['_id'] for r in self.collection.find({**self.filter, '_fold': 'valid'},
                                                   {'_id': 1}).sort('_id', pymongo.ASCENDING)
        ]
        self.validation_sets = validation_sets
        self.n_epochs = n_epochs
        self.n_iterations = n_iterations
        self.download = download
        self.training_id = str(int(time.time() * 100))
        self.projection = projection
        self.train_data = QueryDataset(
            client=self._client,
            database=self._database,
            collection=self._collection,
            filter={**self.filter, '_fold': 'train'},
            transform=self.apply_splitter_and_encoders,
            projection=self.projection,
            features=features,
        )
        self.valid_data = QueryDataset(
            client=self._client,
            database=self._database,
            collection=self._collection,
            filter={**self.filter, '_fold': 'valid'},
            download=True,
            transform=self.apply_splitter_and_encoders,
            projection=self.projection,
            features=features,
        )
        self.best = []
        self.metrics = metrics if metrics is not None else {}
        if isinstance(self.keys, str):  # pragma: no cover
            self.keys = (self.keys,)
        if not isinstance(self.models, tuple) and not isinstance(self.models, list):  # pragma: no cover
            self.models = [self.models, ]
        self.models = list(self.models)
        self._send_to_device()
        self.learn_fields = self.keys
        self.learn_encoders = self.models
        if len(self.keys) == 1:
            self.learn_fields = (self.keys[0], self.keys[0])
            self.learn_encoders = (self.models[0], self.models[0])
        self.optimizers = optimizers if optimizers else [
            self._get_optimizer(model, lr) for model, mn in zip(self.models, self.model_names)
            if isinstance(model, torch.nn.Module) and list(model.parameters())
               and self.use_grads[mn]
        ]
        self.watch = watch
        self.metric_values = {}
        for ds in self.validation_sets:
            self.metric_values[ds] = {me: [] for me in self.metrics}
        self.metric_values['objective'] = []
        self.lr = lr
        self.weights_dict = defaultdict(lambda: [])
        self._weights_choices = {}
        self._init_weight_traces()
        self._log_weights = log_weights
        self.save = save
        self.validation_interval = validation_interval
        self.no_improve_then_stop = no_improve_then_stop

    def _send_to_device(self):
        if not torch.cuda.is_available():
            return
        if torch.cuda.device_count() == 1:
            for m in self.models:
                if isinstance(m, torch.nn.Module):
                    m.to('cuda')
            return
        for i, m in enumerate(self.models):
            if isinstance(m, torch.nn.Module):
                self.models[i] = torch.nn.DataParallel(m)

    def _early_stop(self):
        if self.watch == 'objective':
            to_watch = [-x for x in self.metric_values['objective']]
        else:  # pragma: no cover
            to_watch = self.metric_values[self.watch]
        if max(to_watch[-self.no_improve_then_stop:]) < max(to_watch):  # pragma: no cover
            print('early stopping triggered!')
            return True
        return False

    def _save_weight_traces(self):
        self.collection['_objects'].update_one(
            {'name': self.train_name, 'varieties': self.variety},
            {'$set': {'weights': self.weights_dict}}
        )

    def _init_weight_traces(self):
        for i, e in enumerate(self.models):
            try:
                sd = e.state_dict()
            except AttributeError:
                continue
            self._weights_choices[i] = {}
            for p in sd:
                if len(sd[p].shape) == 2:
                    indexes = [
                        (random.randrange(sd[p].shape[0]), random.randrange(sd[p].shape[1]))
                        for _ in range(min(10, max(sd[p].shape[0], sd[p].shape[1])))
                    ]
                else:
                    assert len(sd[p].shape) == 1
                    indexes = [
                        (random.randrange(sd[p].shape[0]),)
                        for _ in range(min(10, sd[p].shape[0]))
                    ]
                self._weights_choices[i][p] = indexes

    def log_weight_traces(self):
        for i, f in enumerate(self._weights_choices):
            sd = self.models[i].state_dict()
            for p in self._weights_choices[f]:
                indexes = self._weights_choices[f][p]
                tmp = []
                for ind in indexes:
                    param = sd[p]
                    if len(ind) == 1:
                        tmp.append(param[ind[0]].item())
                    elif len(ind) == 2:
                        tmp.append(param[ind[0], ind[1]].item())
                    else:  # pragma: no cover
                        raise Exception('3d tensors not supported')
                self.weights_dict[f'{f}.{p}'].append(tmp)

    def _save_metrics(self):
        self.collection['_objects'].update_one(
            {'name': self.train_name, 'varieties': self.variety},
            {'$set': {'metric_values': self.metric_values}}
        )

    def _save_best_model(self):
        agg = min if self.watch == 'objective' else max
        if self.watch == 'objective' \
                and self.metric_values['objective'][-1] == agg(self.metric_values['objective']):
            print('saving')
            for sn, encoder in zip(self.model_names, self.models):
                if sn in self.collection.list_models():
                    self.save(sn, encoder)
        else:  # pragma: no cover
            print('no best model found...')

    def calibrate(self, encoder):
        if not hasattr(encoder, 'calibrate'):
            return
        raise NotImplementedError  # pragma: no cover

    @property
    def client(self):
        return client.SuperDuperClient(**self._client)

    @property
    def database(self):
        return self.client[self._database]

    @property
    def collection(self):
        if self._collection_object is None:
            self._collection_object = self.database[self._collection]
        return self._collection_object

    def _get_optimizer(self, encoder, lr):
        learnable_parameters = [x for x in encoder.parameters() if x.requires_grad]
        return torch.optim.Adam(learnable_parameters, lr=lr)

    def apply_splitter_and_encoders(self, sample):
        if hasattr(self, 'splitter') and self.splitter is not None:
            sample = self.splitter(sample)
        else:
            sample = [sample for _ in self.learn_fields]
        return _Mapped([
            x.preprocess if hasattr(x, 'preprocess') else lambda x: x
            for x in self.learn_encoders
        ], self.learn_fields)(sample)

    @property
    def data_loaders(self):
        return (
            torch.utils.data.DataLoader(self.train_data, batch_size=self.batch_size,
                                        num_workers=self.num_workers, shuffle=True),
            torch.utils.data.DataLoader(self.valid_data, batch_size=self.batch_size,
                                        num_workers=self.num_workers),
        )

    def log_progress(self, **kwargs):
        out = ''
        for k, v in kwargs.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    out += f'{k}/{kk}: {vv}; '
            else:
                out += f'{k}: {v}; '
        print(out)

    @staticmethod
    def apply_models_to_batch(batch, models):
        output = []
        for subbatch, model in list(zip(batch, models)):
            if isinstance(model, torch.nn.Module) and hasattr(model, 'train_forward'):
                output.append(model.train_forward(subbatch))
            elif isinstance(model, torch.nn.Module):
                output.append(model(subbatch))
            else:
                output.append(subbatch)
        return output

    def take_step(self, objective):
        for opt in self.optimizers:
            opt.zero_grad()
        objective.backward()
        for opt in self.optimizers:
            opt.step()
        if self._log_weights:
            self.log_weight_traces()
        return objective

    def validate_model(self, dataloader, model):
        raise NotImplementedError  # pragma: no cover

    def train(self):

        for encoder in self.models:
            self.calibrate(encoder)

        it = 0
        epoch = 0
        for m in self.models:
            if hasattr(m, 'eval'):
                m.eval()

        metrics = self.validate_model(self.data_loaders[1], -1)
        for k in metrics:
            if k == 'objective':
                self.metric_values[k].append(metrics['objective'])
                continue
            for me in metrics[k]:
                self.metric_values[k][me].append(metrics[k][me])
        self.log_progress(fold='VALID', iteration=it, epoch=epoch, **metrics)
        self._save_metrics()

        while True:
            for m in self.models:
                if hasattr(m, 'train'):
                    m.train()

            train_loader, valid_loader = self.data_loaders

            for batch in train_loader:
                outputs = self.apply_models_to_batch(batch, self.learn_encoders)
                l_ = self.take_step(self.objective(*outputs))
                self.log_progress(fold='TRAIN', iteration=it, epoch=epoch, objective=l_.item())
                it += 1
                if it % self.validation_interval == 0:
                    print('validating model...')
                    metrics = self.validate_model(valid_loader, epoch)
                    for k in metrics:
                        if k == 'objective':
                            self.metric_values[k].append(metrics['objective'])
                            continue
                        for me in metrics[k]:
                            self.metric_values[k][me].append(metrics[k][me])
                    self._save_weight_traces()
                    self._save_best_model()
                    self.log_progress(fold='VALID', iteration=it, epoch=epoch, **metrics)
                    self._save_metrics()
                    stop = self._early_stop()
                    if stop:  # pragma: no cover
                        return
                if self.n_iterations is not None and it >= self.n_iterations:
                    return

            epoch += 1  # pragma: no cover
            if self.n_epochs is not None and epoch >= self.n_epochs:  # pragma: no cover
                return


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
                results[vs] = validate_imputation(self.collection, vs, self.train_name,
                                                  self.metrics, model=self.models[0],
                                                  features=self.features)
        objective_values = []
        for batch in data_loader:
            outputs = self.apply_models_to_batch(batch, self.learn_encoders)
            objective_values.append(self.objective(*outputs).item())
        results['objective'] = sum(objective_values) / len(objective_values)

        for e in self.models:
            if hasattr(e, 'train'):
                e.train()
        return results


class _Mapped:
    def __init__(self, functions, keys):
        self.functions = functions
        self.keys = keys

    def __call__(self, args):
        args = [MongoStyleDict(r) for r in args]
        inputs = [r[k] if k != '_base' else r for r, k in zip(args, self.keys)]
        return [f(i) for i, f in zip(inputs, self.functions)]


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
                results[vs] = validate_representations(
                    self.collection, vs, self.train_name,
                    self.metrics, self.models, features=self.features,
                    refresh=True
                )
        objective_values = []
        for batch in data_loader:
            outputs = self.apply_models_to_batch(batch, self.learn_encoders)
            objective_values.append(self.objective(*outputs).item())
        results['objective'] = sum(objective_values) / len(objective_values)
        for m in self.models:
            if hasattr(m, 'train'):
                m.train()
        return results

