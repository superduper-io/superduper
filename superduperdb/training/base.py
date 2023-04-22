import random
import uuid
from collections import defaultdict

import torch.cuda
import torch.nn
import torch.optim
import torch.utils

from superduperdb.training.loading import QueryDataset
from superduperdb.special_dicts import MongoStyleDict
from superduperdb.database import get_database_from_database_type
from superduperdb.models.utils import to_device


class _Mapped:
    def __init__(self, functions, keys):
        self.functions = functions
        self.keys = keys

    def __call__(self, args):
        args = [MongoStyleDict(r) for r in args]
        inputs = [r[k] if k != '_base' else r for r, k in zip(args, self.keys)]
        return [f(i) for i, f in zip(inputs, self.functions)]


class Trainer:
    def __init__(self,
                 identifier,
                 models,
                 keys,
                 model_names,
                 database_type,
                 database,
                 query_params,
                 use_grads=None,
                 splitter=None,
                 validation_sets=(),
                 metrics=None,
                 objective=None,
                 batch_size=100,
                 optimizers=(),
                 lr=0.0001,
                 betas=(0.5, 0.999),
                 num_workers=0,
                 projection=None,
                 features=None,
                 n_epochs=None,
                 n_iterations=None,
                 save=None,
                 watch='objective',
                 log_weights=False,
                 validation_interval=1000,
                 log_interval=1,
                 no_improve_then_stop=10,
                 download=False):

        self.id = uuid.uuid4()
        self.train_name = identifier

        if use_grads is None:
            self.use_grads = {mn: True for mn in model_names}
        else:
            self.use_grads = use_grads
            for mn in model_names:
                if mn not in use_grads:
                    self.use_grads[mn] = True

        self.splitter = splitter
        self.models = models
        self.keys = keys
        self.model_names = model_names

        self.objective = objective

        self.features = features
        self.projection = projection

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.validation_sets = validation_sets
        self.n_epochs = n_epochs
        self.n_iterations = n_iterations
        self.download = download

        self._database = None
        self._database_type = database_type
        self._database_name = database
        self.query_params = query_params

        self.best = []
        self.metrics = metrics if metrics is not None else {}
        self.train_data, self.valid_data = self._get_data()

        self.optimizers = optimizers if optimizers else [
            self._get_optimizer(model, lr, betas) for model, mn in zip(self.models, self.model_names)
            if isinstance(model, torch.nn.Module) and list(model.parameters())
               and self.use_grads[mn]
        ]
        self.watch = watch
        self.metric_values = {}
        for ds in self.validation_sets:
            self.metric_values[ds] = {me: [] for me in self.metrics}
        self.metric_values['objective'] = []
        self.lr = lr
        if log_weights:
            self.weights_dict = defaultdict(lambda: [])
            self._weights_choices = {}
            self._init_weight_traces()
        self._log_weights = log_weights
        self.save = save
        self.validation_interval = validation_interval
        self.log_interval = log_interval

        self.no_improve_then_stop = no_improve_then_stop

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    @property
    def database(self):
        if self._database is None:
            self._database = get_database_from_database_type(self._database_type, self._database_name)
        return self._database

    def _get_data(self):
        train_data = QueryDataset(self._database_type,
                                  self._database_name,
                                  self.query_params,
                                  keys=self.keys,
                                  fold='train',
                                  transform=self.apply_splitter_and_encoders,
                                  features=self.features)
        valid_data = QueryDataset(self._database_type,
                                  self._database_name,
                                  self.query_params,
                                  keys=self.keys,
                                  fold='valid',
                                  transform=self.apply_splitter_and_encoders,
                                  features=self.features)
        return train_data, valid_data

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
        self.database.save_weight_traces(identifier=self.train_name,
                                         variety=self.variety,
                                         weights=self.weights_dict)

    def _init_weight_traces(self):
        for i, e in enumerate(self.models):
            try:
                sd = e.state_dict()
            except AttributeError:
                continue
            self._weights_choices[i] = {}
            for p in sd:
                if len(sd[p].shape) == 1:
                    assert len(sd[p].shape) == 1
                    indexes = [
                        (random.randrange(sd[p].shape[0]),)
                        for _ in range(min(10, sd[p].shape[0]))
                    ]
                    self._weights_choices[i][p] = indexes
                elif len(sd[p].shape) == 2:
                    indexes = [
                        (random.randrange(sd[p].shape[0]), random.randrange(sd[p].shape[1]))
                        for _ in range(min(10, max(sd[p].shape[0], sd[p].shape[1])))
                    ]
                    self._weights_choices[i][p] = indexes
                elif len(sd[p].shape) == 3:
                    indexes = [
                        (random.randrange(sd[p].shape[0]),
                         random.randrange(sd[p].shape[1]),
                         random.randrange(sd[p].shape[2]))
                        for _ in range(min(10, max(sd[p].shape)))
                    ]
                    self._weights_choices[i][p] = indexes
                elif len(sd[p].shape) == 4:
                    indexes = [
                        (random.randrange(sd[p].shape[0]),
                         random.randrange(sd[p].shape[1]),
                         random.randrange(sd[p].shape[2]),
                         random.randrange(sd[p].shape[3]))
                        for _ in range(min(10, max(sd[p].shape)))
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
                    elif len(ind) == 3:
                        tmp.append(param[ind[0], ind[1], ind[2]].item())
                    elif len(ind) == 4:
                        tmp.append(param[ind[0], ind[1], ind[2], ind[3]].item())
                    else:  # pragma: no cover
                        raise Exception('3d tensors not supported')
                self.weights_dict[f'{f}.{p}'].append(tmp)

    def _save_metrics(self):
        self.database.save_metrics(
            self.train_name,
            'learning_task',
            self.metric_values
        )

    def _save_best_model(self):
        agg = min if self.watch == 'objective' else max
        if self.watch == 'objective' \
                and self.metric_values['objective'][-1] == agg(self.metric_values['objective']):
            print('saving')
            for sn, encoder in zip(self.model_names, self.models):
                # only able to save objects of variety "model"
                if sn in self.database.list_models():
                    self.save(sn, encoder)
        else:  # pragma: no cover
            print('no best model found...')

    def calibrate(self, encoder):
        if not hasattr(encoder, 'calibrate'):
            return
        raise NotImplementedError  # pragma: no cover

    def _get_optimizer(self, encoder, lr, betas):
        learnable_parameters = [x for x in encoder.parameters() if x.requires_grad]
        return torch.optim.Adam(learnable_parameters, lr=lr, betas=betas)

    def apply_splitter_and_encoders(self, sample):
        if hasattr(self, 'splitter') and self.splitter is not None:
            sample = [r[k] for r, k in zip(self.splitter(sample), self.keys)]
        else:
            sample = [sample[k] for k in self.keys]

        out = []
        for s, m in zip(sample, self.models):
            try:
                preprocess = m.preprocess
            except AttributeError:
                preprocess = lambda x: x
            out.append(preprocess(s))

        return out

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
    def apply_models_to_batch(batch, models, device):
        batch = to_device(batch, device)
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
        for m in self.models:
            if isinstance(m, torch.nn.Module):
                m.to(self.device)

        for m in self.models:
            self.calibrate(m)

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
                outputs = self.apply_models_to_batch(batch, self.models, self.device)
                l_ = self.take_step(self.objective(*outputs))
                if it % self.log_interval == 0:
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
                    if self._log_weights:
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