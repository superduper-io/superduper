from .base import Trainer
import numpy


def arrays_from_iterator(it, keys):
    out = {k: [] for k in keys}
    for r in it:
        for k in keys:
            out[k].append(r[k])
    for k in out:
        out = numpy.stack(out[k])
    return out


class TrainerForModelWithBatchFitMethod(Trainer):
    def _get_data(self):
        cursor = self.database.execute_query(
            self.database._format_fold_to_query(self.query_params, 'train')
        )
        train_data = arrays_from_iterator(cursor, self.keys)
        cursor = self.database.execute_query(
            self.database._format_fold_to_query(self.query_params, 'valid')
        )
        valid_data = arrays_from_iterator(cursor, self.keys)
        return train_data, valid_data

    def train(self):
        train_data, valid_data = self._get_data()
        print('training with batch fit method...')
        self.models[0].fit(*[train_data[k] for k in self.keys])
        print('saving...')
        self.save(self.model_names[0], self.models[0])
