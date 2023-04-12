from .base import Trainer
from ..datasets.misc import arrays_from_iterator


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
