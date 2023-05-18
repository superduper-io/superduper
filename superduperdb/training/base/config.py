from superduperdb.training.query_dataset import QueryDataset


class TrainerConfiguration:
    def __init__(self, **parameters):
        for k, v in parameters.items():
            setattr(self, k, v)

    @classmethod
    def split_and_preprocess(cls, r, models):
        raise NotImplementedError

    @classmethod
    def save_models(cls, database, models, model_names):
        for model, mn in zip(models, model_names):
            database._replace_model(model, mn)

    @classmethod
    def _get_data(
        cls,
        database_type,
        database_name,
        query_params,
        keys,
        features,
        transform,
    ):
        train_data = QueryDataset(
            query_params,
            database_name,
            database_type,
            keys=keys,
            fold='train',
            transform=transform,
            features=features,
        )

        valid_data = QueryDataset(
            query_params,
            database_name,
            database_type,
            keys=keys,
            fold='valid',
            transform=transform,
            features=features,
        )

        return train_data, valid_data

    def get(self, k, default=None):
        try:
            return getattr(self, k)
        except:
            return default

    def __call__(
        self,
        identifier,
        models,
        keys,
        model_names,
        database_type,
        database_name,
        query_params,
        validation_sets=(),
        metrics=None,
        features=None,
        download=False,
    ):
        raise NotImplementedError
