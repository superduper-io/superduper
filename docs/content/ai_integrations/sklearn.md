# Scikit-learn

`superduperdb` allows users to work with arbitrary `sklearn` estimators, with additional support for pre-, post-processing and input/ output data-types.

Read more about this [here](/docs/docs/walkthrough/ai_models#scikit-learn).

| Class | Description | GitHub | API-docs |
| --- | --- | --- | --- |
| `superduperdb.ext.sklearn.model.Estimator` | Wraps a scikit-learn estimator | [Code](https://github.com/SuperDuperDB/superduperdb/blob/main/superduperdb/ext/sklearn/model.py) | [Docs](/docs/api/ext/sklearn/model#estimator) |
| `superduperdb.ext.sklearn.model.SklearnTrainer` | May be attached to an `Estimator` for training | [Code](https://github.com/SuperDuperDB/superduperdb/blob/main/superduperdb/ext/sklearn/model.py) | [Docs](/docs/api/ext/sklearn/model#sklearntrainer) |