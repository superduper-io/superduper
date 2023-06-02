from collections import defaultdict

import numpy


def validate_imputation(
    validation_data, models, keys, metrics, predict_kwargs=None
):
    inputs = []
    targets = []
    for i in range(len(validation_data)):
        r = validation_data[i]
        inputs.append(r[keys[0]] if keys[0] != '_base' else r)
        targets.append(r[keys[1]] if keys[1] != '_base' else r)
    predictions = models[0].predict(inputs, **(predict_kwargs or {}))
    metric_values = defaultdict(lambda: [])
    for o, t in zip(predictions, targets):
        for metric in metrics:
            metric_values[metric].append(metrics[metric](o, t))
    for k in metric_values:
        metric_values[k] = sum(metric_values[k]) / len(metric_values[k])
    return metric_values


def validate_vector_search(
    validation_data,
    models,
    keys,
    metrics,
    hash_set_cls,
    measure,
    splitter=None,
    predict_kwargs=None,
):
    inputs = [[] for _ in models]
    for i in range(len(validation_data)):
        r = validation_data[i]
        if splitter is not None:
            all_r = splitter(r)
        else:
            all_r = [r for _ in models]
        for j, m in enumerate(models):
            inputs[j].append(all_r[j][keys[j]])

    random_order = numpy.random.permutation(len(inputs[0]))
    inputs = [[x[i] for i in random_order] for x in inputs]
    predictions = [
        model.predict(inputs[i], **(predict_kwargs or {}))
        for i, model in enumerate(models)
    ]
    h = hash_set_cls(predictions[0], list(range(len(predictions[0]))), measure)
    metric_values = defaultdict(lambda: [])
    for i in range(len(predictions[0])):
        ix, _ = h.find_nearest_from_hash(predictions[0][i], n=100)
        for metric in metrics:
            metric_values[metric.identifier].append(metric(ix, i))

    for k in metric_values:
        metric_values[k] = sum(metric_values[k]) / len(metric_values[k])

    return metric_values
