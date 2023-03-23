from collections import defaultdict

from superduperdb.models.utils import apply_model


def validate_imputation(database, validation_set, imputation, metrics, model=None, features=None):

    info = database.get_object_info(identifier=imputation, variety='imputation')
    if model is None:
        model = database.models[info['model']]
    model_key = info['model_key']
    target_key = info['target_key']
    loader_kwargs = info.get('loader_kwargs') or {}

    if isinstance(metrics, list):
        _save = metrics[:]
        metrics = {}
        for m in _save:
            metrics[m] = database.metrics[m]

    query_params = database.get_query_params_for_validation_set(validation_set)
    docs = list(database.execute_query(*query_params, features=features))
    if model_key != '_base':
        inputs_ = [r[model_key] for r in docs]
    elif '_base' in features:
        inputs_ = [r['_base'] for r in docs]
    else:  # pragma: no cover
        inputs_ = docs

    if target_key != '_base':
        targets = [r[target_key] for r in docs]
    else:  # pragma: no cover
        targets = docs

    outputs = apply_model(
        model,
        inputs_,
        single=False,
        **loader_kwargs,
    )
    metric_values = defaultdict(lambda: [])
    for o, t in zip(outputs, targets):
        for metric in metrics:
            metric_values[metric].append(metrics[metric](o, t))
    for k in metric_values:
        metric_values[k] = sum(metric_values[k]) / len(metric_values[k])
    return metric_values


def validate_representations(database, validation_set, semantic_index,
                             metrics, encoders=None):

    info = database.get_object_info(identifier=semantic_index, variety='semantic_index')

    encoder_names = info['models']
    if encoders is None:
        encoders = []
        for e in encoder_names:
            encoders.append(database.models[e])

    if isinstance(metrics, list):
        _save = metrics[:]
        metrics = {}
        for m in _save:
            metrics[m] = database.metrics[m]

    try:
        database.unset_hash_set(semantic_index)
    except KeyError as e:  # pragma: no cover
        if not 'semantic_index' in str(e):
            raise e

    database.remote = False
    database._process_documents_with_watcher(f'{semantic_index}/{validation_set}', model=encoders[0])
    query_params = database.get_query_params_for_validation_set(validation_set)

    anchors = list(database.execute_query(*query_params))
    _ids = database.get_ids_from_result(query_params, anchors)

    metric_values = defaultdict(lambda: [])
    for _id, r in zip(_ids, anchors):
        query_part, r = database.separate_query_part_from_validation_record(r)
        result = list(database.execute_query(*query_params, like=query_part, n=100,
                                             semantic_index=f'{semantic_index}/{validation_set}'))
        result = sorted(result, key=lambda r: -r['_score'])
        result = database.get_ids_from_result(query_params, result)
        for metric in metrics:
            metric_values[metric].append(metrics[metric](result, _id))

    for k in metric_values:
        metric_values[k] = sum(metric_values[k]) / len(metric_values[k])

    return metric_values
