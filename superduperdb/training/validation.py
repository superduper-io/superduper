from collections import defaultdict

import pymongo

from superduperdb.models.utils import apply_model
from superduperdb.utils import progressbar


def validate_imputation(collection, validation_set, imputation, metrics, model=None, features=None):

    info = collection.database['_objects'].find_one({'name': imputation, 'variety': 'imputation'})
    if model is None:
        model = collection.models[info['model']]
    model_key = info['model_key']
    target_key = info['target_key']
    loader_kwargs = info.get('loader_kwargs') or {}

    if isinstance(metrics, list):
        _save = metrics[:]
        metrics = {}
        for m in _save:
            metrics[m] = collection.metrics[m]

    docs = list(collection.database['_validation_sets'].find(
        {'_validation_set': validation_set, 'collection': collection.name},
        features=features,
    ))
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


def validate_representations(collection, validation_set, semantic_index,
                             metrics, encoders=None,
                             features=None, refresh=False):

    info = collection.database['_objects'].find_one({'name': semantic_index, 'variety': 'semantic_index',
                                                     'collection': collection.name})
    encoder_names = info['models']
    projection = info.get('projection', {})
    if encoders is None:
        encoders = []
        for e in encoder_names:
            encoders.append(collection.models[e])

    if isinstance(metrics, list):
        _save = metrics[:]
        metrics = {}
        for m in _save:
            metrics[m] = collection.metrics[m]

    try:
        collection.unset_hash_set()
    except KeyError as e:  # pragma: no cover
        if not 'semantic_index' in str(e):
            raise e

    if refresh:
        _ids = collection['_validation_sets'].distinct('_id', {'_validation_set': validation_set,
                                                               'collection': collection.name})
    else:
        _ids = collection['_validation_sets'].distinct(
            '_id',
            {
                '_validation_set': validation_set,
                f'_outputs.{info["keys"][0]}.{encoder_names[0]}': {'$exists': 0},
                'collection': collection.name
            },
        )

    collection.remote = False
    collection.database._process_documents_with_watcher(
        '_validation_sets',
        encoder_names[0], info['keys'][0], _ids, verbose=True, model=encoders[0],
        recompute=True,
    )
    valid_coll = collection.database['_validation_sets']
    valid_coll.semantic_index = semantic_index
    anchors = progressbar(
        valid_coll.find({'_validation_set': validation_set, 'collection': collection.name},
                        projection,
                        features=features).sort('_id', pymongo.ASCENDING),
        total=valid_coll.count_documents({'_validation_set': validation_set}),
    )

    metric_values = defaultdict(lambda: [])
    for r in anchors:
        _id = r['_id']
        if '_other' in r:
            r = r['_other']
        if '_id' in r:
            del r['_id']
        result = list(valid_coll.find({}, {'_id': 1}, like=r, n=100))
        result = sorted(result, key=lambda r: -r['_score'])
        result = [r['_id'] for r in result]
        for metric in metrics:
            metric_values[metric].append(metrics[metric](result, _id))

    for k in metric_values:
        metric_values[k] = sum(metric_values[k]) / len(metric_values[k])

    return metric_values
