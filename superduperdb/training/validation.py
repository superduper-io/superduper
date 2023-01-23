from collections import defaultdict

import pymongo

from superduperdb.utils import progressbar


def validate_representations(collection, validation_set, semantic_index,
                             metrics, encoders=None,
                             features=None, refresh=False):

    info = collection['_semantic_indexes'].find_one({'name': semantic_index})
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

    for m in encoders:
        if hasattr(m, 'eval'):
            m.eval()
    try:
        collection.unset_hash_set()
    except KeyError as e:  # pragma: no cover
        if not 'semantic_index' in str(e):
            raise e

    active_model = \
        collection.list_models(**{'active': True, 'name': {'$in': encoder_names}})[0]
    key = collection['_models'].find_one({'name': active_model}, {'key': 1})['key']

    if refresh:
        _ids = collection['_validation_sets'].distinct('_id', {'_validation_set': validation_set})
    else:
        _ids = collection['_validation_sets'].distinct('_id', {'_validation_set': validation_set,
                                                               f'_outputs.{key}.{active_model}': {'$exists': 0}})

    collection.remote = False
    for i, m in enumerate(collection.list_models(
        **{'name': {'$in': encoder_names}, 'active': True}
    )):
        collection['_validation_sets']._process_documents_with_model(
            m, _ids, verbose=True, model=encoders[i],
        )
    valid_coll = collection['_validation_sets']
    valid_coll.semantic_index = semantic_index
    anchors = progressbar(
        valid_coll.find({'_validation_set': validation_set},
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
        result = list(valid_coll.find({'$like': {'document': r, 'n': 100}}, {'_id': 1}))
        result = sorted(result, key=lambda r: -r['_score'])
        result = [r['_id'] for r in result]
        for metric in metrics:
            metric_values[metric].append(metrics[metric](result, _id))

    for k in metric_values:
        metric_values[k] = sum(metric_values[k]) / len(metric_values[k])

    return metric_values
