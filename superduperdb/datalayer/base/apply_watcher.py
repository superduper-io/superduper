import math
import typing as t

from .query import Select

from superduperdb.cluster.job_submission import work
from superduperdb.misc.logger import logging


@work
def apply_watcher(
    db,
    identifier,
    ids: t.Optional[t.List[str]] = None,
    verbose=False,
    max_chunk_size=5000,
    model=None,
    recompute=False,
    watcher_info=None,
    **kwargs,
):
    if watcher_info is None:
        watcher_info = db.metadata.get_component('watcher', identifier)
    select = db.db.select_cls(**watcher_info['select'])  # type: ignore
    if ids is None:
        ids = db.db.get_ids_from_select(select.select_only_id)
        ids = [str(id) for id in ids]
    if max_chunk_size is not None:
        for it, i in enumerate(range(0, len(ids), max_chunk_size)):
            logging.info(
                'computing chunk ' f'({it + 1}/{math.ceil(len(ids) / max_chunk_size)})'
            )
            apply_watcher(
                db,
                identifier,
                ids=ids[i : i + max_chunk_size],
                verbose=verbose,
                max_chunk_size=None,
                model=model,
                recompute=recompute,
                watcher_info=watcher_info,
                remote=False,
                **kwargs,
            )
        return

    model_info = db.metadata.get_component('model', watcher_info['model'])
    outputs = _compute_model_outputs(
        db,
        model_info,
        ids,
        select,
        key=watcher_info['key'],
        features=watcher_info.get('features', {}),
        model=model,
        predict_kwargs=watcher_info.get('predict_kwargs', {}),
    )
    type = model_info.get('type')
    if type is not None:
        type = db.types[type]
        outputs = [type(x).encode() for x in outputs]
    db.db.write_outputs(watcher_info, outputs, ids)
    return outputs


def _compute_model_outputs(
    db,
    model_info,
    _ids,
    select: Select,
    key='_base',
    features=None,
    model=None,
    predict_kwargs=None,
):
    logging.info('finding documents under filter')
    features = features or {}
    model_identifier = model_info['identifier']
    if features is None:
        features = {}  # pragma: no cover
    documents = list(db.execute(select.select_using_ids(_ids, features=features)))
    logging.info('done.')
    documents = [x.unpack() for x in documents]
    if key != '_base' or '_base' in features:
        passed_docs = [r[key] for r in documents]
    else:  # pragma: no cover
        passed_docs = documents
    if model is None:
        model = db.models[model_identifier]
    return model.predict(passed_docs, **(predict_kwargs or {}))
