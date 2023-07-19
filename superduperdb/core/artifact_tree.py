import typing as t

from superduperdb.datalayer.base.artifacts import ArtifactStore
from superduperdb.misc.serialization import serializers
from .artifact import Artifact


def put_artifacts_back(d, lookup, artifact_store: t.Optional[ArtifactStore] = None):
    if isinstance(d, dict):
        for k, v in d.items():
            if isinstance(v, dict) and 'file_id' in set(v.keys()):
                if v['file_id'] in lookup:
                    d[k] = lookup[v['file_id']]
                else:
                    d[k] = Artifact(
                        artifact_store.load_artifact(
                            file_id=v['file_id'],
                            serializer=v['serializer'],
                        ),
                        serializer=v['serializer'],
                        info=v.get('info'),
                    )
                    lookup[v['file_id']] = d[k]
            else:
                d[k] = put_artifacts_back(
                    v, lookup=lookup, artifact_store=artifact_store
                )
    elif isinstance(d, list):
        for i, x in enumerate(d):
            d[i] = put_artifacts_back(x, lookup=lookup, artifact_store=artifact_store)
    return d


def get_artifacts(r):
    out = []
    if isinstance(r, Artifact):
        out.extend([r])
    elif isinstance(r, dict):
        out.extend(sum([get_artifacts(v) for v in r.values()], []))
    elif isinstance(r, list):
        out.extend(sum([get_artifacts(x) for x in r], []))
    return out


def infer_artifacts(r):
    if isinstance(r, dict):
        out = []
        for k, v in r.items():
            if isinstance(v, dict) and 'file_id' in v:
                out.append(v['file_id'])
            else:
                out.extend(infer_artifacts(v))
    if isinstance(r, list):
        return sum([infer_artifacts(x) for x in r], [])
    return []


def replace_artifacts_with_dict(d, info):
    if isinstance(d, dict):
        items = d.items() if isinstance(d, dict) else enumerate(d)
        for k, v in items:
            if isinstance(v, Artifact):
                d[k] = {
                    'file_id': info[v],
                    'sha1': v.sha1,
                    'serializer': v.serializer,
                }
            elif isinstance(v, dict) or isinstance(v, list):
                d[k] = replace_artifacts_with_dict(v, info)
    return d


def replace_artifacts(d, info):
    if isinstance(d, dict):
        for k, v in d.items():
            if isinstance(v, Artifact):
                d[k] = info[hash(v)]
            elif isinstance(v, dict):
                replace_artifacts(v, info)
            elif isinstance(v, list):
                replace_artifacts(v, info)
    if isinstance(d, list):
        for x in d:
            replace_artifacts(x, info)


def load_artifacts(d, getter, cache):
    if isinstance(d, dict) or isinstance(d, list):
        items = d.items() if isinstance(d, dict) else enumerate(d)
        for k, v in items:
            if isinstance(v, dict) and 'file_id' in v:
                if v['file_id'] not in cache:
                    bytes = getter(v['file_id'])
                    cache[v['file_id']] = Artifact(
                        artifact=serializers[v['serializer']].decode(bytes),
                        serializer=v['serializer'],
                    )
                d[k] = cache[v['file_id']]
            elif isinstance(v, dict) or isinstance(v, list):
                d[k] = load_artifacts(v, getter, cache)
    return d
