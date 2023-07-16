from superduperdb.core.artifact import Artifact, InMemoryArtifacts, load_artifact
from superduperdb.core.encoder import Encodable
from superduperdb.datalayer.base.artifacts import ArtifactStore
import bson
import typing as t

ContentType = t.Union[t.Dict, Encodable]


class ArtifactDocument:
    def __init__(self, content):
        self.content = content

    def load_artifacts(self, artifact_store: ArtifactStore, cache: t.Dict):
        return self._load_artifacts(self.content, artifact_store, cache)

    @staticmethod
    def _load_artifacts(
        d: t.Any,
        artifact_store: t.Union[ArtifactStore, InMemoryArtifacts],
        cache: t.Dict,
    ):
        if isinstance(d, dict):
            if 'file_id' in d and 'serializer' in d:
                return load_artifact(d, artifact_store, cache)  # type: ignore[arg-type]
            else:
                for k, v in d.items():
                    if isinstance(v, dict) or isinstance(v, list):
                        d[k] = ArtifactDocument._load_artifacts(
                            v, artifact_store, cache
                        )
        if isinstance(d, list):
            for i, x in enumerate(d):
                d[i] = ArtifactDocument._load_artifacts(x, artifact_store, cache)
        return d

    # ruff: noqa: E501
    @staticmethod
    def _save_artifacts(
        d: t.Any,
        cache: t.Dict,
        artifact_store: t.Union[ArtifactStore, InMemoryArtifacts],
        replace: bool = False,
    ):
        if isinstance(d, dict):
            keys = list(d.keys())
            for k in keys:
                v = d[k]
                if isinstance(v, dict) or isinstance(v, list):
                    ArtifactDocument._save_artifacts(v, cache, artifact_store)
                if isinstance(v, Artifact):
                    v.save(cache=cache, artifact_store=artifact_store, replace=replace)  # type: ignore[arg-type]
                    d[k] = cache[id(v.artifact)]
        if isinstance(d, list):
            for i, x in enumerate(d):
                if isinstance(x, Artifact):
                    x.save(cache=cache, artifact_store=artifact_store)  # type: ignore[arg-type]
                    d[i] = cache[id(x.artifact)]
                ArtifactDocument._save_artifacts(x, cache, artifact_store)

    def save_artifacts(
        self,
        artifact_store: t.Union[ArtifactStore, InMemoryArtifacts],
        cache: t.Dict,
        replace: bool = False,
    ):
        return self._save_artifacts(
            self.content, cache, artifact_store, replace=replace
        )


class Document:
    """
    A wrapper around an instance of dict or a Encodable which may be used to dump
    that resource to a mix of jsonable content or `bytes`
    """

    def __init__(self, content: t.Dict):
        self.content = content

    def __hash__(self):
        return super().__hash__()

    def dump_bson(self):
        return bson.encode(self.encode())

    @staticmethod
    def load_bson(content, encoders):
        return Document(Document.decode(bson.decode(content), encoders=encoders))  # type: ignore[arg-type]

    @staticmethod
    def dump_bsons(documents):
        return bytes(bson.encode({'docs': [d.encode() for d in documents]}))

    @staticmethod
    def load_bsons(content: bytearray, encoders: t.Dict):
        documents = bson.decode(content)['docs']  #  type: ignore[arg-type]
        return [Document(Document.decode(r, encoders=encoders)) for r in documents]

    def _encode(self, r: t.Any):
        if isinstance(r, dict):
            return {k: self._encode(v) for k, v in r.items()}
        elif isinstance(r, Encodable):
            return r.encode()
        return r

    def encode(self):
        return self._encode(self.content)

    @classmethod
    def decode(cls, r: t.Dict, encoders: t.Dict):
        if isinstance(r, Document):
            return Document(cls._decode(r, encoders))
        elif isinstance(r, dict):
            return cls._decode(r, encoders)
        raise NotImplementedError(f'type {type(r)} is not supported')

    @classmethod
    def _decode(cls, r: t.Dict, encoders: t.Dict):
        if isinstance(r, dict) and '_content' in r:
            type = encoders[r['_content']['encoder']]
            try:
                return type.decode(r['_content']['bytes'])
            except KeyError:
                return r
        elif isinstance(r, list):
            return [cls._decode(x, encoders) for x in r]
        elif isinstance(r, dict):
            for k in r:
                r[k] = cls._decode(r[k], encoders)
        return r

    def __repr__(self):
        return f'Document({self.content.__repr__()})'

    def __getitem__(self, item: str):
        assert isinstance(self.content, dict)
        return self.content[item]

    def __setitem__(self, key: str, value: t.Any):
        assert isinstance(self.content, dict)
        self.content[key] = value

    @classmethod
    def _unpack_datavars(cls, item: t.Any):
        if isinstance(item, Encodable):
            return item.x
        elif isinstance(item, dict):
            return {k: cls._unpack_datavars(v) for k, v in item.items()}
        elif isinstance(item, list):
            return [cls._unpack_datavars(x) for x in item]
        else:
            return item

    def unpack(self):
        return self._unpack_datavars(self.content)
