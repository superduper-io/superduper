import dataclasses as dc
import typing as t

from superduperdb.base.artifact import Artifact
from superduperdb.components.model import Model


@dc.dataclass(kw_only=True)
class SentenceTransformer(Model):
    object: t.Union[Artifact, t.Callable, None] = None
    model: t.Optional[str] = None

    def __post_init__(self):
        super().__post_init__()
        if self.model is None:
            self.model = self.identifier

        if self.object is None:
            import sentence_transformers

            self.object = Artifact(
                artifact=sentence_transformers.SentenceTransformer(
                    self.identifier, device=self.device
                )
            )
        self.object.artifact = self.object.artifact.to(self.device)
        self.model_to_device_method = '_to'
        self.predict_method = 'encode'
        self.batch_predict = True

    def _to(self, device):
        self.object.artifact = self.object.artifact.to(device)
        self.object.artifact._target_device = device
