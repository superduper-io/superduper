from superduperdb.container.model import Model


class SentenceTransformer(Model):
    def __post_init__(self):
        self.model_to_device_method = '_to'
        super().__post_init__()

    def _to(self, device):
        self.object.artifact.to(device)
        self.object.artifact._target_device = device
