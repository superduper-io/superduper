import clip
import torch

from superduper import Model


class TextModel(Model):
    model: str = 'Vit-B/32'
    device: str = 'cpu'
    
    def setup(self):
        self.vit = clip.load("ViT-B/32", device='cpu')[0]
        self.vit.to(self.device)

    def predict(self, text):
        preprocessed = clip.tokenize(text)
        activations = self.vit.encode_text(preprocessed)[0]
        return activations.detach().numpy()


class ImageModel(Model):
    model: str = 'Vit-B/32'
    device: str = 'cpu'
    batch_size: int = 10

    def setup(self):
        tmp = clip.load("ViT-B/32", device='cpu')
        self.visual_model = tmp[0].visual
        self.preprocess = tmp[1]

    def predict(self, image):
        preprocessed = self.preprocess(image)[None, :]
        activations = self.visual_model(preprocessed)[0]
        return activations.detach().numpy()

    def predict_batches(self, images):
        out = []
        for i in range(0, len(images), self.batch_size):
            sub = images[i: i + self.batch_size]
            preprocessed = [self.preprocess(img) for img in sub]
            activations = self.visual_model(torch.stack(preprocessed, 0))
            out.extend([x.detach().numpy() for x in activations])
        return out