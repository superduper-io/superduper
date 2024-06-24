---
sidebar_label: Build image embedding model
filename: build_image_embedding_model.md
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import DownloadButton from '../downloadButton.js';


<!-- TABS -->
# Build image embedding model
Construct a neural network architecture to project high-dimensional image data into a lower-dimensional, dense vector representation
(embedding) that preserves relevant semantic and visual information within a learned latent space.

```python
!wget https://raw.githubusercontent.com/openai/CLIP/main/CLIP.png
```

```python
image_path = "CLIP.png"
```


<Tabs>
    <TabItem value="TorchVision" label="TorchVision" default>
        ```python
        
        import torch
        import torchvision.models as models
        from torchvision import transforms
        from superduperdb.ext.torch import TorchModel
        
        class TorchVisionEmbedding:
            def __init__(self):
                # Load the pre-trained ResNet-18 model
                self.resnet = models.resnet18(pretrained=True)
                
                # Set the model to evaluation mode
                self.resnet.eval()
                
            def preprocess(self, image):
                # Preprocess the image
                preprocess = preprocess = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
                tensor_image = preprocess(image)
                return tensor_image
                
        embedding_model = TorchVisionEmbedding()
        superdupermodel = TorchModel(identifier='my-vision-model-torch', object=embedding_model.resnet, preprocess=embedding_model.preprocess)        
        ```
    </TabItem>
    <TabItem value="CLIP-multimodal" label="CLIP-multimodal" default>
        ```python
        import torch
        import clip
        from torchvision import transforms
        from superduperdb.ext.torch import TorchModel
        
        class CLIPVisionEmbedding:
            def __init__(self):
                # Load the CLIP model
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                self.model, self.preprocess = clip.load("RN50", device=self.device)
                
            def preprocess(self, image):
                # Load and preprocess the image
                image = self.preprocess(image).unsqueeze(0).to(self.device)
                return image
                
        embedding_model = CLIPVisionEmbedding()
        superdupermodel = TorchModel(identifier='my-vision-model-clip', object=model.model, preprocess=model.preprocess, forward_method='encode_image')        
        ```
    </TabItem>
    <TabItem value="HuggingFace (ViT)" label="HuggingFace (ViT)" default>
        ```python
        import torch
        from transformers import AutoImageProcessor, AutoModel, AutoFeatureExtractor
        import torchvision.transforms as T
        from superduperdb.ext.torch import TorchModel
        
        
        class HuggingFaceEmbeddings(torch.nn.Module):
            def __init__(self):
                super().__init__()
                model_ckpt = "nateraw/vit-base-beans"
                processor = AutoImageProcessor.from_pretrained(model_ckpt)
                self.extractor = AutoFeatureExtractor.from_pretrained(model_ckpt)
                self.model = AutoModel.from_pretrained(model_ckpt)
        
            def forward(self, x):
                return self.model(pixel_values=x).last_hidden_state[:, 0].cpu()
                
                
        class Preprocessor:
            def __init__(self, extractor):
                self.device = 'cpu'
                # Data transformation chain.
                self.transformation_chain = T.Compose(
                    [
                        # We first resize the input image to 256x256 and then we take center crop.
                        T.Resize(int((256 / 224) * extractor.size["height"])),
                        T.CenterCrop(extractor.size["height"]),
                        T.ToTensor(),
                        T.Normalize(mean=extractor.image_mean, std=extractor.image_std),
                    ]
                )
            def __call__(self, image):
                return self.transformation_chain(image).to(self.device)
        
            
        embedding_model = HuggingFaceEmbeddings()
        superdupermodel = TorchModel(identifier='my-vision-model-huggingface', object=embedding_model, preprocess=Preprocessor(embedding_model.extractor))        
        ```
    </TabItem>
</Tabs>
```python
embedding_model.predict(Image.open(image_path))
```

<DownloadButton filename="build_image_embedding_model.md" />
