---
sidebar_label: Build multimodal embedding models
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

<!-- TABS -->
# Build multimodal embedding models

Some embedding models such as [CLIP](https://github.com/openai/CLIP) come in pairs of `model` and `compatible_model`.
Otherwise:

```python
compatible_model = None
```


<Tabs>
    <TabItem value="Text" label="Text" default>
        ```python
        from superduperdb.ext.sentence_transformers import SentenceTransformer
        
        # Load the pre-trained sentence transformer model
        model = SentenceTransformer(identifier='all-MiniLM-L6-v2')        
        ```
    </TabItem>
    <TabItem value="Image" label="Image" default>
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
                
        model = CLIPVisionEmbedding()
        model = TorchModel(identifier='clip-vision', object=model.model, preprocess=model.preprocess, forward_method='encode_image')        
        ```
    </TabItem>
    <TabItem value="Text+Image" label="Text+Image" default>
        ```python
        
        import torch
        import clip
        from torchvision import transforms
        from superduperdb import Model
        from superduperdb.ext.torch import TorchModel
        
        class CLIPTextEmbedding:
            def __init__(self):
                # Load the CLIP model
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                self.model, _ = clip.load("RN50", device=self.device)
                
            def __call__(self, text):
                features = clip.tokenize([text])
                return self.model.encode_text(features)
                
        model = CLIPTextEmbedding()
        superdupermodel_text = Model(identifier='clip-text', object=model)
        
        class CLIPVisionEmbedding:
            def __init__(self):
                # Load the CLIP model
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                self.model, self.preprocess = clip.load("RN50", device=self.device)
                
            def preprocess(self, image):
                # Load and preprocess the image
                image = self.preprocess(image).unsqueeze(0).to(self.device)
                return image
                
        model = TorchModel(identifier='clip-vision', object=model.model, preprocess=model.preprocess, forward_method='encode_image')
        compatible_model = CLIPVisionEmbedding()        
        ```
    </TabItem>
    <TabItem value="Audio" label="Audio" default>
        ```python
        !pip install librosa
        import librosa
        import numpy as np
        from superduperdb import Model
        
        def audio_embedding(audio_file):
            # Load the audio file
            y, sr = librosa.load(audio_file)
            mfccs = librosa.feature.mfcc(y=y, sr=sr)
            return mfccs
        
        model= Model(identifier='my-model-audio', object=audio_embedding)        
        ```
    </TabItem>
</Tabs>
