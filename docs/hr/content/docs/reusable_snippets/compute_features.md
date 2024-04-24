---
sidebar_label: Compute features
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

<!-- TABS -->
# Compute features


<Tabs>
    <TabItem value="Text" label="Text" default>
        ```python
        
        key = 'txt'
        
        import sentence_transformers
        from superduperdb import vector, Listener
        from superduperdb.ext.sentence_transformers import SentenceTransformer
        
        superdupermodel = SentenceTransformer(
            identifier="embedding",
            object=sentence_transformers.SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2"),
            datatype=vector(shape=(384,)),
            postprocess=lambda x: x.tolist(),
        )
        
        jobs, listener = db.apply(
            Listener(
                model=superdupermodel,
                select=select,
                key=key,
                identifier="features"
            )
        )        
        ```
    </TabItem>
    <TabItem value="Image" label="Image" default>
        ```python
        
        key = 'image'
        
        import torchvision.models as models
        from torchvision import transforms
        from superduperdb.ext.torch import TorchModel
        from superduperdb import Listener
        from PIL import Image
        
        class TorchVisionEmbedding:
            def __init__(self):
                # Load the pre-trained ResNet-18 model
                self.resnet = models.resnet18(pretrained=True)
                
                # Set the model to evaluation mode
                self.resnet.eval()
                
            def preprocess(self, image_array):
                # Preprocess the image
                image = Image.fromarray(image_array.astype(np.uint8))
                preprocess = preprocess = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
                tensor_image = preprocess(image)
                return tensor_image
                
        model = TorchVisionEmbedding()
        superdupermodel = TorchModel(identifier='my-vision-model-torch', object=model.resnet, preprocess=model.preprocess, postprocess=lambda x: x.numpy().tolist())
        
        jobs, listener = db.apply(
            Listener(
                model=superdupermodel,
                select=select,
                key=key,
                identifier="features"
            )
        )        
        ```
    </TabItem>
    <TabItem value="Text-And-Image" label="Text-And-Image" default>
        ```python
        import torch
        import clip
        from torchvision import transforms
        from superduperdb import ObjectModel
        from superduperdb import Listener
        
        import torch
        import clip
        from PIL import Image
        
        key={'txt': 'txt', 'image': 'image'}
        
        class CLIPModel:
            def __init__(self):
                # Load the CLIP model
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                self.model, self.preprocess = clip.load("RN50", device=self.device)
        
            def __call__(self, text, image):
                with torch.no_grad():
                    text = clip.tokenize([text]).to(self.device)
                    image = self.preprocess(Image.fromarray(image.astype(np.uint8))).unsqueeze(0).to(self.device)
                    image_features = self.model.encode_image(image)[0].numpy().tolist()
                    text_features = self.model.encode_text(text)[0].numpy().tolist()
                return [image_features, text_features]
                
        model = CLIPModel()
        
        superdupermodel = ObjectModel(identifier="clip", object=model, signature="**kwargs", flatten=True, model_update_kwargs={"document_embedded": False})
        
        jobs, listener = db.apply(
            Listener(
                model=superdupermodel,
                select=select,
                key=key
                identifier="features"
            )
        )
        
        ```
    </TabItem>
    <TabItem value="Random" label="Random" default>
        ```python
        
        key = 'random'
        
        import numpy as np
        from superduperdb import superduper, ObjectModel, Listener
        
        def random(*args, **kwargs):
            return np.random.random(1024).tolist()
        
        superdupermodel = ObjectModel(identifier="random", object=random)
        
        jobs, listener = db.apply(
            Listener(
                model=superdupermodel,
                select=select,
                key=key,
                identifier="features"
            )
        )        
        ```
    </TabItem>
    <TabItem value="Custom" label="Custom" default>
        ```python
        import numpy as np
        from superduperdb import superduper, ObjectModel, Listener
        
        key = 'custom'
        
        # Define any feature calculation function
        def calc_fake_feature(input_data):
            fake_feature = list(range(10))
            return fake_feature
        
        superdupermodel = ObjectModel(identifier="fake_feature", object=calc_fake_feature)
        
        jobs, listener = db.apply(
            Listener(
                model=superdupermodel,
                select=select,
                # key of input_data
                key=key,
                identifier="features"
            )
        )        
        ```
    </TabItem>
</Tabs>
```python
# <testing>
datas = list(db.execute(select.outputs("features::0")))
for data in datas:
    print(len(data["_outputs.features::0"]))
```

