---
sidebar_label: Build multimodal embedding models
filename: build_multimodal_embedding_models.md
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import DownloadButton from '../downloadButton.js';


<!-- TABS -->
# Build multimodal embedding models

Some embedding models such as [CLIP](https://github.com/openai/CLIP) come in pairs of `model` and `compatible_model`.
Otherwise:


<Tabs>
    <TabItem value="Text" label="Text" default>
        ```python
        from superduperdb.ext.sentence_transformers import SentenceTransformer
        
        # Load the pre-trained sentence transformer model
        model = SentenceTransformer(
            identifier='all-MiniLM-L6-v2',
            postprocess=lambda x: x.tolist(),
        )        
        ```
    </TabItem>
    <TabItem value="Image" label="Image" default>
        ```python
        from torchvision import transforms
        import torch
        import torch.nn as nn
        import torchvision.models as models
        
        import warnings
        
        # Import custom modules
        from superduperdb.ext.torch import TorchModel, tensor
        
        # Define a series of image transformations using torchvision.transforms.Compose
        t = transforms.Compose([
            transforms.Resize((224, 224)),   # Resize the input image to 224x224 pixels (must same as here)
            transforms.CenterCrop((224, 224)),  # Perform a center crop on the resized image
            transforms.ToTensor(),  # Convert the image to a PyTorch tensor
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize the tensor with specified mean and standard deviation
        ])
        
        # Define a preprocess function that applies the defined transformations to an input image
        def preprocess(x):
            try:
                return t(x)
            except Exception as e:
                # If an exception occurs during preprocessing, issue a warning and return a tensor of zeros
                warnings.warn(str(e))
                return torch.zeros(3, 224, 224)
        
        # Load the pre-trained ResNet-50 model from torchvision
        resnet50 = models.resnet50(pretrained=True)
        
        # Extract all layers of the ResNet-50 model except the last one
        modules = list(resnet50.children())[:-1]
        resnet50 = nn.Sequential(*modules)
        
        # Create a TorchModel instance with the ResNet-50 model, preprocessing function, and postprocessing lambda
        model = TorchModel(
            identifier='resnet50',
            preprocess=preprocess,
            object=resnet50,
            postprocess=lambda x: x[:, 0, 0],  # Postprocess by extracting the top-left element of the output tensor
            datatype=tensor(dtype='float', shape=(2048,))  # Specify the encoder configuration
        )        
        ```
    </TabItem>
    <TabItem value="Text-Image" label="Text-Image" default>
        ```python
        !pip install git+https://github.com/openai/CLIP.git
        import clip
        from superduperdb import vector
        from superduperdb.ext.torch import TorchModel
        
        # Load the CLIP model and obtain the preprocessing function
        model, preprocess = clip.load("ViT-B/32", device='cpu')
        
        # Define a vector with shape (1024,)
        
        output_datatpye = vector(shape=(1024,))
        
        # Create a TorchModel for text encoding
        compatible_model = TorchModel(
            identifier='clip_text', # Unique identifier for the model
            object=model, # CLIP model
            preprocess=lambda x: clip.tokenize(x)[0],  # Model input preprocessing using CLIP 
            postprocess=lambda x: x.tolist(), # Convert the model output to a list
            datatype=output_datatpye,  # Vector encoder with shape (1024,)
            forward_method='encode_text', # Use the 'encode_text' method for forward pass 
        )
        
        # Create a TorchModel for visual encoding
        model = TorchModel(
            identifier='clip_image',  # Unique identifier for the model
            object=model.visual,  # Visual part of the CLIP model    
            preprocess=preprocess, # Visual preprocessing using CLIP
            postprocess=lambda x: x.tolist(), # Convert the output to a list 
            datatype=output_datatpye, # Vector encoder with shape (1024,)
        )        
        ```
    </TabItem>
    <TabItem value="Audio" label="Audio" default>
        ```python
        !pip install librosa
        import librosa
        import numpy as np
        from superduperdb import ObjectModel
        from superduperdb import vector
        
        def audio_embedding(audio_file):
            # Load the audio file
            MAX_SIZE= 10000
            y, sr = librosa.load(audio_file)
            y = y[:MAX_SIZE]
            mfccs = librosa.feature.mfcc(y=y, sr=44000, n_mfcc=1)
            mfccs =  mfccs.squeeze().tolist()
            return mfccs
        
        if not get_chunking_datatype:
            e =  vector(shape=(1000,))
        else:
            e = get_chunking_datatype(1000)
        
        model= ObjectModel(identifier='my-model-audio', object=audio_embedding, datatype=e)        
        ```
    </TabItem>
</Tabs>
<DownloadButton filename="build_multimodal_embedding_models.md" />
