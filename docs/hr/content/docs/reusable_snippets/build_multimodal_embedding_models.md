---
sidebar_label: Build multimodal embedding models
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

<!-- TABS -->
# Build multimodal embedding models


<Tabs>
    <TabItem value="Text" label="Text" default>
        ```python
        from superduperdb.ext.sentence_transformers import SentenceTransformer
        
        # Load the pre-trained sentence transformer model
        superdupermodel = SentenceTransformer(identifier='all-MiniLM-L6-v2')        
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
        superdupermodel = TorchModel(identifier='clip-vision', object=model.model, preprocess=model.preprocess, forward_method='encode_image')        
        ```
    </TabItem>
    <TabItem value="Text-2-Image" label="Text-2-Image" default>
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
                
        model = CLIPVisionEmbedding()
        superdupermodel_image = TorchModel(identifier='clip-vision', object=model.model, preprocess=model.preprocess, forward_method='encode_image')        
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
        superdupermodel = Model(identifier='my-model-audio', object=audio_embedding)        
        ```
    </TabItem>
</Tabs>
```python
# <testing:>
import wave
import struct

sample_rate = 44100 
duration = 1 
frequency = 440
amplitude = 0.5

# Generate the sine wave
num_samples = int(sample_rate * duration)
t = np.linspace(0, duration, num_samples, False)
signal = amplitude * np.sin(2 * np.pi * frequency * t)

# Open a new WAV file
output_file = 'dummy_audio.wav'
wav_file = wave.open(output_file, 'w')

# Set the parameters for the WAV file
nchannels = 1  # Mono audio
sampwidth = 2  # Sample width in bytes (2 for 16-bit audio)
framerate = sample_rate
nframes = num_samples

# Set the parameters for the WAV file
wav_file.setparams((nchannels, sampwidth, framerate, nframes, 'NONE', 'not compressed'))

# Write the audio data to the WAV file
for sample in signal:
    wav_file.writeframes(struct.pack('h', int(sample * (2 ** 15 - 1))))

# Close the WAV file
wav_file.close()

# Test
superdupermodel.predict_one(output_file)
```

