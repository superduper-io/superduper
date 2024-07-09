---
sidebar_label: Build and train classifier
filename: build_and_train_classifier.md
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import DownloadButton from '../downloadButton.js';


<!-- TABS -->
# Build and train classifier


<Tabs>
    <TabItem value="Scikit-Learn" label="Scikit-Learn" default>
        ```python
        from superduper.ext.sklearn import Estimator, SklearnTrainer
        from sklearn.svm import SVC
        
        model = Estimator(
            identifier="my-model",
            object=SVC(),
            trainer=SklearnTrainer(
                "my-trainer",
                key=(input_key, "label"),
                select=training_select,
            ),
        )        
        ```
    </TabItem>
    <TabItem value="Torch" label="Torch" default>
        ```python
        import torch
        from torch import nn
        from superduper.ext.torch.model import TorchModel
        from superduper.ext.torch.training import TorchTrainer
        from torch.nn.functional import cross_entropy
        
        
        class SimpleModel(nn.Module):
            def __init__(self, input_size=16, hidden_size=32, num_classes=3):
                super(SimpleModel, self).__init__()
                self.fc1 = nn.Linear(input_size, hidden_size)
                self.relu = nn.ReLU()
                self.fc2 = nn.Linear(hidden_size, num_classes)
        
            def forward(self, x):
                out = self.fc1(x)
                out = self.relu(out)
                out = self.fc2(out)
                return out
        
        preprocess = lambda x: torch.tensor(x)
        
        # Postprocess function for the model output    
        def postprocess(x):
            return int(x.topk(1)[1].item())
        
        def data_transform(features, label):
            return torch.tensor(features), label
        
        # Create a Logistic Regression model
        # feature_length is the input feature size
        model = SimpleModel(feature_size, num_classes=num_classes)
        model = TorchModel(
            identifier='my-model',
            object=model,         
            preprocess=preprocess,
            postprocess=postprocess,
            trainer=TorchTrainer(
                key=(input_key, 'label'),
                identifier='my_trainer',
                objective=cross_entropy,
                loader_kwargs={'batch_size': 10},
                max_iterations=1000,
                validation_interval=100,
                select=select,
                transform=data_transform,
            ),
        )        
        ```
    </TabItem>
</Tabs>
<DownloadButton filename="build_and_train_classifier.md" />
