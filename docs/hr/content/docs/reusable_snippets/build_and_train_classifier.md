---
sidebar_label: Build and train classifier
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

<!-- TABS -->
# Build and train classifier


<Tabs>
    <TabItem value="sklearn" label="sklearn" default>
        ```python
        
        from sklearn.linear_model import LogisticRegression
        from superduperdb.ext.sklearn.model import SklearnTrainer, Estimator
        
        # Create a Logistic Regression model
        model = LogisticRegression()
        model = Estimator(object=model, identifier='my-model', train_X=('X', 'y'), train_select=Collection('clt').find())        
        ```
    </TabItem>
    <TabItem value="Torch" label="Torch" default>
        ```python
        from torch import nn
        
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
        
        from superduperdb.ext.torch.model import TorchModel
        from superduperdb.ext.torch.training import TorchTrainer
        # Loss function
        def my_loss(X, y):
            return torch.nn.functional.binary_cross_entropy_with_logits(
                X[:, 0], y.type(torch.float)
            )
        
        
        # Create a Logistic Regression model
        model = SimpleModel()
        model = TorchModel(object=model,         
                    trainer=TorchTrainer(
                    identifier='my_trainer',
                    objective=my_loss,
                    loader_kwargs={'batch_size': 10},
                    max_iterations=100,
                    validation_interval=10,
                ),
                identifier='my-model', train_X=('X', 'y'), train_select=Collection('clt').find())        
        ```
    </TabItem>
</Tabs>
