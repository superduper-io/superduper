---
sidebar_label: Compute features
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

<!-- TABS -->
# Compute features

# Auto generated
This snippet builds on top of:

- [Connect to SuperDuperDB](connect_to_superduperdb)
- [Build a query](build_a_query)

```python
Computing features combines Listener with different model varieties
```


<Tabs>
    <TabItem value="Text" label="Text" default>
        ```python
        from superduperdb.ext.sentence_transformers import Pipeline
        
        db.add(
            Listener(
                model=model,
                select=select
            )
        )        
        ```
    </TabItem>
    <TabItem value="Image" label="Image" default>
        ```python
        from torchvision import resnet50
        
        db.add(
            Listener(
                model=TorchModel(resnet50),
                select=select
            )
        )        
        ```
    </TabItem>
    <TabItem value="Text-2-Image" label="Text-2-Image" default>
        ```python
        from CLIP import ...        
        ```
    </TabItem>
    <TabItem value="Random" label="Random" default>
        ```python
        import numpy
        
        ...        
        ```
    </TabItem>
    <TabItem value="Custom" label="Custom" default>
        ```python
        
        ...        
        ```
    </TabItem>
</Tabs>
