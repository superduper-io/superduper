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
        from superduperdb.ext.sentence_transformers import Pipeline
        
        ...        
        ```
    </TabItem>
    <TabItem value="Image" label="Image" default>
        ```python
        from torchvision import resnet50        
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
