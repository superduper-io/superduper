---
sidebar_label: Create datatype
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

<!-- TABS -->
# Create datatype


<Tabs>
    <TabItem value="Vector" label="Vector" default>
        ```python
        from superduperdb import vector
        
        db.add(vector)        
        ```
    </TabItem>
    <TabItem value="Tensor" label="Tensor" default>
        ```python
        from superduperdb.ext.torch import tensor
        
        db.add()        
        ```
    </TabItem>
    <TabItem value="Array" label="Array" default>
        ```python
        ...        
        ```
    </TabItem>
    <TabItem value="Image" label="Image" default>
        ```python
        from superduperdb.ext.pillow import pil_image        
        ```
    </TabItem>
    <TabItem value="Audio" label="Audio" default>
        ```python
        ...        
        ```
    </TabItem>
    <TabItem value="Video" label="Video" default>
        ```python
        ...        
        ```
    </TabItem>
    <TabItem value="Custom-in-DB" label="Custom-in-DB" default>
        ```python
        ...        
        ```
    </TabItem>
    <TabItem value="Custom-Artifact" label="Custom-Artifact" default>
        ```python
        ...        
        ```
    </TabItem>
</Tabs>
