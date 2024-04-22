---
sidebar_label: Create Model Output Type
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

<!-- TABS -->
# Create Model Output Type


<Tabs>
    <TabItem value="MongoDB" label="MongoDB" default>
        ```python
        model_output_dtype = None        
        ```
    </TabItem>
    <TabItem value="SQL" label="SQL" default>
        ```python
        from superduperdb.backends.ibis.field_types import dtype
        model_output_dtype = dtype('str')        
        ```
    </TabItem>
</Tabs>
