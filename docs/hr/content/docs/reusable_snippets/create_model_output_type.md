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
        chunked_model_datatype = None        
        ```
    </TabItem>
    <TabItem value="SQL" label="SQL" default>
        ```python
        from superduperdb.backends.ibis.field_types import dtype
        chunked_model_datatype = dtype('str')        
        ```
    </TabItem>
</Tabs>
