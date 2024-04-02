---
sidebar_label: Build simple select queries
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

<!-- TABS -->
# Build simple select queries


<Tabs>
    <TabItem value="MongoDB" label="MongoDB" default>
        ```python
        
        from superduperdb.backends.mongodb import Collection
        
        select = collection.find({})        
        ```
    </TabItem>
    <TabItem value="Ibis" label="Ibis" default>
        ```python
        
        select = table        
        ```
    </TabItem>
</Tabs>
