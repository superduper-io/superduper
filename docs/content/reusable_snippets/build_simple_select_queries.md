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
        
        select = table_or_collection.find({})        
        ```
    </TabItem>
    <TabItem value="SQL" label="SQL" default>
        ```python
        
        select = table_or_collection.to_query()        
        ```
    </TabItem>
</Tabs>
