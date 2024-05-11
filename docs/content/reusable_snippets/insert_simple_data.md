---
sidebar_label: Insert simple data
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

<!-- TABS -->
# Insert simple data

In order to create data, we need to create a `Schema` for encoding our special `Datatype` column(s) in the databackend.


<Tabs>
    <TabItem value="MongoDB" label="MongoDB" default>
        ```python
        from superduperdb import Document
        
        ids, _ = db.execute(table_or_collection.insert_many(datas))        
        ```
    </TabItem>
    <TabItem value="SQL" label="SQL" default>
        ```python
        ids, _ = db.execute(table_or_collection.insert(datas))        
        ```
    </TabItem>
</Tabs>
