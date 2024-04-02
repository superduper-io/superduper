---
sidebar_label: Insert data
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

<!-- TABS -->
# Insert data

In order to create data, we need to create a `Schema` for encoding our special `Datatype` column(s) in the databackend.

```python
N_DATA = round(len(data) - len(data) // 4)
```


<Tabs>
    <TabItem value="MongoDB" label="MongoDB" default>
        ```python
        from superduperdb import Document
        
        if schema is None:
            data = Document([{'x': datatype(x)} for x in data])    
            db.execute(collection.insert_many(data[:N_DATA]))
        else:
            data = Document([{'x': x} for x in data])    
            db.execute(collection.insert_many(data[:N_DATA], schema='my_schema'))        
        ```
    </TabItem>
    <TabItem value="SQL" label="SQL" default>
        ```python
        from superduperdb import Document
        
        db.execute(table.insert([Document({'x': x}) for x in data[:N_DATA]]))        
        ```
    </TabItem>
</Tabs>
```python
sample_datapoint = data[-1]
```

