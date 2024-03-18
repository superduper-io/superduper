---
sidebar_label: Insert data
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

<!-- TABS -->
# Insert data

In order to create data, we need create a `Schema` for encoding our special `Datatype` column(s) in the databackend.

Here's some sample data to work with:


<Tabs>
    <TabItem value="Text" label="Text" default>
        ```python
        !curl -O https://jupyter-sessions.s3.us-east-2.amazonaws.com/text.json
        
        import json
        with open('text.json') as f:
            data = json.load(f)        
        ```
    </TabItem>
    <TabItem value="Images" label="Images" default>
        ```python
        !curl -O https://jupyter-sessions.s3.us-east-2.amazonaws.com/images.zip
        !unzip images.zip
        
        import os
        data = [{'image': f'file://image/{file}'} for file in os.listdir('./images')]        
        ```
    </TabItem>
    <TabItem value="Audio" label="Audio" default>
        ```python
        !curl -O https://jupyter-sessions.s3.us-east-2.amazonaws.com/audio.zip
        !unzip audio.zip
        
        import os
        data = [{'audio': f'file://audio/{file}'} for file in os.listdir('./audio')]        
        ```
    </TabItem>
</Tabs>
The next code-block is only necessary if you're working with a custom `DataType`:

```python
from superduperdb import Schema, Document

schema = Schema(
    'my_schema',
    fields={
        'my_key': dt
    }
)

data = [
    Document({'my_key': item}) for item in data
]
```


<Tabs>
    <TabItem value="MongoDB" label="MongoDB" default>
        ```python
        from superduperdb.backends.mongodb import Collection
        
        collection = Collection('documents')
        
        db.execute(collection.insert_many(data))        
        ```
    </TabItem>
    <TabItem value="SQL" label="SQL" default>
        ```python
        from superduperdb.backends.ibis import Table
        
        table = Table(
            'my_table',
            schema=schema,
        )
        
        db.add(table)
        db.execute(table.insert(data))        
        ```
    </TabItem>
</Tabs>
